"""Round 19: default-account switching, auto credential rotation, and
browser login for existing accounts (all offline — fake tabs, no network).
"""

from __future__ import annotations

import unittest

from nomorals.accounts import (
    AccountManager,
    CredentialVault,
    LoginCaptchaRequired,
    LoginConfig,
    LoginFailed,
    SessionManager,
    login_with_vault,
)
from nomorals.accounts.browser_login import _TOKEN_KINDS
from nomorals.core.errors import NotFound
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test-pass")


# ── default account switching ──────────────────────────────────────────────


class DefaultAccountTests(unittest.TestCase):
    def setUp(self):
        self.mgr = AccountManager(_vault())
        self.mgr.vault.store("github", "alice", "p1")
        self.mgr.vault.store("github", "bob", "p2")

    def test_no_default_initially(self):
        self.assertIsNone(self.mgr.get_default("github"))
        self.assertIsNone(self.mgr.default_credential("github"))

    def test_set_and_get_default(self):
        self.mgr.set_default("github", "bob")
        self.assertEqual("bob", self.mgr.get_default("github"))
        cred = self.mgr.default_credential("github")
        self.assertEqual("bob", cred.username)
        self.assertEqual("p2", cred.password)

    def test_set_default_unknown_credential_fails_fast(self):
        with self.assertRaises(NotFound):
            self.mgr.set_default("github", "mallory")
        self.assertIsNone(self.mgr.get_default("github"))

    def test_set_default_empty_fails_fast(self):
        with self.assertRaises(ValueError):
            self.mgr.set_default("", "bob")

    def test_clear_default(self):
        self.mgr.set_default("github", "alice")
        self.assertTrue(self.mgr.clear_default("github"))
        self.assertIsNone(self.mgr.get_default("github"))
        self.assertFalse(self.mgr.clear_default("github"))

    def test_resolve_prefers_explicit_username(self):
        self.mgr.set_default("github", "bob")
        cred = self.mgr.resolve_account("github", "alice")
        self.assertEqual("alice", cred.username)

    def test_resolve_falls_back_to_default(self):
        self.mgr.set_default("github", "bob")
        cred = self.mgr.resolve_account("github")
        self.assertEqual("bob", cred.username)

    def test_resolve_single_account_needs_no_default(self):
        mgr = AccountManager(_vault())
        mgr.vault.store("solo", "only", "p")
        self.assertEqual("only", mgr.resolve_account("solo").username)

    def test_resolve_ambiguous_without_default_fails_fast(self):
        with self.assertRaises(NotFound) as ctx:
            self.mgr.resolve_account("github")
        self.assertIn("no default", str(ctx.exception))

    def test_resolve_unknown_service_fails_fast(self):
        with self.assertRaises(NotFound):
            self.mgr.resolve_account("nope")

    def test_switching_default_changes_resolution(self):
        self.mgr.set_default("github", "alice")
        self.assertEqual("alice", self.mgr.resolve_account("github").username)
        self.mgr.set_default("github", "bob")
        self.assertEqual("bob", self.mgr.resolve_account("github").username)


# ── auto rotation ──────────────────────────────────────────────────────────


class RotateAutoTests(unittest.TestCase):
    def setUp(self):
        self.mgr = AccountManager(_vault())
        self.mgr.vault.store("svc", "u", "old-secret")

    def test_rotates_to_fresh_secret(self):
        cred = self.mgr.rotate_credential_auto("svc", "u")
        self.assertNotEqual("old-secret", cred.password)
        self.assertEqual(32, len(cred.password))
        self.assertEqual(cred.password,
                         self.mgr.vault.get("svc", "u").password)

    def test_rotation_is_random(self):
        a = self.mgr.rotate_credential_auto("svc", "u").password
        b = self.mgr.rotate_credential_auto("svc", "u").password
        self.assertNotEqual(a, b)

    def test_custom_length(self):
        cred = self.mgr.rotate_credential_auto("svc", "u", length=16)
        self.assertEqual(16, len(cred.password))


# ── browser login ──────────────────────────────────────────────────────────


class _FakeTab:
    """Duck-typed tab satisfying the browser_login protocol."""

    def __init__(self, fields=("email", "password"), captcha=(),
                 page_text="Welcome back, you are logged in", url_after=""):
        self.fields = set(fields)
        self.fills = {}
        self.submits = 0
        self.captcha = list(captcha)
        self.page_text = page_text
        self.url = "https://svc.example/login"
        self._url_after = url_after or "https://svc.example/home"
        self.closed = False
        self.evaluated = []
        self.jar = [{"name": "sessionid", "value": "abc123"}]

    def navigate(self, url):
        self.url = url
        return {"ok": True}

    def fill(self, name, value):
        if name not in self.fields:
            raise Exception(f"no field {name!r}")
        self.fills[name] = value
        return {"ok": True}

    def submit(self, target=""):
        self.submits += 1
        self.url = self._url_after
        return {"ok": True}

    def text(self, max_chars=20000):
        return {"text": self.page_text}

    def cookies(self):
        return list(self.jar)

    def check_captcha(self):
        return {"challenges": list(self.captcha)}

    def evaluate(self, js):
        self.evaluated.append(js)

    def wait_for_text(self, text, timeout=5000):
        if text not in self.page_text:
            raise TimeoutError("nope")

    def close(self):
        self.closed = True


def _solver_ok(challenge):
    kind = challenge["kind"]
    if kind in _TOKEN_KINDS:
        return {"ok": True, "kind": kind, "backend": "service",
                "token": "TOKEN123", "takeover": False}
    return {"ok": True, "kind": kind, "backend": "service",
            "text": "x7y9", "takeover": False}


class BrowserLoginTests(unittest.TestCase):
    def setUp(self):
        self.vault = _vault()
        self.mgr = AccountManager(self.vault)
        self.sessions = SessionManager(self.vault)
        self.vault.store(
            "acme", "owner@example.com", "s3cret-pw",
            metadata={"login_url": "https://svc.example/login"})

    def _login(self, tab, **kw):
        opened = []

        def factory():
            opened.append(tab)
            return tab

        kw.setdefault("captcha_solver", _solver_ok)
        out = login_with_vault(self.mgr, self.sessions, factory,
                               service="acme", **kw)
        return out, opened

    def test_happy_path_saves_cookies(self):
        tab = _FakeTab()
        out, _ = self._login(tab)
        self.assertTrue(out["ok"])
        self.assertEqual("owner@example.com", out["username"])
        self.assertEqual(1, out["cookies_saved"])
        self.assertTrue(tab.closed)
        sess = self.sessions.get_session("acme", "owner@example.com")
        self.assertEqual("abc123", sess.cookies["sessionid"])

    def test_field_discovery_tries_candidates(self):
        tab = _FakeTab(fields=("login", "passwd"))
        out, _ = self._login(tab)
        self.assertTrue(out["ok"])
        self.assertEqual("owner@example.com", tab.fills["login"])
        self.assertEqual("s3cret-pw", tab.fills["passwd"])

    def test_missing_login_url_fails_fast(self):
        self.vault.store("nourl", "u", "p")
        with self.assertRaises(LoginFailed) as ctx:
            self._login(_FakeTab(), service="nourl")
        self.assertIn("login_url", str(ctx.exception))

    def test_non_password_credential_rejected(self):
        self.vault.store("tok", "u", "k", credential_type="api_key",
                         metadata={"login_url": "https://x.example/"})
        with self.assertRaises(LoginFailed) as ctx:
            self._login(_FakeTab(), service="tok")
        self.assertIn("not a password", str(ctx.exception))

    def test_rejected_password_fails_fast(self):
        tab = _FakeTab(page_text="Incorrect password. Try again.")
        with self.assertRaises(LoginFailed) as ctx:
            self._login(tab)
        self.assertIn("rejected", str(ctx.exception))

    def test_uses_default_account(self):
        self.vault.store("acme", "second@example.com", "pw2",
                         metadata={"login_url": "https://svc.example/login"})
        self.mgr.set_default("acme", "second@example.com")
        tab = _FakeTab()
        out, _ = self._login(tab)
        self.assertEqual("second@example.com", out["username"])

    def test_token_captcha_solved_and_injected(self):
        tab = _FakeTab(captcha=[{
            "kind": "recaptcha_v2", "sitekey": "k",
            "page_url": "https://svc.example/login"}])
        out, _ = self._login(tab)
        self.assertTrue(out["ok"])
        self.assertEqual(1, len(tab.evaluated))
        self.assertIn("TOKEN123", tab.evaluated[0])
        self.assertIn("g-recaptcha-response", tab.evaluated[0])
        # solved pre-submit: exactly one submit happened
        self.assertEqual(1, tab.submits)
        self.assertEqual(1, len(out["captcha_solved"]))

    def test_image_captcha_text_typed_into_field(self):
        tab = _FakeTab(fields=("email", "password", "captcha_code"),
                       captcha=[{"kind": "image_captcha"}])
        out, _ = self._login(tab)
        self.assertTrue(out["ok"])
        self.assertEqual("x7y9", tab.fills["captcha_code"])

    def test_captcha_without_solver_raises_takeover(self):
        tab = _FakeTab(captcha=[{"kind": "recaptcha_v2"}])
        with self.assertRaises(LoginCaptchaRequired) as ctx:
            self._login(tab, captcha_solver=None)
        self.assertEqual(1, len(ctx.exception.challenges))

    def test_solver_failure_raises_takeover(self):
        def bad_solver(ch):
            return {"ok": False, "kind": ch["kind"], "backend": "service",
                    "takeover": True, "detail": "owner must click"}

        tab = _FakeTab(captcha=[{"kind": "hcaptcha"}])
        with self.assertRaises(LoginCaptchaRequired) as ctx:
            self._login(tab, captcha_solver=bad_solver)
        self.assertIn("human hand", str(ctx.exception))

    def test_success_text_verified(self):
        tab = _FakeTab(page_text="dashboard home")
        out, _ = self._login(
            tab, config=LoginConfig(
                login_url="https://svc.example/login",
                success_text="dashboard"))
        self.assertTrue(out["ok"])

    def test_success_text_missing_fails(self):
        tab = _FakeTab(page_text="something else")
        with self.assertRaises(LoginFailed) as ctx:
            self._login(tab, config=LoginConfig(
                login_url="https://svc.example/login",
                success_text="dashboard"))
        self.assertIn("unverified", str(ctx.exception))

    def test_tab_closed_on_failure(self):
        tab = _FakeTab(page_text="Invalid credentials")
        with self.assertRaises(LoginFailed):
            self._login(tab)
        self.assertTrue(tab.closed)


if __name__ == "__main__":
    unittest.main()
