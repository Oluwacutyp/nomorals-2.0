"""Tests for account operations round 20: session-restore login, on-site
password rotation, session health tracking, account health checks, and
vault secret hygiene.

Offline by design — all HTTP is mocked. No network, no real secrets.
"""

from __future__ import annotations

import io
import time
import unittest
from unittest import mock

from nomorals.accounts import (
    AccountHealth,
    AccountManager,
    CredentialVault,
    LoginFailed,
    PasswordChangeConfig,
    Session,
    SessionInvalid,
    SessionManager,
    change_password_on_site,
    check_account_health,
    check_all_health,
    ensure_login,
)
from nomorals.accounts.browser_login import _TOKEN_KINDS
from nomorals.core.errors import NotFound
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test-pass")


def _manager() -> AccountManager:
    return AccountManager(_vault())


class _FakeTab:
    """Duck-typed tab satisfying the browser_login protocol."""

    def __init__(self, fields=("email", "password"),
                 page_text="Welcome back, you are logged in", url_after=""):
        self.fields = set(fields)
        self.fills = {}
        self.submits = 0
        self.page_text = page_text
        self.url = "https://svc.example/login"
        self._url_after = url_after or "https://svc.example/home"
        self.closed = False
        self.evaluated = []
        self.navigated = []
        self.jar = [{"name": "sessionid", "value": "abc123"}]

    def navigate(self, url):
        self.navigated.append(url)
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
        return {"challenges": []}

    def evaluate(self, js):
        self.evaluated.append(js)

    def wait_for_text(self, text, timeout=5000):
        if text not in self.page_text:
            raise TimeoutError("nope")

    def close(self):
        self.closed = True


def _seed_login_cred(mgr: AccountManager, service="svc", user="u"):
    mgr.vault.store(service, user, "s3cret",
                    credential_type="password",
                    metadata={"login_url": "https://svc.example/login",
                              "change_password_url":
                                  "https://svc.example/change-password"})


# ── ensure_login: session restore ──────────────────────────────────────────


class EnsureLoginTests(unittest.TestCase):
    def setUp(self):
        self.mgr = _manager()
        _seed_login_cred(self.mgr)
        self.sessions = SessionManager(self.mgr.vault)
        self.tabs_opened: list[_FakeTab] = []

    def _open_tab(self):
        tab = _FakeTab()
        self.tabs_opened.append(tab)
        return tab

    def test_valid_session_skips_browser(self):
        self.sessions.set_cookies("svc", "u", {"sessionid": "abc"})
        result = ensure_login(self.mgr, self.sessions, self._open_tab,
                              service="svc")
        self.assertTrue(result["ok"])
        self.assertTrue(result["from_session"])
        self.assertEqual([], self.tabs_opened)

    def test_no_session_logs_in(self):
        result = ensure_login(self.mgr, self.sessions, self._open_tab,
                              service="svc")
        self.assertTrue(result["ok"])
        self.assertFalse(result["from_session"])
        self.assertEqual(1, len(self.tabs_opened))
        self.assertEqual(1, result["cookies_saved"])

    def test_stale_session_relogs_in(self):
        self.sessions.set_cookies("svc", "u", {"sessionid": "abc"})
        sess = self.sessions.peek_session("svc", "u")
        self.assertIsNotNone(sess)
        sess.last_used = time.time() - 25 * 3600  # past the 24h window
        self.sessions._save_session(sess)
        self.sessions._sessions.pop(self.sessions._session_key("svc", "u"))
        result = ensure_login(self.mgr, self.sessions, self._open_tab,
                              service="svc")
        self.assertTrue(result["ok"])
        self.assertFalse(result["from_session"])
        self.assertEqual(1, len(self.tabs_opened))

    def test_empty_session_relogs_in(self):
        # A session with no cookies and no token cannot authenticate.
        self.sessions.get_session("svc", "u")
        result = ensure_login(self.mgr, self.sessions, self._open_tab,
                              service="svc")
        self.assertFalse(result["from_session"])
        self.assertEqual(1, len(self.tabs_opened))

    def test_force_bypasses_restore(self):
        self.sessions.set_cookies("svc", "u", {"sessionid": "abc"})
        result = ensure_login(self.mgr, self.sessions, self._open_tab,
                              service="svc", force=True)
        self.assertFalse(result["from_session"])
        self.assertEqual(1, len(self.tabs_opened))

    def test_expired_oauth_relogs_in(self):
        from nomorals.accounts import OAuthToken
        expired = OAuthToken(access_token="tok", expires_at=time.time() - 10)
        self.sessions.set_oauth_token("svc", "u", expired)
        result = ensure_login(self.mgr, self.sessions, self._open_tab,
                              service="svc")
        self.assertFalse(result["from_session"])
        self.assertEqual(1, len(self.tabs_opened))

    def test_missing_service_fails_fast(self):
        with self.assertRaises(LoginFailed):
            ensure_login(self.mgr, self.sessions, self._open_tab, service="")


# ── change_password_on_site ───────────────────────────────────────────────


class ChangePasswordOnSiteTests(unittest.TestCase):
    def setUp(self):
        self.mgr = _manager()
        _seed_login_cred(self.mgr)
        self.sessions = SessionManager(self.mgr.vault)
        self.tabs: list[_FakeTab] = []

    def _open_tab(self, page_text="password changed successfully"):
        def factory():
            tab = _FakeTab(
                fields=("email", "password", "current_password",
                        "new_password", "confirm_password"),
                page_text=page_text)
            self.tabs.append(tab)
            return tab
        return factory

    def test_happy_path_updates_vault(self):
        result = change_password_on_site(
            self.mgr, self.sessions, self._open_tab(),
            service="svc", username="u")
        self.assertTrue(result["ok"])
        self.assertTrue(result["rotated"])
        cred = self.mgr.vault.get("svc", "u", mark_used=False)
        self.assertNotEqual("s3cret", cred.password)
        self.assertIn("last_rotation_at", cred.metadata)
        # The tab saw the old password as current and the new one twice.
        tab = self.tabs[-1]
        self.assertEqual("s3cret", tab.fills["current_password"])
        self.assertEqual(tab.fills["new_password"],
                         tab.fills["confirm_password"])
        self.assertEqual(cred.password, tab.fills["new_password"])

    def test_explicit_new_password_used(self):
        change_password_on_site(
            self.mgr, self.sessions, self._open_tab(),
            service="svc", username="u", new_password="pinned-secret-1")
        cred = self.mgr.vault.get("svc", "u", mark_used=False)
        self.assertEqual("pinned-secret-1", cred.password)

    def test_rejected_change_leaves_vault_alone(self):
        with self.assertRaises(LoginFailed):
            change_password_on_site(
                self.mgr, self.sessions,
                self._open_tab(page_text="passwords do not match, try again"),
                service="svc", username="u")
        cred = self.mgr.vault.get("svc", "u", mark_used=False)
        self.assertEqual("s3cret", cred.password)

    def test_missing_change_url_fails_fast(self):
        self.mgr.vault.store("nosite", "u", "pw",
                             metadata={"login_url": "https://x.example/"})
        with self.assertRaises(LoginFailed):
            change_password_on_site(self.mgr, self.sessions, self._open_tab(),
                                    service="nosite", username="u")
        self.assertEqual([], self.tabs)  # never opened a browser

    def test_success_text_enforced(self):
        with self.assertRaises(LoginFailed):
            change_password_on_site(
                self.mgr, self.sessions, self._open_tab(page_text="meh"),
                service="svc", username="u",
                config=PasswordChangeConfig(success_text="all done"))
        cred = self.mgr.vault.get("svc", "u", mark_used=False)
        self.assertEqual("s3cret", cred.password)

    def test_metadata_config_used(self):
        # change_password_url comes from credential metadata (seeded).
        change_password_on_site(self.mgr, self.sessions, self._open_tab(),
                                service="svc", username="u")
        self.assertEqual("https://svc.example/change-password",
                         self.tabs[-1].navigated[0])


# ── session health ────────────────────────────────────────────────────────


class SessionHealthTests(unittest.TestCase):
    def setUp(self):
        self.sessions = SessionManager(_vault())

    def test_health_ok(self):
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        sess = self.sessions.peek_session("svc", "u")
        report = sess.health_report()
        self.assertEqual("ok", report["status"])
        self.assertEqual([], report["reasons"])

    def test_health_stale(self):
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        sess = self.sessions.peek_session("svc", "u")
        sess.last_used = time.time() - 25 * 3600
        report = sess.health_report()
        self.assertEqual("stale", report["status"])
        self.assertIn("idle_over_24h", report["reasons"])

    def test_health_empty(self):
        self.sessions.get_session("svc", "u")
        sess = self.sessions.peek_session("svc", "u")
        report = sess.health_report()
        self.assertEqual("empty", report["status"])

    def test_health_expired_oauth(self):
        from nomorals.accounts import OAuthToken
        self.sessions.set_oauth_token(
            "svc", "u",
            OAuthToken(access_token="t", expires_at=time.time() - 5))
        sess = self.sessions.peek_session("svc", "u")
        report = sess.health_report()
        self.assertEqual("expired", report["status"])
        self.assertIn("oauth_token_expired", report["reasons"])

    def test_peek_does_not_touch(self):
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        before = self.sessions.peek_session("svc", "u").last_used
        time.sleep(0.01)
        after = self.sessions.peek_session("svc", "u").last_used
        self.assertEqual(before, after)


class EnsureAuthenticatedTests(unittest.TestCase):
    def setUp(self):
        self.sessions = SessionManager(_vault())

    def _expire(self, service="svc", user="u"):
        from nomorals.accounts import OAuthToken
        self.sessions.set_oauth_token(
            service, user,
            OAuthToken(access_token="tok", expires_at=time.time() - 10))

    def test_valid_session_no_reauth_needed(self):
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        called = []
        sess = self.sessions.ensure_authenticated(
            "svc", "u", reauth=lambda: called.append(1))
        self.assertEqual([], called)
        self.assertEqual("u", sess.username)

    def test_reauth_restores_expired_session(self):
        self._expire()

        def reauth():
            self.sessions.set_cookies("svc", "u", {"a": "b"})
            # swap the dead token for a live one
            from nomorals.accounts import OAuthToken
            self.sessions.set_oauth_token(
                "svc", "u",
                OAuthToken(access_token="new",
                           expires_at=time.time() + 3600))

        sess = self.sessions.ensure_authenticated("svc", "u", reauth=reauth)
        self.assertFalse(sess.oauth_token.is_expired())

    def test_reauth_failure_raises(self):
        self._expire()

        def reauth():
            raise RuntimeError("browser exploded")

        with self.assertRaises(SessionInvalid):
            self.sessions.ensure_authenticated("svc", "u", reauth=reauth)

    def test_no_reauth_raises(self):
        self._expire()
        with self.assertRaises(SessionInvalid):
            self.sessions.ensure_authenticated("svc", "u")


class _FakeResponse:
    def __init__(self, body, status=200):
        self._body = body.encode("utf-8")
        self.status = status

    def read(self, n=-1):
        return self._body if n is None or n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.sessions = SessionManager(_vault())
        self.sessions.set_cookies("svc", "u", {"sessionid": "abc"})

    def _patched(self, body, status=200):
        opener = mock.Mock()
        opener.open.return_value = _FakeResponse(body, status)
        return mock.patch("urllib.request.build_opener",
                          return_value=opener)

    def test_logged_in(self):
        with self._patched("<html>Welcome back</html>"):
            res = self.sessions.probe(
                "svc", "u", "https://svc.example/me",
                ok_markers=["Welcome back"])
        self.assertTrue(res["ok"])
        self.assertEqual("logged_in", res["status"])

    def test_logout_marker(self):
        with self._patched("<html>Please log in to continue</html>"):
            res = self.sessions.probe(
                "svc", "u", "https://svc.example/me",
                bad_markers=["log in to continue"])
        self.assertFalse(res["ok"])
        self.assertEqual("logged_out", res["status"])

    def test_locked_classification(self):
        with self._patched("<html>Your account is locked</html>"):
            res = self.sessions.probe(
                "svc", "u", "https://svc.example/me",
                bad_markers=["log in to continue"],
                marker_status={"locked": ["account is locked"]})
        self.assertFalse(res["ok"])
        self.assertEqual("locked", res["status"])

    def test_http_401_is_logged_out(self):
        with self._patched("unauthorized", status=401):
            res = self.sessions.probe("svc", "u", "https://svc.example/me")
        self.assertEqual("logged_out", res["status"])

    def test_transport_error_is_unknown(self):
        opener = mock.Mock()
        opener.open.side_effect = OSError("dns down")
        with mock.patch("urllib.request.build_opener",
                        return_value=opener):
            res = self.sessions.probe("svc", "u", "https://svc.example/me")
        self.assertFalse(res["ok"])
        self.assertEqual("unknown", res["status"])


# ── account health ────────────────────────────────────────────────────────


class AccountHealthTests(unittest.TestCase):
    def setUp(self):
        self.mgr = _manager()
        self.mgr.vault.store("svc", "u", "pw")
        self.sessions = SessionManager(self.mgr.vault)

    def test_missing_credential(self):
        report = check_account_health(self.mgr, self.sessions, "nope")
        self.assertEqual("missing", report.status)
        self.assertFalse(report.ok)

    def test_degraded_without_session(self):
        report = check_account_health(self.mgr, self.sessions, "svc", "u")
        self.assertEqual("degraded", report.status)
        self.assertIn("no session stored", " ".join(report.issues))

    def test_healthy_with_valid_session(self):
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        report = check_account_health(self.mgr, self.sessions, "svc", "u")
        self.assertEqual("healthy", report.status)
        self.assertTrue(report.ok)

    def test_disabled_credential(self):
        self.mgr.vault.deactivate("svc", "u")
        report = check_account_health(self.mgr, self.sessions, "svc", "u")
        self.assertEqual("disabled", report.status)

    def test_expired_credential(self):
        self.mgr.vault.store("svc", "u", "pw",
                             expires_at=time.time() - 10)
        report = check_account_health(self.mgr, self.sessions, "svc", "u")
        self.assertEqual("expired", report.status)

    def test_locked_detected_by_probe(self):
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        with mock.patch.object(
            self.sessions, "probe",
            return_value={"ok": False, "status": "locked",
                          "http_status": 200, "reason": "marker"}):
            report = check_account_health(
                self.mgr, self.sessions, "svc", "u",
                probe_url="https://svc.example/me")
        self.assertEqual("locked", report.status)

    def test_needs_verification_detected_by_probe(self):
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        with mock.patch.object(
            self.sessions, "probe",
            return_value={"ok": False, "status": "needs_verification",
                          "http_status": 200, "reason": "marker"}):
            report = check_account_health(
                self.mgr, self.sessions, "svc", "u",
                probe_url="https://svc.example/me")
        self.assertEqual("needs_verification", report.status)

    def test_logged_out_detected_by_probe(self):
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        with mock.patch.object(
            self.sessions, "probe",
            return_value={"ok": False, "status": "logged_out",
                          "http_status": 200, "reason": "marker"}):
            report = check_account_health(
                self.mgr, self.sessions, "svc", "u",
                probe_url="https://svc.example/me")
        self.assertEqual("logged_out", report.status)

    def test_report_is_json_safe_and_secret_free(self):
        import json
        self.sessions.set_cookies("svc", "u", {"a": "b"})
        report = check_account_health(self.mgr, self.sessions, "svc", "u")
        blob = json.dumps(report.to_dict())
        self.assertNotIn("pw", blob)


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.mgr = _manager()
        self.mgr.vault.store("svc", "u1", "pw")
        self.mgr.vault.store("svc", "u2", "pw")
        self.sessions = SessionManager(self.mgr.vault)
        self.sessions.set_cookies("svc", "u1", {"a": "b"})

    def test_sweep_summary(self):
        sweep = check_all_health(self.mgr, self.sessions)
        self.assertEqual(2, len(sweep["accounts"]))
        self.assertEqual(1, sweep["summary"].get("healthy", 0))
        self.assertEqual(1, sweep["summary"].get("degraded", 0))
        self.assertEqual(1, len(sweep["needs_attention"]))
        self.assertEqual("u2", sweep["needs_attention"][0]["username"])

    def test_sweep_survives_one_bad_account(self):
        import nomorals.accounts.health as health_mod
        real = health_mod.check_account_health
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        with mock.patch.object(health_mod, "check_account_health",
                               side_effect=flaky):
            sweep = check_all_health(self.mgr, self.sessions)
        self.assertEqual(2, len(sweep["accounts"]))
        statuses = {a["username"]: a["status"] for a in sweep["accounts"]}
        self.assertEqual("unknown", statuses["u1"])  # crashed -> unknown
        self.assertIn(statuses["u2"], ("healthy", "degraded"))

    def test_metadata_probe_hints_used(self):
        self.mgr.vault.store(
            "svc", "u2", "pw",
            metadata={"health_probe_url": "https://svc.example/me",
                      "health_ok_markers": ["Welcome"]})
        self.sessions.set_cookies("svc", "u2", {"a": "b"})
        with mock.patch.object(self.sessions, "probe",
                               return_value={"ok": True,
                                             "status": "logged_in",
                                             "http_status": 200,
                                             "reason": "ok"}) as p:
            check_all_health(self.mgr, self.sessions)
        calls = [c for c in p.call_args_list if c.args[1] == "u2"]
        self.assertTrue(calls)
        self.assertEqual("https://svc.example/me", calls[0].args[2])


# ── secret hygiene ────────────────────────────────────────────────────────


class SecretHygieneTests(unittest.TestCase):
    def test_credential_repr_hides_password(self):
        vault = _vault()
        vault.store("s", "u", "super-secret-pw")
        cred = vault.get("s", "u")
        self.assertNotIn("super-secret-pw", repr(cred))
        self.assertIn("s", repr(cred))

    def test_session_repr_hides_cookies(self):
        sessions = SessionManager(_vault())
        sessions.set_cookies("s", "u", {"sessionid": "abc123-secret"})
        sess = sessions.peek_session("s", "u")
        self.assertNotIn("abc123-secret", repr(sess))

    def test_oauth_repr_hides_tokens(self):
        from nomorals.accounts import OAuthToken
        tok = OAuthToken(access_token="tok-secret", refresh_token="ref-secret")
        self.assertNotIn("tok-secret", repr(tok))
        self.assertNotIn("ref-secret", repr(tok))

    def test_redact_scrubs_password_assignments(self):
        from nomorals.core.logging_setup import redact
        out = redact("login attempt password=s3cret-value-here failed")
        self.assertNotIn("s3cret-value-here", out)

    def test_redact_scrubs_cookie_headers(self):
        from nomorals.core.logging_setup import redact
        out = redact("Cookie: sessionid=abc123secretvalue")
        self.assertNotIn("abc123secretvalue", out)


if __name__ == "__main__":
    unittest.main()
