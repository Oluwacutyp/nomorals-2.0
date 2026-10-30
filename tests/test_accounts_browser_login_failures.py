"""Tail-62 pins for nomorals/accounts/browser_login.py —
1 convert / 5 exclusions (217, flagged by the batch agent,
CONVERTED by parent amendment — its two tests, one per consumer
surface, pin the NEW fail-closed behavior). 7 tests for 6 sites
(counts differ; the converted site carries both of its
in-module consumers). Each test pins one site with a control
half proving the guarded operation was actually ATTEMPTED via
the double's own call log. The file contains no exception
handlers, so it adds zero census sites.

browser_login.py calls no failures-ledger record function
anywhere (grep-proven), so NOT-recording is pinned against the
accounts package ledger (nomorals/accounts/failures.py):
recent_failures() snapshot before/after every trigger.

Verdict notes (full grounds in the tail-62 SURVEY.md):
- 177: focus failure -> pass; focus is not the operation —
  typing proceeds and the type/fill calls themselves decide.
- 185: per-char type failure -> the handler IS the fallback
  operation (tab.fill with the full value); a fill failure
  propagates to _fill_first -> LoginFailed.
- 217 CONVERTED (parent amendment): _page_text failure -> None
  (debug-logged), distinguishable from a genuinely textless
  page (""). Both consumers fail closed as UNVERIFIED: login
  raises LoginFailed before any cookies are saved; password
  change raises LoginFailed with the vault untouched — the
  module's own "vault is never updated before the site
  confirms" invariant, and the same honesty the
  success_text-configured path already had. Readable and
  genuinely-textless pages behave exactly as before.
- 426: tab.url failure -> "" — cosmetic: the error message's
  own fallback is the literal "unknown url"; no decision
  reads the value.
- 456 / 715: tab.close() teardown in finally -> pass; both
  results were already decided before the finally block.
"""

import random
import time

import pytest

from nomorals.accounts import browser_login as BL
from nomorals.accounts import failures as acct_fail


def _snap():
    return list(acct_fail.recent_failures(200))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


class Cred:
    credential_type = "password"
    username = "u@example.com"
    password = "pw123"
    metadata = {"login_url": "https://svc.test/login"}
    tags = []
    expires_at = None


class FakeVault:
    def __init__(self):
        self.stored = []

    def store(self, service, username, secret, **kw):
        self.stored.append((service, username, secret))


class FakeManager:
    def __init__(self):
        self.refreshed = []
        self.vault = FakeVault()

    def resolve_account(self, service, username):
        return Cred()

    def refresh_credential(self, service, username, fresh):
        self.refreshed.append((service, username, fresh))
        return Cred()


class FakeSessions:
    def __init__(self, session=None):
        self.jars = []
        self._session = session

    def set_cookies(self, service, username, jar):
        self.jars.append((service, username, dict(jar)))

    def get_valid_session(self, service, username):
        return self._session


class LoginTab:
    def __init__(self, text_mode="welcome", close_raises=False,
                 url_raises=False):
        self.log = []
        self.text_mode = text_mode
        self.close_raises = close_raises
        self.url_raises = url_raises
        self.url_accessed = 0

    def navigate(self, url):
        self.log.append(("navigate", url))

    def fill(self, name, value):
        self.log.append(("fill", name))
        return {}

    def submit(self, target=""):
        self.log.append(("submit", target))
        return {}

    def text(self, max_chars=20000):
        self.log.append(("text", max_chars))
        if self.text_mode == "raise":
            raise RuntimeError("text read boom")
        if self.text_mode == "rejected":
            return {"text": "Sorry, incorrect password. Try again."}
        if self.text_mode == "empty":
            return {"text": ""}
        return {"text": "Welcome back, u"}

    def cookies(self):
        return [{"name": "sess", "value": "abc"}]

    def close(self):
        self.log.append(("close",))
        if self.close_raises:
            raise RuntimeError("close boom")

    @property
    def url(self):
        self.url_accessed += 1
        if self.url_raises:
            raise RuntimeError("url boom")
        return "https://svc.test/app"


class PwTab:
    def __init__(self, text_mode="changed", close_raises=False):
        self.log = []
        self.text_mode = text_mode
        self.close_raises = close_raises

    def navigate(self, url):
        self.log.append(("navigate", url))

    def fill(self, name, value):
        self.log.append(("fill", name))
        return {}

    def submit(self, target=""):
        self.log.append(("submit", target))
        return {}

    def text(self, max_chars=20000):
        self.log.append(("text", max_chars))
        if self.text_mode == "raise":
            raise RuntimeError("text read boom")
        if self.text_mode == "rejected":
            return {"text": "Sorry, incorrect current password."}
        if self.text_mode == "empty":
            return {"text": ""}
        return {"text": "Your password was changed."}

    def cookies(self):
        return [{"name": "sess", "value": "abc"}]

    def close(self):
        self.log.append(("close",))
        if self.close_raises:
            raise RuntimeError("close boom")


class _Session:
    cookies = {"sess": "abc"}
    oauth_token = None


def _change(tab, mgr, sess, **cfgkw):
    return BL.change_password_on_site(
        mgr, sess, lambda: tab, service="svc",
        config=BL.PasswordChangeConfig(
            change_password_url="https://svc.test/pw", **cfgkw),
        new_password="Fresh123!")


def test_177_focus_failure_typing_proceeds():
    before = _snap()

    class Tab:
        def __init__(self):
            self.log = []

        def focus(self, field):
            self.log.append(("focus", field))
            raise RuntimeError("focus boom")

        def type(self, field, ch):
            self.log.append(("type", field, ch))

        def fill(self, name, value):
            self.log.append(("fill", name, value))

    tab = Tab()
    BL.type_like_human(tab, "email", "ab", rng=random.Random(0))
    assert ("focus", "email") in tab.log  # focus was attempted
    assert [e for e in tab.log if e[0] == "type"] == [
        ("type", "email", "a"), ("type", "email", "b")]
    assert _snap() == before


def test_185_type_failure_falls_back_to_fill():
    before = _snap()

    class Tab:
        def __init__(self, type_raises):
            self.log = []
            self.type_raises = type_raises

        def focus(self, field):
            self.log.append(("focus", field))

        def type(self, field, ch):
            self.log.append(("type", field, ch))
            if self.type_raises:
                raise RuntimeError("type boom")

        def fill(self, name, value):
            self.log.append(("fill", name, value))

    bad = Tab(type_raises=True)
    BL.type_like_human(bad, "email", "ab", rng=random.Random(0))
    assert ("type", "email", "a") in bad.log  # typing was attempted
    assert ("fill", "email", "ab") in bad.log  # fallback, full value

    good = Tab(type_raises=False)
    BL.type_like_human(good, "email", "ab", rng=random.Random(0))
    assert not [e for e in good.log if e[0] == "fill"]  # control
    assert _snap() == before


def test_217_login_unreadable_fails_unverified():
    before = _snap()
    # control: a READABLE rejection page fails the login honestly
    with pytest.raises(BL.LoginFailed):
        BL.login_with_vault(FakeManager(), FakeSessions(),
                            lambda: LoginTab(text_mode="rejected"),
                            service="svc")
    # control: a readable welcome page succeeds
    sess_ok = FakeSessions()
    out_ok = BL.login_with_vault(FakeManager(), sess_ok,
                                 lambda: LoginTab(text_mode="welcome"),
                                 service="svc")
    assert out_ok["ok"] is True
    # control: a genuinely TEXTLESS page (the read succeeds, the
    # page has no text) behaves exactly as before — no markers
    # seen, the login proceeds and cookies are saved
    sess_empty = FakeSessions()
    out_empty = BL.login_with_vault(FakeManager(), sess_empty,
                                    lambda: LoginTab(text_mode="empty"),
                                    service="svc")
    assert out_empty["ok"] is True
    assert sess_empty.jars and sess_empty.jars[0][2] == {"sess": "abc"}
    # CONVERTED (parent amendment): the read fails -> None ->
    # UNVERIFIED — LoginFailed raised BEFORE any cookies are
    # persisted, with no success_text configured (the exact
    # provisioning configuration that used to return ok True)
    tab_bad = LoginTab(text_mode="raise")
    sess_bad = FakeSessions()
    with pytest.raises(BL.LoginFailed, match="could not be read"):
        BL.login_with_vault(FakeManager(), sess_bad,
                            lambda: tab_bad, service="svc")
    assert ("text", 20000) in tab_bad.log  # the read was attempted
    assert sess_bad.jars == []  # no session persisted
    assert _snap() == before


def test_217_change_unreadable_vault_untouched():
    before = _snap()
    # control: a readable confirmation page rotates the vault
    mgr_ok = FakeManager()
    out_ok = _change(PwTab(text_mode="changed"), mgr_ok,
                     FakeSessions(session=_Session()))
    assert out_ok["ok"] is True and out_ok["rotated"] is True
    assert mgr_ok.refreshed == [("svc", "u@example.com", "Fresh123!")]
    # control: a readable REJECTION page fails, vault untouched
    mgr_rej = FakeManager()
    with pytest.raises(BL.LoginFailed):
        _change(PwTab(text_mode="rejected"), mgr_rej,
                FakeSessions(session=_Session()))
    assert mgr_rej.refreshed == [] and mgr_rej.vault.stored == []
    # control: a genuinely TEXTLESS page behaves exactly as
    # before — read succeeds with no markers, rotation proceeds
    mgr_empty = FakeManager()
    out_empty = _change(PwTab(text_mode="empty"), mgr_empty,
                        FakeSessions(session=_Session()))
    assert out_empty["ok"] is True and out_empty["rotated"] is True
    # CONVERTED (parent amendment): the verification read fails
    # -> None -> UNVERIFIED — LoginFailed, and the vault is NEVER
    # updated (the module's own invariant); no cookies refreshed
    mgr = FakeManager()
    sess = FakeSessions(session=_Session())
    tab = PwTab(text_mode="raise")
    with pytest.raises(BL.LoginFailed, match="could not be read"):
        _change(tab, mgr, sess)
    assert ("text", 20000) in tab.log  # verification read attempted
    assert mgr.refreshed == [] and mgr.vault.stored == []
    assert sess.jars == []
    assert _snap() == before


def test_426_url_failure_yields_empty_url():
    before = _snap()
    tab = LoginTab(text_mode="welcome", url_raises=True)
    out = BL.login_with_vault(
        FakeManager(), FakeSessions(), lambda: tab, service="svc",
        config=BL.LoginConfig(success_text="Welcome"))
    assert tab.url_accessed >= 1  # the url read was attempted
    assert out["ok"] is True and out["url"] == ""  # OBSERVED cosmetic
    out2 = BL.login_with_vault(
        FakeManager(), FakeSessions(),
        lambda: LoginTab(text_mode="welcome"), service="svc",
        config=BL.LoginConfig(success_text="Welcome"))
    assert out2["url"] == "https://svc.test/app"  # control
    assert _snap() == before


def test_456_login_close_failure_result_stands():
    before = _snap()
    tab = LoginTab(text_mode="welcome", close_raises=True)
    out = BL.login_with_vault(FakeManager(), FakeSessions(),
                              lambda: tab, service="svc")
    assert ("close",) in tab.log  # close was attempted
    assert out["ok"] is True
    assert _snap() == before


def test_715_change_close_failure_result_stands():
    before = _snap()
    mgr = FakeManager()
    sess = FakeSessions(session=_Session())
    tab = PwTab(text_mode="changed", close_raises=True)
    out = _change(tab, mgr, sess, success_text="password was changed")
    assert ("close",) in tab.log  # close was attempted
    assert out["ok"] is True and out["rotated"] is True
    assert _snap() == before
