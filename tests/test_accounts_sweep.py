"""Sweep tests for the accounts module upgrade.

Covers the new behavior added in the system-wide accounts sweep:
mail.tm provider, vault passphrase rotation + encrypted backup,
secret-strength + reuse detection, OAuth proactive refresh + rotation
safety + storage_state interop, richer identity-bank personas,
temp-SMS ranking/freshness, human-like typing, and the styled
renderers.
"""

import json
import random
import time
import urllib.error

import pytest

from nomorals.storage.db import Database

from nomorals.accounts.vault import CredentialVault
from nomorals.accounts.manager import (
    AccountManager,
    estimate_secret_strength,
)
from nomorals.accounts.sessions import (
    OAuthToken,
    Session,
    SessionInvalid,
    SessionManager,
    TokenRefreshError,
)
from nomorals.accounts.temp_mail import (
    CASCADE_PROVIDERS,
    MailMessage,
    MailTmProvider,
    PROVIDERS,
    TempAddress,
    grab_address_cascade,
)
from nomorals.accounts.temp_sms import (
    SmsMessage,
    TempNumber,
    TempSmsProvider,
)


# ---------------------------------------------------------------- fixtures

@pytest.fixture()
def vault():
    return CredentialVault(Database(":memory:"),
                           master_passphrase="test-pass")


@pytest.fixture()
def manager(vault):
    return AccountManager(vault)


@pytest.fixture()
def sessions(vault):
    return SessionManager(vault)


# ---------------------------------------------------------------- strength

class TestEstimateSecretStrength:
    def test_weak_password(self):
        r = estimate_secret_strength("password")
        assert r["label"] == "weak"
        assert r["bits"] < 40

    def test_strong_generated(self):
        from nomorals.accounts.creator import generate_password
        r = estimate_secret_strength(generate_password(24))
        assert r["label"] in ("strong", "very_strong")

    def test_empty(self):
        assert estimate_secret_strength("")["label"] == "weak"

    def test_low_entropy_repetition(self):
        # charset×length alone would rate this highly; Shannon catches it
        assert estimate_secret_strength("a" * 24)["label"] == "weak"


# ------------------------------------------------- vault rotation + backup

class TestVaultRotation:
    def test_change_passphrase_reencrypts(self, vault):
        vault.store("github", "devon-bot", "s3cr3t-value-xyz")
        vault.change_passphrase("brand-new-pass")
        # old passphrase can no longer open the rows
        with pytest.raises(Exception):
            bad = CredentialVault.__new__(CredentialVault)
            # simulate: a fresh vault object with the old passphrase
            fresh = CredentialVault(vault.db,
                                    master_passphrase="test-pass")
            fresh.get("github", "devon-bot")
        # new passphrase works
        fresh = CredentialVault(vault.db,
                                master_passphrase="brand-new-pass")
        assert fresh.get("github", "devon-bot",
                         mark_used=False).password == "s3cr3t-value-xyz"

    def test_change_passphrase_empty_rejected(self, vault):
        with pytest.raises(ValueError):
            vault.change_passphrase("")

    def test_export_import_roundtrip(self, vault, tmp_path):
        vault.store("gmail", "bot@example.com", "pw-one",
                    tags=["email"])
        path = str(tmp_path / "backup.enc")
        vault.export_encrypted(path, "backup-pass")
        assert open(path).read() != ""  # opaque blob, not plaintext
        assert "pw-one" not in open(path).read()

        vault2 = CredentialVault(Database(":memory:"),
                                 master_passphrase="other")
        n = vault2.import_encrypted(path, "backup-pass")
        assert n == 1
        assert vault2.get("gmail", "bot@example.com",
                          mark_used=False).password == "pw-one"

    def test_import_wrong_passphrase(self, vault, tmp_path):
        vault.store("x", "y", "z")
        path = str(tmp_path / "b.enc")
        vault.export_encrypted(path, "right")
        vault2 = CredentialVault(Database(":memory:"),
                                 master_passphrase="other")
        with pytest.raises(Exception):
            vault2.import_encrypted(path, "wrong")


# ------------------------------------------------- reused secrets

class TestReusedSecrets:
    def test_detects_reuse(self, manager):
        manager.vault.store("github", "bot", "same-secret-123")
        manager.vault.store("gitlab", "bot", "same-secret-123")
        manager.vault.store("gmail", "bot", "unique-secret-999")
        groups = manager.reused_secrets()
        assert len(groups) == 1
        assert groups[0]["count"] == 2
        svcs = {a["service"] for a in groups[0]["accounts"]}
        assert svcs == {"github", "gitlab"}
        # the secret itself never appears
        blob = json.dumps(groups)
        assert "same-secret-123" not in blob

    def test_health_flags_weak_and_reused(self, manager):
        manager.vault.store("a", "u1", "123456")          # weak
        manager.vault.store("b", "u2", "shared-thing-1")  # reused
        manager.vault.store("c", "u3", "shared-thing-1")  # reused
        issues = manager.health_check()
        types = {i["type"] for i in issues}
        assert "weak_secret" in types
        assert "reused_secret" in types


# ------------------------------------------------- account board rendering

class TestAccountBoard:
    def test_board_renders(self, manager):
        manager.vault.store("github", "devon-bot", "x" * 20)
        manager.vault.store("gmail", "bot@example.com", "y" * 20,
                            expires_at=time.time() - 10)
        board = manager.render_account_board()
        assert "🔐 VAULT" in board
        assert "github" in board and "gmail" in board
        assert "🔴" in board  # expired gmail
        assert "🟢" in board  # active github
        assert "x" * 20 not in board  # no secrets in output

    def test_board_empty(self, manager):
        assert "empty" in manager.render_account_board()


# ------------------------------------------------- OAuth upgrades

class TestOAuthUpgrades:
    def test_needs_refresh_buffer(self):
        tok = OAuthToken(access_token="a", expires_at=time.time() + 30)
        assert not tok.is_expired()
        assert tok.needs_refresh(skew_s=60)
        assert not tok.needs_refresh(skew_s=10)

    def test_seconds_until_expiry(self):
        tok = OAuthToken(access_token="a",
                         expires_at=time.time() + 100)
        assert 90 < tok.seconds_until_expiry() <= 100
        assert OAuthToken(access_token="a").seconds_until_expiry() is None

    def test_refresh_rotation_safe(self, sessions, monkeypatch):
        # response omits refresh_token -> old one must be kept
        body = json.dumps({"access_token": "new-at",
                           "expires_in": 3600,
                           "token_type": "Bearer"}).encode()

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return body

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())
        tok = sessions.refresh_oauth_token("svc", "u", "old-rt",
                                           "cid", "cs",
                                           "https://x/token")
        assert tok.access_token == "new-at"
        assert tok.refresh_token == "old-rt"  # rotation-safe
        assert tok.expires_at <= time.time() + 3600  # skew applied

    def test_refresh_http_error_typed(self, sessions, monkeypatch):
        def boom(*a, **k):
            raise urllib.error.HTTPError("https://x", 400, "bad",
                                         {}, None)
        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(TokenRefreshError) as ei:
            sessions.refresh_oauth_token("svc", "u", "rt", "cid", "cs",
                                         "https://x/token")
        assert ei.value.recoverable is False
        assert isinstance(ei.value, SessionInvalid)

    def test_refresh_retries_transient(self, sessions, monkeypatch):
        calls = {"n": 0}
        body = json.dumps({"access_token": "at",
                            "expires_in": 60}).encode()

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return body

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError("boom")
            return Resp()

        monkeypatch.setattr("urllib.request.urlopen", flaky)
        tok = sessions.refresh_oauth_token("svc", "u", "rt", "cid", "cs",
                                           "https://x/token")
        assert tok.access_token == "at"
        assert calls["n"] == 2


# ------------------------------------------------- storage_state interop

class TestStorageState:
    STATE = {
        "cookies": [
            {"name": "sessionid", "value": "abc123", "domain": "x.com"},
            {"name": "pref", "value": "dark", "domain": "x.com"},
        ],
        "origins": [
            {"origin": "https://x.com",
             "localStorage": [{"name": "k", "value": "v"}]},
        ],
    }

    def test_roundtrip(self):
        s = Session.from_storage_state(self.STATE, service="svc",
                                       username="u")
        assert s.cookies == {"sessionid": "abc123", "pref": "dark"}
        assert s.metadata["local_storage"]["https://x.com"] == [
            {"name": "k", "value": "v"}]
        back = s.to_storage_state()
        s2 = Session.from_storage_state(back, service="svc",
                                        username="u")
        assert s2.cookies == s.cookies

    def test_manager_import(self, sessions):
        s = sessions.import_storage_state(self.STATE, service="gh",
                                          username="bot")
        assert s.cookies["sessionid"] == "abc123"
        peeked = sessions.peek_session("gh", "bot")
        assert peeked is not None
        assert peeked.cookies["pref"] == "dark"

    def test_health_expiring_soon(self):
        tok = OAuthToken(access_token="a",
                         expires_at=time.time() + 300)
        s = Session(service="x", username="y", oauth_token=tok,
                    cookies={"c": "1"})
        assert s.health_report()["status"] == "expiring_soon"


# ------------------------------------------------- mail.tm provider

class TestMailTmProvider:
    def test_registered_first_in_cascade(self):
        assert PROVIDERS["mailtm"] is MailTmProvider
        assert CASCADE_PROVIDERS[0] == "mailtm"

    def test_grab_flow_mocked(self, monkeypatch):
        import urllib.request
        calls = []

        def fake_fetch_json(url, timeout=20, headers=None):
            if url.endswith("/domains"):
                return {"hydra:member": [{"domain": "mail.tm"}]}
            if url.endswith("/messages/1"):
                return {"id": "1", "text": "code 482910",
                        "from": {"address": "n@x.com", "name": "X"},
                        "subject": "verify"}
            if url.endswith("/messages"):
                return {"hydra:member": [
                    {"id": "1", "from": {"address": "n@x.com"},
                     "subject": "verify", "intro": "code",
                     "createdAt": "now"}]}
            raise AssertionError(url)

        class PostResp:
            def __init__(self, status, payload):
                self.status = status
                self._p = payload
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(self._p).encode()

        def fake_urlopen(req, timeout=20):
            url = req.full_url
            calls.append(url)
            if url.endswith("/accounts"):
                return PostResp(201, {})
            if url.endswith("/token"):
                return PostResp(200, {"token": "jwt-123"})
            raise AssertionError(url)

        monkeypatch.setattr("nomorals.accounts.temp_mail._fetch_json",
                            fake_fetch_json)
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        prov = MailTmProvider()
        addr = prov.grab_address()
        assert addr.address.endswith("@mail.tm")
        assert addr.token == "jwt-123"
        msgs = prov.get_messages(addr)
        assert len(msgs) == 1 and msgs[0].id == "1"
        full = prov.read_message(addr, msgs[0])
        assert full.code == "482910"

    def test_cascade_skips_failed_probe(self, monkeypatch):
        class Dead(MailTmProvider):
            name = "dead"
            def probe(self): return False
            def grab_address(self):
                raise AssertionError("should be skipped")

        monkeypatch.setitem(PROVIDERS, "dead", Dead)
        monkeypatch.setattr("nomorals.accounts.temp_mail._fetch_json",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("net down")))
        with pytest.raises(RuntimeError):
            grab_address_cascade(providers=["dead"])


# ------------------------------------------------- temp SMS upgrades

class TestTempSmsUpgrades:
    def test_rank_numbers(self):
        prov = TempSmsProvider()
        stale = TempNumber(number="", masked="+1***", online=True,
                           freshness=0.1)
        good = TempNumber(number="+15551234567", masked="+1555***",
                          online=True, freshness=0.9)
        off = TempNumber(number="+15557654321", online=False,
                         freshness=1.0)
        ranked = prov.rank_numbers([stale, off, good])
        assert ranked[0] is good  # online + resolved wins

    def test_age_seconds(self):
        assert SmsMessage(received="23 minutes ago").age_seconds() == 1380
        assert SmsMessage(received="just now").age_seconds() == 0.0
        assert SmsMessage(received="2 hours ago").age_seconds() == 7200
        assert SmsMessage(received="???").age_seconds() is None

    def test_wait_for_code_seen_store(self):
        class FakeProv(TempSmsProvider):
            name = "fake"
            def get_messages(self, number, limit=20):
                return [SmsMessage(sender="X", body="code 111222")]
        prov = FakeProv()
        store: dict = {}
        num = TempNumber(number="+1000", inbox_id="i1")
        code = prov.wait_for_code(num, timeout=2, poll_every=0.05,
                                 seen_store=store)
        assert code == "111222"
        # second wait on the same store sees the message as already seen
        code2 = prov.wait_for_code(num, timeout=0.3, poll_every=0.05,
                                   seen_store=store)
        assert code2 == ""


# ------------------------------------------------- identity bank upgrades

class TestIdentityBank:
    def test_mint_locales(self):
        from nomorals.accounts.identity_bank import (
            IdentityBank, LOCALE_FIRST_NAMES)
        bank = IdentityBank()
        for locale, pool in LOCALE_FIRST_NAMES.items():
            p = bank.mint("svc-" + locale, rng=random.Random(42),
                          locale=locale)
            assert p.first_name in pool
            assert p.locale == locale
            assert p.address and "Lagos" in p.address
            assert p.phone.startswith("+234")
            assert p.bio
            assert p.handle_variants
            # disposable-surname policy intact
            from nomorals.accounts.identity_bank import DISPOSABLE_SURNAMES
            assert p.last_name in DISPOSABLE_SURNAMES

    def test_mint_seeded_reproducible(self):
        from nomorals.accounts.identity_bank import IdentityBank
        bank = IdentityBank()
        a = bank.mint("s", rng=random.Random(7))
        b = bank.mint("s", rng=random.Random(7))
        assert a.first_name == b.first_name
        assert a.address == b.address
        assert a.phone == b.phone

    def test_persona_card_rich(self):
        from nomorals.accounts.identity_bank import (
            IdentityBank, render_persona_card)
        bank = IdentityBank()
        p = bank.mint("github", rng=random.Random(1))
        card = render_persona_card(p, "github")
        assert "SIGNUP IDENTITY DRAFT" in card
        assert p.phone in card and p.address in card
        assert "DISPOSABLE" in card

    def test_persona_roundtrip_new_fields(self):
        from nomorals.accounts.identity_bank import Persona
        from nomorals.accounts.identity_bank import IdentityBank
        p = IdentityBank().mint("s", rng=random.Random(3))
        d = p.to_dict()
        assert d["phone"] == p.phone and d["locale"] == p.locale
        p2 = Persona.from_dict(d)
        assert p2.address == p.address
        # old dicts without new fields still load
        p3 = Persona.from_dict({"id": "x", "service": "s",
                                "first_name": "A", "last_name": "Trial"})
        assert p3.phone == "" and p3.locale == "ng_yoruba"


# ------------------------------------------------- human typing

class TestHumanTyping:
    def test_type_like_human_uses_tab_type(self):
        from nomorals.accounts.browser_login import type_like_human
        seen = []

        class Tab:
            def focus(self, field): seen.append(("focus", field))
            def type(self, field, ch): seen.append(("type", field, ch))
            def fill(self, field, value):
                raise AssertionError("fill should not be used")

        import random as _r
        type_like_human(Tab(), "username", "ab",
                        rng=_r.Random(0))
        assert ("focus", "username") in seen
        typed = "".join(t[2] for t in seen if t[0] == "type")
        assert typed == "ab"

    def test_type_like_human_falls_back_to_fill(self):
        from nomorals.accounts.browser_login import type_like_human
        filled = {}

        class Tab:
            def focus(self, field): pass
            def fill(self, field, value): filled[field] = value
        type_like_human(Tab(), "pw", "secret")
        assert filled == {"pw": "secret"}

    def test_login_config_flag(self):
        from nomorals.accounts.browser_login import (
            LoginConfig, _config_from_metadata)
        cfg = _config_from_metadata({}, LoginConfig(human_typing=True))
        assert cfg.human_typing is True
        cfg2 = _config_from_metadata({"human_typing": True}, None)
        assert cfg2.human_typing is True


# ------------------------------------------------- health board

class TestHealthBoard:
    def test_render(self, manager, sessions):
        from nomorals.accounts.health import (
            check_all_health, render_health_board)
        manager.vault.store("github", "bot", "x" * 30)
        sweep = check_all_health(manager, sessions)
        board = render_health_board(
            [__import__("nomorals.accounts.health", fromlist=["AccountHealth"])
             .AccountHealth(**{k: v for k, v in a.items()
                               if k in ("service", "username", "status",
                                        "issues")})
             for a in sweep["accounts"]])
        assert "ACCOUNT HEALTH" in board
        assert "github/bot" in board

    def test_render_dicts(self):
        from nomorals.accounts.health import render_health_board
        board = render_health_board([
            {"service": "x", "username": "u", "status": "locked",
             "issues": ["account suspended", "other"]},
        ])
        assert "🔒" in board and "(+1 more)" in board


# ------------------------------------------------- signup progress

class TestSignupProgress:
    def test_progress_ladder(self):
        from nomorals.accounts.signup_driver import (
            SignupAttempt, SignupStage, WallKind, render_attempt_progress)
        a = SignupAttempt(id="attempt-1", service="github",
                          persona_name="X Trial", persona_id="p1",
                          stage=SignupStage.VERIFIED)
        out = render_attempt_progress(a)
        assert "🧭 SIGNUP" in out and "github" in out
        assert "👉 ✅ verified" in out

    def test_progress_stopped(self):
        from nomorals.accounts.signup_driver import (
            SignupAttempt, SignupStage, WallKind, render_attempt_progress)
        a = SignupAttempt(id="attempt-2", service="x",
                          persona_name="X Trial", persona_id="p1",
                          stage=SignupStage.STOPPED_AT_WALL,
                          wall_kind=WallKind.CAPTCHA,
                          wall_detail="recaptcha v2")
        out = render_attempt_progress(a)
        assert "🧱" in out and "CAPTCHA" in out
