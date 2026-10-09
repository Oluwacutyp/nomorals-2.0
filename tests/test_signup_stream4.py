"""Tests for Stream 4: identity bank, signup driver, challenge cascade.

Offline by design — the browser page, temp-mail inbox, temp-SMS
providers and the CAPTCHA solver are all faked.  What the tests prove:

* identity bank: consistent reusable personas, Nigerian pool,
  vault-side persistence, disposable marking;
* confirmation gate: NEVER proceeds without explicit confirmation;
* wall classification: captcha / sms / review / id / age / rate-limit;
* stage machine: legal transitions only, terminal states stick;
* challenge-solving cascade: solver → temp-mail → temp-SMS → owner
  ping as the LAST resort (never the first);
* never-use-real-contact-details invariant: the owner's real email /
  phone fail the drive closed, temp numbers matching them are skipped.
"""

from __future__ import annotations

import asyncio
import random
import unittest
from unittest import mock

from nomorals.accounts import (
    AccountCheckpointPending,
    ConfirmationGate,
    ConfirmationRequired,
    CredentialVault,
    IdentityBank,
    Persona,
    SignupAttemptStore,
    SignupDriver,
    SignupStage,
    WallKind,
    classify_wall,
    render_persona_card,
)
from nomorals.accounts.creator import AccountCreator, CreatedAccount
from nomorals.core.errors import NoMoralsError
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"),
                           master_passphrase="test-pass")


def _persona(**kw) -> Persona:
    bank = IdentityBank(db=Database(":memory:"))
    p = bank.mint("testsvc", rng=random.Random(42))
    for k, v in kw.items():
        setattr(p, k, v)
    return p


class FakePage:
    """Duck-typed page driver. Flip ``text_value``/``html_value`` to
    script walls; ``clear_wall_on_code`` simulates a passed SMS gate."""

    def __init__(self, *, clear_wall_on_code: bool = False):
        self.text_value = ""
        self.html_value = ""
        self.captcha = {"count": 0, "challenges": []}
        self.fills: list[tuple[str, str]] = []
        self.clicks: list[str] = []
        self.navigated: list[str] = []
        self.submitted = 0
        self.fail_fields: set[str] = set()
        self.clear_wall_on_code = clear_wall_on_code

    def navigate(self, url: str):
        self.navigated.append(url)
        return {}

    def fill(self, name: str, value: str):
        if name in self.fail_fields:
            raise RuntimeError(f"no field {name!r}")
        self.fills.append((name, value))
        if self.clear_wall_on_code and "code" in name:
            self.text_value = ("Welcome! Check your email to verify "
                               "your account.")
        return {"ok": True}

    def click(self, target: str):
        self.clicks.append(target)
        return {}

    def submit(self, target: str = ""):
        self.submitted += 1
        return {}

    def text(self, max_chars: int = 40000):
        return self.text_value

    def html(self):
        return self.html_value

    def check_captcha(self, *, fetch_bytes: bool = False):
        return self.captcha


async def _no_sleep(delay):
    return None


def _driver(db=None, creator=None, **kw):
    db = db or Database(":memory:")
    vault = _vault()
    # note: vault is separate from db here on purpose — the attempt
    # store needs a db; the creator needs a vault.
    creator = creator or AccountCreator(vault, db=db)
    attempts = SignupAttemptStore(db)
    notifies: list[tuple[str, str, str]] = []
    kw.setdefault("notify", lambda t, b, aid: notifies.append((t, b, aid)))
    driver = SignupDriver(creator, attempts, **kw)
    return driver, attempts, notifies


def _email_account(email="tmp123@tempmail.plus"):
    return CreatedAccount(service="email_tempmail", username=email,
                          password="", email=email, status="created",
                          notes="fake")


# ── identity bank ────────────────────────────────────────────────


class IdentityBankTests(unittest.TestCase):
    def test_mint_is_disposable_with_username_variants(self):
        bank = IdentityBank(db=Database(":memory:"))
        p = bank.mint("github", rng=random.Random(7))
        self.assertTrue(p.disposable)
        self.assertTrue(p.name)
        self.assertGreaterEqual(len(p.username_variants), 3)
        self.assertEqual(len(set(p.username_variants)),
                         len(p.username_variants))
        # adult dob
        year = int(p.dob.split("-")[0])
        import datetime
        age = datetime.date.today().year - year
        self.assertGreaterEqual(age, 21)
        self.assertLessEqual(age, 45)

    def test_nigerian_pool_used(self):
        from nomorals.accounts import NIGERIAN_FIRST_NAMES
        bank = IdentityBank(db=Database(":memory:"))
        rng = random.Random(1234)
        firsts = {bank.mint("s", rng=rng).first_name for _ in range(30)}
        self.assertTrue(firsts & set(NIGERIAN_FIRST_NAMES),
                        f"no Nigerian names in {sorted(firsts)}")

    def test_get_or_mint_reuses_persona(self):
        db = Database(":memory:")
        bank = IdentityBank(db=db)
        first = bank.get_or_mint("github", rng=random.Random(1))
        second = bank.get_or_mint("github", rng=random.Random(2))
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.name, second.name)
        # a different service gets its own persona
        other = bank.get_or_mint("x", rng=random.Random(1))
        self.assertNotEqual(first.id, other.id)

    def test_vault_side_persistence(self):
        vault = _vault()
        bank = IdentityBank(db=Database(":memory:"), vault=vault)
        p = bank.mint("github", rng=random.Random(9))
        bank2 = IdentityBank(db=Database(":memory:"), vault=vault)
        same = bank2.get_or_mint("github", rng=random.Random(10))
        self.assertEqual(p.id, same.id)
        self.assertEqual(bank2.get(p.id).name, p.name)

    def test_card_marks_disposable_and_warns(self):
        p = _persona()
        card = render_persona_card(p, "github")
        self.assertIn("DISPOSABLE", card)
        self.assertIn("confirm", card.lower())
        self.assertIn(p.name, card)


# ── confirmation gate ────────────────────────────────────────────


class ConfirmationGateTests(unittest.TestCase):
    def test_explicit_yes_confirms_and_consumes_once(self):
        gate = ConfirmationGate()
        token = gate.request(subject="s", card="c",
                             payload={"a": 1})
        self.assertTrue(gate.confirm(token, "yes"))
        self.assertEqual(gate.consume(token), {"a": 1})
        # one-shot: second consume never returns the payload
        self.assertIsNone(gate.consume(token))

    def test_no_or_ambiguous_never_confirms(self):
        gate = ConfirmationGate()
        for reply in ("no", "maybe", "", "   ", "yess", "confirm later"):
            token = gate.request(subject="s", card="c")
            self.assertFalse(gate.confirm(token, reply), reply)
            self.assertIsNone(gate.consume(token))

    def test_consume_without_confirm_returns_none(self):
        gate = ConfirmationGate()
        token = gate.request(subject="s", card="c",
                             payload={"a": 1})
        self.assertIsNone(gate.consume(token))

    def test_unknown_and_expired_tokens(self):
        gate = ConfirmationGate(ttl_seconds=0.01)
        token = gate.request(subject="s", card="c")
        import time
        time.sleep(0.02)
        self.assertFalse(gate.confirm(token, "yes"))
        self.assertIsNone(gate.consume(token))
        self.assertFalse(gate.confirm("nope", "yes"))

    def test_driver_refuses_without_confirmation(self):
        driver, _, _ = _driver()
        with self.assertRaises(ConfirmationRequired):
            asyncio.run(driver.adrive(service="github", persona=_persona(),
                                      confirmed=False,
                                      page=FakePage()))

    def test_driver_refuses_non_disposable_persona(self):
        driver, _, _ = _driver()
        p = _persona(disposable=False)
        with self.assertRaises(NoMoralsError):
            asyncio.run(driver.adrive(service="github", persona=p,
                                      confirmed=True, page=FakePage()))


# ── wall classification ──────────────────────────────────────────


class ClassifyWallTests(unittest.TestCase):
    def test_captcha_html_marker(self):
        wall, ev = classify_wall(
            page_html='<div class="g-recaptcha" data-sitekey="x"></div>')
        self.assertEqual(wall, WallKind.CAPTCHA)
        self.assertIn("g-recaptcha", ev)

    def test_captcha_text(self):
        wall, _ = classify_wall(page_text="Please verify you are human")
        self.assertEqual(wall, WallKind.CAPTCHA)

    def test_sms_phone_gate(self):
        wall, _ = classify_wall(
            page_text="Verify your phone number — we will text you a code")
        self.assertEqual(wall, WallKind.SMS_PHONE)

    def test_manual_review(self):
        wall, _ = classify_wall(
            page_text="Your application is under review. We'll be in touch.")
        self.assertEqual(wall, WallKind.MANUAL_REVIEW)

    def test_real_id(self):
        wall, _ = classify_wall(
            page_text="Please upload a government-issued ID to continue")
        self.assertEqual(wall, WallKind.REAL_ID)

    def test_age_gate(self):
        wall, _ = classify_wall(
            page_text="You must be 18 years old to use this service")
        self.assertEqual(wall, WallKind.AGE_GATE)

    def test_rate_limit(self):
        wall, _ = classify_wall(
            page_text="Too many attempts. Try again later.")
        self.assertEqual(wall, WallKind.RATE_LIMIT)

    def test_email_verification_is_not_a_wall(self):
        wall, ev = classify_wall(
            page_text="Thanks! Check your inbox to verify your email.")
        self.assertEqual(wall, WallKind.NONE)
        self.assertEqual(ev, "")

    def test_empty_page_no_wall(self):
        self.assertEqual(classify_wall()[0], WallKind.NONE)


# ── stage machine ────────────────────────────────────────────────


class StageMachineTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.store = SignupAttemptStore(self.db)

    def test_happy_path_transitions(self):
        a = self.store.create(service="github", persona=_persona(),
                              signup_url="https://github.com/signup")
        self.assertEqual(a.stage, SignupStage.STARTED)
        for nxt in (SignupStage.FORM_FILLED, SignupStage.EMAIL_SENT,
                    SignupStage.VERIFIED, SignupStage.COMPLETE):
            a = self.store.transition(a.id, nxt)
            self.assertEqual(a.stage, nxt)

    def test_illegal_transitions_raise(self):
        a = self.store.create(service="github", persona=_persona())
        with self.assertRaises(NoMoralsError):
            self.store.transition(a.id, SignupStage.COMPLETE)
        with self.assertRaises(NoMoralsError):
            self.store.transition(a.id, SignupStage.VERIFIED)

    def test_terminal_states_stick(self):
        a = self.store.create(service="github", persona=_persona())
        a = self.store.transition(a.id, SignupStage.FAILED)
        with self.assertRaises(NoMoralsError):
            self.store.transition(a.id, SignupStage.STARTED)
        b = self.store.create(service="x", persona=_persona())
        b = self.store.transition(b.id, SignupStage.STOPPED_AT_WALL)
        with self.assertRaises(NoMoralsError):
            self.store.transition(b.id, SignupStage.FORM_FILLED)

    def test_wall_and_credential_recorded(self):
        a = self.store.create(service="github", persona=_persona())
        self.store.set_wall(a.id, WallKind.SMS_PHONE, "need a code")
        self.store.set_credential(a.id, "cred-1")
        got = self.store.get(a.id)
        self.assertEqual(got.wall_kind, WallKind.SMS_PHONE)
        self.assertEqual(got.credential_ref, "cred-1")

    def test_list_filters_service(self):
        self.store.create(service="github", persona=_persona())
        self.store.create(service="x", persona=_persona())
        self.assertEqual(len(self.store.list(service="github")), 1)
        self.assertEqual(len(self.store.list()), 2)


# ── challenge-solving cascade ────────────────────────────────────


class CascadeTests(unittest.TestCase):
    def _patched_driver(self, page, **kw):
        driver, attempts, notifies = _driver(**kw)

        async def fake_email(*a, **k):
            return _email_account()

        driver.creator.create_email_account = fake_email
        # inbox immediately yields a verification link
        driver._read_inbox = lambda cred, provider: [{
            "subject": "verify your account",
            "text": ("welcome! confirm: "
                     "https://example.com/verify?token=abc123"),
            "html": "",
        }]
        return driver, attempts, notifies

    def test_full_drive_completes_without_owner_ping(self):
        page = FakePage()
        driver, attempts, notifies = self._patched_driver(page)
        persona = _persona()
        with mock.patch("asyncio.sleep", new=_no_sleep):
            attempt = asyncio.run(driver.adrive(
                service="github", persona=persona, confirmed=True,
                page=page))
        self.assertEqual(attempt.stage, SignupStage.COMPLETE)
        self.assertTrue(attempt.credential_ref)
        # owner pinged for ACTION only when stuck — a completion report
        # is fine, a "needs you" is not.
        self.assertFalse(any("needs you" in t for t, _, _ in notifies))
        self.assertTrue(any("signup complete" in t for t, _, _ in notifies))
        self.assertTrue(page.navigated)  # followed the verify link
        self.assertIn("https://example.com/verify?token=abc123",
                      page.navigated)
        # form was actually filled
        filled = dict(page.fills)
        self.assertIn(persona.name, filled.values())

    def test_captcha_solved_by_solver_continues(self):
        page = FakePage()
        page.html_value = '<div class="g-recaptcha"></div>'
        driver, attempts, notifies = self._patched_driver(page)
        driver.creator.attempt_captcha_solve = lambda **k: "tok123"
        with mock.patch("asyncio.sleep", new=_no_sleep):
            attempt = asyncio.run(driver.adrive(
                service="github", persona=_persona(), confirmed=True,
                page=page))
        self.assertEqual(attempt.stage, SignupStage.COMPLETE)
        # solver success is reported, but no owner ACTION requested
        self.assertFalse(any("needs you" in t for t, _, _ in notifies))

    def test_captcha_unsolvable_pings_owner_last_resort(self):
        page = FakePage()
        page.html_value = '<div class="g-recaptcha"></div>'
        driver, attempts, notifies = self._patched_driver(page)

        def fake_solve(**k):
            cp = driver.creator.checkpoints.create(
                __import__(
                    "nomorals.accounts.creator",
                    fromlist=["CheckpointKind"]).CheckpointKind.CAPTCHA,
                "solve it", "do the captcha", service="github")
            raise AccountCheckpointPending(cp)

        driver.creator.attempt_captcha_solve = fake_solve
        with mock.patch("asyncio.sleep", new=_no_sleep):
            with self.assertRaises(AccountCheckpointPending):
                asyncio.run(driver.adrive(
                    service="github", persona=_persona(), confirmed=True,
                    page=page))
        attempt = attempts.list(service="github")[0]
        self.assertEqual(attempt.stage, SignupStage.STOPPED_AT_WALL)
        self.assertEqual(attempt.wall_kind, WallKind.CAPTCHA)
        self.assertTrue(any("needs you" in t for t, _, _ in notifies),
                        notifies)
        # exact status: what's done + what's needed
        body = " ".join(b for _, b, _ in notifies)
        self.assertIn("attempt recorded", body)
        self.assertIn("solve the CAPTCHA", body)

    def test_sms_wall_solved_via_temp_number_automatically(self):
        page = FakePage(clear_wall_on_code=True)
        page.text_value = ("Verify your phone number. "
                           "Enter your phone number below.")
        driver, attempts, notifies = self._patched_driver(page)
        driver.creator.get_temp_number_cascade = lambda country="us", \
                providers=None: {
            "status": "ok", "number": "+15551234567",
            "masked": "+1555****", "country": "us",
            "provider": "simcodes", "inbox_id": "1"}
        driver.creator.poll_sms_code = lambda info, **k: "654321"
        with mock.patch("asyncio.sleep", new=_no_sleep):
            attempt = asyncio.run(driver.adrive(
                service="github", persona=_persona(), confirmed=True,
                page=page))
        self.assertEqual(attempt.stage, SignupStage.COMPLETE)
        filled = dict(page.fills)
        self.assertIn("+15551234567", filled.values())
        self.assertIn("654321", filled.values())
        self.assertFalse(any("needs you" in t for t, _, _ in notifies))

    def test_sms_cascade_exhausted_pings_owner(self):
        page = FakePage()
        page.text_value = "Verify your phone number."
        driver, attempts, notifies = self._patched_driver(page)
        driver.creator.get_temp_number_cascade = lambda country="us", \
                providers=None: {"status": "failed",
                                 "notes": "all temp-sms sources exhausted"}
        with mock.patch("asyncio.sleep", new=_no_sleep):
            with self.assertRaises(AccountCheckpointPending):
                asyncio.run(driver.adrive(
                    service="github", persona=_persona(), confirmed=True,
                    page=page))
        attempt = attempts.list(service="github")[0]
        self.assertEqual(attempt.stage, SignupStage.STOPPED_AT_WALL)
        self.assertEqual(attempt.wall_kind, WallKind.SMS_PHONE)
        body = " ".join(b for _, b, _ in notifies)
        self.assertIn("temp-number", body)
        self.assertIn("What I need from you", body)

    def test_rate_limit_retried_then_handed_to_owner(self):
        page = FakePage()
        page.text_value = "Too many attempts. Try again later."
        driver, attempts, notifies = self._patched_driver(page)
        with mock.patch("asyncio.sleep", new=_no_sleep):
            with self.assertRaises(AccountCheckpointPending):
                asyncio.run(driver.adrive(
                    service="github", persona=_persona(), confirmed=True,
                    page=page, rate_limit_waits=(0, 0)))
        attempt = attempts.list(service="github")[0]
        self.assertEqual(attempt.wall_kind, WallKind.RATE_LIMIT)
        self.assertTrue(any("rate limit" in t for t, _, _ in notifies))

    def test_rate_limit_cleared_continues(self):
        page = FakePage()
        page.text_value = "Too many attempts. Try again later."
        orig_navigate = page.navigate

        def navigate(url):
            orig_navigate(url)
            page.text_value = "Create your account"  # limit cleared

        page.navigate = navigate
        driver, attempts, notifies = self._patched_driver(page)
        with mock.patch("asyncio.sleep", new=_no_sleep):
            attempt = asyncio.run(driver.adrive(
                service="github", persona=_persona(), confirmed=True,
                page=page, rate_limit_waits=(0,)))
        self.assertEqual(attempt.stage, SignupStage.COMPLETE)

    def test_unknown_service_hands_to_owner_without_guessing(self):
        page = FakePage()
        driver, attempts, notifies = self._patched_driver(page)
        with mock.patch("asyncio.sleep", new=_no_sleep):
            with self.assertRaises(AccountCheckpointPending):
                asyncio.run(driver.adrive(
                    service="someservice", persona=_persona(),
                    confirmed=True, page=page))
        self.assertEqual(page.navigated, [])  # never guessed a URL
        self.assertTrue(any("needs you" in t for t, _, _ in notifies))


# ── never-use-real-contact-details invariant ─────────────────────


class RealContactInvariantTests(unittest.TestCase):
    def test_owner_email_in_persona_fails_closed(self):
        driver, _, _ = _driver()
        persona = _persona()
        persona.email = "owner@real.com"
        persona.email_provider = "tempmail"
        with self.assertRaises(NoMoralsError):
            asyncio.run(driver.adrive(
                service="github", persona=persona, confirmed=True,
                page=FakePage(),
                owner_contacts={"emails": ["owner@real.com"],
                                "phones": []}))

    def test_freshly_minted_email_checked_against_owner(self):
        driver, _, _ = _driver()

        async def fake_email(*a, **k):
            return _email_account(email="owner@real.com")

        driver.creator.create_email_account = fake_email
        with self.assertRaises(NoMoralsError):
            asyncio.run(driver.adrive(
                service="github", persona=_persona(), confirmed=True,
                page=FakePage(),
                owner_contacts={"emails": ["owner@real.com"],
                                "phones": []}))

    def test_temp_number_matching_owner_phone_is_skipped(self):
        page = FakePage(clear_wall_on_code=True)
        page.text_value = "Verify your phone number."
        driver, attempts, notifies = _driver()

        async def fake_email(*a, **k):
            return _email_account()

        driver.creator.create_email_account = fake_email
        driver._read_inbox = lambda cred, provider: [{
            "subject": "verify", "text": "code 112233", "html": ""}]
        numbers = iter([
            {"status": "ok", "number": "+2348012345678",
             "masked": "+234****", "provider": "simcodes",
             "inbox_id": "1"},
            {"status": "ok", "number": "+15559876543",
             "masked": "+1555****", "provider": "7sim",
             "inbox_id": "http://x/2"},
        ])
        driver.creator.get_temp_number_cascade = lambda country="us", \
            providers=None: next(numbers)
        driver.creator.poll_sms_code = lambda info, **k: "112233"
        with mock.patch("asyncio.sleep", new=_no_sleep):
            attempt = asyncio.run(driver.adrive(
                service="github", persona=_persona(), confirmed=True,
                page=page,
                owner_contacts={"emails": [], "phones": ["+2348012345678"]}))
        self.assertEqual(attempt.stage, SignupStage.COMPLETE)
        filled = dict(page.fills)
        # owner's real number never touched the form
        self.assertNotIn("+2348012345678", filled.values())
        self.assertIn("+15559876543", filled.values())


# ── temp-sms cascade ─────────────────────────────────────────────


class TempSmsCascadeTests(unittest.TestCase):
    def test_cascade_skips_failed_probe(self):
        from nomorals.accounts import temp_sms

        class DeadProvider(temp_sms.TempSmsProvider):
            name = "dead"

            def probe(self):
                return False

            def list_numbers(self, country="us", limit=20):
                raise AssertionError("should be skipped")

        class LiveProvider(temp_sms.TempSmsProvider):
            name = "live"

            def list_numbers(self, country="us", limit=20):
                return [temp_sms.TempNumber(number="+10000000001",
                                            masked="+1000****",
                                            country=country,
                                            provider="live")]

        with mock.patch.dict(temp_sms.PROVIDERS,
                             {"dead": DeadProvider, "live": LiveProvider}):
            out = temp_sms.grab_number_cascade(
                providers=("dead", "live"))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["number"], "+10000000001")

    def test_cascade_all_exhausted_reports(self):
        from nomorals.accounts import temp_sms
        out = temp_sms.grab_number_cascade(providers=("nonexistent",))
        self.assertEqual(out["status"], "failed")
        self.assertIn("notes", out)


if __name__ == "__main__":
    unittest.main()
