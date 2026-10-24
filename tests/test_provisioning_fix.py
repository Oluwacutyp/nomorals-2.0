"""Pins for the 2026-10-23 provisioning fix (owner-flagged).

The shipped flow violated the owner's rules three ways (his
screenshot): ``/provision desk`` and ``/provision odds_api``
hard-failed with "I don't know how to provision …" (exact-string key
gate), and the odds-api creation stopped at a manual_step and handed
the OWNER a 5-step chore list ("Create the account yourself: 1…5")
instead of Devon driving the signup.

Pinned here:
* forgiving service resolution — case/underscores/hyphens/spaces are
  equivalent, and ``desk`` resolves to the desk bundle (odds-api +
  football-data); unknown services still fail honestly;
* Devon ATTEMPTS the automated signup for services with a registered
  signup flow — a mocked automatable signup completes end-to-end
  with NO manual checkpoint; a verification wall or hard wall raises
  exactly ONE single-action checkpoint; a page that doesn't match
  the flow fails loudly at the named signup step (manual_step as the
  FIRST response is gone);
* identity behaviour (owner's ruling, re-confirmed 2026-10-23): a
  recorded owner identity is used preferentially; with none recorded
  the drive proceeds under the service's self-generated IdentityBank
  persona and the reply shows the pick — never presented as the
  owner's identity.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nomorals.accounts.creator import AccountCreator, CreatedAccount
from nomorals.accounts.identity_bank import IdentityBank
from nomorals.accounts.provisioning import (
    ProvisionDriver,
    ProvisionOutcome,
    get_provision_bundle,
    get_provision_service,
    resolve_provision_services,
)
from nomorals.accounts.vault import CredentialVault

GEN_EMAIL = "gen.persona@mailinator.com"


def _vault(tmp: str) -> CredentialVault:
    from nomorals.storage.db import Database

    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()  # kv_store + friends: the creator/provisioning organs need them
    return CredentialVault(db, master_passphrase="test-pass")


def _fake_email_mint(calls: list):
    async def _mint(self, *, provider="guerrilla", username=None):
        calls.append(provider)
        return CreatedAccount(
            service="email_tempmail", username=username or "genuser",
            password="", email=GEN_EMAIL, status="created")

    return _mint


class FakeOddsTab:
    """Scripted stand-in for the live Odds API pages.

    The get-access signup form asks for an email only; submitting it
    lands on ``after``: "verify" (check-your-email), "created"
    (welcome) or "review" (manual-review wall). The login page is a
    form until submitted; the account page carries the API key.
    """

    KEY = "0123456789abcdef0123456789abcdef"

    def __init__(self, after: str = "verify"):
        self.after = after
        self.url = ""
        self.submitted = False
        self.logged_in = False
        self.fills: dict = {}

    def navigate(self, url):
        self.url = url
        return {"ok": True, "url": url}

    def fill(self, name, value):
        if "#get-access" in self.url:
            if name != "email":
                raise Exception(
                    f"no field {name!r} on the get-access form")
        elif "/login" not in self.url:
            raise Exception(f"no form on {self.url}")
        self.fills[name] = value
        return {"ok": True}

    def click(self, target):
        label = str(target).lower()
        if "#get-access" in self.url and "access" in label:
            self.submitted = True
            return {"ok": True}
        if "/login" in self.url:
            self.logged_in = True
            return {"ok": True}
        raise Exception(f"no clickable {target!r} on {self.url}")

    def submit(self, target=""):
        if "#get-access" in self.url:
            self.submitted = True
            return {"ok": True}
        if "/login" in self.url:
            self.logged_in = True
            return {"ok": True}
        raise Exception(f"no form to submit on {self.url}")

    def text(self, max_chars: int = 20000):
        if "/account/" in self.url:
            body = (f"Your account — API Key: {self.KEY} "
                    "— manage your plan")
        elif "#get-access" in self.url:
            if not self.submitted:
                body = ("Get Access — enter your email address and "
                        "we'll send you an API key")
            elif self.after == "created":
                body = "Welcome! Your account is ready."
            elif self.after == "review":
                body = ("Thanks — your signup is under review. We "
                        "will manually review your account before "
                        "activating it.")
            else:
                body = ("Thanks for signing up! Check your email to "
                        "complete your registration and verify your "
                        "account.")
        elif "/login" in self.url and not self.logged_in:
            body = "Log In — email, password"
        elif self.logged_in:
            body = "The Odds API — you are logged in"
        else:
            body = "The Odds API"
        return {"text": body[:max_chars]}

    def evaluate(self, js):
        return None

    def cookies(self):
        return [{"name": "session", "value": "x"}]

    def check_captcha(self):
        return {"challenges": []}

    def close(self):
        return None


def _creds(vault, service, credential_type):
    return [c for c in vault.list_all(service=service)
            if c.credential_type == credential_type]


def _secret(vault, service, credential_type):
    """The decrypted secret of the first stored cred of a type
    (list_all redacts; get() decrypts)."""
    creds = _creds(vault, service, credential_type)
    if not creds:
        return ""
    return vault.get(service, creds[0].username,
                     mark_used=False).password


def _driver(vault, tab):
    return ProvisionDriver(
        vault,
        open_tab=lambda service: tab,
        http_get=lambda url, headers: (200, '{"data": []}'),
        environ={})


class ResolutionTests(unittest.TestCase):
    """Fix 1: forgiving resolution + the desk bundle."""

    @classmethod
    def setUpClass(cls):
        from nomorals.agents.betdesk.sources import ensure_provisioning

        ensure_provisioning()

    def test_spellings_resolve_to_the_same_service(self):
        for spelling in ("odds_api", "Odds API", "odds-api", "ODDS-API",
                         "the_odds_api", "the-odds-api", "oddsapi",
                         "odds api"):
            svc = get_provision_service(spelling)
            self.assertIsNotNone(svc, spelling)
            self.assertEqual(svc.id, "odds-api", spelling)
        for spelling in ("football_data", "Football Data",
                         "football-data", "FOOTBALL_DATA"):
            svc = get_provision_service(spelling)
            self.assertIsNotNone(svc, spelling)
            self.assertEqual(svc.id, "football-data", spelling)
        self.assertEqual(get_provision_service("HF").id, "huggingface")
        self.assertEqual(get_provision_service("hf").id, "huggingface")

    def test_desk_bundle_resolves_both_services(self):
        ids = [s.id for s in resolve_provision_services("desk")]
        self.assertEqual(ids, ["odds-api", "football-data"])
        ids = [s.id for s in resolve_provision_services("Desk")]
        self.assertEqual(ids, ["odds-api", "football-data"])
        bundle = get_provision_bundle("desk")
        self.assertIsNotNone(bundle)
        self.assertEqual([s.id for s in bundle],
                         ["odds-api", "football-data"])
        # a single service still resolves to exactly itself
        self.assertEqual(
            [s.id for s in resolve_provision_services("odds_api")],
            ["odds-api"])

    def test_unknown_still_honest(self):
        self.assertIsNone(get_provision_service("nosuchthing"))
        self.assertEqual(resolve_provision_services("nosuchthing"), [])
        self.assertIsNone(get_provision_bundle("nosuchthing"))


class OddsApiDriveTests(unittest.TestCase):
    """Fixes 3+4 on the real desk registry entry (odds-api)."""

    @classmethod
    def setUpClass(cls):
        from nomorals.agents.betdesk.sources import ensure_provisioning

        ensure_provisioning()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-test-provfix-")
        self.vault = _vault(self.tmp)

    def test_signup_attempt_verification_wall_then_provisioned(self):
        # No owner identity recorded: the drive proceeds under the
        # self-generated persona (owner's rule), ATTEMPTS the signup,
        # and the only pause is the service's own email verification
        # — a single action, not a chore list, not manual_step.
        tab = FakeOddsTab(after="verify")
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            first = driver.provision("odds_api")
        self.assertEqual(first.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         first.message)
        self.assertEqual(first.checkpoint["kind"], "email_verify")
        self.assertNotEqual(first.checkpoint["kind"], "manual_step")
        self.assertNotIn("Create the account yourself", first.message)
        self.assertNotIn("\n1.", first.message)
        # the account is shown as a plain fact: persona name +
        # minted address, no provenance lecture (PREMISE CORRECTION
        # 2026-10-23 purge: the "self-generated" narration was the
        # boilerplate the owner ordered off the system)
        personas = IdentityBank(db=self.vault.db, vault=self.vault).list(
            "the-odds-api")
        self.assertEqual(len(personas), 1)
        self.assertEqual(personas[0].email, GEN_EMAIL)
        self.assertNotIn("self-generated", first.message)
        self.assertIn(personas[0].name, first.message)
        self.assertIn(GEN_EMAIL, first.message)
        # the submitted login is already vaulted for the resume
        self.assertTrue(_creds(self.vault, "the-odds-api", "account"))
        self.assertTrue(any(
            r.get("step") == "create"
            and r.get("identity") == "generated"
            for r in driver.audit_trail), driver.audit_trail)
        # verification "clicked" → the next run drives login → key
        second = driver.provision("odds-api")
        self.assertEqual(second.outcome, ProvisionOutcome.PROVISIONED,
                         second.message)
        keys = _creds(self.vault, "the-odds-api", "api_key")
        self.assertEqual(len(keys), 1)
        self.assertEqual(_secret(self.vault, "the-odds-api", "api_key"),
                         FakeOddsTab.KEY)
        self.assertIn("the-odds-api", second.message)
        # ...and the final summary still names the account as a
        # plain fact, lecture-free (2026-10-23 purge)
        self.assertNotIn("self-generated", second.message)
        self.assertIn(GEN_EMAIL, second.message)

    def test_signup_created_end_to_end_no_checkpoint(self):
        # A mocked automatable signup that lands created completes
        # end-to-end with NO manual checkpoint at all.
        tab = FakeOddsTab(after="created")
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("odds_api")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertTrue(result.account_created, result.message)
        self.assertIsNone(result.checkpoint)
        self.assertEqual(
            _secret(self.vault, "the-odds-api", "api_key"),
            FakeOddsTab.KEY)

    def test_owner_identity_preferred_when_recorded(self):
        creator = AccountCreator(self.vault, db=self.vault.db)
        creator.set_owner_identity("Owner Name", "owner@example.com")
        tab = FakeOddsTab(after="created")
        driver = _driver(self.vault, tab)
        result = driver.provision("odds_api")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertIn("Account: Owner Name.", result.message)
        self.assertNotIn("self-generated", result.message)
        # no persona was minted: the recorded identity won
        self.assertEqual(
            IdentityBank(db=self.vault.db, vault=self.vault).list(
                "the-odds-api"), [])
        self.assertTrue(any(
            r.get("step") == "create" and r.get("identity") == "owner"
            for r in driver.audit_trail), driver.audit_trail)

    def test_hard_wall_raises_one_single_action_checkpoint(self):
        tab = FakeOddsTab(after="review")
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("odds_api")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         result.message)
        self.assertEqual(result.checkpoint["kind"], "manual_step")
        # ONE smallest action — the reply names a single step and
        # carries no chore list and no "create it yourself"
        self.assertIn("exactly one step", result.message)
        self.assertNotIn("Create the account yourself", result.message)
        self.assertNotIn("\n1.", result.message)
        # the signup WAS submitted before the wall: its login is
        # vaulted, so a resume continues instead of restarting
        self.assertTrue(_creds(self.vault, "the-odds-api", "account"))

    def test_signup_page_mismatch_fails_at_named_step(self):
        class _BrokenTab(FakeOddsTab):
            def fill(self, name, value):
                raise Exception("no such field")

        driver = _driver(self.vault, _BrokenTab())
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("odds_api")
        self.assertEqual(result.outcome, ProvisionOutcome.PAGE_CHANGED,
                         result.message)
        self.assertEqual(result.step, "signup:fill_email")
        self.assertIsNone(result.checkpoint)


if __name__ == "__main__":
    unittest.main()
