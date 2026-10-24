"""Pins for the 2026-10-23 boilerplate purge (owner's order, verbatim:
"Get all this your stupid limitations and boiler plates off my system
in every where").

Trigger: after EC2 updated, ``/provision odds-api`` STILL answered
with the old world — "account creation for The Odds API is paused at
a human step (manual_step): Create the-odds-api account … — finish
it, then run /provision odds-api again and I'll pick up from there"
plus a full "Identity: self-generated — no owner identity is
recorded, so…" lecture paragraph.

Pinned here, using the owner's pasted output as the negative fixture:

* NO driven-path output contains the fixture strings: "paused at a
  human step", "finish it, then run", "Identity: self-generated",
  "self-generated", "manual_step", "Create the account yourself", a
  numbered chore list, or a "run /provision … again" instruction.
* The owner's exact case (no Odds API account, no recorded owner
  identity) ends driven: a created account, or ONE smallest owner
  action at a genuine wall — the account itself reported as a plain
  fact (name + address), never a provenance lecture.
* A service carrying a signup URL but no named signup flow is DRIVEN
  with the generic form flow, never handed the creator's checkpoint
  as a first response.
* The creator's per-service checkpoint instructions are ONE plain
  action each — no numbered lists anywhere in the class.
* Identity BEHAVIOUR is unchanged (owner's own rule): a recorded
  owner identity is preferred; with none recorded the generated
  persona carries the creation and is reused on retries.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nomorals.accounts import provisioning
from nomorals.accounts.creator import (
    AccountCreator,
    CheckpointStore,
    CreatedAccount,
)
from nomorals.accounts.identity_bank import IdentityBank
from nomorals.accounts.provisioning import (
    PROVISION_REGISTRY,
    ProvisionDriver,
    ProvisionOutcome,
    ProvisionService,
    register_provision_service,
)
from nomorals.accounts.vault import CredentialVault

GEN_EMAIL = "gen.persona@mailinator.com"

#: The owner's pasted output, as a negative fixture: none of these
#: may appear in ANY driven-path result message.
FORBIDDEN = (
    "paused at a human step",
    "finish it, then run",
    "Identity: self-generated",
    "self-generated",
    "manual_step",
    "Create the account yourself",
    "create it yourself",
    "run /provision",
    "\n1.",
    "\n2.",
)


def _assert_clean(case: unittest.TestCase, message: str) -> None:
    for fixture in FORBIDDEN:
        case.assertNotIn(fixture, message)


def _vault(tmp: str) -> CredentialVault:
    from nomorals.storage.db import Database

    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    return CredentialVault(db, master_passphrase="test-pass")


def _fake_email_mint(calls: list):
    async def _mint(self, *, provider="guerrilla", username=None):
        calls.append(provider)
        return CreatedAccount(
            service="email_tempmail", username=username or "genuser",
            password="", email=GEN_EMAIL, status="created")

    return _mint


def _driver(vault, tab):
    return ProvisionDriver(
        vault,
        open_tab=lambda service: tab,
        http_get=lambda url, headers: (200, '{"data": []}'),
        environ={})


class OwnerExactCaseTests(unittest.TestCase):
    """The owner's case, end to end: odds-api, nothing recorded."""

    @classmethod
    def setUpClass(cls):
        from nomorals.agents.betdesk.sources import ensure_provisioning

        ensure_provisioning()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-test-purge-")
        self.vault = _vault(self.tmp)

    def _odds_tab(self, after: str):
        from tests.test_provisioning_fix import FakeOddsTab

        return FakeOddsTab(after=after)

    def test_verification_wall_is_one_action_and_plain_fact(self):
        tab = self._odds_tab("verify")
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            first = driver.provision("odds_api")
        self.assertEqual(first.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         first.message)
        self.assertEqual(first.checkpoint["kind"], "email_verify")
        _assert_clean(self, first.message)
        # the wall is the service's own inbox step — and for a
        # generated identity it is not even the owner's step
        self.assertIn("no step for you", first.message)
        # the account is a plain fact: name + address, no lecture
        personas = IdentityBank(db=self.vault.db, vault=self.vault).list(
            "the-odds-api")
        self.assertEqual(len(personas), 1)
        self.assertIn(
            f"Account: {personas[0].name} <{GEN_EMAIL}>.", first.message)

    def test_created_end_to_end_is_clean(self):
        tab = self._odds_tab("created")
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("odds-api")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertTrue(result.account_created, result.message)
        _assert_clean(self, result.message)
        personas = IdentityBank(db=self.vault.db, vault=self.vault).list(
            "the-odds-api")
        self.assertIn(
            f"Account: {personas[0].name} <{GEN_EMAIL}>.",
            result.message)

    def test_hard_wall_is_exactly_one_action(self):
        tab = self._odds_tab("review")
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("odds_api")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         result.message)
        _assert_clean(self, result.message)
        self.assertIn("exactly one step", result.message)
        # the one action ends with Devon continuing, not the owner
        # re-running anything
        self.assertIn("tell me", result.message)

    def test_owner_identity_preferred_and_plainly_reported(self):
        creator = AccountCreator(self.vault, db=self.vault.db)
        creator.set_owner_identity("Owner Name", "owner@example.com")
        tab = self._odds_tab("created")
        driver = _driver(self.vault, tab)
        result = driver.provision("odds_api")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        _assert_clean(self, result.message)
        self.assertIn("Account: Owner Name.", result.message)
        # behaviour unchanged: the recorded identity won, no persona
        self.assertEqual(
            IdentityBank(db=self.vault.db, vault=self.vault).list(
                "the-odds-api"), [])


class PendingRereportTests(unittest.TestCase):
    """A pending creator checkpoint re-reported by a later run:
    same checkpoint, one plain action, no chore list, no lecture."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-test-purge-")
        self.vault = _vault(self.tmp)
        svc = register_provision_service(ProvisionService(
            id="purgeflowless", display="PurgeFlowless",
            vault_service="purgeflowless",
            env_keys=("PURGEFLOWLESS_KEY",),
            login_url="https://purgeflowless.example/login",
            token_url="https://purgeflowless.example/keys",
            validator="purgeflowless_ok", flow="purgeflowless_flow"))
        provisioning._PAGE_FLOWS["purgeflowless_flow"] = (
            lambda tab, s, **kw: "purgeflowless-secret")
        provisioning._VALIDATORS["purgeflowless_ok"] = (
            lambda token, http_get: provisioning.ValidationResult(
                "ok", "fine"))
        self.addCleanup(PROVISION_REGISTRY.pop, "purgeflowless", None)
        self.addCleanup(provisioning._PAGE_FLOWS.pop,
                        "purgeflowless_flow", None)
        self.addCleanup(provisioning._VALIDATORS.pop,
                        "purgeflowless_ok", None)
        self.svc = svc

    def test_pending_rereport_is_clean(self):
        tab = mock.Mock()
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            first = driver.provision("purgeflowless")
            second = driver.provision("purgeflowless")
        self.assertEqual(first.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         first.message)
        _assert_clean(self, first.message)
        self.assertEqual(second.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT)
        self.assertEqual(second.checkpoint["id"],
                         first.checkpoint["id"])
        _assert_clean(self, second.message)
        self.assertIn("one step only you can do", second.message)
        self.assertIn(GEN_EMAIL, second.message)


class UrlOnlyServiceDrivenTests(unittest.TestCase):
    """A service with a signup URL but NO named signup flow (the
    Helius/PumpPortal registration shape) is driven with the generic
    form flow — the creator's checkpoint is never the first answer."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-test-purge-")
        self.vault = _vault(self.tmp)
        register_provision_service(ProvisionService(
            id="purgeurlonly", display="PurgeUrlOnly",
            vault_service="purgeurlonly",
            env_keys=("PURGEURLONLY_KEY",),
            login_url="https://purgeurlonly.example/login",
            token_url="",
            signup_url="https://purgeurlonly.example/signup",
            validator="", flow=""))
        self.addCleanup(PROVISION_REGISTRY.pop, "purgeurlonly", None)

    def test_drive_attempted_loud_failure_no_handoff(self):
        class _BrokenTab:
            def __init__(self):
                self.navigations: list = []

            def navigate(self, url):
                self.navigations.append(url)
                return {"ok": True, "url": url}

            def fill(self, name, value):
                raise Exception("no such field")

        tab = _BrokenTab()
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("purgeurlonly")
        # Devon WENT to the signup page herself…
        self.assertEqual(tab.navigations,
                         ["https://purgeurlonly.example/signup"])
        # …and the mismatch failed loudly at the named step instead
        # of handing the owner a creation checkpoint
        self.assertEqual(result.outcome, ProvisionOutcome.PAGE_CHANGED,
                         result.message)
        self.assertEqual(result.step, "signup:fill_email")
        self.assertIsNone(result.checkpoint)
        _assert_clean(self, result.message)
        self.assertEqual(
            CheckpointStore(self.vault.db).list_pending(
                service="purgeurlonly"), [])


class CreatorInstructionTests(unittest.TestCase):
    """The creator's per-service checkpoint instructions: ONE plain
    action each — the numbered chore lists are gone at the root."""

    def test_flow_instructions_are_single_actions(self):
        for service in ("github", "gmail", "twitter", "nosuchsvc"):
            flow = AccountCreator._flow_for(service)
            instructions = flow["instructions"].format(
                service=service, username="someuser",
                email="someone@example.com", password="pw")
            self.assertNotIn("\n1.", instructions, service)
            self.assertNotIn("yourself", instructions, service)
            self.assertIn("the one step only you can do",
                          instructions, service)


if __name__ == "__main__":
    unittest.main()
