"""Pins for the stale-pending-echo fix (owner's 2026-10-24 report).

The 2026-10-23 purge removed the GENERATORS of the old manual-step /
identity-lecture output, but never invalidated STORED pending
checkpoints. A pre-purge pending record (the owner's case: the-odds-api,
persona Ayodele Guest, stored 2026-10-23) was therefore re-reported
verbatim forever by the provision step-3 path — and permanently blocked
a fresh drive — even though the code that generated it was gone.

Pinned here:

* A stored account_create checkpoint whose title or instructions carry
  purged artifacts ("Create the account yourself", "Identity:
  self-generated", "paused at a human step", "finish it, then run", a
  "manual_step" tag, or a numbered chore list) is INVALIDATED (cancelled
  + audited) and the drive proceeds fresh — the old text never surfaces.
* A genuine-wall checkpoint (manual review, real ID, age gate, rate
  limit, CAPTCHA, phone/email verification) keeps reporting exactly as
  today, with its single action — never deleted.
* A stale creation-marker note never leaks into a driven result either
  (step-3 report or a resumed login drive): it is dropped, the
  checkpoint it belonged to is judged on its own text.
* An unknown service never gets a whole-job hand-back from the
  creator's default flow: the checkpoint names the SPECIFIC wall that
  blocks driving (no signup URL) plus the single action (send the URL).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nomorals.accounts import provisioning
from nomorals.accounts.creator import (
    AccountCreator,
    CheckpointKind,
    CheckpointStore,
    CreatedAccount,
)
from nomorals.accounts.provisioning import (
    PROVISION_REGISTRY,
    ProvisionDriver,
    ProvisionOutcome,
    ProvisionService,
    register_provision_service,
)
from nomorals.accounts.vault import CredentialVault
from nomorals.storage.kv import KVStore

GEN_EMAIL = "gen.persona@mailinator.com"

#: The driver's marker key uses the registry id ("odds-api"), NOT the
#: alias the owner typed ("odds_api") — the lecture in the owner's
#: report rode in under this exact key.
MARKER_KEY = "provisioning.created_pending.odds-api"

#: Pre-purge output, as stored on 2026-10-23 (the owner's report).
OLD_TITLE = "Create the-odds-api account ayodele.guest5907"
OLD_INSTRUCTIONS = (
    "Create the account yourself:\n"
    "1. Go to the-odds-api's signup page\n"
    "2. Use email: ayodele.guest5907@mailto.plus\n"
    "3. Pick username: ayodele.guest5907 (or your own)\n"
    "4. Complete any verification step\n"
    "5. Tell me the final username — I'll store the credentials"
)
OLD_NOTE = (
    "Identity: self-generated — no owner identity is recorded, so this "
    "account is being created as Ayodele Guest "
    "<ayodele.guest5907@mailto.plus> (a disposable persona from the vault "
    "identity bank, reused for The Odds API on retries; a recorded owner "
    "identity stays preferred whenever one is set)."
)

PURGED_STRINGS = (
    "Create the account yourself",
    "Identity: self-generated",
    "paused at a human step",
    "finish it, then run",
    "manual_step",
    "\n1.",
    "\n2.",
)


def _assert_clean(case: unittest.TestCase, message: str) -> None:
    for fixture in PURGED_STRINGS:
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
            password="test-password-1",
            email=GEN_EMAIL,
            status="created",
        )

    return _mint


def _driver(vault, tab):
    return ProvisionDriver(
        vault,
        open_tab=lambda service: tab,
        http_get=lambda url, headers: (200, '{"data": []}'),
        environ={})


def _seed_checkpoint(vault, *, title, instructions, note, kind):
    store = CheckpointStore(vault.db)
    cp = store.create(
        kind, title, instructions,
        service="the-odds-api",
        resume_state={
            "flow": "account_create",
            "service": "the-odds-api",
            "username": "ayodele.guest5907",
            "password": "test-password-1",
            "email": "ayodele.guest5907@mailto.plus",
        },
    )
    if note is not None:
        KVStore(vault.db).set(
            MARKER_KEY,
            {"checkpoint_id": cp.id,
             "email": "ayodele.guest5907@mailto.plus",
             "created_at": 0.0, "creation_note": note})
    return cp


class StaleEchoTests(unittest.TestCase):
    """The owner's case: a pre-purge pending must not echo forever."""

    @classmethod
    def setUpClass(cls):
        from nomorals.agents.betdesk.sources import ensure_provisioning

        ensure_provisioning()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-test-stale-echo-")
        self.vault = _vault(self.tmp)

    def _odds_tab(self, after: str):
        from tests.test_provisioning_fix import FakeOddsTab

        return FakeOddsTab(after=after)

    def test_stale_pending_is_invalidated_and_drive_proceeds(self):
        stale = _seed_checkpoint(
            self.vault, title=OLD_TITLE, instructions=OLD_INSTRUCTIONS,
            note=OLD_NOTE, kind=CheckpointKind.MANUAL_STEP)
        tab = self._odds_tab("created")
        driver = _driver(self.vault, tab)
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("odds_api")
        # The stale record is cancelled, not re-reported…
        self.assertEqual(
            CheckpointStore(self.vault.db).get(stale.id).state.value,
            "cancelled")
        # …the old text never surfaces…
        _assert_clean(self, result.message)
        # …and the drive proceeded fresh instead of blocking on it.
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertIn("Account:", result.message)
        audit = [a for a in driver.audit_trail
                 if a.get("outcome") == "pending_stale_invalidated"]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0].get("checkpoint"), stale.id)

    def test_genuine_wall_pending_still_reports(self):
        genuine = _seed_checkpoint(
            self.vault,
            title="Complete email verification for the-odds-api",
            instructions=(
                "the one step only you can do: open the inbox for "
                "gen.persona@mailinator.com and click the verification "
                "link — tell me when it's done and I'll continue."),
            note="Account: Ayodele Guest <ayodele.guest5907@mailto.plus>.",
            kind=CheckpointKind.EMAIL_VERIFY)
        driver = _driver(self.vault, mock.Mock())
        result = driver.provision("odds_api")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         result.message)
        self.assertEqual(result.checkpoint["id"], genuine.id)
        self.assertEqual(
            CheckpointStore(self.vault.db).get(genuine.id).state.value,
            "pending")
        self.assertIn("one step only you can do", result.message)
        self.assertIn("verification link", result.message)
        _assert_clean(self, result.message)

    def test_stale_marker_note_dropped_but_genuine_checkpoint_kept(self):
        genuine = _seed_checkpoint(
            self.vault,
            title="Complete email verification for the-odds-api",
            instructions=(
                "the one step only you can do: open the inbox for "
                "gen.persona@mailinator.com and click the verification "
                "link — tell me when it's done and I'll continue."),
            note=OLD_NOTE,
            kind=CheckpointKind.EMAIL_VERIFY)
        driver = _driver(self.vault, mock.Mock())
        result = driver.provision("odds_api")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT)
        self.assertEqual(result.checkpoint["id"], genuine.id)
        # The checkpoint is judged on ITS text (clean) — but the stale
        # marker note it carried must not leak into the report.
        self.assertNotIn("Identity: self-generated", result.message)
        self.assertNotIn("self-generated", result.message)

    def test_stale_then_genuine_pending_reports_genuine(self):
        stale = _seed_checkpoint(
            self.vault, title=OLD_TITLE, instructions=OLD_INSTRUCTIONS,
            note=OLD_NOTE, kind=CheckpointKind.MANUAL_STEP)
        genuine = _seed_checkpoint(
            self.vault,
            title="Complete email verification for the-odds-api",
            instructions=(
                "the one step only you can do: open the inbox for "
                "gen.persona@mailinator.com and click the verification "
                "link — tell me when it's done and I'll continue."),
            note="Account: Ayodele Guest <ayodele.guest5907@mailto.plus>.",
            kind=CheckpointKind.EMAIL_VERIFY)
        driver = _driver(self.vault, mock.Mock())
        result = driver.provision("odds_api")
        self.assertEqual(
            CheckpointStore(self.vault.db).get(stale.id).state.value,
            "cancelled")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT)
        self.assertEqual(result.checkpoint["id"], genuine.id)
        _assert_clean(self, result.message)

    def test_stale_marker_note_not_adopted_on_login_drive(self):
        # A login exists, so step 2 drives — and the stale marker note
        # it consumes must not ride into the provisioned message.
        self.vault.store("the-odds-api", "ayodele.guest5907",
                         "test-password-1", credential_type="account")
        KVStore(self.vault.db).set(
            MARKER_KEY,
            {"checkpoint_id": "", "email": "ayodele.guest5907@mailto.plus",
             "created_at": 0.0, "creation_note": OLD_NOTE})
        tab = self._odds_tab("created")
        driver = _driver(self.vault, tab)
        with mock.patch.object(
                provisioning, "ensure_login",
                return_value={"username": "ayodele.guest5907",
                              "from_session": False}):
            result = driver.provision("odds_api")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertNotIn("Identity: self-generated", result.message)
        self.assertNotIn("self-generated", result.message)


class PurgedTextDetectorTests(unittest.TestCase):
    """The stale detector itself: purged text in, genuine text out."""

    def test_detector(self):
        is_stale = provisioning._contains_purged_text
        self.assertTrue(is_stale(OLD_TITLE + "\n" + OLD_INSTRUCTIONS))
        self.assertTrue(is_stale("Identity: self-generated — lecture"))
        self.assertTrue(is_stale("paused at a human step (manual_step)"))
        self.assertTrue(is_stale("finish it, then run /provision x again"))
        self.assertTrue(is_stale("do this:\n1. first\n2. second"))
        self.assertFalse(is_stale(
            "the one step only you can do: open the inbox and click "
            "the verification link"))
        self.assertFalse(is_stale("Account: Ayodele Guest <a@b.plus>."))
        self.assertFalse(is_stale(""))


class UnknownServiceDefaultTests(unittest.TestCase):
    """The creator's default flow must name the wall, not the job."""

    def test_default_names_the_wall_not_the_job(self):
        flow = AccountCreator._flow_for("nosuchsvc")
        instructions = flow["instructions"].format(
            service="nosuchsvc", username="someuser",
            email="someone@example.com", password="<redacted>")
        self.assertIn("the one step only you can do", instructions)
        self.assertNotIn("yourself", instructions)
        self.assertNotIn("Create the account yourself", instructions)
        self.assertNotIn("create the nosuchsvc account yourself",
                         instructions)
        # The SPECIFIC wall: Devon has no signup URL for this service.
        self.assertIn("signup", instructions.lower())
        self.assertIn("URL", instructions)

    def test_no_url_service_checkpoint_is_wall_not_handback(self):
        tmp = tempfile.mkdtemp(prefix="nm-test-stale-echo-")
        vault = _vault(tmp)
        register_provision_service(ProvisionService(
            id="echonourl", display="EchoNoUrl",
            vault_service="echonourl",
            env_keys=("ECHONOURL_KEY",),
            login_url="https://echonourl.example/login",
            token_url="https://echonourl.example/keys",
            validator="", flow=""))
        self.addCleanup(PROVISION_REGISTRY.pop, "echonourl", None)
        driver = _driver(vault, mock.Mock())
        mints: list = []
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("echonourl")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         result.message)
        _assert_clean(self, result.message)
        self.assertIn("signup", result.message.lower())
        # The wall is named (no signup URL) and the job is NOT handed
        # back (no "tell me the final username when it's created").
        self.assertIn("no signup URL", result.message)
        self.assertNotIn("Tell me the final username", result.message)
        self.assertNotIn("Create the account yourself", result.message)


if __name__ == "__main__":
    unittest.main()
