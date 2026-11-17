"""Gapfix R3-48: TrialFlow mints disposable identities via the sealed
TrialIdentityBinder — not inline through IdentityBank.

Failing-first: before the wiring, TrialFlow's disposable mint never
touches TrialIdentityBinder, so every binder assertion here fails.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.trial.flow import TrialFlow
from nomorals.agents.trial.identity import TrialIdentityBinder
from nomorals.core.errors import ToolError
from nomorals.storage.db import Database

_KV_DDL = (
    "CREATE TABLE IF NOT EXISTS kv_store ("
    "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
    "kind TEXT NOT NULL DEFAULT 'json', updated_at REAL NOT NULL, "
    "expires_at REAL)")


def _flow(tmp, db):
    ctx = SimpleNamespace(settings=SimpleNamespace(home=tmp), db=db)
    return TrialFlow(ctx)


class BinderMintTests(unittest.TestCase):
    def setUp(self):
        # recovery passes are one-shot per process; simulate a process
        # where they already ran so persisted rows keep their states.
        TrialFlow._assist_recovery_done = True
        TrialFlow._sms_recovery_done = True
        self.tmp = tempfile.mkdtemp()
        self.db = Database(":memory:")
        self.db.execute(_KV_DDL)
        self.env = mock.patch.dict(
            os.environ, {"NM_VAULT_PASSPHRASE": "test-pass"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _binder(self):
        return TrialIdentityBinder(self.tmp)

    def test_assist_mints_via_binder(self):
        """A trial run mints its identity through TrialIdentityBinder:
        the persona the flow drafts carries the binder's identity."""
        flow = _flow(self.tmp, self.db)
        card = flow.assist("example-svc")
        self.assertIn("SIGNUP IDENTITY DRAFT", card)
        binder = self._binder()
        self.assertTrue(binder.has("example-svc"))
        identity = binder.identity_for("example-svc")
        persona, gen = flow._generate_disposable_identity("example-svc")
        self.assertEqual(identity["first_name"], persona.first_name)
        self.assertEqual(identity["last_name"], persona.last_name)
        self.assertEqual(identity["email"], persona.email)
        self.assertEqual(identity["phone"], persona.phone)
        self.assertEqual(identity["password"], persona.password)
        self.assertEqual(identity["username"], persona.username_variants[0])
        self.assertTrue(persona.disposable)
        # the confirmation payload carries an id only — never the secret
        self.assertNotIn("password", gen)
        self.assertNotIn(identity["password"], card)

    def test_two_services_get_different_identities(self):
        """One identity per service: different services never share one."""
        flow = _flow(self.tmp, self.db)
        flow.assist("service-a")
        flow.assist("service-b")
        binder = self._binder()
        a = binder.identity_for("service-a")
        b = binder.identity_for("service-b")
        self.assertNotEqual(a["email"], b["email"])
        self.assertNotEqual(a["username"], b["username"])
        self.assertEqual(a["service"], "service-a")
        self.assertEqual(b["service"], "service-b")

    def test_same_service_rebind_returns_same_identity(self):
        """The binder's pinned guarantee: bind twice for one service,
        same identity — never a second one (test_step8_captcha pins
        this; the flow must not break it)."""
        flow = _flow(self.tmp, self.db)
        flow.assist("example-svc")
        first, _ = flow._generate_disposable_identity("example-svc")
        second, _ = flow._generate_disposable_identity("example-svc")
        self.assertEqual(first.email, second.email)
        self.assertEqual(first.password, second.password)
        self.assertEqual(first.id, second.id)

    def test_fresh_identity_after_removal(self):
        """Sequential trials separated by removal get different
        identities — the per-trial freshness the binder allows."""
        flow = _flow(self.tmp, self.db)
        flow.assist("example-svc")
        before = self._binder().identity_for("example-svc")
        flow.remove("example-svc")
        self.assertFalse(self._binder().has("example-svc"))
        flow.assist("example-svc")
        after = self._binder().identity_for("example-svc")
        self.assertNotEqual(before["email"], after["email"])
        self.assertNotEqual(before["username"], after["username"])

    def test_draft_is_not_a_completed_account(self):
        """A binder draft in the vault must not read as a completed
        account: no duplicate refusal, no credential delivery, and the
        listing marks it a draft."""
        flow = _flow(self.tmp, self.db)
        flow.assist("example-svc")
        self.assertEqual(("none", None),
                         flow._existing_account("example-svc"))
        with self.assertRaises(ToolError) as caught:
            flow.deliver("example-svc")
        self.assertIn("not finished", str(caught.exception))
        listing = flow.list()
        self.assertIn("example-svc", listing)
        self.assertIn("draft", listing)
        self.assertNotIn("trial-example-svc-", listing)

    def test_identity_sealed_not_plaintext(self):
        """The identity is sealed in the TrialVault: the vault file
        carries neither the email nor the password in plaintext."""
        flow = _flow(self.tmp, self.db)
        flow.assist("example-svc")
        identity = self._binder().identity_for("example-svc")
        raw = (Path(self.tmp) / "trial_accounts.json").read_text("utf-8")
        self.assertNotIn(identity["password"], raw)
        self.assertNotIn(identity["email"], raw)
        self.assertNotIn(identity["phone"], raw)

    def test_remove_retires_binder_identity(self):
        """Removing the trial retires the bound identity: no live
        identity outlives its trial."""
        flow = _flow(self.tmp, self.db)
        flow.assist("example-svc")
        self.assertTrue(self._binder().has("example-svc"))
        text = flow.remove("example-svc")
        self.assertIn("draft", text)
        self.assertFalse(self._binder().has("example-svc"))

    def test_driver_resolves_binder_persona(self):
        """The persona the signup driver will fetch by persona_id is
        the binder-derived one — the driver sees the binder identity,
        not a second bank mint."""
        flow = _flow(self.tmp, self.db)
        flow.assist("example-svc")
        _, gen = flow._generate_disposable_identity("example-svc")
        bank = flow._identity_bank()
        persona = bank.get(gen["persona_id"])
        self.assertIsNotNone(persona)
        identity = self._binder().identity_for("example-svc")
        assert persona is not None
        self.assertEqual(identity["email"], persona.email)
        self.assertEqual(identity["password"], persona.password)
        self.assertEqual(identity["first_name"], persona.first_name)

    def test_save_replaces_draft_with_account(self):
        """Saving the completed account overwrites the draft: the
        binder no longer claims the service and the real account is
        what the flow sees from then on."""
        flow = _flow(self.tmp, self.db)
        flow.assist("example-svc")
        self.assertTrue(self._binder().has("example-svc"))
        flow.save("example-svc", "realuser", "realpw")
        self.assertFalse(self._binder().has("example-svc"))
        self.assertEqual(("found", ("trial", "realuser")),
                         flow._existing_account("example-svc"))
        text = flow.deliver("example-svc")
        self.assertIn("realuser", text)
        self.assertIn("realpw", text)


if __name__ == "__main__":
    unittest.main()
