"""Round 19: /trial resume — continuing a paused account flow from chat."""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.accounts.creator import (
    AccountCreator,
    CheckpointKind,
    CheckpointState,
)
from nomorals.accounts.vault import CredentialVault
from nomorals.agents.trial.flow import TrialFlow
from nomorals.storage.db import Database


def _flow():
    tmp = tempfile.mkdtemp()
    db = Database(":memory:")
    db.execute(
        "CREATE TABLE IF NOT EXISTS kv_store ("
        "key TEXT PRIMARY KEY, value TEXT, kind TEXT, updated_at REAL)")
    ctx = SimpleNamespace(settings=SimpleNamespace(home=tmp), db=db)
    flow = TrialFlow(ctx)
    return flow, db, tmp


class TrialResumeTests(unittest.TestCase):
    def setUp(self):
        self.flow, self.db, self.tmp = _flow()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._env = mock.patch.dict(
            os.environ, {"NM_VAULT_PASSPHRASE": "test-pass"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _creator(self):
        vault = CredentialVault(
            self.db, master_passphrase="test-pass")
        return AccountCreator(vault, db=self.db)

    def _checkpoint(self, flow="account_create", **state):
        creator = self._creator()
        rs = {"flow": flow, "service": "github", "username": "octo",
              "password": "pw123", "email": "o@example.com"}
        rs.update(state)
        return creator.checkpoints.create(
            CheckpointKind.MANUAL_STEP, "t", "do the thing",
            service="github", resume_state=rs)

    def test_resume_unknown_id(self):
        out = self.flow.resume("nope")
        self.assertIn("no checkpoint", out)

    def test_resume_empty_id_usage(self):
        self.assertIn("usage", self.flow.resume(""))

    def test_resume_vault_locked(self):
        with mock.patch.dict(os.environ, {"NM_VAULT_PASSPHRASE": ""}):
            out = self.flow.resume("x")
        self.assertIn("vault is locked", out)

    def test_resume_already_resolved(self):
        cp = self._checkpoint()
        creator = self._creator()
        creator.checkpoints.resolve(cp.id, "done")
        out = self.flow.resume(cp.id)
        self.assertIn("already resolved", out)

    def test_resume_account_create_finalizes(self):
        cp = self._checkpoint()
        out = self.flow.resume(cp.id)
        self.assertIn("account ready", out)
        self.assertIn("octo", out)
        # credentials actually landed in the vault
        vault = CredentialVault(self.db, master_passphrase="test-pass")
        cred = vault.get("github", "octo")
        self.assertEqual("pw123", cred.password)

    def test_resume_identity_flow_reports_next_step(self):
        cp = self._checkpoint(flow="need_identity", service="github")
        # identity bank empty -> MissingOwnerIdentity path; set identity
        creator = self._creator()
        creator.set_owner_identity("Owner", "o@example.com")
        out = self.flow.resume(cp.id)
        self.assertIn("identity recorded", out)
        self.assertIn("/trial assist github", out)


if __name__ == "__main__":
    unittest.main()
