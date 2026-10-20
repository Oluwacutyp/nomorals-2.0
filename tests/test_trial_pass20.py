"""Pass #20 — trial + accounts: two verified bugs.

1. **Duplicate assist launches for the same platform.** ``TrialFlow``
   guards one-account-per-service by checking the VAULTS
   (``_existing_account``), but nothing checked whether a signup for
   the same platform was already IN FLIGHT.  Two ``/trial assist X``
   commands in quick succession (or two confirmed persona tokens)
   both passed the vault check — no account lands in the vault until
   a drive finishes — and both launched driver threads, so two
   signups raced for one service and both submitted forms.
   ``_launch_assist`` is the single choke point (assist / confirm /
   resume re-drive all pass through it); it now refuses when a
   non-terminal run for the platform exists, including runs persisted
   by another TrialFlow instance (the runtime builds one per command).

2. **TrialVault silently rotated a corrupt master key.**
   ``_master()`` treated a key file of the wrong length as "no key"
   and overwrote it with a fresh random key.  One truncated/corrupt
   ``trial.key`` read therefore destroyed the key material for every
   stored credential — silently: new stores kept working under the
   new key while all previous credentials became permanently
   unreadable.  A malformed key file now fails loud and is left
   untouched, so the original key can be restored and the
   credentials recovered.
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.trial.flow import TrialFlow
from nomorals.agents.trial.vault import TrialVault
from nomorals.storage.db import Database

_KV_DDL = (
    "CREATE TABLE IF NOT EXISTS kv_store ("
    "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
    "kind TEXT NOT NULL DEFAULT 'json', updated_at REAL NOT NULL, "
    "expires_at REAL)")


class _FakeThread:
    """Records thread starts without running the target."""

    started: list = []

    def __init__(self, target=None, args=(), name=None, daemon=None):
        self._args = args

    def start(self):
        type(self).started.append(self._args)


def _flow(tmp, db):
    ctx = SimpleNamespace(settings=SimpleNamespace(home=tmp), db=db)
    return TrialFlow(ctx)


class DuplicateLaunchTests(unittest.TestCase):
    def setUp(self):
        # recovery passes are one-shot per process; simulate a process
        # where they already ran so persisted rows keep their states.
        TrialFlow._assist_recovery_done = True
        TrialFlow._sms_recovery_done = True
        _FakeThread.started = []
        self.tmp = tempfile.mkdtemp()
        self.db = Database(":memory:")
        self.db.execute(_KV_DDL)
        self.env = mock.patch.dict(
            os.environ, {"NM_VAULT_PASSPHRASE": "test-pass"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _launch(self, flow, platform="github"):
        return flow._launch_assist(
            platform, chat_key="telegram:1",
            identity={"name": "P", "email": "p@mail.tm",
                      "disposable": "true"},
            persona_id="p1")

    def test_second_launch_same_platform_refused(self):
        flow = _flow(self.tmp, self.db)
        with mock.patch("nomorals.agents.trial.flow.threading.Thread",
                        _FakeThread):
            first = self._launch(flow)
            second = self._launch(flow)
        self.assertTrue(first.startswith("assisted signup"), first)
        self.assertFalse(second.startswith("assisted signup"), second)
        self.assertIn("already", second.lower())
        self.assertEqual(1, len(_FakeThread.started),
                         "two signup threads launched for one service")

    def test_second_instance_sees_inflight_run(self):
        """The runtime builds a fresh TrialFlow per command — the
        in-flight guard must see runs persisted by other instances."""
        flow1 = _flow(self.tmp, self.db)
        with mock.patch("nomorals.agents.trial.flow.threading.Thread",
                        _FakeThread):
            first = self._launch(flow1)
            flow2 = _flow(self.tmp, self.db)  # fresh instance, same db
            second = self._launch(flow2)
        self.assertTrue(first.startswith("assisted signup"), first)
        self.assertFalse(second.startswith("assisted signup"), second)
        self.assertEqual(1, len(_FakeThread.started))

    def test_different_platform_still_launches(self):
        flow = _flow(self.tmp, self.db)
        with mock.patch("nomorals.agents.trial.flow.threading.Thread",
                        _FakeThread):
            self._launch(flow, "github")
            other = self._launch(flow, "gitlab")
        self.assertTrue(other.startswith("assisted signup"), other)
        self.assertEqual(2, len(_FakeThread.started))

    def test_terminal_run_does_not_block_relaunch(self):
        flow = _flow(self.tmp, self.db)
        with mock.patch("nomorals.agents.trial.flow.threading.Thread",
                        _FakeThread):
            self._launch(flow)
            (run_id,) = list(flow._assist_runs)
            flow._set_assist_run_state(run_id, "done")
            flow._assist_sem.release()  # what the finished thread does
            again = self._launch(flow)
        self.assertTrue(again.startswith("assisted signup"), again)
        self.assertEqual(2, len(_FakeThread.started))


class CorruptKeyTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_corrupt_key_is_not_overwritten(self):
        v = TrialVault(self.home)
        v.store("github", "octocat", "pw123")
        key_path = self.home / "trial.key"
        good_key = key_path.read_bytes()
        corrupt = good_key[:10]
        key_path.write_bytes(corrupt)
        entry = v.get("github")
        # the entry is honestly unreadable…
        self.assertTrue(entry.get("unreadable"))
        # …but the key file must NOT have been silently replaced.
        self.assertEqual(corrupt, key_path.read_bytes(),
                         "corrupt key file was overwritten — the "
                         "original key material is destroyed")
        # restoring the original key recovers the credential.
        key_path.write_bytes(good_key)
        self.assertEqual("pw123", v.get("github")["secret"])

    def test_store_with_corrupt_key_fails_loud(self):
        v = TrialVault(self.home)
        v.store("github", "octocat", "pw123")
        key_path = self.home / "trial.key"
        key_path.write_bytes(b"short")
        with self.assertRaises(ValueError):
            v.store("gitlab", "u", "p")
        self.assertEqual(b"short", key_path.read_bytes())

    def test_fresh_vault_still_creates_key(self):
        v = TrialVault(self.home)
        v.store("github", "octocat", "pw123")
        self.assertEqual("pw123", v.get("github")["secret"])


if __name__ == "__main__":
    unittest.main()
