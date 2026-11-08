"""Deferred-restore PREMISE fix — the 2026-11-07 EC2 incident.

The live sequence on the owner's EC2 host:

1. A self-update failed on an earlier day and rolled back; the
   database restore was blocked by the live process lock, so the
   updater deferred it via ``pending-db-restore.json`` — correct.
2. The owner updates by HAND (``git pull``), so no update ever
   completes in the update journal on his install: the journal
   supersede check — the only stale-marker defence at boot — can
   never fire for him.
3. At the next boot, recovery tried to apply the dead update's
   restore and paged a failure only because the database file
   happened to be locked (WinError 32). Had the file been free, the
   restore would have APPLIED and rewound the live database to
   serve a rollback whose premise no longer existed: the running
   code (hand-pulled, healthy) was provably NOT the rolled-back
   version the marker pairs with — a fact the boot code already
   computed for its honest state note, then ignored for the
   decision.

The rule pinned here (recorded facts only): the deferred restore
exists to pair the pre-update snapshot with the pre-update CODE —
``from_sha``, the version the failed run started from and the
rollback restored the tree to. At boot, before any restore attempt:

* running code == ``from_sha`` → premise HOLDS: the restore
  applies exactly as before (and a lock failure keeps today's
  defer + failure page);
* ``from_sha`` recorded, running version known, and != ``from_sha``
  → premise GONE: the marker is discarded (journal event +
  operator notice as information, not a page), the live database
  is kept untouched, and no restore is attempted — so neither the
  wrongful apply nor the every-boot failure page can occur;
* provenance insufficient to judge (marker records no ``from_sha``,
  running version unknowable, marker malformed) → today's
  behaviour exactly. A discard on a guess is the one unacceptable
  outcome.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40


class _FakeSnapshots:
    def __init__(self, home: str, restore_exc: Exception | None = None):
        self.home = home
        self.db_path = f"{home}/devon.db"
        self.restored: list[str] = []
        self._restore_exc = restore_exc

    def create(self, label: str = "") -> SimpleNamespace:
        return SimpleNamespace(id="snap-1")

    def restore(self, snap_id: str, force: bool = False) -> None:
        if self._restore_exc is not None:
            raise self._restore_exc
        self.restored.append(snap_id)


def _locked_exc() -> PermissionError:
    exc = PermissionError("the database file is locked")
    exc.winerror = 32  # WinError 32: file in use by another process
    return exc


def _manager(tmp: str, *, health=(), start: str = SHA_A,
             move_to: str | None = SHA_B, snaps=None):
    """An UpdateManager with scripted git/migrate seams (the
    supersede suite's idiom). ``start`` is the HEAD the boot
    resolves — the running version in every premise judgement."""
    from nomorals.os.update import UpdateManager

    snaps = snaps or _FakeSnapshots(tmp)
    mgr = UpdateManager(tmp, snaps, health_checks=list(health))
    state = {"head": start}
    mgr._preflight_pull = lambda: (True, "")
    mgr._git_sha = lambda: state["head"]

    def _pull():
        if move_to is not None:
            state["head"] = move_to
        return True, "ok"

    mgr._git_pull = _pull
    mgr._migrate = lambda: ([], 88)
    return mgr, snaps


def _tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR"))


def _run_failed_update_a(tmp: str) -> dict:
    """Update A: SHA_A → SHA_B, fails health, rolls back against a
    locked database → deferred-restore marker with real provenance
    and a journal whose last run terminated ``rolled_back`` (no
    completion — nothing supersedes the marker)."""
    snaps = _FakeSnapshots(tmp, restore_exc=_locked_exc())
    mgr, _ = _manager(tmp, health=[lambda: (False, "v2 is broken")],
                      start=SHA_A, move_to=SHA_B, snaps=snaps)
    mgr._restore_tree = lambda sha: (True, "tree restored")
    report = mgr.run()
    assert not report.ok and report.rolled_back, report.render()
    assert report.db_restore_deferred, report.render()
    return json.loads(
        Path(mgr.pending_restore_path).read_text(encoding="utf-8"))


def _write_marker(tmp: str, payload: dict) -> Path:
    from nomorals.os.update import PENDING_RESTORE_FILENAME

    path = Path(tmp) / PENDING_RESTORE_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _provenance_payload(**over) -> dict:
    payload = {"snapshot": "snap-1",
               "reason": "PermissionError: locked",
               "ts": time.time() - 3600, "run_id": "run-a",
               "target_sha": SHA_B, "from_sha": SHA_A}
    payload.update(over)
    return payload


class TestPremiseGoneDiscards(unittest.TestCase):
    def test_incident_replay_hand_pulled_code_discards(self):
        with _tmpdir() as tmp:
            marker = _run_failed_update_a(tmp)
            self.assertEqual(marker["from_sha"], SHA_A)
            live_db = Path(tmp) / "devon.db"
            live_db.write_bytes(b"live data, post-pull")
            # The owner hand-pulled past the marker's pairing: the
            # boot resolves SHA_C. No journal completion exists.
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_C)
            receipt = mgr_boot.recover_interrupted()
            self.assertTrue(receipt["recovered"], receipt)
            self.assertEqual(receipt["action"],
                             "pending_restore_discarded")
            self.assertEqual(receipt["update_sha"], SHA_B)
            self.assertEqual(receipt["from_sha"], SHA_A)
            self.assertEqual(receipt["running_sha"], SHA_C)
            self.assertIn("NOT the rolled-back version",
                          receipt["state_note"])
            # No restore attempted; marker gone; live DB untouched.
            self.assertEqual(snaps_boot.restored, [])
            self.assertFalse(Path(mgr_boot.pending_restore_path)
                             .exists())
            self.assertEqual(live_db.read_bytes(),
                             b"live data, post-pull")
            journal = Path(mgr_boot.journal_path).read_text(
                encoding="utf-8")
            self.assertIn("pending_restore_discarded_stale", journal)
            self.assertIn('"discarded_reason": "premise_gone"',
                          journal)
            # The notice is information, not a failure page.
            from nomorals.agents.partner.runtime import _update_recovery_notice
            title, body = _update_recovery_notice(receipt)
            self.assertNotIn("FAILED", title)
            self.assertIn("discarded", title)
            self.assertIn("live database is kept", body)
            self.assertIn("No action needed", body)
            self.assertIn(SHA_C[:12], body)

    def test_discard_precedes_attempt_even_when_db_locked(self):
        # The repeat-page shape: a premise-gone marker + a locked
        # database must NOT page at every boot — the discard is
        # decided before the restore is ever attempted.
        with _tmpdir() as tmp:
            marker_path = _write_marker(tmp, _provenance_payload())
            snaps = _FakeSnapshots(tmp, restore_exc=_locked_exc())
            mgr_boot, _ = _manager(tmp, start=SHA_C, snaps=snaps)
            out = mgr_boot.apply_pending_restore()
            self.assertFalse(out["applied"])
            self.assertTrue(out["discarded_premise"])
            self.assertNotIn("error", out)
            self.assertEqual(snaps.restored, [])
            self.assertFalse(marker_path.exists())

    def test_running_the_failed_target_itself_discards(self):
        # The owner hand-pulled the very version the update failed
        # on and it runs: the rollback's premise is equally dead.
        with _tmpdir() as tmp:
            _run_failed_update_a(tmp)
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_B)
            receipt = mgr_boot.recover_interrupted()
            self.assertEqual(receipt["action"],
                             "pending_restore_discarded")
            self.assertEqual(snaps_boot.restored, [])


class TestPremiseHoldsStillApplies(unittest.TestCase):
    def test_current_marker_applies_on_rolled_back_code(self):
        with _tmpdir() as tmp:
            _run_failed_update_a(tmp)
            # Boot on the rolled-back code (SHA_A == from_sha): the
            # pairing the restore serves is the real one.
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_A)
            receipt = mgr_boot.recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"],
                             "pending_restore_applied")
            self.assertEqual(snaps_boot.restored, ["snap-1"])
            self.assertFalse(Path(mgr_boot.pending_restore_path)
                             .exists())


class TestCurrentMarkerLockBehaviourPreserved(unittest.TestCase):
    def test_locked_current_marker_keeps_todays_failure_page(self):
        with _tmpdir() as tmp:
            _run_failed_update_a(tmp)
            snaps = _FakeSnapshots(tmp, restore_exc=_locked_exc())
            mgr_boot, _ = _manager(tmp, start=SHA_A, snaps=snaps)
            receipt = mgr_boot.recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"],
                             "pending_restore_failed")
            self.assertIn("locked", receipt["error"])
            self.assertTrue(Path(mgr_boot.pending_restore_path)
                            .exists())
            self.assertIn("rolled-back pre-update version",
                          receipt["state_note"])


class TestAmbiguousProvenanceKeepsTodaysBehaviour(unittest.TestCase):
    def test_marker_without_from_sha_applies_as_today(self):
        # Written by an older build: no recorded pairing, so the
        # premise cannot be judged — the restore is attempted
        # exactly as today, even with moved-on code.
        with _tmpdir() as tmp:
            payload = _provenance_payload()
            del payload["from_sha"]
            _write_marker(tmp, payload)
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_C)
            receipt = mgr_boot.recover_interrupted()
            self.assertEqual(receipt["action"],
                             "pending_restore_applied")
            self.assertEqual(snaps_boot.restored, ["snap-1"])

    def test_marker_without_from_sha_locked_fails_as_today(self):
        with _tmpdir() as tmp:
            payload = _provenance_payload()
            del payload["from_sha"]
            marker_path = _write_marker(tmp, payload)
            snaps = _FakeSnapshots(tmp, restore_exc=_locked_exc())
            mgr_boot, _ = _manager(tmp, start=SHA_C, snaps=snaps)
            receipt = mgr_boot.recover_interrupted()
            self.assertEqual(receipt["action"],
                             "pending_restore_failed")
            self.assertTrue(marker_path.exists())

    def test_unknown_running_sha_never_discards(self):
        # Full provenance, but the boot cannot resolve its own
        # version (not a repo, git missing): no discard on a guess.
        with _tmpdir() as tmp:
            _write_marker(tmp, _provenance_payload())
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_C)
            mgr_boot._git_sha = lambda: ""
            out = mgr_boot.apply_pending_restore()
            self.assertTrue(out["applied"])
            self.assertEqual(snaps_boot.restored, ["snap-1"])

    def test_malformed_marker_is_left_alone(self):
        with _tmpdir() as tmp:
            from nomorals.os.update import PENDING_RESTORE_FILENAME

            marker_path = Path(tmp) / PENDING_RESTORE_FILENAME
            marker_path.write_bytes(b"{not json at all")
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_C)
            receipt = mgr_boot.recover_interrupted()
            self.assertFalse(receipt["recovered"], receipt)
            self.assertEqual(snaps_boot.restored, [])
            self.assertTrue(marker_path.exists())

    def test_marker_naming_no_snapshot_is_left_alone(self):
        with _tmpdir() as tmp:
            marker_path = _write_marker(tmp, {})
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_C)
            out = mgr_boot.apply_pending_restore()
            self.assertFalse(out["applied"])
            self.assertIn("no snapshot", out["reason"])
            self.assertNotIn("discarded_premise", out)
            self.assertEqual(snaps_boot.restored, [])
            self.assertTrue(marker_path.exists())


if __name__ == "__main__":
    unittest.main()
