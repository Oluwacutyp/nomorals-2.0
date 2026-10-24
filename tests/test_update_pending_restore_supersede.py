"""Stale deferred-restore markers — the second EC2 /update incident.

The sequence, proven live on the owner's EC2 host on 2026-10-23:

1. Update A (417405c → 80522fa) FAILED health and rolled back; the
   database restore was blocked by the live process lock, so the
   updater deferred it via ``pending-db-restore.json`` — correct.
2. Devon kept running on the rolled-back code + its LIVE database
   for ~5 hours, accumulating real data.
3. Update B (→ 7fd9bce) SUCCEEDED — it migrated and health-checked
   the live database — but never cleared A's marker.
4. At B's boot, recovery tried to apply A's deferred restore, hit
   the lock again, and paged a FAILED alert whose premise was stale:
   "booting on the post-update database paired with rolled-back
   code". The running code was B (newest) and the live database was
   the CORRECT pairing — had the restore succeeded, it would have
   regressed the live database to A's pre-update snapshot and wiped
   ~5 hours of data to serve a superseded rollback.

The fixes pinned here:

* the marker records provenance (originating run, targeted version,
  starting version);
* a successful update clears a stale marker it did not write, and
  says so in its report; its own marker is never cleared; a no-op
  update (which proves nothing about the database) clears nothing;
* boot recovery checks the journal: a marker a later successful
  update superseded is DISCARDED — live database untouched, no
  failure alert; a no-op completion does not supersede;
* a CURRENT marker still applies at boot, and when it cannot, the
  failure carries the actual state (the marker's update, the running
  version, and whether the running code really is the rolled-back
  one) so the owner notice never asserts a false pairing.
"""

from __future__ import annotations

import json
import tempfile
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
    hardening suite's idiom). ``move_to=None`` simulates an
    up-to-date pull (HEAD stays where it started)."""
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


def _step(report, name: str):
    return next((s for s in report.steps if s["step"] == name), None)


def _run_failed_update_a(tmp: str) -> dict:
    """Update A: pulls SHA_B, fails health, rolls back against a
    locked database → deferred-restore marker. Returns the marker
    payload exactly as written."""
    snaps = _FakeSnapshots(tmp, restore_exc=_locked_exc())
    mgr, _ = _manager(tmp, health=[lambda: (False, "v2 is broken")],
                      start=SHA_A, move_to=SHA_B, snaps=snaps)
    mgr._restore_tree = lambda sha: (True, "tree restored")
    report = mgr.run()
    assert not report.ok and report.rolled_back, report.render()
    assert report.db_restore_deferred, report.render()
    marker = json.loads(
        Path(mgr.pending_restore_path).read_text(encoding="utf-8"))
    return marker


class TestMarkerProvenance(unittest.TestCase):
    def test_marker_records_originating_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = _run_failed_update_a(tmp)
            self.assertEqual(marker["snapshot"], "snap-1")
            self.assertEqual(marker["target_sha"], SHA_B)
            self.assertEqual(marker["from_sha"], SHA_A)
            self.assertTrue(marker["run_id"])
            self.assertIn("reason", marker)
            self.assertIn("ts", marker)


class TestSuccessfulUpdateClearsStaleMarker(unittest.TestCase):
    def test_ec2_replay_b_success_clears_and_boot_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_failed_update_a(tmp)
            live_db = Path(tmp) / "devon.db"
            live_db.write_bytes(b"live data from ~5h of running")
            # Update B: rolled-back code at SHA_A, pulls SHA_C,
            # health green. It must adopt the live DB and clear A's
            # marker, saying so.
            mgr_b, snaps_b = _manager(
                tmp, health=[lambda: (True, "fine")],
                start=SHA_A, move_to=SHA_C)
            report_b = mgr_b.run()
            self.assertTrue(report_b.ok, report_b.error)
            step = _step(report_b, "pending_restore")
            self.assertIsNotNone(step, report_b.render())
            self.assertTrue(step["ok"])
            self.assertIn("cleared stale", step["detail"])
            self.assertIn(SHA_B[:12], step["detail"])
            self.assertFalse(Path(mgr_b.pending_restore_path).exists())
            self.assertEqual(snaps_b.restored, [])
            self.assertEqual(live_db.read_bytes(),
                             b"live data from ~5h of running")
            # B's boot: no marker, last run terminated cleanly — no
            # recovery action, no failure alert, DB untouched.
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_C)
            receipt = mgr_boot.recover_interrupted()
            self.assertFalse(receipt["recovered"], receipt)
            self.assertEqual(snaps_boot.restored, [])
            self.assertEqual(live_db.read_bytes(),
                             b"live data from ~5h of running")

    def test_own_marker_is_never_cleared_by_a_successful_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp, health=[], start=SHA_A,
                              move_to=SHA_B)

            def side_effect_health():
                mgr._write_pending_restore("snap-x", "written mid-run")
                return True, "fine"

            mgr.health_checks = [side_effect_health]
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertIsNone(_step(report, "pending_restore"))
            self.assertTrue(Path(mgr.pending_restore_path).exists())

    def test_noop_update_clears_nothing(self):
        # A no-op update runs no migration and no health gate, so it
        # proves nothing about the database: the marker survives it.
        with tempfile.TemporaryDirectory() as tmp:
            _run_failed_update_a(tmp)
            mgr, _ = _manager(tmp, start=SHA_A, move_to=None)
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertIsNotNone(_step(report, "up_to_date"))
            self.assertIsNone(_step(report, "pending_restore"))
            self.assertTrue(Path(mgr.pending_restore_path).exists())


class TestRecoveryDiscardsSupersededMarker(unittest.TestCase):
    def test_boot_discards_marker_that_survived_a_successful_update(self):
        # The marker B was expected to clear but didn't (older build,
        # failed unlink, marker restored onto the box): its bytes are
        # A's, its timestamp predates B's completion. Recovery must
        # discard it — never apply it.
        with tempfile.TemporaryDirectory() as tmp:
            marker_a = _run_failed_update_a(tmp)
            mgr_b, _ = _manager(tmp, health=[lambda: (True, "fine")],
                                start=SHA_A, move_to=SHA_C)
            report_b = mgr_b.run()
            self.assertTrue(report_b.ok, report_b.error)
            marker_path = Path(mgr_b.pending_restore_path)
            self.assertFalse(marker_path.exists())  # B cleared it
            marker_path.write_text(json.dumps(marker_a),
                                   encoding="utf-8")
            live_db = Path(tmp) / "devon.db"
            live_db.write_bytes(b"live data")
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_C)
            receipt = mgr_boot.recover_interrupted()
            self.assertNotEqual(receipt.get("action"),
                                "pending_restore_failed")
            self.assertFalse(receipt["recovered"], receipt)
            self.assertEqual(snaps_boot.restored, [])
            self.assertFalse(marker_path.exists())
            self.assertEqual(live_db.read_bytes(), b"live data")
            journal = Path(mgr_boot.journal_path).read_text(
                encoding="utf-8")
            self.assertIn("pending_restore_discarded_stale", journal)
            self.assertIn(SHA_C, journal)

    def test_noop_completion_does_not_supersede_at_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp)
            mgr._write_pending_restore(
                "snap-1", "PermissionError: locked", run_id="run-a",
                target_sha=SHA_B, from_sha=SHA_A)
            # A no-op update completes AFTER the marker was written.
            noop, _ = _manager(tmp, start=SHA_A, move_to=None)
            report = noop.run()
            self.assertTrue(report.ok, report.error)
            # Recovery: the marker is still current — it applies.
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_A)
            receipt = mgr_boot.recover_interrupted()
            self.assertEqual(receipt["action"],
                             "pending_restore_applied")
            self.assertEqual(snaps_boot.restored, ["snap-1"])
            self.assertFalse(Path(mgr_boot.pending_restore_path)
                             .exists())


class TestGenuineMarkerStillWorks(unittest.TestCase):
    def test_genuine_marker_applies_at_next_boot(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_failed_update_a(tmp)
            mgr_boot, snaps_boot = _manager(tmp, start=SHA_A)
            receipt = mgr_boot.recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"],
                             "pending_restore_applied")
            self.assertEqual(receipt["restored_snapshot"], "snap-1")
            self.assertEqual(snaps_boot.restored, ["snap-1"])
            self.assertFalse(Path(mgr_boot.pending_restore_path)
                             .exists())

    def test_current_marker_lock_at_boot_fails_with_actual_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_failed_update_a(tmp)
            # Boot on the rolled-back code (SHA_A), database STILL
            # locked: the failure is genuine and must say so, naming
            # the marker's update and the running version.
            snaps = _FakeSnapshots(tmp, restore_exc=_locked_exc())
            mgr_boot, _ = _manager(tmp, start=SHA_A, snaps=snaps)
            receipt = mgr_boot.recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"],
                             "pending_restore_failed")
            self.assertTrue(Path(mgr_boot.pending_restore_path)
                            .exists())
            self.assertEqual(receipt["update_sha"], SHA_B)
            self.assertEqual(receipt["running_sha"], SHA_A)
            note = receipt["state_note"]
            self.assertIn(SHA_B[:12], note)
            self.assertIn(SHA_A[:12], note)
            self.assertIn("rolled-back pre-update version", note)
            from nomorals.agents.partner.runtime import (
                _update_recovery_notice)
            _title, body = _update_recovery_notice(receipt)
            self.assertIn("Actual state:", body)
            self.assertIn(SHA_B[:12], body)

    def test_failure_never_claims_rolled_back_code_when_code_moved_on(self):
        # Marker is journal-current (no successful update after it),
        # but the code on disk is NOT the rolled-back version (e.g.
        # the owner hand-pulled). The notice must describe that —
        # never assert the rolled-back pairing.
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp)
            mgr._write_pending_restore(
                "snap-1", "PermissionError: locked", run_id="run-a",
                target_sha=SHA_B, from_sha=SHA_A)
            snaps = _FakeSnapshots(
                tmp, restore_exc=RuntimeError("io error"))
            mgr_boot, _ = _manager(tmp, start=SHA_C, snaps=snaps)
            receipt = mgr_boot.recover_interrupted()
            self.assertEqual(receipt["action"],
                             "pending_restore_failed")
            self.assertEqual(receipt["running_sha"], SHA_C)
            self.assertIn("NOT the rolled-back version",
                          receipt["state_note"])
            from nomorals.agents.partner.runtime import (
                _update_recovery_notice)
            _title, body = _update_recovery_notice(receipt)
            self.assertNotIn("paired with rolled-back code", body)
            self.assertIn("NOT the rolled-back version", body)
            self.assertIn(SHA_C[:12], body)


if __name__ == "__main__":
    unittest.main()
