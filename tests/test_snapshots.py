"""Tests for snapshots, transactional restore, and self-update rollback."""

import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.os.snapshots import (
    SnapshotManager,
    SnapshotRefused,
    system_state,
)
from nomorals.os.update import UpdateManager
from nomorals.storage.db import Database


def make_state(test=None):
    """A fake live state dir: migrated DB + blob dir + a probe table."""
    tmp = tempfile.mkdtemp(prefix="snap-test-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    home = Path(tmp)
    db_path = home / "state.db"
    blob_dir = home / "blobs"
    blob_dir.mkdir()
    (blob_dir / "payload.bin").write_bytes(b"x" * 1024)

    db = Database(str(db_path))
    db.migrate()
    db.execute("CREATE TABLE IF NOT EXISTS probe (k TEXT PRIMARY KEY, v TEXT)")
    db.execute("INSERT OR REPLACE INTO probe (k, v) VALUES ('marker', 'original')")
    db.close()
    return home, db_path, blob_dir


def make_manager(home, db_path, blob_dir):
    return SnapshotManager(home=home, db_path=db_path, blob_dir=blob_dir,
                           settings_dict={"profile": "test"})


def read_marker(db_path):
    db = Database(str(db_path))
    try:
        return db.scalar("SELECT v FROM probe WHERE k = 'marker'", default=None)
    finally:
        db.close()


class TestSnapshots(unittest.TestCase):
    def test_create_list_verify_round_trip(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        snap = mgr.create(label="round-trip")
        self.assertTrue((Path(snap.path) / "db.sqlite").exists())
        self.assertTrue((Path(snap.path) / "blobs").exists())
        self.assertEqual(mgr.verify(snap.id), [])
        ids = [s.id for s in mgr.list()]
        self.assertIn(snap.id, ids)

    def test_restore_preserves_state(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        snap = mgr.create()

        # Damage the live state: wipe the marker row, trash the blob dir.
        db = Database(str(db_path))
        db.execute("DELETE FROM probe WHERE k = 'marker'")
        db.close()
        shutil.rmtree(blob_dir)

        mgr.restore(snap.id, force=True)

        self.assertEqual(read_marker(db_path), "original")
        self.assertTrue((blob_dir / "payload.bin").exists())
        # Restored DB is healthy.
        db = Database(str(db_path))
        try:
            self.assertEqual(db.integrity_check(), "ok")
        finally:
            db.close()

    def test_verify_catches_tampering(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        snap = mgr.create()
        # Corrupt the snapshot's database copy.
        with open(Path(snap.path) / "db.sqlite", "r+b") as fh:
            fh.seek(100)
            fh.write(b"\x00\xff\x00\xff")
        problems = mgr.verify(snap.id)
        self.assertTrue(problems, "tampered snapshot must not verify clean")

    def test_restore_refuses_dirty_without_force(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        snap = mgr.create()
        # Simulate an un-checkpointed WAL sidecar.
        (Path(str(db_path) + "-wal")).write_bytes(b"\x00" * 64)
        try:
            state = system_state(home, db_path)
            self.assertTrue(state["dirty"])
            with self.assertRaises(SnapshotRefused):
                mgr.restore(snap.id)
            # With force it goes through.
            mgr.restore(snap.id, force=True)
            self.assertEqual(read_marker(db_path), "original")
        finally:
            wal = Path(str(db_path) + "-wal")
            if wal.exists():
                wal.unlink()

    def test_restore_refuses_running_system_without_force(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        snap = mgr.create()
        # Another *process* holds the lock (POSIX locks are per-process).
        holder = subprocess.Popen(
            ["python3", "-c",
             "import fcntl, time, sys; "
             f"fd = open({str(home / 'nomorals.lock')!r}, 'w'); "
             "fcntl.flock(fd, fcntl.LOCK_EX); time.sleep(15)"],
        )
        try:
            deadline = time.time() + 10
            while time.time() < deadline:
                if system_state(home, db_path)["running"]:
                    break
                time.sleep(0.1)
            self.assertTrue(system_state(home, db_path)["running"])
            with self.assertRaises(SnapshotRefused):
                mgr.restore(snap.id)
        finally:
            holder.terminate()
            holder.wait(timeout=10)
        # Lock released: restore proceeds without force.
        mgr.restore(snap.id)
        self.assertEqual(read_marker(db_path), "original")

    def test_delete(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        snap = mgr.create()
        mgr.delete(snap.id)
        self.assertEqual(mgr.list(), [])


class TestSelfUpdate(unittest.TestCase):
    def test_failed_pull_leaves_tree_untouched(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        updater = UpdateManager(home, mgr)  # not a git repo
        report = updater.run(pull=True)
        self.assertFalse(report.ok)
        self.assertIn("git_pull", [s["step"] for s in report.steps])
        self.assertFalse(any(s["step"] == "git_pull" and s["ok"]
                             for s in report.steps))
        self.assertFalse(report.rolled_back)  # nothing changed: nothing to undo
        self.assertTrue(report.pre_update_snapshot)  # snapshot still taken
        self.assertEqual(read_marker(db_path), "original")

    def test_broken_health_check_triggers_rollback(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)

        def evil_check():
            # Simulate update damage, then fail the health check.
            db = Database(str(db_path))
            try:
                db.execute("DELETE FROM probe WHERE k = 'marker'")
            finally:
                db.close()
            return False, "simulated post-update breakage"

        updater = UpdateManager(home, mgr, health_checks=[evil_check])
        report = updater.run(pull=False)

        self.assertFalse(report.ok)
        self.assertTrue(report.rolled_back, "failed health check must roll back")
        self.assertIn("simulated post-update breakage", report.error)
        # The rollback restored the pre-update state.
        self.assertEqual(read_marker(db_path), "original")

    def test_clean_update_succeeds(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        updater = UpdateManager(
            home, mgr,
            health_checks=[lambda: (True, "all good")],
        )
        report = updater.run(pull=False)
        self.assertTrue(report.ok)
        self.assertFalse(report.rolled_back)
        self.assertTrue(all(s["ok"] for s in report.steps))

    def test_snapshot_failure_aborts_before_changes(self):
        home, db_path, blob_dir = make_state(self)
        mgr = make_manager(home, db_path, blob_dir)
        # Make snapshot creation impossible: snapshots root is a file.
        (home / "snapshots").write_text("not a dir")
        updater = UpdateManager(
            home, mgr, health_checks=[lambda: (True, "ok")])
        report = updater.run(pull=False)
        self.assertFalse(report.ok)
        self.assertFalse(report.rolled_back)
        self.assertIn("snapshot", report.error)


if __name__ == "__main__":
    unittest.main()
