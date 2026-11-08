"""Disk discipline — the 2026-11-08 EC2 full-disk /update failure, pinned.

What happened live: the owner's /update failed at its snapshot
step — "OperationalError: database or disk is full" — because
pre-update snapshots (each a full database + blob-store copy)
had accumulated with NO retention bound, and nothing anywhere
checked free disk space before an update started. The hardening
stopped the update safely; the disk had still filled silently
for weeks until the update itself discovered it.

The three repairs, each pinned at its real seam:

1. Retention: ``SnapshotManager.prune_pre_update`` bounds the
   pre-update pile to the newest 3 (+ a protected rollback
   anchor), and ``UpdateManager`` runs it the moment a new
   pre-update snapshot lands. Pruning failures journal, never
   fail the update.
2. Preflight: ``UpdateManager._preflight_disk`` runs before the
   snapshot, the pull and the quiesce, and refuses early with
   both numbers (free X · needs ~Y) when the volume cannot hold
   the update. A measurement failure never bricks an update.
3. Surfacing: /doctor (chat + nm doctor) carries a disk line,
   and the housekeeping beat journals one ``disk_low`` row per
   crossing per day into the personal journal.

Everything runs against temp dirs and patched ``free_bytes`` —
never the real disk's free space, never a live install.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SHA_A = "a" * 40
SHA_B = "b" * 40


def _step(report, name: str):
    return next((s for s in report.steps if s["step"] == name), None)


def _make_db(path: Path, payload_kb: int = 256) -> Path:
    """A real SQLite database file of a non-trivial size."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, body BLOB)")
        blob = b"x" * 4096
        for i in range(max(1, payload_kb // 4)):
            conn.execute("INSERT INTO t VALUES (?, ?)", (i, blob))
        conn.commit()
    finally:
        conn.close()
    return path


def _snapshots(tmp: str):
    from nomorals.os.snapshots import SnapshotManager

    home = Path(tmp)
    db = _make_db(home / "devon.db")
    return SnapshotManager(home, db)


def _scripted(tmp: str, snapshots):
    """An UpdateManager with a REAL snapshot manager and scripted
    git/migrate seams (the updater-hardening idiom); the pull
    lands SHA_B. Returns (manager, pull_calls)."""
    from nomorals.os.update import UpdateManager

    mgr = UpdateManager(tmp, snapshots,
                        health_checks=[lambda: (True, "fine")])
    state = {"head": SHA_A}
    pull_calls: list[str] = []
    mgr._preflight_pull = lambda: (True, "")
    mgr._git_sha = lambda: state["head"]

    def _pull():
        pull_calls.append("pull")
        state["head"] = SHA_B
        return True, "ok"

    mgr._git_pull = _pull
    mgr._migrate = lambda: ([], 3)
    return mgr, pull_calls


# ── Fix 1a: the retention bound itself ───────────────────────────────

class TestPrunePreUpdate(unittest.TestCase):
    def test_keeps_newest_three_and_spares_other_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _snapshots(tmp)
            for _ in range(5):
                mgr.create(label="pre-update")
            manual = mgr.create(label="manual")
            ordered = [s.id for s in mgr.list()
                       if s.label == "pre-update"]
            deleted = mgr.prune_pre_update()
            remaining = [s.id for s in mgr.list()
                         if s.label == "pre-update"]
            self.assertEqual(remaining, ordered[-3:])
            self.assertEqual(deleted, ordered[:-3])
            self.assertIn(manual.id, [s.id for s in mgr.list()])

    def test_protect_keeps_the_named_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _snapshots(tmp)
            for _ in range(5):
                mgr.create(label="pre-update")
            ordered = [s.id for s in mgr.list()
                       if s.label == "pre-update"]
            mgr.prune_pre_update(keep=1, protect={ordered[0]})
            remaining = [s.id for s in mgr.list()
                         if s.label == "pre-update"]
            self.assertEqual(remaining, [ordered[0], ordered[-1]])

    def test_delete_failure_never_raises_and_keeps_going(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _snapshots(tmp)
            for _ in range(5):
                mgr.create(label="pre-update")
            ordered = [s.id for s in mgr.list()
                       if s.label == "pre-update"]
            real_delete = mgr.delete

            def flaky(snap_id: str) -> None:
                if snap_id == ordered[0]:
                    raise OSError("file locked")
                real_delete(snap_id)

            with mock.patch.object(mgr, "delete", side_effect=flaky):
                deleted = mgr.prune_pre_update()
            self.assertEqual(deleted, [ordered[1]])
            remaining = [s.id for s in mgr.list()
                         if s.label == "pre-update"]
            self.assertEqual(remaining,
                             [ordered[0]] + ordered[-3:])


# ── The shared measure ───────────────────────────────────────────────

class TestDiskReport(unittest.TestCase):
    def test_estimate_counts_db_wal_and_blobs_need_is_twice(self):
        from nomorals.core import diskspace

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db = _make_db(home / "devon.db", payload_kb=128)
            Path(str(db) + "-wal").write_bytes(b"w" * 10_000)
            blobs = home / "blobs"
            blobs.mkdir()
            (blobs / "a.bin").write_bytes(b"b" * 20_000)
            snap = diskspace.snapshot_bytes_estimate(db, blobs)
            self.assertEqual(
                snap, db.stat().st_size + 10_000 + 20_000)
            self.assertEqual(diskspace.update_need_bytes(db, blobs),
                             2 * snap)

    def test_low_is_free_below_need(self):
        from nomorals.core import diskspace

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db = _make_db(home / "devon.db", payload_kb=128)
            need = diskspace.update_need_bytes(db)
            with mock.patch.object(diskspace, "free_bytes",
                                   return_value=need - 1):
                self.assertTrue(
                    diskspace.disk_report(home, db)["low"])
            with mock.patch.object(diskspace, "free_bytes",
                                   return_value=need):
                self.assertFalse(
                    diskspace.disk_report(home, db)["low"])

    def test_store_stats_counts_snapshot_dirs(self):
        from nomorals.core import diskspace

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "snapshots"
            for name, size in (("s1", 100), ("s2", 250)):
                d = root / name
                d.mkdir(parents=True)
                (d / "f.bin").write_bytes(b"x" * size)
            count, total = diskspace.snapshot_store_stats(home)
            self.assertEqual((count, total), (2, 350))


# ── Fix 2: the update preflight ──────────────────────────────────────

class TestUpdateDiskPreflight(unittest.TestCase):
    def test_refuses_early_with_numbers_and_changes_nothing(self):
        from nomorals.core import diskspace

        with tempfile.TemporaryDirectory() as tmp:
            mgr, pull_calls = _scripted(tmp, _snapshots(tmp))
            with mock.patch.object(diskspace, "free_bytes",
                                   return_value=1):
                report = mgr.run()
            self.assertFalse(report.ok)
            self.assertIn("not enough free disk", report.error)
            self.assertIn("free 1B", report.error)
            self.assertIn("needs ~", report.error)
            step = _step(report, "disk_space")
            self.assertIsNotNone(step)
            self.assertFalse(step["ok"])
            # Refused BEFORE the snapshot, the pull, the quiesce:
            self.assertEqual(pull_calls, [])
            self.assertEqual(mgr.snapshots.list(), [])
            self.assertIsNone(_step(report, "snapshot"))
            self.assertIsNone(_step(report, "quiesce"))

    def test_ample_space_records_ok_and_update_completes(self):
        from nomorals.core import diskspace

        with tempfile.TemporaryDirectory() as tmp:
            mgr, pull_calls = _scripted(tmp, _snapshots(tmp))
            with mock.patch.object(diskspace, "free_bytes",
                                   return_value=10 ** 15):
                report = mgr.run()
            self.assertTrue(report.ok, report.error)
            step = _step(report, "disk_space")
            self.assertIsNotNone(step)
            self.assertTrue(step["ok"])
            self.assertIn("free ", step["detail"])
            self.assertIn("needs ~", step["detail"])
            self.assertEqual(pull_calls, ["pull"])

    def test_measurement_failure_proceeds_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _scripted(tmp, _snapshots(tmp))
            with mock.patch(
                    "nomorals.core.diskspace.disk_report",
                    side_effect=OSError("no such volume")):
                ok, detail = mgr._preflight_disk()
            self.assertTrue(ok)
            self.assertIn("could not measure", detail)


# ── Fix 1b: retention enforced by the update run ─────────────────────

class TestUpdateRetention(unittest.TestCase):
    def _run_with_pile(self, tmp: str, pile: int = 4):
        from nomorals.core import diskspace

        snapshots = _snapshots(tmp)
        for _ in range(pile):
            snapshots.create(label="pre-update")
        before = [s.id for s in snapshots.list()
                  if s.label == "pre-update"]
        mgr, _ = _scripted(tmp, snapshots)
        with mock.patch.object(diskspace, "free_bytes",
                               return_value=10 ** 15):
            report = mgr.run()
        self.assertTrue(report.ok, report.error)
        return mgr, snapshots, before, report

    def test_run_prunes_the_pile_to_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, snapshots, before, report = self._run_with_pile(tmp)
            remaining = [s.id for s in snapshots.list()
                         if s.label == "pre-update"]
            # The 4 old + this run's new one, bounded to newest 3.
            self.assertEqual(len(remaining), 3)
            self.assertIn(report.pre_update_snapshot, remaining)
            self.assertNotIn(before[0], remaining)
            self.assertNotIn(before[1], remaining)
            journal = Path(mgr.journal_path).read_text(
                encoding="utf-8")
            self.assertIn('"retention"', journal)

    def test_pending_restore_anchor_survives_the_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            from nomorals.core import diskspace

            snapshots = _snapshots(tmp)
            for _ in range(4):
                snapshots.create(label="pre-update")
            before = [s.id for s in snapshots.list()
                      if s.label == "pre-update"]
            mgr, _ = _scripted(tmp, snapshots)
            mgr._write_pending_restore(before[0], "pin: deferred")
            with mock.patch.object(diskspace, "free_bytes",
                                   return_value=10 ** 15):
                report = mgr.run()
            self.assertTrue(report.ok, report.error)
            remaining = [s.id for s in snapshots.list()
                         if s.label == "pre-update"]
            self.assertIn(before[0], remaining)
            self.assertEqual(len(remaining), 4)

    def test_prune_failure_journals_and_update_still_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            from nomorals.core import diskspace

            snapshots = _snapshots(tmp)
            mgr, _ = _scripted(tmp, snapshots)
            with mock.patch.object(
                    snapshots, "prune_pre_update",
                    side_effect=OSError("locked")), \
                    mock.patch.object(diskspace, "free_bytes",
                                      return_value=10 ** 15):
                report = mgr.run()
            self.assertTrue(report.ok, report.error)
            journal = Path(mgr.journal_path).read_text(
                encoding="utf-8")
            self.assertIn('"retention_failed"', journal)


# ── Fix 3a: the doctor disk line ─────────────────────────────────────

class TestDoctorDiskLine(unittest.TestCase):
    def _runtime(self, tmp_home: str):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self):
                super().__init__(media_dir="/tmp/nm-test-media")

            def run(self, handler):
                pass

            def send(self, chat, text, *, reply_to="", buttons=None,
                     **kwargs):
                return SendResult(ok=True, platform="telegram",
                                  message_id="m1")

        settings = load_settings(overrides={
            "home": tmp_home,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)
        gateway = ChatGateway({"telegram": FakeAdapter()},
                              db=context.db)
        return PartnerRuntime(context, gateway=gateway)

    def test_doctor_disk_line_ok_when_ample(self):
        from nomorals.core import diskspace

        with tempfile.TemporaryDirectory(
                prefix="nm-test-doctor-") as tmp:
            rt = self._runtime(tmp)
            with mock.patch.dict(__import__("os").environ, {},
                                 clear=True), \
                    mock.patch.object(diskspace, "free_bytes",
                                      return_value=10 ** 15):
                out = rt.handle_control("/doctor",
                                        "telegram:5478650254")
        self.assertIn("disk:", out)
        self.assertNotIn("FAIL disk:", out)

    def test_doctor_disk_line_fails_low_with_the_numbers(self):
        from nomorals.core import diskspace

        with tempfile.TemporaryDirectory(
                prefix="nm-test-doctor-") as tmp:
            rt = self._runtime(tmp)
            with mock.patch.dict(__import__("os").environ, {},
                                 clear=True), \
                    mock.patch.object(diskspace, "free_bytes",
                                      return_value=1):
                out = rt.handle_control("/doctor",
                                        "telegram:5478650254")
        self.assertIn("FAIL disk:", out)
        self.assertIn("LOW", out)


# ── Fix 3b: the housekeeping disk watch ──────────────────────────────

class TestDiskWatch(unittest.TestCase):
    def test_journals_once_per_crossing_per_day(self):
        from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
        from nomorals.core import diskspace

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db = _make_db(home / "devon.db", payload_kb=128)
            notes: list[dict] = []
            personal = SimpleNamespace(
                note_disk_low=lambda **kw: notes.append(kw))
            fake_self = SimpleNamespace(context=SimpleNamespace(
                settings=SimpleNamespace(
                    home_path=home, db_path=db, blob_dir=None)))
            svc: dict = {"personal": personal}
            watch = RuntimeProvisionMixin._disk_watch
            with mock.patch.object(diskspace, "free_bytes",
                                   return_value=1):
                watch(fake_self, svc)
                watch(fake_self, svc)
            self.assertEqual(len(notes), 1)
            self.assertGreater(notes[0]["need_bytes"], 0)
            self.assertEqual(notes[0]["free_bytes"], 1)
            # Recovery clears the stamp; a re-crossing re-journals.
            with mock.patch.object(diskspace, "free_bytes",
                                   return_value=10 ** 15):
                watch(fake_self, svc)
            self.assertNotIn("disk_low_day", svc)
            with mock.patch.object(diskspace, "free_bytes",
                                   return_value=1):
                watch(fake_self, svc)
            self.assertEqual(len(notes), 2)


class TestNoteDiskLow(unittest.TestCase):
    def test_row_lands_in_the_vault_journal(self):
        from nomorals.social.personal import PersonalSurfaces, PersonalVault

        with tempfile.TemporaryDirectory() as tmp:
            vault = PersonalVault(Path(tmp) / "personal")
            fake_self = SimpleNamespace(vault=vault, _clock=time.time)
            entry = PersonalSurfaces.note_disk_low(
                fake_self, free_bytes=1234, need_bytes=999_999,
                store_count=7, store_bytes=555_000)
            self.assertEqual(entry["kind"], "disk_low")
            rows = vault.journal("", "disk_low")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["free_bytes"], 1234)
            self.assertEqual(rows[0]["store_count"], 7)
            self.assertIn("disk low", rows[0]["note"])


if __name__ == "__main__":
    unittest.main()
