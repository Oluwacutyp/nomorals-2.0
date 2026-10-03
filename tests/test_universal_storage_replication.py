"""Tests for the Litestream-style SQLite replicator."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nomorals.core.errors import StorageError
from nomorals.storage.db import Database
from nomorals.storage.replication import SnapshotInfo, SqliteReplicator


def make_db(path: Path) -> Database:
    db = Database(str(path))
    db.migrate()
    return db


class TestSqliteReplicator(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = make_db(self.root / "source.db")
        self.db.insert(
            "memories",
            {
                "id": "mem-1",
                "kind": "fact",
                "content": "replication test",
                "created_at": 1_700_000_000.0,
                "updated_at": 1_700_000_000.0,
            },
        )
        self.replicator = SqliteReplicator(
            self.db, self.root / "replicas", name="test", keep=3
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_snapshot_is_consistent_and_verified(self) -> None:
        info = self.replicator.snapshot()
        self.assertTrue(Path(info.path).is_file())
        self.assertGreater(info.pages, 0)
        self.assertGreater(info.size, 0)
        self.assertEqual(len(info.sha256), 64)
        # Sidecar written alongside.
        sidecar = Path(info.path + ".replica.json")
        self.assertTrue(sidecar.is_file())
        reloaded = SnapshotInfo.from_dict(json.loads(sidecar.read_text()))
        self.assertEqual(reloaded.sha256, info.sha256)

    def test_snapshot_data_matches_source(self) -> None:
        info = self.replicator.snapshot()
        conn = sqlite3.connect(info.path)
        try:
            row = conn.execute(
                "SELECT content FROM memories WHERE id = 'mem-1'"
            ).fetchone()
            self.assertEqual(row[0], "replication test")
            result = conn.execute("PRAGMA integrity_check").fetchone()
            self.assertEqual(result[0], "ok")
        finally:
            conn.close()

    def test_latest_and_list(self) -> None:
        self.assertIsNone(self.replicator.latest())
        first = self.replicator.snapshot()
        time.sleep(1.1)
        second = self.replicator.snapshot()
        snapshots = self.replicator.list_snapshots()
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0].name, first.name)
        latest = self.replicator.latest()
        assert latest is not None
        self.assertEqual(latest.name, second.name)

    def test_rotate_keeps_n_generations(self) -> None:
        for _ in range(5):
            self.replicator.snapshot()
            time.sleep(1.1)
        removed = self.replicator.rotate()
        self.assertEqual(len(removed), 2)
        self.assertEqual(len(self.replicator.list_snapshots()), 3)

    def test_restore(self) -> None:
        info = self.replicator.snapshot()
        target = self.root / "restored.db"
        out = self.replicator.restore(info, target=target)
        self.assertEqual(out, target)
        conn = sqlite3.connect(str(target))
        try:
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()

    def test_restore_latest_by_default(self) -> None:
        self.replicator.snapshot()
        out = self.replicator.restore()
        self.assertTrue(out.is_file())

    def test_restore_missing_snapshot_raises(self) -> None:
        with self.assertRaises(StorageError):
            self.replicator.restore("nope-00000000-000000.db")

    def test_replicate_once_ships(self) -> None:
        shipped: list[Path] = []
        replicator = SqliteReplicator(
            self.db, self.root / "replicas2", name="ship", keep=3,
            ship=shipped.append,
        )
        info = replicator.replicate_once()
        self.assertEqual(shipped, [Path(info.path)])
        self.assertEqual(replicator.stats["shipped"], 1)

    def test_replicate_once_ship_failure_raises(self) -> None:
        def boom(path: Path) -> None:
            raise RuntimeError("disk full")

        replicator = SqliteReplicator(
            self.db, self.root / "replicas3", ship=boom
        )
        with self.assertRaises(StorageError):
            replicator.replicate_once()

    def test_run_loop_replicates_until_stopped(self) -> None:
        stop = threading.Event()
        replicator = SqliteReplicator(
            self.db, self.root / "replicas4", name="loop", keep=5
        )
        thread = threading.Thread(
            target=replicator.run_loop, args=(1,), kwargs={"should_stop": stop.is_set},
            daemon=True,
        )
        thread.start()
        time.sleep(2.6)
        stop.set()
        thread.join(timeout=10)
        snapshots = replicator.list_snapshots()
        self.assertGreaterEqual(len(snapshots), 2)
        self.assertGreaterEqual(replicator.stats["snapshots"], 2)

    def test_snapshot_on_closed_db_raises(self) -> None:
        self.db.close()
        with self.assertRaises(StorageError):
            self.replicator.snapshot()

    def test_stats_snapshot(self) -> None:
        self.replicator.snapshot()
        snap = self.replicator.stats_snapshot()
        self.assertEqual(snap["generations"], 1)
        self.assertGreater(snap["bytes"], 0)
        self.assertIsNotNone(snap["latest"])
        self.assertEqual(snap["keep"], 3)


if __name__ == "__main__":
    unittest.main()
