"""Tests for multi-device sync: store, LWW, engine."""
from __future__ import annotations

import time
import unittest

from nomorals.storage.db import Database
from nomorals.sync import LocalPeer, SyncEngine, SyncRecord, SyncStore


def _db() -> Database:
    return Database(":memory:")


class SyncRecordTests(unittest.TestCase):
    def test_wins_last_write(self):
        a = SyncRecord("k", {"v": 1}, updated_at=100.0, device_id="a")
        b = SyncRecord("k", {"v": 2}, updated_at=200.0, device_id="b")
        self.assertIs(SyncRecord.wins(a, b), b)
        self.assertIs(SyncRecord.wins(b, a), b)

    def test_wins_tie_breaks_on_device_id(self):
        a = SyncRecord("k", {"v": 1}, updated_at=100.0, device_id="a")
        b = SyncRecord("k", {"v": 2}, updated_at=100.0, device_id="b")
        # Deterministic: higher device_id wins ties.
        self.assertIs(SyncRecord.wins(a, b), b)
        self.assertIs(SyncRecord.wins(b, a), b)


class SyncStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = SyncStore(_db(), device_id="dev1")

    def test_put_and_get(self):
        self.store.put("theme", {"mode": "dark"})
        rec = self.store.get("theme")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.value["mode"], "dark")
        self.assertEqual(rec.device_id, "dev1")

    def test_put_requires_key(self):
        with self.assertRaises(ValueError):
            self.store.put("", {"v": 1})

    def test_delete_is_tombstone(self):
        self.store.put("k", {"v": 1})
        self.store.delete("k")
        self.assertIsNone(self.store.get("k"))
        # Tombstone still visible to replication.
        rec = self.store.get_record("k")
        self.assertIsNotNone(rec)
        self.assertTrue(rec.deleted)

    def test_apply_newer_wins(self):
        self.store.put("k", {"v": 1}, updated_at=100.0)
        remote = SyncRecord("k", {"v": 2}, updated_at=200.0,
                            device_id="dev2")
        self.assertTrue(self.store.apply(remote))
        self.assertEqual(self.store.get("k").value["v"], 2)

    def test_apply_older_loses(self):
        self.store.put("k", {"v": 2}, updated_at=200.0)
        remote = SyncRecord("k", {"v": 1}, updated_at=100.0,
                            device_id="dev2")
        self.assertFalse(self.store.apply(remote))
        self.assertEqual(self.store.get("k").value["v"], 2)

    def test_changed_since(self):
        t0 = time.time()
        self.store.put("a", {"v": 1})
        time.sleep(0.01)
        t1 = time.time()
        self.store.put("b", {"v": 2})
        changed = self.store.list_changed_since(t1)
        self.assertEqual([r.key for r in changed], ["b"])
        changed = self.store.list_changed_since(t0 - 1)
        self.assertEqual(len(changed), 2)


class SyncEngineTests(unittest.TestCase):
    def _pair(self):
        db_a, db_b = _db(), _db()
        store_a = SyncStore(db_a, device_id="a")
        store_b = SyncStore(db_b, device_id="b")
        eng_a = SyncEngine(db_a, store_a)
        return eng_a, store_a, store_b

    def test_two_way_sync(self):
        eng_a, store_a, store_b = self._pair()
        store_a.put("from_a", {"v": 1})
        store_b.put("from_b", {"v": 2})
        result = eng_a.sync(LocalPeer(store_b))
        self.assertEqual(result.pushed, 1)
        self.assertEqual(result.pulled, 1)
        # Both sides now have both keys.
        self.assertIsNotNone(store_a.get("from_b"))
        self.assertIsNotNone(store_b.get("from_a"))

    def test_sync_is_incremental(self):
        eng_a, store_a, store_b = self._pair()
        store_a.put("k", {"v": 1})
        r1 = eng_a.sync(LocalPeer(store_b))
        self.assertEqual(r1.pushed, 1)
        r2 = eng_a.sync(LocalPeer(store_b))
        self.assertEqual(r2.pushed, 0)
        self.assertEqual(r2.pulled, 0)

    def test_conflict_last_write_wins(self):
        eng_a, store_a, store_b = self._pair()
        store_a.put("k", {"v": "a"}, updated_at=100.0)
        store_b.put("k", {"v": "b"}, updated_at=200.0)
        result = eng_a.sync(LocalPeer(store_b))
        self.assertEqual(result.conflicts_resolved, 1)
        self.assertEqual(store_a.get("k").value["v"], "b")

    def test_delete_replicates(self):
        eng_a, store_a, store_b = self._pair()
        store_a.put("k", {"v": 1})
        eng_a.sync(LocalPeer(store_b))
        self.assertIsNotNone(store_b.get("k"))
        store_a.delete("k")
        eng_a.sync(LocalPeer(store_b))
        self.assertIsNone(store_b.get("k"))

    def test_status(self):
        eng_a, store_a, _ = self._pair()
        store_a.put("k", {"v": 1})
        st = eng_a.status()
        self.assertEqual(st["device_id"], "a")
        self.assertEqual(st["local_keys"], 1)
        self.assertEqual(st["pending_push"], 1)


class SyncSeqTests(unittest.TestCase):
    """Seq-based replication cursors (R22): the cursor is a monotonic local
    sequence, not a timestamp, so backdated writes and clock skew can never
    slip past it unseen."""

    def _pair(self):
        db_a, db_b = _db(), _db()
        store_a = SyncStore(db_a, device_id="a")
        store_b = SyncStore(db_b, device_id="b")
        eng_a = SyncEngine(db_a, store_a)
        return eng_a, store_a, store_b

    def test_seq_assigned_in_write_order(self):
        store = SyncStore(_db(), device_id="a")
        r1 = store.put("a", {"v": 1})
        r2 = store.put("b", {"v": 2})
        self.assertEqual((r1.seq, r2.seq), (1, 2))
        got = store.list_since_seq(1)
        self.assertEqual([r.key for r in got], ["b"])

    def test_backdated_write_still_pushes(self):
        # Under the old timestamp cursor this write would be silently
        # dropped: its updated_at predates the push mark.
        eng_a, store_a, store_b = self._pair()
        store_a.put("k1", {"v": 1})
        eng_a.sync(LocalPeer(store_b))
        store_a.put("k2", {"v": 2}, updated_at=100.0)  # clock skew / import
        result = eng_a.sync(LocalPeer(store_b))
        self.assertEqual(result.pushed, 1)
        rec = store_b.get("k2")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.value["v"], 2)
        # ...and LWW still resolves by (updated_at, device_id), not seq:
        # the peer's newer record wins on the next pull.
        store_b.put("k2", {"v": "peer"}, updated_at=time.time())
        result = eng_a.sync(LocalPeer(store_b))
        self.assertEqual(store_a.get("k2").value["v"], "peer")

    def test_hub_fans_out_backdated_write(self):
        # The old timestamp cursor missed this: a backdated record pulled
        # by the hub after a peer's pull cursor had already advanced past
        # its timestamp. The seq cursor cannot miss it.
        db_hub, db_a, db_b = _db(), _db(), _db()
        hub = SyncStore(db_hub, device_id="hub")
        store_a = SyncStore(db_a, device_id="a")
        store_b = SyncStore(db_b, device_id="b")
        eng_hub = SyncEngine(db_hub, hub)
        eng_b = SyncEngine(db_b, store_b)
        # Seed a normal record so b's pull cursor advances past the
        # backdated timestamp used below.
        hub.put("seed", {"v": 0})
        r = eng_b.sync(LocalPeer(hub), peer_id="hub")
        self.assertEqual(r.pulled, 1)
        # Hub pulls a backdated record from a (clock skew / import).
        store_a.put("k", {"v": 1}, updated_at=100.0)
        eng_a = SyncEngine(db_a, store_a)
        r = eng_hub.sync(LocalPeer(store_a), peer_id="a")
        self.assertEqual(r.pulled, 1)
        # b must still receive it.
        r = eng_b.sync(LocalPeer(hub), peer_id="hub")
        self.assertEqual(r.pulled, 1)
        self.assertEqual(store_b.get("k").value["v"], 1)

    def test_apply_advances_local_seq_but_identical_is_noop(self):
        store = SyncStore(_db(), device_id="a")
        store.put("k", {"v": 1})
        before = store.max_seq()
        store.apply(SyncRecord("j", {"v": 2}, updated_at=500.0,
                               device_id="b"))
        self.assertGreater(store.max_seq(), before)
        # Re-applying the identical record changes nothing (no echo churn).
        cur = store.max_seq()
        changed = store.apply(SyncRecord("j", {"v": 2}, updated_at=500.0,
                                         device_id="b"))
        self.assertFalse(changed)
        self.assertEqual(store.max_seq(), cur)

    def test_legacy_peer_falls_back_to_timestamp_fetch(self):
        from nomorals.sync.engine import SyncPeer

        class LegacyPeer(SyncPeer):
            """Implements only the original timestamp fetch."""
            def __init__(self, store):
                self.store = store

            def push_records(self, records):
                n = 0
                for rec in records:
                    if self.store.apply(rec):
                        n += 1
                return n

            def fetch_since(self, since):
                return self.store.list_changed_since(since)

        eng_a, store_a, store_b = self._pair()
        store_a.put("k", {"v": 1})
        store_b.put("j", {"v": 2})
        result = eng_a.sync(LegacyPeer(store_b))
        self.assertEqual(result.pushed, 1)
        self.assertEqual(result.pulled, 1)
        self.assertIsNotNone(store_a.get("j"))
        self.assertIsNotNone(store_b.get("k"))

    def test_seq_migration_from_legacy_schema(self):
        # Simulate a pre-R22 database: no seq column, timestamp progress.
        db = Database(":memory:")
        db.execute(
            """CREATE TABLE sync_records (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL DEFAULT 0,
                device_id TEXT NOT NULL DEFAULT '',
                deleted INTEGER NOT NULL DEFAULT 0)""")
        db.execute(
            "INSERT INTO sync_records (key, value, updated_at, device_id)"
            " VALUES ('a', '{}', 100.0, 'x')")
        db.execute(
            "INSERT INTO sync_records (key, value, updated_at, device_id)"
            " VALUES ('b', '{}', 200.0, 'x')")
        db.execute(
            """CREATE TABLE sync_progress (
                peer_id TEXT PRIMARY KEY,
                last_push REAL NOT NULL DEFAULT 0,
                last_pull REAL NOT NULL DEFAULT 0)""")
        db.execute(
            "INSERT INTO sync_progress (peer_id, last_push, last_pull)"
            " VALUES ('hub', 150.0, 0)")

        store = SyncStore(db, device_id="x")
        recs = store.list_since_seq(0)
        self.assertEqual([r.key for r in recs], ["a", "b"])
        self.assertEqual([r.seq for r in recs], [1, 2])

        eng = SyncEngine(db, store)
        st = eng.status("hub")
        # push cursor migrated: only 'a' (updated_at 100 <= 150) counts.
        self.assertEqual(st["push_seq"], 1)
        self.assertEqual(st["pending_push"], 1)  # 'b' still unsynced

        # New writes continue the sequence with no collisions.
        rec = store.put("c", {"v": 3})
        self.assertEqual(rec.seq, 3)


class SyncCLITests(unittest.TestCase):
    """R22: the sync CLI store verbs and the peer-db sync path."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        db = Database(str(self.root / "dev.db"))
        self.ctx = SimpleNamespace(
            db=db, device_id="dev",
            settings=SimpleNamespace(workspace_dir=str(self.root)))

    def _run(self, *words, **kw):
        import io
        from contextlib import redirect_stdout, redirect_stderr
        from types import SimpleNamespace
        from nomorals.cmdline.commands.sync import _cmd_sync
        args = SimpleNamespace(task=list(words), json=kw.get("json", False),
                               peer_db=kw.get("peer_db", ""))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = _cmd_sync(args, self.ctx)
        return rc, out.getvalue(), err.getvalue()

    def test_put_get_delete_keys(self):
        rc, out, _ = self._run("put", "theme", '{"mode": "dark"}')
        self.assertEqual(rc, 0)
        self.assertIn("seq=1", out)
        rc, out, _ = self._run("get", "theme")
        self.assertEqual(rc, 0)
        self.assertIn("dark", out)
        rc, out, _ = self._run("keys")
        self.assertEqual(rc, 0)
        self.assertIn("theme", out)
        rc, out, _ = self._run("delete", "theme")
        self.assertEqual(rc, 0)
        self.assertIn("tombstone", out)
        rc, _, err = self._run("get", "theme")
        self.assertEqual(rc, 1)
        self.assertIn("no such key", err)

    def test_put_rejects_bad_json(self):
        rc, _, err = self._run("put", "k", "not-json")
        self.assertEqual(rc, 2)
        self.assertIn("invalid JSON", err)
        rc, _, err = self._run("put", "k", "[1, 2]")
        self.assertEqual(rc, 2)
        self.assertIn("must be a JSON object", err)

    def test_push_pull_two_devices(self):
        # R22: peer_id used to be a PosixPath → SQL bind error.
        from types import SimpleNamespace
        from nomorals.cmdline.commands.sync import _cmd_sync
        import io
        from contextlib import redirect_stdout, redirect_stderr
        peer_path = str(self.root / "hub.db")
        peer_db = Database(peer_path)
        peer_ctx = SimpleNamespace(
            db=peer_db, device_id="hub",
            settings=SimpleNamespace(workspace_dir=str(self.root)))

        def run(ctx, *words, **kw):
            args = SimpleNamespace(task=list(words),
                                   json=kw.get("json", False),
                                   peer_db=kw.get("peer_db", ""))
            out = io.StringIO()
            with redirect_stdout(out):
                rc = _cmd_sync(args, ctx)
            return rc, out.getvalue()

        self._run("put", "reminder", '{"text": "hi"}')
        rc, out = run(self.ctx, "push", peer_db=peer_path)
        self.assertEqual(rc, 0, out)
        self.assertIn("pushed 1", out)
        rc, out = run(peer_ctx, "get", "reminder")
        self.assertEqual(rc, 0, out)
        self.assertIn("hi", out)
        # Incremental: nothing new → zero counts, not an error.
        rc, out = run(self.ctx, "push", peer_db=peer_path)
        self.assertIn("pushed 0", out)


if __name__ == "__main__":
    unittest.main()
