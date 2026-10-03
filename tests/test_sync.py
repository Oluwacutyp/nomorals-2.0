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


if __name__ == "__main__":
    unittest.main()
