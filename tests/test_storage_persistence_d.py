"""Section D (persistence) tests: KVStore, migration-engine hardening, Database upgrades."""

import os
import tempfile
import threading
import time
import unittest

from nomorals.storage.db import Database
from nomorals.storage.kv import KVStore
from nomorals.storage.schema import Migration, MigrationRunner


def make_db(**kwargs):
    db = Database(":memory:", **kwargs)
    db.execute(
        "CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'json', updated_at REAL NOT NULL, "
        "expires_at REAL)"
    )
    return db


class TestKVStore(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.kv = KVStore(self.db)

    def tearDown(self):
        self.db.close()

    def test_set_get_types(self):
        self.kv.set("s", "hello")
        self.kv.set("n", 42)
        self.kv.set("f", 1.5)
        self.kv.set("lst", [1, 2, 3])
        self.kv.set("d", {"a": 1})
        self.assertEqual(self.kv.get("s"), "hello")
        self.assertEqual(self.kv.get("n"), 42)
        self.assertEqual(self.kv.get("lst"), [1, 2, 3])
        self.assertEqual(self.kv.get("d"), {"a": 1})
        self.assertEqual(self.kv.get("missing", default="dflt"), "dflt")
        self.assertIsNone(self.kv.get("missing"))

    def test_typed_getters(self):
        self.kv.set("i", 7)
        self.kv.set("f", 2.5)
        self.kv.set("b", True)
        self.kv.set("s", "txt")
        self.assertEqual(self.kv.get_int("i"), 7)
        self.assertEqual(self.kv.get_float("f"), 2.5)
        self.assertTrue(self.kv.get_bool("b"))
        self.assertEqual(self.kv.get_str("s"), "txt")
        self.assertEqual(self.kv.get_int("nope", default=99), 99)
        # Legacy plain-string rows written by raw SQL callers.
        self.db.execute(
            "INSERT INTO kv_store (key, value, kind, updated_at) VALUES ('leg', '123', 'text', 0)"
        )
        self.assertEqual(self.kv.get_int("leg"), 123)
        # "123" is valid JSON, so get() decodes it; get_str keeps it verbatim.
        self.assertEqual(self.kv.get("leg"), 123)
        self.assertEqual(self.kv.get_str("leg"), "123")

    def test_namespaces(self):
        a = self.kv.namespaced("arena")
        b = self.kv.namespaced("ops")
        a.set("k", "va")
        b.set("k", "vb")
        self.kv.set("k", "global")
        self.assertEqual(a.get("k"), "va")
        self.assertEqual(b.get("k"), "vb")
        self.assertEqual(self.kv.get("k"), "global")
        # Raw key layout stays greppable.
        self.assertEqual(self.kv.get("arena:k"), "va")
        self.assertEqual(a.keys(), ["k"])
        nested = a.namespaced("sub")
        nested.set("x", 1)
        self.assertEqual(nested.get("x"), 1)
        self.assertEqual(self.kv.get("arena:sub:x"), 1)

    def test_ttl_expiry_and_sweep(self):
        tick = [1000.0]
        kv = KVStore(self.db, now_fn=lambda: tick[0])
        kv.set("immortal", "x")
        kv.set("short", "y", ttl=10)
        self.assertEqual(kv.get("short"), "y")
        self.assertAlmostEqual(kv.ttl("short"), 10, places=1)
        self.assertIsNone(kv.ttl("immortal"))
        tick[0] += 11
        self.assertIsNone(kv.get("short"))  # lazy expiry on read
        self.assertFalse(kv.exists("short"))
        self.assertEqual(kv.delete_expired(), 1)
        self.assertEqual(kv.get("immortal"), "x")
        # ttl=0 deletes immediately
        kv.set("z", 1, ttl=0)
        self.assertFalse(kv.exists("z"))

    def test_expire_and_persist(self):
        tick = [500.0]
        kv = KVStore(self.db, now_fn=lambda: tick[0])
        kv.set("k", "v")
        self.assertTrue(kv.expire("k", 30))
        self.assertAlmostEqual(kv.ttl("k"), 30, places=1)
        self.assertTrue(kv.persist("k"))
        self.assertIsNone(kv.ttl("k"))
        self.assertFalse(kv.expire("missing", 10))

    def test_batch(self):
        self.assertEqual(self.kv.set_many({"a": 1, "b": "two", "c": [3]}), 3)
        got = self.kv.get_many(["a", "b", "c", "missing"])
        self.assertEqual(got, {"a": 1, "b": "two", "c": [3]})
        self.assertEqual(self.kv.get_many([]), {})

    def test_cas(self):
        self.assertTrue(self.kv.compare_and_set("k", None, "first"))
        self.assertFalse(self.kv.compare_and_set("k", None, "second"))
        self.assertTrue(self.kv.compare_and_set("k", "first", "second"))
        self.assertFalse(self.kv.compare_and_set("k", "first", "third"))
        self.assertEqual(self.kv.get("k"), "second")
        self.assertFalse(self.kv.compare_and_set("absent", "x", "y"))

    def test_incr_decr(self):
        self.assertEqual(self.kv.incr("n"), 1)
        self.assertEqual(self.kv.incr("n", 4), 5)
        self.assertEqual(self.kv.decr("n", 2), 3)
        # Tolerates legacy plain-string numbers.
        self.db.execute(
            "INSERT INTO kv_store (key, value, kind, updated_at) VALUES ('m', '10', 'text', 0)"
        )
        self.assertEqual(self.kv.incr("m"), 11)

    def test_scan_count_delete(self):
        ns = self.kv.namespaced("s")
        ns.set_many({"a1": 1, "a2": 2, "b1": 3})
        self.assertEqual(ns.count(), 3)
        self.assertEqual(ns.count("a"), 2)
        self.assertEqual([k for k, _ in ns.scan("a")], ["a1", "a2"])
        self.assertEqual(ns.delete_prefix("a"), 2)
        self.assertEqual(ns.count(), 1)
        self.assertTrue(ns.delete("b1"))
        self.assertFalse(ns.delete("b1"))
        self.assertEqual(ns.clear_namespace(), 0)

    def test_delete_expired_scoped(self):
        tick = [0.0]
        a = KVStore(self.db, namespace="a", now_fn=lambda: tick[0])
        b = KVStore(self.db, namespace="b", now_fn=lambda: tick[0])
        a.set("x", 1, ttl=5)
        b.set("x", 2, ttl=5)
        tick[0] = 10
        self.assertEqual(a.delete_expired(), 1)
        # b's row is untouched by a's sweep (still in the table, merely expired
        # so lazy reads hide it) — namespace isolation holds.
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM kv_store"), 1)
        self.assertFalse(b.exists("x"))
        self.assertEqual(b.delete_expired(), 1)
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM kv_store"), 0)

    def test_stats_and_namespaces(self):
        self.kv.namespaced("n1").set("k", 1, ttl=60)
        self.kv.namespaced("n2").set("k", 2)
        self.assertEqual(sorted(self.kv.namespaces()), ["n1", "n2"])
        stats = self.kv.namespaced("n1").stats()
        self.assertEqual(stats["keys"], 1)
        self.assertEqual(stats["with_ttl"], 1)


class TestMigrationHardening(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.runner = MigrationRunner(self.db)

    def tearDown(self):
        self.db.close()

    def test_source_checksum_column(self):
        cols = {r["name"] for r in self.db.query("PRAGMA table_info(schema_migrations)")}
        self.assertIn("source_checksum", cols)
        self.runner.apply_all([Migration(1, "a", sql="CREATE TABLE t(x)")])
        row = self.db.query_one("SELECT checksum, source_checksum FROM schema_migrations")
        self.assertEqual(row["checksum"], row["source_checksum"])
        self.assertEqual(len(row["source_checksum"]), 16)

    def test_repair_flow(self):
        self.runner.apply_all([Migration(1, "a", sql="CREATE TABLE t(x)")])
        edited = Migration(1, "a", sql="CREATE TABLE t(x, y)")
        self.assertTrue(self.runner.validate([edited]))
        with self.assertRaises(Exception):
            self.runner.apply_all([edited], strict=True)
        repaired = self.runner.repair([edited])
        self.assertEqual(repaired, ["0001_a"])
        self.assertEqual(self.runner.validate([edited]), [])
        self.assertEqual(self.runner.apply_all([edited]), [])

    def test_baseline(self):
        ms = [
            Migration(1, "a", sql="CREATE TABLE t1(x)"),
            Migration(2, "b", sql="CREATE TABLE t2(x)"),
            Migration(3, "c", sql="CREATE TABLE t3(x)"),
        ]
        stamped = self.runner.baseline(ms, 2)
        self.assertEqual(len(stamped), 2)
        self.assertEqual(self.runner.current_version(), 2)
        # Pending now only contains migration 3.
        self.assertEqual([m.version for m in self.runner.pending(ms)], [3])
        with self.assertRaises(Exception):
            self.runner.baseline(ms, 3)  # already has history, no force

    def test_dry_run_and_status(self):
        ms = [Migration(1, "a", sql="CREATE TABLE t(x); CREATE INDEX i ON t(x);")]
        plan = self.runner.dry_run(ms)
        self.assertEqual(plan[0]["statement_count"], 2)
        self.assertFalse(self.db.table_exists("t"))  # nothing executed
        status = self.runner.status(ms)
        self.assertEqual(status["current_version"], 0)
        self.assertEqual(status["pending"], ["0001_a"])
        self.assertEqual(status["problems"], [])

    def test_concurrent_apply_runs_once(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "race.db")
        db = Database(path)
        ms = [
            Migration(
                1, "t",
                sql="CREATE TABLE IF NOT EXISTS lt(x); INSERT INTO lt VALUES (1)",
            )
        ]
        results = []
        lock = threading.Lock()

        def worker():
            try:
                out = MigrationRunner(db).apply_all(ms, lock_timeout_s=15)
                with lock:
                    results.append(out)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    results.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        applied_total = sum(len(r) for r in results if isinstance(r, list))
        self.assertEqual(applied_total, 1)
        self.assertEqual(db.scalar("SELECT COUNT(*) FROM lt"), 1)
        db.close()


class TestDatabaseUpgrades(unittest.TestCase):
    def test_backup_to_verifies(self):
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "src.db")
        dst = os.path.join(tmp, "dst.db")
        db = Database(src)
        db.execute("CREATE TABLE t(x)")
        db.execute("INSERT INTO t VALUES (1)")
        out = db.backup_to(dst)
        self.assertTrue(out.exists())
        probe = Database(dst, readonly=True)
        self.assertEqual(probe.scalar("SELECT COUNT(*) FROM t"), 1)
        self.assertEqual(probe.integrity_check().strip().lower(), "ok")
        probe.close()
        db.close()

    def test_backup_to_memory_db(self):
        tmp = tempfile.mkdtemp()
        db = Database(":memory:")
        db.execute("CREATE TABLE t(x)")
        out = db.backup_to(os.path.join(tmp, "m.db"))
        self.assertTrue(out.exists())
        db.close()

    def test_wal_autocheckpoint_and_journal_mode(self):
        tmp = tempfile.mkdtemp()
        db = Database(os.path.join(tmp, "w.db"), wal_autocheckpoint=500)
        self.assertEqual(db.journal_mode().lower(), "wal")
        self.assertEqual(int(db.scalar("PRAGMA wal_autocheckpoint")), 500)
        db.close()

    def test_slow_query_stat(self):
        db = Database(":memory:", slow_query_threshold_s=0.0)
        db.execute("SELECT 1")
        self.assertGreaterEqual(db.stats["slow_queries"], 1)
        db.close()
        db2 = Database(":memory:", slow_query_threshold_s=None)
        db2.execute("SELECT 1")
        self.assertEqual(db2.stats["slow_queries"], 0)
        db2.close()

    def test_stats_snapshot_fields(self):
        tmp = tempfile.mkdtemp()
        db = Database(os.path.join(tmp, "s.db"))
        snap = db.stats_snapshot()
        for key in ("page_count", "freelist_count", "page_size", "wal_size_bytes",
                    "journal_mode", "slow_queries"):
            self.assertIn(key, snap)
        db.close()

    def test_optimize_runs(self):
        db = Database(":memory:")
        db.execute("CREATE TABLE t(x)")
        db.optimize()  # must not raise
        db.close()

    def test_migrate_reaches_86(self):
        from nomorals.storage.migrations import MIGRATIONS, latest_version

        self.assertEqual(latest_version(), 86)
        tmp = tempfile.mkdtemp()
        db = Database(os.path.join(tmp, "mig.db"))
        summary = db.migrate()
        self.assertEqual(summary.version, 86)
        cols = {c["name"] for c in db.table_info("kv_store")}
        self.assertIn("expires_at", cols)
        db.close()


if __name__ == "__main__":
    unittest.main()
