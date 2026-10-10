"""Sweep tests for the nomorals.storage system-wide upgrade.

Covers the new capability added in the storage sweep: WAL health monitoring
and async offload (db), Redis-grade KV ergonomics (kv), range reads / trash /
cross-store sync (blob), presigned URLs + multipart (s3blob), real k-means
IVF with persistence + recall eval (vectors), tokenizer choice / facets /
suggestions / drift check (fts), borg-style retention + restore safety
(backup), queueing locks / cron / lease heartbeat / DLQ replay (queue),
point-in-time restore + compressed snapshots (replication), joins /
aggregates / upsert_many / streaming (repository), aliases / search / tags /
bundles (artifacts), analytical helpers (analytics), repeatable migrations /
targets / clean (schema), generic telemetry instruments, and the new
MissionRecord model.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import os
import sqlite3
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.storage.analytics import SQLiteAnalytics, open_analytics
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.backup import BackupInfo, BackupManager, RetentionPolicy
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import AsyncDatabase, Database
from nomorals.storage.fts import (
    FTSIndex,
    build_match_query,
    escape_fts,
    near_query,
    phrase_query,
)
from nomorals.storage.kv import KVStore
from nomorals.storage.models import AgentRecord, MissionRecord, TaskRecord
from nomorals.storage.queue import (
    QueueFull,
    WorkQueue,
    next_cron_fire,
    parse_cron,
)
from nomorals.storage.replication import SqliteReplicator
from nomorals.storage.repository import Query, Repository
from nomorals.storage import router_telemetry
from nomorals.storage.s3blob import S3Config, presign_url
from nomorals.storage.schema import Migration, MigrationRunner
from nomorals.storage.vectors import VectorStore, normalize


def make_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class DatabaseSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nm-dbsweep-")
        cls.path = str(Path(cls.tmp.name) / "sweep.db")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_journal_size_limit_applied(self):
        db = Database(self.path)
        try:
            self.assertEqual(db.journal_size_limit, 64 * 1024 * 1024)
            self.assertEqual(
                db.scalar("PRAGMA journal_size_limit"), 64 * 1024 * 1024)
            self.assertEqual(db.journal_mode(), "wal")
        finally:
            db.close()

    def test_wal_health(self):
        db = Database(self.path)
        try:
            health = db.wal_health()
            for key in ("journal_mode", "wal_size_bytes", "busy", "log_pages",
                        "checkpointed_pages", "lag_pages"):
                self.assertIn(key, health)
            self.assertEqual(health["journal_mode"], "wal")
            self.assertGreaterEqual(health["lag_pages"], 0)
        finally:
            db.close()

    def test_checkpoint_maintenance(self):
        db = Database(self.path)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
            db.execute("INSERT INTO t (a) VALUES (1)")
            report = db.checkpoint_maintenance()
            self.assertIn("reclaimed_bytes", report)
            self.assertGreaterEqual(report["reclaimed_bytes"], 0)
        finally:
            db.close()

    def test_explain(self):
        db = make_db()
        try:
            plan = db.explain("SELECT * FROM kv_store WHERE key = ?", ("x",))
            self.assertTrue(plan)
            self.assertIn("detail", plan[0])
            self.assertTrue(any("kv_store" in r["detail"] for r in plan))
        finally:
            db.close()

    def test_foreign_key_check_clean(self):
        db = make_db()
        try:
            self.assertEqual(db.foreign_key_check(), [])
        finally:
            db.close()

    def test_format_stats(self):
        db = make_db()
        try:
            text = db.format_stats()
            self.assertIn("database", text)
            self.assertIn("queries", text)
        finally:
            db.close()

    def test_async_database(self):
        async def main():
            db = Database(":memory:")
            adb = AsyncDatabase(db)
            try:
                await adb.execute("CREATE TABLE t (a INTEGER)")
                await adb.execute("INSERT INTO t (a) VALUES (?)", (41,))
                rows = await adb.query("SELECT a FROM t")
                self.assertEqual(rows[0]["a"], 41)
                async with adb.transaction():
                    await adb.execute("INSERT INTO t (a) VALUES (?)", (1,))
                self.assertEqual(await adb.scalar("SELECT COUNT(*) FROM t"), 2)
            finally:
                db.close()

        asyncio.run(main())


class KVSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)

    def setUp(self):
        self.kv = KVStore(self.db, namespace=f"t{time.time_ns()}")

    def test_touch_sliding_expiry(self):
        self.kv.set("sess", "abc", ttl=100)
        ttl1 = self.kv.ttl("sess")
        self.assertTrue(self.kv.touch("sess", 1000))
        ttl2 = self.kv.ttl("sess")
        self.assertGreater(ttl2, ttl1)
        self.assertFalse(self.kv.touch("missing", 10))

    def test_get_and_touch(self):
        self.kv.set("k", "v", ttl=100)
        self.assertEqual(self.kv.get_and_touch("k", 1000), "v")
        self.assertGreater(self.kv.ttl("k"), 500)

    def test_setnx(self):
        self.assertTrue(self.kv.setnx("nx", "first"))
        self.assertFalse(self.kv.setnx("nx", "second"))
        self.assertEqual(self.kv.get("nx"), "first")

    def test_getset_getdel_pop(self):
        self.kv.set("g", "old")
        self.assertEqual(self.kv.getset("g", "new"), "old")
        self.assertEqual(self.kv.get("g"), "new")
        self.assertEqual(self.kv.getdel("g"), "new")
        self.assertIsNone(self.kv.get("g"))
        self.assertEqual(self.kv.pop("absent", default="dflt"), "dflt")

    def test_append(self):
        self.assertEqual(self.kv.append("log", "a"), 1)
        self.assertEqual(self.kv.append("log", "bc"), 3)
        self.assertEqual(self.kv.get("log"), "abc")

    def test_rename_keeps_ttl(self):
        self.kv.set("src", "data", ttl=500)
        self.assertTrue(self.kv.rename("src", "dst"))
        self.assertEqual(self.kv.get("dst"), "data")
        self.assertFalse(self.kv.exists("src"))
        self.assertGreater(self.kv.ttl("dst"), 400)
        self.assertFalse(self.kv.rename("nope", "dst2"))

    def test_get_or_set(self):
        calls = []
        v1 = self.kv.get_or_set("c", lambda: calls.append(1) or "computed")
        v2 = self.kv.get_or_set("c", lambda: calls.append(1) or "other")
        self.assertEqual((v1, v2), ("computed", "computed"))
        self.assertEqual(len(calls), 1)

    def test_scan_cursor(self):
        for i in range(10):
            self.kv.set(f"item:{i:02d}", i)
        seen = []
        cursor = ""
        while True:
            cursor, batch = self.kv.scan_cursor("item:", cursor=cursor, count=3)
            seen.extend(k for k, _ in batch)
            if not cursor:
                break
        self.assertEqual(len(seen), 10)
        self.assertEqual(len(set(seen)), 10)

    def test_hashes(self):
        self.assertEqual(self.kv.hset("user:1", "name", "ada"), 1)
        self.assertEqual(self.kv.hset("user:1", "name", "grace"), 0)
        self.assertEqual(self.kv.hget("user:1", "name"), "grace")
        self.assertEqual(self.kv.hget("user:1", "missing", "d"), "d")
        self.assertEqual(self.kv.hgetall("user:1"), {"name": "grace"})
        self.assertEqual(self.kv.hincrby("user:1", "visits"), 1)
        self.assertEqual(self.kv.hincrby("user:1", "visits", 4), 5)
        self.assertEqual(sorted(self.kv.hkeys("user:1")), ["name", "visits"])
        self.assertEqual(self.kv.hlen("user:1"), 2)
        self.assertEqual(self.kv.hdel("user:1", "name", "nope"), 1)
        self.assertEqual(self.kv.hgetall("user:1"), {"visits": 5})
        self.kv.set("plain", "x")
        with self.assertRaises(TypeError):
            self.kv.hget("plain", "f")

    def test_memory_usage_and_format(self):
        self.kv.set("a", "b")
        mem = self.kv.memory_usage()
        self.assertGreaterEqual(mem["keys"], 1)
        self.assertGreater(mem["bytes"], 0)
        text = self.kv.format_stats()
        self.assertIn("kv store", text)


class BlobSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)
        cls.tmp = tempfile.TemporaryDirectory(prefix="nm-blobsweep-")
        cls.addClassCleanup(cls.tmp.cleanup)

    def setUp(self):
        self.store = BlobStore(self.db, Path(self.tmp.name) / f"s{time.time_ns()}")

    def test_read_range(self):
        data = b"0123456789" * 100
        info = self.store.put_bytes(data, mime="application/octet-stream")
        self.assertEqual(self.store.read_range(info.sha256, 5, 10), data[5:15])
        chunks = list(self.store.stream_range(info.sha256, 0, 25, chunk=7))
        self.assertEqual(b"".join(chunks), data[:25])
        with self.assertRaises(ValueError):
            self.store.read_range(info.sha256, -1, 5)

    def test_trash_cycle(self):
        info = self.store.put_bytes(b"trash me", mime="text/plain")
        sha = info.sha256
        self.assertTrue(self.store.trash(sha))
        self.assertFalse(self.store.exists(sha))
        trashed = self.store.list_trash()
        self.assertEqual(len(trashed), 1)
        self.assertEqual(trashed[0]["sha256"], sha)
        # Trashed files are not orphans.
        self.assertEqual(self.store.orphans(), [])
        self.assertTrue(self.store.restore_trash(sha))
        self.assertTrue(self.store.exists(sha))
        self.assertEqual(self.store.get_bytes(sha), b"trash me")
        # Trash again, then empty.
        self.store.trash(sha)
        self.assertEqual(self.store.empty_trash(), 1)
        self.assertEqual(self.store.list_trash(), [])
        self.assertFalse(self.store.restore_trash(sha))

    def test_empty_trash_age_filter(self):
        info = self.store.put_bytes(b"old", mime="text/plain")
        self.store.trash(info.sha256)
        # Nothing is old enough yet.
        self.assertEqual(self.store.empty_trash(older_than_s=3600), 0)
        self.assertEqual(len(self.store.list_trash()), 1)
        self.assertEqual(self.store.empty_trash(older_than_s=0), 1)

    def test_sync_to(self):
        # sync_to walks the whole blobs table, so the source needs its own
        # database — one db per store root is the real deployment shape.
        src_db = Database(":memory:")
        src_db.migrate()
        self.addCleanup(src_db.close)
        src = BlobStore(src_db, Path(self.tmp.name) / f"s{time.time_ns()}")
        src_data = b"sync payload " * 50
        info = src.put_bytes(src_data, mime="application/octet-stream")
        other_db = Database(":memory:")
        other_db.migrate()
        self.addCleanup(other_db.close)
        other = BlobStore(other_db, Path(self.tmp.name) / f"o{time.time_ns()}")
        report = src.sync_to(other)
        self.assertEqual(report["pushed"], 1)
        self.assertEqual(report["skipped"], 0)
        self.assertTrue(other.exists(info.sha256))
        self.assertEqual(other.get_bytes(info.sha256), src_data)
        # Second sync is a no-op delta.
        report2 = src.sync_to(other)
        self.assertEqual(report2["pushed"], 0)
        self.assertEqual(report2["skipped"], 1)

    def test_put_file_progress(self):
        src = Path(self.tmp.name) / "big.bin"
        src.write_bytes(b"x" * 5000)
        seen = []
        info = self.store.put_file(src, mime="application/octet-stream",
                                   progress=lambda d, t: seen.append((d, t)))
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], 5000)
        self.assertEqual(info.size, 5000)

    def test_describe_and_format(self):
        self.store.put_bytes(b"a" * 100, mime="text/plain")
        self.store.put_bytes(b"a" * 100, mime="text/plain")  # dedup hit
        desc = self.store.describe()
        self.assertEqual(desc["blobs"], 1)
        self.assertEqual(desc["dedup_hits"], 1)
        self.assertGreaterEqual(desc["dedup_ratio"], 1.0)
        self.assertIn("blob store", self.store.format_stats())


class S3BlobSweepTest(unittest.TestCase):
    def _config(self) -> S3Config:
        return S3Config(
            endpoint="https://s3.example.com",
            bucket="test-bucket",
            access_key="AKID",
            secret_key="SECRET",
            region="us-east-1",
        )

    def test_presign_url_deterministic(self):
        url = "https://s3.example.com/test-bucket/ab/cd/abc123"
        a = presign_url("GET", url, access_key="AKID", secret_key="SECRET",
                        region="us-east-1", expires_in=3600,
                        timestamp="20260101T000000Z")
        b = presign_url("GET", url, access_key="AKID", secret_key="SECRET",
                        region="us-east-1", expires_in=3600,
                        timestamp="20260101T000000Z")
        self.assertEqual(a, b)
        self.assertIn("X-Amz-Algorithm=AWS4-HMAC-SHA256", a)
        self.assertIn("X-Amz-Expires=3600", a)
        self.assertIn("X-Amz-Signature=", a)
        self.assertTrue(a.startswith(url + "?"))

    def test_presign_url_expiry_bounds(self):
        with self.assertRaises(ValueError):
            presign_url("GET", "https://x/y", access_key="a", secret_key="s",
                        region="r", expires_in=0)
        with self.assertRaises(ValueError):
            presign_url("GET", "https://x/y", access_key="a", secret_key="s",
                        region="r", expires_in=999999)

    def test_presigned_put_shape(self):
        from nomorals.storage.s3blob import S3BlobStore

        store = S3BlobStore(self._config())
        result = store.presigned_put("report.pdf", expires_in=600)
        self.assertIn("url", result)
        self.assertIn("key", result)
        self.assertIn("X-Amz-Signature=", result["url"])
        self.assertIn("incoming/", result["key"])

    def test_s3config_validation(self):
        from nomorals.core.errors import ConfigError

        with self.assertRaises(ConfigError):
            S3Config(endpoint="", bucket="b", access_key="a", secret_key="s")
        with self.assertRaises(ConfigError):
            S3Config(endpoint="https://x", bucket="b", access_key="", secret_key="")

    def test_xml_text_helper(self):
        from nomorals.storage.s3blob import S3BlobStore

        body = (b'<?xml version="1.0"?><InitiateMultipartUploadResult>'
                b"<UploadId>upload-123</UploadId></InitiateMultipartUploadResult>")
        self.assertEqual(S3BlobStore._xml_text(body, "UploadId"), "upload-123")
        self.assertEqual(S3BlobStore._xml_text(b"garbage", "UploadId"), "")


class VectorSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import random as _random

        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)
        rng = _random.Random(7)
        dim = 16
        items = []
        for i in range(600):
            vec = [rng.gauss(0, 1) for _ in range(dim)]
            items.append((vec, "doc", f"doc-{i}"))
        cls.store = VectorStore(cls.db, use_numpy=True)
        cls.store.put_many(items)
        cls.dim = dim

    def test_build_index_kmeans(self):
        k = self.store.build_index(seed=7, iterations=5)
        self.assertGreater(k, 0)
        total_members = sum(len(c.members) for c in self.store._centroids)
        self.assertEqual(total_members, 600)
        # Centroids are normalized.
        import math

        for centroid in self.store._centroids:
            norm = math.sqrt(sum(v * v for v in centroid.vector))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_index_persistence(self):
        self.store.build_index(seed=7, iterations=3)
        fresh = VectorStore(self.db, use_numpy=True)
        loaded = fresh.load_index()
        self.assertGreater(loaded, 0)
        self.assertEqual(len(fresh._centroids), loaded)

    def test_stale_index_falls_back(self):
        self.store.build_index(seed=7, iterations=3)
        self.store.put([0.1] * self.dim, owner_type="doc", owner_id="new")
        fresh = VectorStore(self.db, use_numpy=True)
        # Persisted index covers 600 vectors; table now has 601 → stale.
        self.assertEqual(fresh.load_index(), 0)
        # In-memory staleness: search with nprobe still returns results.
        hits = self.store.search([0.1] * self.dim, limit=5, nprobe=4)
        self.assertEqual(len(hits), 5)

    def test_evaluate_recall(self):
        self.store.build_index(seed=7, iterations=5)
        queries = [[0.5] * self.dim, [0.1] * self.dim, [-0.3] * self.dim]
        report = self.store.evaluate_recall(queries, k=5, nprobe=1000)
        self.assertAlmostEqual(report["recall_at_k"], 1.0, places=2)
        self.assertEqual(report["queries"], 3)

    def test_search_many(self):
        batches = self.store.search_many([[0.2] * self.dim] * 3, limit=4)
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 4)

    def test_score_mode_distance(self):
        hits = self.store.search([0.2] * self.dim, limit=5,
                                 score_mode="distance")
        self.assertEqual(len(hits), 5)
        scores = [h.score for h in hits]
        self.assertTrue(all(0.0 <= s <= 2.0 for s in scores))
        self.assertEqual(scores, sorted(scores))
        with self.assertRaises(Exception):
            self.store.search([0.2] * self.dim, score_mode="weird")

    def test_owner_ids_filter(self):
        hits = self.store.search([0.2] * self.dim, limit=10,
                                 owner_ids={"doc-1", "doc-2"})
        self.assertTrue(hits)
        self.assertTrue(all(h.owner_id in {"doc-1", "doc-2"} for h in hits))

    def test_nprobe_clamped(self):
        self.store.build_index(seed=7, iterations=3)
        hits = self.store.search([0.2] * self.dim, limit=5, nprobe=10**9)
        self.assertEqual(len(hits), 5)

    def test_drop_index(self):
        self.store.build_index(seed=7, iterations=2)
        self.store.drop_index()
        self.assertEqual(self.store._centroids, [])
        self.assertFalse(self.db.table_exists("embeddings_index"))

    def test_format_stats(self):
        self.assertIn("vector index", self.store.format_stats())


class FTSSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)

    def setUp(self):
        name = f"fts_{time.time_ns()}"
        self.index = FTSIndex(self.db, name, columns=["title", "body"],
                              auto_create=True)
        self.assertTrue(self.index.available)
        self.index.put(1, ["hello world", "the quick brown fox jumps"])
        self.index.put(2, ["goodbye world", "the lazy dog sleeps"])
        self.index.put(3, ["hello again", "another hello world document"])

    def test_match_query_facets(self):
        q = build_match_query("title:hello world", columns=["title", "body"])
        self.assertIn('{title : "hello"*}', q)
        self.assertIn('"world"*', q)
        # Unknown facet falls through as free text.
        q2 = build_match_query("nope:hello", columns=["title"])
        self.assertIn('"nope:hello"*', q2)

    def test_phrase_and_near(self):
        self.assertEqual(phrase_query("quick brown"), '"quick brown"')
        self.assertEqual(near_query(["quick", "fox"], 5), "NEAR(\"quick\" \"fox\", 5)")
        self.assertEqual(near_query(["solo"]), '"solo"')
        self.assertEqual(near_query([]), "")

    def test_search_weights_and_highlight(self):
        hits = self.index.search("hello", weights={"title": 10.0, "body": 1.0},
                                 highlight_tags=("<b>", "</b>"))
        self.assertTrue(hits)
        titles = [h.columns["title"] for h in hits]
        self.assertIn("hello world", titles)
        self.assertTrue(any("<b>" in h.snippet for h in hits))

    def test_suggest(self):
        self.assertTrue(self.index.ensure_vocab())
        suggestions = self.index.suggest("hel")
        self.assertIn("hello", suggestions)
        self.assertEqual(self.index.suggest("x"), [])

    def test_check_drift(self):
        report = self.index.check_drift(3)
        self.assertTrue(report["healthy"])
        self.assertEqual(report["drift"], 0)
        bad = self.index.check_drift(99)
        self.assertFalse(bad["healthy"])

    def test_build_ddl(self):
        ddl = self.index.build_ddl()
        self.assertIn("USING fts5", ddl)
        self.assertIn("unicode61", ddl)

    def test_format_stats(self):
        self.assertIn("fts index", self.index.format_stats())
        self.assertIn("unicode61", self.index.format_stats())


class BackupSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nm-backupsweep-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)
        cls.db.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
        cls.db.execute("INSERT INTO t VALUES (1)")

    def _manager(self, **kwargs):
        d = Path(self.tmp.name) / f"mgr{time.time_ns()}"
        return BackupManager(self.db, d, **kwargs)

    def test_retention_policy_select(self):
        now = time.time()
        backups = [
            BackupInfo(name=f"b{i}", path=f"/tmp/b{i}", size=10,
                       sha256="x", pages=1, created_at=now - i * 86400)
            for i in range(40)
        ]
        policy = RetentionPolicy(keep_last=7, keep_daily=7, keep_weekly=2,
                                 keep_monthly=2)
        keepers = policy.select_keepers(backups)
        # Last 7 always kept.
        for b in backups[:7]:
            self.assertIn(b.path, keepers)
        # One per day beyond that keeps several more.
        self.assertGreater(len(keepers), 7)
        self.assertLess(len(keepers), 40)

    def test_rotate_dry_run(self):
        mgr = self._manager(keep=1)
        for _ in range(3):
            mgr.create(label="dry")
            time.sleep(1.05)
        doomed = mgr.rotate(dry_run=True)
        self.assertTrue(doomed)
        # Nothing deleted on dry run.
        self.assertEqual(len(mgr.list()), 3)
        removed = mgr.rotate()
        self.assertEqual(sorted(removed), sorted(doomed))
        self.assertLessEqual(len(mgr.list()), 2)

    def test_verify_all(self):
        mgr = self._manager()
        mgr.create(label="v")
        self.assertEqual(mgr.verify_all(), {})

    def test_restore_safety(self):
        mgr = self._manager(compress=False)
        info = mgr.create(label="r")
        target = Path(self.tmp.name) / f"restored{time.time_ns()}.db"
        target.write_bytes(b"precious")
        with self.assertRaises(Exception):
            mgr.restore(info, target=target)
        restored = mgr.restore(info, target=target, overwrite=True)
        self.assertTrue(restored.is_file())
        spares = list(target.parent.glob(target.name + ".pre-restore-*"))
        self.assertEqual(len(spares), 1)
        self.assertEqual(spares[0].read_bytes(), b"precious")

    def test_estimate_and_format(self):
        mgr = self._manager()
        est = mgr.estimate()
        self.assertGreater(est["db_pages"], 0)
        table = mgr.format_table()
        self.assertIn("backups", table)


class QueueSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)

    def setUp(self):
        self.q = WorkQueue(self.db)
        self.topic = f"topic-{time.time_ns()}"

    def test_queueing_lock_dedup(self):
        j1 = self.q.enqueue(self.topic, {"n": 1}, queueing_lock="report-42")
        j2 = self.q.enqueue(self.topic, {"n": 2}, queueing_lock="report-42")
        self.assertEqual(j1, j2)
        self.q.complete(j1, {"ok": True})
        j3 = self.q.enqueue(self.topic, {"n": 3}, queueing_lock="report-42")
        self.assertNotEqual(j3, j1)

    def test_max_depth_backpressure(self):
        with self.assertRaises(QueueFull):
            for _ in range(3):
                self.q.enqueue(self.topic, {}, max_depth=2)

    def test_extend_lease(self):
        jid = self.q.enqueue(self.topic, {})
        job = self.q.lease_one(self.topic, worker="w1")
        self.assertTrue(self.q.extend_lease(jid, "w1", 600))
        self.assertFalse(self.q.extend_lease(jid, "w2", 600))

    def test_cancel(self):
        jid = self.q.enqueue(self.topic, {})
        self.assertTrue(self.q.cancel(jid))
        self.assertIsNone(self.q.lease_one(self.topic))
        self.assertEqual(self.q.get(jid).status, "cancelled")

    def test_wait_for_result(self):
        jid = self.q.enqueue(self.topic, {"x": 1})
        job = self.q.lease_one(self.topic)
        self.q.complete(job.id, {"answer": 42})
        self.assertEqual(self.q.wait_for(jid, timeout=5), {"answer": 42})
        self.assertEqual(self.q.result_of(jid), {"answer": 42})

    def test_wait_for_dead_raises(self):
        jid = self.q.enqueue(self.topic, {}, max_attempts=1)
        job = self.q.lease_one(self.topic)
        self.q.fail(job.id, "boom", retry=False)
        with self.assertRaises(Exception):
            self.q.wait_for(jid, timeout=5)

    def test_dead_letter_replay(self):
        jid = self.q.enqueue(self.topic, {}, max_attempts=1)
        job = self.q.lease_one(self.topic)
        self.q.fail(job.id, "boom", retry=False)
        self.assertEqual(len(self.q.dead_jobs(self.topic)), 1)
        self.assertTrue(self.q.retry_dead(jid))
        self.assertEqual(self.q.get(jid).status, "ready")
        self.assertEqual(self.q.get(jid).attempts, 0)

    def test_parse_cron(self):
        fields = parse_cron("*/15 9-17 * * 1-5")
        self.assertEqual(len(fields), 5)
        self.assertIn(0, fields[0])
        self.assertIn(30, fields[0])
        with self.assertRaises(ValueError):
            parse_cron("not a cron")
        with self.assertRaises(ValueError):
            parse_cron("* * * * 99")

    def test_next_cron_fire(self):
        nxt = next_cron_fire("* * * * *")
        self.assertIsNotNone(nxt)
        self.assertGreater(nxt, time.time())
        self.assertLess(nxt - time.time(), 120)
        # Feb 30 never fires.
        self.assertIsNone(next_cron_fire("0 0 30 2 *"))

    def test_recurring_tick_idempotent(self):
        name = f"cron-{time.time_ns()}"
        topic = f"rt-{time.time_ns()}"
        self.q.schedule_recurring(name, topic, "* * * * *", {"ping": 1})
        # Force due.
        self.db.execute(
            "UPDATE recurring_jobs SET next_fire_at = ? WHERE name = ?",
            (time.time() - 1, name),
        )
        fired = self.q.tick_recurring()
        self.assertIn(name, fired)
        self.assertEqual(self.q.pending(topic), 1)
        # A second tick with the same due time must not double-fire.
        self.db.execute(
            "UPDATE recurring_jobs SET next_fire_at = ? WHERE name = ?",
            (time.time() - 1, name),
        )
        fired2 = self.q.tick_recurring()
        self.assertIn(name, fired2)
        self.assertEqual(self.q.pending(topic), 1)
        schedules = self.q.list_recurring()
        self.assertTrue(any(s["name"] == name for s in schedules))
        self.assertTrue(self.q.unschedule_recurring(name))
        self.assertFalse(self.q.unschedule_recurring(name))

    def test_format_status(self):
        self.q.enqueue(self.topic, {})
        text = self.q.format_status()
        self.assertIn("work queue", text)
        self.assertIn(self.topic, text)


class ReplicationSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nm-replsweep-")
        cls.addClassCleanup(cls.tmp.cleanup)

    def _db(self):
        db = Database(":memory:")
        db.execute("CREATE TABLE t (a INTEGER)")
        db.execute("INSERT INTO t VALUES (7)")
        return db

    def test_snapshot_restore_cycle(self):
        for compress in (False, True):
            db = self._db()
            try:
                rep = SqliteReplicator(
                    db, Path(self.tmp.name) / f"r{time.time_ns()}{compress}",
                    compress=compress)
                info = rep.snapshot()
                if compress:
                    self.assertTrue(info.path.endswith(".gz"))
                restored = rep.restore(info, target=Path(self.tmp.name) /
                                       f"out{time.time_ns()}{compress}.db")
                probe = sqlite3.connect(str(restored))
                try:
                    val = probe.execute("SELECT a FROM t").fetchone()[0]
                finally:
                    probe.close()
                self.assertEqual(val, 7)
                self.assertEqual(rep.verify_all(), {})
            finally:
                db.close()

    def test_restore_at_point_in_time(self):
        db = self._db()
        try:
            rep = SqliteReplicator(db, Path(self.tmp.name) / f"p{time.time_ns()}")
            first = rep.snapshot()
            time.sleep(1.05)
            db.execute("INSERT INTO t VALUES (8)")
            second = rep.snapshot()
            # Restore to just before the second snapshot → first generation.
            target = Path(self.tmp.name) / f"pit{time.time_ns()}.db"
            rep.restore_at(second.created_at - 0.5, target=target)
            probe = sqlite3.connect(str(target))
            try:
                rows = probe.execute("SELECT a FROM t ORDER BY a").fetchall()
            finally:
                probe.close()
            self.assertEqual([r[0] for r in rows], [7])
            with self.assertRaises(Exception):
                rep.restore_at(first.created_at - 100)
        finally:
            db.close()

    def test_lag_and_status(self):
        db = self._db()
        try:
            rep = SqliteReplicator(db, Path(self.tmp.name) / f"l{time.time_ns()}")
            self.assertIsNone(rep.lag_seconds())
            rep.snapshot()
            lag = rep.lag_seconds()
            self.assertIsNotNone(lag)
            self.assertLess(lag, 60)
            text = rep.status()
            self.assertIn("replication", text)
            self.assertIn("lag", text)
        finally:
            db.close()

    def test_restore_safety_copy(self):
        db = self._db()
        try:
            rep = SqliteReplicator(db, Path(self.tmp.name) / f"s{time.time_ns()}")
            info = rep.snapshot()
            target = Path(self.tmp.name) / f"safe{time.time_ns()}.db"
            target.write_bytes(b"live data")
            rep.restore(info, target=target)
            spares = list(target.parent.glob(target.name + ".pre-restore-*"))
            self.assertEqual(len(spares), 1)
            self.assertEqual(spares[0].read_bytes(), b"live data")
        finally:
            db.close()


class RepositorySweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = Database(":memory:")
        cls.addClassCleanup(cls.db.close)
        cls.db.execute(
            "CREATE TABLE authors (id TEXT PRIMARY KEY, name TEXT, country TEXT, "
            "created_at REAL, updated_at REAL)"
        )
        cls.db.execute(
            "CREATE TABLE books (id TEXT PRIMARY KEY, author_id TEXT, "
            "title TEXT, pages INTEGER, rating REAL, created_at REAL, "
            "updated_at REAL)"
        )
        # Scratch table for the mutating tests (pytest runs unittest methods
        # alphabetically, so mutation tests must not pollute the shared rows).
        cls.db.execute(
            "CREATE TABLE books_mut (id TEXT PRIMARY KEY, author_id TEXT, "
            "title TEXT, pages INTEGER, rating REAL, created_at REAL, "
            "updated_at REAL)"
        )
        cls.db.execute(
            "INSERT INTO books_mut (id, title, pages) VALUES "
            "('m0', 't0', 10), ('m1', 't1', 20)"
        )
        repo_a = Repository(cls.db, "authors", pk="id", auto_id=False)
        repo_b = Repository(cls.db, "books", pk="id", auto_id=False)
        for i, (name, country) in enumerate(
                [("a1", "NG"), ("a2", "US"), ("a3", "NG")]):
            repo_a.create({"id": f"a{i}", "name": name, "country": country})
        books = [
            ("b0", "a0", "t0", 100, 4.5), ("b1", "a0", "t1", 200, 3.5),
            ("b2", "a1", "t2", 300, 5.0), ("b3", "a2", "t3", 150, 4.0),
        ]
        for bid, aid, title, pages, rating in books:
            repo_b.create({"id": bid, "author_id": aid, "title": title,
                           "pages": pages, "rating": rating})
        cls.authors = repo_a
        cls.books = repo_b

    def test_join(self):
        q = (self.books.query()
             .select("books.title", "authors.name")
             .join("authors", "authors.id = books.author_id"))
        sql, params = q.build()
        rows = self.db.query(sql, params)
        self.assertEqual(len(rows), 4)
        self.assertIn("name", rows[0])

    def test_group_by_having(self):
        q = (self.books.query()
             .select("author_id", "COUNT(*) AS n")
             .group_by("author_id")
             .having("COUNT(*) > ?", 1))
        sql, params = q.build()
        rows = self.db.query(sql, params)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["author_id"], "a0")

    def test_where_variants(self):
        q = self.books.query().where_in("id", ["b0", "b1"])
        self.assertEqual(len(self.db.query(*q.build())), 2)
        q = self.books.query().where_like("title", "t%")
        self.assertEqual(len(self.db.query(*q.build())), 4)
        q = self.books.query().where_between("pages", 100, 200)
        self.assertEqual(len(self.db.query(*q.build())), 3)
        q = self.books.query().where_null("rating")
        self.assertEqual(len(self.db.query(*q.build())), 0)
        self.assertTrue(self.books.query().where("pages > 1000").exists(self.db)
                        is False)
        self.assertTrue(self.books.query().where("pages > 10").exists(self.db))

    def test_first_pluck_distinct(self):
        row = self.books.query().order_by("pages DESC").first(self.db)
        self.assertEqual(row["id"], "b2")
        ids = self.books.query().order_by("id").pluck(self.db, "id")
        self.assertEqual(ids, ["b0", "b1", "b2", "b3"])
        countries = (self.authors.query().select("country").distinct()
                     .pluck(self.db, "country"))
        self.assertEqual(sorted(countries), ["NG", "US"])

    def test_aggregate(self):
        self.assertEqual(self.books.aggregate("COUNT", "*"), 4)
        self.assertEqual(self.books.aggregate("SUM", "pages"), 750)
        self.assertAlmostEqual(self.books.aggregate("AVG", "rating"), 4.25)
        self.assertEqual(self.books.aggregate("MAX", "pages", author_id="a0"), 200)
        with self.assertRaises(Exception):
            self.books.aggregate("MEDIAN", "pages")

    def test_update_where_and_touch(self):
        repo = Repository(self.db, "books_mut", pk="id", auto_id=False)
        n = repo.update_where({"rating": 1.0}, "id = ?", ("m1",))
        self.assertEqual(n, 1)
        self.assertEqual(repo.get("m1")["rating"], 1.0)
        before = repo.get("m0")["updated_at"]
        time.sleep(0.01)
        repo.touch("m0")
        self.assertGreater(repo.get("m0")["updated_at"], before)

    def test_paginate_with_total(self):
        page = self.books.paginate_with_total(1, 3, order_by="id")
        self.assertEqual(page["total"], 4)
        self.assertEqual(page["pages"], 2)
        self.assertTrue(page["has_next"])
        self.assertFalse(page["has_prev"])
        self.assertEqual(len(page["rows"]), 3)

    def test_first_or_create_update_or_create(self):
        repo = Repository(self.db, "books_mut", pk="id", auto_id=False)
        row = repo.first_or_create({"title": "t9", "pages": 9}, id="m9")
        self.assertEqual(row["id"], "m9")
        again = repo.first_or_create({"title": "t9"}, id="m9")
        self.assertEqual(again["id"], "m9")
        updated = repo.update_or_create({"rating": 2.5}, id="m9")
        self.assertEqual(updated["rating"], 2.5)

    def test_bulk_update_and_upsert_many(self):
        repo = Repository(self.db, "books_mut", pk="id", auto_id=False)
        n = repo.bulk_update(
            [{"id": "m0", "rating": 4.9}, {"id": "m1", "rating": 4.8}])
        self.assertEqual(n, 2)
        keys = repo.upsert_many([
            {"id": "m0", "title": "t0x", "pages": 101},
            {"id": "mx", "title": "new", "pages": 10},
        ])
        self.assertIn("m0", keys)
        self.assertIn("mx", keys)
        self.assertEqual(repo.get("m0")["title"], "t0x")
        self.assertEqual(repo.get("mx")["pages"], 10)

    def test_stream(self):
        rows = list(self.books.stream(batch_size=2, order_by="id"))
        self.assertGreaterEqual(len(rows), 4)
        ids = [r["id"] for r in rows]
        self.assertEqual(ids, sorted(ids))


class ArtifactSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)
        cls.tmp = tempfile.TemporaryDirectory(prefix="nm-artsweep-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.blobs = BlobStore(cls.db, Path(cls.tmp.name) / "blobs")

    def setUp(self):
        self.store = ArtifactStore(self.db, self.blobs)

    def test_search(self):
        marker = f"probe-{time.time_ns()}"
        a1 = self.store.put_text("report one", type=f"{marker}-report",
                                 creator=f"{marker}-analyst", mission_id=marker,
                                 metadata={"lang": "en", marker: "yes"})
        self.store.put_text("note", type=f"{marker}-note",
                            creator=f"{marker}-analyst", mission_id=marker)
        self.assertEqual(len(self.store.search(type=f"{marker}-report")), 1)
        self.assertEqual(len(self.store.search(creator=f"{marker}-analyst")), 2)
        self.assertEqual(len(self.store.search(mission_id=marker)), 2)
        self.assertEqual(
            len(self.store.search(metadata={marker: "yes"})), 1)
        self.assertEqual(
            self.store.search(type=f"{marker}-report")[0].id, a1.id)

    def test_tag_untag(self):
        art = self.store.put_text("t", type="note")
        self.assertTrue(self.store.tag(art.id, "priority", "high"))
        self.assertEqual(self.store.get(art.id).metadata["priority"], "high")
        self.assertTrue(self.store.untag(art.id, "priority"))
        self.assertNotIn("priority", self.store.get(art.id).metadata)
        self.assertFalse(self.store.untag(art.id, "nope"))
        self.assertFalse(self.store.tag("missing", "k", "v"))

    def test_delete_drops_blob_refcount(self):
        art = self.store.put_text("bye", type="note")
        sha = art.content_hash
        before = self.blobs.info(sha).refcount
        self.assertTrue(self.store.delete(art.id))
        self.assertIsNone(self.store.get(art.id))
        after = self.blobs.info(sha)
        self.assertTrue(after is None or after.refcount == before - 1)
        self.assertFalse(self.store.delete("missing"))

    def test_aliases(self):
        a1 = self.store.put_text("v1", type="report")
        a2 = self.store.put_text("v2", type="report")
        self.store.set_alias("latest", a1.id)
        self.assertEqual(self.store.resolve_alias("latest").id, a1.id)
        self.store.set_alias("latest", a2.id)  # move the pointer
        self.assertEqual(self.store.resolve_alias("latest").id, a2.id)
        self.assertEqual(self.store.aliases(), {"latest": a2.id})
        self.assertTrue(self.store.delete_alias("latest"))
        self.assertIsNone(self.store.resolve_alias("latest"))
        with self.assertRaises(ValueError):
            self.store.set_alias("", a1.id)
        with self.assertRaises(KeyError):
            self.store.set_alias("x", "missing-id")

    def test_export_import_bundle(self):
        a1 = self.store.put_text("bundle me", type="report", creator="me",
                                 metadata={"k": "v"})
        dest = Path(self.tmp.name) / f"bundle{time.time_ns()}.tar.gz"
        bundle = self.store.export_bundle([a1.id, "missing-id"], dest)
        self.assertTrue(bundle.is_file())
        with tarfile.open(bundle, "r:gz") as tar:
            self.assertIn("manifest.json", tar.getnames())
        # Import into a fresh store (fresh db + blob root).
        fresh_db = Database(":memory:")
        fresh_db.migrate()
        self.addCleanup(fresh_db.close)
        other_blobs = BlobStore(fresh_db, Path(self.tmp.name) / f"b{time.time_ns()}")
        other = ArtifactStore(fresh_db, other_blobs)
        imported = other.import_bundle(bundle)
        self.assertEqual(len(imported), 1)
        self.assertEqual(other.read_text(imported[0].id), "bundle me")
        self.assertEqual(imported[0].metadata["k"], "v")

    def test_format_card_and_stats(self):
        art = self.store.put_text("card", type="report", creator="c")
        card = self.store.format_card(art)
        self.assertIn(art.id, card)
        self.assertIn("artifact", card)
        stats = self.store.stats()
        self.assertGreaterEqual(stats["total"], 1)
        self.assertIn("report", stats["by_type"])


class AnalyticsSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)
        cls.db.execute("CREATE TABLE IF NOT EXISTS nums (id INTEGER, v REAL)")
        cls.db.executemany(
            "INSERT INTO nums (id, v) VALUES (?, ?)",
            [(i, float(i * 10)) for i in range(100)],
        )

    def setUp(self):
        self.engine = SQLiteAnalytics(self.db)

    def test_profile(self):
        plan = self.engine.profile("SELECT * FROM nums WHERE v > ?", (10,))
        self.assertTrue(plan)

    def test_query_cached(self):
        rows = self.engine.query_cached("SELECT COUNT(*) AS n FROM nums")
        self.assertEqual(rows[0]["n"], 100)

    def test_describe(self):
        desc = self.engine.describe("nums")
        self.assertEqual(desc["rows"], 100)
        self.assertTrue(any(c["name"] == "v" for c in desc["columns"]))

    def test_sample(self):
        rows = self.engine.sample("nums", 10)
        self.assertEqual(len(rows), 10)

    def test_histogram(self):
        bins = self.engine.histogram("nums", "v", bins=5)
        self.assertTrue(bins)
        self.assertEqual(sum(b["count"] for b in bins), 100)

    def test_iter_query(self):
        total = 0
        for batch in self.engine.iter_query("SELECT * FROM nums", batch=30):
            total += len(batch)
        self.assertEqual(total, 100)

    def test_open_analytics_prefers_sqlite(self):
        engine = open_analytics(self.db, prefer="sqlite")
        self.assertEqual(engine.name, "sqlite")
        with self.assertRaises(Exception):
            open_analytics(self.db, prefer="duckdb")

    def test_format_stats(self):
        self.assertIn("analytics", self.engine.format_stats())


class SchemaSweepTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.close)
        self.runner = MigrationRunner(self.db)

    def _versioned(self, n=3):
        return [
            Migration(i, f"m{i}", sql=f"CREATE TABLE t{i} (a INTEGER);")
            for i in range(1, n + 1)
        ]

    def test_repeatable_reapplies_on_change(self):
        # Repeatable SQL must be re-runnable (DROP + CREATE): Flyway's
        # repeatable contract — the definition, not a delta.
        view_sql = lambda n: (  # noqa: E731
            f"DROP VIEW IF EXISTS v1; CREATE VIEW v1 AS SELECT {n} AS x;")
        migs = self._versioned(1) + [
            Migration(0, "latest_view", sql=view_sql(1), repeatable=True)
        ]
        first = self.runner.apply_all(migs)
        self.assertIn("R__latest_view", first)
        # No change → not re-applied.
        second = self.runner.apply_all(migs)
        self.assertNotIn("R__latest_view", second)
        # Changed definition → re-applied.
        migs2 = self._versioned(1) + [
            Migration(0, "latest_view", sql=view_sql(2), repeatable=True)
        ]
        third = self.runner.apply_all(migs2)
        self.assertIn("R__latest_view", third)
        self.assertEqual(self.db.scalar("SELECT x FROM v1"), 2)

    def test_repeatable_runs_after_versioned(self):
        order = []

        def _fn(db):
            order.append("fn")

        migs = [
            Migration(1, "base", sql="CREATE TABLE b (a INTEGER);"),
            Migration(0, "rep", fn=_fn, repeatable=True),
        ]
        self.runner.apply_all(migs)
        self.assertEqual(order, ["fn"])
        self.assertTrue(self.db.table_exists("b"))

    def test_target_version(self):
        applied = self.runner.apply_all(self._versioned(3), target=2)
        self.assertEqual(self.runner.current_version(), 2)
        self.assertEqual(len(applied), 2)
        # Resume to the latest.
        rest = self.runner.apply_all(self._versioned(3))
        self.assertEqual(rest, ["0003_m3"])

    def test_out_of_order_strict(self):
        # Apply v2 only (v1 skipped) → v1 pending below current version.
        self.runner.apply_all([self._versioned(2)[1]])
        migs = self._versioned(3)
        with self.assertRaises(Exception):
            self.runner.apply_all(migs, out_of_order=False)
        # Default (lenient) applies it.
        applied = self.runner.apply_all(migs)
        self.assertIn("0001_m1", applied)

    def test_validate_detects_removed(self):
        self.runner.apply_all(self._versioned(2))
        problems = self.runner.validate(self._versioned(1))
        self.assertTrue(any("no matching migration" in p for p in problems))

    def test_clean(self):
        self.runner.apply_all(self._versioned(1))
        with self.assertRaises(Exception):
            self.runner.clean(confirm="nope")
        dropped = self.runner.clean(confirm="CLEAN")
        self.assertIn("t1", dropped)
        self.assertFalse(self.db.table_exists("t1"))

    def test_dry_run_includes_repeatable(self):
        migs = self._versioned(1) + [
            Migration(0, "rep",
                      sql="DROP VIEW IF EXISTS v; CREATE VIEW v AS SELECT 1;",
                      repeatable=True)
        ]
        plan = self.runner.dry_run(migs)
        self.assertEqual(len(plan), 2)
        self.assertTrue(any(p["repeatable"] for p in plan))

    def test_status_and_format(self):
        self.runner.apply_all(self._versioned(2))
        status = self.runner.status(self._versioned(3))
        self.assertEqual(status["current_version"], 2)
        self.assertEqual(status["pending"], ["0003_m3"])
        text = self.runner.format_status(self._versioned(3))
        self.assertIn("migrations", text)

    def test_repeatable_version_must_be_zero(self):
        with self.assertRaises(Exception):
            Migration(5, "bad", sql="SELECT 1;", repeatable=True)


class TelemetrySweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = make_db()
        cls.addClassCleanup(cls.db.close)

    def setUp(self):
        self.db.execute("DELETE FROM coremind_telemetry")

    def test_counter_gauge_timing(self):
        router_telemetry.record_counter(self.db, "jobs_done", 2)
        router_telemetry.record_counter(self.db, "jobs_done", 3)
        router_telemetry.record_gauge(self.db, "queue_depth", 7.0)
        router_telemetry.record_timing(self.db, "route_latency", 0.5)
        router_telemetry.record_timing(self.db, "route_latency", 1.5)
        snap = router_telemetry.snapshot(self.db)
        self.assertEqual(snap["gauges"]["jobs_done"], 5.0)
        self.assertEqual(snap["gauges"]["queue_depth"], 7.0)
        timing = snap["timings"]["route_latency"]
        self.assertEqual(timing["count"], 2)
        self.assertAlmostEqual(timing["avg"], 1.0)
        self.assertAlmostEqual(timing["min"], 0.5)
        self.assertAlmostEqual(timing["max"], 1.5)

    def test_top_routes_and_reset(self):
        router_telemetry.record_route(self.db, "coding")
        router_telemetry.record_route(self.db, "coding")
        router_telemetry.record_route(self.db, "brain")
        top = router_telemetry.top_routes(self.db, limit=5)
        self.assertEqual(top[0], ("coding", 2))
        self.assertTrue(router_telemetry.reset(self.db, "route:coding"))
        self.assertFalse(router_telemetry.reset(self.db, "route:coding"))
        snap = router_telemetry.snapshot(self.db)
        self.assertNotIn("coding", snap["routes"])

    def test_format_snapshot(self):
        router_telemetry.record_route(self.db, "research_swarm")
        router_telemetry.record_plan_error(self.db, "boom", route="coding")
        router_telemetry.record_timing(self.db, "plan_s", 2.0)
        text = router_telemetry.format_snapshot(self.db)
        self.assertIn("router telemetry", text)
        self.assertIn("research_swarm", text)
        self.assertIn("boom", text)


class ModelsSweepTest(unittest.TestCase):
    def test_mission_record(self):
        row = {
            "id": "m1", "name": "n", "goal": "g", "status": "running",
            "state": '{"step": 2}', "budget_wall": 100.0, "budget_tokens": 1000,
            "spent_wall": 30.0, "spent_tokens": 400, "iterations": 3,
            "success": None, "created_at": 1.0, "updated_at": 2.0,
            "finished_at": None, "metadata": '{"a": 1}',
        }
        rec = MissionRecord.from_row(row)
        self.assertEqual(rec.state, {"step": 2})
        self.assertEqual(rec.budget_wall_remaining, 70.0)
        self.assertEqual(rec.budget_tokens_remaining, 600)
        self.assertFalse(rec.is_terminal)
        out = rec.to_row()
        self.assertEqual(out["id"], "m1")
        # JSON round trip.
        rec2 = MissionRecord.from_json(rec.to_json())
        self.assertEqual(rec2.id, "m1")
        self.assertEqual(rec2.state, {"step": 2})
        # Diff.
        rec3 = MissionRecord.from_row({**row, "status": "done"})
        diff = rec.changed_fields(rec3)
        self.assertEqual(diff["status"], ("running", "done"))

    def test_agent_record_helpers(self):
        rec = AgentRecord(id="a1", status="running")
        self.assertTrue(rec.is_running)
        self.assertFalse(rec.is_terminal)
        rec2 = AgentRecord(id="a1", status="done")
        self.assertEqual(rec.changed_fields(rec2)["status"], ("running", "done"))
        rec3 = AgentRecord.from_json(rec.to_json())
        self.assertEqual(rec3.id, "a1")

    def test_task_elapsed(self):
        rec = TaskRecord(id="t1", name="n", started_at=100.0, finished_at=160.0)
        self.assertEqual(rec.elapsed, 60.0)
        rec2 = TaskRecord(id="t2", name="n")
        self.assertEqual(rec2.elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
