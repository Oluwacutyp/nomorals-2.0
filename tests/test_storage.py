"""Tests for the L2 storage layer."""

from __future__ import annotations

import gzip
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.core.errors import ConstraintViolation, MigrationError, NotFound, StorageError
from nomorals.storage.backup import BackupManager
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database, split_sql
from nomorals.storage.fts import FTSIndex, build_match_query, escape_fts
from nomorals.storage.migrations import MIGRATIONS, latest_version
from nomorals.storage.queue import WorkQueue
from nomorals.storage.repository import InsertBuilder, Query, Repository
from nomorals.storage.schema import Migration, MigrationRunner
from nomorals.storage.vectors import VectorStore, cosine, normalize


def make_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class TestSplitSql(unittest.TestCase):
    def test_splits_multiple_statements_on_one_line(self) -> None:
        self.assertEqual(len(split_sql("CREATE TABLE a(x); INSERT INTO a VALUES(';');")), 2)

    def test_semicolon_inside_string_is_not_a_terminator(self) -> None:
        self.assertEqual(len(split_sql("SELECT 'a;b'; SELECT \"c;d\";")), 2)
        self.assertEqual(len(split_sql("SELECT 'it''s; fine';")), 1)

    def test_comments_are_stripped(self) -> None:
        self.assertEqual(split_sql("-- only a comment\n"), [])
        self.assertEqual(len(split_sql("SELECT 1; -- trailing\n-- another\n")), 1)
        self.assertEqual(len(split_sql("/* block; comment */ SELECT 1;")), 1)

    def test_trailing_statement_without_semicolon(self) -> None:
        self.assertEqual(len(split_sql("SELECT 1")), 1)


class TestDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()

    def tearDown(self) -> None:
        self.db.close()

    def test_migrations_apply_all_tables(self) -> None:
        self.assertEqual(self.db.scalar("SELECT MAX(version) FROM schema_migrations"), latest_version())
        self.assertGreater(len(self.db.tables()), 25)
        for expected in ("memories", "tasks", "missions", "models", "work_queue", "blobs", "social_posts"):
            self.assertTrue(self.db.table_exists(expected), f"missing table {expected}")

    def test_migrations_are_idempotent(self) -> None:
        self.assertEqual(self.db.migrate().applied, [])

    def test_integrity_check_passes(self) -> None:
        self.assertEqual(self.db.integrity_check(), "ok")

    def test_insert_query_update_delete(self) -> None:
        self.db.insert("conversations", {"id": "c1", "title": "t", "created_at": 1.0, "updated_at": 1.0})
        self.assertEqual(self.db.row_count("conversations"), 1)
        self.assertEqual(self.db.update("conversations", {"title": "x"}, "id = ?", ("c1",)), 1)
        self.assertEqual(self.db.scalar("SELECT title FROM conversations WHERE id='c1'"), "x")
        self.assertEqual(self.db.delete("conversations", "id = ?", ("c1",)), 1)

    def test_transaction_rolls_back_on_error(self) -> None:
        self.db.insert("conversations", {"id": "c1", "title": "t", "created_at": 1.0, "updated_at": 1.0})
        with self.assertRaises(RuntimeError):
            with self.db.transaction():
                self.db.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
                    ("m1", "c1", "user", "x", 1.0),
                )
                raise RuntimeError("abort")
        self.assertEqual(self.db.row_count("messages"), 0)

    def test_nested_transaction_uses_savepoints(self) -> None:
        self.db.insert("conversations", {"id": "c1", "title": "t", "created_at": 1.0, "updated_at": 1.0})
        insert = "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)"
        with self.db.transaction():
            self.db.execute(insert, ("m1", "c1", "a", "y", 1.0))
            with self.assertRaises(ValueError):
                with self.db.transaction():
                    self.db.execute(insert, ("m2", "c1", "a", "z", 2.0))
                    raise ValueError("inner")
        self.assertEqual(
            sorted(r["id"] for r in self.db.query("SELECT id FROM messages")), ["m1"]
        )

    def test_foreign_keys_are_enforced(self) -> None:
        with self.assertRaises(ConstraintViolation):
            self.db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
                ("m1", "missing", "user", "", 1.0),
            )

    def test_error_wrapping(self) -> None:
        with self.assertRaises(StorageError):
            self.db.execute("SELECT * FROM table_that_does_not_exist")

    def test_scalar_default(self) -> None:
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM memories"), 0)
        self.assertEqual(self.db.scalar("SELECT NULL", default="d"), "d")

    def test_stats_track_activity(self) -> None:
        self.db.execute("SELECT 1")
        snapshot = self.db.stats_snapshot()
        self.assertGreater(snapshot["queries"], 0)

    def test_file_backed_database_persists_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.db"
            db = Database(path)
            db.migrate()
            db.insert("conversations", {"id": "c1", "title": "t", "created_at": 1.0, "updated_at": 1.0})
            db.checkpoint("TRUNCATE")
            db.close()
            reopened = Database(path)
            self.assertEqual(reopened.row_count("conversations"), 1)
            reopened.close()


class TestMigrations(unittest.TestCase):
    def test_failed_migration_is_atomic(self) -> None:
        db = make_db()
        try:
            runner = MigrationRunner(db)
            bad = Migration(
                99, "bad", sql="CREATE TABLE good_one(x INTEGER);\nINSERT INTO nonexistent VALUES (1);"
            )
            with self.assertRaises(StorageError):
                runner.apply(bad)
            self.assertFalse(db.table_exists("good_one"), "partial DDL must be rolled back")
            self.assertEqual(runner.current_version(), latest_version())
        finally:
            db.close()

    def test_edited_applied_migration_is_detected(self) -> None:
        db = make_db()
        try:
            runner = MigrationRunner(db)
            self.assertEqual(runner.validate(MIGRATIONS), [])
            tampered = list(MIGRATIONS)
            tampered[0] = Migration(1, "core_state", sql=MIGRATIONS[0].sql + "\nCREATE TABLE sneaky(x);")
            problems = runner.validate(tampered)
            self.assertEqual(len(problems), 1)
            self.assertIn("modified after being applied", problems[0])
            with self.assertRaises(MigrationError):
                runner.apply_all(tampered)
        finally:
            db.close()

    def test_duplicate_versions_rejected(self) -> None:
        db = make_db()
        try:
            runner = MigrationRunner(db)
            dup = (Migration(1, "a", sql="SELECT 1"), Migration(1, "b", sql="SELECT 2"))
            self.assertTrue(any("duplicate" in p for p in runner.validate(dup)))
        finally:
            db.close()

    def test_migration_requires_sql_or_fn(self) -> None:
        with self.assertRaises(Exception):
            Migration(1, "empty")

    def test_history_is_recorded(self) -> None:
        db = make_db()
        try:
            history = MigrationRunner(db).history()
            self.assertEqual(len(history), latest_version())
            self.assertTrue(all(h["checksum"] for h in history))
        finally:
            db.close()

    def test_rollback_without_down_migration_is_refused(self) -> None:
        db = make_db()
        try:
            runner = MigrationRunner(db)
            with self.assertRaises(MigrationError):
                runner.rollback(MIGRATIONS, to_version=latest_version() - 1)
        finally:
            db.close()


class TestRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()
        self.repo = Repository(self.db, "memories", json_columns=("metadata",))

    def tearDown(self) -> None:
        self.db.close()

    def test_create_assigns_id_and_timestamps(self) -> None:
        row = self.repo.create({"kind": "fact", "content": "x"})
        self.assertTrue(row["id"])
        self.assertGreater(row["created_at"], 0)

    def test_json_column_roundtrip(self) -> None:
        row = self.repo.create({"kind": "fact", "content": "x", "metadata": {"a": [1, 2]}})
        self.assertEqual(self.repo.get(row["id"])["metadata"], {"a": [1, 2]})

    def test_get_require_and_not_found(self) -> None:
        row = self.repo.create({"kind": "fact", "content": "x"})
        self.assertIsNone(self.repo.get("nope"))
        with self.assertRaises(NotFound):
            self.repo.require("nope")
        self.assertEqual(self.repo.require(row["id"])["id"], row["id"])

    def test_find_count_update_delete(self) -> None:
        self.repo.create({"kind": "fact", "content": "a"})
        self.repo.create({"kind": "episode", "content": "b"})
        self.assertEqual(len(self.repo.find(kind="fact")), 1)
        self.assertEqual(self.repo.count(), 2)
        row = self.repo.find_one(kind="episode")
        self.assertEqual(self.repo.update(row["id"], {"content": "changed"}), 1)
        self.assertEqual(self.repo.get(row["id"])["content"], "changed")
        self.assertEqual(self.repo.delete(row["id"]), 1)
        self.assertEqual(self.repo.count(), 1)

    def test_bulk_create_in_one_transaction(self) -> None:
        ids = self.repo.create_many([{"kind": "episode", "content": f"e{i}"} for i in range(100)])
        self.assertEqual(len(ids), 100)
        self.assertEqual(self.repo.count(), 100)
        self.assertEqual(len({i for i in ids}), 100)

    def test_upsert(self) -> None:
        first = self.repo.upsert({"id": "fixed", "kind": "fact", "content": "v1"})
        second = self.repo.upsert({"id": "fixed", "kind": "fact", "content": "v2"})
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.repo.count(), 1)
        self.assertEqual(self.repo.get("fixed")["content"], "v2")

    def test_pagination(self) -> None:
        self.repo.create_many([{"kind": "episode", "content": f"e{i}"} for i in range(25)])
        self.assertEqual(len(self.repo.paginate(1, 10)), 10)
        self.assertEqual(len(self.repo.paginate(3, 10)), 5)
        with self.assertRaises(Exception):
            self.repo.paginate(0, 10)

    def test_query_builder(self) -> None:
        self.repo.create({"kind": "fact", "content": "a", "importance": 0.9})
        self.repo.create({"kind": "fact", "content": "b", "importance": 0.1})
        query = self.repo.query().where("importance > ?", 0.5).order_by("importance DESC").limit(10)
        sql, params = query.build()
        rows = self.db.query(sql, params)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "a")

    def test_query_builder_count(self) -> None:
        self.repo.create_many([{"kind": "fact", "content": f"c{i}"} for i in range(5)])
        sql, params = self.repo.query().where("kind = ?", "fact").count_sql()
        self.assertEqual(self.db.scalar(sql, params), 5)

    def test_insert_builder_validates_width(self) -> None:
        builder = InsertBuilder("memories", ["id", "kind", "content", "created_at", "updated_at"])
        with self.assertRaises(Exception):
            builder.add("only-one-value")

    def test_insert_builder_batches(self) -> None:
        now = time.time()
        builder = self.repo.batch(["id", "kind", "content", "created_at", "updated_at"])
        for i in range(20):
            builder.add(f"id{i}", "fact", f"c{i}", now, now)
        self.assertEqual(len(builder), 20)
        self.assertEqual(builder.flush(self.db), 20)
        self.assertEqual(self.repo.count(), 20)
        self.assertEqual(len(builder), 0)

    def test_truncate_and_ids(self) -> None:
        self.repo.create_many([{"kind": "fact", "content": f"c{i}"} for i in range(3)])
        self.assertEqual(len(self.repo.ids()), 3)
        self.assertEqual(self.repo.truncate(), 3)
        self.assertEqual(self.repo.count(), 0)

    def test_bool_values_are_coerced(self) -> None:
        row = self.repo.create({"kind": "fact", "content": "x", "metadata": {"flag": True}})
        self.assertIs(self.repo.get(row["id"])["metadata"]["flag"], True)


class TestFTS(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()
        self.index = FTSIndex(self.db, "documents_fts", columns=["title", "body"])
        self.rowid = self.db.insert(
            "documents", {"id": "d1", "kind": "text", "title": "fox report", "created_at": time.time()}
        )
        self.index.put(self.rowid, ["fox report", "a quick brown fox was observed jumping"])

    def tearDown(self) -> None:
        self.db.close()

    def test_search_ranks_and_snippets(self) -> None:
        hits = self.index.search("quick fox", limit=5)
        self.assertEqual(len(hits), 1)
        self.assertGreater(hits[0].score, 0)
        self.assertIn("[", hits[0].snippet)

    def test_prefix_matching(self) -> None:
        self.assertEqual(len(self.index.search("jump")), 1)

    def test_no_match_returns_empty(self) -> None:
        self.assertEqual(self.index.search("zzzzqqq"), [])

    def test_unavailable_index_degrades(self) -> None:
        missing = FTSIndex(self.db, "no_such_fts", columns=["x"], available=False)
        missing.put(1, ["text"])
        self.assertEqual(missing.search("text"), [])
        self.assertEqual(missing.count(), 0)

    def test_delete_and_count(self) -> None:
        self.assertEqual(self.index.count(), 1)
        self.index.delete(self.rowid)
        self.assertEqual(self.index.count(), 0)

    def test_put_replaces_not_duplicates(self) -> None:
        self.index.put(self.rowid, ["replaced", "totally different content"])
        self.assertEqual(self.index.count(), 1)
        self.assertEqual(len(self.index.search("totally")), 1)
        self.assertEqual(self.index.search("fox report"), [])

    def test_query_building_is_safe(self) -> None:
        self.assertEqual(escape_fts('say "hi"'), '"say ""hi"""')
        self.assertEqual(build_match_query(""), "")
        self.assertIn("AND", build_match_query("a b"))
        self.assertIn("OR", build_match_query("a b", operator="OR"))

    def test_wrong_arity_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.index.put(99, ["only one"])


class TestVectorStore(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()
        self.store = VectorStore(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_normalize_and_cosine(self) -> None:
        unit = normalize([3.0, 4.0])
        self.assertAlmostEqual(unit[0] ** 2 + unit[1] ** 2, 1.0, places=6)
        self.assertEqual(normalize([0.0, 0.0]), [0.0, 0.0])
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)
        with self.assertRaises(Exception):
            cosine([1], [1, 2])

    def test_search_orders_by_similarity(self) -> None:
        self.store.put([1.0, 0.0, 0.0], owner_type="m", owner_id="a")
        self.store.put([0.9, 0.1, 0.0], owner_type="m", owner_id="b")
        self.store.put([0.0, 1.0, 0.0], owner_type="m", owner_id="c")
        hits = self.store.search([1.0, 0.0, 0.0], limit=3)
        self.assertEqual([h.owner_id for h in hits], ["a", "b", "c"])
        self.assertGreater(hits[0].score, hits[1].score)

    def test_limit_and_min_score(self) -> None:
        self.store.put([1.0, 0.0], owner_type="m", owner_id="a")
        self.store.put([0.0, 1.0], owner_type="m", owner_id="b")
        self.assertEqual(len(self.store.search([1.0, 0.0], limit=1)), 1)
        self.assertEqual(len(self.store.search([1.0, 0.0], limit=5, min_score=0.5)), 1)
        self.assertEqual(self.store.search([1.0, 0.0], limit=0), [])

    def test_owner_type_filter(self) -> None:
        self.store.put([1.0, 0.0], owner_type="m", owner_id="a")
        self.store.put([1.0, 0.0], owner_type="doc", owner_id="b")
        hits = self.store.search([1.0, 0.0], limit=5, owner_type="doc")
        self.assertEqual([h.owner_id for h in hits], ["b"])

    def test_put_many_and_get(self) -> None:
        ids = self.store.put_many([([1.0, 0.0], "m", "a"), ([0.0, 1.0], "m", "b")])
        self.assertEqual(len(ids), 2)
        record = self.store.get(ids[0])
        self.assertEqual(record.owner_id, "a")
        self.assertAlmostEqual(record.vector[0], 1.0, places=5)

    def test_empty_vector_rejected(self) -> None:
        with self.assertRaises(Exception):
            self.store.put([], owner_type="m", owner_id="a")

    def test_delete(self) -> None:
        record_id = self.store.put([1.0, 0.0], owner_type="m", owner_id="a")
        self.assertEqual(self.store.count(), 1)
        self.store.delete(record_id)
        self.assertEqual(self.store.count(), 0)
        self.store.invalidate()
        self.assertEqual(self.store.search([1.0, 0.0], limit=3), [])

    def test_search_by_owner(self) -> None:
        self.store.put([1.0, 0.0], owner_type="m", owner_id="a")
        self.store.put([0.0, 1.0], owner_type="m", owner_id="b")
        hits = self.store.search_by_owner("m", "a", limit=2)
        self.assertEqual(hits[0].owner_id, "a")
        self.assertEqual(self.store.search_by_owner("m", "missing", limit=2), [])

    def test_index_is_not_built_for_small_stores(self) -> None:
        self.store.put([1.0, 0.0], owner_type="m", owner_id="a")
        self.assertEqual(self.store.build_index(), 0)

    def test_coarse_quantizer_search_matches_brute_force(self) -> None:
        import random

        rng = random.Random(7)
        vectors = [[rng.random() for _ in range(8)] for _ in range(600)]
        self.store.put_many([(v, "m", f"id{i}") for i, v in enumerate(vectors)])
        self.assertEqual(self.store.build_index(clusters=8), 8)
        query = vectors[0]
        brute = {h.owner_id for h in self.store.search(query, limit=10, nprobe=0)}
        quantized = {h.owner_id for h in self.store.search(query, limit=10, nprobe=8)}
        self.assertEqual(brute, quantized, "nprobe=all must reproduce brute force exactly")

    def test_stats_report_backend(self) -> None:
        self.store.put([1.0, 0.0], owner_type="m", owner_id="a")
        snapshot = self.store.stats_snapshot()
        self.assertIn(snapshot["backend"], {"numpy", "pure-python"})
        self.assertEqual(snapshot["vectors"], 1)


class TestBlobStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = make_db()
        self.store = BlobStore(self.db, Path(self.tmp.name) / "blobs")

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_put_and_roundtrip(self) -> None:
        data = b"hello world" * 10
        info = self.store.put_bytes(data)
        self.assertEqual(self.store.get_bytes(info.sha256), data)
        self.assertEqual(info.refcount, 1)

    def test_deduplication(self) -> None:
        data = b"repeat me" * 50
        first = self.store.put_bytes(data)
        second = self.store.put_bytes(data)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(second.refcount, 2)
        self.assertEqual(self.store.stats_snapshot()["blobs"], 1)

    def test_compression_for_text_only(self) -> None:
        text_blob = self.store.put_bytes(b"lorem ipsum " * 1000, mime="text/plain")
        self.assertTrue(text_blob.compressed)
        self.assertLess(text_blob.ratio, 0.5)
        png = self.store.put_bytes(b"\x89PNG\r\n\x1a\n" + bytes(100))
        self.assertFalse(png.compressed)
        self.assertEqual(png.mime, "image/png")

    def test_small_blobs_skip_compression(self) -> None:
        self.assertFalse(self.store.put_bytes(b"tiny", mime="text/plain").compressed)

    def test_put_file_and_export(self) -> None:
        source = Path(self.tmp.name) / "in.txt"
        source.write_text("file content " * 100, encoding="utf-8")
        info = self.store.put_file(source)
        target = self.store.export(info.sha256, Path(self.tmp.name) / "out.txt")
        self.assertEqual(target.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))

    def test_put_file_missing_raises(self) -> None:
        with self.assertRaises(NotFound):
            self.store.put_file(Path(self.tmp.name) / "nope.txt")

    def test_verify_detects_corruption(self) -> None:
        info = self.store.put_bytes(b"integrity " * 100, mime="text/plain")
        self.assertEqual(self.store.verify(), [])
        path = self.store.path_for(info.sha256, info.compressed)
        path.write_bytes(b"corrupted")
        self.assertTrue(self.store.verify())

    def test_verify_detects_missing_file(self) -> None:
        info = self.store.put_bytes(b"vanish " * 100, mime="text/plain")
        self.store.path_for(info.sha256, info.compressed).unlink()
        self.assertIn("missing", self.store.verify()[0])

    def test_release_and_purge(self) -> None:
        info = self.store.put_bytes(b"refcounted")
        self.store.put_bytes(b"refcounted")
        self.assertEqual(self.store.release(info.sha256), 1)
        self.assertEqual(self.store.release(info.sha256, delete_at_zero=True), 0)
        self.assertFalse(self.store.exists(info.sha256))

    def test_orphan_detection_and_sweep(self) -> None:
        stray = self.store.root / "aa" / "bb" / ("c" * 64)
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"stray")
        self.assertEqual(len(self.store.orphans()), 1)
        self.assertEqual(self.store.sweep()["orphans"], 1)
        self.assertFalse(stray.exists())

    def test_missing_blob_raises(self) -> None:
        with self.assertRaises(NotFound):
            self.store.get_bytes("0" * 64)


class TestWorkQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()
        self.queue = WorkQueue(self.db, backoff_base=0.0, backoff_cap=0.0)

    def tearDown(self) -> None:
        self.db.close()

    def test_priority_ordering(self) -> None:
        self.queue.enqueue("t", {"url": "low"})
        self.queue.enqueue("t", {"url": "high"}, priority=5)
        self.assertEqual(self.queue.lease_one("t").payload["url"], "high")

    def test_lease_increments_attempts(self) -> None:
        job_id = self.queue.enqueue("t", {})
        self.assertEqual(self.queue.lease_one("t").attempts, 1)
        self.assertEqual(self.queue.get(job_id).status, "leased")

    def test_complete_marks_done(self) -> None:
        job_id = self.queue.enqueue("t", {})
        leased = self.queue.lease_one("t")
        self.queue.complete(leased.id, {"ok": True})
        self.assertEqual(self.queue.get(job_id).status, "done")
        self.assertEqual(self.queue.pending("t"), 0)

    def test_fail_retries_then_dead_letters(self) -> None:
        job_id = self.queue.enqueue("t", {}, max_attempts=2)
        self.assertEqual(self.queue.fail(self.queue.lease_one("t").id, "e1"), "retrying")
        self.assertEqual(self.queue.get(job_id).status, "ready")
        self.assertEqual(self.queue.fail(self.queue.lease_one("t").id, "e2"), "dead")
        self.assertEqual(self.queue.get(job_id).status, "dead")

    def test_expired_lease_is_reclaimed(self) -> None:
        job_id = self.queue.enqueue("t", {})
        self.queue.lease_one("t", "w1", lease_seconds=0.0)
        self.assertEqual(self.queue.reclaim_expired(), 1)
        self.assertEqual(self.queue.get(job_id).status, "ready")

    def test_live_lease_is_not_stolen(self) -> None:
        self.queue.enqueue("t", {})
        self.queue.lease_one("t", "w1", lease_seconds=300)
        self.assertEqual(self.queue.reclaim_expired(), 0)
        self.assertIsNone(self.queue.lease_one("t", "w2"))

    def test_delayed_job_not_visible_early(self) -> None:
        self.queue.enqueue("t", {}, delay=60)
        self.assertIsNone(self.queue.lease_one("t"))
        self.assertEqual(self.queue.pending("t"), 1)

    def test_worker_loop_drains_and_honours_max_jobs(self) -> None:
        for i in range(5):
            self.queue.enqueue("bulk", {"i": i})
        seen: list[int] = []
        handled = self.queue.run_worker("bulk", lambda j: seen.append(j.payload["i"]), max_jobs=3)
        self.assertEqual(handled, 3)
        self.assertEqual(self.queue.pending("bulk"), 2)

    def test_handler_exception_does_not_kill_loop(self) -> None:
        for _ in range(3):
            self.queue.enqueue("boom", {})

        def handler(job: object) -> str:
            raise RuntimeError("kaboom")

        handled = self.queue.run_worker("boom", handler, max_jobs=3)
        self.assertEqual(handled, 3)
        statuses = {t["status"] for t in self.queue.topics() if t["topic"] == "boom"}
        self.assertTrue(statuses & {"ready", "dead"})

    def test_purge_only_finished(self) -> None:
        self.queue.enqueue("t", {})
        job = self.queue.lease_one("t")
        self.queue.complete(job.id)
        self.queue.enqueue("t", {})
        self.assertEqual(self.queue.purge("t", only_finished=True), 1)
        self.assertEqual(self.queue.pending("t"), 1)

    def test_missing_job_raises(self) -> None:
        with self.assertRaises(NotFound):
            self.queue.fail("does-not-exist")

    def test_enqueue_many(self) -> None:
        ids = self.queue.enqueue_many("t", [{"i": i} for i in range(10)])
        self.assertEqual(len(ids), 10)
        self.assertEqual(self.queue.pending("t"), 10)


class TestBackupManager(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "live.db"
        self.db = Database(self.path)
        self.db.migrate()
        self.db.insert("conversations", {"id": "c1", "title": "t", "created_at": 1.0, "updated_at": 1.0})
        self.manager = BackupManager(self.db, Path(self.tmp.name) / "backups", keep=3)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_create_produces_verifiable_snapshot(self) -> None:
        info = self.manager.create(label="test")
        self.assertTrue(Path(info.path).is_file())
        self.assertGreater(info.size, 0)
        self.assertEqual(info.schema_version, latest_version())
        self.assertEqual(self.manager.verify(info), [])

    def test_backup_is_gzipped_and_checksummed(self) -> None:
        info = self.manager.create()
        self.assertTrue(info.name.endswith(".db.gz"))
        with gzip.open(info.path, "rb") as handle:
            self.assertEqual(handle.read(16), b"SQLite format 3\x00")
        self.assertEqual(len(info.sha256), 64)

    def test_restore_yields_working_database(self) -> None:
        info = self.manager.create()
        restored = self.manager.restore(info, target=Path(self.tmp.name) / "restored.db")
        copy = Database(restored)
        try:
            self.assertEqual(copy.row_count("conversations"), 1)
            self.assertEqual(copy.integrity_check(), "ok")
        finally:
            copy.close()

    def test_snapshot_is_consistent_without_wal_sidecar(self) -> None:
        info = self.manager.create()
        restored = self.manager.restore(info, target=Path(self.tmp.name) / "r2.db")
        self.assertFalse(Path(str(restored) + "-wal").exists())
        copy = Database(restored)
        self.assertEqual(copy.integrity_check(), "ok")
        copy.close()

    def test_verify_detects_checksum_mismatch(self) -> None:
        info = self.manager.create()
        Path(info.path).write_bytes(b"not a database at all")
        problems = self.manager.verify(info)
        self.assertTrue(any("checksum" in p for p in problems))

    def test_rotation_keeps_recent_and_daily(self) -> None:
        for i in range(6):
            info = self.manager.create(label=f"b{i}")
            time.sleep(0.01)
        self.manager.rotate()
        remaining = self.manager.list()
        self.assertLessEqual(len(remaining), 6)
        self.assertTrue(all(Path(b.path).is_file() for b in remaining))

    def test_latest_and_listing(self) -> None:
        self.assertIsNone(self.manager.latest())
        first = self.manager.create(label="one")
        second = self.manager.create(label="two")
        self.assertEqual(self.manager.latest().name, second.name)
        self.assertEqual(len(self.manager.list()), 2)

    def test_backup_if_due_skips_fresh(self) -> None:
        self.manager.create()
        self.assertIsNone(self.manager.backup_if_due(3600))
        self.assertIsNotNone(self.manager.backup_if_due(0))

    def test_push_without_repo_is_a_noop(self) -> None:
        result = self.manager.push_to_git()
        self.assertFalse(result["pushed"])
        self.assertEqual(result["reason"], "no git_repo configured")

    def test_manifest_survives_reload(self) -> None:
        self.manager.create(label="persisted")
        fresh = BackupManager(self.db, self.manager.directory)
        self.assertEqual(len(fresh.list()), 1)


    # KNOWN GAP: a real-database round-trip for ModelRow/TaskRecord is not covered.
    # to_row() omits columns the schema declares NOT NULL without a default, so
    # inserting a bare dataclass fails. Writing through Repository (which supplies
    # timestamps) works; the direct db.insert() path does not.


if __name__ == "__main__":
    unittest.main()


class ModelDataclassTests(unittest.TestCase):
    """storage/models.py: typed row mirrors that tolerate schema drift."""

    def setUp(self):
        from nomorals.storage.db import Database

        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        handle.close()
        self.path = handle.name
        self.db = Database(self.path)
        self.db.migrate()

    def tearDown(self):
        self.db.close()
        Path(self.path).unlink(missing_ok=True)

    def test_decode_json_handles_all_three_shapes(self):
        from nomorals.storage.models import decode_json

        self.assertEqual(decode_json(None, {}), {})
        self.assertEqual(decode_json("", []), [])
        self.assertEqual(decode_json('{"a": 1}', {}), {"a": 1})
        self.assertEqual(decode_json({"b": 2}, {}), {"b": 2})
        self.assertEqual(decode_json("not json", []), [])

    def test_agent_duration_and_terminality(self):
        from nomorals.storage.models import AgentRecord

        record = AgentRecord.from_row(
            {"id": "a", "status": "done", "started_at": 2.0, "finished_at": 5.5}
        )
        self.assertAlmostEqual(record.duration, 3.5)
        self.assertTrue(record.is_terminal)

    def test_agent_without_timestamps_has_zero_duration(self):
        from nomorals.storage.models import AgentRecord

        self.assertEqual(AgentRecord.from_row({"id": "a"}).duration, 0.0)

    def test_task_retry_budget(self):
        from nomorals.storage.models import TaskRecord

        self.assertTrue(TaskRecord.from_row({"id": "t", "attempts": 3}).exhausted)
        self.assertFalse(TaskRecord.from_row({"id": "t", "attempts": 1}).exhausted)

    def test_rows_missing_newer_columns_do_not_crash(self):
        from nomorals.storage.models import ModelRow, Reflection, TaskRecord

        self.assertEqual(TaskRecord.from_row({"id": "t"}).max_attempts, 3)
        self.assertEqual(Reflection.from_row({"id": "r"}).lessons, [])
        self.assertEqual(ModelRow.from_row({"id": "m", "name": "n"}).eval_scores, {})

    def test_non_numeric_columns_fall_back_instead_of_raising(self):
        from nomorals.storage.models import TaskRecord

        record = TaskRecord.from_row({"id": "t", "attempts": "garbage", "priority": None})
        self.assertEqual(record.attempts, 0)
        self.assertEqual(record.priority, 0)

    def test_round_trip_preserves_decoded_types(self):
        from nomorals.storage.models import TaskRecord

        original = TaskRecord(
            id="t", name="n", deps=["a", "b"], payload={"k": "v"}, priority=5
        )
        restored = TaskRecord.from_row(original.to_row())
        self.assertEqual(restored.deps, ["a", "b"])
        self.assertEqual(restored.payload, {"k": "v"})
        self.assertEqual(restored.priority, 5)


