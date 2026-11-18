"""DB stall fix (2026-11-18): unbounded scans + uncheckpointed WAL growth.

The 2026-11-18 EC2 alerts: morning-pulse timed out after 1800s with
``NewsAgent._known_urls`` ~155s and a CostLog write retrying ~190s, and
the failure-loop tick timed out after 300s with
``TrajectoryStore.failure_clusters`` ~115s.  The WAL sat at 26.7MB /
64MB (its journal_size_limit cap) with a reader pinning checkpoints.

What the probes proved (~/workspace/dbfix-verify/probe.py, probe3/4):
- ``SELECT url ... WHERE url IN (...)`` already SEARCHes the UNIQUE(url)
  autoindex (0.002s at 120k rows) — there was never a missing url index,
  so none is added.  The news growth problem is the TABLE growing
  forever (no prune) plus the delivery-state queries scanning it.
- ``failure_clusters`` full-scans: the (success, created_at) index alone
  does not fix it (0.097s vs 0.087s) — the 90-day bound is the fix
  (0.020s, SEARCH using the new index).
- ``wal_checkpoint(TRUNCATE)`` stalls out the 5s busy_timeout when a
  reader pins the log; PASSIVE returns at once with busy=1.  The old
  ``Database.checkpoint`` returned (log, checkpointed) under a
  "(busy, log_pages)" docstring, so ``checkpoint_maintenance`` treated
  any non-empty WAL as busy and retried TRUNCATE into the timeout.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.news import (
    NEWS_RETENTION_DAYS,
    STATE_ARCHIVED,
    STATE_DIGESTED,
    STATE_PUSHED,
    STATE_UNDELIVERED,
    NewsAgent,
)
from nomorals.cognition.trajectories import (
    DEFAULT_RETENTION_DAYS,
    TrajectoryStore,
)
from nomorals.storage.db import (
    WAL_CHECKPOINT_JOB,
    WAL_CHECKPOINT_SPEC,
    WAL_CHECKPOINT_TOOL,
    Database,
    ensure_wal_checkpoint_job,
)
from nomorals.storage.migrations import latest_version
from nomorals.tools.autonomy import register as register_autonomy
from nomorals.tools.registry import ToolRegistry

DAY = 86400.0


def _mem_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _plan_details(db: Database, sql: str, params=()) -> list[str]:
    return [str(r.get("detail", "")) for r in db.explain(sql, params)]


def _bulk_insert(db: Database, sql: str, rows: list[tuple]) -> None:
    """Fixture seeding: one transaction on the caller's connection."""
    conn = db._connection()  # noqa: SLF001 - test fixture setup only
    conn.execute("BEGIN")
    try:
        conn.executemany(sql, rows)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    db.execute("ANALYZE")


def _seed_news(db: Database, n: int = 3000) -> None:
    now = time.time()
    rows = []
    for i in range(n):
        age = (i % 400) + (i // 400) * 0.001
        state = ("undelivered" if i % 50 == 0
                 else "pushed" if i % 2 else "archived")
        rows.append((f"n{i}", f"src{i % 86}", f"Title {i} - Reuters",
                     f"https://example.com/story/{i}", f"s {i}", "2026-01-01",
                     now - age * DAY, state))
    _bulk_insert(
        db,
        "INSERT INTO news_items (id, source, title, url, summary,"
        " published, created_at, delivery_state) VALUES (?,?,?,?,?,?,?,?)",
        rows)


def _seed_traj(db: Database, n: int = 3000) -> None:
    now = time.time()
    rows = []
    for i in range(n):
        age = (i % 400) + (i // 400) * 0.001
        rows.append(
            (f"t{i}", f"kind{i % 12}", f"cap{i % 5}", f"model{i % 4}",
             f"skill{i % 6}", "", 0 if i % 5 < 2 else 1, 0.5, 0.0,
             "boom" if i % 5 < 2 else "", "runtime" if i % 5 < 2 else "",
             now - age * DAY))
    _bulk_insert(
        db,
        "INSERT INTO cog_trajectories (id, task_kind, capability,"
        " model_id, skill_id, tool, success, latency_s, cost, error,"
        " failure_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows)


class TestMigrationV89(unittest.TestCase):
    def test_v89_applies_and_indexes_news_delivery(self) -> None:
        db = _mem_db()
        try:
            self.assertEqual(latest_version(), 89)
            idx = {r["name"] for r in db.query("PRAGMA index_list(news_items)")}
            self.assertIn("idx_news_items_delivery", idx)
            cols = {r["name"] for r in db.query("PRAGMA table_info(news_items)")}
            self.assertIn("delivery_state", cols)
            # Idempotent: a second migrate applies nothing.
            self.assertEqual(db.migrate().applied, [])
        finally:
            db.close()

    def test_news_delivery_plan_is_search(self) -> None:
        db = _mem_db()
        try:
            _seed_news(db)
            for sql, params in (
                ("SELECT COUNT(*) AS n FROM news_items WHERE delivery_state = ?",
                 ("undelivered",)),
                ("SELECT * FROM news_items WHERE delivery_state = ? "
                 "ORDER BY created_at ASC LIMIT 200", ("undelivered",)),
            ):
                details = _plan_details(db, sql, params)
                self.assertTrue(
                    any("SEARCH news_items USING" in d
                        and "idx_news_items_delivery" in d
                        for d in details),
                    f"expected SEARCH on idx_news_items_delivery: {details}")
                self.assertFalse(any("SCAN news_items" in d for d in details),
                                 f"unexpected full scan: {details}")
        finally:
            db.close()

    def test_url_lookup_needs_no_new_index(self) -> None:
        # The UNIQUE(url) autoindex already serves _known_urls; a
        # redundant idx_news_items_url must NOT exist.
        db = _mem_db()
        try:
            _seed_news(db)
            chunk = tuple(f"https://example.com/story/{i}" for i in range(500))
            placeholders = ",".join("?" * len(chunk))
            details = _plan_details(
                db, f"SELECT url FROM news_items WHERE url IN ({placeholders})",
                chunk)
            self.assertTrue(any("SEARCH news_items USING COVERING INDEX" in d
                                and "url=?" in d for d in details),
                            f"expected covering-index SEARCH: {details}")
            idx = {r["name"] for r in db.query("PRAGMA index_list(news_items)")}
            self.assertNotIn("idx_news_items_url", idx)
        finally:
            db.close()


class TestFailureClustersBound(unittest.TestCase):
    def _store_with_clusters(self) -> tuple[Database, TrajectoryStore]:
        db = _mem_db()
        store = TrajectoryStore(db)
        now = time.time()

        def fail(i: int, error: str, age_days: float) -> None:
            db.execute(
                "INSERT INTO cog_trajectories (id, task_kind, capability,"
                " model_id, skill_id, tool, success, latency_s, cost, error,"
                " failure_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"c{i}", "kind", "cap", "m", "s", "", 0, 0.5, 0.0,
                 error, "runtime", now - age_days * DAY))
        for i in range(5):
            fail(i, "boom A", 1.0)
        for i in range(5, 8):
            fail(i, "boom A", 100.0)
        for i in range(8, 10):
            fail(i, "boom B", 100.0)
        db.execute("ANALYZE")
        return db, store

    def test_bounded_matches_unbounded_on_window(self) -> None:
        db, store = self._store_with_clusters()
        try:
            bounded = {c["error_signature"]: c["count"]
                       for c in store.failure_clusters()}
            self.assertEqual(bounded.get("boom A"), 5)
            self.assertNotIn("boom B", bounded)
            unbounded = {c["error_signature"]: c["count"]
                         for c in store.failure_clusters(since_days=None)}
            self.assertEqual(unbounded.get("boom A"), 8)
            self.assertEqual(unbounded.get("boom B"), 2)
        finally:
            db.close()

    def test_bounded_query_plan_is_search(self) -> None:
        db = _mem_db()
        try:
            store = TrajectoryStore(db)
            _seed_traj(db)
            cutoff = time.time() - DEFAULT_RETENTION_DAYS * DAY
            details = _plan_details(
                db,
                "SELECT id, task_kind, model_id, skill_id, error,"
                " failure_class, created_at FROM cog_trajectories"
                " WHERE success = 0 AND created_at >= ?",
                (cutoff,))
            self.assertTrue(
                any("SEARCH cog_trajectories USING INDEX "
                    "idx_cog_traj_success_created" in d for d in details),
                f"expected SEARCH on idx_cog_traj_success_created: {details}")
            self.assertFalse(any("SCAN cog_trajectories" in d for d in details),
                             f"unexpected full scan: {details}")
            # The method itself returns clusters from the bounded window.
            self.assertTrue(store.failure_clusters())
        finally:
            db.close()


class TestNewsPrune(unittest.TestCase):
    def _agent(self, db: Database) -> NewsAgent:
        ctx = SimpleNamespace(db=db, settings=None)
        return NewsAgent(ctx)

    def _insert(self, db: Database, row_id: str, age_days: float,
                state: str | None) -> None:
        db.execute(
            "INSERT INTO news_items (id, source, title, url, summary,"
            " published, created_at, delivery_state) VALUES (?,?,?,?,?,?,?,?)",
            (row_id, "src", f"Title {row_id}", f"https://example.com/{row_id}",
             "s", "2026-01-01", time.time() - age_days * DAY, state))

    def test_prune_drops_settled_old_keeps_live(self) -> None:
        db = _mem_db()
        try:
            agent = self._agent(db)
            self._insert(db, "old-pushed", 100.0, STATE_PUSHED)
            self._insert(db, "old-archived", 100.0, STATE_ARCHIVED)
            self._insert(db, "old-digested", 100.0, STATE_DIGESTED)
            self._insert(db, "old-legacy-null", 100.0, None)
            self._insert(db, "old-undelivered", 100.0, STATE_UNDELIVERED)
            self._insert(db, "new-pushed", 10.0, STATE_PUSHED)
            self._insert(db, "new-undelivered", 1.0, STATE_UNDELIVERED)
            dropped = agent.prune_news_items()
            self.assertEqual(dropped, 4)
            remaining = {r["id"] for r in db.query("SELECT id FROM news_items")}
            self.assertEqual(remaining, {"old-undelivered", "new-pushed",
                                         "new-undelivered"})
        finally:
            db.close()

    def test_prune_keeps_14_day_titles_and_recent_urls(self) -> None:
        db = _mem_db()
        try:
            agent = self._agent(db)
            self._insert(db, "fortnight", 13.0, STATE_PUSHED)
            self._insert(db, "old-url", 100.0, STATE_PUSHED)
            agent.prune_news_items()
            titles = [r["title"] for r in db.query(
                "SELECT title FROM news_items WHERE created_at >= ?",
                (time.time() - 14 * DAY,))]
            self.assertIn("Title fortnight", titles)
            known = agent._known_urls(["https://example.com/fortnight"])
            self.assertEqual(known, {"https://example.com/fortnight"})
        finally:
            db.close()

    def test_maybe_prune_is_hourly_gated(self) -> None:
        db = _mem_db()
        try:
            agent = self._agent(db)
            self._insert(db, "g1", 100.0, STATE_PUSHED)
            agent._maybe_prune_news()
            self.assertEqual(
                db.query("SELECT COUNT(*) AS n FROM news_items")[0]["n"], 0)
            self._insert(db, "g2", 100.0, STATE_PUSHED)
            agent._maybe_prune_news()  # gated: still within the hour
            self.assertEqual(
                db.query("SELECT COUNT(*) AS n FROM news_items")[0]["n"], 1)
            agent._last_prune_news = float("-inf")
            agent._maybe_prune_news()
            self.assertEqual(
                db.query("SELECT COUNT(*) AS n FROM news_items")[0]["n"], 0)
        finally:
            db.close()

    def test_prune_retention_constant(self) -> None:
        self.assertEqual(NEWS_RETENTION_DAYS, DEFAULT_RETENTION_DAYS)


class TestCheckpointTupleAndMaintenance(unittest.TestCase):
    def _file_db(self, tmp: str) -> Database:
        return Database(os.path.join(tmp, "wal.db"))

    def test_checkpoint_returns_busy_and_log_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._file_db(tmp)
            try:
                db.migrate()
                # Migration itself leaves WAL frames; drain first.
                db.checkpoint("TRUNCATE")
                busy, log_pages = db.checkpoint("PASSIVE")
                self.assertEqual((busy, log_pages), (0, 0))
                db.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
                db.execute("INSERT INTO t (a) VALUES (1)")
                # Unpinned writes: busy must be 0 while log_pages > 0.
                # (The old tuple returned (log, checkpointed) here, so
                # its first element was > 0 even with nothing pinning.)
                busy, log_pages = db.checkpoint("PASSIVE")
                self.assertEqual(busy, 0)
                self.assertGreater(log_pages, 0)
            finally:
                db.close()

    def test_maintenance_never_blocks_on_pinned_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._file_db(tmp)
            try:
                db.migrate()
                db.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
                _bulk_insert(
                    db, "INSERT INTO t (a) VALUES (?)",
                    [(i,) for i in range(2000)])
                db.checkpoint("TRUNCATE")
                reader = sqlite3.connect(db.path)
                try:
                    reader.execute("BEGIN")
                    reader.execute("SELECT COUNT(*) FROM t").fetchone()
                    self.assertTrue(reader.in_transaction)
                    _bulk_insert(
                        db, "INSERT INTO t (a) VALUES (?)",
                        [(i,) for i in range(2000, 4000)])
                    before = db.wal_size_bytes()
                    self.assertGreater(before, 0)
                    t0 = time.perf_counter()
                    report = db.checkpoint_maintenance()
                    dt = time.perf_counter() - t0
                    self.assertLess(dt, 2.0, "maintenance stalled on a pin")
                    self.assertTrue(report["busy"])
                    self.assertEqual(db.wal_size_bytes(), before)
                finally:
                    reader.execute("ROLLBACK")
                    reader.close()
                report = db.checkpoint_maintenance()
                self.assertFalse(report["busy"])
                self.assertGreater(report["reclaimed_bytes"], 0)
                self.assertEqual(db.wal_size_bytes(), 0)
            finally:
                db.close()

    def test_maintenance_reclaims_when_unpinned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._file_db(tmp)
            try:
                db.migrate()
                db.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
                _bulk_insert(
                    db, "INSERT INTO t (a) VALUES (?)",
                    [(i,) for i in range(2000)])
                self.assertGreater(db.wal_size_bytes(), 0)
                report = db.checkpoint_maintenance()
                self.assertFalse(report["busy"])
                self.assertGreater(report["reclaimed_bytes"], 0)
                self.assertEqual(db.wal_size_bytes(), 0)
            finally:
                db.close()


class TestWalCheckpointJob(unittest.TestCase):
    def _fake_scheduler(self) -> SimpleNamespace:
        jobs: list[dict] = []

        def list_jobs() -> list[dict]:
            return list(jobs)

        def add(name: str, spec: str, kind: str, payload: dict,
                **kwargs: object) -> dict:
            job = {"id": f"job-{len(jobs)}", "name": name, "spec": spec,
                   "kind": kind, "payload": payload, "kwargs": kwargs}
            jobs.append(job)
            return job

        def remove(job_id: str) -> None:
            jobs[:] = [j for j in jobs if j["id"] != job_id]

        return SimpleNamespace(list_jobs=list_jobs, add=add, remove=remove)

    def test_ensure_job_idempotent_and_replaces_stale(self) -> None:
        sched = self._fake_scheduler()
        job = ensure_wal_checkpoint_job(sched)
        self.assertEqual(job["name"], WAL_CHECKPOINT_JOB)
        self.assertEqual(job["spec"], WAL_CHECKPOINT_SPEC)
        self.assertEqual(job["payload"]["tool"], WAL_CHECKPOINT_TOOL)
        self.assertTrue(job["kwargs"].get("routine"))
        again = ensure_wal_checkpoint_job(sched)
        self.assertEqual(again["id"], job["id"])
        self.assertEqual(len(sched.list_jobs()), 1)
        stale = self._fake_scheduler()
        stale.add(WAL_CHECKPOINT_JOB, "every 5m", "tool",
                  {"tool": WAL_CHECKPOINT_TOOL, "args": {}})
        replaced = ensure_wal_checkpoint_job(stale)
        self.assertEqual(replaced["spec"], WAL_CHECKPOINT_SPEC)
        self.assertEqual(len(stale.list_jobs()), 1)
        self.assertIsNone(ensure_wal_checkpoint_job(None))

    def test_wal_checkpoint_tool_registered_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(os.path.join(tmp, "tool.db"))
            try:
                db.migrate()
                db.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
                db.execute("INSERT INTO t (a) VALUES (1)")
                ctx = SimpleNamespace(db=db, settings=None)
                reg = ToolRegistry(ctx)
                register_autonomy(reg)
                self.assertIn(WAL_CHECKPOINT_TOOL, reg._tools)
                result = reg._tools[WAL_CHECKPOINT_TOOL].fn()
                self.assertTrue(result["ok"])
                self.assertIn("reclaimed_bytes", result)
                self.assertEqual(db.wal_size_bytes(), 0)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
