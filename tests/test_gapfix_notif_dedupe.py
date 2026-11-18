"""Notifier dedupe stall (2026-11-18): the 40s publish-time scan.

The owner's EC2 screenshot: ``WARN storage.db slow query (39.98s >
5.00s): SELECT id FROM notifications WHERE kind = ? AND title = ?
AND created_at > ? AND COALESCE(delivery_state, '') NOT IN (?, ?, ?)``.
That is ``Notifier._is_duplicate`` — the dedupe check that runs on
EVERY publish.  The notifications table had only two indexes
(``idx_notifications_read`` on (read), ``idx_notifications_retry`` on
(delivered, next_retry_at)) — nothing on (kind, title, created_at) —
so every publish paid a full table scan, and the table is NEVER pruned
(no DELETE FROM notifications anywhere; the ``pruned_at`` column added
by v37 is dead — nothing writes it).  Same disease as the news_items /
failure_clusters stalls fixed in 796aa152, third table, not covered by
that ship.

What the probes proved (~/workspace/notif-verify/probe.py, 3000 rows):
- dedupe query WITHOUT the index: SCAN, 0.000146s best-of-5.
- dedupe query WITH idx_notifications_dedupe (kind, title, created_at):
  SEARCH USING INDEX (kind=? AND title=? AND created_at>?), 0.000003s.
  The delivery_state NOT IN exclusion is a residual filter on the
  handful of rows the prefix narrows to — exactly as designed.
"""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from nomorals.agents import notifier as N
from nomorals.agents.notifier import (
    STATE_DEAD,
    STATE_DEDUPED,
    STATE_DIGEST,
    STATE_DISABLED,
    STATE_FAILED,
    STATE_HELD_BRAIN,
    STATE_HELD_FOCUS,
    STATE_HELD_QUIET,
    STATE_LOGGED,
    STATE_MUTED,
    STATE_PENDING,
    STATE_SENT,
    STATE_SUPERSEDED,
    Notifier,
)
from nomorals.storage.db import Database
from nomorals.storage.migrations import latest_version

DAY = 86400.0

DEDUPE_SQL = (
    "SELECT id FROM notifications WHERE kind = ? AND title = ? "
    "AND created_at > ? AND COALESCE(delivery_state, '') "
    "NOT IN (?, ?, ?)")
DEDUPE_PARAMS = ("alert", "Disk full on EC2", time.time() - 600.0,
                 STATE_DIGEST, STATE_LOGGED, STATE_SUPERSEDED)


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


def _seed_notif(db: Database, n: int = 3000) -> None:
    now = time.time()
    rows = []
    for i in range(n):
        age = (i % 400) + (i // 400) * 0.001
        state = (STATE_PENDING if i % 50 == 0
                 else STATE_SENT if i % 2 else STATE_LOGGED)
        rows.append((f"n{i}", f"kind{i % 12}", f"Title {i}",
                     f"body {i}", 0, state, now - age * DAY))
    _bulk_insert(
        db,
        "INSERT INTO notifications (id, kind, title, body, delivered,"
        " delivery_state, created_at) VALUES (?,?,?,?,?,?,?)",
        rows)


def _ctx(db):
    return SimpleNamespace(db=db, settings=SimpleNamespace(partner=None))


class TestMigrationV90(unittest.TestCase):
    def test_v90_applies_and_indexes_dedupe(self) -> None:
        db = _mem_db()
        try:
            self.assertEqual(latest_version(), 90)
            rows = db.query(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND name = 'idx_notifications_dedupe'")
            self.assertEqual(len(rows), 1)
            cols = {r["name"] for r in db.query(
                "PRAGMA table_info(notifications)")}
            for col in ("kind", "title", "created_at"):
                self.assertIn(col, cols)
        finally:
            db.close()


class TestDedupePlan(unittest.TestCase):
    def test_dedupe_query_searches_the_new_index(self) -> None:
        db = _mem_db()
        try:
            _seed_notif(db)
            plan = _plan_details(db, DEDUPE_SQL, DEDUPE_PARAMS)
            self.assertTrue(
                any("SEARCH" in d and "idx_notifications_dedupe" in d
                    for d in plan),
                f"expected SEARCH on idx_notifications_dedupe, got {plan}")
            self.assertFalse(any("SCAN" in d for d in plan), plan)
        finally:
            db.close()

    def test_dedupe_query_scans_without_the_index(self) -> None:
        """Baseline: the same query on the same rows, index dropped."""
        db = _mem_db()
        try:
            _seed_notif(db)
            db.execute("DROP INDEX idx_notifications_dedupe")
            db.execute("ANALYZE")
            plan = _plan_details(db, DEDUPE_SQL, DEDUPE_PARAMS)
            self.assertTrue(any("SCAN" in d for d in plan), plan)
        finally:
            db.close()


class TestPrune(unittest.TestCase):
    def setUp(self) -> None:
        N._LAST_PRUNE_NOTIFICATIONS_MONO = 0.0  # noqa: SLF001 - test gate reset

    def tearDown(self) -> None:
        N._LAST_PRUNE_NOTIFICATIONS_MONO = 0.0  # noqa: SLF001 - test gate reset

    def _seed_states(self, db: Database, *, with_fold: bool = True) -> None:
        now = time.time()
        old = now - 100 * DAY
        recent = now - 1 * DAY
        rows = [
            # (id, state, created_at, digest_folded_at)
            ("old-sent", STATE_SENT, old, None),
            ("old-dead", STATE_DEAD, old, None),
            ("old-superseded", STATE_SUPERSEDED, old, None),
            ("old-disabled", STATE_DISABLED, old, None),
            ("old-muted", STATE_MUTED, old, None),
            ("old-deduped", STATE_DEDUPED, old, None),
            ("old-logged", STATE_LOGGED, old, None),
            ("old-held-brain", STATE_HELD_BRAIN, old, None),
            ("old-null-state", "", old, None),
            ("old-digest-unfolded", STATE_DIGEST, old, None),
            ("old-digest-folded", STATE_DIGEST, old, now - 50 * DAY),
            ("old-pending", STATE_PENDING, old, None),
            ("old-failed", STATE_FAILED, old, None),
            ("old-held-quiet", STATE_HELD_QUIET, old, None),
            ("old-held-focus", STATE_HELD_FOCUS, old, None),
            ("recent-sent", STATE_SENT, recent, None),
            ("recent-pending", STATE_PENDING, recent, None),
        ]
        if with_fold:
            _bulk_insert(
                db,
                "INSERT INTO notifications (id, kind, title, body, delivered,"
                " delivery_state, created_at, digest_folded_at)"
                " VALUES (?, 'alert', 't', 'b', 0, ?, ?, ?)",
                [(rid, state, ts, folded)
                 for rid, state, ts, folded in rows])
        else:
            _bulk_insert(
                db,
                "INSERT INTO notifications (id, kind, title, body, delivered,"
                " delivery_state, created_at)"
                " VALUES (?, 'alert', 't', 'b', 0, ?, ?)",
                [(rid, state, ts) for rid, state, ts, _ in rows])

    def _ids(self, db: Database) -> set[str]:
        return {str(r["id"]) for r in db.query("SELECT id FROM notifications")}

    def test_prune_drops_settled_keeps_live_work(self) -> None:
        db = _mem_db()
        try:
            self._seed_states(db)
            dropped = Notifier(_ctx(db)).prune_notifications()
            self.assertEqual(dropped, 10)
            self.assertEqual(
                self._ids(db),
                {"old-digest-unfolded", "old-pending", "old-failed",
                 "old-held-quiet", "old-held-focus", "recent-sent",
                 "recent-pending"})
        finally:
            db.close()

    def test_prune_keeps_every_digest_row_without_fold_column(self) -> None:
        """Pre-F7 databases have no digest_folded_at: the fallback
        treats every digest row as live work instead of raising."""
        db = _mem_db()
        try:
            db.execute(
                "ALTER TABLE notifications DROP COLUMN digest_folded_at")
            self._seed_states(db, with_fold=False)
            # The seeded folded digest row is now just another digest
            # row — the fallback keeps it, so one fewer row drops.
            dropped = Notifier(_ctx(db)).prune_notifications()
            self.assertEqual(dropped, 9)
            self.assertIn("old-digest-unfolded", self._ids(db))
            self.assertIn("old-digest-folded", self._ids(db))
        finally:
            db.close()

    def test_prune_no_db_is_zero(self) -> None:
        self.assertEqual(Notifier(SimpleNamespace(db=None)).prune_notifications(), 0)

    def test_maybe_prune_runs_at_most_hourly(self) -> None:
        db = _mem_db()
        try:
            self._seed_states(db)
            notifier = Notifier(_ctx(db))
            notifier._maybe_prune_notifications()
            self.assertEqual(len(self._ids(db)), 7)
            # A second old settled row added after the first prune
            # must survive the second call: the hourly gate holds.
            _bulk_insert(
                db,
                "INSERT INTO notifications (id, kind, title, body,"
                " delivered, delivery_state, created_at)"
                " VALUES (?, 'alert', 't', 'b', 0, ?, ?)",
                [("old-sent-2", STATE_SENT, time.time() - 100 * DAY)])
            notifier._maybe_prune_notifications()
            self.assertIn("old-sent-2", self._ids(db))
        finally:
            db.close()

    def test_publish_prunes_before_storing(self) -> None:
        db = _mem_db()
        try:
            self._seed_states(db)
            notifier = Notifier(_ctx(db))
            notifier.observe_candidates = False
            notifier.publish("alert", "fresh title", "fresh body")
            self.assertNotIn("old-sent", self._ids(db))
            self.assertIn("old-pending", self._ids(db))
            self.assertIn("old-digest-unfolded", self._ids(db))
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
