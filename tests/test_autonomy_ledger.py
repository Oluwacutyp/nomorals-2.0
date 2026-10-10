"""Autonomy ledger: the unified fail-open journal of autonomous work."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from nomorals.agents.autonomy_ledger import (
    AutonomyLedger,
    record_ledger,
)
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.ledger = AutonomyLedger(self.db)

    def test_record_and_recent(self):
        self.ledger.record("scheduler", "run", "job-1", "ran fine",
                           cost_seconds=1.5, cost_tokens=100, ok=True,
                           learned="cached the weather")
        rows = self.ledger.recent(limit=10)
        self.assertEqual(len(rows), 1)
        e = rows[0]
        self.assertEqual(e["system"], "scheduler")
        self.assertEqual(e["kind"], "run")
        self.assertEqual(e["ref_id"], "job-1")
        self.assertEqual(e["summary"], "ran fine")
        self.assertEqual(e["cost_seconds"], 1.5)
        self.assertEqual(e["cost_tokens"], 100)
        self.assertTrue(e["ok"])
        self.assertEqual(e["learned"], "cached the weather")
        self.assertGreater(e["ts"], time.time() - 60)

    def test_record_never_raises(self):
        class Boom:
            def execute(self, *a, **k):
                raise RuntimeError("db down")

            def transaction(self):
                raise RuntimeError("db down")

        ledger = AutonomyLedger(Boom())
        ledger.record("x", "y", "z", "w")  # must not raise

    def test_recent_filters(self):
        self.ledger.record("scheduler", "run", "a", "ok")
        self.ledger.record("mission", "terminal", "b", "failed", ok=False)
        self.ledger.record("scheduler", "skip", "c", "overlap")
        only_sched = self.ledger.recent(system="scheduler")
        self.assertEqual(len(only_sched), 2)
        only_bad = self.ledger.recent(ok=False)
        self.assertEqual(len(only_bad), 1)
        self.assertEqual(only_bad[0]["system"], "mission")
        only_kind = self.ledger.recent(kind="skip")
        self.assertEqual(len(only_kind), 1)
        self.assertEqual(only_kind[0]["ref_id"], "c")

    def test_summary(self):
        self.ledger.record("scheduler", "run", "a", "ok", cost_seconds=2.0,
                           cost_tokens=10, ok=True)
        self.ledger.record("scheduler", "run", "b", "fail", ok=False)
        self.ledger.record("pulse", "run", "p", "ok", cost_seconds=5.0,
                           cost_tokens=500, ok=True)
        s = self.ledger.summary(window_hours=1)
        self.assertEqual(s["totals"]["runs"], 3)
        self.assertEqual(s["totals"]["failures"], 1)
        self.assertEqual(s["totals"]["cost_seconds"], 7.0)
        self.assertEqual(s["totals"]["cost_tokens"], 510)
        self.assertEqual(s["systems"]["scheduler"]["runs"], 2)
        self.assertEqual(s["systems"]["pulse"]["failures"], 0)

    def test_summary_window(self):
        self.ledger.record("scheduler", "run", "a", "old")
        self.db.execute(
            "UPDATE autonomy_ledger SET ts = ?", (time.time() - 7200,))
        s = self.ledger.summary(window_hours=1)
        self.assertEqual(s["totals"]["runs"], 0)

    def test_purge(self):
        self.ledger.record("x", "y", "a", "old")
        self.db.execute(
            "UPDATE autonomy_ledger SET ts = ?", (time.time() - 40 * 86400,))
        self.ledger.record("x", "y", "b", "fresh")
        n = self.ledger.purge(older_than_days=30)
        self.assertEqual(n, 1)
        rows = self.ledger.recent(limit=10)
        self.assertEqual([r["ref_id"] for r in rows], ["b"])

    def test_convenience(self):
        ctx = SimpleNamespace(db=self.db)
        record_ledger(ctx, "trigger", "fired", "t1", "matched bus event")
        rows = self.ledger.recent(system="trigger")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "fired")


if __name__ == "__main__":
    unittest.main()
