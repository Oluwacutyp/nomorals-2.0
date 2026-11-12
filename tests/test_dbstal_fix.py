"""DB stall fix (2026-11-12): abandoned workers must not poison the process.

Proven by the reproduction rig (~/workspace/dbstal-fix-verify/rig/rig.py):
- E6ii: a thread abandoned inside transaction() held the process-wide
  _write_lock forever; a second transaction() blocked past a 60s wall
  (timeout rc=124). Reads were unaffected (E6i 0.02s).
- E8: lock contention on current code fails at ~32s (6 x 5s busy_timeout
  + backoff), never 300s — so the news poll's 300s dwell in fetchall
  was NOT ordinary contention, and the retry bound never covered it.
- E9b: conn.interrupt() aborts a long pure-stepping fetchall in 0.19s
  with OperationalError("interrupted"), which _is_retryable_sqlite
  does NOT retry ("locked"/"busy" only).
- E9 (honest limit): interrupt() does NOT abort a step blocked inside
  a user-defined function — SQLite only checks the flag between VDBE
  steps. No test pretends otherwise.

The fix: transaction() acquires _write_lock bounded (30s) with the
holder named; the scheduler interrupts an abandoned worker's DB
connection on abandonment so its transaction unwinds and the lock is
released; the watchdog timeout report carries a sample histogram and
the DB lock state so the next alert diagnoses instead of merely
locating.
"""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.scheduler import Scheduler
from nomorals.core.errors import StorageError
from nomorals.storage.db import Database


def _mem_db() -> Database:
    db = Database(":memory:")
    return db


class WriteLockBoundedTests(unittest.TestCase):
    def test_second_transaction_fails_fast_naming_holder(self):
        db = _mem_db()
        entered = threading.Event()
        release = threading.Event()

        def holder():
            with db.transaction():
                entered.set()
                release.wait(30)

        t = threading.Thread(target=holder, name="dbstal-holder",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        try:
            with mock.patch(
                    "nomorals.storage.db._WRITE_LOCK_TIMEOUT_S", 0.5):
                t0 = time.perf_counter()
                with self.assertRaises(StorageError) as cm, db.transaction():
                    pass
                waited = time.perf_counter() - t0
            self.assertLess(waited, 10)
            self.assertIn("dbstal-holder", str(cm.exception))
            self.assertIn("held by thread", str(cm.exception))
        finally:
            release.set()
            t.join(timeout=10)
        # The lock is released when the holder unwinds: a fresh
        # transaction succeeds promptly.
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
            db.execute("INSERT INTO t VALUES (1)")
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 1}])
        db.close()

    def test_normal_and_nested_transactions_still_work(self):
        db = _mem_db()
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
            with db.transaction():
                db.execute("INSERT INTO t VALUES (7)")
        self.assertEqual(db.query("SELECT a FROM t"), [{"a": 7}])
        state = db.lock_state()
        self.assertIsNone(state["write_lock_holder"])
        db.close()

    def test_lock_state_reports_holder_while_held(self):
        db = _mem_db()
        entered = threading.Event()
        release = threading.Event()

        def holder():
            with db.transaction():
                entered.set()
                release.wait(30)

        t = threading.Thread(target=holder, name="dbstal-state",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        try:
            state = db.lock_state()
            self.assertIsNotNone(state["write_lock_holder"])
            self.assertEqual(state["write_lock_holder"]["thread"],
                             "dbstal-state")
            self.assertGreaterEqual(
                state["write_lock_holder"]["held_s"], 0)
            self.assertEqual(state["open_transactions"], 1)
            self.assertIn("journal_mode", state)
            self.assertIn("wal_bytes", state)
        finally:
            release.set()
            t.join(timeout=10)
        db.close()


class InterruptThreadTests(unittest.TestCase):
    def test_interrupt_aborts_stuck_fetchall_and_releases_lock(self):
        db = _mem_db()
        entered = threading.Event()
        done = threading.Event()
        outcome: dict[str, str] = {}

        def worker():
            try:
                with db.transaction():
                    entered.set()
                    cur = db.execute(
                        "WITH RECURSIVE c(n) AS "
                        "(SELECT 1 UNION ALL SELECT n+1 FROM c "
                        "LIMIT 30000000) SELECT n FROM c")
                    rows = cur.fetchall()
                    outcome["rows"] = str(len(rows))
            except Exception as exc:  # noqa: BLE001 - the point is the abort
                outcome["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                done.set()

        t = threading.Thread(target=worker, name="dbstal-stuck",
                             daemon=True)
        t.start()
        self.assertTrue(entered.wait(10))
        time.sleep(1.0)  # worker is deep in stepping
        self.assertTrue(db.interrupt_thread(t.ident or 0))
        self.assertTrue(done.wait(10), "worker did not unwind after interrupt")
        self.assertIn("interrupted", outcome.get("error", ""))
        # The transaction unwound (rollback) and the lock is released:
        # a fresh transaction succeeds promptly.
        with db.transaction():
            db.execute("CREATE TABLE t (a INTEGER)")
        self.assertIsNone(db.lock_state()["write_lock_holder"])
        db.close()

    def test_interrupt_unknown_thread_is_false(self):
        db = _mem_db()
        self.assertFalse(db.interrupt_thread(999999999))
        db.close()


def _sched_ctx(**kw):
    db = Database(":memory:")
    db.migrate()
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw2: SimpleNamespace(
            ok=True, value="ok", error=None)
    )
    ctx = SimpleNamespace(db=db, tools=tools, settings=SimpleNamespace(),
                          extras={})
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


class WatchdogHistogramTests(unittest.TestCase):
    def test_timeout_report_carries_histogram_and_db_state(self):
        ctx = _sched_ctx()
        block = threading.Event()

        def slow_call(tool, capabilities=None, **kw):
            block.wait(30)
            return SimpleNamespace(ok=True, value="slow", error=None)

        ctx.tools = SimpleNamespace(call=slow_call)
        sched = Scheduler(ctx, wall_seconds=300.0)
        job = sched.add("slow-hist", "every 1h", "tool",
                        {"tool": "slowtool", "args": {}},
                        run_timeout_s=6)
        try:
            out = sched.run_now(job["id"])
        finally:
            block.set()
        self.assertFalse(out["ok"])
        self.assertTrue(out["timed_out"])
        self.assertIn("timed out", out["result"])
        self.assertIn("worker abandoned", out["result"])
        # The histogram names where the wall went (the slow frame).
        self.assertIn("where the time went:", out["result"])
        self.assertIn("slow_call", out["result"])
        # The DB lock state rides along.
        self.assertIn("db:", out["result"])
        self.assertIn("db interrupt", out["result"])
        samples = sched._last_guard_timeout.get("samples") or {}
        self.assertTrue(samples)
        self.assertIn("db_lock", sched._last_guard_timeout)


if __name__ == "__main__":
    unittest.main()
