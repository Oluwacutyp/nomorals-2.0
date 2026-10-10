"""Scheduler deep upgrades: backoff strategies, missed-fire policies,
overlap policies, run timeouts, multi-dependency gates, resource
deferral, run history, bus events, ledger."""

from __future__ import annotations

import random
import threading
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.scheduler import (
    Scheduler,
    _backoff_delay,
    _parse_depends,
)
from nomorals.core.events import EventBus, global_bus
from nomorals.storage.db import Database


def _ctx(**kw):
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


class BackoffTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(1234)

    def test_constant(self):
        ds = [_backoff_delay("constant", 60, a, jitter=0, rng=self.rng)
              for a in (1, 2, 3, 4)]
        self.assertEqual(ds, [60.0, 60.0, 60.0, 60.0])

    def test_linear(self):
        ds = [_backoff_delay("linear", 60, a, jitter=0, rng=self.rng)
              for a in (1, 2, 3)]
        self.assertEqual(ds, [60.0, 120.0, 180.0])

    def test_exponential(self):
        ds = [_backoff_delay("exponential", 60, a, jitter=0, rng=self.rng)
              for a in (1, 2, 3, 4)]
        self.assertEqual(ds, [60.0, 120.0, 240.0, 480.0])

    def test_cap(self):
        d = _backoff_delay("exponential", 60, 10, max_s=300, jitter=0,
                           rng=self.rng)
        self.assertEqual(d, 300.0)

    def test_jitter_bounded(self):
        for _ in range(50):
            d = _backoff_delay("exponential", 100, 2, jitter=0.25,
                               rng=self.rng)
            self.assertGreaterEqual(d, 100 * 2 * 0.75)
            self.assertLessEqual(d, 100 * 2 * 1.25)

    def test_unknown_strategy_degrades_to_exponential(self):
        d = _backoff_delay("bogus", 60, 3, jitter=0, rng=self.rng)
        self.assertEqual(d, 240.0)

    def test_never_below_one_second(self):
        self.assertGreaterEqual(
            _backoff_delay("constant", 0, 1, jitter=0, rng=self.rng), 1.0)


class ParseDependsTests(unittest.TestCase):
    def test_single(self):
        self.assertEqual(_parse_depends("abc"), ["abc"])

    def test_list(self):
        self.assertEqual(_parse_depends(["a", "b"]), ["a", "b"])

    def test_json(self):
        self.assertEqual(_parse_depends('["a", "b"]'), ["a", "b"])

    def test_csv(self):
        self.assertEqual(_parse_depends("a, b"), ["a", "b"])

    def test_empty(self):
        self.assertEqual(_parse_depends(""), [])
        self.assertEqual(_parse_depends(None), [])


class PolicyValidationTests(unittest.TestCase):
    def setUp(self):
        self.sched = Scheduler(_ctx())

    def test_bad_policies_rejected(self):
        for kw in ({"missed_fire_policy": "nope"},
                   {"overlap_policy": "nope"},
                   {"backoff": "nope"},
                   {"depends_policy": "nope"}):
            with self.assertRaises(ValueError, msg=kw):
                self.sched.add("x", "every 1h", "message", {"text": "x"},
                               **kw)

    def test_policies_stored(self):
        job = self.sched.add("x", "every 1h", "message", {"text": "x"},
                             missed_fire_policy="skip",
                             overlap_policy="skip",
                             backoff="linear",
                             run_timeout_s=45,
                             heavy=True,
                             depends_policy="any_ok")
        f = self.sched._format_job(self.sched._find(job["id"]))
        self.assertEqual(f["missed_fire_policy"], "skip")
        self.assertEqual(f["overlap_policy"], "skip")
        self.assertEqual(f["backoff"], "linear")
        self.assertEqual(f["run_timeout_s"], 45.0)
        self.assertTrue(f["heavy"])
        self.assertEqual(f["depends_policy"], "any_ok")


class MissedFireTests(unittest.TestCase):
    def setUp(self):
        self.sched = Scheduler(_ctx(), missed_grace_s=60)

    def _stale_job(self, **kw):
        job = self.sched.add("stale", "every 1h", "message",
                             {"text": "stale"}, **kw)
        self.sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 3600, job["id"]))
        return job

    def test_fire_now_runs(self):
        self._stale_job(missed_fire_policy="fire_now")
        outcomes = self.sched.tick()
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["ok"])

    def test_skip_advances_without_running(self):
        job = self._stale_job(missed_fire_policy="skip")
        outcomes = self.sched.tick()
        self.assertEqual(len(outcomes), 0)
        row = self.sched._find(job["id"])
        # rescheduled from now (every 1h), not from the stale firing
        self.assertGreater(row["next_run"], time.time() + 3500)
        runs = self.sched.recent_runs(job["id"])
        self.assertEqual(len(runs), 1)
        self.assertIn("skipped", runs[0]["result"])

    def test_next_only_keeps_phase(self):
        job = self._stale_job(missed_fire_policy="next_only")
        outcomes = self.sched.tick()
        self.assertEqual(len(outcomes), 0)
        row = self.sched._find(job["id"])
        # stepped forward from the missed firing: within the next hour
        self.assertLess(row["next_run"], time.time() + 3600)

    def test_merely_late_runs_normally(self):
        job = self.sched.add("late", "every 1h", "message", {"text": "late"},
                             missed_fire_policy="skip")
        self.sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 30, job["id"]))  # inside the 60s grace
        outcomes = self.sched.tick()
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["ok"])


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.sched = Scheduler(_ctx())

    def test_overlap_skip(self):
        job = self.sched.add("ov", "every 1h", "message", {"text": "ov"},
                             overlap_policy="skip")
        # fake an in-flight run
        self.sched._active.add(job["id"])
        try:
            self.sched.db.execute(
                "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                (time.time() - 1, job["id"]))
            outcomes = self.sched.tick()
        finally:
            self.sched._active.discard(job["id"])
        self.assertEqual(len(outcomes), 0)
        runs = self.sched.recent_runs(job["id"])
        self.assertEqual(len(runs), 1)
        self.assertIn("overlap", runs[0]["result"])

    def test_overlap_queue_leaves_for_next_tick(self):
        job = self.sched.add("ov", "every 1h", "message", {"text": "ov"},
                             overlap_policy="queue")
        self.sched._active.add(job["id"])
        try:
            self.sched.db.execute(
                "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                (time.time() - 1, job["id"]))
            outcomes = self.sched.tick()
        finally:
            self.sched._active.discard(job["id"])
        self.assertEqual(len(outcomes), 0)
        # still due — nothing consumed it
        row = self.sched._find(job["id"])
        self.assertLessEqual(row["next_run"], time.time())


class TimeoutTests(unittest.TestCase):
    def test_timeout_marks_failed(self):
        ctx = _ctx()
        block = threading.Event()

        def slow_call(tool, capabilities=None, **kw):
            block.wait(30)
            return SimpleNamespace(ok=True, value="slow", error=None)

        ctx.tools = SimpleNamespace(call=slow_call)
        sched = Scheduler(ctx, wall_seconds=300.0)
        job = sched.add("slow", "every 1h", "tool",
                        {"tool": "slowtool", "args": {}},
                        run_timeout_s=0.3)
        out = sched.run_now(job["id"])
        block.set()
        self.assertFalse(out["ok"])
        self.assertTrue(out["timed_out"])
        self.assertIn("timed out", out["result"])

    def test_no_timeout_when_fast(self):
        sched = Scheduler(_ctx())
        job = sched.add("fast", "every 1h", "message", {"text": "fast"},
                        run_timeout_s=30)
        out = sched.run_now(job["id"])
        self.assertFalse(out["timed_out"])


class DependsPolicyTests(unittest.TestCase):
    def setUp(self):
        self.sched = Scheduler(_ctx())

    def _run_tool_job(self, name):
        job = self.sched.add(name, "every 1h", "tool",
                             {"tool": "t", "args": {}})
        return job

    def test_all_ok_needs_every_dependency(self):
        a = self._run_tool_job("a")
        b = self._run_tool_job("b")
        c = self.sched.add("c", "every 1h", "tool", {"tool": "t", "args": {}},
                           depends_on=[a["id"], b["id"]],
                           depends_policy="all_ok")
        self.sched.run_now(a["id"])  # only a succeeded
        row = self.sched._find(c["id"])
        self.assertFalse(self.sched._dependency_ok(dict(row)))

    def test_any_ok(self):
        a = self._run_tool_job("a")
        b = self._run_tool_job("b")
        c = self.sched.add("c", "every 1h", "tool", {"tool": "t", "args": {}},
                           depends_on=[a["id"], b["id"]],
                           depends_policy="any_ok")
        self.sched.run_now(a["id"])
        row = self.sched._find(c["id"])
        self.assertTrue(self.sched._dependency_ok(dict(row)))

    def test_latest_ok(self):
        a = self._run_tool_job("a")
        b = self._run_tool_job("b")
        c = self.sched.add("c", "every 1h", "tool", {"tool": "t", "args": {}},
                           depends_on=[a["id"], b["id"]],
                           depends_policy="latest_ok")
        self.sched.run_now(a["id"])
        time.sleep(0.02)
        # b fails (tool registry stub always ok — force a failure via bad payload)
        self.sched.db.execute(
            "UPDATE schedule_jobs SET last_result = 'job failed: boom', "
            "last_run = ? WHERE id = ?", (time.time(), b["id"]))
        row = self.sched._find(c["id"])
        # latest run (b) failed → gate shut
        self.assertFalse(self.sched._dependency_ok(dict(row)))

    def test_deleted_dependency_keeps_gate_shut(self):
        a = self._run_tool_job("a")
        c = self.sched.add("c", "every 1h", "tool", {"tool": "t", "args": {}},
                           depends_on=a["id"])
        self.sched.remove(a["id"])
        row = self.sched._find(c["id"])
        self.assertFalse(self.sched._dependency_ok(dict(row)))


class ResourceDeferralTests(unittest.TestCase):
    def test_heavy_defers_under_pressure(self):
        ctx = _ctx()
        advisor = SimpleNamespace(
            consult=lambda: {"ok": False, "throttled": True,
                             "reasons": ["cpu hot"]})
        sched = Scheduler(ctx, resources=advisor)
        job = sched.add("heavy", "every 1h", "message", {"text": "h"},
                        heavy=True)
        sched.db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                         (time.time() - 1, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 0)
        self.assertEqual(sched._deferrals.get(job["id"]), 1)
        # light jobs still run
        light = sched.add("light", "every 1h", "message", {"text": "l"})
        sched.db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                         (time.time() - 1, light["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 1)

    def test_no_advisor_no_gating(self):
        sched = Scheduler(_ctx())  # no resources
        job = sched.add("heavy", "every 1h", "message", {"text": "h"},
                        heavy=True)
        sched.db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                         (time.time() - 1, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 1)


class RunHistoryTests(unittest.TestCase):
    def test_every_execution_lands_a_row(self):
        sched = Scheduler(_ctx())
        job = sched.add("h", "every 1h", "message", {"text": "h"})
        sched.run_now(job["id"])
        sched.run_now(job["id"])
        runs = sched.recent_runs(job["id"])
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["trigger_source"], "manual")
        self.assertTrue(all(r["ok"] for r in runs))

    def test_health_includes_runs_and_overlaps(self):
        sched = Scheduler(_ctx())
        job = sched.add("h", "every 1h", "message", {"text": "h"})
        sched.run_now(job["id"])
        h = sched.health()
        self.assertIn("recent_runs", h)
        self.assertTrue(h["recent_runs"])
        self.assertIn("overlaps", h)
        self.assertEqual(h["overlaps"]["count"], 0)
        self.assertIn("deferred", h)


class BusEventTests(unittest.TestCase):
    def test_started_finished_events(self):
        bus = EventBus()
        seen = []
        bus.subscribe("scheduler.job.*", lambda e: seen.append(e.topic),
                      sync=True)
        import nomorals.agents.scheduler as sched_mod

        orig = sched_mod.global_bus
        sched_mod.global_bus = bus
        try:
            sched = Scheduler(_ctx())
            job = sched.add("b", "every 1h", "message", {"text": "b"})
            sched.run_now(job["id"])
        finally:
            sched_mod.global_bus = orig
        self.assertEqual(seen, ["scheduler.job.started",
                               "scheduler.job.finished"])


class RetryBackoffTests(unittest.TestCase):
    def test_exponential_retry_schedule(self):
        ctx = _ctx()
        calls = []

        def failing_call(tool, capabilities=None, **kw):
            calls.append(time.time())
            return SimpleNamespace(ok=False, value=None, error="boom")

        ctx.tools = SimpleNamespace(call=failing_call)
        sched = Scheduler(ctx)
        sched._rng = random.Random(0)
        job = sched.add("r", "every 1h", "tool", {"tool": "t", "args": {}},
                        max_retries=2, retry_delay=60, backoff="exponential")
        out = sched.run_now(job["id"])
        self.assertFalse(out["ok"])
        self.assertTrue(out["will_retry"])
        self.assertIn("exponential backoff", out["result"])
        row = sched._find(job["id"])
        # first retry: ~60s out (jitter=0.25 default may shrink it)
        self.assertGreater(row["next_run"], time.time() + 30)


class ProfileGatingTests(unittest.TestCase):
    def test_termux_defaults_to_single_concurrency(self):
        import os as _os

        ctx = _ctx()
        old = _os.environ.get("NM_PROFILE")
        _os.environ["NM_PROFILE"] = "termux"
        try:
            sched = Scheduler(ctx)
        finally:
            if old is None:
                del _os.environ["NM_PROFILE"]
            else:
                _os.environ["NM_PROFILE"] = old
        self.assertEqual(sched.max_concurrent, 1)
        # explicit wins
        _os.environ["NM_PROFILE"] = "termux"
        try:
            sched2 = Scheduler(ctx, max_concurrent=4)
        finally:
            if old is None:
                del _os.environ["NM_PROFILE"]
            else:
                _os.environ["NM_PROFILE"] = old
        self.assertEqual(sched2.max_concurrent, 4)


if __name__ == "__main__":
    unittest.main()
