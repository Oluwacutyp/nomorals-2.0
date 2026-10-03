"""Scheduler resource-aware execution: heavy tasks defer under pressure.

The Scheduler (L4) never imports nomorals.os (L6); the resource advisor is
injected via the ``resources`` constructor kwarg and only needs to satisfy
the ``consult()`` protocol.  These tests use a deterministic stub advisor
so pressure is scripted, not sampled.
"""

import asyncio
import json
import time
import unittest

from nomorals.scheduler.scheduler import (
    HEAVY_WEIGHT_THRESHOLD,
    Scheduler,
)
from nomorals.storage.db import Database


class StubAdvisor:
    """Duck-typed ResourceManager stand-in with a scripted consult()."""

    def __init__(self, *, ok=True, throttled=False, reasons=None):
        self.ok = ok
        self.throttled = throttled
        self.reasons = list(reasons or [])
        self.calls = 0

    def consult(self, mission=None):
        self.calls += 1
        return {
            "ok": self.ok,
            "throttled": self.throttled,
            "reasons": list(self.reasons),
            "pressure": {"overall": 0.9 if self.throttled else 0.1},
            "sample": {},
        }


class ExplodingAdvisor:
    """An advisor that raises: the tick must survive it."""

    def consult(self, mission=None):  # noqa: D102 - test stub
        raise RuntimeError("boom")


def _make_scheduler(advisor=None):
    db = Database(":memory:")
    sched = Scheduler(db, resources=advisor)
    calls = []

    async def handler(**kwargs):
        calls.append(kwargs)

    sched.register_action("test_action", handler)
    return sched, db, calls


class TestResourceGating(unittest.TestCase):
    def test_light_one_time_runs_under_pressure(self):
        sched, _db, calls = _make_scheduler(
            StubAdvisor(throttled=True, reasons=["thermal state warm"])
        )
        asyncio.run(sched.schedule_once(None, time.time() - 1, "test_action"))
        sched._tick()
        self.assertEqual(len(calls), 1)

    def test_heavy_one_time_deferred_under_throttle(self):
        advisor = StubAdvisor(throttled=True, reasons=["thermal state warm"])
        sched, db, calls = _make_scheduler(advisor)
        task = asyncio.run(
            sched.schedule_once(None, time.time() - 1, "test_action", heavy=True)
        )
        sched._tick()
        self.assertEqual(calls, [])
        # Not dropped: still pending in the DB, deferral counted.
        row = db.query(
            "SELECT status FROM scheduled_tasks WHERE task_id = ?", (task.task_id,)
        )[0]
        self.assertEqual(row["status"], "pending")
        self.assertEqual(sched.deferral_counts.get(task.task_id), 1)
        # One consult per tick even with several due tasks.
        self.assertEqual(advisor.calls, 1)

    def test_heavy_task_picked_up_when_pressure_eases(self):
        advisor = StubAdvisor(throttled=True, reasons=["cpu pressure 0.80 >= throttle"])
        sched, db, calls = _make_scheduler(advisor)
        task = asyncio.run(
            sched.schedule_once(None, time.time() - 1, "test_action", heavy=True)
        )
        sched._tick()
        self.assertEqual(calls, [])
        # Pressure eases -> next tick runs it.
        advisor.throttled = False
        advisor.reasons = []
        sched._tick()
        self.assertEqual(len(calls), 1)
        row = db.query(
            "SELECT status FROM scheduled_tasks WHERE task_id = ?", (task.task_id,)
        )[0]
        self.assertEqual(row["status"], "completed")

    def test_critical_ok_false_defers_heavy_light_still_runs(self):
        advisor = StubAdvisor(ok=False, reasons=["battery 10% <= minimum 15%"])
        sched, _db, calls = _make_scheduler(advisor)
        light = asyncio.run(sched.schedule_once(None, time.time() - 1, "test_action"))
        heavy = asyncio.run(
            sched.schedule_once(None, time.time() - 1, "test_action", heavy=True)
        )
        sched._tick()
        self.assertEqual(len(calls), 1)
        self.assertEqual(sched.deferral_counts.get(heavy.task_id), 1)
        self.assertIsNone(sched.deferral_counts.get(light.task_id))

    def test_weight_threshold_marks_heavy(self):
        advisor = StubAdvisor(throttled=True, reasons=["cpu pressure high"])
        sched, _db, calls = _make_scheduler(advisor)
        heavy = asyncio.run(
            sched.schedule_once(
                None, time.time() - 1, "test_action",
                weight=HEAVY_WEIGHT_THRESHOLD + 1.0,
            )
        )
        feather = asyncio.run(
            sched.schedule_once(
                None, time.time() - 1, "test_action", weight=0.5
            )
        )
        sched._tick()
        self.assertEqual(len(calls), 1)  # only the light one ran
        self.assertEqual(sched.deferral_counts.get(heavy.task_id), 1)
        self.assertIsNone(sched.deferral_counts.get(feather.task_id))

    def test_reminder_default_light_runs_under_pressure(self):
        sched, _db, calls = _make_scheduler(
            StubAdvisor(throttled=True, reasons=["memory pressure 0.85 >= throttle"])
        )
        reminder = asyncio.run(
            sched.create_reminder(
                "drink water", time.time() - 1, "u1", action="test_action"
            )
        )
        self.assertFalse(reminder.metadata["heavy"])
        sched._tick()
        self.assertEqual(len(calls), 1)

    def test_heavy_reminder_marked_heavy_defers(self):
        sched, _db, calls = _make_scheduler(
            StubAdvisor(throttled=True, reasons=["thermal state hot"])
        )
        reminder = asyncio.run(
            sched.create_reminder(
                "big report", time.time() - 1, "u1", heavy=True
            )
        )
        sched._tick()
        self.assertEqual(calls, [])
        self.assertEqual(sched.deferral_counts.get(reminder.task_id), 1)

    def test_heavy_cron_deferred_without_advancing(self):
        advisor = StubAdvisor(throttled=True, reasons=["metered network"])
        sched, db, calls = _make_scheduler(advisor)
        job = asyncio.run(
            sched.schedule_cron(None, "0 9 * * *", "test_action", heavy=True)
        )
        db.execute(
            "UPDATE cron_jobs SET next_run = ? WHERE task_id = ?",
            (time.time() - 1, job.task_id),
        )
        next_run_before = db.query(
            "SELECT next_run, run_count FROM cron_jobs WHERE task_id = ?",
            (job.task_id,),
        )[0]
        sched._tick()
        self.assertEqual(calls, [])
        after = db.query(
            "SELECT next_run, run_count FROM cron_jobs WHERE task_id = ?",
            (job.task_id,),
        )[0]
        # Deferred cron keeps its schedule: next_run untouched, no run counted.
        self.assertEqual(after["next_run"], next_run_before["next_run"])
        self.assertEqual(after["run_count"], 0)
        # Pressure eases -> cron fires and advances normally.
        advisor.throttled = False
        advisor.reasons = []
        sched._tick()
        self.assertEqual(len(calls), 1)
        fired = db.query(
            "SELECT run_count FROM cron_jobs WHERE task_id = ?", (job.task_id,)
        )[0]
        self.assertEqual(fired["run_count"], 1)

    def test_no_advisor_runs_everything_ungated(self):
        sched, _db, calls = _make_scheduler(None)  # backwards compatible
        asyncio.run(sched.schedule_once(None, time.time() - 1, "test_action", heavy=True))
        sched._tick()
        self.assertEqual(len(calls), 1)
        self.assertEqual(sched.deferral_counts, {})

    def test_consult_failure_defers_heavy_but_tick_survives(self):
        sched, _db, calls = _make_scheduler(ExplodingAdvisor())
        asyncio.run(sched.schedule_once(None, time.time() - 1, "test_action", heavy=True))
        asyncio.run(sched.schedule_once(None, time.time() - 1, "test_action"))
        sched._tick()  # must not raise
        self.assertEqual(len(calls), 1)  # light ran, heavy deferred

    def test_deferral_is_logged(self):
        advisor = StubAdvisor(throttled=True, reasons=["thermal state warm"])
        sched, _db, _calls = _make_scheduler(advisor)
        task = asyncio.run(
            sched.schedule_once(None, time.time() - 1, "test_action", heavy=True)
        )
        with self.assertLogs("nomorals.scheduler.scheduler", level="WARNING") as logs:
            sched._tick()
        output = "\n".join(logs.output)
        self.assertIn(task.task_id, output)
        self.assertIn("Deferring heavy", output)
        self.assertIn("thermal state warm", output)

    def test_heavy_metadata_persists(self):
        sched, db, _calls = _make_scheduler()
        task = asyncio.run(
            sched.schedule_once(None, time.time() + 60, "test_action",
                                heavy=True, weight=2.5)
        )
        row = db.query(
            "SELECT metadata FROM scheduled_tasks WHERE task_id = ?", (task.task_id,)
        )[0]
        meta = json.loads(row["metadata"])
        self.assertTrue(meta["heavy"])
        self.assertEqual(meta["weight"], 2.5)
        # And on the returned dataclass.
        self.assertTrue(task.metadata["heavy"])

    def test_is_heavy_classification(self):
        self.assertTrue(Scheduler._is_heavy({"heavy": True}))
        self.assertTrue(Scheduler._is_heavy({"weight": HEAVY_WEIGHT_THRESHOLD}))
        self.assertFalse(Scheduler._is_heavy({"weight": 0.5}))
        self.assertFalse(Scheduler._is_heavy({}))
        self.assertFalse(Scheduler._is_heavy({"heavy": False, "weight": 0.0}))
        self.assertFalse(Scheduler._is_heavy({"weight": "not-a-number"}))

    def test_row_metadata_never_raises(self):
        self.assertEqual(Scheduler._row_metadata({"metadata": None}), {})
        self.assertEqual(Scheduler._row_metadata({"metadata": "garbage{"}), {})
        self.assertEqual(Scheduler._row_metadata({}), {})
        self.assertEqual(
            Scheduler._row_metadata({"metadata": {"heavy": True}}), {"heavy": True}
        )


if __name__ == "__main__":
    unittest.main()
