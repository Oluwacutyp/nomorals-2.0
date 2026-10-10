"""Tests for power monitor and power-aware scheduler."""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from nomorals.power import PowerAwareScheduler, PowerMonitor, PowerStatus
from nomorals.storage.db import Database


def _db() -> Database:
    from nomorals.storage.migrations import MIGRATIONS
    from nomorals.storage.schema import MigrationRunner
    db = Database(":memory:")
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


@dataclass
class _FakeSample:
    battery_percent: float | None = None
    thermal_state: str | None = None


class _FakeManager:
    """Deterministic ResourceManager stand-in."""

    def __init__(self, battery=None, thermal=None, throttled=False, ok=True,
                 reasons=None, degradation_level=0, degradation_name="full",
                 allowed_tiers=None, budgets=None):
        self._sample = _FakeSample(battery, thermal)
        self._advisory = {
            "throttled": throttled, "ok": ok, "reasons": reasons or [],
            "degradation_level": degradation_level,
            "degradation": {
                "level": degradation_level, "name": degradation_name,
                "allowed_tiers": allowed_tiers or
                ["critical", "important", "background", "bulk"],
            },
            "battery_forecast": {},
        }
        self.budgets = budgets

    def sample(self):
        return self._sample

    def consult(self, mission=None):
        return dict(self._advisory)


class PowerMonitorTests(unittest.TestCase):
    def _monitor(self, **kw):
        return PowerMonitor(sampler=lambda: _FakeManager(**kw))

    def test_status_good_power(self):
        st = self._monitor(battery=80.0, thermal="nominal").status()
        self.assertIsInstance(st, PowerStatus)
        self.assertEqual(st.battery_pct, 80.0)
        self.assertFalse(st.should_defer_heavy)

    def test_low_battery_defers_heavy(self):
        st = self._monitor(battery=5.0, thermal="nominal",
                           throttled=True, reasons=["battery low"]).status()
        self.assertTrue(st.throttled)
        self.assertTrue(st.should_defer_heavy)

    def test_hot_thermal_defers_heavy(self):
        st = self._monitor(battery=80.0, thermal="hot", ok=False,
                           reasons=["thermal hot"]).status()
        self.assertFalse(st.ok)
        self.assertTrue(st.should_defer_heavy)

    def test_no_sampler_fails_open(self):
        st = PowerMonitor().status()
        self.assertTrue(st.ok)
        self.assertIsNone(st.battery_pct)

    def test_sampling_failure_never_raises(self):
        class Boom:
            def sample(self):
                raise RuntimeError("nope")
        st = PowerMonitor(sampler=Boom).status()
        self.assertTrue(st.ok)  # fail-open, logged


class PowerAwareSchedulerTests(unittest.TestCase):
    def _sched(self, **mgr_kw):
        monitor = PowerMonitor(sampler=lambda: _FakeManager(**mgr_kw))
        return PowerAwareScheduler(_db(), monitor=monitor)

    def test_light_always_flows(self):
        s = self._sched(battery=5.0, throttled=True)
        s.dispatch("ping", {}, power_class="light")
        tasks = s.poll()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].power_class, "light")

    def test_heavy_deferred_when_constrained(self):
        s = self._sched(battery=5.0, throttled=True)
        s.dispatch("train", {}, power_class="heavy")
        self.assertEqual(s.poll(), [])
        self.assertEqual(s.deferred_heavy_count(), 1)

    def test_heavy_flows_when_ok(self):
        s = self._sched(battery=80.0, thermal="nominal")
        s.dispatch("train", {}, power_class="heavy")
        tasks = s.poll()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].power_class, "heavy")

    def test_force_heavy_override(self):
        s = self._sched(battery=5.0, throttled=True)
        s.dispatch("train", {}, power_class="heavy")
        tasks = s.poll(force_heavy=True)
        self.assertEqual(len(tasks), 1)

    def test_bad_power_class_rejected(self):
        s = self._sched()
        with self.assertRaises(ValueError):
            s.dispatch("t", {}, power_class="ultra")

    def test_medium_power_class_accepted(self):
        s = self._sched()
        jid = s.dispatch("t", {}, power_class="medium")
        self.assertTrue(jid)

    def test_bad_tier_rejected(self):
        s = self._sched()
        with self.assertRaises(ValueError):
            s.dispatch("t", {}, tier="whatever")

    def test_complete_and_fail(self):
        s = self._sched()
        jid = s.dispatch("t", {}, power_class="light")
        tasks = s.poll()
        s.complete(tasks[0].job_id)
        self.assertEqual(jid, tasks[0].job_id)

    def test_medium_flows_at_level_0_and_1(self):
        for level in (0, 1):
            s = PowerAwareScheduler(
                _db(), monitor=PowerMonitor(
                    sampler=lambda: _FakeManager(degradation_level=level)))
            s.dispatch("m", {}, power_class="medium")
            tasks = s.poll()
            self.assertEqual(len(tasks), 1, f"level {level}")
            self.assertEqual(tasks[0].power_class, "medium")

    def test_medium_deferred_at_level_2(self):
        s = PowerAwareScheduler(
            _db(), monitor=PowerMonitor(
                sampler=lambda: _FakeManager(degradation_level=2,
                                             degradation_name="degraded",
                                             throttled=True)))
        s.dispatch("m", {}, power_class="medium")
        self.assertEqual(s.poll(), [])
        self.assertEqual(s.deferred_medium_count(), 1)

    def test_tier_shed_defers_bulk_light_task(self):
        s = PowerAwareScheduler(
            _db(), monitor=PowerMonitor(
                sampler=lambda: _FakeManager(
                    degradation_level=1, degradation_name="light-shed",
                    allowed_tiers=["critical", "important", "background"],
                    throttled=True)))
        s.dispatch("bulk-job", {}, power_class="light", tier="bulk")
        s.dispatch("chat", {}, power_class="light", tier="critical")
        tasks = s.poll()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_type, "chat")

    def test_budget_exhaustion_defers_task(self):
        from nomorals.os.resources import ResourceManager

        real = ResourceManager()
        real.budgets.define("llm", max_concurrent=1)
        self.assertTrue(real.budgets.acquire("llm"))  # occupy the slot
        s = PowerAwareScheduler(
            _db(),
            monitor=PowerMonitor(sampler=lambda: _FakeManager()),
            resources=real,
            budget_backoff_s=0.0)  # zero backoff so the test can re-poll
        s.dispatch("infer", {}, power_class="light", subsystem="llm")
        tasks = s.poll()
        self.assertEqual(tasks, [])  # budget exhausted -> requeued, not run
        real.budgets.release("llm")
        tasks = s.poll()
        self.assertEqual(len(tasks), 1)  # slot free -> flows
        s.complete(tasks[0].job_id)  # releases the acquired slot
        snap = s.budget_snapshot()
        self.assertEqual(snap["llm"]["running"], 0)

    def test_budget_backoff_delays_requeue(self):
        from nomorals.os.resources import ResourceManager

        real = ResourceManager()
        real.budgets.define("llm", max_concurrent=1)
        real.budgets.acquire("llm")  # occupy the slot
        s = PowerAwareScheduler(
            _db(),
            monitor=PowerMonitor(sampler=lambda: _FakeManager()),
            resources=real,
            budget_backoff_s=60.0)
        jid = s.dispatch("infer", {}, power_class="light", subsystem="llm")
        self.assertEqual(s.poll(), [])
        row = s.queue.get(jid)
        self.assertIsNotNone(row)
        # requeued with backoff: not available yet
        import time as _t
        self.assertGreater(row.payload.get("enqueued_at", 0), 0)
        db_row = s.db.query_one(
            f"SELECT available_at FROM {s.queue.TABLE} WHERE id = ?", (jid,))
        self.assertGreater(db_row["available_at"], _t.time())

    def test_resources_builds_monitor_when_missing(self):
        from nomorals.os.resources import ResourceManager

        s = PowerAwareScheduler(_db(), resources=ResourceManager())
        self.assertIsNotNone(s.monitor)
        st = s.monitor.status()
        self.assertIn("degradation_level", st.to_dict())

    def test_stats_shape(self):
        s = self._sched()
        s.dispatch("a", {}, power_class="light")
        s.dispatch("b", {}, power_class="heavy")
        stats = s.stats()
        self.assertEqual(stats["pending"]["light"], 1)
        self.assertEqual(stats["pending"]["heavy"], 1)
        self.assertIn("budgets", stats)

    def test_monitor_reports_degradation_and_charging(self):
        st = PowerMonitor(
            sampler=lambda: _FakeManager(
                degradation_level=2, degradation_name="degraded")).status()
        self.assertEqual(st.degradation_level, 2)
        self.assertEqual(st.degradation_name, "degraded")
        self.assertTrue(st.should_defer_medium)
        self.assertTrue(st.tier_allowed("critical"))
        d = st.to_dict()
        self.assertIn("time_to_empty_min", d)
        self.assertIn("allowed_tiers", d)


if __name__ == "__main__":
    unittest.main()
