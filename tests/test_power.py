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
                 reasons=None):
        self._sample = _FakeSample(battery, thermal)
        self._advisory = {"throttled": throttled, "ok": ok,
                          "reasons": reasons or []}

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
            s.dispatch("t", {}, power_class="medium")

    def test_complete_and_fail(self):
        s = self._sched()
        jid = s.dispatch("t", {}, power_class="light")
        tasks = s.poll()
        s.complete(tasks[0].job_id)
        self.assertEqual(jid, tasks[0].job_id)


if __name__ == "__main__":
    unittest.main()
