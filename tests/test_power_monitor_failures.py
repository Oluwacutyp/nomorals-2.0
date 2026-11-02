"""Pins for nomorals/power/monitor.py (tail-116): behavior + NOT-recording.

Sites pinned (all exclusions — probe ladders and best-effort sampling):

* 283 — ``_collect_battery``: a raising ``sensors_battery()`` yields
  ``b = None`` and the sysfs fallback rung runs; the sample carries no
  battery value rather than a fabricated one.
* 314 — ``_collect_thermal``: a raising ``sensors_temperatures()``
  yields ``temps = {}`` and the sysfs thermal rung runs.
* 332 — per-entry threshold guard: an entry whose ``high`` cannot be
  converted is skipped for thresholds only; its ``current`` still
  counts toward the max and earlier thresholds survive.
* 528 — ``_record_sample``: a non-numeric ``battery_pct`` on a
  duck-typed status is skipped; the in-memory history (the drain
  forecast's only input) just misses that sample.

NOT-recording: the power package has no failures ledger; these guards
bind nothing and emit nothing. ``monitor._log`` is patched with a
MagicMock around the crash triggers and ``method_calls`` must stay
empty (tail-17 precedent). This file contains NO except handlers:
every trigger is a mock side_effect raised inside the target's own
handler.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.power.monitor import PowerMonitor, PowerStatus, _LocalSampler


def _boom(exc):
    def _raise(*a, **k):
        raise exc
    return _raise


def _entry(current, high, critical):
    return SimpleNamespace(current=current, high=high,
                           critical=critical, label="cpu")


class TestBatteryProbe283(unittest.TestCase):
    def test_raising_sensors_battery_degrades_to_no_value(self):
        fake = SimpleNamespace(
            sensors_battery=_boom(RuntimeError("psutil dead")))
        sampler = _LocalSampler(fake, sysfs_root="/nonexistent")
        with mock.patch("nomorals.power.monitor._log") as log:
            s = sampler.sample()
        self.assertIsNone(s.battery_percent)
        self.assertEqual(log.method_calls, [])

    def test_healthy_battery_control(self):
        batt = SimpleNamespace(percent=77.7, power_plugged=True,
                               secsleft=3600)
        sampler = _LocalSampler(SimpleNamespace(sensors_battery=lambda: batt),
                                sysfs_root="/nonexistent")
        s = sampler.sample()
        self.assertEqual(s.battery_percent, 77.7)
        self.assertEqual(s.power_plugged, True)


class TestThermalProbe314(unittest.TestCase):
    def test_raising_sensors_temperatures_degrades_to_no_value(self):
        fake = SimpleNamespace(
            sensors_temperatures=_boom(RuntimeError("psutil dead")))
        sampler = _LocalSampler(fake, sysfs_root="/nonexistent")
        with mock.patch("nomorals.power.monitor._log") as log:
            s = sampler.sample()
        self.assertIsNone(s.thermal_max_c)
        self.assertEqual(log.method_calls, [])

    def test_healthy_thermal_control(self):
        fake = SimpleNamespace(sensors_temperatures=lambda: {
            "cpu": [_entry(55.0, 80.0, 100.0)]})
        sampler = _LocalSampler(fake, sysfs_root="/nonexistent")
        s = sampler.sample()
        self.assertEqual(s.thermal_max_c, 55.0)


class TestThresholdGuard332(unittest.TestCase):
    def test_unconvertible_high_skips_threshold_only(self):
        fake = SimpleNamespace(sensors_temperatures=lambda: {
            "cpu": [_entry(50.0, 90.0, 110.0), _entry(60.0, "junk", None)]})
        sampler = _LocalSampler(fake, sysfs_root="/nonexistent")
        with mock.patch("nomorals.power.monitor._log") as log:
            s = sampler.sample()
        # the second entry's current still counts; the first entry's
        # thresholds survive (state computed, not defaulted away)
        self.assertEqual(s.thermal_max_c, 60.0)
        self.assertEqual(s.thermal_state, "nominal")
        self.assertEqual(log.method_calls, [])


class TestRecordSample528(unittest.TestCase):
    def test_non_numeric_pct_skipped(self):
        mon = PowerMonitor()
        st = PowerStatus(battery_pct="high", thermal_state=None,
                         throttled=False, ok=True, reasons=[], charging=False)
        with mock.patch("nomorals.power.monitor._log") as log:
            mon._record_sample(st)
        self.assertEqual(len(mon._history), 0)
        self.assertEqual(log.method_calls, [])

    def test_numeric_pct_recorded_control(self):
        mon = PowerMonitor()
        st = PowerStatus(battery_pct=50.0, thermal_state=None,
                         throttled=False, ok=True, reasons=[], charging=False)
        mon._record_sample(st)
        self.assertEqual(len(mon._history), 1)


if __name__ == "__main__":
    unittest.main()
