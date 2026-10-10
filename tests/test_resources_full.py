"""Full-signal ResourceManager: PSI, steal, cgroup, forecast, ladder, budgets."""

import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.core.platform import Platform, PlatformCapabilities
from nomorals.os import resources
from nomorals.os.resources import (
    DegradationLadder,
    ResourceManager,
    ResourceSample,
    cgroup_cpu_quota_cores,
    cgroup_memory_limit_bytes,
    detect_environment,
    reset_environment_cache,
    sample_psi,
)


def _pc_platform():
    return Platform(name="linux", capabilities=PlatformCapabilities(), detail="test")


def _calm(**over):
    kw = {
        "cpu_percent": 10.0, "mem_percent": 20.0, "disk_percent": 30.0,
        "battery_percent": None, "thermal_state": "nominal",
        "network_online": True, "metered": None, "ts": 0.0,
    }
    kw.update(over)
    return ResourceSample(**kw)


class TestPSI(unittest.TestCase):
    def test_parse_psi_files(self):
        canned = {
            "/proc/pressure/cpu":
                "some avg10=2.10 avg60=1.40 avg300=0.80 total=1240182",
            "/proc/pressure/memory":
                "some avg10=64.20 avg60=42.10 avg300=12.00 total=8923011\n"
                "full avg10=38.50 avg60=18.30 avg300=4.10 total=4129841",
            "/proc/pressure/io":
                "some avg10=8.40 avg60=5.10 avg300=2.00 total=3120914\n"
                "full avg10=1.20 avg60=0.80 avg300=0.20 total=410920",
        }

        def fake_read(path):
            return canned.get(path)

        with patch.object(resources, "_read_text", side_effect=fake_read):
            psi = sample_psi()
        self.assertIsNotNone(psi)
        self.assertAlmostEqual(psi["memory"].some_avg10, 64.20)
        self.assertAlmostEqual(psi["memory"].full_avg10, 38.50)
        self.assertIsNone(psi["cpu"].full_avg10)  # cpu has no full line
        self.assertAlmostEqual(psi["io"].some_avg300, 2.00)

    def test_psi_unavailable_returns_none(self):
        with patch.object(resources, "_read_text", return_value=None):
            self.assertIsNone(sample_psi())

    def test_psi_never_raises(self):
        with patch.object(resources, "_read_text",
                          side_effect=RuntimeError("gone")):
            self.assertIsNone(sample_psi())


class TestStealAndUtil(unittest.TestCase):
    def test_first_call_primes(self):
        mgr = ResourceManager(platform=_pc_platform())
        with patch.object(resources, "_read_cpu_stat",
                          return_value=(1000, 800, 10)):
            util, steal = mgr._sample_cpu_deltas()
        self.assertIsNone(util)
        self.assertIsNone(steal)

    def test_delta_math(self):
        mgr = ResourceManager(platform=_pc_platform())
        reads = iter([(1000, 800, 10), (2000, 1500, 110)])
        with patch.object(resources, "_read_cpu_stat",
                          side_effect=lambda: next(reads)):
            mgr._sample_cpu_deltas()  # prime
            util, steal = mgr._sample_cpu_deltas()
        # dt=1000, idle delta=700 -> busy 300 -> 30%; steal delta=100 -> 10%
        self.assertAlmostEqual(util, 30.0)
        self.assertAlmostEqual(steal, 10.0)

    def test_steal_critical_fails_consult(self):
        mgr = ResourceManager(platform=_pc_platform())
        s = _calm(steal_percent=30.0)
        with patch.object(mgr, "sample", return_value=s):
            r = mgr.consult()
        self.assertFalse(r["ok"])
        self.assertTrue(any("steal" in x for x in r["reasons"]))


class TestCgroup(unittest.TestCase):
    def test_memory_limit_walk(self):
        def fake_read(path):
            if path.endswith("memory.max"):
                return "1073741824"  # 1 GiB at the leaf
            return None

        with patch.object(resources, "_cgroup_mount",
                          return_value="/sys/fs/cgroup"), \
             patch.object(resources, "_self_cgroup_relpath",
                          return_value="/kubepods/pod123"), \
             patch.object(resources, "_read_text", side_effect=fake_read):
            self.assertEqual(cgroup_memory_limit_bytes(), 1073741824)

    def test_max_sentinel_is_unlimited(self):
        with patch.object(resources, "_cgroup_mount",
                          return_value="/sys/fs/cgroup"), \
             patch.object(resources, "_self_cgroup_relpath",
                          return_value="/"), \
             patch.object(resources, "_read_text", return_value="max"):
            self.assertIsNone(cgroup_memory_limit_bytes())

    def test_v1_sentinel_is_unlimited(self):
        def fake_read(path):
            if path.endswith("memory.limit_in_bytes"):
                return str(1 << 63)  # LONG_MAX sentinel
            return None

        with patch.object(resources, "_cgroup_mount",
                          return_value="/sys/fs/cgroup"), \
             patch.object(resources, "_self_cgroup_relpath",
                          return_value="/docker/abc"), \
             patch.object(resources, "_read_text", side_effect=fake_read):
            self.assertIsNone(cgroup_memory_limit_bytes())

    def test_cpu_quota_v2(self):
        def fake_read(path):
            if path.endswith("cpu.max"):
                return "50000 100000"  # half a core
            return None

        with patch.object(resources, "_cgroup_mount",
                          return_value="/sys/fs/cgroup"), \
             patch.object(resources, "_self_cgroup_relpath",
                          return_value="/"), \
             patch.object(resources, "_read_text", side_effect=fake_read), \
             patch.object(resources, "_read_float", return_value=None):
            self.assertAlmostEqual(cgroup_cpu_quota_cores(), 0.5)


class TestEnvironment(unittest.TestCase):
    def tearDown(self):
        reset_environment_cache()

    def test_termux_detected_via_prefix(self):
        with patch.dict(os.environ,
                        {"PREFIX": "/data/data/com.termux/files/usr"}):
            env = detect_environment(refresh=True)
        self.assertEqual(env.kind, "termux")
        self.assertIn("prefix", env.signals)

    def test_container_detected_via_dockerenv(self):
        with patch.object(resources.os.path, "exists",
                          side_effect=lambda p: p == "/.dockerenv"), \
             patch.object(resources, "_read_text", return_value=None), \
             patch.dict(os.environ, {}, clear=False):
            # PREFIX may leak termux; ensure it's absent
            os.environ.pop("PREFIX", None)
            env = detect_environment(refresh=True)
        self.assertTrue(env.containerized)

    def test_detection_never_raises(self):
        with patch.object(resources, "_read_text",
                          side_effect=RuntimeError("x")):
            env = detect_environment(refresh=True)
        self.assertEqual(env.kind, "unknown")


class TestTrendForecast(unittest.TestCase):
    def _mgr_with_history(self):
        mgr = ResourceManager(platform=_pc_platform())
        now = time.time()
        for i in range(5):
            mgr._history.append({
                "ts": now - (4 - i) * 60.0,
                "cpu": 10.0, "mem": 20.0 + i * 10.0, "disk": 30.0,
                "steal": None, "psi_mem": None, "composite": 0.1,
                "battery": None, "battery_status": None,
            })
        return mgr

    def test_trend_positive(self):
        mgr = self._mgr_with_history()
        slope = mgr.trend("mem", window_s=600.0)
        self.assertIsNotNone(slope)
        # +10 per 60s = +1/6 per second
        self.assertAlmostEqual(slope, 10.0 / 60.0, places=4)

    def test_trend_needs_two_points(self):
        mgr = ResourceManager(platform=_pc_platform())
        self.assertIsNone(mgr.trend("mem"))

    def test_forecast_minutes_to_critical(self):
        mgr = self._mgr_with_history()
        fc = mgr.forecast(300.0)
        mins = fc["minutes_to_critical"]["mem"]
        self.assertIsNotNone(mins)
        # mem now 60%, critical 93%, slope 10/60 per s
        self.assertAlmostEqual(mins, (93.0 - 60.0) / (10.0 / 60.0) / 60.0,
                               places=1)

    def test_forecast_flat_is_none(self):
        mgr = ResourceManager(platform=_pc_platform())
        now = time.time()
        for i in range(3):
            mgr._history.append({
                "ts": now - (2 - i) * 60.0, "cpu": 10.0, "mem": 20.0,
                "disk": 30.0, "steal": None, "psi_mem": None,
                "composite": 0.1, "battery": None, "battery_status": None,
            })
        fc = mgr.forecast(300.0)
        self.assertIsNone(fc["minutes_to_critical"]["mem"])


class TestLadder(unittest.TestCase):
    def test_step_up_needs_sustained_ticks(self):
        lad = DegradationLadder(up_ticks=2, calm_seconds=60.0)
        self.assertEqual(lad.update(2), 0)   # first tick: streak 1
        self.assertEqual(lad.update(2), 1)   # second tick: step up one
        self.assertEqual(lad.update(2), 1)   # streak resets; needs 2 more
        self.assertEqual(lad.update(2), 2)

    def test_critical_jumps_immediately(self):
        lad = DegradationLadder(up_ticks=5, calm_seconds=60.0)
        self.assertEqual(lad.update(4, critical=True), 4)

    def test_slow_start_recovery(self):
        lad = DegradationLadder(up_ticks=1, calm_seconds=60.0)
        lad.update(3, critical=True)
        t0 = time.time()
        self.assertEqual(lad.update(0, now=t0), 3)        # calm starts
        self.assertEqual(lad.update(0, now=t0 + 30), 3)   # not calm long enough
        self.assertEqual(lad.update(0, now=t0 + 61), 2)   # one step down
        self.assertEqual(lad.update(0, now=t0 + 90), 2)   # needs another window
        self.assertEqual(lad.update(0, now=t0 + 122), 1)

    def test_allowed_tiers(self):
        lad = DegradationLadder(up_ticks=1, calm_seconds=0.0)
        lad.update(1, critical=True)
        self.assertNotIn("bulk", lad.allowed_tiers())
        self.assertIn("critical", lad.allowed_tiers())
        lad.update(4, critical=True)
        self.assertEqual(lad.allowed_tiers(), ("critical",))

    def test_ladder_never_raises(self):
        lad = DegradationLadder()
        self.assertIsInstance(lad.update("bogus"), int)


class TestBudgets(unittest.TestCase):
    def test_acquire_release(self):
        mgr = ResourceManager(platform=_pc_platform())
        mgr.budgets.define("llm", max_mem_mb=100.0, max_concurrent=1)
        with patch.object(mgr.budgets, "machine_headroom_mb",
                          return_value=10000.0):
            self.assertTrue(mgr.budgets.acquire("llm", mem_mb=50.0))
            # concurrent slot exhausted
            self.assertFalse(mgr.budgets.acquire("llm", mem_mb=10.0))
            mgr.budgets.release("llm", mem_mb=50.0)
            self.assertTrue(mgr.budgets.acquire("llm", mem_mb=10.0))

    def test_headroom_reserve_blocks(self):
        mgr = ResourceManager(platform=_pc_platform())
        mgr.budgets.define("media", max_mem_mb=100000.0)
        with patch.object(mgr.budgets, "machine_headroom_mb",
                          return_value=100.0):
            ok, reasons = mgr.budgets.admission("media", mem_mb=500.0)
        self.assertFalse(ok)
        self.assertTrue(any("headroom" in r for r in reasons))

    def test_defaults_sized_from_effective(self):
        mgr = ResourceManager(platform=_pc_platform())
        with patch.object(mgr, "effective_memory_mb", return_value=2000.0), \
             patch.object(mgr, "effective_cpu_count", return_value=4.0):
            mgr.budgets.ensure_defaults()
            llm = mgr.budgets.get("llm")
        self.assertAlmostEqual(llm.max_mem_mb, 1400.0)  # 70% of 2000
        self.assertIsNotNone(mgr.budgets.get("media"))

    def test_consult_subsystem_budget_denial_throttles(self):
        mgr = ResourceManager(platform=_pc_platform())
        mgr.budgets.define("llm", max_concurrent=1)
        mgr.budgets.acquire("llm")  # occupy the only slot
        with patch.object(mgr, "sample", return_value=_calm()):
            r = mgr.consult(subsystem="llm")
        self.assertTrue(r["ok"])  # budget denial throttles, never fails
        self.assertTrue(r["throttled"])
        self.assertFalse(r["budget_ok"])
        self.assertTrue(r["budget_reasons"])

    def test_budgets_never_raise(self):
        mgr = ResourceManager(platform=_pc_platform())
        # acquire() wraps admission(); an internal explosion degrades to
        # "not admitted", never an exception
        with patch.object(mgr.budgets, "admission",
                          side_effect=RuntimeError("x")):
            self.assertFalse(mgr.budgets.acquire("llm", mem_mb=1.0))
            mgr.budgets.release("llm", mem_mb=1.0)  # must not raise either
        snap = mgr.budgets.snapshot()
        self.assertIsInstance(snap, dict)


class TestBatteryCharging(unittest.TestCase):
    def test_charging_relaxes_battery_floor(self):
        mgr = ResourceManager(platform=_pc_platform(), battery_min=15.0)
        s = _calm(battery_percent=10.0, battery_status="Charging")
        with patch.object(mgr, "sample", return_value=s):
            r = mgr.consult()
        self.assertTrue(r["ok"])  # 10 > 15/2 while charging

    def test_discharging_keeps_floor(self):
        mgr = ResourceManager(platform=_pc_platform(), battery_min=15.0)
        s = _calm(battery_percent=10.0, battery_status="Discharging")
        with patch.object(mgr, "sample", return_value=s):
            r = mgr.consult()
        self.assertFalse(r["ok"])

    def test_battery_forecast_drain(self):
        mgr = ResourceManager(platform=_pc_platform())
        now = time.time()
        mgr._history.append({"ts": now - 3600.0, "cpu": 1, "mem": 1,
                             "disk": 1, "steal": None, "psi_mem": None,
                             "composite": 0, "battery": 80.0,
                             "battery_status": "Discharging"})
        mgr._history.append({"ts": now, "cpu": 1, "mem": 1, "disk": 1,
                             "steal": None, "psi_mem": None,
                             "composite": 0, "battery": 60.0,
                             "battery_status": "Discharging"})
        fc = mgr.battery_forecast()
        self.assertAlmostEqual(fc["drain_pct_per_h"], 20.0)
        self.assertAlmostEqual(fc["time_to_empty_min"], 180.0)


class TestConsultShape(unittest.TestCase):
    def test_new_keys_present(self):
        mgr = ResourceManager(platform=_pc_platform())
        with patch.object(mgr, "sample", return_value=_calm()):
            r = mgr.consult()
        for key in ("degradation_level", "degradation", "forecast",
                    "battery_forecast", "psi", "environment", "budgets",
                    "budget_ok", "budget_reasons", "composite"):
            self.assertIn(key, r)
        self.assertEqual(r["degradation_level"], 0)
        self.assertTrue(r["budget_ok"])

    def test_composite_pressure_weights(self):
        mgr = ResourceManager(platform=_pc_platform())
        psi = {"memory": {"some_avg10": 100.0, "some_avg60": 0,
                         "some_avg300": 0, "full_avg10": 0,
                         "full_avg60": 0, "full_avg300": 0}}
        s = _calm(mem_percent=0.0, psi=psi, swap_used_percent=0.0)
        with patch.object(mgr, "_sample_swap_io_rate", return_value=0.0):
            score = mgr.composite_pressure(s)
        self.assertAlmostEqual(score, 0.55, places=2)


if __name__ == "__main__":
    unittest.main()
