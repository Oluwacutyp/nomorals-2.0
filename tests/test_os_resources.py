"""ResourceManager: advisory resource sampling, stdlib-only, never raises."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.core.platform import Platform, PlatformCapabilities
from nomorals.os import resources
from nomorals.os.resources import (
    ResourceManager,
    ResourceSample,
    advisor_callable,
    default_manager,
    reset_default_manager,
)


def _calm_sample(**over):
    kw = {
        "cpu_percent": 10.0,
        "mem_percent": 20.0,
        "disk_percent": 30.0,
        "battery_percent": None,
        "thermal_state": "nominal",
        "network_online": True,
        "metered": None,
        "ts": 0.0,
    }
    kw.update(over)
    return ResourceSample(**kw)


def _pc_platform():
    return Platform(name="linux", capabilities=PlatformCapabilities(), detail="test")


def _termux_platform():
    return Platform(
        name="termux",
        capabilities=PlatformCapabilities(process_pool=False, wake_lock=True,
                                          low_memory=True),
        detail="test",
    )


class TestSample(unittest.TestCase):
    def test_sample_sane_ranges(self):
        mgr = ResourceManager(platform=_pc_platform())
        s = mgr.sample()
        for name in ("cpu_percent", "mem_percent", "disk_percent", "battery_percent"):
            v = getattr(s, name)
            self.assertTrue(v is None or 0.0 <= v <= 100.0, (name, v))
        self.assertIn(s.thermal_state, (None, "nominal", "warm", "hot"))
        self.assertIsInstance(s.network_online, bool)
        self.assertTrue(s.metered is None or isinstance(s.metered, bool))

    def test_every_sampler_fails_but_sample_still_returns(self):
        # each underlying source fails -> every sampler degrades gracefully
        mgr = ResourceManager(platform=_pc_platform())
        with patch("nomorals.os.resources.os.getloadavg",
                   side_effect=OSError("no procfs")), \
             patch("nomorals.os.resources.shutil.disk_usage",
                   side_effect=OSError("no disk")), \
             patch("builtins.open", side_effect=OSError("denied")), \
             patch("nomorals.os.resources._read_float", side_effect=OSError("x")), \
             patch("nomorals.os.resources._thermal_zone_temps",
                   side_effect=RuntimeError("x")), \
             patch("nomorals.os.resources.os.listdir",
                   side_effect=OSError("no sysfs")), \
             patch("nomorals.os.resources.socket.create_connection",
                   side_effect=OSError("offline")):
            s = mgr.sample()
        self.assertIsNone(s.cpu_percent)
        self.assertIsNone(s.mem_percent)
        self.assertIsNone(s.disk_percent)
        self.assertIsNone(s.battery_percent)
        self.assertIsNone(s.thermal_state)
        self.assertFalse(s.network_online)

    def test_sample_never_raises(self):
        mgr = ResourceManager(platform=_pc_platform())
        with patch.object(ResourceManager, "_sample_cpu",
                          side_effect=RuntimeError("cpu gone")), \
             patch.object(ResourceManager, "_sample_mem",
                          side_effect=ValueError("mem gone")):
            s = mgr.sample()  # sample() itself degrades to None on Exception
        self.assertIsNone(s.cpu_percent)
        self.assertIsNone(s.mem_percent)
        self.assertFalse(s.network_online is None)
        # and consult survives even the pathological case
        with patch.object(ResourceManager, "sample",
                          side_effect=RuntimeError("all gone")):
            r = mgr.consult()
        self.assertIn("ok", r)


class TestPressure(unittest.TestCase):
    def test_pressure_math_on_synthetic_sample(self):
        mgr = ResourceManager(platform=_pc_platform())
        p = mgr.pressure(_calm_sample(cpu_percent=50.0, mem_percent=25.0,
                                      disk_percent=100.0, battery_percent=80.0,
                                      thermal_state="warm"))
        self.assertAlmostEqual(p["cpu"], 0.50)
        self.assertAlmostEqual(p["mem"], 0.25)
        self.assertAlmostEqual(p["disk"], 1.0)
        self.assertAlmostEqual(p["battery"], 0.20)   # 1 - 0.8
        self.assertAlmostEqual(p["thermal"], 0.5)
        self.assertAlmostEqual(p["overall"], 1.0)

    def test_pressure_thermal_map(self):
        mgr = ResourceManager(platform=_pc_platform())
        self.assertEqual(mgr.pressure(_calm_sample(thermal_state="nominal"))["thermal"], 0.0)
        self.assertEqual(mgr.pressure(_calm_sample(thermal_state="hot"))["thermal"], 1.0)
        self.assertEqual(mgr.pressure(_calm_sample(thermal_state=None))["thermal"], 0.0)

    def test_pressure_unknown_is_zero(self):
        mgr = ResourceManager(platform=_pc_platform())
        s = _calm_sample(cpu_percent=None, mem_percent=None, disk_percent=None)
        p = mgr.pressure(s)
        self.assertEqual(p["cpu"], 0.0)
        self.assertEqual(p["mem"], 0.0)
        self.assertEqual(p["disk"], 0.0)

    def test_pressure_never_raises(self):
        mgr = ResourceManager(platform=_pc_platform())
        with patch.object(mgr, "sample", side_effect=RuntimeError("gone")):
            p = mgr.pressure()
        self.assertIn("overall", p)


class TestConsult(unittest.TestCase):
    def _consult(self, mgr, sample, mission=None):
        with patch.object(mgr, "sample", return_value=sample):
            return mgr.consult(mission)

    def test_calm_is_ok(self):
        mgr = ResourceManager(platform=_pc_platform())
        r = self._consult(mgr, _calm_sample())
        self.assertTrue(r["ok"])
        self.assertFalse(r["throttled"])
        self.assertIn("pressure", r)
        self.assertIn("sample", r)

    def test_cpu_critical_fails_ok(self):
        mgr = ResourceManager(platform=_pc_platform())
        r = self._consult(mgr, _calm_sample(cpu_percent=98.0))
        self.assertFalse(r["ok"])
        self.assertTrue(any("cpu" in x for x in r["reasons"]))

    def test_cpu_throttle(self):
        mgr = ResourceManager(platform=_pc_platform())
        r = self._consult(mgr, _calm_sample(cpu_percent=75.0))
        self.assertTrue(r["ok"])
        self.assertTrue(r["throttled"])

    def test_low_battery_fails_ok(self):
        mgr = ResourceManager(platform=_pc_platform())
        r = self._consult(mgr, _calm_sample(battery_percent=5.0))
        self.assertFalse(r["ok"])
        self.assertTrue(any("battery" in x for x in r["reasons"]))

    def test_hot_thermal_fails_ok(self):
        mgr = ResourceManager(platform=_pc_platform())
        r = self._consult(mgr, _calm_sample(thermal_state="hot"))
        self.assertFalse(r["ok"])

    def test_metered_throttles(self):
        mgr = ResourceManager(platform=_pc_platform(), metered=True)
        r = self._consult(mgr, _calm_sample())
        self.assertTrue(r["throttled"])
        self.assertTrue(any("metered" in x for x in r["reasons"]))

    def test_mission_pressure_cap(self):
        mgr = ResourceManager(platform=_pc_platform())
        mission = SimpleNamespace(max_pressure=0.1)
        r = self._consult(mgr, _calm_sample(cpu_percent=50.0), mission=mission)
        self.assertTrue(r["ok"])  # cap throttles, never fails
        self.assertTrue(r["throttled"])
        self.assertTrue(any("cap" in x for x in r["reasons"]))

    def test_consult_never_raises_even_when_samplers_explode(self):
        mgr = ResourceManager(platform=_pc_platform())
        with patch.object(mgr, "sample", side_effect=RuntimeError("all down")), \
             patch.object(mgr, "pressure", side_effect=RuntimeError("also down")):
            r = mgr.consult()
        self.assertFalse(r["ok"])
        self.assertTrue(r["throttled"])
        self.assertTrue(r["reasons"])


class TestOverrides(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        reset_default_manager()

    def test_constructor_kwargs_respected(self):
        mgr = ResourceManager(platform=_pc_platform(), cpu_throttle=0.01)
        with patch.object(mgr, "sample",
                          return_value=_calm_sample(cpu_percent=10.0)):
            r = mgr.consult()
        self.assertTrue(r["throttled"])

    def test_env_overrides_win(self):
        os.environ["NM_RESOURCE_CPU_THROTTLE"] = "0.05"
        os.environ["NM_RESOURCE_BATTERY_MIN"] = "99"
        os.environ["NM_RESOURCE_METERED"] = "1"
        mgr = ResourceManager(platform=_pc_platform(), cpu_throttle=0.99,
                              battery_min=1.0, metered=False)
        self.assertAlmostEqual(mgr.cpu_throttle, 0.05)
        self.assertAlmostEqual(mgr.battery_min, 99.0)
        self.assertTrue(mgr.metered)
        with patch.object(mgr, "sample",
                          return_value=_calm_sample(cpu_percent=10.0)):
            r = mgr.consult()
        self.assertTrue(r["throttled"])  # via env cpu throttle

    def test_bad_env_values_fall_back_to_kwargs(self):
        os.environ["NM_RESOURCE_CPU_THROTTLE"] = "not-a-number"
        os.environ["NM_RESOURCE_METERED"] = "maybe"
        mgr = ResourceManager(platform=_pc_platform(), cpu_throttle=0.42)
        self.assertAlmostEqual(mgr.cpu_throttle, 0.42)
        self.assertIsNone(mgr.metered)

    def test_termux_is_more_conservative(self):
        pc = ResourceManager(platform=_pc_platform())
        phone = ResourceManager(platform=_termux_platform())
        self.assertGreaterEqual(phone.battery_throttle, pc.battery_throttle)
        self.assertLessEqual(phone.cpu_throttle, pc.cpu_throttle)
        self.assertLessEqual(phone.mem_throttle, pc.mem_throttle)


class TestAdvisor(unittest.TestCase):
    def tearDown(self):
        reset_default_manager()

    def test_advisor_callable_returns_working_callable(self):
        mgr = ResourceManager(platform=_pc_platform())
        advise = advisor_callable(mgr)
        self.assertTrue(callable(advise))
        with patch.object(mgr, "sample", return_value=_calm_sample()):
            r = advise(SimpleNamespace(name="demo"))
        self.assertTrue(r["ok"])
        self.assertIn("pressure", r)

    def test_advisor_callable_defaults_to_singleton(self):
        advise = advisor_callable()
        r = advise(None)
        self.assertIn("ok", r)

    def test_advisor_never_raises(self):
        mgr = ResourceManager(platform=_pc_platform())
        with patch.object(mgr, "consult", side_effect=RuntimeError("down")):
            r = advisor_callable(mgr)(None)
        self.assertFalse(r["ok"])
        self.assertTrue(r["throttled"])
        self.assertTrue(r["reasons"])

    def test_default_manager_is_cached_singleton(self):
        a = default_manager()
        b = default_manager()
        self.assertIs(a, b)
        reset_default_manager()
        c = default_manager()
        self.assertIsNot(c, a)


if __name__ == "__main__":
    unittest.main()
