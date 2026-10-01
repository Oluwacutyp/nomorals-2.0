"""Platform backend: capabilities as data, Termux first-class."""

import unittest
from types import SimpleNamespace

from nomorals.core.platform import (
    Platform,
    PlatformCapabilities,
    detect_platform,
    reset_platform_cache,
)


def prof(kind):
    return SimpleNamespace(kind=kind, detail=f"test {kind}")


class TestPlatformBackend(unittest.TestCase):
    def tearDown(self):
        reset_platform_cache()

    def test_termux_declares_limits(self):
        plat = detect_platform(profile=prof("termux"), refresh=True)
        self.assertEqual(plat.name, "termux")
        self.assertFalse(plat.supports("process_pool"))
        self.assertFalse(plat.supports("gpu"))
        self.assertFalse(plat.supports("background_service"))
        self.assertTrue(plat.capabilities.wake_lock)
        self.assertTrue(plat.capabilities.low_memory)
        self.assertEqual(plat.capabilities.max_workers, 4)

    def test_unknown_capability_defaults_to_true(self):
        plat = detect_platform(profile=prof("termux"), refresh=True)
        self.assertTrue(plat.supports("something_not_listed"))

    def test_linux_declares_process_pool(self):
        plat = detect_platform(profile=prof("pc"), refresh=True)
        # on this CI/dev machine the profile kind pc maps to a desktop OS
        self.assertIn(plat.name, {"linux", "macos", "windows"})
        self.assertTrue(plat.supports("process_pool"))

    def test_detection_is_cached(self):
        a = detect_platform(profile=prof("termux"), refresh=True)
        b = detect_platform()
        self.assertIs(a, b)

    def test_to_dict_shape(self):
        plat = detect_platform(profile=prof("termux"), refresh=True)
        d = plat.to_dict()
        self.assertEqual(d["name"], "termux")
        self.assertIn("process_pool", d["capabilities"])
        self.assertIn("max_workers", d["capabilities"])

    def test_platform_value_object(self):
        plat = Platform(name="x", capabilities=PlatformCapabilities(process_pool=False))
        self.assertFalse(plat.supports("process_pool"))
        self.assertTrue(plat.supports("gpu") is False)  # declared False stays False


if __name__ == "__main__":
    unittest.main()
