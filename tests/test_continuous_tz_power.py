"""Offline tests: timezone robustness + power-restore boot order.

- nomorals.core.tz.safe_zoneinfo never raises, even with no tzdata at all
- morning_briefing / watchers / weather timezone paths survive missing tzdata
- PartnerRuntime._adopt_power_mode never crashes on a None gateway, and the
  boot order guarantees the gateway exists before it runs
"""

import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch


class TestSafeZoneinfo(unittest.TestCase):
    def test_valid_name(self):
        from nomorals.core.tz import safe_zoneinfo

        tz = safe_zoneinfo("America/Denver")
        # works whether or not tzdata is installed
        self.assertIsNotNone(tz)
        datetime.now(tz)

    def test_garbage_name_falls_back(self):
        from nomorals.core.tz import safe_zoneinfo

        tz = safe_zoneinfo("Not/AZone")
        self.assertIsNotNone(tz)
        datetime.now(tz)

    def test_none_and_empty(self):
        from nomorals.core.tz import safe_zoneinfo

        for bad in (None, "", "   "):
            tz = safe_zoneinfo(bad)
            self.assertIsNotNone(tz)
            datetime.now(tz)

    def test_total_tzdata_absence(self):
        """Simulate Termux with no tzdata: ZoneInfo raises for EVERYTHING."""
        from zoneinfo import ZoneInfoNotFoundError

        import nomorals.core.tz as tzmod

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            mod = real_import(name, *args, **kwargs)
            if name == "zoneinfo":
                class BrokenZoneInfo:
                    def __init__(self, key):
                        raise ZoneInfoNotFoundError(
                            f"No time zone found with key {key}")

                mod = type(mod)("zoneinfo")
                mod.ZoneInfo = BrokenZoneInfo
                mod.ZoneInfoNotFoundError = ZoneInfoNotFoundError
            return mod

        with patch("builtins.__import__", side_effect=fake_import):
            # reload to pick up the patched import machinery path
            self.assertEqual(tzmod.safe_zoneinfo("UTC"), timezone.utc)
            self.assertEqual(tzmod.safe_zoneinfo("America/Denver"), timezone.utc)
            self.assertEqual(tzmod.safe_zoneinfo(None), timezone.utc)

    def test_utc_fallback(self):
        from nomorals.core.tz import utc_fallback

        self.assertEqual(utc_fallback(), timezone.utc)


class TestBriefingTzPaths(unittest.TestCase):
    def test_today_str_no_tzdata(self):
        from zoneinfo import ZoneInfoNotFoundError
        from nomorals.agents import morning_briefing as mb

        real_zoneinfo = sys.modules.get("zoneinfo")

        class BrokenZoneInfo:
            def __init__(self, key):
                raise ZoneInfoNotFoundError(
                    f"No time zone found with key {key}")

        fake = type("zoneinfo", (), {
            "ZoneInfo": BrokenZoneInfo,
            "ZoneInfoNotFoundError": ZoneInfoNotFoundError,
        })
        try:
            sys.modules["zoneinfo"] = fake
            # must not raise even though every ZoneInfo() explodes
            day = mb._today_str(context=None)
            self.assertRegex(day, r"^\d{4}-\d{2}-\d{2}$")
        finally:
            if real_zoneinfo is not None:
                sys.modules["zoneinfo"] = real_zoneinfo
            else:
                sys.modules.pop("zoneinfo", None)

    def test_weather_zone_no_tzdata(self):
        from zoneinfo import ZoneInfoNotFoundError
        from nomorals.agents import weather as wx

        real_zoneinfo = sys.modules.get("zoneinfo")

        class BrokenZoneInfo:
            def __init__(self, key):
                raise ZoneInfoNotFoundError(
                    f"No time zone found with key {key}")

        fake = type("zoneinfo", (), {
            "ZoneInfo": BrokenZoneInfo,
            "ZoneInfoNotFoundError": ZoneInfoNotFoundError,
        })
        try:
            sys.modules["zoneinfo"] = fake
            tz = wx._zone("America/Denver")
            self.assertEqual(tz, timezone.utc)
        finally:
            if real_zoneinfo is not None:
                sys.modules["zoneinfo"] = real_zoneinfo
            else:
                sys.modules.pop("zoneinfo", None)


class TestPowerRestoreBootOrder(unittest.TestCase):
    def test_adopt_power_mode_none_gateway_no_crash(self):
        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.gateway = None
        # must not raise "'NoneType' object has no attribute 'set_rate_limit'"
        rt._adopt_power_mode()

    def test_boot_order_gateway_before_power_adopt(self):
        """The __init__ source must call _adopt_power_mode only after the
        gateway is guaranteed to exist."""
        import inspect

        from nomorals.agents.partner import runtime as rtmod

        src = inspect.getsource(rtmod.PartnerRuntime.__init__)
        adopt_pos = src.find("_adopt_power_mode()")
        self.assertGreater(adopt_pos, 0, "_adopt_power_mode not called in __init__")
        # the gateway construction block must come first
        gateway_pos = src.find("self.gateway = ChatGateway(")
        self.assertGreater(gateway_pos, 0, "ChatGateway construction not found")
        self.assertLess(gateway_pos, adopt_pos,
                        "boot order wrong: _adopt_power_mode runs before the "
                        "gateway is built")


if __name__ == "__main__":
    unittest.main()
