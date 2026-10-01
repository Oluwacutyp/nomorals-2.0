"""Tests for the weather-awareness module (agents/weather.py) and the
weather tool.  All parsing tests run on fixtures — no network."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import nomorals.agents.weather as wx
from nomorals.agents.weather import (
    CurrentConditions,
    DayForecast,
    FeedEntry,
    KnowledgeFeed,
    Place,
    USASituations,
    Weather,
    WeatherAlert,
    convert_time,
    format_with_tz,
    now_in_tz,
    owner_tz,
    parse_current,
    parse_daily,
    parse_geocode,
    parse_nws_alerts,
    tz_abbr,
    tz_note,
)

GEOCODE_FIXTURE = {
    "results": [{
        "name": "New York", "latitude": 40.71427, "longitude": -74.00597,
        "timezone": "America/New_York", "country": "United States",
        "admin1": "New York",
    }]
}

FORECAST_FIXTURE = {
    "timezone": "America/New_York",
    "current": {
        "time": "2026-10-01T10:30", "temperature_2m": 68.5,
        "relative_humidity_2m": 55, "apparent_temperature": 69.1,
        "weather_code": 3, "wind_speed_10m": 8.2,
        "wind_direction_10m": 270,
    },
    "daily": {
        "time": ["2026-10-01", "2026-10-02"],
        "weather_code": [3, 61],
        "temperature_2m_max": [72.0, 70.0],
        "temperature_2m_min": [58.0, 57.0],
        "precipitation_probability_max": [10, 60],
    },
}

NWS_FIXTURE = {
    "features": [
        {"properties": {
            "event": "Flood Warning", "severity": "Severe",
            "messageType": "Alert",
            "headline": "Flood Warning issued Oct 1",
            "areaDesc": "Daviess, MO", "onset": "2026-10-01T12:33:00-05:00",
            "ends": "2026-10-03T03:24:00-05:00",
            "description": "River flooding.", "instruction": "Move to high ground.",
            "senderName": "NWS Kansas City"}},
        {"properties": {
            "event": "Heat Advisory", "severity": "Moderate",
            "messageType": "Alert", "headline": "Heat Advisory",
            "areaDesc": "Maricopa, AZ", "onset": "2026-10-01T09:00:00-07:00",
            "ends": "2026-10-01T20:00:00-07:00"}},
        {"properties": {
            "event": "Test Message", "severity": "Unknown",
            "messageType": "Test", "headline": "weekly test"}},
        {"properties": {
            "event": "Flood Warning", "severity": "Severe",
            "messageType": "Cancel", "headline": "cancelled"}},
    ]
}


class WmoTests(unittest.TestCase):
    def test_known_code(self):
        self.assertEqual(wx.WMO[0][0], "Clear sky")

    def test_current_falls_back_on_unknown_code(self):
        cur = CurrentConditions(temp=70.0, feels_like=None, humidity=None,
                                weather_code=12345, wind_speed=None,
                                wind_direction=None)
        self.assertEqual(cur.description, "Unknown")
        self.assertIn("70", cur.one_liner("X"))


class GeocodeTests(unittest.TestCase):
    def test_parse(self):
        p = parse_geocode(GEOCODE_FIXTURE)
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.name, "New York")
        self.assertAlmostEqual(p.latitude, 40.71427)
        self.assertEqual(p.timezone, "America/New_York")
        self.assertIn("New York", p.label)

    def test_empty_results(self):
        self.assertIsNone(parse_geocode({"results": []}))
        self.assertIsNone(parse_geocode({}))


class ForecastParseTests(unittest.TestCase):
    def test_current(self):
        cur = parse_current(FORECAST_FIXTURE)
        self.assertIsNotNone(cur)
        assert cur is not None
        self.assertAlmostEqual(cur.temp, 68.5)
        self.assertEqual(cur.description, "Overcast")
        line = cur.one_liner("New York, NY")
        self.assertIn("68", line)
        self.assertIn("Overcast", line)
        self.assertIn("humidity 55%", line)

    def test_current_missing(self):
        self.assertIsNone(parse_current({}))

    def test_daily(self):
        days = parse_daily(FORECAST_FIXTURE)
        self.assertEqual(len(days), 2)
        self.assertEqual(days[0].date, "2026-10-01")
        self.assertEqual(days[1].description, "Light rain")
        self.assertEqual(days[1].precip_prob, 60)
        self.assertIn("2026-10-02", days[1].one_liner())


class NwsParseTests(unittest.TestCase):
    def test_filters_and_ranks(self):
        alerts = parse_nws_alerts(NWS_FIXTURE)
        # Test message + Cancel dropped; 2 real alerts remain
        self.assertEqual(len(alerts), 2)
        # Severe outranks Moderate
        self.assertEqual(alerts[0].event, "Flood Warning")
        self.assertEqual(alerts[1].event, "Heat Advisory")
        self.assertEqual(alerts[0].rank, 3)
        self.assertIn("Move to high ground", alerts[0].instruction)

    def test_one_liner(self):
        alerts = parse_nws_alerts(NWS_FIXTURE)
        line = alerts[0].one_liner()
        self.assertIn("Flood Warning", line)
        self.assertIn("Severe", line)
        self.assertIn("Daviess, MO", line)

    def test_empty(self):
        self.assertEqual(parse_nws_alerts({}), [])
        self.assertEqual(parse_nws_alerts({"features": []}), [])


class TzTests(unittest.TestCase):
    def test_owner_tz_from_settings(self):
        s = SimpleNamespace(timezone="America/Chicago")
        self.assertEqual(owner_tz(s), "America/Chicago")

    def test_owner_tz_from_partner(self):
        s = SimpleNamespace(partner=SimpleNamespace(tz="Europe/London"))
        self.assertEqual(owner_tz(s), "Europe/London")

    def test_owner_tz_env_fallback(self):
        with mock.patch.dict(os.environ, {"TZ": "Asia/Tokyo"}):
            self.assertEqual(owner_tz(SimpleNamespace()), "Asia/Tokyo")

    def test_owner_tz_utc_default(self):
        env = {k: v for k, v in os.environ.items() if k != "TZ"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(owner_tz(None), "UTC")

    def test_convert_ny_to_london(self):
        # 2026-10-01: EDT (UTC-4) vs BST (UTC+1)
        res = convert_time("2026-10-01 15:00",
                           "America/New_York", "Europe/London")
        self.assertTrue(res["ok"])
        self.assertIn("15:00 EDT", res["from"])
        self.assertIn("20:00 BST", res["to"])

    def test_convert_unix_ts(self):
        res = convert_time(0, "UTC", "America/New_York")
        self.assertTrue(res["ok"])
        self.assertIn("1969-12-31 19:00 EST", res["to"])

    def test_convert_bad_input(self):
        res = convert_time("not a time", "UTC", "UTC")
        self.assertFalse(res["ok"])

    def test_dst_spring_forward(self):
        # 2026-03-08 02:00 → 03:00 in America/New_York; 01:30 is EST
        res = convert_time("2026-03-08 01:30",
                           "America/New_York", "UTC")
        self.assertTrue(res["ok"])
        self.assertIn("01:30 EST", res["from"])
        self.assertIn("06:30 UTC", res["to"])

    def test_dst_fall_back_first_occurrence(self):
        # 2026-11-01 01:30 happens twice; fold=0 = EDT occurrence
        res = convert_time("2026-11-01 01:30",
                           "America/New_York", "UTC")
        self.assertTrue(res["ok"])
        self.assertIn("01:30 EDT", res["from"])
        self.assertIn("05:30 UTC", res["to"])

    def test_format_with_tz(self):
        from datetime import datetime, timezone
        m = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(format_with_tz(m, "America/New_York"),
                         "2026-10-01 08:00 EDT")

    def test_tz_note(self):
        note = tz_note("America/New_York")
        self.assertIn("America/New_York", note)
        self.assertRegex(note, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")

    def test_now_in_tz(self):
        now = now_in_tz("Asia/Tokyo")
        self.assertEqual(tz_abbr(now), "JST")

    def test_bad_zone_falls_back_to_utc(self):
        now = now_in_tz("Not/AZone")
        self.assertEqual(now.tzinfo.key, "UTC")


class KnowledgeFeedTests(unittest.TestCase):
    def test_ttl_caching(self):
        calls = {"n": 0}

        class FakeWeather:
            def now(self, place=None):
                calls["n"] += 1
                return {"ok": True, "text": f"fake now {calls['n']}"}

            def forecast(self, place=None, days=3):
                return {"ok": True, "text": "fake outlook"}

        feed = KnowledgeFeed(weather=FakeWeather(),  # type: ignore[arg-type]
                             default_place="X", tz_name="UTC")
        first = feed.pull(["weather-now"])
        second = feed.pull(["weather-now"])
        self.assertEqual(calls["n"], 1)  # cached
        self.assertEqual(first[0].body, second[0].body)
        feed.invalidate("weather-now")
        feed.pull(["weather-now"])
        self.assertEqual(calls["n"], 2)  # refreshed

    def test_entries_labeled_with_tz(self):
        class FakeWeather:
            def now(self, place=None):
                return {"ok": True, "text": "sunny"}

            def forecast(self, place=None, days=3):
                return {"ok": True, "text": "sunny tomorrow"}

        feed = KnowledgeFeed(weather=FakeWeather(),  # type: ignore[arg-type]
                             default_place="X", tz_name="America/New_York")
        entries = feed.pull()
        kinds = {e.kind for e in entries}
        self.assertEqual(kinds, {"weather-now", "today-outlook",
                                "usa-situation", "tz-note"})
        for e in entries:
            self.assertTrue(e.tz_abbr)  # every entry tz-labeled
            self.assertIn(e.tz_abbr, e.chat_line())
        # usa-situation works without network in the degraded path
        usa = next(e for e in entries if e.kind == "usa-situation")
        self.assertTrue(usa.body)

    def test_unknown_kind_raises(self):
        feed = KnowledgeFeed(tz_name="UTC")
        with self.assertRaises(ValueError):
            feed.refresh("bogus-kind")


class ProviderTests(unittest.TestCase):
    def _ctx(self):
        return SimpleNamespace(
            settings=SimpleNamespace(
                weather=SimpleNamespace(location="Springfield")))

    def test_weather_provider(self):
        class FakeWeather:
            def __init__(self, default_place=""):
                self.default_place = default_place

            def now(self, place=None):
                return {"ok": True, "text": "☀️ Springfield: Clear sky, 75°F"}

            def forecast(self, place=None, days=3):
                return {"ok": True,
                        "text": "📅 Springfield — 1-day outlook:\n• ☀️ day"}

        with mock.patch.object(wx, "Weather", FakeWeather):
            prov = wx.WeatherProvider()
            sec = prov.collect(self._ctx(), 0.0)
        self.assertIsNotNone(sec)
        assert sec is not None
        self.assertEqual(sec.name, "weather")
        self.assertTrue(sec.lines)

    def test_weather_provider_network_failure(self):
        class BoomWeather:
            def __init__(self, default_place=""):
                pass

            def now(self, place=None):
                raise RuntimeError("net down")

        with mock.patch.object(wx, "Weather", BoomWeather):
            prov = wx.WeatherProvider()
            self.assertIsNone(prov.collect(self._ctx(), 0.0))

    def test_usa_provider(self):
        fake_res = {"ok": True,
                    "text": "🇺🇸 USA right now:\n• 3 active weather alerts",
                    "top_events": [{"event": "Flood Warning"}],
                    "alert_count": 3, "cities": [], "usa_news": []}

        class FakeSituations:
            def __init__(self, weather=None):
                pass

            def overview(self, news_items=None):
                return fake_res

        with mock.patch.object(wx, "USASituations", FakeSituations), \
             mock.patch("nomorals.agents.news.NewsAgent") as na:
            na.return_value.recent.return_value = []
            prov = wx.USASituationsProvider()
            sec = prov.collect(self._ctx(), 0.0)
        self.assertIsNotNone(sec)
        assert sec is not None
        self.assertEqual(sec.name, "usa")
        self.assertTrue(any("alerts" in ln for ln in sec.lines))

    def test_usa_keyword_filter(self):
        self.assertTrue(USASituations._is_usa(
            {"title": "Senate passes infrastructure bill", "summary": ""}))
        self.assertFalse(USASituations._is_usa(
            {"title": "Lagos market rally", "summary": ""}))


class ToolRegistrationTests(unittest.TestCase):
    def test_weather_tool_registered(self):
        from nomorals.tools.registry import ToolRegistry
        reg = ToolRegistry(context=SimpleNamespace(settings=None))
        reg.register_builtins()
        names = set()
        try:
            names = set(reg.names())
        except Exception:  # noqa: BLE001 — fall back to tools dict
            names = set(getattr(reg, "tools", {}).keys())
        self.assertIn("weather", names)

    def test_briefing_composer_includes_providers(self):
        from nomorals.agents.morning_briefing import BriefingComposer
        comp = BriefingComposer()
        names = {p.name for p in comp.providers}
        self.assertIn("weather", names)
        self.assertIn("usa", names)


if __name__ == "__main__":
    unittest.main()
