"""Live flight tracking (52-websites #21 FlightRadar24).

Devon's flight knowledge was trip-shaped (airport table, fare
watchers, itineraries) — nothing answered "where is flight BA249
right now?" from live data. This suite pins the live-tracking
organ: OpenSky ADS-B states normalised to ``AircraftState``,
flight-number → callsign resolution, freshness that is stated
never faked, area search, airport coordinates from the same
OurAirports dump as the country table, great-circle ETA, the
durable track journal + the track map she draws herself, and the
flight watcher (landed / late / diverted) under the fare watchers'
alert discipline — one alert per event, never spam.

Fixtures are recorded-shape OpenSky payloads (the live endpoint
was probed 2026-11-17: ``{"time": ..., "states": null}`` for an
empty box — ``states`` is null, not [], when nothing is airborne
there). No test ever touches the live feed; the feed is injected.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.travel.tracking import (
    AIRLINE_ICAO,
    AREA_PRESETS,
    FLIGHT_TRACK_JOB,
    FLIGHT_TRACK_SPEC,
    FLIGHT_TRACK_TOOL,
    LAGOS_BBOX,
    AircraftState,
    FlightWatcher,
    OpenSkyFeed,
    TrackStore,
    aircraft_in_area,
    airport_info,
    bearing_deg,
    callsigns_for_flight,
    coords_for_iata,
    ensure_flight_track_job,
    eta_minutes,
    find_flight,
    haversine_km,
    nearest_airport,
    parse_flight_query,
    parse_opensky_states,
    render_track_map,
    run_flight_track_tick_once,
    track_flight,
)

_NOW = 1794941424.0

# A recorded-shape OpenSky payload: [icao24, callsign,
# origin_country, time_position, last_contact, longitude, latitude,
# baro_altitude, on_ground, velocity, true_track, vertical_rate,
# sensors, geo_altitude, squawk, spi, position_source, category].
_FIXTURE_PAYLOAD = {
    "time": 1794941424,
    "states": [
        ["4b1814", "BAW249  ", "United Kingdom", 1794941400,
         1794941400, -20.0, 45.0, 10668.0, False, 250.0, 300.0, 0.0,
         None, 11000.0, "1234", False, 0, 0],
        ["7a1b2c", "APK1234 ", "Nigeria", 1794941410, 1794941410,
         3.32116, 6.57737, 0.0, True, 0.0, 0.0, 0.0,
         None, 0.0, "4321", False, 0, 0],
        ["4d2200", "ETH501  ", "Ethiopia", 1794941390, 1794941390,
         None, None, 11500.0, False, 240.0, 90.0, 0.0,
         None, 12000.0, "5678", False, 0, 0],
    ],
}

_EMPTY_PAYLOAD = {"time": 1794941424, "states": None}


class _FakeFeed:
    """A feed returning canned states, in the real feed's shape."""

    def __init__(self, states, meta=None):
        self._states = list(states)
        self._meta = meta or {"source": "opensky",
                              "fetched_at": _NOW, "error": "",
                              "cached": False}
        self.calls = 0

    def fetch(self, bbox=None):
        self.calls += 1
        if bbox is None:
            return list(self._states), dict(self._meta)
        lamin, lomin, lamax, lomax = bbox
        kept = [s for s in self._states
                if s.latitude is not None and s.longitude is not None
                and lamin <= s.latitude <= lamax
                and lomin <= s.longitude <= lomax]
        return kept, dict(self._meta)


def _states(now=_NOW):
    return parse_opensky_states(_FIXTURE_PAYLOAD, now)


def _los_state(now=_NOW):
    return [s for s in _states(now) if s.callsign == "APK1234"][0]


def _baw_state(now=_NOW):
    return [s for s in _states(now) if s.callsign == "BAW249"][0]


class ParseTests(unittest.TestCase):
    def test_payload_parses_three_states(self):
        states = _states()
        self.assertEqual(3, len(states))
        self.assertEqual(["BAW249", "APK1234", "ETH501"],
                         [s.callsign for s in states])

    def test_callsign_trailing_spaces_stripped(self):
        self.assertEqual("BAW249", _baw_state().callsign)

    def test_null_states_is_empty_not_an_error(self):
        self.assertEqual([], parse_opensky_states(_EMPTY_PAYLOAD, _NOW))

    def test_missing_position_kept_without_position(self):
        eth = [s for s in _states() if s.callsign == "ETH501"][0]
        self.assertIsNone(eth.latitude)
        self.assertIsNone(eth.longitude)
        self.assertFalse(eth.has_position)

    def test_short_rows_and_bad_rows_skipped(self):
        payload = {"time": 1794941424,
                   "states": [["abc", "X"], None, "junk",
                              ["4b1814", "BAW249  ", "United Kingdom",
                               1794941400, 1794941400, -20.0, 45.0,
                               10668.0, False, 250.0, 300.0, 0.0,
                               None, 11000.0, "1234", False, 0, 0]]}
        states = parse_opensky_states(payload, _NOW)
        self.assertEqual(1, len(states))
        self.assertEqual("BAW249", states[0].callsign)

    def test_altitude_feet_and_speed_knots(self):
        baw = _baw_state()
        self.assertAlmostEqual(10668.0 * 3.28084, baw.altitude_ft, places=0)
        self.assertAlmostEqual(250.0 * 1.94384, baw.speed_kts, places=0)

    def test_freshness_live_when_recent(self):
        self.assertEqual("live", _baw_state().freshness(_NOW))

    def test_freshness_stale_when_old(self):
        self.assertEqual("stale", _baw_state().freshness(_NOW + 400.0))

    def test_freshness_unknown_without_contact_time(self):
        state = AircraftState(icao24="abc", callsign="TEST1",
                              last_contact=None, fetched_at=_NOW)
        self.assertEqual("unknown", state.freshness(_NOW))

    def test_age_is_now_minus_last_contact(self):
        self.assertAlmostEqual(24.0, _baw_state().age_s(_NOW))


class CallsignTests(unittest.TestCase):
    def test_ba249_resolves_to_baw249(self):
        self.assertEqual(["BAW249"], callsigns_for_flight("BA249"))

    def test_spaced_and_lowercase_flight_numbers(self):
        self.assertEqual(["BAW249"], callsigns_for_flight("ba 249"))

    def test_three_letter_callsign_used_as_given(self):
        self.assertEqual(["BAW249"], callsigns_for_flight("BAW249"))

    def test_rano_air_verified_designator(self):
        self.assertEqual(["RAN312"], callsigns_for_flight("R4312"))

    def test_air_peace_verified_designator(self):
        self.assertEqual(["APK123"], callsigns_for_flight("P4123"))

    def test_unknown_airline_has_no_callsign(self):
        self.assertEqual([], callsigns_for_flight("XX999"))

    def test_icao24_hex_query(self):
        parsed = parse_flight_query("4b1814")
        self.assertEqual("icao24", parsed["kind"])
        self.assertEqual("4b1814", parsed["icao24"])

    def test_table_covers_the_repo_curated_carriers(self):
        self.assertEqual("BAW", AIRLINE_ICAO["BA"])
        self.assertEqual("APK", AIRLINE_ICAO["P4"])
        self.assertEqual("RAN", AIRLINE_ICAO["R4"])
        # VJT is VistaJet, not ValueJet — ValueJet stays unmapped
        # rather than carrying a wrong designator.
        self.assertNotIn("VK", AIRLINE_ICAO)

    def test_find_flight_by_number(self):
        self.assertEqual("BAW249",
                         find_flight(_states(), "BA249").callsign)

    def test_find_flight_by_icao24(self):
        self.assertEqual("BAW249",
                         find_flight(_states(), "4b1814").callsign)

    def test_find_flight_unknown_returns_none(self):
        self.assertIsNone(find_flight(_states(), "XX999"))

    def test_find_flight_matches_callsign_not_substring(self):
        self.assertIsNone(find_flight(_states(), "BAW24"))


class TrackFlightTests(unittest.TestCase):
    def test_live_position_answer(self):
        out = track_flight(_FakeFeed(_states()), "BA249", now=_NOW)
        self.assertTrue(out["ok"])
        self.assertEqual("BAW249", out["callsign"])
        self.assertEqual(45.0, out["latitude"])
        self.assertEqual(-20.0, out["longitude"])
        self.assertEqual("live", out["freshness"])
        self.assertEqual("opensky", out["source"])
        self.assertLessEqual(out["age_s"], 60.0)

    def test_not_found_names_what_was_tried(self):
        out = track_flight(_FakeFeed(_states()), "XX999", now=_NOW)
        self.assertFalse(out["ok"])
        self.assertIn("XX999", out["reason"])

    def test_feed_error_is_a_result_not_an_exception(self):
        feed = _FakeFeed([], meta={"source": "opensky",
                                   "fetched_at": _NOW,
                                   "error": "boom", "cached": False})
        out = track_flight(feed, "BA249", now=_NOW)
        self.assertFalse(out["ok"])
        self.assertIn("boom", out["reason"])

    def test_eta_needs_destination_coords_and_speed(self):
        out = track_flight(_FakeFeed(_states()), "BA249",
                           destination="LHR", now=_NOW)
        self.assertTrue(out["ok"])
        self.assertIsNotNone(out["distance_km"])
        self.assertIsNotNone(out["eta_minutes"])
        self.assertGreater(out["eta_minutes"], 0.0)
        self.assertIsNotNone(out["bearing"])
        self.assertIn(out["on_course"], (True, False))

    def test_eta_unknown_without_destination(self):
        out = track_flight(_FakeFeed(_states()), "BA249", now=_NOW)
        self.assertIsNone(out["eta_minutes"])
        self.assertIsNone(out["distance_km"])

    def test_eta_unknown_for_grounded_flight(self):
        out = track_flight(_FakeFeed(_states()), "APK1234",
                           destination="ABV", now=_NOW)
        self.assertTrue(out["ok"])
        self.assertTrue(out["on_ground"])
        self.assertIsNone(out["eta_minutes"])

    def test_area_lagos_finds_the_grounded_flight(self):
        out = track_flight(_FakeFeed(_states()), "BA249", now=_NOW)
        self.assertTrue(out["ok"])
        states, _meta = _FakeFeed(_states()).fetch(LAGOS_BBOX)
        self.assertEqual(1, len(states))
        self.assertEqual("APK1234", states[0].callsign)


class AreaTests(unittest.TestCase):
    def test_area_filter_keeps_only_states_in_box(self):
        found = aircraft_in_area(_states(), LAGOS_BBOX)
        self.assertEqual(["APK1234"], [s.callsign for s in found])

    def test_area_presets_name_lagos(self):
        self.assertIn("lagos", AREA_PRESETS)
        self.assertEqual(LAGOS_BBOX, AREA_PRESETS["lagos"])

    def test_area_empty_box_is_empty(self):
        found = aircraft_in_area(_states(), (80.0, 80.0, 81.0, 81.0))
        self.assertEqual([], found)


class CoordsTests(unittest.TestCase):
    def test_lagos_coords_from_the_dump(self):
        lat, lon = coords_for_iata("LOS")
        self.assertAlmostEqual(6.57737, lat, places=4)
        self.assertAlmostEqual(3.32116, lon, places=4)

    def test_heathrow_coords_from_the_dump(self):
        lat, lon = coords_for_iata("lhr")
        self.assertAlmostEqual(51.4707, lat, places=3)
        self.assertAlmostEqual(-0.4599, lon, places=3)

    def test_unknown_code_is_none(self):
        self.assertIsNone(coords_for_iata("XXX"))
        self.assertIsNone(coords_for_iata(""))

    def test_airport_info_names_the_airport(self):
        info = airport_info("LOS")
        self.assertEqual("Murtala Muhammed International Airport",
                         info["name"])
        self.assertEqual("Lagos", info["municipality"])
        self.assertEqual("NG", info["country"])

    def test_nearest_airport_to_lagos_position(self):
        near = nearest_airport(6.6, 3.3)
        self.assertEqual("LOS", near["iata"])
        self.assertLess(near["distance_km"], 10.0)

    def test_nearest_airport_far_from_anywhere_is_none(self):
        # Mid-Atlantic: no airport within the 300 km honesty radius.
        self.assertIsNone(nearest_airport(30.0, -40.0))


class GeoTests(unittest.TestCase):
    def test_los_to_lhr_distance(self):
        d = haversine_km(6.57737, 3.32116, 51.470748, -0.459909)
        self.assertGreater(d, 4700.0)
        self.assertLess(d, 5100.0)

    def test_los_to_lhr_bearing_is_north_northwest(self):
        b = bearing_deg(6.57737, 3.32116, 51.470748, -0.459909)
        self.assertGreater(b, 330.0)
        self.assertLess(b, 365.0)

    def test_same_point_is_zero(self):
        self.assertEqual(0.0, haversine_km(6.0, 3.0, 6.0, 3.0))

    def test_eta_minutes_from_speed(self):
        self.assertAlmostEqual(60.0, eta_minutes(900.0, 250.0), places=1)
        self.assertIsNone(eta_minutes(900.0, 0.0))
        self.assertIsNone(eta_minutes(0.0, 250.0))


class TrackStoreTests(unittest.TestCase):
    def test_record_and_read_back(self):
        with tempfile.TemporaryDirectory() as home:
            store = TrackStore(home)
            store.record("BAW249", _baw_state(), now=_NOW)
            store.record("BAW249", _baw_state(now=_NOW + 60.0),
                         now=_NOW + 60.0)
            points = store.points("BAW249")
            self.assertEqual(2, len(points))
            self.assertEqual(45.0, points[0]["latitude"])

    def test_prune_drops_old_points(self):
        with tempfile.TemporaryDirectory() as home:
            store = TrackStore(home)
            store.record("BAW249", _baw_state(), now=_NOW)
            store.record("BAW249", _baw_state(now=_NOW + 90000.0),
                         now=_NOW + 90000.0)
            store.prune(now=_NOW + 90000.0)
            self.assertEqual(1, len(store.points("BAW249")))

    def test_positions_without_coordinates_are_not_recorded(self):
        with tempfile.TemporaryDirectory() as home:
            store = TrackStore(home)
            eth = [s for s in _states() if s.callsign == "ETH501"][0]
            store.record("ETH501", eth, now=_NOW)
            self.assertEqual([], store.points("ETH501"))

    def test_store_survives_reload(self):
        with tempfile.TemporaryDirectory() as home:
            TrackStore(home).record("BAW249", _baw_state(), now=_NOW)
            self.assertEqual(1, len(TrackStore(home).points("BAW249")))


class TrackMapTests(unittest.TestCase):
    def _points(self):
        return [
            {"ts": _NOW, "latitude": 45.0, "longitude": -20.0,
             "altitude_ft": 35000.0},
            {"ts": _NOW + 60.0, "latitude": 46.0, "longitude": -15.0,
             "altitude_ft": 35000.0},
            {"ts": _NOW + 120.0, "latitude": 47.0, "longitude": -10.0,
             "altitude_ft": 35000.0},
        ]

    def test_map_renders_real_png_bytes(self):
        png, digest = render_track_map(
            self._points(), origin="LOS", destination="LHR",
            current=self._points()[-1])
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertGreater(len(png), 1000)
        self.assertEqual(hashlib.sha256(png).hexdigest(), digest)

    def test_map_is_deterministic(self):
        first = render_track_map(self._points(), origin="LOS",
                                 destination="LHR")[1]
        second = render_track_map(self._points(), origin="LOS",
                                  destination="LHR")[1]
        self.assertEqual(first, second)

    def test_map_with_no_points_is_a_blank_honest_map(self):
        png, _digest = render_track_map([])
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_map_with_unknown_airports_names_nothing(self):
        png, _digest = render_track_map(self._points(), origin="XXX",
                                        destination="YYY")
        self.assertTrue(png.startswith(b"\x89PNG"))


class WatcherTests(unittest.TestCase):
    def _watcher(self, home, feed):
        return FlightWatcher(home, feed=feed, now=lambda: _NOW)

    def test_watch_landed_fires_once(self):
        with tempfile.TemporaryDirectory() as home:
            airborne = _FakeFeed([_baw_state()])
            watcher = self._watcher(home, airborne)
            watch = watcher.watch("BA249", destination="LHR")
            self.assertIsNone(watcher.check(watch))
            # Now on the ground near Heathrow.
            landed = AircraftState(
                icao24="4b1814", callsign="BAW249",
                origin_country="United Kingdom",
                latitude=51.47, longitude=-0.46,
                baro_altitude=0.0, on_ground=True, velocity=0.0,
                true_track=0.0, last_contact=_NOW + 60.0,
                fetched_at=_NOW + 60.0)
            watcher.feed = _FakeFeed([landed])
            alert = watcher.check(watch)
            self.assertIsNotNone(alert)
            self.assertEqual("landed", alert["kind"])
            self.assertIn("Heathrow", alert["text"])
            # Second check: no repeat.
            self.assertIsNone(watcher.check(watch))

    def test_watch_late_fires_past_expected_time(self):
        with tempfile.TemporaryDirectory() as home:
            watcher = self._watcher(home, _FakeFeed([_baw_state()]))
            watch = watcher.watch("BA249", destination="LHR",
                                  expect_at=_NOW - 600.0)
            alert = watcher.check(watch)
            self.assertIsNotNone(alert)
            self.assertEqual("late", alert["kind"])

    def test_watch_diverted_when_landing_elsewhere(self):
        with tempfile.TemporaryDirectory() as home:
            watcher = self._watcher(home, _FakeFeed([_baw_state()]))
            watch = watcher.watch("BA249", destination="LHR")
            watcher.check(watch)
            # Lands at Paris CDG instead.
            landed = AircraftState(
                icao24="4b1814", callsign="BAW249",
                origin_country="France",
                latitude=49.0097, longitude=2.5479,
                baro_altitude=0.0, on_ground=True, velocity=0.0,
                true_track=0.0, last_contact=_NOW + 60.0,
                fetched_at=_NOW + 60.0)
            watcher.feed = _FakeFeed([landed])
            alert = watcher.check(watch)
            self.assertIsNotNone(alert)
            self.assertEqual("diverted", alert["kind"])
            self.assertIn("Paris", alert["text"])

    def test_watch_vanished_is_lost_not_landed(self):
        with tempfile.TemporaryDirectory() as home:
            watcher = self._watcher(home, _FakeFeed([_baw_state()]))
            watch = watcher.watch("BA249", destination="LHR")
            watcher.check(watch)
            watcher.feed = _FakeFeed([])
            # One miss is a coverage gap, not an event.
            self.assertIsNone(watcher.check(watch))
            self.assertIsNone(watcher.check(watch))
            alert = watcher.check(watch)
            self.assertIsNotNone(alert)
            self.assertEqual("lost", alert["kind"])

    def test_watch_needs_a_flight(self):
        with tempfile.TemporaryDirectory() as home:
            watcher = self._watcher(home, _FakeFeed([]))
            with self.assertRaises(ValueError):
                watcher.watch("")

    def test_unwatch_and_list(self):
        with tempfile.TemporaryDirectory() as home:
            watcher = self._watcher(home, _FakeFeed([]))
            first = watcher.watch("BA249")
            watcher.watch("APK1234")
            self.assertEqual(2, len(watcher.list_watches()))
            self.assertTrue(watcher.unwatch(first.id))
            self.assertEqual(1, len(watcher.list_watches()))

    def test_tick_checks_all_watches(self):
        with tempfile.TemporaryDirectory() as home:
            watcher = self._watcher(home, _FakeFeed([_baw_state()]))
            watcher.watch("BA249", destination="LHR",
                          expect_at=_NOW - 600.0)
            alerts = watcher.check_all()
            self.assertEqual(1, len(alerts))
            self.assertEqual("late", alerts[0]["kind"])


class FeedTests(unittest.TestCase):
    def test_feed_parses_live_shape(self):
        class _Client:
            def get(self, url, **kw):
                class _Resp:
                    text = json.dumps(_FIXTURE_PAYLOAD)

                    def raise_for_status(self):
                        pass

                return _Resp()

        feed = OpenSkyFeed(http_client=_Client())
        states, meta = feed.fetch()
        self.assertEqual(3, len(states))
        self.assertEqual("opensky", meta["source"])
        self.assertEqual("", meta["error"])

    def test_feed_error_never_raises(self):
        class _Boom:
            def get(self, url, **kw):
                raise RuntimeError("network down")

        feed = OpenSkyFeed(http_client=_Boom())
        states, meta = feed.fetch()
        self.assertEqual([], states)
        self.assertIn("network down", meta["error"])

    def test_feed_snapshot_cached_within_ttl(self):
        class _Counting:
            def __init__(self):
                self.calls = 0

            def get(self, url, **kw):
                self.calls += 1

                class _Resp:
                    text = json.dumps(_FIXTURE_PAYLOAD)

                    def raise_for_status(self):
                        pass

                return _Resp()

        client = _Counting()
        clock = [_NOW]
        feed = OpenSkyFeed(http_client=client, now=lambda: clock[0])
        feed.fetch()
        feed.fetch()
        self.assertEqual(1, client.calls)
        clock[0] = _NOW + 120.0
        feed.fetch()
        self.assertEqual(2, client.calls)


class TickTests(unittest.TestCase):
    def test_tick_quiet_when_no_watches(self):
        with tempfile.TemporaryDirectory() as home:
            out = run_flight_track_tick_once(context=None, now=_NOW,
                                             home=home)
            self.assertTrue(out["ok"])
            self.assertEqual(0, out["checked"])

    def test_tick_alerts_through_notifier(self):
        with tempfile.TemporaryDirectory() as home:
            feed = _FakeFeed([_baw_state()])
            watcher = FlightWatcher(home, feed=feed, now=lambda: _NOW)
            watcher.watch("BA249", destination="LHR",
                          expect_at=_NOW - 600.0)
            seen = []

            class _Notifier:
                def publish(self, kind, title="", body="", **kw):
                    seen.append((kind, title, body))
                    return {"ok": True}

            out = run_flight_track_tick_once(
                context=None, notifier=_Notifier(), feed=feed, now=_NOW,
                home=home)
            self.assertEqual(1, out["alerts"])
            self.assertEqual(1, len(seen))
            self.assertEqual("flight_track", seen[0][0])

    def test_job_registers_every_15_minutes(self):
        self.assertEqual("*/15 * * * *", FLIGHT_TRACK_SPEC)
        self.assertEqual("flight track tick", FLIGHT_TRACK_JOB)
        self.assertEqual("__flight_track_tick__", FLIGHT_TRACK_TOOL)


class RefreshCoordsTests(unittest.TestCase):
    _DUMP = (
        "id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,"
        "continent,iso_country,iso_region,municipality,"
        "scheduled_service,gps_code,iata_code,local_code,home_link,"
        "wikipedia_link,keywords\n"
        "1,AAAA,small_airport,Alpha Field,10.0,20.0,0,AF,US,US-CA,"
        "Alpha,0,,AAA,AAA,,\n"
        "2,BBBB,small_airport,Beta Field,-5.5,100.25,0,AF,NG,NG-LA,"
        "Beta,0,,BBB,BBB,,\n"
        "3,CCCC,small_airport,Bad Lat Field,not-a-number,20.0,0,AF,"
        "NG,NG-LA,Gamma,0,,CCC,CCC,,\n"
        "4,DDDD,small_airport,No Code Field,10.0,20.0,0,AF,NG,NG-LA,"
        "Delta,0,,,,,\n"
        "5,EEEE,small_airport,Conflict Field,11.0,21.0,0,AF,US,US-CA,"
        "Epsilon,0,,AAA,AAA,,\n"
        "6,FFFF,small_airport,Same Field,10.0,20.0,0,AF,US,US-CA,"
        "Zeta,0,,AAA,AAA,,\n"
    )

    def test_build_coords_rules(self):
        from nomorals.travel.refresh import build_coords

        coords, stats = build_coords(self._DUMP)
        self.assertEqual(2, len(coords))
        self.assertEqual((10.0, 20.0), coords["AAA"][:2])
        self.assertEqual(2, stats["dropped"])
        self.assertEqual(6, stats["total_rows"])

    def test_validate_coords_floors(self):
        from nomorals.travel.refresh import build_coords, validate_coords

        coords, stats = build_coords(self._DUMP)
        self.assertNotEqual("", validate_coords(coords, stats))
        big = {f"{a}{b}{c}": (0.0, 0.0, "", "", "")
               for a in "ABCDEFGH" for b in "ABCDEFGH"
               for c in "ABCDEFGH"}
        # 8^3 = 512 codes is still below the floor on purpose: the
        # validation must refuse a dump that lost most of its rows.
        big_stats = {"total_rows": 6000, "dropped": 0,
                     "codes": len(big)}
        self.assertNotEqual("", validate_coords(big, big_stats))
        good = {}
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(6000):
            code = (letters[(i // 676) % 26] + letters[(i // 26) % 26]
                    + letters[i % 26])
            good[code] = (0.0, 0.0, "", "", "")
        good_stats = {"total_rows": 6000, "dropped": 0,
                      "codes": len(good)}
        self.assertEqual("", validate_coords(good, good_stats))

    @staticmethod
    def _big_dump(n_codes: int = 6000) -> str:
        """A well-formed dump with n_codes distinct IATA codes and
        real lat/lon, cycling 120 countries like the real dump."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "ident", "type", "name", "latitude_deg",
                    "longitude_deg", "elevation_ft", "continent",
                    "iso_country", "iso_region", "municipality",
                    "scheduled_service", "gps_code", "iata_code",
                    "local_code", "home_link", "wikipedia_link",
                    "keywords"])
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(n_codes):
            code = (letters[(i // 676) % 26]
                    + letters[(i // 26) % 26] + letters[i % 26])
            place = letters[(i // 26) % 26] + letters[i % 26]
            w.writerow([i, code, "small_airport", f"Field {i}",
                        10.0 + (i % 80), -170.0 + (i % 340), 0, "AF",
                        place, f"{place}-X", f"Town {i}", 0, "",
                        code, code, "", "", ""])
        return buf.getvalue()

    def test_refresh_rebuilds_coords_beside_the_table(self):
        from nomorals.travel.refresh import refresh_airports

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "country.csv"
            coords_path = Path(tmp) / "coords.csv"
            out = refresh_airports(
                fetcher=lambda: self._big_dump(), data_path=data,
                coords_path=coords_path,
                state_path=Path(tmp) / "state.json", now=_NOW)
            self.assertTrue(out["ok"])
            self.assertTrue(coords_path.exists())
            text = coords_path.read_text(encoding="utf-8")
            self.assertIn("AAA,10.0,-170.0,Field 0,Town 0,AA", text)

    def test_refresh_without_coords_path_leaves_coords_alone(self):
        from nomorals.travel.refresh import refresh_airports

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "country.csv"
            coords_path = Path(tmp) / "coords.csv"
            coords_path.write_text("sentinel", encoding="utf-8")
            out = refresh_airports(
                fetcher=lambda: self._big_dump(), data_path=data,
                state_path=Path(tmp) / "state.json", now=_NOW)
            self.assertTrue(out["ok"])
            self.assertEqual("sentinel",
                             coords_path.read_text(encoding="utf-8"))


class FlightRegistrationTests(unittest.TestCase):
    """``/flight`` is registered everywhere, owner-only, and the
    public count does not move (it is not in PUBLIC_COMMANDS)."""

    def test_registered_everywhere(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
            parse_control,
        )

        self.assertIn("flight", CONTROL_COMMANDS)
        self.assertIn("flight", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["flight"].get("what"))
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("flight", grouped)
        self.assertEqual(grouped.count("flight"), 1)
        self.assertIn("/flight <flight>",
                      COMMAND_DETAILS["flight"]["usage"])
        self.assertNotIn("flight", PUBLIC_COMMANDS)
        self.assertEqual(274, len(PUBLIC_COMMANDS))
        self.assertFalse(caller_may_run("flight", False))
        self.assertTrue(caller_may_run("flight", True))
        self.assertIn("/flight", list_catalog())
        cmd = parse_control("/flight BA249")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "flight")
        self.assertEqual(cmd.tail, "BA249")

    def test_menu_still_validates(self):
        from nomorals.social.chat.telegram_menu import validate_menu

        ok, problems = validate_menu()
        self.assertTrue(ok, problems)


class _DecoRegistry:
    """A registry double that captures register() calls."""

    def __init__(self, context: Any = None) -> None:
        self.context = context
        self.registered: dict[str, Any] = {}

    def register(self, name: str, fn: Any = None, **_kw: Any) -> Any:
        def deco(f: Any) -> Any:
            self.registered[name] = f
            return f

        if fn is not None:
            return deco(fn)
        return deco


class FlightToolTests(unittest.TestCase):
    def test_tool_registered_by_builtins(self):
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register_builtins()
        self.assertIn("flight_track", reg._tools)

    def test_track_through_tool_with_fake_feed(self):
        from unittest import mock

        from nomorals.tools.flight_track import register

        with tempfile.TemporaryDirectory() as home:
            ctx = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            deco = _DecoRegistry(context=ctx)
            register(deco)
            self.assertIn("flight_track", deco.registered)
            fn = deco.registered["flight_track"]
            with mock.patch(
                    "nomorals.tools.flight_track._feed",
                    return_value=_FakeFeed([_baw_state()])):
                out = fn(action="track", flight="BA249")
            self.assertTrue(out["ok"])
            self.assertEqual("BAW249", out["callsign"])
            self.assertAlmostEqual(35000, out["altitude_ft"], places=0)

    def test_watch_through_tool(self):
        from nomorals.tools.flight_track import register

        with tempfile.TemporaryDirectory() as home:
            ctx = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            deco = _DecoRegistry(context=ctx)
            register(deco)
            fn = deco.registered["flight_track"]
            out = fn(action="watch", flight="BA249",
                     destination="LHR")
            self.assertTrue(out["ok"])
            self.assertEqual("BA249", out["watch"]["flight"])
            self.assertEqual("LHR", out["watch"]["destination"])
            listed = fn(action="watches")
            self.assertEqual(1, len(listed["watches"]))

    def test_area_and_status_through_tool(self):
        from unittest import mock

        from nomorals.tools.flight_track import register

        with tempfile.TemporaryDirectory() as home:
            ctx = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            deco = _DecoRegistry(context=ctx)
            register(deco)
            fn = deco.registered["flight_track"]
            with mock.patch(
                    "nomorals.tools.flight_track._feed",
                    return_value=_FakeFeed(_states())):
                area = fn(action="area", near="LOS")
                self.assertTrue(area["ok"])
                self.assertEqual(1, area["count"])
                self.assertEqual("APK1234",
                                 area["aircraft"][0]["callsign"])
                status = fn(action="status")
                self.assertTrue(status["ok"])
                self.assertEqual("", status["meta"]["error"])


class FlightChatTests(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from tests.test_gapfix_museum_command import _msg

        scratch = Path.home() / "workspace" / "flightradar-verify"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-flight-chat-", dir=str(scratch))
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(
            settings, with_executor=False, with_router=False,
            with_tools=True)
        self.context.db.migrate()

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.runtime = PartnerRuntime(self.context, gateway=None)
        self._msg = _msg

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str, *, owner: bool = True,
             chat_key: str = "telegram:999") -> str:
        return self.runtime.handle_control(
            text, chat_key, message=self._msg(
                text, chat_key=chat_key, owner=owner))

    def test_track_reports_live_position_and_draws_map(self):
        from unittest import mock

        from nomorals.agents.partner.runtime import PartnerRuntime

        with mock.patch(
                "nomorals.travel.tracking.OpenSkyFeed",
                return_value=_FakeFeed([_baw_state()])), \
                mock.patch.object(PartnerRuntime, "_scene_send_file",
                                  return_value=None):
            reply = self._run("/flight BA249")
        self.assertIn("BAW249", reply)
        self.assertIn("35000", reply)
        self.assertIn("s old", reply)
        self.assertIn("map sent.", reply)
        maps = list(Path(self.tmp.name).rglob("track_*.png"))
        self.assertEqual(1, len(maps))
        self.assertGreater(maps[0].stat().st_size, 1000)

    def test_track_not_found_is_honest(self):
        from unittest import mock

        with mock.patch(
                "nomorals.travel.tracking.OpenSkyFeed",
                return_value=_FakeFeed([])):
            reply = self._run("/flight XX999")
        self.assertIn("no live position", reply)
        self.assertIn("XX999", reply)

    def test_watch_verbs_round_trip(self):
        reply = self._run("/flight watch BA249 LHR")
        self.assertIn("watching BA249", reply)
        self.assertIn("LHR", reply)
        reply = self._run("/flight watches")
        self.assertIn("BA249", reply)
        reply = self._run("/flight unwatch BA249")
        self.assertIn("no longer watching", reply)
        self.assertIn("no flights watched", self._run("/flight watches"))

    def test_area_lists_aircraft_over_airport(self):
        from unittest import mock

        with mock.patch(
                "nomorals.travel.tracking.OpenSkyFeed",
                return_value=_FakeFeed(_states())):
            reply = self._run("/flight area LOS")
        self.assertIn("APK1234", reply)
        self.assertIn("live aircraft", reply)


class FlightCliTests(unittest.TestCase):
    def _ctx(self, home: str) -> Any:
        return SimpleNamespace(
            settings=SimpleNamespace(home=home, home_path=home))

    def _args(self, action: str, rest: list[str] | None = None,
              as_json: bool = False) -> Any:
        return SimpleNamespace(flight_action=action,
                               flight_rest=list(rest or []),
                               json=as_json)

    def test_watch_watches_unwatch(self):
        from nomorals.cmdline.commands.flight import _cmd_flight

        with tempfile.TemporaryDirectory() as home:
            ctx = self._ctx(home)
            self.assertEqual(
                0, _cmd_flight(self._args("watch", ["BA249", "LHR"]),
                               ctx))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(
                    0, _cmd_flight(self._args("watches"), ctx))
            self.assertIn("BA249", out.getvalue())
            self.assertEqual(
                0, _cmd_flight(self._args("unwatch", ["BA249"]), ctx))

    def test_track_with_fake_feed(self):
        from unittest import mock

        from nomorals.cmdline.commands.flight import _cmd_flight

        with tempfile.TemporaryDirectory() as home:
            ctx = self._ctx(home)
            out = io.StringIO()
            with mock.patch(
                    "nomorals.travel.tracking.OpenSkyFeed",
                    return_value=_FakeFeed([_baw_state()])), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(
                    0, _cmd_flight(self._args("BA249"), ctx))
            text = out.getvalue()
            self.assertIn("BAW249", text)
            self.assertIn("35000", text)
            self.assertIn("map:", text)

    def test_tick_with_no_watches_is_quiet(self):
        from nomorals.cmdline.commands.flight import _cmd_flight

        with tempfile.TemporaryDirectory() as home:
            ctx = self._ctx(home)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(0, _cmd_flight(self._args("tick"), ctx))
            self.assertIn("0 watched", out.getvalue())


class FlightTickToolTests(unittest.TestCase):
    def test_tick_tool_registered_and_never_raises(self):
        from nomorals.tools.autonomy import register

        deco = _DecoRegistry(context=None)
        register(deco)
        self.assertIn("__flight_track_tick__", deco.registered)
        out = deco.registered["__flight_track_tick__"]()
        self.assertTrue(out["ok"])
        self.assertEqual(0, out["checked"])

    def test_job_registered_every_15m(self):
        self.assertEqual("flight track tick", FLIGHT_TRACK_JOB)
        self.assertEqual("*/15 * * * *", FLIGHT_TRACK_SPEC)
        self.assertEqual("__flight_track_tick__", FLIGHT_TRACK_TOOL)

        class _Sched:
            def __init__(self) -> None:
                self.jobs: list[dict[str, Any]] = []

            def list_jobs(self) -> list[dict[str, Any]]:
                return list(self.jobs)

            def add(self, name: str, spec: str, kind: str,
                    payload: dict[str, Any], **_kw: Any) -> dict[str, Any]:
                job = {"name": name, "spec": spec, "kind": kind,
                       "payload": payload}
                self.jobs.append(job)
                return job

            def remove(self, job_id: Any) -> None:
                self.jobs = [j for j in self.jobs
                             if j.get("id") != job_id]

        sched = _Sched()
        self.assertIsNone(ensure_flight_track_job(None))
        first = ensure_flight_track_job(sched)
        self.assertEqual(FLIGHT_TRACK_JOB, first["name"])
        self.assertEqual(FLIGHT_TRACK_SPEC, first["spec"])
        self.assertEqual({"tool": FLIGHT_TRACK_TOOL, "args": {}},
                         first["payload"])
        second = ensure_flight_track_job(sched)
        self.assertIs(second, first)
        self.assertEqual(1, len(sched.jobs))
        third = ensure_flight_track_job(sched, spec="*/5 * * * *")
        self.assertIsNot(third, first)
        self.assertEqual(1, len(sched.jobs))
        self.assertEqual("*/5 * * * *", third["spec"])

    def test_boot_ensures_flight_track_job(self):
        import inspect

        from nomorals.agents.partner import runtime as _rt

        src = inspect.getsource(_rt.PartnerRuntime.start)
        self.assertIn("ensure_flight_track_job", src)


if __name__ == "__main__":
    unittest.main()
