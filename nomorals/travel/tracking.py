"""Live flight tracking — the FlightRadar24 organ (52-websites #21).

Devon's flight knowledge was trip-shaped (airport table, fare
watchers, itineraries) — nothing answered "where is flight BA249
right now?" from live data. This organ is the live picture:

* ``AircraftState`` — one ADS-B state, normalised from the feed.
* ``OpenSkyFeed`` — the live feed: the OpenSky Network public
  states API (https://opensky-network.org/api/states/all), keyless
  and anonymous (the anonymous tier is rate-limited — the feed
  caches each snapshot for 60 s so a burst of questions costs one
  call, and every answer names its source and its age). A dead
  network is a result (``error`` in the meta), never an exception.
* Flight-number → callsign resolution against ``AIRLINE_ICAO`` —
  the IATA prefix → ICAO designator table, verified live against
  the FlightAware operator read on 2026-11-17 (BA→BAW, P4→APK,
  R4→RAN, OF→OLA, …). An airline the table does not know has no
  callsign guessed for it: the answer says so and asks for the
  callsign (``BAW249``) or the hex (``4b1814``) instead.
* Airport coordinates from ``data/airports_coords.csv`` — generated
  from the same OurAirports dump as the country table (86,371 rows
  → 9,053 IATA codes, 0 dropped), rebuilt by the monthly refresh
  beside the country table (``travel/refresh.py``).
* ``TrackStore`` — the durable per-flight position journal; the
  track map is drawn from it, never from a single fix.
* ``render_track_map`` — the moving map she draws herself: PIL,
  equirectangular, graticule, origin/destination, the journalled
  track polyline, the current position. Deterministic
  (``render_hash`` = SHA-256 of the PNG bytes).
* ``FlightWatcher`` — "tell me when it lands": persistent watches
  under the fare watchers' alert discipline (one alert per event,
  never spam). Events the live feed can honestly judge: landed
  (on the ground near an airport — named), late (still airborne
  past the expected time the OWNER gave — the feed carries no
  schedule, so no delay is ever invented), diverted (landed at a
  different airport than the expected destination), lost (gone
  from the feed for three straight checks after being seen
  airborne — a coverage gap, reported as one, never as a landing).

Honesty rules, throughout: every answer names its source
(``opensky``) and the age of the position; a position older than
120 s is ``stale``, never presented as live; a flight with no live
state is "not on the live feed right now", never a guessed
position; coverage is whatever the public aggregators carry
(notably thin over oceans and parts of Africa).
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "AIRLINE_ICAO",
    "AIRPORTS_COORDS_PATH",
    "AREA_PRESETS",
    "FLIGHT_TRACK_JOB",
    "FLIGHT_TRACK_SPEC",
    "FLIGHT_TRACK_TOOL",
    "LAGOS_BBOX",
    "OPENSKY_URL",
    "AircraftState",
    "FlightWatch",
    "FlightWatcher",
    "OpenSkyFeed",
    "TrackStore",
    "airport_info",
    "aircraft_in_area",
    "bearing_deg",
    "callsigns_for_flight",
    "coords_for_iata",
    "ensure_flight_track_job",
    "eta_minutes",
    "find_flight",
    "haversine_km",
    "invalidate_coords_cache",
    "nearest_airport",
    "parse_flight_query",
    "parse_opensky_states",
    "render_track_map",
    "run_flight_track_tick_once",
    "track_flight",
]

#: The live feed — OpenSky Network public states API. Keyless,
#: anonymous; the anonymous tier is rate-limited, so the feed caches.
OPENSKY_URL = "https://opensky-network.org/api/states/all"

#: Snapshot cache: one live call serves every question inside this
#: window (OpenSky itself refreshes ~every 10 s; 60 s keeps the
#: answer fresh while respecting the anonymous tier).
SNAPSHOT_TTL_S = 60.0

#: A position this old is still presented as live; older is stale.
LIVE_MAX_AGE_S = 120.0

#: HTTP ceiling for one feed call.
OPENSKY_TIMEOUT_S = 20.0

#: Misses in a row before a vanished flight is "lost" (one miss is a
#: coverage gap, not an event).
LOST_AFTER_MISSES = 3

#: "Landed near an airport" radius, and the radius inside which the
#: nearest airport is named at all.
LANDED_NEAR_KM = 60.0
NEAREST_AIRPORT_MAX_KM = 300.0

#: The flight-track tick: every 15 minutes, but only while a watch
#: is active (a quiet tick makes no network call at all).
FLIGHT_TRACK_JOB = "flight track tick"
FLIGHT_TRACK_SPEC = "*/15 * * * *"
FLIGHT_TRACK_TOOL = "__flight_track_tick__"

#: Lagos area preset — the "any aircraft over Lagos?" box.
LAGOS_BBOX = (6.0, 3.0, 7.0, 4.0)
AREA_PRESETS = {"lagos": LAGOS_BBOX}

#: IATA flight-number prefix → ICAO designator. Every row was
#: verified live against the FlightAware operator read on
#: 2026-11-17 (``flightaware operator <icao>`` returns the IATA).
#: ValueJet (VK) is deliberately absent: VJT is VistaJet, and no
#: verified designator for ValueJet was found — the resolver says
#: so instead of carrying a wrong code. Overland is OF (the repo's
#: itinerary table says OJ — FlightAware says OF; the discrepancy
#: is recorded, the itinerary table is not this lane's to fix).
AIRLINE_ICAO = {
    # Nigeria
    "P4": "APK", "W3": "ARA", "9J": "DAN", "QI": "IAN",
    "VM": "NGL", "OF": "OLA", "Q9": "GWG", "R4": "RAN",
    # Africa / Middle East / Europe / Americas majors
    "BA": "BAW", "VS": "VIR", "AF": "AFR", "KL": "KLM",
    "LH": "DLH", "EK": "UAE", "QR": "QTR", "ET": "ETH",
    "TK": "THY", "MS": "MSR", "AT": "RAM", "SA": "SAA",
    "KQ": "KQA", "DL": "DAL", "UA": "UAL", "AA": "AAL",
    "WN": "SWA", "B6": "JBU", "AC": "ACA",
}

#: The airport coordinates table, next to this module (same pattern
#: as ``travel/airports.py`` — generated from the OurAirports dump,
#: gitignored by the bare ``data/`` rule, committed with ``git add
#: -f`` like the country table).
AIRPORTS_COORDS_PATH = (
    Path(__file__).parent / "data" / "airports_coords.csv")

_FLIGHT_RE = re.compile(r"^([A-Z0-9]{2})(\d{1,4})$")
_CALLSIGN_RE = re.compile(r"^([A-Z]{3})(\d{1,4})$")
_ICAO24_RE = re.compile(r"^[0-9A-F]{6}$")

_EARTH_RADIUS_KM = 6371.0


# ── the state ────────────────────────────────────────────────────────

@dataclass
class AircraftState:
    """One normalised ADS-B state, from whichever feed carried it."""

    icao24: str = ""
    callsign: str = ""
    origin_country: str = ""
    latitude: float | None = None
    longitude: float | None = None
    baro_altitude: float | None = None
    geo_altitude: float | None = None
    on_ground: bool = False
    velocity: float | None = None
    true_track: float | None = None
    vertical_rate: float | None = None
    squawk: str = ""
    last_contact: float | None = None
    time_position: float | None = None
    source: str = "opensky"
    fetched_at: float = 0.0

    @property
    def has_position(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def altitude_ft(self) -> float | None:
        """Barometric altitude in feet, else geometric, else None."""
        metres = self.baro_altitude
        if metres is None:
            metres = self.geo_altitude
        if metres is None:
            return None
        return metres * 3.28084

    @property
    def speed_kts(self) -> float | None:
        if self.velocity is None:
            return None
        return self.velocity * 1.94384

    def age_s(self, now: float | None = None) -> float | None:
        """Seconds since the last contact, or None when unknown."""
        if self.last_contact is None:
            return None
        return max(0.0, float(now if now is not None else time.time())
                   - self.last_contact)

    def freshness(self, now: float | None = None) -> str:
        """``live`` | ``stale`` | ``unknown`` — never faked."""
        age = self.age_s(now)
        if age is None:
            return "unknown"
        return "live" if age <= LIVE_MAX_AGE_S else "stale"

    def to_dict(self) -> dict[str, Any]:
        return {
            "icao24": self.icao24, "callsign": self.callsign,
            "origin_country": self.origin_country,
            "latitude": self.latitude, "longitude": self.longitude,
            "baro_altitude": self.baro_altitude,
            "geo_altitude": self.geo_altitude,
            "on_ground": self.on_ground, "velocity": self.velocity,
            "true_track": self.true_track,
            "vertical_rate": self.vertical_rate,
            "squawk": self.squawk,
            "last_contact": self.last_contact,
            "time_position": self.time_position,
            "source": self.source, "fetched_at": self.fetched_at,
        }


def parse_opensky_states(
        payload: Any, fetched_at: float,
        source: str = "opensky") -> list[AircraftState]:
    """Normalise an OpenSky ``/states/all`` payload. Pure.

    ``states`` is null (not []) when nothing is airborne in the
    box — that is an empty list, not an error. Short rows, non-list
    rows and rows with no callsign are skipped; a state without a
    position is kept (it can still answer "on the ground", just
    without coordinates).
    """
    states: list[AircraftState] = []
    if not isinstance(payload, dict):
        return states
    raw = payload.get("states")
    if not raw:
        return states
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 17:
            continue
        callsign = str(row[1] or "").strip().upper()
        if not callsign:
            continue

        def _num(value: Any) -> float | None:
            if isinstance(value, bool) or value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        states.append(AircraftState(
            icao24=str(row[0] or "").strip().lower(),
            callsign=callsign,
            origin_country=str(row[2] or ""),
            time_position=_num(row[3]),
            last_contact=_num(row[4]),
            longitude=_num(row[5]),
            latitude=_num(row[6]),
            baro_altitude=_num(row[7]),
            on_ground=bool(row[8]),
            velocity=_num(row[9]),
            true_track=_num(row[10]),
            vertical_rate=_num(row[11]),
            geo_altitude=_num(row[13]),
            squawk=str(row[14] or ""),
            source=source,
            fetched_at=float(fetched_at),
        ))
    return states


# ── flight-number resolution ─────────────────────────────────────────

def parse_flight_query(query: Any) -> dict[str, Any]:
    """Classify a tracking query. Never raises, never guesses.

    ``BA249`` → flight (IATA + number); ``BAW249`` → callsign;
    ``4b1814`` → icao24 hex. Anything else is ``unknown`` with the
    reason named, so the caller can report it honestly.
    """
    raw = (query or "")
    text = str(raw).strip().upper().replace(" ", "")
    if not text:
        return {"kind": "unknown", "flight": "", "callsigns": [],
                "icao24": "", "reason": "no flight given"}
    if _CALLSIGN_RE.match(text):
        return {"kind": "callsign", "flight": text,
                "callsigns": [text], "icao24": "", "reason": ""}
    match = _FLIGHT_RE.match(text)
    if match and match.group(1) in AIRLINE_ICAO:
        return {"kind": "flight", "flight": text,
                "callsigns": [AIRLINE_ICAO[match.group(1)]
                              + match.group(2)],
                "icao24": "", "reason": ""}
    if _ICAO24_RE.match(text):
        # A 6-hex string is an aircraft id — unless it parsed as a
        # flight with a known airline above (BA1234 is a flight;
        # 4B1814 is an id: "4B" is no airline).
        return {"kind": "icao24", "flight": text, "callsigns": [],
                "icao24": text.lower(), "reason": ""}
    if match:
        return {"kind": "flight", "flight": text, "callsigns": [],
                "icao24": "",
                "reason": (f"airline {match.group(1)!r} has no "
                           "verified ICAO designator — give the "
                           "callsign (e.g. BAW249) or the hex "
                           "(e.g. 4b1814) instead")}
    return {"kind": "unknown", "flight": text, "callsigns": [],
            "icao24": "",
            "reason": (f"{text!r} is not a flight number "
                       "(e.g. BA249), a callsign (e.g. BAW249) or "
                       "a 6-hex aircraft id (e.g. 4b1814)")}


def callsigns_for_flight(flight: Any) -> list[str]:
    """Flight number → candidate callsigns (usually one).

    ``BA249`` → ``["BAW249"]``; a callsign passed as-is is returned
    as-is; an airline the table does not know yields [] — never a
    guessed designator.
    """
    text = str(flight or "").strip().upper().replace(" ", "")
    if _CALLSIGN_RE.match(text):
        return [text]
    match = _FLIGHT_RE.match(text)
    if not match:
        return []
    designator = AIRLINE_ICAO.get(match.group(1))
    if designator is None:
        return []
    return [designator + match.group(2)]


def find_flight(states: list[AircraftState],
                query: Any) -> AircraftState | None:
    """The live state for a query, or None. Exact match only — a
    callsign is never matched by substring (``BAW24`` is not
    ``BAW249``)."""
    parsed = parse_flight_query(query)
    if parsed["kind"] == "icao24":
        for state in states:
            if state.icao24 == parsed["icao24"]:
                return state
        return None
    for wanted in parsed["callsigns"]:
        for state in states:
            if state.callsign == wanted:
                return state
    return None


# ── the feed ─────────────────────────────────────────────────────────

class OpenSkyFeed:
    """The OpenSky public states feed. Never raises.

    ``fetch`` returns ``(states, meta)``; ``meta`` carries
    ``source`` / ``fetched_at`` / ``error`` / ``cached``. A dead
    network, a bad status or an unreadable body is an empty list
    with the error named — the caller reports it, the feed never
    throws. Snapshots are cached per box for ``SNAPSHOT_TTL_S``.
    """

    def __init__(self, http_client: Any = None, *,
                 cache_ttl_s: float = SNAPSHOT_TTL_S,
                 now: Callable[[], float] | None = None) -> None:
        self._client = http_client
        self._ttl = float(cache_ttl_s)
        self._now = now or time.time
        self._cache: dict[str, tuple[float, list[AircraftState]]] = {}

    def _http(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = HttpClient(timeout=OPENSKY_TIMEOUT_S)
        return self._client

    def fetch(self, bbox: tuple[float, float, float, float] | None = None
              ) -> tuple[list[AircraftState], dict[str, Any]]:
        now = float(self._now())
        key = "all" if bbox is None else (
            f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}")
        hit = self._cache.get(key)
        if hit is not None and now - hit[0] <= self._ttl:
            return list(hit[1]), {"source": "opensky",
                                  "fetched_at": hit[0], "error": "",
                                  "cached": True}
        url = OPENSKY_URL
        if bbox is not None:
            lamin, lomin, lamax, lomax = bbox
            url = (f"{url}?lamin={lamin}&lomin={lomin}"
                   f"&lamax={lamax}&lomax={lomax}")
        try:
            response = self._http().get(url)
            response.raise_for_status()
            payload = json.loads(response.text)
        except Exception as exc:  # noqa: BLE001 - feed failure is data
            _log.debug("opensky fetch failed", exc_info=True)
            return [], {"source": "opensky", "fetched_at": now,
                        "error": f"live feed unreachable: {exc}",
                        "cached": False}
        states = parse_opensky_states(payload, now)
        self._cache[key] = (now, states)
        return list(states), {"source": "opensky", "fetched_at": now,
                              "error": "", "cached": False}


def aircraft_in_area(
        states: list[AircraftState],
        bbox: tuple[float, float, float, float]) -> list[AircraftState]:
    """States with a position inside (lamin, lomin, lamax, lomax)."""
    lamin, lomin, lamax, lomax = bbox
    return [s for s in states
            if s.has_position
            and lamin <= s.latitude <= lamax
            and lomin <= s.longitude <= lomax]


# ── airport coordinates ──────────────────────────────────────────────

_coords_cache: dict[str, tuple[float, float, str, str, str]] | None = None


def _load_coords() -> dict[str, tuple[float, float, str, str, str]]:
    """IATA → (lat, lon, name, municipality, country). Empty on any
    failure — unknown airports degrade to 'unknown', never raise."""
    global _coords_cache
    if _coords_cache is not None:
        return _coords_cache
    table: dict[str, tuple[float, float, str, str, str]] = {}
    try:
        with open(AIRPORTS_COORDS_PATH, newline="",
                  encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                code = (row.get("iata") or "").strip().upper()
                try:
                    lat = float(row.get("latitude") or "")
                    lon = float(row.get("longitude") or "")
                except (TypeError, ValueError):
                    continue
                if len(code) != 3 or not code.isalpha():
                    continue
                table.setdefault(code, (
                    lat, lon,
                    (row.get("name") or "").strip(),
                    (row.get("municipality") or "").strip(),
                    (row.get("iso_country") or "").strip().upper()))
    except Exception:  # noqa: BLE001 - data problems degrade, never raise
        _log.debug("airport coordinates unavailable", exc_info=True)
    _coords_cache = table
    return table


def invalidate_coords_cache() -> None:
    """Drop the cached coordinates table (called by the refresh
    right after an atomic replace)."""
    global _coords_cache
    _coords_cache = None


def coords_for_iata(iata: Any) -> tuple[float, float] | None:
    """IATA code → (latitude, longitude), or None when the table
    does not know the code. Never raises, never guesses."""
    if not isinstance(iata, str):
        return None
    row = _load_coords().get(iata.strip().upper())
    if row is None:
        return None
    return row[0], row[1]


def airport_info(iata: Any) -> dict[str, Any] | None:
    """Full descriptor for an IATA code, or None when unknown."""
    if not isinstance(iata, str):
        return None
    row = _load_coords().get(iata.strip().upper())
    if row is None:
        return None
    lat, lon, name, city, country = row
    return {"iata": iata.strip().upper(), "latitude": lat,
            "longitude": lon, "name": name, "municipality": city,
            "country": country}


def nearest_airport(latitude: float, longitude: float, *,
                    max_km: float = NEAREST_AIRPORT_MAX_KM
                    ) -> dict[str, Any] | None:
    """Nearest airport to a position, or None when none is within
    ``max_km`` — a landing far from any known airport is reported
    as a position, never forced onto the nearest field."""
    best: dict[str, Any] | None = None
    best_d = float("inf")
    for code, (lat, lon, name, city, country) in (
            _load_coords().items()):
        dist = haversine_km(latitude, longitude, lat, lon)
        if dist < best_d:
            best_d = dist
            best = {"iata": code, "latitude": lat, "longitude": lon,
                    "name": name, "municipality": city,
                    "country": country, "distance_km": dist}
    if best is None or best_d > max_km:
        return None
    return best


# ── geography ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(dlambda / 2) ** 2)
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """Initial bearing in degrees (0 = north, clockwise)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def eta_minutes(distance_km: float, speed_ms: float | None) -> float | None:
    """Minutes to cover a distance at a ground speed, or None when
    the speed is unknown or zero — an ETA is never invented."""
    if speed_ms is None or speed_ms <= 0 or distance_km <= 0:
        return None
    return distance_km * 1000.0 / speed_ms / 60.0


# ── the live answer ──────────────────────────────────────────────────

def track_flight(feed: Any, query: Any, *,
                 destination: Any = None,
                 now: float | None = None,
                 store: Any = None) -> dict[str, Any]:
    """Where a flight is right now — the live answer, with its age.

    Returns one JSON-plain dict: ``ok`` + the state (callsign,
    position, altitude, speed, heading, on_ground, age, freshness,
    source) + the destination leg (distance, bearing, ETA, on-course)
    when the destination's coordinates are known and the aircraft
    is moving. ``ok: False`` carries the reason — an unknown airline,
    a flight with no live state, a dead feed — never a guessed
    position. A successful lookup journals the fix into ``store``
    when one is given (the track map is drawn from the journal).
    """
    now_f = float(now if now is not None else time.time())
    parsed = parse_flight_query(query)
    if parsed["kind"] == "unknown":
        return {"ok": False, "found": False,
                "flight": str(query or ""),
                "reason": parsed["reason"],
                "source": "opensky", "freshness": "unknown",
                "age_s": None, "tried": []}
    states, meta = feed.fetch()
    if meta.get("error"):
        return {"ok": False, "found": False,
                "flight": parsed["flight"],
                "reason": meta["error"],
                "source": "opensky", "freshness": "unknown",
                "age_s": None,
                "tried": parsed["callsigns"] or [parsed["icao24"]]}
    state = find_flight(states, query)
    if state is None:
        tried = parsed["callsigns"] or [parsed["icao24"]]
        return {"ok": False, "found": False,
                "flight": parsed["flight"],
                "reason": (f"no live position for "
                           f"{parsed['flight']} right now "
                           f"(looked for {', '.join(tried)} on the "
                           f"live feed — the flight may not be "
                           f"airborne, or the feed may not cover it)"),
                "source": "opensky", "freshness": "unknown",
                "age_s": None, "tried": tried}
    if store is not None and state.has_position:
        with contextlib.suppress(Exception):
            store.record(state.callsign, state, now=now_f)
    out: dict[str, Any] = {
        "ok": True, "found": True, "flight": parsed["flight"],
        "callsign": state.callsign, "icao24": state.icao24,
        "origin_country": state.origin_country,
        "latitude": state.latitude, "longitude": state.longitude,
        "altitude_ft": state.altitude_ft,
        "speed_kts": state.speed_kts,
        "heading": state.true_track,
        "vertical_rate": state.vertical_rate,
        "on_ground": state.on_ground,
        "squawk": state.squawk,
        "age_s": state.age_s(now_f),
        "freshness": state.freshness(now_f),
        "source": "opensky",
        "destination": None, "distance_km": None,
        "eta_minutes": None, "bearing": None, "on_course": None,
    }
    dest = (destination or "")
    dest = str(dest).strip().upper() if isinstance(dest, str) else ""
    if dest and state.has_position:
        coords = coords_for_iata(dest)
        if coords is not None:
            dist = haversine_km(state.latitude, state.longitude,
                                coords[0], coords[1])
            bearing = bearing_deg(state.latitude, state.longitude,
                                  coords[0], coords[1])
            out["destination"] = dest
            out["distance_km"] = round(dist, 1)
            out["bearing"] = round(bearing, 1)
            if not state.on_ground and state.velocity:
                out["eta_minutes"] = eta_minutes(dist, state.velocity)
                if state.true_track is not None:
                    diff = abs((state.true_track - bearing + 180.0)
                               % 360.0 - 180.0)
                    out["on_course"] = diff <= 45.0
    return out


# ── the track journal ────────────────────────────────────────────────

def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = (getattr(settings, "home_path", None)
            or getattr(settings, "home", None))
    return Path(home) if home else None


def _resolve_home(home: Any, context: Any) -> Path:
    if home is not None:
        return Path(home)
    found = _home_path(context)
    if found is not None:
        return found
    return Path("~/.nomorals").expanduser()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no store is an empty store
        return {}
    return raw if isinstance(raw, dict) else {}


class TrackStore:
    """Durable per-flight position journal.

    ``<home>/data/flight_tracks.json`` — one entry per callsign,
    points oldest-first, pruned past 24 h / 500 points. The track
    map and the watcher's "last seen" both read from here, so a
    restart never loses the track. Positions without coordinates
    are never recorded (there is nothing to plot).
    """

    _NAME = "flight_tracks.json"
    _MAX_AGE_S = 24 * 3600
    _MAX_POINTS = 500

    def __init__(self, home: Any) -> None:
        self._path = Path(home) / "data" / self._NAME
        self._data = _read_json(self._path)

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            _atomic_json_write(self._path, self._data)

    def record(self, flight: str, state: AircraftState,
               *, now: float | None = None) -> None:
        if not state.has_position:
            return
        now_f = float(now if now is not None else time.time())
        flights = self._data.setdefault("flights", {})
        entry = flights.setdefault(str(flight).upper(), {"points": []})
        entry["points"].append({
            "ts": now_f, "latitude": state.latitude,
            "longitude": state.longitude,
            "altitude_ft": state.altitude_ft,
            "speed_kts": state.speed_kts,
            "heading": state.true_track,
        })
        self.prune(now=now_f)
        self._save()

    def points(self, flight: str) -> list[dict[str, Any]]:
        flights = self._data.get("flights") or {}
        entry = flights.get(str(flight).upper()) or {}
        return list(entry.get("points") or [])

    def prune(self, *, now: float | None = None) -> None:
        now_f = float(now if now is not None else time.time())
        cutoff = now_f - self._MAX_AGE_S
        flights = self._data.get("flights") or {}
        for entry in flights.values():
            pts = [p for p in (entry.get("points") or [])
                   if float(p.get("ts") or 0.0) >= cutoff]
            entry["points"] = pts[-self._MAX_POINTS:]


# ── the track map ────────────────────────────────────────────────────

def render_track_map(
        points: list[dict[str, Any]], *,
        origin: Any = None, destination: Any = None,
        current: dict[str, Any] | None = None,
        size: tuple[int, int] = (960, 600)) -> tuple[bytes, str]:
    """Draw the moving map: the journalled track on a graticule.

    Equirectangular projection with a cos(latitude) correction, so
    the shape is honest at any latitude. Origin/destination are
    IATA codes — drawn only when the coordinates table knows them.
    Returns ``(png_bytes, sha256_hex)``; the same input renders the
    same bytes (deterministic — no clock, no randomness). An empty
    track is a blank graticule with the fact written on it, never a
    faked route.
    """
    from PIL import Image, ImageDraw

    width, height = size
    img = Image.new("RGB", (width, height), (250, 250, 245))
    draw = ImageDraw.Draw(img)
    margin = 60
    origin_code = (str(origin).strip().upper()
                   if isinstance(origin, str) and str(origin).strip()
                   else None)
    destination_code = (str(destination).strip().upper()
                        if isinstance(destination, str)
                        and str(destination).strip() else None)
    lats = [float(p["latitude"]) for p in points
            if p.get("latitude") is not None]
    lons = [float(p["longitude"]) for p in points
            if p.get("longitude") is not None]
    marks: dict[str, tuple[float, float]] = {}
    for code in (origin_code, destination_code):
        if code:
            coords = coords_for_iata(code)
            if coords is not None:
                marks[code] = coords
                lats.append(coords[0])
                lons.append(coords[1])
    if current is not None:
        if current.get("latitude") is not None:
            lats.append(float(current["latitude"]))
        if current.get("longitude") is not None:
            lons.append(float(current["longitude"]))
    if lats and lons:
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
    else:
        lat_min, lat_max, lon_min, lon_max = 0.0, 10.0, 0.0, 10.0
    lat_pad = max(2.0, (lat_max - lat_min) * 0.25)
    lon_pad = max(2.0, (lon_max - lon_min) * 0.25)
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad
    mean_lat = math.radians((lat_min + lat_max) / 2.0)
    x_scale = (width - 2 * margin) / max(
        0.001, (lon_max - lon_min) * max(0.2, math.cos(mean_lat)))
    y_scale = (height - 2 * margin) / max(0.001, lat_max - lat_min)
    scale = min(x_scale, y_scale)
    x_off = (width - (lon_max - lon_min)
             * max(0.2, math.cos(mean_lat)) * scale) / 2.0
    y_off = (height - (lat_max - lat_min) * scale) / 2.0

    def _xy(lat: float, lon: float) -> tuple[float, float]:
        x = x_off + (lon - lon_min) * max(0.2, math.cos(mean_lat)) * scale
        y = y_off + (lat_max - lat) * scale
        return x, y

    # Graticule every 5 degrees, labelled on the edges.
    lat = math.ceil(lat_min / 5.0) * 5.0
    while lat <= lat_max:
        x1, y = _xy(lat, lon_min)
        x2, _ = _xy(lat, lon_max)
        draw.line([(x1, y), (x2, y)], fill=(220, 220, 215), width=1)
        draw.text((6, y - 6), f"{lat:.0f}°", fill=(140, 140, 130))
        lat += 5.0
    lon = math.ceil(lon_min / 5.0) * 5.0
    while lon <= lon_max:
        x, y1 = _xy(lat_min, lon)
        _, y2 = _xy(lat_max, lon)
        draw.line([(x, y1), (x, y2)], fill=(220, 220, 215), width=1)
        draw.text((x - 8, height - 16), f"{lon:.0f}°",
                  fill=(140, 140, 130))
        lon += 5.0
    # Route: origin → destination, dashed, when both are known.
    if origin_code in marks and destination_code in marks:
        x1, y1 = _xy(*marks[origin_code])
        x2, y2 = _xy(*marks[destination_code])
        steps = 24
        for i in range(steps):
            if i % 2:
                continue
            ax = x1 + (x2 - x1) * i / steps
            ay = y1 + (y2 - y1) * i / steps
            bx = x1 + (x2 - x1) * (i + 1) / steps
            by = y1 + (y2 - y1) * (i + 1) / steps
            draw.line([(ax, ay), (bx, by)], fill=(150, 150, 145),
                      width=2)
    # The journalled track.
    track = [p for p in points if p.get("latitude") is not None
             and p.get("longitude") is not None]
    if len(track) >= 2:
        xy = [_xy(float(p["latitude"]), float(p["longitude"]))
              for p in track]
        draw.line(xy, fill=(37, 99, 235), width=3)
        for x, y in xy[:-1]:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3],
                         fill=(37, 99, 235))
    # Airport markers.
    for code, (alat, alon) in marks.items():
        x, y = _xy(alat, alon)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7],
                     outline=(34, 139, 34), width=3)
        info = airport_info(code) or {}
        label = f"{code} — {info.get('name') or ''}".strip(" —")
        draw.text((x + 10, y - 7), label, fill=(34, 100, 34))
    # Current position.
    if current is not None and current.get("latitude") is not None \
            and current.get("longitude") is not None:
        x, y = _xy(float(current["latitude"]),
                   float(current["longitude"]))
        draw.ellipse([x - 9, y - 9, x + 9, y + 9],
                     fill=(220, 38, 38))
        draw.ellipse([x - 14, y - 14, x + 14, y + 14],
                     outline=(220, 38, 38), width=2)
        draw.text((x + 16, y - 7), "now", fill=(180, 30, 30))
    if not track and not marks:
        draw.text((margin, height // 2 - 8),
                  "no track points yet — the map fills in as fixes arrive",
                  fill=(140, 140, 130))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    return png, hashlib.sha256(png).hexdigest()


# ── the watcher ──────────────────────────────────────────────────────

@dataclass
class FlightWatch:
    """One persistent flight watch: "tell me when it lands"."""

    id: str
    flight: str
    destination: str = ""
    expect_at: float = 0.0
    notify: str = "any"  # landed | late | diverted | lost | any
    active: bool = True
    created_at: float = 0.0
    last_checked: float = 0.0
    last_seen: dict[str, Any] | None = None
    miss_count: int = 0
    alerted: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        dest = f" → {self.destination}" if self.destination else ""
        return f"{self.flight}{dest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "flight": self.flight,
            "destination": self.destination,
            "expect_at": self.expect_at, "notify": self.notify,
            "active": self.active, "created_at": self.created_at,
            "last_checked": self.last_checked,
            "last_seen": self.last_seen,
            "miss_count": self.miss_count,
            "alerted": list(self.alerted),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FlightWatch:
        return cls(
            id=str(raw.get("id") or ""),
            flight=str(raw.get("flight") or ""),
            destination=str(raw.get("destination") or ""),
            expect_at=float(raw.get("expect_at") or 0.0),
            notify=str(raw.get("notify") or "any"),
            active=bool(raw.get("active", True)),
            created_at=float(raw.get("created_at") or 0.0),
            last_checked=float(raw.get("last_checked") or 0.0),
            last_seen=raw.get("last_seen"),
            miss_count=int(raw.get("miss_count") or 0),
            alerted=list(raw.get("alerted") or []),
        )


class FlightWatcher:
    """Persistent flight watches, under the fare watchers' alert
    discipline: one alert per event, never spam.

    Store: ``<home>/data/flight_watches.json`` (atomic JSON, same
    shape as the other durable JSON stores). The feed is injected
    (a fake in tests, the live OpenSky feed in production); the
    sender is an optional ``sender(text, buttons=...)`` callable.
    A check that finds nothing never raises.
    """

    _NAME = "flight_watches.json"

    def __init__(self, home: Any, *, feed: Any = None,
                 sender: Callable[..., Any] | None = None,
                 now: Callable[[], float] | None = None) -> None:
        self._path = Path(home) / "data" / self._NAME
        self._data = _read_json(self._path)
        self.feed = feed
        self._sender = sender
        self._now = now or time.time

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            _atomic_json_write(self._path, self._data)

    def _rows(self) -> list[dict[str, Any]]:
        return list(self._data.get("watches") or [])

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        self._data["watches"] = rows
        self._save()

    def watch(self, flight: Any, *, destination: Any = "",
              expect_at: float = 0.0,
              notify: str = "any") -> FlightWatch:
        parsed = parse_flight_query(flight)
        if parsed["kind"] == "unknown":
            raise ValueError(parsed["reason"])
        dest = ""
        if destination is not None and str(destination).strip():
            dest = str(destination).strip().upper()
            if not re.fullmatch(r"[A-Z]{3}", dest):
                raise ValueError(
                    "destination must be a 3-letter IATA code")
        mode = (notify or "any").strip().lower()
        if mode not in ("any", "landed", "late", "diverted", "lost"):
            raise ValueError("notify must be any | landed | late | "
                             "diverted | lost")
        watch = FlightWatch(
            id="fw_" + uuid.uuid4().hex[:8],
            flight=parsed["flight"],
            destination=dest,
            expect_at=float(expect_at or 0.0),
            notify=mode,
            created_at=self._now(),
        )
        rows = self._rows()
        rows.append(watch.to_dict())
        self._write_rows(rows)
        return watch

    def unwatch(self, watch_id: str) -> bool:
        rows = self._rows()
        kept = [r for r in rows if r.get("id") != watch_id]
        if len(kept) == len(rows):
            return False
        self._write_rows(kept)
        return True

    def list_watches(self, *, active_only: bool = True
                     ) -> list[FlightWatch]:
        watches = [FlightWatch.from_dict(r) for r in self._rows()]
        if active_only:
            watches = [w for w in watches if w.active]
        watches.sort(key=lambda w: w.created_at)
        return watches

    def get(self, watch_id: str) -> FlightWatch | None:
        for raw in self._rows():
            if raw.get("id") == watch_id:
                return FlightWatch.from_dict(raw)
        return None

    def _store_watch(self, watch: FlightWatch) -> None:
        rows = [watch.to_dict() if r.get("id") == watch.id else r
                for r in self._rows()]
        self._write_rows(rows)

    def _allowed(self, watch: FlightWatch, kind: str) -> bool:
        return watch.notify in ("any", kind)

    def _airport_label(self, info: dict[str, Any]) -> str:
        """``Charles de Gaulle International Airport (CDG), Paris``."""
        label = f"{info.get('name') or ''} ({info.get('iata') or ''})"
        city = (info.get("municipality") or "").strip()
        if city:
            label += f", {city}"
        return label.strip(" ,")

    def _alert(self, watch: FlightWatch, kind: str,
               text: str) -> dict[str, Any] | None:
        if kind in watch.alerted or not self._allowed(watch, kind):
            return None
        watch.alerted.append(kind)
        alert = {"watch_id": watch.id, "flight": watch.flight,
                 "kind": kind, "text": text,
                 "at": float(self._now())}
        if self._sender is not None:
            with contextlib.suppress(Exception):
                self._sender(text, buttons=[])
        return alert

    def check(self, watch: FlightWatch) -> dict[str, Any] | None:
        """One watch check. Returns the alert when an event fires,
        else None. Never raises."""
        now = float(self._now())
        fresh = self.get(watch.id)
        if fresh is not None:
            watch = fresh
        if not watch.active:
            return None
        watch.last_checked = now
        feed = self.feed if self.feed is not None else OpenSkyFeed()
        try:
            states, meta = feed.fetch()
        except Exception:  # noqa: BLE001 - a dead feed is a miss
            states, meta = [], {"error": "feed failed"}
        if meta.get("error"):
            self._store_watch(watch)
            return None
        state = find_flight(states, watch.flight)
        alert: dict[str, Any] | None = None
        if state is None:
            watch.miss_count += 1
            if (watch.miss_count >= LOST_AFTER_MISSES
                    and watch.last_seen is not None
                    and watch.last_seen.get("on_ground") is False):
                last = watch.last_seen
                when = time.strftime(
                    "%Y-%m-%d %H:%M",
                    time.localtime(float(last.get("ts") or 0.0)))
                alert = self._alert(
                    watch, "lost",
                    (f"📡 {watch.flight} dropped off the live feed — "
                     f"last seen {when} at "
                     f"{last.get('latitude')}, {last.get('longitude')} "
                     f"— a coverage gap, not a landing."))
            self._store_watch(watch)
            return alert
        watch.miss_count = 0
        watch.last_seen = {
            "ts": now, "latitude": state.latitude,
            "longitude": state.longitude,
            "on_ground": state.on_ground,
            "altitude_ft": state.altitude_ft,
        }
        if state.on_ground and state.has_position:
            near = nearest_airport(state.latitude, state.longitude,
                                   max_km=LANDED_NEAR_KM)
            if near is not None:
                dest = watch.destination
                if (dest and coords_for_iata(dest) is not None
                        and near["iata"] != dest):
                    dest_info = airport_info(dest) or {}
                    alert = self._alert(
                        watch, "diverted",
                        (f"⚠️ {watch.flight} diverted — landed at "
                         f"{self._airport_label(near)} instead of "
                         f"{self._airport_label(dest_info)}."))
                else:
                    alert = self._alert(
                        watch, "landed",
                        (f"✈️ {watch.flight} landed at "
                         f"{self._airport_label(near)}."))
            else:
                alert = self._alert(
                    watch, "landed",
                    (f"✈️ {watch.flight} is on the ground at "
                     f"{state.latitude}, {state.longitude} — no known "
                     f"airport within {LANDED_NEAR_KM:.0f} km."))
        elif (watch.expect_at and now > watch.expect_at
                and not state.on_ground):
            expected = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(watch.expect_at))
            alert = self._alert(
                watch, "late",
                (f"⏰ {watch.flight} is running late — expected by "
                 f"{expected}, still airborne at "
                 f"{state.altitude_ft or 0:.0f} ft."))
        self._store_watch(watch)
        return alert

    def check_all(self) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        for watch in self.list_watches():
            try:
                found = self.check(watch)
            except Exception:  # noqa: BLE001 - one bad watch never kills
                _log.debug("flight watch check failed", exc_info=True)
                continue
            if found is not None:
                alerts.append(found)
        return alerts


# ── the scheduled tick ───────────────────────────────────────────────

def run_flight_track_tick_once(
        *, context: Any = None, notifier: Any = None,
        feed: Any = None, now: float | None = None,
        home: Any = None) -> dict[str, Any]:
    """One flight-track tick: check every active watch, publish
    what fired. Never raises. A quiet tick (no watches) makes no
    network call at all — the feed is only built when a watch
    exists.
    """
    now_f = float(now if now is not None else time.time())
    try:
        home_path = _resolve_home(home, context)
        watcher = FlightWatcher(
            home_path, feed=feed, now=lambda: now_f)
        alerts = watcher.check_all()
    except Exception as exc:  # noqa: BLE001 - tick must never raise
        return {"ok": False, "checked": 0, "alerts": 0,
                "ran_at": now_f, "error": f"tick failed: {exc}"}
    for alert in alerts:
        if notifier is None:
            continue
        with contextlib.suppress(Exception):
            notifier.publish(
                "flight_track",
                title=f"flight {alert['flight']}: {alert['kind']}",
                body=alert["text"])
    return {"ok": True, "checked": len(watcher.list_watches()),
            "alerts": len(alerts), "ran_at": now_f, "error": ""}


def ensure_flight_track_job(scheduler: Any, *,
                            spec: str = FLIGHT_TRACK_SPEC
                            ) -> Any | None:
    """Register the 15-minute flight-track job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job
    that already matches is returned untouched. Best-effort — a
    scheduler that cannot store the job returns None rather than
    raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"),
                                          job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != FLIGHT_TRACK_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            FLIGHT_TRACK_JOB, spec, "tool",
            {"tool": FLIGHT_TRACK_TOOL, "args": {}}, routine=True)
    except Exception:
        return None
