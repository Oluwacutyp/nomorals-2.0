"""Live geocode providers: the federation rungs behind the local gazetteer.

Doc 21 S3 left the federation seam unwired ("a real provider is a
separate owner decision"); census B9 is that decision.  Two keyless
providers, both fallback-only — they run only when the local index
has no answer, and only when owner policy
(``NM_MAPS_GEOCODE_FEDERATION``) permits:

* :class:`NominatimProvider` — the primary rung.  OpenStreetMap's
  Nominatim search/reverse, keyless under its usage policy: at most
  1 request per second, an identifying User-Agent, attribution,
  ODbL data.  The policy is enforced in code (a shared 1-second
  throttle), not just documented.
* :class:`OverpassProvider` — the second rung, for POI/place
  search where Nominatim is thin, and for reverse around a point.
  Its one honest limit, verified live 2026-11-12: a global
  name-regex search is a planet scan the public server will not
  answer in time (40 s, zero bytes), and the public instance
  502s at capacity under load — so forward search uses the
  exact-name form only, and the reliable Overpass shape is the
  scoped ``around`` reverse.  A rung failure is raised, never a
  fake answer.

:class:`LiveGeocodeProvider` chains the two in that order behind
the :class:`~nomorals.maps.gazetteer.GeocodeProvider` protocol:
Nominatim first; Overpass when Nominatim is empty or fails.  Its
``name`` reports the rung that actually answered (``"nominatim"``
or ``"overpass"``), so the gazetteer's result says plainly which
rung answered; before any answer it reads ``"nominatim+overpass"``.

This module is deliberately separate from
:mod:`nomorals.maps.gazetteer`: the gazetteer carries a no-network
AST pin (no ``socket``/``urllib``/``requests``/``http.client``
imports), so the providers — the only maps code that opens a
socket — live here.  The routing providers below ride the same
home for the same reason: :mod:`nomorals.maps.routing` carries
the same pin, and its :class:`~nomorals.maps.routing.RouteProvider`
seam is what they implement.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from ..core.logging_setup import get_logger
from ..location.primitives import LatLon, haversine_m
from .gazetteer import (
    KIND_WEIGHTS,
    POSTURE_FEDERATED,
    REVERSE_LADDER_KM,
    GeocodeCandidate,
    ReverseCandidate,
)
from .routing import RouteResult, RouteStep

__all__ = [
    "LiveGeocodeError",
    "LiveGeocodeProvider",
    "LiveRouteError",
    "LiveRouteProvider",
    "NominatimProvider",
    "OSRMProvider",
    "OverpassProvider",
    "ValhallaProvider",
    "live_geocode_provider",
    "live_route_provider",
]

_log = get_logger(__name__)

#: User-Agent identifying the application, as Nominatim's usage
#: policy requires (stock library User-Agents will not do).
USER_AGENT = (
    "Devon-Atlas/1.0 (personal AI assistant;"
    " https://github.com/Oluwacutyp/nomorals-2.0)")

#: Nominatim's documented ceiling: at most 1 request per second.
NOMINATIM_MIN_INTERVAL_S = 1.0

#: Per-request ceilings: Nominatim answers fast; Overpass queries
#: run server-side for up to their 25 s query timeout.
NOMINATIM_TIMEOUT_S = 10.0
OVERPASS_TIMEOUT_S = 30.0
OVERPASS_QUERY_TIMEOUT_S = 25

NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Attribution the providers' data carries (ODbL).  Surfaces
#: that render a federated answer should carry it; the registry
#: rows in definitions.py record it too.
ATTRIBUTION = "© OpenStreetMap contributors"


class LiveGeocodeError(Exception):
    """A live rung failed (transport, status, or payload) — the
    local answer stands behind it.  Never a fake candidate."""


# ── rate limiting (Nominatim policy) ─────────────────────────────


class _RateLimiter:
    """One shared throttle: at most one call per ``min_interval_s``.

    The limit is per application, not per object, so the default
    Nominatim provider instances share one module-level limiter.
    Tests pass a zero-interval limiter instead of waiting.
    """

    def __init__(self, min_interval_s: float = 1.0) -> None:
        self._min = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._last: float | None = None

    @property
    def min_interval_s(self) -> float:
        return self._min

    def wait(self) -> None:
        """Sleep until the interval since the last call elapsed."""
        if self._min <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            if self._last is not None:
                due = self._last + self._min - now
                if due > 0.0:
                    time.sleep(due)
                    now = time.monotonic()
            self._last = now


#: The shared Nominatim throttle (1 request/second, per policy).
_NOMINATIM_LIMITER = _RateLimiter(NOMINATIM_MIN_INTERVAL_S)


# ── HTTP plumbing (urllib, the repo's own idiom) ─────────────────


def _http_json(url: str, *, data: bytes | None = None,
               timeout_s: float, who: str):
    """GET (``data`` None) or POST ``url`` as JSON, or raise
    :class:`LiveGeocodeError`.  The User-Agent identifies Devon
    Atlas, per Nominatim's usage policy."""
    headers = {"User-Agent": USER_AGENT,
               "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = (
            "application/x-www-form-urlencoded")
    request = urllib.request.Request(
        url, data=data, headers=headers,
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(
                request, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise LiveGeocodeError(
            who + " answered HTTP " + str(exc.code)
            + " for " + url.split("?", 1)[0]) from exc
    except (urllib.error.URLError, TimeoutError,
            OSError) as exc:
        raise LiveGeocodeError(
            who + " unreachable (" + type(exc).__name__
            + ": " + str(exc) + ")") from exc
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise LiveGeocodeError(
            who + " returned unparseable JSON") from exc


def _as_float(value) -> float | None:
    """A coordinate as a float, or None when it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _score(position: int) -> float:
    """A provider candidate's display score: position-based, the
    first result 400, descending by 10 — the order the provider
    ranked them in.  Provider candidates never mix with local
    ones (federation runs only on a local miss), so the score
    only orders within one provider answer."""
    return 400.0 - 10.0 * max(0, int(position))


def _nominatim_kind(item: dict) -> str:
    """A Nominatim result's gazetteer kind: its address type when
    that is a known kind, a POI class for POI categories, else
    ``"place"``."""
    addresstype = str(item.get("addresstype") or "")
    if addresstype in KIND_WEIGHTS:
        return addresstype
    category = str(item.get("category") or "")
    if category in ("amenity", "shop", "tourism", "leisure",
                    "railway", "aeroway", "natural", "manmade",
                    "building", "office", "craft", "healthcare",
                    "education"):
        return "poi"
    type_name = str(item.get("type") or "")
    if type_name in KIND_WEIGHTS:
        return type_name
    return "place"


def _nominatim_name(item: dict, query: str) -> str:
    """The short name of a Nominatim result: ``name`` when the
    service carries it, else the first segment of its display
    name, else the query itself."""
    name = str(item.get("name") or "").strip()
    if name:
        return name
    display = str(item.get("display_name") or "")
    if display:
        return display.split(",", 1)[0].strip() or query
    return query


# ── Nominatim ────────────────────────────────────────────────────


@dataclass
class NominatimProvider:
    """OpenStreetMap Nominatim: the primary live rung.

    Keyless under the usage policy (max 1 request/second, an
    identifying User-Agent, attribution, ODbL).  The throttle is
    shared across instances; pass a zero-interval limiter only
    in tests.
    """

    name: str = "nominatim"
    timeout_s: float = NOMINATIM_TIMEOUT_S
    limiter: _RateLimiter | None = None

    def _limiter(self) -> _RateLimiter:
        return self.limiter if self.limiter is not None \
            else _NOMINATIM_LIMITER

    def _search_url(self, query: str, limit: int) -> str:
        params = urllib.parse.urlencode({
            "q": query, "format": "jsonv2",
            "limit": max(1, int(limit)), "addressdetails": "0"})
        return NOMINATIM_BASE + "/search?" + params

    def _reverse_url(self, lat: float, lon: float) -> str:
        params = urllib.parse.urlencode({
            "lat": format(float(lat), ".6f"),
            "lon": format(float(lon), ".6f"),
            "format": "jsonv2", "zoom": "18"})
        return NOMINATIM_BASE + "/reverse?" + params

    def search(self, query: str, *,
               limit: int = 8) -> tuple[GeocodeCandidate, ...]:
        """Forward-geocode ``query``.  Raises
        :class:`LiveGeocodeError` on any rung failure."""
        text = (query or "").strip()
        if not text:
            return ()
        self._limiter().wait()
        payload = _http_json(
            self._search_url(text, limit), timeout_s=self.timeout_s,
            who="nominatim")
        if isinstance(payload, dict):
            raise LiveGeocodeError(
                "nominatim search failed: "
                + str(payload.get("error", payload))[:200])
        if not isinstance(payload, list):
            raise LiveGeocodeError(
                "nominatim search returned an unexpected payload")
        found: list[GeocodeCandidate] = []
        for position, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            lat = _as_float(item.get("lat"))
            lon = _as_float(item.get("lon"))
            if lat is None or lon is None:
                continue
            found.append(GeocodeCandidate(
                name=_nominatim_name(item, text),
                kind=_nominatim_kind(item), lat=lat, lon=lon,
                pack_id="", score=_score(position),
                matched_as="provider"))
        return tuple(found[:max(1, int(limit))])

    def reverse(self, lat: float, lon: float, *,
                limit: int = 8) -> tuple[ReverseCandidate, ...]:
        """Reverse-geocode one point.  A documented ``error``
        payload ("Unable to geocode") is an empty answer, not a
        failure."""
        self._limiter().wait()
        payload = _http_json(
            self._reverse_url(lat, lon), timeout_s=self.timeout_s,
            who="nominatim")
        if not isinstance(payload, dict):
            raise LiveGeocodeError(
                "nominatim reverse returned an unexpected payload")
        if payload.get("error"):
            return ()
        rlat = _as_float(payload.get("lat"))
        rlon = _as_float(payload.get("lon"))
        if rlat is None or rlon is None:
            return ()
        distance = haversine_m(
            LatLon(lat=float(lat), lon=float(lon)),
            LatLon(lat=rlat, lon=rlon)) / 1000.0
        return (ReverseCandidate(
            name=_nominatim_name(payload, ""),
            kind=_nominatim_kind(payload), lat=rlat, lon=rlon,
            pack_id="", distance_km=distance),)


# ── Overpass ─────────────────────────────────────────────────────


def _ql_escape(text: str) -> str:
    """Escape ``text`` for an Overpass double-quoted string."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _overpass_kind(tags: dict) -> str:
    """An Overpass element's gazetteer kind: its ``place`` tag
    when that is a known kind, a POI class for POI tags, else
    ``"place"``."""
    place = str(tags.get("place") or "")
    if place in KIND_WEIGHTS:
        return place
    for key in ("amenity", "shop", "tourism", "leisure",
                "natural", "building", "office", "craft",
                "railway", "aeroway"):
        if tags.get(key):
            return "poi"
    return "place"


@dataclass
class OverpassProvider:
    """Overpass API: the second live rung.

    Keyless.  Forward search uses the exact-name form only —
    ``nwr["name"="<query>",i]`` — because a global name-regex
    search is a planet scan the public server will not answer in
    time (verified live: 40 s, zero bytes), and the public
    instance 502s at capacity under load, so the rung may fail
    honestly where Nominatim succeeds.  Reverse uses the scoped
    ``around`` query, stepping the local ladder's radii — the
    reliable Overpass shape.
    """

    name: str = "overpass"
    timeout_s: float = OVERPASS_TIMEOUT_S

    def _post(self, query: str):
        body = urllib.parse.urlencode({"data": query}).encode("utf-8")
        return _http_json(OVERPASS_URL, data=body,
                          timeout_s=self.timeout_s, who="overpass")

    def _search_ql(self, query: str, limit: int) -> str:
        return (
            "[out:json][timeout:" + str(OVERPASS_QUERY_TIMEOUT_S)
            + "];(nwr[\"name\"=\"" + _ql_escape(query) + "\",i];);"
            "out center " + str(max(1, int(limit))) + ";")

    def _around_ql(self, lat: float, lon: float, radius_m: int,
                   limit: int) -> str:
        return (
            "[out:json][timeout:" + str(OVERPASS_QUERY_TIMEOUT_S)
            + "];(nwr(around:" + str(int(radius_m)) + ","
            + format(float(lat), ".6f") + ","
            + format(float(lon), ".6f") + ")[\"name\"];);"
            "out center " + str(max(1, int(limit))) + ";")

    @staticmethod
    def _element_point(element: dict) -> tuple | None:
        """An element's anchor: a node's own coordinates, or a
        way/relation's ``center``.  None when it has neither."""
        if element.get("type") == "node":
            lat = _as_float(element.get("lat"))
            lon = _as_float(element.get("lon"))
        else:
            center = element.get("center") or {}
            lat = _as_float(center.get("lat"))
            lon = _as_float(center.get("lon"))
        if lat is None or lon is None:
            return None
        return lat, lon

    def _parse_search(self, payload, limit: int):
        if not isinstance(payload, dict) \
                or not isinstance(payload.get("elements"), list):
            raise LiveGeocodeError(
                "overpass search returned an unexpected payload")
        found: list[GeocodeCandidate] = []
        for position, element in enumerate(payload["elements"]):
            if not isinstance(element, dict):
                continue
            tags = element.get("tags") or {}
            if not isinstance(tags, dict):
                continue
            name = str(tags.get("name") or "").strip()
            point = self._element_point(element)
            if not name or point is None:
                continue
            lat, lon = point
            found.append(GeocodeCandidate(
                name=name, kind=_overpass_kind(tags), lat=lat,
                lon=lon, pack_id="", score=_score(position),
                matched_as="provider"))
            if len(found) >= max(1, int(limit)):
                break
        return tuple(found)

    def search(self, query: str, *,
               limit: int = 8) -> tuple[GeocodeCandidate, ...]:
        """Forward-geocode ``query`` by exact name.  Raises
        :class:`LiveGeocodeError` on any rung failure (including
        the public instance answering 502 at capacity)."""
        text = (query or "").strip()
        if not text:
            return ()
        payload = self._post(self._search_ql(text, limit))
        return self._parse_search(payload, limit)

    def reverse(self, lat: float, lon: float, *,
                limit: int = 8) -> tuple[ReverseCandidate, ...]:
        """Reverse-geocode one point: named features around it,
        stepping the local ladder's radii (1/5/25/100 km) and
        stopping at the first rung with an answer — each query
        stays scoped, so the server never scans a planet."""
        point = LatLon(lat=float(lat), lon=float(lon))
        for radius_km in REVERSE_LADDER_KM:
            payload = self._post(self._around_ql(
                lat, lon, int(radius_km * 1000.0), limit))
            if not isinstance(payload, dict) \
                    or not isinstance(payload.get("elements"), list):
                raise LiveGeocodeError(
                    "overpass reverse returned an unexpected"
                    " payload")
            found: list[ReverseCandidate] = []
            for element in payload["elements"]:
                if not isinstance(element, dict):
                    continue
                tags = element.get("tags") or {}
                if not isinstance(tags, dict):
                    continue
                name = str(tags.get("name") or "").strip()
                anchor = self._element_point(element)
                if not name or anchor is None:
                    continue
                rlat, rlon = anchor
                distance = haversine_m(
                    point, LatLon(lat=rlat, lon=rlon)) / 1000.0
                if distance > radius_km:
                    continue
                found.append(ReverseCandidate(
                    name=name, kind=_overpass_kind(tags), lat=rlat,
                    lon=rlon, pack_id="", distance_km=distance))
            if found:
                found.sort(key=lambda c: (
                    c.distance_km, -KIND_WEIGHTS.get(c.kind, 8),
                    c.name.casefold()))
                return tuple(found[:max(1, int(limit))])
        return ()


# ── the chain ────────────────────────────────────────────────────


class LiveGeocodeProvider:
    """Nominatim, then Overpass, behind the gazetteer's
    :class:`~nomorals.maps.gazetteer.GeocodeProvider` seam.

    ``name`` reports the rung that actually answered
    (``"nominatim"`` / ``"overpass"``); before any answer it
    reads ``"nominatim+overpass"``.  A rung that fails after the
    other already answered (even with an empty answer) is logged
    and the standing answer wins; only a total failure — both
    rungs down — raises, so the gazetteer can name it.
    """

    def __init__(self, *, nominatim: NominatimProvider | None = None,
                 overpass: OverpassProvider | None = None) -> None:
        self._nominatim = nominatim if nominatim is not None \
            else NominatimProvider()
        self._overpass = overpass if overpass is not None \
            else OverpassProvider()
        self._answered_rung: str | None = None

    @property
    def name(self) -> str:
        return self._answered_rung or "nominatim+overpass"

    def search(self, query: str, *,
               limit: int = 8) -> tuple[GeocodeCandidate, ...]:
        """Nominatim first; Overpass when it is empty or fails."""
        try:
            found = self._nominatim.search(query, limit=limit)
        except LiveGeocodeError as exc:
            _log.warning("nominatim search failed for %r: %s",
                         query, exc)
            return self._overpass_search(query, limit, exc)
        if found:
            self._answered_rung = "nominatim"
            return found
        return self._overpass_search(query, limit, None)

    def _overpass_search(self, query: str, limit: int,
                         nominatim_error):
        try:
            found = self._overpass.search(query, limit=limit)
        except LiveGeocodeError as exc:
            if nominatim_error is None:
                # Nominatim answered (empty); the second rung's
                # failure is logged and the standing empty answer
                # wins — it is not a reason to fail the query.
                _log.warning("overpass search failed for %r: %s",
                             query, exc)
                return ()
            raise LiveGeocodeError(
                "both live geocode rungs failed: nominatim ("
                + str(nominatim_error) + ") and overpass ("
                + str(exc) + ")") from exc
        if found:
            self._answered_rung = "overpass"
        return found

    def reverse(self, lat: float, lon: float, *,
                limit: int = 8) -> tuple[ReverseCandidate, ...]:
        """Nominatim first; Overpass (around-ladder) when it is
        empty or fails."""
        try:
            found = self._nominatim.reverse(lat, lon, limit=limit)
        except LiveGeocodeError as exc:
            _log.warning("nominatim reverse failed for %r,%r: %s",
                         lat, lon, exc)
            return self._overpass_reverse(lat, lon, limit, exc)
        if found:
            self._answered_rung = "nominatim"
            return found
        return self._overpass_reverse(lat, lon, limit, None)

    def _overpass_reverse(self, lat: float, lon: float, limit: int,
                          nominatim_error):
        try:
            found = self._overpass.reverse(lat, lon, limit=limit)
        except LiveGeocodeError as exc:
            if nominatim_error is None:
                _log.warning("overpass reverse failed for %r,%r: %s",
                             lat, lon, exc)
                return ()
            raise LiveGeocodeError(
                "both live geocode rungs failed: nominatim ("
                + str(nominatim_error) + ") and overpass ("
                + str(exc) + ")") from exc
        if found:
            self._answered_rung = "overpass"
        return found


def live_geocode_provider() -> LiveGeocodeProvider:
    """The wired federation chain: Nominatim, then Overpass.

    Construction opens no socket; the rungs run only when the
    gazetteer's federation posture calls the provider (local
    miss + ``NM_MAPS_GEOCODE_FEDERATION`` on)."""
    return LiveGeocodeProvider()


# ── live routing (census C3) ─────────────────────────────────────
#
# Two keyless routers, both fallback-only — they run only when
# the local road graph has no route, and only when owner policy
# (``NM_MAPS_ROUTING_FEDERATION``) permits:
#
# * :class:`OSRMProvider` — the primary rung.  The OSRM public
#   demo server (router.project-osrm.org), keyless, driving
#   profile.  The demo is for light/demo use, not production
#   traffic — the registry row says so.  Its one honest limit,
#   verified live 2026-11-12: the demo snaps a waypoint to the
#   nearest road with NO distance limit (two mid-ocean points
#   routed a 606 m hop on an island 800 km away), so the rung
#   passes ``radiuses=500;500`` — the local plane's snap bound
#   — and a point with no road inside the radius answers
#   ``NoSegment``.  A disconnected pair answers ``NoRoute``.
# * :class:`ValhallaProvider` — the second rung.  The public
#   Valhalla demo instance (valhalla1.openstreetmap.de),
#   costing ``auto`` (driving), keyless.  Slow — 14 s observed
#   for a Lagos hop, against OSRM's 2 s — and its no-route
#   vocabulary is HTTP 400 error codes (170 unconnected
#   regions, 171 no suitable edges, 441 unreachable, 442 no
#   path), verified live for 171 and 442 on 2026-11-12.
#   Its snap bound is its own ``search_cutoff`` (35 km); the
#   request carries ``radius: 500`` as a preference, and the
#   returned snap distances ride the result, labelled like
#   the local plane's snaps.
#
# Both rungs return a :class:`~nomorals.maps.routing.RouteResult`
# of their own (``pack_id`` empty — no pack served it; the
# Router mints the receipt for whichever plane answered), or
# None for an honest empty answer (no route), and raise
# :class:`LiveRouteError` on any rung failure.  A rung failure
# is raised, never a fake route.


class LiveRouteError(Exception):
    """A live routing rung failed (transport, status, or
    payload) — the local answer stands behind it.  Never a
    fake route."""


OSRM_BASE = "https://router.project-osrm.org"
OSRM_TIMEOUT_S = 15.0

#: The snap bound the OSRM rung enforces, in metres — the local
#: plane's :data:`~nomorals.maps.routing.SNAP_MAX_M`, so a
#: provider route never teleports an endpoint farther than the
#: local plane would snap one.
OSRM_SNAP_RADIUS_M = 500

VALHALLA_URL = "https://valhalla1.openstreetmap.de/route"
VALHALLA_TIMEOUT_S = 60.0

#: Valhalla's documented no-route error codes (the route API
#: reference's error table): 170 unconnected regions, 171 no
#: suitable edges near a location, 441 location unreachable,
#: 442 no path could be found.  171 and 442 verified live
#: 2026-11-12; 170 and 441 rest on the documented table.
#: Any other error code is a rung failure, not an empty answer.
VALHALLA_NO_ROUTE_CODES = frozenset((170, 171, 441, 442))

#: OSRM's no-route codes: ``NoRoute`` (no path between the
#: snapped points) and ``NoSegment`` (a point with no road
#: inside the snap radius) — both verified live 2026-11-12.
#: Any other code is a rung failure.
OSRM_NO_ROUTE_CODES = frozenset(("NoRoute", "NoSegment"))


def _route_http_json(url: str, *, data: bytes | None = None,
                     timeout_s: float, who: str):
    """GET (``data`` None) or POST ``url`` and return
    ``(http_status, payload)`` — the status rides along so a
    rung can tell an honest empty answer (an HTTP 400 carrying
    a no-route code) from a failure.  Transport errors and
    unparseable bodies raise :class:`LiveRouteError`."""
    headers = {"User-Agent": USER_AGENT,
               "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers,
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(
                request, timeout=timeout_s) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:  # noqa: BLE001 — no body to read
            raw = b""
        try:
            payload = json.loads(raw.decode("utf-8",
                                            errors="replace")) \
                if raw else {}
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return int(exc.code), payload
    except (urllib.error.URLError, TimeoutError,
            OSError) as exc:
        raise LiveRouteError(
            who + " unreachable (" + type(exc).__name__
            + ": " + str(exc) + ")") from exc
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise LiveRouteError(
            who + " returned unparseable JSON") from exc
    if not isinstance(payload, dict):
        raise LiveRouteError(
            who + " returned an unexpected payload")
    return status, payload


def _decode_polyline(shape: str, *,
                     precision: int = 6,
                     ) -> tuple[tuple[float, float], ...]:
    """Decode an encoded polyline (Valhalla's leg ``shape``) to
    ``(lat, lon)`` points.  The standard Google polyline
    algorithm at the given decimal precision (Valhalla's is 6);
    an empty shape decodes to nothing.  Proven against the
    live Valhalla shape in the federation suite."""
    factor = 10.0 ** precision
    points: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lon = 0
    length = len(shape)
    while index < length:
        for which in (0, 1):
            shift = 0
            value = 0
            while True:
                if index >= length:
                    raise LiveRouteError(
                        "valhalla shape is truncated")
                byte = ord(shape[index]) - 63
                index += 1
                value |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(value >> 1) if value & 1 else value >> 1
            if which == 0:
                lat += delta
            else:
                lon += delta
        points.append((lat / factor, lon / factor))
    return tuple(points)


def _with_endpoints(geometry: tuple[tuple[float, float], ...],
                    start: LatLon, end: LatLon,
                    ) -> tuple[tuple[float, float], ...]:
    """A provider geometry with the requested endpoints on it:
    the route's drawn line starts at the requested start and
    ends at the requested end, the way the local plane's does
    (its snap legs ride the geometry, never a silent teleport).
    Consecutive duplicates are dropped, as the local plane
    drops them."""
    points: list[tuple[float, float]] = [
        (start.lat, start.lon)]
    points.extend(geometry)
    points.append((end.lat, end.lon))
    return tuple(
        point for at, point in enumerate(points)
        if at == 0 or point != points[at - 1])


def _route_detail(who: str, distance_m: float,
                  duration_s: float) -> str:
    """One provider route's detail line: the distance and the
    provider's own duration — a live estimate, never the local
    plane's class-speed arithmetic."""
    return ("route via " + who + ": "
            + format(distance_m, ",.0f") + " m by road, ~"
            + str(int(round(duration_s)))
            + " s (provider estimate)")


@dataclass
class OSRMProvider:
    """The OSRM public demo server: the primary live rung.

    Keyless, driving profile, snap radius 500 m (the local
    plane's bound).  ``NoRoute``/``NoSegment`` are empty
    answers (None); anything else is a :class:`LiveRouteError`.
    """

    name: str = "osrm"
    timeout_s: float = OSRM_TIMEOUT_S

    def _route_url(self, start: LatLon, end: LatLon) -> str:
        return (
            OSRM_BASE + "/route/v1/driving/"
            + format(start.lon, ".6f") + ","
            + format(start.lat, ".6f") + ";"
            + format(end.lon, ".6f") + ","
            + format(end.lat, ".6f")
            + "?overview=full&geometries=geojson&steps=true"
            + "&radiuses=" + str(OSRM_SNAP_RADIUS_M) + ";"
            + str(OSRM_SNAP_RADIUS_M))

    def route(self, start: LatLon,
              end: LatLon) -> RouteResult | None:
        """Route ``start`` → ``end`` via the OSRM demo server.
        Returns None for an honest empty answer."""
        status, payload = _route_http_json(
            self._route_url(start, end), timeout_s=self.timeout_s,
            who="osrm")
        code = str(payload.get("code") or "")
        if status != 200 or code != "Ok":
            if code in OSRM_NO_ROUTE_CODES:
                return None
            raise LiveRouteError(
                "osrm answered HTTP " + str(status) + " ("
                + (code or "no code") + ": "
                + str(payload.get("message", ""))[:200] + ")")
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            raise LiveRouteError(
                "osrm answered Ok with no routes")
        route = routes[0]
        if not isinstance(route, dict):
            raise LiveRouteError(
                "osrm route is not an object")
        distance = _as_float(route.get("distance"))
        duration = _as_float(route.get("duration"))
        geometry = route.get("geometry") or {}
        coords = geometry.get("coordinates") \
            if isinstance(geometry, dict) else None
        if distance is None or duration is None \
                or not isinstance(coords, list) or not coords:
            raise LiveRouteError(
                "osrm route is missing distance, duration or"
                " geometry")
        points: list[tuple[float, float]] = []
        for pair in coords:
            if not isinstance(pair, (list, tuple)) \
                    or len(pair) < 2:
                raise LiveRouteError(
                    "osrm geometry coordinate is not a pair")
            lon = _as_float(pair[0])
            lat = _as_float(pair[1])
            if lat is None or lon is None:
                raise LiveRouteError(
                    "osrm geometry coordinate is not numeric")
            points.append((lat, lon))
        steps: list[RouteStep] = []
        legs = route.get("legs")
        if not isinstance(legs, list):
            raise LiveRouteError("osrm route has no legs")
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            for step in leg.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                name = str(step.get("name") or "").strip()
                mode = str(step.get("mode") or "driving").strip() \
                    or "driving"
                length = _as_float(step.get("distance"))
                if length is None:
                    raise LiveRouteError(
                        "osrm step is missing its distance")
                steps.append(RouteStep(
                    name=name or mode, road_class=mode,
                    length_m=length, named=bool(name)))
        from_snap = None
        to_snap = None
        waypoints = payload.get("waypoints")
        if isinstance(waypoints, list) and len(waypoints) >= 2:
            from_snap = _as_float(waypoints[0].get("distance")) \
                if isinstance(waypoints[0], dict) else None
            to_snap = _as_float(waypoints[1].get("distance")) \
                if isinstance(waypoints[1], dict) else None
        return RouteResult(
            ok=True, from_point=(start.lat, start.lon),
            to_point=(end.lat, end.lon),
            geometry=_with_endpoints(tuple(points), start, end),
            distance_m=distance, duration_s=duration,
            steps=tuple(steps), from_snap_m=from_snap,
            to_snap_m=to_snap, pack_id="",
            posture=POSTURE_FEDERATED, provider="osrm",
            detail=_route_detail("OSRM demo server",
                                 distance, duration))


@dataclass
class ValhallaProvider:
    """The public Valhalla demo instance: the second live rung.

    Keyless, costing ``auto`` (driving).  Its no-route
    vocabulary is HTTP 400 error codes
    (:data:`VALHALLA_NO_ROUTE_CODES`); anything else is a
    :class:`LiveRouteError`.  The leg ``shape`` is a precision-6
    encoded polyline, decoded by :func:`_decode_polyline`.
    """

    name: str = "valhalla"
    timeout_s: float = VALHALLA_TIMEOUT_S

    def _request_body(self, start: LatLon, end: LatLon) -> bytes:
        return json.dumps({
            "locations": [
                {"lat": float(start.lat), "lon": float(start.lon),
                 "radius": OSRM_SNAP_RADIUS_M},
                {"lat": float(end.lat), "lon": float(end.lon),
                 "radius": OSRM_SNAP_RADIUS_M}],
            "costing": "auto"}).encode("utf-8")

    def route(self, start: LatLon,
              end: LatLon) -> RouteResult | None:
        """Route ``start`` → ``end`` via the Valhalla demo
        instance.  Returns None for an honest empty answer."""
        status, payload = _route_http_json(
            VALHALLA_URL, data=self._request_body(start, end),
            timeout_s=self.timeout_s, who="valhalla")
        if status != 200:
            error_code = payload.get("error_code")
            if isinstance(error_code, int) \
                    and error_code in VALHALLA_NO_ROUTE_CODES:
                return None
            raise LiveRouteError(
                "valhalla answered HTTP " + str(status) + " ("
                + str(payload.get("error", ""))[:200] + ")")
        trip = payload.get("trip")
        if not isinstance(trip, dict):
            raise LiveRouteError(
                "valhalla answer has no trip")
        if trip.get("status") != 0:
            raise LiveRouteError(
                "valhalla trip status is "
                + str(trip.get("status")) + ": "
                + str(trip.get("status_message", ""))[:200])
        legs = trip.get("legs")
        if not isinstance(legs, list) or not legs \
                or not isinstance(legs[0], dict):
            raise LiveRouteError(
                "valhalla trip has no legs")
        leg = legs[0]
        shape = leg.get("shape")
        if not isinstance(shape, str) or not shape:
            raise LiveRouteError(
                "valhalla leg has no shape")
        points = _decode_polyline(shape)
        if not points:
            raise LiveRouteError(
                "valhalla shape decoded to nothing")
        summary = leg.get("summary") or trip.get("summary") or {}
        if not isinstance(summary, dict):
            raise LiveRouteError(
                "valhalla trip has no summary")
        seconds = _as_float(summary.get("time"))
        length_km = _as_float(summary.get("length"))
        if seconds is None or length_km is None:
            raise LiveRouteError(
                "valhalla summary is missing time or length")
        steps: list[RouteStep] = []
        for maneuver in leg.get("maneuvers") or []:
            if not isinstance(maneuver, dict):
                continue
            instruction = str(
                maneuver.get("instruction") or "").strip()
            travel_mode = str(
                maneuver.get("travel_mode") or "drive").strip() \
                or "drive"
            length = _as_float(maneuver.get("length"))
            if length is None:
                raise LiveRouteError(
                    "valhalla maneuver is missing its length")
            steps.append(RouteStep(
                name=instruction or travel_mode,
                road_class=travel_mode,
                length_m=length * 1000.0,
                named=bool(instruction)))
        distance_m = length_km * 1000.0
        from_snap = haversine_m(
            start, LatLon(*points[0])) if points else None
        to_snap = haversine_m(
            end, LatLon(*points[-1])) if points else None
        return RouteResult(
            ok=True, from_point=(start.lat, start.lon),
            to_point=(end.lat, end.lon),
            geometry=_with_endpoints(points, start, end),
            distance_m=distance_m, duration_s=seconds,
            steps=tuple(steps), from_snap_m=from_snap,
            to_snap_m=to_snap, pack_id="",
            posture=POSTURE_FEDERATED, provider="valhalla",
            detail=_route_detail("Valhalla demo server",
                                 distance_m, seconds))


class LiveRouteProvider:
    """OSRM, then Valhalla, behind the router's
    :class:`~nomorals.maps.routing.RouteProvider` seam.

    ``name`` reports the rung that actually answered
    (``"osrm"`` / ``"valhalla"``); before any answer it reads
    ``"osrm+valhalla"``.  A rung that fails after the other
    already answered (even with an empty answer) is logged and
    the standing answer wins; only a total failure — both
    rungs down — raises, so the router can name it.
    """

    def __init__(self, *, osrm: OSRMProvider | None = None,
                 valhalla: ValhallaProvider | None = None) -> None:
        self._osrm = osrm if osrm is not None else OSRMProvider()
        self._valhalla = valhalla if valhalla is not None \
            else ValhallaProvider()
        self._answered_rung: str | None = None

    @property
    def name(self) -> str:
        return self._answered_rung or "osrm+valhalla"

    def route(self, start: LatLon,
              end: LatLon) -> RouteResult | None:
        """OSRM first; Valhalla when it is empty or fails."""
        try:
            found = self._osrm.route(start, end)
        except LiveRouteError as exc:
            _log.warning("osrm route failed for %r -> %r: %s",
                         start, end, exc)
            return self._valhalla_route(start, end, exc)
        if found is not None:
            self._answered_rung = "osrm"
            return found
        return self._valhalla_route(start, end, None)

    def _valhalla_route(self, start: LatLon, end: LatLon,
                        osrm_error):
        try:
            found = self._valhalla.route(start, end)
        except LiveRouteError as exc:
            if osrm_error is None:
                # OSRM answered (empty); the second rung's
                # failure is logged and the standing empty answer
                # wins — it is not a reason to fail the route.
                _log.warning("valhalla route failed for %r -> %r: %s",
                             start, end, exc)
                return None
            raise LiveRouteError(
                "both live routing rungs failed: osrm ("
                + str(osrm_error) + ") and valhalla ("
                + str(exc) + ")") from exc
        if found is not None:
            self._answered_rung = "valhalla"
        return found


def live_route_provider() -> LiveRouteProvider:
    """The wired federation chain: OSRM, then Valhalla.

    Construction opens no socket; the rungs run only when the
    router's federation posture calls the provider (local
    no-route + ``NM_MAPS_ROUTING_FEDERATION`` on)."""
    return LiveRouteProvider()
