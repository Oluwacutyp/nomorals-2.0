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
socket — live here.
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
    REVERSE_LADDER_KM,
    GeocodeCandidate,
    ReverseCandidate,
)

__all__ = [
    "LiveGeocodeError",
    "LiveGeocodeProvider",
    "NominatimProvider",
    "OverpassProvider",
    "live_geocode_provider",
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
