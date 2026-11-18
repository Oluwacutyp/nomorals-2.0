"""Radio station registry + place index (Radio.Garden core).

The system under the globe is a station directory indexed by place
plus a stream player.  This module is the directory: stations as
registry entries with city/country coordinates, place-first
browsing ("what's on in Accra?"), health-checked like every source
class, seeded from the open radio-browser community directory and
grown by discovery.

Layer 0: stdlib only.  Network is an injected fetcher — with none
wired, directory growth and stream checks answer honestly instead
of fabricating.

She indexes and plays; she does not rebroadcast.  A live stream
has no end, so "play" through the spine is a BOUNDED live sample:
the first seconds of the stream, saved as a file and sent, with
the direct stream URL carried in the reply so the owner can open
the full live stream in their own player.  A dead stream degrades
via source health — the play refuses with the reason named and
tries the next station in the same place — never a silent hang.

Seeds were verified live against the radio-browser directory on
2026-11-18 (every stream URL returned audio bytes on a real GET).
City anchors come from her own airport table
(nomorals/travel/data/airports_coords.csv, municipality rows);
where the directory itself carries a station's coordinates those
are kept as the station's own geo and preferred for distance.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Station",
    "StationStore",
    "RadioBrowserClient",
    "Sample",
    "default_store_path",
    "open_store",
    "normalize_place",
    "haversine_km",
    "stations_in_place",
    "stations_near",
    "stations_by_genre",
    "search_stations",
    "station_by_id",
    "station_by_name",
    "grow_from_directory",
    "check_stream",
    "check_station",
    "check_all",
    "capture_sample",
    "globe_dots",
    "format_station_line",
    "format_radio_reply",
    "parse_radio_request",
    "RADIO_BROWSER_BASE",
    "SEED_STATIONS",
    "SEED_CHECKED",
    "DEFAULT_SAMPLE_SECONDS",
    "SAMPLE_MAX_BYTES",
    "SAMPLE_MAX_SECONDS",
    "HEALTH_FRESH_SECONDS",
]

RADIO_BROWSER_BASE = "https://de1.api.radio-browser.info/json"
SEED_CHECKED = "2026-11-18"

# A live stream has no end, so a play is a bounded sample: 45 s of
# audio by default, never more than 2 MB, never more than 2 minutes
# of reading, whichever bound is hit first.
DEFAULT_SAMPLE_SECONDS = 45
SAMPLE_MAX_BYTES = 2_000_000
SAMPLE_MAX_SECONDS = 120

# Health re-check cadence for a station, matching the platform_api
# volatility class in the music registry (6 h).
HEALTH_FRESH_SECONDS = 6 * 60 * 60

_HEALTH_OK = ("unchecked", "up", "down", "unknown")

_GENRE_ALIASES = {
    "hiphop": "hip hop",
    "hip-hop": "hip hop",
    "r&b": "r&b",
    "rnb": "r&b",
    "dancehall": "dancehall",
}


def normalize_place(text: str) -> str:
    """Fold a place name for matching: casefold, strip accents and
    the Yoruba/Igbo dotted letters (ẹ→e, ọ→o, ṣ→s), collapse space.

    Same folding the maps gazetteer uses, so "Lagos", "lagos" and
    "Lágoṣ" all resolve to the same place.
    """
    folded = unicodedata.normalize("NFKD", (text or "").casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.replace("ẹ", "e").replace("ọ", "o").replace("ṣ", "s")
    return re.sub(r"\s+", " ", folded).strip(" .")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points."""
    r = math.radians
    dlat = r(lat2 - lat1)
    dlon = r(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2
         + math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_place(text)).strip("-")
    return slug or "station"


@dataclass
class Station:
    """One broadcast station: her own stream, indexed by place.

    ``lat``/``lon`` are the CITY anchor (from her airport table, so
    a station with no coordinates of its own still sits on the
    globe at its city).  ``station_lat``/``station_lon`` are the
    coordinates the directory itself carries for the station, when
    it carries any — preferred for distance.  ``health`` is one of
    "unchecked" | "up" | "down" | "unknown".
    """

    name: str
    city: str
    country: str
    lat: float
    lon: float
    stream_url: str
    id: str = ""
    region: str = ""
    country_code: str = ""
    station_lat: float | None = None
    station_lon: float | None = None
    genres: list[str] = field(default_factory=list)
    language: str = ""
    votes: int = 0
    health: str = "unchecked"
    checked_at: float = 0.0
    source: str = "seed"
    uuid: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _slug(f"{self.name} {self.city}")
        if not self.name.strip():
            raise ValueError("station name must not be empty")
        if not self.city.strip():
            raise ValueError("station city must not be empty")
        if not self.country.strip():
            raise ValueError("station country must not be empty")
        for value, label, low, high in (
                (self.lat, "lat", -90.0, 90.0),
                (self.lon, "lon", -180.0, 180.0)):
            if not low <= value <= high:
                raise ValueError(f"station {label} {value} out of range")
        for value, label, low, high in (
                (self.station_lat, "station_lat", -90.0, 90.0),
                (self.station_lon, "station_lon", -180.0, 180.0)):
            if value is not None and not low <= value <= high:
                raise ValueError(f"station {label} {value} out of range")
        if not self.stream_url.startswith(("http://", "https://")):
            raise ValueError("stream_url must be http(s)")
        if self.health not in _HEALTH_OK:
            raise ValueError(f"unknown health {self.health!r}")
        self.genres = [g.strip().lower() for g in self.genres if g.strip()]
        self.votes = int(self.votes)

    @property
    def anchor(self) -> tuple[float, float]:
        """Best coordinates for distance: the station's own geo when
        the directory carries it, else the city anchor."""
        if self.station_lat is not None and self.station_lon is not None:
            return (self.station_lat, self.station_lon)
        return (self.lat, self.lon)

    @property
    def has_coords(self) -> bool:
        """True when the station sits somewhere real on the globe.
        A grown directory row with no coordinates at all anchors at
        (0, 0) — Null Island — and must never be plotted or returned
        by a distance query."""
        if self.station_lat is not None and self.station_lon is not None:
            return True
        return not (self.lat == 0.0 and self.lon == 0.0)

    @property
    def health_stale(self) -> bool:
        if self.health == "unchecked" or not self.checked_at:
            return True
        return (time.time() - self.checked_at) > HEALTH_FRESH_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "city": self.city,
            "region": self.region,
            "country": self.country,
            "country_code": self.country_code,
            "lat": self.lat,
            "lon": self.lon,
            "station_lat": self.station_lat,
            "station_lon": self.station_lon,
            "stream_url": self.stream_url,
            "genres": list(self.genres),
            "language": self.language,
            "votes": self.votes,
            "health": self.health,
            "checked_at": self.checked_at,
            "source": self.source,
            "uuid": self.uuid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Station:
        return cls(
            name=str(data.get("name", "") or ""),
            city=str(data.get("city", "") or ""),
            region=str(data.get("region", "") or ""),
            country=str(data.get("country", "") or ""),
            country_code=str(data.get("country_code", "") or ""),
            lat=float(data.get("lat", 0.0) or 0.0),
            lon=float(data.get("lon", 0.0) or 0.0),
            station_lat=(None if data.get("station_lat") is None
                         else float(data["station_lat"])),
            station_lon=(None if data.get("station_lon") is None
                         else float(data["station_lon"])),
            stream_url=str(data.get("stream_url", "") or ""),
            id=str(data.get("id", "") or ""),
            genres=list(data.get("genres") or []),
            language=str(data.get("language", "") or ""),
            votes=int(data.get("votes", 0) or 0),
            health=str(data.get("health", "unchecked") or "unchecked"),
            checked_at=float(data.get("checked_at", 0.0) or 0.0),
            source=str(data.get("source", "seed") or "seed"),
            uuid=str(data.get("uuid", "") or ""),
        )


def _s(name: str, city: str, country: str, code: str, lat: float, lon: float,
       stream_url: str, genres: list[str], votes: int,
       station_lat: float | None = None, station_lon: float | None = None,
       region: str = "", language: str = "",
       uuid: str = "") -> Station:
    return Station(
        name=name, city=city, region=region or city, country=country,
        country_code=code, lat=lat, lon=lon, stream_url=stream_url,
        genres=genres, votes=votes, station_lat=station_lat,
        station_lon=station_lon, language=language, uuid=uuid,
    )


# Seeds verified live against the radio-browser directory on
# 2026-11-18: every stream URL below returned audio bytes on a
# real GET from this box.  City anchors are the municipality rows
# of her airport table (travel/data/airports_coords.csv); station
# geo is the directory's own geo_lat/geo_long where it carries any.
# Genres are the directory's own tags for the row.
SEED_STATIONS: list[Station] = [
    _s("METRO FM LAGOS", "Lagos", "Nigeria", "NG",
       6.57737, 3.32116,
       "https://go.webgateready.com/metrofm/radio.mp3", [], 2067),
    _s("Wazobia FM: Lagos", "Lagos", "Nigeria", "NG",
       6.57737, 3.32116,
       "https://wazobiafmlagos951-atunwadigital.streamguys1.com/wazobiafmlagos951",
       [], 144),
    _s("YFm Accra", "Accra", "Ghana", "GH",
       5.605189800262451, -0.16678600013256073,
       "https://atunwadigital.streamguys1.com/yfm1079accra",
       ["entertainment", "music", "news", "variety"], 28),
    _s("Sompa 106.5FM Accra", "Accra", "Ghana", "GH",
       5.605189800262451, -0.16678600013256073,
       "https://stream-281.zeno.fm/55sg1lh4pqquv",
       ["highlife", "music", "news", "pop", "talk", "variety"], 23,
       station_lat=5.6424482, station_lon=-0.2435186),
    _s("AFN GO TOKYO", "Tokyo", "Japan", "JP",
       35.549678, 139.786958,
       "http://18433.live.streamtheworld.com:3690/AFNP_TKO_SC",
       ["military"], 235, region="Tokyo"),
    _s("Capital FM London", "London", "United Kingdom", "GB",
       51.470748, -0.459909,
       "http://media-ice.musicradio.com/CapitalMP3", ["pop"], 40598),
    _s("Jazz London Radio", "London", "United Kingdom", "GB",
       51.470748, -0.459909,
       "http://radio.canstream.co.uk:8075/live.mp3", ["jazz"], 12624),
    _s("France Bleu Paris", "Paris", "France", "FR",
       49.00896, 2.554117,
       "http://direct.francebleu.fr/live/fb1071-midfi.mp3",
       ["public radio", "radio france", "regional"], 2094,
       region="Île-de-France"),
    _s("FIP Jazz", "Paris", "France", "FR",
       49.00896, 2.554117,
       "http://icecast.radiofrance.fr/fipjazz-hifi.aac", ["jazz"], 11130),
    _s("WNYC 93.9 FM", "New York", "United States", "US",
       40.639447, -73.779317,
       "http://fm939.wnyc.org/wnycfm.aac", ["news"], 1319,
       station_lat=40.64730356252251, station_lon=-73.98742675781251),
    _s("WWOZ 90.7 New Orleans", "New Orleans", "United States", "US",
       29.993401, -90.264673,
       "http://wwoz-sc.streamguys.com/wwoz-hi.mp3",
       ["blues", "jazz", "new orleans", "non-commercial"], 1519,
       region="Louisiana"),
    _s("Berliner Rundfunk 91.4", "Berlin", "Germany", "DE",
       52.361738, 13.502341,
       "http://stream.berliner-rundfunk.de/brf/mp3-128/internetradio/",
       ["pop"], 4291),
    _s("Radio Maria Kenya - Nairobi", "Nairobi", "Kenya", "KE",
       -1.318886, 36.928233,
       "http://141.95.98.87:8069/stream", [], 275),
    _s("702 Johannesburg", "Johannesburg", "South Africa", "ZA",
       -26.140081, 28.246801,
       "http://25683.live.streamtheworld.com/FM702_SC",
       ["news talk"], 4305, region="Gauteng"),
    _s("Alpha Boys School Radio", "Kingston", "Jamaica", "JM",
       17.935699462890625, -76.7874984741211,
       "http://alphaboys-live.streamguys1.com/alphaboys.mp3",
       ["jazz", "reggae", "ska"], 5399),
    _s("Antena 1 São Paulo", "São Paulo", "Brazil", "BR",
       -23.431274, -46.469954,
       "http://antena1.newradio.it/stream?ext=.mp3",
       ["adult contemporary", "jazz", "pop", "smooth jazz"], 33543),
    _s("Radio DS Mumbai", "Mumbai", "India", "IN",
       19.088699, 72.867897,
       "https://streamasiacdn.atc-labs.com/radiods.aac", ["music"], 135),
    _s("Triple M Sydney", "Sydney", "Australia", "AU",
       -33.946098, 151.177002,
       "http://legacy.scahw.com.au/2mmm_128", [], 635, region="NSW"),
    _s("Radio Hits 88.2 Cairo", "Cairo", "Egypt", "EG",
       30.111534, 31.396694,
       "https://radiohits882.radioca.st/;", ["arabic music"], 3903),
    _s("CFRB News/Talk 1010", "Toronto", "Canada", "CA",
       43.675935, -79.629421,
       "http://26103.live.streamtheworld.com/CFRBAMAAC_SC",
       ["commercial", "local news", "news", "news talk"], 5450,
       region="Ontario"),
    _s("ORF Radio Wien", "Vienna", "Austria", "AT",
       48.110298, 16.5697,
       "http://orf-live.ors-shoutcast.at/wie-q1a", ["orf"], 2954,
       station_lat=48.1776994, station_lon=16.2877707),
    _s("Amsterdam Funk Channel", "Amsterdam", "Netherlands", "NL",
       52.308601, 4.76389,
       "https://stream.afc.fm/",
       ["dutch", "funk"], 220, region="Noord-Holland",
       station_lat=52.35211857272093, station_lon=4.833984375000001),
    _s("SomaFM Groove Salad", "San Francisco", "United States", "US",
       37.619806, -122.374821,
       "https://ice5.somafm.com/groovesalad-128-mp3",
       ["ambient", "chillout", "downtempo", "groove", "lounge"], 47830,
       region="California",
       station_lat=37.809784, station_lon=-121.286316),
    _s("Radio Paradise Main Mix", "Eureka", "United States", "US",
       40.785114, -124.183922,
       "http://stream-uk1.radioparadise.com/aac-320",
       ["eclectic"], 323472, region="California",
       station_lat=40.785114, station_lon=-124.183922),
]


def default_store_path(home: str | Path | None = None) -> Path:
    """The durable store: <home>/data/radio_stations.json."""
    base = Path(home) if home else Path.home() / ".nomorals"
    return base / "data" / "radio_stations.json"


class StationStore:
    """JSON station index. Atomic writes; a corrupt store is
    quarantined aside and re-seeded, never silently emptied."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stations: dict[str, Station] = {}
        self._load()

    # -- persistence -------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            self._seed()
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            rows = raw.get("stations", []) if isinstance(raw, dict) else []
            for row in rows:
                station = Station.from_dict(row)
                self._stations[station.id] = station
        except Exception:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            quarantine = self.path.with_name(
                f"{self.path.name}.corrupt-{stamp}")
            with contextlib.suppress(OSError):
                self.path.rename(quarantine)
            self._stations = {}
            self._seed()

    def _seed(self) -> None:
        for station in SEED_STATIONS:
            self._stations.setdefault(station.id, station)
        self.save()

    def save(self) -> None:
        payload = {
            "checked": SEED_CHECKED,
            "stations": [s.to_dict() for s in self._stations.values()],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.path)

    # -- registry ----------------------------------------------------
    def add(self, station: Station) -> bool:
        """Add a station; False when its id or stream URL is already
        indexed (discovery never duplicates a station)."""
        if station.id in self._stations:
            return False
        if any(s.stream_url.rstrip("/") == station.stream_url.rstrip("/")
               for s in self._stations.values()):
            return False
        self._stations[station.id] = station
        self.save()
        return True

    def remove(self, station_id: str) -> bool:
        if station_id not in self._stations:
            return False
        del self._stations[station_id]
        self.save()
        return True

    def update(self, station: Station) -> None:
        self._stations[station.id] = station
        self.save()

    def get(self, station_id: str) -> Station | None:
        return self._stations.get(station_id)

    def all(self) -> list[Station]:
        return sorted(self._stations.values(),
                      key=lambda s: (-s.votes, s.name.casefold()))

    def count(self) -> int:
        return len(self._stations)

    def health_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {"up": 0, "down": 0, "unknown": 0,
                                   "unchecked": 0}
        for station in self._stations.values():
            summary[station.health] = summary.get(station.health, 0) + 1
        return summary


def open_store(home: str | Path | None = None) -> StationStore:
    """Open (creating and seeding when missing) the station store."""
    return StationStore(default_store_path(home))


# --------------------------------------------------------------------
# Queries: place-first browsing
# --------------------------------------------------------------------

def _genre_matches(query: str, genres: list[str]) -> bool:
    want = _GENRE_ALIASES.get(query.strip().lower(),
                              query.strip().lower())
    if not want:
        return True
    for genre in genres:
        genre = genre.lower()
        if want in genre or genre in want:
            return True
    return False


def _place_rank(query: str, station: Station) -> int:
    """0 = exact city, 1 = exact region, 2 = exact country,
    3 = substring of city/region/country, -1 = no match."""
    q = normalize_place(query)
    if not q:
        return -1
    if normalize_place(station.city) == q:
        return 0
    if station.region and normalize_place(station.region) == q:
        return 1
    if normalize_place(station.country) == q:
        return 2
    for haystack in (station.city, station.region, station.country):
        if q in normalize_place(haystack):
            return 3
    return -1


def stations_in_place(store: StationStore, place: str,
                      *, genre: str | None = None,
                      limit: int = 20) -> list[Station]:
    """Stations for a place, best match first.  An empty list is an
    honest answer: the index does not have that place yet."""
    ranked: list[tuple[int, int, Station]] = []
    for station in store.all():
        rank = _place_rank(place, station)
        if rank < 0:
            continue
        if genre and not _genre_matches(genre, station.genres):
            continue
        ranked.append((rank, -station.votes, station))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [station for _, _, station in ranked[:limit]]


def stations_near(store: StationStore, lat: float, lon: float,
                  *, radius_km: float = 100.0,
                  genre: str | None = None,
                  limit: int = 20) -> list[Station]:
    """Stations within ``radius_km`` of a point, nearest first,
    measured on each station's best coordinates."""
    scored: list[tuple[float, Station]] = []
    for station in store.all():
        if not station.has_coords:
            continue
        alat, alon = station.anchor
        distance = haversine_km(lat, lon, alat, alon)
        if distance > radius_km:
            continue
        if genre and not _genre_matches(genre, station.genres):
            continue
        scored.append((distance, station))
    scored.sort(key=lambda item: item[0])
    return [station for _, station in scored[:limit]]


def stations_by_genre(store: StationStore, genre: str,
                      *, limit: int = 20) -> list[Station]:
    """Stations carrying a genre, most-voted first."""
    found = [s for s in store.all() if _genre_matches(genre, s.genres)]
    return found[:limit]


def search_stations(store: StationStore, *, place: str | None = None,
                    genre: str | None = None,
                    lat: float | None = None, lon: float | None = None,
                    radius_km: float = 100.0,
                    limit: int = 20) -> list[Station]:
    """Combined query: place and genre intersect; a lat/lon narrows
    by distance instead of by place name.  No filter at all lists
    the index, most-voted first."""
    if lat is not None and lon is not None:
        return stations_near(store, lat, lon, radius_km=radius_km,
                             genre=genre, limit=limit)
    if place:
        return stations_in_place(store, place, genre=genre, limit=limit)
    if genre:
        return stations_by_genre(store, genre, limit=limit)
    return store.all()[:limit]


def station_by_id(store: StationStore, station_id: str) -> Station | None:
    return store.get(station_id)


def station_by_name(store: StationStore, name: str) -> Station | None:
    want = normalize_place(name)
    for station in store.all():
        if normalize_place(station.name) == want:
            return station
    for station in store.all():
        if want and want in normalize_place(station.name):
            return station
    return None


# --------------------------------------------------------------------
# The globe, as data: one dot per city, sized by broadcaster count
# --------------------------------------------------------------------

def globe_dots(store: StationStore) -> list[dict[str, Any]]:
    """Dots for the globe view: one per city, with its coordinates
    and how many broadcasters sit there."""
    dots: dict[str, dict[str, Any]] = {}
    for station in store.all():
        if not station.has_coords:
            continue
        key = normalize_place(f"{station.city}|{station.country}")
        dot = dots.get(key)
        if dot is None:
            dots[key] = {
                "city": station.city,
                "country": station.country,
                "lat": station.lat,
                "lon": station.lon,
                "count": 1,
                "stations": [station.name],
            }
        else:
            dot["count"] += 1
            dot["stations"].append(station.name)
    return sorted(dots.values(), key=lambda d: (-d["count"], d["city"]))


# --------------------------------------------------------------------
# Radio-browser directory (open community database)
# --------------------------------------------------------------------

def _default_fetch(url: str, timeout_s: float = 20.0) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Devon/1.0 (radio directory)"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


class RadioBrowserClient:
    """Thin client over the radio-browser community API.

    Keyless, public, community-run.  A fetcher can be injected for
    tests; the default does a real GET.  Every call returns [] on
    failure and records ``last_error`` — the directory being down
    is reported, never papered over.
    """

    def __init__(self, fetch_fn: Callable[[str], str] | None = None,
                 base: str = RADIO_BROWSER_BASE) -> None:
        self._fetch = fetch_fn or _default_fetch
        self.base = base.rstrip("/")
        self.last_error: str = ""

    def _get(self, path: str) -> list[dict[str, Any]]:
        try:
            raw = self._fetch(self.base + path)
            rows = json.loads(raw)
            if not isinstance(rows, list):
                self.last_error = "directory returned a non-list"
                return []
            return [row for row in rows if isinstance(row, dict)]
        except Exception as exc:  # noqa: BLE001 - directory failure is data
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []

    def search(self, name: str, *, limit: int = 20) -> list[Station]:
        import urllib.parse
        path = ("/stations/search?name="
                + urllib.parse.quote(name)
                + f"&limit={limit}&order=votes&reverse=true&hidebroken=true")
        return [self.row_to_station(row) for row in self._get(path)
                if self.row_to_station(row) is not None]

    def by_country(self, code: str, *, limit: int = 50) -> list[Station]:
        path = (f"/stations/bycountrycodeexact/{code.upper()}"
                f"?limit={limit}&order=votes&reverse=true&hidebroken=true")
        return [self.row_to_station(row) for row in self._get(path)
                if self.row_to_station(row) is not None]

    def by_tag(self, tag: str, *, limit: int = 50) -> list[Station]:
        import urllib.parse
        path = ("/stations/bytag/" + urllib.parse.quote(tag)
                + f"?limit={limit}&order=votes&reverse=true&hidebroken=true")
        return [self.row_to_station(row) for row in self._get(path)
                if self.row_to_station(row) is not None]

    def top_voted(self, *, limit: int = 50) -> list[Station]:
        path = (f"/stations/topvote/{limit}"
                "?order=votes&reverse=true&hidebroken=true")
        return [self.row_to_station(row) for row in self._get(path)
                if self.row_to_station(row) is not None]

    def row_to_station(self, row: dict[str, Any]) -> Station | None:
        """Map one directory row to a Station.  None when the row
        has no usable stream URL or no place at all — the directory
        is noisy (a city name matches station names anywhere, and
        most rows carry no coordinates), so the mapping is strict
        about what it accepts and honest about what it drops."""
        url = (row.get("url_resolved") or row.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return None
        name = (row.get("name") or "").strip()
        if not name:
            return None
        country = (row.get("country") or "").strip()
        region = (row.get("state") or "").strip()
        if not region and not country:
            # A station with no place at all cannot sit on the globe.
            return None
        # The directory carries no city field of its own; the state
        # is the finest place it gives, and it is often empty.  The
        # city is therefore the finest place the directory gives —
        # state when it has one, else the country — and nothing is
        # invented about where the station sits.  The place query
        # matches city, region and country alike, so the station
        # stays findable either way.
        city = region or country
        lat = row.get("geo_lat")
        lon = row.get("geo_long")
        try:
            station_lat = float(lat) if lat is not None else None
            station_lon = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            station_lat = station_lon = None
        if (station_lat is not None and station_lon is not None
                and not (-90.0 <= station_lat <= 90.0
                         and -180.0 <= station_lon <= 180.0)):
            station_lat = station_lon = None
        tags = [t.strip().lower()
                for t in str(row.get("tags") or "").split(",") if t.strip()]
        return Station(
            name=name,
            city=city,
            region=region,
            country=country,
            country_code=str(row.get("countrycode") or ""),
            lat=station_lat if station_lat is not None else 0.0,
            lon=station_lon if station_lon is not None else 0.0,
            station_lat=station_lat,
            station_lon=station_lon,
            stream_url=url,
            genres=tags,
            language=str(row.get("language") or ""),
            votes=int(row.get("votes") or 0),
            source="radio-browser",
            uuid=str(row.get("stationuuid") or ""),
        )


def grow_from_directory(store: StationStore, *,
                        client: RadioBrowserClient | None = None,
                        limit: int = 50) -> dict[str, Any]:
    """Grow the index from the open directory: the most-voted
    stations, which are the ones listeners actually use.  New
    stations only — discovery never duplicates one already indexed.
    Returns the counts, and the directory's error when it failed."""
    client = client or RadioBrowserClient()
    grown = client.top_voted(limit=limit)
    if not grown and client.last_error:
        return {"added": 0, "seen": 0, "error": client.last_error}
    added = 0
    for station in grown:
        if store.add(station):
            added += 1
    return {"added": added, "seen": len(grown), "error": ""}


# --------------------------------------------------------------------
# Health: a dead stream degrades, never hangs
# --------------------------------------------------------------------

def _default_probe(url: str, timeout_s: float = 8.0
                   ) -> tuple[int, str, int, float]:
    started = time.monotonic()
    req = urllib.request.Request(
        url, headers={"User-Agent": "Devon/1.0 (radio health)",
                      "Icy-MetaData": "0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        chunk = resp.read(16384)
        return (resp.status, resp.headers.get_content_type() or "",
                len(chunk), time.monotonic() - started)


def check_stream(station: Station, *,
                 read_fn: Callable[[str], tuple[int, str, int, float]]
                 | None = None,
                 timeout_s: float = 8.0) -> tuple[str, str]:
    """Probe a stream: read the first 16 KiB and say whether it
    answers.  Returns ("up"|"down", detail).  Never raises — a
    failing probe is a "down" verdict with the reason named."""
    probe = read_fn or (lambda url: _default_probe(url, timeout_s))
    try:
        status, content_type, n_bytes, elapsed = probe(station.stream_url)
    except Exception as exc:  # noqa: BLE001 - probe failure is data
        return ("down", f"{type(exc).__name__}: {exc}")
    if status != 200:
        return ("down", f"HTTP {status}")
    if n_bytes == 0:
        return ("down", "stream answered but sent no bytes")
    return ("up", f"HTTP 200 {content_type}, {n_bytes} bytes "
                  f"in {elapsed:.1f}s")


def check_station(store: StationStore, station_id: str, *,
                  read_fn: Callable[[str], tuple[int, str, int, float]]
                  | None = None) -> tuple[str, str] | None:
    """Probe one station and persist the verdict.  None when the
    station is not in the index."""
    station = store.get(station_id)
    if station is None:
        return None
    status, detail = check_stream(station, read_fn=read_fn)
    station.health = status
    station.checked_at = time.time()
    store.update(station)
    return (status, detail)


def check_all(store: StationStore, *,
              read_fn: Callable[[str], tuple[int, str, int, float]]
              | None = None) -> dict[str, Any]:
    """Probe every station in the index.  One dead stream never
    stops the sweep."""
    up = down = 0
    failures: list[str] = []
    for station in store.all():
        status, detail = check_stream(station, read_fn=read_fn)
        station.health = status
        station.checked_at = time.time()
        store.update(station)
        if status == "up":
            up += 1
        else:
            down += 1
            failures.append(f"{station.name}: {detail}")
    return {"up": up, "down": down, "failures": failures}


# --------------------------------------------------------------------
# Play: a bounded live sample of the stream
# --------------------------------------------------------------------

@dataclass
class Sample:
    ok: bool
    path: str = ""
    n_bytes: int = 0
    seconds: float = 0.0
    content_type: str = ""
    error: str = ""
    station_id: str = ""


def _extension_for(first_bytes: bytes, content_type: str) -> str:
    if first_bytes.startswith(b"ID3"):
        return ".mp3"
    if len(first_bytes) >= 2 and first_bytes[0] == 0xFF:
        if first_bytes[1] in (0xFB, 0xF3, 0xF2):
            return ".mp3"
        if first_bytes[1] in (0xF1, 0xF9):
            return ".aac"
    if first_bytes.startswith(b"OggS"):
        return ".ogg"
    if first_bytes.startswith(b"fLaC"):
        return ".flac"
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/aac": ".aac",
        "audio/aacp": ".aac",
        "audio/ogg": ".ogg",
        "audio/x-wav": ".wav",
        "audio/wav": ".wav",
    }
    return mapping.get((content_type or "").lower().split(";")[0].strip(),
                       ".mp3")


def _default_stream(url: str, timeout_s: float = 10.0):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Devon/1.0 (radio sample)",
                      "Icy-MetaData": "0"})
    return urllib.request.urlopen(req, timeout=timeout_s)


def capture_sample(station: Station, *, seconds: float = DEFAULT_SAMPLE_SECONDS,
                   max_bytes: int = SAMPLE_MAX_BYTES,
                   dest_dir: str | Path | None = None,
                   read_fn: Callable[[str], Any] | None = None) -> Sample:
    """Record a bounded live sample of a stream: at most ``seconds``
    of reading and at most ``max_bytes`` of audio, whichever comes
    first.  The sample lands as a file under <home>/radio_samples/
    (or ``dest_dir``) and is what the play spine sends; the reply
    always carries the direct stream URL too, so the owner can open
    the full live stream in their own player.  Never reads
    unbounded — a stream that never ends still stops at the cap."""
    seconds = min(max(1.0, float(seconds)), float(SAMPLE_MAX_SECONDS))
    max_bytes = min(max(1, int(max_bytes)), SAMPLE_MAX_BYTES)
    dest = Path(dest_dir) if dest_dir else (
        Path.home() / ".nomorals" / "radio_samples")
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    try:
        resp = (read_fn(station.stream_url) if read_fn is not None
                else _default_stream(station.stream_url))
    except Exception as exc:  # noqa: BLE001 - stream failure is data
        return Sample(ok=False, station_id=station.id,
                      error=f"{type(exc).__name__}: {exc}")
    content_type = ""
    try:
        headers = getattr(resp, "headers", None)
        if headers is not None:
            content_type = headers.get_content_type() or ""
    except Exception:  # noqa: BLE001 - header failure is data
        content_type = ""
    started = time.monotonic()
    written = 0
    first = b""
    path: Path | None = None
    try:
        with resp as stream:
            chunk = stream.read(32768)
            first = chunk[:8]
            ext = _extension_for(first, content_type)
            path = dest / f"{station.id}-{stamp}{ext}"
            with path.open("wb") as handle:
                while chunk:
                    take = min(len(chunk), max_bytes - written)
                    handle.write(chunk[:take])
                    written += take
                    if written >= max_bytes:
                        break
                    if time.monotonic() - started >= seconds:
                        break
                    chunk = stream.read(32768)
    except Exception as exc:  # noqa: BLE001 - stream failure is data
        if path is not None and path.exists():
            with contextlib.suppress(OSError):
                path.unlink()
        return Sample(ok=False, station_id=station.id,
                      error=f"{type(exc).__name__}: {exc}")
    if written == 0:
        if path is not None and path.exists():
            with contextlib.suppress(OSError):
                path.unlink()
        return Sample(ok=False, station_id=station.id,
                      error="stream sent no bytes")
    return Sample(ok=True, path=str(path), n_bytes=written,
                  seconds=round(time.monotonic() - started, 1),
                  content_type=content_type or "audio/mpeg",
                  station_id=station.id)


# --------------------------------------------------------------------
# Replies and natural-language parsing
# --------------------------------------------------------------------

def format_station_line(station: Station) -> str:
    genres = ", ".join(station.genres) if station.genres else "radio"
    health = {"up": "live", "down": "down", "unknown": "unverified",
              "unchecked": "unverified"}.get(station.health, "unverified")
    return (f"• {station.name} — {station.city}, {station.country} "
            f"({genres}) [{health}]")


def format_radio_reply(stations: list[Station], *, place: str = "",
                       genre: str = "") -> str:
    """The reply for a place/genre listing.  An empty index for a
    place says so honestly — it never invents a station."""
    if not stations:
        what = f"{genre} stations in {place}" if genre and place else (
            f"stations in {place}" if place else
            f"{genre} stations" if genre else "stations")
        return (f"📻 I don't have {what} in my radio index yet — "
                "the index grows from the open radio directory, and "
                "this place isn't in it. Try another city, or ask me "
                "to grow the index.")
    title = place or "radio"
    if genre:
        title = f"{genre} in {title}"
    lines = [f"📻 Radio in {title} — {len(stations)} station(s):"]
    lines += [format_station_line(s) for s in stations]
    lines.append("Say 'play <station name>' to hear a live sample.")
    return "\n".join(lines)


_PLAY_PLACE = re.compile(
    r"^(?:play|listen to|tune in to|tune into)\s+(?:a\s+|the\s+)?"
    r"(?:radio\s+station|radio|station)\s+from\s+(.+)$", re.IGNORECASE)
_PLAY_GENRE_PLACE = re.compile(
    r"^(?:play|listen to)\s+(.+?)\s+radio\s+from\s+(.+)$", re.IGNORECASE)
_WHATS_ON = re.compile(
    r"^(?:what'?s on|what is playing|stations?|radio|what'?s playing)"
    r"\s+(?:in|from|on)\s+(.+)$", re.IGNORECASE)
_GENRE_IN_PLACE = re.compile(
    r"^(.+?)\s+(?:stations?|radio)\s+in\s+(.+)$", re.IGNORECASE)
_RADIO_IN = re.compile(
    r"^radio\s+(?:in|from)\s+(.+)$", re.IGNORECASE)
_GENRE_ONLY = re.compile(r"^(.+?)\s+radio$", re.IGNORECASE)


def parse_radio_request(text: str, stations: list[Station]
                        ) -> dict[str, Any] | None:
    """Parse a radio request.  Returns {"intent": "play"|"stations",
    "place": ..., "genre": ..., "station_id": ...} or None when the
    text is not a radio request at all.  A bare "play <something>"
    only parses when <something> names a station in the index — so
    "play despacito" never becomes a radio play."""
    cleaned = (text or "").strip().rstrip("?!.").strip()
    if not cleaned:
        return None
    match = _PLAY_GENRE_PLACE.match(cleaned)
    if match:
        return {"intent": "play", "genre": match.group(1).strip(),
                "place": match.group(2).strip(), "station_id": ""}
    match = _PLAY_PLACE.match(cleaned)
    if match:
        return {"intent": "play", "genre": "",
                "place": match.group(1).strip(), "station_id": ""}
    match = _WHATS_ON.match(cleaned)
    if match:
        return {"intent": "stations", "genre": "",
                "place": match.group(1).strip(), "station_id": ""}
    match = _GENRE_IN_PLACE.match(cleaned)
    if match:
        return {"intent": "stations", "genre": match.group(1).strip(),
                "place": match.group(2).strip(), "station_id": ""}
    match = _RADIO_IN.match(cleaned)
    if match:
        return {"intent": "stations", "genre": "",
                "place": match.group(1).strip(), "station_id": ""}
    match = _GENRE_ONLY.match(cleaned)
    if match and len(cleaned.split()) <= 3:
        # "play the radio" is not a genre request — the play-name
        # branch below owns anything starting with "play".
        if cleaned.lower().startswith(("play ", "listen to ",
                                        "tune in ")):
            pass
        else:
            return {"intent": "stations",
                    "genre": match.group(1).strip(),
                    "place": "", "station_id": ""}
    # "play <station name>" — only when the name is in the index.
    lowered = cleaned.lower()
    if lowered.startswith("play "):
        want = normalize_place(cleaned[5:].strip())
        if want.endswith(" radio"):
            # "play jazz radio" — a genre, not a station name.
            genre = want[:-len(" radio")].strip()
            if genre:
                for station in stations:
                    if genre in [g.lower() for g in station.genres] or any(
                            genre in g.lower() for g in station.genres):
                        return {"intent": "play", "genre": genre,
                                "place": "", "station_id": station.id}
        for station in stations:
            if normalize_place(station.name) == want:
                return {"intent": "play", "genre": "", "place": "",
                        "station_id": station.id}
        for station in stations:
            if want and want in normalize_place(station.name):
                return {"intent": "play", "genre": "", "place": "",
                        "station_id": station.id}
    return None
