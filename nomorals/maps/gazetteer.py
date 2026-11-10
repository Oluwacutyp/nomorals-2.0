"""The local geocoder: a gazetteer indexed from Devon's own packs.

Doc 21 S3 superseded doc 19's hosted-geocoder posture: the
geocoder is **local first** — an index Devon builds from the packs
she already holds — and federation to an outside provider happens
only when owner policy permits (``MapSettings.geocode_federation``,
default OFF).  This module is that index and its two queries.

**The index.**  A pack's vector tiles carry the named world: the
``place`` and ``poi`` layers hold a feature per named place, with
its name, its class and sometimes its population as tags.  The
gazetteer decodes those layers (M3's decoder) into one row per
name form — the canonical ``name`` plus every ``name:*`` variant,
so a place is findable in the languages its mappers recorded —
each row carrying the feature's coordinate, converted from tile
space through the Web-Mercator inverse.  The same place sighted
at several zooms collapses to one row — the coarsest (most
prominent) sighting wins — where "same" means same folded name
within ~110 m: tile quantization shifts a label's decoded
coordinate per zoom, so identity is proximity, never exact
float equality.  Storage is a sqlite file in the packs
directory: derived data, rebuildable at any time, keyed per pack
by the pack digest (M1's contract) so an unchanged pack is never
re-scanned and a changed pack always is.  Raster (png) packs
carry no feature data and index to zero rows — stated in the
build result, never faked.

**The scan discipline (stated, because it is the cost).**  Places
live at zooms 0–14 in the planetiler/OpenMapTiles schema the packs
are built in: countries and cities enter at the top, villages and
POIs arrive by z14, and deeper tiles only repeat the same labels
at finer grain.  The scan therefore reads in-band members only
(``tiles/{z}/…`` with ``z <= GAZETTEER_MAX_ZOOM``), decodes each
once, and keeps only the place/poi layers — one pass, no
re-reads.  A member that will not decode is counted and named as
corrupt in the build receipt; it never fails the index.

**Matching.**  Names compare normalized: casefolded, NFKD-folded
with combining marks stripped — so ``Ìláwé`` matches ``ilawe``
and ``Ìláwé`` alike, and the Yoruba underdot letters (ẹ/ọ/ṣ) fold
to e/o/s.  Ranking is lexicographic by design: match class first
(exact > prefix > word-boundary > substring), then kind weight
(city above town above village above suburb above POI classes),
then population where the data carries it.  No match is an honest
empty result — never a guessed coordinate.

**Reverse.**  Named places are points, so reverse geocoding is
nearest-by-distance over a stated ladder (1/5/25/100 km), the
distance printed on every candidate; beyond the widest rung the
answer is "no named place within 100 km", not the nearest thing
on Earth.

**Federation.**  When it is off, the local answer — found or
empty — is final, and the result says so.  When it is on and the
local index has no answer, the query falls to a
:class:`GeocodeProvider` if one is wired; none ships in this
pass (wiring a real provider is a separate owner decision), so
the honest result names that: federation on, no provider
configured.  The package stays network-free either way — the
no-network AST pin covers this module.
"""

from __future__ import annotations

import gzip
import math
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..core.failures import record_failure
from ..location.primitives import LatLon, haversine_m
from .manifest import PackManifest
from .mvt import GEOM_POINT, MvtError, decode_mvt
from .receipts import (
    OP_GEOCODE,
    OUTCOME_OK,
    OUTCOME_UNAVAILABLE,
    POSTURE_OFFLINE,
    POSTURE_ONLINE,
    MapReceipt,
    MapReceiptStore,
)

__all__ = [
    "GAZETTEER_MAX_ZOOM", "INDEX_FILENAME", "KIND_WEIGHTS",
    "POSTURE_FEDERATED", "POSTURE_LOCAL", "REVERSE_LADDER_KM",
    "Gazetteer", "GazetteerEntry", "GeocodeCandidate",
    "GeocodeProvider", "GeocodeResult", "IndexBuildResult",
    "PackIndexResult", "ReverseCandidate", "ReverseResult",
    "match_rank", "normalize_name",
]

#: The geocoder's answer came from Devon's own index / a provider.
POSTURE_LOCAL = "local"
POSTURE_FEDERATED = "federated"

#: The index file's name inside the packs directory.
INDEX_FILENAME = "gazetteer.sqlite3"

#: Deepest zoom the index scan reads (see the module docstring:
#: place/POI features live at z0–z14 in the pack schema; deeper
#: tiles repeat the same labels).
GAZETTEER_MAX_ZOOM = 14

#: Source layers the index reads: the style's place vocabulary
#: (``place`` + its aliases) and the ``poi`` layer.
PLACE_LAYERS = ("place", "place_label", "places")
POI_LAYERS = ("poi",)
INDEX_LAYERS = PLACE_LAYERS + POI_LAYERS

#: The reverse-geocode distance ladder, kilometres.  The widest
#: rung bounds the search; answers state their distances.
REVERSE_LADDER_KM = (1.0, 5.0, 25.0, 100.0)

#: Kind weights for ranking: settlements in the doc's order, then
#: areas, then everything else (POI classes land at the default).
KIND_WEIGHTS = {
    "continent": 65, "country": 60, "state": 55, "city": 50,
    "town": 42, "village": 34, "hamlet": 28, "island": 24,
    "suburb": 22, "neighbourhood": 20, "neighborhood": 20,
    "quarter": 18, "place": 15, "poi": 10,
}
_DEFAULT_KIND_WEIGHT = 8

#: Match-class ranks and their spoken names.  The gap between
#: bases exceeds any kind weight, so classes never bleed.
_MATCH_BASES = {4: 400.0, 3: 300.0, 2: 200.0, 1: 100.0}
_MATCH_NAMES = {4: "exact", 3: "prefix", 2: "word", 1: "substring"}

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS gazetteer_packs ("
    " pack_id TEXT PRIMARY KEY, digest TEXT NOT NULL,"
    " built_at REAL NOT NULL, entries INTEGER NOT NULL,"
    " tiles_scanned INTEGER NOT NULL,"
    " corrupt_tiles INTEGER NOT NULL);"
    "CREATE TABLE IF NOT EXISTS gazetteer_entries ("
    " pack_id TEXT NOT NULL, name TEXT NOT NULL,"
    " indexed_name TEXT NOT NULL, norm TEXT NOT NULL,"
    " kind TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,"
    " source_zoom INTEGER NOT NULL,"
    " population INTEGER NOT NULL DEFAULT 0,"
    " PRIMARY KEY (pack_id, norm, lat, lon));"
    "CREATE INDEX IF NOT EXISTS gazetteer_norm_idx"
    " ON gazetteer_entries(norm);"
    "CREATE INDEX IF NOT EXISTS gazetteer_pack_idx"
    " ON gazetteer_entries(pack_id);"
)


# ── name handling ─────────────────────────────────────────────────


def normalize_name(text: str) -> str:
    """Fold a name for comparison: case, accents, punctuation.

    Casefolded, NFKD-decomposed with every combining mark
    stripped (Ìláwé → ilawe; ẹ/ọ/ṣ → e/o/s), non-alphanumerics
    collapsed to single spaces.  Two names that fold equal are
    the same name to the geocoder.
    """
    decomposed = unicodedata.normalize(
        "NFKD", str(text).casefold())
    stripped = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = "".join(
        ch if ch.isalnum() else " " for ch in stripped)
    return " ".join(cleaned.split())


def match_rank(norm_query: str, norm_name: str) -> int:
    """How ``norm_query`` matches ``norm_name``: 4 exact, 3 prefix,
    2 at a word boundary, 1 as a bare substring, 0 no match."""
    if not norm_query or not norm_name:
        return 0
    if norm_name == norm_query:
        return 4
    if norm_name.startswith(norm_query):
        return 3
    if (" " + norm_query) in (" " + norm_name):
        return 2
    if norm_query in norm_name:
        return 1
    return 0


def _kind_weight(kind: str) -> int:
    return KIND_WEIGHTS.get(kind, _DEFAULT_KIND_WEIGHT)


def _score(rank: int, kind: str, population: int) -> float:
    """A display score that mirrors the lexicographic ranking."""
    frac = 0.0
    if population > 0:
        frac = min(0.99, math.log10(population + 1) / 7.0)
    return _MATCH_BASES[rank] + _kind_weight(kind) + frac


# ── typed values ──────────────────────────────────────────────────


@dataclass(frozen=True)
class GazetteerEntry:
    """One indexed name form of one named place in one pack."""

    name: str            # the canonical display name
    indexed_name: str    # the name form this row indexes
    kind: str
    lat: float
    lon: float
    pack_id: str
    source_zoom: int
    population: int = 0


@dataclass(frozen=True)
class PackIndexResult:
    """What indexing one pack did (or honestly did not do)."""

    pack_id: str
    digest: str
    entries: int = 0
    tiles_scanned: int = 0
    corrupt_tiles: tuple[str, ...] = ()
    skipped: bool = False       # digest unchanged — a true no-op
    note: str = ""


@dataclass(frozen=True)
class IndexBuildResult:
    """A whole build/ensure pass over the installed shelf."""

    packs: tuple[PackIndexResult, ...] = ()
    total_entries: int = 0
    built_any: bool = False
    receipt: MapReceipt | None = None


@dataclass(frozen=True)
class GeocodeCandidate:
    """One forward-geocode answer, ranked."""

    name: str
    kind: str
    lat: float
    lon: float
    pack_id: str
    score: float
    matched_as: str      # exact | prefix | word | substring | provider


@dataclass(frozen=True)
class ReverseCandidate:
    """One reverse-geocode answer, with its distance stated."""

    name: str
    kind: str
    lat: float
    lon: float
    pack_id: str
    distance_km: float


@dataclass(frozen=True)
class GeocodeResult:
    """A forward query's whole answer: candidates + posture."""

    query: str
    candidates: tuple[GeocodeCandidate, ...] = ()
    posture: str = POSTURE_LOCAL
    provider: str = "local"
    note: str = ""
    receipt: MapReceipt | None = None


@dataclass(frozen=True)
class ReverseResult:
    """A reverse query's whole answer: candidates + posture."""

    lat: float = 0.0
    lon: float = 0.0
    candidates: tuple[ReverseCandidate, ...] = ()
    searched_km: float = REVERSE_LADDER_KM[-1]
    rung_km: float = REVERSE_LADDER_KM[-1]
    posture: str = POSTURE_LOCAL
    provider: str = "local"
    note: str = ""
    receipt: MapReceipt | None = None


class GeocodeProvider(Protocol):
    """The federation seam (doc 21 S3): an outside geocoder.

    Implementations answer with the same candidate types the
    local index returns.  No provider ships in this pass — the
    seam, the owner's setting, and the honest "no provider
    configured" answer are the deliverable; a real provider is
    wired by a later, separately-decided pass.
    """

    name: str

    def search(self, query: str, *,
               limit: int = 8) -> tuple[GeocodeCandidate, ...]:
        ...

    def reverse(self, lat: float, lon: float, *,
                limit: int = 8) -> tuple[ReverseCandidate, ...]:
        ...


# ── tile → entry extraction ───────────────────────────────────────


def _tile_point_to_latlon(z: int, x: int, y: int, extent: int,
                          px: float, py: float) -> tuple[float, float]:
    """One tile-space point → (lat, lon), the Mercator inverse."""
    fx = (x + px / float(extent)) / (1 << z)
    fy = (y + py / float(extent)) / (1 << z)
    lon = fx * 360.0 - 180.0
    lat = math.degrees(
        math.atan(math.sinh(math.pi * (1.0 - 2.0 * fy))))
    return lat, lon


def _feature_anchor(feature) -> tuple[float, float] | None:
    """The tile-space anchor of a feature, or None if it has none.

    Points anchor at their first point; lines and polygons at the
    centroid of their first path's points (a place label's anchor
    is what a gazetteer wants either way).
    """
    if not feature.paths:
        return None
    points = feature.paths[0].points
    if not points:
        return None
    if feature.geom_type == GEOM_POINT:
        return float(points[0][0]), float(points[0][1])
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return sx / len(points), sy / len(points)


def _population(tags: dict) -> int:
    raw = tags.get("population")
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, (int, float)):
        return max(0, int(raw))
    if isinstance(raw, str):
        try:
            return max(0, int(float(
                raw.replace(",", "").strip())))
        except ValueError:
            return 0
    return 0


def _kind_of(layer_name: str, tags: dict) -> str:
    klass = tags.get("class") or tags.get("subclass")
    if klass:
        return str(klass)
    return "poi" if layer_name in POI_LAYERS else "place"


def _name_forms(tags: dict) -> list[str]:
    """The canonical ``name`` plus every ``name:*`` variant."""
    forms: list[str] = []
    canonical = tags.get("name")
    if canonical:
        forms.append(str(canonical))
    for key in sorted(tags):
        if key.startswith("name:"):
            value = tags[key]
            if value and str(value) not in forms:
                forms.append(str(value))
    return forms


def _parse_tile_member(path: str) -> tuple[int, int, int] | None:
    """``tiles/{z}/{x}/{y}.<fmt>`` → (z, x, y), else None."""
    parts = path.split("/")
    if len(parts) != 4 or parts[0] != "tiles":
        return None
    try:
        return int(parts[1]), int(parts[2]), int(parts[3].split(".")[0])
    except ValueError:
        return None


# ── the gazetteer ─────────────────────────────────────────────────


class Gazetteer:
    """The local geocoder over the packs in one packs directory."""

    def __init__(self, packs_dir: Path | str, *,
                 store: MapReceiptStore | None = None,
                 provider: GeocodeProvider | None = None,
                 federation: bool = False) -> None:
        self._root = Path(packs_dir)
        self._store = store
        self._provider = provider
        self._federation = bool(federation)

    @property
    def index_path(self) -> Path:
        return self._root / INDEX_FILENAME

    # ── sqlite plumbing ─────────────────────────────────────────

    def _connect(self, *, create: bool) -> sqlite3.Connection | None:
        """Open the index, or None when read-only and absent.

        A corrupt index file is deleted and rebuilt: the index is
        derived data and a fresh build is always its true state.
        """
        if not create and not self.index_path.is_file():
            return None
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(self.index_path))
            conn.executescript(_SCHEMA)
            conn.commit()
            return conn
        except sqlite3.DatabaseError:
            try:
                self.index_path.unlink()
            except OSError as exc:
                record_failure("maps.gazetteer_index_reset", exc)
            conn = sqlite3.connect(str(self.index_path))
            conn.executescript(_SCHEMA)
            conn.commit()
            return conn

    def _recorded(self, conn: sqlite3.Connection) -> dict[str, tuple]:
        rows = conn.execute(
            "SELECT pack_id, digest, entries FROM gazetteer_packs"
        ).fetchall()
        return {str(r[0]): (str(r[1]), int(r[2])) for r in rows}

    def _indexed_pack_ids(self, conn: sqlite3.Connection) -> tuple:
        rows = conn.execute(
            "SELECT pack_id FROM gazetteer_packs ORDER BY pack_id"
        ).fetchall()
        return tuple(str(r[0]) for r in rows)

    # ── building ────────────────────────────────────────────────

    def _scan_pack(self, pack_dir: Path,
                   manifest: PackManifest,
                   ) -> tuple[PackIndexResult, list[tuple]]:
        """Decode one pack's in-band place/poi tiles: the build
        result plus the entry rows to persist."""
        if manifest.tile_format != "mvt":
            return PackIndexResult(
                pack_id=manifest.pack_id, digest=manifest.digest,
                note=(manifest.tile_format
                      + " tiles carry no feature data"
                      " — nothing to index")), []
        found: dict[tuple, GazetteerEntry] = {}
        tiles_scanned = 0
        corrupt: list[str] = []
        for member in manifest.members:
            parsed = _parse_tile_member(member.path)
            if parsed is None:
                continue
            z, x, y = parsed
            if z > GAZETTEER_MAX_ZOOM:
                continue
            try:
                raw = (pack_dir / member.path).read_bytes()
            except OSError:
                corrupt.append(member.path)
                continue
            tiles_scanned += 1
            if raw[:2] == b"\x1f\x8b":
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    corrupt.append(member.path)
                    continue
            try:
                tile = decode_mvt(raw)
            except MvtError:
                corrupt.append(member.path)
                continue
            for layer in tile.layers:
                if layer.name not in INDEX_LAYERS:
                    continue
                for feature in layer.features:
                    self._harvest(found, manifest, layer, feature,
                                  z, x, y)
        rows = [
            (entry.pack_id, entry.name, entry.indexed_name,
             normalize_name(entry.indexed_name), entry.kind,
             entry.lat, entry.lon, entry.source_zoom,
             entry.population)
            for entry in found.values()]
        return PackIndexResult(
            pack_id=manifest.pack_id, digest=manifest.digest,
            entries=len(found), tiles_scanned=tiles_scanned,
            corrupt_tiles=tuple(corrupt),
            note="" if found else "no named places in band"), rows

    def _harvest(self, found: dict, manifest: PackManifest, layer,
                 feature, z: int, x: int, y: int) -> None:
        forms = _name_forms(feature.tags)
        if not forms:
            return
        anchor = _feature_anchor(feature)
        if anchor is None:
            return
        lat, lon = _tile_point_to_latlon(
            z, x, y, layer.extent, anchor[0], anchor[1])
        kind = _kind_of(layer.name, feature.tags)
        population = _population(feature.tags)
        for form in forms:
            norm = normalize_name(form)
            key = self._merge_key(found, norm, lat, lon)
            existing = found.get(key)
            if existing is not None \
                    and (existing.source_zoom < z
                         or (existing.source_zoom == z
                             and existing.population >= population)):
                continue
            found[key] = GazetteerEntry(
                name=forms[0], indexed_name=form, kind=kind,
                lat=lat, lon=lon, pack_id=manifest.pack_id,
                source_zoom=z, population=population)

    @staticmethod
    def _merge_key(found: dict, norm: str, lat: float,
                   lon: float) -> tuple:
        """The dedupe key for a sighting: its own 3-decimal cell,
        or the key of a same-name entry already indexed within
        ~110 m (the multi-zoom merge — see the module docstring).
        """
        cell = (norm, round(lat, 3), round(lon, 3))
        for dlat in (-0.001, 0.0, 0.001):
            for dlon in (-0.001, 0.0, 0.001):
                key = (norm, round(lat + dlat, 3),
                       round(lon + dlon, 3))
                existing = found.get(key)
                if existing is not None \
                        and abs(existing.lat - lat) <= 0.0012 \
                        and abs(existing.lon - lon) <= 0.0012:
                    return key
        return cell

    def _run_build(self, packs, recorded: dict) -> list:
        results: list[PackIndexResult] = []
        conn = self._connect(create=True)
        assert conn is not None  # create=True always yields a conn
        try:
            for pack_dir, manifest in packs:
                prior = recorded.get(manifest.pack_id)
                if prior is not None and prior[0] == manifest.digest:
                    results.append(PackIndexResult(
                        pack_id=manifest.pack_id,
                        digest=manifest.digest,
                        entries=prior[1], skipped=True,
                        note="digest unchanged — no-op"))
                    continue
                result, rows = self._scan_pack(
                    Path(pack_dir), manifest)
                conn.execute(
                    "DELETE FROM gazetteer_entries WHERE pack_id = ?",
                    (manifest.pack_id,))
                conn.executemany(
                    "INSERT OR REPLACE INTO gazetteer_entries"
                    " (pack_id, name, indexed_name, norm, kind,"
                    " lat, lon, source_zoom, population)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
                conn.execute(
                    "INSERT OR REPLACE INTO gazetteer_packs"
                    " (pack_id, digest, built_at, entries,"
                    " tiles_scanned, corrupt_tiles)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (manifest.pack_id, manifest.digest, time.time(),
                     result.entries, result.tiles_scanned,
                     len(result.corrupt_tiles)))
                results.append(result)
            conn.commit()
        finally:
            conn.close()
        return results

    def build(self, packs, *, force: bool = False) -> IndexBuildResult:
        """Index every pack whose digest moved (all, when forced),
        and mint the build receipt."""
        packs = list(packs)
        conn = self._connect(create=True)
        assert conn is not None
        try:
            recorded = {} if force else self._recorded(conn)
        finally:
            conn.close()
        results = self._run_build(packs, recorded)
        return self._finish_build(results)

    def ensure(self, packs) -> IndexBuildResult:
        """Build only what is stale; silent (no receipt) when the
        index is already current — the first-use surface path."""
        packs = list(packs)
        conn = self._connect(create=False)
        if conn is None:
            stale = True
            recorded: dict = {}
        else:
            try:
                recorded = self._recorded(conn)
            finally:
                conn.close()
            stale = any(
                recorded.get(m.pack_id, ("", 0))[0] != m.digest
                for _d, m in packs)
        if not stale:
            results = [
                PackIndexResult(
                    pack_id=m.pack_id, digest=m.digest,
                    entries=recorded.get(m.pack_id, ("", 0))[1],
                    skipped=True, note="digest unchanged — no-op")
                for _d, m in packs]
            return IndexBuildResult(
                packs=tuple(results),
                total_entries=sum(r.entries for r in results),
                built_any=False, receipt=None)
        results = self._run_build(packs, recorded)
        return self._finish_build(results)

    def _finish_build(self, results: list) -> IndexBuildResult:
        built_any = any(not r.skipped for r in results)
        receipt = None
        if self._store is not None:
            detail = " · ".join(
                self._build_line(r) for r in results) or "no packs"
            receipt = MapReceipt.mint(
                OP_GEOCODE, OUTCOME_OK,
                pack_refs=tuple(r.pack_id for r in results),
                pack_digests=tuple(r.digest for r in results),
                detail="gazetteer index: " + detail)
            self._store.append(receipt)
        conn = self._connect(create=False)
        total = 0
        if conn is not None:
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(entries), 0)"
                    " FROM gazetteer_packs").fetchone()
                total = int(row[0]) if row else 0
            finally:
                conn.close()
        return IndexBuildResult(
            packs=tuple(results), total_entries=total,
            built_any=built_any, receipt=receipt)

    @staticmethod
    def _build_line(result: PackIndexResult) -> str:
        if result.skipped:
            return (result.pack_id + ": unchanged (digest match,"
                    " no-op)")
        line = (result.pack_id + ": " + str(result.entries)
                + " places from " + str(result.tiles_scanned)
                + " tiles")
        if result.corrupt_tiles:
            named = ", ".join(result.corrupt_tiles[:5])
            extra = len(result.corrupt_tiles) - 5
            line += (" · corrupt: " + named
                     + (" +" + str(extra) + " more" if extra > 0
                        else ""))
        if result.note:
            line += " · " + result.note
        return line

    # ── forward geocode ─────────────────────────────────────────

    def search(self, query: str, *, limit: int = 8,
               quiet: bool = False) -> GeocodeResult:
        """Forward-geocode ``query`` against the local index."""
        limit = max(1, int(limit))
        norm_q = normalize_name(query)
        conn = self._connect(create=False)
        built = False
        pack_ids: tuple = ()
        local: list[GeocodeCandidate] = []
        if conn is not None:
            try:
                pack_ids = self._indexed_pack_ids(conn)
                built = bool(pack_ids)
                if built and norm_q:
                    local = self._search_local(conn, norm_q, limit)
            finally:
                conn.close()
        posture, provider, note = POSTURE_LOCAL, "local", ""
        candidates = local
        if not local:
            posture, provider, note, candidates = \
                self._federate_search(query, limit, built)
        outcome = OUTCOME_OK if (built or candidates) \
            else OUTCOME_UNAVAILABLE
        receipt = self._query_receipt(
            "find " + repr(query), candidates, posture, provider,
            pack_ids, outcome, quiet)
        if isinstance(candidates, list):
            candidates = tuple(candidates)
        return GeocodeResult(
            query=query, candidates=tuple(candidates),
            posture=posture, provider=provider, note=note,
            receipt=receipt)

    def _search_local(self, conn: sqlite3.Connection, norm_q: str,
                      limit: int) -> list[GeocodeCandidate]:
        escaped = norm_q.replace("\\", "\\\\").replace(
            "%", "\\%").replace("_", "\\_")
        rows = conn.execute(
            "SELECT name, kind, lat, lon, pack_id, population, norm"
            " FROM gazetteer_entries"
            " WHERE norm LIKE ? ESCAPE '\\'",
            ("%" + escaped + "%",)).fetchall()
        best: dict[tuple, tuple] = {}
        for name, kind, lat, lon, pack_id, population, norm in rows:
            rank = match_rank(norm_q, str(norm))
            if rank == 0:
                continue
            key = (str(name), round(float(lat), 4),
                   round(float(lon), 4), str(pack_id))
            order = (-rank, -_kind_weight(str(kind)),
                     -int(population))
            if key in best and best[key][0] <= order:
                continue
            best[key] = (order, GeocodeCandidate(
                name=str(name), kind=str(kind), lat=float(lat),
                lon=float(lon), pack_id=str(pack_id),
                score=_score(rank, str(kind), int(population)),
                matched_as=_MATCH_NAMES[rank]))
        ranked = sorted(
            best.values(),
            key=lambda item: (item[0], item[1].name.casefold(),
                              item[1].lat, item[1].lon))
        return [cand for _order, cand in ranked[:limit]]

    def _federate_search(self, query: str, limit: int,
                         built: bool):
        """The no-local-answer path, per the federation posture.

        Order: federation off → the local answer (or its absence)
        is final.  Federation on but nothing wired → that is
        named, with the missing index named too when there is
        one.  A wired provider answers — its failure is named,
        never fatal, and the local answer stands behind it.
        """
        if not self._federation:
            note = ("no match in the local Atlas index —"
                    " local-first (federation off): nothing was"
                    " fetched from a provider") if built else (
                    "no gazetteer index has been built from the"
                    " installed packs yet")
            return (POSTURE_LOCAL, "local", note, [])
        if self._provider is None:
            note = ("no match in the local Atlas index; federation"
                    " is on but no geocode provider is configured"
                    " — the local answer stands") if built else (
                    "no gazetteer index has been built yet, and"
                    " federation is on but no geocode provider is"
                    " configured")
            return (POSTURE_FEDERATED, "none", note, [])
        try:
            found = self._provider.search(query, limit=limit)
        except Exception as exc:  # noqa: BLE001 — a provider's
            # failure is an answer about the provider, not a crash
            return (POSTURE_FEDERATED, str(self._provider.name),
                    "provider " + str(self._provider.name)
                    + " failed (" + type(exc).__name__
                    + ") — the local answer stands", [])
        why = ("after the local index had no match" if built
               else "— the local index has not been built")
        return (POSTURE_FEDERATED, str(self._provider.name),
                "answered by provider " + str(self._provider.name)
                + " " + why, list(found))

    # ── reverse geocode ─────────────────────────────────────────

    def reverse(self, lat: float, lon: float, *, limit: int = 8,
                quiet: bool = False) -> ReverseResult:
        """Reverse-geocode a point: nearest named places, with
        distances, inside the stated ladder."""
        limit = max(1, int(limit))
        point = LatLon(lat=float(lat), lon=float(lon))
        widest = REVERSE_LADDER_KM[-1]
        conn = self._connect(create=False)
        built = False
        pack_ids: tuple = ()
        local: list[ReverseCandidate] = []
        if conn is not None:
            try:
                pack_ids = self._indexed_pack_ids(conn)
                built = bool(pack_ids)
                if built:
                    local = self._reverse_local(
                        conn, point, widest, limit)
            finally:
                conn.close()
        posture, provider, note = POSTURE_LOCAL, "local", ""
        candidates = local
        if not local:
            posture, provider, note, candidates = \
                self._federate_reverse(point, built)
        rung = widest
        if candidates:
            nearest = min(c.distance_km for c in candidates)
            rung = next(
                (r for r in REVERSE_LADDER_KM if nearest <= r),
                widest)
        outcome = OUTCOME_OK if (built or candidates) \
            else OUTCOME_UNAVAILABLE
        receipt = self._query_receipt(
            "near " + format(point.lat, ".4f") + ","
            + format(point.lon, ".4f"), candidates, posture,
            provider, pack_ids, outcome, quiet,
            nearest_first=True)
        return ReverseResult(
            lat=point.lat, lon=point.lon,
            candidates=tuple(candidates), searched_km=widest,
            rung_km=rung, posture=posture, provider=provider,
            note=note, receipt=receipt)

    def _reverse_local(self, conn: sqlite3.Connection,
                       point: LatLon, widest_km: float,
                       limit: int) -> list[ReverseCandidate]:
        lat_d = widest_km / 110.0
        cos_lat = max(0.05, math.cos(math.radians(point.lat)))
        lon_d = min(180.0, widest_km / (100.0 * cos_lat))
        lat_lo = max(-90.0, point.lat - lat_d)
        lat_hi = min(90.0, point.lat + lat_d)
        lon_lo, lon_hi = point.lon - lon_d, point.lon + lon_d
        if lon_lo < -180.0:
            lon_clause = "(lon >= ? OR lon <= ?)"
            lon_params = (lon_lo + 360.0, lon_hi)
        elif lon_hi > 180.0:
            lon_clause = "(lon >= ? OR lon <= ?)"
            lon_params = (lon_lo, lon_hi - 360.0)
        else:
            lon_clause = "lon BETWEEN ? AND ?"
            lon_params = (lon_lo, lon_hi)
        rows = conn.execute(
            "SELECT name, kind, lat, lon, pack_id"
            " FROM gazetteer_entries"
            " WHERE lat BETWEEN ? AND ? AND " + lon_clause,
            (lat_lo, lat_hi) + tuple(lon_params)).fetchall()
        best: dict[tuple, ReverseCandidate] = {}
        for name, kind, rlat, rlon, pack_id in rows:
            distance = haversine_m(
                point, LatLon(lat=float(rlat), lon=float(rlon))) / 1000.0
            if distance > widest_km:
                continue
            key = (str(name), round(float(rlat), 4),
                   round(float(rlon), 4), str(pack_id))
            cand = ReverseCandidate(
                name=str(name), kind=str(kind), lat=float(rlat),
                lon=float(rlon), pack_id=str(pack_id),
                distance_km=distance)
            if key not in best or distance < best[key].distance_km:
                best[key] = cand
        ranked = sorted(
            best.values(),
            key=lambda c: (c.distance_km, -_kind_weight(c.kind),
                           c.name.casefold()))
        return ranked[:limit]

    def _federate_reverse(self, point: LatLon, built: bool):
        """The no-local-answer path for reverse, mirroring search."""
        widest = format(REVERSE_LADDER_KM[-1], ".0f")
        if not self._federation:
            note = ("no named place within " + widest
                    + " km in the local Atlas index — local-first"
                    " (federation off): nothing was fetched from"
                    " a provider") if built else (
                    "no gazetteer index has been built from the"
                    " installed packs yet")
            return (POSTURE_LOCAL, "local", note, [])
        if self._provider is None:
            note = ("no named place within " + widest
                    + " km locally; federation is on but no"
                    " geocode provider is configured — the local"
                    " answer stands") if built else (
                    "no gazetteer index has been built yet, and"
                    " federation is on but no geocode provider is"
                    " configured")
            return (POSTURE_FEDERATED, "none", note, [])
        try:
            found = self._provider.reverse(
                point.lat, point.lon, limit=8)
        except Exception as exc:  # noqa: BLE001
            return (POSTURE_FEDERATED, str(self._provider.name),
                    "provider " + str(self._provider.name)
                    + " failed (" + type(exc).__name__
                    + ") — the local answer stands", [])
        why = ("after the local index had no answer" if built
               else "— the local index has not been built")
        return (POSTURE_FEDERATED, str(self._provider.name),
                "answered by provider " + str(self._provider.name)
                + " " + why, list(found))

    # ── receipts ────────────────────────────────────────────────

    def _query_receipt(self, label: str, candidates, posture: str,
                       provider: str, pack_ids: tuple, outcome: str,
                       quiet: bool, *,
                       nearest_first: bool = False) -> MapReceipt | None:
        if quiet or self._store is None:
            return None
        detail = (label + ": " + str(len(candidates))
                  + " candidate(s)")
        if candidates:
            top = candidates[0]
            detail += " · top: " + top.name
            if nearest_first:
                detail += (" at " + format(top.distance_km, ".1f")
                           + " km")
        if posture == POSTURE_FEDERATED:
            detail += " · via provider " + provider
        receipt = MapReceipt.mint(
            OP_GEOCODE, outcome,
            posture=(POSTURE_ONLINE if posture == POSTURE_FEDERATED
                     and provider not in ("local", "none")
                     else POSTURE_OFFLINE),
            provider=provider, pack_refs=tuple(pack_ids),
            detail=detail)
        self._store.append(receipt)
        return receipt
