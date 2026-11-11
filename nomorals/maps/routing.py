"""The routing plane: a local road graph + Dijkstra, over packs.

Doc 21 S3 superseded doc 19's hosted-router posture the same way
it superseded the hosted geocoder (M4): the router is **local
first** — a graph Devon extracts from the packs she already
holds — and federation to an outside routing provider happens
only when owner policy permits (``MapSettings.routing_federation``,
default OFF).  No provider ships in this pass; the seam, the
setting, and the honest "no provider configured" answer are the
deliverable, exactly as the geocoder's federation landed in M4.

**The graph.**  A pack's vector tiles carry the road network in
the ``transportation`` layer (the style's own vocabulary, aliases
``road``/``roads``): each feature is one road's geometry with its
``class``, its ``name`` when the mappers recorded one, and a
``oneway`` tag when travel is restricted.  The graph's nodes are
the geometry *vertices*, identified by their exact integer
coordinates in global pixel space (``tile_x * extent + px``,
``tile_y * extent + py``) — the join tolerance is therefore
zero, stated plainly: two vertices are the same node iff the
source data quantized them to the same pixel.  That is the
honest join for OSM-derived tiles: ways that meet at a real
junction share a node in the source, so both geometries carry
the identical vertex — across a tile boundary as much as inside
one tile, because the boundary vertex has one global pixel
coordinate whichever tile's copy is read.  A crossing with no
shared vertex (a bridge over a road) has no node in the source
either, and correctly does not connect.  Edges join consecutive
vertices along a feature; an edge's length is the haversine
along that segment; a ``oneway`` edge arcs in its permitted
direction only (digitized direction for ``1``/``yes``, against
it for ``-1``), and an untagged edge arcs both ways.

**Seam bridging (the routing-design pass).**  Exact joins alone
shatter the real graph: planetiler's simplification drops shared
junction vertices — the through road loses the vertex where a
side road joins — so the side road's END is left dangling a few
metres from the segment it should meet.  Measured on real
corridors before this pass (Kano: 963 components over 19,784
nodes), most dangling ends sit ≤5 m from a through *segment*
while the nearest surviving vertex is tens of metres away —
which is why the bridge attaches an end to a segment, not only
to a node.  At graph build, each way END whose nearest
attachment — a node, or the interior of a segment — lies within
:data:`SEAM_JOIN_MAX_M` **in a different component** is joined
to it: a segment attachment first splits the segment at the
projected point (a real node on the road, on the same global
pixel grid as every other node; the split lengths are
haversine and sum to the original segment's), and the end is
then joined by a **zero-length seam arc** (road class
:data:`SEAM_CLASS`).  The guards are conservative: only way
ends initiate, each end bridges at most once and only to its
*nearest* attachment (an end already at its junction — nearest
attachment in its own component — bridges nothing), and the
tolerance is small (10 m) beside real separations.  A seam arc
is a digitization repair, never a road: it contributes **zero**
distance and zero duration, it never appears as a route step,
and every route that crosses one counts it
(``RouteResult.seam_joins``, named in the route's detail and
its receipt) — the same labelled-honesty discipline the snap
distances follow.  Crossings still do not connect: a bridge
deck's segment is an attachment only for a way END that stops
within the tolerance of it, and way ends do not occur at
grade-separated crossings in the source.

**Corridor discipline (the gazetteer's zoom-band lesson).**  A
country graph is never materialized: the graph builds for one
route's *corridor* — the endpoints' bounding box padded by
:data:`CORRIDOR_PAD_FRACTION` of its span (at least
:data:`CORRIDOR_PAD_MIN_DEG`) — at zoom :data:`ROUTE_ZOOM` (or
the pack's own ceiling, when the pack is shallower), capped at
:data:`MAX_CORRIDOR_TILES` tiles.  A corridor over the cap is a
typed refusal naming the bound, never an out-of-memory death.
The built graph is cached in a sqlite store beside the
gazetteer, keyed by (pack digest, zoom, corridor): an unchanged
pack re-routes from the cache, a changed pack rebuilds, and
stale graphs of superseded pack digests are dropped at build
time.  The cache is derived data, rebuildable at any time.

**Corridor widening (the corridor-pad pass).**  The shipped
corridor is a pure function of the endpoints: it cannot see
that the road a route needs bulges past its edge — lagoon
crossings, causeways and one-way circulations do exactly that,
and the in-corridor optimum was returned silently as *the*
route (measured on the real pack: a Lagos hop served 11,702 m
at ratio 4.00 while 9,370 m ran just outside the corridor's
east edge).  So :meth:`Router.route` keeps its first attempt
byte-identical to the shipped corridor, and only when that
attempt is *suspect* — a found route whose ratio exceeds
:data:`ROUTE_RATIO_SUSPECT`, or a no-route of a class widening
can honestly retry (disconnected, snap, no road data) — it
re-materializes at geometrically growing pads
(:data:`CORRIDOR_WIDEN_FACTOR`, ceiling
:data:`CORRIDOR_WIDEN_MAX_DEG`), checking the tile cap
*before* each step and stopping at the first attempt that is
not suspect.  A bigger corridor's graph is a superset, so
steps can only tie or improve; the shortest route measured
across attempts is the answer regardless.  The result records
the ladder it climbed (``ladder_steps``, ``corridor_pad_deg``,
``widened``, ``ladder_stop``) and a ``detour_class`` computed
from the ladder's own measurements — ``corridor_bound`` when
widening materially improved the answer (the shipped corridor
was the binding constraint), ``network_detour`` when the
ladder exhausted and the ratio stands as the network's real
shape, ``unremarkable`` inside the healthy envelope.  The
classification labels measurements; it never moves the
battery's ratio bar, and a route the ladder never touches is
byte-identical to the route before this pass.

**Routing.**  Dijkstra over distance — exact, and the corridor
cap keeps the graph small enough that A*'s bookkeeping buys
nothing; duration is then a stated *estimate*: each arc's length
over the speed its road class carries in :data:`SPEED_KMH`
(defaults in km/h, data in this module, not measurements of any
real road).  Endpoints snap *multi-candidate*: each endpoint
offers its :data:`SNAP_CANDIDATES` nearest nodes inside the
snap bound and a multi-source Dijkstra takes the pair that
routes best — a place label anchored a few metres from a stub
vertex (the M8 finding: Ado Ekiti's nearest node was a 3-node
stub while the town network ran just beyond it) is no longer
stranded by its single nearest node.  The snap distances that
ride the result are the *chosen* pair's, and an endpoint with
no node at all inside :data:`SNAP_MAX_M` is a typed no-route
naming the nearest distance — never a silent teleport.  A disconnected
graph answers "disconnected" just as plainly.  Every attempt —
success and no-route alike — mints an ``OP_ROUTE`` MapReceipt:
``ok`` when the plane answered (a route, or the truthful
"disconnected"), ``unavailable`` when the plane could not serve
the request at all (snap, no road data, corridor cap, raster
pack, unwired federation).  The package stays network-free —
the no-network AST pin covers this module.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from ..core.failures import record_failure
from ..location.primitives import EARTH_RADIUS_M, LatLon, haversine_m
from .coords import BBox
from .gazetteer import POSTURE_FEDERATED, POSTURE_LOCAL
from .manifest import PackManifest, tile_relpath
from .mvt import GEOM_LINESTRING, MvtError, decode_mvt
from .receipts import (
    OP_ROUTE,
    OUTCOME_OK,
    OUTCOME_UNAVAILABLE,
    POSTURE_OFFLINE,
    POSTURE_ONLINE,
    MapReceipt,
    MapReceiptStore,
)

__all__ = [
    "CORRIDOR_PAD_FRACTION", "CORRIDOR_PAD_MIN_DEG",
    "CORRIDOR_WIDEN_FACTOR", "CORRIDOR_WIDEN_MAX_DEG",
    "DEFAULT_SPEED_KMH", "DETOUR_CORRIDOR_BOUND",
    "DETOUR_NETWORK_DETOUR", "DETOUR_UNREMARKABLE",
    "GRAPH_FILENAME", "MAX_CORRIDOR_TILES",
    "NON_ROAD_CLASSES", "REASON_CORRIDOR_TOO_LARGE",
    "REASON_DISCONNECTED", "REASON_NO_ROAD_DATA",
    "REASON_NOT_VECTOR", "REASON_SNAP_FROM", "REASON_SNAP_TO",
    "ROUTE_RATIO_SUSPECT", "ROUTE_ZOOM", "SEAM_CLASS",
    "SEAM_JOIN_MAX_M",
    "SNAP_CANDIDATES", "SNAP_MAX_M", "SPEED_KMH", "TRANSPORT_LAYERS",
    "GraphStats", "LadderStep", "RouteProvider", "RouteResult",
    "RouteStep", "Router",
]

#: The graph store's filename inside the packs directory.
GRAPH_FILENAME = "routing.sqlite3"

#: The zoom road detail is read at (the pack's own ceiling when
#: the pack is shallower — generalized roads still route).
ROUTE_ZOOM = 14

#: One corridor never spans more tiles than this at the graph
#: zoom.  The bound is the memory contract: a route that needs
#: more is refused by name, not attempted into an OOM.
MAX_CORRIDOR_TILES = 1024

#: Corridor padding: this fraction of the endpoints' span, each
#: side, with a floor so a short hop still gets breathing room.
CORRIDOR_PAD_FRACTION = 0.30
CORRIDOR_PAD_MIN_DEG = 0.01

#: A first-attempt route whose distance/haversine ratio exceeds
#: this is *suspect*: the corridor, not the network, may be the
#: binding constraint, so the widening ladder climbs.  Set under
#: the acceptance battery's 3.0 bar and above the healthy legs'
#: measured envelope (1.23–2.16 on the real pack) — a healthy
#: route never pays for a second materialization.
ROUTE_RATIO_SUSPECT = 2.5

#: The widening ladder: the pad floor grows by this factor per
#: step from CORRIDOR_PAD_MIN_DEG (0.01° → 0.02° → 0.04° →
#: 0.08°), never past CORRIDOR_WIDEN_MAX_DEG, and never into a
#: corridor over MAX_CORRIDOR_TILES (checked before the step is
#: attempted, exactly as the first attempt's cap is checked).
CORRIDOR_WIDEN_FACTOR = 2.0
CORRIDOR_WIDEN_MAX_DEG = 0.08

#: Detour classes — the route ticket's measured classification
#: of a high ratio (the corridor-pad pass; see the docstring).
#: corridor_bound: widening materially improved the answer, so
#: the shipped corridor was the binding constraint.
#: network_detour: the ladder exhausted and the ratio stands —
#: the detour lives in the network (terrain, one-way shape).
#: unremarkable: the ratio sits inside the healthy envelope.
DETOUR_CORRIDOR_BOUND = "corridor_bound"
DETOUR_NETWORK_DETOUR = "network_detour"
DETOUR_UNREMARKABLE = "unremarkable"

#: The relative shortening that counts as the corridor having
#: bound the route: a widened step must beat the first attempt
#: by more than this fraction before the ticket calls the route
#: corridor_bound (superset steps tie often; ties are not
#: evidence the corridor bound anything).
_IMPROVEMENT_MIN_FRACTION = 0.01

#: An endpoint farther than this from the nearest road node is
#: not snapped — the route refuses, naming the distance.
SNAP_MAX_M = 500.0

#: Seam bridging: a way END whose nearest attachment (a node, or
#: the interior of a segment) lies within this many metres, in a
#: different component, is joined to it by a zero-length seam
#: arc.  Sized from the measured digitization gaps on real
#: corridors: the dangling-end gap mass sits at ≤5 m, and 10 m
#: stays well under real separations (parallel carriageways,
#: frontage roads).  See the module docstring.
SEAM_JOIN_MAX_M = 10.0

#: The road class synthetic seam arcs carry — never a real
#: road's class (the transportation vocabulary has no "seam"),
#: and the marker that keeps them labelled through the graph
#: cache: a loaded arc of this class is a seam, not a road.
SEAM_CLASS = "seam"

#: Snap candidates per endpoint: the nearest this many nodes
#: inside SNAP_MAX_M are offered to the router, which takes the
#: pair that routes best (multi-source Dijkstra).
SNAP_CANDIDATES = 8

#: The graph-build version riding the cache key: a graph built
#: by an older builder (exact joins only, no seam bridging) is
#: never served for a corridor the new builder would bridge.
_GRAPH_BUILD_TAG = "seamjoin-v1"

#: Source layers the graph reads: the style's transportation
#: vocabulary and its aliases.
TRANSPORT_LAYERS = ("transportation", "road", "roads")

#: Classes in the transportation layer that are not roads a
#: route may travel: rails and guideways carry their own planes.
NON_ROAD_CLASSES = ("rail", "transit", "aerialway")

#: Travel speeds by road class, km/h — ESTIMATES, data in this
#: module: planning defaults over the class vocabulary the packs
#: carry, not measurements and never presented as live speeds.
SPEED_KMH = {
    "motorway": 100.0, "trunk": 80.0, "primary": 60.0,
    "secondary": 50.0, "tertiary": 40.0, "minor": 30.0,
    "residential": 30.0, "unclassified": 30.0, "service": 15.0,
    "track": 15.0, "cycleway": 15.0, "ferry": 20.0,
    "path": 5.0, "pedestrian": 5.0, "footway": 5.0,
}

#: The speed an unlisted road class is estimated at, km/h.
DEFAULT_SPEED_KMH = 25.0

#: No-route reason classes — the receipt vocabulary.
REASON_SNAP_FROM = "snap-from"
REASON_SNAP_TO = "snap-to"
REASON_DISCONNECTED = "disconnected"
REASON_NO_ROAD_DATA = "no-road-data"
REASON_CORRIDOR_TOO_LARGE = "corridor-too-large"
REASON_NOT_VECTOR = "pack-not-vector"

#: No-route reasons the widening ladder may honestly retry: a
#: bigger corridor can hold roads the shipped one cut (a path,
#: a snap node, any road data at all).  Corridor-too-large is
#: deliberately absent — widening only adds tiles, so a retry
#: there is futile by construction — and pack-not-vector never
#: reaches the ladder (it is refused before the first attempt).
_LADDER_TRIGGER_REASONS = frozenset((
    REASON_DISCONNECTED, REASON_SNAP_FROM, REASON_SNAP_TO,
    REASON_NO_ROAD_DATA))

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS routing_graphs ("
    " cache_key TEXT PRIMARY KEY, pack_id TEXT NOT NULL,"
    " digest TEXT NOT NULL, zoom INTEGER NOT NULL,"
    " corridor TEXT NOT NULL, built_at REAL NOT NULL,"
    " nodes INTEGER NOT NULL, arcs INTEGER NOT NULL,"
    " tiles_present INTEGER NOT NULL,"
    " corrupt_tiles INTEGER NOT NULL);"
    "CREATE TABLE IF NOT EXISTS routing_nodes ("
    " cache_key TEXT NOT NULL, node TEXT NOT NULL,"
    " lat REAL NOT NULL, lon REAL NOT NULL,"
    " PRIMARY KEY (cache_key, node));"
    "CREATE TABLE IF NOT EXISTS routing_arcs ("
    " cache_key TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL,"
    " road_class TEXT NOT NULL, name TEXT NOT NULL,"
    " length_m REAL NOT NULL,"
    " PRIMARY KEY (cache_key, a, b, road_class, name));"
    "CREATE INDEX IF NOT EXISTS routing_arcs_cache_idx"
    " ON routing_arcs(cache_key);"
    "CREATE INDEX IF NOT EXISTS routing_nodes_cache_idx"
    " ON routing_nodes(cache_key);"
)


# ── typed values ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteStep:
    """One leg of a route: consecutive same-road arcs, merged."""

    name: str            # display: the road's name, else its class
    road_class: str
    length_m: float
    named: bool          # the name came from a tag (else the class)


@dataclass(frozen=True)
class LadderStep:
    """One corridor attempt's measurements (the widening ladder).

    ``pad_deg`` is the pad floor the attempt's corridor was
    built with; ``tiles`` the corridor's tile count at the
    graph zoom; ``distance_m``/``ratio`` are the attempt's
    route measurements (None when the attempt found no route);
    ``outcome`` is "route" or the attempt's no-route reason.
    Every field is measured by the attempt itself — the ladder
    records what it did, never what it hoped.
    """

    pad_deg: float
    tiles: int
    distance_m: float | None = None
    ratio: float | None = None
    outcome: str = ""


@dataclass(frozen=True)
class RouteResult:
    """A route attempt's whole answer: the route, or the refusal.

    ``ok`` distinguishes the two; on a refusal ``reason`` is one
    of the REASON_* classes and ``detail`` says it in words.
    Snap distances are the endpoint→node gaps the route actually
    bridged (None when no snap happened); ``distance_m`` is road
    distance only — snap legs are reported, never folded in.
    ``seam_joins`` counts the zero-length synthetic seam arcs
    the path crossed: labelled like the snaps, contributing no
    distance, and named in ``detail`` whenever nonzero.

    The corridor fields are the widening ladder's record:
    ``corridor_pad_deg`` is the pad floor of the corridor that
    produced this answer, ``ladder_steps`` every attempt's
    measurements in order, ``widened`` whether the answer came
    from a widened attempt, ``ladder_stop`` why the ladder
    stopped ("" when it never climbed; "not-suspect",
    "ceiling", "tile-cap" or "corridor-unchanged"), and
    ``detour_class`` the measured classification of the ratio
    (one of the DETOUR_* values; "" when no corridor attempt
    ran at all, e.g. a raster pack refused up front).
    """

    ok: bool
    from_label: str = ""
    to_label: str = ""
    from_point: tuple[float, float] | None = None
    to_point: tuple[float, float] | None = None
    geometry: tuple[tuple[float, float], ...] = ()
    distance_m: float = 0.0
    duration_s: float = 0.0
    steps: tuple[RouteStep, ...] = ()
    seam_joins: int = 0
    from_snap_m: float | None = None
    to_snap_m: float | None = None
    reason: str = ""
    detail: str = ""
    pack_id: str = ""
    posture: str = POSTURE_LOCAL
    provider: str = "local"
    receipt: MapReceipt | None = None
    corridor_pad_deg: float = 0.0
    ladder_steps: tuple[LadderStep, ...] = ()
    widened: bool = False
    ladder_stop: str = ""
    detour_class: str = ""


@dataclass(frozen=True)
class GraphStats:
    """What one corridor graph is: the build/introspection view."""

    zoom: int
    node_count: int
    arc_count: int
    tiles_present: int
    corrupt_tiles: int
    cached: bool
    seam_arcs: int = 0


class RouteProvider(Protocol):
    """The federation seam (doc 21 S3): an outside router.

    Implementations answer with a :class:`RouteResult` of their
    own (its receipt left None — the Router mints the receipt for
    whichever plane answered).  No provider ships in this pass.
    """

    name: str

    def route(self, start: LatLon,
              end: LatLon) -> RouteResult | None:
        ...


@dataclass
class _Graph:
    """One built corridor graph, in memory.

    Adjacency arcs are (node, length_m, road_class, name,
    synthetic) — the flag marks seam arcs (road class
    :data:`SEAM_CLASS`, length zero) so routing, steps and
    statistics can keep them labelled as repairs, never roads.
    ``terminals`` (way-end node keys) and ``extent`` are build
    scratch for the seam bridger; a cache-loaded graph needs
    neither.
    """

    nodes: dict[str, tuple[float, float]] = field(default_factory=dict)
    adj: dict[str, list[tuple[str, float, str, str, bool]]] = field(
        default_factory=dict)
    tiles_present: int = 0
    corrupt_tiles: int = 0
    cached: bool = False
    seam_arcs: int = 0
    terminals: set[str] = field(default_factory=set)
    extent: int = 0


# ── geometry helpers ──────────────────────────────────────────────


def _global_px_to_latlon(gx: float, gy: float, zoom: int,
                         extent: int) -> tuple[float, float]:
    """Global extent-pixel coordinates → (lat, lon), normalized."""
    world = float(extent) * (1 << zoom)
    lon = gx / world * 360.0 - 180.0
    lon = (lon + 180.0) % 360.0 - 180.0
    lat = math.degrees(
        math.atan(math.sinh(math.pi * (1.0 - 2.0 * gy / world))))
    return lat, lon


def _corridor(start: LatLon, end: LatLon, *,
              pad_floor: float = CORRIDOR_PAD_MIN_DEG) -> BBox:
    """The route corridor: the endpoints' box, padded.

    Longitude is computed in whichever frame (direct or wrapped)
    is shorter, so a route across the antimeridian gets the
    corridor it actually travels; latitudes clamp to the
    Mercator band the tiles cover.  ``pad_floor`` is the pad's
    floor in degrees (the shipped CORRIDOR_PAD_MIN_DEG unless
    the widening ladder is climbing); the fraction term rides
    on top of it exactly as before, so the default call is
    byte-identical to the shipped corridor.
    """
    lat_lo = min(start.lat, end.lat)
    lat_hi = max(start.lat, end.lat)
    pad_lat = max(pad_floor,
                  CORRIDOR_PAD_FRACTION * (lat_hi - lat_lo))
    south = max(-85.05112878, lat_lo - pad_lat)
    north = min(85.05112878, lat_hi + pad_lat)
    lon_a, lon_b = start.lon, end.lon
    if lon_b - lon_a > 180.0:
        lon_a += 360.0
    elif lon_a - lon_b > 180.0:
        lon_b += 360.0
    lon_lo = min(lon_a, lon_b)
    lon_hi = max(lon_a, lon_b)
    pad_lon = max(pad_floor,
                  CORRIDOR_PAD_FRACTION * (lon_hi - lon_lo))
    west_raw = lon_lo - pad_lon
    east_raw = lon_hi + pad_lon
    if east_raw - west_raw >= 360.0:
        return BBox(west=-180.0, south=south, east=180.0, north=north)
    west = (west_raw + 180.0) % 360.0 - 180.0
    east = (east_raw + 180.0) % 360.0 - 180.0
    return BBox(west=west, south=south, east=east, north=north)


def _graph_zoom(manifest: PackManifest) -> int:
    """The zoom the graph reads: ROUTE_ZOOM, inside the pack."""
    return max(manifest.zoom_min,
               min(ROUTE_ZOOM, manifest.zoom_max))


def _oneway(tags: dict) -> int:
    """A feature's travel restriction: 1 forward-only (along the
    digitized direction), -1 reverse-only, 0 two-way (the absent
    tag's stated meaning)."""
    raw = tags.get("oneway")
    if raw is None:
        return 0
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, (int, float)):
        if raw > 0:
            return 1
        if raw < 0:
            return -1
        return 0
    text = str(raw).strip().lower()
    if text in ("yes", "true", "1"):
        return 1
    if text in ("-1", "reverse"):
        return -1
    return 0


def _speed_ms(road_class: str) -> float:
    """The class's estimated speed in m/s (SPEED_KMH as data)."""
    return SPEED_KMH.get(road_class, DEFAULT_SPEED_KMH) / 3.6


# ── the router ────────────────────────────────────────────────────


@dataclass
class _Attempt:
    """One widening-ladder attempt's local answer and its
    measurements (internal: the ladder's working record; the
    ticket carries LadderStep copies out)."""

    pad_deg: float
    tiles: int
    result: RouteResult
    outcome: str
    ratio: float | None


class Router:
    """Local-first routing over the packs in one packs directory."""

    def __init__(self, packs_dir: Path | str, *,
                 store: MapReceiptStore | None = None,
                 provider: RouteProvider | None = None,
                 federation: bool = False) -> None:
        self._root = Path(packs_dir)
        self._store = store
        self._provider = provider
        self._federation = bool(federation)

    @property
    def graph_path(self) -> Path:
        return self._root / GRAPH_FILENAME

    # ── sqlite plumbing ─────────────────────────────────────────

    def _connect(self, *, create: bool) -> sqlite3.Connection | None:
        """Open the graph store, or None when read-only and absent.

        A corrupt store file is deleted and rebuilt: the graphs
        are derived data and a fresh build is always their true
        state (the gazetteer's discipline).
        """
        if not create and not self.graph_path.is_file():
            return None
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(self.graph_path))
            conn.executescript(_SCHEMA)
            conn.commit()
            return conn
        except sqlite3.DatabaseError:
            try:
                self.graph_path.unlink()
            except OSError as exc:
                record_failure("maps.routing_store_reset", exc)
            conn = sqlite3.connect(str(self.graph_path))
            conn.executescript(_SCHEMA)
            conn.commit()
            return conn

    # ── graph build / cache ─────────────────────────────────────

    @staticmethod
    def _cache_key(manifest: PackManifest, zoom: int,
                   corridor: BBox) -> str:
        payload = "|".join((
            manifest.pack_id, manifest.digest, str(int(zoom)),
            format(corridor.west, ".6f"), format(corridor.south, ".6f"),
            format(corridor.east, ".6f"), format(corridor.north, ".6f"),
            _GRAPH_BUILD_TAG, format(SEAM_JOIN_MAX_M, ".1f")))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _load_cached(self, cache_key: str,
                     manifest: PackManifest) -> _Graph | None:
        conn = self._connect(create=False)
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT digest, tiles_present, corrupt_tiles"
                " FROM routing_graphs WHERE cache_key = ?",
                (cache_key,)).fetchone()
            if row is None or str(row[0]) != manifest.digest:
                return None
            graph = _Graph(tiles_present=int(row[1]),
                           corrupt_tiles=int(row[2]), cached=True)
            for node, lat, lon in conn.execute(
                    "SELECT node, lat, lon FROM routing_nodes"
                    " WHERE cache_key = ?", (cache_key,)):
                graph.nodes[str(node)] = (float(lat), float(lon))
            for a, b, klass, name, length in conn.execute(
                    "SELECT a, b, road_class, name, length_m"
                    " FROM routing_arcs WHERE cache_key = ?",
                    (cache_key,)):
                synthetic = str(klass) == SEAM_CLASS
                graph.adj.setdefault(str(a), []).append(
                    (str(b), float(length), str(klass), str(name),
                     synthetic))
                if synthetic:
                    graph.seam_arcs += 1
            return graph
        finally:
            conn.close()

    def _persist(self, cache_key: str, manifest: PackManifest,
                 zoom: int, corridor: BBox, graph: _Graph) -> None:
        conn = self._connect(create=True)
        assert conn is not None  # create=True always yields a conn
        try:
            # Stale graphs of superseded digests of this pack go:
            # a changed pack never leaves its old roads behind.
            stale = [str(r[0]) for r in conn.execute(
                "SELECT cache_key FROM routing_graphs"
                " WHERE pack_id = ? AND digest != ?",
                (manifest.pack_id, manifest.digest)).fetchall()]
            for key in stale:
                conn.execute(
                    "DELETE FROM routing_nodes WHERE cache_key = ?",
                    (key,))
                conn.execute(
                    "DELETE FROM routing_arcs WHERE cache_key = ?",
                    (key,))
                conn.execute(
                    "DELETE FROM routing_graphs WHERE cache_key = ?",
                    (key,))
            conn.execute(
                "DELETE FROM routing_nodes WHERE cache_key = ?",
                (cache_key,))
            conn.execute(
                "DELETE FROM routing_arcs WHERE cache_key = ?",
                (cache_key,))
            conn.execute(
                "INSERT OR REPLACE INTO routing_graphs"
                " (cache_key, pack_id, digest, zoom, corridor,"
                " built_at, nodes, arcs, tiles_present, corrupt_tiles)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cache_key, manifest.pack_id, manifest.digest,
                 int(zoom),
                 json.dumps([corridor.west, corridor.south,
                             corridor.east, corridor.north]),
                 time.time(), len(graph.nodes),
                 sum(len(v) for v in graph.adj.values()),
                 graph.tiles_present, graph.corrupt_tiles))
            conn.executemany(
                "INSERT OR REPLACE INTO routing_nodes"
                " (cache_key, node, lat, lon) VALUES (?, ?, ?, ?)",
                [(cache_key, node, lat, lon)
                 for node, (lat, lon) in graph.nodes.items()])
            conn.executemany(
                "INSERT OR REPLACE INTO routing_arcs"
                " (cache_key, a, b, road_class, name, length_m)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(cache_key, a, b, klass, name, length)
                 for a, arcs in graph.adj.items()
                 for (b, length, klass, name, _syn) in arcs])
            conn.commit()
        finally:
            conn.close()

    def _build_graph(self, pack_dir: Path, manifest: PackManifest,
                     corridor: BBox, zoom: int) -> _Graph:
        graph = _Graph()
        tiles = corridor.tiles(zoom)
        n = 1 << zoom
        first_x = tiles[0].x if tiles else 0
        crosses = corridor.crosses_antimeridian
        arcs: dict[tuple, float] = {}
        for tile in tiles:
            rel = tile_relpath(tile.z, tile.x, tile.y,
                               manifest.tile_format)
            path = pack_dir / rel
            if not path.is_file():
                continue
            graph.tiles_present += 1
            raw = path.read_bytes()
            if raw[:2] == b"\x1f\x8b":
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    graph.corrupt_tiles += 1
                    continue
            try:
                mvt_tile = decode_mvt(raw)
            except MvtError:
                graph.corrupt_tiles += 1
                continue
            unit_x = tile.x + n if (crosses and tile.x < first_x) \
                else tile.x
            for layer in mvt_tile.layers:
                if layer.name not in TRANSPORT_LAYERS:
                    continue
                for feature in layer.features:
                    self._harvest(graph, arcs, feature, layer.extent,
                                  zoom, unit_x, tile.y)
        for (a, b, klass, name), length in arcs.items():
            graph.adj.setdefault(a, []).append(
                (b, length, klass, name, False))
        self._bridge_seams(graph, zoom)
        return graph

    def _harvest(self, graph: _Graph, arcs: dict, feature,
                 extent: int, zoom: int, unit_x: int,
                 tile_y: int) -> None:
        """One transportation feature → nodes and arcs."""
        if feature.geom_type != GEOM_LINESTRING:
            return
        klass = str(feature.tags.get("class")
                    or feature.tags.get("subclass") or "")
        if klass in NON_ROAD_CLASSES:
            return
        name = str(feature.tags.get("name") or "")
        oneway = _oneway(feature.tags)
        if not graph.extent:
            graph.extent = extent
        for path in feature.paths:
            keys: list[str] = []
            for px, py in path.points:
                gx = unit_x * extent + px
                gy = tile_y * extent + py
                key = str(gx) + "," + str(gy)
                if key not in graph.nodes:
                    graph.nodes[key] = _global_px_to_latlon(
                        gx, gy, zoom, extent)
                keys.append(key)
            if keys:
                # a way's ends are the seam bridger's candidates:
                # simplification breaks joins at way ends
                graph.terminals.add(keys[0])
                graph.terminals.add(keys[-1])
            for at in range(len(keys) - 1):
                a, b = keys[at], keys[at + 1]
                if a == b:
                    continue
                length = haversine_m(
                    LatLon(*graph.nodes[a]), LatLon(*graph.nodes[b]))
                if length <= 0.0:
                    continue
                if oneway >= 0:
                    self._arc(arcs, a, b, klass, name, length)
                if oneway <= 0:
                    self._arc(arcs, b, a, klass, name, length)

    @staticmethod
    def _arc(arcs: dict, a: str, b: str, klass: str, name: str,
             length: float) -> None:
        """One directed arc; a duplicated arc (tile overlap) keeps
        the shorter reading of the same segment."""
        key = (a, b, klass, name)
        if key not in arcs or length < arcs[key]:
            arcs[key] = length

    # ── seam bridging ───────────────────────────────────────────

    @staticmethod
    def _px(key: str) -> tuple[int, int]:
        """A node key's global-pixel coordinates."""
        gx, _, gy = key.partition(",")
        return int(gx), int(gy)

    def _bridge_seams(self, graph: _Graph, zoom: int) -> None:
        """Join dangling way ends to the network they stop short
        of (the routing-design pass; the module docstring states
        the law).  Runs once per fresh graph build, before the
        graph is cached: bridges are baked into the stored arcs
        (seam arcs under :data:`SEAM_CLASS`, split segments as
        ordinary arcs through a real new node), so a cache-loaded
        graph bridges identically.

        Method: components over the real arcs; every way END
        (harvest-recorded terminal with at least one real arc)
        finds its nearest attachment — node or segment interior,
        in global-pixel space — inside :data:`SEAM_JOIN_MAX_M`;
        an attachment in the end's own component means the end is
        already at its junction and bridges nothing; a segment
        attachment is recorded as a split at the projected point;
        the join itself is a zero-length seam arc each way.
        """
        if not graph.terminals or not graph.adj:
            return
        extent = graph.extent or 4096
        world = float(extent) * (1 << zoom)
        neighbors: dict[str, set[str]] = {k: set() for k in graph.nodes}
        segments: set[tuple[str, str]] = set()
        for a, arcs in graph.adj.items():
            for b, _len, _k, _n, syn in arcs:
                if syn:
                    continue
                neighbors[a].add(b)
                neighbors.setdefault(b, set()).add(a)
                segments.add((a, b) if a < b else (b, a))
        comp: dict[str, int] = {}
        for key in graph.nodes:
            if key in comp:
                continue
            cid = max(comp.values(), default=-1) + 1
            queue = deque([key])
            comp[key] = cid
            while queue:
                node = queue.popleft()
                for nxt in neighbors[node]:
                    if nxt not in comp:
                        comp[nxt] = cid
                        queue.append(nxt)
        terminals = sorted(k for k in graph.terminals
                           if neighbors.get(k))
        if not terminals:
            return
        px = {k: self._px(k) for k in graph.nodes}
        min_cos = min(
            math.cos(math.radians(lat))
            for lat, _lon in graph.nodes.values())
        cell = int(SEAM_JOIN_MAX_M * world
                   / (2.0 * math.pi * EARTH_RADIUS_M * min_cos)) + 1

        def cell_of(pt: tuple[int, int]) -> tuple[int, int]:
            return (pt[0] // cell, pt[1] // cell)

        node_cells: dict[tuple[int, int], list[str]] = {}
        for key, pt in px.items():
            node_cells.setdefault(cell_of(pt), []).append(key)
        seg_cells: dict[tuple[int, int], list[tuple[str, str]]] = {}
        for seg in sorted(segments):
            a, b = seg
            ax, ay = px[a]
            bx, by = px[b]
            steps = max(1, math.ceil(
                math.hypot(bx - ax, by - ay) / cell))
            seen_cells: set[tuple[int, int]] = set()
            for i in range(steps + 1):
                c = cell_of((round(ax + (bx - ax) * i / steps),
                             round(ay + (by - ay) * i / steps)))
                if c not in seen_cells:
                    seen_cells.add(c)
                    seg_cells.setdefault(c, []).append(seg)

        splits: dict[tuple[str, str], dict[str, float]] = {}
        seams: set[tuple[str, str]] = set()
        for t in terminals:
            tx, ty = px[t]
            lat_t = graph.nodes[t][0]
            metres_per_unit = (2.0 * math.pi * EARTH_RADIUS_M
                               / world) * math.cos(
                                   math.radians(lat_t))
            tol_units = SEAM_JOIN_MAX_M / metres_per_unit
            ci, cj = cell_of((tx, ty))
            best: tuple[float, str, Any] | None = None
            seen_segs: set[tuple[str, str]] = set()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for u in node_cells.get((ci + di, cj + dj), ()):
                        if u == t or u in neighbors[t]:
                            continue
                        dist = math.hypot(px[u][0] - tx,
                                          px[u][1] - ty)
                        if dist <= tol_units and (
                                best is None or dist < best[0]):
                            best = (dist, "node", u)
                    for seg in seg_cells.get((ci + di, cj + dj), ()):
                        if seg in seen_segs or t in seg:
                            continue
                        seen_segs.add(seg)
                        a, b = seg
                        ax, ay = px[a]
                        bx, by = px[b]
                        dx, dy = bx - ax, by - ay
                        span = dx * dx + dy * dy
                        if span == 0:
                            continue
                        frac = ((tx - ax) * dx
                                + (ty - ay) * dy) / span
                        frac = min(1.0, max(0.0, frac))
                        dist = math.hypot(tx - (ax + frac * dx),
                                          ty - (ay + frac * dy))
                        if dist <= tol_units and (
                                best is None or dist < best[0]):
                            best = (dist, "seg", (a, b, frac))
            if best is None:
                continue
            _dist, kind, payload = best
            if kind == "node":
                u = payload
                if comp[u] != comp[t]:
                    seams.add((t, u) if t < u else (u, t))
                continue
            a, b, frac = payload
            if comp[a] == comp[t]:
                continue    # already at its junction; bridge nothing
            if frac <= 0.02:
                if a not in neighbors[t]:
                    seams.add((t, a) if t < a else (a, t))
                continue
            if frac >= 0.98:
                if b not in neighbors[t]:
                    seams.add((t, b) if t < b else (b, t))
                continue
            ax, ay = px[a]
            bx, by = px[b]
            pkey = (str(round(ax + frac * (bx - ax))) + ","
                    + str(round(ay + frac * (by - ay))))
            if pkey in (a, b):
                if pkey not in neighbors[t]:
                    seams.add((t, pkey) if t < pkey else (pkey, t))
                continue
            seg = (a, b) if a < b else (b, a)
            splits.setdefault(seg, {})[pkey] = (
                frac if seg[0] == a else 1.0 - frac)
            if pkey != t:
                seams.add((t, pkey) if t < pkey else (pkey, t))

        for seg, proj in splits.items():
            a, b = seg
            chain = [a] + [k for k, _f in sorted(
                proj.items(), key=lambda kv: kv[1])] + [b]
            for key in proj:
                if key not in graph.nodes:
                    pgx, pgy = self._px(key)
                    graph.nodes[key] = _global_px_to_latlon(
                        pgx, pgy, zoom, extent)
            fwd = [(length, klass, name)
                   for (dst, length, klass, name, syn)
                   in graph.adj.get(a, []) if dst == b and not syn]
            rev = [(length, klass, name)
                   for (dst, length, klass, name, syn)
                   in graph.adj.get(b, []) if dst == a and not syn]
            if a in graph.adj:
                graph.adj[a] = [
                    e for e in graph.adj[a]
                    if not (e[0] == b and not e[4])]
            if b in graph.adj:
                graph.adj[b] = [
                    e for e in graph.adj[b]
                    if not (e[0] == a and not e[4])]
            for arcs_dir, nodes_seq in ((fwd, chain),
                                        (rev, chain[::-1])):
                for _len, klass, name in arcs_dir:
                    for i in range(len(nodes_seq) - 1):
                        u, v = nodes_seq[i], nodes_seq[i + 1]
                        graph.adj.setdefault(u, []).append((
                            v,
                            haversine_m(LatLon(*graph.nodes[u]),
                                        LatLon(*graph.nodes[v])),
                            klass, name, False))
        for t, u in sorted(seams):
            if t == u:
                continue
            graph.adj.setdefault(t, []).append(
                (u, 0.0, SEAM_CLASS, "", True))
            graph.adj.setdefault(u, []).append(
                (t, 0.0, SEAM_CLASS, "", True))
            graph.seam_arcs += 2

    def _graph(self, pack_dir: Path, manifest: PackManifest,
               corridor: BBox, zoom: int) -> _Graph:
        cache_key = self._cache_key(manifest, zoom, corridor)
        cached = self._load_cached(cache_key, manifest)
        if cached is not None:
            return cached
        graph = self._build_graph(pack_dir, manifest, corridor, zoom)
        self._persist(cache_key, manifest, zoom, corridor, graph)
        return graph

    def graph_stats(self, pack_dir: Path | str,
                    manifest: PackManifest, start: LatLon,
                    end: LatLon) -> GraphStats:
        """The corridor graph for a would-be route, as statistics.

        Raises ValueError naming the bound when the corridor
        spans more tiles than the cap — the same refusal
        :meth:`route` returns as a typed result.
        """
        corridor = _corridor(start, end)
        zoom = _graph_zoom(manifest)
        count = len(corridor.tiles(zoom))
        if count > MAX_CORRIDOR_TILES:
            raise ValueError(
                "corridor spans " + str(count) + " tiles at z"
                + str(zoom) + ", over the " + str(MAX_CORRIDOR_TILES)
                + "-tile corridor bound")
        graph = self._graph(Path(pack_dir), manifest, corridor, zoom)
        return GraphStats(
            zoom=zoom, node_count=len(graph.nodes),
            arc_count=sum(len(v) for v in graph.adj.values()),
            tiles_present=graph.tiles_present,
            corrupt_tiles=graph.corrupt_tiles, cached=graph.cached,
            seam_arcs=graph.seam_arcs)

    # ── routing ─────────────────────────────────────────────────

    @staticmethod
    def _nearest(graph: _Graph, point: LatLon) -> tuple[str, float]:
        """The nearest node key and its distance in metres."""
        best_key = ""
        best = math.inf
        for key, (lat, lon) in graph.nodes.items():
            dist = haversine_m(point, LatLon(lat=lat, lon=lon))
            if dist < best:
                best = dist
                best_key = key
        return best_key, best

    @staticmethod
    def _snap_candidates(
            graph: _Graph, point: LatLon, *,
            limit: int = SNAP_CANDIDATES,
            max_m: float = SNAP_MAX_M,
            ) -> tuple[list[tuple[str, float]], float]:
        """The endpoint's snap candidates: the ``limit`` nearest
        node keys inside ``max_m``, nearest first, each with its
        snap distance — plus the single nearest distance overall
        (the refusal's number when no candidate is in bounds)."""
        scored: list[tuple[float, str]] = []
        nearest = math.inf
        for key, (lat, lon) in graph.nodes.items():
            dist = haversine_m(point, LatLon(lat=lat, lon=lon))
            if dist < nearest:
                nearest = dist
            if dist <= max_m:
                scored.append((dist, key))
        scored.sort()
        return ([(key, dist) for dist, key in scored[:limit]],
                nearest)

    @staticmethod
    def _dijkstra(graph: _Graph, start_key: str, goal_key: str):
        """Shortest path by road distance: (node keys, arcs) or
        None when the goal is unreachable.  Arcs ride along as
        (length, class, name) triples parallel to the node list's
        steps."""
        found = Router._dijkstra_multi(
            graph, [start_key], [goal_key])
        if found is None:
            return None
        keys, arcs, _root, _goal = found
        return keys, [(length, klass, name)
                      for length, klass, name, _syn in arcs]

    @staticmethod
    def _dijkstra_multi(graph: _Graph, sources: list[str],
                        goals: list[str]):
        """Multi-source Dijkstra: the shortest path from any of
        ``sources`` to any of ``goals`` — (node keys, arcs, the
        source key used, the goal key reached) or None when no
        goal is reachable.  Arcs ride along as (length, class,
        name, synthetic) tuples parallel to the node list's
        steps.  This is what lets a snap take the candidate pair
        that actually routes, not only the nearest pair."""
        source_set = set(sources)
        goal_set = set(goals)
        dist: dict[str, float] = {key: 0.0 for key in sources}
        prev: dict[str, tuple[str, float, str, str, bool]] = {}
        heap: list[tuple[float, str]] = [(0.0, key) for key in sources]
        heapq.heapify(heap)
        settled: set[str] = set()
        goal: str | None = None
        while heap:
            at_dist, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled.add(node)
            if node in goal_set:
                goal = node
                break
            for nxt, length, klass, name, syn in graph.adj.get(
                    node, ()):
                cand = at_dist + length
                if cand < dist.get(nxt, math.inf):
                    dist[nxt] = cand
                    prev[nxt] = (node, length, klass, name, syn)
                    heapq.heappush(heap, (cand, nxt))
        if goal is None:
            return None
        keys = [goal]
        arcs: list[tuple[float, str, str, bool]] = []
        node = goal
        while node not in source_set:
            prior, length, klass, name, syn = prev[node]
            arcs.append((length, klass, name, syn))
            keys.append(prior)
            node = prior
        keys.reverse()
        arcs.reverse()
        return keys, arcs, node, goal

    @staticmethod
    def _steps(arcs: list[tuple[float, str, str, bool]],
               ) -> tuple[RouteStep, ...]:
        """Merge consecutive arcs of the same road into steps.
        Seam arcs are skipped: a synthetic join is not a road and
        never becomes a step (its zero length would add nothing;
        hiding it here keeps the step list a list of roads)."""
        merged: list[list] = []
        for length, klass, name, syn in arcs:
            if syn:
                continue
            if merged and merged[-1][0] == name \
                    and merged[-1][1] == klass:
                merged[-1][2] += length
            else:
                merged.append([name, klass, length])
        return tuple(
            RouteStep(name=name or klass or "road", road_class=klass,
                      length_m=length, named=bool(name))
            for name, klass, length in merged)

    # ── the route operation ─────────────────────────────────────

    @staticmethod
    def _ladder_floors() -> tuple[float, ...]:
        """The widening ladder's pad floors, shipped pad first:
        CORRIDOR_PAD_MIN_DEG growing by CORRIDOR_WIDEN_FACTOR to
        the CORRIDOR_WIDEN_MAX_DEG ceiling."""
        floors = [CORRIDOR_PAD_MIN_DEG]
        while floors[-1] < CORRIDOR_WIDEN_MAX_DEG:
            floors.append(min(
                CORRIDOR_WIDEN_MAX_DEG,
                floors[-1] * CORRIDOR_WIDEN_FACTOR))
        return tuple(floors)

    @staticmethod
    def _suspect(result: RouteResult, ratio: float | None) -> bool:
        """Whether one corridor attempt's answer is suspect: a
        route whose ratio says the corridor may have bound it,
        or a no-route of a class a bigger corridor can honestly
        answer differently."""
        if result.ok:
            return ratio is not None and ratio > ROUTE_RATIO_SUSPECT
        return result.reason in _LADDER_TRIGGER_REASONS

    @staticmethod
    def _detour_class(attempts: list[_Attempt],
                      best: _Attempt | None) -> str:
        """Classify the answer's ratio from the ladder's own
        measurements — never from the ratio alone: widening must
        have *moved* the answer (a route where the shipped
        corridor had none, or a materially shorter one) before
        the corridor is called the binding constraint; a suspect
        answer the ladder could not move is the network's own
        shape; anything else sits in the healthy envelope."""
        first = attempts[0]
        if best is not None and best is not first:
            if not first.result.ok:
                return DETOUR_CORRIDOR_BOUND
            if best.result.distance_m < first.result.distance_m * (
                    1.0 - _IMPROVEMENT_MIN_FRACTION):
                return DETOUR_CORRIDOR_BOUND
        winner = best if best is not None else attempts[-1]
        if Router._suspect(winner.result, winner.ratio):
            return DETOUR_NETWORK_DETOUR
        return DETOUR_UNREMARKABLE

    @staticmethod
    def _ladder_sentence(attempts: list[_Attempt],
                         detour: str) -> str:
        """The ladder's line in the route's detail: the attempts
        and their measured ratios, named like the seam joins —
        a widened answer is labelled, never silent."""
        ratios = ", ".join(
            "no route" if attempt.ratio is None
            else format(attempt.ratio, ".2f")
            for attempt in attempts)
        gloss = {
            DETOUR_CORRIDOR_BOUND:
                "the shipped corridor was the binding constraint",
            DETOUR_NETWORK_DETOUR:
                "the detour is the road network's own shape",
            DETOUR_UNREMARKABLE:
                "inside the healthy envelope",
        }[detour]
        return (" · corridor widening: the first attempt was"
                " suspect, so the corridor was re-materialized"
                " over " + str(len(attempts)) + " attempts (pad "
                + format(attempts[0].pad_deg, ".2f") + "° → "
                + format(attempts[-1].pad_deg, ".2f")
                + "°, ratios " + ratios + ") and the best"
                " measured answer stands — detour class: "
                + detour + " (" + gloss + ")")

    def route(self, pack_dir: Path | str, manifest: PackManifest,
              start: LatLon, end: LatLon, *,
              from_label: str = "", to_label: str = "",
              ) -> RouteResult:
        """Route ``start`` → ``end`` over one pack's road graph.

        Local first (doc 21 S3): the local answer — a route or a
        named no-route — is final unless federation is on, in
        which case a local no-route may fall to the wired
        provider (none ships; the fall is named when unwired).
        Every route mints its OP_ROUTE receipt when a store is
        attached.

        The first corridor attempt is the shipped corridor,
        byte-identical to the router before the widening ladder;
        the ladder (see the module docstring) climbs only from a
        suspect first answer, and the receipt is minted once,
        for the final answer the ladder settles on.
        """
        pack_dir = Path(pack_dir)
        base = RouteResult(
            ok=False, from_label=from_label, to_label=to_label,
            from_point=(start.lat, start.lon),
            to_point=(end.lat, end.lon), pack_id=manifest.pack_id)
        if manifest.tile_format != "mvt":
            return self._no_route(
                base, manifest, REASON_NOT_VECTOR,
                "pack " + manifest.name + " is a "
                + manifest.tile_format + " pack — raster tiles"
                " carry no road graph to route over",
                OUTCOME_UNAVAILABLE)
        zoom = _graph_zoom(manifest)
        straight = haversine_m(start, end)
        attempts: list[_Attempt] = []
        best: _Attempt | None = None
        ladder_stop = ""
        previous: BBox | None = None
        floors = self._ladder_floors()
        for step, pad_floor in enumerate(floors):
            corridor = _corridor(start, end, pad_floor=pad_floor)
            if previous is not None and corridor == previous:
                # The fraction term already pads past this floor:
                # a wider floor would rebuild the same graph.
                ladder_stop = "corridor-unchanged"
                break
            tile_count = len(corridor.tiles(zoom))
            if tile_count > MAX_CORRIDOR_TILES:
                if step == 0:
                    return self._no_route(
                        base, manifest, REASON_CORRIDOR_TOO_LARGE,
                        "the route corridor spans " + str(tile_count)
                        + " tiles at z" + str(zoom) + ", over the "
                        + str(MAX_CORRIDOR_TILES)
                        + "-tile corridor bound"
                        " — route in legs, or install a deeper pack",
                        OUTCOME_UNAVAILABLE)
                # Checked before the step is attempted, exactly
                # as the first attempt's cap is: the best answer
                # so far stands.
                ladder_stop = "tile-cap"
                break
            result, outcome = self._route_local(
                pack_dir, manifest, base, start, end, corridor, zoom)
            ratio: float | None = None
            if result.ok:
                ratio = (result.distance_m / straight
                         if straight > 0.0 else 0.0)
            attempt = _Attempt(pad_deg=pad_floor, tiles=tile_count,
                               result=result, outcome=outcome,
                               ratio=ratio)
            attempts.append(attempt)
            previous = corridor
            if result.ok and (best is None or result.distance_m
                              < best.result.distance_m):
                best = attempt
            if not self._suspect(result, ratio):
                if step > 0:
                    ladder_stop = "not-suspect"
                break
            if step == len(floors) - 1:
                ladder_stop = "ceiling"
        winner = best if best is not None else attempts[-1]
        detour = self._detour_class(attempts, best)
        ticket = {
            "corridor_pad_deg": winner.pad_deg,
            "ladder_steps": tuple(
                LadderStep(
                    pad_deg=attempt.pad_deg, tiles=attempt.tiles,
                    distance_m=(attempt.result.distance_m
                                if attempt.result.ok else None),
                    ratio=attempt.ratio,
                    outcome=("route" if attempt.result.ok
                             else attempt.result.reason))
                for attempt in attempts),
            "widened": winner is not attempts[0],
            "ladder_stop": ladder_stop,
            "detour_class": detour,
        }
        sentence = ""
        if len(attempts) > 1:
            sentence = self._ladder_sentence(attempts, detour)
        if winner.result.ok:
            result = replace(winner.result,
                             detail=winner.result.detail + sentence,
                             **ticket)
            receipt = self._mint(manifest, OUTCOME_OK, POSTURE_LOCAL,
                                 "local", result.detail)
            return replace(result, receipt=receipt)
        return replace(
            self._no_route(
                base, manifest, winner.result.reason,
                winner.result.detail + sentence, winner.outcome,
                from_snap=winner.result.from_snap_m,
                to_snap=winner.result.to_snap_m),
            **ticket)

    def _route_local(self, pack_dir: Path, manifest: PackManifest,
                     base: RouteResult, start: LatLon, end: LatLon,
                     corridor: BBox, zoom: int,
                     ) -> tuple[RouteResult, str]:
        """One corridor attempt's local answer: the shipped
        routing body over one materialized corridor — the route,
        or the named no-route — with no receipt minted and no
        federation posture applied (:meth:`route` owns both,
        once, for the ladder's final answer).  The second return
        is the receipt outcome class a no-route carries
        (OUTCOME_OK for the truthful "disconnected",
        OUTCOME_UNAVAILABLE when the plane could not serve);
        ignored for a route.  The corridor's tile cap is the
        caller's check, made before the attempt."""
        graph = self._graph(pack_dir, manifest, corridor, zoom)
        if graph.tiles_present == 0:
            return (replace(
                base, reason=REASON_NO_ROAD_DATA,
                detail="pack " + manifest.name + " holds no tiles"
                " for the route corridor — no road data there"),
                OUTCOME_UNAVAILABLE)
        if not graph.nodes:
            return (replace(
                base, reason=REASON_NO_ROAD_DATA,
                detail="no roads in the route corridor of pack "
                + manifest.name),
                OUTCOME_UNAVAILABLE)
        from_cands, from_nearest = self._snap_candidates(graph, start)
        if not from_cands:
            return (replace(
                base, reason=REASON_SNAP_FROM,
                detail="the start point cannot snap to the road"
                " network: its nearest road node is "
                + format(from_nearest, ",.0f") + " m away, beyond"
                " the " + format(SNAP_MAX_M, ",.0f")
                + " m snap bound"),
                OUTCOME_UNAVAILABLE)
        to_cands, to_nearest = self._snap_candidates(graph, end)
        if not to_cands:
            return (replace(
                base, reason=REASON_SNAP_TO,
                detail="the end point cannot snap to the road"
                " network: its nearest road node is "
                + format(to_nearest, ",.0f") + " m away, beyond"
                " the " + format(SNAP_MAX_M, ",.0f")
                + " m snap bound"),
                OUTCOME_UNAVAILABLE)
        found = self._dijkstra_multi(
            graph, [key for key, _d in from_cands],
            [key for key, _d in to_cands])
        if found is None:
            return (replace(
                base, reason=REASON_DISCONNECTED,
                detail="the road graph has no connected path"
                " between the two points — the roads near each"
                " end do not meet in pack " + manifest.name,
                from_snap_m=from_cands[0][1],
                to_snap_m=to_cands[0][1]),
                OUTCOME_OK)
        keys, arcs, from_key, to_key = found
        from_snap = dict(from_cands)[from_key]
        to_snap = dict(to_cands)[to_key]
        seam_joins = sum(1 for _l, _k, _n, syn in arcs if syn)
        distance = sum(length for length, _k, _n, _s in arcs)
        duration = sum(
            length / _speed_ms(klass)
            for length, klass, _n, _s in arcs)
        points: list[tuple[float, float]] = [(start.lat, start.lon)]
        points.extend(graph.nodes[key] for key in keys)
        points.append((end.lat, end.lon))
        geometry = tuple(
            point for at, point in enumerate(points)
            if at == 0 or point != points[at - 1])
        detail = ("route over pack " + manifest.name + ": "
                  + format(distance, ",.0f") + " m by road, ~"
                  + str(int(round(duration)))
                  + " s at class speeds (estimates)")
        if seam_joins:
            detail += (
                " · crosses " + str(seam_joins) + " seam join"
                + ("s" if seam_joins != 1 else "")
                + " (zero-distance digitization repairs, labelled,"
                " never counted as road)")
        result = replace(
            base, ok=True, geometry=geometry, distance_m=distance,
            duration_s=duration, steps=self._steps(arcs),
            from_snap_m=from_snap, to_snap_m=to_snap,
            seam_joins=seam_joins, detail=detail)
        return result, OUTCOME_OK

    def _no_route(self, base: RouteResult, manifest: PackManifest,
                  reason: str, detail: str, outcome: str, *,
                  from_snap: float | None = None,
                  to_snap: float | None = None) -> RouteResult:
        """A local no-route, with the federation posture applied.

        Federation off → the local answer is final, and says so.
        Federation on, nothing wired → the fall is named, the
        local answer stands.  A wired provider that answers ok
        takes the result (its own receipt is minted here); a
        provider that fails or has no route is named, and the
        local answer stands behind it.
        """
        result = replace(
            base, reason=reason, detail=detail,
            from_snap_m=from_snap, to_snap_m=to_snap)
        if not self._federation:
            return replace(
                result,
                detail=detail + " Local-first (federation off):"
                " nothing was fetched from a provider.",
                receipt=self._mint(manifest, outcome, POSTURE_LOCAL,
                                   "local",
                                   reason + ": " + detail))
        if self._provider is None:
            note = ("federation is on but no route provider is"
                    " configured — the local answer stands")
            return replace(
                result, posture=POSTURE_FEDERATED, provider="none",
                detail=detail + " " + note + ".",
                receipt=self._mint(
                    manifest, outcome, POSTURE_FEDERATED, "none",
                    reason + ": " + detail + " · " + note))
        try:
            found = self._provider.route(
                LatLon(*base.from_point), LatLon(*base.to_point))
        except Exception as exc:  # noqa: BLE001 — a provider's
            # failure is an answer about the provider, not a crash
            note = ("provider " + str(self._provider.name)
                    + " failed (" + type(exc).__name__
                    + ") — the local answer stands")
            return replace(
                result, posture=POSTURE_FEDERATED,
                provider=str(self._provider.name),
                detail=detail + " " + note + ".",
                receipt=self._mint(
                    manifest, outcome, POSTURE_FEDERATED,
                    str(self._provider.name),
                    reason + ": " + detail + " · " + note))
        if found is not None and found.ok:
            note = ("answered by provider "
                    + str(self._provider.name)
                    + " after the local graph had no route")
            adopted = replace(
                found, from_label=base.from_label,
                to_label=base.to_label,
                from_point=base.from_point, to_point=base.to_point,
                pack_id=manifest.pack_id,
                posture=POSTURE_FEDERATED,
                provider=str(self._provider.name),
                detail=found.detail + " " + note + ".")
            return replace(
                adopted,
                receipt=self._mint(
                    manifest, OUTCOME_OK, POSTURE_FEDERATED,
                    str(self._provider.name),
                    "route via provider " + str(self._provider.name)
                    + ": " + format(found.distance_m, ",.0f")
                    + " m"))
        note = ("provider " + str(self._provider.name)
                + " had no route either — the local answer stands")
        return replace(
            result, posture=POSTURE_FEDERATED,
            provider=str(self._provider.name),
            detail=detail + " " + note + ".",
            receipt=self._mint(
                manifest, outcome, POSTURE_FEDERATED,
                str(self._provider.name),
                reason + ": " + detail + " · " + note))

    def _mint(self, manifest: PackManifest, outcome: str,
              posture: str, provider: str,
              detail: str) -> MapReceipt | None:
        """Mint + store the OP_ROUTE receipt, when a store rides."""
        if self._store is None:
            return None
        receipt = MapReceipt.mint(
            OP_ROUTE, outcome,
            posture=(POSTURE_ONLINE if posture == POSTURE_FEDERATED
                     else POSTURE_OFFLINE),
            provider=provider, pack_refs=(manifest.pack_id,),
            pack_digests=(manifest.digest,), detail=detail)
        self._store.append(receipt)
        return receipt
