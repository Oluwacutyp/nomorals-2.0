"""Devon Atlas: Devon's own map system (docs 19+21, adopted 2026-11-10).

Doc 21 governs the map leg: Devon-built packs are the basemap
(S4), releases are signed and hash-pinned (S2), the geocoder and
router are local first (S3), providers are optional adapters (S1),
and any renderer — MapLibre included — sits behind Devon's own
boundary (S5).  The product is Atlas: data, style, consent, agent.

M1 is the foundation pass: the coordinate/GeoJSON schemas
(:mod:`nomorals.maps.coords`, :mod:`nomorals.maps.geojson`), the
pack manifest contract (:mod:`nomorals.maps.manifest`), the
MapReceipt + its store (:mod:`nomorals.maps.receipts`), the
renderer boundary proven by an offline fixture renderer
(:mod:`nomorals.maps.render`), pack operations
(:mod:`nomorals.maps.packs`) and settings
(:mod:`nomorals.maps.settings`).  M3 adds the vector plane: the
MVT decoder (:mod:`nomorals.maps.mvt`), the Devon Style document
(:mod:`nomorals.maps.style`) and the vector renderer behind the
same boundary (:mod:`nomorals.maps.vector_render`).  M4 adds the
local geocoder (:mod:`nomorals.maps.gazetteer`): a gazetteer
indexed from the packs themselves, forward/reverse queries, and
the owner-policy federation seam.  M5 adds the routing plane
(:mod:`nomorals.maps.routing`): a corridor-scoped road graph
extracted from a pack's own tiles, Dijkstra over it, route
receipts, and the same federation posture for the router; the
renderer gains the overlay seam routes draw through
(:class:`~nomorals.maps.vector_render.MapOverlay`).  The
basemap pipeline (M2), the live-location broker (M6, on
``location/``'s ConsentStore) and the MapAgent (M7) build on
these contracts in their own passes.
"""

from __future__ import annotations

from .coords import (
                     MAX_ZOOM,
                     MERCATOR_MAX_LAT,
                     BBox,
                     TileCoord,
                     clamp_mercator_lat,
                     lat_to_tile_y,
                     lat_to_y_float,
                     lon_to_tile_x,
                     lon_to_x_float,
                     tile_x_to_lon,
                     tile_y_to_lat,
                     validate_lat,
                     validate_lon,
)
from .gazetteer import (
    GAZETTEER_MAX_ZOOM,
    INDEX_FILENAME,
    KIND_WEIGHTS,
    POSTURE_FEDERATED,
    POSTURE_LOCAL,
    REVERSE_LADDER_KM,
    Gazetteer,
    GazetteerEntry,
    GeocodeCandidate,
    GeocodeProvider,
    GeocodeResult,
    IndexBuildResult,
    PackIndexResult,
    ReverseCandidate,
    ReverseResult,
    match_rank,
    normalize_name,
)
from .geojson import GeoValidation, validate_geojson
from .manifest import (
                     FORMAT_VERSION,
                     MANIFEST_NAME,
                     PackError,
                     PackManifest,
                     PackMember,
                     PackVerification,
                     build_manifest,
                     hash_bytes,
                     load_manifest,
                     tile_relpath,
                     verify_pack,
                     verify_signature,
                     write_manifest,
)
from .mvt import (
    MvtError,
    MvtFeature,
    MvtLayer,
    MvtPath,
    MvtTile,
    decode_mvt,
)
from .packs import render_pack_preview, verify_pack_dir
from .receipts import OPERATIONS, MapReceipt, MapReceiptStore, new_receipt_id
from .render import FixtureRenderer, MapRenderer, RenderResult, StdlibPngBackend
from .routing import (
    GRAPH_FILENAME,
    MAX_CORRIDOR_TILES,
    ROUTE_ZOOM,
    SNAP_MAX_M,
    SPEED_KMH,
    GraphStats,
    RouteProvider,
    Router,
    RouteResult,
    RouteStep,
)
from .settings import MapSettings, map_settings
from .style import DEFAULT_STYLE, MapStyle, StyleError, StyleLayer, style_hash
from .vector_render import LabelSpec, MapOverlay, PilTextBackend, TextBackend, VectorRenderer

__all__ = [
    "DEFAULT_STYLE", "FORMAT_VERSION", "GAZETTEER_MAX_ZOOM",
    "GRAPH_FILENAME", "INDEX_FILENAME", "KIND_WEIGHTS",
    "MANIFEST_NAME", "MAX_CORRIDOR_TILES", "MAX_ZOOM",
    "MERCATOR_MAX_LAT", "OPERATIONS", "POSTURE_FEDERATED",
    "POSTURE_LOCAL", "REVERSE_LADDER_KM", "ROUTE_ZOOM", "BBox",
    "FixtureRenderer", "Gazetteer", "GazetteerEntry", "GeoValidation",
    "GeocodeCandidate", "GeocodeProvider", "GeocodeResult",
    "GraphStats", "IndexBuildResult", "LabelSpec", "MapOverlay",
    "MapReceipt", "MapRenderer", "MapReceiptStore", "MapSettings",
    "MapStyle", "MvtError", "MvtFeature", "MvtLayer", "MvtPath",
    "MvtTile", "PackError", "PackIndexResult", "PackManifest",
    "PackMember", "PackVerification", "PilTextBackend", "RenderResult",
    "ReverseCandidate", "ReverseResult", "RouteProvider",
    "RouteResult", "RouteStep", "Router", "SNAP_MAX_M", "SPEED_KMH",
    "StdlibPngBackend", "StyleError", "StyleLayer", "TextBackend",
    "TileCoord", "VectorRenderer", "build_manifest",
    "clamp_mercator_lat", "decode_mvt", "hash_bytes", "lat_to_tile_y",
    "lat_to_y_float", "load_manifest", "lon_to_tile_x",
    "lon_to_x_float", "map_settings", "match_rank", "new_receipt_id",
    "normalize_name", "render_pack_preview", "style_hash",
    "tile_relpath", "tile_x_to_lon", "tile_y_to_lat",
    "validate_geojson", "validate_lat", "validate_lon",
    "verify_pack", "verify_pack_dir", "verify_signature",
    "write_manifest",
]
