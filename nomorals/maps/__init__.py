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
(:mod:`nomorals.maps.settings`).  M2 adds the basemap pipeline
(:mod:`nomorals.maps.pipeline` over the source/pack definitions
in :mod:`nomorals.maps.definitions`, the release registry in
:mod:`nomorals.maps.registry`, and the Planetiler/MBTiles +
Natural Earth raster builders).  M3 adds the vector plane: the
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
(:class:`~nomorals.maps.vector_render.MapOverlay`).  M7 adds
the owner's overlays (:mod:`nomorals.maps.overlays`): named
marker/polyline/polygon documents, validated through the M1
GeoJSON gate before decoded data becomes a layer, stored as
data and converted onto the same overlay seam.  The
live-location broker (M6, on ``location/``'s ConsentStore)
builds on these contracts in its own pass.  M9 adds personal
marks (:mod:`nomorals.maps.marks`): the owner's own saved
places, captured only through an armed ``/map mark`` (the
command message's location, or the pin that completes a durable
pending mark), stored beside the overlays in the same idiom,
and resolved ahead of the gazetteer everywhere places are
looked up.
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
from .definitions import PACKS, SOURCES, MapSource, PackDefinition, get_pack, get_source
from .gazetteer import (
    GAZETTEER_MAX_ZOOM,
    INDEX_BUILD_COMMAND,
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
    PackIndexStatus,
    ReverseCandidate,
    ReverseResult,
    index_status_line,
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
from .marks import (
    MAX_LABEL_CHARS,
    METHOD_MESSAGE,
    METHOD_PENDING,
    PENDING_TTL_S,
    Mark,
    MarkError,
    MarkStore,
    PendingMark,
    normalize_label,
    pending_key,
)
from .mvt import (
    MvtError,
    MvtFeature,
    MvtLayer,
    MvtPath,
    MvtTile,
    decode_mvt,
)
from .overlays import (
    Overlay,
    OverlayError,
    OverlayFeature,
    OverlayStore,
    geojson_to_features,
    overlay_bbox,
    overlay_to_map_overlay,
    validate_features,
)
from .packs import render_pack_preview, verify_pack_dir
from .pipeline import BuildReport, PipelineError, run_build
from .receipts import OPERATIONS, MapReceipt, MapReceiptStore, new_receipt_id
from .registry import ReleaseEntry, ReleaseRegistry
from .render import FixtureRenderer, MapRenderer, RenderResult, StdlibPngBackend
from .routing import (
    GRAPH_FILENAME,
    MAX_CORRIDOR_TILES,
    ROUTE_ZOOM,
    SEAM_CLASS,
    SEAM_JOIN_MAX_M,
    SNAP_CANDIDATES,
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
    "MANIFEST_NAME", "MAX_CORRIDOR_TILES", "MAX_LABEL_CHARS",
    "MAX_ZOOM", "METHOD_MESSAGE", "METHOD_PENDING",
    "MERCATOR_MAX_LAT", "OPERATIONS", "Overlay", "OverlayError",
    "OverlayFeature", "OverlayStore", "PACKS", "PENDING_TTL_S",
    "POSTURE_FEDERATED", "POSTURE_LOCAL", "REVERSE_LADDER_KM",
    "ROUTE_ZOOM", "SOURCES",
    "BBox", "BuildReport",
    "FixtureRenderer", "Gazetteer", "GazetteerEntry", "GeoValidation",
    "GeocodeCandidate", "GeocodeProvider", "GeocodeResult",
    "GraphStats", "INDEX_BUILD_COMMAND", "IndexBuildResult",
    "LabelSpec", "MapOverlay",
    "MapReceipt", "MapRenderer", "MapReceiptStore", "MapSettings",
    "MapSource", "MapStyle", "Mark", "MarkError", "MarkStore",
    "MvtError", "MvtFeature", "MvtLayer",
    "MvtPath", "MvtTile", "PackDefinition", "PackError",
    "PackIndexResult", "PackIndexStatus", "PackManifest", "PackMember",
    "PackVerification", "PendingMark", "PilTextBackend",
    "PipelineError",
    "ReleaseEntry", "ReleaseRegistry", "RenderResult",
    "ReverseCandidate", "ReverseResult", "RouteProvider",
    "RouteResult", "RouteStep", "Router", "SEAM_CLASS",
    "SEAM_JOIN_MAX_M", "SNAP_CANDIDATES", "SNAP_MAX_M", "SPEED_KMH",
    "StdlibPngBackend", "StyleError", "StyleLayer", "TextBackend",
    "TileCoord", "VectorRenderer", "build_manifest",
    "clamp_mercator_lat", "decode_mvt", "geojson_to_features",
    "get_pack", "get_source", "hash_bytes", "lat_to_tile_y",
    "lat_to_y_float", "load_manifest", "lon_to_tile_x",
    "index_status_line", "lon_to_x_float", "map_settings",
    "match_rank", "new_receipt_id",
    "normalize_label", "normalize_name", "overlay_bbox",
    "overlay_to_map_overlay", "pending_key",
    "render_pack_preview", "run_build", "style_hash",
    "tile_relpath", "tile_x_to_lon", "tile_y_to_lat",
    "validate_features", "validate_geojson", "validate_lat",
    "validate_lon",
    "verify_pack", "verify_pack_dir", "verify_signature",
    "write_manifest",
]
