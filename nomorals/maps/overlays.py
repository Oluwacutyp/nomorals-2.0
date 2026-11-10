"""User overlays: the owner's own map layers, as data (docs 19+21 M7).

An overlay is a named document of features — markers, polylines,
polygons — that rides over the basemap packs through the M5
overlay seam (:class:`nomorals.maps.vector_render.MapOverlay`).
Overlays are the places/tasks/geofences layer of Devon Atlas:
authored by the owner (the /map overlay verbs, the agent, or a
validated GeoJSON import), stored as data, rendered on demand.
Nothing here fetches: pure data + validation + storage +
conversion, inside the maps package's no-network discipline.

Validation follows the M1 GeoJSON gate's discipline
(:mod:`nomorals.maps.geojson`): every position is validated
through :mod:`nomorals.maps.coords`, counts are capped so a
bloated or hostile document cannot make the store or the render
the denial of service, and every defect is a NAMED error — an
overlay that fails validation is refused whole, never partially
kept.

Decoded GeoJSON becomes an overlay only through
:func:`geojson_to_features`, which runs the M1
:func:`~nomorals.maps.geojson.validate_geojson` gate FIRST and
refuses the payload whole when the gate refuses it (the decoder
gate of doc 21 §6: nothing decoded becomes a layer unvalidated).

Storage follows the glossary store's idiom
(``llm/glossary.py``): when the context carries the shared db,
overlays land in the runtime-created ``map_overlays`` table
(migrations are frozen); with no db they live in
``<home>/maps/overlays.json``, written atomically (tmp +
replace).  Reads never raise.

RENDERING HONESTY: the MapOverlay seam carries markers,
polylines and polygons.  A polygon feature converts to a seam
polygon (outer ring + hole rings, carried whole from the
GeoJSON) and renders FILLED — fill color/opacity and the ring
stroke are the style document's overlay paint, never hardcoded
here or in the renderer.  A polygon whose geometry cannot be
filled honestly (a self-intersecting or degenerate ring, a
hole outside its outer ring) converts to its closed outline
instead, with a note naming the defect; surfaces and tool
results carry the note.  A polygon is never silently dropped
and never dressed up as filled when it is not.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger
from .coords import BBox, validate_lat, validate_lon
from .geojson import validate_geojson
from .vector_render import MapOverlay, OverlayPolygon, polygon_defects

_log = get_logger(__name__)

__all__ = [
    "KIND_MARKER", "KIND_POLYGON", "KIND_POLYLINE",
    "MAX_OVERLAY_FEATURES", "MAX_OVERLAY_POINTS",
    "Overlay", "OverlayError", "OverlayFeature", "OverlayStore",
    "geojson_to_features", "overlay_bbox", "overlay_to_map_overlay",
    "validate_features",
]

KIND_MARKER = "marker"
KIND_POLYLINE = "polyline"
KIND_POLYGON = "polygon"
_FEATURE_KINDS = (KIND_MARKER, KIND_POLYLINE, KIND_POLYGON)

#: Owner-data bounds, in the M1 GeoJSON gate's spirit: generous
#: for real layers, fatal for bombs.  Exceeding either is a
#: validation error naming the cap, never a crash.
MAX_OVERLAY_FEATURES = 500
MAX_OVERLAY_POINTS = 20_000

_DDL = (
    "CREATE TABLE IF NOT EXISTS map_overlays ("
    "overlay_id TEXT PRIMARY KEY, name TEXT, payload TEXT)"
)


class OverlayError(ValueError):
    """A named overlay defect: invalid data, a name clash, a miss."""


@dataclass(frozen=True)
class OverlayFeature:
    """One drawn thing: a marker (1 point), a polyline (>=2), or
    a polygon ring (>=3).  ``points`` are (lat, lon) pairs;
    ``holes`` (polygons only) are the polygon's interior rings,
    each a ring of (lat, lon) pairs — carried whole so the fill
    can leave them unfilled."""

    kind: str
    points: tuple[tuple[float, float], ...] = ()
    label: str = ""
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "points": [[lat, lon] for lat, lon in self.points],
            "label": self.label}
        if self.holes:
            data["holes"] = [[[lat, lon] for lat, lon in ring]
                             for ring in self.holes]
        return data

    @classmethod
    def from_dict(cls, data: Any) -> OverlayFeature:
        if not isinstance(data, dict):
            raise OverlayError("overlay feature must be an object")

        def _ring(raw_ring: Any) -> tuple[tuple[float, float], ...]:
            points = []
            for raw in raw_ring or ():
                if not isinstance(raw, (list, tuple)) \
                        or len(raw) != 2:
                    raise OverlayError(
                        "overlay feature point must be [lat, lon]")
                points.append((float(raw[0]), float(raw[1])))
            return tuple(points)

        return cls(kind=str(data.get("kind", "")),
                   points=_ring(data.get("points") or ()),
                   label=str(data.get("label", "") or ""),
                   holes=tuple(_ring(r) for r in data.get("holes")
                               or ()))


@dataclass(frozen=True)
class Overlay:
    """One named overlay document: features + provenance."""

    overlay_id: str
    name: str
    created_at: float
    features: tuple[OverlayFeature, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"overlay_id": self.overlay_id, "name": self.name,
                "created_at": self.created_at,
                "features": [f.to_dict() for f in self.features]}

    @classmethod
    def from_dict(cls, data: dict) -> Overlay:
        return cls(
            overlay_id=str(data.get("overlay_id", "")),
            name=str(data.get("name", "")),
            created_at=float(data.get("created_at", 0.0)),
            features=tuple(OverlayFeature.from_dict(f)
                           for f in data.get("features", ())))


def validate_features(features: Iterable[OverlayFeature],
                      ) -> tuple[str, ...]:
    """Every defect in a feature set, named.  Empty tuple = valid."""
    errors: list[str] = []
    feats = list(features)
    if not feats:
        errors.append("an overlay holds at least one feature")
    if len(feats) > MAX_OVERLAY_FEATURES:
        errors.append(
            "feature count exceeds the cap of "
            + str(MAX_OVERLAY_FEATURES))
    total = 0
    for i, feat in enumerate(feats):
        where = "feature " + str(i)
        if feat.kind not in _FEATURE_KINDS:
            errors.append(where + ": unknown kind " + repr(feat.kind)
                          + " (marker | polyline | polygon)")
            continue
        if not isinstance(feat.label, str):
            errors.append(where + ": label must be text")
        need = {KIND_MARKER: 1, KIND_POLYLINE: 2,
                KIND_POLYGON: 3}[feat.kind]
        if feat.kind == KIND_MARKER and len(feat.points) != 1:
            errors.append(where + ": a marker holds exactly 1 point, "
                          "got " + str(len(feat.points)))
        elif feat.kind != KIND_MARKER and len(feat.points) < need:
            errors.append(where + ": a " + feat.kind + " holds at "
                          "least " + str(need) + " points, got "
                          + str(len(feat.points)))
        total += len(feat.points)
        for j, (lat, lon) in enumerate(feat.points):
            try:
                validate_lat(float(lat))
                validate_lon(float(lon))
            except (TypeError, ValueError) as exc:
                errors.append(where + " point " + str(j) + ": "
                              + str(exc))
        if feat.holes and feat.kind != KIND_POLYGON:
            errors.append(where + ": holes belong to a polygon")
        for k, hole in enumerate(feat.holes):
            if len(hole) < 3:
                errors.append(where + " hole " + str(k) + ": a "
                              "hole ring holds at least 3 points, "
                              "got " + str(len(hole)))
            total += len(hole)
            for j, (lat, lon) in enumerate(hole):
                try:
                    validate_lat(float(lat))
                    validate_lon(float(lon))
                except (TypeError, ValueError) as exc:
                    errors.append(where + " hole " + str(k)
                                  + " point " + str(j) + ": "
                                  + str(exc))
    if total > MAX_OVERLAY_POINTS:
        errors.append("point count exceeds the cap of "
                      + str(MAX_OVERLAY_POINTS))
    return tuple(errors)


def overlay_bbox(overlay: Overlay) -> BBox:
    """The box covering every feature point (unpadded)."""
    lats = [p[0] for f in overlay.features for p in f.points]
    lons = [p[1] for f in overlay.features for p in f.points]
    if not lats:
        raise OverlayError("overlay " + repr(overlay.name)
                           + " has no points to bound")
    return BBox(west=min(lons), south=min(lats),
                east=max(lons), north=max(lats))


def overlay_to_map_overlay(overlay: Overlay,
                           ) -> tuple[MapOverlay, tuple[str, ...]]:
    """Convert to the renderer's overlay seam, honestly.

    Markers ride as markers, polylines as polylines; a polygon
    rides as a seam polygon — outer ring + holes — and renders
    filled in the style document's overlay paint.  A polygon
    whose rings cannot be filled honestly
    (:func:`polygon_defects`: a self-intersecting or degenerate
    ring, a hole outside its outer ring) rides as its CLOSED
    outline instead and earns a note naming the defect — never
    a silent wrong fill.
    """
    markers: list[tuple[float, float]] = []
    polylines: list[tuple[tuple[float, float], ...]] = []
    polygons: list[OverlayPolygon] = []
    notes: list[str] = []
    for feat in overlay.features:
        if feat.kind == KIND_MARKER:
            markers.extend(feat.points)
        elif feat.kind == KIND_POLYLINE:
            polylines.append(feat.points)
        elif feat.kind == KIND_POLYGON:
            rings = (feat.points,) + tuple(feat.holes)
            defects = polygon_defects(rings)
            label = feat.label or "unnamed"
            if defects:
                ring = feat.points
                if ring and ring[0] != ring[-1]:
                    ring = ring + (ring[0],)
                polylines.append(ring)
                notes.append(
                    "polygon " + repr(label) + " drawn as an "
                    "outline only — " + defects[0] + ", so its "
                    "interior is not shaded")
            else:
                polygons.append(OverlayPolygon(
                    rings=rings, label=feat.label))
    return (MapOverlay(polylines=tuple(polylines),
                       markers=tuple(markers),
                       polygons=tuple(polygons)),
            tuple(notes))


def geojson_to_features(data: Any,
                        ) -> tuple[tuple[OverlayFeature, ...],
                                   tuple[str, ...]]:
    """The decoder gate: decoded GeoJSON -> overlay features.

    The M1 validation gate runs FIRST; a refused payload is
    refused whole (OverlayError carrying the gate's named
    defects).  Convertible geometries become features (Point ->
    marker, LineString -> polyline, Polygon -> polygon with its
    outer ring as points and its inner rings carried as holes);
    multi-geometries fan out; a null geometry is
    skipped with a note.  GeoJSON positions are [lon, lat];
    overlay points are (lat, lon) — the conversion happens here,
    once.
    """
    verdict = validate_geojson(data)
    if not verdict.ok:
        raise OverlayError(
            "GeoJSON refused by the validation gate: "
            + "; ".join(verdict.errors[:5]))
    features: list[OverlayFeature] = []
    notes: list[str] = []

    def label_of(props: Any) -> str:
        if isinstance(props, dict):
            for key in ("name", "label", "title"):
                value = props.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def point(raw: Any) -> tuple[float, float]:
        return (float(raw[1]), float(raw[0]))   # [lon, lat] -> (lat, lon)

    def add_geometry(geom: Any, props: Any) -> None:
        if geom is None:
            notes.append("a feature with no geometry was skipped")
            return
        kind = geom.get("type")
        coords = geom.get("coordinates")
        label = label_of(props)
        if kind == "Point":
            features.append(OverlayFeature(
                KIND_MARKER, (point(coords),), label))
        elif kind == "MultiPoint":
            for raw in coords or ():
                features.append(OverlayFeature(
                    KIND_MARKER, (point(raw),), label))
        elif kind == "LineString":
            features.append(OverlayFeature(
                KIND_POLYLINE,
                tuple(point(p) for p in coords or ()), label))
        elif kind == "MultiLineString":
            for line in coords or ():
                features.append(OverlayFeature(
                    KIND_POLYLINE,
                    tuple(point(p) for p in line), label))
        elif kind == "Polygon":
            rings = coords or ()
            if rings:
                features.append(OverlayFeature(
                    KIND_POLYGON,
                    tuple(point(p) for p in rings[0]), label,
                    holes=tuple(
                        tuple(point(p) for p in ring)
                        for ring in rings[1:])))
        elif kind == "MultiPolygon":
            for poly in coords or ():
                if poly:
                    features.append(OverlayFeature(
                        KIND_POLYGON,
                        tuple(point(p) for p in poly[0]), label,
                        holes=tuple(
                            tuple(point(p) for p in ring)
                            for ring in poly[1:])))
            notes.append(
                "a MultiPolygon became one polygon feature per "
                "polygon")
        elif kind == "GeometryCollection":
            for child in geom.get("geometries") or ():
                add_geometry(child, props)
        else:   # unreachable: the gate refuses unknown types
            notes.append("geometry type " + repr(kind)
                         + " was skipped")

    root_kind = data.get("type") if isinstance(data, dict) else ""
    if root_kind == "FeatureCollection":
        for feat in data.get("features") or ():
            if isinstance(feat, dict):
                add_geometry(feat.get("geometry"),
                             feat.get("properties"))
    elif root_kind == "Feature":
        add_geometry(data.get("geometry"), data.get("properties"))
    else:
        add_geometry(data, None)
    errors = validate_features(features)
    if errors:
        raise OverlayError(
            "GeoJSON converts to an invalid overlay: "
            + "; ".join(errors[:5]))
    return tuple(features), tuple(notes)


class OverlayStore:
    """Durable overlay documents.  Reads never raise."""

    def __init__(self, db: Any = None, home: str = "") -> None:
        self._db = db
        self._table_ready = False
        self._path = (Path(home) / "maps" / "overlays.json"
                      if db is None and home else None)

    # ── storage backends ─────────────────────────────────────

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        self._db.execute(_DDL)
        self._table_ready = True

    def _load_file(self) -> dict[str, dict[str, Any]]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001 — reads fail open
            record_failure("maps.overlays_load_file", exc)
            return {}

    def _save_file(self, data: dict[str, dict[str, Any]]) -> bool:
        if self._path is None:
            return False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False,
                                      indent=1), encoding="utf-8")
            tmp.replace(self._path)
            return True
        except Exception as exc:  # noqa: BLE001
            record_failure("maps.overlays_save_file", exc)
            return False

    def _all(self) -> list[Overlay]:
        out: list[Overlay] = []
        if self._db is not None:
            try:
                self._ensure_table()
                rows = self._db.query(
                    "SELECT payload FROM map_overlays", ())
                payloads = [json.loads(r.get("payload") or "{}")
                            for r in rows]
            except Exception as exc:  # noqa: BLE001
                record_failure("maps.overlays_read", exc)
                return []
        else:
            payloads = list(self._load_file().values())
        for payload in payloads:
            try:
                out.append(Overlay.from_dict(payload))
            except (OverlayError, TypeError, ValueError) as exc:
                record_failure("maps.overlays_row", exc)
                continue
        return sorted(out, key=lambda o: o.name.casefold())

    # ── CRUD ─────────────────────────────────────────────────

    def create(self, name: str,
               features: Iterable[OverlayFeature]) -> Overlay:
        """Validate + persist a new overlay.  Names are unique
        (case-insensitively): a clash is a named refusal."""
        name = (name or "").strip()
        if not name:
            raise OverlayError("an overlay needs a name")
        feats = tuple(features)
        errors = validate_features(feats)
        if errors:
            raise OverlayError("invalid overlay " + repr(name)
                               + ": " + "; ".join(errors[:5]))
        if self.get(name) is not None:
            raise OverlayError(
                "an overlay named " + repr(name) + " already "
                "exists — drop it first, or pick another name")
        overlay = Overlay(
            overlay_id="ov-" + uuid.uuid4().hex[:12], name=name,
            created_at=time.time(), features=feats)
        if self._db is not None:
            try:
                self._ensure_table()
                self._db.execute(
                    "INSERT INTO map_overlays "
                    "(overlay_id, name, payload) VALUES (?, ?, ?)",
                    (overlay.overlay_id, overlay.name,
                     json.dumps(overlay.to_dict(),
                                ensure_ascii=False)))
            except Exception as exc:  # noqa: BLE001
                record_failure("maps.overlays_create", exc)
                raise OverlayError(
                    "overlay " + repr(name) + " could not be "
                    "stored: " + str(exc)) from exc
        else:
            data = self._load_file()
            data[overlay.overlay_id] = overlay.to_dict()
            if not self._save_file(data):
                raise OverlayError(
                    "overlay " + repr(name) + " could not be "
                    "stored (no store is attached)")
        return overlay

    def list(self) -> tuple[Overlay, ...]:
        return tuple(self._all())

    def get(self, name: str) -> Overlay | None:
        """By name: exact match first, then case-insensitive."""
        wanted = (name or "").strip()
        if not wanted:
            return None
        folded = wanted.casefold()
        fallback = None
        for overlay in self._all():
            if overlay.name == wanted:
                return overlay
            if overlay.name.casefold() == folded:
                fallback = overlay
        return fallback

    def delete(self, name: str) -> bool:
        """A real delete of the owner's own overlay data."""
        overlay = self.get(name)
        if overlay is None:
            return False
        if self._db is not None:
            try:
                self._ensure_table()
                self._db.execute(
                    "DELETE FROM map_overlays WHERE overlay_id = ?",
                    (overlay.overlay_id,))
                return True
            except Exception as exc:  # noqa: BLE001
                record_failure("maps.overlays_delete", exc)
                return False
        data = self._load_file()
        data.pop(overlay.overlay_id, None)
        return self._save_file(data)
