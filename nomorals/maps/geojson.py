"""GeoJSON validation: structure, positions, and hostile-input bounds.

M7 will let decoded GeoJSON/KML/GPX become map layers only after it
passes this gate, so the validator is written for untrusted input
from the start: it never raises on malformed data (every defect is a
named error in the result), and it refuses unbounded payloads —
feature and position counts are capped, so a hostile file cannot
make validation itself the denial of service.

A *position* is ``[lon, lat]`` (GeoJSON order), each validated by
:mod:`nomorals.maps.coords`.  Extra ordinates (altitude, …) are
tolerated and ignored, per RFC 7946.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .coords import validate_lat, validate_lon

__all__ = [
    "GEOJSON_TYPES", "MAX_FEATURES", "MAX_POSITIONS",
    "GeoValidation", "validate_geojson",
]

GEOJSON_TYPES = (
    "Point", "MultiPoint", "LineString", "MultiLineString",
    "Polygon", "MultiPolygon", "GeometryCollection",
    "Feature", "FeatureCollection",
)

_GEOMETRY_TYPES = frozenset(GEOJSON_TYPES) - {"Feature", "FeatureCollection"}

#: Bounds for untrusted payloads: generous for real layers, fatal
#: for bombs.  Exceeding either is a validation error, not a crash.
MAX_FEATURES = 10_000
MAX_POSITIONS = 100_000


@dataclass(frozen=True)
class GeoValidation:
    """The verdict: usable or not, and exactly why not."""

    ok: bool
    kind: str
    feature_count: int
    position_count: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "kind": self.kind,
                "feature_count": self.feature_count,
                "position_count": self.position_count,
                "errors": list(self.errors)}


class _Walk:
    """One validation pass: accumulates errors + counts, never raises."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.features = 0
        self.n_positions = 0
        self._cap_noted = False

    def error(self, path: str, message: str) -> None:
        self.errors.append(path + ": " + message)

    def count_position(self) -> bool:
        """Count one position; False when the cap is blown."""
        self.n_positions += 1
        if self.n_positions > MAX_POSITIONS and not self._cap_noted:
            self._cap_noted = True
            self.error("$", "position count exceeds the cap of "
                       + str(MAX_POSITIONS))
        return self.n_positions <= MAX_POSITIONS

    def position(self, value: Any, path: str) -> None:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            self.error(path, "position must be [lon, lat, ...]")
            return
        lon, lat = value[0], value[1]
        if not isinstance(lon, (int, float)) or isinstance(lon, bool):
            self.error(path, "longitude is not a number")
            return
        if not isinstance(lat, (int, float)) or isinstance(lat, bool):
            self.error(path, "latitude is not a number")
            return
        try:
            validate_lon(float(lon))
            validate_lat(float(lat))
        except ValueError as exc:
            self.error(path, str(exc))
            return
        self.count_position()

    def positions(self, value: Any, path: str, depth: int) -> None:
        """Walk a coordinate nest of ``depth`` levels down to positions."""
        if depth == 0:
            self.position(value, path)
            return
        if not isinstance(value, (list, tuple)):
            self.error(path, "coordinates must be an array")
            return
        for index, item in enumerate(value):
            self.positions(item, path + "[" + str(index) + "]", depth - 1)

    def geometry(self, value: Any, path: str) -> None:
        if not isinstance(value, dict):
            self.error(path, "geometry must be an object")
            return
        kind = value.get("type")
        if kind == "Point":
            self.positions(value.get("coordinates"), path + ".coordinates", 0)
        elif kind in ("MultiPoint", "LineString"):
            self.positions(value.get("coordinates"), path + ".coordinates", 1)
        elif kind in ("MultiLineString", "Polygon"):
            self.positions(value.get("coordinates"), path + ".coordinates", 2)
        elif kind == "MultiPolygon":
            self.positions(value.get("coordinates"), path + ".coordinates", 3)
        elif kind == "GeometryCollection":
            children = value.get("geometries")
            if not isinstance(children, list):
                self.error(path + ".geometries", "must be an array")
                return
            for index, child in enumerate(children):
                self.geometry(child, path + ".geometries[" + str(index) + "]")
        else:
            self.error(path, "unknown geometry type: " + repr(kind))

    def feature(self, value: Any, path: str) -> None:
        if not isinstance(value, dict):
            self.error(path, "feature must be an object")
            return
        self.features += 1
        if self.features > MAX_FEATURES:
            self.error(path, "feature count exceeds the cap of "
                       + str(MAX_FEATURES))
            return
        geometry = value.get("geometry")
        if geometry is not None:
            self.geometry(geometry, path + ".geometry")
        properties = value.get("properties")
        if properties is not None and not isinstance(properties, dict):
            self.error(path + ".properties", "must be an object or null")


def validate_geojson(data: Any) -> GeoValidation:
    """Validate a decoded GeoJSON value; the verdict names every defect."""
    walk = _Walk()
    kind = ""
    if not isinstance(data, dict):
        walk.error("$", "GeoJSON root must be an object")
    else:
        kind = str(data.get("type", ""))
        if kind in _GEOMETRY_TYPES:
            walk.geometry(data, "$")
            walk.features = 1 if kind != "GeometryCollection" else 0
        elif kind == "Feature":
            walk.feature(data, "$")
        elif kind == "FeatureCollection":
            children = data.get("features")
            if not isinstance(children, list):
                walk.error("$.features", "must be an array")
            else:
                for index, child in enumerate(children):
                    walk.feature(child, "$.features[" + str(index) + "]")
        else:
            walk.error("$", "unknown GeoJSON type: " + repr(kind))
        bbox = data.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                walk.error("$.bbox", "must be [west, south, east, north, ...]")
            else:
                for label, raw, fn in (
                        ("west", bbox[0], validate_lon),
                        ("south", bbox[1], validate_lat),
                        ("east", bbox[2], validate_lon),
                        ("north", bbox[3], validate_lat)):
                    try:
                        fn(float(raw))
                    except (TypeError, ValueError):
                        walk.error("$.bbox", label + " is out of range: "
                                   + repr(raw))
    return GeoValidation(
        ok=not walk.errors, kind=kind, feature_count=walk.features,
        position_count=walk.n_positions, errors=tuple(walk.errors))
