"""Geofences: circle + polygon, enter/exit evaluated on supplied updates.

A geofence is pure geometry: ``contains(point) -> bool``. It never
knows where anyone is until the caller hands it a point (see
``triggers.evaluate_update``). Circle uses haversine distance;
polygon uses the ray-casting (even-odd) rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from .primitives import LatLon, haversine_m

__all__ = ["Geofence", "circle_fence", "polygon_fence"]


@dataclass(frozen=True)
class Geofence:
    """A named region. ``kind`` is ``circle`` or ``polygon``."""

    id: str
    name: str
    kind: str
    center: LatLon | None = None
    radius_m: float = 0.0
    vertices: tuple[LatLon, ...] = ()
    locale: str = "ng"

    def __post_init__(self) -> None:
        if self.kind == "circle":
            if self.center is None:
                raise ValueError("circle fence needs a center")
            if self.radius_m <= 0:
                raise ValueError("circle fence needs radius_m > 0")
        elif self.kind == "polygon":
            if len(self.vertices) < 3:
                raise ValueError("polygon fence needs >= 3 vertices")
        else:
            raise ValueError(f"unknown fence kind {self.kind!r}")

    def contains(self, point: LatLon) -> bool:
        """True iff ``point`` is inside the fence (boundary counts as in)."""
        if self.kind == "circle":
            assert self.center is not None
            return haversine_m(self.center, point) <= self.radius_m
        return _point_in_polygon(point, self.vertices)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "kind": self.kind,
            "center": self.center.to_dict() if self.center else None,
            "radius_m": self.radius_m,
            "vertices": [v.to_dict() for v in self.vertices],
            "locale": self.locale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Geofence":
        center = data.get("center")
        return cls(
            id=str(data["id"]), name=str(data["name"]), kind=str(data["kind"]),
            center=LatLon.from_dict(center) if center else None,
            radius_m=float(data.get("radius_m", 0.0)),
            vertices=tuple(LatLon.from_dict(v)
                           for v in data.get("vertices", [])),
            locale=str(data.get("locale", "ng")),
        )


def circle_fence(fence_id: str, name: str, center: LatLon,
                 radius_m: float, locale: str = "ng") -> Geofence:
    """Build a circle fence: everything within ``radius_m`` of ``center``."""
    return Geofence(id=fence_id, name=name, kind="circle",
                    center=center, radius_m=radius_m, locale=locale)


def polygon_fence(fence_id: str, name: str, vertices: list[LatLon],
                  locale: str = "ng") -> Geofence:
    """Build a polygon fence from >= 3 vertices (lat/lon, any order)."""
    return Geofence(id=fence_id, name=name, kind="polygon",
                    vertices=tuple(vertices), locale=locale)


def _point_in_polygon(point: LatLon, vertices: tuple[LatLon, ...]) -> bool:
    """Even-odd ray casting. Points exactly on an edge count as inside."""
    x, y = point.lon, point.lat
    inside = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i].lon, vertices[i].lat
        x2, y2 = vertices[(i + 1) % n].lon, vertices[(i + 1) % n].lat
        # On-edge check (with tolerance): boundary counts as inside.
        if _on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_int:
                inside = not inside
    return inside


def _on_segment(px: float, py: float, x1: float, y1: float,
                x2: float, y2: float, eps: float = 1e-9) -> bool:
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
    return dot <= eps
