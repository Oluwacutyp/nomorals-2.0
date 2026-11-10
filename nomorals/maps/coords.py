"""Devon Atlas coordinate core: tiles, bounding boxes, validation.

The typed substrate every Atlas plane shares — the renderer (M1/M3),
the local geocoder (M4) and the routing plane (M5) all speak these
types, so a coordinate is validated exactly once, here, at the
boundary.  Pure stdlib; no network, no data files.

Latitude/longitude validation reuses :class:`nomorals.location.
primitives.LatLon` — one validation truth for the whole repo.  Tile
math is the standard Web-Mercator slippy scheme: at zoom ``z`` the
world is a ``2**z`` by ``2**z`` tile grid, x growing east from the
antimeridian, y growing south from the Mercator north edge
(±85.05112878°, beyond which the projection does not reach).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..location.primitives import LatLon

__all__ = [
    "MAX_ZOOM", "MERCATOR_MAX_LAT", "BBox", "TileCoord",
    "clamp_mercator_lat", "lat_to_tile_y", "lat_to_y_float",
    "lon_to_tile_x", "lon_to_x_float",
    "tile_x_to_lon", "tile_y_to_lat", "validate_lat", "validate_lon",
]

#: Deepest zoom the Atlas schemas admit (slippy convention).
MAX_ZOOM = 22

#: The Web-Mercator latitude edge: the projection's north/south limit.
MERCATOR_MAX_LAT = 85.05112878


def validate_lat(lat: float) -> float:
    """Return ``lat`` as a float, refusing anything outside ±90."""
    value = float(lat)
    if not (-90.0 <= value <= 90.0):
        raise ValueError("latitude out of range: " + repr(value))
    return value


def validate_lon(lon: float) -> float:
    """Return ``lon`` as a float, refusing anything outside ±180."""
    value = float(lon)
    if not (-180.0 <= value <= 180.0):
        raise ValueError("longitude out of range: " + repr(value))
    return value


def clamp_mercator_lat(lat: float) -> float:
    """Clamp a valid latitude to the Mercator band the tiles cover."""
    return max(-MERCATOR_MAX_LAT, min(MERCATOR_MAX_LAT, validate_lat(lat)))


def lon_to_x_float(lon: float, zoom: int) -> float:
    """Fractional tile column of ``lon`` at ``zoom`` (sub-tile exact)."""
    return (validate_lon(lon) + 180.0) / 360.0 * (1 << zoom)


def lat_to_y_float(lat: float, zoom: int) -> float:
    """Fractional tile row of ``lat`` at ``zoom`` (sub-tile exact)."""
    phi = math.radians(clamp_mercator_lat(lat))
    return (1.0 - math.log(math.tan(phi) + 1.0 / math.cos(phi)) / math.pi) \
        / 2.0 * (1 << zoom)


def lon_to_tile_x(lon: float, zoom: int) -> int:
    """Tile column containing ``lon`` at ``zoom`` (clamped to the grid)."""
    n = 1 << zoom
    return max(0, min(n - 1, int(math.floor(lon_to_x_float(lon, zoom)))))


def lat_to_tile_y(lat: float, zoom: int) -> int:
    """Tile row containing ``lat`` at ``zoom`` (clamped to the grid)."""
    n = 1 << zoom
    return max(0, min(n - 1, int(math.floor(lat_to_y_float(lat, zoom)))))


def tile_x_to_lon(x: int, zoom: int) -> float:
    """Longitude of tile column ``x``'s west edge at ``zoom``."""
    return x / (1 << zoom) * 360.0 - 180.0


def tile_y_to_lat(y: int, zoom: int) -> float:
    """Latitude of tile row ``y``'s north edge at ``zoom``."""
    n = 1 << zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))


@dataclass(frozen=True)
class TileCoord:
    """One tile in the slippy grid: ``z`` zoom, ``x`` east, ``y`` south."""

    z: int
    x: int
    y: int

    def __post_init__(self) -> None:
        if not (0 <= int(self.z) <= MAX_ZOOM):
            raise ValueError("zoom out of range: " + repr(self.z))
        n = 1 << int(self.z)
        if not (0 <= int(self.x) < n):
            raise ValueError(
                "tile x out of range for zoom " + repr(self.z)
                + ": " + repr(self.x))
        if not (0 <= int(self.y) < n):
            raise ValueError(
                "tile y out of range for zoom " + repr(self.z)
                + ": " + repr(self.y))

    @classmethod
    def from_latlon(cls, lat: float, lon: float, zoom: int) -> TileCoord:
        """The tile containing a point (validated via ``LatLon``)."""
        point = LatLon(lat=float(lat), lon=float(lon))
        return cls(z=int(zoom), x=lon_to_tile_x(point.lon, int(zoom)),
                   y=lat_to_tile_y(point.lat, int(zoom)))

    def bbox(self) -> BBox:
        """The geographic box this tile covers."""
        return BBox(west=tile_x_to_lon(self.x, self.z),
                    south=tile_y_to_lat(self.y + 1, self.z),
                    east=tile_x_to_lon(self.x + 1, self.z),
                    north=tile_y_to_lat(self.y, self.z))

    def parent(self) -> TileCoord | None:
        """The zoom-``z-1`` tile containing this one (None at zoom 0)."""
        if self.z == 0:
            return None
        return TileCoord(z=self.z - 1, x=self.x // 2, y=self.y // 2)

    def children(self) -> tuple[TileCoord, ...]:
        """The four zoom-``z+1`` tiles inside this one (empty at MAX)."""
        if self.z >= MAX_ZOOM:
            return ()
        return tuple(
            TileCoord(z=self.z + 1, x=2 * self.x + dx, y=2 * self.y + dy)
            for dy in (0, 1) for dx in (0, 1))

    def to_dict(self) -> dict[str, int]:
        return {"z": int(self.z), "x": int(self.x), "y": int(self.y)}

    @classmethod
    def from_dict(cls, data: dict) -> TileCoord:
        return cls(z=int(data["z"]), x=int(data["x"]), y=int(data["y"]))


@dataclass(frozen=True)
class BBox:
    """A geographic rectangle.  ``west > east`` crosses the antimeridian.

    Latitudes must satisfy ``south <= north``; longitudes are plain
    ±180 values — the antimeridian is expressed by ordering, never by
    out-of-range numbers.
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        validate_lon(self.west)
        validate_lon(self.east)
        validate_lat(self.south)
        validate_lat(self.north)
        if float(self.south) > float(self.north):
            raise ValueError(
                "bbox south above north: " + repr(self.south)
                + " > " + repr(self.north))

    @property
    def crosses_antimeridian(self) -> bool:
        return float(self.west) > float(self.east)

    @property
    def width_deg(self) -> float:
        """Longitudinal span in degrees, wrap-aware (0..360)."""
        span = float(self.east) - float(self.west)
        return span + 360.0 if span < 0.0 else span

    @property
    def height_deg(self) -> float:
        return float(self.north) - float(self.south)

    def contains(self, lat: float, lon: float) -> bool:
        """True when the (validated) point falls inside this box."""
        point = LatLon(lat=float(lat), lon=float(lon))
        if not (float(self.south) <= point.lat <= float(self.north)):
            return False
        if self.crosses_antimeridian:
            return point.lon >= float(self.west) or point.lon <= float(self.east)
        return float(self.west) <= point.lon <= float(self.east)

    def tiles(self, zoom: int) -> tuple[TileCoord, ...]:
        """Every tile covering this box at ``zoom``, north row first.

        For an antimeridian box the columns wrap: they start at the
        west edge's column and run through the grid edge back to the
        east edge's column, in that order.
        """
        z = int(zoom)
        if not (0 <= z <= MAX_ZOOM):
            raise ValueError("zoom out of range: " + repr(zoom))
        n = 1 << z
        x_first = lon_to_tile_x(self.west, z)
        x_last = lon_to_tile_x(self.east, z)
        if self.crosses_antimeridian:
            xs = list(range(x_first, n)) + list(range(0, x_last + 1))
        else:
            xs = list(range(x_first, x_last + 1))
        y_first = lat_to_tile_y(self.north, z)
        y_last = lat_to_tile_y(self.south, z)
        return tuple(
            TileCoord(z=z, x=x, y=y)
            for y in range(y_first, y_last + 1) for x in xs)

    def as_list(self) -> list[float]:
        """``[west, south, east, north]`` — the manifest/GeoJSON order."""
        return [float(self.west), float(self.south),
                float(self.east), float(self.north)]

    def to_dict(self) -> dict[str, float]:
        return {"west": float(self.west), "south": float(self.south),
                "east": float(self.east), "north": float(self.north)}

    @classmethod
    def from_list(cls, data: list | tuple) -> BBox:
        west, south, east, north = data
        return cls(west=float(west), south=float(south),
                   east=float(east), north=float(north))

    @classmethod
    def from_dict(cls, data: dict) -> BBox:
        return cls(west=float(data["west"]), south=float(data["south"]),
                   east=float(data["east"]), north=float(data["north"]))
