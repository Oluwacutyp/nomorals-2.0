"""The Natural Earth raster builder: Devon's own tile painter.

One of the M2 pipeline's two builders (the other drives
Planetiler).  It renders overview raster tiles from Natural Earth
10m **vector** data — land, lakes, admin-0 boundary lines — with a
pure-stdlib scanline rasterizer over the shipped PNG codec, so a
world overview pack (Pack B) is Devon-built end to end: Devon's
pipeline fetched the public-domain vectors, Devon's code painted
the tiles, Devon's manifest pins them.  At z0–z7 an overview needs
exactly this fidelity — continents, lakes, borders — and PNG
tiles are what the M1 renderer already composites.

The rasterizer is exact about the slippy scheme: pixel centers are
projected through :mod:`nomorals.maps.coords`, polygon fill is
even-odd per polygon (outer ring minus its holes), and a tile no
polygon edge touches is decided by one point-in-polygon test at
its center — fully-in or fully-out, never guessed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .coords import BBox, lat_to_y_float, lon_to_x_float
from .manifest import PackError, tile_relpath
from .pngcodec import encode_rgb_png

__all__ = [
    "BOUNDARY_RGB", "LAND_RGB", "OCEAN_RGB", "TILE_PX",
    "build_ne_tiles", "load_polygons", "load_polylines",
]

#: Overview palette (the Devon style pass, M3, restyles renders;
#: the pack itself just needs a legible, honest base).
OCEAN_RGB = (168, 200, 216)
LAND_RGB = (240, 236, 224)
BOUNDARY_RGB = (148, 140, 126)

#: Slippy tiles are 256 px; the pack layout assumes nothing else.
TILE_PX = 256

Ring = tuple[tuple[float, float], ...]
Edge = tuple[float, float, float, float]


class _Polygon:
    """One polygon: rings (outer first), edges, extent — precomputed."""

    __slots__ = ("bbox", "edges", "rings")

    def __init__(self, rings: tuple[Ring, ...]) -> None:
        self.rings = rings
        edges: list[Edge] = []
        west = south = float("inf")
        east = north = float("-inf")
        for ring in rings:
            for idx, (lon, lat) in enumerate(ring):
                west = min(west, lon)
                east = max(east, lon)
                south = min(south, lat)
                north = max(north, lat)
                lon2, lat2 = ring[(idx + 1) % len(ring)]
                if (lon, lat) != (lon2, lat2):
                    edges.append((lon, lat, lon2, lat2))
        self.edges = tuple(edges)
        self.bbox = (west, south, east, north)

    def contains(self, lon: float, lat: float) -> bool:
        """Even-odd point test across all rings (holes subtract)."""
        inside = False
        for ring in self.rings:
            count = len(ring)
            for idx in range(count):
                lon1, lat1 = ring[idx]
                lon2, lat2 = ring[(idx + 1) % count]
                if (lat1 <= lat < lat2) or (lat2 <= lat < lat1):
                    cross = lon1 + (lat - lat1) * (lon2 - lon1) \
                        / (lat2 - lat1)
                    if cross > lon:
                        inside = not inside
        return inside


def _rings_of(geometry: dict) -> list[tuple[Ring, ...]]:
    """Every polygon of a GeoJSON Polygon/MultiPolygon geometry."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    polygons: list[tuple[Ring, ...]] = []
    if kind == "Polygon":
        polygons.append(tuple(
            tuple((float(lon), float(lat)) for lon, lat in ring)
            for ring in coords))
    elif kind == "MultiPolygon":
        for polygon in coords:
            polygons.append(tuple(
                tuple((float(lon), float(lat)) for lon, lat in ring)
                for ring in polygon))
    return polygons


def _lines_of(geometry: dict) -> list[Ring]:
    """Every polyline of a GeoJSON LineString/MultiLineString."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "LineString":
        return [tuple((float(lon), float(lat)) for lon, lat
                      in coords)]
    if kind == "MultiLineString":
        return [tuple((float(lon), float(lat)) for lon, lat in line)
                for line in coords]
    return []


def _load_geojson(path: Path) -> list[dict]:
    """The geometries of a GeoJSON file (features or bare geometry)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackError(
            "geojson unreadable: " + str(path) + ": " + str(exc)
        ) from exc
    if data.get("type") == "FeatureCollection":
        return [f.get("geometry") or {} for f in data.get("features", [])]
    if data.get("type") == "Feature":
        return [data.get("geometry") or {}]
    return [data]


def load_polygons(path: Path | str) -> list[_Polygon]:
    """All polygons of a GeoJSON dataset, edges precomputed."""
    out: list[_Polygon] = []
    for geometry in _load_geojson(Path(path)):
        for rings in _rings_of(geometry):
            if rings and len(rings[0]) >= 3:
                out.append(_Polygon(rings))
    return out


def load_polylines(path: Path | str) -> list[Ring]:
    """All polylines of a GeoJSON dataset."""
    out: list[Ring] = []
    for geometry in _load_geojson(Path(path)):
        out.extend(line for line in _lines_of(geometry)
                   if len(line) >= 2)
    return out


def _tile_lat(y_frac: float) -> float:
    """Latitude of a fractional tile row (inverse Mercator)."""
    return math.degrees(math.atan(math.sinh(
        math.pi * (1.0 - 2.0 * y_frac))))


def _fill_span(row: bytearray, px_from: float, px_to: float,
               rgb: tuple[int, int, int]) -> None:
    """Fill one pixel row between two fractional pixel columns."""
    start = max(0, int(math.floor(px_from)))
    end = min(TILE_PX - 1, int(math.floor(px_to)))
    for px in range(start, end + 1):
        off = px * 3
        row[off:off + 3] = bytes(rgb)


def _paint_polygon(buf: bytearray, poly: _Polygon, z: int, x: int,
                   y: int, rgb: tuple[int, int, int]) -> None:
    """Scanline-fill one polygon into a tile buffer."""
    west = x * 360.0 / (1 << z) - 180.0
    east = (x + 1) * 360.0 / (1 << z) - 180.0
    north = _tile_lat(y / (1 << z))
    south = _tile_lat((y + 1) / (1 << z))
    pw, ps, pe, pn = poly.bbox
    if pe < west or pw > east or pn < south or ps > north:
        return
    edges = [e for e in poly.edges
             if max(e[1], e[3]) >= south and min(e[1], e[3]) <= north
             and max(e[0], e[2]) >= west and min(e[0], e[2]) <= east]
    if not edges:
        # No edge touches the tile: wholly in or wholly out.
        mid_lon = (west + east) / 2.0
        mid_lat = _tile_lat((y + 0.5) / (1 << z))
        if poly.contains(mid_lon, mid_lat):
            row = bytearray(bytes(rgb) * TILE_PX)
            for py in range(TILE_PX):
                buf[py * TILE_PX * 3:(py + 1) * TILE_PX * 3] = row
        return
    for py in range(TILE_PX):
        lat_c = _tile_lat((y + (py + 0.5) / TILE_PX) / (1 << z))
        crossings: list[float] = []
        for lon1, lat1, lon2, lat2 in edges:
            if (lat1 <= lat_c < lat2) or (lat2 <= lat_c < lat1):
                crossings.append(
                    lon1 + (lat_c - lat1) * (lon2 - lon1) / (lat2 - lat1))
        if not crossings:
            continue
        crossings.sort()
        row = bytearray(buf[py * TILE_PX * 3:(py + 1) * TILE_PX * 3])
        for idx in range(0, len(crossings) - 1, 2):
            px_a = (lon_to_x_float(crossings[idx], z) - x) * TILE_PX
            px_b = (lon_to_x_float(crossings[idx + 1], z) - x) * TILE_PX
            _fill_span(row, px_a, px_b, rgb)
        buf[py * TILE_PX * 3:(py + 1) * TILE_PX * 3] = row


def _paint_line(buf: bytearray, line: Ring, z: int, x: int, y: int,
                rgb: tuple[int, int, int]) -> None:
    """Rasterize one polyline into a tile buffer (pixel sampling)."""
    for idx in range(len(line) - 1):
        lon1, lat1 = line[idx]
        lon2, lat2 = line[idx + 1]
        fx1 = (lon_to_x_float(lon1, z) - x) * TILE_PX
        fy1 = (lat_to_y_float(lat1, z) - y) * TILE_PX
        fx2 = (lon_to_x_float(lon2, z) - x) * TILE_PX
        fy2 = (lat_to_y_float(lat2, z) - y) * TILE_PX
        if max(fx1, fx2) < 0 or min(fx1, fx2) > TILE_PX \
                or max(fy1, fy2) < 0 or min(fy1, fy2) > TILE_PX:
            continue
        steps = int(max(abs(fx2 - fx1), abs(fy2 - fy1))) + 1
        for step in range(steps + 1):
            px = int(fx1 + (fx2 - fx1) * step / steps)
            py = int(fy1 + (fy2 - fy1) * step / steps)
            if 0 <= px < TILE_PX and 0 <= py < TILE_PX:
                off = (py * TILE_PX + px) * 3
                buf[off:off + 3] = bytes(rgb)


def _render_tile(z: int, x: int, y: int, land: list[_Polygon],
                 lakes: list[_Polygon],
                 lines: list[tuple[Ring, tuple[float, ...]]]) -> bytes:
    """One tile: ocean, then land, lakes cut back, boundaries on top."""
    buf = bytearray(bytes(OCEAN_RGB) * (TILE_PX * TILE_PX))
    west = x * 360.0 / (1 << z) - 180.0
    east = (x + 1) * 360.0 / (1 << z) - 180.0
    north = _tile_lat(y / (1 << z))
    south = _tile_lat((y + 1) / (1 << z))

    def _hits(bbox: tuple[float, ...]) -> bool:
        return not (bbox[2] < west or bbox[0] > east
                    or bbox[3] < south or bbox[1] > north)

    for poly in land:
        if _hits(poly.bbox):
            _paint_polygon(buf, poly, z, x, y, LAND_RGB)
    for poly in lakes:
        if _hits(poly.bbox):
            _paint_polygon(buf, poly, z, x, y, OCEAN_RGB)
    for line, line_bbox in lines:
        if _hits(line_bbox):
            _paint_line(buf, line, z, x, y, BOUNDARY_RGB)
    return encode_rgb_png(TILE_PX, TILE_PX, bytes(buf))


def build_ne_tiles(*, land_path: Path | str,
                   lakes_path: Path | str = "",
                   lines_path: Path | str = "",
                   out_dir: Path | str, bbox: BBox,
                   zoom_min: int, zoom_max: int) -> int:
    """Paint every tile of ``bbox`` at z[``zoom_min``..``zoom_max``].

    Returns the number of tiles written into ``out_dir`` in the
    pack layout.  A missing land dataset is a named PackError —
    an overview with no land is not a basemap.
    """
    land = load_polygons(Path(land_path))
    if not land:
        raise PackError(
            "no land polygons in dataset: " + str(land_path))
    lakes = load_polygons(Path(lakes_path)) if lakes_path else []
    raw_lines = load_polylines(Path(lines_path)) if lines_path else []
    lines = [(line, (min(p[0] for p in line), min(p[1] for p in line),
                     max(p[0] for p in line), max(p[1] for p in line)))
             for line in raw_lines]
    root = Path(out_dir)
    count = 0
    for z in range(int(zoom_min), int(zoom_max) + 1):
        for tile in bbox.tiles(z):
            png = _render_tile(z, tile.x, tile.y, land, lakes, lines)
            dest = root / tile_relpath(z, tile.x, tile.y, "png")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(png)
            count += 1
    return count
