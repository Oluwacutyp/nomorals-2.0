"""The vector renderer: MVT packs + the Devon Style → a map image.

M3's half of the renderer boundary (doc 21 S5).  Where M1's
:class:`~nomorals.maps.render.FixtureRenderer` composites a PNG
pack's tiles verbatim, this renderer *draws*: it decodes a vector
pack's tiles (:mod:`nomorals.maps.mvt`), projects their geometry
from tile space into the output through the same Web-Mercator
math the fixture renderer crops by (:mod:`nomorals.maps.coords`),
and rasterizes it in the order a style document dictates
(:mod:`nomorals.maps.style`) — polygon fills by even-odd scanline,
line strokes as round-joined disc stamps, points as markers —
into the stdlib PNG codec's buffer.  The style is data: swap the
document and the same code draws a different map (pinned).

Labels ride an injectable :class:`TextBackend` — Devon's own
glyph work is a typography-organ matter, so the shipped backend
draws through Pillow with a face the typography service selects
from the host's catalogue.  A profile without Pillow, or without
a resolvable font, does not crash and does not silently ship a
label-less map: the geometry renders and the receipt *says* the
labels were skipped, with the count.  Honesty rules otherwise
match M1 exactly: missing or corrupt tiles render as the style's
no-data color and are named in the receipt; a view the pack holds
nothing for, a zoom outside the pack, an oversize view — all are
typed unavailable receipts, never a blank image.

M5 adds one input to the boundary: an optional
:class:`MapOverlay` of finished (lat, lon) geometry — a route
polyline plus its endpoint markers — painted after the style
layers and before the labels, so the caller's geometry reads as
part of the map rather than a sticker on it.  The overlay is
*input data*, not style: a route wears the house gold whatever
the style document says.  A render without an overlay is
byte-identical to the render before the seam existed (pinned).
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .coords import BBox, lat_to_y_float, lon_to_x_float
from .manifest import PackManifest, load_manifest, tile_relpath
from .mvt import MvtError, MvtFeature, MvtTile, decode_mvt
from .receipts import OP_RENDER, OUTCOME_OK, OUTCOME_UNAVAILABLE, MapReceipt, MapReceiptStore
from .render import MAX_OUTPUT_PX, MAX_RENDER_TILES, PngBackend, RenderResult, StdlibPngBackend
from .style import (
    DEFAULT_STYLE,
    KIND_FILL,
    KIND_LABEL,
    KIND_LINE,
    KIND_POINT,
    ROAD_MOTORWAY,
    MapStyle,
    StyleError,
    StyleLayer,
    parse_color,
    style_hash,
)

__all__ = [
    "LabelSpec", "MapOverlay", "PilTextBackend", "TextBackend",
    "VectorRenderer",
]


# ── the text seam ─────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelSpec:
    """One label to draw: anchor (output px), text, size, colors."""

    x: float
    y: float
    text: str
    size_px: int
    color: tuple[int, int, int]
    halo: tuple[int, int, int] | None = None
    halo_width: int = 2


# ── the overlay seam (M5) ─────────────────────────────────────────


@dataclass(frozen=True)
class MapOverlay:
    """Finished geometry drawn over the styled map (M5).

    Input data, not style: a caller hands the renderer (lat, lon)
    polylines and markers; the renderer projects and stamps them
    after the style layers and before the labels.  Routes wear
    the house gold (the style's motorway gold, :data:`ROAD_MOTORWAY`)
    whatever the style document says; endpoint markers are gold
    discs with an ink centre so both ends read at a glance.
    """

    polylines: tuple[tuple[tuple[float, float], ...], ...] = ()
    markers: tuple[tuple[float, float], ...] = ()
    line_color: str = ROAD_MOTORWAY
    line_width: float = 5.0
    marker_color: str = ROAD_MOTORWAY

    def __post_init__(self) -> None:
        parse_color(self.line_color)   # a bad color is a caller
        parse_color(self.marker_color)  # error, raised here
        if self.line_width <= 0:
            raise StyleError("overlay line_width must be positive")


@runtime_checkable
class TextBackend(Protocol):
    """Draws text onto an RGB buffer — atomically, or not at all.

    ``draw_labels`` returns the new buffer, or None when this
    profile cannot draw; it never returns a half-labeled buffer.
    """

    def available(self) -> bool:
        ...

    def draw_labels(self, width: int, height: int, rgb: bytes,
                    labels: tuple[LabelSpec, ...]) -> bytes | None:
        ...


class PilTextBackend:
    """The shipped text backend: Pillow + the typography service.

    The font is a real face from the host's font catalogue,
    selected per label text by ``media/typography`` (the organ
    that owns glyphs), loaded lazily — importing this module
    costs nothing and a Pillow-less profile simply reports
    unavailable.
    """

    def __init__(self) -> None:
        self._verdict: bool | None = None
        self._fonts: dict[tuple[int, str], object] = {}

    def _font(self, size_px: int, text: str) -> object | None:
        key = (int(size_px), text)
        if key in self._fonts:
            return self._fonts[key]
        font = None
        try:
            from ..media.typography.service import get_service

            font = get_service().pil_font(
                "text", max(6, int(size_px)), text=text)
        except Exception:  # noqa: BLE001 — no font is an incapacity
            font = None
        if len(self._fonts) < 128:
            self._fonts[key] = font
        return font

    def available(self) -> bool:
        if self._verdict is None:
            try:
                import PIL.Image  # noqa: F401
                import PIL.ImageDraw  # noqa: F401

                self._verdict = self._font(12, "Ag") is not None
            except Exception:  # noqa: BLE001
                self._verdict = False
        return bool(self._verdict)

    def draw_labels(self, width: int, height: int, rgb: bytes,
                    labels: tuple[LabelSpec, ...]) -> bytes | None:
        try:
            from PIL import Image, ImageDraw

            image = Image.frombytes("RGB", (int(width), int(height)),
                                    bytes(rgb))
            draw = ImageDraw.Draw(image)
            for spec in labels:
                font = self._font(spec.size_px, spec.text)
                if font is None:
                    return None
                draw.text(
                    (float(spec.x), float(spec.y)), spec.text,
                    font=font, fill=tuple(spec.color), anchor="mm",
                    stroke_width=int(spec.halo_width) if spec.halo
                    else 0,
                    stroke_fill=tuple(spec.halo) if spec.halo
                    else tuple(spec.color))
            return image.tobytes()
        except Exception:  # noqa: BLE001 — incapacity, named upstream
            return None


# ── the renderer ──────────────────────────────────────────────────


class VectorRenderer:
    """Draws a vector (``mvt``) pack into a static map image."""

    def __init__(self, pack_dir: Path | str,
                 manifest: PackManifest | None = None, *,
                 store: MapReceiptStore | None = None,
                 backend: PngBackend | None = None,
                 style: MapStyle | None = None,
                 text_backend: TextBackend | None = None) -> None:
        self._dir = Path(pack_dir)
        self._manifest = manifest if manifest is not None \
            else load_manifest(self._dir)
        self._store = store
        self._backend: PngBackend = backend if backend is not None \
            else StdlibPngBackend()
        self._style = style if style is not None else DEFAULT_STYLE
        self._text_backend = text_backend

    @property
    def manifest(self) -> PackManifest:
        return self._manifest

    @property
    def style(self) -> MapStyle:
        return self._style

    def _finish(self, outcome: str, detail: str,
                image: bytes | None = None, width: int = 0,
                height: int = 0,
                missing: tuple[str, ...] = ()) -> RenderResult:
        receipt = MapReceipt.mint(
            OP_RENDER, outcome, pack_refs=(self._manifest.pack_id,),
            pack_digests=(self._manifest.digest,), detail=detail)
        if self._store is not None:
            self._store.append(receipt)
        return RenderResult(image=image, receipt=receipt, width=width,
                            height=height, missing_tiles=missing)

    # ── tile loading ──────────────────────────────────────────────

    def _load_tiles(self, tiles: tuple) -> tuple[dict, list, list]:
        """Decode the view's tiles.

        Returns ``(decoded, missing, corrupt)``: decoded maps
        ``(x, y)`` → :class:`MvtTile`; missing names members the
        pack does not hold; corrupt names members that are not
        readable MVT.  A gzip-wrapped member (the common tile
        transport encoding) is unwrapped first — transport, not
        data.
        """
        decoded: dict[tuple[int, int], MvtTile] = {}
        missing: list[str] = []
        corrupt: list[str] = []
        for tile in tiles:
            rel = tile_relpath(tile.z, tile.x, tile.y,
                               self._manifest.tile_format)
            path = self._dir / rel
            if not path.is_file():
                missing.append(rel)
                continue
            raw = path.read_bytes()
            if raw[:2] == b"\x1f\x8b":
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    corrupt.append(rel)
                    continue
            try:
                decoded[(tile.x, tile.y)] = decode_mvt(raw)
            except MvtError:
                corrupt.append(rel)
        return decoded, missing, corrupt

    # ── the render ────────────────────────────────────────────────

    def render(self, bbox: BBox, zoom: int,
               size: tuple[int, int] = (512, 512), *,
               overlay: MapOverlay | None = None) -> RenderResult:
        manifest = self._manifest
        style = self._style
        out_w, out_h = int(size[0]), int(size[1])
        head = "style " + style.name + " (" \
            + style_hash(style)[:12] + ")"
        if not (1 <= out_w <= MAX_OUTPUT_PX
                and 1 <= out_h <= MAX_OUTPUT_PX):
            return self._finish(
                OUTCOME_UNAVAILABLE,
                head + " · output size out of bounds: " + repr(size))
        if not (manifest.zoom_min <= int(zoom) <= manifest.zoom_max):
            return self._finish(
                OUTCOME_UNAVAILABLE,
                head + " · zoom " + str(int(zoom))
                + " outside pack range " + str(manifest.zoom_min)
                + ".." + str(manifest.zoom_max))
        if manifest.tile_format != "mvt":
            return self._finish(
                OUTCOME_UNAVAILABLE,
                head + " · vector renderer draws mvt packs, this "
                "pack is " + repr(manifest.tile_format))
        tiles = bbox.tiles(int(zoom))
        if len(tiles) > MAX_RENDER_TILES:
            return self._finish(
                OUTCOME_UNAVAILABLE,
                head + " · view spans " + str(len(tiles))
                + " tiles, over the " + str(MAX_RENDER_TILES)
                + "-tile render bound")
        try:
            return self._draw(bbox, int(zoom), tiles, out_w, out_h,
                              overlay)
        except Exception as exc:  # noqa: BLE001 — a drawing defect is
            # an incapacity answer here, never a crash through the seam
            return self._finish(
                OUTCOME_UNAVAILABLE,
                head + " · renderer failed: " + type(exc).__name__
                + ": " + str(exc))

    def _draw(self, bbox: BBox, zoom: int, tiles: tuple,
              out_w: int, out_h: int,
              overlay: MapOverlay | None = None) -> RenderResult:
        style = self._style
        head = "style " + style.name + " (" \
            + style_hash(style)[:12] + ")"
        decoded, missing, corrupt = self._load_tiles(tiles)
        absent = missing + corrupt
        if not decoded:
            return self._finish(
                OUTCOME_UNAVAILABLE,
                head + " · pack holds no tiles for this view ("
                + str(len(absent)) + " missing)",
                missing=tuple(absent))
        # The projection: fractional tile space → output pixels,
        # linear in Mercator space — the fixture renderer's crop
        # math, applied to geometry instead of tile images.
        n = 1 << zoom
        west_f = lon_to_x_float(bbox.west, zoom)
        east_f = lon_to_x_float(bbox.east, zoom)
        if bbox.crosses_antimeridian:
            east_f += n
        north_f = lat_to_y_float(bbox.north, zoom)
        south_f = lat_to_y_float(bbox.south, zoom)
        span_x = max(1e-9, east_f - west_f)
        span_y = max(1e-9, south_f - north_f)

        def project(x_units: float, y_units: float) -> tuple[float, float]:
            return ((x_units - west_f) / span_x * out_w,
                    (y_units - north_f) / span_y * out_h)

        # Antimeridian views wrap tile columns: a wrapped tile's
        # unit-x is its column + the grid width, so it projects to
        # the right of the west edge instead of back at column 0
        # (the fixture renderer's ordered-columns math, in units).
        def unit_x(tile) -> float:
            if bbox.crosses_antimeridian and tile.x < tiles[0].x:
                return float(tile.x + n)
            return float(tile.x)

        buf = bytearray(bytes(parse_color(style.background))
                        * (out_w * out_h))
        no_data = parse_color(style.no_data)
        for tile in tiles:
            if (tile.x, tile.y) in decoded:
                continue
            x0, y0 = project(unit_x(tile), tile.y)
            x1, y1 = project(unit_x(tile) + 1, tile.y + 1)
            self._fill_rect(buf, out_w, out_h, x0, y0, x1, y1, no_data)
        labels: list[LabelSpec] = []
        for rule in style.layers:
            if not rule.applies_at(zoom):
                continue
            for tile in tiles:
                mvt_tile = decoded.get((tile.x, tile.y))
                if mvt_tile is None:
                    continue
                for mvt_layer in mvt_tile.layers:
                    if not rule.matches(mvt_layer.name):
                        continue
                    for feature in mvt_layer.features:
                        self._paint(buf, out_w, out_h, rule, feature,
                                    mvt_layer.extent, unit_x(tile),
                                    tile.y, project, zoom, labels)
        if overlay is not None:
            self._paint_overlay(buf, out_w, out_h, overlay, bbox,
                                zoom, project)
        detail = head + " · tiles " + str(len(decoded)) + "/" \
            + str(len(tiles)) + " present"
        if overlay is not None:
            detail += "; overlay: " + str(len(overlay.polylines)) \
                + " line(s), " + str(len(overlay.markers)) \
                + " marker(s)"
        if missing:
            detail += "; missing: " + ", ".join(missing[:5])
            if len(missing) > 5:
                detail += " (+" + str(len(missing) - 5) + " more)"
        if corrupt:
            detail += "; corrupt: " + ", ".join(corrupt[:5])
        detail += self._apply_labels(buf, out_w, out_h, labels)
        image = self._backend.encode_png(out_w, out_h, bytes(buf))
        return self._finish(OUTCOME_OK, detail, image=image,
                            width=out_w, height=out_h,
                            missing=tuple(absent))

    def _paint_overlay(self, buf: bytearray, out_w: int,
                       out_h: int, overlay: MapOverlay, bbox: BBox,
                       zoom: int, project) -> None:
        """Stamp the overlay's polylines + markers onto the map.

        Points arrive as (lat, lon); projection runs through the
        same unit math as the tiles, with the antimeridian wrap
        applied the same way (a wrapped point's unit-x shifts by
        the grid width).  Markers are a gold disc ringed around an
        ink centre — the style's background — so an endpoint
        reads on any road color beneath it.
        """
        n = float(1 << zoom)
        west_f = lon_to_x_float(bbox.west, zoom)

        def to_px(point: tuple[float, float]) -> tuple[float, float]:
            lat, lon = point
            ux = lon_to_x_float(lon, zoom)
            if bbox.crosses_antimeridian and ux < west_f:
                ux += n
            return project(ux, lat_to_y_float(lat, zoom))

        line_rgb = parse_color(overlay.line_color)
        for line in overlay.polylines:
            pts = [to_px(point) for point in line]
            if len(pts) >= 2:
                self._stroke_path(buf, out_w, out_h, pts, line_rgb,
                                  float(overlay.line_width),
                                  close=False)
        marker_rgb = parse_color(overlay.marker_color)
        ink = parse_color(self._style.background)
        for marker in overlay.markers:
            cx, cy = to_px(marker)
            self._stamp_disc(buf, out_w, out_h, cx, cy, 7.0,
                             marker_rgb)
            self._stamp_disc(buf, out_w, out_h, cx, cy, 3.0, ink)

    def _apply_labels(self, buf: bytearray, out_w: int, out_h: int,
                      labels: list[LabelSpec]) -> str:
        """Draw the collected labels, or say plainly why not."""
        if not labels:
            return "; labels: none in view"
        backend = self._text_backend
        if backend is None:
            backend = PilTextBackend()
            self._text_backend = backend
        if not backend.available():
            return "; labels skipped: no text backend on this " \
                "profile (" + str(len(labels)) + " not drawn)"
        drawn = backend.draw_labels(out_w, out_h, bytes(buf),
                                    tuple(labels))
        if drawn is None or len(drawn) != out_w * out_h * 3:
            return "; labels skipped: the text backend could not " \
                "draw (" + str(len(labels)) + " not drawn)"
        buf[:] = drawn
        return "; labels: " + str(len(labels)) + " drawn"

    # ── painting ──────────────────────────────────────────────────

    def _paint(self, buf: bytearray, out_w: int, out_h: int,
               rule: StyleLayer, feature: MvtFeature, extent: int,
               tile_ux: float, tile_y: int, project, zoom: int,
               labels: list[LabelSpec]) -> None:
        def to_px(point: tuple[int, int]) -> tuple[float, float]:
            return project(tile_ux + point[0] / extent,
                           tile_y + point[1] / extent)

        if rule.kind == KIND_FILL:
            if feature.geom_type != "polygon":
                return
            color = rule.fill_for(feature.tags)
            if color is not None:
                rings = [[to_px(p) for p in path.points]
                         for path in feature.paths if path.points]
                self._fill_rings(buf, out_w, out_h, rings, color)
            stroke = parse_color(rule.stroke) if rule.stroke else None
            if stroke is not None:
                for path in feature.paths:
                    pts = [to_px(p) for p in path.points]
                    if len(pts) >= 2:
                        self._stroke_path(buf, out_w, out_h, pts,
                                          stroke, max(1.0,
                                                     float(rule.width)),
                                          close=True)
        elif rule.kind == KIND_LINE:
            color = rule.stroke_for(feature.tags)
            if color is None:
                return
            width = rule.width_for(feature.tags, zoom)
            if feature.geom_type == "linestring":
                for path in feature.paths:
                    pts = [to_px(p) for p in path.points]
                    if len(pts) >= 2:
                        self._stroke_path(buf, out_w, out_h, pts,
                                          color, width, close=False)
            elif feature.geom_type == "polygon":
                for path in feature.paths:
                    pts = [to_px(p) for p in path.points]
                    if len(pts) >= 2:
                        self._stroke_path(buf, out_w, out_h, pts,
                                          color, width, close=True)
        elif rule.kind == KIND_POINT:
            if feature.geom_type != "point":
                return
            color = rule.fill_for(feature.tags)
            if color is None:
                return
            for path in feature.paths:
                for point in path.points:
                    cx, cy = to_px(point)
                    self._stamp_disc(buf, out_w, out_h, cx, cy,
                                     float(rule.radius), color)
        elif rule.kind == KIND_LABEL:
            text = str(feature.tags.get(rule.text_field) or "").strip()
            if not text:
                return
            anchor = self._anchor(feature, to_px)
            if anchor is None:
                return
            labels.append(LabelSpec(
                x=anchor[0], y=anchor[1], text=text,
                size_px=max(6, round(rule.text_size_for(zoom))),
                color=parse_color(rule.text_color),
                halo=parse_color(rule.halo_color)
                if rule.halo_color else None,
                halo_width=int(rule.halo_width)))

    @staticmethod
    def _anchor(feature: MvtFeature, to_px) -> tuple[float, float] | None:
        """Where a feature's label sits: point → the point;
        line → the middle vertex of its longest path;
        polygon → the centre of its bounding box."""
        points: list[tuple[int, int]] = []
        for path in feature.paths:
            points.extend(path.points)
        if not points:
            return None
        if feature.geom_type == "point":
            return to_px(points[0])
        if feature.geom_type == "linestring":
            longest = max(feature.paths, key=lambda p: len(p.points),
                          default=None)
            if longest is None or not longest.points:
                return None
            return to_px(longest.points[len(longest.points) // 2])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return to_px(((min(xs) + max(xs)) // 2,
                      (min(ys) + max(ys)) // 2))

    # ── raster primitives ─────────────────────────────────────────

    @staticmethod
    def _set(buf: bytearray, w: int, h: int, x: int, y: int,
             rgb: tuple[int, int, int]) -> None:
        if 0 <= x < w and 0 <= y < h:
            at = (y * w + x) * 3
            buf[at:at + 3] = bytes(rgb)

    def _fill_rect(self, buf: bytearray, w: int, h: int,
                   x0: float, y0: float, x1: float, y1: float,
                   rgb: tuple[int, int, int]) -> None:
        xa, xb = max(0, math.floor(min(x0, x1))), min(w, math.ceil(max(x0, x1)))
        ya, yb = max(0, math.floor(min(y0, y1))), min(h, math.ceil(max(y0, y1)))
        row = bytes(rgb) * max(0, xb - xa)
        for y in range(ya, yb):
            at = (y * w + xa) * 3
            buf[at:at + len(row)] = row

    def _stamp_disc(self, buf: bytearray, w: int, h: int,
                    cx: float, cy: float, radius: float,
                    rgb: tuple[int, int, int]) -> None:
        r = max(0.5, float(radius))
        xa = max(0, int(math.floor(cx - r)))
        xb = min(w - 1, int(math.ceil(cx + r)))
        ya = max(0, int(math.floor(cy - r)))
        yb = min(h - 1, int(math.ceil(cy + r)))
        for y in range(ya, yb + 1):
            for x in range(xa, xb + 1):
                dx, dy = x + 0.5 - cx, y + 0.5 - cy
                if dx * dx + dy * dy <= r * r:
                    self._set(buf, w, h, x, y, rgb)

    def _stroke_path(self, buf: bytearray, w: int, h: int,
                     pts: list[tuple[float, float]],
                     rgb: tuple[int, int, int], width: float,
                     close: bool) -> None:
        """A polyline as overlapping discs: round joins by shape.

        Honest approximation, stated: joins and caps are round
        because the stamp is a disc; there is no miter math and
        no dash pattern — the style's widths are honored, its
        geometry is walked at sub-width steps so no gap opens
        along a fast diagonal.
        """
        radius = max(0.5, float(width) / 2.0)
        step = max(0.4, radius / 2.0)
        chain = list(pts)
        if close and len(chain) >= 2 and chain[0] != chain[-1]:
            chain.append(chain[0])
        for at in range(len(chain) - 1):
            ax, ay = chain[at]
            bx, by = chain[at + 1]
            length = math.hypot(bx - ax, by - ay)
            stamps = max(1, int(math.ceil(length / step)))
            for i in range(stamps + 1):
                t = i / stamps
                self._stamp_disc(buf, w, h, ax + (bx - ax) * t,
                                 ay + (by - ay) * t, radius, rgb)

    def _fill_rings(self, buf: bytearray, w: int, h: int,
                    rings: list[list[tuple[float, float]]],
                    rgb: tuple[int, int, int]) -> None:
        """Even-odd scanline fill over all of a feature's rings:
        outer rings fill, inner rings (holes) empty — the rule
        the MVT winding order implies, applied geometrically."""
        edges: list[tuple[float, float, float, float]] = []
        for ring in rings:
            for at in range(len(ring)):
                ax, ay = ring[at]
                bx, by = ring[(at + 1) % len(ring)]
                if ay != by:
                    edges.append((ax, ay, bx, by))
        if not edges:
            return
        y_lo = max(0, int(math.floor(
            min(min(ay, by) for _ax, ay, _bx, by in edges))))
        y_hi = min(h - 1, int(math.ceil(
            max(max(ay, by) for _ax, ay, _bx, by in edges))))
        for y in range(y_lo, y_hi + 1):
            scan = y + 0.5
            crossings: list[float] = []
            for ax, ay, bx, by in edges:
                if (ay <= scan < by) or (by <= scan < ay):
                    crossings.append(
                        ax + (scan - ay) * (bx - ax) / (by - ay))
            crossings.sort()
            for at in range(0, len(crossings) - 1, 2):
                xa = max(0, int(math.ceil(crossings[at] - 0.5)))
                xb = min(w - 1,
                         int(math.floor(crossings[at + 1] - 0.5)))
                for x in range(xa, xb + 1):
                    self._set(buf, w, h, x, y, rgb)
