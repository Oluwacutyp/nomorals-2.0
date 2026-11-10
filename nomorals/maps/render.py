"""The renderer boundary (doc 21 S5) + the fixture renderer.

Doc 21 demoted MapLibre from "the product" to *an initial,
replaceable renderer behind Devon's own boundary*; the product is
Devon Atlas.  This module is that boundary: :class:`MapRenderer`
is the protocol every renderer satisfies — the M3 web renderer
included — taking a typed :class:`~nomorals.maps.coords.BBox`, a
zoom and an output size, and returning image bytes plus a
:class:`~nomorals.maps.receipts.MapReceipt`.  A renderer never
raises for an expected incapacity: it answers with a typed
unavailable receipt, so "Atlas cannot draw this here" is never
indistinguishable from a crash or faked with a blank image.

:class:`FixtureRenderer` is the M1 proof of the boundary: it
composites a pack's PNG tiles (the member layout the manifest
contract defines) entirely offline.  Tiles the pack does not hold
render as the named no-data color and are counted in the receipt;
a view the pack holds nothing for is unavailable, not an image of
nothing.  PNG work goes through an injectable :class:`PngBackend`
— the stdlib codec by default — so a profile without an image
backend degrades to the honest receipt, by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from . import pngcodec
from .coords import BBox, lat_to_y_float, lon_to_x_float
from .manifest import PackManifest, load_manifest, tile_relpath
from .receipts import OP_RENDER, OUTCOME_OK, OUTCOME_UNAVAILABLE, MapReceipt, MapReceiptStore

__all__ = [
    "MAX_OUTPUT_PX", "MAX_RENDER_TILES", "NO_DATA_RGB",
    "FixtureRenderer", "MapRenderer", "PngBackend", "RenderResult",
    "StdlibPngBackend",
]

#: The color a missing tile renders as — named, so a surface can
#: legend it instead of a user guessing what the dark patch is.
NO_DATA_RGB = (26, 26, 32)

#: Bounds: one render never composites more tiles or emits a
#: larger image than this.  Wider views are a later pass's tiling
#: problem, refused honestly here rather than attempted badly.
MAX_RENDER_TILES = 256
MAX_OUTPUT_PX = 4096


@runtime_checkable
class PngBackend(Protocol):
    """The image codec seam: bytes ↔ (width, height, RGB pixels)."""

    def decode_png(self, data: bytes) -> tuple[int, int, bytes]:
        ...

    def encode_png(self, width: int, height: int, rgb: bytes) -> bytes:
        ...


class StdlibPngBackend:
    """The shipped codec: :mod:`nomorals.maps.pngcodec`, stdlib only."""

    def decode_png(self, data: bytes) -> tuple[int, int, bytes]:
        return pngcodec.decode_png(data)

    def encode_png(self, width: int, height: int, rgb: bytes) -> bytes:
        return pngcodec.encode_rgb_png(width, height, rgb)


@runtime_checkable
class MapRenderer(Protocol):
    """The replaceable renderer boundary every Atlas renderer meets."""

    def render(self, bbox: BBox, zoom: int,
               size: tuple[int, int]) -> RenderResult:
        ...


@dataclass(frozen=True)
class RenderResult:
    """A render's answer: image bytes (or None) + its receipt."""

    image: bytes | None
    receipt: MapReceipt
    width: int = 0
    height: int = 0
    missing_tiles: tuple[str, ...] = ()


class FixtureRenderer:
    """Composites a fixture pack's PNG tiles into a static image."""

    def __init__(self, pack_dir: Path | str,
                 manifest: PackManifest | None = None, *,
                 store: MapReceiptStore | None = None,
                 backend: PngBackend | None = None) -> None:
        self._dir = Path(pack_dir)
        self._manifest = manifest if manifest is not None \
            else load_manifest(self._dir)
        self._store = store
        self._backend: PngBackend = backend if backend is not None \
            else StdlibPngBackend()

    @property
    def manifest(self) -> PackManifest:
        return self._manifest

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

    def render(self, bbox: BBox, zoom: int,
               size: tuple[int, int] = (512, 512)) -> RenderResult:
        manifest = self._manifest
        out_w, out_h = int(size[0]), int(size[1])
        if not (1 <= out_w <= MAX_OUTPUT_PX
                and 1 <= out_h <= MAX_OUTPUT_PX):
            return self._finish(
                OUTCOME_UNAVAILABLE,
                "output size out of bounds: " + repr(size))
        if not (manifest.zoom_min <= int(zoom) <= manifest.zoom_max):
            return self._finish(
                OUTCOME_UNAVAILABLE,
                "zoom " + str(int(zoom)) + " outside pack range "
                + str(manifest.zoom_min) + ".." + str(manifest.zoom_max))
        if manifest.tile_format != "png":
            return self._finish(
                OUTCOME_UNAVAILABLE,
                "fixture renderer draws png packs, this pack is "
                + repr(manifest.tile_format))
        tiles = bbox.tiles(int(zoom))
        if len(tiles) > MAX_RENDER_TILES:
            return self._finish(
                OUTCOME_UNAVAILABLE,
                "view spans " + str(len(tiles)) + " tiles, over the "
                + str(MAX_RENDER_TILES) + "-tile render bound")
        try:
            return self._composite(bbox, int(zoom), tiles, out_w, out_h)
        except Exception as exc:  # noqa: BLE001 — backend defects are
            # an incapacity answer here, never a crash through the seam
            return self._finish(
                OUTCOME_UNAVAILABLE,
                "image backend failed: " + type(exc).__name__
                + ": " + str(exc))

    def _composite(self, bbox: BBox, zoom: int, tiles: tuple,
                   out_w: int, out_h: int) -> RenderResult:
        manifest = self._manifest
        decoded: dict[tuple[int, int], tuple[int, int, bytes]] = {}
        missing: list[str] = []
        tile_px = 0
        for tile in tiles:
            rel = tile_relpath(tile.z, tile.x, tile.y,
                               manifest.tile_format)
            path = self._dir / rel
            if not path.is_file():
                missing.append(rel)
                continue
            width, height, rgb = self._backend.decode_png(
                path.read_bytes())
            if width != height or (tile_px and width != tile_px):
                missing.append(rel)  # misshapen tile: treated as no data
                continue
            tile_px = width
            decoded[(tile.x, tile.y)] = (width, height, rgb)
        if not decoded:
            return self._finish(
                OUTCOME_UNAVAILABLE,
                "pack holds no tiles for this view ("
                + str(len(missing)) + " missing)",
                missing=tuple(missing))
        n = 1 << zoom
        world_px = n * tile_px
        west_px = lon_to_x_float(bbox.west, zoom) * tile_px
        east_px = lon_to_x_float(bbox.east, zoom) * tile_px
        if bbox.crosses_antimeridian:
            east_px += world_px
        north_px = lat_to_y_float(bbox.north, zoom) * tile_px
        south_px = lat_to_y_float(bbox.south, zoom) * tile_px
        xs: list[int] = []
        for tile in tiles:
            if tile.x not in xs:
                xs.append(tile.x)
        ys = sorted({tile.y for tile in tiles})
        x_first, y_first = xs[0], ys[0]
        cols, rows = len(xs), len(ys)
        canvas_w, canvas_h = cols * tile_px, rows * tile_px
        canvas = bytearray(bytes(NO_DATA_RGB) * (canvas_w * canvas_h))
        for (x, y), (width, height, rgb) in decoded.items():
            col = xs.index(x)
            row = ys.index(y)
            for line in range(height):
                src = line * width * 3
                dst = ((row * tile_px + line) * canvas_w
                       + col * tile_px) * 3
                canvas[dst:dst + width * 3] = rgb[src:src + width * 3]
        origin_x = x_first * tile_px
        origin_y = y_first * tile_px
        cx0 = max(0, min(canvas_w - 1, round(west_px - origin_x)))
        cy0 = max(0, min(canvas_h - 1, round(north_px - origin_y)))
        cx1 = max(cx0 + 1, min(canvas_w, round(east_px - origin_x)))
        cy1 = max(cy0 + 1, min(canvas_h, round(south_px - origin_y)))
        crop_w, crop_h = cx1 - cx0, cy1 - cy0
        out = bytearray(out_w * out_h * 3)
        for oy in range(out_h):
            sy = cy0 + (oy * crop_h) // out_h
            for ox in range(out_w):
                sx = cx0 + (ox * crop_w) // out_w
                src = (sy * canvas_w + sx) * 3
                dst = (oy * out_w + ox) * 3
                out[dst:dst + 3] = canvas[src:src + 3]
        image = self._backend.encode_png(out_w, out_h, bytes(out))
        detail = "tiles " + str(len(decoded)) + "/" + str(len(tiles)) \
            + " present"
        if missing:
            detail += "; missing: " + ", ".join(missing[:5])
            if len(missing) > 5:
                detail += " (+" + str(len(missing) - 5) + " more)"
        return self._finish(OUTCOME_OK, detail, image=image,
                            width=out_w, height=out_h,
                            missing=tuple(missing))
