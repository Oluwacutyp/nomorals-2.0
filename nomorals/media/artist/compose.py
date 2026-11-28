"""Render a scene to pixels — 2D compositing and 3D staging.

Two renders, both deterministic and CPU-only (PIL + numpy, no GPU):

* :func:`flatten_layers` — the 2D compositor: the scene's layers in
  order, with opacity, rotation and blend modes (normal / multiply /
  screen), exactly the ArtCraft 2D canvas.
* :func:`render_staged_frame` — the 3D stage: the location backdrop,
  then every staged item projected through the scene camera
  (``scale = focal / (focal + depth * DEPTH_UNIT)``), drawn far to
  near so nearer items occlude farther ones. A character with no
  reference image renders as a posed mannequin placeholder — a
  blocking aid, labelled with the character's name, never presented
  as the character.

A missing image file is an error naming the path, never a blank
substitute: a staged frame that silently drops a layer is a lie.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont

from ..typography import get_service, pil_font_for
from .scene import ArtistError, Camera, Scene

_log = logging.getLogger(__name__)

#: Pixels per depth unit in the staging projection.
DEPTH_UNIT = 100.0

#: A staged item with no image renders this size (before projection).
MANNEQUIN_SIZE = (120, 240)

#: Mannequin placeholder colour (warm amber, distinct from any backdrop).
MANNEQUIN_COLOR = (230, 180, 60)


def _parse_color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _font(size: int) -> Any:
    """Cover-art labels wear the cover context (Anton): the
    scene's titles are display type, not the old DejaVu door
    (gapfix 164). A total miss is logged and the Pillow bitmap
    default carries the label."""
    size = max(8, int(size))
    try:
        return pil_font_for(get_service(), "cover", size, weight=700)
    except Exception as exc:  # noqa: BLE001 - named miss
        _log.warning("artist font could not resolve: %s", exc)
        return ImageFont.load_default()


def _load_image(path: str) -> Image.Image:
    if not path:
        raise ArtistError("an image layer needs a path")
    file = Path(path)
    if not file.exists():
        raise ArtistError(f"no such image: {path}")
    try:
        with Image.open(file) as opened:
            return opened.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise ArtistError(f"cannot read image {path}: {exc}") from exc


def _apply_opacity(tile: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1.0:
        return tile
    alpha = tile.split()[3].point(lambda v: int(v * opacity))
    tile = tile.copy()
    tile.putalpha(alpha)
    return tile


def _render_tile(layer: Any) -> Image.Image:
    """Render one layer to an RGBA tile of its own box size."""
    width = max(1, int(round(layer.width)))
    height = max(1, int(round(layer.height)))
    tile = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if layer.kind == "solid":
        tile = Image.new("RGBA", (width, height),
                         _parse_color(layer.color) + (255,))
    elif layer.kind == "shape":
        draw = ImageDraw.Draw(tile)
        fill = _parse_color(layer.color) + (255,)
        if layer.shape == "rect":
            draw.rectangle([0, 0, width - 1, height - 1], fill=fill)
        elif layer.shape == "ellipse":
            draw.ellipse([0, 0, width - 1, height - 1], fill=fill)
        elif layer.shape == "line":
            draw.line([0, height // 2, width - 1, height // 2],
                      fill=fill, width=max(1, height // 8))
    elif layer.kind == "image":
        source = _load_image(layer.path)
        tile = source.resize((width, height), Image.LANCZOS)
    elif layer.kind == "text":
        draw = ImageDraw.Draw(tile)
        font = _font(layer.font_size)
        draw.text((0, 0), layer.text, font=font,
                  fill=_parse_color(layer.color) + (255,))
    else:  # pragma: no cover — Layer.validate guards this
        raise ArtistError(f"unknown layer kind {layer.kind!r}")
    if layer.rotation:
        rotated = tile.rotate(-layer.rotation, expand=True,
                              resample=Image.BICUBIC)
        return rotated
    return tile


def _paste_blend(canvas: Image.Image, tile: Image.Image, box: tuple[int, int],
                 blend: str) -> None:
    """Paste ``tile`` at ``box`` honouring the layer blend mode."""
    if blend == "normal":
        canvas.alpha_composite(tile, box)
        return
    x, y = box
    width, height = tile.size
    region = canvas.crop((x, y, x + width, y + height)).convert("RGB")
    tile_rgb = tile.convert("RGB")
    if blend == "multiply":
        blended = ImageChops.multiply(region, tile_rgb)
    else:  # screen
        blended = ImageChops.screen(region, tile_rgb)
    # The blended result is opaque; re-apply the tile's own alpha so a
    # translucent layer stays translucent over the blended region.
    out = blended.convert("RGBA")
    out.putalpha(tile.split()[3])
    canvas.alpha_composite(out, (x, y))


def flatten_layers(scene: Scene,
                   size: tuple[int, int] | None = None) -> Image.Image:
    """Composite the scene's 2D layers into one RGB image."""
    width, height = size or (scene.width, scene.height)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for layer in scene.layers:
        tile = _render_tile(layer)
        tile = _apply_opacity(tile, layer.opacity)
        if layer.rotation:
            # The rotated tile is larger than the layer box; re-centre it.
            box_w, box_h = int(round(layer.width)), int(round(layer.height))
            dx = (tile.size[0] - box_w) // 2
            dy = (tile.size[1] - box_h) // 2
            box = (int(round(layer.x)) - dx, int(round(layer.y)) - dy)
        else:
            box = (int(round(layer.x)), int(round(layer.y)))
        _paste_blend(canvas, tile, box, layer.blend)
    return canvas.convert("RGB")


def _cover_fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize ``image`` to cover ``size``, centre-cropped."""
    width, height = size
    scale = max(width / image.size[0], height / image.size[1])
    resized = image.resize(
        (max(1, int(round(image.size[0] * scale))),
         max(1, int(round(image.size[1] * scale)))),
        Image.LANCZOS)
    left = (resized.size[0] - width) // 2
    top = (resized.size[1] - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _project(camera: Camera, size: tuple[int, int], depth: float,
             world_x: float, world_y: float) -> tuple[float, float, float]:
    """Camera projection: (screen_x, screen_y, scale) for a world point."""
    width, height = size
    half_fov = math.radians(camera.fov) / 2.0
    focal = (width / 2.0) / max(0.05, math.tan(half_fov))
    scale = focal / (focal + depth * DEPTH_UNIT)
    screen_x = width / 2.0 + (world_x - camera.x) * scale
    screen_y = height / 2.0 + (world_y - camera.y) * scale
    return screen_x, screen_y, scale


def _mannequin(name: str, size: tuple[int, int]) -> Image.Image:
    """A posed-man placeholder for a character with no reference image."""
    width, height = size
    tile = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    head_r = max(4, width // 6)
    cx = width // 2
    draw.ellipse([cx - head_r, 0, cx + head_r, head_r * 2],
                 fill=MANNEQUIN_COLOR + (255,))
    draw.rounded_rectangle(
        [width // 4, head_r * 2 + 4, 3 * width // 4, height - 4],
        radius=8, fill=MANNEQUIN_COLOR + (255,))
    font = _font(max(10, width // 8))
    draw.text((4, height - max(14, width // 6)), name, font=font,
              fill=(20, 20, 30, 255))
    return tile


def render_staged_frame(scene: Scene,
                        size: tuple[int, int] | None = None,
                        camera: Camera | None = None) -> Image.Image:
    """Render the 3D stage: backdrop + depth-ordered items + camera.

    The 2D layers are NOT part of this render — the staged frame is
    the scene before generation, and the layers are composited onto
    the generated image afterwards (see :func:`flatten_layers`).
    """
    width, height = size or (scene.width, scene.height)
    cam = camera or scene.camera
    if scene.location.backdrop:
        backdrop = _cover_fit(
            _load_image(scene.location.backdrop), (width, height))
        canvas = backdrop.convert("RGBA")
    else:
        canvas = Image.new("RGBA", (width, height),
                           _parse_color(scene.backdrop_color) + (255,))
    items = sorted(scene.staging, key=lambda item: item.depth, reverse=True)
    for item in items:
        if item.image:
            source = _load_image(item.image)
            base_w, base_h = source.size
        else:
            source = None
            base_w, base_h = MANNEQUIN_SIZE
        screen_x, screen_y, scale = _project(
            cam, (width, height), item.depth, item.x, item.y)
        tile_w = max(1, int(round(base_w * item.scale * scale)))
        tile_h = max(1, int(round(base_h * item.scale * scale)))
        if source is not None:
            tile = source.resize((tile_w, tile_h), Image.LANCZOS)
        else:
            name = item.character or item.name
            tile = _mannequin(name, (tile_w, tile_h))
        box = (int(round(screen_x - tile_w / 2)),
               int(round(screen_y - tile_h / 2)))
        canvas.alpha_composite(tile, box)
    return canvas.convert("RGB")
