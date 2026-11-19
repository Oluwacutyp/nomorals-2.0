"""The shared compositor — one numpy float RGBA pipeline for both
``PsdFile.flatten_layers`` and ``PhotoDocument.flatten``.

Layers composite bottom-first (PSD file order) onto a straight-
alpha float canvas: pixel layers (with masks, opacity, fill and
blend modes), adjustment layers (applied to the canvas below
them), fill layers (solid / gradient / pattern, rendered), layer
effects (exterior below the layer, interior with it), live text
(the stored raster; a vector re-render only where the layer has
no raster at all), and smart objects (their stored pixels — the
smart filters are already baked into them by Photoshop, so the
compositor does not re-apply them). Groups composite children
directly when pass-through, else onto a buffer blended as one.
A clipped layer is masked to the alpha of the layer below it.

The compositor never guesses: an unparsable adjustment/effect is
reported through the layer's ``unrendered`` note and the canvas
keeps the stored pixels (the oracle names the cause), and an
unknown blend key falls back to normal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .adjust import ADJUSTMENT_NAMES, apply_adjustment, parse_adjustment
from .blend import (
    TEXT_GAMMA,
    blend_rgb,
    composite,
    composite_fill,
    dissolve_mask,
)
from .effects import composite_with_effects, parse_effects
from .fill import FILL_KEYS, parse_fill_layer, render_fill
from .smart import parse_plld, parse_sold
from .text import parse_tysh, render_text

__all__ = ["CompositeResult", "apply_mask", "composite_layers",
           "describe_layer", "mask_to_canvas", "pil_to_array",
           "array_to_pil"]


def pil_to_array(image: Any) -> np.ndarray:
    """A PIL RGBA image -> float64 (H, W, 4) in 0..1."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    arr = np.asarray(image, dtype=np.float64) / 255.0
    return np.ascontiguousarray(arr)


def array_to_pil(arr: np.ndarray) -> Any:
    """A float (H, W, 4) array in 0..1 -> a PIL RGBA image
    (rounded to whole levels, as Photoshop's composite is)."""
    clipped = np.clip(np.asarray(arr, dtype=np.float64),
                      0.0, 1.0)
    bytes_arr = np.round(clipped * 255.0).astype(np.uint8)
    return Image.fromarray(bytes_arr, mode="RGBA")


def apply_mask(rgba: np.ndarray, mask: bytes | None,
               mask_rect: tuple[int, int, int, int] | None,
               layer_rect: tuple[int, int, int, int],
               *, mask_default: int = 255,
               disabled: bool = False) -> np.ndarray:
    """A layer mask over float RGBA: the mask plane (8-bit,
    mask-rect-sized) multiplies the layer's alpha where it
    covers, the mask default fills the rest of the layer
    rect. A disabled mask or no mask returns the input."""
    if mask is None or mask_rect is None or disabled:
        return rgba
    top, left, bottom, right = layer_rect
    height = max(0, bottom - top)
    width = max(0, right - left)
    if rgba.shape[:2] != (height, width):
        return rgba
    full = np.full((height, width), mask_default / 255.0,
                   dtype=np.float64)
    mtop, mleft, mbottom, mright = mask_rect
    mw = max(0, mright - mleft)
    mh = max(0, mbottom - mtop)
    if mw and mh and len(mask) >= mw * mh:
        plane = np.frombuffer(mask[:mw * mh], dtype=np.uint8)
        plane = plane.reshape((mh, mw)).astype(np.float64) \
            / 255.0
        y0 = max(0, mtop - top)
        x0 = max(0, mleft - left)
        y1 = min(height, y0 + mh)
        x1 = min(width, x0 + mw)
        if y1 > y0 and x1 > x0:
            full[y0:y1, x0:x1] = plane[:y1 - y0, :x1 - x0]
    out = rgba.copy()
    out[..., 3] *= full
    return out


def mask_to_canvas(mask: bytes | None,
                   mask_rect: tuple[int, int, int, int] | None,
                   width: int, height: int, *,
                   mask_default: int = 255,
                   disabled: bool = False) -> np.ndarray | None:
    """A layer mask as a full-canvas float array (1.0 where the
    mask is white): the mask plane pasted at its rect, the
    mask default elsewhere. None when there is no mask or it
    is disabled."""
    if mask is None or mask_rect is None or disabled:
        return None
    full = np.full((height, width), mask_default / 255.0,
                   dtype=np.float64)
    mtop, mleft, mbottom, mright = mask_rect
    mw = max(0, mright - mleft)
    mh = max(0, mbottom - mtop)
    if mw and mh and len(mask) >= mw * mh:
        plane = np.frombuffer(mask[:mw * mh], dtype=np.uint8)
        plane = plane.reshape((mh, mw)).astype(np.float64) \
            / 255.0
        y0 = max(0, mtop)
        x0 = max(0, mleft)
        y1 = min(height, mbottom)
        x1 = min(width, mright)
        if y1 > y0 and x1 > x0:
            full[y0:y1, x0:x1] = plane[y0 - mtop:y1 - mtop,
                                       x0 - mleft:x1 - mleft]
    return full


def describe_layer(*, rgba: np.ndarray | None = None,
                   left: int = 0, top: int = 0,
                   width: int = 0, height: int = 0,
                   opacity: int = 255, fill: float = 1.0,
                   blend: str = "normal", visible: bool = True,
                   clipping: int = 0, kind: str = "pixels",
                   name: str = "Layer",
                   raw_blocks: list | dict | None = None,
                   doc_width: int = 0, doc_height: int = 0,
                   advanced: dict | None = None,
                   patterns: dict | None = None,
                   global_light: tuple[float, float] = (120.0,
                                                         30.0),
                   mask: np.ndarray | None = None
                   ) -> dict:
    """One layer as the compositor reads it: the pixels plus
    the parsed adjustment / fill / effects / text / smart
    features from its raw blocks. Text layers with no stored
    raster are re-rendered from the ``TySh`` spec (the only
    place the vector path runs); every other feature is
    parsed here so the compositor never re-reads blocks."""
    blocks = dict(raw_blocks or [])
    adjustment: dict | None = None
    fill_spec: dict | None = None
    effects: dict | None = None
    text_spec: dict | None = None
    smart: dict | None = None
    is_text = "TySh" in blocks
    if kind == "adjustment":
        for key, data in blocks.items():
            if key in ADJUSTMENT_NAMES:
                cged = blocks.get("CgEd")
                adjustment = parse_adjustment(
                    key, bytes(data),
                    bytes(cged) if cged is not None else None)
                break
    elif kind == "fill":
        for key, data in blocks.items():
            if key in FILL_KEYS:
                fill_spec = parse_fill_layer(key, bytes(data))
                break
    if "lfx2" in blocks or "lrFX" in blocks:
        effects = parse_effects(blocks)
    if is_text:
        try:
            text_spec = parse_tysh(bytes(blocks["TySh"]))
        except Exception:
            text_spec = None
    if "SoLd" in blocks:
        try:
            smart = parse_sold(bytes(blocks["SoLd"]))
        except Exception:
            smart = None
    elif "PlLd" in blocks:
        try:
            smart = parse_plld(bytes(blocks["PlLd"]))
        except Exception:
            smart = None
    if rgba is None and text_spec is not None \
            and doc_width > 0 and doc_height > 0:
        full = render_text(text_spec, doc_width, doc_height,
                           left=0, top=0)
        if full is not None:
            rgba = np.zeros((max(0, height), max(0, width), 4),
                            dtype=np.float64)
            y0 = max(0, top)
            x0 = max(0, left)
            y1 = min(doc_height, top + height)
            x1 = min(doc_width, left + width)
            if y1 > y0 and x1 > x0:
                rgba[y0 - top:y1 - top,
                     x0 - left:x1 - left] = \
                    full[y0:y1, x0:x1]
    return {
        "rgba": rgba,
        "left": left,
        "top": top,
        "opacity": opacity,
        "fill": fill,
        "blend": blend,
        "visible": visible,
        "clipping": clipping,
        "kind": kind,
        "name": name,
        "adjustment": adjustment,
        "fill_spec": fill_spec,
        "effects": effects,
        "is_text": is_text,
        "text": text_spec,
        "smart": smart,
        "advanced": advanced or {},
        "patterns": patterns or {},
        "global_light": global_light,
        "mask": mask,
    }


class CompositeResult:
    """What a composite produced: the image, and the layers it
    could not render (each a human line for the oracle/notes)."""

    def __init__(self, image: Any, unrendered: list[str]) -> None:
        self.image = image
        self.unrendered = unrendered


def _place(canvas: np.ndarray, layer_rgba: np.ndarray,
           left: int, top: int) -> tuple[np.ndarray, int, int, int,
                                         int]:
    """The overlap of a layer rect with the canvas: returns the
    canvas view, the layer view, and the canvas origin. Empty
    overlap returns views with a zero height."""
    height, width = canvas.shape[:2]
    lh, lw = layer_rgba.shape[:2]
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(width, left + lw)
    y1 = min(height, top + lh)
    if x1 <= x0 or y1 <= y0:
        empty = np.zeros((0, 0, 4), dtype=np.float64)
        return empty, empty, x0, y0
    canvas_view = canvas[y0:y1, x0:x1]
    layer_view = layer_rgba[y0 - top:y1 - top, x0 - left:x1 - left]
    return canvas_view, layer_view, x0, y0


def _composite_layer(canvas: np.ndarray, layer_rgba: np.ndarray,
                     left: int, top: int, *, blend: str = "normal",
                     opacity: float = 1.0, fill: float = 1.0,
                     gamma: float = 1.0) -> None:
    """Composite one layer's RGBA onto the canvas in place."""
    canvas_view, layer_view, x0, y0 = _place(canvas, layer_rgba,
                                             left, top)
    if canvas_view.shape[0] == 0:
        return
    if blend == "dissolve":
        # Dissolve resolves per pixel before compositing: the
        # kept pixels composite as normal at full coverage.
        keep = dissolve_mask(layer_view[..., 3], x0, y0)
        kept = layer_view.copy()
        kept[..., 3] = np.where(keep, layer_view[..., 3], 0.0)
        result = composite("normal", canvas_view, kept, opacity)
        canvas_view[:, :, :] = result
        return
    if gamma != 1.0:
        # Text gamma: the coverage mix happens in the text
        # blending space (linear light raised to 1/gamma),
        # not in sRGB.
        from .blend import composite_gamma as _comp_gamma

        result = _comp_gamma(blend, canvas_view, layer_view,
                             opacity, gamma)
        canvas_view[:, :, :] = result
        return
    result = composite_fill(blend, canvas_view, layer_view,
                            opacity, fill)
    canvas_view[:, :, :] = result


def composite_layers(layers: list[dict], width: int, height: int,
                     *, depth: str = "u8", mode: str = "RGB",
                     effects: Any = None) -> CompositeResult:
    """Composite ``layers`` (bottom-first dicts) onto a canvas.

    Each layer dict carries: ``rgba`` (float (h, w, 4) at its own
    rect, or None for a non-pixel layer), ``left``/``top``,
    ``opacity`` (0..255), ``fill`` (0..1, default 1), ``blend``
    (mode name), ``visible``, ``clipping`` (0/1), ``kind``
    (``pixels`` / ``adjustment`` / ``fill`` / ``group-start`` /
    ``group-end``), ``adjustment`` (parsed dict or None),
    ``fill_spec`` (parsed dict or None), ``effects`` (parsed dict
    or None), ``name``. ``effects`` (the module, for rendering)
    is passed in to avoid a circular import.
    """
    canvas = np.zeros((height, width, 4), dtype=np.float64)
    unrendered: list[str] = []
    # Group stack: (parent canvas, group dict, buffered?) — a
    # pass-through group composites its children directly onto
    # the parent canvas; any other blend buffers them first.
    stack: list[tuple[np.ndarray, dict, bool]] = []
    base_alpha: np.ndarray | None = None
    for layer in layers:
        kind = layer.get("kind", "pixels")
        if kind == "group-start":
            buffered = layer.get("blend", "normal") \
                != "pass_through"
            stack.append((canvas, layer, buffered))
            if buffered:
                canvas = np.zeros((height, width, 4),
                                  dtype=np.float64)
            continue
        if kind == "group-end":
            if not stack:
                continue
            parent, group, buffered = stack.pop()
            if not buffered:
                # Pass-through: the children already went onto
                # the parent canvas; nothing to merge.
                continue
            group_canvas = canvas
            canvas = parent
            _composite_layer(
                canvas, group_canvas, 0, 0,
                blend=group.get("blend", "normal"),
                opacity=group.get("opacity", 255) / 255.0)
            continue
        if not layer.get("visible", True):
            continue
        opacity = layer.get("opacity", 255) / 255.0
        fill = layer.get("fill", 1.0)
        blend = layer.get("blend", "normal")
        clipping = layer.get("clipping", 0)
        name = layer.get("name", "Layer")
        if kind == "adjustment":
            adj = layer.get("adjustment")
            if adj is None or adj.get("kind") == "unsupported":
                reason = (adj or {}).get("reason", "unparsed")
                unrendered.append(
                    f"adjustment layer {name!r} is not rendered "
                    f"({reason})")
                continue
            base = canvas[..., :3].copy()
            adjusted = apply_adjustment(
                base, canvas[..., 3], adj, depth=depth, mode=mode)
            # The layer's blend mode blends the adjusted
            # result with the canvas below it (a color-blend
            # hue/saturation layer does not replace the RGB).
            if blend != "normal":
                adjusted = blend_rgb(blend, base, adjusted)
            # Opacity, layer mask and clipping all scale the
            # same mix.
            mix = np.full(canvas.shape[:2], opacity,
                          dtype=np.float64)
            layer_mask = layer.get("mask")
            if layer_mask is not None:
                mix = mix * layer_mask
            if clipping and base_alpha is not None:
                mix = mix * base_alpha
            canvas[..., :3] = (base * (1.0 - mix[..., None])
                               + adjusted * mix[..., None])
            continue
        if kind == "fill":
            spec = layer.get("fill_spec")
            if spec is None:
                unrendered.append(
                    f"fill layer {name!r} is not rendered "
                    "(unparsed)")
                continue
            rgba = _render_fill(spec, width, height)
            if rgba is None:
                unrendered.append(
                    f"fill layer {name!r} is not rendered "
                    "(unsupported fill)")
                continue
            layer_mask = layer.get("mask")
            if layer_mask is not None:
                rgba = rgba.copy()
                rgba[..., 3] *= layer_mask
            before = canvas.copy()
            _composite_layer(canvas, rgba, 0, 0, blend=blend,
                             opacity=opacity, fill=fill)
            if clipping and base_alpha is not None:
                mask = base_alpha[..., None]
                canvas[:, :, :] = (
                    before * (1.0 - mask) + canvas * mask)
            base_alpha = rgba[..., 3].copy()
            continue
        rgba = layer.get("rgba")
        if rgba is None:
            continue
        # Layer effects: the whole EffectPass (exterior
        # shadows/glows, the layer pixels, interior overlays
        # and strokes, bevels) in one call. A clipped layer's
        # effects are clipped with it, so the snapshot is
        # taken before the pass.
        fx = layer.get("effects")
        if fx is not None:
            before = canvas.copy()
            render = (effects.composite_with_effects
                      if effects is not None
                      and hasattr(effects,
                                  "composite_with_effects")
                      else composite_with_effects)
            render(canvas, rgba, layer.get("left", 0),
                   layer.get("top", 0), blend=blend,
                   opacity=opacity, fill=fill, effects=fx,
                   is_text=bool(layer.get("is_text")),
                   advanced=layer.get("advanced") or {},
                   patterns=layer.get("patterns") or {},
                   global_light=layer.get("global_light")
                   or (120.0, 30.0))
            if clipping and base_alpha is not None:
                mask = base_alpha[..., None]
                canvas[:, :, :] = (
                    before * (1.0 - mask) + canvas * mask)
            base_alpha = _layer_alpha(rgba, layer.get("left", 0),
                                      layer.get("top", 0),
                                      width, height)
            continue
        before = canvas.copy()
        _composite_layer(canvas, rgba, layer.get("left", 0),
                         layer.get("top", 0), blend=blend,
                         opacity=opacity, fill=fill,
                         gamma=TEXT_GAMMA if layer.get("is_text")
                         else 1.0)
        if clipping and base_alpha is not None:
            mask = base_alpha[..., None]
            canvas[:, :, :] = (
                before * (1.0 - mask) + canvas * mask)
        base_alpha = _layer_alpha(rgba, layer.get("left", 0),
                                  layer.get("top", 0), width, height)
    image = array_to_pil(canvas)
    return CompositeResult(image, unrendered)


def _layer_alpha(rgba: np.ndarray, left: int, top: int,
                 width: int, height: int) -> np.ndarray:
    """A layer's alpha placed on a full-canvas array."""
    out = np.zeros((height, width), dtype=np.float64)
    _, layer_view, x0, y0 = _place(
        np.zeros((height, width, 4)), rgba, left, top)
    if layer_view.shape[0] == 0:
        return out
    h, w = layer_view.shape[:2]
    out[y0:y0 + h, x0:x0 + w] = layer_view[..., 3]
    return out


def _render_fill(spec: dict, width: int, height: int
                 ) -> np.ndarray | None:
    """A fill layer's RGBA at canvas size (solid / gradient /
    pattern), or None when the fill cannot be rendered."""

    return render_fill(spec, width, height)


def flatten_pil_layers(layers: list[dict], width: int, height: int,
                       *, depth: str = "u8",
                       effects: Any = None) -> Any:
    """Composite PIL-based layer dicts (``pil`` RGBA images) — the
    ``PhotoDocument`` path."""
    prepared = []
    for layer in layers:
        item = dict(layer)
        pil = layer.get("pil")
        if pil is not None:
            item["rgba"] = pil_to_array(pil)
        prepared.append(item)
    return composite_layers(prepared, width, height, depth=depth,
                            effects=effects).image
