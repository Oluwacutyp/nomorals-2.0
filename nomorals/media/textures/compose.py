"""CPU layer-stack compositor (TextureCraft port, census 3A.15).

The CPU oracle (upstream AGENTS.md §3): layers live as sparse
copy-on-write 64x64 tiles and compositing runs on the CPU per dirty
tile. ``composite`` flattens one channel of one texture set,
bottom -> top, through each layer's per-channel blend mode and
opacity, its mask, and its content effects, and returns a
``TileImage`` with one extra coverage component (``channel
components + 1``) so callers can tell covered texels from empty
ones. Group layers flatten their children (Passthrough blends
straight onto the stack below); instance layers render their source
layer's content.

Coverage: a paint layer's image carries an alpha component (the
channel's components + 1); a fill layer covers every texel of its
allocated tiles. A texel no layer covers reads the channel's
upstream default value with coverage 0.
"""
from __future__ import annotations

from .blend import BlendMode, blend1, blend3, mix3
from .errors import TextureError, TextureErrorKind
from .filters import apply_filter
from .model import Channel, TextureDocument, TextureSet
from .tiles import TILE, TileImage, from_u16, to_u16


def _channel_comps(channel: str) -> int:
    ch = Channel.parse(channel)
    return ch.components() if ch is not None else 1


def _default_u16(channel: str) -> list[int]:
    ch = Channel.parse(channel)
    vals = ch.default_value() if ch is not None else [0.0, 0.0, 0.0]
    return [to_u16(v) for v in vals]


def _mask_value(layer, x: int, y: int) -> float:
    """Mask coverage 0..1 at a texel (background when no mask image)."""
    mask = layer.mask
    if mask is None:
        return 1.0
    img = layer.mask_image()
    if img is None:
        return 1.0 if mask.get("background", "white") == "white" else 0.0
    v = from_u16(img.get(x, y)[0])
    levels = mask.get("levels") or {}
    in_low = float(levels.get("in_low", 0.0))
    in_high = float(levels.get("in_high", 1.0))
    gamma = float(levels.get("gamma", 1.0))
    out_low = float(levels.get("out_low", 0.0))
    out_high = float(levels.get("out_high", 1.0))
    if in_high > in_low:
        v = (v - in_low) / (in_high - in_low)
    v = min(1.0, max(0.0, v)) ** gamma if gamma > 0 else 0.0
    return out_low + v * (out_high - out_low)


def _layer_content(doc: TextureDocument, tset: TextureSet, layer,
                   channel: str) -> tuple[TileImage, list[int]]:
    """The layer's content image for ``channel`` plus its per-texel
    alpha (u16) as a flat list is avoided: instead the caller reads
    alpha from the image itself for paint layers (last component)
    and treats fill layers as fully covered. Returns (image,
    alpha_index) where alpha_index is the component holding alpha,
    or -1 for fully-covered."""
    img = layer.image(channel)
    if layer.kind == "paint":
        return img, img.comps - 1
    return img, -1


def _apply_effects(img: TileImage, effects: list) -> TileImage:
    out = img
    for spec in effects:
        if isinstance(spec, (list, tuple)) and spec:
            out = apply_filter(out, tuple(spec))
    return out


def composite(doc: TextureDocument, set_name: str,
              channel: str) -> TileImage:
    """Flatten one channel of one texture set to a ``TileImage``."""
    tset = None
    for s in doc.sets:
        if s.name == set_name:
            tset = s
            break
    if tset is None:
        raise TextureError(TextureErrorKind.NOT_FOUND,
                           f"no texture set {set_name!r}")
    comps = _channel_comps(channel)
    out = TileImage(tset.width, tset.height, comps + 1)
    default = _default_u16(channel)
    _composite_into(doc, tset, out, channel, comps, default)
    out.prune_empty()
    return out


def _composite_into(doc: TextureDocument, tset: TextureSet,
                    out: TileImage, channel: str, comps: int,
                    default: list[int]) -> None:
    top_level = [layer for layer in tset.layers
                 if not _is_child(tset, layer.id)]
    for layer in top_level:
        _composite_layer(doc, tset, layer, out, channel, comps, default)


def _is_child(tset: TextureSet, layer_id: int) -> bool:
    return any(layer.kind == "group" and layer_id in layer.children
               for layer in tset.layers)


def _composite_layer(doc: TextureDocument, tset: TextureSet, layer,
                     out: TileImage, channel: str, comps: int,
                     default: list[int]) -> None:
    if not layer.visible:
        return
    if layer.kind == "group":
        _composite_group(doc, tset, layer, out, channel, comps, default)
        return
    if layer.kind == "instance":
        source = _find_layer(doc, layer.source) if layer.source else None
        if source is None:
            return
        _composite_layer(doc, tset, source, out, channel, comps, default)
        return
    blend = layer.channels.get(channel)
    mode = blend.mode if blend is not None else BlendMode.NORMAL
    opacity = blend.opacity if blend is not None else 1.0
    if mode is BlendMode.PASSTHROUGH:
        mode = BlendMode.NORMAL
    img, alpha_idx = _layer_content(doc, tset, layer, channel)
    img = _apply_effects(img, layer.effects)
    _blend_image(out, img, channel, comps, mode, opacity, alpha_idx,
                 lambda x, y: _mask_value(layer, x, y), default)


def _composite_group(doc: TextureDocument, tset: TextureSet, group,
                     out: TileImage, channel: str, comps: int,
                     default: list[int]) -> None:
    blend = group.channels.get(channel)
    mode = blend.mode if blend is not None else BlendMode.PASSTHROUGH
    opacity = blend.opacity if blend is not None else 1.0
    if mode is BlendMode.PASSTHROUGH:
        for child_id in group.children:
            child = tset.layer_by_id(child_id)
            _composite_layer(doc, tset, child, out, channel, comps, default)
        return
    tmp = TileImage(tset.width, tset.height, comps + 1)
    for child_id in group.children:
        child = tset.layer_by_id(child_id)
        _composite_layer(doc, tset, child, tmp, channel, comps, default)
    # Blend the group's flattened result over the stack.
    _blend_image(out, tmp, channel, comps, mode, opacity, comps,
                 lambda x, y: _mask_value(group, x, y), default)


def _find_layer(doc: TextureDocument, layer_id: int):
    for tset in doc.sets:
        for layer in tset.layers:
            if layer.id == layer_id:
                return layer
    return None


def _blend_image(out: TileImage, img: TileImage, channel: str, comps: int,
                 mode: BlendMode, opacity: float, alpha_idx: int,
                 mask_fn, default: list[int]) -> None:
    """Blend ``img`` over ``out`` per texel, over the tiles ``img``
    actually allocates. A destination texel with coverage 0 reads
    the channel's default value as its base."""
    for ty in range(img.tiles_y):
        for tx in range(img.tiles_x):
            if not img.is_allocated(tx, ty):
                continue
            out_tile = out.tile_mut(tx, ty)
            for ly in range(TILE):
                gy = ty * TILE + ly
                if gy >= out.height:
                    break
                for lx in range(TILE):
                    gx = tx * TILE + lx
                    if gx >= out.width:
                        break
                    src = img.get(gx, gy)
                    alpha = (src[alpha_idx] / 65535.0
                             if alpha_idx >= 0 else 1.0)
                    if alpha <= 0.0:
                        continue
                    w = opacity * alpha * mask_fn(gx, gy)
                    if w <= 0.0:
                        continue
                    dst = out.get(gx, gy)
                    cov0 = dst[comps] / 65535.0
                    base = [from_u16(v) for v in dst[:comps]] \
                        if cov0 > 0.0 else \
                        [from_u16(v) for v in default[:comps]]
                    top = [from_u16(v) for v in src[:comps]]
                    if comps == 3:
                        blended = blend3(mode, tuple(base), tuple(top))
                        mixed = mix3(tuple(base), blended, w)
                    else:
                        blended = tuple(
                            blend1(mode, base[i], top[i]) for i in range(comps))
                        mixed = tuple(base[i] + (blended[i] - base[i]) * w
                                      for i in range(comps))
                    off = (ly * TILE + lx) * out.comps * 2
                    for i in range(comps):
                        v = max(0, min(65535,
                                       int(mixed[i] * 65535.0 + 0.5)))
                        out_tile[off + i * 2] = v & 0xFF
                        out_tile[off + i * 2 + 1] = (v >> 8) & 0xFF
                    # Coverage accumulates as a union.
                    new_cov = int((1.0 - (1.0 - cov0) * (1.0 - w))
                                  * 65535.0 + 0.5)
                    out_tile[off + comps * 2] = new_cov & 0xFF
                    out_tile[off + comps * 2 + 1] = (new_cov >> 8) & 0xFF
