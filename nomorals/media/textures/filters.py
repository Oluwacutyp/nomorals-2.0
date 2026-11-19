"""Image filters on tile images (TextureCraft port, census 3A.15).

The filter subset this organ applies (upstream ``Filter`` in
``crates/core/src/layer.rs`` also has Hsl, ColorBalance, GradientMap
and more — those are named as not yet ported in the survey, not
faked): blur, sharpen, invert, threshold, clamp, posterize, levels.
Every filter reads through :meth:`TileImage.get`, so tile
boundaries are invisible to it, and writes a fresh sparse image.
"""
from __future__ import annotations

from .errors import TextureError, TextureErrorKind
from .tiles import TILE, TileImage


def apply_filter(img: TileImage, spec: tuple) -> TileImage:
    """Apply one filter spec to ``img`` and return a new image.
    Specs: ``("blur", radius)``, ``("sharpen", amount)``,
    ``("invert",)``, ``("threshold", level)``, ``("clamp", lo, hi)``,
    ``("posterize", levels)``, ``("levels", in_low, in_high, gamma)``."""
    if not spec:
        raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                           "empty filter spec")
    name = str(spec[0]).lower()
    if name == "blur":
        return _blur(img, float(spec[1]) if len(spec) > 1 else 1.0)
    if name == "sharpen":
        return _sharpen(img, float(spec[1]) if len(spec) > 1 else 1.0)
    if name == "invert":
        return _per_texel(img, lambda v: 65535 - v)
    if name == "threshold":
        level = float(spec[1]) if len(spec) > 1 else 0.5
        cut = int(level * 65535.0)
        return _per_texel(img, lambda v: 65535 if v >= cut else 0)
    if name == "clamp":
        lo = int(float(spec[1]) * 65535.0) if len(spec) > 1 else 0
        hi = int(float(spec[2]) * 65535.0) if len(spec) > 2 else 65535
        return _per_texel(img, lambda v: min(hi, max(lo, v)))
    if name == "posterize":
        levels = max(2, int(spec[1])) if len(spec) > 1 else 2
        step = 65535.0 / (levels - 1)
        return _per_texel(
            img, lambda v: int(round(v / step) * step + 0.5)
            if step > 0 else 0)
    if name == "levels":
        in_low = float(spec[1]) if len(spec) > 1 else 0.0
        in_high = float(spec[2]) if len(spec) > 2 else 1.0
        gamma = float(spec[3]) if len(spec) > 3 else 1.0

        def remap(v: int) -> int:
            t = (v / 65535.0 - in_low) / max(1e-6, in_high - in_low)
            t = min(1.0, max(0.0, t))
            if gamma > 0:
                t = t ** gamma
            return int(t * 65535.0 + 0.5)

        return _per_texel(img, remap)
    raise TextureError(TextureErrorKind.UNSUPPORTED,
                       f"filter not ported: {name!r}")


def _per_texel(img: TileImage, fn) -> TileImage:
    out = TileImage(img.width, img.height, img.comps)
    for ty in range(img.tiles_y):
        for tx in range(img.tiles_x):
            if not img.is_allocated(tx, ty):
                continue
            tile = out.tile_mut(tx, ty)
            for ly in range(TILE):
                gy = ty * TILE + ly
                if gy >= img.height:
                    break
                for lx in range(TILE):
                    gx = tx * TILE + lx
                    if gx >= img.width:
                        break
                    src = img.get(gx, gy)
                    off = (ly * TILE + lx) * img.comps * 2
                    for c in range(img.comps):
                        v = max(0, min(65535, int(fn(src[c]))))
                        tile[off + c * 2] = v & 0xFF
                        tile[off + c * 2 + 1] = (v >> 8) & 0xFF
    out.prune_empty()
    return out


def _blur(img: TileImage, radius: float) -> TileImage:
    r = max(1, int(round(radius)))
    out = TileImage(img.width, img.height, img.comps)
    size = 2 * r + 1
    for ty in range(img.tiles_y):
        for tx in range(img.tiles_x):
            # Blur reaches r texels past the tile edge.
            if not img.any_allocated_in(tx * TILE - r, ty * TILE - r,
                                        tx * TILE + TILE - 1 + r,
                                        ty * TILE + TILE - 1 + r):
                continue
            tile = out.tile_mut(tx, ty)
            for ly in range(TILE):
                gy = ty * TILE + ly
                if gy >= img.height:
                    break
                for lx in range(TILE):
                    gx = tx * TILE + lx
                    if gx >= img.width:
                        break
                    off = (ly * TILE + lx) * img.comps * 2
                    for c in range(img.comps):
                        total = 0
                        for dy in range(-r, r + 1):
                            for dx in range(-r, r + 1):
                                total += img.get(gx + dx, gy + dy)[c]
                        v = (total + size * size // 2) // (size * size)
                        tile[off + c * 2] = v & 0xFF
                        tile[off + c * 2 + 1] = (v >> 8) & 0xFF
    out.prune_empty()
    return out


def _sharpen(img: TileImage, amount: float) -> TileImage:
    blurred = _blur(img, 1.0)
    out = TileImage(img.width, img.height, img.comps)
    for ty in range(img.tiles_y):
        for tx in range(img.tiles_x):
            if not img.is_allocated(tx, ty):
                continue
            tile = out.tile_mut(tx, ty)
            for ly in range(TILE):
                gy = ty * TILE + ly
                if gy >= img.height:
                    break
                for lx in range(TILE):
                    gx = tx * TILE + lx
                    if gx >= img.width:
                        break
                    src = img.get(gx, gy)
                    blur = blurred.get(gx, gy)
                    off = (ly * TILE + lx) * img.comps * 2
                    for c in range(img.comps):
                        v = int(src[c] + amount * (src[c] - blur[c]) + 0.5)
                        v = max(0, min(65535, v))
                        tile[off + c * 2] = v & 0xFF
                        tile[off + c * 2 + 1] = (v >> 8) & 0xFF
    out.prune_empty()
    return out


SUPPORTED_FILTERS = ("blur", "sharpen", "invert", "threshold", "clamp",
                     "posterize", "levels")
