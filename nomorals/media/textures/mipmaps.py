"""Mipmaps and texture sampling (TextureCraft port, census 3A.15).

A mipmap chain is built by box-averaging 2x2 texel blocks (real
downsampling, not a stub); each level is a sparse ``TileImage`` of
its own. Sampling honours the wrap mode (repeat / mirror / clamp)
and the filter (nearest / bilinear / trilinear — trilinear
interpolates between the two nearest levels).
"""
from __future__ import annotations

from enum import StrEnum

from .tiles import TileImage, bilinear


class WrapMode(StrEnum):
    REPEAT = "repeat"
    MIRROR = "mirror"
    CLAMP = "clamp"

    @classmethod
    def parse(cls, name: str) -> WrapMode | None:
        needle = name.strip().lower().replace(" ", "_").replace("-", "_")
        for m in cls:
            if m.value == needle:
                return m
        return None


class FilterMode(StrEnum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    TRILINEAR = "trilinear"

    @classmethod
    def parse(cls, name: str) -> FilterMode | None:
        needle = name.strip().lower().replace(" ", "_").replace("-", "_")
        for m in cls:
            if m.value == needle:
                return m
        return None


def mipmap_chain(img: TileImage) -> list[TileImage]:
    """Box-downsample chain: level 0 is ``img`` itself, each next
    level halves both dimensions until 1x1."""
    chain = [img]
    cur = img
    while cur.width > 1 or cur.height > 1:
        nw = max(1, cur.width // 2)
        nh = max(1, cur.height // 2)
        nxt = TileImage(nw, nh, cur.comps)
        for ty in range(nxt.tiles_y):
            for tx in range(nxt.tiles_x):
                # Allocate the tile only when its source 2x2 blocks
                # hold data.
                if not cur.any_allocated_in(tx * 128, ty * 128,
                                            tx * 128 + 127, ty * 128 + 127):
                    continue
                tile = nxt.tile_mut(tx, ty)
                for ly in range(64):
                    gy = ty * 64 + ly
                    if gy >= nh:
                        break
                    for lx in range(64):
                        gx = tx * 64 + lx
                        if gx >= nw:
                            break
                        vals = []
                        for c in range(cur.comps):
                            s = (cur.get(gx * 2, gy * 2)[c]
                                 + cur.get(gx * 2 + 1, gy * 2)[c]
                                 + cur.get(gx * 2, gy * 2 + 1)[c]
                                 + cur.get(gx * 2 + 1, gy * 2 + 1)[c])
                            vals.append((s + 2) // 4)
                        off = (ly * 64 + lx) * cur.comps * 2
                        for c, v in enumerate(vals):
                            tile[off + c * 2] = v & 0xFF
                            tile[off + c * 2 + 1] = (v >> 8) & 0xFF
        nxt.prune_empty()
        chain.append(nxt)
        cur = nxt
    return chain


def _wrap(coord: float, mode: WrapMode) -> float:
    if mode is WrapMode.REPEAT:
        return coord % 1.0
    if mode is WrapMode.MIRROR:
        t = coord % 2.0
        return t if t <= 1.0 else 2.0 - t
    return min(1.0, max(0.0, coord))


def _sample_level(img: TileImage, u: float, v: float,
                  wrap: WrapMode, filter_mode: FilterMode) -> list[int]:
    u = _wrap(u, wrap)
    v = _wrap(v, wrap)
    if filter_mode is FilterMode.NEAREST:
        x = min(img.width - 1, max(0, int(u * img.width)))
        y = min(img.height - 1, max(0, int(v * img.height)))
        return img.get(x, y)
    fx = u * img.width
    fy = v * img.height
    vals = bilinear(img, fx, fy)
    return (vals + [0, 0, 0, 0])[:4]


def sample(chain: list[TileImage], u: float, v: float, level: float,
           wrap: WrapMode, filter_mode: FilterMode) -> list[int]:
    """Sample the mipmap chain at normalised UV (0..1) and a
    (possibly fractional) mip level. Returns ``[u16; 4]``."""
    if not chain:
        raise ValueError("empty mipmap chain")
    if filter_mode is FilterMode.TRILINEAR:
        lo = int(level // 1)
        hi = lo + 1
        frac = level - lo
        lo = min(max(0, lo), len(chain) - 1)
        hi = min(max(0, hi), len(chain) - 1)
        a = _sample_level(chain[lo], u, v, wrap, FilterMode.BILINEAR)
        b = _sample_level(chain[hi], u, v, wrap, FilterMode.BILINEAR)
        return [int(a[i] * (1.0 - frac) + b[i] * frac + 0.5)
                for i in range(4)]
    lvl = min(max(0, int(round(level))), len(chain) - 1)
    return _sample_level(chain[lvl], u, v, wrap, filter_mode)
