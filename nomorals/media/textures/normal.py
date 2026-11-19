"""Normal-map derivation (TextureCraft port, census 3A.15).

``height_to_normal`` turns a height channel into a tangent-space
normal map with real Sobel gradients (edge texels replicate their
nearest neighbour). ``combine_normals`` re-orients a detail normal
onto a base normal with upstream RNM (``blend.rnm``).
"""
from __future__ import annotations

import math

from .blend import rnm
from .tiles import TileImage, to_u16


def height_to_normal(height: TileImage, strength: float = 1.0,
                     flip_y: bool = False) -> TileImage:
    """Height (1 component, u16) -> normal (3 components, packed
    0..1). The Sobel kernel measures the height gradient; the
    normal is ``normalize(-dx, -dy, 1)`` scaled by ``strength``
    (``flip_y`` flips the green channel for DirectX-style maps)."""
    out = TileImage(height.width, height.height, 3)
    # A normal map is dense by nature: every texel has a normal,
    # so every tile is allocated (flat up where the height is empty).
    for ty in range(out.tiles_y):
        for tx in range(out.tiles_x):
            tile = out.tile_mut(tx, ty)
            for ly in range(64):
                gy = ty * 64 + ly
                if gy >= height.height:
                    break
                for lx in range(64):
                    gx = tx * 64 + lx
                    if gx >= height.width:
                        break
                    def h(x: int, y: int) -> float:
                        return height.get(
                            min(max(0, x), height.width - 1),
                            min(max(0, y), height.height - 1))[0] / 65535.0
                    dx = ((h(gx + 1, gy - 1) + 2 * h(gx + 1, gy)
                           + h(gx + 1, gy + 1))
                          - (h(gx - 1, gy - 1) + 2 * h(gx - 1, gy)
                             + h(gx - 1, gy + 1))) / 8.0 * strength
                    dy = ((h(gx - 1, gy + 1) + 2 * h(gx, gy + 1)
                           + h(gx + 1, gy + 1))
                          - (h(gx - 1, gy - 1) + 2 * h(gx, gy - 1)
                             + h(gx + 1, gy - 1))) / 8.0 * strength
                    if flip_y:
                        dy = -dy
                    nx, ny, nz = -dx, -dy, 1.0
                    n = math.sqrt(nx * nx + ny * ny + nz * nz)
                    vals = [to_u16(nx / n * 0.5 + 0.5),
                            to_u16(ny / n * 0.5 + 0.5),
                            to_u16(nz / n * 0.5 + 0.5)]
                    off = (ly * 64 + lx) * 3 * 2
                    for c, v in enumerate(vals):
                        tile[off + c * 2] = v & 0xFF
                        tile[off + c * 2 + 1] = (v >> 8) & 0xFF
    return out


def flat_normal(width: int, height: int) -> TileImage:
    """A flat-up normal map (all texels (0.5, 0.5, 1.0))."""
    img = TileImage(width, height, 3)
    for ty in range(img.tiles_y):
        for tx in range(img.tiles_x):
            img.tile_mut(tx, ty, [32768, 32768, 65535])
    return img


def combine_normals(base: tuple[float, float, float],
                    detail: tuple[float, float, float]
                    ) -> tuple[float, float, float]:
    """Reorient ``detail`` onto ``base`` (upstream RNM)."""
    return rnm(base, detail)
