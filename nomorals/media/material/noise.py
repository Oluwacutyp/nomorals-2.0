"""Tileable noise primitives — the material organ's generators.

MaterialCraft port of upstream's ``crates/nodes/src/noise.rs`` and
``noise3d.rs`` (Perlin gradient noise, Worley/cellular noise, fractal
sums) plus the value-noise core. Everything is deterministic per
seed: the permutation table is built from the seed with SplitMix64,
never from Python's randomised ``hash()`` or a global RNG, so the
same seed cooks the same pixels on every machine and every run.

Tiling follows upstream: lattice coordinates wrap by the cell period
when tiling is on, so a noise tile repeats seamlessly. Pixel-based
parameters scale with ``w / 256`` (upstream's preview rule) through
the caller's ``scale`` argument.
"""

from __future__ import annotations

import numpy as np

_GRADIENTS = np.array([
    [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
], dtype=np.float64)


def _splitmix64(state: int) -> int:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def permutation_table(seed: int) -> np.ndarray:
    """A 512-entry permutation table, deterministic per seed."""
    table = np.arange(256, dtype=np.uint8)
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(255, 0, -1):
        state = _splitmix64(state)
        j = state % (i + 1)
        table[i], table[j] = table[j], table[i]
    return np.concatenate([table, table]).astype(np.int64)


def hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic [0, 1) hash of integer lattice points + seed."""
    x = np.asarray(ix, dtype=np.uint64)
    y = np.asarray(iy, dtype=np.uint64)
    h = (x * np.uint64(0x9E3779B97F4A7C15)
         ^ y * np.uint64(0xC2B2AE3D27D4EB4F)
         ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF))
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> np.uint64(33)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def perlin(w: int, h: int, cells: float, seed: int,
           tiling: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """Tileable Perlin gradient noise in [0, 1], shape (h, w)."""
    perm = permutation_table(seed)
    period_x = max(1, int(round(cells)))
    period_y = max(1, int(round(cells)))
    xs = (np.arange(w, dtype=np.float64) + 0.5) / w * cells
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h * cells
    xx, yy = np.meshgrid(xs, ys)
    xi = np.floor(xx).astype(np.int64)
    yi = np.floor(yy).astype(np.int64)
    xf = xx - xi
    yf = yy - yi
    if tiling[0]:
        xi = np.mod(xi, period_x)
    if tiling[1]:
        yi = np.mod(yi, period_y)
    xi1 = xi + 1
    yi1 = yi + 1
    if tiling[0]:
        xi1 = np.mod(xi1, period_x)
    if tiling[1]:
        yi1 = np.mod(yi1, period_y)
    g00 = _GRADIENTS[perm[(perm[xi] + yi) % 256] % 8]
    g10 = _GRADIENTS[perm[(perm[xi1] + yi) % 256] % 8]
    g01 = _GRADIENTS[perm[(perm[xi] + yi1) % 256] % 8]
    g11 = _GRADIENTS[perm[(perm[xi1] + yi1) % 256] % 8]
    d00 = g00[..., 0] * xf + g00[..., 1] * yf
    d10 = g10[..., 0] * (xf - 1.0) + g10[..., 1] * yf
    d01 = g01[..., 0] * xf + g01[..., 1] * (yf - 1.0)
    d11 = g11[..., 0] * (xf - 1.0) + g11[..., 1] * (yf - 1.0)
    u = _fade(xf)
    v = _fade(yf)
    value = _lerp(_lerp(d00, d10, u), _lerp(d01, d11, u), v)
    return ((value / np.sqrt(2.0)) * 0.5 + 0.5).astype(np.float32)


def value_noise(w: int, h: int, cells: float, seed: int,
                tiling: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """Tileable value noise in [0, 1], shape (h, w)."""
    period_x = max(1, int(round(cells)))
    period_y = max(1, int(round(cells)))
    xs = (np.arange(w, dtype=np.float64) + 0.5) / w * cells
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h * cells
    xx, yy = np.meshgrid(xs, ys)
    xi = np.floor(xx).astype(np.int64)
    yi = np.floor(yy).astype(np.int64)
    xf = xx - xi
    yf = yy - yi
    if tiling[0]:
        xi = np.mod(xi, period_x)
    if tiling[1]:
        yi = np.mod(yi, period_y)
    xi1 = xi + 1
    yi1 = yi + 1
    if tiling[0]:
        xi1 = np.mod(xi1, period_x)
    if tiling[1]:
        yi1 = np.mod(yi1, period_y)
    v00 = hash2(xi, yi, seed)
    v10 = hash2(xi1, yi, seed)
    v01 = hash2(xi, yi1, seed)
    v11 = hash2(xi1, yi1, seed)
    u = _fade(xf)
    v = _fade(yf)
    return _lerp(_lerp(v00, v10, u), _lerp(v01, v11, u), v).astype(np.float32)


def fbm(w: int, h: int, cells: float, octaves: int, seed: int,
        lacunarity: float = 2.0, gain: float = 0.5,
        kind: str = "perlin",
        tiling: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """Fractal sum of octaves of Perlin or value noise, in [0, 1]."""
    total = np.zeros((h, w), dtype=np.float64)
    amplitude = 1.0
    norm = 0.0
    freq = float(cells)
    for octave in range(max(1, int(octaves))):
        if kind == "value":
            layer = value_noise(w, h, freq, seed + octave * 7919, tiling)
        else:
            layer = perlin(w, h, freq, seed + octave * 7919, tiling)
        total += amplitude * layer.astype(np.float64)
        norm += amplitude
        amplitude *= gain
        freq *= lacunarity
    return (total / norm).astype(np.float32)


def worley(w: int, h: int, cells: float, seed: int,
           mode: str = "f2-f1",
           tiling: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """Worley/cellular noise in [0, 1], shape (h, w).

    Modes: ``f1`` (distance to nearest feature), ``f2`` (second
    nearest), ``f2-f1`` (crack/edge field, upstream's default) and
    ``cell`` (a flat random value per cell).
    """
    period_x = max(1, int(round(cells)))
    period_y = max(1, int(round(cells)))
    xs = (np.arange(w, dtype=np.float64) + 0.5) / w * cells
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h * cells
    xx, yy = np.meshgrid(xs, ys)
    cx = np.floor(xx).astype(np.int64)
    cy = np.floor(yy).astype(np.int64)
    f1 = np.full((h, w), np.inf)
    f2 = np.full((h, w), np.inf)
    cell_id = np.zeros((h, w))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx = cx + dx
            ny = cy + dy
            if tiling[0]:
                nx = np.mod(nx, period_x)
            if tiling[1]:
                ny = np.mod(ny, period_y)
            px = hash2(nx, ny, seed)
            py = hash2(nx, ny, seed + 104729)
            ddx = nx + px - xx
            ddy = ny + py - yy
            dist = np.sqrt(ddx * ddx + ddy * ddy)
            nearer = dist < f1
            f2 = np.where(nearer, f1, np.minimum(f2, dist))
            f1 = np.where(nearer, dist, f1)
            cell_id = np.where(nearer, hash2(nx, ny, seed + 31), cell_id)
    if mode == "f1":
        out = np.clip(f1, 0.0, 1.0)
    elif mode == "f2":
        out = np.clip(f2, 0.0, 1.0)
    elif mode == "cell":
        out = cell_id
    else:
        out = np.clip(f2 - f1, 0.0, 1.0)
    return out.astype(np.float32)


def warp_field(w: int, h: int, cells: float, seed: int,
               tiling: tuple[bool, bool] = (True, True)
               ) -> tuple[np.ndarray, np.ndarray]:
    """Two Perlin fields (dx, dy) in [-1, 1] for domain warping."""
    dx = perlin(w, h, cells, seed, tiling).astype(np.float64) * 2.0 - 1.0
    dy = perlin(w, h, cells, seed + 4099, tiling).astype(np.float64) * 2.0 - 1.0
    return dx, dy
