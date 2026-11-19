"""The shared noise engines (Craft-Libs port, census 3A.20).

Two engines lived in two organs doing the same job with different
algorithms, and the materialcraft lane flagged the duplication:

* the **point engine** — TextureCraft's f32-exact PCG3D value /
  Perlin / Worley / cellular noise, ported from upstream
  ``crates/texel/src/noise.rs`` (commit 37ad2112). Every arithmetic
  step is done in IEEE float32 exactly as the Rust f32 code does
  it, so values match the upstream binary bit-for-bit.
* the **grid engine** — MaterialCraft's f64 SplitMix64 tileable
  Perlin / value / Worley / fbm / warp fields, ported from
  upstream ``crates/nodes/src/noise.rs`` and ``noise3d.rs``
  (commit 4e8bb9ca).

Both engines moved here byte-for-byte in behaviour (the parity
pins prove it); the organ modules are thin re-exports now. Point
arguments are ``(x, y, z)`` float triples; ``seed`` is a u32.
Grid functions return ``(h, w)`` float32 arrays.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "grid_fbm",
    "grid_hash2",
    "grid_perlin",
    "grid_permutation_table",
    "grid_value_noise",
    "grid_warp_field",
    "grid_worley",
    "point_cell_hash",
    "point_cellular",
    "point_fbm",
    "point_pcg3d",
    "point_perlin",
    "point_value",
    "point_worley",
]

_M32 = 0xFFFFFFFF
_F32_MAX = np.float32(4294967295.0)  # u32::MAX as f32 rounds up to 2**32
_F32_INF = np.float32(3.4028235e38)  # f32::MAX


# ── point engine (textures; texel noise.rs) ───────────────────────

def _f32(v: float) -> np.float32:
    return np.float32(v)


def _u32(v: int) -> int:
    return int(v) & _M32


def point_pcg3d(x: int, y: int, z: int) -> tuple[int, int, int]:
    """PCG3D hash (Jarzynski & Olano 2020), u32 wrapping."""
    v0 = _u32(_u32(x) * 1664525 + 1013904223)
    v1 = _u32(_u32(y) * 1664525 + 1013904223)
    v2 = _u32(_u32(z) * 1664525 + 1013904223)
    v0 = _u32(v0 + _u32(v1 * v2))
    v1 = _u32(v1 + _u32(v2 * v0))
    v2 = _u32(v2 + _u32(v0 * v1))
    v0 ^= v0 >> 16
    v1 ^= v1 >> 16
    v2 ^= v2 >> 16
    v0 = _u32(v0 + _u32(v1 * v2))
    v1 = _u32(v1 + _u32(v2 * v0))
    v2 = _u32(v2 + _u32(v0 * v1))
    return (v0, v1, v2)


def _cell(p: float) -> tuple[int, np.float32]:
    f = np.floor(_f32(p))
    return int(f), _f32(_f32(p) - f)


def _hash01(x: int, y: int, z: int, seed: int) -> np.float32:
    h = point_pcg3d(_u32(x) ^ _u32(seed * 0x9E3779B9), _u32(y), _u32(z))[0]
    return _f32(_f32(h) / _F32_MAX)


def _smooth(t: np.float32) -> np.float32:
    return _f32(t * t * t * (t * (t * _f32(6.0) - _f32(15.0)) + _f32(10.0)))


def _lerp(a: np.float32, b: np.float32, t: np.float32) -> np.float32:
    return _f32(a + (b - a) * t)


def point_value(p: tuple[float, float, float], seed: int) -> float:
    """Value noise in 0..1."""
    ix, fx = _cell(p[0])
    iy, fy = _cell(p[1])
    iz, fz = _cell(p[2])
    sx, sy, sz = _smooth(fx), _smooth(fy), _smooth(fz)

    def h(dx: int, dy: int, dz: int) -> np.float32:
        return _hash01(ix + dx, iy + dy, iz + dz, seed)

    return float(_lerp(
        _lerp(_lerp(h(0, 0, 0), h(1, 0, 0), sx),
              _lerp(h(0, 1, 0), h(1, 1, 0), sx), sy),
        _lerp(_lerp(h(0, 0, 1), h(1, 0, 1), sx),
              _lerp(h(0, 1, 1), h(1, 1, 1), sx), sy),
        sz))


def _grad(x: int, y: int, z: int, seed: int,
          f: tuple[np.float32, np.float32, np.float32]) -> np.float32:
    h = point_pcg3d(_u32(x) ^ _u32(seed * 0x85EBCA6B), _u32(y), _u32(z))[1] % 12
    pairs = [
        (f[0], f[1]), (-f[0], f[1]), (f[0], -f[1]), (-f[0], -f[1]),
        (f[0], f[2]), (-f[0], f[2]), (f[0], -f[2]), (-f[0], -f[2]),
        (f[1], f[2]), (-f[1], f[2]), (f[1], -f[2]), (-f[1], -f[2]),
    ]
    a, b = pairs[h]
    return _f32(a + b)


def point_perlin(p: tuple[float, float, float], seed: int) -> float:
    """Gradient (Perlin) noise in about 0..1."""
    ix, fx = _cell(p[0])
    iy, fy = _cell(p[1])
    iz, fz = _cell(p[2])
    sx, sy, sz = _smooth(fx), _smooth(fy), _smooth(fz)

    def g(dx: int, dy: int, dz: int) -> np.float32:
        return _grad(ix + dx, iy + dy, iz + dz, seed,
                     (_f32(fx - _f32(dx)), _f32(fy - _f32(dy)),
                      _f32(fz - _f32(dz))))

    v = _lerp(
        _lerp(_lerp(g(0, 0, 0), g(1, 0, 0), sx),
              _lerp(g(0, 1, 0), g(1, 1, 0), sx), sy),
        _lerp(_lerp(g(0, 0, 1), g(1, 0, 1), sx),
              _lerp(g(0, 1, 1), g(1, 1, 1), sx), sy),
        sz)
    return float(min(max(_f32(v * _f32(0.5) + _f32(0.5)),
                         _f32(0.0)), _f32(1.0)))


def point_cellular(p: tuple[float, float, float], seed: int
                   ) -> tuple[float, float, float]:
    """Cellular noise: (f1, f2, id) — distances to the nearest and
    second-nearest feature points and a random id of the nearest
    point's cell."""
    ix, fx = _cell(p[0])
    iy, fy = _cell(p[1])
    iz, fz = _cell(p[2])
    f1 = _F32_INF
    f2 = _F32_INF
    ident = 0
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                h = point_pcg3d(_u32(ix + dx) ^ _u32(seed * 0xC2B2AE35),
                                _u32(iy + dy), _u32(iz + dz))
                o = [_f32(_f32(v) / _F32_MAX) for v in h]
                d = [_f32(_f32(dx) + o[0] - fx),
                     _f32(_f32(dy) + o[1] - fy),
                     _f32(_f32(dz) + o[2] - fz)]
                d2 = _f32(_f32(d[0] * d[0]) + _f32(d[1] * d[1])
                          + _f32(d[2] * d[2]))
                if d2 < f1:
                    f2, f1, ident = f1, d2, h[0] ^ h[1]
                elif d2 < f2:
                    f2 = d2
    f1s = min(_f32(np.sqrt(f1)), _f32(1.0))
    f2s = min(_f32(np.sqrt(f2)), _f32(1.0))
    return (float(f1s), float(f2s), float(_f32(_f32(ident) / _F32_MAX)))


def point_worley(p: tuple[float, float, float], seed: int) -> float:
    """Worley noise: distance to the nearest feature point."""
    return point_cellular(p, seed)[0]


def point_cell_hash(x: int, y: int, z: int, seed: int) -> float:
    """Random 0..1 value of an integer lattice cell."""
    return float(_hash01(x, y, z, seed))


def point_fbm(p: tuple[float, float, float], octaves: int, seed: int,
              base) -> float:
    """Fractal sum of ``octaves`` layers of ``base`` noise,
    normalised to 0..1."""
    total = _f32(0.0)
    amp = _f32(0.5)
    norm = _f32(0.0)
    q = (_f32(p[0]), _f32(p[1]), _f32(p[2]))
    for o in range(max(1, min(12, octaves))):
        total = _f32(total + _f32(amp * _f32(base(q, _u32(seed + o * 31)))))
        norm = _f32(norm + amp)
        q = tuple(_f32(_f32(v * _f32(2.0)) + _f32(17.13)) for v in q)
        amp = _f32(amp * _f32(0.5))
    return float(_f32(total / norm))


# ── grid engine (material; nodes noise.rs / noise3d.rs) ───────────

_GRID_GRADIENTS = np.array([
    [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
], dtype=np.float64)


def _splitmix64(state: int) -> int:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def grid_permutation_table(seed: int) -> np.ndarray:
    """A 512-entry permutation table, deterministic per seed."""
    table = np.arange(256, dtype=np.uint8)
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(255, 0, -1):
        state = _splitmix64(state)
        j = state % (i + 1)
        table[i], table[j] = table[j], table[i]
    return np.concatenate([table, table]).astype(np.int64)


def grid_hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
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


def _grid_fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _grid_lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def grid_perlin(w: int, h: int, cells: float, seed: int,
                tiling: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """Tileable Perlin gradient noise in [0, 1], shape (h, w)."""
    perm = grid_permutation_table(seed)
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
    g00 = _GRID_GRADIENTS[perm[(perm[xi] + yi) % 256] % 8]
    g10 = _GRID_GRADIENTS[perm[(perm[xi1] + yi) % 256] % 8]
    g01 = _GRID_GRADIENTS[perm[(perm[xi] + yi1) % 256] % 8]
    g11 = _GRID_GRADIENTS[perm[(perm[xi1] + yi1) % 256] % 8]
    d00 = g00[..., 0] * xf + g00[..., 1] * yf
    d10 = g10[..., 0] * (xf - 1.0) + g10[..., 1] * yf
    d01 = g01[..., 0] * xf + g01[..., 1] * (yf - 1.0)
    d11 = g11[..., 0] * (xf - 1.0) + g11[..., 1] * (yf - 1.0)
    u = _grid_fade(xf)
    v = _grid_fade(yf)
    value = _grid_lerp(_grid_lerp(d00, d10, u), _grid_lerp(d01, d11, u), v)
    return ((value / np.sqrt(2.0)) * 0.5 + 0.5).astype(np.float32)


def grid_value_noise(w: int, h: int, cells: float, seed: int,
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
    v00 = grid_hash2(xi, yi, seed)
    v10 = grid_hash2(xi1, yi, seed)
    v01 = grid_hash2(xi, yi1, seed)
    v11 = grid_hash2(xi1, yi1, seed)
    u = _grid_fade(xf)
    v = _grid_fade(yf)
    return _grid_lerp(_grid_lerp(v00, v10, u),
                      _grid_lerp(v01, v11, u), v).astype(np.float32)


def grid_fbm(w: int, h: int, cells: float, octaves: int, seed: int,
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
            layer = grid_value_noise(w, h, freq, seed + octave * 7919, tiling)
        else:
            layer = grid_perlin(w, h, freq, seed + octave * 7919, tiling)
        total += amplitude * layer.astype(np.float64)
        norm += amplitude
        amplitude *= gain
        freq *= lacunarity
    return (total / norm).astype(np.float32)


def grid_worley(w: int, h: int, cells: float, seed: int,
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
            px = grid_hash2(nx, ny, seed)
            py = grid_hash2(nx, ny, seed + 104729)
            ddx = nx + px - xx
            ddy = ny + py - yy
            dist = np.sqrt(ddx * ddx + ddy * ddy)
            nearer = dist < f1
            f2 = np.where(nearer, f1, np.minimum(f2, dist))
            f1 = np.where(nearer, dist, f1)
            cell_id = np.where(nearer, grid_hash2(nx, ny, seed + 31), cell_id)
    if mode == "f1":
        out = np.clip(f1, 0.0, 1.0)
    elif mode == "f2":
        out = np.clip(f2, 0.0, 1.0)
    elif mode == "cell":
        out = cell_id
    else:
        out = np.clip(f2 - f1, 0.0, 1.0)
    return out.astype(np.float32)


def grid_warp_field(w: int, h: int, cells: float, seed: int,
                    tiling: tuple[bool, bool] = (True, True)
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Two Perlin fields (dx, dy) in [-1, 1] for domain warping."""
    dx = grid_perlin(w, h, cells, seed, tiling).astype(np.float64) * 2.0 - 1.0
    dy = grid_perlin(w, h, cells, seed + 4099,
                     tiling).astype(np.float64) * 2.0 - 1.0
    return dx, dy
