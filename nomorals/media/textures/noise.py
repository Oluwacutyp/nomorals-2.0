"""Deterministic 3D noise (TextureCraft port, census 3A.15).

Faithful port of upstream ``crates/texel/src/noise.rs`` (commit
37ad2112): PCG3D integer hashing (Jarzynski & Olano 2020), value,
Perlin, Worley/cellular and fractal sums. Every arithmetic step is
done in IEEE float32 (``numpy.float32``) exactly as the Rust f32
code does it, so values match the upstream binary bit-for-bit;
inputs are rounded to f32 on entry, because a Rust caller passes
f32 coordinates.

Point arguments are ``(x, y, z)`` float triples; ``seed`` is a u32.
"""
from __future__ import annotations

import numpy as np

_M32 = 0xFFFFFFFF
_F32_MAX = np.float32(4294967295.0)  # u32::MAX as f32 rounds up to 2**32
_F32_INF = np.float32(3.4028235e38)  # f32::MAX


def _f32(v: float) -> np.float32:
    return np.float32(v)


def _u32(v: int) -> int:
    return int(v) & _M32


def pcg3d(x: int, y: int, z: int) -> tuple[int, int, int]:
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
    h = pcg3d(_u32(x) ^ _u32(seed * 0x9E3779B9), _u32(y), _u32(z))[0]
    return _f32(_f32(h) / _F32_MAX)


def _smooth(t: np.float32) -> np.float32:
    return _f32(t * t * t * (t * (t * _f32(6.0) - _f32(15.0)) + _f32(10.0)))


def _lerp(a: np.float32, b: np.float32, t: np.float32) -> np.float32:
    return _f32(a + (b - a) * t)


def value(p: tuple[float, float, float], seed: int) -> float:
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
    h = pcg3d(_u32(x) ^ _u32(seed * 0x85EBCA6B), _u32(y), _u32(z))[1] % 12
    pairs = [
        (f[0], f[1]), (-f[0], f[1]), (f[0], -f[1]), (-f[0], -f[1]),
        (f[0], f[2]), (-f[0], f[2]), (f[0], -f[2]), (-f[0], -f[2]),
        (f[1], f[2]), (-f[1], f[2]), (f[1], -f[2]), (-f[1], -f[2]),
    ]
    a, b = pairs[h]
    return _f32(a + b)


def perlin(p: tuple[float, float, float], seed: int) -> float:
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


def cellular(p: tuple[float, float, float], seed: int
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
                h = pcg3d(_u32(ix + dx) ^ _u32(seed * 0xC2B2AE35),
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


def worley(p: tuple[float, float, float], seed: int) -> float:
    """Worley noise: distance to the nearest feature point."""
    return cellular(p, seed)[0]


def cell_hash(x: int, y: int, z: int, seed: int) -> float:
    """Random 0..1 value of an integer lattice cell."""
    return float(_hash01(x, y, z, seed))


def fbm(p: tuple[float, float, float], octaves: int, seed: int,
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
