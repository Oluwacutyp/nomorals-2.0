"""Procedural patterns (TextureCraft port, census 3A.15).

Faithful port of upstream ``crates/compose/src/pattern.rs``
``raw()`` / ``shape()`` / ``procedural()`` (commit 37ad2112) and of
``ProceduralKind`` (upstream ``crates/core/src/layer.rs``): all 23
patterns, each a deterministic function of position and seed.
``raw()`` evaluates the 3D pattern at a point; ``procedural()``
samples a 2D UV slice of it and shapes the result with contrast /
balance / invert (upstream ``Procedural`` defaults: scale 4.0,
seed 1, contrast 1.0, balance 0.0, invert false).

Noise inputs go through :mod:`nomorals.media.textures.noise`, whose
float32 arithmetic matches the Rust f32 code bit-for-bit; the pattern
arithmetic itself is plain float math, well inside the 2e-6 parity
tolerance of the pinned upstream values.
"""
from __future__ import annotations

import math
from enum import StrEnum

from . import noise

_LABELS = [
    ("clouds", "Clouds"),
    ("perlin", "Perlin noise"),
    ("noise", "Fine noise"),
    ("cells", "Cells"),
    ("grunge", "Grunge"),
    ("scratches", "Scratches"),
    ("checker", "Checker"),
    ("stripes", "Stripes"),
    ("gradient", "Gradient"),
    ("wood", "Wood grain"),
    ("bricks", "Bricks"),
    ("tiles", "Tiles"),
    ("weave", "Fabric weave"),
    ("twill", "Twill weave"),
    ("leather", "Leather grain"),
    ("hammered", "Hammered"),
    ("brushed", "Brushed lines"),
    ("marble", "Marble veins"),
    ("voronoi", "Voronoi"),
    ("streaks", "Streaks"),
    ("smudges", "Fingerprints"),
    ("pores", "Pores"),
    ("chips", "Paint chips"),
]


class ProceduralKind(StrEnum):
    """One of the 23 procedural patterns (upstream ``ProceduralKind``)."""

    CLOUDS = "clouds"
    PERLIN = "perlin"
    NOISE = "noise"
    CELLS = "cells"
    GRUNGE = "grunge"
    SCRATCHES = "scratches"
    CHECKER = "checker"
    STRIPES = "stripes"
    GRADIENT = "gradient"
    WOOD = "wood"
    BRICKS = "bricks"
    TILES = "tiles"
    WEAVE = "weave"
    TWILL = "twill"
    LEATHER = "leather"
    HAMMERED = "hammered"
    BRUSHED = "brushed"
    MARBLE = "marble"
    VORONOI = "voronoi"
    STREAKS = "streaks"
    SMUDGES = "smudges"
    PORES = "pores"
    CHIPS = "chips"

    @classmethod
    def all(cls) -> list[ProceduralKind]:
        return [cls(v) for v, _ in _LABELS]

    def label(self) -> str:
        return dict(_LABELS)[self.value]

    @classmethod
    def parse(cls, name: str) -> ProceduralKind | None:
        needle = name.lower().replace(" ", "").replace("_", "").replace("-", "")
        for v, label in _LABELS:
            if v.replace("_", "") == needle or \
                    label.lower().replace(" ", "") == needle:
                return cls(v)
        return None


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _rem(x: float, d: float) -> float:
    """Rust ``f32::rem_euclid`` for a positive divisor."""
    return x % d


def raw(kind: ProceduralKind, x: float, y: float, z: float,
        seed: int) -> float:
    """Raw pattern value at a 3D point (upstream ``raw()``)."""
    p = (x, y, z)
    k = kind.value
    if k == "clouds":
        return noise.fbm(p, 6, seed, noise.value)
    if k == "perlin":
        return noise.fbm(p, 4, seed, noise.perlin)
    if k == "noise":
        return noise.value((x * 16.0, y * 16.0, z * 16.0), seed)
    if k == "cells":
        return noise.worley(p, seed)
    if k == "grunge":
        n = noise.fbm(p, 6, seed, noise.perlin) * 0.7 + \
            noise.fbm((x * 3.7, y * 3.7, z * 3.7), 3, seed ^ 0x55,
                      noise.value) * 0.3
        return _clamp01((n - 0.45) * 4.0 + 0.5)
    if k == "scratches":
        n = noise.perlin((x * 0.15, y * 6.0, z * 6.0), seed)
        line = 1.0 - min(abs(n - 0.5) * 40.0, 1.0)
        gate = noise.value((x * 0.8, y * 0.8, z * 0.8), seed ^ 0x99)
        return line * _clamp01((gate - 0.45) * 4.0)
    if k == "checker":
        c = (math.floor(x), math.floor(y), math.floor(z))
        return 1.0 if (c[0] + c[1] + c[2]) % 2 < 1.0 else 0.0
    if k == "stripes":
        return 1.0 if _rem(x, 1.0) < 0.5 else 0.0
    if k == "gradient":
        return _rem(x, 1.0)
    if k == "wood":
        warp = noise.fbm((x * 0.18, y * 0.6, z), 3, seed, noise.perlin)
        t = _rem((y + (warp - 0.5) * 0.9) * 5.0, 1.0)
        ring = 1.0 - smoothstep(0.45, 1.0, t)
        fiber = noise.value((x * 2.0, y * 40.0, z), seed ^ 0x3C)
        tone = noise.fbm((x * 0.6, y * 3.0, z), 2, seed ^ 0x71, noise.value)
        return _clamp01(ring * 0.6 + fiber * 0.25 + tone * 0.15)
    if k == "bricks":
        row = math.floor(y * 2.0)
        xx = x + (0.5 if row % 2 == 1 else 0.0)
        col = math.floor(xx)
        fx = _rem(xx, 1.0)
        fy = _rem(y * 2.0, 1.0)
        d = min(fx, 1.0 - fx, min(fy, 1.0 - fy) * 0.5)
        tone = noise.cell_hash(col, row, 0, seed) * 0.6 + \
            noise.value((x * 9.0, y * 9.0, z * 9.0), seed) * 0.4
        return smoothstep(0.02, 0.045, d) * (0.85 + 0.15 * tone)
    if k == "tiles":
        c = (math.floor(x), math.floor(y), math.floor(z))
        f = (x - c[0], y - c[1], z - c[2])
        d = min(f[0], 1.0 - f[0], min(f[1], 1.0 - f[1]))
        return smoothstep(0.015, 0.035, d) * (
            0.9 + 0.1 * noise.cell_hash(c[0], c[1], 0, seed))
    if k in ("weave", "twill"):
        twill = k == "twill"
        c = (math.floor(x * 2.0), math.floor(y * 2.0), math.floor(z * 2.0))
        f = (x * 2.0 - c[0], y * 2.0 - c[1], z * 2.0 - c[2])
        pi = math.pi
        period, warp_cells = (4, 2) if twill else (2, 1)
        warp_on_top = (c[0] + c[1]) % period < warp_cells
        across, along = (f[1], f[0]) if warp_on_top else (f[0], f[1])
        thread = max(math.sin(pi * across), 0.0) ** 0.5 * (
            0.8 + 0.2 * math.sin(pi * along))
        tone = 0.55 if (twill and not warp_on_top) else 1.0
        fiber = noise.value((x * 40.0, y * 40.0, z), seed)
        return _clamp01(thread * tone * (0.88 + 0.12 * fiber))
    if k == "leather":
        w = noise.perlin((x * 0.7, y * 0.7, z * 0.7), seed ^ 0x2D) - 0.5
        f1, f2, _ = noise.cellular((x + w * 0.4, y - w * 0.4, z), seed)
        crease = smoothstep(0.0, 0.16, f2 - f1)
        pebble = 1.0 - f1 * 0.5
        fine = noise.value((x * 7.0, y * 7.0, z * 7.0), seed ^ 0x13)
        return _clamp01(crease * pebble * (0.85 + 0.15 * fine))
    if k == "hammered":
        def dent(q: tuple[float, float, float], s: int) -> float:
            f1, _, _ = noise.cellular(q, s)
            return min(f1 * 1.25, 1.0) ** 2
        return _clamp01(dent(p, seed) * 0.75
                        + dent((x * 2.3 + 5.1, y * 2.3 + 5.1,
                                z * 2.3 + 5.1), seed ^ 0x5A) * 0.25)
    if k == "brushed":
        a = noise.value((x * 1.5, y * 60.0, z), seed)
        b = noise.value((x * 0.4, y * 18.0, z), seed ^ 0x77)
        return _clamp01(a * 0.6 + b * 0.4)
    if k == "marble":
        turb = noise.fbm(p, 5, seed, noise.perlin)
        veins = abs(math.sin((x + y * 0.5) * 2.5 + turb * 9.0)) ** 0.3
        cloud = noise.fbm((x * 2.0, y * 2.0, z * 2.0), 3, seed ^ 0x19,
                          noise.value)
        return _clamp01(veins * 0.85 + cloud * 0.15)
    if k == "voronoi":
        return noise.cellular(p, seed)[2]
    if k == "streaks":
        def drip(px: float, salt: int) -> float:
            cx = px * 6.0
            col = math.floor(cx)

            def h(key: int) -> float:
                return noise.cell_hash(col, key, salt, seed)

            width = 0.08 + 0.25 * h(0)
            across = 1.0 - smoothstep(
                width * 0.3, width,
                abs(cx - math.floor(cx) - 0.5 - (h(1) - 0.5) * 0.5))
            yy = y * (0.35 + 0.4 * h(2)) + h(3) * 7.0
            t = yy - math.floor(yy)
            on = 1.0 if noise.cell_hash(col, math.floor(yy), 3 + salt,
                                       seed) > 0.5 else 0.0
            return across * on * t ** 1.6 * (1.0 - smoothstep(0.9, 1.0, t))

        fine = noise.value((x * 37.0, y * 1.3, z), seed ^ 0x2B)
        return _clamp01(max(drip(x, 0), drip(x + 0.37, 1) * 0.8)
                        * (0.6 + 0.4 * fine))
    if k == "smudges":
        w = noise.perlin((x * 1.3, y * 1.3, z * 1.3), seed ^ 0x61) - 0.5
        q = (x * 1.15 + w * 0.3, y * 0.85 + w * 0.3, z)
        f1, _, ident = noise.cellular(q, seed)
        printed = (1.0 - smoothstep(0.18 + 0.1 * ident, 0.3 + 0.12 * ident,
                                    f1)) * (1.0 if ident > 0.4 else 0.0)
        ridges = 0.5 + 0.5 * math.sin(f1 * 190.0 + w * 12.0)
        smudge = _clamp01((noise.fbm((x * 1.7, y * 1.7, z * 1.7), 4,
                                     seed ^ 0x33, noise.value) - 0.55) * 3.0)
        return _clamp01(printed * ridges * 0.9 + smudge * 0.4)
    if k == "pores":
        f1, _, _ = noise.cellular(p, seed)
        g1, _, _ = noise.cellular((x * 2.7 + 13.0, y * 2.7 + 13.0,
                                   z * 2.7 + 13.0), seed ^ 0x6B)
        return min(smoothstep(0.08, 0.16, f1),
                   smoothstep(0.06, 0.12, g1))
    if k == "chips":
        f1, _, _ = noise.cellular((x * 2.0, y * 2.0, z * 2.0), seed ^ 0x45)
        n = noise.fbm(p, 5, seed, noise.perlin) + (f1 - 0.5) * 0.25
        return smoothstep(0.55, 0.58, n)
    raise ValueError(f"unknown procedural kind: {kind!r}")


class Procedural:
    """A procedural grayscale pattern (upstream ``Procedural``).

    Defaults match ``Procedural::new``: scale 4.0, seed 1,
    contrast 1.0, balance 0.0, invert False.
    """

    __slots__ = ("kind", "scale", "seed", "contrast", "balance", "invert")

    def __init__(self, kind: ProceduralKind, scale: float = 4.0,
                 seed: int = 1, contrast: float = 1.0,
                 balance: float = 0.0, invert: bool = False) -> None:
        self.kind = kind
        self.scale = float(scale)
        self.seed = int(seed)
        self.contrast = float(contrast)
        self.balance = float(balance)
        self.invert = bool(invert)

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "scale": self.scale,
                "seed": self.seed, "contrast": self.contrast,
                "balance": self.balance, "invert": self.invert}

    @classmethod
    def from_dict(cls, data: dict) -> Procedural:
        kind = ProceduralKind.parse(str(data.get("kind", "clouds")))
        if kind is None:
            raise ValueError(f"unknown procedural kind: {data.get('kind')!r}")
        return cls(kind, scale=float(data.get("scale", 4.0)),
                   seed=int(data.get("seed", 1)),
                   contrast=float(data.get("contrast", 1.0)),
                   balance=float(data.get("balance", 0.0)),
                   invert=bool(data.get("invert", False)))


def shape(pr: Procedural, v: float) -> float:
    """Contrast / balance / invert shaping (upstream ``shape()``)."""
    v = _clamp01((v - 0.5) * max(pr.contrast, 0.0) + 0.5 + pr.balance)
    return 1.0 - v if pr.invert else v


def procedural(pr: Procedural, u: float, v: float) -> float:
    """Sample a 2D UV slice of the 3D pattern and shape it
    (upstream ``procedural()`` with the default UV projection)."""
    s = max(pr.scale, 1e-4)
    return shape(pr, raw(pr.kind, u * s, v * s, 0.0, pr.seed))
