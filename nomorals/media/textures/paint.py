"""Brush engine (TextureCraft port, census 3A.15).

Strokes, stamps and fills on a layer's tile image (upstream
``crates/paint``). A stamp is a soft round brush: coverage falls off
from the centre by hardness, and the stamp blends its colour over
the texels it touches by coverage x opacity (normal blend). A
stroke walks its path in steps of ``size * spacing`` and stamps at
each step. Fills write rectangles and procedural patterns.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from .patterns import Procedural
from .patterns import procedural as _procedural
from .tiles import TileImage


@dataclass
class Brush:
    """A soft round brush."""

    size: float = 16.0
    hardness: float = 0.5
    opacity: float = 1.0
    spacing: float = 0.25

    def falloff(self, t: float) -> float:
        """Coverage at normalised radius ``t`` (0 centre, 1 edge)."""
        if t >= 1.0:
            return 0.0
        h = min(1.0, max(0.0, self.hardness))
        # Hardness 1 = a hard disc; 0 = a linear cone.
        inner = h
        if t <= inner:
            return 1.0
        return max(0.0, 1.0 - (t - inner) / max(1e-6, 1.0 - inner))


def stamp(img: TileImage, cx: float, cy: float, brush: Brush,
          color: Sequence[int]) -> None:
    """Stamp the brush once at (cx, cy). ``color`` is per-component
    u16 values; for a paint image the last component is alpha."""
    radius = brush.size / 2.0
    if radius <= 0.0:
        return
    x0 = max(0, int(math.floor(cx - radius)))
    y0 = max(0, int(math.floor(cy - radius)))
    x1 = min(img.width - 1, int(math.ceil(cx + radius)))
    y1 = min(img.height - 1, int(math.ceil(cy + radius)))
    comps = img.comps
    vals = [int(v) & 0xFFFF for v in color]
    vals = (vals + [0] * comps)[:comps]
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            dx = (x + 0.5 - cx) / radius
            dy = (y + 0.5 - cy) / radius
            t = math.sqrt(dx * dx + dy * dy)
            cov = brush.falloff(t) * brush.opacity
            if cov <= 0.0:
                continue
            dst = img.get(x, y)
            out = []
            for c in range(comps):
                if c == comps - 1 and img.comps > 1 and \
                        len(color) == img.comps:
                    # Alpha component: coverage accumulates as a union.
                    a0 = dst[c] / 65535.0
                    a1 = vals[c] / 65535.0 * cov
                    out.append(int((1.0 - (1.0 - a0) * (1.0 - a1))
                                   * 65535.0 + 0.5))
                else:
                    out.append(int(dst[c] * (1.0 - cov)
                                   + vals[c] * cov + 0.5))
            img.set(x, y, out)


def stroke(img: TileImage, path: Sequence[tuple[float, float]],
           brush: Brush, color: Sequence[int]) -> None:
    """Paint a stroke along ``path`` (a list of (x, y) points)."""
    if not path:
        return
    step = max(1.0, brush.size * brush.spacing)
    stamp(img, path[0][0], path[0][1], brush, color)
    for (x0, y0), (x1, y1) in pairwise(path):
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(dist / step))
        for i in range(1, n + 1):
            t = i / n
            stamp(img, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, brush, color)


def fill_rect(img: TileImage, x: int, y: int, w: int, h: int,
              color: Sequence[int]) -> None:
    """Fill a rectangle with ``color`` (u16 per component)."""
    vals = [int(v) & 0xFFFF for v in color]
    vals = (vals + [0] * img.comps)[: img.comps]
    for yy in range(max(0, y), min(img.height, y + h)):
        for xx in range(max(0, x), min(img.width, x + w)):
            img.set(xx, yy, vals)


def fill_pattern(img: TileImage, pr: Procedural, u0: float = 0.0,
                 v0: float = 0.0, scale: float = 1.0) -> None:
    """Fill the image with a procedural pattern (UV 0..1 across the
    image, offset by (u0, v0), scaled by ``scale``)."""
    for y in range(img.height):
        for x in range(img.width):
            v = _procedural(pr, (u0 + x / img.width) * scale,
                            (v0 + y / img.height) * scale)
            u = int(v * 65535.0 + 0.5)
            img.set(x, y, [u] * img.comps)
