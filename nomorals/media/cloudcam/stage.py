"""Texture staging — the clone's per-frame Tick as pure data.

Pattern from storytold/cloudcamplugin (PATTERN ONLY): every
frame, the actor copies the ring's points into two textures the
Niagara renderer consumes — a position texture (one float4 per
grid cell) and a colour texture (one BGRA byte-quad per cell).
Devon stages the same two buffers, as bytes, with the receipt
naming what happened:

* Point ``i`` lands in texel ``i`` (row-major, ``id = x + y*W``).
* Texels with no point are hidden: position far away
  (``-300000`` on every axis, the clone's sentinel) and alpha 0.
* Points past the grid are dropped — counted in the receipt,
  never silently truncated the way the clone truncates them.
* Until the first frame arrives, the stage is the clone's flat
  gradient plane, marked ``placeholder=True`` — the renderer
  shows a plane, never a black void, and the receipt says so.

No GPU is touched here: staging produces the byte buffers and
the receipt; a renderer consumes them (see ``render.py`` for
the PNG projection, and the session's ``tick`` for the live
path). Positions are millimetres, colours BGRA bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .protocol import CloudCamPoint

__all__ = [
    "HIDDEN_POSITION",
    "StagedFrame",
    "frame_bounds",
    "placeholder_grid",
    "stage_frame",
]

#: Where hidden texels sit — far outside any real scene, the
#: clone's sentinel, kept verbatim so a staged buffer renders
#: exactly like the clone's.
HIDDEN_POSITION = -300000.0

_FLOAT = struct.Struct("<f")
_UBYTE = struct.Struct("<B")


@dataclass
class StagedFrame:
    """One staged grid: the two texture buffers + the receipt."""

    width: int
    height: int
    positions: bytes
    colors: bytes
    packed: int
    hidden: int
    dropped: int
    placeholder: bool
    point_count: int

    @property
    def cells(self) -> int:
        """Texels in the grid."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """The receipt: JSON-safe, no buffers."""
        return {
            "width": self.width,
            "height": self.height,
            "cells": self.cells,
            "packed": self.packed,
            "hidden": self.hidden,
            "dropped": self.dropped,
            "placeholder": self.placeholder,
            "point_count": self.point_count,
            "positions_bytes": len(self.positions),
            "colors_bytes": len(self.colors),
        }


def stage_frame(points: list[CloudCamPoint], width: int,
                height: int) -> StagedFrame:
    """Pack points into the position + colour buffers."""
    cells = int(width) * int(height)
    if cells < 1:
        raise ValueError("stage grid must have at least one cell")
    packed = min(len(points), cells)
    dropped = len(points) - packed
    positions = bytearray(cells * 4 * 4)
    colors = bytearray(cells * 4)
    for i in range(packed):
        p = points[i]
        struct.pack_into("<ffff", positions, i * 16,
                         float(p.x), float(p.y), float(p.z), 1.0)
        colors[i * 4 + 0] = p.b
        colors[i * 4 + 1] = p.g
        colors[i * 4 + 2] = p.r
        colors[i * 4 + 3] = p.a
    for i in range(packed, cells):
        struct.pack_into("<ffff", positions, i * 16,
                         HIDDEN_POSITION, HIDDEN_POSITION,
                         HIDDEN_POSITION, 1.0)
        # colours stay zero: alpha 0 hides the texel.
    return StagedFrame(width=int(width), height=int(height),
                       positions=bytes(positions), colors=bytes(colors),
                       packed=packed, hidden=cells - packed,
                       dropped=dropped, placeholder=False,
                       point_count=len(points))


def placeholder_grid(width: int, height: int) -> StagedFrame:
    """The flat gradient plane the clone renders until the first
    frame arrives: x ramps -500..500 across, y ramps -500..500
    down, z = 0; colours ramp red across, green down, alpha 10."""
    cells = int(width) * int(height)
    if cells < 1:
        raise ValueError("stage grid must have at least one cell")
    positions = bytearray(cells * 4 * 4)
    colors = bytearray(cells * 4)
    for y in range(int(height)):
        for x in range(int(width)):
            i = x + y * int(width)
            struct.pack_into(
                "<ffff", positions, i * 16,
                float(x) / int(width) * 1000 - 500,
                float(y) / int(height) * 1000 - 500,
                0.0, 1.0)
            colors[i * 4 + 0] = 0
            colors[i * 4 + 1] = int(float(y) / int(height) * 255)
            colors[i * 4 + 2] = int(float(x) / int(width) * 255)
            colors[i * 4 + 3] = 10
    return StagedFrame(width=int(width), height=int(height),
                       positions=bytes(positions), colors=bytes(colors),
                       packed=0, hidden=cells, dropped=0,
                       placeholder=True, point_count=0)


def frame_bounds(points: list[CloudCamPoint]) -> dict:
    """Min/max/centroid of a frame, millimetres. Empty frames
    report zeros, not an error — a frame with no points is a
    fact, not a failure."""
    if not points:
        return {"count": 0,
                "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0],
                "centroid": [0.0, 0.0, 0.0]}
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    n = float(len(points))
    return {"count": len(points),
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
            "centroid": [sum(xs) / n, sum(ys) / n, sum(zs) / n]}
