"""UV padding (TextureCraft port, census 3A.15).

Faithful port of upstream ``crates/texel/src/dilate.rs`` (commit
37ad2112): a breadth-first dilation map that fills uncovered texels
from their nearest covered texel. ``NONE`` (``u32::MAX``) marks a
texel with no source. The block-bounded variant limits neighbours to
the same ``bsz`` square block.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TypeVar

NONE: int = 0xFFFFFFFF

T = TypeVar("T")


def _neighbours(i: int, w: int, h: int, bsz: tuple[int, int]):
    """The 8-neighbours of texel ``i`` inside the image and inside its
    ``bsz`` square block."""
    wu = max(1, w)
    x, y = i % wu, i // wu
    left = x > 0 and x % bsz[0] != 0
    right = x + 1 < w and (x + 1) % bsz[0] != 0
    up = y > 0 and y % bsz[1] != 0
    down = y + 1 < h and (y + 1) % bsz[1] != 0
    for dx, dy, ok in ((-1, 0, left), (1, 0, right), (0, -1, up),
                       (0, 1, down), (-1, -1, left and up),
                       (1, -1, right and up), (-1, 1, left and down),
                       (1, 1, right and down)):
        if ok:
            yield i + dy * w + dx


class DilationMap:
    """``src[i]`` is the index of the covered texel nearest to ``i``
    (or ``NONE`` when none is within ``max_dist``)."""

    __slots__ = ("width", "height", "src")

    def __init__(self, covered: Sequence[bool], width: int, height: int,
                 max_dist: int | None = None,
                 block: tuple[int, int] | None = None) -> None:
        if len(covered) != width * height:
            raise ValueError("covered mask does not match dimensions")
        self.width = int(width)
        self.height = int(height)
        bsz = block if block is not None else (width, height)
        src = [NONE] * (width * height)
        frontier: deque[int] = deque()
        for i, c in enumerate(covered):
            if c:
                src[i] = i
                frontier.append(i)
        dist = 0
        while frontier and (max_dist is None or dist < max_dist):
            nxt: deque[int] = deque()
            for i in frontier:
                for j in _neighbours(i, width, height, bsz):
                    if src[j] == NONE:
                        src[j] = src[i]
                        nxt.append(j)
            frontier = nxt
            dist += 1
        self.src = src

    def source(self, x: int, y: int) -> tuple[int, int] | None:
        """The source texel of (x, y), if any."""
        if x >= self.width or y >= self.height or x < 0 or y < 0:
            return None
        s = self.src[y * self.width + x]
        if s == NONE:
            return None
        return (s % self.width, s // self.width)

    def apply(self, data: list, comps: int) -> None:
        """Copy padded values in place: ``data`` holds ``comps``
        components per texel. Texels with no source keep their value."""
        for i, s in enumerate(self.src):
            if s in (i, NONE):
                continue
            for c in range(comps):
                data[i * comps + c] = data[s * comps + c]
