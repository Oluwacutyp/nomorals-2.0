"""PointStream — the galvo multi-object drawing algorithm.

Ported from ``pointstream.py`` in echelon/laser-asteroids (MIT, Copyright
(c) 2012 Brandon Thomas), translated to Python 3. One pass over the
objects is one frame: cache every object's first point, then per object
blank in (dark points), draw its points, blank out, and track (dark,
interpolated) to the next object. Objects marked ``destroy`` are culled
at the end of the pass.

Two changes from the original, both deliberate: the original's
``produce()`` was one infinite generator whose only error handling was
an infinite print loop — here a pass is a finite ``frame()`` and errors
propagate to the caller; and an empty object list yields one dark point
per pass instead of hanging ``read(n)`` forever.
"""
from __future__ import annotations

from collections.abc import Iterator

from .constants import (
    BLANK_SAMPLE_PTS,
    CMAX,
    SHOW_TRACKING_PATH,
    TRACKING_SAMPLE_PTS,
)
from .entities import Entity, Point

__all__ = ["PointStream"]

#: A dark point (laser off) at a position.
_BLANK = (0, 0, 0)


class PointStream:
    """Turns a list of entities into a stream of laser points."""

    def __init__(self, objects: list[Entity] | None = None) -> None:
        self.objects: list[Entity] = list(objects or [])
        self.stream = self.produce()

    def frame(self) -> list[Point]:
        """One full pass over the objects: exactly one frame of points."""
        return list(self._pass())

    def _pass(self) -> Iterator[Point]:
        # Cache the first points upfront: the galvos need to know where
        # the next object starts before they track over to it.
        for obj in self.objects:
            obj.cacheFirstPt()

        destroy: list[int] = []
        if not self.objects:
            yield (0, 0, 0, 0, 0)
        for i in range(len(self.objects)):
            cur = self.objects[i]
            nxt = self.objects[(i + 1) % len(self.objects)]

            if cur.skipDraw:
                continue
            if cur.destroy:
                destroy.append(i)

            # Blanking on the way in.
            if cur.doBlanking:
                p = cur.firstPt
                for _ in range(BLANK_SAMPLE_PTS):
                    yield (p[0], p[1], 0, 0, 0)

            # Draw the object.
            if not cur.drawn:
                yield cur.firstPt
                yield from cur.produce()

            # Blanking on the way out.
            if cur.doBlanking:
                p = cur.lastPt
                for _ in range(BLANK_SAMPLE_PTS):
                    yield (p[0], p[1], 0, 0, 0)

            # Track to the next object.
            last_x = cur.lastPt[0]
            last_y = cur.lastPt[1]
            x_diff = cur.lastPt[0] - nxt.firstPt[0]
            y_diff = cur.lastPt[1] - nxt.firstPt[1]
            mv = TRACKING_SAMPLE_PTS
            for j in range(mv):
                percent = j / float(mv)
                xb = int(last_x - x_diff * percent)
                yb = int(last_y - y_diff * percent)
                if SHOW_TRACKING_PATH:
                    yield (xb, yb, 0, CMAX, 0)
                else:
                    yield (xb, yb, 0, 0, 0)

        # Reset the drawn flags (the caching hack) and cull the dead.
        for obj in self.objects:
            obj.drawn = False
        destroy.sort(reverse=True)
        for i in destroy:
            self.objects.pop(i)

    def produce(self) -> Iterator[Point]:
        """The infinite point generator — one frame after another."""
        while True:
            yield from self._pass()

    def read(self, n: int) -> list[Point]:
        """Pull ``n`` points off the stream."""
        return [next(self.stream) for _ in range(n)]
