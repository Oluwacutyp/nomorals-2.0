"""SoundCraft automation — parameter lanes (lane 107).

Ported from storytold/soundcraft's ``crates/model`` automation and
``crates/mix`` automation: a lane is a list of (sample, value)
points; the value at any sample is the linear interpolation
between the surrounding points, clamped at the ends. Sample-accurate:
the mixer evaluates the lane per sample with ``numpy.interp``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["AutomationLane"]


class AutomationLane:
    """One automated parameter: points of (sample, value)."""

    def __init__(self, param: str, points: list[tuple[int, float]]):
        if not points:
            raise ValueError("automation lane needs at least one point")
        self.param = param
        self.points = sorted((int(s), float(v)) for s, v in points)

    def value_at(self, sample: int) -> float:
        points = self.points
        if sample <= points[0][0]:
            return points[0][1]
        if sample >= points[-1][0]:
            return points[-1][1]
        for (s0, v0), (s1, v1) in zip(points, points[1:], strict=False):
            if s0 <= sample <= s1:
                if s1 == s0:
                    return v1
                t = (sample - s0) / (s1 - s0)
                return v0 + t * (v1 - v0)
        return points[-1][1]

    def values(self, start: int, count: int) -> np.ndarray:
        """Sample-accurate values for ``count`` samples from ``start``."""
        xs = np.asarray([p[0] for p in self.points], dtype=np.float64)
        ys = np.asarray([p[1] for p in self.points], dtype=np.float64)
        query = np.arange(start, start + count, dtype=np.float64)
        return np.interp(query, xs, ys)

    def to_list(self) -> list[list[float]]:
        return [[float(s), float(v)] for s, v in self.points]

    @classmethod
    def from_list(cls, param: str,
                  raw: list[list[float]]) -> AutomationLane:
        return cls(param, [(int(s), float(v)) for s, v in raw])
