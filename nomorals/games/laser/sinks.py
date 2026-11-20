"""Laser output sinks — where a frame of points goes.

Devon's own abstraction, replacing the original's ``dac_thread`` (which
drove the Ether Dream DAC through the GPL ``daclib`` — carved out, never
ported). A sink takes points; the game renders into whatever sink the
caller hands it. The only honest sink for a live projector refuses by
name: no DAC driver exists in the repo and no projector hardware exists
here or on EC2.
"""
from __future__ import annotations

from typing import Protocol

from .entities import Point

__all__ = [
    "FrameSink",
    "LaserSink",
    "NullSink",
    "projector_sink",
    "projector_unavailable",
]


class LaserError(Exception):
    """A named laser failure. ``code`` is the refusal name."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def projector_unavailable() -> LaserError:
    """No laser projector is attached to this box — named, never faked."""
    return LaserError(
        "projector_unavailable",
        "no laser projector is attached: the game renders to a frame, "
        "a text board, or a PNG instead — a live DAC stream needs a "
        "projector and its driver on this machine.")


class LaserSink(Protocol):
    """Anything a frame of laser points can be written to."""

    def write(self, points: list[Point]) -> None: ...


class NullSink:
    """Discard points. The headless default."""

    def write(self, points: list[Point]) -> None:
        return None


class FrameSink:
    """Keep the latest frame in memory for inspection or rendering."""

    def __init__(self) -> None:
        self.frames: list[list[Point]] = []
        self.points_written = 0

    def write(self, points: list[Point]) -> None:
        self.frames.append(list(points))
        self.points_written += len(points)

    @property
    def latest(self) -> list[Point]:
        return self.frames[-1] if self.frames else []

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def projector_sink() -> LaserSink:
    """A sink for a live laser projector. Refuses by name: there is no
    DAC driver in the repo (the original's was GPL and carved out) and
    no projector on this machine."""
    raise projector_unavailable()
