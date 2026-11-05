"""Rational frame time (Echelon Wave 1.2a — FilmCraft ``Tick`` pattern).

Edit math in integers, seconds only at the API edges. FilmCraft's rule
— "never use f64 seconds for edit math" — exists because float sums
drift: ten thousand cuts assembled from float durations land measurably
off the frame grid, and xfade offsets computed from accumulated floats
wander with them.

Devon's canonical model (Wave 1, D1) stores seconds in
:class:`~.timeline.Clip`/:class:`~.document.Document` dicts — that
storage is pinned by the round-trip corpus and does not change. What
changes here is the *arithmetic*: every planning computation that used
to accumulate floats (cut-grid plans, boundary folds, duration sums)
goes through :class:`FrameGrid`, whose values are integer frame counts
and exact :class:`fractions.Fraction` seconds. A frame count never
drifts; a sum of frame counts is exact at any length.

``FrameGrid(fps)`` accepts an int (24/25/30/60), a
:class:`~fractions.Fraction` (30000/1001 for NTSC), or a float that
names a standard rate (29.97 → 30000/1001, 59.94 → 60000/1001,
23.976 → 24000/1001); any other float is taken at face value as an
exact decimal fraction.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable

__all__ = [
    "FrameGrid",
    "frame_grid",
    "quantize_seconds",
    "sum_frame_durations",
]

#: float fps spellings that mean a rational broadcast rate.
_FLOAT_RATES: dict[float, Fraction] = {
    23.976: Fraction(24000, 1001),
    29.97: Fraction(30000, 1001),
    47.952: Fraction(48000, 1001),
    59.94: Fraction(60000, 1001),
    119.88: Fraction(120000, 1001),
}


def _fps_fraction(fps: Any) -> Fraction:
    if isinstance(fps, Fraction):
        out = fps
    elif isinstance(fps, bool):  # bool is an int subclass — never a rate
        raise ValueError(f"invalid fps {fps!r}")
    elif isinstance(fps, int):
        out = Fraction(fps, 1)
    elif isinstance(fps, float):
        out = _FLOAT_RATES.get(fps)
        if out is None:
            out = Fraction(str(fps))
    else:
        raise ValueError(f"invalid fps {fps!r}")
    if out <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return out


class FrameGrid:
    """The frame grid of one timeline: exact seconds ↔ frame counts."""

    __slots__ = ("_fps",)

    def __init__(self, fps: Any = 30):
        self._fps = _fps_fraction(fps)

    @property
    def fps(self) -> Fraction:
        return self._fps

    @property
    def fps_float(self) -> float:
        return float(self._fps)

    def frames(self, seconds: Any) -> int:
        """Seconds → nearest frame count (exact rational rounding)."""
        exact = Fraction(seconds) if not isinstance(seconds, Fraction) \
            else seconds
        return int(round(exact * self._fps))

    def seconds(self, frames: int) -> Fraction:
        """Frame count → exact seconds (a Fraction, never a float)."""
        return Fraction(int(frames), 1) / self._fps

    def quantize(self, seconds: Any) -> Fraction:
        """Snap seconds to the frame grid; exact Fraction out."""
        return self.seconds(self.frames(seconds))

    def quantize_float(self, seconds: Any) -> float:
        """Snap seconds to the frame grid; float out (API edges only)."""
        return float(self.quantize(seconds))

    def span_frames(self, start_s: Any, end_s: Any) -> int:
        """Whole frames in [start, end) on this grid."""
        return max(0, self.frames(end_s) - self.frames(start_s))

    def sum_durations(self, durations: Iterable[Any]) -> Fraction:
        """Exact total of frame-quantized durations, as Fraction seconds."""
        total = 0
        for d in durations:
            total += self.frames(d)
        return self.seconds(total)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FrameGrid(fps={self._fps})"


def frame_grid(fps: Any = 30) -> FrameGrid:
    """A :class:`FrameGrid` for ``fps`` (int, float, or Fraction)."""
    return FrameGrid(fps)


def quantize_seconds(seconds: Any, fps: Any = 30) -> Fraction:
    """One-shot: snap ``seconds`` to the ``fps`` frame grid (Fraction)."""
    return FrameGrid(fps).quantize(seconds)


def sum_frame_durations(durations: Iterable[Any], fps: Any = 30) -> Fraction:
    """One-shot exact frame-sum of ``durations`` at ``fps``."""
    return FrameGrid(fps).sum_durations(durations)
