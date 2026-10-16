"""Tier 1 — Motion / camera as structured params.

A camera move is data, not prose: kind (dolly/pan/orbit/crane/push-in/
pull-out/tilt/static), direction, speed, start/end framing, easing.
:func:`camera_plan` validates a move and expands it into per-second
keyframe samples (first / intermediate / last) the renderer can
consume. A :class:`MotionBrush` paints motion onto a region: region
mask + direction + strength, applied per frame by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

MOVES = ("dolly", "pan", "orbit", "crane", "push-in", "pull-out", "tilt", "static")
DIRECTIONS = ("in", "out", "left", "right", "up", "down", "cw", "ccw", "none")
EASINGS = ("linear", "ease-in", "ease-out", "ease-in-out")


@dataclass
class CameraMove:
    kind: str
    direction: str = "none"
    speed: float = 1.0  # 0.1..3, 1 = normal
    duration_s: float = 2.0
    start_frame: str = "wide"
    end_frame: str = "medium"
    easing: str = "ease-in-out"

    def __post_init__(self) -> None:
        if self.kind not in MOVES:
            raise ValueError(f"unknown camera move {self.kind!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {self.direction!r}")
        if self.easing not in EASINGS:
            raise ValueError(f"unknown easing {self.easing!r}")
        if not 0.1 <= self.speed <= 3.0:
            raise ValueError("speed must be 0.1..3")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "direction": self.direction, "speed": self.speed,
                "duration_s": self.duration_s, "start_frame": self.start_frame,
                "end_frame": self.end_frame, "easing": self.easing}


@dataclass
class MotionBrush:
    region: str  # mask kind or geometric descriptor, e.g. "subject" / "rect:0,0,0.5,1"
    direction: str = "right"
    strength: float = 0.5  # 0..1

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {self.direction!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be 0..1")

    def to_dict(self) -> dict[str, Any]:
        return {"region": self.region, "direction": self.direction,
                "strength": self.strength}


def _ease(easing: str, t: float) -> float:
    if easing == "linear":
        return t
    if easing == "ease-in":
        return t * t
    if easing == "ease-out":
        return 1.0 - (1.0 - t) ** 2
    return t * t * (3.0 - 2.0 * t)  # ease-in-out


def camera_plan(move: CameraMove, *, fps: int = 24) -> dict[str, Any]:
    """Expand a move into first/intermediate/last keyframe samples.

    ``progress`` 0..1 is the eased move amount at each sampled frame;
    ``zoom`` is the derived scale factor (push-in grows, pull-out
    shrinks, others stay 1). First sample is always progress 0, last
    is always progress 1 — keyframes first/last/intermediate.
    """
    n = max(2, int(round(move.duration_s * fps)))
    samples = []
    for i in range(n):
        t = i / (n - 1)
        p = _ease(move.easing, t)
        if move.kind == "push-in":
            zoom = 1.0 + 0.25 * p * move.speed
        elif move.kind == "pull-out":
            zoom = 1.0 - 0.20 * p * min(move.speed, 1.5)
        else:
            zoom = 1.0
        samples.append({"frame": i, "t_s": round(i / fps, 3),
                        "progress": round(p, 4), "zoom": round(zoom, 4)})
    return {"move": move.to_dict(), "fps": fps,
            "keyframes": [samples[0], samples[len(samples) // 2], samples[-1]],
            "samples": samples}


def keyframe_plan(moves: list[CameraMove], *, fps: int = 24) -> list[dict[str, Any]]:
    """Concatenate plans for a sequence of moves (a shot list)."""
    return [camera_plan(m, fps=fps) for m in moves]
