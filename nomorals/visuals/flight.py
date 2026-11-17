"""The Bezier flight: the buddy's smooth flight to a point, as data.

farzaa's `animateBezierFlightArc` (OverlayWindow.swift:495) is the
reference: a quadratic Bezier, duration scaled with distance,
smoothstep easing, the triangle rotated to the curve tangent, a
scale pulse mid-flight. clickyX computes the same arc (its control
point is a perpendicular offset of distance x 0.3; farzaa lifts the
midpoint straight up by min(distance x 0.2, 80) px).

Devon's flight is better than both in one way that matters: it is
pure data. :func:`flight_frames` takes a start and an end in
scene-spec (0..1) coordinates and returns the whole flight — every
frame's time, position, tangent angle and scale — as plain dicts.
The live HTML page interpolates between the frames; the code-video
rung can sequence them; a test can assert them. Same start, same
end, same frames, every time — no timer, no randomness, no window
needed to prove the motion is right.

The math runs in pixels (against the frame size, default 1280x720)
and converts back to 0..1, so the arc height and the duration mean
the same thing at any frame size.
"""

from __future__ import annotations

import math

__all__ = ["flight_duration", "flight_frames"]

#: Duration floor/ceiling in seconds (farzaa: clamp(dist/800, 0.6, 1.4)).
_MIN_DURATION_S = 0.6
_MAX_DURATION_S = 1.4
#: Pixels per second of flight (farzaa: distance / 800).
_SPEED_PX_PER_S = 800.0
#: Arc lift: min(distance x 0.2, 80) px (farzaa), applied along the
#: perpendicular so the arc works in any direction (clickyX's fix
#: for the vertical-lift case, where "up" is along the flight).
_ARC_FRACTION = 0.2
_MAX_ARC_PX = 80.0
#: Mid-flight scale pulse (farzaa: 1 + 0.3 * sin(pi * progress)).
_PULSE = 0.3


def _px(start: tuple[float, float], end: tuple[float, float],
        width: int, height: int) -> tuple[float, float, float, float]:
    return (start[0] * width, start[1] * height,
            end[0] * width, end[1] * height)


def flight_duration(start: tuple[float, float],
                    end: tuple[float, float], *,
                    width: int = 1280, height: int = 720) -> float:
    """Seconds for a flight: distance-scaled, clamped to 0.6..1.4s."""
    x1, y1, x2, y2 = _px(start, end, width, height)
    distance = math.hypot(x2 - x1, y2 - y1)
    return min(_MAX_DURATION_S,
               max(_MIN_DURATION_S, distance / _SPEED_PX_PER_S))


def _smoothstep(t: float) -> float:
    return 3 * t * t - 2 * t * t * t


def flight_frames(start: tuple[float, float],
                  end: tuple[float, float], *,
                  width: int = 1280, height: int = 720,
                  fps: int = 60) -> list[dict[str, float]]:
    """The whole flight as frames: ``t`` (seconds), ``x``/``y``
    (0..1), ``angle`` (degrees, tangent + 90deg — the glyph points
    up at 0deg), ``scale`` (1 at both ends, pulsing mid-flight).

    Pure and deterministic: the same start and end always produce
    the same frames.
    """
    x1, y1, x2, y2 = _px(start, end, width, height)
    dx, dy = x2 - x1, y2 - y1
    distance = math.hypot(dx, dy)
    duration = flight_duration(start, end, width=width, height=height)
    lift = min(distance * _ARC_FRACTION, _MAX_ARC_PX)
    # Perpendicular unit vector — (dy, -dx) lifts "up" on screen
    # for a left-to-right flight (screen y grows downward).
    perp = ((dy / distance, -dx / distance) if distance > 0
            else (0.0, -1.0))
    cx = (x1 + x2) / 2 + perp[0] * lift
    cy = (y1 + y2) / 2 + perp[1] * lift
    steps = max(1, int(round(duration * fps)))
    frames: list[dict[str, float]] = []
    for i in range(steps + 1):
        progress = i / steps
        eased = _smoothstep(progress)
        # Quadratic Bezier B(t) and its derivative B'(t).
        bx = ((1 - eased) ** 2 * x1 + 2 * (1 - eased) * eased * cx
              + eased ** 2 * x2)
        by = ((1 - eased) ** 2 * y1 + 2 * (1 - eased) * eased * cy
              + eased ** 2 * y2)
        dbx = 2 * (1 - eased) * (cx - x1) + 2 * eased * (x2 - cx)
        dby = 2 * (1 - eased) * (cy - y1) + 2 * eased * (y2 - cy)
        angle = math.degrees(math.atan2(dby, dbx)) + 90.0
        frames.append({
            "t": round(progress * duration, 6),
            "x": round(bx / width, 6),
            "y": round(by / height, 6),
            "angle": round(angle, 3),
            "scale": round(1.0 + _PULSE * math.sin(math.pi * progress), 6),
        })
    return frames
