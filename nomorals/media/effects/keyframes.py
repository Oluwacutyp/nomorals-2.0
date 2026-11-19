"""Property keyframes — EffectCraft ``keyframe`` crate's surface.

A ``PropertyTrack`` is one animatable property of one layer (opacity,
position_x, scale …): a sorted list of keyframes sampled through the
house easings, plus an optional safe value expression (see
``expressions.py``) that drives the value when set. Wraps
``edit_engine.keyframes.KeyframeTrack`` instead of duplicating its
sample/easing/ffmpeg-expr machinery.
"""

from __future__ import annotations

from typing import Any

from ..edit_engine.keyframes import KeyframeTrack
from . import expressions

__all__ = ["PropertyTrack"]


class PropertyTrack:
    """One animatable property: keyframes, optionally expression-driven."""

    def __init__(self, name: str, *, fps: float = 24.0) -> None:
        self.name = name
        self.fps = float(fps)
        self._track = KeyframeTrack(clip_id="", effect=name, param=name)
        self._expression: str | None = None

    # -- keyframes -----------------------------------------------------
    def add(self, time: float, value: float, easing: str = "smooth") -> None:
        self._track.add(time, value, easing)

    def remove_at(self, time: float) -> bool:
        return self._track.remove_at(time)

    @property
    def keys(self) -> list:
        return list(self._track.keys)

    def sample(self, t: float) -> float:
        """Value at time ``t`` (seconds) — expression wins if one is set."""
        if self._expression is not None:
            return float(expressions.evaluate(self._expression, self._ctx(t)))
        return float(self._track.sample(t))

    def sample_frame(self, frame: int, fps: float | None = None) -> float:
        rate = float(fps) if fps is not None else self.fps
        return self.sample(frame / rate)

    # -- expressions ---------------------------------------------------
    def set_expression(self, expr: str) -> None:
        expressions.validate(expr)
        self._expression = expr

    def clear_expression(self) -> None:
        self._expression = None

    @property
    def expression(self) -> str | None:
        return self._expression

    def _ctx(self, t: float) -> dict[str, float]:
        base = float(self._track.sample(t)) if self._track.keys else 0.0
        return {
            "time": float(t),
            "t": float(t),
            "frame": float(t) * self.fps,
            "fps": self.fps,
            "value": base,
        }

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fps": self.fps,
            "expression": self._expression,
            "keys": [
                {"t": k.time, "value": k.value, "easing": k.easing}
                for k in self._track.keys
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PropertyTrack:
        track = cls(str(data.get("name", "value")),
                    fps=float(data.get("fps", 24.0)))
        for key in data.get("keys", []):
            track.add(float(key["t"]), float(key["value"]),
                      str(key.get("easing", "smooth")))
        if data.get("expression"):
            track.set_expression(str(data["expression"]))
        return track
