"""Devon Studio — keyframe animation for the edit engine.

Keyframe tracks animate any numeric effect parameter over a clip's
play time: zoom ramps, opacity fades, position moves, color shifts.
Interpolation is real math, not presets.

A KeyframeTrack binds to (clip_id, effect_name, param_name) and holds
time→value points. The renderer samples it per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ── interpolation ──────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 2 * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 2) / 2


_EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda t: max(0.0, min(1.0, t)),
    "smooth": _smoothstep,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
}


@dataclass
class Keyframe:
    """One point: at `time` seconds, the value is `value`."""
    time: float
    value: float
    easing: str = "smooth"  # easing INTO this keyframe from the previous

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.time, "value": self.value, "easing": self.easing}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Keyframe":
        return cls(time=float(d["time"]), value=float(d["value"]),
                   easing=str(d.get("easing", "smooth")))


@dataclass
class KeyframeTrack:
    """Animates one parameter of one effect on one clip."""
    clip_id: str
    effect: str
    param: str
    keys: list[Keyframe] = field(default_factory=list)

    def add(self, time: float, value: float,
            easing: str = "smooth") -> "KeyframeTrack":
        if easing not in _EASINGS:
            raise ValueError(f"unknown easing {easing!r}; pick from {sorted(_EASINGS)}")
        self.keys.append(Keyframe(time=time, value=value, easing=easing))
        self.keys.sort(key=lambda k: k.time)
        return self

    def sample(self, t: float) -> float:
        """Value at time t (seconds into the clip's play time)."""
        if not self.keys:
            raise ValueError("track has no keyframes")
        if t <= self.keys[0].time:
            return self.keys[0].value
        if t >= self.keys[-1].time:
            return self.keys[-1].value
        for prev, nxt in zip(self.keys, self.keys[1:]):
            if prev.time <= t <= nxt.time:
                span = nxt.time - prev.time
                local = 0.0 if span <= 0 else (t - prev.time) / span
                eased = _EASINGS.get(nxt.easing, _EASINGS["smooth"])(local)
                return _lerp(prev.value, nxt.value, eased)
        return self.keys[-1].value

    def duration(self) -> float:
        return self.keys[-1].time if self.keys else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"clip_id": self.clip_id, "effect": self.effect,
                "param": self.param,
                "keys": [k.to_dict() for k in self.keys]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KeyframeTrack":
        return cls(clip_id=d["clip_id"], effect=d["effect"], param=d["param"],
                   keys=[Keyframe.from_dict(k) for k in d.get("keys", [])])


# ── ffmpeg expression builder ──────────────────────────────────────
# Turns a track into an ffmpeg filter expression so the animation
# renders frame-accurately without per-frame Python.

def _ffmpeg_expr(track: KeyframeTrack) -> str:
    """Build an ffmpeg `if(lt(t,...),...)` expression from keyframes."""
    ks = track.keys
    if len(ks) == 1:
        return f"{ks[0].value:.6f}"
    parts: list[str] = []
    # Build nested ifs from the last segment backwards.
    expr = f"{ks[-1].value:.6f}"
    for prev, nxt in zip(reversed(ks[:-1]), reversed(ks[1:])):
        span = nxt.time - prev.time
        if span <= 0:
            expr = f"{nxt.value:.6f}"
            continue
        # linear in ffmpeg expr; easing baked by subdividing
        a, b = prev.value, nxt.value
        local = f"(({nxt.time:.3f}-t)/{span:.6f})"
        seg = f"({b:.6f}+({a:.6f}-{b:.6f})*{local})"
        # smoothstep approximation via the eased flag: subdivide segment
        if nxt.easing in ("smooth", "ease_in_out"):
            # 4 subdivisions approximate smoothstep closely
            seg = _subdivided(prev, nxt, 4)
        expr = f"if(lt(t,{nxt.time:.3f}),{seg},{expr})"
    parts.append(expr)
    return parts[0]


def _subdivided(prev: Keyframe, nxt: Keyframe, n: int) -> str:
    """Piecewise-linear smoothstep approximation as nested ifs."""
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        s = t * t * (3 - 2 * t)
        pts.append((prev.time + (nxt.time - prev.time) * t,
                    prev.value + (nxt.value - prev.value) * s))
    expr = f"{pts[-1][1]:.6f}"
    for (t0, v0), (t1, v1) in zip(reversed(pts[:-1]), reversed(pts[1:])):
        span = t1 - t0
        local = f"(({t1:.3f}-t)/{span:.6f})"
        seg = f"({v1:.6f}+({v0:.6f}-{v1:.6f})*{local})"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return expr


def track_to_filter(track: KeyframeTrack, filter_template: str) -> str:
    """Render a track into an ffmpeg filter string.

    filter_template uses {v} where the animated value goes, e.g.
    "zoompan=z='{v}':d=1" or "format=yuv444p,eq=brightness={v}".
    """
    return filter_template.replace("{v}", _ffmpeg_expr(track))


# ── convenience builders ───────────────────────────────────────────

def fade_in(track_id: str, duration: float = 1.0) -> KeyframeTrack:
    """Opacity 0 → 1 over `duration`."""
    return KeyframeTrack(clip_id=track_id, effect="opacity", param="alpha",
                         keys=[Keyframe(0.0, 0.0, "smooth"),
                               Keyframe(duration, 1.0, "smooth")])


def fade_out(track_id: str, start: float, duration: float = 1.0) -> KeyframeTrack:
    return KeyframeTrack(clip_id=track_id, effect="opacity", param="alpha",
                         keys=[Keyframe(start, 1.0, "smooth"),
                               Keyframe(start + duration, 0.0, "smooth")])


def zoom_ramp(track_id: str, start: float, end: float,
              z0: float = 1.0, z1: float = 1.5,
              easing: str = "smooth") -> KeyframeTrack:
    """Punch-zoom 1.0 → 1.5 between start and end."""
    return KeyframeTrack(clip_id=track_id, effect="zoom", param="z",
                         keys=[Keyframe(start, z0, easing),
                               Keyframe(end, z1, easing)])


def list_easings() -> list[str]:
    return sorted(_EASINGS)
