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

    def __post_init__(self) -> None:
        # sample()/duration() assume chronological keys — enforce it
        # for every construction path (from_dict, direct), not just add().
        self.keys.sort(key=lambda k: k.time)

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


def ffmpeg_expr(track: KeyframeTrack) -> str:
    """The track's ffmpeg expression (public name for the builder)."""
    return _ffmpeg_expr(track)


# ── Wave 1.2c: the renderer's keyframe wire ────────────────────────
# KeyframeTrack declarations live in the effects property registry
# (effects.EFFECT_PROPERTIES, from which KEYFRAMABLE_PARAMS derives —
# each effect states which params are keyframable: the property-path
# idea). Four application families:
#
# * ``expression`` — zoom-family tracks become ONE ffmpeg filter via
#   :func:`track_to_filter` with :data:`ZOOM_TEMPLATE` (scale with
#   eval=frame is a proven per-frame expression in this engine —
#   the punch_zoom/ken_burns builders use the same primitive);
# * ``opacity`` — alpha curves are applied by the renderer's
#   finishing pass, discretized into constant-alpha windows (the
#   _chain_rgb_animated doctrine: this ffmpeg rejects per-frame
#   expressions in several filters, so curves are sampled);
# * ``position`` — overlay placement: the composite's overlay()
#   call takes the track's expression for x/y directly;
# * ``sampled`` (Wave 5) — every other declared param (blur radius,
#   grain strength, rgb shift, ken-burns pan, …): the renderer
#   discretizes the curve into constant-value windowed sub-effects
#   of the SAME effect, built by the SAME chain builder a static
#   effect uses — one registry, one rung, sampled where ffmpeg
#   expressions cannot go.

#: scale-eval template for zoom-family tracks; ``{v}`` is the zoom
#: factor expression, ``{w}``/``{h}`` the canvas. The static
#: punch_zoom builder converges on the same shape with a linear {v}.
ZOOM_TEMPLATE = (
    "scale=eval=frame:w='iw*({v})':h='ih*({v})':flags=bilinear,"
    "crop={w}:{h}:x='(iw-ow)/2':y='(ih-oh)/2',setsar=1"
)

_EXPRESSION_FAMILY = {("zoom", "z"), ("ken_burns", "zoom"),
                      ("punch_zoom", "zmax")}


def classify_track(track: KeyframeTrack) -> str:
    """Which application family a track belongs to.

    Raises ValueError when (effect, param) is not declared keyframable
    in :data:`effects.KEYFRAMABLE_PARAMS` — an undeclared animation
    target is a build error, never a silent no-op at render time.
    A track naming a deferred effect raises with the deferral's
    reason (a named refusal, not an unknown target).
    """
    from .effects import KEYFRAMABLE_PARAMS, deferred_effect_reason
    reason = deferred_effect_reason(track.effect)
    if reason is not None:
        raise ValueError(
            f"keyframe target {track.effect}.{track.param}: effect "
            f"{track.effect!r} is deferred, not unknown: {reason}")
    allowed = KEYFRAMABLE_PARAMS.get(track.effect)
    if allowed is None or track.param not in allowed:
        declared = ", ".join(
            f"{e}.{p}" for e, ps in sorted(KEYFRAMABLE_PARAMS.items())
            for p in ps)
        raise ValueError(
            f"keyframe target {track.effect}.{track.param} is not "
            f"keyframable; declared: {declared}")
    if (track.effect, track.param) in _EXPRESSION_FAMILY:
        return "expression"
    if track.effect == "opacity":
        return "opacity"
    if track.effect == "position":
        return "position"
    return "sampled"


def keyframe_filter(track: KeyframeTrack, *, width: int,
                    height: int) -> str:
    """The ffmpeg filter for an expression-family track on a canvas.

    This is the renderer's half of the two-paths-one-result contract:
    it is exactly ``track_to_filter(track, ZOOM_TEMPLATE)`` with the
    canvas substituted — a caller applying the same track standalone
    through :func:`track_to_filter` gets the identical filter.
    """
    if classify_track(track) != "expression":
        raise ValueError(
            f"keyframe_filter only serves expression-family tracks, "
            f"got {track.effect}.{track.param}")
    return track_to_filter(
        track, ZOOM_TEMPLATE.format(v="{v}", w=width, h=height))


def shifted(track: KeyframeTrack, offset: float,
            duration: float) -> KeyframeTrack:
    """The track re-based to a play window [offset, offset+duration].

    Segment chains see segment-local time (trim + setpts restarts the
    clock), so a track bound to a clip that plays in several segments
    is re-based per segment: keys inside the window shift by
    ``-offset``; the curve's value at the window edges becomes
    synthetic boundary keys (sampled, so the visible curve is the
    same curve, windowed). A window covering the whole track returns
    an equal track.
    """
    if duration <= 0:
        raise ValueError("shift window duration must be > 0")
    end = offset + duration
    keys: list[Keyframe] = [Keyframe(0.0, track.sample(offset),
                                     track.keys[0].easing
                                     if track.keys else "smooth")]
    for k in track.keys:
        if offset < k.time < end:
            keys.append(Keyframe(k.time - offset, k.value, k.easing))
    keys.append(Keyframe(float(duration), track.sample(end), "smooth"))
    # collapse duplicates at the same instant (window edges on keys)
    out: list[Keyframe] = []
    for k in keys:
        if out and abs(k.time - out[-1].time) < 1e-9:
            out[-1] = k
        else:
            out.append(k)
    return KeyframeTrack(clip_id=track.clip_id, effect=track.effect,
                         param=track.param, keys=out)


def discretize(track: KeyframeTrack, *, step_s: float = 0.125,
               max_steps: int = 64, fps: Any = None
               ) -> list[tuple[float, float, float]]:
    """Sample a curve into constant (start, end, value) windows.

    For filters without per-frame expression support (the opacity
    pass): each window takes the curve's value at its midpoint, the
    same sampling doctrine as render.py's animated rgb_split.

    With ``fps`` given (the sampled keyframe family's binding), the
    windows are computed in INTEGER FRAMES on the timeline's
    :class:`~.frametime.FrameGrid`: every boundary is an exact frame
    multiple and each window's value is the curve sampled at the
    window's middle frame — keyframe binding pinned at exact frames,
    no float-step drift.
    """
    if not track.keys:
        raise ValueError("track has no keyframes")
    start = track.keys[0].time
    end = track.keys[-1].time
    if end <= start:
        return [(start, end, track.keys[0].value)]
    if fps is not None:
        from fractions import Fraction

        from .frametime import FrameGrid
        grid = FrameGrid(fps)
        f0, f1 = grid.frames(start), grid.frames(end)
        total = f1 - f0
        if total <= 0:
            return [(start, end, track.keys[0].value)]
        step_f = max(1, round(step_s * grid.fps_float))
        if (total + step_f - 1) // step_f > max_steps:
            step_f = max(1, (total + max_steps - 1) // max_steps)
        out_f: list[tuple[float, float, float]] = []
        fa = f0
        while fa < f1:
            fb = min(fa + step_f, f1)
            mid = float(grid.seconds(Fraction(fa + fb, 2)))
            out_f.append((float(grid.seconds(fa)),
                          float(grid.seconds(fb)),
                          track.sample(mid)))
            fa = fb
        return out_f
    steps = max(1, min(max_steps, int((end - start) / step_s) or 1))
    span = (end - start) / steps
    out: list[tuple[float, float, float]] = []
    for i in range(steps):
        a = start + span * i
        b = start + span * (i + 1)
        out.append((a, b, track.sample((a + b) / 2.0)))
    return out


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
