"""Style-agnostic video effect primitives → ffmpeg filter chains.

Each builder: ``(params, ctx) -> str`` where ``ctx`` is
``{"w","h","fps","duration","seed","beats"}``. Builders are pure
parameterized transforms — no aesthetic choices, no style names.

Proven ffmpeg truths (verified against ffmpeg 8.1.2, migrated):
- no ``rand()`` in crop exprs → seeded sin-hash pseudo-random offsets
  (deterministic per seed — a feature, not a workaround);
- ``rgbashift`` instead of ``mergeplanes`` for channel splits
  (mergeplanes rejects packed rgb24; gbrp mapping is fragile);
- crop w/h are init-evaluated (``t`` fails there) → all zoom motion
  uses ``scale=eval=frame`` (truly per-frame) + static center crop.
"""

from __future__ import annotations

import math
from typing import Any

from ...media_edit.videos import MediaEditError


def _seed_phases(seed: int) -> tuple[float, float]:
    """Deterministic pseudo-random phases from ``seed``."""
    s = int(seed) & 0xFFFFFFFF
    p1 = ((s * 2654435761) % 100000) / 100000.0 * 2 * math.pi
    p2 = ((s * 40503 + 17) % 100000) / 100000.0 * 2 * math.pi
    return p1, p2


def _fx_shake(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    amp = float(params.get("amplitude", 14.0))
    w, h = ctx["w"], ctx["h"]
    margin = int(math.ceil(amp)) + 4
    s1, s2 = _seed_phases(int(ctx["seed"]) + int(params.get("seed_offset", 0)))
    return (
        f"crop={w - 2 * margin}:{h - 2 * margin}:"
        f"x='(iw-ow)/2+{amp:.1f}*sin(n*12.9898+{s1:.4f})':"
        f"y='(ih-oh)/2+{amp:.1f}*sin(n*78.233+{s2:.4f})',"
        f"scale={w}:{h}:flags=bilinear,setsar=1"
    )


def _fx_punch_zoom(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    # zoom 1 → zmax across the segment. scale=eval=frame is the robust
    # primitive: crop w/h are init-evaluated in this ffmpeg build.
    zmax = float(params.get("zmax", 1.18))
    w, h, dur = ctx["w"], ctx["h"], max(ctx["duration"], 0.04)
    k = zmax - 1.0
    z = f"(1+{k:.4f}*min(t/{dur:.4f}\\,1))"
    return (
        f"scale=eval=frame:w='iw*{z}':h='ih*{z}':flags=bilinear,"
        f"crop={w}:{h}:x='(iw-ow)/2':y='(ih-oh)/2',setsar=1"
    )


def _fx_flash(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    dur = ctx["duration"]
    at = params.get("time", params.get("at", dur / 2.0))
    d = float(params.get("duration", 0.12))
    level = float(params.get("level", 0.85))
    t0 = max(0.0, float(at) - d / 2.0)
    # triangular brightness pulse — a white blink without concat surgery
    return (
        f"eq=brightness='if(between(t\\,{t0:.3f}\\,{t0 + d:.3f})\\,"
        f"{level:.2f}*(1-abs(t-{t0 + d / 2:.3f})/({d / 2:.3f}))\\,0)'"
    )


def _fx_grain(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    strength = float(params.get("strength", 9.0))
    return f"noise=alls={strength:.1f}:allf=t+u"


def _fx_vignette(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    angle = params.get("angle", "PI/4.2")
    return f"vignette={angle}"


def _fx_rgb_split(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    shift = float(params.get("shift", 10.0))
    if params.get("animate"):
        # NOTE: per-frame expressions are rejected by rgbashift on some
        # ffmpeg builds — render.py intercepts animate=True and
        # discretizes via _chain_rgb_animated. This branch is only a
        # fallback for direct builder use.
        raise MediaEditError(
            "animated rgb_split must go through the chain builder "
            "(_chain_effect), not the raw filter builder")
    return f"rgbashift=rh={shift:.1f}:bh={-shift:.1f}"


def _fx_ken_burns(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    zoom = float(params.get("zoom", 1.25))
    direction = str(params.get("direction", "in")).lower()
    pan_x = float(params.get("pan_x", 0.0))  # px/sec drift
    pan_y = float(params.get("pan_y", 0.0))
    w, h, dur = ctx["w"], ctx["h"], max(ctx["duration"], 0.04)
    k = zoom - 1.0
    if direction == "out":
        z = f"({zoom:.4f}-{k:.4f}*min(t/{dur:.4f}\\,1))"
    else:
        z = f"(1+{k:.4f}*min(t/{dur:.4f}\\,1))"
    return (
        f"scale=eval=frame:w='iw*{z}':h='ih*{z}':flags=bilinear,"
        f"crop={w}:{h}:x='(iw-ow)/2+{pan_x:.1f}*t':"
        f"y='(ih-oh)/2+{pan_y:.1f}*t',setsar=1"
    )


def _fx_beat_pulse(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    beats = [float(b) for b in (params.get("beats") or ctx.get("beats") or [])]
    beats = beats[:10]  # keep the expression small
    if not beats:
        raise MediaEditError("beat_pulse needs beats= (or ctx beats)")
    strength = float(params.get("strength", 0.10))
    width = float(params.get("width", 0.14))
    w, h = ctx["w"], ctx["h"]
    terms = "+".join(
        f"max(0\\,1-abs(t-{b:.3f})/{width:.3f})" for b in beats)
    z = f"(1+{strength:.3f}*({terms}))"
    return (
        f"scale=eval=frame:w='iw*{z}':h='ih*{z}':flags=bilinear,"
        f"crop={w}:{h}:x='(iw-ow)/2':y='(ih-oh)/2',setsar=1"
    )


def _fx_blur(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    radius = float(params.get("radius", 8.0))
    if radius <= 0:
        raise MediaEditError("blur radius must be > 0")
    return f"boxblur=luma_radius={radius:.1f}:luma_power=1"


#: named color grades → ffmpeg filter chains. Grades are looks, not
#: styles: a documentary and a vlog can both ask for "warm".
_GRADES: dict[str, str] = {
    "warm": "eq=saturation=1.15,colorbalance=rs=0.08:gs=0.03:bs=-0.06",
    "cool": "eq=saturation=1.05,colorbalance=rs=-0.06:gs=0.00:bs=0.08",
    "teal_orange": ("colorbalance=rs=0.10:gs=0.02:bs=-0.02:"
                    "rm=0.06:gm=-0.02:bm=-0.08,eq=saturation=1.2:"
                    "contrast=1.05"),
    "noir": "eq=saturation=0:contrast=1.15:brightness=-0.02",
    "faded": "eq=saturation=0.75:contrast=0.90:brightness=0.04",
    "vibrant": "eq=saturation=1.35:contrast=1.08",
}


def _fx_color_grade(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    grade = str(params.get("grade", "warm")).strip().lower()
    chain = _GRADES.get(grade)
    if chain is None:
        raise MediaEditError(
            f"unknown color grade {grade!r}; use: {sorted(_GRADES)}")
    return chain


#: effect name → filter builder. velocity_ramp is structural (rewrites
#: the chain via trim/setpts/concat) and handled in render.py.
EFFECTS: dict[str, Any] = {
    "shake": _fx_shake,
    "punch_zoom": _fx_punch_zoom,
    "flash": _fx_flash,
    "grain": _fx_grain,
    "vignette": _fx_vignette,
    "rgb_split": _fx_rgb_split,
    "ken_burns": _fx_ken_burns,
    "beat_pulse": _fx_beat_pulse,
    "blur": _fx_blur,
    "color_grade": _fx_color_grade,
}


def _effect_filter(name: str, params: dict[str, Any],
                   ctx: dict[str, Any]) -> str:
    from .timeline import canonical_effect
    name = canonical_effect(name)
    builder = EFFECTS.get(name)
    if builder is None:
        raise MediaEditError(
            f"unknown effect {name!r}; use: {sorted(EFFECTS)}")
    return builder(params or {}, ctx)


def _ctx(w: int, h: int, fps: int, duration: float, seed: int,
         beats: list[float] | None = None) -> dict[str, Any]:
    return {"w": w, "h": h, "fps": fps, "duration": duration,
            "seed": seed, "beats": beats or []}


def list_effects() -> list[str]:
    """All registered effect names (plus the structural velocity_ramp)."""
    return sorted(EFFECTS) + ["velocity_ramp"]
