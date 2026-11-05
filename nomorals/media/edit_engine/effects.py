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
from dataclasses import dataclass
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


def _fx_opacity(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    """Constant alpha — the still point of an opacity keyframe curve.

    The stream gains an alpha channel at ``alpha`` opacity. On an
    overlay track the composite reads it directly; on the base track
    the renderer composites over black, so it reads as a fade.
    """
    alpha = float(params.get("alpha", 1.0))
    if not 0.0 <= alpha <= 1.0:
        raise MediaEditError("opacity alpha must be in [0, 1]")
    return f"format=yuva420p,colorchannelmixer=aa={alpha:.4f}"


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


#: the scalar keyframe paths into a CDL decision (Wave 5 registry
#: extension closing the "animated CDL components" remainder): each
#: component of slope/offset/power is one scalar property, and
#: saturation is the tenth. A keyframe track binds ONE component;
#: the renderer samples every bound component jointly into
#: constant-value windowed sub-effects whose params carry all ten,
#: and the builder below composes them into the one CDL that gets
#: baked — components are never applied as sequential grades.
CDL_COMPONENT_PARAMS: dict[str, tuple[str, int | None]] = {
    "cdl_slope_r": ("slope", 0), "cdl_slope_g": ("slope", 1),
    "cdl_slope_b": ("slope", 2),
    "cdl_offset_r": ("offset", 0), "cdl_offset_g": ("offset", 1),
    "cdl_offset_b": ("offset", 2),
    "cdl_power_r": ("power", 0), "cdl_power_g": ("power", 1),
    "cdl_power_b": ("power", 2),
    "cdl_saturation": ("saturation", None),
}

_CDL_IDENTITY: dict[str, Any] = {
    "slope": (1.0, 1.0, 1.0), "offset": (0.0, 0.0, 0.0),
    "power": (1.0, 1.0, 1.0), "saturation": 1.0}


def compose_cdl(params: dict[str, Any]) -> dict[str, Any] | None:
    """The CDL decision a color_grade param set names, or None.

    Precedence: explicit ``cdl_*`` component params compose over
    the ``cdl`` dict (which composes over the CDL identity);
    a bare ``cdl`` dict stands alone; neither → None (a named
    ``grade`` applies instead). Component params and a named
    ``grade`` are two different decisions — carrying both raises,
    naming the conflict, instead of silently dropping one.
    """
    comps = {k: params[k] for k in CDL_COMPONENT_PARAMS
             if params.get(k) is not None}
    cdl = params.get("cdl")
    if comps:
        if params.get("grade") is not None:
            raise MediaEditError(
                "color_grade: cdl_* component params and a named "
                "grade are two different decisions — author one "
                "or the other (components compose over a cdl dict, "
                "never over a named grade)")
        out: dict[str, Any] = dict(_CDL_IDENTITY)
        if isinstance(cdl, dict):
            out.update({k: cdl[k] for k in
                        ("slope", "offset", "power", "saturation")
                        if k in cdl})
        vec: dict[str, list[float]] = {}
        for key, (field, idx) in CDL_COMPONENT_PARAMS.items():
            if key not in comps:
                continue
            if idx is None:
                out[field] = float(comps[key])
            else:
                if field not in vec:
                    vec[field] = [float(v) for v in out[field]]
                vec[field][idx] = float(comps[key])
        out.update(vec)
        return out
    if isinstance(cdl, dict) and cdl:
        return dict(cdl)
    return None


def _fx_color_grade(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    cdl = compose_cdl(params)
    if cdl is not None:
        return color_grade_cdl_chain(cdl)
    return color_grade_chain(str(params.get("grade", "warm")))


def color_grade_cdl_chain(cdl: dict[str, Any], *,
                          cache_dir: str | None = None) -> str:
    """ffmpeg chain for an ASC-CDL grade carried as timeline state.

    A ``color_grade`` effect may carry ``params={"cdl": {...}}``
    (slope/offset/power/saturation, e.g. from a Document look_lock)
    instead of a named grade. The CDL is baked to a content-addressed
    .cube by the canonical color engine (``media/studio_color``) and
    applied via lut3d — the same single render path as every other
    grade surface, never a separate re-encode.
    """
    from ..studio_color import CDL, StudioColorError, bake_cdl_cube_file
    try:
        decision = CDL.from_dict(dict(cdl))
    except StudioColorError as exc:
        raise MediaEditError(f"bad cdl grade: {exc}") from exc
    path = bake_cdl_cube_file(decision, cache_dir=cache_dir)
    return f"lut3d='{path}'"


def list_color_grades() -> list[str]:
    """Canonical color-grade names (the single source of truth for the
    shared grade definitions — motion_studio.grading delegates here)."""
    return sorted(_GRADES)


def color_grade_chain(name: str) -> str:
    """ffmpeg filter chain for a canonical color grade.

    Raises MediaEditError on unknown names.
    """
    key = str(name or "").strip().lower()
    chain = _GRADES.get(key)
    if chain is None:
        raise MediaEditError(
            f"unknown color grade {name!r}; use: {sorted(_GRADES)}")
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
    "opacity": _fx_opacity,
}

#: ── the property registry (Echelon Wave 5 — EffectCraft's property
#: tree, Devon-shaped). Every effect is a set of Properties at paths
#: ``effects/<name>/<param>``; each param declares its kind, default,
#: range and whether a :class:`~.keyframes.KeyframeTrack` may animate
#: it (and through which application family). A param outside this
#: registry is an authoring error — :func:`validate_effect_params`
#: raises naming the path, never a silent no-op. ``KEYFRAMABLE_PARAMS``
#: below is DERIVED from this table (single source of truth).
#:
#: Application families: ``expression`` (one per-frame ffmpeg
#: expression — the zoom family), ``opacity`` (alpha windows over the
#: composite), ``position`` (overlay placement expressions) and
#: ``sampled`` (the renderer discretizes the curve into constant-
#: value windowed sub-effects — the doctrine this ffmpeg's
#: expression-rejecting filters already forced for animated
#: rgb_split and opacity). Params that are placement, identity or
#: shape rather than value curves (a flash's centre, a shake's seed)
#: declare ``keyframable=None`` and say why.
@dataclass(frozen=True)
class ParamSpec:
    """One declared property of one effect."""
    kind: str                    # number|int|bool|string|expr|enum|list|dict
    default: Any = None
    lo: float | None = None
    hi: float | None = None
    keyframable: str | None = None   # family name, or None
    choices: tuple = ()
    doc: str = ""


#: effect name → {param name: ParamSpec}. Covers every param every
#: builder reads (the builders' ``params.get`` defaults ARE the
#: registry defaults — the table describes the code, and
#: :func:`validate_effect_params` enforces it at the chain funnel).
EFFECT_PROPERTIES: dict[str, dict[str, ParamSpec]] = {
    "shake": {
        "amplitude": ParamSpec(
            "number", 14.0, 0.0, 500.0, "sampled",
            doc="peak displacement (px) of the seeded sine wobble"),
        "seed_offset": ParamSpec(
            "int", 0,
            doc="deterministic phase shift of the wobble — identity, "
                "not a value curve; not keyframable"),
    },
    "punch_zoom": {
        "zmax": ParamSpec(
            "number", 1.18, 1.0, 8.0, "expression",
            doc="end zoom factor; animates through the zoom "
                "expression family"),
    },
    "flash": {
        "time": ParamSpec(
            "number", None,
            doc="flash centre (s, clip-relative); placement, not a "
                "value curve — not keyframable"),
        "at": ParamSpec(
            "number", None,
            doc="alias of ``time`` (the older spelling)"),
        "duration": ParamSpec(
            "number", 0.12, 0.0, 10.0,
            doc="pulse width (s) — the shape of the blink, authored "
                "statically; the level carries the curve"),
        "level": ParamSpec(
            "number", 0.85, 0.0, 1.0, "sampled",
            doc="peak brightness of the triangular pulse"),
    },
    "grain": {
        "strength": ParamSpec(
            "number", 9.0, 0.0, 100.0, "sampled",
            doc="ffmpeg noise alls strength"),
    },
    "vignette": {
        "angle": ParamSpec(
            "expr", "PI/4.2",
            doc="ffmpeg angle expression (e.g. 'PI/4.2') — a string "
                "expression, not a scalar curve; not keyframable"),
    },
    "rgb_split": {
        "shift": ParamSpec(
            "number", 10.0, -200.0, 200.0, "sampled",
            doc="horizontal channel shift (px): red +, blue −"),
        "animate": ParamSpec(
            "bool", False,
            doc="sine-wobble mode — render.py discretizes it; a "
                "``shift`` keyframe track samples the static shift "
                "instead"),
        "period": ParamSpec(
            "number", 0.5, 0.0, 60.0,
            doc="wobble period (s); animate mode only"),
    },
    "ken_burns": {
        "zoom": ParamSpec(
            "number", 1.25, 1.0, 8.0, "expression",
            doc="end zoom factor over the clip (direction in) or "
                "start factor (out)"),
        "direction": ParamSpec(
            "enum", "in", choices=("in", "out"),
            doc="zoom direction"),
        "pan_x": ParamSpec(
            "number", 0.0, -1000.0, 1000.0, "sampled",
            doc="horizontal drift (px/sec)"),
        "pan_y": ParamSpec(
            "number", 0.0, -1000.0, 1000.0, "sampled",
            doc="vertical drift (px/sec)"),
    },
    "beat_pulse": {
        "beats": ParamSpec(
            "list", None,
            doc="beat times (s, clip-relative); falls back to the "
                "ctx beats — the grid is data, not a curve"),
        "strength": ParamSpec(
            "number", 0.10, 0.0, 2.0, "sampled",
            doc="zoom bump height per beat"),
        "width": ParamSpec(
            "number", 0.14, 0.0, 5.0, "sampled",
            doc="bump width (s)"),
    },
    "blur": {
        "radius": ParamSpec(
            "number", 8.0, 0.0, 200.0, "sampled",
            doc="boxblur luma radius (px); the builder additionally "
                "requires > 0"),
    },
    "color_grade": {
        "grade": ParamSpec(
            "enum", "warm",
            choices=tuple(sorted(_GRADES)),
            doc="canonical grade name — resolved by "
                "color_grade_chain(), never re-implemented. A "
                "named grade is a cut, not a curve: not keyframable "
                "(animate the CDL components instead)"),
        "cdl": ParamSpec(
            "dict", None,
            doc="ASC-CDL decision dict — baked to a .cube by "
                "studio_color (the canonical chain) and applied via "
                "lut3d. The dict as a whole is a cut; its scalar "
                "components are keyframable through the cdl_* "
                "paths below"),
        # the per-component keyframe paths (the §1.2b remainder,
        # closed): each is one scalar of the CDL decision, bound
        # by one keyframe track and sampled jointly with its
        # siblings into windowed sub-effects (see render.py's
        # joint sampled binding). The ranges are the CDL's own
        # working domain; power's > 0 law stays in studio_color's
        # CDL constructor, which refuses by name at the bake.
        "cdl_slope_r": ParamSpec(
            "number", 1.0, 0.0, 10.0, "sampled",
            doc="CDL slope, red channel"),
        "cdl_slope_g": ParamSpec(
            "number", 1.0, 0.0, 10.0, "sampled",
            doc="CDL slope, green channel"),
        "cdl_slope_b": ParamSpec(
            "number", 1.0, 0.0, 10.0, "sampled",
            doc="CDL slope, blue channel"),
        "cdl_offset_r": ParamSpec(
            "number", 0.0, -2.0, 2.0, "sampled",
            doc="CDL offset, red channel"),
        "cdl_offset_g": ParamSpec(
            "number", 0.0, -2.0, 2.0, "sampled",
            doc="CDL offset, green channel"),
        "cdl_offset_b": ParamSpec(
            "number", 0.0, -2.0, 2.0, "sampled",
            doc="CDL offset, blue channel"),
        "cdl_power_r": ParamSpec(
            "number", 1.0, 0.0, 10.0, "sampled",
            doc="CDL power, red channel (the CDL constructor "
                "refuses 0 by name at the bake)"),
        "cdl_power_g": ParamSpec(
            "number", 1.0, 0.0, 10.0, "sampled",
            doc="CDL power, green channel"),
        "cdl_power_b": ParamSpec(
            "number", 1.0, 0.0, 10.0, "sampled",
            doc="CDL power, blue channel"),
        "cdl_saturation": ParamSpec(
            "number", 1.0, 0.0, 10.0, "sampled",
            doc="CDL saturation"),
    },
    "opacity": {
        "alpha": ParamSpec(
            "number", 1.0, 0.0, 1.0, "opacity",
            doc="constant alpha; keyframes render as alpha windows "
                "over the composite"),
    },
    "velocity_ramp": {
        "speed": ParamSpec(
            "number", 2.0, 0.0, 16.0,
            doc="constant speed factor — structural: unmasked, "
                "render.py rewrites the chain (trim/setpts/concat); "
                "masked (regional), render.py's time-remap "
                "resampler retimes the branch back onto the base "
                "timeline (hold/cut absorbed at the window edge, "
                "inside the region only). No value curve to bind "
                "either way; not keyframable"),
        "ramps": ParamSpec(
            "list", None,
            doc="explicit (start, end, speed) pieces"),
    },
}

#: Keyframe-layer pseudo-effects: animation targets that are not
#: chain builders. ``zoom`` is the keyframe layer's own zoom property
#: (the ``zoom_ramp`` helper's effect name); ``position`` addresses
#: an overlay clip's placement in the composite.
PROPERTY_EFFECTS: dict[str, dict[str, ParamSpec]] = {
    "zoom": {
        "z": ParamSpec(
            "number", 1.0, 0.05, 20.0, "expression",
            doc="zoom factor (the keyframe layer's own property)"),
    },
    "position": {
        "x": ParamSpec(
            "number", 0.0, -8192.0, 8192.0, "position",
            doc="overlay placement x (px)"),
        "y": ParamSpec(
            "number", 0.0, -8192.0, 8192.0, "position",
            doc="overlay placement y (px)"),
    },
}

#: Effects the plan carries as named deferrals. They are NOT in
#: EFFECTS and never render — naming one raises with the reason
#: (a named refusal, never a stub, never a silent unknown-effect).
DEFERRED_EFFECTS: dict[str, str] = {
    "camera3d": (
        "the 3D rung is the owner's unmade choice (plan §1.5e/§6.3: "
        "a Blender-class external engine vs a Rust rung is a "
        "hardware-bearing decision) — no 3D renderer exists in the "
        "engine and none is substituted"),
}


def effect_property_path(name: str, param: str) -> str:
    """EffectCraft-shaped property path: ``effects/<name>/<param>``."""
    return f"effects/{name}/{param}"


def _derive_keyframable() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for src in (EFFECT_PROPERTIES, PROPERTY_EFFECTS):
        for name, params in src.items():
            ks = tuple(p for p, s in params.items() if s.keyframable)
            if ks:
                out[name] = ks
    return out


#: Keyframable params per effect — DERIVED from EFFECT_PROPERTIES +
#: PROPERTY_EFFECTS (Wave 1.2c declarations, Wave 5 registry).
KEYFRAMABLE_PARAMS: dict[str, tuple[str, ...]] = _derive_keyframable()


def keyframable_params() -> dict[str, tuple[str, ...]]:
    """Effect → keyframable params (derived from the registry)."""
    return dict(KEYFRAMABLE_PARAMS)


def keyframable_family(name: str, param: str) -> str | None:
    """The application family a keyframe on ``name.param`` renders
    through (``expression`` | ``opacity`` | ``position`` |
    ``sampled``), or None when the target is not keyframable."""
    for src in (EFFECT_PROPERTIES, PROPERTY_EFFECTS):
        spec = src.get(name, {}).get(param)
        if spec is not None:
            return spec.keyframable
    return None


def effect_rung(name: str) -> str:
    """The code rung that produces ``name``'s pixels (honesty
    records name it): the builder function, the structural chain
    builder, or the keyframe layer for pseudo-effects."""
    builder = EFFECTS.get(name)
    if builder is not None:
        return f"edit_engine.effects.{builder.__name__}"
    if name == "velocity_ramp":
        return "edit_engine.render._chain_velocity"
    if name in PROPERTY_EFFECTS:
        return "edit_engine.keyframes (keyframe layer)"
    return "unknown"


def validate_effect_params(name: str, params: dict[str, Any]) -> None:
    """Enforce the property registry for one effect's params.

    Raises MediaEditError naming the property path on: an undeclared
    param, a kind mismatch, an out-of-range number, an unlisted enum
    value. Reserved keys pass: ``mask`` (a MaskRef dict — the
    regional anchor, resolved by the renderer), ``mask_track``
    (the renderer's tracking switch for that anchor) and
    ``_``-prefixed renderer-internal keys. A param explicitly None
    is accepted only where the declared default is None.
    """
    specs = EFFECT_PROPERTIES.get(name)
    if specs is None:
        return  # unknown effect — the builder lookup raises its error
    for key, value in (params or {}).items():
        if key in ("mask", "mask_track") or key.startswith("_"):
            continue
        spec = specs.get(key)
        path = effect_property_path(name, key)
        if spec is None:
            raise MediaEditError(
                f"effect {name!r} has no property {path}; declared: "
                f"{sorted(specs)}")
        if value is None:
            if spec.default is None:
                continue
            raise MediaEditError(f"{path} may not be null")
        kind = spec.kind
        ok = True
        if kind == "number":
            ok = isinstance(value, (int, float)) \
                and not isinstance(value, bool)
        elif kind == "int":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif kind == "bool":
            ok = isinstance(value, bool)
        elif kind == "string":
            ok = isinstance(value, str)
        elif kind == "expr":
            ok = isinstance(value, (str, int, float)) \
                and not isinstance(value, bool)
        elif kind == "enum":
            ok = isinstance(value, str) \
                and value.strip().lower() in spec.choices
        elif kind == "list":
            ok = isinstance(value, (list, tuple))
        elif kind == "dict":
            ok = isinstance(value, dict)
        if not ok:
            raise MediaEditError(
                f"{path} must be {kind}"
                + (f" (one of {list(spec.choices)})"
                   if kind == "enum" else "")
                + f", got {value!r}")
        if kind in ("number", "int") and (
                spec.lo is not None or spec.hi is not None):
            v = float(value)
            if (spec.lo is not None and v < spec.lo) or (
                    spec.hi is not None and v > spec.hi):
                raise MediaEditError(
                    f"{path}={v} outside the declared range "
                    f"[{spec.lo}, {spec.hi}]")


def effect_properties() -> dict[str, Any]:
    """The whole property registry as comparable data (JSON-safe).

    The brain/QC surface over the registry: per effect, its rung and
    per param its path, kind, default, range, keyframable family and
    doc — the same discipline §1.1e gave the generation backends.
    Deferred effects are listed separately, with their reasons.
    """
    def dump(src: dict[str, dict[str, ParamSpec]], *,
             structural: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, params in src.items():
            out[name] = {
                "rung": effect_rung(name),
                "params": {
                    p: {"path": effect_property_path(name, p),
                        "kind": s.kind, "default": s.default,
                        "min": s.lo, "max": s.hi,
                        "keyframable": s.keyframable,
                        "choices": list(s.choices),
                        "doc": s.doc}
                    for p, s in params.items()},
            }
        return out

    return {"effects": dump(EFFECT_PROPERTIES),
            "keyframe_targets": dump(PROPERTY_EFFECTS),
            "deferred": dict(DEFERRED_EFFECTS),
            "reserved_params": {
                "mask": "a MaskRef dict anchoring the effect to a "
                        "region (imggen.masks.MaskRef); resolved by "
                        "the renderer, which records the mask rung",
                "mask_track": "bool — ask the renderer to track the "
                        "resolved mask across the clip's frames "
                        "(edit_engine.mask_track, translation "
                        "only); the effect record states whether "
                        "the mask tracked, partially tracked or "
                        "held, and why held"}}


def deferred_effect_reason(name: str) -> str | None:
    """The named reason an effect name is a deferral, or None.

    Registry-aware (the §1.5e wiring): while no 3D rung is
    registered, ``camera3d``'s reason is the historical
    :data:`DEFERRED_EFFECTS` text, verbatim. Once a rung serves the
    ``scene3d`` surface, the reason becomes the accurate one — the
    rung exists, but ``camera3d`` is not a 2D chain effect: 3D
    composes as a Scene3D document (:mod:`.scene3d`) or a
    MotionProject ``camera3d`` group, both resolved through the
    same registry. Either way it stays a named refusal — no builder
    is substituted here.
    """
    reason = DEFERRED_EFFECTS.get(name)
    if reason is None or name != "camera3d":
        return reason
    from .scene3d import Scene3DError, registered_renderers, resolve_renderer
    if not registered_renderers():
        return reason
    try:
        renderer, _record = resolve_renderer(surface="scene3d")
    except Scene3DError:
        return reason
    return (f"the 3D rung is chosen ({renderer.name!r} serves "
            f"Scene3D documents through edit_engine.scene3d's "
            f"registry) — but 'camera3d' is not a 2D chain effect "
            f"and has no builder here; 3D composes as a Scene3D "
            f"document or a MotionProject camera3d group")


def _effect_filter(name: str, params: dict[str, Any],
                   ctx: dict[str, Any]) -> str:
    from .timeline import canonical_effect
    name = canonical_effect(name)
    reason = deferred_effect_reason(name)
    if reason is not None:
        raise MediaEditError(
            f"effect {name!r} is deferred, not unknown: {reason}")
    builder = EFFECTS.get(name)
    if builder is None:
        raise MediaEditError(
            f"unknown effect {name!r}; use: {sorted(EFFECTS)}")
    validate_effect_params(name, params or {})
    return builder(params or {}, ctx)


def _ctx(w: int, h: int, fps: int, duration: float, seed: int,
         beats: list[float] | None = None) -> dict[str, Any]:
    return {"w": w, "h": h, "fps": fps, "duration": duration,
            "seed": seed, "beats": beats or []}


def list_effects() -> list[str]:
    """All registered effect names (plus the structural velocity_ramp)."""
    return sorted(EFFECTS) + ["velocity_ramp"]
