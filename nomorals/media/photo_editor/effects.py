"""Layer effects (layer styles) — parse and render.

Parse: ``lfx2`` (a versioned descriptor; authoritative) with
the ``lrFX`` legacy binary layout as the fallback, covering
drop / inner shadows, outer / inner glows, bevel & emboss,
satin, strokes and colour / gradient / pattern overlays.

Render: the CPU reference model — exterior effects (drop
shadows, outer glows, outer bevel) behind the layer, the
layer's own pixels at fill opacity, interior effects
(overlays, satin, inner glows, inner shadows, inside strokes,
inner bevel) over them, outside strokes beneath the layer,
emboss shading over the composited stack, all at layer
opacity. Distances are the exact Euclidean transform for
precise glows / chiselled bevels / wide strokes and the 5 x 5
chamfer for spread / choke / narrow strokes; soft falloffs
are two box passes (a tent), as Photoshop does.

Clean-room reimplementation of the documented model
(photocraft ``effects.rs`` / ``paint.rs`` / ``bevel.rs`` /
``blur.rs`` / ``distance.rs`` / ``composite.rs``), written
from the public description of the behaviour.
"""

from __future__ import annotations

import math

import numpy as np

from .adjust import _spline_eval
from .blend import (
    TEXT_GAMMA,
    composite,
    composite_gamma,
)
from .descriptor import (
    Desc,
    color_from_desc,
    desc_bool,
    desc_desc,
    desc_enum,
    desc_list,
    desc_num,
    desc_text,
    parse_versioned,
)
from .distance import (
    INSIDE_EPS,
    dist_inside,
    dist_inside_chamfer,
    dist_outside,
    dist_outside_chamfer,
)
from .fill import gradient_from_desc
from .gradient import (
    gradient_layout,
    gradient_t_array,
    sample_opacity_array,
    sample_stops_array,
)

__all__ = ["composite_with_effects", "effect_margin",
           "parse_effects", "parse_lfx2", "parse_lrfx"]

MAX_REACH = 512.0
BEVEL_H_EPS = 1e-5
STROKE_CHAMFER_MAX = 20.0

# Effect keys in an lfx2 descriptor: (single key, multi key).
_KINDS = (
    ("DrSh", "dropShadowMulti"),
    ("IrSh", "innerShadowMulti"),
    ("OrGl", ""),
    ("IrGl", ""),
    ("ebbl", ""),
    ("ChFX", ""),
    ("SoFi", "solidFillMulti"),
    ("GrFl", "gradientFillMulti"),
    ("patternFill", ""),
    ("FrFX", "frameFXMulti"),
)

_BLEND_LONG = {
    "normal": "normal", "dissolve": "dissolve",
    "darken": "darken", "multiply": "multiply",
    "colorBurn": "color_burn", "linearBurn": "linear_burn",
    "lighten": "lighten", "screen": "screen",
    "colorDodge": "color_dodge", "linearDodge": "linear_dodge",
    "overlay": "overlay", "softLight": "soft_light",
    "hardLight": "hard_light", "vividLight": "vivid_light",
    "linearLight": "linear_light", "pinLight": "pin_light",
    "hardMix": "hard_mix", "difference": "difference",
    "exclusion": "exclusion", "subtract": "subtract",
    "divide": "divide", "hue": "hue",
    "saturation": "saturation", "color": "color",
    "luminosity": "luminosity", "darkerColor": "darker_color",
    "lighterColor": "lighter_color",
}


def _num(desc: Desc | None, key: str, default: float = 0.0
         ) -> float:
    if desc is None:
        return default
    value = desc_num(desc, key)
    return default if value is None else value


def _pct(desc: Desc, key: str, default: float) -> float:
    return _num(desc, key, default * 100.0) / 100.0


def _bool(desc: Desc | None, key: str,
          default: bool = False) -> bool:
    if desc is None:
        return default
    value = desc_bool(desc, key)
    return default if value is None else value


def _text(desc: Desc | None, key: str) -> str:
    if desc is None:
        return ""
    value = desc_text(desc, key)
    return value if value is not None else ""


def _enum(desc: Desc | None, key: str) -> str:
    if desc is None:
        return ""
    pair = desc_enum(desc, key)
    return pair[1] if pair is not None else ""


def _desc_of(desc: Desc | None, key: str) -> Desc | None:
    if desc is None:
        return None
    return desc_desc(desc, key)


def _blend_from_id(value: str) -> str:
    if value in _BLEND_LONG:
        return _BLEND_LONG[value]
    from .blend import mode_name

    return mode_name(value)


def _blend_of(desc: dict, key: str, default: str) -> str:
    raw = _enum(desc, key)
    if not raw:
        return default
    return _blend_from_id(raw)


def _color(desc: Desc, key: str) -> tuple[float, float, float]:
    colour = _desc_of(desc, key)
    if colour is None:
        return (0.0, 0.0, 0.0)
    rgb = color_from_desc(colour)
    return rgb if rgb is not None else (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- contours

def contour_from_desc(desc: Desc | None) -> dict | None:
    """A contour dict from a descriptor, or None for Linear."""
    points: list[tuple[float, float]] = []
    corners: list[bool] = []
    if desc is None:
        return None
    raw = desc_list(desc, "Crv ")
    if raw is not None:
        for item in raw:
            if not isinstance(item, Desc):
                continue
            points.append((_num(item, "Hrzn", 0.0) / 255.0,
                           _num(item, "Vrtc", 0.0) / 255.0))
            corners.append(desc_bool(item, "Cnty") is False)
    name = _text(desc, "Nm  ")
    if not corners or not any(corners):
        corners = []
    linear = (len(points) == 2 and points[0] == (0.0, 0.0)
              and points[1] == (1.0, 1.0))
    if not points or (linear and (not name or name == "Linear")):
        return None
    return {"name": name, "points": points, "corners": corners}


def _curve_lut(points: list[tuple[float, float]],
               corners: list[bool]) -> np.ndarray:
    """A contour's 4096-entry LUT: the natural cubic spline
    through its points, split at corner points (each segment
    splined on its own, later segments overwriting earlier
    ones from the corner on)."""
    size = 4096
    ident = np.linspace(0.0, 1.0, size)
    if not points:
        return ident
    if not corners or not any(corners):
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        return _spline_eval(list(zip(xs.tolist(),
                                 ys.tolist(), strict=True)),
                            ident)
    pts = sorted(zip(points, corners, strict=True),
                 key=lambda pc: pc[0][0])
    dedup: list[tuple[tuple[float, float], bool]] = []
    for pc in pts:
        if dedup and abs(pc[0][0] - dedup[-1][0][0]) < 1e-6:
            continue
        dedup.append(pc)
    out = np.array([], dtype=np.float64)
    start = 0
    last = len(dedup) - 1
    for i in range(1, last + 1):
        corner = dedup[i][1]
        if not (corner or i == last):
            continue
        seg = dedup[start:i + 1]
        xs = np.array([p[0][0] for p in seg])
        ys = np.array([p[0][1] for p in seg])
        lut = _spline_eval(list(zip(xs.tolist(),
                                  ys.tolist(), strict=True)),
                           ident)
        if out.size == 0:
            out = lut
        else:
            start_in = seg[0][0][0]
            mask = ident >= start_in
            out = np.where(mask, lut, out)
        start = i
    if out.size == 0:
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        return _spline_eval(list(zip(xs.tolist(),
                                 ys.tolist(), strict=True)),
                            ident)
    return out


def contour_lut(contour: dict | None) -> np.ndarray | None:
    """A contour's LUT, or None for Linear (identity)."""
    if contour is None:
        return None
    return _curve_lut(contour["points"], contour["corners"])


def _lut_at(lut: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Interpolated lookup in a 0..=1 table."""
    x = np.clip(v, 0.0, 1.0) * (len(lut) - 1)
    i = np.floor(x).astype(int)
    j = np.minimum(i + 1, len(lut) - 1)
    f = x - i
    return lut[i] * (1.0 - f) + lut[j] * f


def _ranged_lut(contour: dict | None, range_: float
                ) -> np.ndarray | None:
    """A contour applied over a Range
    (``contour(min(v / range, 1))``); None = identity."""
    base = contour_lut(contour)
    r = min(max(range_, 0.01), 1.0) if math.isfinite(range_) \
        else 1.0
    if r >= 0.999:
        return base
    n = 4096
    v = (np.arange(n) / (n - 1) / r)
    v = np.minimum(v, 1.0)
    if base is None:
        return v
    return _lut_at(base, v)


def _satin_lut(contour: dict | None) -> np.ndarray | None:
    """Satin's coverage transfer: the contour read from its far
    end, ``|c(1 - v) - c(1)|``."""
    lut = contour_lut(contour)
    if lut is None:
        return None
    end = lut[-1]
    return np.abs(lut[::-1] - end)


def _bevel_contour_lut(contour: dict | None) -> np.ndarray:
    """The bevel Contour element's height transfer: the
    contour alone (its Range does not stretch it)."""
    lut = contour_lut(contour)
    if lut is None:
        return np.linspace(0.0, 1.0, 4096)
    return lut


# ------------------------------------------------------------------- parse

def _common(desc: Desc, default_blend: str,
            default_opacity: float) -> dict:
    return {
        "enabled": _bool(desc, "enab", True),
        "blend": _blend_of(desc, "Md  ", default_blend),
        "opacity": _pct(desc, "Opct", default_opacity),
    }


def _parse_shadow(desc: Desc, inner: bool) -> dict:
    common = _common(desc, "multiply", 0.75)
    return {
        "kind": "inner_shadow" if inner else "drop_shadow",
        "enabled": common["enabled"],
        "blend": common["blend"],
        "opacity": common["opacity"],
        "color": _color(desc, "Clr "),
        "angle": _num(desc, "lagl", 120.0),
        "use_global_light": _bool(desc, "uglg", True),
        "distance": _num(desc, "Dstn", 5.0),
        "spread": _pct(desc, "Ckmt", 0.0),
        "size": _num(desc, "blur", 5.0),
        "contour": contour_from_desc(_desc_of(desc, "TrnS")),
        "anti_alias": _bool(desc, "AntA"),
        "noise": _pct(desc, "Nose", 0.0),
        "knocks_out": (not inner
                       and _bool(desc, "layerConceals", True)),
    }


def _paint_from_desc(desc: Desc) -> dict:
    """An effect paint: colour, gradient or pattern."""
    if _desc_of(desc, "Grad") is not None:
        return {"kind": "gradient",
                "gradient": gradient_from_desc(desc)}
    pattern = _desc_of(desc, "Ptrn")
    if pattern is not None:
        return {"kind": "pattern",
                "name": _text(pattern, "Nm  "),
                "id": _text(pattern, "Idnt"),
                "scale": _pct(desc, "Scl ", 1.0)}
    return {"kind": "color", "rgb": _color(desc, "Clr ")}


def _parse_glow(desc: Desc, inner: bool) -> dict:
    common = _common(desc, "screen", 0.75)
    paint = _paint_from_desc(desc)
    return {
        "kind": "inner_glow" if inner else "outer_glow",
        "enabled": common["enabled"],
        "blend": common["blend"],
        "opacity": common["opacity"],
        "paint": paint,
        "technique": ("precise" if _enum(desc, "GlwT") == "PrBL"
                      else "softer"),
        "spread": _pct(desc, "Ckmt", 0.0),
        "size": _num(desc, "blur", 5.0),
        "contour": contour_from_desc(_desc_of(desc, "TrnS")),
        "anti_alias": _bool(desc, "AntA"),
        "range": _pct(desc, "Inpr", 0.5),
        "jitter": _pct(desc, "ShdN", 0.0),
        "noise": _pct(desc, "Nose", 0.0),
        "source": ("center" if inner
                   and _enum(desc, "glwS") == "SrcC" else "edge"),
    }


def _parse_bevel(desc: Desc) -> dict:
    style = {"OtrB": "outer", "Embs": "emboss",
             "PlEb": "pillow", "strokeEmboss": "stroke"}.get(
                 _enum(desc, "bvlS"), "inner")
    technique = {"PrBL": "chisel_hard",
                 "Slmt": "chisel_soft"}.get(_enum(desc, "bvlT"),
                                            "smooth")
    contour_desc = _desc_of(desc, "MpgS")
    contour = None
    contour_range = 0.5
    contour_aa = False
    if _bool(desc, "useShape") and contour_desc is not None:
        contour = contour_from_desc(contour_desc)
        contour_range = _pct(desc, "Inpr", 0.5)
        contour_aa = _bool(desc, "AntA")
    hglc = _desc_of(desc, "hglC")
    highlight_color = color_from_desc(hglc) \
        if hglc is not None else None
    if highlight_color is None:
        highlight_color = (1.0, 1.0, 1.0)
    texture = None
    if _bool(desc, "useTexture"):
        pattern = _desc_of(desc, "Ptrn")
        texture = {
            "name": _text(pattern, "Nm  "),
            "id": _text(pattern, "Idnt"),
            "scale": _pct(desc, "Scl ", 1.0),
            "depth": _pct(desc, "textureDepth", 1.0),
            "invert": _bool(desc, "InvT"),
            "link": _bool(desc, "Lnkd", True),
            "phase": (0.0, 0.0),
        }
    return {
        "kind": "bevel",
        "enabled": _bool(desc, "enab", True),
        "style": style,
        "technique": technique,
        "depth": _pct(desc, "srgR", 1.0),
        "up": _enum(desc, "bvlD") != "Out ",
        "size": _num(desc, "blur", 5.0),
        "soften": _num(desc, "Sftn", 0.0),
        "angle": _num(desc, "lagl", 120.0),
        "altitude": _num(desc, "Lald", 30.0),
        "use_global_light": _bool(desc, "uglg", True),
        "gloss_contour": contour_from_desc(_desc_of(desc, "TrnS")),
        "gloss_anti_alias": _bool(desc, "antialiasGloss"),
        "highlight": {
            "blend": _blend_of(desc, "hglM", "screen"),
            "opacity": _pct(desc, "hglO", 0.75)},
        "highlight_color": highlight_color,
        "shadow": {
            "blend": _blend_of(desc, "sdwM", "multiply"),
            "opacity": _pct(desc, "sdwO", 0.75)},
        "shadow_color": _color(desc, "sdwC"),
        "contour": contour,
        "contour_range": contour_range,
        "contour_anti_alias": contour_aa,
        "texture": texture,
    }


def _parse_satin(desc: Desc) -> dict:
    common = _common(desc, "multiply", 0.5)
    return {
        "kind": "satin",
        "enabled": common["enabled"],
        "blend": common["blend"],
        "opacity": common["opacity"],
        "color": _color(desc, "Clr "),
        "angle": _num(desc, "lagl", 120.0),
        "distance": _num(desc, "Dstn", 11.0),
        "size": _num(desc, "blur", 14.0),
        "invert": _bool(desc, "Invr"),
        "contour": contour_from_desc(_desc_of(desc, "MpgS")),
        "anti_alias": _bool(desc, "AntA"),
    }


def _parse_stroke(desc: Desc) -> dict:
    common = _common(desc, "normal", 1.0)
    paint = _paint_from_desc(desc)
    if _enum(desc, "PntT") == "Ptrn":
        pattern = _desc_of(desc, "Ptrn")
        paint = {"kind": "pattern",
                 "name": _text(pattern, "Nm  "),
                 "id": _text(pattern, "Idnt"),
                 "scale": _pct(desc, "Scl ", 1.0)}
    position = {"InsF": "inside", "CtrF": "center"}.get(
        _enum(desc, "Styl"), "outside")
    return {
        "kind": "stroke",
        "enabled": common["enabled"],
        "blend": common["blend"],
        "opacity": common["opacity"],
        "size": _num(desc, "Sz  ", 3.0),
        "position": position,
        "paint": paint,
        "overprint": _bool(desc, "overprint"),
    }


def _parse_one(single: str, desc: Desc) -> dict | None:
    if single == "DrSh":
        return _parse_shadow(desc, inner=False)
    if single == "IrSh":
        return _parse_shadow(desc, inner=True)
    if single == "OrGl":
        return _parse_glow(desc, inner=False)
    if single == "IrGl":
        return _parse_glow(desc, inner=True)
    if single == "ebbl":
        return _parse_bevel(desc)
    if single == "ChFX":
        return _parse_satin(desc)
    if single == "SoFi":
        common = _common(desc, "normal", 1.0)
        return {"kind": "color_overlay",
                "enabled": common["enabled"],
                "blend": common["blend"],
                "opacity": common["opacity"],
                "color": _color(desc, "Clr ")}
    if single == "GrFl":
        common = _common(desc, "normal", 1.0)
        return {"kind": "gradient_overlay",
                "enabled": common["enabled"],
                "blend": common["blend"],
                "opacity": common["opacity"],
                "gradient": gradient_from_desc(desc)}
    if single == "patternFill":
        common = _common(desc, "normal", 1.0)
        pattern = _desc_of(desc, "Ptrn")
        return {"kind": "pattern_overlay",
                "enabled": common["enabled"],
                "blend": common["blend"],
                "opacity": common["opacity"],
                "name": _text(pattern, "Nm  "),
                "id": _text(pattern, "Idnt"),
                "scale": _pct(desc, "Scl ", 1.0),
                "link": _bool(desc, "Lnkd", True),
                "angle": _num(desc, "Angl", 0.0),
                "phase": (0.0, 0.0)}
    if single == "FrFX":
        return _parse_stroke(desc)
    return None


def effects_from_descriptor(desc: Desc) -> dict:
    """(master switch, effects) of an effects descriptor."""
    master = _bool(desc, "masterFXSwitch", True)
    items: list[dict] = []
    for single, multi in _KINDS:
        if multi:
            raw = desc_list(desc, multi)
            if raw is not None:
                for item in raw:
                    if isinstance(item, Desc):
                        parsed = _parse_one(single, item)
                        if parsed is not None:
                            items.append(parsed)
                continue
        sub = _desc_of(desc, single)
        if sub is not None:
            parsed = _parse_one(single, sub)
            if parsed is not None:
                items.append(parsed)
    return {"master": master, "items": items,
            "scale": _pct(desc, "Scl ", 1.0)}


def parse_lfx2(data: bytes) -> dict | None:
    """An ``lfx2`` block: 4-byte version + versioned
    descriptor -> an effects dict, or None."""
    try:
        desc, _ = parse_versioned(data, 4)
    except Exception:
        return None
    return effects_from_descriptor(desc)


def _be16(data: bytes, at: int) -> int | None:
    if at + 2 > len(data):
        return None
    return int.from_bytes(data[at:at + 2], "big")


def _be32(data: bytes, at: int) -> int | None:
    if at + 4 > len(data):
        return None
    return int.from_bytes(data[at:at + 4], "big")


def _fixed(value: int) -> float:
    return value / 65536.0


def parse_lrfx(data: bytes) -> dict | None:
    """The legacy ``lrFX`` layout: count + per-effect
    ``8BIM`` records. Only shadows, glows and colour overlays
    carry enough in the legacy layout; bevels, satin, strokes
    and gradient overlays live in ``lfx2`` alone."""
    count = _be16(data, 2)
    if count is None:
        return None
    items: list[dict] = []
    at = 4
    for _ in range(count):
        if data[at:at + 4] != b"8BIM":
            return None
        key = bytes(data[at + 4:at + 8])
        size = _be32(data, at + 8)
        if size is None:
            return None
        b = at + 12

        def color_at(offset: int) -> tuple[float, float, float]:
            def comp(k: int) -> float:
                v = _be16(data, offset + 2 + k * 2)
                return (v >> 8) / 255.0 if v is not None else 0.0
            return (comp(0), comp(1), comp(2))

        def blend_at(offset: int) -> str:
            raw = bytes(data[offset + 4:offset + 8])
            return _blend_from_id(raw.decode("latin-1"))

        if key in (b"dsdw", b"isdw"):
            blur = _be32(data, b + 4)
            angle = _be32(data, b + 12)
            distance = _be32(data, b + 16)
            if blur is None or angle is None or distance is None:
                return None
            opacity = data[b + 40] / 255.0 if b + 40 < len(data) \
                else 0.75
            items.append({
                "kind": ("inner_shadow" if key == b"isdw"
                         else "drop_shadow"),
                "enabled": data[b + 38] != 0 if b + 38 < len(data)
                else True,
                "blend": blend_at(b + 30),
                "opacity": opacity,
                "color": color_at(b + 20),
                "angle": _fixed(angle),
                "use_global_light": data[b + 39] != 0
                if b + 39 < len(data) else True,
                "distance": _fixed(distance),
                "spread": 0.0,
                "size": _fixed(blur),
                "contour": None,
                "anti_alias": False,
                "noise": 0.0,
                "knocks_out": key == b"dsdw",
            })
        elif key in (b"oglw", b"iglw"):
            blur = _be32(data, b + 4)
            if blur is None:
                return None
            items.append({
                "kind": ("inner_glow" if key == b"iglw"
                         else "outer_glow"),
                "enabled": data[b + 30] != 0 if b + 30 < len(data)
                else True,
                "blend": blend_at(b + 22),
                "opacity": data[b + 31] / 255.0
                if b + 31 < len(data) else 0.75,
                "paint": {"kind": "color",
                          "rgb": color_at(b + 12)},
                "technique": "softer",
                "spread": 0.0,
                "size": _fixed(blur),
                "contour": None,
                "anti_alias": False,
                "range": 0.5,
                "jitter": 0.0,
                "noise": 0.0,
                "source": "edge",
            })
        elif key == b"sofi":
            items.append({
                "kind": "color_overlay",
                "enabled": data[b + 23] != 0 if b + 23 < len(data)
                else True,
                "blend": blend_at(b + 4),
                "opacity": data[b + 22] / 255.0
                if b + 22 < len(data) else 1.0,
                "color": color_at(b + 12),
            })
        at = b + size
    return {"master": True, "items": items, "scale": 1.0}


def parse_effects(raw_blocks: dict) -> dict | None:
    """A layer's effects from its raw blocks (``lfx2`` first,
    ``lrFX`` as the fallback), or None when it has none."""
    data = raw_blocks.get("lfx2")
    if data is not None:
        parsed = parse_lfx2(bytes(data))
        if parsed is not None:
            return parsed
    data = raw_blocks.get("lrFX")
    if data is not None:
        return parse_lrfx(bytes(data))
    return None


def effect_margin(effects: dict | None) -> int:
    """Pixels an effect stack can reach beyond the layer's
    shape (bounded by MAX_REACH, +2)."""
    if not effects or not effects.get("master", True):
        return 0
    margin = 0.0
    out_stroke = 0.0
    for effect in effects.get("items", []):
        if not effect.get("enabled", True):
            continue
        if effect.get("kind") == "stroke":
            out_stroke = max(out_stroke,
                             _stroke_widths(effect)[1])
    for effect in effects.get("items", []):
        if not effect.get("enabled", True):
            continue
        kind = effect.get("kind")
        if kind in ("drop_shadow", "inner_shadow"):
            reach = effect["distance"] + effect["size"]
        elif kind in ("outer_glow", "inner_glow") or kind == "stroke":
            reach = effect["size"]
        elif kind == "satin":
            reach = max(effect["distance"], 1.0) + effect["size"]
        elif kind == "bevel":
            reach = effect["size"] + effect["soften"]
            if effect.get("style") == "stroke":
                # A stroke bevel is bevelled on the stroke
                # band, so it reaches the band's outside
                # width past the shape plus its own size.
                reach += out_stroke
        else:
            reach = 0.0
        if math.isfinite(reach):
            margin = max(margin, reach)
    return int(math.ceil(min(max(margin, 0.0), MAX_REACH))) + 2


# ------------------------------------------------------------------ blur

def _box_geom(w: float) -> tuple[int, float, float]:
    """Box geometry for a (fractional) width: (``r``, end-tap
    weight ``f``, ``1 / width``) — the box taps are ``r - 1``
    full ones each side of the centre plus two end taps at
    ``f``."""
    if math.isnan(w):
        w = 1.0
    w = min(max(w, 1.0), MAX_REACH)
    half = w / 2.0
    r = int(math.ceil(half - 0.5))
    if r < 0:
        r = 0
    f = min(max(half - (r - 0.5), 0.0), 1.0)
    return r, f, 1.0 / w


def _box_kernel(arr: np.ndarray, w: float) -> np.ndarray:
    """One box pass evaluated at every index of ``arr`` that
    has ``r`` neighbours each side: out[j] = norm * (sum of
    the ``2r - 1`` inner taps + ``f`` * the two end taps).
    Length ``len(arr) - 2r``; ``r == 0`` is the identity."""
    r, f, norm = _box_geom(w)
    if r == 0:
        return arr.copy()
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    inner = cs[2 * r:len(arr)] - cs[1:len(arr) - 2 * r + 1]
    # inner[j] = sum(arr[j+1 : j+2r]) for j in 0..len-2r-1.
    out = (inner + f * arr[:len(arr) - 2 * r]
           + f * arr[2 * r:]) * norm
    return out


def _tent_line(src: np.ndarray, w: float) -> np.ndarray:
    """Two box passes along a line, exactly the convolution
    with the tent kernel (zero outside): the first pass is
    evaluated ``r`` samples beyond each end so the second
    sees what lies there."""
    r, _, _ = _box_geom(w)
    if r == 0:
        return src.copy()
    padded = np.concatenate([np.zeros(2 * r), src,
                             np.zeros(2 * r)])
    mid = _box_kernel(padded, w)
    return _box_kernel(mid, w)


def _tent(arr: np.ndarray, w: float) -> np.ndarray:
    """Two box passes of width ``w`` over both axes."""
    if not math.isfinite(w):
        return arr.copy()
    out = np.apply_along_axis(_tent_line, 1, arr, w)
    out = np.apply_along_axis(_tent_line, 0, out, w)
    return out


def _shift(arr: np.ndarray, dx: int, dy: int,
           fill: float) -> np.ndarray:
    """An integer shift (whole pixels, as effect offsets are
    rounded); outside reads ``fill``."""
    h, w = arr.shape
    out = np.full_like(arr, fill)
    x0 = max(0, dx)
    x1 = min(w, w + dx)
    y0 = max(0, dy)
    y1 = min(h, h + dy)
    if x1 <= x0 or y1 <= y0:
        return out
    out[y0:y1, x0:x1] = arr[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def _dilate(alpha: np.ndarray, r: float) -> np.ndarray:
    """Grows the shape by ``r`` pixels (anti-aliased),
    keeping the original soft edge. Distances use the 5 x 5
    chamfer metric."""
    if r <= 0.0:
        return alpha.copy()
    d = dist_outside_chamfer(alpha)
    out = alpha.copy()
    inside = d < 0.0
    out = np.where(inside, np.minimum(alpha + r, 1.0),
                   np.maximum(alpha, np.clip(r + 0.5 - d,
                                             0.0, 1.0)))
    return out


def _hash_noise_array(shape: tuple[int, int], x0: int,
                      y0: int) -> np.ndarray:
    """Deterministic per-pixel noise in 0..=1, hashed from
    document coordinates (stable across renders)."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    xs = (xs + x0).astype(np.int64)
    ys = (ys + y0).astype(np.int64)
    h = ((xs.astype(np.uint64) * np.uint64(0x8DA6B343))
         ^ (ys.astype(np.uint64) * np.uint64(0xD8163841))
         ^ np.uint64(0x9E3779B9)).astype(np.uint32)
    h = h ^ (h >> np.uint32(15))
    h = (h.astype(np.uint64) * np.uint64(0x2C1B3C6D)
         ).astype(np.uint32)
    h = h ^ (h >> np.uint32(12))
    return (h & np.uint32(0xFFFF)).astype(np.float64) / 65536.0


def _apply_noise(coverage: np.ndarray, amount: float,
                 x0: int, y0: int) -> np.ndarray:
    """Effect noise: monochrome speckle in the coverage,
    applied after the contour, scaled by the coverage so it
    fades out with the effect."""
    if not math.isfinite(amount) or amount <= 0.0:
        return coverage
    noise = _hash_noise_array(coverage.shape, x0, y0)
    out = coverage.copy()
    mask = coverage > 0.0
    out[mask] = np.clip(
        coverage[mask] * (1.0 + (noise[mask] - 0.5) * 2.0
                          * amount), 0.0, 1.0)
    return out


def _offset(angle: float, distance: float) -> tuple[int, int]:
    """Offset of an effect lit from ``angle`` at ``distance``
    (shadow falls away from the light), rounded to whole
    pixels as Photoshop does."""
    a = math.radians(angle)
    return (int(round(-math.cos(a) * distance)),
            int(round(math.sin(a) * distance)))


def _spread_split(size: float, spread: float
                  ) -> tuple[float, float]:
    """Spread / choke of a soft effect: (dilation radius,
    blur width). The dilation is whole pixels (``size x
    spread`` rounded), the blur keeps ``size x (1-spread)``."""
    return (round(size * spread), size * (1.0 - spread))


# ------------------------------------------------------------------- maps

def _shadow_map(shape: np.ndarray, shadow: dict,
                light_angle: float, inner: bool,
                origin: tuple[int, int]) -> np.ndarray:
    angle = light_angle if shadow["use_global_light"] \
        else shadow["angle"]
    dx, dy = _offset(angle, shadow["distance"])
    src = (1.0 - shape) if inner else shape.copy()
    m = _shift(src, dx, dy, 1.0 if inner else 0.0)
    r, bw = _spread_split(shadow["size"], shadow["spread"])
    m = _dilate(m, r)
    m = _tent(m, bw)
    lut = contour_lut(shadow["contour"])
    if lut is not None:
        m = _lut_at(lut, m)
    m = _apply_noise(m, shadow["noise"], origin[0], origin[1])
    if inner:
        m = m * shape
    return m


def _glow_map(shape: np.ndarray, glow: dict, inner: bool,
              origin: tuple[int, int]) -> np.ndarray:
    src = (1.0 - shape) if inner else shape.copy()
    if glow["technique"] == "precise":
        # Coverage falls linearly from 0.5 at the edge to 0 at
        # ``size + 1`` px, solid within the spread / choke.
        d = dist_inside(shape) if inner else dist_outside(src)
        solid = glow["size"] * glow["spread"]
        soft = max(glow["size"] + 1.0 - solid, 1e-3)
        dist = d + 0.5
        m = np.where(dist <= solid, 1.0,
                     0.5 * np.clip(1.0 - (dist - solid) / soft,
                                   0.0, 1.0))
    else:
        r, bw = _spread_split(glow["size"], glow["spread"])
        m = _dilate(src, r)
        m = _tent(m, bw)
    lut = _ranged_lut(glow["contour"], glow["range"])
    if lut is not None:
        m = _lut_at(lut, m)
    if inner and glow["source"] == "center":
        m = 1.0 - m
    m = _apply_noise(m, glow["noise"], origin[0], origin[1])
    if inner:
        m = m * shape
    return m


def _satin_map(shape: np.ndarray, satin: dict) -> np.ndarray:
    distance = max(satin["distance"], 1.0)
    dx, dy = _offset(satin["angle"], distance)
    a = _shift(shape, dx, dy, 0.0)
    b = _shift(shape, -dx, -dy, 0.0)
    a = _tent(a, satin["size"])
    b = _tent(b, satin["size"])
    m = np.abs(a - b)
    lut = _satin_lut(satin["contour"])
    if lut is not None:
        m = _lut_at(lut, m)
    if satin["invert"]:
        # Photoshop applies the contour to the raw map and
        # inverts the result: the inverted file's map matches
        # 5x better this way (0.032 vs 0.235 mean map diff)
        # than inverting first and then reading the contour.
        m = 1.0 - m
    return m * shape


def _bevel_geom(bevel: dict) -> dict:
    size = max(bevel["size"], 1.0)
    emboss = bevel["style"] in ("emboss", "pillow")
    paint = {"outer": "outer", "emboss": "both",
             "pillow": "both"}.get(bevel["style"], "inner")
    width = math.ceil(size / 2.0) if emboss else size
    smooth = bevel["technique"] == "smooth"
    sign = 1.0 if bevel["up"] else -1.0
    depth = bevel["depth"] * (width if smooth else size / 2.0) \
        * sign
    return {
        "paint": paint,
        "width": width,
        "depth": depth,
        "pillow": bevel["style"] == "pillow",
        "chisel_soft": (max(size / 4.0, 1.0)
                        if bevel["technique"] == "chisel_soft"
                        else 0.0),
    }


def _bevel_chisel_h(paint: str, size: float, din: np.ndarray,
                    dout: np.ndarray, soft: bool) -> np.ndarray:
    """Chiselled height from the inside / outside distances."""
    ramp = size + 1.0
    if paint == "outer" and not soft:
        top = (size + 2.0) / ramp
        return np.where(dout > 0.0,
                        np.clip((size + 1.5 - dout) / ramp,
                                0.0, top), top)
    if paint == "outer":
        return np.where(dout > 0.0,
                        1.0 - np.clip((dout + 0.5) / ramp,
                                      0.0, 1.0),
                        (size + 2.0) / ramp)
    if paint == "both":
        half = size / 2.0 + 0.5
        return np.where(din > 0.0,
                        0.5 + 0.5 * np.clip(din / half, 0.0, 1.0),
                        0.5 - 0.5 * np.clip(dout / half, 0.0, 1.0))
    if soft:
        return np.where(din > 0.0,
                        np.clip((din + 1.5) / ramp, 0.0,
                                (size + 2.0) / ramp), 0.0)
    return np.where(din > 0.0,
                    np.clip((din + 0.5) / ramp, 0.0, 1.0), 0.0)


def _bevel_height(shape: np.ndarray, bevel: dict,
                  geom: dict) -> np.ndarray:
    size = max(bevel["size"], 1.0)
    if bevel["technique"] == "smooth":
        h = _tent(shape.copy(), geom["width"])
    else:
        din = dist_inside(shape)
        dout = dist_outside(shape)
        h = _bevel_chisel_h(geom["paint"], size, din, dout,
                            geom["chisel_soft"] > 0.0)
        if geom["chisel_soft"] > 0.0 \
                and geom["paint"] == "both":
            h = _tent(h, geom["chisel_soft"])
    contour = bevel["contour"]
    if contour is not None:
        h = _lut_at(_bevel_contour_lut(contour), h)
    if bevel["soften"] >= 1.0:
        h = _tent(h, bevel["soften"])
    return h


def _bevel_maps(shape: np.ndarray, bevel: dict,
                light: tuple[float, float]
                ) -> tuple[list[np.ndarray], str]:
    """Bevel highlight and shadow maps: [hi, sh]."""
    geom = _bevel_geom(bevel)
    hmap = _bevel_height(shape, bevel, geom)
    angle, altitude = (light if bevel["use_global_light"]
                       else (bevel["angle"], bevel["altitude"]))
    sa, ca = math.sin(math.radians(angle)), math.cos(
        math.radians(angle))
    se, ce = math.sin(math.radians(altitude)), math.cos(
        math.radians(altitude))
    light_v = np.array([ca * ce, -sa * ce, se])
    gloss = contour_lut(bevel["gloss_contour"])

    def height_at(dx: int, dy: int) -> np.ndarray:
        return _shift(hmap, dx, dy, 0.0)

    def shade_into(depth: float, region_kind: int
                   ) -> tuple[np.ndarray, np.ndarray]:
        # The height gradient scaled by this half's depth.
        # ``_shift(h, 1, 0)`` samples h[x - 1] (a content
        # shift), so the forward difference is
        # height_at(-1, 0) - height_at(1, 0) — the Rust
        # samples h[x + 1] - h[x - 1] directly.
        gx = (height_at(-1, 0) - height_at(1, 0)) * 0.5 * depth
        gy = (height_at(0, -1) - height_at(0, 1)) * 0.5 * depth
        nx = -gx
        ny = -gy
        length = np.sqrt(nx * nx + ny * ny + 1.0)
        shade = (nx * light_v[0] + ny * light_v[1]
                 + light_v[2]) / length
        # The gloss contour remaps the shading ``n . light``
        # before it splits into highlight and shadow.
        # The contour maps the shading, then the flat-face
        # shade itself (not its mapped value) is the split:
        # measured on bevel-inner-gloss-ramp-peak, the flat
        # face (shade = sin(altitude) = 0.5, contour peak
        # 1.0) takes the FULL highlight, which the
        # c(shade) - c(se) form cannot give. A Linear
        # contour still changes nothing either way.
        k = (_lut_at(gloss, shade) - se if gloss is not None
             else shade - se)
        # region 0 = inside (x the shape's alpha), 1 = under
        # the layer's edge and outside.
        if region_kind == 0:
            region = shape
        else:
            on_bevel = hmap > BEVEL_H_EPS
            if geom["paint"] == "both" \
                    or (geom["paint"] == "outer"
                        and bevel["technique"] != "smooth"):
                neighbours = (
                    _shift(hmap, 1, 0, 0.0) > BEVEL_H_EPS)
                neighbours = neighbours | (
                    _shift(hmap, -1, 0, 0.0) > BEVEL_H_EPS)
                neighbours = neighbours | (
                    _shift(hmap, 0, 1, 0.0) > BEVEL_H_EPS)
                neighbours = neighbours | (
                    _shift(hmap, 0, -1, 0.0) > BEVEL_H_EPS)
                on_bevel = on_bevel | neighbours
            region = ((shape < 1.0 - INSIDE_EPS)
                      & on_bevel).astype(np.float64)
        hi = np.where(k > 0.0,
                      np.clip(k / max(1.0 - se, 1e-3), 0.0, 1.0)
                      * region, 0.0)
        sh = np.where(k <= 0.0,
                      np.clip(-k / max(se, 1e-3), 0.0, 1.0)
                      * region, 0.0)
        return hi, sh

    if geom["paint"] in ("inner", "outer"):
        hi, sh = shade_into(
            geom["depth"],
            1 if geom["paint"] == "outer" else 0)
        return [hi, sh], geom["paint"]
    hi, sh = shade_into(geom["depth"], 0)
    depth_out = -geom["depth"] if geom["pillow"] \
        else geom["depth"]
    ho, so = shade_into(depth_out, 1)

    def mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        s = np.clip(shape, 0.0, 1.0)
        return np.where(s > INSIDE_EPS, a, 0.0) \
            + b * (1.0 - s)

    return [mix(hi, ho), mix(sh, so)], geom["paint"]


# ------------------------------------------------------------------ paint

def _paint_color(dst: np.ndarray, coverage: np.ndarray,
                 rgb: tuple[float, float, float], blend: str,
                 opacity: float,
                 origin: tuple[int, int]) -> None:
    """Composite ``rgb x coverage`` into ``dst`` in place."""
    a = coverage * opacity
    if not np.any(a > 0.0):
        return
    if blend == "dissolve":
        keep = _hash_noise_array(
            coverage.shape, origin[0], origin[1]) < a
        src = np.zeros_like(dst)
        src[..., 0] = rgb[0]
        src[..., 1] = rgb[1]
        src[..., 2] = rgb[2]
        src[..., 3] = np.where(keep, 1.0, 0.0)
        dst[:, :, :] = composite("normal", dst, src, 1.0)
        return
    src = np.zeros_like(dst)
    src[..., 0] = rgb[0]
    src[..., 1] = rgb[1]
    src[..., 2] = rgb[2]
    src[..., 3] = a
    dst[:, :, :] = composite(blend, dst, src, 1.0)


def _sample_paint_rgba(paint: dict, shape_bounds: tuple[int,
                                                        int,
                                                        int,
                                                        int],
                       anchor: tuple[float, float],
                       big: tuple[int, int, int, int],
                       height: int, width: int,
                       patterns: dict | None) -> np.ndarray | None:
    """A paint's RGBA over the ``big`` region (document
    coords), or None when a pattern is missing."""
    kind = paint.get("kind")
    if kind == "color":
        rgb = paint["rgb"]
        out = np.zeros((height, width, 4), dtype=np.float64)
        out[..., 0] = rgb[0]
        out[..., 1] = rgb[1]
        out[..., 2] = rgb[2]
        out[..., 3] = 1.0
        return out
    if kind == "gradient":
        gradient = paint["gradient"]
        angle, scale, offset = gradient_layout(
            gradient.get("style", "linear"),
            gradient.get("angle", 90.0),
            gradient.get("scale", 1.0),
            gradient.get("offset", (0.0, 0.0)), shape_bounds)
        t = gradient_t_array(
            gradient.get("style", "linear"), angle, scale,
            gradient.get("reverse", False), offset, shape_bounds,
            width, height, origin=(big[0], big[1]))
        rgb = sample_stops_array(gradient.get("stops", []), t)
        alpha = sample_opacity_array(
            gradient.get("opacity_stops", []), t)
        out = np.zeros((height, width, 4), dtype=np.float64)
        out[..., :3] = rgb
        out[..., 3] = alpha
        return out
    if kind == "pattern":
        if not patterns:
            return None
        tile = patterns.get(paint.get("id", ""))
        if tile is None:
            return None
        scale = max(paint.get("scale", 1.0), 1e-3)
        th, tw = tile.shape[:2]
        tw = max(1, int(round(tw * scale)))
        th = max(1, int(round(th * scale)))
        ys = np.arange(height)
        xs = np.arange(width)
        # Patterns tile from the anchor (the effects
        # reference point, else the layer's top-left).
        src_y = ((ys + big[1] - int(anchor[1])) % th
                 * tile.shape[0] // th)
        src_x = ((xs + big[0] - int(anchor[0])) % tw
                 * tile.shape[1] // tw)
        return tile[np.ix_(src_y, src_x)]
    return None


def _paint_fx(dst: np.ndarray, coverage: np.ndarray,
              paint: dict, shape_bounds: tuple[int, int, int,
                                               int],
              anchor: tuple[float, float],
              big: tuple[int, int, int, int], blend: str,
              opacity: float, origin: tuple[int, int],
              patterns: dict | None) -> None:
    """Paint through ``coverage`` with a colour / gradient /
    pattern paint."""
    rgba = _sample_paint_rgba(paint, shape_bounds, anchor, big,
                              dst.shape[0], dst.shape[1],
                              patterns)
    if rgba is None:
        return
    a = coverage * opacity
    if not np.any(a > 0.0):
        return
    src = rgba.copy()
    src[..., 3] = rgba[..., 3] * a
    if blend == "dissolve":
        keep = _hash_noise_array(
            coverage.shape, origin[0], origin[1]) < src[..., 3]
        src[..., 3] = np.where(keep, 1.0, 0.0)
        dst[:, :, :] = composite("normal", dst, src, 1.0)
        return
    dst[:, :, :] = composite(blend, dst, src, 1.0)


def _paint_glow(dst: np.ndarray, coverage: np.ndarray,
                glow: dict, shape_bounds: tuple[int, int, int,
                                                int],
                anchor: tuple[float, float],
                big: tuple[int, int, int, int],
                origin: tuple[int, int],
                patterns: dict | None) -> None:
    """A glow's paint over its map. A gradient runs along the
    glow rather than across the canvas: Photoshop colours the
    glow where its strength is ``v`` with the gradient at
    ``1 - v``, and the glow is opaque from ``v = range^2``
    on."""
    paint = glow["paint"]
    if paint.get("kind") != "gradient":
        _paint_fx(dst, coverage, paint, shape_bounds, anchor,
                  big, glow["blend"], glow["opacity"], origin,
                  patterns)
        return
    gradient = paint["gradient"]
    gain = 1.0 / (min(max(glow["range"], 0.01), 1.0) ** 2) \
        if math.isfinite(glow["range"]) else 1.0
    strength = np.minimum(coverage * gain, 1.0)
    # The gradient is sampled at 1 - coverage, not at the
    # canvas position: sample the stops directly.
    stops = gradient.get("stops", [])
    rgb = sample_stops_array(stops, 1.0 - np.clip(coverage,
                                                  0.0, 1.0))
    alpha = sample_opacity_array(
        gradient.get("opacity_stops", []),
        1.0 - np.clip(coverage, 0.0, 1.0))
    src = np.zeros_like(dst)
    src[..., :3] = rgb
    src[..., 3] = alpha * strength * glow["opacity"]
    if glow["blend"] == "dissolve":
        keep = _hash_noise_array(
            coverage.shape, origin[0], origin[1]) < src[..., 3]
        src[..., 3] = np.where(keep, 1.0, 0.0)
        dst[:, :, :] = composite("normal", dst, src, 1.0)
        return
    dst[:, :, :] = composite(glow["blend"], dst, src, 1.0)


# ------------------------------------------------------- the effect pass

def _stroke_widths(stroke: dict) -> tuple[float, float]:
    if stroke["position"] == "outside":
        return (0.0, stroke["size"])
    if stroke["position"] == "inside":
        return (stroke["size"], 0.0)
    return (stroke["size"] / 2.0, stroke["size"] / 2.0)


def _stroke_band_shape(shape: np.ndarray,
                       strokes: list[dict],
                       din: np.ndarray | None,
                       dout: np.ndarray | None,
                       dout_exact: np.ndarray | None
                       ) -> np.ndarray | None:
    """The union of a layer's strokes' painted coverage, or
    None when no stroke reaches outside the shape. A
    stroke-style bevel is bevelled on this band. The band
    keeps the shape's interior at 1.0: the bevel's height
    rises across the band and plateaus inside, so the
    gradient (and the shading) lives on the band alone."""
    if not strokes:
        return None
    band = np.zeros(shape.shape, dtype=np.float64)
    any_out = False
    inside = shape > INSIDE_EPS
    for stroke in strokes:
        in_w, out_w = _stroke_widths(stroke)
        if in_w > 0.0 and din is not None:
            band = np.maximum(
                band, np.where(
                    inside,
                    np.clip(in_w + 0.5 - din, 0.0, 1.0),
                    0.0))
        if out_w > 0.0:
            any_out = True
            exact = out_w >= STROKE_CHAMFER_MAX
            d = dout_exact if exact else dout
            if d is None:
                continue
            band = np.maximum(
                band, np.clip(out_w + 0.5 - d, 0.0, 1.0))
    if not any_out:
        return None
    return band


def _stroke_visible_band(shape: np.ndarray,
                         strokes: list[dict],
                         din: np.ndarray | None,
                         dout: np.ndarray | None,
                         dout_exact: np.ndarray | None
                         ) -> np.ndarray | None:
    """A stroke band's VISIBLE coverage: the outside part
    only where the shape does not cover, the inside part
    where the stroke runs inside. A stroke bevel's shading
    is clipped to this: the height field rises a few pixels
    inside the shape's edge (the tent's spread), but
    Photoshop shades the stroke alone, never the layer's
    interior."""
    if not strokes:
        return None
    band = np.zeros(shape.shape, dtype=np.float64)
    any_out = False
    inside = shape > INSIDE_EPS
    for stroke in strokes:
        in_w, out_w = _stroke_widths(stroke)
        if in_w > 0.0 and din is not None:
            band = np.maximum(
                band, np.where(
                    inside,
                    np.clip(in_w + 0.5 - din, 0.0, 1.0),
                    0.0))
        if out_w > 0.0:
            any_out = True
            exact = out_w >= STROKE_CHAMFER_MAX
            d = dout_exact if exact else dout
            if d is None:
                continue
            band = np.maximum(
                band, np.clip(out_w + 0.5 - d, 0.0, 1.0)
                * (1.0 - shape))
    if not any_out:
        return None
    return band


def _relative_to_shape(m: np.ndarray, shape: np.ndarray
                       ) -> np.ndarray:
    inside = shape > INSIDE_EPS
    return np.where(inside, np.minimum(m / np.maximum(shape,
                                                      1e-9),
                                       1.0), 0.0)


def _knock_out_exterior(coverage: np.ndarray, shape: np.ndarray,
                        fill_opacity: float) -> np.ndarray:
    see_through = 1.0 - min(max(fill_opacity, 0.0), 1.0)
    return coverage * (1.0 - shape * see_through)


def _shadow_see_through(blend: str, fill: float) -> float:
    if blend in ("normal", "dissolve", "pass_through"):
        return 1.0 - min(max(fill, 0.0), 1.0)
    return 1.0


def _blends_inside(effects: dict) -> bool:
    for effect in effects.get("items", []):
        if not effect.get("enabled", True):
            continue
        kind = effect.get("kind")
        if (kind in ("color_overlay", "gradient_overlay",
                     "pattern_overlay", "satin", "inner_glow",
                     "inner_shadow")
                and effect["blend"] != "normal"):
            return True
        if kind == "stroke":
            in_w, _ = _stroke_widths(effect)
            if in_w > 0.0 and effect["blend"] != "normal":
                return True
        if kind == "bevel":
            geom = _bevel_geom(effect)
            if geom["paint"] == "inner" and (
                    effect["highlight"]["blend"] != "normal"
                    or effect["shadow"]["blend"] != "normal"):
                return True
    return False


def _premerges_layer(blend: str, fill: float, effects: dict,
                     blend_interior: bool) -> bool:
    if blend_interior:
        return False
    if blend in ("pass_through", "dissolve"):
        return False
    if blend == "normal":
        return _blends_inside(effects)
    return True


def composite_with_effects(canvas: np.ndarray, rgba: np.ndarray,
                           left: int, top: int, *,
                           blend: str = "normal",
                           opacity: float = 1.0,
                           fill: float = 1.0,
                           effects: dict | None = None,
                           is_text: bool = False,
                           advanced: dict | None = None,
                           patterns: dict | None = None,
                           global_light: tuple[float, float]
                           = (120.0, 30.0)) -> None:
    """Composite one layer with its effects onto ``canvas``
    in place. ``rgba`` is the layer's float RGBA at its own
    rect; ``canvas`` is the full float RGBA canvas."""
    if effects is None or not effects.get("master", True):
        return
    items = [e for e in effects.get("items", [])
             if e.get("enabled", True)]
    if not items:
        return
    advanced = advanced or {}
    blend_interior = advanced.get("blend_interior", False)
    transparency_shapes = advanced.get("transparency_shapes",
                                       True)
    height, width = canvas.shape[:2]
    margin = effect_margin(effects)
    # The region effects paint over: the layer's rect grown
    # by the margin, clipped to the canvas.
    bx0 = max(0, left - margin)
    by0 = max(0, top - margin)
    bx1 = min(width, left + rgba.shape[1] + margin)
    by1 = min(height, top + rgba.shape[0] + margin)
    if bx1 <= bx0 or by1 <= by0:
        return
    big = (bx0, by0, bx1, by1)
    bh, bw = by1 - by0, bx1 - bx0
    # The shape: the layer's alpha over ``big``.
    shape = np.zeros((bh, bw), dtype=np.float64)
    lx0 = max(0, left - bx0)
    ly0 = max(0, top - by0)
    lx1 = min(bw, left + rgba.shape[1] - bx0)
    ly1 = min(bh, top + rgba.shape[0] - by0)
    if lx1 > lx0 and ly1 > ly0:
        shape[ly0:ly1, lx0:lx1] = rgba[
            ly0 - (top - by0):ly1 - (top - by0),
            lx0 - (left - bx0):lx1 - (left - bx0), 3]
    # The content over ``big`` (the layer's pixels).
    content = np.zeros((bh, bw, 4), dtype=np.float64)
    if lx1 > lx0 and ly1 > ly0:
        content[ly0:ly1, lx0:lx1] = rgba[
            ly0 - (top - by0):ly1 - (top - by0),
            lx0 - (left - bx0):lx1 - (left - bx0)]
    sb = (left, top, left + rgba.shape[1],
          top + rgba.shape[0])
    anchor = (float(sb[0]), float(sb[1]))
    origin = (bx0, by0)
    # Stroke distance fields (needed before the effect
    # maps: a stroke-style bevel is bevelled on the
    # stroke band, not on the layer's shape).
    strokes = [e for e in items if e.get("kind") == "stroke"]
    din = dout = dout_exact = None
    if strokes:
        din = dist_inside_chamfer(shape)
        out_widths = [_stroke_widths(s)[1] for s in strokes]
        if any(w > 0.0 and w < STROKE_CHAMFER_MAX
               for w in out_widths):
            dout = dist_outside_chamfer(shape)
        if any(w >= STROKE_CHAMFER_MAX for w in out_widths):
            dout_exact = dist_outside(shape)
    band = _stroke_band_shape(shape, strokes, din, dout,
                              dout_exact)
    visible_band = _stroke_visible_band(shape, strokes, din,
                                        dout, dout_exact)
    # The effect maps, in ``items`` order.
    per: list[list[np.ndarray]] = []
    bevel_paint: list[str] = []
    for effect in items:
        kind = effect.get("kind")
        if kind == "drop_shadow":
            per.append([_shadow_map(shape, effect,
                                    global_light[0], False,
                                    origin)])
            bevel_paint.append("inner")
        elif kind == "inner_shadow":
            per.append([_shadow_map(shape, effect,
                                    global_light[0], True,
                                    origin)])
            bevel_paint.append("inner")
        elif kind == "outer_glow":
            per.append([_glow_map(shape, effect, False, origin)])
            bevel_paint.append("inner")
        elif kind == "inner_glow":
            per.append([_glow_map(shape, effect, True, origin)])
            bevel_paint.append("inner")
        elif kind == "satin":
            per.append([_satin_map(shape, effect)])
            bevel_paint.append("inner")
        elif kind == "bevel":
            if effect.get("style") == "stroke" \
                    and band is not None:
                # Photoshop bevels the stroke itself: the
                # maps are built on the stroke band and
                # painted over it (see below), never on the
                # layer's interior.
                maps, _ = _bevel_maps(band, effect,
                                      global_light)
                per.append(maps)
                bevel_paint.append("stroke")
            else:
                maps, paint = _bevel_maps(shape, effect,
                                          global_light)
                per.append(maps)
                bevel_paint.append(paint)
        else:
            per.append([])
            bevel_paint.append("inner")
    # Exterior: drop shadows, then outer glows, on a copy of
    # the backdrop over ``big``.
    work = canvas[by0:by1, bx0:bx1].copy()
    before = work.copy()
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "drop_shadow":
            continue
        m = per[i][0].copy()
        if effect["knocks_out"]:
            m = _knock_out_exterior(
                m, shape,
                1.0 - _shadow_see_through(blend, fill))
        _paint_color(work, m, effect["color"], effect["blend"],
                     effect["opacity"], origin)
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "outer_glow":
            continue
        m = _knock_out_exterior(per[i][0], shape, fill)
        _paint_glow(work, m, effect, sb, anchor, big, origin,
                    patterns)
    # The layer: content at fill opacity, alpha relative to
    # the shape.
    interior_group = blend_interior and fill < 1.0
    content_fill = 1.0 if interior_group else fill
    lay = content.copy()
    inside = shape > INSIDE_EPS
    if transparency_shapes:
        alpha = np.where(
            inside,
            content_fill * np.minimum(
                np.where(inside, content[..., 3]
                         / np.maximum(shape, 1e-9), 0.0), 1.0),
            0.0)
    else:
        alpha = np.where(inside, content_fill, 0.0)
    lay[..., 3] = alpha
    gamma = TEXT_GAMMA if is_text else 1.0
    premerge = _premerges_layer(blend, fill, effects,
                                blend_interior)
    if premerge:
        lay[:, :, :] = composite_gamma(blend, work, lay, 1.0,
                                       gamma)
    # Interior effects (other than strokes and bevels):
    # overlays, satin, inner glow, inner shadow.
    full = np.where(inside, 1.0, 0.0)
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "pattern_overlay":
            continue
        _paint_fx(lay, full, {"kind": "pattern",
                              "name": effect["name"],
                              "id": effect["id"],
                              "scale": effect["scale"]},
                  sb, anchor, big, effect["blend"],
                  effect["opacity"], origin, patterns)
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "gradient_overlay":
            continue
        _paint_fx(lay, full, {"kind": "gradient",
                              "gradient": effect["gradient"]},
                  sb, anchor, big, effect["blend"],
                  effect["opacity"], origin, patterns)
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "color_overlay":
            continue
        _paint_color(lay, full, effect["color"],
                     effect["blend"], effect["opacity"], origin)
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "satin":
            continue
        _paint_color(lay, _relative_to_shape(per[i][0], shape),
                     effect["color"], effect["blend"],
                     effect["opacity"], origin)
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "inner_glow":
            continue
        _paint_glow(lay, _relative_to_shape(per[i][0], shape),
                    effect, sb, anchor, big, origin, patterns)
    if interior_group:
        lay[..., 3] = lay[..., 3] * max(fill, 0.0)
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "inner_shadow":
            continue
        _paint_color(lay, _relative_to_shape(per[i][0], shape),
                     effect["color"], effect["blend"],
                     effect["opacity"], origin)
    # Inside strokes (bottom instance first).
    for stroke in reversed(strokes):
        in_w, _ = _stroke_widths(stroke)
        if in_w <= 0.0 or din is None:
            continue
        m = np.where(inside,
                     np.clip(in_w + 0.5 - din, 0.0, 1.0), 0.0)
        frame = _stroke_frame(shape, big, _stroke_widths(
            stroke)[1])
        _paint_fx(lay, m, stroke["paint"], frame, anchor, big,
                  stroke["blend"], stroke["opacity"], origin,
                  patterns)
    # Inner bevel.
    _paint_bevel(lay, items, per, bevel_paint, "inner", shape,
                 origin)
    # A stroke-style bevel shades the stroke band: its
    # inside half rides in the layer, its outside half in
    # the exterior work (each half is overwritten where the
    # other surface covers it). The shading is clipped to
    # the band's visible coverage, never the layer's
    # interior (the height field's tent spread reaches a
    # few pixels inside the shape's edge).
    if band is not None and visible_band is not None:
        _paint_bevel(lay, items, per, bevel_paint, "stroke",
                     visible_band, origin)
    # The shape's own alpha.
    lay[..., 3] = lay[..., 3] * np.minimum(shape, 1.0)
    # Outside strokes beneath the layer.
    _paint_outside_strokes(work, lay, shape, strokes, din,
                           dout, dout_exact, sb, anchor, big,
                           origin, patterns)
    if band is not None and visible_band is not None:
        _paint_bevel(work, items, per, bevel_paint, "stroke",
                     visible_band, origin)
    # Outer bevel.
    _paint_bevel(work, items, per, bevel_paint, "outer", None,
                 origin)
    # The layer (with interior effects) onto the exterior
    # result, in the layer's mode.
    mode = "normal" if (premerge or blend == "pass_through") \
        else blend
    mask = lay[..., 3] > 0.0
    if np.any(mask):
        merged = composite_gamma(mode, work, lay, 1.0, gamma)
        work[mask] = merged[mask]
    # Emboss styles shade the composited layer.
    _paint_bevel(work, items, per, bevel_paint, "both", None,
                 origin)
    # Write back at layer opacity.
    region = canvas[by0:by1, bx0:bx1]
    if blend == "dissolve":
        keep = _hash_noise_array((bh, bw), bx0, by0) < opacity
        region[:, :, :] = np.where(keep[..., None], work,
                                   before)
    else:
        region[:, :, :] = _mix_premul(before, work, opacity)


def _stroke_frame(shape: np.ndarray,
                  big: tuple[int, int, int, int],
                  out_w: float) -> tuple[int, int, int, int]:
    """The frame a gradient stroke aligned with the layer is
    laid out in: the pixel bounds of the shape grown by the
    stroke's outside width less one pixel."""
    ys, xs = np.where(shape > INSIDE_EPS)
    if len(xs) == 0:
        return big
    grow = max(int(math.ceil(min(out_w, MAX_REACH))) - 1, 0) \
        if math.isfinite(out_w) else 0
    return (big[0] + int(xs.min()) - grow,
            big[1] + int(ys.min()) - grow,
            big[0] + int(xs.max()) + 1 + grow,
            big[1] + int(ys.max()) + 1 + grow)


def _paint_bevel(target: np.ndarray, items: list[dict],
                 per: list[list[np.ndarray]],
                 bevel_paint: list[str], kind: str,
                 shape: np.ndarray | None,
                 origin: tuple[int, int]) -> None:
    for i in range(len(items) - 1, -1, -1):
        effect = items[i]
        if effect.get("kind") != "bevel":
            continue
        if bevel_paint[i] != kind:
            continue
        for k, (color, common) in enumerate((
                (effect["highlight_color"],
                 effect["highlight"]),
                (effect["shadow_color"], effect["shadow"]))):
            m = per[i][k]
            if shape is not None:
                m = _relative_to_shape(m, shape)
            _paint_color(target, m, color, common["blend"],
                         common["opacity"], origin)


def _paint_outside_strokes(work: np.ndarray, lay: np.ndarray,
                           shape: np.ndarray,
                           strokes: list[dict],
                           din: np.ndarray | None,
                           dout: np.ndarray | None,
                           dout_exact: np.ndarray | None,
                           sb: tuple[int, int, int, int],
                           anchor: tuple[float, float],
                           big: tuple[int, int, int, int],
                           origin: tuple[int, int],
                           patterns: dict | None) -> None:
    if not any(_stroke_widths(s)[1] > 0.0 for s in strokes):
        return
    h, w = shape.shape
    base = work.copy()
    acc = np.zeros((h, w, 4), dtype=np.float64)
    cover = np.zeros((h, w), dtype=np.float64)
    for stroke in strokes:
        _, out_w = _stroke_widths(stroke)
        if out_w <= 0.0:
            continue
        exact = out_w >= STROKE_CHAMFER_MAX
        d = dout_exact if exact else dout
        if d is None:
            continue
        band = np.clip(out_w + 0.5 - d, 0.0, 1.0)
        inside = shape > INSIDE_EPS
        k = np.where(inside,
                     band * _outline_share(shape, lay[..., 3]),
                     band)
        k = k * stroke["opacity"]
        if stroke["blend"] == "dissolve":
            keep = _hash_noise_array((h, w), origin[0],
                                     origin[1]) < k
            k = np.where((k > 0.0) & keep, 1.0, 0.0)
        share = k * (1.0 - cover)
        cover = cover + share
        # The stroke blended over the backdrop at full
        # coverage.
        blended = base.copy()
        ones = np.where(share > 0.0, 1.0, 0.0)
        paint_blend = "normal" if stroke["blend"] == "dissolve" \
            else stroke["blend"]
        frame = _stroke_frame(shape, big, out_w)
        _paint_fx(blended, ones, stroke["paint"], frame,
                  anchor, big, paint_blend, 1.0, origin,
                  patterns)
        acc[..., :3] += (share[..., None] * blended[..., :3]
                         * blended[..., 3:4])
        acc[..., 3] += share * blended[..., 3]
    mask = cover > 0.0
    if not np.any(mask):
        return
    k = 1.0 - np.minimum(cover, 1.0)
    alpha = work[..., 3] * k + acc[..., 3]
    safe = np.where(alpha > 0.0, alpha, 1.0)
    out = work.copy()
    out[..., :3] = (work[..., :3] * work[..., 3:4] * k[..., None]
                    + acc[..., :3]) / safe[..., None]
    out[..., 3] = np.minimum(alpha, 1.0)
    work[mask] = out[mask]


def _outline_share(cov: np.ndarray, lay: np.ndarray
                   ) -> np.ndarray:
    """Coverage of an outside stroke beneath a filled shape's
    edge pixel."""
    outside = np.maximum(1.0 - np.clip(cov, 0.0, 1.0), 0.0)
    rest = 1.0 - np.clip(lay, 0.0, 1.0)
    share = np.where(outside <= 1e-6, 0.0,
                     np.where(rest <= 1e-6, 1.0,
                              np.minimum(outside / np.maximum(
                                  rest, 1e-9), 1.0)))
    return share


def _mix_premul(a: np.ndarray, b: np.ndarray,
                k: float) -> np.ndarray:
    """Premultiplied linear interpolation between two
    straight-alpha pixels."""
    alpha = a[..., 3] + (b[..., 3] - a[..., 3]) * k
    safe = np.where(alpha <= 0.0, 1.0, alpha)
    out = np.zeros_like(a)
    out[..., :3] = (a[..., :3] * a[..., 3:4]
                    + (b[..., :3] * b[..., 3:4]
                       - a[..., :3] * a[..., 3:4]) * k) / safe[..., None]
    out[..., 3] = alpha
    out[alpha <= 0.0] = 0.0
    return out
