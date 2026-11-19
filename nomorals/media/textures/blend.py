"""W3C blend modes (TextureCraft port, census 3A.15).

Faithful port of upstream ``crates/core/src/blend.rs`` (commit
37ad2112): the 32 modes in menu order with their upstream labels and
four-letter shorts, ``blend1``/``blend3``/``mix3``, the W3C luminance
(0.3 / 0.59 / 0.11), ``clip_color`` / ``set_lum`` / ``set_sat``,
``unpack``/``pack`` and RNM (reoriented normal mapping, Barré-
Brisebois & Hill). Inputs and outputs are 0..1 floats; upstream
clamps nothing except where its formulas do, so this port does not
clamp either — callers that need display range clamp at the edge.
"""
from __future__ import annotations

import math
from enum import StrEnum

# -- mode menu (upstream BlendMode, menu order) -------------------------------
_MENU = [
    ("normal", "Normal", "Norm"),
    ("passthrough", "Passthrough", "Pass"),
    ("disable", "Disable", "Dis"),
    ("replace", "Replace", "Repl"),
    ("multiply", "Multiply", "Mult"),
    ("divide", "Divide", "Div"),
    ("inverse_divide", "Inverse divide", "IDiv"),
    ("darken", "Darken (Min)", "Dark"),
    ("lighten", "Lighten (Max)", "Ligh"),
    ("linear_dodge", "Linear dodge (Add)", "Add"),
    ("subtract", "Subtract", "Sub"),
    ("inverse_subtract", "Inverse Subtract", "ISub"),
    ("difference", "Difference", "Diff"),
    ("exclusion", "Exclusion", "Excl"),
    ("signed_addition", "Signed addition (AddSub)", "AdSb"),
    ("overlay", "Overlay", "Over"),
    ("screen", "Screen", "Scrn"),
    ("linear_burn", "Linear burn", "LBrn"),
    ("color_burn", "Color burn", "CBrn"),
    ("color_dodge", "Color dodge", "CDdg"),
    ("soft_light", "Soft light", "SftL"),
    ("hard_light", "Hard light", "HrdL"),
    ("vivid_light", "Vivid light", "VivL"),
    ("linear_light", "Linear light", "LinL"),
    ("pin_light", "Pin light", "PinL"),
    ("tint", "Tint", "Tint"),
    ("saturation", "Saturation", "Sat"),
    ("color", "Color", "Colr"),
    ("value", "Value", "Val"),
    ("normal_map_combine", "Normal map combine", "NmCb"),
    ("normal_map_detail", "Normal map detail", "NmDt"),
    ("normal_map_inverse_detail", "Normal map inverse detail", "NmID"),
]

_NON_SEPARABLE = {"tint", "saturation", "color", "value",
                  "normal_map_combine", "normal_map_detail",
                  "normal_map_inverse_detail"}


class BlendMode(StrEnum):
    """One of the 32 blend modes (upstream ``BlendMode``)."""

    NORMAL = "normal"
    PASSTHROUGH = "passthrough"
    DISABLE = "disable"
    REPLACE = "replace"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    INVERSE_DIVIDE = "inverse_divide"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    LINEAR_DODGE = "linear_dodge"
    SUBTRACT = "subtract"
    INVERSE_SUBTRACT = "inverse_subtract"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    SIGNED_ADDITION = "signed_addition"
    OVERLAY = "overlay"
    SCREEN = "screen"
    LINEAR_BURN = "linear_burn"
    COLOR_BURN = "color_burn"
    COLOR_DODGE = "color_dodge"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    VIVID_LIGHT = "vivid_light"
    LINEAR_LIGHT = "linear_light"
    PIN_LIGHT = "pin_light"
    TINT = "tint"
    SATURATION = "saturation"
    COLOR = "color"
    VALUE = "value"
    NORMAL_MAP_COMBINE = "normal_map_combine"
    NORMAL_MAP_DETAIL = "normal_map_detail"
    NORMAL_MAP_INVERSE_DETAIL = "normal_map_inverse_detail"

    @classmethod
    def all(cls) -> list[BlendMode]:
        return [cls(v) for v, _, _ in _MENU]

    def label(self) -> str:
        return {v: label for v, label, _ in _MENU}[self.value]

    def short(self) -> str:
        return {v: s for v, _, s in _MENU}[self.value]

    @classmethod
    def parse(cls, name: str) -> BlendMode | None:
        """Parse a label, short label or variant name
        (case-insensitive; punctuation ignored)."""
        needle = "".join(ch for ch in name.lower()
                         if ch not in " _-()")
        for v, label, short in _MENU:
            for cand in (v, label, short):
                if "".join(ch for ch in cand.lower()
                           if ch not in " _-()") == needle:
                    return cls(v)
        return None

    def is_non_separable(self) -> bool:
        return self.value in _NON_SEPARABLE


def _color_burn(a: float, b: float) -> float:
    if a >= 1.0:
        return 1.0
    if b <= 0.0:
        return 0.0
    return 1.0 - min((1.0 - a) / b, 1.0)


def _color_dodge(a: float, b: float) -> float:
    if a <= 0.0:
        return 0.0
    if b >= 1.0:
        return 1.0
    return min(a / (1.0 - b), 1.0)


def _soft_light(a: float, b: float) -> float:
    a = min(1.0, max(0.0, a))
    b = min(1.0, max(0.0, b))
    if b <= 0.5:
        return a - (1.0 - 2.0 * b) * a * (1.0 - a)
    d = ((16.0 * a - 12.0) * a + 4.0) * a if a <= 0.25 else math.sqrt(a)
    return a + (2.0 * b - 1.0) * (d - a)


def _overlay(a: float, b: float) -> float:
    if a < 0.5:
        return 2.0 * a * b
    return 1.0 - 2.0 * (1.0 - a) * (1.0 - b)


def blend1(mode: BlendMode, a: float, b: float) -> float:
    """Blend one channel: ``a`` is the base, ``b`` the blend layer."""
    m = mode.value
    if m in ("normal", "passthrough", "replace"):
        return b
    if m == "disable":
        return a
    if m == "multiply":
        return a * b
    if m == "divide":
        return a / max(b, 1e-6)
    if m == "inverse_divide":
        return b / max(a, 1e-6)
    if m == "darken":
        return min(a, b)
    if m == "lighten":
        return max(a, b)
    if m == "linear_dodge":
        return a + b
    if m == "subtract":
        return a - b
    if m == "inverse_subtract":
        return b - a
    if m == "difference":
        return abs(a - b)
    if m == "exclusion":
        return a + b - 2.0 * a * b
    if m in ("signed_addition", "linear_light"):
        return a + 2.0 * b - 1.0
    if m == "overlay":
        return _overlay(a, b)
    if m == "screen":
        return 1.0 - (1.0 - a) * (1.0 - b)
    if m == "linear_burn":
        return a + b - 1.0
    if m == "color_burn":
        return _color_burn(a, b)
    if m == "color_dodge":
        return _color_dodge(a, b)
    if m == "soft_light":
        return _soft_light(a, b)
    if m == "hard_light":
        return _overlay(b, a)
    if m == "vivid_light":
        if b < 0.5:
            return _color_burn(a, 2.0 * b)
        return _color_dodge(a, 2.0 * (b - 0.5))
    if m == "pin_light":
        if b < 0.5:
            return min(a, 2.0 * b)
        return max(a, 2.0 * b - 1.0)
    if m in _NON_SEPARABLE:
        return b
    return b


def lum(c: tuple[float, float, float]) -> float:
    """W3C luminance (0.3 / 0.59 / 0.11)."""
    return 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]


def clip_color(c: tuple[float, float, float]
               ) -> tuple[float, float, float]:
    value = lum(c)
    n = min(c)
    x = max(c)
    out = c
    if n < 0.0 and value - n > 1e-6:
        out = tuple(value + (v - value) * value / (value - n) for v in out)
    if x > 1.0 and x - value > 1e-6:
        out = tuple(value + (v - value) * (1.0 - value) / (x - value)
                    for v in out)
    return out


def set_lum(c: tuple[float, float, float], target: float
            ) -> tuple[float, float, float]:
    d = target - lum(c)
    return clip_color(tuple(v + d for v in c))


def sat(c: tuple[float, float, float]) -> float:
    return max(c) - min(c)


def set_sat(c: tuple[float, float, float], s: float
            ) -> tuple[float, float, float]:
    mx = max(c)
    mn = min(c)
    if mx - mn <= 1e-6:
        return (0.0, 0.0, 0.0)
    return tuple((v - mn) * s / (mx - mn) for v in c)


def unpack(n: tuple[float, float, float]) -> tuple[float, float, float]:
    """Packed 0..1 normal -> unpacked -1..1."""
    return tuple(v * 2.0 - 1.0 for v in n)


def pack(n: tuple[float, float, float]) -> tuple[float, float, float]:
    """Unpacked normal -> packed 0..1 (normalised first)."""
    length = max(math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), 1e-6)
    return tuple(v / length * 0.5 + 0.5 for v in n)


def rnm(base: tuple[float, float, float],
        detail: tuple[float, float, float]
        ) -> tuple[float, float, float]:
    """Reoriented normal mapping (Barré-Brisebois & Hill): ``detail``
    re-oriented onto ``base`` (both packed into 0..1, as is the
    result)."""
    t = unpack(base)
    t = (t[0], t[1], t[2] + 1.0)
    u = unpack(detail)
    u = (-u[0], -u[1], u[2])
    d = t[0] * u[0] + t[1] * u[1] + t[2] * u[2]
    return pack((t[0] * d - u[0] * t[2],
                 t[1] * d - u[1] * t[2],
                 t[2] * d - u[2] * t[2]))


def blend3(mode: BlendMode, a: tuple[float, float, float],
           b: tuple[float, float, float]
           ) -> tuple[float, float, float]:
    """Blend three channels. Non-separable modes (hue/saturation/
    color/value/tint and the normal-map modes) work on the colour as
    a whole; the rest apply ``blend1`` per channel."""
    m = mode.value
    if m == "color":
        return set_lum(b, lum(a))
    if m == "saturation":
        return set_lum(set_sat(a, sat(b)), lum(a))
    if m == "value":
        va = max(a)
        vb = max(b)
        if va <= 1e-6:
            return (vb, vb, vb)
        return tuple(v * vb / va for v in a)
    if m == "tint":
        lum_a = lum(a)
        return tuple(v * lum_a for v in b)
    if m == "normal_map_combine":
        na = unpack(a)
        nb = unpack(b)
        return pack((na[0] + nb[0], na[1] + nb[1], na[2] * nb[2]))
    if m == "normal_map_detail":
        return rnm(a, b)
    if m == "normal_map_inverse_detail":
        return rnm(b, a)
    return tuple(blend1(mode, a[i], b[i]) for i in range(3))


def mix3(a: tuple[float, float, float], blended: tuple[float, float, float],
         w: float) -> tuple[float, float, float]:
    """Mix base ``a`` with ``blended`` by coverage ``w``."""
    return tuple(a[i] + (blended[i] - a[i]) * w for i in range(3))
