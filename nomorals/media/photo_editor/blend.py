"""Blend modes and compositing — PhotoCraft's ``crates/color`` blend
math plus its Photoshop-exact ``psblend`` variants, in numpy.

The formulas are the W3C/PDF separable modes with Photoshop's
measured variants: Vivid Light lets the source extremes win,
Hard Mix thresholds the *generic* Vivid Light, Color Burn/Dodge
treat backdrop values within 1e-4 of 0/1 as exact, and Soft Light
is the W3C curve with the cubic ``D(cb)`` below a quarter. The
composite is the W3C general formula over straight alpha; the
"special eight" fill semantics (a layer's Fill below 100% pulls
the source toward the mode's neutral colour) and the text gamma
(1.45) composite live here too.

All functions take float32 RGB(A) numpy arrays in 0..1 and return
the same shape; scalar helpers take plain floats. Nothing here
raises on out-of-range input — values are clamped where the
formula clips, as Photoshop does.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BLEND_KEYS",
    "blend_rgb",
    "clip_color",
    "cmyk_to_rgb",
    "composite",
    "composite_fill",
    "dissolve_mask",
    "fill_is_special",
    "hsb_to_rgb",
    "lab_to_srgb",
    "linear_to_srgb",
    "lum",
    "rgb_to_cmyk",
    "rgb_to_lab",
    "set_lum",
    "set_sat",
    "srgb_to_lab",
    "srgb_to_linear",
]

#: Photoshop blend keys (4-byte codes as stored in the layer
#: record) -> mode name. ``pass`` is a group pass-through (the
#: group composites as normal); ``norm`` is normal.
BLEND_KEYS: dict[str, str] = {
    "norm": "normal",
    "diss": "dissolve",
    "dark": "darken",
    "mul ": "multiply",
    "idiv": "color_burn",
    "lbrn": "linear_burn",
    "lite": "lighten",
    "scrn": "screen",
    "div ": "color_dodge",
    "over": "overlay",
    "sLit": "soft_light",
    "hLit": "hard_light",
    "vLit": "vivid_light",
    "lLit": "linear_light",
    "pLit": "pin_light",
    "hMix": "hard_mix",
    "diff": "difference",
    "smud": "exclusion",
    "fsub": "subtract",
    "fdiv": "divide",
    "hue ": "hue",
    "sat ": "saturation",
    "colr": "color",
    "lum ": "luminosity",
    "dkCl": "darker_color",
    "lgCl": "lighter_color",
    "pass": "pass_through",
}

#: The "special eight": Fill below 100% changes the blend, not
#: just the coverage.
_SPECIAL_EIGHT = frozenset({
    "color_burn", "linear_burn", "color_dodge", "linear_dodge",
    "vivid_light", "linear_light", "hard_mix", "difference",
})

#: Text layers composite in gamma 1.45 space (measured).
TEXT_GAMMA = 1.45

#: D50 white point (CIE XYZ).
D50 = (0.96422, 1.0, 0.82521)

#: sRGB -> XYZ (D50) and back (Bradford-adapted, as PhotoCraft's
#: ``convert.rs`` carries them).
SRGB_TO_XYZ_D50 = np.array([
    [0.4360657, 0.3851515, 0.1430784],
    [0.2224884, 0.7169030, 0.0606087],
    [0.0139160, 0.0970816, 0.7140944],
], dtype=np.float64)
XYZ_D50_TO_SRGB = np.array([
    [3.1338561, -1.6168667, -0.4906146],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0719453, -0.2289914, 1.4052427],
], dtype=np.float64)


def mode_name(key: str) -> str:
    """A stored 4-byte blend key -> mode name (``normal`` for an
    unknown key — never a crash on a file Photoshop wrote)."""
    return BLEND_KEYS.get((key or "norm")[:4].ljust(4), "normal")


def fill_is_special(mode: str) -> bool:
    return mode in _SPECIAL_EIGHT


# -- colour helpers -----------------------------------------------------

def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """sRGB-encoded -> linear light (the piecewise sRGB EOTF)."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92,
                    ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    """Linear light -> sRGB-encoded."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4)
                    - 0.055)


def lum(rgb: np.ndarray) -> np.ndarray:
    """W3C luminosity: 0.30 R + 0.59 G + 0.11 B."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]


def _sat(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return (rgb.max(axis=-1) - rgb.min(axis=-1))


def clip_color(rgb: np.ndarray, top: float = 1.0) -> np.ndarray:
    """Move ``rgb`` toward its grey until every channel is within
    0..``top``, keeping the luminosity (W3C ClipColor). Outside the
    gamut's luminosity range a colour becomes its grey instead —
    the formula's value as the luminosity reaches the bound."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lum_v = lum(rgb)[..., None]
    n = rgb.min(axis=-1, keepdims=True)
    x = rgb.max(axis=-1, keepdims=True)
    out = rgb.copy()
    neg = n < 0.0
    if np.any(neg):
        d = lum_v - n
        factor = np.where((lum_v > 0.0) & (d > 0.0),
                          lum_v / np.where(
            d != 0, d, 1.0), 0.0)
        fixed = np.where((lum_v > 0.0) & (d > 0.0),
                         lum_v + (out - lum_v) * factor, lum_v)
        out = np.where(neg, fixed, out)
    over = x > top
    if np.any(over):
        d = x - lum_v
        factor = np.where((lum_v < top) & (d > 0.0),
                          (top - lum_v) / np.where(d != 0, d, 1.0),
                          0.0)
        fixed = np.where((lum_v < top) & (d > 0.0),
                         lum_v + (out - lum_v) * factor, lum_v)
        out = np.where(over, fixed, out)
    return out


def set_lum(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Set the luminosity of ``rgb`` to ``target`` (W3C SetLum)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    d = target - lum(rgb)
    return clip_color(rgb + d[..., None])


def set_sat(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Set the saturation of ``rgb`` to ``target`` (W3C SetSat)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    order = np.argsort(rgb, axis=-1)
    imin = order[..., 0]
    imid = order[..., 1]
    imax = order[..., 2]
    mn = np.take_along_axis(rgb, imin[..., None], axis=-1)[..., 0]
    md = np.take_along_axis(rgb, imid[..., None], axis=-1)[..., 0]
    mx = np.take_along_axis(rgb, imax[..., None], axis=-1)[..., 0]
    out = np.zeros_like(rgb)
    span = mx - mn
    safe = span > 0
    mid_val = np.where(safe, (md - mn) * target / np.where(
        safe, span, 1.0), 0.0)
    np.put_along_axis(out, imid[..., None], mid_val[..., None],
                      axis=-1)
    np.put_along_axis(out, imax[..., None], target[..., None],
                      axis=-1)
    return out


def hsb_to_rgb(hue: float, sat: float, bri: float
               ) -> tuple[float, float, float]:
    """HSB (hue degrees, saturation / brightness 0..1) -> sRGB."""
    h = (hue % 360.0) / 60.0
    s = min(1.0, max(0.0, sat))
    v = min(1.0, max(0.0, bri))
    f = h - int(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    sector = int(h) % 6
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][sector]


def cmyk_to_rgb(cmyk: tuple[float, float, float, float]
                ) -> tuple[float, float, float]:
    """Naive CMYK (0..1 inks, Photoshop's no-profile fallback)."""
    c, m, y, k = cmyk
    return ((1.0 - c) * (1.0 - k), (1.0 - m) * (1.0 - k),
            (1.0 - y) * (1.0 - k))


def rgb_to_cmyk(rgb: tuple[float, float, float]
                ) -> tuple[float, float, float, float]:
    r, g, b = rgb
    k = 1.0 - max(r, g, b)
    if k >= 1.0:
        return (0.0, 0.0, 0.0, 1.0)
    denom = 1.0 - k
    return ((1.0 - r - k) / denom, (1.0 - g - k) / denom,
            (1.0 - b - k) / denom, k)


def cmyk_to_rgb_arr(cmyk: np.ndarray) -> np.ndarray:
    """Vectorized naive CMYK (0..1 inks) -> sRGB."""
    cmyk = np.asarray(cmyk, dtype=np.float64)
    c, m, y, k = (cmyk[..., i] for i in range(4))
    return np.stack([(1.0 - c) * (1.0 - k),
                     (1.0 - m) * (1.0 - k),
                     (1.0 - y) * (1.0 - k)], axis=-1)


def rgb_to_cmyk_arr(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB -> naive CMYK (0..1 inks)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    k = 1.0 - rgb.max(axis=-1)
    denom = np.where(1.0 - k > 0.0, 1.0 - k, 1.0)
    return np.stack([(1.0 - rgb[..., 0] - k) / denom,
                     (1.0 - rgb[..., 1] - k) / denom,
                     (1.0 - rgb[..., 2] - k) / denom, k],
                    axis=-1)


def lab_to_rgb_arr(lab: np.ndarray) -> np.ndarray:
    """Vectorized CIELAB (L 0..1, a/b 0..255 with 128 neutral)
    -> sRGB, through D50 XYZ."""
    lab = np.asarray(lab, dtype=np.float64)
    lum_v = lab[..., 0] * 100.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    fy = (lum_v + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xr, yr, zr = D50
    x = xr * _lab_f_inv(fx)
    y = yr * _lab_f_inv(fy)
    z = zr * _lab_f_inv(fz)
    xyz = np.stack([x, y, z], axis=-1)
    lin = xyz @ XYZ_D50_TO_SRGB.T
    return np.clip(linear_to_srgb(lin), 0.0, 1.0)


def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t),
                    t / (3.0 * delta * delta) + 4.0 / 29.0)


def _lab_f_inv(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta, t ** 3,
                    3.0 * delta * delta * (t - 4.0 / 29.0))


def lab_to_srgb(lab: tuple[float, float, float]
                ) -> tuple[float, float, float]:
    """CIELAB (L 0..1, a/b stored 0..255 with 128 neutral, as PSD
    pixels carry them) -> sRGB, through D50 XYZ."""
    lum_, a, b = lab
    lum_v = lum_ * 100.0
    a = a - 128.0
    b = b - 128.0
    fy = (lum_v + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xr, yr, zr = D50
    x = xr * _lab_f_inv(np.array([fx]))[0]
    y = yr * _lab_f_inv(np.array([fy]))[0]
    z = zr * _lab_f_inv(np.array([fz]))[0]
    lin = XYZ_D50_TO_SRGB @ np.array([x, y, z])
    srgb = linear_to_srgb(lin)
    return (float(np.clip(srgb[0], 0.0, 1.0)),
            float(np.clip(srgb[1], 0.0, 1.0)),
            float(np.clip(srgb[2], 0.0, 1.0)))


def srgb_to_lab(rgb: tuple[float, float, float]
                ) -> tuple[float, float, float]:
    """sRGB -> CIELAB (L 0..1, a/b 0..255 with 128 neutral)."""
    lin = srgb_to_linear(np.array(rgb, dtype=np.float64))
    xyz = SRGB_TO_XYZ_D50 @ lin
    xr, yr, zr = D50
    fx = _lab_f(np.array([xyz[0] / xr]))[0]
    fy = _lab_f(np.array([xyz[1] / yr]))[0]
    fz = _lab_f(np.array([xyz[2] / zr]))[0]
    lum_v = (116.0 * fy - 16.0) / 100.0
    return (float(lum_v), float(500.0 * (fx - fy) + 128.0),
            float(200.0 * (fy - fz) + 128.0))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB array (..., 3) -> Lab (..., 3) with L 0..1
    and a/b 0..255 (128 neutral)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = srgb_to_linear(rgb)
    xyz = lin @ SRGB_TO_XYZ_D50.T
    xr, yr, zr = D50
    fx = _lab_f(xyz[..., 0] / xr)
    fy = _lab_f(xyz[..., 1] / yr)
    fz = _lab_f(xyz[..., 2] / zr)
    lum_v = (116.0 * fy - 16.0) / 100.0
    a = 500.0 * (fx - fy) + 128.0
    b = 200.0 * (fy - fz) + 128.0
    return np.stack([lum_v, a, b], axis=-1)


# -- separable blend channels (vectorized) -----------------------------

def _blend_channel(mode: str, cb: np.ndarray, cs: np.ndarray
                   ) -> np.ndarray:
    """One channel of a separable mode (Photoshop variants)."""
    cb = np.asarray(cb, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)
    edge = 1e-4
    if mode in ("normal", "dissolve", "pass_through"):
        return cs
    if mode == "darken":
        return np.minimum(cb, cs)
    if mode == "multiply":
        return cb * cs
    if mode == "color_burn":
        return np.where(cb >= 1.0 - edge, 1.0,
                        np.where(cs <= 0.0, 0.0,
                                 1.0 - np.minimum(
                                     (1.0 - cb) / np.where(
                                         cs <= 0.0, 1.0, cs), 1.0)))
    if mode == "linear_burn":
        return np.maximum(cb + cs - 1.0, 0.0)
    if mode == "lighten":
        return np.maximum(cb, cs)
    if mode == "screen":
        return cb + cs - cb * cs
    if mode == "color_dodge":
        return np.where(cb <= edge, 0.0,
                        np.where(cs >= 1.0, 1.0,
                                 np.minimum(
                                     cb / np.where(
                                         cs >= 1.0, 1.0, 1.0 - cs),
                                     1.0)))
    if mode == "linear_dodge":
        return np.minimum(cb + cs, 1.0)
    if mode == "overlay":
        return _hard_light(cs, cb)
    if mode == "soft_light":
        return _soft_light_ps(cb, cs)
    if mode == "hard_light":
        return _hard_light(cb, cs)
    if mode == "vivid_light":
        # Source extremes win (Photoshop).
        return np.where(
            cs <= 0.5,
            np.where(cs <= 0.0, 0.0,
                     1.0 - np.minimum((1.0 - cb) / np.where(
                         cs <= 0.0, 1.0, 2.0 * cs), 1.0)),
            np.where(cs >= 1.0, 1.0,
                     np.minimum(cb / np.where(
                         cs >= 1.0, 1.0, 2.0 * (1.0 - cs)), 1.0)))
    if mode == "linear_light":
        return np.clip(cb + 2.0 * cs - 1.0, 0.0, 1.0)
    if mode == "pin_light":
        return np.where(cs <= 0.5, np.minimum(cb, 2.0 * cs),
                        np.maximum(cb, 2.0 * cs - 1.0))
    if mode == "hard_mix":
        vl = _vivid_light_generic(cb, cs)
        return np.where(vl >= 0.5 - 1e-6, 1.0, 0.0)
    if mode == "difference":
        return np.abs(cb - cs)
    if mode == "exclusion":
        return cb + cs - 2.0 * cb * cs
    if mode == "subtract":
        return np.maximum(cb - cs, 0.0)
    if mode == "divide":
        return np.where(cs <= 0.0,
                        np.where(cb <= 0.0, 0.0, 1.0),
                        np.minimum(cb / np.where(cs <= 0.0, 1.0, cs),
                                   1.0))
    if mode == "reflect":
        return np.where(cs >= 1.0, 1.0,
                        np.minimum(cb * cb / np.where(
                            cs >= 1.0, 1.0, 1.0 - cs), 1.0))
    if mode == "glow":
        return np.where(cb >= 1.0, 1.0,
                        np.minimum(cs * cs / np.where(
                            cb >= 1.0, 1.0, 1.0 - cb), 1.0))
    if mode == "negation":
        return 1.0 - np.abs(1.0 - cb - cs)
    if mode == "xor":
        b = np.round(np.clip(cb, 0.0, 1.0) * 255.0).astype(np.uint8)
        s = np.round(np.clip(cs, 0.0, 1.0) * 255.0).astype(np.uint8)
        return (b ^ s).astype(np.float64) / 255.0
    # Non-separable modes fall back to the source per channel;
    # blend_rgb handles them as triples.
    return cs


def _hard_light(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return np.where(cs <= 0.5, cb * 2.0 * cs,
                    cb + (2.0 * cs - 1.0)
                    - cb * (2.0 * cs - 1.0))


def _soft_light_ps(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    b = np.maximum(cb, 0.0)
    d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b,
                 np.sqrt(b))
    return np.where(cs <= 0.5,
                    2.0 * cb * cs + cb * cb * (1.0 - 2.0 * cs),
                    2.0 * cb * (1.0 - cs) + d * (2.0 * cs - 1.0))


def _vivid_light_generic(cb: np.ndarray, cs: np.ndarray
                         ) -> np.ndarray:
    edge = 1e-4
    return np.where(
        cs <= 0.5,
        np.where(cb >= 1.0 - edge, 1.0,
                 np.where(cs <= 0.0, 0.0,
                          1.0 - np.minimum((1.0 - cb) / np.where(
                              cs <= 0.0, 1.0, 2.0 * cs), 1.0))),
        np.where(cb <= edge, 0.0,
                 np.where(cs >= 1.0, 1.0,
                          np.minimum(cb / np.where(
                              cs >= 1.0, 1.0, 2.0 * (1.0 - cs)),
                              1.0))))


def blend_rgb(mode: str, cb: np.ndarray, cs: np.ndarray
              ) -> np.ndarray:
    """``B(Cb, Cs)`` for a whole RGB triple, including the
    non-separable modes (Hue / Saturation / Color / Luminosity /
    Darker / Lighter Color)."""
    cb = np.asarray(cb, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)
    if mode == "hue":
        return set_lum(set_sat(cs, _sat(cb)), lum(cb))
    if mode == "saturation":
        return set_lum(set_sat(cb, _sat(cs)), lum(cb))
    if mode == "color":
        return set_lum(cs, lum(cb))
    if mode == "luminosity":
        return set_lum(cb, lum(cs))
    if mode == "darker_color":
        return np.where((lum(cs) < lum(cb))[..., None], cs, cb)
    if mode == "lighter_color":
        return np.where((lum(cs) > lum(cb))[..., None], cs, cb)
    out = np.empty_like(cb)
    for i in range(3):
        out[..., i] = _blend_channel(mode, cb[..., i], cs[..., i])
    return out


def _hash_noise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Deterministic per-pixel value in 0..1, hashed from document
    coordinates (the same hash as PhotoCraft's ``dissolve_noise``),
    so Dissolve speckle is stable across renders."""
    h = ((x.astype(np.int64) * 0x8DA6B343)
         ^ (y.astype(np.int64) * 0xD8163841)
         ^ 0x9E3779B9).astype(np.uint32)
    h = (h ^ (h >> np.uint32(15))).astype(np.uint32)
    h = (h * np.uint32(0x2C1B3C6D)).astype(np.uint32)
    h = (h ^ (h >> np.uint32(12))).astype(np.uint32)
    return (h & np.uint32(0xFFFF)).astype(np.float64) / 65536.0


def dissolve_mask(coverage: np.ndarray, x0: int, y0: int
                  ) -> np.ndarray:
    """Boolean mask of the pixels Dissolve keeps at ``coverage``
    (document coordinates start at ``x0``/``y0``)."""
    height, width = coverage.shape[:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    noise = _hash_noise(xs + x0, ys + y0)
    return noise < coverage


# -- compositing --------------------------------------------------------

def text_encode(v: np.ndarray, gamma: float) -> np.ndarray:
    """Photoshop's text blending space: linear light raised to
    1 / gamma (``srgb_to_linear`` then ``**(1/gamma)``)."""
    return np.power(srgb_to_linear(np.maximum(v, 0.0)),
                    1.0 / gamma)


def text_decode(v: np.ndarray, gamma: float) -> np.ndarray:
    """Inverse of ``text_encode``."""
    return linear_to_srgb(np.power(np.maximum(v, 0.0), gamma))


def composite_gamma(mode: str, backdrop: np.ndarray,
                    source: np.ndarray, opacity: float,
                    gamma: float) -> np.ndarray:
    """``composite`` with the coverage mix done in the text
    blending space of ``gamma`` (gamma == 1 is ``composite``)."""
    if gamma == 1.0:
        return composite(mode, backdrop, source, opacity)
    ab = backdrop[..., 3:4]
    as_ = source[..., 3:4] * opacity
    ao = as_ + ab * (1.0 - as_)
    safe_ao = np.where(ao <= 0.0, 1.0, ao)
    cb = backdrop[..., :3]
    cs = source[..., :3]
    b = blend_rgb(mode, cb, cs)
    wb = (1.0 - as_) * ab / safe_ao
    ws = (1.0 - ab) * as_ / safe_ao
    wbs = as_ * ab / safe_ao
    out = np.zeros_like(backdrop)
    mixed = (wb * text_encode(cb, gamma)
             + ws * text_encode(cs, gamma)
             + wbs * text_encode(b, gamma))
    out[..., :3] = text_decode(mixed, gamma)
    out[..., 3] = np.where(ao[..., 0] <= 0.0, 0.0, ao[..., 0])
    keep = (as_[..., 0] <= 0.0)
    out[keep] = backdrop[keep]
    return out


def composite(mode: str, backdrop: np.ndarray, source: np.ndarray,
              opacity: float = 1.0) -> np.ndarray:
    """Straight-alpha source over straight-alpha backdrop with a
    blend mode and opacity (W3C/PDF general formula). Both arrays
    are (H, W, 4) float in 0..1. Dissolve resolves per pixel
    before this call (the caller applies :func:`dissolve_mask`)."""
    mode = mode if mode != "pass_through" else "normal"
    backdrop = np.asarray(backdrop, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    ab = backdrop[..., 3]
    a_s = source[..., 3] * opacity
    ao = a_s + ab * (1.0 - a_s)
    out = backdrop.copy()
    live = (a_s > 0.0) & (ao > 0.0)
    if not np.any(live):
        return out
    cb = backdrop[..., :3]
    cs = source[..., :3]
    # The divide runs over fully transparent pixels too
    # (ao == 0 there); the ``where`` below discards those
    # results, so the warning is silenced, not hidden.
    with np.errstate(invalid="ignore", divide="ignore"):
        if mode == "normal":
            mix = (ab[..., None] * (1.0 - a_s[..., None]) * cb
                   + a_s[..., None] * cs) / ao[..., None]
        else:
            b = blend_rgb(mode, cb, cs)
            mix = ((1.0 - a_s[..., None]) * ab[..., None] * cb
                   + (1.0 - ab[..., None]) * a_s[..., None] * cs
                   + a_s[..., None] * ab[..., None] * b) \
                / ao[..., None]
    out[..., :3] = np.where(live[..., None], mix, out[..., :3])
    out[..., 3] = np.where(a_s > 0.0, ao, ab)
    return out


def composite_atop(mode: str, backdrop: np.ndarray,
                   source: np.ndarray, opacity: float = 1.0
                   ) -> np.ndarray:
    """Source-atop composite (Lock Transparent Pixels / Fill's
    Preserve Transparency): the backdrop's alpha is kept."""
    backdrop = np.asarray(backdrop, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    a_s = source[..., 3] * opacity
    out = backdrop.copy()
    live = a_s > 0.0
    if not np.any(live):
        return out
    b = blend_rgb(mode, backdrop[..., :3], source[..., :3])
    mix = (1.0 - a_s[..., None]) * backdrop[..., :3] \
        + a_s[..., None] * b
    out[..., :3] = np.where(live[..., None], mix, out[..., :3])
    return out


def blend_rgb_fill(mode: str, cb: np.ndarray, cs: np.ndarray,
                   fill: float) -> np.ndarray:
    """``blend_rgb`` for a layer at ``fill`` (0..1): the special
    eight pull the source toward the mode's neutral colour."""
    fill = min(1.0, max(0.0, fill))
    if fill >= 1.0 or not fill_is_special(mode):
        return blend_rgb(mode, cb, cs)
    cb = np.asarray(cb, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)
    if mode == "hard_mix":
        return np.clip((cb - fill * (1.0 - cs)) / (1.0 - fill),
                       0.0, 1.0)
    if mode in ("color_burn", "linear_burn"):
        neutral = 1.0
    elif mode in ("vivid_light", "linear_light"):
        neutral = 0.5
    else:
        neutral = 0.0
    toward = neutral + (cs - neutral) * fill
    return blend_rgb(mode, cb, toward)


def composite_fill(mode: str, backdrop: np.ndarray,
                   source: np.ndarray, opacity: float = 1.0,
                   fill: float = 1.0, gamma: float = 1.0
                   ) -> np.ndarray:
    """Composite with separate opacity and fill (the special eight
    below 100% Fill blend at full strength over a neutral-pulled
    source; over transparency Fill stays coverage). ``gamma`` is
    the text gamma (1.45) when compositing text, else 1.0."""
    if fill >= 1.0 or not fill_is_special(mode):
        return composite(mode, backdrop, source,
                         opacity * min(1.0, max(0.0, fill)))
    backdrop = np.asarray(backdrop, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    fill = min(1.0, max(0.0, fill))
    ab = backdrop[..., 3]
    over = source[..., 3] * opacity
    alone = over * fill
    out = backdrop.copy()
    live = (over > 0.0)
    if not np.any(live):
        return out
    ao = ab + (1.0 - ab) * alone
    cb = backdrop[..., :3]
    cs = source[..., :3]
    b = blend_rgb_fill(mode, cb, cs, fill)
    wb = ab * (1.0 - over) / np.where(ao > 0, ao, 1.0)
    wbs = ab * over / np.where(ao > 0, ao, 1.0)
    ws = (1.0 - ab) * alone / np.where(ao > 0, ao, 1.0)
    if gamma == 1.0:
        mix = (wb[..., None] * cb + wbs[..., None] * b
               + ws[..., None] * cs)
    else:
        def enc(v):
            return np.power(np.clip(v, 0.0, 1.0), gamma)

        def dec(v):
            return np.power(np.clip(v, 0.0, 1.0), 1.0 / gamma)
        mix = dec(wb[..., None] * enc(cb) + wbs[..., None] * enc(b)
                  + ws[..., None] * enc(cs))
    out[..., :3] = np.where(live[..., None], mix, out[..., :3])
    out[..., 3] = np.where(over > 0.0, ao, ab)
    return out
