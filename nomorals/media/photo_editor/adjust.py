"""Adjustment layers — PhotoCraft's ``crates/io`` block parsers and
``crates/compose`` evaluation math, in numpy.

Every adjustment Photoshop writes as a tagged block (``brit``,
``curv``, ``levl``, ``hue2``, ``mixr``, ``blnc``, ``phfl``,
``blwh``, ``grdm``, ``post``, ``thrs``, ``selc``, ``vibA``,
``expA``, ``nvrt``) parses into a plain dict (``kind`` + params)
and evaluates over a float RGB canvas (0..1, sRGB-encoded, straight
alpha alongside). The evaluation is the documented math: modern
brightness (Hermite knee at 126/255, power roll-off) and contrast
(cubic Hermite S-curve), legacy brightness/contrast pivoting at
128/255, Levels with whole-level rounding and the gamma toe (power
2.2), Curves as a natural cubic spline (or a 256-byte pencil map),
Hue/Saturation with its six range weights, Color Balance folded
into one Levels-like curve per channel, Selective Color by range
weights, Vibrance in linear light, Photo Filter as a linear-light
matrix, Black & White by primary/secondary weights, Gradient Map
driven by luminosity, Exposure in linear 2.2, Threshold on the
0.30/0.59/0.11 luminosity, Posterize by whole-level bins.

Depths: the document model is always 8-bit pixels, but the
evaluation follows the *source* depth's rules (16-bit posterize /
threshold bins, 32-bit float Levels and unclipped HDR paths) where
the source depth is known — the corpus carries 16- and 32-bit
files, and the oracle measures against Photoshop's own render of
them. A block that cannot be parsed is ``{"kind": "unsupported"}``
with the reason named — never a crash, never a silent identity.
"""

from __future__ import annotations

import struct

import numpy as np

from .blend import (
    cmyk_to_rgb,
    cmyk_to_rgb_arr,
    hsb_to_rgb,
    lab_to_rgb_arr,
    lab_to_srgb,
    linear_to_srgb,
    lum,
    rgb_to_cmyk_arr,
    rgb_to_lab,
    set_lum,
    srgb_to_linear,
)
from .descriptor import (
    DescError,
    desc_bool,
    desc_desc,
    desc_num,
    parse_versioned,
)
from .gradient import bake_stops, method_from_code

__all__ = [
    "ADJUSTMENT_NAMES",
    "apply_adjustment",
    "parse_adjustment",
]

#: Human names for the adjustment keys (the flatten's notes).
ADJUSTMENT_NAMES: dict[str, str] = {
    "brit": "brightness/contrast",
    "curv": "curves",
    "levl": "levels",
    "hue2": "hue/saturation",
    "mixr": "channel mixer",
    "blnc": "color balance",
    "phfl": "photo filter",
    "blwh": "black and white",
    "grdm": "gradient map",
    "post": "posterize",
    "thrs": "threshold",
    "selc": "selective color",
    "vibA": "vibrance",
    "expA": "exposure",
    "nvrt": "invert",
    "clrL": "color lookup",
}


# -- binary readers ------------------------------------------------------

def _be16(data: bytes, at: int) -> int | None:
    if at < 0 or at + 2 > len(data):
        return None
    return struct.unpack(">H", data[at:at + 2])[0]


def _bei16(data: bytes, at: int) -> int | None:
    if at < 0 or at + 2 > len(data):
        return None
    return struct.unpack(">h", data[at:at + 2])[0]


def _be32(data: bytes, at: int) -> int | None:
    if at < 0 or at + 4 > len(data):
        return None
    return struct.unpack(">I", data[at:at + 4])[0]


def _bef32(data: bytes, at: int) -> float | None:
    if at < 0 or at + 4 > len(data):
        return None
    return struct.unpack(">f", data[at:at + 4])[0]


# -- parsers --------------------------------------------------------------

def _levels_rec(data: bytes, at: int) -> dict | None:
    in_black = _be16(data, at)
    in_white = _be16(data, at + 2)
    out_black = _be16(data, at + 4)
    out_white = _be16(data, at + 6)
    gamma = _be16(data, at + 8)
    if None in (in_black, in_white, out_black, out_white, gamma):
        return None
    return {
        "in_black": in_black / 255.0,
        "in_white": in_white / 255.0,
        "out_black": out_black / 255.0,
        "out_white": out_white / 255.0,
        "gamma": gamma / 100.0,
    }


def _parse_curves(data: bytes) -> dict | None:
    """``curv``: flag(1) version(2) bitmap(4), then per set bit a
    point list (u16 count, then (out, in) u16 pairs) or, for a
    pencil-drawn map, 256 output bytes per channel."""
    if not data:
        return None
    pencil = data[0] != 0
    version = _be16(data, 1)
    if version not in (1, 4):
        return None
    bits = _be32(data, 3)
    if bits is None:
        return None
    at = 7
    identity = [(0.0, 0.0), (1.0, 1.0)]
    curves: list[list[tuple[float, float]]] = [
        list(identity) for _ in range(5)]
    for bit in range(32):
        if not (bits & (1 << bit)):
            continue
        if bit >= 5:
            return None
        if pencil:
            if at + 256 > len(data):
                return None
            mapping = list(data[at:at + 256])
            at += 256
            curves[bit] = [(i / 255.0, mapping[i] / 255.0)
                           for i in range(256)]
            continue
        count = _be16(data, at)
        if count is None:
            return None
        at += 2
        points = []
        for _ in range(count):
            out = _be16(data, at)
            inp = _be16(data, at + 2)
            if out is None or inp is None:
                return None
            at += 4
            points.append((inp / 255.0, out / 255.0))
        curves[bit] = points
    return {
        "kind": "curves",
        "master": curves[0],
        "per_channel": curves[1:4],
        "black": curves[4],
        "pencil": pencil,
    }


def _parse_selective(data: bytes) -> dict | None:
    if _be16(data, 0) != 1 or len(data) < 84:
        return None
    relative = _be16(data, 2) == 0
    adjustments = []
    for record in range(9):
        row = []
        for k in range(4):
            value = _bei16(data, 4 + (record + 1) * 8 + k * 2)
            if value is None:
                return None
            row.append(float(value))
        adjustments.append(row)
    return {"kind": "selective_color", "relative": relative,
            "adjustments": adjustments}


def _parse_gradient_map(data: bytes) -> dict | None:
    """``grdm``: version (1, or 3 with a method key), reverse,
    dither, [method], name, colour stops (location /4096, midpoint
    %, colour space + four u16), transparency stops, smoothness
    (/4096)."""
    version = _be16(data, 0)
    if version not in (1, 3):
        return None
    reverse = data[2] != 0 if len(data) > 2 else False
    dither = data[3] != 0 if len(data) > 3 else False
    at = 8 if version == 3 else 4
    name_len = _be32(data, at)
    if name_len is None or name_len > 100000:
        return None
    at = at + 4 + name_len * 2
    count = _be16(data, at)
    if count is None:
        return None
    at += 2
    stops = []
    mids = []
    for _ in range(count):
        loc = _be32(data, at)
        mid = _be32(data, at + 4)
        space = _be16(data, at + 8)
        if None in (loc, mid, space):
            return None
        comps = [_be16(data, at + 10 + 2 * k) for k in range(4)]
        if None in comps:
            return None
        color = _decode_color(space, comps)
        if color is None:
            return None
        stops.append((loc / 4096.0, color))
        mids.append(mid / 100.0)
        at += 20
    if len(stops) < 2:
        return None
    trans_count = _be16(data, at)
    if trans_count is None:
        return None
    at += 2 + trans_count * 10 + 2
    smooth = _be16(data, at)
    smooth = (smooth / 4096.0) if smooth is not None else 0.0
    # The interpolation method lives in the version-3 header
    # (bytes 4..8: 'Gcls' classic, 'Perc', 'Lnr ', 'Smoo'); a
    # version-1 block is always classic. The bake reproduces
    # Photoshop's interpolation: Catmull-Rom blended with linear
    # by the smoothness, in the method's colour space — a plain
    # linear bake is up to 0.05 off on a three-stop ramp.
    method = "classic"
    if version == 3 and len(data) >= 8:
        method = method_from_code(data[4:8].decode("latin-1"))
    order = sorted(range(len(stops)), key=lambda i: stops[i][0])
    ordered = [stops[i] for i in order]
    # Midpoint k applies to the segment after stop k (in location
    # order): the warp for segment i uses the mid of stop i+1.
    seg_mids = [mids[order[i + 1]] if i + 1 < len(mids) else 0.5
                for i in range(len(ordered) - 1)]
    baked = bake_stops(ordered, seg_mids, smooth, method)
    return {"kind": "gradient_map", "stops": baked,
            "reverse": reverse, "dither": dither}


def _decode_color(space: int, comps: list[int]
                  ) -> tuple[float, float, float] | None:
    """A PSD colour structure (0 RGB, 1 HSB, 2 CMYK, 7 Lab, 8
    Grayscale) as an sRGB triple in 0..1."""
    n = lambda v: v / 65535.0  # noqa: E731 - tiny local
    if space == 0:
        return (n(comps[0]), n(comps[1]), n(comps[2]))
    if space == 1:
        return hsb_to_rgb(n(comps[0]) * 360.0, n(comps[1]),
                          n(comps[2]))
    if space == 2:
        return cmyk_to_rgb((1.0 - n(comps[0]), 1.0 - n(comps[1]),
                            1.0 - n(comps[2]), 1.0 - n(comps[3])))
    if space == 7:
        lum_ = min(comps[0], 10000) / 10000.0
        a = float(struct.unpack(">h", struct.pack(
            ">H", comps[1]))[0]) / 100.0
        b = float(struct.unpack(">h", struct.pack(
            ">H", comps[2]))[0]) / 100.0
        return lab_to_srgb((lum_, a + 128.0, b + 128.0))
    if space == 8:
        value = 1.0 - min(comps[0], 10000) / 10000.0
        return (value, value, value)
    return None


def _parse_channel_mixer(data: bytes) -> dict | None:
    if _be16(data, 0) != 1:
        return None
    monochrome = _be16(data, 2) != 0
    matrix = []
    for record in range(3):
        at = 4 + record * 10
        row = []
        for k in (0, 1, 2, 4):
            value = _bei16(data, at + k * 2)
            if value is None:
                return None
            row.append(max(-200.0, min(200.0, float(value)))
                       / 100.0)
        matrix.append(row)
    return {"kind": "channel_mixer", "matrix": matrix,
            "monochrome": monochrome}


def _parse_photo_filter(data: bytes) -> dict | None:
    version = _be16(data, 0)
    if version == 2:
        space = _be16(data, 2)
        if space not in (0, 2, 7, 8):
            return None
        comps = [_be16(data, 4 + 2 * k) for k in range(4)]
        if None in comps:
            return None
        color = _decode_color(space, comps)
        if color is None:
            return None
        at = 12
    elif version == 3:
        xyz = [_be32(data, 2 + 4 * k) for k in range(3)]
        if None in xyz:
            return None
        rel = [v / 32768.0 for v in xyz]
        if any(v > 2.0 for v in rel):
            return None
        # Relative to the D50 white: back to absolute D50 XYZ,
        # then to sRGB through the linear matrix.
        xyz_abs = [rel[0] * 0.96422, rel[1], rel[2] * 0.82521]
        from .blend import XYZ_D50_TO_SRGB

        lin = XYZ_D50_TO_SRGB @ np.array(xyz_abs, dtype=np.float64)
        srgb = linear_to_srgb(np.clip(lin, 0.0, 1.0))
        color = (float(srgb[0]), float(srgb[1]), float(srgb[2]))
        at = 14
    else:
        return None
    density = _be32(data, at)
    if density is None or len(data) <= at + 4:
        return None
    preserve = data[at + 4] != 0
    return {"kind": "photo_filter",
            "color": tuple(min(1.0, max(0.0, c)) for c in color),
            "density": min(1.0, max(0.0, density / 100.0)),
            "preserve_luminosity": preserve}


def _parse_color_balance(data: bytes) -> dict | None:
    tones = []
    for k in range(3):
        row = []
        for i in range(3):
            value = _bei16(data, (k * 3 + i) * 2)
            if value is None:
                return None
            row.append(max(-100.0, min(100.0, float(value))))
        tones.append(row)
    preserve = len(data) > 18 and data[18] != 0
    return {"kind": "color_balance",
            "shadows": tones[0], "midtones": tones[1],
            "highlights": tones[2],
            "preserve_luminosity": preserve}


def _parse_vibrance(data: bytes) -> dict | None:
    try:
        desc, _ = parse_versioned(data)
    except DescError:
        return None
    get = lambda k: (desc_num(desc, k) or 0.0)  # noqa: E731
    return {"kind": "vibrance",
            "vibrance": max(-100.0, min(100.0, get("vibrance"))),
            "saturation": max(-100.0, min(100.0, get("Strt")))}


def _parse_black_white(data: bytes) -> dict | None:
    try:
        desc, _ = parse_versioned(data)
    except DescError:
        return None
    defaults = [40.0, 60.0, 40.0, 60.0, 20.0, 80.0]
    keys = ["Rd  ", "Yllw", "Grn ", "Cyn ", "Bl  ", "Mgnt"]
    weights = []
    for default, key in zip(defaults, keys, strict=True):
        value = desc_num(desc, key)
        weights.append(default if value is None else
                       max(-200.0, min(300.0, value)))
    tint = None
    if desc_bool(desc, "useTint"):
        tint_desc = desc_desc(desc, "tintColor")
        if tint_desc is not None:
            from .descriptor import color_from_desc

            tint = color_from_desc(tint_desc)
        if tint is None:
            tint = (225.0 / 255.0, 211.0 / 255.0, 179.0 / 255.0)
    return {"kind": "black_white", "weights": weights,
            "tint": tint}


def parse_adjustment(key: str, data: bytes,
                     cged: bytes | None = None) -> dict:
    """Parse one adjustment block into an adjustment dict. An
    unparsable block is ``{"kind": "unsupported"}`` with the key
    and reason — the caller renders it as unrendered, never as
    identity."""
    def unsupported(reason: str) -> dict:
        return {"kind": "unsupported", "key": key,
                "reason": reason}

    if key == "nvrt":
        return {"kind": "invert"}
    if key == "thrs":
        level = _be16(data, 0)
        if level is None:
            return unsupported("threshold block is truncated")
        return {"kind": "threshold", "level": level / 255.0}
    if key == "post":
        levels = _be16(data, 0)
        if levels is None:
            return unsupported("posterize block is truncated")
        return {"kind": "posterize", "levels": levels}
    if key == "brit":
        # Modern: the CgEd descriptor carries Brgh/Cntr/useLegacy;
        # without it, the block is two i16 (legacy).
        if cged:
            try:
                desc, _ = parse_versioned(cged)
                brightness = desc_num(desc, "Brgh")
                contrast = desc_num(desc, "Cntr")
                if brightness is not None \
                        and contrast is not None:
                    return {
                        "kind": "brightness_contrast",
                        "brightness": brightness,
                        "contrast": contrast,
                        "legacy": bool(
                            desc_bool(desc, "useLegacy")),
                    }
            except DescError:
                pass
        brightness = _bei16(data, 0)
        contrast = _bei16(data, 2)
        if brightness is None or contrast is None:
            return unsupported(
                "brightness/contrast block is truncated")
        return {"kind": "brightness_contrast",
                "brightness": float(brightness),
                "contrast": float(contrast), "legacy": True}
    if key == "hue2":
        if len(data) < 16:
            return unsupported("hue/saturation block is truncated")
        colorize = data[2] != 0
        base = 4 if colorize else 10
        hue = _bei16(data, base)
        sat = _bei16(data, base + 2)
        light = _bei16(data, base + 4)
        if None in (hue, sat, light):
            return unsupported("hue/saturation master is truncated")
        ranges = []
        for i in range(6):
            at = 16 + i * 14
            vals = [_bei16(data, at + 2 * k) for k in range(7)]
            if None in vals:
                ranges.append(_default_hue_range(i))
                continue
            bounds = _canonical_bounds(
                [float(v) for v in vals[:4]])
            ranges.append({
                "hue": float(vals[4]),
                "saturation": float(vals[5]),
                "lightness": float(vals[6]),
                "bounds": bounds,
            })
        return {"kind": "hue_saturation",
                "hue": (hue % 360.0) if colorize else float(hue),
                "saturation": float(sat),
                "lightness": float(light),
                "colorize": colorize, "ranges": ranges}
    if key == "expA":
        exposure = _bef32(data, 2)
        offset = _bef32(data, 6)
        gamma = _bef32(data, 10)
        if None in (exposure, offset, gamma):
            return unsupported("exposure block is truncated")
        return {"kind": "exposure", "exposure": exposure,
                "offset": offset, "gamma": gamma}
    if key == "levl":
        master = _levels_rec(data, 2)
        per_channel = [_levels_rec(data, at)
                       for at in (12, 22, 32)]
        black = _levels_rec(data, 42)
        if master is None or None in per_channel:
            return unsupported("levels block is truncated")
        return {"kind": "levels", "master": master,
                "per_channel": per_channel,
                "black": black or _levels_identity()}
    if key == "curv":
        parsed = _parse_curves(data)
        if parsed is None:
            return unsupported("curves block is unreadable")
        return parsed
    if key == "selc":
        parsed = _parse_selective(data)
        if parsed is None:
            return unsupported("selective color block is truncated")
        return parsed
    if key == "grdm":
        parsed = _parse_gradient_map(data)
        if parsed is None:
            return unsupported("gradient map block is unreadable")
        return parsed
    if key == "mixr":
        parsed = _parse_channel_mixer(data)
        if parsed is None:
            return unsupported("channel mixer block is truncated")
        return parsed
    if key == "phfl":
        parsed = _parse_photo_filter(data)
        if parsed is None:
            return unsupported("photo filter block is unreadable")
        return parsed
    if key == "blnc":
        parsed = _parse_color_balance(data)
        if parsed is None:
            return unsupported(
                "color balance block is truncated")
        return parsed
    if key == "vibA":
        parsed = _parse_vibrance(data)
        if parsed is None:
            return unsupported("vibrance block is unreadable")
        return parsed
    if key == "blwh":
        parsed = _parse_black_white(data)
        if parsed is None:
            return unsupported("black and white block is unreadable")
        return parsed
    return unsupported(f"unknown adjustment key {key!r}")


def _levels_identity() -> dict:
    return {"in_black": 0.0, "in_white": 1.0, "out_black": 0.0,
            "out_white": 1.0, "gamma": 1.0}


def _default_hue_range(index: int) -> dict:
    defaults = [(-45.0, -15.0, 15.0, 45.0),
                (15.0, 45.0, 75.0, 105.0),
                (75.0, 105.0, 135.0, 165.0),
                (135.0, 165.0, 195.0, 225.0),
                (195.0, 225.0, 255.0, 285.0),
                (255.0, 285.0, 315.0, 345.0)]
    return {"hue": 0.0, "saturation": 0.0, "lightness": 0.0,
            "bounds": list(defaults[index])}


def _canonical_bounds(values: list[float]) -> list[float]:
    """Fold four range bounds into canonical form (monotone,
    centred on the range's middle, within one turn)."""
    if any(v != v or v in (float("inf"), float("-inf"))
           for v in values):
        return [-45.0, -15.0, 15.0, 45.0]
    out = [values[0]] * 4
    for k in range(1, 4):
        out[k] = min(out[k - 1] + (values[k] - out[k - 1]) % 360.0,
                     out[0] + 359.0)
    centre = (out[1] + out[2]) / 2.0
    shift = (centre + 30.0) % 360.0 - 30.0 - centre
    return [v + shift for v in out]


# -- evaluation ----------------------------------------------------------

def _round_levels(values: np.ndarray, quantum: float | None
                  ) -> np.ndarray:
    if quantum is None:
        return values
    return np.round(values * quantum) / quantum


def _levels_gamma(u: np.ndarray, gamma: float) -> np.ndarray:
    """Levels' midtone gamma: ``u^(1/g)`` above 1, with a toe
    ``a·u − b·u^2.2`` of slope ``2^g`` joining it below."""
    u = np.asarray(u, dtype=np.float64)
    gamma = max(gamma, 0.01)
    if gamma <= 1.0:
        return np.power(np.clip(u, 0.0, None), 1.0 / gamma)
    c = 1.0 / gamma
    a = 2.0 ** gamma
    q = 2.2
    u1 = ((1.0 - c / q) / (a * (1.0 - 1.0 / q))) ** (
        1.0 / (1.0 - c)) if c != 1.0 else 0.0
    out = np.power(np.clip(u, 0.0, None), c)
    if u1 > 0.0:
        below = u < u1
        if np.any(below):
            b = (a * u1 - u1 ** c) / (u1 ** q) \
                if u1 ** q > 0 else 0.0
            if np.isfinite(b):
                toe = a * u - b * np.power(
                    np.clip(u, 0.0, None), q)
                out = np.where(below, toe, out)
    return out


def _levels_channel(channel: dict, values: np.ndarray,
                    quantum: float | None, is_float: bool
                    ) -> np.ndarray:
    """One Levels channel over ``values`` (0..1)."""
    values = np.asarray(values, dtype=np.float64)
    if is_float:
        span = max(channel["in_white"] - channel["in_black"], 1e-6)
        t = (values - channel["in_black"]) / span
        if channel["gamma"] != 1.0:
            t = np.sign(t) * np.power(
                np.abs(t), 1.0 / max(channel["gamma"], 0.01))
        return channel["out_black"] + t * (
            channel["out_white"] - channel["out_black"])
    span = max(channel["in_white"] - channel["in_black"], 1e-6)
    if quantum is not None:
        v = np.round(values * quantum) / quantum
        u = (np.floor(((v - channel["in_black"]) / span)
                      .clip(0.0, 1.0) * quantum + 0.5 + 1e-3)
             / quantum)
    else:
        u = ((values - channel["in_black"]) / span).clip(0.0, 1.0)
    t = _levels_gamma(u, channel["gamma"])
    if quantum is not None:
        t = np.floor(t * quantum + 0.5) / quantum
    out = channel["out_black"] + t * (
        channel["out_white"] - channel["out_black"])
    if quantum is not None:
        out = np.floor(out * quantum + 0.5) / quantum
    return out


def _spline_second_derivatives(points: list
                               ) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray]:
    """Natural cubic spline second derivatives (f64, Thomas
    algorithm) for sorted (input, output) points."""
    pts = sorted(points)
    dedup = [pts[0]]
    for point in pts[1:]:
        if abs(point[0] - dedup[-1][0]) >= 1e-6:
            dedup.append(point)
    xs = np.array([p[0] for p in dedup], dtype=np.float64)
    ys = np.array([p[1] for p in dedup], dtype=np.float64)
    n = len(xs)
    m2 = np.zeros(n, dtype=np.float64)
    if n > 2:
        c = np.zeros(n, dtype=np.float64)
        d = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            h0 = xs[i] - xs[i - 1]
            h1 = xs[i + 1] - xs[i]
            a = h0 / 6.0
            b = (h0 + h1) / 3.0
            cc = h1 / 6.0
            r = ((ys[i + 1] - ys[i]) / h1
                 - (ys[i] - ys[i - 1]) / h0)
            denom = b - a * c[i - 1]
            c[i] = cc / denom if denom != 0 else 0.0
            d[i] = (r - a * d[i - 1]) / denom if denom != 0 else 0.0
        for i in range(n - 2, 0, -1):
            m2[i] = d[i] - c[i] * m2[i + 1]
    return xs, ys, m2


def _spline_eval(points: list, values: np.ndarray) -> np.ndarray:
    """A natural cubic spline through ``points`` at ``values``
    (constant outside the first/last point, clamped 0..1)."""
    values = np.asarray(values, dtype=np.float64)
    if len(points) < 2:
        return values.copy()
    xs, ys, m2 = _spline_second_derivatives(points)
    out = np.empty_like(values)
    below = values <= xs[0]
    above = values >= xs[-1]
    out = np.where(below, ys[0], out)
    out = np.where(above, ys[-1], out)
    mid = ~(below | above)
    if np.any(mid):
        v = values[mid]
        idx = np.searchsorted(xs, v, side="right") - 1
        idx = np.clip(idx, 0, len(xs) - 2)
        x0, x1 = xs[idx], xs[idx + 1]
        y0, y1 = ys[idx], ys[idx + 1]
        h = x1 - x0
        a = (x1 - v) / h
        b = (v - x0) / h
        y = (a * y0 + b * y1
             + ((a ** 3 - a) * m2[idx]
                + (b ** 3 - b) * m2[idx + 1]) * h * h / 6.0)
        out[mid] = y
    return np.clip(out, 0.0, 1.0)


def _curve_eval(points: list, values: np.ndarray, pencil: bool
                ) -> np.ndarray:
    """One Curves curve: a pencil map interpolates its 256
    entries; a point list is a natural spline."""
    values = np.asarray(values, dtype=np.float64)
    if pencil:
        table = np.array([p[1] for p in points], dtype=np.float64)
        if len(table) < 256:
            table = np.interp(np.linspace(0, 255, 256),
                              np.linspace(0, len(table) - 1,
                                          len(table)), table)
        return np.clip(np.interp(values * 255.0,
                                 np.arange(256), table), 0.0, 1.0)
    return _spline_eval(points, values)


def _modern_brightness_raised(values: np.ndarray,
                              brightness: float) -> np.ndarray:
    """Modern Brightness raised (brightness > 0): a line of slope
    ``1.375^(b/50)`` to the knee at 126/255, then a cubic Hermite
    (or a power roll-off when the Hermite would overshoot)."""
    b = abs(brightness)
    s = 1.375 ** (b / 50.0)
    e = 1.0 / (1.0 + 0.084 * b)
    knee = 126.0 / 255.0
    x0 = min(knee / s, 1.0)
    secant = (1.0 - knee) / max(1.0 - x0, 1e-6)
    alpha = s / secant
    beta = e / secant
    hermite = alpha * alpha + beta * beta <= 9.0
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    low = values <= x0
    out = np.where(low, np.minimum(knee, s * values), out)
    high = ~low
    if np.any(high):
        h = max(1.0 - x0, 1e-6)
        t = np.clip((values[high] - x0) / h, 0.0, 1.0)
        if hermite:
            t2, t3 = t * t, t * t * t
            y = ((2.0 * t3 - 3.0 * t2 + 1.0) * knee
                 + (t3 - 2.0 * t2 + t) * h * s
                 + (-2.0 * t3 + 3.0 * t2)
                 + (t3 - t2) * h * e)
        else:
            y = 1.0 - (1.0 - knee) * (1.0 - t) ** alpha
        out[high] = np.clip(y, 0.0, 1.0)
    return out


def _modern_brightness(values: np.ndarray, brightness: float
                       ) -> np.ndarray:
    """Modern Brightness (both signs); lowered is the inverse of
    the raised curve, by monotone interpolation."""
    values = np.asarray(values, dtype=np.float64)
    if brightness == 0.0 or not np.isfinite(brightness):
        return values
    if brightness > 0.0:
        return _modern_brightness_raised(values, brightness)
    if np.all((values <= 0.0) | (values >= 1.0)):
        return values
    fine = 16384
    xs = np.linspace(0.0, 1.0, fine)
    ys = _modern_brightness_raised(xs, -brightness)
    return np.interp(values, ys, xs)


def _modern_contrast(values: np.ndarray, contrast: float
                     ) -> np.ndarray:
    """Modern Contrast: a symmetric cubic-Hermite S-curve pivoting
    at 0.5 (endpoint slope ``1 − c/128``, centre slope
    ``1 + c/128``)."""
    values = np.asarray(values, dtype=np.float64)
    if contrast == 0.0:
        return values
    k = contrast / 128.0
    end, mid = 1.0 - k, 1.0 + k

    def half(t: np.ndarray) -> np.ndarray:
        return (end * 0.5 * (t ** 3 - 2.0 * t ** 2 + t)
                + (-2.0 * t ** 3 + 3.0 * t ** 2) * 0.5
                + mid * 0.5 * (t ** 3 - t ** 2))

    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.5, half(values / 0.5),
                    1.0 - half((1.0 - values) / 0.5))


def _legacy_bc(values: np.ndarray, brightness: float,
               contrast: float, pivot: float = 128.0 / 255.0
               ) -> np.ndarray:
    """Legacy Brightness/Contrast: brightness added before the
    contrast scale when contrast is raised, after it when lowered;
    contrast pivots at 128/255."""
    b = brightness / 255.0
    c = min(99.0, max(-100.0, contrast))
    k = 1.0 / (1.0 - c / 100.0) if c >= 0.0 else 1.0 + c / 100.0
    pre, post = (b, 0.0) if c >= 0.0 else (0.0, b)
    return np.clip((values + pre - pivot) * k + pivot + post,
                   0.0, 1.0)


def _hue_weight(bounds: list[float], degrees: np.ndarray
                ) -> np.ndarray:
    """How much a pixel of hue ``degrees`` belongs to a range."""
    a, b, c, d = [float(v) for v in bounds]
    def rel(x):
        return np.mod(x - a, 360.0)

    b_r = rel(b)
    c_r = np.maximum(rel(c), b_r)
    d_r = np.maximum(rel(d), c_r)
    h = rel(degrees)
    out = np.zeros_like(h)
    rising = h <= b_r
    out = np.where(rising, np.where(b_r <= 0.0, 1.0, h / np.where(
        b_r <= 0.0, 1.0, b_r)), out)
    flat = (h > b_r) & (h <= c_r)
    out = np.where(flat, 1.0, out)
    falling = (h > c_r) & (h <= d_r)
    out = np.where(falling, np.where(
        d_r <= c_r, 1.0, (d_r - h) / np.where(
            d_r <= c_r, 1.0, d_r - c_r)), out)
    return out


def _rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.float64)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    light = (mx + mn) / 2.0
    d = mx - mn
    sat = np.where(d < 1e-7, 0.0,
                   np.where(light > 0.5, d / np.where(
                       2.0 - mx - mn == 0, 1.0, 2.0 - mx - mn),
                       d / np.where(mx + mn == 0, 1.0, mx + mn)))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.zeros_like(mx)
    mask_r = (mx == r) & (d >= 1e-7)
    mask_g = (mx == g) & (d >= 1e-7) & ~mask_r
    mask_b = (mx == b) & (d >= 1e-7) & ~mask_r & ~mask_g
    hue = np.where(mask_r, np.mod((g - b) / np.where(
        d == 0, 1.0, d), 6.0), hue)
    hue = np.where(mask_g, (b - r) / np.where(d == 0, 1.0, d)
                   + 2.0, hue)
    hue = np.where(mask_b, (r - g) / np.where(d == 0, 1.0, d)
                   + 4.0, hue)
    return hue / 6.0, sat, light


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray,
                light: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    light = np.asarray(light, dtype=np.float64)
    q = np.where(light < 0.5, light * (1.0 + s),
                 light + s - light * s)
    p = 2.0 * light - q
    def channel(t: np.ndarray) -> np.ndarray:
        t = np.mod(t, 1.0)
        return np.where(
            t < 1.0 / 6.0, p + (q - p) * 6.0 * t,
            np.where(t < 0.5, q,
                     np.where(t < 2.0 / 3.0,
                              p + (q - p) * (2.0 / 3.0 - t) * 6.0,
                              p)))
    r = channel(h + 1.0 / 3.0)
    g = channel(h)
    b = channel(h - 1.0 / 3.0)
    out = np.stack([r, g, b], axis=-1)
    return np.where((s <= 0.0)[..., None], light[..., None], out)


def _hue_rgb(h: np.ndarray, p: np.ndarray, q: np.ndarray
             ) -> np.ndarray:
    def channel(t: np.ndarray) -> np.ndarray:
        t = np.mod(t, 1.0)
        return np.where(
            t < 1.0 / 6.0, p + (q - p) * 6.0 * t,
            np.where(t < 0.5, q,
                     np.where(t < 2.0 / 3.0,
                              p + (q - p) * (2.0 / 3.0 - t) * 6.0,
                              p)))
    return np.stack([channel(h + 1.0 / 3.0), channel(h),
                     channel(h - 1.0 / 3.0)], axis=-1)


def _black_white_gray(rgb: np.ndarray, weights: list[float]
                      ) -> np.ndarray:
    """Black & White grey value: smallest channel + weighted
    secondary (middle − smallest) + primary (largest − middle)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    mid = r + g + b - mx - mn
    primary = np.where((r >= g) & (r >= b), 0,
                       np.where(g >= b, 2, 4))
    secondary = np.where((b <= r) & (b <= g), 1,
                         np.where(r <= g, 3, 5))
    w = np.array(weights, dtype=np.float64)
    w_primary = w[primary]
    w_secondary = w[secondary]
    return (mn + (mid - mn) * w_secondary / 100.0
            + (mx - mid) * w_primary / 100.0)


def _balance_curve(tones: list[list[float]],
                   preserve: bool, channel: int
                   ) -> tuple[float, float, float,
                              tuple | None]:
    """One channel's Color Balance curve: black/white points, the
    power exponent, and the gamma > 1 knee (Hermite join)."""
    shadows, midtones, highlights = [
        [min(100.0, max(-100.0, s)) for s in t] for t in tones]
    def own(t):
        return t[channel]

    def mx(t):
        return max(t)

    def mn(t):
        return min(t)
    if preserve:
        black = mx(shadows) - own(shadows)
        white = own(highlights) - mn(highlights)
        m = own(midtones) - (mx(midtones) + mn(midtones)) / 2.0
    else:
        black = max(-own(shadows), 0.0)
        white = max(own(highlights), 0.0)
        m = own(midtones) + (own(shadows) + own(highlights)) / 2.0
    gamma = 2.0 ** (m / 100.0)
    e = 1.0 / max(gamma, 0.01)
    knee = None
    if gamma > 1.0:
        s0 = 2.0 ** gamma
        u0 = min((2.0 * s0 ** (1.0 / (e - 1.0))), 1.0) \
            if e != 1.0 else 0.0
        if u0 > 0.0 and np.isfinite(u0):
            knee = (u0, s0, u0 ** e, e * u0 ** (e - 1.0))
    return black / 255.0, 1.0 - white / 255.0, e, knee


def _balance_eval(curve: tuple, values: np.ndarray) -> np.ndarray:
    b, w, e, knee = curve
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    out = np.ones_like(values)
    out = np.where(values <= b, 0.0, out)
    mid = (values > b) & (values < w)
    if np.any(mid):
        u = (values[mid] - b) / (w - b)
        if knee is not None:
            u0, s0, p, d = knee
            below = u < u0
            if np.any(below):
                t = u[below] / u0
                t2, t3 = t * t, t * t * t
                y = ((t3 - 2.0 * t2 + t) * u0 * s0
                     + (3.0 * t2 - 2.0 * t3) * p
                     + (t3 - t2) * u0 * d)
                # Scatter the knee values back into the full
                # array (``y`` holds only the below-knee
                # entries — a ``where`` over it would not
                # broadcast).
                u = u.copy()
                u[below] = np.clip(y, 0.0, 1.0)
                rest = ~below
                u[rest] = np.power(np.clip(u[rest], 0.0, None),
                                   e)
            else:
                u = np.power(np.clip(u, 0.0, None), e)
        else:
            u = np.power(np.clip(u, 0.0, None), e)
        out[mid] = u
    return out


def _selective_weights(rgb: np.ndarray) -> np.ndarray:
    """How much a colour belongs to each Selective Color range:
    reds, yellows, greens, cyans, blues, magentas, whites,
    neutrals, blacks."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    mid = rgb[..., 0] + rgb[..., 1] + rgb[..., 2] - mx - mn
    def top(i):
        return np.where(rgb[..., i] >= mx, mx - mid, 0.0)

    def bottom(i):
        return np.where(rgb[..., i] <= mn, mid - mn, 0.0)
    weights = np.stack([
        top(0), bottom(2), top(1), bottom(0), top(2), bottom(1),
        np.maximum((mn - 0.5) * 2.0, 0.0),
        np.clip(1.0 - np.abs(mx - 0.5) - np.abs(mn - 0.5),
                0.0, 1.0),
        np.maximum((0.5 - mx) * 2.0, 0.0),
    ], axis=-1)
    return weights


def _photo_filter_matrix(color: tuple[float, float, float],
                         density: float) -> np.ndarray:
    """Photo Filter's linear-light RGB matrix: the filter colour's
    D50 XYZ (mixed in by density) folded into one 3x3 map."""
    from .blend import D50, SRGB_TO_XYZ_D50, XYZ_D50_TO_SRGB

    d = min(1.0, max(0.0, density))
    lin = srgb_to_linear(np.array(color, dtype=np.float64))
    xyz = SRGB_TO_XYZ_D50 @ lin
    s = np.array([1.0 - d + d * xyz[i] / D50[i]
                  for i in range(3)])
    return np.array([
        [sum(XYZ_D50_TO_SRGB[r][k] * s[k] * SRGB_TO_XYZ_D50[k][c]
             for k in range(3)) for c in range(3)]
        for r in range(3)], dtype=np.float64)


def _vibrance_lin(rgb: np.ndarray, v: float) -> np.ndarray:
    """The Vibrance slider in linear light (HSV change keeping
    the hue; positive boost damped on reds and skin)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    out = rgb.copy()
    live = (mx > 0.0) & ((mx - mn) > 1e-9)
    if not np.any(live):
        return out
    t = 1.0 - mn / np.where(mx > 0, mx, 1.0)
    shape = (t * (1.0 - t) * (2.0 - t + t * t) * mx
             * np.maximum(1.0 - mx, 0.0))
    if v < 0.0:
        a = -v
        t2 = t * (1.0 - a / 4.0) * (
            1.0 - a + a * t * (1.0 + t) / 2.0)
        v2 = mx - a * shape
    else:
        c0, c1, p, q, r0, ra, rt = (
            0.5001, 0.0887, 0.4374, 0.7573, -0.8113, 0.5154,
            2.8026)
        # HSV hue in degrees, for the skin damping.
        d = mx - mn
        hue = np.zeros_like(mx)
        mask_r = (mx == rgb[..., 0]) & (d > 0)
        mask_g = (mx == rgb[..., 1]) & (d > 0) & ~mask_r
        mask_b = (mx == rgb[..., 2]) & (d > 0) & ~mask_r & ~mask_g
        hue = np.where(mask_r, np.mod(
            (rgb[..., 1] - rgb[..., 2]) / np.where(d == 0, 1.0, d),
            6.0) * 60.0, hue)
        hue = np.where(mask_g, ((rgb[..., 2] - rgb[..., 0])
                                / np.where(d == 0, 1.0, d) + 2.0)
                       * 60.0, hue)
        hue = np.where(mask_b, ((rgb[..., 0] - rgb[..., 1])
                                / np.where(d == 0, 1.0, d) + 4.0)
                       * 60.0, hue)
        skin = np.where(hue <= 30.0, 1.0,
                        np.where(hue < 45.0, (45.0 - hue) / 15.0,
                                 np.where(hue < 300.0, 0.0,
                                          (hue - 300.0) / 60.0)))
        a = v * (1.0 - skin) + 0.64 * (v ** 1.6) * skin
        boost = ((c0 * a + c1 * a * a) * np.power(t, p)
                 * np.power(1.0 - t, q)
                 * np.exp((r0 + ra * a + rt * (1.0 - t))
                          * np.maximum(1.0 - mx, 0.0)))
        t2 = np.minimum(t * (1.0 + boost), 1.0)
        v2 = mx + v / 4.0 * shape
    m2 = v2 * (1.0 - t2)
    k = (v2 - m2) / np.where(mx - mn == 0, 1.0, mx - mn)
    out = np.where(live[..., None],
                   m2[..., None] + (rgb - mn[..., None])
                   * k[..., None], out)
    return out


def _bayer4(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    matrix = np.array([0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0,
                       3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0])
    return (matrix[(y & 3) * 4 + (x & 3)] + 0.5) / 16.0 - 0.5


def _tone_row(adj: dict, kind: str, index: int, values: np.ndarray,
              quantum: float | None, is_float: bool,
              lab: bool) -> np.ndarray:
    """One channel's tone record composed with the master record
    (the master is ignored in Lab, which has no composite
    record). ``index`` 3 is the CMYK black record."""
    record = (adj.get("black") if index == 3
              else adj["per_channel"][index])
    if kind == "levels":
        if record is None or record == _levels_identity():
            out = values
        else:
            out = _levels_channel(record, values, quantum,
                                  is_float)
        master = adj["master"]
        if not lab and master != _levels_identity():
            out = _levels_channel(master, out, quantum, is_float)
        return out
    out = (values if record is None
           else _curve_eval(record, values, adj["pencil"]))
    master = adj["master"]
    if not lab and len(master) >= 2 \
            and master != [(0.0, 0.0), (1.0, 1.0)]:
        out = _curve_eval(master, out, adj["pencil"])
    return out


def _tone_in_space(rgb: np.ndarray, adj: dict, kind: str,
                   mode: str, quantum: float | None,
                   is_float: bool) -> np.ndarray:
    """Levels/Curves in a CMYK or Lab document: the records run
    over the native channels and the canvas is shifted by the
    difference the native change makes (the Rust's
    ``tone_in_space``)."""
    if mode == "CMYK":
        ink = rgb_to_cmyk_arr(rgb)
        bright = 1.0 - ink
        out_bright = np.empty_like(bright)
        for i in range(4):
            out_bright[..., i] = _tone_row(
                adj, kind, i, bright[..., i], quantum, is_float,
                lab=False)
        if np.array_equal(out_bright, bright):
            return rgb
        out_ink = 1.0 - out_bright
        before = cmyk_to_rgb_arr(ink)
        after = cmyk_to_rgb_arr(out_ink)
        return np.clip(rgb + after - before, 0.0, 1.0)
    lab = rgb_to_lab(rgb)
    n = np.stack([lab[..., 0], lab[..., 1] / 255.0,
                  lab[..., 2] / 255.0], axis=-1)
    o = np.empty_like(n)
    for i in range(3):
        o[..., i] = _tone_row(adj, kind, i, n[..., i], quantum,
                              is_float, lab=True)
    if np.array_equal(o, n):
        return rgb
    before = lab_to_rgb_arr(lab)
    after = lab_to_rgb_arr(np.stack(
        [o[..., 0], o[..., 1] * 255.0, o[..., 2] * 255.0],
        axis=-1))
    return np.clip(rgb + after - before, 0.0, 1.0)


def apply_adjustment(rgb: np.ndarray, alpha: np.ndarray,
                     adj: dict, *, depth: str = "u8",
                     mode: str = "RGB",
                     x0: int = 0, y0: int = 0) -> np.ndarray:
    """Apply one parsed adjustment to an RGB canvas (H, W, 3)
    float in 0..1, with ``alpha`` (H, W) alongside for the
    alpha-gated adjustments. ``depth`` is the source document's
    depth (``u8`` / ``u16`` / ``f32``) — it picks the
    depth-appropriate rules; ``mode`` is the document's colour
    mode (``RGB`` / ``L`` / ``CMYK`` / ``Lab``) — it picks the
    mode-appropriate transfer (Exposure's grey gamma).
    Returns a new RGB array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    kind = adj.get("kind")
    is_float = depth == "f32"
    quantum = None if is_float else (32768.0 if depth == "u16"
                                     else 255.0)
    if kind == "exposure":
        exposure = adj["exposure"]
        offset = adj["offset"]
        gamma = max(adj["gamma"], 0.01)
        m = 2.0 ** exposure
        if is_float:
            # 32-bit samples are linear already; Photoshop's
            # merged Exposure oracle keeps negative results
            # with a signed gamma.
            lin = rgb * m + offset
            if gamma == 1.0:
                return lin
            return np.sign(lin) * np.abs(lin) ** (1.0 / gamma)
        # RGB documents linearise through a pure 2.2 power
        # (not the sRGB curve); grey documents through
        # their dot-gain gamma (~1.732).
        g = 1.732 if mode in ("L", "Grayscale", "Gray") else 2.2
        lin = np.clip(rgb, 0.0, 1.0) ** g * m + offset
        lin = np.maximum(lin, 0.0)
        if gamma != 1.0:
            lin = lin ** (1.0 / gamma)
        return np.clip(lin ** (1.0 / g), 0.0, 1.0)
    if kind == "unsupported":
        return rgb
    if kind == "invert":
        # Photoshop inverts in the document's native space: the
        # inks of a CMYK document, the L/a/b samples of a Lab
        # document — inverting the RGB canvas instead is up to
        # 0.33 off on a CMYK file (invert-in-CMYK measures 0.00009
        # on the corpus). In native samples the inversion is
        # 1 - value per channel.
        # Photoshop inverts in the document's native space. For
        # Lab the RGB->Lab->RGB round trip is (nearly) lossless,
        # so inverting the Lab samples matches the corpus (0.0029
        # mean). For CMYK the round trip is lossy — four inks
        # collapse to three RGB channels and cannot be recovered —
        # so inverting the reconstructed inks is worse (0.64) than
        # inverting the RGB canvas (0.33); the native-ink invert
        # needs a native-space compositor, which this lane does
        # not build. CMYK stays RGB-space here.
        if mode == "Lab":
            # Lab samples are L 0..1, a/b 0..255 (128 neutral);
            # inverting the samples flips each one.
            lab = rgb_to_lab(rgb)
            inv = np.stack([1.0 - lab[..., 0],
                            255.0 - lab[..., 1],
                            255.0 - lab[..., 2]], axis=-1)
            return np.clip(lab_to_rgb_arr(inv), 0.0, 1.0)
        return 1.0 - rgb
    if kind == "threshold":
        level = adj["level"]
        t = round(level * 255.0)
        if depth == "u16":
            sums = np.round(rgb * 32768.0)
            total = 30.0 * sums[..., 0] + 59.0 * sums[..., 1] \
                + 11.0 * sums[..., 2]
            white = total >= 100.0 * round(t * 32768.0 / 255.0)
        elif is_float:
            white = (0.3 * rgb[..., 0] + 0.59 * rgb[..., 1]
                     + 0.11 * rgb[..., 2]) * 255.0 >= t
        else:
            sums = np.round(np.clip(rgb, 0.0, 1.0) * 255.0)
            total = 30.0 * sums[..., 0] + 59.0 * sums[..., 1] \
                + 11.0 * sums[..., 2]
            white = total > 100.0 * t - 50.0
        return np.where(white[..., None], 1.0, 0.0)
    if kind == "posterize":
        levels = max(2, min(255, adj["levels"]))
        n = float(levels)
        if depth == "u16":
            x = np.round(np.clip(rgb, 0.0, 1.0) * 32768.0)
            bins = np.minimum(np.floor(x * n / 32768.0), n - 1.0)
            step = np.floor(bins * 32768.0 / (n - 1.0))
            return np.floor((step * 255.0 + 16384.0) / 32768.0) \
                / 255.0
        x = np.round(np.clip(rgb, 0.0, 1.0) * 255.0)
        bins = np.minimum(np.floor(x * n / 256.0), n - 1.0)
        return np.floor(bins * 255.0 / (n - 1.0)) / 255.0
    if kind == "brightness_contrast":
        brightness = adj["brightness"]
        contrast = adj["contrast"]
        if adj.get("legacy"):
            if is_float:
                b = brightness / 255.0
                c = min(99.0, max(-100.0, contrast))
                k = 1.0 / (1.0 - c / 100.0) if c >= 0.0 \
                    else 1.0 + c / 100.0
                pre, post = (b, 0.0) if c >= 0.0 else (0.0, b)
                pivot = 127.0 / 255.0
                return (rgb + pre - pivot) * k + pivot + post
            if mode == "CMYK":
                # Legacy Brightness/Contrast in a CMYK document
                # runs over the inks, not the RGB canvas (5x
                # closer on the corpus's legacy file).
                ink = rgb_to_cmyk_arr(rgb)
                return np.clip(cmyk_to_rgb_arr(
                    _legacy_bc(ink, brightness, contrast)),
                    0.0, 1.0)
            return _legacy_bc(rgb, brightness, contrast)
        out = _modern_brightness(rgb, brightness)
        out = _modern_contrast(out, contrast)
        return np.clip(out, 0.0, 1.0)
    if kind == "levels":
        # Levels and Curves in a CMYK or Lab document run in the
        # document's native space, as a delta on the RGB canvas:
        # convert to native, apply each channel's record, convert
        # back, and shift the canvas by the difference (the Rust's
        # tone_in_space). Applying the records to RGB directly is
        # up to 0.06 off on these files.
        if mode in ("CMYK", "Lab"):
            return _tone_in_space(rgb, adj, "levels", mode,
                                  quantum, is_float)
        out = np.empty_like(rgb)
        # A grey document has one channel: Photoshop runs the
        # grey record (per_channel[0], then master) over the grey
        # value, so all three replicated channels take it.
        grey = mode in ("L", "Grayscale", "Gray")
        for i in range(3):
            channel = adj["per_channel"][0] if grey \
                else adj["per_channel"][i]
            identity = channel == _levels_identity()
            value = rgb[..., i] if identity else \
                _levels_channel(channel, rgb[..., i], quantum,
                                is_float)
            out[..., i] = value
        master = adj["master"]
        if master != _levels_identity():
            for i in range(3):
                out[..., i] = _levels_channel(
                    master, out[..., i], quantum, is_float)
        return out
    if kind == "curves":
        if mode in ("CMYK", "Lab"):
            return _tone_in_space(rgb, adj, "curves", mode,
                                  quantum, is_float)
        out = np.empty_like(rgb)
        grey = mode in ("L", "Grayscale", "Gray")
        for i in range(3):
            channel = adj["per_channel"][0] if grey \
                else adj["per_channel"][i]
            value = _curve_eval(channel, rgb[..., i],
                                adj["pencil"])
            out[..., i] = value
        master = adj["master"]
        if len(master) >= 2 and master != [(0.0, 0.0), (1.0, 1.0)]:
            for i in range(3):
                out[..., i] = _curve_eval(master, out[..., i],
                                          adj["pencil"])
        return out
    if kind == "hue_saturation":
        hue = adj["hue"]
        sat = adj["saturation"] / 100.0
        light = adj["lightness"] / 100.0
        colorize = adj["colorize"]
        ranges = adj["ranges"]
        has_ranges = (not colorize and any(
            r["hue"] != 0.0 or r["saturation"] != 0.0
            or r["lightness"] != 0.0 for r in ranges))
        out = rgb
        if has_ranges:
            h0, _, _ = _rgb_to_hsl(out)
            degrees = h0 * 360.0
            dh = np.zeros_like(degrees)
            ds = np.zeros_like(degrees)
            dl = np.zeros_like(degrees)
            for r in ranges:
                if r["hue"] == 0.0 and r["saturation"] == 0.0 \
                        and r["lightness"] == 0.0:
                    continue
                w = _hue_weight(r["bounds"], degrees)
                dh = dh + w * r["hue"]
                ds = ds + w * r["saturation"] / 100.0
                dl = dl + w * r["lightness"] / 100.0
            sat_eff = min(1.0, max(-1.0, sat + ds))
            out = _hue_saturation(out, hue + dh, sat_eff, light,
                                  colorize, is_float)
            out = _range_lightness(out, dl)
            return out
        sat_eff = sat
        if colorize and is_float:
            sat_eff = min(1.0, max(-1.0, sat)) \
                * (1.0 - min(1.0, abs(light)))
        return _hue_saturation(out, hue, sat_eff, light, colorize,
                               is_float)
    if kind == "vibrance":
        v = adj["vibrance"] / 100.0
        s = adj["saturation"] / 100.0
        hi = float("inf") if is_float else 1.0
        lin = srgb_to_linear(rgb)
        if v != 0.0:
            lin = _vibrance_lin(lin, v)
        if s != 0.0:
            grey = 0.288 * lin[..., 0] + 0.712 * lin[..., 1]
            lin = np.clip(grey[..., None]
                          + (1.0 + s) * (lin - grey[..., None]),
                          0.0, hi)
        return linear_to_srgb(np.clip(lin, 0.0, hi))
    if kind == "channel_mixer":
        matrix = np.array(adj["matrix"], dtype=np.float64)
        mono = adj["monochrome"]
        if is_float:
            def cut(v):
                return np.trunc(v * 1024.0) / 1024.0
            m = np.array([[cut(matrix[r][0]), cut(matrix[r][1]),
                           cut(matrix[r][2]),
                           matrix[r][3] * 255.0 / 512.0]
                          for r in range(3)])
            mix = (rgb @ m[:, :3].T) + m[:, 3]
            return np.stack([mix[..., 0]] * 3, axis=-1) \
                if mono else mix
        mix = (rgb @ matrix[:, :3].T) + matrix[:, 3]
        mix = np.clip(mix, 0.0, 1.0)
        return np.stack([mix[..., 0]] * 3, axis=-1) \
            if mono else mix
    if kind == "photo_filter":
        matrix = _photo_filter_matrix(adj["color"], adj["density"])
        if is_float:
            out = rgb @ matrix.T
            return out
        lin = srgb_to_linear(rgb)
        out = linear_to_srgb(lin @ matrix.T)
        if adj["preserve_luminosity"]:
            # Photoshop clips the matrix output into gamut BEFORE
            # SetLum: SetLum on the unclipped colour (the Rust's
            # model) pulls out-of-gamut colours far off — a saturated
            # red filters to G = -0.25, and ClipColor then drags B
            # from 0.007 to 0.22. Clipping first matches the corpus
            # to 0.0013 mean / 0.0176 max with no pixel over 0.05.
            out = set_lum(np.clip(out, 0.0, 1.0), lum(rgb))
        return np.clip(out, 0.0, 1.0)
    if kind == "black_white":
        hi = 16.0 if is_float else 1.0
        grey = np.clip(_black_white_gray(rgb, adj["weights"]),
                       0.0, hi)
        tint = adj["tint"]
        if tint is None:
            return np.stack([grey] * 3, axis=-1)
        if is_float:
            from .blend import blend_rgb as _blend_rgb
            return _blend_rgb("color",
                              np.stack([grey] * 3, axis=-1),
                              np.broadcast_to(
                                  np.array(tint, dtype=np.float64),
                                  rgb.shape))
        return np.clip(set_lum(np.broadcast_to(
            np.array(tint, dtype=np.float64), rgb.shape), grey),
            0.0, 1.0)
    if kind == "gradient_map":
        opaque = alpha > 0.0
        t = lum(rgb)
        if adj["reverse"]:
            t = 1.0 - t
        stops = adj["stops"]
        locs = np.array([s[0] for s in stops], dtype=np.float64)
        cols = np.array([s[1] for s in stops], dtype=np.float64)
        out = np.empty_like(rgb)
        for i in range(3):
            out[..., i] = np.interp(t, locs, cols[:, i])
        if adj["dither"]:
            height, width = rgb.shape[:2]
            xs, ys = np.meshgrid(np.arange(width), np.arange(height))
            dither = _bayer4(xs + x0, ys + y0) / 255.0
            out = np.clip(out + dither[..., None], 0.0, 1.0)
        return np.where(opaque[..., None], out, rgb)
    if kind == "color_balance":
        tones = [adj["shadows"], adj["midtones"], adj["highlights"]]
        preserve = adj["preserve_luminosity"]
        out = np.empty_like(rgb)
        for i in range(3):
            curve = _balance_curve(tones, preserve, i)
            out[..., i] = _balance_eval(curve, rgb[..., i])
        return out
    if kind == "selective_color":
        weights = _selective_weights(rgb)
        adj_arr = np.array(adj["adjustments"], dtype=np.float64)
        relative = adj["relative"]
        # Per range, per channel: the shift scales by the ink
        # already present (relative) or by the room the ink has
        # (absolute), weighted by the colour's range membership.
        delta = np.zeros_like(rgb)
        for r in range(9):
            w = weights[..., r]
            if not np.any(w > 0.0):
                continue
            a = adj_arr[r] / 100.0
            for i in range(3):
                ink = 1.0 - rgb[..., i]
                d = a[i] + a[3]
                if relative:
                    d = d * ink
                else:
                    d = np.where(d > 0.0,
                                 d * np.minimum(2.0 * (1.0 - ink),
                                                1.0),
                                 d * np.minimum(2.0 * ink, 1.0))
                delta[..., i] = delta[..., i] + d * w
        return np.clip(rgb - delta, 0.0, 1.0)
    return rgb


def _hue_saturation(rgb: np.ndarray, hue: np.ndarray | float,
                    sat: np.ndarray | float, light: float,
                    colorize: bool, is_float: bool) -> np.ndarray:
    """One Hue/Saturation evaluation (master): lightness first,
    then hue rotation and saturation scaling; colorize sets the
    hue and saturation instead of shifting them."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lighten = (lambda v: v + (1.0 - v) * light) if light > 0.0 \
        else (lambda v: v * (1.0 + light))
    if colorize:
        _, _, ll = _rgb_to_hsl(rgb)
        sat_value = min(1.0, abs(float(sat))) \
            if np.ndim(sat) == 0 else np.minimum(1.0, np.abs(sat))
        return _hsl_to_rgb(np.mod(hue, 360.0) / 360.0,
                           sat_value, lighten(ll))
    lit = lighten(rgb)
    hh, ss, ll = _rgb_to_hsl(lit)
    hh = np.mod(hh + np.asarray(hue, dtype=np.float64) / 360.0,
                1.0)
    if is_float:
        d = (lit.max(axis=-1) - lit.min(axis=-1))
        if np.ndim(sat) == 0:
            factor = _hdr_sat_factor(float(sat))
        else:
            factor = np.vectorize(_hdr_sat_factor)(sat)
        d = d * factor
        return _hue_rgb(hh, ll - d / 2.0, ll + d / 2.0)
    if np.ndim(sat) == 0:
        ss = ss / max(1.0 - float(sat), 1.0 / 512.0) \
            if sat > 0 else ss * (1.0 + float(sat))
    else:
        ss = np.where(sat > 0, ss / np.maximum(1.0 - sat,
                                               1.0 / 512.0),
                      ss * (1.0 + sat))
    ss = np.clip(ss, 0.0, 1.0)
    return _hsl_to_rgb(hh, ss, ll)


def _hdr_sat_factor(s: float) -> float:
    s = min(1.0, max(-1.0, s))
    if s <= 0.0:
        return 1.0 - int((-s * 256.0 + 1e-4) // 1) / 256.0
    steps = int((s * 254.0 + 1e-4) // 1)
    return min(65280 // max(255 - steps, 1), 32767) / 256.0


def _range_lightness(rgb: np.ndarray, light: np.ndarray
                     ) -> np.ndarray:
    """A range's Lightness: toward the pixel's brightest channel
    when raised, its darkest when lowered."""
    rgb = np.asarray(rgb, dtype=np.float64)
    light = np.asarray(light, dtype=np.float64)
    target = np.where(light > 0.0, rgb.max(axis=-1),
                      rgb.min(axis=-1))
    return rgb + (target[..., None] - rgb) * np.abs(light)[..., None]
