"""Photoshop gradient appearance — dense-stop baking, the
parameter ``t`` of a gradient at a pixel, and sampling.

Photoshop interpolates gradient colours with a Smoothness
setting and (since 2023) an interpolation method (Smooth /
Perceptual / Linear / Classic). The compositors interpolate
stops linearly in sRGB, so on import the curve is sampled
into dense stops: Smoothness blends linear interpolation with
a Catmull-Rom spline through the stop colours (end tangents
at half the chord slope); Perceptual interpolates in Oklab,
Linear in linear light, Classic in sRGB. Stop midpoints
remap each segment piecewise-linearly. Smooth's progress is
the measured table from Photoshop 27.11 — the curve fits no
simple formula.

Clean-room reimplementation of the documented model
(photocraft ``gradient_bake.rs`` / ``effects::gradient_t``),
written from the public description of the behaviour.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["bake_stops", "bake_opacity_stops",
           "gradient_layout", "gradient_t_array",
           "render_gradient", "sample_stops_array"]

# Photoshop 27.11's measured Smooth progress across a segment
# at u = 0, 1/16, ..., 1.
_SMOOTH_EASE = (0.0, 0.0438, 0.099, 0.1644, 0.2303, 0.2988,
                0.3667, 0.433, 0.4974, 0.5642, 0.6302, 0.6985,
                0.7656, 0.8333, 0.8945, 0.9517, 1.0)

_OPACITY_RESPONSE = 0.65


def _lin(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)


def _enc(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _to_oklab(rgb: np.ndarray) -> np.ndarray:
    lin = _lin(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    lum_v = np.cbrt(0.41222146 * r + 0.53633255 * g
                    + 0.051445995 * b)
    m = np.cbrt(0.2119035 * r + 0.6806995 * g + 0.10739696 * b)
    s = np.cbrt(0.08830246 * r + 0.28171885 * g + 0.6299787 * b)
    return np.stack([
        0.21045426 * lum_v + 0.7936178 * m - 0.004072047 * s,
        1.9779985 * lum_v - 2.4285922 * m + 0.4505937 * s,
        0.025904037 * lum_v + 0.78277177 * m - 0.80867577 * s,
    ], axis=-1)


def _from_oklab(p: np.ndarray) -> np.ndarray:
    lum_v = (p[..., 0] + 0.39633778 * p[..., 1]
             + 0.21580376 * p[..., 2]) ** 3
    m = (p[..., 0] - 0.105561346 * p[..., 1]
         - 0.06385417 * p[..., 2]) ** 3
    s = (p[..., 0] - 0.08948418 * p[..., 1]
         - 1.2914855 * p[..., 2]) ** 3
    lin = np.stack([
        4.0767417 * lum_v - 3.3077116 * m + 0.23096994 * s,
        -1.268438 * lum_v + 2.6097574 * m - 0.34131938 * s,
        -0.0041960863 * lum_v - 0.7034186 * m + 1.7076147 * s,
    ], axis=-1)
    return _enc(lin)


def _into_space(method: str, rgb: np.ndarray) -> np.ndarray:
    if method == "linear":
        return _lin(rgb)
    if method in ("perceptual", "smooth"):
        return _to_oklab(rgb)
    return rgb


def _out_of_space(method: str, p: np.ndarray) -> np.ndarray:
    if method == "linear":
        return _enc(p)
    if method in ("perceptual", "smooth"):
        return _from_oklab(p)
    return np.clip(p, 0.0, 1.0)


def _smooth_ease(u: np.ndarray) -> np.ndarray:
    x = np.clip(u, 0.0, 1.0) * 16.0
    i = np.minimum(np.floor(x).astype(int), 15)
    a = np.array(_SMOOTH_EASE)[i]
    b = np.array(_SMOOTH_EASE)[i + 1]
    return a + (b - a) * (x - i)


def _eval_segment(pos: np.ndarray, cols: np.ndarray,
                  mids: list[float], smooth: float, t: np.ndarray,
                  method: str) -> np.ndarray:
    """Colour at ``t`` for sorted stops (positions in 0..1)."""
    n = len(pos)
    out = np.zeros(t.shape + (3,), dtype=np.float64)
    out[:] = cols[0]
    if n < 2:
        return out
    idx = np.searchsorted(pos, t, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    t0 = pos[idx]
    t1 = pos[idx + 1]
    h = np.maximum(t1 - t0, 1e-6)
    u = (t - t0) / h
    mid = np.array([mids[i] if i < len(mids) else 0.5
                    for i in idx])
    mid = np.clip(mid, 0.05, 0.95)
    u = np.where(u <= mid, 0.5 * u / mid,
                 0.5 + 0.5 * (u - mid) / (1.0 - mid))
    # Outside the stop range the colour is the nearest stop.
    u = np.clip(u, 0.0, 1.0)
    p0 = cols[idx]
    p1 = cols[idx + 1]
    if method == "smooth":
        e = _smooth_ease(u)[..., None]
        return p0 + (p1 - p0) * e

    def tangent(k: np.ndarray) -> np.ndarray:
        # Catmull-Rom tangents per unit t; ends use half the
        # chord slope.
        a = np.where(k == 0, 0, np.where(k == n - 1, n - 2, k - 1))
        bidx = np.where(k == 0, 1, np.where(k == n - 1, n - 1,
                                            k + 1))
        d = np.maximum(pos[bidx] - pos[a], 1e-6)
        half = np.where((k == 0) | (k == n - 1), 0.5, 1.0)
        return ((cols[bidx] - cols[a]) / d[..., None]
                * half[..., None])

    m0 = tangent(idx)
    m1 = tangent(idx + 1)
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    spline = (h00[..., None] * p0 + h10[..., None] * h[..., None]
              * m0 + h01[..., None] * p1
              + h11[..., None] * h[..., None] * m1)
    linear = p0 + (p1 - p0) * u[..., None]
    value = linear + (spline - linear) * smooth
    lo = np.minimum(p0, p1)
    hi = np.maximum(p0, p1)
    return np.clip(value, lo, hi)


def method_from_code(code: str | None) -> str:
    """The interpolation method for a four-character code."""
    return {"Perc": "perceptual", "Lnr ": "linear",
            "Smoo": "smooth"}.get(code or "", "classic")


def bake_stops(stops: list[tuple[float, tuple[float, float,
                                             float]]],
               midpoints: list[float], smoothness: float,
               method: str) -> list[tuple[float, tuple[float,
                                                      float,
                                                      float]]]:
    """Dense linear stops reproducing Photoshop's interpolation.
    Returns the input unchanged when it already interpolates
    linearly in sRGB."""
    if len(stops) < 2:
        return stops
    plain_mids = all(abs(m - 0.5) < 1e-3 for m in midpoints)
    if (smoothness <= 0.0 and method == "classic"
            and plain_mids):
        return stops
    ordered = sorted(stops, key=lambda s: s[0])
    pos = np.array([s[0] for s in ordered], dtype=np.float64)
    cols = np.array([s[1] for s in ordered], dtype=np.float64)
    space = _into_space(method, cols)
    dense_t = np.linspace(0.0, 1.0, 97)
    values = _eval_segment(pos, space, midpoints,
                           min(max(smoothness, 0.0), 1.0),
                           dense_t, method)
    rgb = _out_of_space(method, values)
    out: list[tuple[float, tuple[float, float, float]]] = []
    for i, t in enumerate(dense_t):
        if out and abs(float(t) - out[-1][0]) < 1e-6:
            continue
        out.append((float(t), (float(rgb[i][0]),
                               float(rgb[i][1]),
                               float(rgb[i][2]))))
    return out


def bake_opacity_stops(stops: list[tuple[float, float]],
                       midpoints: list[float], smoothness: float,
                       method: str) -> list[tuple[float, float]]:
    """Dense linear opacity stops (Photoshop's opacity response
    is ``y**0.65`` out of Linear / Perceptual space)."""
    if method == "smooth":
        method = "classic"
    plain_mids = all(abs(m - 0.5) < 1e-3 for m in midpoints)
    smoothness = min(max(smoothness, 0.0), 1.0)
    if (len(stops) < 2 or (smoothness <= 0.0
                           and method == "classic"
                           and plain_mids)):
        return stops
    ordered = sorted(stops, key=lambda s: s[0])
    pos = np.array([s[0] for s in ordered], dtype=np.float64)
    vals = np.array([s[1] for s in ordered], dtype=np.float64)

    def into(o: np.ndarray) -> np.ndarray:
        o = np.clip(o, 0.0, 1.0)
        if method == "linear":
            return _lin(o)
        if method == "perceptual":
            return np.cbrt(_lin(o))
        return o

    def out(v: np.ndarray) -> np.ndarray:
        if method == "linear":
            y = v
        elif method == "perceptual":
            y = np.maximum(v, 0.0) ** 3
        else:
            return np.clip(v, 0.0, 1.0)
        return np.clip(y, 0.0, 1.0) ** _OPACITY_RESPONSE

    space = into(vals)[..., None].repeat(3, axis=-1)
    dense_t = np.linspace(0.0, 1.0, 97)
    values = _eval_segment(pos, space, midpoints, smoothness,
                           dense_t, "classic")
    result = out(values[..., 0])
    baked: list[tuple[float, float]] = []
    for i, t in enumerate(dense_t):
        if baked and abs(float(t) - baked[-1][0]) < 1e-6:
            continue
        baked.append((float(t), float(result[i])))
    return baked


def sample_stops_array(stops: list[tuple[float, tuple[float,
                                                    float,
                                                    float]]],
                       t: np.ndarray) -> np.ndarray:
    """Linear sRGB sampling of dense (baked) colour stops."""
    if not stops:
        return np.zeros(t.shape + (3,), dtype=np.float64)
    if len(stops) == 1:
        return np.tile(np.array(stops[0][1]), t.shape + (1,))
    pos = np.array([s[0] for s in stops], dtype=np.float64)
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    out = np.zeros(t.shape + (3,), dtype=np.float64)
    for c in range(3):
        out[..., c] = np.interp(t, pos, cols[:, c])
    return out


def sample_opacity_array(stops: list[tuple[float, float]],
                         t: np.ndarray) -> np.ndarray:
    """Linear sampling of (baked) opacity stops."""
    if not stops:
        return np.ones_like(t)
    if len(stops) == 1:
        return np.full_like(t, stops[0][1])
    pos = np.array([s[0] for s in stops], dtype=np.float64)
    vals = np.array([s[1] for s in stops], dtype=np.float64)
    return np.interp(t, pos, vals)


def _gradient_units(angle: float, w: float, h: float
                    ) -> tuple[float, float]:
    """(chord, length) of a gradient at ``angle`` over ``w``x``h``:
    the chord (Linear / Reflected) and the ellipse-norm length
    (Radial, Angle, Diamond — its half is the radius)."""
    a = math.radians(angle)
    s, c = math.sin(a), math.cos(a)
    chord = min(w / max(abs(c), 1e-6), h / max(abs(s), 1e-6))
    chord = max(chord, 1.0)
    length = max(math.sqrt((c * w) ** 2 + (s * h) ** 2), 1.0)
    return chord, length


def gradient_layout(style: str, angle: float, scale: float,
                      offset: tuple[float, float],
                      frame: tuple[int, int, int, int]
                      ) -> tuple[float, float, tuple[float, float]]:
    """Effective ``(angle, scale, centre offset)`` for a
    gradient with ``style`` / ``angle`` / ``scale`` laid out in
    ``frame``: Photoshop lays Linear and Reflected gradients
    out between whole-pixel end points — the chord through the
    frame's centre along the angle ends at points truncated to
    the pixel grid. On large frames that is invisible, on
    small ones it changes the effective angle."""
    if (style not in ("linear", "reflected")
            or not math.isfinite(angle)
            or not math.isfinite(scale)
            or not math.isfinite(offset[0])
            or not math.isfinite(offset[1])):
        return (angle, scale, offset)
    x0, y0, x1, y1 = frame
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    def chord_of(a: float) -> float:
        s, c = math.sin(a), math.cos(a)
        return max(min(w / max(abs(c), 1e-6),
                       h / max(abs(s), 1e-6)), 1.0)

    cx = x0 + w / 2.0 + offset[0] * w
    cy = y0 + h / 2.0 + offset[1] * h
    a = math.radians(angle)
    scale64 = max(scale, 1e-3)
    half = chord_of(a) * scale64 / 2.0
    s, c = math.sin(a), math.cos(a)
    dx, dy = c * half, -s * half

    def snap(v):
        return math.floor(v + 1e-4)

    end = (snap(cx + dx), snap(cy + dy))
    if style == "reflected":
        start = (cx, cy)
        mid = (cx, cy)
    else:
        start = (snap(cx - dx), snap(cy - dy))
        mid = ((start[0] + end[0]) / 2.0,
               (start[1] + end[1]) / 2.0)
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(vx, vy)
    if length < 0.5:
        return (angle, scale, offset)
    a2 = math.atan2(-vy, vx)
    span = 2.0 * length if style == "reflected" else length
    scale2 = span / chord_of(a2)
    shift = ((mid[0] - cx) / w, (mid[1] - cy) / h)
    return (math.degrees(a2), scale2,
            (offset[0] + shift[0], offset[1] + shift[1]))


def gradient_t_array(style: str, angle: float, scale: float,
                     reverse: bool, offset: tuple[float, float],
                     bounds: tuple[int, int, int, int], width: int,
                     height: int, *,
                     origin: tuple[float, float] = (0.0, 0.0)
                     ) -> np.ndarray:
    """The gradient parameter ``t`` at every pixel of a
    ``width``x``height`` canvas, for a gradient laid out over
    ``bounds`` (x0, y0, x1, y1). Pixel ``(col, row)`` sits at
    document point ``origin + (col + 0.5, row + 0.5)``."""
    x0, y0, x1, y1 = bounds
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    xs = xs + 0.5 + origin[0]
    ys = ys + 0.5 + origin[1]
    cx = x0 + w / 2.0 + offset[0] * w
    cy = y0 + h / 2.0 + offset[1] * h
    a = math.radians(angle)
    s, c = math.sin(a), math.cos(a)
    dx = xs - cx
    dy = ys - cy
    along = dx * c - dy * s
    across = dx * s + dy * c
    chord, length = _gradient_units(angle, w, h)
    chord = chord * max(scale, 1e-3)
    length = length * max(scale, 1e-3)
    # Linear / Reflected sample the pixel's top-left corner,
    # half a pixel before its centre.
    corner = 0.5 * (c - s)
    if style == "linear":
        t = (along - corner) / chord + 0.5
    elif style == "reflected":
        t = np.abs((along - corner) / (chord / 2.0))
    elif style == "radial":
        t = np.sqrt(dx * dx + dy * dy) / (length / 2.0)
    elif style == "diamond":
        t = (np.abs(along) + np.abs(across)) / (length / 2.0)
    elif style == "angle":
        t = ((a - np.arctan2(-dy, dx)) / (2.0 * math.pi)) % 1.0
    else:
        t = (along - corner) / chord + 0.5
    t = np.clip(t, 0.0, 1.0)
    if reverse:
        t = 1.0 - t
    return t


def render_gradient(gradient: dict, width: int, height: int,
                    *, bounds: tuple[int, int, int, int] | None
                    = None) -> np.ndarray:
    """A gradient dict (``stops``, ``opacity_stops``, ``style``,
    ``angle``, ``scale``, ``reverse``, ``offset``) rendered as
    float RGBA over ``width``x``height``."""
    if bounds is None:
        bounds = (0, 0, width, height)
    angle, scale, offset = gradient_layout(
        gradient.get("style", "linear"),
        gradient.get("angle", 90.0), gradient.get("scale", 1.0),
        gradient.get("offset", (0.0, 0.0)), bounds)
    t = gradient_t_array(gradient.get("style", "linear"),
                         angle, scale,
                         gradient.get("reverse", False), offset,
                         bounds, width, height)
    rgb = sample_stops_array(gradient.get("stops", []), t)
    alpha = sample_opacity_array(
        gradient.get("opacity_stops", []), t)
    out = np.zeros((height, width, 4), dtype=np.float64)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out
