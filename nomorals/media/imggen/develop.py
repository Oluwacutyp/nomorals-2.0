"""The develop stage — LightCraft-style non-destructive develop
controls as pure, deterministic tone math (Echelon Wave 2 · plan
§2.1b: develop settings live in the ImageDocument's op list as a
*stage*, not in a separate sidecar file).

Every control is a slider in ``[-1, 1]`` (``exposure`` is in EV
stops, ``[-2, 2]``); all-zero is the identity. The math is numpy
float tone curves — the same document develops to the same pixels
on every machine and every replay, which is what lets the image
document promise byte-identical renders. No neural rung, no hidden
state: the rung label is always ``develop (numpy, deterministic)``.

Control order is fixed and part of the contract (a develop stage
means the same thing wherever it replays):

1. exposure (EV) → 2. temperature/tint → 3. brightness/contrast →
4. shadows/highlights (luminance-weighted) → 5. saturation/vibrance
→ 6. sharpness.
"""

from __future__ import annotations

from . import ImgGenError
from .edit import NUMPY_AVAILABLE, PIL_AVAILABLE

__all__ = [
    "DEVELOP_FIELDS",
    "develop_image",
    "develop_defaults",
]

#: The develop vocabulary, in application order.
DEVELOP_FIELDS = (
    "exposure", "temperature", "tint",
    "brightness", "contrast",
    "shadows", "highlights",
    "saturation", "vibrance",
    "sharpness",
)

_RANGES = {
    "exposure": (-2.0, 2.0),
    **{f: (-1.0, 1.0) for f in DEVELOP_FIELDS if f != "exposure"},
}


def develop_defaults() -> dict:
    """The identity develop settings (every slider at zero)."""
    return {f: 0.0 for f in DEVELOP_FIELDS}


def _validate(settings: dict) -> dict:
    out = develop_defaults()
    for k, v in settings.items():
        if k not in DEVELOP_FIELDS:
            raise ImgGenError(
                f"unknown develop control {k!r}; "
                f"use one of {list(DEVELOP_FIELDS)}")
        lo, hi = _RANGES[k]
        fv = float(v)
        if not lo <= fv <= hi:
            raise ImgGenError(
                f"develop {k} must be in [{lo}, {hi}], got {fv}")
        out[k] = fv
    return out


def develop_image(img, **settings):
    """Apply develop controls to a PIL image. Pure: returns a new
    image, never mutates the input. All-zero settings return a copy
    of the input (bit-identical pixels)."""
    if not PIL_AVAILABLE:
        raise ImgGenError(
            "develop needs Pillow (pip install Pillow)")
    if not NUMPY_AVAILABLE:
        raise ImgGenError(
            "develop needs numpy (pip install numpy)")
    s = _validate(settings)
    from PIL import Image

    if all(v == 0.0 for v in s.values()):
        return img.copy()

    import numpy as np

    x = np.asarray(img.convert("RGB")).astype(np.float64) / 255.0

    # 1. exposure — EV stops are a plain gain in linear-ish space.
    if s["exposure"]:
        x = x * (2.0 ** s["exposure"])

    # 2. white balance — temperature warms (R up, B down), tint
    # shifts the green↔magenta axis.
    if s["temperature"] or s["tint"]:
        t = s["temperature"]
        x[..., 0] *= 1.0 + 0.18 * t
        x[..., 2] *= 1.0 - 0.18 * t
        g = s["tint"]
        x[..., 1] *= 1.0 - 0.14 * g
        x[..., 0] *= 1.0 + 0.05 * g
        x[..., 2] *= 1.0 + 0.05 * g

    # 3. brightness / contrast around middle gray.
    if s["brightness"]:
        x = x + 0.22 * s["brightness"]
    if s["contrast"]:
        c = s["contrast"]
        factor = (1.0 + c) if c >= 0 else 1.0 / (1.0 - c)
        x = (x - 0.5) * factor + 0.5

    # 4. shadows / highlights — luminance-weighted so the lift
    # lands where the control's name says and leaves the opposite
    # end (and the hue) alone.
    if s["shadows"] or s["highlights"]:
        lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1]
               + 0.0722 * x[..., 2])
        if s["shadows"]:
            w = (1.0 - np.clip(lum, 0.0, 1.0)) ** 2
            x = x + (0.30 * s["shadows"] * w)[..., None]
        if s["highlights"]:
            lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1]
                   + 0.0722 * x[..., 2])
            w = np.clip(lum, 0.0, 1.0) ** 2
            x = x + (0.30 * s["highlights"] * w)[..., None]

    # 5. saturation / vibrance — vibrance scales by how unsaturated
    # a pixel already is, so skin-tone-ish muted colors move and
    # already-vivid colors stay put.
    if s["saturation"] or s["vibrance"]:
        lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1]
               + 0.0722 * x[..., 2])[..., None]
        if s["saturation"]:
            x = lum + (x - lum) * (1.0 + s["saturation"])
        if s["vibrance"]:
            sat = (x.max(axis=2, keepdims=True)
                   - x.min(axis=2, keepdims=True))
            gain = 1.0 + s["vibrance"] * (1.0 - np.clip(sat, 0.0, 1.0))
            x = lum + (x - lum) * gain

    out = Image.fromarray(
        np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")

    # 6. sharpness — deterministic unsharp via PIL's enhancer.
    if s["sharpness"]:
        from PIL import ImageEnhance

        out = ImageEnhance.Sharpness(out).enhance(
            1.0 + s["sharpness"])
    return out
