"""Raster effects — EffectCraft ``effects`` crate's surface, real impls.

EffectCraft ships 308 named effects; Devon's registry carries the
families that have a real, deterministic raster implementation here
(21 effects), each one a pure function ``PIL.Image -> PIL.Image``.
The upstream catalogue is named honestly in the survey — effects
with no real implementation are NOT listed, so a name in
``list_effects()`` always does what it says.

Every effect validates its parameters (unknown name or an
out-of-range value is an :class:`EffectError`, never a silent
clamp) and is deterministic: the same input + params give the same
pixels, every time.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

__all__ = [
    "EffectError",
    "apply_effect",
    "effect_params",
    "list_effects",
]


class EffectError(ValueError):
    """Unknown effect, unknown parameter, or an out-of-range value."""


# -- helpers ---------------------------------------------------------------

def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float64)


def _from(arr: np.ndarray, mode: str = "RGB") -> Image.Image:
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode=mode)


def _rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def _check(params: dict[str, Any], spec: dict[str, tuple[Any, Any, Any]]) -> dict[str, Any]:
    """Validate ``params`` against ``spec`` {name: (default, min, max)}."""
    unknown = set(params) - set(spec)
    if unknown:
        raise EffectError(f"unknown parameter(s): {sorted(unknown)}")
    out: dict[str, Any] = {}
    for name, (default, lo, hi) in spec.items():
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EffectError(f"parameter {name!r} must be a number, got {value!r}")
        value = float(value)
        if lo is not None and value < lo or hi is not None and value > hi:
            raise EffectError(
                f"parameter {name!r}={value} out of range [{lo}, {hi}]")
        out[name] = value
    return out


# -- individual effects ----------------------------------------------------

def _gaussian_blur(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"radius": (2.0, 0.0, 50.0)})
    return _rgb(img).filter(ImageFilter.GaussianBlur(v["radius"]))


def _brightness(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"amount": (1.2, 0.0, 3.0)})
    return ImageEnhance.Brightness(_rgb(img)).enhance(v["amount"])


def _contrast(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"amount": (1.2, 0.0, 3.0)})
    return ImageEnhance.Contrast(_rgb(img)).enhance(v["amount"])


def _saturation(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"amount": (1.2, 0.0, 3.0)})
    return ImageEnhance.Color(_rgb(img)).enhance(v["amount"])


def _hue_shift(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"degrees": (30.0, -180.0, 180.0)})
    hsv = _rgb(img).convert("HSV")
    h, s, val = hsv.split()
    shift = int(round(v["degrees"] / 360.0 * 255.0))
    h = h.point(lambda x: (x + shift) % 256)
    return Image.merge("HSV", (h, s, val)).convert("RGB")


def _invert(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    _check(p, {})
    return ImageOps.invert(_rgb(img))


def _grayscale(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    _check(p, {})
    return ImageOps.grayscale(_rgb(img)).convert("RGB")


def _sepia(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"amount": (1.0, 0.0, 1.0)})
    base = _arr(_rgb(img))
    sepia = np.stack([
        base[:, :, 0] * 0.393 + base[:, :, 1] * 0.769 + base[:, :, 2] * 0.189,
        base[:, :, 0] * 0.349 + base[:, :, 1] * 0.686 + base[:, :, 2] * 0.168,
        base[:, :, 0] * 0.272 + base[:, :, 1] * 0.534 + base[:, :, 2] * 0.131,
    ], axis=2)
    return _from(base * (1 - v["amount"]) + sepia * v["amount"])


def _posterize(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"bits": (2.0, 1.0, 8.0)})
    return ImageOps.posterize(_rgb(img), int(v["bits"]))


def _sharpen(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"amount": (1.5, 0.0, 3.0)})
    return _rgb(img).filter(ImageFilter.UnsharpMask(
        radius=2.0, percent=int(v["amount"] * 100), threshold=3))


def _vignette(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"strength": (0.6, 0.0, 1.0)})
    base = _rgb(img)
    h, w = base.size[1], base.size[0]
    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx - w / 2) / (w / 2)
    dy = (yy - h / 2) / (h / 2)
    dist = np.sqrt(dx * dx + dy * dy)
    mask = 1.0 - v["strength"] * np.clip(dist - 0.45, 0, 1) ** 2
    return _from(_arr(base) * mask[:, :, None])


def _drop_shadow(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {
        "offset_x": (8.0, -100.0, 100.0),
        "offset_y": (8.0, -100.0, 100.0),
        "blur": (6.0, 0.0, 50.0),
        "opacity": (0.6, 0.0, 1.0),
    })
    base = img.convert("RGBA")
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dark = Image.new("RGBA", base.size, (0, 0, 0,
                                         int(255 * v["opacity"])))
    shadow.paste(dark, (0, 0), base.split()[3])
    shadow = shadow.filter(ImageFilter.GaussianBlur(v["blur"]))
    canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
    canvas.paste(shadow, (int(v["offset_x"]), int(v["offset_y"])))
    canvas.alpha_composite(base)
    return canvas.convert("RGB")


def _glow(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"radius": (8.0, 0.0, 50.0), "strength": (0.5, 0.0, 1.0)})
    base = _arr(_rgb(img))
    blurred = np.asarray(
        _rgb(img).filter(ImageFilter.GaussianBlur(v["radius"])),
        dtype=np.float64)
    bright = np.clip(blurred - 110.0, 0, None) / 145.0
    screen = 255.0 - (255.0 - base) * (255.0 - bright * 255.0) / 255.0
    return _from(base * (1 - v["strength"]) + screen * v["strength"])


def _noise(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"amount": (0.1, 0.0, 1.0)})
    rng = np.random.default_rng(0)  # deterministic by design
    base = _arr(_rgb(img))
    sigma = v["amount"] * 40.0
    return _from(base + rng.normal(0.0, sigma, base.shape))


def _pixelate(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"size": (8.0, 2.0, 64.0)})
    base = _rgb(img)
    size = int(v["size"])
    small = base.resize((max(1, base.size[0] // size),
                         max(1, base.size[1] // size)),
                        Image.BILINEAR)
    return small.resize(base.size, Image.NEAREST)


def _edge_detect(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    _check(p, {})
    return _rgb(img).filter(ImageFilter.FIND_EDGES)


def _emboss(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    _check(p, {})
    return _rgb(img).filter(ImageFilter.EMBOSS)


def _color_balance(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"red": (0.0, -1.0, 1.0), "green": (0.0, -1.0, 1.0),
                   "blue": (0.0, -1.0, 1.0)})
    base = _arr(_rgb(img))
    gains = np.array([1.0 + v["red"], 1.0 + v["green"], 1.0 + v["blue"]])
    return _from(base * gains)


def _levels(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"black": (0.0, 0.0, 255.0), "white": (255.0, 0.0, 255.0)})
    if v["white"] <= v["black"]:
        raise EffectError("levels: white must be above black")
    base = _arr(_rgb(img))
    scaled = (base - v["black"]) * 255.0 / (v["white"] - v["black"])
    return _from(scaled)


def _chroma_key(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"threshold": (0.25, 0.0, 1.0),
                   "smoothness": (0.1, 0.0, 1.0)})
    color = p.get("color", [0, 200, 0])
    if (not isinstance(color, (list, tuple)) or len(color) != 3
            or any(not isinstance(c, (int, float)) or c < 0 or c > 255
                   for c in color)):
        raise EffectError("chroma_key: color must be [r, g, b] 0..255")
    base = _arr(img.convert("RGBA"))
    rgb = base[:, :, :3]
    key = np.array(color, dtype=np.float64)
    dist = np.sqrt(((rgb - key) ** 2).sum(axis=2)) / (255.0 * math.sqrt(3.0))
    lo = v["threshold"]
    hi = min(1.0, v["threshold"] + max(v["smoothness"], 1e-6))
    alpha = np.clip((dist - lo) / (hi - lo), 0.0, 1.0) * 255.0
    out = np.dstack([rgb, alpha])
    return Image.fromarray(out.astype(np.uint8), mode="RGBA")


def _unsharp_mask(img: Image.Image, p: dict[str, Any]) -> Image.Image:
    v = _check(p, {"radius": (2.0, 0.0, 20.0),
                   "percent": (150.0, 0.0, 500.0),
                   "threshold": (3.0, 0.0, 50.0)})
    return _rgb(img).filter(ImageFilter.UnsharpMask(
        radius=v["radius"], percent=int(v["percent"]),
        threshold=int(v["threshold"])))


_EFFECTS: dict[str, Callable[[Image.Image, dict[str, Any]], Image.Image]] = {
    "gaussian_blur": _gaussian_blur,
    "brightness": _brightness,
    "contrast": _contrast,
    "saturation": _saturation,
    "hue_shift": _hue_shift,
    "invert": _invert,
    "grayscale": _grayscale,
    "sepia": _sepia,
    "posterize": _posterize,
    "sharpen": _sharpen,
    "vignette": _vignette,
    "drop_shadow": _drop_shadow,
    "glow": _glow,
    "noise": _noise,
    "pixelate": _pixelate,
    "edge_detect": _edge_detect,
    "emboss": _emboss,
    "color_balance": _color_balance,
    "levels": _levels,
    "chroma_key": _chroma_key,
    "unsharp_mask": _unsharp_mask,
}

_PARAM_SPECS: dict[str, dict[str, tuple[Any, Any, Any]]] = {
    "gaussian_blur": {"radius": (2.0, 0.0, 50.0)},
    "brightness": {"amount": (1.2, 0.0, 3.0)},
    "contrast": {"amount": (1.2, 0.0, 3.0)},
    "saturation": {"amount": (1.2, 0.0, 3.0)},
    "hue_shift": {"degrees": (30.0, -180.0, 180.0)},
    "invert": {},
    "grayscale": {},
    "sepia": {"amount": (1.0, 0.0, 1.0)},
    "posterize": {"bits": (2.0, 1.0, 8.0)},
    "sharpen": {"amount": (1.5, 0.0, 3.0)},
    "vignette": {"strength": (0.6, 0.0, 1.0)},
    "drop_shadow": {"offset_x": (8.0, -100.0, 100.0),
                    "offset_y": (8.0, -100.0, 100.0),
                    "blur": (6.0, 0.0, 50.0),
                    "opacity": (0.6, 0.0, 1.0)},
    "glow": {"radius": (8.0, 0.0, 50.0), "strength": (0.5, 0.0, 1.0)},
    "noise": {"amount": (0.1, 0.0, 1.0)},
    "pixelate": {"size": (8.0, 2.0, 64.0)},
    "edge_detect": {},
    "emboss": {},
    "color_balance": {"red": (0.0, -1.0, 1.0), "green": (0.0, -1.0, 1.0),
                      "blue": (0.0, -1.0, 1.0)},
    "levels": {"black": (0.0, 0.0, 255.0), "white": (255.0, 0.0, 255.0)},
    "chroma_key": {"threshold": (0.25, 0.0, 1.0),
                   "smoothness": (0.1, 0.0, 1.0)},
    "unsharp_mask": {"radius": (2.0, 0.0, 20.0),
                     "percent": (150.0, 0.0, 500.0),
                     "threshold": (3.0, 0.0, 50.0)},
}


def list_effects() -> list[str]:
    """Names of every effect with a real implementation, sorted."""
    return sorted(_EFFECTS)


def effect_params(name: str) -> dict[str, dict[str, Any]]:
    """Parameter spec for ``name``: {param: {default, min, max}}."""
    if name not in _PARAM_SPECS:
        raise EffectError(f"unknown effect {name!r}")
    return {k: {"default": d, "min": lo, "max": hi}
            for k, (d, lo, hi) in _PARAM_SPECS[name].items()}


def apply_effect(name: str, img: Image.Image,
                 params: dict[str, Any] | None = None) -> Image.Image:
    """Apply effect ``name`` to ``img`` with validated ``params``."""
    if name not in _EFFECTS:
        raise EffectError(
            f"unknown effect {name!r}; known: {', '.join(list_effects())}")
    return _EFFECTS[name](img, dict(params or {}))
