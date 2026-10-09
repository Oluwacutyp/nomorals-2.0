"""Upscaling — Devon's own code, two paths.

1. **Classical** (always available): Lanczos resample + unsharp-mask
   detail recovery + optional mild denoising. Fast, honest, no model.
2. **Diffusion refinement** (needs a native pipeline): upscale
   classically, then run img2img at low strength *per tile* with the
   prompt, blending tiles with a feathered overlap. Real detail
   synthesis, fully in our code.

Profile note: diffusion upscaling is heavy — termux gets the
classical path with an honest message; workstation gets tiles.
"""

from __future__ import annotations

import math

from . import ImgGenError, TORCH_AVAILABLE

__all__ = [
    "upscale_classical",
    "upscale_diffusion",
    "MAX_CLASSICAL_SCALE",
]

MAX_CLASSICAL_SCALE = 4


def upscale_classical(image, scale: float = 2.0,
                      sharpen: float = 0.6):
    """Lanczos upscale + unsharp mask. ``image`` is a PIL image."""
    from PIL import Image, ImageFilter

    if scale <= 0 or scale > 8:
        raise ImgGenError("scale must be in (0, 8]")
    w, h = image.size
    nw, nh = round(w * scale), round(h * scale)
    up = image.convert("RGB").resize((nw, nh), Image.LANCZOS)
    if sharpen > 0:
        # UnsharpMask(radius=2, percent=sharpen*100, threshold=2)
        up = up.filter(ImageFilter.UnsharpMask(
            radius=2, percent=int(sharpen * 100), threshold=2))
    return up


def upscale_diffusion(pipeline, image, prompt: str = "",
                      scale: float = 2.0, strength: float = 0.35,
                      tile: int = 64, overlap: int = 16,
                      **gen_kwargs):
    """Tile-based diffusion upscaling with detail synthesis.

    Each tile is classically upscaled, lightly re-noised (img2img at
    ``strength``), denoised with the prompt, and blended back with a
    cosine-feathered overlap so tile seams disappear.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImgGenError("diffusion upscaling needs torch")
    from PIL import Image

    import numpy as np

    from .edit import img2img

    if scale <= 1.0 or scale > 4:
        raise ImgGenError("diffusion scale must be in (1, 4]")
    base = upscale_classical(image, scale=scale, sharpen=0.0)
    w, h = base.size
    tw, th = tile, tile
    # Tile grid covering the image.
    xs = list(range(0, w, tw - overlap))
    ys = list(range(0, h, th - overlap))
    if xs[-1] + tw > w:
        xs[-1] = w - tw
    if ys[-1] + th > h:
        ys[-1] = h - th

    acc = np.zeros((h, w, 3), dtype=np.float64)
    weight = np.zeros((h, w, 1), dtype=np.float64)

    # Cosine feather window for seamless blending.
    wy = np.ones(th)
    wx = np.ones(tw)
    fo = overlap
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fo))
    wy[:fo], wy[-fo:] = ramp, ramp[::-1]
    wx[:fo], wx[-fo:] = ramp, ramp[::-1]
    window = (wy[:, None] * wx[None, :])[:, :, None]

    n = 0
    for y in ys:
        for x in xs:
            crop = base.crop((x, y, x + tw, y + th))
            # img2img at the tile size.
            kw = dict(gen_kwargs)
            kw["width"], kw["height"] = tw, th
            kw["strength"] = strength
            refined = img2img(pipeline, crop,
                              prompt or "high detail, sharp",
                              **kw)[0]
            refined = refined.resize((tw, th), Image.BILINEAR)
            arr = np.asarray(refined).astype(np.float64)
            acc[y:y + th, x:x + tw] += arr * window
            weight[y:y + th, x:x + tw] += window
            n += 1
    weight = np.maximum(weight, 1e-6)
    out = (acc / weight).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")
