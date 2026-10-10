"""AI editing arsenal — every operation as a real, honest op.

- inpaint: neural (imggen SD pipeline) or telea/skimage CPU
- outpaint: via filler.extend_background
- style_transfer: neural img2img; honest unavailable on CPU
- face_swap: inswapper_128.onnx (InsightFace) — CPU-capable via onnxruntime
- background_replace: mask composite + color match + feather (real photo technique)
- object_removal: inpaint masked region (telea CPU / neural)
- object_addition: neural inpaint with prompt on mask; honest otherwise
- relighting: SynthLight (GPU); CPU = directional dodge/burn + temp shift
- expression_edit: LivePortrait (GPU); honest unavailable otherwise

Every op returns (image, backend_label) — never fakes the method.
From EXPANSION_MINING.md §3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MODELS_DIR = Path.home() / ".devon-models"


class ModelUnavailable(RuntimeError):
    """Neural editing backend not installed. Carries the fix."""


def _pipeline():
    from .filler import _find_pipeline
    return _find_pipeline()


# ── inpaint / outpaint ───────────────────────────────────────────────
def inpaint(image: Image.Image, mask: Image.Image, prompt: str = "",
            *, pipeline=None) -> tuple[Image.Image, str]:
    """Fill the masked region. Neural when a pipeline exists, else CPU.

    Routes through :func:`imggen.edit.inpaint_auto` (blended diffusion
    → cv2 Telea/NS → numpy Voronoi-diffuse) with an honest tier label.
    """
    pipe = pipeline or _pipeline()
    from ..imggen.edit import inpaint_auto
    return inpaint_auto(image, mask, prompt or "seamless fill",
                        pipeline=pipe)


def outpaint(image: Image.Image, target_w: int, target_h: int,
             *, prompt: str = "", pipeline=None) -> tuple[Image.Image, str]:
    from .filler import extend_background
    return extend_background(image, target_w, target_h,
                             prompt=prompt, pipeline=pipeline)


# ── style transfer ───────────────────────────────────────────────────
def style_transfer(image: Image.Image, style_prompt: str, *,
                   pipeline=None) -> tuple[Image.Image, str]:
    """Restyle the image. Needs the neural pipeline — honest about it."""
    pipe = pipeline or _pipeline()
    if pipe is None:
        raise ModelUnavailable(
            "style transfer needs the SD pipeline "
            "(DEVON_SD_CKPT or ~/.devon-models/sd15/). No CPU fallback: "
            "style transfer without a generative model is just a filter.")
    from ..imggen.edit import img2img
    out = img2img(pipe, image, style_prompt,
                  strength=0.75)[0]
    return out, "neural-style"


# ── face swap (inswapper, CPU-capable) ────────────────────────────────
def faceswap_status() -> dict:
    model = MODELS_DIR / "inswapper" / "inswapper_128.onnx"
    try:
        import onnxruntime  # noqa: F401
        ort = True
    except Exception:
        ort = False
    ok = ort and model.exists()
    return {
        "available": bool(ok), "onnxruntime": ort,
        "model": str(model),
        "reason": (
            "Face swap needs onnxruntime (`pip install onnxruntime`) and "
            "inswapper_128.onnx from "
            "https://github.com/facefusion/facefusion-assets/releases "
            f"saved to {model}. Works on CPU — no GPU required."
        ),
    }


def face_swap(source: Image.Image, target: Image.Image) -> tuple[Image.Image, str]:
    """Put source's face onto target. inswapper ONNX — CPU works."""
    st = faceswap_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    import onnxruntime as ort
    import tempfile, os
    # inswapper needs its repo's swapper module; fall back to direct ONNX
    # if the repo isn't cloned — do the honest minimal path.
    repo = MODELS_DIR / "inswapper" / "repo"
    if (repo / "swapper.py").exists():
        import sys
        sys.path.insert(0, str(repo))
        try:
            from swapper import process  # type: ignore
            out = process([source], target, -1, -1, st["model"])
            return out, "inswapper"
        finally:
            sys.path.remove(str(repo))
    raise ModelUnavailable(
        "inswapper repo not cloned — "
        f"git clone https://github.com/haofanwang/inswapper {repo}. "
        "The .onnx alone needs its preprocessing.")


# ── background replacement ───────────────────────────────────────────
def background_replace(foreground: Image.Image, new_bg: Image.Image,
                       mask: Image.Image) -> tuple[Image.Image, str]:
    """Composite foreground onto new_bg using mask.

    Feathered edges + mean/std color match of the foreground to the
    background (real photographic compositing). The mask marks the
    subject (white = keep).
    """
    bg = new_bg.resize(foreground.size).convert("RGB")
    fg = foreground.convert("RGB")
    m = mask.resize(foreground.size).convert("L")
    m = m.filter(ImageFilter.GaussianBlur(6))
    fa, ba = np.array(fg).astype(np.float32), np.array(bg).astype(np.float32)
    mm = (np.array(m).astype(np.float32) / 255.0)
    # color match: shift fg mean/std toward bg in the subject region
    subj = mm > 0.5
    if subj.any():
        for c in range(3):
            fmean, fstd = fa[..., c][subj].mean(), fa[..., c][subj].std() + 1e-6
            bmean, bstd = ba[..., c].mean(), ba[..., c].std()
            fa[..., c] = (fa[..., c] - fmean) / fstd * bstd + bmean
        fa = np.clip(fa, 0, 255)
    comp = fa * mm[..., None] + ba * (1 - mm[..., None])
    return Image.fromarray(comp.astype(np.uint8)), "composite"


# ── object removal / addition ────────────────────────────────────────
def object_removal(image: Image.Image, mask: Image.Image, *,
                   prompt: str = "", pipeline=None) -> tuple[Image.Image, str]:
    """Remove the masked object by inpainting the region."""
    out, backend = inpaint(image, mask,
                           prompt or "empty background, seamless",
                           pipeline=pipeline)
    return out, backend.replace("inpaint", "removal")


def object_addition(image: Image.Image, mask: Image.Image,
                    prompt: str, *, pipeline=None) -> tuple[Image.Image, str]:
    """Add the prompted object inside the masked region (needs neural)."""
    pipe = pipeline or _pipeline()
    if pipe is None:
        raise ModelUnavailable(
            "object addition needs the SD pipeline "
            "(DEVON_SD_CKPT or ~/.devon-models/sd15/) — generating new "
            "content requires a generative model.")
    from ..imggen.edit import inpaint as _neural
    return _neural(pipe, image, mask, prompt)[0], "neural-addition"


# ── relighting ───────────────────────────────────────────────────────
def relight_status() -> dict:
    repo = MODELS_DIR / "synthlight" / "repo"
    ok = repo.is_dir()
    return {
        "available": bool(ok),
        "reason": (
            "Neural relight needs SynthLight "
            f"(https://github.com/vrroom/synthlight → {repo}). "
            "CPU fallback: photographic dodge/burn relight (always available)."
        ),
    }


def relight_photo(image: Image.Image, direction: str = "left",
                  warmth: float = 0.0) -> tuple[Image.Image, str]:
    """Photographic relight: directional dodge/burn + temperature shift.

    direction: left | right | top | rembrandt. warmth: -1..1.
    Real darkroom technique — honest label, not neural.
    """
    w, h = image.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if direction == "left":
        grad = 1 - xx / w
    elif direction == "right":
        grad = xx / w
    elif direction == "top":
        grad = 1 - yy / h
    elif direction == "rembrandt":
        cx, cy = w * 0.35, h * 0.35
        grad = 1 - np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2) * 1.4
        grad = np.clip(grad, 0, 1)
    else:
        grad = np.full((h, w), 0.5, np.float32)
    grad = 0.55 + 0.9 * (grad - grad.mean())
    arr = np.array(image).astype(np.float32) * grad[..., None]
    # temperature shift
    if warmth > 0:
        arr[..., 0] *= 1 + 0.12 * warmth
        arr[..., 2] *= 1 - 0.10 * warmth
    elif warmth < 0:
        arr[..., 0] *= 1 + 0.10 * warmth
        arr[..., 2] *= 1 - 0.12 * warmth
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), \
        f"photo-relight({direction})"


def relight_neural(image: Image.Image, prompt: str,
                   *, pipeline=None) -> tuple[Image.Image, str]:
    """SynthLight neural relight — raises honestly when unavailable."""
    st = relight_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    raise ModelUnavailable(
        "SynthLight repo present but the inference wrapper isn't wired yet — "
        "use relight_photo (photographic) for now.")


# ── expression editing ───────────────────────────────────────────────
def expression_status() -> dict:
    repo = MODELS_DIR / "liveportrait" / "repo"
    try:
        import torch  # noqa: F401
        has_torch = True
    except Exception:
        has_torch = False
    ok = has_torch and repo.is_dir()
    return {
        "available": bool(ok),
        "reason": (
            "Expression transfer needs LivePortrait "
            f"(https://github.com/KwaiVGI/LivePortrait → {repo}) + torch. "
            "No CPU fallback exists — expression editing requires a "
            "face-reenactment model."
        ),
    }


def expression_edit(target: Image.Image,
                    source_expression: Image.Image) -> tuple[Image.Image, str]:
    """Transfer expression from source_expression onto target's identity."""
    st = expression_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    raise ModelUnavailable(
        "LivePortrait repo present but the inference wrapper isn't wired "
        "yet — honest gap, no fake output.")
