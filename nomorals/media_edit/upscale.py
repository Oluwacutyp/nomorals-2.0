"""Real super-resolution upscaling: Real-ESRGAN + SUPIR rescue mode.

``op_upscale`` in generate.py is Lanczos resampling — fine for enlarging
a generation, but this module is the real thing: Real-ESRGAN x4 with
tiling for large images, and SUPIR as the workstation-only rescue mode
for badly degraded inputs.

Honesty contract: when the heavy library is missing the error names the
pip install. ``allow_fallback=True`` is the ONLY way to get Lanczos
instead, and it is opt-in per call — never a silent downgrade.

Profile gating:
- ``upscale(..., model="supir")`` → workstation profile only.
- Real-ESRGAN runs on laptop+; on termux it raises a clear "needs a
  laptop/workstation" error (Real-ESRGAN wants real RAM/VRAM).
"""

from __future__ import annotations

import os
import sys
from typing import Any

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind
from . import _model_cache as mc

_log = get_logger(__name__)

#: model key → weights file + arch params
MODELS = {
    "RealESRGAN_x4plus": {
        "file": "RealESRGAN_x4plus.pth",
        "url": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.1.0/RealESRGAN_x4plus.pth"),
        "size_mb": 64.0,
        "scale": 4,
        "num_feat": 64, "num_block": 23, "num_grow_ch": 32,
        "note": "general x4 upscaler",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "url": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.2.2.3/RealESRGAN_x4plus_anime_6B.pth"),
        "size_mb": 18.0,
        "scale": 4,
        "num_feat": 64, "num_block": 6, "num_grow_ch": 32,
        "note": "anime/illustration x4",
    },
    "RealESRGAN_x2plus": {
        "file": "RealESRGAN_x2plus.pth",
        "url": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.2.1/RealESRGAN_x2plus.pth"),
        "size_mb": 64.0,
        "scale": 2,
        "num_feat": 64, "num_block": 23, "num_grow_ch": 32,
        "note": "general x2 upscaler",
    },
}

_UPSAMPLERS: dict[str, Any] = {}


class UpscaleError(Exception):
    """Super-resolution unavailable or failed — the real reason."""


def list_models() -> list[dict[str, Any]]:
    """Registered SR models with sizes and notes."""
    return [{"name": name, **info} for name, info in MODELS.items()]


def _require_realesrgan() -> Any:
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as exc:
        raise UpscaleError(
            "super-resolution needs realesrgan: "
            "pip install realesrgan basicsr "
            "(or pass allow_fallback=True for plain Lanczos)") from exc
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise UpscaleError(
            "super-resolution needs torch: pip install torch") from exc
    return RealESRGANer, RRDBNet


def _upsampler(model: str, tile: int) -> Any:
    key = f"{model}:tile{tile}"
    if key in _UPSAMPLERS:
        return _UPSAMPLERS[key]
    RealESRGANer, RRDBNet = _require_realesrgan()
    info = MODELS[model]
    import torch
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=info["num_feat"],
                  num_block=info["num_block"],
                  num_grow_ch=info["num_grow_ch"], scale=info["scale"])
    weights = mc.model_path(info["file"], url=info["url"],
                            size_mb=info["size_mb"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log.info("loading Real-ESRGAN %s on %s (tile=%d)", model, device, tile)
    up = RealESRGANer(scale=info["scale"], model_path=str(weights),
                      model=net, tile=tile, tile_pad=10, pre_pad=0,
                      half=(device == "cuda"))
    _UPSAMPLERS[key] = up
    return up


def _auto_tile(w: int, h: int, tile: int) -> int:
    """tile=0 → automatic: tile when the image is large, else whole-image.

    Real-ESRGAN tiles to bound VRAM; ~512px tiles are the sweet spot.
    Images under ~1 MP go through whole (tile=0 in realesrgan means no
    tiling) for speed.
    """
    if tile:
        return tile
    return 512 if w * h > 1024 * 1024 else 0


def _lanczos(img: Any, scale: float) -> Any:
    from .images import _require_pillow
    Image = _require_pillow()
    w, h = img.size
    return img.resize((max(1, int(round(w * scale))),
                       max(1, int(round(h * scale)))), Image.LANCZOS)


def upscale(img: Any, scale: float = 4.0, *,
            model: str = "RealESRGAN_x4plus", tile: int = 0,
            allow_fallback: bool = False) -> Any:
    """Super-resolve ``img`` with Real-ESRGAN (or SUPIR rescue mode).

    ``scale`` must match the model's native scale (4 for x4 models, 2
    for x2) — mismatch raises. ``tile=0`` picks tiling automatically.
    ``allow_fallback=True`` permits plain Lanczos when realesrgan is
    missing; the default (False) raises with the pip hint instead.
    ``model="supir"`` is the workstation-only rescue mode for heavily
    degraded inputs.
    """
    model = (model or "RealESRGAN_x4plus").strip()
    if model == "supir":
        return _supir(img, scale)
    if model not in MODELS:
        raise UpscaleError(
            f"unknown SR model {model!r}; use: {sorted(MODELS)} or 'supir'")
    info = MODELS[model]
    if scale != info["scale"]:
        raise UpscaleError(
            f"model {model!r} is x{info['scale']}, not x{scale:g} — "
            f"use scale={info['scale']} or a matching model")
    kind = get_profile_kind()
    if kind == "termux":
        raise UpscaleError(
            "Real-ESRGAN needs a laptop/workstation (RAM/VRAM); "
            "on termux use the plain 'upscale' op (Lanczos)")
    if not allow_fallback:
        _require_realesrgan()  # fail fast with the pip hint
    else:
        try:
            _require_realesrgan()
        except UpscaleError:
            _log.warning("realesrgan missing — Lanczos fallback "
                         "(allow_fallback=True)")
            return _lanczos(img, scale)
    import numpy as np
    up = _upsampler(model, _auto_tile(*img.size, tile))
    arr = np.asarray(img.convert("RGB"))
    try:
        out, _ = up.enhance(arr, outscale=info["scale"])
    except Exception as exc:  # noqa: BLE001 - wrap with context
        raise UpscaleError(f"Real-ESRGAN ({model}) failed: {exc}") from exc
    from .images import _require_pillow
    Image = _require_pillow()
    return Image.fromarray(out)


#: Asset type → (model, why). The upscaler choice is asset-dependent:
#: photos get the general GAN, anime/illustration gets the anime-tuned
#: net, and documents/diagrams/screenshots get x2 (fidelity over
#: invention — diffusion SR hallucinates text, so SUPIR is never routed
#: here; call ``upscale(model="supir")`` explicitly for rescue jobs).
ASSET_MODELS: dict[str, dict[str, str]] = {
    "photo": {"model": "RealESRGAN_x4plus",
              "why": "general x4 GAN — fast, geometry-preserving"},
    "anime": {"model": "RealESRGAN_x4plus_anime_6B",
              "why": "anime/illustration-tuned x4 (photo models smear "
                     "flat art)"},
    "document": {"model": "RealESRGAN_x2plus",
                 "why": "x2 fidelity for text/diagrams — never invent "
                        "detail in text"},
}


def upscale_auto(img: Any, *, kind: str = "photo",
                 allow_fallback: bool = False) -> tuple[Any, dict[str, Any]]:
    """Upscale with the right model for the asset type.

    ``kind``: "photo" | "anime" | "document". Returns (image, info) where
    info names the model chosen and why — so the choice is inspectable,
    not magic.
    """
    kind = (kind or "photo").strip().lower()
    if kind not in ASSET_MODELS:
        raise UpscaleError(
            f"unknown asset kind {kind!r}; use {sorted(ASSET_MODELS)}")
    pick = ASSET_MODELS[kind]
    info = MODELS[pick["model"]]
    out = upscale(img, scale=float(info["scale"]), model=pick["model"],
                  allow_fallback=allow_fallback)
    return out, {"kind": kind, "model": pick["model"],
                 "scale": info["scale"], "why": pick["why"]}


def _supir(img: Any, scale: float) -> Any:
    """SUPIR rescue mode — workstation profile only, drives the SUPIR
    repo's own CLI.

    SUPIR has no pip package and no stable Python API (Fanghua-Yu/SUPIR
    is a research repo: SDXL base + CLIP encoders + LLaVA 13B + the
    SUPIR-v0Q/v0F checkpoints from Google Drive, ~10 GB total). So this
    drives the repo's real ``test.py`` CLI via subprocess instead of
    guessing at internals. Set ``SUPIR_REPO`` to the cloned repo path.

    Missing anything → fail closed with the exact setup steps, never a
    fake restoration.
    """
    if get_profile_kind() != "workstation":
        raise UpscaleError(
            "SUPIR rescue mode needs the workstation profile "
            f"(current: {get_profile_kind()}); use RealESRGAN_x4plus instead")
    repo = (os.environ.get("SUPIR_REPO") or "").strip()
    test_py = Path(repo) / "test.py" if repo else None
    if not test_py or not test_py.is_file():
        raise UpscaleError(
            "SUPIR rescue mode needs the SUPIR repo checked out:\n"
            "  git clone https://github.com/Fanghua-Yu/SUPIR.git\n"
            "  # install requirements.txt, download the checkpoints\n"
            "  # (SDXL base, CLIP encoders, LLaVA, SUPIR-v0Q — see the\n"
            "  # repo README), then:\n"
            "  export SUPIR_REPO=/path/to/SUPIR\n"
            "Until then, RealESRGAN_x4plus is the working upscaler.")
    import shutil
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="supir_"))
    try:
        in_dir, out_dir = tmp / "in", tmp / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        src = in_dir / "input.png"
        img.convert("RGB").save(src)
        cmd = [sys.executable, str(test_py),
               "--img_dir", str(in_dir), "--save_dir", str(out_dir)]
        _log.info("SUPIR rescue: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(test_py.parent),
                              capture_output=True, text=True,
                              timeout=3600)
        if proc.returncode != 0:
            raise UpscaleError(
                f"SUPIR test.py failed (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-2000:]}")
        outs = sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.jpg"))
        if not outs:
            raise UpscaleError(
                "SUPIR ran but produced no output image — check the repo's "
                "checkpoint paths (options/SUPIR_v0.yaml, CKPT_PTH.py)")
        from .images import _require_pillow
        Image = _require_pillow()
        return Image.open(outs[0]).convert("RGB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def op_upscale_sr(img: Any, *, scale: float = 4.0,
                  model: str = "RealESRGAN_x4plus") -> Any:
    """Chain op: real super-resolution (Real-ESRGAN x4 default).

    Registered as ``upscale_sr``; the plain ``upscale`` op stays Lanczos.
    """
    return upscale(img, scale=scale, model=model)
