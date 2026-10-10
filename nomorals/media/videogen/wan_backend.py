"""Wan 2.2 backend — the quality-finals neural engine.

Target: **Wan 2.2 TI2V-5B** (``Wan-AI/Wan2.2-TI2V-5B-Diffusers``) —
720p @ 24fps, T2V + I2V in one checkpoint, Apache 2.0 (the cleanest
licence of any strong open video model). The tradeoff is speed:
~4–9 min per 5s clip on a 4090 — so the pipeline uses LTX for previz
and Wan for finals.

Same interface as :class:`ltx_backend.LTXBackend` so the pipeline and
chainer can swap backends without caring. Import-gated like LTX:
no diffusers / no CUDA ≥ 16GB → honest error, never a fake render.

(Verified 2026-10-09: diffusers ``WanPipeline`` /
``WanImageToVideoPipeline``; the 14B MoE variants need 48–80GB and are
deliberately NOT targeted.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capabilities import (
    WAN_MIN_VRAM_GB,
    VideogenError,
    diffusers_available,
    neural_capability,
    torch_cuda_available,
)
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["WanBackend", "WanClipRequest", "WAN_DEFAULTS"]

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

WAN_DEFAULTS: dict[str, Any] = {
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "fps": 24,
}


@dataclass
class WanClipRequest:
    prompt: str
    mode: str = "t2v"                    # t2v | i2v
    image: str | os.PathLike | None = None
    negative_prompt: str = ("worst quality, inconsistent motion, blurry, "
                            "jittery, distorted, watermark, logo")
    duration_s: float = 5.0
    width: int = 1280
    height: int = 720
    fps: int = 24
    seed: int = 0
    out: str | os.PathLike | None = None
    model_id: str = ""
    num_inference_steps: int = 0
    guidance_scale: float = 0.0


def _snap_frames(duration_s: float, fps: int) -> int:
    want = max(17, int(round(duration_s * fps)))
    return min(want, 161)  # TI2V-5B is happiest ≤ ~6.7s per clip


class WanBackend:
    """Wan 2.2 TI2V-5B via diffusers. Lazy-loads on first use."""

    name = "wan"
    min_vram_gb = WAN_MIN_VRAM_GB

    def __init__(self, *, model_id: str = ""):
        self.model_id = model_id or MODEL_ID
        self._pipe: Any = None
        self._i2v_pipe: Any = None

    def check(self) -> dict[str, Any]:
        cap = neural_capability(prefer="wan")
        ok = cap.available and cap.backend == "wan"
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "available": ok,
            "vram_gb": cap.vram_gb,
            "reason": cap.reason if not ok else
                      f"ready — {self.model_id} ({cap.reason})",
        }

    def require(self) -> None:
        info = self.check()
        if not info["available"]:
            raise VideogenError(
                f"Wan backend unavailable: {info['reason']}")

    def _load(self, for_i2v: bool = False) -> Any:
        if not diffusers_available():
            raise VideogenError(
                "the `diffusers` package is not installed — "
                "`pip install diffusers accelerate` then retry")
        if not torch_cuda_available():
            raise VideogenError("no CUDA GPU — Wan 2.2 needs CUDA ≥ 16GB VRAM")
        import torch
        cached = self._i2v_pipe if for_i2v else self._pipe
        if cached is not None:
            return cached
        from diffusers import WanImageToVideoPipeline, WanPipeline
        cls = WanImageToVideoPipeline if for_i2v else WanPipeline
        _log.info("loading Wan pipeline %s (i2v=%s) …", self.model_id, for_i2v)
        pipe = cls.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        # reference flags: offload model, keep the big umT5 encoder on CPU
        # when VRAM is tight (mirrors --offload_model --t5_cpu)
        try:
            pipe.enable_model_cpu_offload()
        except Exception:  # noqa: BLE001
            pipe.to("cuda")
        try:
            pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001
            pass
        if for_i2v:
            self._i2v_pipe = pipe
        else:
            self._pipe = pipe
        return pipe

    def generate(self, prompt: str, **kw) -> str:
        """Generate a 720p24 clip. Returns the output mp4 path."""
        req = WanClipRequest(prompt=prompt, **kw) if not isinstance(
            prompt, WanClipRequest) else prompt
        self.require()
        if not req.prompt.strip():
            raise VideogenError("empty prompt — nothing to generate")
        mode = req.mode.lower()
        if mode not in ("t2v", "i2v"):
            raise VideogenError(f"unknown Wan mode {mode!r} — t2v | i2v")
        if mode == "i2v":
            if not req.image:
                raise VideogenError("Wan i2v needs an `image` input")
            if not Path(req.image).exists():
                raise VideogenError(f"input image not found: {req.image}")

        import torch
        from diffusers.utils import export_to_video, load_image

        steps = req.num_inference_steps or WAN_DEFAULTS["num_inference_steps"]
        guidance = (req.guidance_scale if req.guidance_scale > 0
                    else WAN_DEFAULTS["guidance_scale"])
        num_frames = _snap_frames(req.duration_s, req.fps)
        generator = torch.Generator(device="cuda").manual_seed(req.seed)

        call_kw: dict[str, Any] = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=req.width, height=req.height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if mode == "i2v":
            pipe = self._load(for_i2v=True)
            call_kw["image"] = load_image(str(req.image))
        else:
            pipe = self._load()
        frames = pipe(**call_kw).frames[0]

        from ..motion_studio._core import new_render_path, record_ledger
        out_path = Path(req.out) if req.out else new_render_path(f"wan-{mode}")
        export_to_video(frames, str(out_path), fps=req.fps)
        record_ledger({"kind": "videogen.wan", "path": str(out_path),
                       "mode": mode, "model": self.model_id,
                       "frames": num_frames, "steps": steps,
                       "prompt": req.prompt[:120]})
        return str(out_path)

    def unload(self) -> None:
        self._pipe = self._i2v_pipe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
