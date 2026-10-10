"""LTX-Video backend — the fast-iteration neural engine.

Target: **LTX-Video 2B distilled** — 8GB VRAM floor, ~real-time on a
4090, T2V + I2V + extension in one checkpoint family, 8 diffusion steps
with no classifier-free guidance.

``diffusers`` is the denoising backend (the trained DiT is the moat —
we don't reimplement it); everything around it is ours: prompt policy,
frame-count snapping, seed control, memory management, chaining.

Import-gated: without ``diffusers``/torch/CUDA this module loads fine
and every call fails honestly with install instructions — it never
pretends to generate.

Model IDs (verified 2026-10-09):
- base 2B (diffusers-native): ``Lightricks/LTX-Video``
- official distilled: ``Lightricks/LTX-Video-0.9.7-distilled``
  (Lightricks also released ``ltxv-2b-0.9.8-distilled`` upstream; the
  diffusers-format HF id for 0.9.8 2B was unverified — override
  ``model_id`` if you have it)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .capabilities import (
    LTX_MIN_VRAM_GB,
    VideogenError,
    cuda_vram_gb,
    diffusers_available,
    neural_capability,
    torch_cuda_available,
)
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["LTXBackend", "LTXClipRequest", "LTX_DEFAULTS"]

BASE_MODEL_ID = "Lightricks/LTX-Video"
DISTILLED_MODEL_ID = "Lightricks/LTX-Video-0.9.7-distilled"

#: sane defaults per mode — distilled = 8 steps, no CFG (that's the point)
LTX_DEFAULTS: dict[str, dict[str, Any]] = {
    "base": {"num_inference_steps": 30, "guidance_scale": 3.0},
    "distilled": {"num_inference_steps": 8, "guidance_scale": 1.0},
}


@dataclass
class LTXClipRequest:
    prompt: str
    mode: str = "t2v"                    # t2v | i2v | extend
    image: str | os.PathLike | None = None   # i2v source / extend seed frame
    negative_prompt: str = ("worst quality, inconsistent motion, blurry, "
                            "jittery, distorted")
    duration_s: float = 5.0
    width: int = 768
    height: int = 512
    fps: int = 24
    seed: int = 0
    out: str | os.PathLike | None = None
    distilled: bool = True
    model_id: str = ""
    num_inference_steps: int = 0         # 0 → mode default
    guidance_scale: float = 0.0          # 0 → mode default


def _snap_frames(duration_s: float, fps: int) -> int:
    """LTX needs num_frames = 8k+1 (VAE temporal compression)."""
    want = max(9, int(round(duration_s * fps)))
    k = max(1, round((want - 1) / 8))
    k = min(k, 32)  # cap at 257 frames (community glitch reports above ~168)
    return 8 * k + 1


def _snap_hw(w: int, h: int) -> tuple[int, int]:
    return max(32, (w // 32) * 32), max(32, (h // 32) * 32)


class LTXBackend:
    """LTX-Video 2B via diffusers. Lazy-loads the pipeline on first use."""

    name = "ltx"
    min_vram_gb = LTX_MIN_VRAM_GB

    def __init__(self, *, model_id: str = "", distilled: bool = True):
        self.model_id = model_id or (DISTILLED_MODEL_ID if distilled
                                     else BASE_MODEL_ID)
        self.distilled = distilled
        self._pipe: Any = None
        self._i2v_pipe: Any = None

    # -- capability ------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """Honest capability report — never raises."""
        cap = neural_capability(prefer="ltx")
        ok = cap.available and cap.backend == "ltx"
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "distilled": self.distilled,
            "available": ok,
            "vram_gb": cap.vram_gb,
            "reason": cap.reason if not ok else
                      f"ready — {self.model_id} ({cap.reason})",
        }

    def require(self) -> None:
        info = self.check()
        if not info["available"]:
            raise VideogenError(
                f"LTX backend unavailable: {info['reason']}")

    # -- pipeline loading -------------------------------------------------

    def _load(self, for_i2v: bool = False) -> Any:
        if not diffusers_available():
            raise VideogenError(
                "the `diffusers` package is not installed — "
                "`pip install diffusers accelerate` then retry")
        if not torch_cuda_available():
            raise VideogenError("no CUDA GPU — LTX needs CUDA ≥ 8GB VRAM")
        import torch
        if for_i2v and self._i2v_pipe is not None:
            return self._i2v_pipe
        if not for_i2v and self._pipe is not None:
            return self._pipe
        from diffusers import LTXPipeline
        _log.info("loading LTX pipeline %s …", self.model_id)
        pipe = LTXPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16)
        # memory plan: offload what doesn't fit, tile the VAE
        try:
            pipe.enable_model_cpu_offload()
        except Exception:  # noqa: BLE001 - older diffusers
            pipe.to("cuda")
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        except Exception:  # noqa: BLE001
            pass
        if for_i2v:
            self._i2v_pipe = pipe
        else:
            self._pipe = pipe
        return pipe

    def _load_i2v(self) -> tuple[Any, str]:
        """Return (pipeline, strategy) — native I2V pipe or conditioning."""
        if not diffusers_available():
            raise VideogenError(
                "the `diffusers` package is not installed — "
                "`pip install diffusers accelerate` then retry")
        try:
            from diffusers import LTXImageToVideoPipeline  # newer diffusers
            import torch
            if self._i2v_pipe is None:
                pipe = LTXImageToVideoPipeline.from_pretrained(
                    self.model_id, torch_dtype=torch.bfloat16)
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:  # noqa: BLE001
                    pipe.to("cuda")
                self._i2v_pipe = pipe
            return self._i2v_pipe, "native-i2v"
        except Exception:  # noqa: BLE001 - fall back to conditioning
            _log.info("LTXImageToVideoPipeline unavailable, "
                      "using LTXConditionPipeline strategy")
            from diffusers import LTXConditionPipeline
            from diffusers.pipelines.ltx.pipeline_ltx_condition import (
                LTXVideoCondition)
            from diffusers.utils import export_to_video, load_image, load_video
            import torch
            pipe = LTXConditionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16)
            try:
                pipe.enable_model_cpu_offload()
            except Exception:  # noqa: BLE001
                pipe.to("cuda")
            return (pipe, "condition",
                    LTXVideoCondition, export_to_video, load_image, load_video)

    # -- generation -------------------------------------------------------

    def generate(self, prompt: str, **kw) -> str:
        """Generate a clip. Returns the output mp4 path.

        ``mode``: t2v | i2v (needs ``image``) | extend (needs ``image`` =
        last frame of the previous clip — orchestration-level extension).
        """
        req = LTXClipRequest(prompt=prompt, **kw) if not isinstance(
            prompt, LTXClipRequest) else prompt
        self.require()
        if not req.prompt.strip():
            raise VideogenError("empty prompt — nothing to generate")
        mode = req.mode.lower()
        if mode not in ("t2v", "i2v", "extend"):
            raise VideogenError(
                f"unknown LTX mode {mode!r} — t2v | i2v | extend")
        if mode in ("i2v", "extend") and not req.image:
            raise VideogenError(f"LTX {mode} needs an `image` input")
        if req.image and not Path(req.image).exists():
            raise VideogenError(f"input image not found: {req.image}")

        import torch
        from diffusers.utils import export_to_video

        defaults = LTX_DEFAULTS["distilled" if req.distilled else "base"]
        steps = req.num_inference_steps or defaults["num_inference_steps"]
        guidance = (req.guidance_scale if req.guidance_scale > 0
                    else defaults["guidance_scale"])
        width, height = _snap_hw(req.width, req.height)
        num_frames = _snap_frames(req.duration_s, req.fps)
        generator = torch.Generator(device="cuda").manual_seed(req.seed)

        if mode == "t2v":
            pipe = self._load()
            out = pipe(prompt=req.prompt,
                       negative_prompt=req.negative_prompt or None,
                       width=width, height=height,
                       num_frames=num_frames,
                       num_inference_steps=steps,
                       guidance_scale=guidance,
                       generator=generator)
            frames = out.frames[0]
        else:
            loaded = self._load_i2v()
            if loaded[1] == "native-i2v":
                pipe = loaded[0]
                from diffusers.utils import load_image
                image = load_image(str(req.image))
                out = pipe(image=image, prompt=req.prompt,
                           negative_prompt=req.negative_prompt or None,
                           width=width, height=height,
                           num_frames=num_frames,
                           num_inference_steps=steps,
                           guidance_scale=guidance,
                           generator=generator)
                frames = out.frames[0]
            else:
                pipe, _, LTXVideoCondition, export_to_video, load_image, load_video = loaded
                image = load_image(str(req.image))
                # the model was trained on videos — compress the still
                video = load_video(export_to_video([image]))
                condition = LTXVideoCondition(video=video, frame_index=0)
                out = pipe(conditions=[condition], prompt=req.prompt,
                           negative_prompt=req.negative_prompt or None,
                           width=width, height=height,
                           num_frames=num_frames,
                           num_inference_steps=steps,
                           guidance_scale=guidance,
                           generator=generator)
                frames = out.frames[0]

        from ..motion_studio._core import new_render_path, record_ledger
        out_path = Path(req.out) if req.out else new_render_path(f"ltx-{mode}")
        export_to_video(frames, str(out_path), fps=req.fps)
        record_ledger({"kind": "videogen.ltx", "path": str(out_path),
                       "mode": mode, "model": self.model_id,
                       "frames": num_frames, "steps": steps,
                       "prompt": req.prompt[:120]})
        return str(out_path)

    def unload(self) -> None:
        """Drop pipelines + CUDA cache (long chaining sessions)."""
        self._pipe = self._i2v_pipe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
