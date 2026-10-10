"""Devon Studio — neural video generation (workstation-gated).

Orchestration is ours; ``diffusers`` is the denoising backend (weights
are data). Two engines:

- :mod:`ltx_backend` — LTX-Video 2B distilled: 8GB VRAM floor,
  ~real-time iteration, T2V + I2V + extension.
- :mod:`wan_backend` — Wan 2.2 TI2V-5B: 720p24 quality finals,
  Apache 2.0, 16GB+ VRAM.

:mod:`pipeline` is the front door (capability check → neural or honest
motion-studio fallback); :mod:`chaining` assembles multi-scene films.

Below the VRAM floor everything routes to
:mod:`nomorals.media.motion_studio` with an honest message — never
silent, never fake.
"""

from .capabilities import (
    VideogenError,
    neural_capability,
    cuda_vram_gb,
    diffusers_available,
)
from .pipeline import generate, request_hero_clip, capability_report, VideoResult
from .chaining import chain_scenes, ChainReport
from .autotune import autotune_request, AUTOTUNE
from .consistency import (
    consistency_pass,
    boundary_metric,
    reinhard_match,
)

__all__ = [
    "VideogenError",
    "neural_capability",
    "cuda_vram_gb",
    "diffusers_available",
    "generate",
    "request_hero_clip",
    "capability_report",
    "VideoResult",
    "chain_scenes",
    "ChainReport",
    "autotune_request",
    "AUTOTUNE",
    "consistency_pass",
    "boundary_metric",
    "reinhard_match",
]
