"""Devon Studio — neural video generation (workstation-gated).

Orchestration is ours; ``diffusers`` is the denoising backend (weights
are data). Two engines:

- :mod:`ltx_backend` — LTX-Video 2B distilled: 8GB VRAM floor,
  ~real-time iteration, T2V + I2V + extension.
- :mod:`wan_backend` — Wan 2.2 TI2V-5B: 720p24 quality finals,
  Apache 2.0, 16GB+ VRAM.
- :mod:`codegen` — the code backend: deterministic seek(t) scene
  rendering, no model, CPU-only; :mod:`formats` holds the reusable
  format presets (``map-route``, ``headline``, ``teardown``).
- :mod:`qc` — the variants-matrix QC organ: the same job rendered
  across rungs/params, a contact sheet per variant, one matrix
  sheet, and a report naming the winning variant and why.

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
from .codegen import render_code_scene
from .formats import build_format, list_formats, render_format
from .presets import build_preset, list_presets, render_preset
from .narration import scene_narration, synthesize_narration
from .captions import (
    caption_narration,
    check_caption_timing,
    draw_captions,
    captions_srt,
)
from .qc import matrix_message, plan_variants, qc_existing, render_matrix
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
    "render_code_scene",
    "build_format",
    "list_formats",
    "render_format",
    "build_preset",
    "list_presets",
    "render_preset",
    "scene_narration",
    "synthesize_narration",
    "caption_narration",
    "check_caption_timing",
    "draw_captions",
    "captions_srt",
    "plan_variants",
    "render_matrix",
    "qc_existing",
    "matrix_message",
    "chain_scenes",
    "ChainReport",
    "autotune_request",
    "AUTOTUNE",
    "consistency_pass",
    "boundary_metric",
    "reinhard_match",
]
