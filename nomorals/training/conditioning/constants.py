"""Conditioning constants — a faithful port of upstream ``constants.py``.

Source: storytold/FineTrainers-Conditioning (upstream aarnphm/FineTrainers
snapshot, commit 9f0fdb5), ``finetrainers/constants.py``. Values are
verbatim; only the module docstring and the Devon additions at the end
(:data:`DEVON_MODEL_IDS`, :data:`PRECOMPUTED_MANIFEST_NAME`) are new.

Upstream notes that survive the port: the bucket lists are the defaults
``--video_resolution_buckets`` falls back to, and
:data:`COMMON_LLM_START_PHRASES` is the exact list
``--remove_common_llm_caption_prefixes`` strips.
"""

from __future__ import annotations

import os

__all__ = [
    "DEFAULT_FRAME_BUCKETS",
    "DEFAULT_HEIGHT_BUCKETS",
    "DEFAULT_IMAGE_RESOLUTION_BUCKETS",
    "DEFAULT_VIDEO_RESOLUTION_BUCKETS",
    "DEFAULT_WIDTH_BUCKETS",
    "COMMON_LLM_START_PHRASES",
    "DEVON_MODEL_IDS",
    "FINETRAINERS_LOG_LEVEL",
    "MODEL_DESCRIPTION",
    "PRECOMPUTED_CONDITIONS_DIR_NAME",
    "PRECOMPUTED_DIR_NAME",
    "PRECOMPUTED_LATENTS_DIR_NAME",
    "PRECOMPUTED_MANIFEST_NAME",
]


DEFAULT_HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
DEFAULT_WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
DEFAULT_FRAME_BUCKETS = [49]

DEFAULT_IMAGE_RESOLUTION_BUCKETS = []
for height in DEFAULT_HEIGHT_BUCKETS:
    for width in DEFAULT_WIDTH_BUCKETS:
        DEFAULT_IMAGE_RESOLUTION_BUCKETS.append((height, width))

DEFAULT_VIDEO_RESOLUTION_BUCKETS = []
for frames in DEFAULT_FRAME_BUCKETS:
    for height in DEFAULT_HEIGHT_BUCKETS:
        for width in DEFAULT_WIDTH_BUCKETS:
            DEFAULT_VIDEO_RESOLUTION_BUCKETS.append((frames, height, width))


FINETRAINERS_LOG_LEVEL = os.environ.get("FINETRAINERS_LOG_LEVEL", "INFO")

PRECOMPUTED_DIR_NAME = "precomputed"
PRECOMPUTED_CONDITIONS_DIR_NAME = "conditions"
PRECOMPUTED_LATENTS_DIR_NAME = "latents"

MODEL_DESCRIPTION = r"""
\# {model_id} {training_type} finetune

<Gallery />

\#\# Model Description

This model is a {training_type} of the `{model_id}` model.

This model was trained using the `fine-video-trainers` library - a repository containing memory-optimized scripts for training video models with [Diffusers](https://github.com/huggingface/diffusers).

\#\# Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

\#\# Usage

Requires [🧨 Diffusers](https://github.com/huggingface/diffusers) installed.

```python
{model_example}
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.

\#\# License

Please adhere to the license of the base model.
""".strip()

_COMMON_BEGINNING_PHRASES = (
    "This video",
    "The video",
    "This clip",
    "The clip",
    "The animation",
    "This image",
    "The image",
    "This picture",
    "The picture",
)
_COMMON_CONTINUATION_WORDS = ("shows", "depicts", "features", "captures", "highlights", "introduces", "presents")

COMMON_LLM_START_PHRASES = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    *(
        f"{beginning} {continuation}"
        for beginning in _COMMON_BEGINNING_PHRASES
        for continuation in _COMMON_CONTINUATION_WORDS
    ),
)

#: Upstream model ids per spec, plus the checkpoint Devon's own videogen
#: organ actually runs for that backend (nomorals/media/videogen/).
#: The LTX distilled id is the one Devon loads; the base id is what the
#: upstream spec defaults to. Both are recorded so a precompute run names
#: the checkpoint it was built against.
DEVON_MODEL_IDS = {
    "cogvideox": {
        "upstream_default": "THUDM/CogVideoX-5b",
        "devon_runs": (),
    },
    "hunyuan_video": {
        "upstream_default": "hunyuanvideo-community/HunyuanVideo",
        "devon_runs": (),
    },
    "ltx_video": {
        "upstream_default": "Lightricks/LTX-Video",
        "devon_runs": (
            "Lightricks/LTX-Video",
            "Lightricks/LTX-Video-0.9.7-distilled",
        ),
    },
}

#: Provenance file this port writes next to a precomputed store. Upstream
#: writes no manifest (it re-derives everything from the directory); the
#: manifest is a Devon addition so a store says which model, codec, and
#: encoder built it instead of forcing the reader to guess from filenames.
PRECOMPUTED_MANIFEST_NAME = "manifest.json"
