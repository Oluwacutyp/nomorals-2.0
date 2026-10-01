"""Fetch open TTS weights from HuggingFace — no API keys, no cloud TTS.

The models live on HF; this module just downloads them into a local
cache so the backends in ``nomorals/voice/tts.py`` can load them.
``huggingface_hub`` is the only dependency, and it is imported lazily
so nothing here breaks when it is not installed.
"""

from __future__ import annotations

import os

__all__ = ["MODEL_REGISTRY", "default_cache_dir", "fetch_model"]

#: What's worth pulling, why, and under what license. ``hf_repo`` is the
#: default; ``nm voice fetch --repo`` can point at any other repo
#: (mirrors, fine-tunes, your own uploads).
MODEL_REGISTRY = {
    "cosyvoice": {
        "hf_repo": "FunAudioLLM/CosyVoice-3",
        "license": "MIT",
        "approx_size": "1–2 GB",
        "notes": (
            "Multilingual (9 languages, 18 dialects), zero-shot cloning "
            "from 3–10s of reference audio, instruction-driven "
            "paralinguistics ([laughter]/[breath] bursts, <laughter>/"
            "<strong> spans, emotion + rate control). ~150ms time to "
            "first token on GPU; works on CPU, slowly."
        ),
        "alt_repos": [
            # older 300M instruct checkpoint this backend's API was
            # written against; layout may differ, verify before use
            "gpustack/CosyVoice-300M",
        ],
    },
}


def default_cache_dir(name: str) -> str:
    """Local home for a backend's weights."""
    return os.path.join(
        os.path.expanduser("~"), ".cache", "nomorals",
        "voice_models", name)


def fetch_model(name: str, dest: str = "", repo: str = "",
                revision: str = "main") -> str:
    """Download a registered model's weights. Returns the directory."""
    key = (name or "").lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; registered: "
            f"{', '.join(sorted(MODEL_REGISTRY))}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed — pip install huggingface_hub"
        ) from exc

    target_repo = repo or MODEL_REGISTRY[key]["hf_repo"]
    target_dir = dest or default_cache_dir(key)
    path = snapshot_download(repo_id=target_repo, revision=revision,
                             local_dir=target_dir)
    return path
