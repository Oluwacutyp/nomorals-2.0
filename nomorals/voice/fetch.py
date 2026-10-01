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
    "fish-s2-pro": {
        "hf_repo": "fishaudio/s2-pro",
        "license": "Fish Audio Research License (non-commercial research; "
                   "read the repo LICENSE before production use)",
        "approx_size": "~8 GB (4B params)",
        "notes": (
            "SOTA open TTS: 10M+ hours, 80+ languages, 91.6% "
            "paralinguistics win rate on EmergentTTS-Eval. 15,000+ "
            "free-form [tag] directions (the director's canonical "
            "markup is near-native here), native multi-speaker "
            "<|speaker:i|> tags, 10–30s reference cloning, SGLang "
            "streaming at 0.195 RTF on H200. Local inference via the "
            "fish-speech repo's SGLang server; or reach it with no GPU "
            "through the 'hf-endpoint' backend."
        ),
        "alt_repos": [],
    },
    "fish-s1-mini": {
        "hf_repo": "fishaudio/openaudio-s1-mini",
        "license": "check the repo LICENSE (open weights)",
        "approx_size": "~1 GB (0.5B params)",
        "notes": (
            "Lightweight 0.5B Fish TTS for CPU/small-GPU boxes. Emotion "
            "and tone markers ((angry), (laughing), (sighing), …), EN/ZH/"
            "JA + more. Good first download to smoke-test the pipeline "
            "before pulling s2-pro."
        ),
        "alt_repos": [],
    },
    "orpheus": {
        "hf_repo": "canopylabs/orpheus-3b-0.1-ft",
        "license": "Apache-2.0",
        "approx_size": "~6 GB fp16 (GGUF Q8 ~3.5 GB, Q4 ~2.1 GB)",
        "notes": (
            "Canopy Labs' Llama-3B speech LLM: native <laugh>/<chuckle>/"
            "<sigh>/<cough>/<sniffle>/<groan>/<yawn>/<gasp> emotion tags, "
            "zero-shot voice cloning, 8 preset voices, ~200ms streaming "
            "latency. Needs the SNAC decoder too (hubertsiuzdak/snac_24khz). "
            "Full weights want a GPU via `pip install orpheus-speech`; "
            "the GGUF quants run on CPU under llama.cpp / Orpheus-FastAPI."
        ),
        "alt_repos": [
            # community GGUF quants for the CPU path
            "lex-au/Orpheus-3b-FT-Q8_0.gguf",
        ],
    },
    "dia": {
        "hf_repo": "nari-labs/Dia-1.6B-0626",
        "license": "Apache-2.0",
        "approx_size": "~3.5 GB",
        "notes": (
            "nari-labs dialogue TTS: ultra-realistic two-speaker dialogue "
            "in one pass, native parenthesized non-verbals (laughs), "
            "(clears throat), (sighs), (gasps), (coughs), (sneezes), "
            "(whistles), (groans)…, voice cloning via audio prompt. "
            "GPU-ONLY (~10GB VRAM); CPU support is on the Dia roadmap, "
            "not here yet. The 'dia' backend renders the director's "
            "script into Dia's [S1]/[S2] format."
        ),
        "alt_repos": [
            "nari-labs/Dia-1.6B",
        ],
    },
    "qwen3-tts": {
        # No verified official HF repo id at the time of writing — the
        # field stays empty on purpose rather than guessing. Use --repo
        # with the official Qwen release once published, or the
        # community C engine (github.com/bon5co/qwen3-tts).
        "hf_repo": "",
        "license": "check the official Qwen release license",
        "approx_size": "0.6B / 1.7B",
        "notes": (
            "Qwen's open TTS with inline [laugh]/[sigh]/[yawn]/[wow]/"
            "[giggle]/[scoff] paralinguistic tags and per-sentence "
            "[emotion] switching. The 0.6B is the most CPU-friendly "
            "expressive open model found. Paralinguistics are alpha "
            "(hit-or-miss across voices). The community pure-C engine "
            "runs it with no Python/PyTorch. Not wired as a backend "
            "yet — the director's onomatopoeia fallback covers the "
            "same ground on plain backends."
        ),
        "alt_repos": [],
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
    target_repo = repo or MODEL_REGISTRY[key]["hf_repo"]
    if not target_repo:
        raise ValueError(
            f"no verified HuggingFace repo for {key!r} yet — pass "
            f"--repo explicitly (see the registry notes for {key!r})")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed — pip install huggingface_hub"
        ) from exc

    target_dir = dest or default_cache_dir(key)
    path = snapshot_download(repo_id=target_repo, revision=revision,
                             local_dir=target_dir)
    return path
