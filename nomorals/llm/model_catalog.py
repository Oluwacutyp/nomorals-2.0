"""Brain upgrade catalog for Devon's local GGUF brain.

The upgrade lineup lives here as data, not tribal knowledge. The
incumbent vs challenger decision (codebeast-3.8b vs qwen3.5-4b) is made
with evidence from :mod:`nomorals.llm.benchmark`, not vibes.

Standing decisions (from the owner):
- Phone brain today: CodeBeast 3.8B (Phi-3.5-mini QLoRA fine-tune).
- Multimodal base: Qwen/Qwen3-VL-4B-Instruct (Apache 2.0, ~3.3GB
  Q4_K_M + mmproj, under the 4GB phone target).
- The Unsloth #3899 tiny-merge → GGUF → mmproj validation is a
  separate side-chat task; qwen3-vl-4b is NOT phone-ready until it
  lands.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: The brain upgrade lineup. ``hf_repo`` is the GGUF source;
#: ``size_gb`` is the Q4_K_M file size; ``context`` is the tested
#: context window. Keep this curated — model repos move fast.
BRAIN_MODELS: dict[str, dict[str, Any]] = {
    "codebeast-3.8b": {
        "hf_repo": "Cutyp/codebeast-3.8b",
        "quant": "Q4_K_M",
        "size_gb": 2.32,
        "context": 4096,
        "license": "MIT (QLoRA fine-tune of Apache-2.0 microsoft/Phi-3.5-mini-instruct)",
        "notes": (
            "INCUMBENT phone brain. Proven on Termux. 30K-row run done "
            "2026-10-05; verdict 'better' but not clearly outclassing the "
            "earlier dataset. Stays the termux default until the benchmark "
            "harness proves a challenger wins on Devon's own task set."
        ),
    },
    "qwen3.5-4b": {
        "hf_repo": "Qwen/Qwen3.5-4B-GGUF",
        "quant": "Q4_K_M",
        "size_gb": 3.4,
        "context": 4096,
        "license": "Apache 2.0",
        "notes": (
            "CHALLENGER. Newer architecture than Phi-3.5; ~3.4GB Q4_K_M "
            "fits the phone budget. Run the benchmark harness before "
            "promoting — newer != better for Devon's workload."
        ),
    },
    "phi-4-mini": {
        "hf_repo": "microsoft/Phi-4-mini-instruct-GGUF",
        "quant": "Q4_K_M",
        "size_gb": 2.5,
        "context": 4096,
        "license": "MIT",
        "notes": (
            "In-family upgrade alternative: same Phi lineage as the "
            "incumbent, so behavior deltas are easier to reason about. "
            "Consider if qwen3.5-4b wins on benchmarks but feels alien."
        ),
    },
    "qwen3-vl-4b": {
        "hf_repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "quant": "Q4_K_M",
        "size_gb": 3.3,
        "context": 8192,
        "license": "Apache 2.0",
        "needs_mmproj": True,
        "notes": (
            "Multimodal base (owner's standing choice): ~3.3GB Q4_K_M + "
            "mmproj under the 4GB phone target. BLOCKED on the Unsloth "
            "#3899 tiny-merge → GGUF → mmproj validation (separate "
            "side-chat task) — do not ship to the phone before that "
            "validation lands."
        ),
    },
}

#: env var that pins the brain explicitly (beats profile defaults).
BRAIN_ENV_VAR = "DEVON_BRAIN"


def list_brains() -> list[dict[str, Any]]:
    """The catalog as a list of dicts (each includes its ``id``)."""
    return [{"id": model_id, **spec} for model_id, spec in BRAIN_MODELS.items()]


def _mmproj_present(cache_dir: str = "models") -> bool:
    """Best-effort check: does an mmproj file exist in the model cache?"""
    from pathlib import Path

    try:
        base = Path(os.path.expanduser(cache_dir))
        return any(base.rglob("*mmproj*"))
    except Exception:  # noqa: BLE001 - best effort only
        return False


def pick_brain(profile: str = "", *, multimodal: bool = False,
               cache_dir: str = "models") -> str:
    """Pick the brain model id for ``profile``.

    - ``DEVON_BRAIN`` env always wins (explicit owner choice).
    - ``multimodal=True`` → qwen3-vl-4b, with a loud warning when no
      mmproj is found (the #3899 validation is still pending).
    - termux → codebeast-3.8b (incumbent, proven on the phone).
    - laptop/workstation → qwen3.5-4b (challenger; benchmark decides).
    """
    override = (os.environ.get(BRAIN_ENV_VAR, "") or "").strip()
    if override:
        if override not in BRAIN_MODELS:
            _log.warning("DEVON_BRAIN=%r not in catalog; using it anyway", override)
        return override
    if multimodal:
        if not _mmproj_present(cache_dir):
            _log.warning(
                "qwen3-vl-4b selected but no mmproj found in %s — the "
                "Unsloth #3899 tiny-merge → GGUF → mmproj validation has "
                "not landed. Vision will not work until it does.",
                cache_dir,
            )
        return "qwen3-vl-4b"
    from ..core.profiles import get_profile_kind

    kind = (profile or "").strip().lower() or get_profile_kind()
    if kind == "termux":
        return "codebeast-3.8b"
    return "qwen3.5-4b"


def brain_spec(model_id: str) -> dict[str, Any]:
    """Catalog entry for ``model_id`` (empty dict when unknown)."""
    spec = BRAIN_MODELS.get(model_id)
    return {"id": model_id, **spec} if spec else {}
