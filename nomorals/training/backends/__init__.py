"""Training backends: the selectable engines behind one pipeline.

The pipeline (``self_improvement`` → ``TrainingRegistry``) asks for a backend
BY NAME (the ``backend`` column on the run row, ``TrainingSettings.backend``,
or ``nm train --run --backend …``) and receives the same contract from all of
them: gate-shaped metrics plus an artifact path. The promotion gate is
backend-blind by design.

    native         pure Python, zero dependencies — runs on a phone
    mlx            LoRA on Apple Silicon via mlx-lm (the only trainer that
                   runs on a Mac — no CUDA needed)
    unsloth        QLoRA on a single CUDA GPU (Colab-friendly) — the production
                   path for 7B–8B finetunes
    axolotl        YAML-configured finetunes, free multi-GPU (FSDP2/DeepSpeed),
                   Liger-Kernel plugin when installed — the reproducible
                   scale-out path
    llama_factory  delegates to the LLaMA-Factory CLI (dataset + YAML + run)

Availability is a property of the MACHINE, not the code: ``available()`` is
checked at run time and a missing dependency degrades to a clear error, never
a crash of the import chain.
"""

from __future__ import annotations

from typing import Any

from .base import BackendResult, TrainingBackend
from .axolotl import AxolotlBackend
from .llama_factory import LlamaFactoryBackend
from .mlx import MLXBackend
from .native import NativeBackend
from .unsloth import UnslothBackend

__all__ = [
    "AxolotlBackend",
    "BackendResult",
    "LlamaFactoryBackend",
    "MLXBackend",
    "NativeBackend",
    "TrainingBackend",
    "UnslothBackend",
    "available_backends",
    "get_backend",
    "KNOWN_BACKENDS",
]

#: name → class. Order is the presentation order for ``available_backends``.
#: The pipeline asks for a backend BY NAME, so this order is advisory, not
#: a default — but it reads as the recommendation ladder:
#: native (runs everywhere) → mlx (the only trainer on Apple Silicon) →
#: unsloth (fastest single-GPU QLoRA) → axolotl (free multi-GPU scale-out,
#: YAML-reproducible) → llama_factory (breadth + web UI).
KNOWN_BACKENDS: dict[str, type[TrainingBackend]] = {
    "native": NativeBackend,
    "mlx": MLXBackend,
    "unsloth": UnslothBackend,
    "axolotl": AxolotlBackend,
    "llama_factory": LlamaFactoryBackend,
}


def get_backend(name: str = "") -> TrainingBackend:
    """Instantiate a backend by name. Unknown names raise ValueError with the
    list of valid ones — a typo must fail loudly, not silently retrain natively."""
    key = (name or "native").strip().lower()
    cls = KNOWN_BACKENDS.get(key)
    if cls is None:
        valid = ", ".join(KNOWN_BACKENDS)
        raise ValueError(f"unknown training backend {name!r} (valid: {valid})")
    return cls()


def available_backends() -> list[dict[str, Any]]:
    """{name, available, reason} for every known backend — doctor output."""
    out: list[dict[str, Any]] = []
    for name in KNOWN_BACKENDS:
        try:
            ok, reason = get_backend(name).available()
        except Exception as exc:  # noqa: BLE001 — an availability probe must never raise
            ok, reason = False, f"probe failed: {exc}"
        out.append({"name": name, "available": ok, "reason": reason})
    return out
