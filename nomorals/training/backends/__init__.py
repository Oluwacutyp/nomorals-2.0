"""Training backends: the selectable engines behind one pipeline.

The pipeline (``self_improvement`` → ``TrainingRegistry``) asks for a backend
BY NAME (the ``backend`` column on the run row, ``TrainingSettings.backend``,
or ``nm train --run --backend …``) and receives the same contract from all of
them: gate-shaped metrics plus an artifact path. The promotion gate is
backend-blind by design.

    native         pure Python, zero dependencies — runs on a phone
    unsloth        QLoRA on a single CUDA GPU (Colab-friendly) — the production
                   path for 7B–8B finetunes
    llama_factory  delegates to the LLaMA-Factory CLI (dataset + YAML + run)

Availability is a property of the MACHINE, not the code: ``available()`` is
checked at run time and a missing dependency degrades to a clear error, never
a crash of the import chain.
"""

from __future__ import annotations

from typing import Any

from .base import BackendResult, TrainingBackend
from .llama_factory import LlamaFactoryBackend
from .native import NativeBackend
from .unsloth import UnslothBackend

__all__ = [
    "BackendResult",
    "LlamaFactoryBackend",
    "NativeBackend",
    "TrainingBackend",
    "UnslothBackend",
    "available_backends",
    "get_backend",
    "KNOWN_BACKENDS",
]

#: name → class. Order is the presentation order for ``available_backends``.
KNOWN_BACKENDS: dict[str, type[TrainingBackend]] = {
    "native": NativeBackend,
    "unsloth": UnslothBackend,
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
