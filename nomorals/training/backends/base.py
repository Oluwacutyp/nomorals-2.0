"""The common interface every training backend implements.

A backend takes the SAME inputs the pipeline already has (curated ``Example``
rows, an output directory, the training settings) and returns the SAME output
the promotion gate understands (a gate-shaped metrics dict plus an artifact
path). The gate does not care who did the gradient descent — only whether the
result beats the incumbent.

Backends must be import-safe on machines that do not have their dependencies:
a phone has no torch, Colab has no LLaMA-Factory. Heavy imports happen inside
``available()`` / ``train()``, never at module top level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from ..dataset import Example

__all__ = ["BackendResult", "TrainingBackend"]


@dataclass
class BackendResult:
    """What a training run produced. ``metrics`` is gate-shaped: it must
    contain a numeric ``score`` in (0, 1] (higher is better) so
    ``TrainingRegistry.evaluate`` can compare it against the incumbent."""

    output_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.output_path)


class TrainingBackend(Protocol):
    """One way to turn a curated corpus into a trained model."""

    name: str

    def available(self) -> tuple[bool, str]:
        """(is usable on this machine, human-readable reason when not)."""
        ...

    def train(
        self,
        train: Sequence[Example],
        evaluation: Sequence[Example] = (),
        *,
        output_dir: str | Path,
        base_model: str = "",
        settings: Any = None,
        on_step: Callable[[int, float], None] | None = None,
    ) -> BackendResult:
        """Train on ``train`` (held-out ``evaluation`` when given).

        ``settings`` is the ``TrainingSettings`` dataclass; backends may read
        sensible defaults from it but must not require it.
        """
        ...
