"""The native backend: pure-Python next-token MLP, zero dependencies.

This is the backend that runs on a phone. It trains a real model with real
gradient descent (see ``trainer.py``) — small, slow, and dependency-free, but
complete enough that the whole collect → train → evaluate → promote loop works
without torch, CUDA, or a Colab session.

The BPE vocabulary is trained from the corpus itself (as the pipeline has
always done), so tiny corpora still produce a working artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from ...core.logging_setup import get_logger
from ..dataset import Example
from ..tokenize import BPETokenizer
from ..trainer import NativeTrainer, TrainConfig
from .base import BackendResult

__all__ = ["NativeBackend", "default_train_config"]

_log = get_logger(__name__)


def default_train_config(settings: Any = None) -> TrainConfig:
    """Pipeline defaults, overridable by TrainingSettings when present."""
    kwargs: dict[str, Any] = dict(hidden_size=32)
    if settings is not None:
        kwargs["context_window"] = min(16, max(2, int(settings.max_seq_len)))
        kwargs["learning_rate"] = max(0.0001, float(settings.learning_rate))
        kwargs["epochs"] = max(1, int(settings.epochs))
        kwargs["batch_size"] = max(1, int(settings.batch_size))
    return TrainConfig(**kwargs)


class NativeBackend:
    """Wraps :class:`NativeTrainer` behind the :class:`TrainingBackend` API."""

    name = "native"

    def available(self) -> tuple[bool, str]:
        return True, "pure Python — runs everywhere"

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
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)

        texts = [example.to_chatml() for example in train]
        if not texts:
            raise ValueError("native backend: training set is empty")
        tokenizer = BPETokenizer.train(texts, vocab_size=512, min_frequency=2)
        tokenizer.save(out / "tokenizer.json")

        config = default_train_config(settings)
        model, metrics = NativeTrainer(tokenizer, config, on_step=on_step).fit(train, evaluation)
        model.save(out)
        _log.info("native backend: model saved to %s", out)
        return BackendResult(
            output_path=str(out),
            metrics=metrics.as_scores(),
            info={
                "architecture": "native-mlp-lm",
                "vocab_size": tokenizer.vocab_size,
                "full_metrics": metrics.to_dict(),
            },
        )
