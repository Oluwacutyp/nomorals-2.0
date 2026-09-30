"""A native pure-Python training loop.

This trains a real model with real gradient descent — a single-hidden-layer
softmax over the BPE vocabulary, predicting the next token. It is not a
transformer and it will not beat one. What it is: a complete, dependency-free
training path that runs on a phone, so the self-improvement loop is exercisable
end to end without torch.

When torch is available, ``TorchTrainer`` is the better backend and this one
steps aside. The point of both is that they produce a model the promotion gate
can evaluate on the same terms.

Gradients are computed analytically for this architecture. That is deliberate: a
hand-rolled autograd engine would be slower to run and far slower to trust.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..core.errors import ValidationError
from ..core.logging_setup import get_logger
from .dataset import Example
from .tokenize import BPETokenizer

__all__ = ["TrainConfig", "TrainMetrics", "NativeTrainer", "TrainedModel"]

_log = get_logger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters. Deliberately small defaults: this runs on a phone."""

    hidden_size: int = 64
    context_window: int = 16
    learning_rate: float = 0.05
    epochs: int = 3
    batch_size: int = 8
    l2: float = 1e-4
    seed: int = 1234
    log_every: int = 50
    max_examples: int = 0

    def validate(self) -> None:
        if self.hidden_size < 1:
            raise ValidationError("hidden_size must be positive", field="hidden_size")
        if self.learning_rate <= 0:
            raise ValidationError("learning_rate must be positive", field="learning_rate")
        if self.epochs < 1:
            raise ValidationError("epochs must be at least 1", field="epochs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size, "context_window": self.context_window,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "l2": self.l2, "seed": self.seed,
            "log_every": self.log_every, "max_examples": self.max_examples,
        }


@dataclass
class TrainMetrics:
    """What a run actually achieved. These feed the promotion gate."""

    steps: int = 0
    epochs: float = 0.0
    final_loss: float = 0.0
    best_eval_loss: float = float("inf")
    train_loss_history: list[float] = field(default_factory=list)
    eval_loss_history: list[float] = field(default_factory=list)
    seconds: float = 0.0
    examples: int = 0
    tokens: int = 0
    backend: str = "native"

    @property
    def perplexity(self) -> float:
        """exp(loss). Reported alongside loss because it is the interpretable one."""
        if not math.isfinite(self.best_eval_loss):
            return float("inf")
        return math.exp(min(20.0, self.best_eval_loss))

    def as_scores(self) -> dict[str, float]:
        """Shaped for ``ModelRegistry.record_eval`` / ``beats_incumbent``."""
        out: dict[str, float] = {"train_loss": round(self.final_loss, 6)}
        if math.isfinite(self.best_eval_loss):
            out["eval_loss"] = round(self.best_eval_loss, 6)
            out["perplexity"] = round(self.perplexity, 4)
            # A single ascending metric, so the gate needs no special-casing:
            # higher is better, bounded in (0, 1].
            out["score"] = round(1.0 / (1.0 + self.best_eval_loss), 6)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.as_scores(),
            "steps": self.steps, "epochs": round(self.epochs, 3),
            "seconds": round(self.seconds, 2), "examples": self.examples,
            "tokens": self.tokens, "backend": self.backend,
            "train_loss_history": [round(v, 5) for v in self.train_loss_history[-20:]],
            "eval_loss_history": [round(v, 5) for v in self.eval_loss_history[-20:]],
        }


@dataclass
class TrainedModel:
    """The artifact: weights plus the tokenizer needed to use them."""

    config: TrainConfig
    vocab_size: int
    input_weights: list[list[float]]
    hidden_weights: list[list[float]]
    output_weights: list[list[float]]
    hidden_bias: list[float]
    output_bias: list[float]
    metrics: TrainMetrics
    tokenizer_path: str = ""

    def save(self, directory: str | os.PathLike[str]) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.json").write_text(
            json.dumps(
                {
                    "architecture": "native-mlp-lm",
                    "vocab_size": self.vocab_size,
                    "config": self.config.to_dict(),
                    "metrics": self.metrics.to_dict(),
                    "tokenizer_path": self.tokenizer_path,
                    "weights": {
                        "input": self.input_weights, "hidden": self.hidden_weights,
                        "output": self.output_weights, "hidden_bias": self.hidden_bias,
                        "output_bias": self.output_bias,
                    },
                }
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> "TrainedModel":
        data = json.loads((Path(directory).expanduser() / "model.json").read_text(encoding="utf-8"))
        weights = data["weights"]
        config_data = data.get("config") or {}
        return cls(
            config=TrainConfig(**{k: v for k, v in config_data.items() if k in TrainConfig.__dataclass_fields__}),
            vocab_size=int(data["vocab_size"]),
            input_weights=weights["input"], hidden_weights=weights["hidden"],
            output_weights=weights["output"], hidden_bias=weights["hidden_bias"],
            output_bias=weights["output_bias"],
            metrics=TrainMetrics(), tokenizer_path=data.get("tokenizer_path", ""),
        )


class NativeTrainer:
    """Trains a next-token MLP with plain Python and analytic gradients."""

    def __init__(
        self,
        tokenizer: BPETokenizer,
        config: TrainConfig | None = None,
        *,
        on_step: Callable[[int, float], None] | None = None,
    ) -> None:
        config = config or TrainConfig()
        config.validate()
        self.tokenizer = tokenizer
        self.config = config
        self.on_step = on_step
        self._rng = random.Random(config.seed)
        self._cancelled = False

    def cancel(self) -> None:
        """Cooperative stop, checked between batches."""
        self._cancelled = True

    def _cpp_kernel(self) -> bool:
        """Whether the C++ mlptrain kernel carries this run's batch math.

        The kernel (nomorals/native/mlptrain.cpp) runs the SAME operation
        order and clamps as :meth:`_batch_gradients` — parity is enforced by
        test_wave92 element-wise — and additionally applies the SGD step so
        one ctypes call covers a whole batch.  Availability is a property of
        the machine; a missing/broken .so silently degrades to pure Python.
        """
        try:
            from .. import native as _native
            return bool(_native.mlp_available())
        except Exception:  # noqa: BLE001 — a broken extension is not a crash
            return False

    @staticmethod
    def _cpp_batch(seq_batch, buf: dict, *, context: int, lr: float, l2: float):
        """One C++ batch (gradients + SGD step when lr > 0)."""
        from array import array

        from .. import native as _native
        flat_ids: array = array("i")
        lens: array = array("i")
        for ids in seq_batch:
            flat_ids.extend(ids)
            lens.append(len(ids))
        return _native.mlp_batch(
            flat_ids, lens,
            buf["embedding"], buf["hidden_w"], buf["hidden_b"],
            buf["out_w"], buf["out_b"],
            context=context, lr=lr, l2=l2)

    def fit(
        self,
        train: Sequence[Example],
        evaluation: Sequence[Example] = (),
    ) -> tuple[TrainedModel, TrainMetrics]:
        config = self.config
        samples = self._tokenize(train)
        if not samples:
            raise ValidationError("training set produced no usable token sequences")
        eval_samples = self._tokenize(evaluation) if evaluation else []

        vocab = self.tokenizer.vocab_size
        context = config.context_window
        hidden = config.hidden_size
        scale = 1.0 / math.sqrt(context)

        # Weight shapes: input[context*vocab -> hidden], output[hidden -> vocab].
        # Dense vocab-sized input would be vocab*context*hidden floats, which is
        # far too large. Use a hashed context feature instead: sum of scaled
        # per-position embeddings. Keeps memory bounded by vocab*hidden.
        embedding = [[self._rng.uniform(-scale, scale) for _ in range(hidden)] for _ in range(vocab)]
        hidden_weights = [[self._rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(hidden)]
        hidden_bias = [0.0] * hidden
        output_weights = [[0.0] * vocab for _ in range(hidden)]
        output_bias = [0.0] * vocab

        buffers = None
        if self._cpp_kernel():
            try:
                buffers = _pack_weights(embedding, hidden_weights, hidden_bias,
                                        output_weights, output_bias)
            except Exception:  # noqa: BLE001 — fall back rather than fail the run
                buffers = None

        metrics = TrainMetrics(examples=len(samples), backend="native")
        started = time.perf_counter()
        batches = max(1, len(samples) // max(1, config.batch_size))
        step = 0

        for epoch in range(config.epochs):
            if self._cancelled:
                break
            self._rng.shuffle(samples)
            running = 0.0
            tokens = 0
            for batch_start in range(0, len(samples), config.batch_size):
                if self._cancelled:
                    break
                batch = samples[batch_start : batch_start + config.batch_size]
                lr = config.learning_rate / (1.0 + 0.01 * step)
                if buffers is not None:
                    loss, count, _ = self._cpp_batch(
                        batch, buffers, context=context, lr=lr, l2=config.l2 * lr)
                else:
                    loss, grad_embed, grad_hidden_w, grad_hidden_b, grad_out_w, grad_out_b, count = (
                        self._batch_gradients(
                            batch, embedding, hidden_weights, hidden_bias, output_weights, output_bias
                        )
                    )
                    _axpy(embedding, grad_embed, -lr, config.l2 * lr)
                    _axpy(hidden_weights, grad_hidden_w, -lr, config.l2 * lr)
                    _axpy(output_weights, grad_out_w, -lr, config.l2 * lr)
                    _add(hidden_bias, grad_hidden_b, -lr)
                    _add(output_bias, grad_out_b, -lr)
                running += loss
                tokens += count
                step += 1
                if self.on_step and step % config.log_every == 0:
                    self.on_step(step, loss)

            if buffers is not None:
                _unpack_weights(buffers, embedding, hidden_weights, hidden_bias,
                                output_weights, output_bias)
            average = running / max(1, batches)
            metrics.train_loss_history.append(average)
            metrics.final_loss = average
            metrics.tokens += tokens
            metrics.epochs = epoch + 1
            _log.info("epoch %d train_loss=%.4f", epoch + 1, average)

            if eval_samples:
                eval_loss = self.evaluate(
                    eval_samples, embedding, hidden_weights, hidden_bias, output_weights, output_bias
                )
                metrics.eval_loss_history.append(eval_loss)
                metrics.best_eval_loss = min(metrics.best_eval_loss, eval_loss)
                _log.info("epoch %d eval_loss=%.4f", epoch + 1, eval_loss)

        metrics.steps = step
        metrics.seconds = time.perf_counter() - started
        if not eval_samples:
            # Without a held-out set the train loss is the only signal. Say so
            # rather than reporting a number that looks like validation.
            metrics.best_eval_loss = metrics.final_loss

        model = TrainedModel(
            config=config, vocab_size=vocab,
            input_weights=embedding, hidden_weights=hidden_weights,
            output_weights=output_weights, hidden_bias=hidden_bias,
            output_bias=output_bias, metrics=metrics,
            tokenizer_path=self.tokenizer.path if hasattr(self.tokenizer, "path") else "",
        )
        _log.info(
            "training done: %d steps in %.1fs, loss %.4f, perplexity %.2f",
            step, metrics.seconds, metrics.final_loss, metrics.perplexity,
        )
        return model, metrics

    # ── forward / backward ───────────────────────────────────────────────────

    def _tokenize(self, examples: Sequence[Example]) -> list[list[int]]:
        limit = self.config.max_examples
        out: list[list[int]] = []
        for index, example in enumerate(examples):
            if limit and index >= limit:
                break
            ids = self.tokenizer.encode(example.to_chatml())
            if len(ids) > self.config.context_window + 1:
                out.append(ids)
        return out

    def _context_vector(
        self, ids: Sequence[int], embedding: list[list[float]], hidden: int
    ) -> list[float]:
        """Mean of the last ``context`` token embeddings. O(context*hidden)."""
        window = ids[-self.config.context_window :]
        if not window:
            return [0.0] * hidden
        accumulator = [0.0] * hidden
        for token_id in window:
            row = embedding[token_id % len(embedding)]
            for j in range(hidden):
                accumulator[j] += row[j]
        factor = 1.0 / len(window)
        return [v * factor for v in accumulator]

    def _forward(
        self,
        context: Sequence[float],
        hidden_weights: list[list[float]],
        hidden_bias: Sequence[float],
        output_weights: list[list[float]],
        output_bias: Sequence[float],
    ) -> tuple[list[float], list[float], list[float]]:
        """Returns (hidden_pre, hidden_post, log_probs)."""
        hidden = len(hidden_bias)
        pre = list(hidden_bias)
        for j in range(hidden):
            row = hidden_weights[j]
            total = hidden_bias[j]
            for k, value in enumerate(context):
                total += value * row[k]
            pre[j] = total
        post = [v if v > 0 else 0.0 for v in pre]  # ReLU
        vocab = len(output_bias)
        logits = list(output_bias)
        for j in range(hidden):
            if post[j] == 0.0:
                continue
            row = output_weights[j]
            for i in range(vocab):
                logits[i] += post[j] * row[i]
        maximum = max(logits)
        exps = [math.exp(min(50.0, v - maximum)) for v in logits]
        total = sum(exps)
        return pre, post, [math.log(max(1e-12, e / total)) for e in exps]

    def _batch_gradients(
        self,
        batch: Sequence[Sequence[int]],
        embedding: list[list[float]],
        hidden_weights: list[list[float]],
        hidden_bias: Sequence[float],
        output_weights: list[list[float]],
        output_bias: Sequence[float],
    ) -> tuple[float, list, list, list, list, list, int]:
        vocab = len(output_bias)
        hidden = len(hidden_bias)
        grad_embed = [[0.0] * hidden for _ in range(vocab)]
        grad_hidden_w = [[0.0] * hidden for _ in range(hidden)]
        grad_hidden_b = [0.0] * hidden
        grad_out_w = [[0.0] * vocab for _ in range(hidden)]
        grad_out_b = [0.0] * vocab
        total_loss = 0.0
        count = 0

        for sequence in batch:
            for position in range(self.config.context_window, len(sequence)):
                history = sequence[:position]
                target = sequence[position]
                context = self._context_vector(history, embedding, hidden)
                pre, post, log_probs = self._forward(
                    context, hidden_weights, hidden_bias, output_weights, output_bias
                )
                total_loss -= log_probs[target]
                count += 1

                # d(loss)/d(logits) = softmax - onehot(target)
                probs = [math.exp(min(50.0, lp)) for lp in log_probs]
                for j in range(hidden):
                    if post[j] == 0.0:
                        continue
                    row = grad_out_w[j]
                    for i in range(vocab):
                        row[i] += post[j] * probs[i]
                    row[target] -= post[j]
                for i in range(vocab):
                    grad_out_b[i] += probs[i]
                grad_out_b[target] -= 1.0

                # Backprop through ReLU into the hidden layer and the embedding.
                grad_hidden = [0.0] * hidden
                for i in range(vocab):
                    if probs[i] == 0.0:
                        continue
                    row = output_weights[i] if i < len(output_weights) else None
                for j in range(hidden):
                    if pre[j] <= 0:
                        continue
                    accumulated = 0.0
                    for i in range(vocab):
                        delta = probs[i] - (1.0 if i == target else 0.0)
                        accumulated += delta * output_weights[j][i]
                    grad_hidden[j] = accumulated
                    grad_hidden_b[j] += accumulated
                    row = grad_hidden_w[j]
                    for k, value in enumerate(context):
                        row[k] += accumulated * value

                window = history[-self.config.context_window :]
                factor = 1.0 / len(window) if window else 0.0
                for token_id in window:
                    row = grad_embed[token_id % vocab]
                    for j in range(hidden):
                        row[j] += grad_hidden[j] * factor

        divisor = max(1, count)
        return (
            total_loss / divisor,
            _scale(grad_embed, divisor), _scale(grad_hidden_w, divisor),
            _scale([grad_hidden_b], divisor)[0],
            _scale(grad_out_w, divisor), _scale([grad_out_b], divisor)[0],
            count,
        )

    def evaluate(
        self,
        samples: Sequence[Sequence[int]],
        embedding: list[list[float]],
        hidden_weights: list[list[float]],
        hidden_bias: Sequence[float],
        output_weights: list[list[float]],
        output_bias: Sequence[float],
    ) -> float:
        """Mean negative log-likelihood over a held-out set."""
        total = 0.0
        count = 0
        for sequence in samples:
            for position in range(self.config.context_window, len(sequence)):
                context = self._context_vector(sequence[:position], embedding, len(hidden_bias))
                _, _, log_probs = self._forward(
                    context, hidden_weights, hidden_bias, output_weights, output_bias
                )
                total -= log_probs[sequence[position]]
                count += 1
        return total / max(1, count)


def _scale(matrix: list[list[float]], divisor: float) -> list[list[float]]:
    return [[v / divisor for v in row] for row in matrix]


def _pack_weights(embedding, hidden_weights, hidden_bias, output_weights,
                  output_bias) -> dict:
    """Flatten the nested weight lists into the float64 buffers the C kernel
    updates in place.  Kept as plain arrays — zero marshalling per batch."""
    from array import array

    def flat(rows) -> array:
        buf: array = array("d")
        for row in rows:
            buf.extend(row)
        return buf

    return {
        "embedding": flat(embedding),
        "hidden_w": flat(hidden_weights),
        "hidden_b": array("d", hidden_bias),
        "out_w": flat(output_weights),
        "out_b": array("d", output_bias),
    }


def _unpack_weights(buf: dict, embedding, hidden_weights, hidden_bias,
                    output_weights, output_bias) -> None:
    """Write the kernel-updated buffers back into the live weight lists."""
    for target, source in ((embedding, buf["embedding"]),
                           (hidden_weights, buf["hidden_w"]),
                           (output_weights, buf["out_w"])):
        width = len(target[0])
        for i, row in enumerate(target):
            base = i * width
            row[:] = [source[base + j] for j in range(width)]
    hidden_bias[:] = list(buf["hidden_b"])
    output_bias[:] = list(buf["out_b"])


def _add(vector: list[float], gradient: Sequence[float], rate: float) -> None:
    for index, value in enumerate(gradient):
        vector[index] += rate * value


def _axpy(matrix: list[list[float]], gradient: list[list[float]], rate: float, l2: float) -> None:
    """In-place SGD step with weight decay."""
    for i, row in enumerate(matrix):
        grad_row = gradient[i]
        for j, value in enumerate(row):
            row[j] = value + rate * grad_row[j] - l2 * value
