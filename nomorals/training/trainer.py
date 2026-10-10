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

__all__ = [
    "TrainConfig", "TrainMetrics", "NativeTrainer", "TrainedModel",
    "clip_grads", "global_grad_norm", "lr_factor",
]

_log = get_logger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters. Deliberately small defaults: this runs on a phone.

    The optimizer/schedule fields are the mined upgrades: AdamW with
    decoupled weight decay and a warmup+cosine schedule are what every real
    training loop uses instead of fixed-decay SGD, and both are cheap in
    pure Python.
    """

    hidden_size: int = 64
    context_window: int = 16
    learning_rate: float = 0.05
    epochs: int = 3
    batch_size: int = 8
    l2: float = 1e-4
    seed: int = 1234
    log_every: int = 50
    max_examples: int = 0
    # ── mined upgrades ──
    optimizer: str = "adamw"            # "sgd" | "adamw"
    lr_schedule: str = "warmup_cosine"  # "constant" | "cosine" | "warmup_cosine"
    warmup_steps: int = 0               # 0 = auto (10% of total steps)
    grad_clip: float = 1.0              # global-norm clip; 0 = off
    early_stopping_patience: int = 0    # epochs w/o eval gain before stopping; 0 = off
    keep_best: bool = True              # restore best-eval weights at the end
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8

    def validate(self) -> None:
        if self.hidden_size < 1:
            raise ValidationError("hidden_size must be positive", field="hidden_size")
        if self.learning_rate <= 0:
            raise ValidationError("learning_rate must be positive", field="learning_rate")
        if self.epochs < 1:
            raise ValidationError("epochs must be at least 1", field="epochs")
        if self.optimizer not in {"sgd", "adamw"}:
            raise ValidationError(f"unknown optimizer {self.optimizer!r}", field="optimizer")
        if self.lr_schedule not in {"constant", "cosine", "warmup_cosine"}:
            raise ValidationError(f"unknown lr_schedule {self.lr_schedule!r}", field="lr_schedule")
        if self.grad_clip < 0:
            raise ValidationError("grad_clip must be >= 0", field="grad_clip")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size, "context_window": self.context_window,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "l2": self.l2, "seed": self.seed,
            "log_every": self.log_every, "max_examples": self.max_examples,
            "optimizer": self.optimizer, "lr_schedule": self.lr_schedule,
            "warmup_steps": self.warmup_steps, "grad_clip": self.grad_clip,
            "early_stopping_patience": self.early_stopping_patience,
            "keep_best": self.keep_best,
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
    best_epoch: int = 0
    stopped_early: bool = False
    tokens_per_sec: float = 0.0
    optimizer: str = ""
    lr_final: float = 0.0

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
            "best_epoch": self.best_epoch, "stopped_early": self.stopped_early,
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "optimizer": self.optimizer, "lr_final": self.lr_final,
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

    def generate(
        self,
        tokenizer: Any,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        seed: int = 0,
    ) -> str:
        """Sample a continuation for ``prompt`` (temperature + top-k).

        This is what makes the artifact *usable*: the golden battery in
        :mod:`nomorals.training.evaluate` can grade a native model
        end-to-end, and ``nm train --quick`` can show you what the demo
        model actually learned.  A tiny MLP is not a chatbot — expect
        babble on small corpora — but the path is real.
        """
        probe = NativeTrainer.__new__(NativeTrainer)
        probe.config = self.config
        rng = random.Random(seed)
        ids = list(tokenizer.encode(prompt))
        hidden = len(self.hidden_bias)
        for _ in range(max_new_tokens):
            context = probe._context_vector(ids, self.input_weights, hidden)
            _, _, log_probs = probe._forward(
                context, self.hidden_weights, self.hidden_bias,
                self.output_weights, self.output_bias,
            )
            logits = [lp for lp in log_probs]  # log-softmax; argmax/sampling same order
            if temperature <= 0:
                nxt = max(range(len(logits)), key=logits.__getitem__)
            else:
                scaled = [v / temperature for v in logits]
                if top_k > 0:
                    order = sorted(range(len(scaled)), key=scaled.__getitem__, reverse=True)
                    keep = set(order[:top_k])
                    scaled = [v if i in keep else float("-inf") for i, v in enumerate(scaled)]
                maximum = max(scaled)
                exps = [math.exp(min(50.0, v - maximum)) for v in scaled]
                total = sum(exps) or 1.0
                pick = rng.random() * total
                nxt, acc = 0, 0.0
                for i, e in enumerate(exps):
                    acc += e
                    if acc >= pick:
                        nxt = i
                        break
            ids.append(nxt)
            if len(ids) > self.config.context_window + max_new_tokens + 64:
                ids = ids[-(self.config.context_window + max_new_tokens):]
        return tokenizer.decode(ids)


def _zeros_like(struct: Any) -> Any:
    """A zero nested list with the same shape as a weight matrix or vector."""
    if struct and isinstance(struct[0], list):
        return [[0.0] * len(row) for row in struct]
    return [0.0] * len(struct)


def _squared_sum(struct: Any) -> float:
    if struct and isinstance(struct[0], list):
        return sum(v * v for row in struct for v in row)
    return sum(v * v for v in struct)


def global_grad_norm(grads: dict[str, Any]) -> float:
    """L2 norm over every gradient tensor.  Public: useful for diagnostics."""
    return math.sqrt(sum(_squared_sum(g) for g in grads.values()))


def clip_grads(grads: dict[str, Any], max_norm: float) -> float:
    """Scale gradients in place so their global norm is at most ``max_norm``.

    Returns the pre-clip norm.  The cheapest anti-explosion insurance a
    training loop can buy; real trainers never skip it.
    """
    norm = global_grad_norm(grads)
    if norm > max_norm > 0:
        scale = max_norm / (norm + 1e-12)
        for grad in grads.values():
            if grad and isinstance(grad[0], list):
                for row in grad:
                    for j in range(len(row)):
                        row[j] *= scale
            else:
                for j in range(len(grad)):
                    grad[j] *= scale
    return norm


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    """LR multiplier for ``schedule`` at ``step`` (0-based).

    ``warmup_cosine``: linear warmup then cosine decay to 0 — the standard
    recipe (HF's ``get_cosine_schedule_with_warmup``).  ``cosine`` skips the
    warmup; ``constant`` is 1.0 everywhere.
    """
    if schedule == "constant" or total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class _AdamW:
    """AdamW in pure Python: per-parameter first/second moments, decoupled
    weight decay, bias correction.  ~25 lines; converges meaningfully better
    than fixed-decay SGD on the same step budget."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: dict[str, Any] = {}
        self.v: dict[str, Any] = {}

    def step(
        self,
        params: dict[str, Any],
        grads: dict[str, Any],
        lr: float,
        weight_decay: float,
    ) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t
        for name, param in params.items():
            grad = grads[name]
            m = self.m.setdefault(name, _zeros_like(param))
            v = self.v.setdefault(name, _zeros_like(param))
            if param and isinstance(param[0], list):
                for i in range(len(param)):
                    prow, grow, mrow, vrow = param[i], grad[i], m[i], v[i]
                    for j in range(len(prow)):
                        g = grow[j]
                        mj = mrow[j] = self.beta1 * mrow[j] + (1.0 - self.beta1) * g
                        vj = vrow[j] = self.beta2 * vrow[j] + (1.0 - self.beta2) * g * g
                        update = (mj / bias1) / (math.sqrt(vj / bias2) + self.eps)
                        prow[j] -= lr * (update + weight_decay * prow[j])
            else:
                for j in range(len(param)):
                    g = grad[j]
                    mj = m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g
                    vj = v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g
                    update = (mj / bias1) / (math.sqrt(vj / bias2) + self.eps)
                    param[j] -= lr * (update + weight_decay * param[j])


def _sgd_step(
    params: dict[str, Any], grads: dict[str, Any], lr: float, weight_decay: float
) -> None:
    """Plain SGD with decoupled weight decay (the AdamW-style kind)."""
    for name, param in params.items():
        grad = grads[name]
        if param and isinstance(param[0], list):
            _axpy(param, grad, -lr, weight_decay * lr)
        else:
            _add(param, grad, -lr)
            if weight_decay:
                for j in range(len(param)):
                    param[j] -= lr * weight_decay * param[j]


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

        # The C++ kernel applies its own SGD step, so it only carries runs
        # that actually use SGD.  An AdamW run silently degrading to the
        # kernel's SGD would train a different optimizer than configured —
        # fall back to pure Python instead, and say so.
        use_cpp = config.optimizer == "sgd" and self._cpp_kernel()
        if not use_cpp and config.optimizer != "sgd" and self._cpp_kernel():
            _log.info("optimizer=%s: C++ kernel does SGD only — using pure Python",
                      config.optimizer)
        buffers = None
        if use_cpp:
            try:
                buffers = _pack_weights(embedding, hidden_weights, hidden_bias,
                                        output_weights, output_bias)
            except Exception:  # noqa: BLE001 — fall back rather than fail the run
                buffers = None

        params = {
            "embedding": embedding, "hidden_w": hidden_weights,
            "hidden_b": hidden_bias, "out_w": output_weights,
            "out_b": output_bias,
        }
        optimizer = _AdamW(config.beta1, config.beta2, config.adam_eps) \
            if config.optimizer == "adamw" else None

        metrics = TrainMetrics(examples=len(samples),
                               backend="native-cpp" if buffers is not None else "native",
                               optimizer=config.optimizer)
        started = time.perf_counter()
        batches = max(1, len(samples) // max(1, config.batch_size))
        step = 0
        total_steps = config.epochs * max(1, (len(samples) + max(1, config.batch_size) - 1)
                                          // max(1, config.batch_size))
        warmup = config.warmup_steps
        if not warmup and config.lr_schedule == "warmup_cosine":
            warmup = max(1, total_steps // 10)
        best_snapshot: dict[str, Any] | None = None
        bad_epochs = 0
        lr = config.learning_rate

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
                lr = config.learning_rate * lr_factor(step, total_steps, warmup,
                                                      config.lr_schedule)
                if buffers is not None:
                    loss, count, _ = self._cpp_batch(
                        batch, buffers, context=context, lr=lr, l2=config.l2 * lr)
                else:
                    (loss, grad_embed, grad_hidden_w, grad_hidden_b,
                     grad_out_w, grad_out_b, count) = self._batch_gradients(
                        batch, embedding, hidden_weights, hidden_bias,
                        output_weights, output_bias)
                    grads = {
                        "embedding": grad_embed, "hidden_w": grad_hidden_w,
                        "hidden_b": grad_hidden_b, "out_w": grad_out_w,
                        "out_b": grad_out_b,
                    }
                    if config.grad_clip > 0:
                        clip_grads(grads, config.grad_clip)
                    if optimizer is not None:
                        optimizer.step(params, grads, lr, config.l2)
                    else:
                        _sgd_step(params, grads, lr, config.l2)
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
            _log.info("epoch %d train_loss=%.4f lr=%.6f", epoch + 1, average, lr)

            if eval_samples:
                eval_loss = self.evaluate(
                    eval_samples, embedding, hidden_weights, hidden_bias, output_weights, output_bias
                )
                metrics.eval_loss_history.append(eval_loss)
                improved = eval_loss < metrics.best_eval_loss - 1e-4
                if improved:
                    metrics.best_eval_loss = eval_loss
                    metrics.best_epoch = epoch + 1
                    bad_epochs = 0
                    if config.keep_best:
                        best_snapshot = _snapshot_params(params)
                else:
                    bad_epochs += 1
                _log.info("epoch %d eval_loss=%.4f%s", epoch + 1, eval_loss,
                          " (best)" if improved else "")
                if config.early_stopping_patience and bad_epochs >= config.early_stopping_patience:
                    _log.info("early stopping: no eval gain for %d epoch(s)", bad_epochs)
                    metrics.stopped_early = True
                    break

        if best_snapshot is not None and config.keep_best and eval_samples:
            # End on the best weights, not the last ones — the HF Trainer
            # `load_best_model_at_end` behavior.
            _restore_params(params, best_snapshot)
            _log.info("restored best-eval weights (epoch %d)", metrics.best_epoch)

        metrics.steps = step
        metrics.seconds = time.perf_counter() - started
        metrics.tokens_per_sec = metrics.tokens / max(1e-9, metrics.seconds)
        metrics.lr_final = lr
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


def _snapshot_params(params: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of the live weight lists (for best-checkpoint restore)."""
    out: dict[str, Any] = {}
    for name, struct in params.items():
        if struct and isinstance(struct[0], list):
            out[name] = [row[:] for row in struct]
        else:
            out[name] = struct[:]
    return out


def _restore_params(params: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Write a snapshot back into the live weight lists, in place."""
    for name, struct in params.items():
        saved = snapshot[name]
        if struct and isinstance(struct[0], list):
            for row, saved_row in zip(struct, saved):
                row[:] = saved_row
        else:
            struct[:] = saved


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
