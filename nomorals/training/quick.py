"""`nm train --quick`: the one-command phone training demo.

A real, end-to-end run of the native model trainer with zero setup:

1. **Dataset** — the bundled seed corpus plus every real conversation pair
   the bot has collected since running (`conversations.jsonl`), so the more
   the bot is used, the more the demo trains on HER actual voice.
2. **Tokenizer** — BPE trained on the corpus (C++ kernel when built).
3. **Training** — the native next-token MLP (C++ kernel when built),
   phone-sized by default: hidden 64, context 16, 3 epochs.
4. **Evaluation** — before/after loss on a held-out split, so the summary
   says what the model actually learned.
5. **Artifact** — ``<training dir>/quick-<ulid>/`` with model.json +
   tokenizer.json + metrics.json, loadable by ``TrainedModel.load``.

One command, a couple of minutes on a phone, nothing left unexplained.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..core.ids import new_id
from .dataset import Example, Turn
from .tokenize import BPETokenizer
from .trainer import NativeTrainer, TrainConfig

__all__ = ["QuickResult", "collect_quick_dataset", "run_quick_training"]


@dataclass
class QuickResult:
    ok: bool
    dataset: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    loss_before: float = 0.0
    loss_after: float = 0.0
    seconds: float = 0.0
    artifact_dir: str = ""
    backend: str = "pure-python"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dataset": self.dataset,
            "config": self.config,
            "loss_before": round(self.loss_before, 5),
            "loss_after": round(self.loss_after, 5),
            "seconds": round(self.seconds, 2),
            "artifact_dir": self.artifact_dir,
            "backend": self.backend,
            "note": self.note,
        }


def _examples_from_pair(user: str, assistant: str) -> Example | None:
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    if len(user) < 8 or len(assistant) < 8:
        return None
    if len(user) > 1500 or len(assistant) > 1500:
        return None
    return Example(turns=[Turn(role="user", content=user),
                          Turn(role="assistant", content=assistant)],
                   source="quick")


def collect_quick_dataset(context: Any,
                          max_pairs: int = 240) -> list[Example]:
    """Seed corpus + real collected conversation pairs, deduped."""
    examples: list[Example] = []
    seen: set[str] = set()

    def add(user: str, assistant: str, source: str) -> None:
        if len(examples) >= max_pairs:
            return
        ex = _examples_from_pair(user, assistant)
        if ex is None:
            return
        key = user.lower()[:80]
        if key in seen:
            return
        seen.add(key)
        ex.source = source
        examples.append(ex)

    # 1. the bundled seed (works on a phone with zero prior use)
    try:
        seed = Path(__file__).resolve().parent / "seeds" / "codebeast_seed.jsonl"
        if seed.exists():
            for line in seed.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                msgs = data.get("messages") or []
                users = [m for m in msgs if m.get("role") == "user"]
                assists = [m for m in msgs if m.get("role") == "assistant"]
                if users and assists:
                    add(users[0].get("content", ""),
                        assists[0].get("content", ""), "seed")
    except OSError:
        pass

    # 2. REAL pairs collected by the running bot (its actual voice)
    try:
        data_dir = Path(context.settings.resolve(
            context.settings.training.data_dir))
        conv = data_dir / "conversations.jsonl"
        if conv.exists():
            for line in conv.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                add(str(rec.get("user", "")), str(rec.get("assistant", "")),
                    "collected")
    except (OSError, AttributeError):
        pass

    return examples


def run_quick_training(context: Any, *,
                       epochs: int = 3,
                       hidden: int = 64,
                       ctx: int = 16,
                       max_pairs: int = 240) -> QuickResult:
    """The whole demo.  Returns a QuickResult; ok=False carries a note."""
    from .. import native

    examples = collect_quick_dataset(context, max_pairs=max_pairs)
    if len(examples) < 12:
        return QuickResult(
            ok=False,
            note=(f"need at least 12 usable examples, found {len(examples)} — "
                  "run the bot for a while (real pairs are collected "
                  "automatically) or check the seed corpus"),
        )

    # held-out split for an honest before/after
    eval_examples = examples[-max(2, len(examples) // 10):]
    train_examples = examples[: len(examples) - len(eval_examples)]

    texts = [e.to_chatml() for e in train_examples]
    eval_texts = [e.to_chatml() for e in eval_examples]

    # 1. tokenizer (C++ kernel when built)
    t0 = time.perf_counter()
    tok = BPETokenizer.train(texts, vocab_size=300, min_frequency=1)
    tok_s = time.perf_counter() - t0

    # 2. trainer
    config = TrainConfig(hidden_size=hidden, context_window=ctx,
                         epochs=epochs, batch_size=8, learning_rate=0.05,
                         seed=1234)
    trainer = NativeTrainer(tok, config)
    use_cpp = trainer._cpp_kernel() if hasattr(trainer, "_cpp_kernel") \
        else False
    backend = "native-cpp" if use_cpp else "pure-python"

    def _loss(trainer_: NativeTrainer, sample_texts: Sequence[str],
              weights: tuple) -> float:
        seqs = [tok.encode(t) for t in sample_texts]
        seqs = [s for s in seqs if len(s) > ctx]
        if not seqs:
            return 0.0
        return trainer_.evaluate(
            seqs, weights[0], weights[1], weights[2], weights[3], weights[4])

    # 3. BEFORE: loss of the freshly-initialized model.  Same seed → the
    # RNG draws are identical → these are exactly the weights the real run
    # started from; one epoch of lr≈0 (no meaningful updates) then evaluate.
    fresh = NativeTrainer(tok, TrainConfig(
        hidden_size=hidden, context_window=ctx, epochs=1, batch_size=8,
        learning_rate=1e-9, seed=1234))
    model0, _ = fresh.fit([train_examples[0]], [])
    before = _loss(
        fresh, eval_texts,
        (model0.input_weights, model0.hidden_weights, model0.hidden_bias,
         model0.output_weights, model0.output_bias))

    # 4. the real run
    model, metrics = trainer.fit(train_examples, eval_examples)
    after_s = time.perf_counter() - t0

    # 4. artifact
    out_root = Path(context.settings.resolve(
        context.settings.training.data_dir))
    artifact = out_root / f"quick-{new_id()}"
    artifact.mkdir(parents=True, exist_ok=True)
    tok.save(artifact / "tokenizer.json")
    model.tokenizer_path = str(artifact / "tokenizer.json")
    model.save(artifact)
    (artifact / "metrics.json").write_text(
        json.dumps({
            "config": config.to_dict(),
            "loss_before": before,
            "loss_after": metrics.best_eval_loss
            if metrics.best_eval_loss != float("inf")
            else metrics.final_loss,
            "seconds": after_s,
            "bpe_train_seconds": tok_s,
            "backend": backend,
            "dataset": {
                "examples": len(examples),
                "train": len(train_examples),
                "eval": len(eval_examples),
                "collected": sum(1 for e in examples
                                 if e.source == "collected"),
                "seed": sum(1 for e in examples if e.source == "seed"),
            },
        }, indent=2),
        encoding="utf-8",
    )

    after = metrics.best_eval_loss if metrics.best_eval_loss != float("inf") \
        else metrics.final_loss
    return QuickResult(
        ok=True,
        dataset={
            "examples": len(examples),
            "train": len(train_examples),
            "eval": len(eval_examples),
            "collected": sum(1 for e in examples if e.source == "collected"),
            "seed": sum(1 for e in examples if e.source == "seed"),
        },
        config=config.to_dict(),
        loss_before=before,
        loss_after=after,
        seconds=after_s,
        artifact_dir=str(artifact),
        backend=backend,
    )
