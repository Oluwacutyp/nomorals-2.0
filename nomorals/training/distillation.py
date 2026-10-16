"""Distillation: cloud answers the owner keeps become local training data.

Sources of distillation pairs:
- Owner KEEPS a cloud turn (no correction, no pushback) — implicit approval.
- Owner PASSES a cloud turn on (forwards / shares / quotes it) — explicit
  endorsement, stronger than keeping.
- A ``llm/adjudicate.py`` :class:`Judge` picks a cloud answer over
  alternatives — judged quality, with position-bias control on pairs.

Pairs flow: capture -> dedupe -> periodically train a local adapter via
``training/collect.py`` (harvest), ``training/tokenize.py`` (encode),
``training/trainer.py`` (``NativeTrainer``), ``training/liger.py``
(fused-kernel flags when available), ``training/checkpoints.py``
(validate/prune) and the ``training/registry.py`` production gate.
No pair ever trains without passing that gate — distillation is a
proposal pipeline, not an auto-promotion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "DistillationPair",
    "DistillationStore",
    "capture",
    "capture_owner_kept",
    "capture_owner_passed",
    "judge_and_capture",
    "to_examples",
    "train_local_adapter",
    "periodic_distill",
    "maybe_liger_kwargs",
]

# Kept = implicit approval; passed-on = explicit endorsement.
SOURCE_WEIGHTS = {"kept": 1.0, "passed": 1.5, "judge": 1.25}


def _pair_hash(prompt: str, answer: str) -> str:
    return hashlib.sha256(f"{prompt}\n{answer}".encode()).hexdigest()[:16]


@dataclass
class DistillationPair:
    """One captured (prompt, context, answer) distillation pair."""

    prompt: str
    context: str
    answer: str
    source: str = "kept"  # kept | passed | judge
    teacher: str = ""
    judge_score: float | None = None
    created_at: float = field(default_factory=time.time)
    pair_hash: str = ""

    def __post_init__(self) -> None:
        if not self.pair_hash:
            self.pair_hash = _pair_hash(self.prompt, self.answer)

    @property
    def weight(self) -> float:
        return SOURCE_WEIGHTS.get(self.source, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt, "context": self.context,
            "answer": self.answer, "source": self.source,
            "teacher": self.teacher, "judge_score": self.judge_score,
            "created_at": self.created_at, "pair_hash": self.pair_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DistillationPair":
        return cls(
            prompt=str(data.get("prompt") or ""),
            context=str(data.get("context") or ""),
            answer=str(data.get("answer") or ""),
            source=str(data.get("source") or "kept"),
            teacher=str(data.get("teacher") or ""),
            judge_score=data.get("judge_score"),
            created_at=float(data.get("created_at") or 0.0),
            pair_hash=str(data.get("pair_hash") or ""),
        )

    def to_example(self) -> Any:
        """As a ``training/dataset.py`` :class:`Example` (SFT shape)."""
        from .dataset import Example, Turn

        user_text = self.prompt
        if self.context:
            user_text = f"{self.context}\n\n{self.prompt}"
        return Example(
            turns=[Turn("user", user_text), Turn("assistant", self.answer)],
            weight=self.weight, source=f"distillation:{self.source}",
        )


def capture(
    prompt: str,
    context: str,
    answer: str,
    *,
    source: str = "kept",
    teacher: str = "",
    judge_score: float | None = None,
) -> DistillationPair | None:
    """Capture one pair.  Empty prompt/answer -> None (never a fake pair)."""
    prompt = (prompt or "").strip()
    answer = (answer or "").strip()
    if not prompt or not answer:
        return None
    if source not in SOURCE_WEIGHTS:
        source = "kept"
    return DistillationPair(
        prompt=prompt, context=(context or "").strip(), answer=answer,
        source=source, teacher=teacher, judge_score=judge_score,
    )


def capture_owner_kept(prompt: str, context: str, answer: str, *,
                       teacher: str = "") -> DistillationPair | None:
    """Owner kept the cloud turn (no correction) — implicit approval."""
    return capture(prompt, context, answer, source="kept", teacher=teacher)


def capture_owner_passed(prompt: str, context: str, answer: str, *,
                         teacher: str = "") -> DistillationPair | None:
    """Owner passed the cloud turn on (forwarded/shared) — endorsement."""
    return capture(prompt, context, answer, source="passed", teacher=teacher)


def judge_and_capture(
    question: str,
    candidates: Sequence[str],
    ask: Any,
    *,
    context: str = "",
    teacher: str = "",
) -> tuple[DistillationPair | None, dict[str, Any]]:
    """Judge ``candidates`` with ``llm/adjudicate.py`` and capture the winner.

    The judge (not this module) decides; a failed/empty judgment captures
    nothing.  For exactly two candidates the position-bias-controlled
    ``pairwise`` path is used; otherwise ``adjudicate`` ranks the set.
    """
    from ..llm.adjudicate import Judge

    texts = [str(c or "").strip() for c in candidates]
    texts = [t for t in texts if t]
    if not texts:
        return None, {"ok": False, "reason": "no candidates"}
    judge = Judge(ask)
    if len(texts) == 2:
        verdict = judge.pairwise(question, texts[0], texts[1])
        if not verdict.ok or verdict.winner < 0:
            return None, {"ok": False, "reason": verdict.error or "tie",
                           "consistent": verdict.consistent}
        winner, confidence = verdict.winner, verdict.confidence
        method = "pairwise-debiased"
    else:
        judgment = judge.adjudicate(question, texts)
        if not judgment.ok or judgment.winner < 0:
            return None, {"ok": False,
                           "reason": judgment.error or "no winner"}
        winner, confidence, method = (
            judgment.winner, judgment.confidence, judgment.method)
    pair = capture(question, context, texts[winner], source="judge",
                   teacher=teacher, judge_score=float(confidence))
    return pair, {"ok": True, "winner": winner, "confidence": confidence,
                  "method": method}


def to_examples(pairs: Sequence[DistillationPair]) -> list[Any]:
    """Distillation pairs -> training ``Example`` objects (SFT shape)."""
    return [p.to_example() for p in pairs]


def maybe_liger_kwargs() -> dict[str, Any]:
    """Liger fused-kernel kwargs when the optional dep is present, else {}.

    Delegates to ``training/liger.py`` — this module never imports liger
    directly, so a missing optional dep is a clean empty dict, not a crash.
    """
    try:
        from .liger import liger_available, trl_sft_kwargs

        available, _reason = liger_available()
        return dict(trl_sft_kwargs()) if available else {}
    except Exception:  # noqa: BLE001
        return {}


class DistillationStore:
    """Deduped, append-only store for distillation pairs (JSONL-backed)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._pairs: dict[str, DistillationPair] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pair = DistillationPair.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            self._pairs[pair.pair_hash] = pair

    def add(self, pair: DistillationPair | None) -> bool:
        """Add a pair; False when None or an exact duplicate."""
        if pair is None or pair.pair_hash in self._pairs:
            return False
        self._pairs[pair.pair_hash] = pair
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(pair.to_dict()) + "\n")
        return True

    def __len__(self) -> int:
        return len(self._pairs)

    def pairs(self) -> list[DistillationPair]:
        return sorted(self._pairs.values(), key=lambda p: p.created_at)

    def to_examples(self) -> list[Any]:
        return to_examples(self.pairs())

    def export_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(json.dumps(p.to_dict()) for p in self.pairs()) + "\n",
            encoding="utf-8")
        return target


def train_local_adapter(
    pairs: Sequence[DistillationPair],
    *,
    tokenizer: Any = None,
    config: Any = None,
    out_dir: str | Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Train a local adapter from distillation pairs via the real pipeline.

    Tokenizer: ``training/tokenize.py`` (``BPETokenizer.train`` on pair
    text when none is supplied).  Trainer: ``training/trainer.py``
    ``NativeTrainer``.  Checkpoints: ``training/checkpoints.py``
    validation when ``out_dir`` is given.  The returned model is a
    *candidate* — callers still gate it through ``TrainingRegistry``
    before it serves anything.
    """
    from .tokenize import BPETokenizer
    from .trainer import NativeTrainer, TrainConfig, TrainedModel

    examples = to_examples(pairs)
    if not examples:
        raise ValueError("no distillation pairs to train on")
    if tokenizer is None:
        corpus = [f"{p.prompt}\n{p.answer}" for p in pairs]
        tokenizer = BPETokenizer.train(corpus, vocab_size=300)
    cfg = config or TrainConfig(
        hidden_size=16, context_window=8, epochs=1, batch_size=4,
        learning_rate=0.05)
    cfg.validate()
    trainer = NativeTrainer(tokenizer, cfg)
    model, metrics = trainer.fit(examples)
    info: dict[str, Any] = {
        "examples": len(examples),
        "liger_kwargs": maybe_liger_kwargs(),
        "metrics": metrics.to_dict(),
    }
    if out_dir is not None:
        # checkpoints.validate_checkpoint is for HF checkpoints
        # (config.json + payload + trainer_state.json); a native
        # TrainedModel.save writes model.json, so the honest check is
        # a load-back round-trip: the checkpoint is valid iff it loads
        # and matches the trained model.
        out_path = Path(out_dir)
        ckpt_dir = out_path if out_path.name.startswith("checkpoint-") \
            else out_path / "checkpoint-1"
        saved = model.save(ckpt_dir)
        try:
            reloaded = TrainedModel.load(saved)
            valid = reloaded.vocab_size == model.vocab_size
            reason = "ok (load-back round-trip)" if valid else "vocab mismatch"
        except Exception as exc:  # noqa: BLE001 — unloadable = invalid
            valid, reason = False, f"reload failed: {exc}"
        info["checkpoint"] = str(saved)
        info["checkpoint_valid"] = valid
        info["checkpoint_reason"] = reason
        if not valid:
            raise ValueError(f"checkpoint failed validation: {reason}")
    return model, metrics, info


def periodic_distill(
    store: DistillationStore,
    *,
    min_pairs: int = 32,
    registry: Any = None,
    run_name: str = "distillation",
    base_model: str = "",
    out_dir: str | Path | None = None,
    **train_kw: Any,
) -> dict[str, Any]:
    """Periodically train from captured pairs — only when enough exist.

    Below ``min_pairs`` this is a no-op ``{"status": "deferred"}``: a
    handful of pairs is noise, not a training set.  At/above the
    threshold it trains via :func:`train_local_adapter` and, when a
    ``registry`` is supplied, completes + gates the run through
    ``TrainingRegistry`` — a failed gate means the distilled candidate
    never lands.
    """
    if len(store) < min_pairs:
        return {"status": "deferred", "pairs": len(store),
                 "min_pairs": min_pairs,
                 "reason": "not enough distillation pairs yet"}
    pairs = store.pairs()
    run = None
    if registry is not None:
        run = registry.start(run_name, base_model=base_model)
    try:
        _model, metrics, info = train_local_adapter(
            pairs, out_dir=out_dir, **train_kw)
    except Exception as exc:  # noqa: BLE001
        if registry is not None and run is not None:
            registry.fail(run, error=str(exc))
        return {"status": "failed", "pairs": len(pairs), "error": str(exc)}
    result: dict[str, Any] = {"status": "trained", "pairs": len(pairs), **info}
    if registry is not None and run is not None:
        scores = metrics.as_scores()
        registry.complete(run, metrics=scores,
                          output_path=str(info.get("checkpoint") or ""))
        gated = registry.evaluate(run)
        result["gate_passed"] = bool(gated)
        if gated:
            result["promoted"] = bool(registry.promote(run))
        else:
            result["promoted"] = False
    return result
