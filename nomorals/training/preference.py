"""Preference learning from the owner's own corrections.

Pipeline (all pieces already exist except this glue):

    ``cognition/representation.py`` ledger
        -- disapproval (-1) hits harder than approval (+1) --
    -> :func:`build_preference_pairs` (pairwise preference pairs)
    -> :class:`PreferenceAdapter` (LoRA-class adapter: a small, servable
       delta — vocab-preference weights + distilled lessons — NOT full
       model weights; weight-level LoRA export stays with
       ``training/backends`` when a GPU backend is available)
    -> ``training/registry.py`` production gate
       (:class:`TrainingRegistry.evaluate` / :meth:`promote`)
    -> ``llm/brain.py`` serves the tuned adapter for owner-directed work
       (``Brain.preference_adapter`` / ``set_preference_adapter``)

The loop MUST reject its own update.  The held-out set is the owner's
ACTUAL past corrections (chronological split — the most recent
corrections are the test, so there is no leakage from training on the
future).  A candidate adapter that does not beat the empty-adapter
baseline on held-out corrections is rejected, however well it fits
training pairs.  Without that reject, a bad update is worse than none:
it would bake the owner's most recent corrections into a model that
scores worse on exactly those corrections.

Preference model (honest, CPU-servable): a linear bag-of-words
preference model.  Each training pair moves weight toward tokens in the
chosen text and away from tokens in the rejected text, scaled by the
pair weight (disapproval pairs count ``DISAPPROVAL_WEIGHT``).  The
DPO-style loss is ``-log sigmoid(beta * (score(chosen) - score(rejected)))``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "APPROVAL_WEIGHT",
    "DISAPPROVAL_WEIGHT",
    "PreferencePair",
    "PreferenceAdapter",
    "build_preference_pairs",
    "split_held_out",
    "dpo_loss",
    "pairwise_loss",
    "train_preference_adapter",
    "evaluate_on_held_out",
    "production_gate",
    "gate_with_registry",
    "lessons_for",
    "serving_config",
]

#: Disapproval (-1) hits harder than approval (+1) helps.  The ledger
#: itself scores a disapproved "success" down by a flat 0.5; here the
#: asymmetry is a 2:1 pair weight, so one owner correction outweighs
#: two approvals when the adapter learns what to prefer.
DISAPPROVAL_WEIGHT = 2.0
APPROVAL_WEIGHT = 1.0
#: Minimum held-out pairs before any update may land.  Fewer than this
#: and the gate cannot tell a good update from noise — it rejects.
MIN_HELD_OUT = 3
#: Default bar: held-out win rate must clear chance (0.5) by this margin.
DEFAULT_MIN_WIN_RATE = 0.6
DEFAULT_MIN_IMPROVEMENT = 0.0

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _pair_id(prompt: str, chosen: str, rejected: str) -> str:
    raw = f"{prompt}\n{chosen}\n{rejected}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class PreferencePair:
    """One pairwise preference: ``chosen`` beats ``rejected`` for ``prompt``."""

    prompt: str
    chosen: str
    rejected: str
    source: str = "owner_correction"  # owner_correction | counterfactual | approved
    weight: float = DISAPPROVAL_WEIGHT
    action_type: str = ""
    created_at: float = field(default_factory=time.time)
    pair_id: str = ""

    def __post_init__(self) -> None:
        if not self.pair_id:
            self.pair_id = _pair_id(self.prompt, self.chosen, self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt, "chosen": self.chosen,
            "rejected": self.rejected, "source": self.source,
            "weight": self.weight, "action_type": self.action_type,
            "created_at": self.created_at, "pair_id": self.pair_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferencePair":
        return cls(
            prompt=str(data.get("prompt") or ""),
            chosen=str(data.get("chosen") or ""),
            rejected=str(data.get("rejected") or ""),
            source=str(data.get("source") or "owner_correction"),
            weight=float(data.get("weight") or DISAPPROVAL_WEIGHT),
            action_type=str(data.get("action_type") or ""),
            created_at=float(data.get("created_at") or 0.0),
            pair_id=str(data.get("pair_id") or ""),
        )


def build_preference_pairs(
    ledger: Any,
    *,
    since_days: float = 90.0,
    limit: int = 500,
) -> tuple[list[PreferencePair], dict[str, Any]]:
    """Build pairwise pairs from the representation ledger.

    Sources, in priority order:
    1. Owner corrections (``owner_feedback == -1`` with ``feedback_text``):
       rejected = what Devon did, chosen = what the owner said instead.
       Weight ``DISAPPROVAL_WEIGHT``.
    2. Counterfactuals attached to a disapproved / failed record:
       rejected = what Devon did, chosen = the recorded better action.
    3. Failed records with a counterfactual but no owner text: same as (2).

    Approved records carry no rejected alternative, so they are counted
    (``approved_skipped``) but never fabricated into pairs — a pair
    needs a real rejected side.
    """
    pairs: list[PreferencePair] = []
    stats = {"corrections": 0, "counterfactuals": 0, "approved_skipped": 0,
             "empty_skipped": 0}
    try:
        records = ledger.recent(limit=limit, since_days=since_days)
    except Exception:  # noqa: BLE001 — ledger read failure -> no pairs, not a crash
        _log.debug("preference: ledger read failed", exc_info=True)
        return [], {**stats, "error": "ledger read failed"}
    counterfactuals: dict[str, list[dict[str, Any]]] = {}
    try:
        for rec in records:
            for cf in ledger._counterfactuals_for(rec.id):
                counterfactuals.setdefault(rec.id, []).append(cf)
    except Exception:  # noqa: BLE001
        pass
    for rec in records:
        desc = (getattr(rec, "description", "") or "").strip()
        feedback = (getattr(rec, "feedback_text", "") or "").strip()
        owner_fb = getattr(rec, "owner_feedback", None)
        outcome = getattr(rec, "outcome", "") or ""
        atype = getattr(rec, "action_type", "") or ""
        created = float(getattr(rec, "created_at", 0.0) or 0.0)
        cfs = counterfactuals.get(getattr(rec, "id", ""), [])
        better = ""
        if cfs:
            better = str(cfs[0].get("better_action") or "").strip()
        if owner_fb == -1 and feedback and desc:
            pairs.append(PreferencePair(
                prompt=f"Act on the owner's behalf ({atype}): {desc}",
                chosen=feedback, rejected=desc,
                source="owner_correction", weight=DISAPPROVAL_WEIGHT,
                action_type=atype, created_at=created))
            stats["corrections"] += 1
        elif better and desc and (owner_fb == -1 or outcome in ("failed", "partial")):
            pairs.append(PreferencePair(
                prompt=f"Act on the owner's behalf ({atype}): {desc}",
                chosen=better, rejected=desc,
                source="counterfactual", weight=DISAPPROVAL_WEIGHT,
                action_type=atype, created_at=created))
            stats["counterfactuals"] += 1
        elif owner_fb == 1:
            stats["approved_skipped"] += 1
        else:
            stats["empty_skipped"] += 1
    # Dedupe on pair id, keep first.
    seen: set[str] = set()
    deduped: list[PreferencePair] = []
    for pair in pairs:
        if pair.pair_id in seen:
            continue
        seen.add(pair.pair_id)
        deduped.append(pair)
    stats["pairs"] = len(deduped)
    return deduped, stats


def split_held_out(
    pairs: Sequence[PreferencePair],
    *,
    held_out_fraction: float = 0.2,
) -> tuple[list[PreferencePair], list[PreferencePair]]:
    """Chronological split: the most recent corrections are the held-out set.

    Chronological (not random) so the gate tests on the owner's LATEST
    corrections — the ones the adapter has never seen and that best
    predict the next correction.  Training on the future to predict the
    past would be leakage.
    """
    ordered = sorted(pairs, key=lambda p: (p.created_at, p.pair_id))
    n_held = int(len(ordered) * held_out_fraction)
    n_held = max(n_held, 1) if len(ordered) >= MIN_HELD_OUT else 0
    if n_held <= 0:
        return list(ordered), []
    return ordered[:-n_held], ordered[-n_held:]


def dpo_loss(chosen_score: float, rejected_score: float, *, beta: float = 1.0) -> float:
    """DPO-style pairwise loss for one pair: ``-log sigmoid(beta * margin)``."""
    margin = beta * (chosen_score - rejected_score)
    if margin >= 0:
        return float(math.log1p(math.exp(-margin)))
    return float(-margin + math.log1p(math.exp(margin)))


def pairwise_loss(
    pairs: Sequence[PreferencePair],
    adapter: "PreferenceAdapter",
    *,
    beta: float = 1.0,
) -> float:
    """Mean DPO loss of ``adapter`` over ``pairs`` (lower is better)."""
    if not pairs:
        return float("inf")
    total = 0.0
    for pair in pairs:
        total += dpo_loss(
            adapter.score(pair.prompt, pair.chosen),
            adapter.score(pair.prompt, pair.rejected), beta=beta)
    return total / len(pairs)


@dataclass
class PreferenceAdapter:
    """LoRA-class adapter: a small servable delta over the base model.

    ``token_weights`` is the learned preference direction (chosen tokens
    up, rejected tokens down); ``lessons`` are the human-readable
    distilled rules from the same pairs (from
    ``RepresentationLedger.distill``); ``action_weights`` is the per-type
    preference strength.  The base model weights are never touched —
    serving adds this delta as prompt steering + scoring, and the
    weight-level LoRA export path stays in ``training/backends``.
    """

    base_model: str = ""
    name: str = "preference-v1"
    token_weights: dict[str, float] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)
    action_weights: dict[str, float] = field(default_factory=dict)
    pair_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def score(self, prompt: str, candidate: str) -> float:
        """Preference score: weighted token overlap + lesson overlap.

        Prompt tokens are ignored (they appear in both sides); only the
        candidate text is scored, so the comparison is apples-to-apples.
        """
        total = 0.0
        for tok in _tokens(candidate):
            total += self.token_weights.get(tok, 0.0)
        cand = set(_tokens(candidate))
        for lesson in self.lessons:
            overlap = len(cand & set(_tokens(lesson)))
            if overlap:
                total += 0.25 * overlap
        return total

    def prefers(self, prompt: str, chosen: str, rejected: str) -> bool:
        return self.score(prompt, chosen) > self.score(prompt, rejected)

    def win_rate(self, pairs: Sequence[PreferencePair]) -> float:
        """Fraction of pairs where chosen outscores rejected (ties = 0.5)."""
        if not pairs:
            return 0.0
        total = 0.0
        for pair in pairs:
            s_c = self.score(pair.prompt, pair.chosen)
            s_r = self.score(pair.prompt, pair.rejected)
            total += 1.0 if s_c > s_r else (0.5 if s_c == s_r else 0.0)
        return total / len(pairs)

    def lora_config(self) -> dict[str, Any]:
        """Adapter descriptor in LoRA terms (rank = delta size class)."""
        return {
            "peft_type": "PREFERENCE_DELTA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["preference_delta"],
            "base_model": self.base_model,
            "trainable_params": len(self.token_weights),
            "note": ("CPU-servable preference delta (vocab weights + lessons). "
                     "Weight-level LoRA export via training/backends when a "
                     "GPU backend is available; this file never rewrites "
                     "base weights."),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model, "name": self.name,
            "token_weights": dict(self.token_weights),
            "lessons": list(self.lessons),
            "action_weights": dict(self.action_weights),
            "pair_count": self.pair_count, "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "lora_config": self.lora_config(),
        }

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "preference_adapter.json"
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PreferenceAdapter":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            base_model=str(data.get("base_model") or ""),
            name=str(data.get("name") or "preference-v1"),
            token_weights={str(k): float(v)
                           for k, v in (data.get("token_weights") or {}).items()},
            lessons=[str(x) for x in (data.get("lessons") or [])],
            action_weights={str(k): float(v)
                            for k, v in (data.get("action_weights") or {}).items()},
            pair_count=int(data.get("pair_count") or 0),
            created_at=float(data.get("created_at") or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def empty(cls, base_model: str = "") -> "PreferenceAdapter":
        """The baseline: no preference direction (every pair ties at 0.5)."""
        return cls(base_model=base_model, name="preference-baseline")


def train_preference_adapter(
    pairs: Sequence[PreferencePair],
    *,
    base_model: str = "",
    name: str = "preference-v1",
    lessons: Sequence[str] = (),
) -> PreferenceAdapter:
    """Learn the preference delta from ``pairs`` (no base-weight updates)."""
    token_weights: dict[str, float] = {}
    action_weights: dict[str, float] = {}
    for pair in pairs:
        w = float(pair.weight or 1.0)
        for tok in set(_tokens(pair.chosen)):
            token_weights[tok] = token_weights.get(tok, 0.0) + w
        for tok in set(_tokens(pair.rejected)):
            token_weights[tok] = token_weights.get(tok, 0.0) - w
        if pair.action_type:
            action_weights[pair.action_type] = (
                action_weights.get(pair.action_type, 0.0) + w)
    return PreferenceAdapter(
        base_model=base_model, name=name,
        token_weights=token_weights, lessons=[str(x) for x in lessons],
        action_weights=action_weights, pair_count=len(pairs),
        metadata={"sources": sorted({p.source for p in pairs})},
    )


def evaluate_on_held_out(
    adapter: PreferenceAdapter,
    held_out: Sequence[PreferencePair],
    *,
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
) -> dict[str, Any]:
    """Held-out verdict: the adapter must beat the empty baseline.

    Returns the report; ``passed`` is False when the held-out set is too
    small to judge (``MIN_HELD_OUT``) — too little evidence is a reject,
    not a pass.
    """
    baseline = PreferenceAdapter.empty(adapter.base_model)
    adapter_rate = adapter.win_rate(held_out)
    baseline_rate = baseline.win_rate(held_out)
    improvement = adapter_rate - baseline_rate
    reasons: list[str] = []
    if len(held_out) < MIN_HELD_OUT:
        reasons.append(
            f"held-out too small ({len(held_out)} < {MIN_HELD_OUT}) — "
            "cannot tell a good update from noise")
    if adapter_rate < min_win_rate:
        reasons.append(
            f"held-out win rate {adapter_rate:.3f} < {min_win_rate:.3f}")
    if improvement < min_improvement:
        reasons.append(
            f"improvement over baseline {improvement:.3f} < {min_improvement:.3f}")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "held_out_n": len(held_out),
        "adapter_win_rate": round(adapter_rate, 4),
        "baseline_win_rate": round(baseline_rate, 4),
        "improvement": round(improvement, 4),
        "adapter_train_loss": None,
        "min_win_rate": min_win_rate,
        "min_improvement": min_improvement,
    }


def production_gate(
    adapter: PreferenceAdapter,
    held_out: Sequence[PreferencePair],
    *,
    train_pairs: Sequence[PreferencePair] = (),
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
) -> tuple[bool, dict[str, Any]]:
    """The reject gate: held-out first, training fit only as context.

    A candidate that fits training pairs but fails held-out corrections
    is rejected — that is the whole point.  Returns ``(passed, report)``.
    """
    report = evaluate_on_held_out(
        adapter, held_out,
        min_win_rate=min_win_rate, min_improvement=min_improvement)
    report["train_loss"] = (pairwise_loss(train_pairs, adapter)
                            if train_pairs else None)
    report["train_n"] = len(train_pairs)
    report["adapter"] = adapter.name
    report["base_model"] = adapter.base_model
    if not report["passed"]:
        _log.warning("preference gate REJECTED %s: %s",
                     adapter.name, "; ".join(report["reasons"]))
    else:
        _log.info("preference gate PASSED %s (held-out %.3f vs baseline %.3f)",
                  adapter.name, report["adapter_win_rate"],
                  report["baseline_win_rate"])
    return bool(report["passed"]), report


def gate_with_registry(
    adapter: PreferenceAdapter,
    held_out: Sequence[PreferencePair],
    *,
    registry: Any,
    run: Any,
    train_pairs: Sequence[PreferencePair] = (),
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    metric: str = "held_out_win_rate",
) -> tuple[bool, dict[str, Any]]:
    """Held-out gate + the existing ``TrainingRegistry`` production gate.

    Order matters: the held-out reject runs FIRST.  A candidate that
    fails held-out never reaches the registry gate, so a registry
    ``evaluate()``/``promote()`` can never rescue a bad update.  When
    held-out passes, the run is completed through the registry's own
    champion-challenger gate (``evaluate`` + ``promote``), and both
    verdicts must pass for the adapter to land.
    """
    passed, report = production_gate(
        adapter, held_out, train_pairs=train_pairs,
        min_win_rate=min_win_rate, min_improvement=min_improvement)
    report["registry_gate"] = None
    if not passed:
        return False, report
    try:
        run.metrics = {
            **(getattr(run, "metrics", None) or {}),
            metric: report["adapter_win_rate"],
            "score": report["adapter_win_rate"],
        }
        registry_gate_passed = registry.evaluate(run, metric=metric)
        report["registry_gate"] = bool(registry_gate_passed)
        if not registry_gate_passed:
            report["reasons"] = [*report.get("reasons", []),
                                  "registry champion-challenger gate failed"]
            return False, report
        promoted = registry.promote(run)
        report["promoted"] = bool(promoted)
        return bool(promoted), report
    except Exception as exc:  # noqa: BLE001 — registry failure is a reject
        _log.warning("preference registry gate failed: %s", exc)
        report["reasons"] = [*report.get("reasons", []),
                              f"registry gate error: {exc}"]
        return False, report


def lessons_for(adapter: PreferenceAdapter, action_type: str = "") -> list[str]:
    """Lessons relevant to one action type (all lessons when untyped)."""
    if not action_type:
        return list(adapter.lessons)
    typed = [lesson for lesson in adapter.lessons
             if action_type.lower() in lesson.lower()]
    return typed or list(adapter.lessons)


def serving_config(adapter: PreferenceAdapter) -> dict[str, Any]:
    """What ``Brain`` needs to serve this adapter for owner-directed work."""
    return {
        "adapter": adapter.name,
        "base_model": adapter.base_model,
        "pair_count": adapter.pair_count,
        "lessons": list(adapter.lessons),
        "lora_config": adapter.lora_config(),
        "steering": ("prepend lessons for the action type + score "
                     "candidate replies with the preference delta"),
    }
