"""Offline policy for deciding whether the training loop should run.

The policy is deliberately data driven and side-effect free.  It can be called
from a scheduler, the CLI, or a test without constructing an agent context.  A
run is eligible when there is enough new data, the periodic interval is due, or
recent mission reflections are getting worse.  The policy never bypasses the
promotion gate; it only decides whether to spend resources trying a run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "PolicyDecision",
    "RetrainingPolicy",
    "RetrainPolicy",
    "RetrainTriggerPolicy",
    "policy_status",
    "psi",
    "drift_band",
]


def psi(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    """Population Stability Index between two samples — the standard
    data-drift number (Evidently/Alibi Detect practice: PSI ≥ 0.2 means
    the distribution moved enough to investigate).

    Pure stdlib: quantile bins from the reference, Laplace-smoothed
    proportions, ``Σ (cur − ref) · ln(cur / ref)``.  Returns ``inf`` when
    the reference is empty.
    """
    ref = [float(v) for v in reference if math.isfinite(float(v))]
    cur = [float(v) for v in current if math.isfinite(float(v))]
    if not ref or not cur or bins < 2:
        return float("inf") if not ref else 0.0
    ordered = sorted(ref)
    edges = [ordered[min(len(ordered) - 1, int(i * len(ordered) / bins))]
             for i in range(bins + 1)]
    edges[-1] = float("inf")
    ref_counts = [0] * bins
    cur_counts = [0] * bins
    for value in ref:
        for b in range(bins):
            if edges[b] <= value < edges[b + 1]:
                ref_counts[b] += 1
                break
    for value in cur:
        for b in range(bins):
            if edges[b] <= value < edges[b + 1]:
                cur_counts[b] += 1
                break
    total = 0.0
    for r, c in zip(ref_counts, cur_counts):
        # Laplace smoothing: an empty bin must not zero the log term.
        rp = (r + 0.5) / (len(ref) + 0.5 * bins)
        cp = (c + 0.5) / (len(cur) + 0.5 * bins)
        total += (cp - rp) * math.log(cp / rp)
    return total


def drift_band(score: float) -> str:
    """Two-tier reading of a PSI score: the MLOps warn/critical split."""
    if not math.isfinite(score):
        return "unknown"
    if score >= 0.25:
        return "critical"
    if score >= 0.1:
        return "warning"
    return "stable"


@dataclass(frozen=True)
class PolicyDecision:
    """Explainable result of one trigger evaluation."""

    should_retrain: bool
    reasons: tuple[str, ...] = ()
    new_examples: int = 0
    elapsed_seconds: float = 0.0
    reflection_scores: tuple[float, ...] = ()
    data_available: bool = True
    drift_score: float | None = None
    response: str = ""  #: what the drift evidence recommends: retrain|expand|investigate|""

    @property
    def trigger(self) -> bool:
        """Alias used by schedulers that call the decision a trigger."""
        return self.should_retrain

    @property
    def eligible(self) -> bool:
        return self.should_retrain

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "trigger": self.trigger,
            "reasons": list(self.reasons),
            "new_examples": self.new_examples,
            "elapsed_seconds": (
                None if not math.isfinite(self.elapsed_seconds) else round(self.elapsed_seconds, 3)
            ),
            "reflection_scores": list(self.reflection_scores),
            "data_available": self.data_available,
            "drift_score": (
                None if self.drift_score is None or not math.isfinite(self.drift_score)
                else round(self.drift_score, 4)
            ),
            "drift_band": drift_band(self.drift_score) if self.drift_score is not None else "unknown",
            "response": self.response,
        }


@dataclass
class RetrainingPolicy:
    """Trigger settings with conservative, offline-safe defaults.

    The alias fields make configuration ergonomic without requiring a config-file
    migration: ``growth_threshold`` and ``min_new_examples`` mean the same thing,
    as do ``min_elapsed_seconds`` and ``interval_seconds``.
    """

    dataset_growth_threshold: int = 100
    interval_seconds: float = 7 * 24 * 3600.0
    reflection_window: int = 5
    reflection_decline: float = 0.10
    cooldown_seconds: float = 0.0
    # Distribution-drift trigger (PSI over reflection scores / eval losses):
    # 0.2 is the Evidently-standard "the distribution moved" line.
    drift_threshold: float = 0.2
    drift_bins: int = 10
    # Compatibility/configuration aliases.
    growth_threshold: int | None = None
    min_new_examples: int | None = None
    min_growth: int | None = None
    min_dataset_growth: int | None = None
    min_elapsed_seconds: float | None = None

    def __post_init__(self) -> None:
        aliases = [
            self.growth_threshold,
            self.min_new_examples,
            self.min_growth,
            self.min_dataset_growth,
        ]
        selected = next((int(value) for value in aliases if value is not None), None)
        if selected is not None:
            self.dataset_growth_threshold = max(0, selected)
        if self.min_elapsed_seconds is not None:
            self.interval_seconds = max(0.0, float(self.min_elapsed_seconds))
        self.dataset_growth_threshold = max(0, int(self.dataset_growth_threshold))
        self.interval_seconds = max(0.0, float(self.interval_seconds))
        self.reflection_window = max(2, int(self.reflection_window))
        self.reflection_decline = max(0.0, float(self.reflection_decline))
        self.cooldown_seconds = max(0.0, float(self.cooldown_seconds))
        self.drift_threshold = max(0.0, float(self.drift_threshold))
        self.drift_bins = max(2, int(self.drift_bins))

    def decide(
        self,
        *,
        new_examples: int = 0,
        elapsed_seconds: float | None = None,
        last_run_at: float | None = None,
        now: float | None = None,
        reflection_scores: Iterable[float] = (),
        data_available: bool | None = None,
        drift_reference: Iterable[float] = (),
        drift_current: Iterable[float] = (),
        drift_score: float | None = None,
    ) -> PolicyDecision:
        """Return a reasoned decision without touching the database.

        ``drift_reference``/``drift_current`` are two samples of the same
        signal (e.g. reflection scores before/after) — their PSI is the
        distribution-drift trigger.  Pass a precomputed ``drift_score``
        instead when the caller already has one.
        """
        current = time.time() if now is None else float(now)
        if elapsed_seconds is None:
            elapsed = float("inf") if last_run_at is None else max(0.0, current - float(last_run_at))
        else:
            elapsed = max(0.0, float(elapsed_seconds))

        scores = tuple(float(score) for score in reflection_scores)
        has_data = bool(new_examples > 0 or scores) if data_available is None else bool(data_available)
        reasons: list[str] = []
        growth = int(new_examples) >= self.dataset_growth_threshold
        interval = elapsed >= self.interval_seconds
        trend = self._declining(scores)

        if drift_score is None:
            reference = tuple(float(v) for v in drift_reference)
            present = tuple(float(v) for v in drift_current)
            drift_score = psi(reference, present, bins=self.drift_bins) \
                if reference and present else None
        drifted = drift_score is not None and math.isfinite(drift_score) \
            and drift_score >= self.drift_threshold

        if growth:
            reasons.append("dataset growth threshold reached")
        if interval:
            reasons.append("retraining interval elapsed")
        if trend:
            reasons.append("reflection scores trending down")
        if drifted:
            assert drift_score is not None
            reasons.append(
                f"distribution drift detected (PSI={drift_score:.3f}, "
                f"band={drift_band(drift_score)})")

        # The drift evidence recommends a RESPONSE, not just a boolean —
        # drift alone triggers an investigation; drift + declining scores
        # recommends retraining; drift + growth recommends dataset expansion.
        response = ""
        if drifted:
            if trend:
                response = "retrain"
            elif growth:
                response = "expand"
            else:
                response = "investigate"

        # Cooldown applies to all automatic reasons.  A caller can still inspect
        # the reasons to explain why a run was deferred.
        if self.cooldown_seconds and elapsed < self.cooldown_seconds:
            reasons.append("cooldown active")
            should = False
        else:
            should = bool(reasons) and has_data
        if not has_data:
            reasons = ["no training data"]
            should = False

        return PolicyDecision(
            should_retrain=should,
            reasons=tuple(reasons),
            new_examples=max(0, int(new_examples)),
            elapsed_seconds=elapsed,
            reflection_scores=scores,
            data_available=has_data,
            drift_score=drift_score,
            response=response,
        )

    def should_retrain(self, **kwargs: Any) -> bool:
        """Boolean convenience wrapper around :meth:`decide`."""
        return self.decide(**kwargs).should_retrain

    def evaluate(self, **kwargs: Any) -> PolicyDecision:
        """Named alias for schedulers that call a policy evaluation."""
        return self.decide(**kwargs)

    def _declining(self, scores: Sequence[float]) -> bool:
        if len(scores) < self.reflection_window:
            return False
        # The public API accepts chronological scores (oldest to newest).  The
        # database adapter below reverses its newest-first SQL result before it
        # calls us, keeping this method unsurprising for direct callers.
        ordered = list(scores[-self.reflection_window :])
        midpoint = len(ordered) // 2
        older = sum(ordered[:midpoint]) / max(1, midpoint)
        recent = sum(ordered[midpoint:]) / max(1, len(ordered) - midpoint)
        return older - recent >= self.reflection_decline

    def evaluate_db(self, db: Any, *, now: float | None = None) -> PolicyDecision:
        """Evaluate against shipped SQLite tables and their durable timestamps."""
        return policy_status(db, policy=self, now=now)["decision"]


# Short spellings for callers and documentation that use "retrain".
RetrainPolicy = RetrainingPolicy
RetrainTriggerPolicy = RetrainingPolicy


def policy_status(
    db: Any,
    *,
    policy: RetrainingPolicy | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Return trigger state and the source metrics used to reach it."""
    policy = policy or RetrainingPolicy()
    current = time.time() if now is None else float(now)

    latest_run = _scalar(
        db,
        "SELECT MAX(COALESCE(finished_at, created_at)) FROM training_runs",
        default=None,
    )
    last_run = None if latest_run is None else float(latest_run)
    if last_run is None:
        new_examples = int(_scalar(db, "SELECT COALESCE(SUM(rows),0) FROM datasets", default=0) or 0)
    else:
        new_examples = int(
            _scalar(
                db,
                "SELECT COALESCE(SUM(rows),0) FROM datasets WHERE created_at > ?",
                (last_run,),
                default=0,
            )
            or 0
        )

    reflection_rows = _query(
        db,
        "SELECT score FROM reflections ORDER BY created_at DESC LIMIT ?",
        (policy.reflection_window,),
    )
    # SQL is newest-first; the policy's direct API is oldest-to-newest.
    scores = [float(row.get("score") or 0.0) for row in reversed(reflection_rows)]
    total_rows = int(_scalar(db, "SELECT COALESCE(SUM(rows),0) FROM datasets", default=0) or 0)
    # Before the first collection there is no dataset row yet, but the live
    # stores may already contain valuable training material.  Treat those rows
    # as available so the automatic job can perform its collect stage.
    live_rows = sum(
        int(_scalar(db, f"SELECT COUNT(*) FROM {table}", default=0) or 0)
        for table in ("memories", "messages", "tool_calls", "reflections")
    )
    decision = policy.decide(
        new_examples=new_examples,
        last_run_at=last_run,
        now=current,
        reflection_scores=scores,
        data_available=(total_rows + live_rows) > 0,
    )
    return {
        "decision": decision,
        "last_run_at": last_run,
        "total_dataset_rows": total_rows,
        "live_rows": live_rows,
        "reflection_scores": scores,
        "policy": policy,
    }


def _scalar(db: Any, sql: str, params: Sequence[Any] = (), *, default: Any = None) -> Any:
    try:
        return db.scalar(sql, params, default=default)
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return default
        raise


def _query(db: Any, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in db.query(sql, params)]
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return []
        raise
