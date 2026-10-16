"""Uncertainty-driven learning, pointed at the model.

``autonomy/weakness.py`` tracks weaknesses in tools, jobs, and systems —
but never the model itself.  This module points the same machinery at
the brain: where the model's stated confidence diverges from its real
success rate, that (task, model) slice is a learning target — collect
data there, distill there, train there.

Uncertainty for a slice = ``1 - empirical_rate`` (how often it actually
fails) plus the calibration gap (how overconfident it is about failing).
Slices are ranked by uncertainty, so the scarcest learning effort goes
where the model is both wrong and sure — the most expensive failure
mode.  Findings are recorded through ``weakness.report_weakness`` so
model weaknesses live in the same ledger as every other weakness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "ModelLearningTarget",
    "rank_uncertainty",
    "learning_targets",
    "record_model_failure",
    "record_model_weakness",
]


@dataclass
class ModelLearningTarget:
    """One (task, model) slice the model should learn about next."""

    task: str
    model: str
    uncertainty: float
    empirical_rate: float | None
    mean_predicted: float | None
    calibration_gap: float | None
    n_samples: int
    suggested_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "model": self.model,
            "uncertainty": round(self.uncertainty, 4),
            "empirical_rate": self.empirical_rate,
            "mean_predicted": self.mean_predicted,
            "calibration_gap": self.calibration_gap,
            "n_samples": self.n_samples,
            "suggested_action": self.suggested_action,
        }


def _suggested_action(uncertainty: float, gap: float | None) -> str:
    if uncertainty >= 0.8:
        return ("collect targeted data + distill judged pairs + "
                "preference-tune (model is wrong AND confident)")
    if uncertainty >= 0.5:
        return "distill judged pairs for this slice; add to replay buffer"
    if (gap or 0.0) > 0.2:
        return ("recalibrate abstention threshold for this slice "
                "(overconfident, rarely wrong)")
    return "monitor — low uncertainty, keep the slice in the eval harness"


def rank_uncertainty(
    store: Any,
    *,
    min_samples: int = 3,
) -> list[ModelLearningTarget]:
    """Rank (task, model) slices by uncertainty, worst first.

    Uses ``cognition/calibration.py`` :class:`CalibrationStore` records.
    Slices below ``min_samples`` are skipped — a single failure is a
    data point, not a learning target.
    """
    targets: list[ModelLearningTarget] = []
    for task, model in store.pairs():
        rows = store.records(task, model)
        if len(rows) < min_samples:
            continue
        empirical = store.empirical_rate(task, model)
        predicted = store.mean_predicted(task, model)
        gap = store.calibration_gap(task, model)
        if empirical is None:
            continue
        # Uncertainty = failure rate + overconfidence gap (clamped).
        uncertainty = (1.0 - empirical) + max(0.0, gap or 0.0)
        uncertainty = min(1.5, max(0.0, uncertainty))
        targets.append(ModelLearningTarget(
            task=task, model=model, uncertainty=uncertainty,
            empirical_rate=round(empirical, 4),
            mean_predicted=(round(predicted, 4)
                            if predicted is not None else None),
            calibration_gap=(round(gap, 4) if gap is not None else None),
            n_samples=len(rows),
            suggested_action=_suggested_action(uncertainty, gap),
        ))
    targets.sort(key=lambda t: (-t.uncertainty, t.task, t.model))
    return targets


def learning_targets(
    store: Any,
    *,
    min_samples: int = 3,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Top learning targets as plain dicts (worst uncertainty first)."""
    return [t.to_dict() for t in rank_uncertainty(store, min_samples=min_samples)[:limit]]


def record_model_failure(
    store: Any,
    *,
    task: str,
    model: str,
    predicted_confidence: float,
    success: bool,
) -> Any:
    """Record one model outcome in the calibration store (the raw signal)."""
    return store.record(task, model, predicted_confidence, success)


def record_model_weakness(
    db: Any,
    *,
    task: str,
    model: str,
    evidence: dict[str, Any] | None = None,
) -> str:
    """File a model weakness in the shared ``autonomy/weakness.py`` ledger.

    Subject is namespaced ``model:<task>:<model>`` so model weaknesses
    cluster with each other and never collide with tool weaknesses.
    Returns the weakness id.
    """
    from .weakness import report_weakness

    return report_weakness(
        db, "model_weakness", f"model:{task}:{model}",
        evidence={"task": task, "model": model, **(evidence or {})},
        fuzzy=False)
