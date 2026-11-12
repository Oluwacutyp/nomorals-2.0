"""Morning-pulse ranker — rank briefing sections by what he reads.

The briefing's hand rule (``morning_briefing._apply_engagement``)
demotes a section when it has been shown a lot and rarely followed
up on — a single threshold on the current rate. This module learns
the ranking: a classifier over per-briefing section history (views,
followups, the section's own followup rate, recency, size), trained
behind a volume gate, whose P(followup) adjusts a section's priority
when trained.

The code rung is the hand rule. It promotes only when it beats the
hand rule on the same held-out slice. Below the gate it is honestly
unavailable and the hand rule carries the briefing, unchanged.

The history is the per-briefing snapshot journal the composer writes
(:func:`morning_briefing._EngagementStore.snapshot_history`): one row
per (date, section) — views/followups as they stood when that
briefing was composed, plus whether a followup followed. A row is
the training example; the label is the followup.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "PULSE_FEATURES", "PULSE_MIN_ROWS", "pulse_features",
    "fit_pulse_ranker", "rank_adjustment", "load_pulse_ranker",
]

_log = get_logger(__name__)

PULSE_FEATURES = ("views", "followups", "followup_rate", "days_since",
                  "n_items")
#: Floor for the pulse ranker (per-briefing section-days).
PULSE_MIN_ROWS = 30


def pulse_features(row: dict[str, Any]) -> dict[str, float]:
    """Feature vector for one (date, section) history row. Views and
    followups are RAW counts — the hand rule reads them raw
    (``views>=7``), so the code rung in :func:`fit_pulse_ranker`
    must see the same numbers the composer sees."""
    views = float(row.get("views", 0) or 0)
    followups = float(row.get("followups", 0) or 0)
    return {
        "views": views,
        "followups": followups,
        "followup_rate": max(0.0, min(1.0, followups / max(views, 1.0))),
        "days_since": max(0.0, min(7.0, float(
            row.get("days_since", 0) or 0))),
        "n_items": float(row.get("n_items", 0) or 0),
    }


def fit_pulse_ranker(history: list[dict[str, Any]] | None = None, *,
                     store: Any = None,
                     backend: str = "gradboost") -> Any:
    """Batch-fit the pulse ranker on per-briefing section history.
    Each row carries ``label`` (1 = a followup followed that
    briefing, 0 = it did not) or ``followed_up``. Returns the
    toolkit's :class:`llm.specialists.FitOutcome` — "trained" (saved
    under ``morning-rank``), "held", "insufficient" (below
    ``PULSE_MIN_ROWS``), or "unavailable". The code rung is the hand
    rule as a followup predictor (it demotes a section when
    views>=7 and rate<0.2 — i.e. it predicts NO followup there) on
    the same held-out slice."""
    from ..llm.specialists import LearnerSpec, ModelStore, fit_if_ready

    rows: list[dict[str, Any]] = []
    for row in (history or []):
        if not isinstance(row, dict):
            continue
        label = row.get("label", row.get("followed_up"))
        if label is None:
            continue
        rows.append({"features": pulse_features(row),
                     "label": 1 if label else 0})

    def _code_score(feats: list[dict[str, Any]],
                    labels: list[float]) -> float:
        if not labels:
            return 0.0
        correct = 0
        for f, y in zip(feats, labels, strict=True):
            # hand rule predicts NO followup exactly where it demotes.
            rate = float(f.get("followup_rate", 0.0) or 0.0)
            views = float(f.get("views", 0.0) or 0.0)
            demote = views >= 7 and rate < 0.2
            pred = 0 if demote else 1
            if pred == int(y):
                correct += 1
        return correct / len(labels)

    spec = LearnerSpec(job="morning-rank", features=PULSE_FEATURES,
                       kind="classifier", backend=backend,
                       min_examples=PULSE_MIN_ROWS, label_name="followed_up")
    return fit_if_ready(rows, spec, code_score_fn=_code_score,
                        store=(store or ModelStore()))


def rank_adjustment(model: Any, features: dict[str, Any]) -> float:
    """Priority delta in [-1, 1] from a fitted ranker: P(followup)
    centred on 0.5. 0.0 when the model is absent — the hand rule
    carries, unchanged."""
    if model is None:
        return 0.0
    try:
        prob = max(0.0, min(1.0, float(model.predict_one(features))))
    except Exception:  # noqa: BLE001 - a bad model never kills a briefing
        return 0.0
    return round((prob - 0.5) * 2.0, 4)


def load_pulse_ranker(store: Any = None) -> Any | None:
    """The fitted pulse ranker, if one was ever promoted."""
    try:
        from ..llm.specialists import ModelStore

        return (store or ModelStore()).load("morning-rank")
    except Exception:  # noqa: BLE001 - a bad store never kills a briefing
        return None
