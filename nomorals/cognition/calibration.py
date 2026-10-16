"""Confidence calibration per (task, model).

A model that says "90% sure" and is right half the time is not 90%
sure — it is overconfident, and routing on its raw confidence will
over-trust it.  This module keeps the receipts: every recorded outcome
pairs the *predicted* confidence with what actually happened, and
from those pairs it computes:

- :meth:`CalibrationStore.brier_score` — mean squared error between
  predicted confidence and the 0/1 outcome (lower is better; 0 is
  perfect, 0.25 is coin-flip-at-0.5).
- :meth:`CalibrationStore.calibrated_confidence` — a corrected
  confidence.  Without a fresh prediction it is the Beta-shrunk
  empirical success rate (prior 0.5, strength 4 — unknown stays
  0.5).  With a fresh prediction it shifts that prediction by the
  model's measured over/under-confidence gap, shrunk by sample size,
  so a handful of observations barely move the number while a long
  track record moves it most of the way.
- :meth:`CalibrationStore.summary` — the owner-facing feed: per
  (task, model) rows with sample size, empirical rate, mean predicted
  confidence, Brier score, and the calibrated Brier (what the score
  *would* have been had the calibrated confidence been used).

Storage is in-memory by design: calibration is a rolling view over
recent outcomes, recomputed from the recorded pairs, not a second
persistent ledger.  Callers that need durability keep the
:class:`TrajectoryStore` as the durable record and replay its rows
into this store.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CalibrationRecord",
    "CalibrationStore",
    "PRIOR_CONFIDENCE",
    "PRIOR_STRENGTH",
    "SHRINKAGE_K",
]

#: Prior confidence for a (task, model) pair with no data.
PRIOR_CONFIDENCE = 0.5
#: Pseudo-observations behind the prior (Beta strength).
PRIOR_STRENGTH = 4.0
#: Shrinkage constant: correction weight is n / (n + SHRINKAGE_K).
SHRINKAGE_K = 5.0


@dataclass
class CalibrationRecord:
    """One predicted-confidence / actual-outcome pair."""

    task: str
    model: str
    predicted: float
    success: bool
    created_at: float = 0.0


def _clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return min(high, max(low, float(value)))


class CalibrationStore:
    """In-memory (task, model) calibration records."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: list[CalibrationRecord] = []

    # ── recording ────────────────────────────────────────────────────
    def record(
        self,
        task: str,
        model: str,
        predicted_confidence: float,
        success: bool,
        *,
        created_at: float | None = None,
    ) -> CalibrationRecord:
        """Record one outcome vs its predicted confidence.

        ``predicted_confidence`` must be in [0, 1]; ``success`` is the
        0/1 outcome.  Returns the stored record.
        """
        predicted = float(predicted_confidence)
        if not 0.0 <= predicted <= 1.0:
            raise ValueError(
                f"predicted_confidence {predicted_confidence!r} must be in [0, 1]")
        rec = CalibrationRecord(
            task=task or "",
            model=model or "",
            predicted=predicted,
            success=bool(success),
            created_at=float(created_at if created_at is not None else time.time()),
        )
        with self._lock:
            self._records.append(rec)
        return rec

    def records(
        self, task: str | None = None, model: str | None = None
    ) -> list[CalibrationRecord]:
        """Records for a (task, model) slice; ``None`` acts as wildcard."""
        with self._lock:
            rows = list(self._records)
        return [
            r for r in rows
            if (task is None or r.task == task)
            and (model is None or r.model == model)
        ]

    def pairs(self) -> list[tuple[str, str]]:
        """Distinct (task, model) pairs with data, sorted."""
        with self._lock:
            seen = {(r.task, r.model) for r in self._records}
        return sorted(seen)

    # ── scoring ──────────────────────────────────────────────────────
    def empirical_rate(self, task: str, model: str) -> float | None:
        """Raw observed success rate for a slice, or None with no data."""
        rows = self.records(task, model)
        if not rows:
            return None
        return sum(1 for r in rows if r.success) / len(rows)

    def mean_predicted(self, task: str, model: str) -> float | None:
        """Mean predicted confidence for a slice, or None with no data."""
        rows = self.records(task, model)
        if not rows:
            return None
        return sum(r.predicted for r in rows) / len(rows)

    def brier_score(self, task: str, model: str) -> float | None:
        """Mean (predicted − outcome)² for a slice; None with no data.

        Lower is better.  A model that always says 0.5 scores 0.25;
        a perfect model scores 0.0.
        """
        rows = self.records(task, model)
        if not rows:
            return None
        total = sum((r.predicted - (1.0 if r.success else 0.0)) ** 2 for r in rows)
        return total / len(rows)

    def calibrated_confidence(
        self,
        task: str,
        model: str,
        predicted: float | None = None,
    ) -> float:
        """Corrected confidence for a (task, model) slice.

        - No data: returns ``predicted`` when given, else the 0.5 prior.
        - ``predicted is None``: Beta-shrunk empirical rate —
          ``(successes + prior * strength) / (n + strength)`` — so a
          1/1 slice reports ~0.5, not 1.0.
        - ``predicted`` given: shifts the prediction by the measured
          calibration gap (``empirical − mean_predicted``) times the
          shrinkage weight ``n / (n + SHRINKAGE_K)``, clamped to
          [0.01, 0.99].  More data → more correction; sparse data →
          prediction barely moves.
        """
        rows = self.records(task, model)
        if not rows:
            if predicted is None:
                return PRIOR_CONFIDENCE
            return _clamp(predicted)
        n = len(rows)
        successes = sum(1 for r in rows if r.success)
        if predicted is None:
            return (successes + PRIOR_CONFIDENCE * PRIOR_STRENGTH) / (
                n + PRIOR_STRENGTH)
        mean_pred = sum(r.predicted for r in rows) / n
        empirical = successes / n
        gap = empirical - mean_pred
        weight = n / (n + SHRINKAGE_K)
        return _clamp(float(predicted) + gap * weight)

    def calibrated_brier(self, task: str, model: str) -> float | None:
        """Brier score had the calibrated confidence been used.

        Each stored prediction is replaced by its calibrated value
        (same gap/shrinkage correction as
        :meth:`calibrated_confidence`) and the mean squared error is
        recomputed.  On systematically over/under-confident data this
        is lower than :meth:`brier_score` — that improvement *is* the
        evidence that calibration helps.  In-sample by construction;
        read it as "how much would correction have helped here", not
        as out-of-sample skill.
        """
        rows = self.records(task, model)
        if not rows:
            return None
        total = 0.0
        for rec in rows:
            calibrated = self.calibrated_confidence(task, model, rec.predicted)
            total += (calibrated - (1.0 if rec.success else 0.0)) ** 2
        return total / len(rows)

    def calibration_gap(self, task: str, model: str) -> float | None:
        """``mean_predicted − empirical``: >0 = overconfident, <0 = under.

        None with no data.
        """
        empirical = self.empirical_rate(task, model)
        mean_pred = self.mean_predicted(task, model)
        if empirical is None or mean_pred is None:
            return None
        return mean_pred - empirical

    # ── owner feed ───────────────────────────────────────────────────
    def summary(
        self,
        task: str | None = None,
        model: str | None = None,
    ) -> str:
        """Owner-facing calibration report, one line per (task, model).

        Feed to the owner (e.g. in the "/how did I do" review): it
        names which models are overconfident on which tasks and how
        much calibration would have helped.  Empty store → honest
        one-liner, never invented numbers.
        """
        pairs = [
            (t, m) for (t, m) in self.pairs()
            if (task is None or t == task) and (model is None or m == model)
        ]
        lines = ["🎯 Confidence calibration"]
        if not pairs:
            lines.append("No calibration data yet — predictions are uncorrected.")
            return "\n".join(lines)
        for t, m in pairs:
            rows = self.records(t, m)
            n = len(rows)
            empirical = self.empirical_rate(t, m) or 0.0
            mean_pred = self.mean_predicted(t, m) or 0.0
            brier = self.brier_score(t, m) or 0.0
            cal_brier = self.calibrated_brier(t, m)
            calibrated = self.calibrated_confidence(t, m)
            gap = self.calibration_gap(t, m) or 0.0
            verdict = (
                "overconfident" if gap > 0.05
                else "underconfident" if gap < -0.05
                else "well-calibrated"
            )
            improvement = ""
            if cal_brier is not None and brier > 0:
                improvement = (
                    f", calibrated Brier {cal_brier:.3f}"
                    f" ({(brier - cal_brier) / brier:+.0%} vs raw)")
            lines.append(
                f"• {t or '(any task)'} / {m or '(any model)'} — n={n},"
                f" empirical {empirical:.0%}, mean predicted {mean_pred:.0%},"
                f" {verdict} (gap {gap:+.0%}), Brier {brier:.3f}"
                f"{improvement}; calibrated confidence {calibrated:.2f}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable snapshot: one entry per (task, model)."""
        out: dict[str, Any] = {}
        for t, m in self.pairs():
            out[f"{t}::{m}"] = {
                "n": len(self.records(t, m)),
                "empirical_rate": self.empirical_rate(t, m),
                "mean_predicted": self.mean_predicted(t, m),
                "brier": self.brier_score(t, m),
                "calibrated_brier": self.calibrated_brier(t, m),
                "calibrated_confidence": self.calibrated_confidence(t, m),
                "calibration_gap": self.calibration_gap(t, m),
            }
        return out
