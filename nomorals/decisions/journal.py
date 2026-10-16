"""Decision journal — calibrate the OWNER's life, not system routing.

``nomorals/cognition/calibration.py`` calibrates *models* (per
task/model confidence). This module calibrates the *owner*: the
decisions they actually make — what they chose, what they predicted
would happen, what actually happened — so the owner learns how good
their own judgement is.

Design rules:

* **Honest emptiness.** No resolved decisions → accuracy, mean
  confidence, and Brier are ``None``, and the summary sentence is
  ``None`` — never 0%. 0% would claim "you are always wrong" on zero
  evidence; ``None`` says "no data yet".
* **Predictions are recorded BEFORE outcomes.** ``record()`` takes the
  prediction; ``resolve()`` takes the outcome. A decision resolved
  without a prediction still counts for outcome stats, just not for
  calibration.
* **Outcomes are append-only history.** Resolving twice appends a
  revision to ``history``; the latest revision is the current outcome.
  The journal never silently rewrites what the owner first recorded.
* **Calibration is descriptive, not moral.** The journal reports the
  gap between confidence and accuracy (over/under-confident) and the
  per-confidence-bucket hit rates, so the owner can see *where* their
  judgement is off — e.g. "your 90%-sure calls land 60% of the time".
* **Universal.** Decisions are free text in any language; the journal
  never parses or judges the content, only the numbers the owner
  supplied. Timezones are the caller's (epoch seconds in, epoch out).

Example::

    >>> journal = DecisionJournal()
    >>> d = journal.record(decision="Ship the trading bot",
    ...                    context="Backtests look strong, but live is untested.",
    ...                    options=["Ship now", "Paper-trade a month", "Kill it"],
    ...                    chosen="Paper-trade a month",
    ...                    prediction="Paper trading will reveal one real bug.",
    ...                    confidence=0.7)
    >>> journal.resolve(d.id, outcome="Paper trading found two bugs.",
    ...                 success=True).status
    'resolved'
    >>> journal.calibration_sentence()
    'Your predictions have been 100% accurate across 1 resolved decision (mean confidence 70% — underconfident).'

Only the standard library is imported (see the layering note in
:mod:`nomorals.commitments.ledger`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Decision",
    "OutcomeRevision",
    "DecisionJournal",
    "CONFIDENCE_BUCKETS",
]


#: Confidence buckets for the calibration table: (label, low, high).
CONFIDENCE_BUCKETS = (
    ("50–60%", 0.50, 0.60),
    ("60–70%", 0.60, 0.70),
    ("70–80%", 0.70, 0.80),
    ("80–90%", 0.80, 0.90),
    ("90–100%", 0.90, 1.01),
)


@dataclass
class OutcomeRevision:
    """One recorded outcome (append-only — revisions never overwrite)."""

    ts: float
    outcome: str
    success: bool | None  # True / False / None (resolved, no verdict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "outcome": self.outcome,
                "success": self.success, "note": self.note}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> OutcomeRevision:
        return cls(ts=float(row.get("ts") or 0.0),
                   outcome=str(row.get("outcome", "")),
                   success=row.get("success"),
                   note=str(row.get("note", "")))


@dataclass
class Decision:
    """One decision: what was chosen, what was predicted, what happened."""

    id: str
    decision: str
    context: str = ""
    options: list[str] = field(default_factory=list)
    chosen: str = ""
    prediction: str = ""
    predicted_outcome: str = ""
    confidence: float | None = None  # 0..1, None = "no number given"
    stakes: str = ""  # free text: "low" / "high" / "irreversible" …
    tags: list[str] = field(default_factory=list)
    status: str = "open"  # open | resolved
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None
    history: list[OutcomeRevision] = field(default_factory=list)

    # ── derived ──────────────────────────────────────────────────
    @property
    def outcome(self) -> str:
        return self.history[-1].outcome if self.history else ""

    @property
    def outcome_success(self) -> bool | None:
        return self.history[-1].success if self.history else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "decision": self.decision,
            "context": self.context, "options": list(self.options),
            "chosen": self.chosen, "prediction": self.prediction,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence, "stakes": self.stakes,
            "tags": list(self.tags), "status": self.status,
            "created_at": self.created_at, "resolved_at": self.resolved_at,
            "history": [h.to_dict() for h in self.history],
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Decision:
        return cls(
            id=str(row.get("id", "")),
            decision=str(row.get("decision", "")),
            context=str(row.get("context", "")),
            options=list(row.get("options") or []),
            chosen=str(row.get("chosen", "")),
            prediction=str(row.get("prediction", "")),
            predicted_outcome=str(row.get("predicted_outcome", "")),
            confidence=row.get("confidence"),
            stakes=str(row.get("stakes", "")),
            tags=list(row.get("tags") or []),
            status=str(row.get("status", "open")),
            created_at=float(row.get("created_at") or 0.0),
            resolved_at=row.get("resolved_at"),
            history=[OutcomeRevision.from_dict(h)
                     for h in (row.get("history") or [])],
        )


class DecisionJournal:
    """The owner's decision record + calibration. JSON-backed, optional."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.decisions: dict[str, Decision] = {}
        if path:
            self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        for row in raw.get("decisions") or []:
            try:
                decision = Decision.from_dict(row)
                self.decisions[decision.id] = decision
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"decisions": [d.to_dict()
                                   for d in self.decisions.values()]},
                    fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ── intake ───────────────────────────────────────────────────
    def record(self, *, decision: str, context: str = "",
               options: list[str] | None = None, chosen: str = "",
               prediction: str = "", predicted_outcome: str = "",
               confidence: float | None = None, stakes: str = "",
               tags: list[str] | None = None,
               ts: float | None = None) -> Decision:
        """Record a decision WITH its prediction, before the outcome.

        ``confidence`` must be in [0, 1] when given. The id is
        deterministic on (decision, timestamp) so re-recording the
        same decision at the same time does not duplicate it.
        """
        if confidence is not None:
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"confidence {confidence!r} must be in [0, 1]")
        now = float(ts if ts is not None else time.time())
        digest_src = f"{decision}|{chosen}|{now}"
        did = "dc-" + hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:12]
        if did in self.decisions:
            return self.decisions[did]
        entry = Decision(
            id=did, decision=decision, context=context,
            options=list(options or []), chosen=chosen,
            prediction=prediction, predicted_outcome=predicted_outcome,
            confidence=confidence, stakes=stakes,
            tags=list(tags or []), created_at=now)
        self.decisions[did] = entry
        self._save()
        return entry

    def resolve(self, decision_id: str, *, outcome: str,
                success: bool | None = None, note: str = "",
                ts: float | None = None) -> Decision:
        """Record what actually happened. Appends a revision; never
        rewrites the previously recorded outcome."""
        entry = self.decisions.get(decision_id)
        if entry is None:
            raise KeyError(f"unknown decision {decision_id!r}")
        now = float(ts if ts is not None else time.time())
        entry.history.append(OutcomeRevision(
            ts=now, outcome=outcome,
            success=(None if success is None else bool(success)),
            note=note))
        entry.status = "resolved"
        entry.resolved_at = now
        self._save()
        return entry

    def get(self, decision_id: str) -> Decision | None:
        return self.decisions.get(decision_id)

    def open_decisions(self, tag: str | None = None) -> list[Decision]:
        rows = [d for d in self.decisions.values() if d.status == "open"]
        if tag is not None:
            rows = [d for d in rows if tag in d.tags]
        return sorted(rows, key=lambda d: d.created_at)

    def resolved(self, since: float | None = None,
                 until: float | None = None) -> list[Decision]:
        rows = [d for d in self.decisions.values()
                if d.status == "resolved"]
        if since is not None:
            rows = [d for d in rows if (d.resolved_at or 0.0) >= since]
        if until is not None:
            rows = [d for d in rows if (d.resolved_at or 0.0) < until]
        return sorted(rows, key=lambda d: d.resolved_at or 0.0)

    # ── calibration ──────────────────────────────────────────────
    def calibration(self, since: float | None = None,
                    until: float | None = None) -> dict[str, Any]:
        """Predicted vs actual, honestly.

        ``accuracy`` / ``mean_confidence`` / ``brier`` are ``None``
        with no usable data — never 0%. ``accuracy`` counts only
        resolutions with a True/False verdict; ``brier`` counts only
        resolutions with both a confidence and a verdict.
        """
        rows = self.resolved(since, until)
        verdicts = [d for d in rows if d.outcome_success is not None]
        scored = [d for d in verdicts if d.confidence is not None]
        accuracy = (sum(1 for d in verdicts if d.outcome_success) / len(verdicts)
                    if verdicts else None)
        mean_conf = (sum(d.confidence for d in scored) / len(scored)
                     if scored else None)
        brier = None
        if scored:
            total = sum((d.confidence - (1.0 if d.outcome_success else 0.0)) ** 2
                        for d in scored)
            brier = total / len(scored)
        gap = (mean_conf - accuracy) if (mean_conf is not None
                                         and accuracy is not None) else None
        buckets: list[dict[str, Any]] = []
        for label, low, high in CONFIDENCE_BUCKETS:
            bucket = [d for d in scored if low <= d.confidence < high]
            bucket_verdicts = [d for d in bucket
                               if d.outcome_success is not None]
            buckets.append({
                "bucket": label, "n": len(bucket_verdicts),
                "hit_rate": (sum(1 for d in bucket_verdicts
                                 if d.outcome_success) / len(bucket_verdicts)
                             if bucket_verdicts else None),
            })
        return {
            "n_resolved": len(rows),
            "n_with_verdict": len(verdicts),
            "n_scored": len(scored),
            "accuracy": accuracy,
            "mean_confidence": mean_conf,
            "brier": brier,
            "calibration_gap": gap,
            "verdict": ("overconfident" if gap is not None and gap > 0.05
                        else "underconfident" if gap is not None and gap < -0.05
                        else "well-calibrated" if gap is not None else None),
            "buckets": buckets,
        }

    def calibration_sentence(self, since: float | None = None,
                             until: float | None = None) -> str | None:
        """Plain words: ``Your predictions have been 70% accurate.``

        ``None`` when there is nothing to calibrate — the caller must
        render "no data yet", not a fabricated percentage.
        """
        stats = self.calibration(since, until)
        if stats["n_with_verdict"] == 0 or stats["accuracy"] is None:
            return None
        n = stats["n_with_verdict"]
        base = (f"Your predictions have been {stats['accuracy']:.0%} "
                f"accurate across {n} resolved decision"
                f"{'s' if n != 1 else ''}")
        if stats["mean_confidence"] is not None:
            base += (f" (mean confidence {stats['mean_confidence']:.0%} — "
                     f"{stats['verdict']})")
        return base + "."

    # ── review ───────────────────────────────────────────────────
    def review(self, period: str = "weekly",
               now: float | None = None) -> dict[str, Any]:
        """Weekly / monthly summary: what was decided, what landed.

        ``period`` is ``"weekly"`` (last 7 days) or ``"monthly"``
        (last 30 days), measured on resolution time. Open decisions
        are always listed in full — they are the review's real payload.
        """
        now = float(now if now is not None else time.time())
        window = 7 * 86400 if period == "weekly" else 30 * 86400
        since = now - window
        stats = self.calibration(since, now)
        recent = self.resolved(since, now)
        by_tag: dict[str, dict[str, int]] = {}
        for decision in recent:
            for tag in decision.tags or ["untagged"]:
                bucket = by_tag.setdefault(
                    tag, {"resolved": 0, "success": 0})
                bucket["resolved"] += 1
                if decision.outcome_success:
                    bucket["success"] += 1
        return {
            "period": period,
            "since": since, "until": now,
            "calibration": stats,
            "calibration_sentence": self.calibration_sentence(since, now),
            "resolved_this_period": [
                {"id": d.id, "decision": d.decision, "chosen": d.chosen,
                 "outcome": d.outcome, "success": d.outcome_success,
                 "confidence": d.confidence}
                for d in recent],
            "open_decisions": [
                {"id": d.id, "decision": d.decision, "chosen": d.chosen,
                 "confidence": d.confidence, "created_at": d.created_at,
                 "stakes": d.stakes}
                for d in self.open_decisions()],
            "by_tag": by_tag,
        }

    def render_review(self, period: str = "weekly",
                      now: float | None = None) -> str:
        """Owner-facing text for :meth:`review` — honest when empty."""
        data = self.review(period, now)
        lines = [f"🧭 Decision review ({period})"]
        if data["calibration_sentence"]:
            lines.append(data["calibration_sentence"])
        else:
            lines.append("No resolved decisions this period — "
                         "nothing to calibrate yet.")
        if data["resolved_this_period"]:
            lines.append(f"Resolved: {len(data['resolved_this_period'])}")
            for row in data["resolved_this_period"]:
                mark = ("✅" if row["success"] is True
                        else "❌" if row["success"] is False else "❔")
                lines.append(f"  {mark} {row['decision']} → {row['chosen']}"
                             f" — {row['outcome'] or 'no outcome noted'}")
        if data["open_decisions"]:
            lines.append(f"Still open: {len(data['open_decisions'])}")
            for row in data["open_decisions"]:
                lines.append(f"  ⏳ {row['decision']} → {row['chosen'] or 'undecided'}")
        return "\n".join(lines)
