"""Calibrated abstention for ordinary brain answers.

Generalizes the pattern already proven in ``legal/research.py``: that
module says "I don't know" instead of fabricating when its corpus has
no coverage, with a calibrated confidence and a 0.55 floor below which
an answer is low-confidence.  The same discipline belongs on ordinary
answers — a brain that never abstains fabricates at its weakest
moments.

Mechanics: :func:`calibrate_confidence` reuses research's weights
(0.45 retrieval strength + 0.35 coverage + 0.20 source quality) for
evidence-backed answers; :class:`CalibratedAbstention` turns a
confidence + evidence count into an answered/abstained decision with a
measurable abstention rate.  Thresholds can be tightened from the
``cognition/calibration.py`` store: when the model's stated confidence
runs ahead of its empirical success rate, the bar rises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "ABSTAIN_TEXT",
    "AbstentionDecision",
    "CalibratedAbstention",
    "calibrate_confidence",
    "threshold_from_calibration",
    "abstention_rate",
]

#: The honest fallback.  Short, explicit, and it names what would help.
ABSTAIN_TEXT = ("I don't know — I don't have reliable evidence for that. "
                "Tell me where to look (a file, a message, a source) and "
                "I'll check it rather than guess.")
#: Mirrors ``legal/research.py``: below this, an answer is low-confidence.
DEFAULT_THRESHOLD = 0.55
#: Below this, an answer is refused outright even with evidence.
HARD_FLOOR = 0.20
#: Weights from ``legal/research.py::_calibrate`` (kept identical on purpose).
_W_STRENGTH = 0.45
_W_COVERAGE = 0.35
_W_SOURCE = 0.20


def calibrate_confidence(
    strength: float,
    coverage: float,
    *,
    source_quality: float = 0.8,
) -> float:
    """Calibrated confidence in [0, 1] — research's formula, generalized.

    ``strength``: how strongly the evidence supports the answer (0..1).
    ``coverage``: how much of the question the evidence covers (0..1).
    ``source_quality``: how trustworthy the sources are (0..1).
    """
    strength = min(1.0, max(0.0, float(strength)))
    coverage = min(1.0, max(0.0, float(coverage)))
    source_quality = min(1.0, max(0.0, float(source_quality)))
    confidence = (_W_STRENGTH * strength + _W_COVERAGE * coverage
                  + _W_SOURCE * source_quality)
    return round(max(0.0, min(1.0, confidence)), 2)


@dataclass
class AbstentionDecision:
    """One answered/abstained verdict."""

    answered: bool
    confidence: float
    threshold: float
    reason: str = ""
    evidence_count: int = 0
    abstention_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answered": self.answered, "confidence": self.confidence,
            "threshold": self.threshold, "reason": self.reason,
            "evidence_count": self.evidence_count,
            "abstention_text": self.abstention_text,
        }


class CalibratedAbstention:
    """Decide answered vs. \"I don't know\" from calibrated confidence."""

    def __init__(self, *, threshold: float = DEFAULT_THRESHOLD,
                 require_evidence: bool = True) -> None:
        self.threshold = float(threshold)
        self.require_evidence = require_evidence
        self._decisions: list[AbstentionDecision] = []

    def decide(
        self,
        confidence: float,
        *,
        evidence_count: int = 0,
        question: str = "",
    ) -> AbstentionDecision:
        """Verdict for one answer.  Never raises; abstains on bad input."""
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        if not 0.0 <= conf <= 1.0:
            conf = min(1.0, max(0.0, conf))
        if conf < HARD_FLOOR:
            decision = AbstentionDecision(
                answered=False, confidence=conf, threshold=self.threshold,
                reason=f"confidence {conf:.2f} below hard floor {HARD_FLOOR}",
                evidence_count=int(evidence_count),
                abstention_text=ABSTAIN_TEXT)
        elif self.require_evidence and evidence_count <= 0:
            decision = AbstentionDecision(
                answered=False, confidence=conf, threshold=self.threshold,
                reason="no evidence behind the answer",
                evidence_count=0, abstention_text=ABSTAIN_TEXT)
        elif conf < self.threshold:
            decision = AbstentionDecision(
                answered=False, confidence=conf, threshold=self.threshold,
                reason=(f"confidence {conf:.2f} below threshold "
                        f"{self.threshold:.2f}"),
                evidence_count=int(evidence_count),
                abstention_text=ABSTAIN_TEXT)
        else:
            decision = AbstentionDecision(
                answered=True, confidence=conf, threshold=self.threshold,
                reason="confidence clears threshold with evidence",
                evidence_count=int(evidence_count))
        self._decisions.append(decision)
        return decision

    def wrap(self, answer: str, confidence: float, *,
             evidence_count: int = 0, question: str = "") -> tuple[str, AbstentionDecision]:
        """Return (answer-or-abstention, decision) for one brain answer."""
        decision = self.decide(confidence, evidence_count=evidence_count,
                               question=question)
        if decision.answered:
            return answer, decision
        return decision.abstention_text, decision

    def rate(self) -> float:
        """Measured abstention rate over decisions so far (0..1)."""
        return abstention_rate(self._decisions)

    def decisions(self) -> list[AbstentionDecision]:
        return list(self._decisions)


def abstention_rate(decisions: Sequence[AbstentionDecision]) -> float:
    """Fraction of ``decisions`` that abstained (0.0 with no decisions)."""
    if not decisions:
        return 0.0
    abstained = sum(1 for d in decisions if not d.answered)
    return abstained / len(decisions)


def threshold_from_calibration(
    store: Any,
    task: str,
    model: str,
    *,
    base_threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """Raise the abstention bar when the model is overconfident.

    Compares the store's mean predicted confidence against its empirical
    success rate (``cognition/calibration.py``).  Overconfidence
    (predicted > empirical) lifts the threshold by the gap, capped at
    +0.25; underconfidence leaves the base threshold alone.  No data ->
    base threshold (honest default, not a stricter one).
    """
    try:
        empirical = store.empirical_rate(task, model)
        predicted = store.mean_predicted(task, model)
    except Exception:  # noqa: BLE001 — calibration is advisory, never fatal
        return base_threshold
    if empirical is None or predicted is None:
        return base_threshold
    gap = float(predicted) - float(empirical)
    if gap <= 0:
        return base_threshold
    return round(min(0.95, base_threshold + min(0.25, gap)), 3)
