"""Calibrated confidence UX (build-map #34) — anti-hallucination.

Every agentic-loop answer gets a self-rated confidence. Below threshold
the reply flags uncertainty explicitly instead of guessing:

- high (>= 0.75):   answer ships unchanged
- medium (>= 0.45): answer ships with a light "double-check" touch
- low (< 0.45):     "I'm not sure about this — …" + what to check
- very low (< 0.25): "I don't know — and I'd rather say so than guess."

The scoring is heuristic, offline, and microseconds-cheap — it runs in
the default path with no extra LLM call. (An LLM-judge upgrade is a
natural follow-up; the ``assess_confidence`` signature already accepts
rich evidence for it.)

Signals:
- verified citations (#20's ``[S<n>]`` / resolved ``[n]`` markers) boost,
  capped — five citations aren't five times more certain than two
- hedge words ("maybe", "probably", "I think") drag the score down
- answers built from actual tool calls outrank pure generation
- bare numbers/dates with no citation drag the score down
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Score bands. High answers ship unchanged; medium gets a light touch;
#: low gets an explicit uncertainty prefix; very low gets the "I don't
#: know" framing and the shaky answer is NOT shipped as fact.
HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.45
VERY_LOW_THRESHOLD = 0.25

#: Citation markers: #20's model-output form [S1] and the resolved [1] form.
_CITE_RE = re.compile(r"\[\s*(?:S\s*)?(\d+)\s*\]", re.IGNORECASE)

#: Bare numbers and dates that ought to be cited if asserted as fact.
_NUMBER_RE = re.compile(
    r"(?:₦|\$|€|£)?\d[\d,]*(?:\.\d+)?%?|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:19|20)\d{2}\b"
)

_HEDGE_WORDS = (
    "maybe", "probably", "perhaps", "possibly", "might be", "could be",
    "i think", "i believe", "i'm not sure", "not certain", "seems like",
    "appears to", "roughly", "around", "somewhere around", "my guess",
)

#: reason code -> concrete check suggestion (Devon-direct, not corporate).
_CHECK_SUGGESTIONS = {
    "no_citations": "pulling up a source for the key claims",
    "thin_evidence": "finding one more source to back this up",
    "hedged": "verifying the uncertain parts against a second source",
    "no_tools": "looking this up instead of answering from memory",
    "uncited_numbers": "confirming the numbers against the source",
    "empty_response": "re-running with a clearer question",
}


@dataclass
class Confidence:
    """Self-rated confidence for one answer."""
    score: float            # 0.0 .. 1.0
    level: str              # "high" | "medium" | "low"
    reasons: list[str] = field(default_factory=list)  # reason codes

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 3), "level": self.level,
                "reasons": list(self.reasons)}


def _count_citations(text: str) -> int:
    """Distinct citation markers in the response."""
    return len(set(_CITE_RE.findall(text or "")))


def assess_confidence(
    response: str,
    *,
    evidence: list[Any] | None = None,
    tools_called: list[str] | None = None,
    question: str | None = None,
) -> Confidence:
    """Heuristic self-rating. Offline, fast, never raises.

    ``evidence``: optional list of claim/citation dicts or objects with a
    ``verified`` attribute (from #20's citation infrastructure). Verified
    evidence boosts more than bare markers.
    ``tools_called``: tool names the loop actually invoked.
    """
    try:
        return _assess(response, evidence=evidence,
                       tools_called=tools_called, question=question)
    except Exception:  # noqa: BLE001 — confidence never breaks a run
        return Confidence(score=0.5, level="medium",
                          reasons=["assessment_failed"])


def _assess(
    response: str,
    *,
    evidence: list[Any] | None,
    tools_called: list[str] | None,
    question: str | None,
) -> Confidence:
    text = (response or "").strip()
    reasons: list[str] = []

    if not text or text == "(no response produced)":
        return Confidence(score=0.1, level="low",
                          reasons=["empty_response"])

    score = 0.55  # thin-signal default: honest medium, not fake certainty

    # — citations (#20 pairing): markers boost, verified evidence boosts more,
    #   capped so citation count can't fake certainty.
    markers = _count_citations(text)
    verified = 0
    for item in evidence or []:
        if isinstance(item, dict):
            if item.get("verified"):
                verified += 1
        elif getattr(item, "verified", False):
            verified += 1
    cite_boost = min(0.10 * markers, 0.15) + min(0.08 * verified, 0.10)
    if cite_boost > 0:
        score += cite_boost
        reasons.append("cited_evidence" if verified else "citations_present")
    elif _NUMBER_RE.search(text):
        score -= 0.15
        reasons.append("uncited_numbers")

    # — tool backing: answers built from real tool output outrank pure
    #   generation.
    n_tools = len(tools_called or [])
    if n_tools:
        score += min(0.05 * n_tools, 0.15)
        reasons.append("tool_backed")
    else:
        reasons.append("no_tools")

    # — hedge words: the model telling on itself.
    lowered = text.lower()
    hedges = sum(1 for h in _HEDGE_WORDS if h in lowered)
    if hedges:
        score -= min(0.05 * hedges, 0.20)
        reasons.append("hedged")

    # — very short answers to substantive questions are usually evasions.
    if question and len(question.split()) > 12 and len(text.split()) < 8:
        score -= 0.10
        reasons.append("thin_answer")

    score = max(0.0, min(1.0, score))
    if score >= HIGH_THRESHOLD:
        level = "high"
    elif score >= MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "low"
    return Confidence(score=score, level=level, reasons=reasons)


def _checks_for(reasons: list[str]) -> list[str]:
    """1-2 concrete check suggestions from the reason codes."""
    seen: list[str] = []
    for code in reasons:
        suggestion = _CHECK_SUGGESTIONS.get(code)
        if suggestion and suggestion not in seen:
            seen.append(suggestion)
        if len(seen) == 2:
            break
    return seen or ["double-checking this against a reliable source"]


def _is_substantive(text: str) -> bool:
    """Does this answer warrant a confidence touch? Very short replies
    ("HI", "done") get no suffix — the touch is for answers with enough
    content to be wrong about."""
    words = text.split()
    return len(words) >= 8 or bool(_NUMBER_RE.search(text)) or \
        bool(_CITE_RE.search(text))


def format_with_confidence(response: str, confidence: Confidence) -> str:
    """Apply the confidence UX. High ships unchanged; lower bands flag
    uncertainty in Devon's voice — direct, not corporate. The medium
    touch only fires on substantive answers, never on chit-chat."""
    text = (response or "").strip()
    score = confidence.score

    if score >= HIGH_THRESHOLD:
        return text

    if score >= MEDIUM_THRESHOLD:
        if not _is_substantive(text):
            return text
        return (text + "\n\n_(Fairly confident here, but double-check the "
                       "key numbers before acting on them.)_")

    checks = _checks_for(confidence.reasons)
    check_line = "Here's what I'd check to be sure: " + \
        " and ".join(checks) + "."

    if score < VERY_LOW_THRESHOLD:
        # Don't ship the shaky answer as fact — own the miss instead.
        return ("I don't know — and I'd rather say so than guess. " +
                check_line)

    return ("I'm not sure about this — " + text +
            "\n\n_" + check_line + "_")
