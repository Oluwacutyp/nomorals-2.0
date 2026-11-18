"""Candidate tiering: the recruitment module's rules, in code.

A candidate is scored 0–100 from what is actually on file — CV,
skills overlap with the vacancy's requirements, experience against
its minimum, contact, location — and lands in a band:

* A "strong" — 75 and up
* B "fit" — 55 and up
* C "weak" — 35 and up
* D "not a fit" — below 35

Two hard rules sit above the arithmetic, and both say why:

* no contact on file (no email, no phone) is always D — a candidate
  Devon cannot reach cannot be hired, however strong the CV;
* no CV on file caps the tier at C — the skills and experience
  signals are claims without the document behind them.

Every result carries its deciding reasons: the score is never a
bare number and the tier is never a bare letter.  Pure functions —
no I/O, no store — so the rules are testable without a home dir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TierResult", "score_candidate", "tier_candidate"]

#: Band floors, in score order.
BANDS: tuple[tuple[str, float], ...] = (
    ("A", 75.0),
    ("B", 55.0),
    ("C", 35.0),
    ("D", 0.0),
)

#: Signal weights — the whole score is 100.
_W_CV = 25.0
_W_SKILLS = 35.0
_W_EXPERIENCE = 25.0
_W_CONTACT = 10.0
_W_LOCATION = 5.0


@dataclass
class TierResult:
    """A scored + tiered candidate, with the reasons attached."""

    score: float
    tier: str
    reasons: list[str] = field(default_factory=list)


def _skills_of(candidate: Any) -> list[str]:
    raw = getattr(candidate, "skills", None) or []
    return [str(s).strip().casefold() for s in raw if str(s).strip()]


def _required_of(vacancy: Any) -> list[str]:
    raw = getattr(vacancy, "required_skills", None) or []
    return [str(s).strip().casefold() for s in raw if str(s).strip()]


def _has_cv(candidate: Any) -> bool:
    return bool(
        (getattr(candidate, "cv_text", "") or "").strip()
        or (getattr(candidate, "cv_artifact_id", "") or "").strip()
        or (getattr(candidate, "cv_url", "") or "").strip()
    )


def _has_contact(candidate: Any) -> bool:
    return bool(
        (getattr(candidate, "email", "") or "").strip()
        or (getattr(candidate, "phone", "") or "").strip()
    )


def score_candidate(candidate: Any, vacancy: Any) -> TierResult:
    """Score + tier one candidate against one vacancy's requirements."""
    reasons: list[str] = []
    score = 0.0

    # ── CV: 25 — the document behind the claims ──────────────────
    if _has_cv(candidate):
        score += _W_CV
        reasons.append("CV on file (+25)")
    else:
        reasons.append("no CV on file (+0)")

    # ── Skills: 35 — the share of required skills matched ────────
    required = _required_of(vacancy)
    have = _skills_of(candidate)
    if required:
        matched = sorted(set(required) & set(have))
        share = len(matched) / len(required)
        score += _W_SKILLS * share
        names = ", ".join(matched) if matched else "none"
        reasons.append(
            f"skills {len(matched)}/{len(required)} matched "
            f"({names}) (+{_W_SKILLS * share:.0f})")
    elif have:
        score += 20.0
        reasons.append(f"skills on file ({len(have)}) (+20)")
    else:
        reasons.append("no skills on file (+0)")

    # ── Experience: 25 — years against the vacancy minimum ───────
    try:
        years = float(getattr(candidate, "years", 0) or 0)
    except (TypeError, ValueError):
        years = 0.0
    try:
        minimum = float(getattr(vacancy, "min_years", 0) or 0)
    except (TypeError, ValueError):
        minimum = 0.0
    if minimum > 0:
        share = min(1.0, max(0.0, years) / minimum)
        score += _W_EXPERIENCE * share
        reasons.append(
            f"experience {years:g}y vs {minimum:g}y required "
            f"(+{_W_EXPERIENCE * share:.0f})")
    elif years > 0:
        score += _W_EXPERIENCE
        reasons.append(f"experience {years:g}y (+25)")
    else:
        reasons.append("no experience on file (+0)")

    # ── Contact: 10 — reachable or not ───────────────────────────
    if _has_contact(candidate):
        score += _W_CONTACT
        reasons.append("contact on file (+10)")
    else:
        reasons.append("no contact on file (+0)")

    # ── Location: 5 — only when both sides name one ──────────────
    cand_loc = (getattr(candidate, "location", "") or "").strip()
    vac_loc = (getattr(vacancy, "location", "") or "").strip()
    if cand_loc and vac_loc:
        if cand_loc.casefold() == vac_loc.casefold():
            score += _W_LOCATION
            reasons.append(f"location matches ({cand_loc}) (+5)")
        else:
            reasons.append(
                f"location {cand_loc} vs {vac_loc} (+0)")
    elif cand_loc or vac_loc:
        reasons.append("location on one side only (+0)")

    score = max(0.0, min(100.0, round(score, 1)))

    # ── Bands, then the hard rules ───────────────────────────────
    tier = "D"
    for band, floor in BANDS:
        if score >= floor:
            tier = band
            break
    if not _has_contact(candidate):
        tier = "D"
        reasons.append(
            "no contact on file — cannot reach this candidate, tier D")
    if not _has_cv(candidate) and tier in ("A", "B"):
        tier = "C"
        reasons.append(
            "no CV on file — capped at weak (C) without the document")
    return TierResult(score=score, tier=tier, reasons=reasons)


def tier_candidate(candidate: Any, vacancy: Any) -> TierResult:
    """The tiering entry point — one name for the one job."""
    return score_candidate(candidate, vacancy)
