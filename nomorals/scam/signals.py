"""General scam signals — descriptor-driven scoring over any text.

``nomorals.property.scam`` already owns rental-listing checks
(``check_listing``). This module generalises *around* it: it scores
arbitrary messages / listings / payment requests against a data table
of signal descriptors, and it delegates listing checks to
``property.scam`` instead of rebuilding them.

Scoring contract:

* :data:`SIGNAL_DESCRIPTORS` is *data*: a list of ``{id, kind,
  description, weight, examples}``. Kinds: ``listing``, ``message``,
  ``payment``, ``impersonation``, ``urgency``, ``too_good``,
  ``advance_fee``, ``off_platform``.
* Matching is **token overlap on descriptor example phrases** — the
  text is tokenised, each example phrase is tokenised, and a
  descriptor matches when an example phrase is (near-)fully covered
  by the text's tokens (or appears as a substring). There is no
  hardcoded keyword-regex list that decides the verdict on its own:
  the descriptors are the signal definitions, and matching is
  mechanical overlap against them.
* ``evidence`` (optional) names descriptor ids the caller already
  established (e.g. from a screenshot, a call, or a listing check);
  evidenced descriptors count as matched with an explicit reason.
* Score is the matched weight over a corroboration scale: one strong
  signal alone lands in ``review`` (corroboration discipline, same as
  the property module); two or more strong signals land in
  ``suspicious``.

:func:`assess_listing` delegates to ``property.scam.check_listing``
and merges its flags into the same report shape (``score`` 0..1 where
1 = most suspicious, ``flags``, ``verdict``, ``reasons``), so callers
handle messages and listings uniformly.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["SIGNAL_DESCRIPTORS", "assess", "assess_listing"]

#: Signal descriptors — data, not code. ``weight`` is 1 (weak) to 3
#: (fires towards a verdict with corroboration). ``examples`` are the
#: phrases token-overlap matching runs against.
SIGNAL_DESCRIPTORS: list[dict[str, Any]] = [
    {"id": "too_good_price", "kind": "too_good", "weight": 3,
     "description": "Price far below the going rate for the same thing — classic bait.",
     "examples": ["far below market price", "half the usual price",
                  "unbelievably cheap", "too good to be true",
                  "price far below market", "fraction of the normal price"]},
    {"id": "guaranteed_returns", "kind": "too_good", "weight": 3,
     "description": "Guaranteed or risk-free outsized returns.",
     "examples": ["guaranteed returns", "double your money",
                  "risk free profit", "guaranteed profit",
                  "earn money fast with no risk"]},
    {"id": "urgency_pressure", "kind": "urgency", "weight": 2,
     "description": "Artificial urgency pushing a fast decision.",
     "examples": ["act now", "limited time only", "urgent hurry",
                  "offer ends today", "first come first served",
                  "decide today or lose it", "going fast"]},
    {"id": "advance_fee", "kind": "advance_fee", "weight": 3,
     "description": "Money demanded before anything is delivered or shown.",
     "examples": ["pay before viewing", "pay upfront fee",
                  "send money upfront", "pay before you receive",
                  "money before delivery", "registration fee to unlock",
                  "pay a fee to release your prize"]},
    {"id": "irreversible_payment", "kind": "payment", "weight": 3,
     "description": "Payment via an irreversible route (crypto, gift card, wire).",
     "examples": ["pay with gift card", "pay in crypto",
                  "pay with bitcoin", "western union transfer",
                  "send crypto", "pay by gift cards"]},
    {"id": "personal_account_payment", "kind": "payment", "weight": 2,
     "description": "Payment to a personal account instead of a verifiable business account.",
     "examples": ["pay to my personal account", "send to my personal account",
                  "pay directly to me", "pay into my own account"]},
    {"id": "impersonation", "kind": "impersonation", "weight": 3,
     "description": "Claims to be an official, a known person, or support — unverified.",
     "examples": ["i am from the bank", "this is customer support",
                  "i am a police officer", "official from the government",
                  "your account will be closed by support"]},
    {"id": "account_threat", "kind": "impersonation", "weight": 2,
     "description": "Threat that an account will be closed/suspended unless you act.",
     "examples": ["your account will be closed", "account suspended unless",
                  "verify your account now or lose access",
                  "your account is compromised"]},
    {"id": "off_platform", "kind": "off_platform", "weight": 2,
     "description": "Push to move chat or payment off the platform / off traceable channels.",
     "examples": ["move to whatsapp", "chat outside the platform",
                  "contact me directly off platform",
                  "do not use the platform payment",
                  "pay outside the platform"]},
    {"id": "no_viewing", "kind": "listing", "weight": 3,
     "description": "No viewing / inspection possible before payment.",
     "examples": ["no viewing possible", "cannot view the property",
                  "money before viewing", "keys will be sent by post",
                  "i am abroad and cannot show it"]},
    {"id": "vague_listing", "kind": "listing", "weight": 1,
     "description": "Listing with landmark-only address and no verifiable details.",
     "examples": ["near the landmark", "address on request only",
                  "no full address given", "landmark only address"]},
    {"id": "prize_fee", "kind": "message", "weight": 3,
     "description": "A prize / inheritance / windfall that needs a fee to claim.",
     "examples": ["you won a prize", "claim your prize",
                  "inheritance waiting for you", "you have won the lottery",
                  "pay to claim your winnings"]},
]

#: Descriptor ids by kind, for callers that want one kind's signals.
_DESCRIPTORS_BY_ID = {d["id"]: d for d in SIGNAL_DESCRIPTORS}

#: Corroboration scale: matched weight >= this → suspicious.
_SUSPICIOUS_AT = 6.0
#: Matched weight >= this → review.
_REVIEW_AT = 2.0
#: Token-overlap threshold for an example phrase to count as matched.
#: 0.75 means a 3-token phrase needs all 3 tokens (2/3 generic tokens
#: like "your money" must NOT fire "double your money").
_OVERLAP_AT = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _example_matches(example: str, text_lower: str, text_tokens: set[str]) -> bool:
    """True when an example phrase is covered by the text.

    Substring hit (the whole phrase appears) or token overlap >=
    threshold. Both are mechanical overlap against descriptor data —
    not a hand-written keyword list.
    """
    example_lower = example.lower().strip()
    if not example_lower:
        return False
    if example_lower in text_lower:
        return True
    example_tokens = _tokens(example_lower)
    if not example_tokens:
        return False
    overlap = len(example_tokens & text_tokens) / len(example_tokens)
    return overlap >= _OVERLAP_AT


def _matched_descriptors(text: str, evidence: list[str] | None) -> tuple[list[dict[str, Any]], list[str]]:
    text_lower = (text or "").lower()
    text_tokens = _tokens(text or "")
    matched: list[dict[str, Any]] = []
    reasons: list[str] = []
    for descriptor in SIGNAL_DESCRIPTORS:
        hit_example = None
        for example in descriptor["examples"]:
            if _example_matches(example, text_lower, text_tokens):
                hit_example = example
                break
        if hit_example is not None:
            matched.append(descriptor)
            reasons.append(
                f"{descriptor['id']} ({descriptor['kind']}): matched example "
                f"{hit_example!r} — {descriptor['description']}")
    for ev in evidence or []:
        descriptor = _DESCRIPTORS_BY_ID.get(str(ev))
        if descriptor is None:
            reasons.append(f"unknown evidence {ev!r} ignored")
            continue
        if descriptor not in matched:
            matched.append(descriptor)
            reasons.append(
                f"{descriptor['id']} ({descriptor['kind']}): supplied as evidence — "
                f"{descriptor['description']}")
    return matched, reasons


def assess(text: str, kind: str = "message",
           evidence: list[str] | None = None) -> dict[str, Any]:
    """Score ``text`` against the signal descriptors.

    Returns ``{score 0..1, flags, verdict, reasons}`` where
    ``verdict`` is ``suspicious`` | ``review`` | ``ok`` and ``flags``
    are the matched descriptor ids. ``kind`` records the caller's
    context (``"message"``, ``"listing"``, …) — every descriptor is
    evaluated regardless, because a message can carry payment,
    urgency and impersonation signals at once.
    """
    matched, reasons = _matched_descriptors(text or "", evidence)
    matched_weight = sum(d["weight"] for d in matched)
    # Corroboration scale: one weight-3 signal alone → review, not
    # suspicious; two strong signals (or a strong + corroboration)
    # → suspicious. Normalised to 0..1 for the report.
    score = round(min(1.0, matched_weight / _SUSPICIOUS_AT), 4)
    if matched_weight >= _SUSPICIOUS_AT:
        verdict = "suspicious"
    elif matched_weight >= _REVIEW_AT:
        verdict = "review"
    else:
        verdict = "ok"
    return {
        "score": score,
        "flags": [d["id"] for d in matched],
        "verdict": verdict,
        "reasons": reasons,
        "kind": kind,
        "matched_weight": matched_weight,
    }


def assess_listing(url_or_text: str, *, check_listing_fn: Any = None,
                   **kwargs: Any) -> dict[str, Any]:
    """Score a listing: generic signals + ``property.scam`` delegation.

    The listing-specific checks come from the injected
    ``check_listing_fn`` — normally
    ``nomorals.property.scam.check_listing``, passed by the caller.
    This module deliberately does NOT import the property module
    itself: ``scam`` is L0 and property is L4, so a static import
    would be an upward layer violation. Without a checker the generic
    signals still run, with a reason saying the property check was
    not supplied.
    """
    generic = assess(url_or_text or "", kind="listing")
    if check_listing_fn is None:
        generic = dict(generic)
        generic["reasons"] = generic["reasons"] + [
            "property.scam.check_listing not supplied "
            "(pass check_listing_fn=...); generic signals only"]
        return generic
    try:
        report = check_listing_fn(url_or_text or "", **kwargs)
    except Exception as exc:  # noqa: BLE001 — property module unavailable
        generic = dict(generic)
        generic["reasons"] = generic["reasons"] + [
            f"property.scam.check_listing unavailable ({exc}); generic signals only"]
        return generic
    listing_flags = [f.code for f in report.flags]
    # Merge: generic descriptor flags + property check codes, deduped.
    flags: list[str] = list(generic["flags"])
    for code in listing_flags:
        if code not in flags:
            flags.append(code)
    reasons = list(generic["reasons"])
    for f in report.flags:
        reasons.append(
            f"property:{f.code} ({f.severity}): {f.message}"
            + (f" — {f.evidence}" if f.evidence else ""))
    # Property score is 0..100 with higher = safer; invert it.
    listing_score = round(1.0 - max(0, min(100, report.score)) / 100.0, 4)
    combined = round(min(1.0, max(generic["score"], listing_score)), 4)
    if report.verdict.startswith("likely scam"):
        verdict = "suspicious"
    elif report.verdict.startswith("check carefully"):
        verdict = "review" if combined < 0.6 else "suspicious"
    elif generic["verdict"] == "suspicious":
        verdict = "suspicious"
    elif generic["verdict"] == "review" or combined >= 0.3:
        verdict = "review"
    else:
        verdict = "ok"
    return {
        "score": combined,
        "flags": flags,
        "verdict": verdict,
        "reasons": reasons,
        "kind": "listing",
        "matched_weight": generic["matched_weight"],
        "listing_score": report.score,
        "listing_verdict": report.verdict,
        "listing_url": report.url,
    }
