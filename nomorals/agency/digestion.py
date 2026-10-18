"""Digestion that changes behavior (audit §4).

Pipeline: source → extraction → claims → provenance → entity/time
normalization → novelty/conflict → relevance → knowledge update →
action consequences → later validation.

Each :class:`Claim` carries source refs, publication/observation time,
geo/jurisdictional scope, confidence basis, and a freshness rule.
Output is THREE deltas:

1. ``knowledge_delta`` — what is newly known or changed.
2. ``decision_delta`` — which plan / watch / recommendation /
   assumption changes (contradictions land here).
3. ``action_delta`` — what should happen next, if anything.
   ``None`` is a VALID answer: record it cheaply, don't manufacture
   work. Judged by improved answers / changed plans / prevented
   mistakes / completed outcomes — not documents read.

Learning loop: callers feed real outcomes back via
:func:`record_outcome` (e.g. gig never paid → source estimates drop;
deal hid shipping → scoring changes). Kept in-memory per digest
call-site; durable calibration lives in ``research`` source-quality.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["Claim", "ThreeDeltas", "digest", "record_outcome"]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


@dataclass
class Claim:
    text: str
    source: str
    observed_at: float
    scope: str = "global"  # geo / jurisdiction scope, e.g. "NG", "US", "global"
    confidence_basis: str = ""  # why this confidence: source, corroboration…
    freshness_s: float = 86400.0  # after this, treat as stale
    published_at: float | None = None
    contradicts: str = ""  # normalized key/text of the claim this refutes
    entities: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return _normalize(self.text)


@dataclass
class ThreeDeltas:
    knowledge_delta: list[Claim] = field(default_factory=list)
    decision_delta: list[str] = field(default_factory=list)
    action_delta: list[str] | None = None


def digest(source: str, claims: list[Claim], *,
           known_claims: list[Claim] | None = None,
           now: float = 0.0,
           relevance: Callable[[Claim], bool] | None = None,
           consequence: Callable[[Claim], str | None] | None = None
           ) -> ThreeDeltas:
    known = {c.key: c for c in (known_claims or [])}
    knowledge: list[Claim] = []
    decisions: list[str] = []
    actions: list[str] = []
    for claim in claims:
        if relevance is not None and not relevance(claim):
            continue
        # Novelty: exact-normalized duplicate of known state → nothing new.
        if claim.key in known:
            continue
        # Conflict: explicit contradiction of a known claim.
        if claim.contradicts and _normalize(claim.contradicts) in known:
            decisions.append(
                f"assumption revised: {claim.contradicts!r} contradicted "
                f"by {claim.source} ({claim.text[:160]})")
            knowledge.append(claim)
            if consequence is not None:
                nxt = consequence(claim)
                if nxt:
                    actions.append(nxt)
            continue
        knowledge.append(claim)
        if consequence is not None:
            nxt = consequence(claim)
            if nxt:
                actions.append(nxt)
    return ThreeDeltas(
        knowledge_delta=knowledge, decision_delta=decisions,
        action_delta=actions or None)


def record_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    """Close the learning loop after a real result.

    ``outcome``: {kind: "gig"|"deal"|"research", expected, actual,
    source, note}. Returns the calibration adjustment the caller
    should apply to its estimates — never silently rewrites history.
    """
    expected = float(outcome.get("expected", 0) or 0)
    actual = float(outcome.get("actual", 0) or 0)
    kind = outcome.get("kind", "")
    if kind == "gig" and expected:
        return {"source": outcome.get("source", ""),
                "adjustment": "downweight source win/payout estimates",
                "ratio": actual / expected if expected else 0.0}
    if kind == "deal" and expected:
        return {"source": outcome.get("source", ""),
                "adjustment": "include hidden shipping/import in landed price",
                "ratio": actual / expected if expected else 0.0}
    return {"source": outcome.get("source", ""), "adjustment": "none",
            "ratio": 1.0}
