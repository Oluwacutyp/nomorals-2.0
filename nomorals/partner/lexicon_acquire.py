"""Research -> partner lexicon acquisition loop (wave E).

Wave D built the pieces — pull (``mine_candidates``), score
(``score_term``), store (``LexiconStore.acquire``, versioned), reload
(``LexiconFeed`` reads the store live) — but nothing connected them.
This module is the loop:

    scored research findings
        -> pull: keep findings at/above ``min_confidence``, mine candidates
        -> score + store: per-category ``acquire`` into the ``partner``
           module; the scorer's relevance component routes each candidate
           to the categories it actually fits (a catchphrase-y span lands
           in ``catchphrase``, a mood-y span in ``mood_expression``,
           everything else is rejected by the threshold)
        -> versioning: every successful acquire bumps ``lexicon_versions``
        -> reload: the partner feed re-reads so new terms take effect
           without a restart

The loop is driven from the Core Mind's research dispatch
(``coremind._dispatch_research``): each finished swarm run feeds its
scored findings through here, best-effort. It can also be driven by the
``research_lexicon`` tool or any caller with findings in hand.

Never raises: a loop failure is a log line, never a broken reply.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..agents.research_lexicon import (
    LexiconStore,
    acquire_from_findings,
    mine_candidates,
)
from ..core.logging_setup import get_logger
from .lexicon_feed import (
    CATEGORIES,
    CATEGORY_KEYWORDS,
    LEXICON_MODULE,
    LexiconFeed,
)

__all__ = [
    "LOOP_THRESHOLD",
    "MIN_FINDING_CONFIDENCE",
    "feed_partner_lexicon",
    "pull_scored_findings",
]

_log = get_logger(__name__)

#: Acquire threshold for loop-fed terms. Same as the partner seed (0.35,
#: not the research default 0.55): colloquial voice terms rarely share
#: content words with the category keywords, so relevance lands at the
#: neutral-to-zero end and scoring leans on novelty + quality. The junk
#: filter (score 0.0) still rejects degenerate terms at any threshold.
LOOP_THRESHOLD = 0.35

#: Minimum finding confidence to mine terms from — the "scored" in
#: "scored research findings". Below this, a finding is too weak to
#: teach the voice anything.
MIN_FINDING_CONFIDENCE = 0.4


def pull_scored_findings(
    findings: Any, *, min_confidence: float = MIN_FINDING_CONFIDENCE
) -> list[Any]:
    """Pull the scored subset of research findings.

    Accepts ``SwarmFinding`` objects or plain dicts (anything with a
    ``claim`` and a ``confidence``). Findings below ``min_confidence``
    or without claim text are dropped. Pure — no DB, no side effects.
    """
    try:
        floor = float(min_confidence)
    except (TypeError, ValueError):
        floor = MIN_FINDING_CONFIDENCE
    out: list[Any] = []
    for f in findings or []:
        claim = getattr(f, "claim", None)
        if claim is None and isinstance(f, dict):
            claim = f.get("claim")
        if not claim:
            continue
        conf = getattr(f, "confidence", None)
        if conf is None and isinstance(f, dict):
            conf = f.get("confidence")
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        if conf >= floor:
            out.append(f)
    return out


def feed_partner_lexicon(
    db: Any,
    findings: Any,
    *,
    source: str = "research",
    min_confidence: float = MIN_FINDING_CONFIDENCE,
    threshold: float = LOOP_THRESHOLD,
    categories: tuple[str, ...] = CATEGORIES,
) -> dict[str, Any]:
    """Run one acquisition-loop pass over scored research findings.

    Returns ``{"findings": n, "candidates": m, "added": [...],
    "per_category": {cat: {"added": [...], ...}}, "version": v,
    "reloaded": {...}}``. When ``db`` is None the loop is a no-op that
    reports why. Never raises.
    """
    report: dict[str, Any] = {
        "findings": 0,
        "candidates": 0,
        "added": [],
        "per_category": {},
        "version": 0,
        "reloaded": {},
    }
    if db is None:
        report["reason"] = "no db"
        return report
    try:
        scored = pull_scored_findings(findings, min_confidence=min_confidence)
        report["findings"] = len(scored)
        candidates = mine_candidates(scored)
        report["candidates"] = len(candidates)
        if not candidates:
            return report
        store = LexiconStore(SimpleNamespace(db=db))
        added_all: list[str] = []
        for category in categories:
            try:
                result = acquire_from_findings(
                    scored,
                    store,
                    module=LEXICON_MODULE,
                    category=category,
                    source=source,
                    category_keywords=CATEGORY_KEYWORDS.get(category),
                    threshold=threshold,
                )
            except Exception as exc:  # noqa: BLE001 - one bad category ≠ dead loop
                _log.warning("lexicon loop: acquire failed for %r: %s", category, exc)
                result = {"added": [], "skipped": [], "version": store.version(LEXICON_MODULE)}
            report["per_category"][category] = {
                "added": list(result.get("added", [])),
                "skipped": len(result.get("skipped", [])),
            }
            added_all.extend(result.get("added", []))
        report["added"] = added_all
        try:
            report["version"] = store.version(LEXICON_MODULE)
        except Exception:  # noqa: BLE001
            report["version"] = 0
        # reload: the feed reads the store live — re-read the version so
        # the loop can confirm the new terms are visible.
        try:
            report["reloaded"] = LexiconFeed(db).reload()
        except Exception as exc:  # noqa: BLE001
            _log.debug("lexicon loop: reload read failed: %s", exc)
    except Exception as exc:  # noqa: BLE001 - the loop must never break a reply
        _log.warning("lexicon acquisition loop failed: %s", exc)
        report["reason"] = f"loop failed: {exc}"
    return report
