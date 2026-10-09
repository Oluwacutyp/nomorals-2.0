"""Deep recall strategies layered on the trusted recall path.

``MemoryManager.recall()`` is the trusted single-shot merge — its scoring
math is inviolable and is never touched here.  This module adds the
*strategies* that sit on top of it:

- **multi-hop recall** (``recall_deep``): hop 1 runs the trusted recall;
  hop 2 re-queries with terms mined from the hop-1 hits (the "I didn't
  know the right words yet" pattern); the two hops fuse with a decay on
  hop 2.  Never worse than hop 1 alone — on any error it *is* hop 1.
- **associative recall** (``related``): "what else connects to this
  memory?" — same scope, shared tags, shared entities, same kind, each a
  strategy with its own weight, not one hardcoded boost.
- **temporal reasoning** (``timeline``, ``decisions_about``,
  ``recall_window``): time as a queryable dimension — "what did we decide
  last month about X", oldest → newest, superseded records included so
  the evolution (not just the current belief) is visible.

All of these call ``manager.recall()`` / ``manager.get()`` and re-rank
*after* scoring.  The trusted path is consumed, never replaced.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.logging_setup import get_logger
from .base import MemoryKind
from .scopes import scope_of

_log = get_logger(__name__)

__all__ = [
    "DeepRecallResult",
    "decisions_about",
    "recall_deep",
    "recall_window",
    "related",
    "timeline",
]

_WORD = re.compile(r"[a-z][a-z0-9\-]{3,}")
_ENTITY = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

_STOP = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "have", "has",
    "had", "will", "would", "should", "could", "about", "what", "when",
    "where", "which", "their", "there", "they", "them", "then", "than",
    "also", "into", "over", "under", "your", "you're", "just", "like",
    "more", "most", "some", "such", "been", "were", "was", "are", "our",
    "devon", "please", "thanks", "thank", "know", "think", "want", "need",
    "make", "made", "does", "did", "doing", "don't", "doesn't", "can't",
    "remember", "recall", "memory", "tell", "told", "said", "says",
})


@dataclass
class DeepRecallResult:
    """A recall with its strategy trace attached."""

    records: list[Any] = field(default_factory=list)
    query: str = ""
    hops: int = 1
    expansion_terms: list[str] = field(default_factory=list)
    strategy: str = "single"
    elapsed_ms: float = 0.0
    fallback: str = ""

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def texts(self) -> list[str]:
        return [r.content for r in self.records]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "hops": self.hops,
            "expansion_terms": self.expansion_terms,
            "fallback": self.fallback,
            "count": len(self.records),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "records": [r.to_dict() for r in self.records],
        }


# ── multi-hop recall ─────────────────────────────────────────────────────

def _content_terms(texts: Sequence[str], exclude: set[str],
                   limit: int = 8) -> list[str]:
    """Salient terms from hop-1 hits that were not in the query."""
    counts: Counter[str] = Counter()
    for text in texts:
        for word in _WORD.findall((text or "").lower()):
            if word in _STOP or word in exclude:
                continue
            counts[word] += 1
    # capitalized entities carry more signal than common nouns
    entities: Counter[str] = Counter()
    for text in texts:
        for match in _ENTITY.findall(text or ""):
            phrase = match.lower()
            if phrase in _STOP or phrase in exclude:
                continue
            entities[phrase] += 2
    merged = counts + entities
    return [term for term, _ in merged.most_common(limit)]


def recall_deep(manager: Any, query: str, *,
                limit: int = 8,
                max_hops: int = 2,
                hop_decay: float = 0.6,
                **recall_kw: Any) -> DeepRecallResult:
    """Multi-hop recall: trusted hop 1, mined-term hop 2, decayed fusion.

    Hop 2 exists for the case the query didn't have the right words —
    the hop-1 hits teach the second query.  Fusion keeps hop-1 scores at
    full weight and hop-2 at ``hop_decay``; a record found in both keeps
    its best fused score.  Any failure degrades to hop 1 alone — the
    trusted path is the floor, never the casualty.
    """
    started = time.perf_counter()
    query = (query or "").strip()
    try:
        hop1 = manager.recall(query, limit=limit, **recall_kw)
    except Exception as exc:  # noqa: BLE001
        _log.warning("deep recall hop 1 failed: %s", exc)
        return DeepRecallResult(query=query, strategy="deep",
                                fallback=f"hop1 failed: {exc}")

    if max_hops < 2 or not hop1.records:
        return DeepRecallResult(
            records=list(hop1.records), query=query, hops=1,
            strategy="deep",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            fallback="single hop: no hits to expand from" if max_hops >= 2 else "")

    try:
        query_words = set(_WORD.findall(query.lower()))
        terms = _content_terms([r.content for r in hop1.records],
                               query_words)
        if not terms:
            raise ValueError("no expansion terms")
        hop2 = manager.recall(" ".join(terms), limit=limit * 2, **recall_kw)
    except Exception as exc:  # noqa: BLE001
        _log.debug("deep recall hop 2 failed (%s); hop 1 only", exc)
        return DeepRecallResult(
            records=list(hop1.records), query=query, hops=1,
            strategy="deep",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            fallback=f"hop2 failed: {exc}")

    fused: dict[str, tuple[Any, float]] = {}
    for record in hop1.records:
        fused[record.id] = (record, record.score)
    for record in hop2.records:
        boosted = record.score * hop_decay
        if record.id in fused:
            best = max(fused[record.id][1], boosted)
            fused[record.id] = (fused[record.id][0], best)
        else:
            fused[record.id] = (record, boosted)
    ordered = sorted(fused.values(), key=lambda pair: -pair[1])[:limit]
    return DeepRecallResult(
        records=[r for r, _ in ordered],
        query=query, hops=2, expansion_terms=terms, strategy="deep",
        elapsed_ms=(time.perf_counter() - started) * 1000.0)


# ── associative recall ───────────────────────────────────────────────────

def _shared_tags(a: Any, b: Any) -> set[str]:
    at = {t.strip() for t in (getattr(a, "tags", "") or "").split(",") if t.strip()}
    bt = {t.strip() for t in (getattr(b, "tags", "") or "").split(",") if t.strip()}
    # scope tags bind records to a space; they are not a topic signal here
    drop = {t for t in at | bt if t.startswith("scope:")}
    return (at & bt) - drop


def _entities(text: str) -> set[str]:
    return {m.lower() for m in _ENTITY.findall(text or "")}


def related(manager: Any, record_id: str, *,
            limit: int = 8) -> list[Any]:
    """Records associated with one record — the "what else connects" view.

    Strategy chain (each with its own weight, summed, not boolean):
    same scope +0.30 · shared tags +0.25/tag (cap 0.5) · shared entities
    +0.20/entity (cap 0.6) · same kind +0.15 · lexical overlap +0.10.
    The seed record itself is excluded.  Never raises.
    """
    try:
        seed = manager.get(record_id)
        if seed is None:
            return []
        seed_scope = scope_of(seed)
        seed_entities = _entities(seed.content)
        seed_words = set(_WORD.findall((seed.content or "").lower())) - _STOP

        pool = manager.recall(seed.content, limit=limit * 4)
        scored: list[tuple[float, Any]] = []
        for record in pool.records:
            if record.id == seed.id:
                continue
            score = 0.0
            if seed_scope and scope_of(record) == seed_scope:
                score += 0.30
            shared = _shared_tags(seed, record)
            score += min(0.5, 0.25 * len(shared))
            ent = seed_entities & _entities(record.content)
            score += min(0.6, 0.20 * len(ent))
            if record.kind == seed.kind:
                score += 0.15
            words = set(_WORD.findall((record.content or "").lower())) - _STOP
            if seed_words and words:
                overlap = len(seed_words & words) / max(len(seed_words), 1)
                score += 0.10 * min(1.0, overlap * 3)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: -pair[0])
        return [record for _, record in scored[:limit]]
    except Exception as exc:  # noqa: BLE001
        _log.debug("related() failed: %s", exc)
        return []


# ── temporal reasoning ───────────────────────────────────────────────────

def recall_window(manager: Any, query: str, *,
                  since: float | None = None,
                  until: float | None = None,
                  limit: int = 8,
                  **recall_kw: Any) -> Any:
    """Trusted recall filtered to a created_at window.

    ``since``/``until`` are unix timestamps (``None`` = open).  Over-fetches
    then filters, so the window is exact even though the index has no time
    dimension.  Returns the manager's own RecallResult type with the
    filtered records — the scoring is untouched.
    """
    result = manager.recall(query, limit=max(limit * 4, limit),
                            **recall_kw)
    if since is None and until is None:
        result.records = result.records[:limit]
        return result
    kept = [r for r in result.records
            if (since is None or r.created_at >= since)
            and (until is None or r.created_at <= until)]
    result.records = kept[:limit]
    return result


def timeline(manager: Any, topic: str, *,
             kinds: Sequence[str] = (MemoryKind.DECISION, MemoryKind.FACT),
             limit: int = 20,
             since: float | None = None,
             until: float | None = None) -> list[Any]:
    """How the belief about ``topic`` evolved — oldest first.

    "What did we decide last month about X" as a first-class query.
    Superseded records are *included* (``include_superseded=True``): a
    timeline that only shows the current belief is not a timeline, it is
    a snapshot.  Never raises.
    """
    try:
        result = manager.recall(
            topic, limit=max(limit * 4, 20),
            include_superseded=True,
            include_private=False)
        kept = [r for r in result.records
                if (not kinds or r.kind in kinds)
                and (since is None or r.created_at >= since)
                and (until is None or r.created_at <= until)]
        kept.sort(key=lambda r: r.created_at)
        return kept[:limit]
    except Exception as exc:  # noqa: BLE001
        _log.debug("timeline() failed: %s", exc)
        return []


def decisions_about(manager: Any, topic: str, *,
                    days: float = 30.0,
                    limit: int = 10,
                    now: float | None = None) -> list[Any]:
    """Decisions about ``topic`` in the last ``days`` days, newest first.

    The direct answer to "what did we decide last month about X".
    """
    moment = now if now is not None else time.time()
    items = timeline(manager, topic, kinds=(MemoryKind.DECISION,),
                     limit=limit, since=moment - days * 86400.0,
                     until=moment)
    items.sort(key=lambda r: -r.created_at)
    return items
