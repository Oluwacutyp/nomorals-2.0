"""Shared result model for federated search.

``SearchResult`` is the one shape every source adapter maps into.
``source`` names the subsystem that produced the hit; ``provenance`` is
the source's own provenance passed through verbatim — never flattened
away, so a wisdom hit still carries work/translator/url/canon_status and
a memory hit still carries its kind/tags.

Ranking scheme (also documented on :func:`federated_search`):
1. Each source scores in its own native scale (FTS5 BM25 negatives, raw
   term frequencies, cosine-ish similarities...), so raw scores are
   min-max normalized *per source* into [0, 1]. A source with a single
   hit normalizes to 1.0.
2. Hits are sorted by normalized score, descending.
3. Ties break on canonical source order, then on newer timestamps first
   (undated hits sort last), then on title — so ordering is fully
   deterministic.

Deduplication: hits whose normalized (title, snippet) content hash
matches are duplicates; the higher-scoring hit is kept. Cross-source
duplicates (e.g. a wisdom passage also indexed as a book passage) keep
the winner's provenance only — the loser is counted in
``SearchResponse.deduped``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def content_hash(title: str, snippet: str) -> str:
    """Stable identity for dedupe: normalized title + snippet."""
    normalized = " ".join(f"{title}\n{snippet}".lower().split())
    return hashlib.sha1(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass
class SearchResult:
    """One hit from one source, in the federated shape."""

    query: str
    title: str
    snippet: str
    source: str  # books | docs | memory | wisdom | timeline | code
    type: str  # book | doc | memory | passage | event | code
    score: float = 0.0  # normalized 0..1 (per-source min-max)
    raw_score: float = 0.0  # the source's native score, for transparency
    provenance: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None  # epoch seconds, when the source knows it
    source_id: str = ""  # stable per-source id (source + native id)

    def dedupe_key(self) -> str:
        return content_hash(self.title, self.snippet)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
            "type": self.type,
            "score": round(self.score, 4),
            "raw_score": self.raw_score,
            "provenance": dict(self.provenance),
            "timestamp": self.timestamp,
            "source_id": self.source_id,
        }


@dataclass
class SearchResponse:
    """The merged answer to one federated query."""

    query: str
    hits: list[SearchResult] = field(default_factory=list)
    sources_searched: list[str] = field(default_factory=list)
    sources_skipped: dict[str, str] = field(default_factory=dict)
    deduped: int = 0

    @property
    def total(self) -> int:
        return len(self.hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "total": self.total,
            "deduped": self.deduped,
            "sources_searched": list(self.sources_searched),
            "sources_skipped": dict(self.sources_skipped),
            "hits": [h.to_dict() for h in self.hits],
        }


def normalize_scores(results: list[SearchResult]) -> None:
    """Min-max normalize ``raw_score`` into ``score`` in place (per source batch).

    A batch with a single hit (or identical scores) normalizes to 1.0 —
    the hit was that source's best answer.
    """
    if not results:
        return
    lo = min(r.raw_score for r in results)
    hi = max(r.raw_score for r in results)
    if hi == lo:
        for r in results:
            r.score = 1.0
        return
    span = hi - lo
    for r in results:
        r.score = (r.raw_score - lo) / span


def dedupe_results(results: list[SearchResult]) -> tuple[list[SearchResult], int]:
    """Drop content duplicates, keeping the highest-scoring hit.

    Returns ``(kept, dropped_count)``.
    """
    best: dict[str, SearchResult] = {}
    dropped = 0
    for r in results:
        key = r.dedupe_key()
        prev = best.get(key)
        if prev is None:
            best[key] = r
        else:
            dropped += 1
            if r.score > prev.score:
                best[key] = r
    return list(best.values()), dropped


def rank_results(
    results: list[SearchResult], source_order: list[str]
) -> list[SearchResult]:
    """Sort by normalized score desc; ties break deterministically."""
    order = {name: i for i, name in enumerate(source_order)}

    def _key(r: SearchResult) -> tuple[float, int, float, str]:
        ts = r.timestamp if r.timestamp is not None else float("-inf")
        return (-r.score, order.get(r.source, 999), -ts, r.title)

    return sorted(results, key=_key)
