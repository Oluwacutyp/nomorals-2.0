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
    #: per-source wall-clock seconds (search only, probes excluded). Additive
    #: metadata for operators — every metasearch reports per-engine latency.
    timings: dict[str, float] = field(default_factory=dict)
    #: total wall-clock seconds for the whole federated call.
    elapsed: float = 0.0

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
            "timings": dict(self.timings),
            "elapsed": self.elapsed,
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


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> tuple[list[SearchResult], int]:
    """Fuse per-source ranked lists with reciprocal rank fusion (RRF).

    Each input list must already be ordered best-first (rank 1 = first
    element); a hit appearing at rank ``r`` in a source contributes
    ``weight * (1 / (k + r))`` to its fused score, summed across every
    source that returned it. Weights default to 1.0 per list — the plain
    Cormack/Clarke/Büttcher form; per-list weights are the Elasticsearch
    weighted-RRF extension, letting a trusted engine count more than a
    long-tail one. Cross-source content duplicates (same
    :meth:`SearchResult.dedupe_key`) fuse into one hit — the kept
    representative is the one with the strongest single-source
    contribution, and its ``score`` is set to the fused total.

    Each hit's ``provenance["source_rank"]`` records its 1-based rank in
    its own source list (additive metadata for debugging and second-stage
    rerankers — never overwritten when already set).

    Unlike per-source min-max normalization, RRF scores are comparable
    *across* sources without assuming anything about the native scales,
    which is why it is the fusion mode for heterogeneous web backends.

    Returns ``(fused, deduped_count)`` where ``deduped_count`` is the
    number of duplicate hits folded away. The fused list is sorted by
    fused score descending; ties break on title (fully deterministic —
    source-order tie-breaking happens downstream in :func:`rank_results`).
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(
            f"RRF weights length {len(weights)} != ranked_lists length "
            f"{len(ranked_lists)}"
        )
    if weights is not None and any(w < 0 for w in weights):
        raise ValueError(f"RRF weights must be non-negative, got {weights}")
    fused: dict[str, float] = {}
    best: dict[str, tuple[SearchResult, float]] = {}
    total = 0
    for list_idx, hits in enumerate(ranked_lists):
        weight = 1.0 if weights is None else weights[list_idx]
        for rank, hit in enumerate(hits, start=1):
            total += 1
            key = hit.dedupe_key()
            contrib = weight * (1.0 / (k + rank))
            fused[key] = fused.get(key, 0.0) + contrib
            hit.provenance.setdefault("source_rank", rank)
            prev = best.get(key)
            if prev is None or contrib > prev[1]:
                best[key] = (hit, contrib)
    out: list[SearchResult] = []
    for key, (hit, _contrib) in best.items():
        hit.score = fused[key]
        out.append(hit)
    out.sort(key=lambda h: (-h.score, h.title or ""))
    return out, total - len(out)
