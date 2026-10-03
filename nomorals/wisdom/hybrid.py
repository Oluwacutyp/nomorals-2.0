"""Hybrid search: fuse keyword (BM25/FTS5) and semantic (vector) rankings.

Fusion is reciprocal rank fusion (Cormack et al., SIGIR 2009):

    score(d) = Σ 1 / (k + rank(d))

over each ranking that contains d. RRF is rank-based, not score-based,
so BM25 scores and cosine similarities never have to be calibrated
against each other — a classic source of hybrid-search brittleness.
Documents found by *both* paths get a natural boost, which is exactly
the behavior we want: lexical + semantic agreement is strong evidence.
"""
from __future__ import annotations

from typing import Any, Callable, Hashable


def reciprocal_rank_fusion(rankings: list[list[Hashable]],
                           k: int = 60) -> dict[Hashable, float]:
    """Fuse ranked key-lists into {key: rrf_score}.

    ``rankings`` is a list of orderings (best first). ``k`` dampens the
    influence of deep ranks; 60 is the literature standard.
    """
    fused: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def fuse_hits(keyword_hits: list[Any], semantic_hits: list[Any], *,
              key_fn: Callable[[Any], Hashable],
              top: int = 8, k: int = 60) -> list[Any]:
    """Merge two hit lists (best-first) into one best-first list via RRF.

    Order within the result follows the fused score. Hits present in
    both lists appear once — the keyword-side object wins the tie, since
    it carries the richer snippet context; the semantic score is folded
    into the fused ordering only.
    """
    key_rank_kw = [key_fn(h) for h in keyword_hits]
    key_rank_sem = [key_fn(h) for h in semantic_hits]
    scores = reciprocal_rank_fusion([key_rank_kw, key_rank_sem], k=k)
    by_key: dict[Hashable, Any] = {}
    for h in keyword_hits:
        by_key.setdefault(key_fn(h), h)
    for h in semantic_hits:
        by_key.setdefault(key_fn(h), h)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [by_key[key] for key in ordered[:max(top, 0)] if key in by_key]
