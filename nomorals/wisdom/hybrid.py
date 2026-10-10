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
    return weighted_rrf(rankings, k=k)


def weighted_rrf(rankings: list[list[Hashable]], *,
                 weights: list[float] | None = None,
                 k: int = 60) -> dict[Hashable, float]:
    """RRF with a per-ranking weight.

    ``weights[i]`` scales ranking ``i``'s contribution. Weaviate-style
    hybrid practice starts around 0.7 vector / 0.3 keyword for general
    text and raises the keyword weight for precise-terminology domains
    (ours: religious terminology rewards exact matches, so keyword
    weight is worth raising). Default weights are all 1.0 — plain RRF.
    """
    weights = list(weights) if weights else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights length {len(weights)} != rankings length "
            f"{len(rankings)}")
    fused: dict[Hashable, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
    return fused


def fuse_hits(keyword_hits: list[Any], semantic_hits: list[Any], *,
              key_fn: Callable[[Any], Hashable],
              top: int = 8, k: int = 60,
              weights: tuple[float, float] = (1.0, 1.0),
              semantic_floor: float = 0.0) -> list[Any]:
    """Merge two hit lists (best-first) into one best-first list via RRF.

    Order within the result follows the fused score. Hits present in
    both lists appear once — the keyword-side object wins the tie, since
    it carries the richer snippet context; the semantic score is folded
    into the fused ordering only.

    ``weights`` is (keyword_weight, semantic_weight) — the Weaviate
    alpha knob. ``semantic_floor`` drops semantic hits below a cosine
    similarity *before* fusion, so the vector half can't smuggle "top K
    of anything" into the fused ranking.

    The winning hit object gets ``rrf_score`` (fused) plus
    ``keyword_rank`` / ``semantic_rank`` attributes (None when the hit
    came from only one side) — score transparency for display.
    """
    if semantic_floor > 0:
        semantic_hits = [h for h in semantic_hits
                         if getattr(h, "score", 0.0) >= semantic_floor]
    key_rank_kw = [key_fn(h) for h in keyword_hits]
    key_rank_sem = [key_fn(h) for h in semantic_hits]
    scores = weighted_rrf([key_rank_kw, key_rank_sem],
                          weights=list(weights), k=k)
    rank_kw = {key: i + 1 for i, key in enumerate(key_rank_kw)}
    rank_sem = {key: i + 1 for i, key in enumerate(key_rank_sem)}
    by_key: dict[Hashable, Any] = {}
    for h in keyword_hits:
        by_key.setdefault(key_fn(h), h)
    for h in semantic_hits:
        by_key.setdefault(key_fn(h), h)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    out = []
    for key in ordered[:max(top, 0)]:
        hit = by_key.get(key)
        if hit is None:
            continue
        try:
            hit.rrf_score = scores[key]
            hit.keyword_rank = rank_kw.get(key)
            hit.semantic_rank = rank_sem.get(key)
        except AttributeError:
            pass  # exotic hit objects stay untouched
        out.append(hit)
    return out
