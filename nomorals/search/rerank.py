"""Local re-ranking for federated search: dependency-free BM25.

Web backends (SearXNG aggregates, DDG snippets, API snippets) each order
by their own opaque relevance. BM25 over the query terms against
``title + snippet`` re-orders one source's hits by plain lexical
relevance — no model download, no API key, pure stdlib. It is the
"re-rank" half of the upgrade (the other half is
:func:`~nomorals.search.model.reciprocal_rank_fusion` across sources).

``bm25_rerank`` overwrites ``raw_score`` with the BM25 score; the
backend's native score is expected to be stashed in
``provenance["backend_score"]`` by the caller before re-ranking, so the
native signal is never silently lost.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle; model never imports here
    from .model import SearchResult

__all__ = ["tokenize", "bm25_scores", "bm25_rerank"]

_TERM_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric terms, dropping single characters."""
    return [t for t in _TERM_RE.findall((text or "").lower()) if len(t) >= 2]


def _doc_text(hit: "SearchResult") -> str:
    return f"{hit.title or ''} {hit.snippet or ''}"


def bm25_scores(
    query_terms: list[str],
    docs: list[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Classic Okapi BM25 for tokenized docs against tokenized query terms.

    ``docs`` is one token list per document. Returns one score per doc.
    Empty query terms or empty corpus → all zeros (never NaN).
    """
    n_docs = len(docs)
    if n_docs == 0 or not query_terms:
        return [0.0] * n_docs
    doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / n_docs if n_docs else 0.0
    # document frequency per query term
    df: dict[str, int] = {}
    for term in set(query_terms):
        df[term] = sum(1 for d in docs if term in d)
    scores = [0.0] * n_docs
    for i, doc in enumerate(docs):
        if not doc:
            continue
        tf: dict[str, int] = {}
        for t in doc:
            if t in df:
                tf[t] = tf.get(t, 0) + 1
        dl = doc_lens[i]
        norm = 1.0 - b + b * (dl / avgdl) if avgdl > 0 else 1.0
        for term, freq in tf.items():
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            scores[i] += idf * (freq * (k1 + 1.0)) / (freq + k1 * norm)
    return scores


def bm25_rerank(
    results: list["SearchResult"],
    query: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list["SearchResult"]:
    """Re-rank one source's hits by BM25(query, title + snippet).

    Returns a NEW list sorted by BM25 score descending (ties break on
    title, so the order is deterministic). Each hit's ``raw_score`` is
    overwritten with its BM25 score — the caller is expected to have
    stashed the backend's native score in provenance first.
    """
    terms = tokenize(query)
    docs = [tokenize(_doc_text(r)) for r in results]
    scores = bm25_scores(terms, docs, k1=k1, b=b)
    ranked = list(zip(results, scores))
    for hit, score in ranked:
        hit.raw_score = score
    ranked.sort(key=lambda pair: (-pair[1], pair[0].title or ""))
    return [hit for hit, _ in ranked]
