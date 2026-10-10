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

Variants supported (mined from the Okapi canon and the rank-bm25 family):

- field weighting (the BM25F pattern): ``title_weight`` multiplies the
  term frequencies contributed by the title, because a query term in the
  title is worth more than one buried in the snippet;
- BM25+ (``delta > 0``, typically 1.0): fixes BM25's tendency to push
  long-document scores toward zero;
- ``k3``: Okapi query-term saturation (``k3=None``/infinite disables it,
  the common default);
- ``tokenizer`` seam: pass your own tokenizer (stemming, stopwords,
  domain rules) instead of hand-rolling preprocessing outside.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle; model never imports here
    from .model import SearchResult

__all__ = [
    "tokenize",
    "bm25_scores",
    "bm25_rerank",
    "bm25_field_scores",
]

_TERM_RE = re.compile(r"[a-z0-9]+")

#: How much more a title term is worth than a snippet term (BM25F pattern).
DEFAULT_TITLE_WEIGHT = 2.0


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
    k3: float | None = None,
    delta: float = 0.0,
) -> list[float]:
    """Classic Okapi BM25 for tokenized docs against tokenized query terms.

    ``docs`` is one token list per document. Returns one score per doc.
    Empty query terms or empty corpus → all zeros (never NaN).

    ``delta`` > 0 switches to BM25+: the score term becomes
    ``idf * (tf * (k1 + 1)) / (tf + k1 * norm) + delta`` per query term,
    which keeps long documents from collapsing to zero. ``k3`` enables
    Okapi query-term saturation (``(k3 + 1) * qtf / (k3 + qtf)``); the
    default ``None`` treats it as infinite (no saturation — every query
    term counts once, the common case for short queries).
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
    # query term frequencies (for k3 saturation)
    qtf: dict[str, int] = {}
    for term in query_terms:
        if term in df:
            qtf[term] = qtf.get(term, 0) + 1
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
            term_score = idf * (freq * (k1 + 1.0)) / (freq + k1 * norm)
            if delta:
                term_score += delta * idf
            if k3 is not None and k3 > 0:
                qf = qtf.get(term, 1)
                term_score *= (k3 + 1.0) * qf / (k3 + qf)
            scores[i] += term_score
    return scores


def bm25_field_scores(
    query_terms: list[str],
    titles: list[list[str]],
    snippets: list[list[str]],
    *,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    k1: float = 1.5,
    b: float = 0.75,
    k3: float | None = None,
    delta: float = 0.0,
) -> list[float]:
    """Field-weighted BM25 (the BM25F pattern) over title + snippet fields.

    Term frequencies from the title are multiplied by ``title_weight``
    (default 2.0 — the industry-standard title boost) before the BM25
    saturation, so a title match outranks a snippet-only match without
    changing the formula's shape. ``titles``/``snippets`` are parallel
    token lists; ``len(titles) == len(snippets)`` is required.
    """
    if len(titles) != len(snippets):
        raise ValueError(
            f"titles ({len(titles)}) and snippets ({len(snippets)}) must align"
        )
    if not query_terms or not titles:
        return [0.0] * len(titles)
    n_docs = len(titles)
    # weighted term frequencies per doc: title tf * title_weight + snippet tf
    doc_lens = [len(t) * title_weight + len(s) for t, s in zip(titles, snippets)]
    avgdl = sum(doc_lens) / n_docs if n_docs else 0.0
    df: dict[str, int] = {}
    for term in set(query_terms):
        df[term] = sum(
            1 for t, s in zip(titles, snippets) if term in t or term in s
        )
    qtf: dict[str, int] = {}
    for term in query_terms:
        if term in df:
            qtf[term] = qtf.get(term, 0) + 1
    scores = [0.0] * n_docs
    for i, (title, snippet) in enumerate(zip(titles, snippets)):
        wtf: dict[str, float] = {}
        for t in title:
            if t in df:
                wtf[t] = wtf.get(t, 0.0) + title_weight
        for t in snippet:
            if t in df:
                wtf[t] = wtf.get(t, 0.0) + 1.0
        if not wtf:
            continue
        dl = doc_lens[i]
        norm = 1.0 - b + b * (dl / avgdl) if avgdl > 0 else 1.0
        for term, freq in wtf.items():
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            term_score = idf * (freq * (k1 + 1.0)) / (freq + k1 * norm)
            if delta:
                term_score += delta * idf
            if k3 is not None and k3 > 0:
                qf = qtf.get(term, 1)
                term_score *= (k3 + 1.0) * qf / (k3 + qf)
            scores[i] += term_score
    return scores


def bm25_rerank(
    results: list["SearchResult"],
    query: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    k3: float | None = None,
    delta: float = 0.0,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> list["SearchResult"]:
    """Re-rank one source's hits by BM25(query, title + snippet).

    Returns a NEW list sorted by BM25 score descending (ties break on
    title, so the order is deterministic). Each hit's ``raw_score`` is
    overwritten with its BM25 score — the caller is expected to have
    stashed the backend's native score in provenance first.

    Title matches are field-weighted by ``title_weight`` (the BM25F
    pattern — default 2.0); pass ``title_weight=1.0`` for the classic
    flat treatment. ``tokenizer`` overrides :func:`tokenize` (stemming,
    stopwords, domain rules); ``delta`` enables BM25+; ``k3`` enables
    Okapi query-term saturation.
    """
    tok = tokenizer or tokenize
    terms = tok(query)
    titles = [tok(r.title or "") for r in results]
    snippets = [tok(r.snippet or "") for r in results]
    scores = bm25_field_scores(
        terms, titles, snippets,
        title_weight=title_weight, k1=k1, b=b, k3=k3, delta=delta,
    )
    ranked = list(zip(results, scores))
    for hit, score in ranked:
        hit.raw_score = score
    ranked.sort(key=lambda pair: (-pair[1], pair[0].title or ""))
    return [hit for hit, _ in ranked]
