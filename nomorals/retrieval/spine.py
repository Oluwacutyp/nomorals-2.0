"""One retrieval spine (L3): embed → index → fuse → rerank.

Finding 2 fix. Before this, retrieval lived in six files with per-organ
variants (storage/vectors, memory/vector_backends, memory/embeddings,
wisdom/embeddings, wisdom/vectorstore, books/library FTS5, two RRF
fusers) and Wisdom never imported the storage or memory mechanisms.

This module is the single spine. Organs supply *corpora* (a list of docs,
a books.Library, a CanonCorpus, anything with a ``search`` method) —
never mechanisms. Embedding backends are resolved lazily with a
deterministic hash fallback, so the spine works with zero optional
dependencies and upgrades to a real model backend when one is present.

Pipeline::

    embed  →  keyword (BM25-lite / FTS) + semantic (cosine KNN)
           →  RRF fuse (with per-source weights + evidence transparency)
           →  MMR rerank (relevance vs novelty) + similarity floor

Every hit carries ``scores`` (per-source scores + fused score) and
``provenance`` (where the passage came from), so callers can cite, not
just rank.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "Hit", "HashEmbedder", "RetrievalSpine",
    "rrf_fuse", "mmr_rerank", "cosine_sim", "tokenize",
]

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an and or of to in on at for with is was were be been are "
    "had has have it its they them he she his her we you your this that "
    "as by from".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed."""
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 on empty / zero vectors (never NaN)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


@dataclass
class Hit:
    """One fused, ranked result with full score transparency."""

    key: str
    text: str
    score: float = 0.0
    source: str = ""  # "keyword" | "semantic" | "fused"
    scores: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "text": self.text,
            "score": round(self.score, 4), "source": self.source,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "provenance": self.provenance, "rank": self.rank,
        }


def rrf_fuse(rankings: list[list[str]], *, k: int = 60,
             weights: list[float] | None = None) -> dict[str, float]:
    """Reciprocal-rank fusion. ``rankings`` = per-source ordered key lists.

    Score(key) = Σ_source weight_source / (k + rank_source(key)).
    Keys absent from a source contribute 0 from that source.
    """
    w = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, w):
        for rank, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + weight / (k + rank + 1)
    return fused


def mmr_rerank(items: list[Hit], *, top: int = 5,
               lambda_: float = 0.6) -> list[Hit]:
    """Maximal-marginal-relevance selection over already-scored hits.

    Picks ``lambda_ * relevance - (1 - lambda_) * max_sim_to_selected``
    with token-Jaccard similarity, so near-duplicates stop crowding out
    coverage. Deterministic: ties break by original score, then key.
    """
    if not items or top <= 0:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_!r}")
    token_sets = {h.key: set(tokenize(h.text)) for h in items}

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    remaining = sorted(items, key=lambda h: (-h.score, h.key))
    selected: list[Hit] = [remaining.pop(0)]
    while remaining and len(selected) < top:
        best: Hit | None = None
        best_val = float("-inf")
        for cand in remaining:
            max_sim = max(_jaccard(token_sets[cand.key],
                                   token_sets[s.key]) for s in selected)
            val = lambda_ * cand.score - (1.0 - lambda_) * max_sim
            if val > best_val or (val == best_val and best is not None
                                  and (cand.score, cand.key)
                                  > (best.score, best.key)):
                best, best_val = cand, val
        assert best is not None
        remaining.remove(best)
        selected.append(best)
    return selected


class HashEmbedder:
    """Deterministic dependency-free embedder (fallback, not a model).

    Token-hashed bag-of-words projected to ``dim`` dims, L2-normalized.
    Similarity is meaningful for near-duplicate / topical overlap, and
    the interface matches the real backends so the spine never branches
    on backend type. Real backends (wisdom/embeddings) are preferred
    whenever importable — see :meth:`RetrievalSpine._resolve_embedder`.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.name = "hash"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
                vec[(h >> 16) % self.dim] += 0.5
            norm = math.sqrt(sum(x * x for x in vec))
            out.append([x / norm for x in vec] if norm else vec)
        return out


class _DocCorpus:
    """In-memory corpus adapter over a list of ``{id, text, ...}`` docs."""

    def __init__(self, docs: list[dict[str, Any]], name: str = "docs") -> None:
        self.name = name
        self.docs = docs
        # IDF over the corpus, computed once.
        df: dict[str, int] = {}
        for d in docs:
            for tok in set(tokenize(d.get("text", ""))):
                df[tok] = df.get(tok, 0) + 1
        n = max(1, len(docs))
        self._idf = {t: math.log(1.0 + n / (1.0 + c)) for t, c in df.items()}

    def keyword(self, query: str, *, top: int) -> list[Hit]:
        q = tokenize(query)
        scored: list[Hit] = []
        for d in self.docs:
            toks = tokenize(d.get("text", ""))
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            score = sum(self._idf.get(t, 0.0)
                        * (counts.get(t, 0) / max(1, len(toks)))
                        for t in set(q))
            if score > 0:
                scored.append(Hit(
                    key=str(d.get("id", "")),
                    text=d.get("text", ""), score=score,
                    source="keyword",
                    scores={"keyword": score},
                    provenance={k: v for k, v in d.items()
                                if k not in ("id", "text")},
                ))
        scored.sort(key=lambda h: (-h.score, h.key))
        return scored[:top]

    def texts(self) -> list[tuple[str, str]]:
        return [(str(d.get("id", "")), d.get("text", "")) for d in self.docs]


class RetrievalSpine:
    """Embed → index → fuse → rerank, over any corpus shape.

    A corpus is one of:
    - a list of ``{id, text, ...}`` dicts (in-memory, BM25-lite + cosine);
    - an object with ``search(query, top=...)`` (books.Library shape);
    - an object with ``ask(query, top=...)`` (CanonCorpus shape);
    - an object with ``keyword(query, top=...)`` / ``semantic(...)``.
    """

    def __init__(self, embedder: Any = None) -> None:
        self._embedder = embedder  # resolved lazily
        self._embed_cache: dict[str, list[float]] = {}

    # ── embedding ────────────────────────────────────────────────────
    def _resolve_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        # Layering: L0 retrieval must not import L5 wisdom embeddings.
        # Default is the hash fallback; callers inject a real embedder.
        self._embedder = HashEmbedder()
        return self._embedder

    def embed_query(self, text: str) -> list[float]:
        return self.embed_docs([text])[0]

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        backend = self._resolve_embedder()
        out: list[list[float]] = []
        missing: list[int] = []
        for i, text in enumerate(texts):
            key = hashlib.sha256(
                f"{getattr(backend, 'name', type(backend).__name__)}"
                f"|{text}".encode()).hexdigest()
            if key in self._embed_cache:
                out.append(self._embed_cache[key])
            else:
                out.append([])
                missing.append(i)
        if missing:
            if isinstance(backend, HashEmbedder):
                vecs = backend.embed([texts[i] for i in missing])
            elif hasattr(backend, "embed_query") and len(missing) == 1:
                vecs = [backend.embed_query(texts[missing[0]])]
            elif hasattr(backend, "embed_documents"):
                vecs = backend.embed_documents([texts[i] for i in missing])
            elif hasattr(backend, "embed"):
                vecs = backend.embed([texts[i] for i in missing])
            else:  # pragma: no cover - unknown backend shape
                vecs = HashEmbedder().embed([texts[i] for i in missing])
            for i, vec in zip(missing, vecs):
                key = hashlib.sha256(
                    f"{getattr(backend, 'name', type(backend).__name__)}"
                    f"|{texts[i]}".encode()).hexdigest()
                self._embed_cache[key] = vec
                out[i] = vec
        return out

    # ── per-source search ────────────────────────────────────────────
    def keyword_search(self, corpus: Any, query: str, *,
                       top: int = 10) -> list[Hit]:
        """Lexical search over any corpus shape."""
        if isinstance(corpus, list):
            return _DocCorpus(corpus).keyword(query, top=top)
        if hasattr(corpus, "keyword"):
            hits = corpus.keyword(query, top=top) or []
            return [h if isinstance(h, Hit) else self._coerce(h)
                    for h in hits]
        if hasattr(corpus, "search"):
            raw = corpus.search(query, top=top) or []
            return [self._coerce(h, source="keyword") for h in raw]
        if hasattr(corpus, "ask"):
            ans = corpus.ask(query, top=top, mode="keyword")
            return [self._coerce(h, source="keyword")
                    for h in getattr(ans, "passages", [])]
        raise TypeError(f"corpus {type(corpus).__name__!r} has no usable "
                        "keyword search (list | search() | ask() | keyword())")

    def semantic_search(self, corpus: Any, query: str, *,
                        top: int = 10,
                        min_score: float = 0.0) -> list[Hit]:
        """Dense cosine search. ``min_score`` is the similarity floor:
        hits below it are dropped, never returned as 'top K of anything'."""
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {min_score!r}")
        if hasattr(corpus, "semantic") and not isinstance(corpus, list):
            hits = corpus.semantic(query, top=top,
                                   min_score=min_score) or []
            return [h if isinstance(h, Hit) else self._coerce(h)
                    for h in hits]
        if isinstance(corpus, list):
            adapter = _DocCorpus(corpus)
            pairs = adapter.texts()
            if not pairs:
                return []
            qvec = self.embed_query(query)
            vecs = self.embed_docs([t for _, t in pairs])
            scored = [Hit(key=k, text=t,
                          score=cosine_sim(qvec, v), source="semantic",
                          scores={"semantic": cosine_sim(qvec, v)},
                          provenance={})
                      for (k, t), v in zip(pairs, vecs)]
            scored = [h for h in scored if h.score >= min_score]
            scored.sort(key=lambda h: (-h.score, h.key))
            return scored[:top]
        if hasattr(corpus, "ask"):
            ans = corpus.ask(query, top=top, mode="semantic",
                             min_score=min_score)
            return [self._coerce(h, source="semantic")
                    for h in getattr(ans, "passages", [])]
        raise TypeError(f"corpus {type(corpus).__name__!r} has no usable "
                        "semantic search")

    # ── fused search ─────────────────────────────────────────────────
    def search(self, query: str, corpora: Any, *,
               top: int = 5, mode: str = "hybrid",
               min_score: float = 0.0, diversify: bool = False,
               mmr_lambda: float = 0.6,
               weights: tuple[float, float] = (1.0, 1.0),
               ) -> list[Hit]:
        """Full spine: embed → keyword + semantic → RRF fuse → rerank.

        ``corpora`` is one corpus or a list of them; hits from several
        corpora fuse into one ranking keyed by ``source:key``.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("search() needs a non-empty query")
        if mode not in ("keyword", "semantic", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        if isinstance(corpora, (list,)) and corpora and isinstance(
                corpora[0], dict):
            corpora = [corpora]  # a single doc-list corpus, not a list of corpora
        if not isinstance(corpora, list):
            corpora = [corpora]
        kw_hits: list[Hit] = []
        sem_hits: list[Hit] = []
        for corpus in corpora:
            if mode in ("keyword", "hybrid"):
                kw_hits.extend(self.keyword_search(corpus, query,
                                                   top=top * 3))
            if mode in ("semantic", "hybrid"):
                sem_hits.extend(self.semantic_search(corpus, query,
                                                     top=top * 3,
                                                     min_score=min_score))
        if mode == "keyword":
            ranked = sorted(kw_hits, key=lambda h: (-h.score, h.key))
            fused = ranked
        elif mode == "semantic":
            fused = sorted(sem_hits, key=lambda h: (-h.score, h.key))
        else:
            fused = self._fuse(kw_hits, sem_hits, weights=weights)
        if diversify:
            fused = mmr_rerank(fused, top=top, lambda_=mmr_lambda)
        else:
            fused = fused[:top]
        for i, h in enumerate(fused, 1):
            h.rank = i
        return fused

    def _fuse(self, kw: list[Hit], sem: list[Hit],
              *, weights: tuple[float, float]) -> list[Hit]:
        by_key: dict[str, Hit] = {}
        for h in kw:
            by_key.setdefault(h.key, Hit(
                key=h.key, text=h.text, source="fused",
                provenance=h.provenance)).scores["keyword"] = h.score
            by_key[h.key].text = by_key[h.key].text or h.text
            by_key[h.key].provenance = (
                by_key[h.key].provenance or h.provenance)
        for h in sem:
            slot = by_key.setdefault(h.key, Hit(
                key=h.key, text=h.text, source="fused",
                provenance=h.provenance))
            slot.scores["semantic"] = h.score
            slot.text = slot.text or h.text
            slot.provenance = slot.provenance or h.provenance
        kw_rank = [h.key for h in sorted(
            kw, key=lambda h: (-h.score, h.key))]
        sem_rank = [h.key for h in sorted(
            sem, key=lambda h: (-h.score, h.key))]
        fused_scores = rrf_fuse([kw_rank, sem_rank], weights=list(weights))
        out = []
        for key, score in fused_scores.items():
            hit = by_key[key]
            hit.score = score
            hit.scores["fused"] = score
            out.append(hit)
        out.sort(key=lambda h: (-h.score, h.key))
        return out

    @staticmethod
    def _coerce(raw: Any, *, source: str = "") -> Hit:
        if isinstance(raw, Hit):
            return raw
        if hasattr(raw, "to_dict"):
            d = raw.to_dict()
        elif isinstance(raw, dict):
            d = raw
        else:
            d = {"text": str(raw)}
        text = (d.get("passage") or d.get("snippet") or d.get("text") or "")
        key = str(d.get("book_slug") or d.get("work") or d.get("key")
                  or d.get("book") or text[:40])
        score = float(d.get("score", 0.0) or 0.0)
        return Hit(key=key, text=text, score=score,
                   source=source or str(d.get("source", "")),
                   scores={source: score} if source else {},
                   provenance={k: v for k, v in d.items()
                               if k not in ("passage", "snippet",
                                            "text", "score")})


def adapt_library(library: Any) -> dict[str, Callable[..., Any]]:
    """Expose a books.Library as spine-compatible callables (no subclass)."""
    return {
        "search": lambda q, top=10: library.search(q, top=top),
    }


def adapt_corpus(corpus: Any) -> dict[str, Callable[..., Any]]:
    """Expose a CanonCorpus as spine-compatible callables (no subclass)."""
    return {
        "ask": lambda q, top=5, mode="keyword", **kw: corpus.ask(
            q, top=top, mode=mode, **kw),
    }
