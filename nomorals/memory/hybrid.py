"""Hybrid retrieval: vector + BM25 fused with RRF (logseq-composer pattern).

Vector-only retrieval is a single signal — vectors blur exact identifiers
("Ada", "2026-10-14") while BM25 catches them. Fusing both with
Reciprocal Rank Fusion gives precision over vibes.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Hit",
    "HybridMemoryIndex",
    "detect_entity_type",
    "hybrid_search",
    "mmr_select",
    "rrf_fuse",
    "type_aware_search",
]

#: Standard RRF constant.
RRF_K = 60


@dataclass
class Hit:
    """One fused retrieval hit with per-signal scores for debugging."""
    id: str
    text: str
    rrf_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    source: str = ""  # e.g. "fact", "event", "entity"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text,
                "rrf_score": round(self.rrf_score, 4),
                "vector_rank": self.vector_rank,
                "bm25_rank": self.bm25_rank, "source": self.source}


def rrf_fuse(vector_ids: Sequence[str], bm25_ids: Sequence[str],
             k: int = RRF_K, *,
             vector_weight: float = 1.0,
             bm25_weight: float = 1.0) -> list[tuple[str, float, int | None, int | None]]:
    """Reciprocal Rank Fusion: score = Σ weight/(k + rank) per signal.

    Returns (id, fused_score, vector_rank, bm25_rank) sorted by score desc,
    ties broken on id for determinism. Lane weights let the caller lean on
    the signal it trusts more for a query (e.g. BM25 up for identifier-
    heavy queries like "Ada" or "2026-10-14").
    """
    scores: dict[str, float] = {}
    vranks: dict[str, int] = {}
    branks: dict[str, int] = {}
    for rank, uid in enumerate(vector_ids):
        scores[uid] = scores.get(uid, 0.0) + vector_weight / (k + rank)
        vranks.setdefault(uid, rank)
    for rank, uid in enumerate(bm25_ids):
        scores[uid] = scores.get(uid, 0.0) + bm25_weight / (k + rank)
        branks.setdefault(uid, rank)
    return sorted(
        ((uid, s, vranks.get(uid), branks.get(uid))
         for uid, s in scores.items()),
        key=lambda t: (-t[1], t[0]))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0.0 or db == 0.0:
        return 0.0
    return max(-1.0, min(1.0, num / (da * db)))


def mmr_select(
    query_vector: Sequence[float],
    candidates: Sequence[Hit],
    vectors: dict[str, Sequence[float]],
    *,
    limit: int = 10,
    lambda_mult: float = 0.6,
) -> list[Hit]:
    """Maximal Marginal Relevance: relevance − redundancy.

    ``lambda_mult`` ∈ [0, 1]: 1.0 = pure relevance (top-k by query
    similarity), 0.0 = pure diversity. 0.6 is the production default —
    recall sets stop being five paraphrases of the same memory and start
    covering the topic. Candidates without a vector keep their RRF order
    (never dropped for a missing embedding).
    """
    if not candidates or limit <= 0:
        return []
    lam = max(0.0, min(1.0, lambda_mult))
    with_vec = [(h, vectors.get(h.id)) for h in candidates]
    rel = {h.id: _cosine(query_vector, v) if v else 0.0
           for h, v in with_vec}
    selected: list[Hit] = []
    remaining = list(with_vec)
    # Seed with the best hit so an empty selection never stalls.
    remaining.sort(key=lambda hv: (-rel[hv[0].id], hv[0].id))
    while remaining and len(selected) < limit:
        best: Hit | None = None
        best_score = float("-inf")
        for hit, vec in remaining:
            if vec is None:
                # No vector — judge on RRF rank alone.
                score = lam * 0.5 - (1.0 - lam) * 0.0
                # Fall back to position: earlier RRF rank wins.
                score += 1e-6 * -candidates.index(hit)
            else:
                redundancy = max(
                    (_cosine(vec, vectors[s.id]) for s in selected
                     if s.id in vectors), default=0.0)
                score = lam * rel[hit.id] - (1.0 - lam) * redundancy
            if score > best_score:
                best, best_score = hit, score
        if best is None:
            break
        selected.append(best)
        remaining = [(h, v) for h, v in remaining if h.id != best.id]
    return selected


def hybrid_search(
    query: str,
    *,
    vector_fn: Callable[[str, int], Sequence[tuple[str, str]]],
    bm25_fn: Callable[[str, int], Sequence[tuple[str, str]]],
    limit: int = 10,
    source: str = "",
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    mmr: Callable[[list[Hit]], list[Hit]] | None = None,
) -> list[Hit]:
    """Fuse vector and BM25 results with RRF (+ optional MMR).

    ``vector_fn``/``bm25_fn`` each take (query, limit) and return
    ``(id, text)`` pairs in rank order. Either may raise — a dead signal
    degrades to the other one, never to empty silence... unless both die,
    in which case we return [] and log.

    ``vector_weight``/``bm25_weight`` tilt the fusion lane by lane.
    ``mmr`` is an optional post-pass ``hits -> hits`` (e.g. a partial
    applying :func:`mmr_select` with the query vector) that diversifies
    the fused ranking.
    """
    texts: dict[str, str] = {}
    try:
        vhits = list(vector_fn(query, limit * 2))
    except Exception:  # noqa: BLE001
        _log.warning("hybrid vector signal failed", exc_info=True)
        vhits = []
    try:
        bhits = list(bm25_fn(query, limit * 2))
    except Exception:  # noqa: BLE001
        _log.warning("hybrid BM25 signal failed", exc_info=True)
        bhits = []
    for uid, text in vhits + bhits:
        texts.setdefault(uid, text)
    fused = rrf_fuse([u for u, _ in vhits], [u for u, _ in bhits],
                     vector_weight=vector_weight, bm25_weight=bm25_weight)
    hits = [Hit(id=uid, text=texts[uid], rrf_score=s,
                vector_rank=vr, bm25_rank=br, source=source)
            for uid, s, vr, br in fused[:limit]]
    if mmr is not None and hits:
        try:
            return list(mmr(hits))[:limit]
        except Exception:  # noqa: BLE001 — MMR is garnish, never load-bearing
            _log.debug("hybrid MMR pass failed; keeping RRF order",
                       exc_info=True)
    return hits


# ── type-aware search ─────────────────────────────────────────────────────────

_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("person", re.compile(
        r"\b(persons?|people|who|someone|guy|woman|man|girl|friend|"
        r"colleague|contact)\b", re.I)),
    ("project", re.compile(
        r"\b(projects?|repos?|repository|codebase|build)\b", re.I)),
    ("place", re.compile(
        r"\b(places?|where|location|address|city|restaurant|office)\b", re.I)),
    ("commitment", re.compile(
        r"\b(commitments?|promise|owe|deadline|due|todos?|tasks?)\b", re.I)),
    ("preference", re.compile(
        r"\b(prefer|preference|like|favorite|favourite)\b", re.I)),
    ("habit", re.compile(
        r"\b(habit|routine|daily|streak)\b", re.I)),
]


def detect_entity_type(query: str) -> str | None:
    """Detect an entity type hint in the query. None when ambiguous."""
    hits = [etype for etype, rx in _TYPE_PATTERNS if rx.search(query or "")]
    return hits[0] if len(hits) == 1 else None


def type_aware_search(
    query: str,
    hits: Sequence[Hit],
    entities: dict[str, str],
) -> list[Hit]:
    """Filter fused hits to the query's detected entity type.

    ``entities`` maps hit id → entity type. No detected type → hits pass
    through unchanged.
    """
    etype = detect_entity_type(query)
    if etype is None:
        return list(hits)
    return [h for h in hits if entities.get(h.id) == etype]


# ── BM25 sidecar over memory units ────────────────────────────────────────────

class HybridMemoryIndex:
    """A BM25 (FTS5) mirror of memory units for the keyword lane.

    Lazily built from facts/events/entities; units added after the build
    are indexed incrementally. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._index: Any = None
        self._indexed: set[str] = set()

    def _ensure(self) -> Any:
        if self._index is None:
            from ..documents.index import DocumentIndex
            from ..documents.model import Document
            self._index = DocumentIndex()
            self._Document = Document
        return self._index

    def index_unit(self, unit_id: str, title: str, text: str) -> None:
        """Add or replace one memory unit in the BM25 index."""
        with self._lock:
            idx = self._ensure()
            doc = self._Document(id=unit_id, title=title, format="memory")
            from ..documents.model import Section
            doc.sections = [Section(heading=title, text=text)]
            try:
                idx.add(doc)
            except Exception:  # noqa: BLE001
                _log.warning("hybrid index add failed for %s", unit_id,
                             exc_info=True)
                return
            self._indexed.add(unit_id)

    def bm25_search(self, query: str,
                    limit: int = 10) -> list[tuple[str, str]]:
        """(id, text) pairs in BM25 rank order. [] when FTS5 is missing."""
        with self._lock:
            try:
                idx = self._ensure()
            except Exception:  # noqa: BLE001 — FTS5 missing, degrade honestly
                _log.warning("hybrid BM25 unavailable (no FTS5)")
                return []
            try:
                results = idx.search(query, limit=limit)
            except Exception:  # noqa: BLE001
                return []
            out = []
            for r in results:
                out.append((r["doc_id"],
                            r.get("snippet") or r.get("title", "")))
            return out

    @property
    def indexed_count(self) -> int:
        return len(self._indexed)
