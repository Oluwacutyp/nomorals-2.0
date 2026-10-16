"""Search spine (Tier 0 wiring): promote ``search/`` to the canonical spine.

:mod:`nomorals.search.federated` already implements fan-out, per-source
normalization, content-hash dedupe, and deterministic ranking, and
:mod:`nomorals.search.rerank` already implements BM25 — but three
subsystems each kept their own fusion copy
(``memory/hybrid.py`` ``rrf_fuse``, ``wisdom/hybrid.py`` RRF,
``books`` FTS5-only) and nothing named one canonical entry point.
This module is that entry point: :class:`SearchSpine` wraps the
federated path with ``fusion="rrf"`` as the canonical default and
exposes the pipeline stages (normalize → BM25 rerank → dedupe →
deterministic rank) for reuse.

The ``*_adapter`` factories are facades, not replacements: each wraps
the ORIGINAL adapter from :mod:`nomorals.search.sources` and runs its
hits through the spine pipeline. The originals are never deleted,
modified, or bypassed — a facade's ``probe()`` is the original's
``probe()``, and its ``search()`` is the original's search plus spine
post-processing.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..tier0_rollback import record_promotion
from .base import SourceAdapter
from .federated import federated_search as _federated_search
from .model import (
    SearchResponse,
    SearchResult,
    dedupe_results,
    normalize_scores,
    rank_results,
    reciprocal_rank_fusion,
)
from .rerank import bm25_rerank

__all__ = [
    "SearchSpine",
    "search",
    "memory_hybrid_adapter",
    "wisdom_hybrid_adapter",
    "books_library_adapter",
    "register_promotion",
    "promotion_active",
]

_log = get_logger(__name__)

PROMOTION_NAME = "search_spine"

_promotion_active = False


def promotion_active() -> bool:
    """True while the search-spine promotion is active (not rolled back)."""
    return _promotion_active


def register_promotion() -> None:
    """Record this wiring in the Tier 0 rollback registry."""
    global _promotion_active
    _promotion_active = True

    def _rollback() -> None:
        global _promotion_active
        _promotion_active = False

    record_promotion(
        PROMOTION_NAME, _rollback,
        description="search/ promoted to canonical spine (fusion=rrf)",
        metadata={"module": __name__},
    )


class SearchSpine:
    """Canonical search entry point: federated search, fusion='rrf'."""

    def __init__(
        self,
        *,
        default_fusion: str = "rrf",
        rrf_k: int = 60,
        source_order: list[str] | None = None,
    ) -> None:
        if default_fusion not in ("rrf", "legacy"):
            raise ValueError(
                f"unknown fusion {default_fusion!r}; expected 'rrf'/'legacy'")
        if rrf_k <= 0:
            raise ValueError(f"RRF k must be positive, got {rrf_k}")
        self.default_fusion = default_fusion
        self.rrf_k = int(rrf_k)
        self.source_order = list(source_order) if source_order else None

    # ── canonical entry point ────────────────────────────────────────────────
    def search(
        self,
        query: str,
        *,
        fusion: str | None = None,
        adapters: dict[str, SourceAdapter] | None = None,
        sources: list[str] | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Federated search with ``fusion="rrf"`` unless overridden.

        Delegates to :func:`nomorals.search.federated.federated_search`
        — the spine adds the default, it does not reimplement the path.
        """
        return _federated_search(
            query,
            sources=sources,
            limit=limit,
            adapters=adapters,
            fusion=fusion or self.default_fusion,
            **kwargs,
        )

    # ── pipeline stages (shared, so subsystems stop copying them) ───────────
    def normalize(self, hits: list[SearchResult]) -> None:
        """Per-source min-max normalize ``raw_score`` → ``score`` in place."""
        normalize_scores(hits)

    def dedupe(
        self, hits: list[SearchResult]
    ) -> tuple[list[SearchResult], int]:
        """Drop content-hash duplicates, keeping the highest-scoring hit."""
        return dedupe_results(hits)

    def rank(
        self,
        hits: list[SearchResult],
        source_order: list[str] | None = None,
    ) -> list[SearchResult]:
        """Deterministic ranking: score desc, source order, newer, title."""
        from .sources import valid_source_names
        order = source_order or self.source_order or valid_source_names()
        return rank_results(hits, order)

    def bm25_rerank(
        self, hits: list[SearchResult], query: str, **kwargs: Any
    ) -> list[SearchResult]:
        """Re-rank one source's hits by BM25(query, title + snippet)."""
        return bm25_rerank(hits, query, **kwargs)

    def pipeline(
        self,
        per_source_hits: list[list[SearchResult]],
        query: str,
        *,
        fusion: str | None = None,
        source_order: list[str] | None = None,
        bm25: bool = False,
    ) -> tuple[list[SearchResult], int]:
        """Run the canonical pipeline over already-fetched per-source hits.

        Per batch: optional BM25 rerank → per-source normalize.
        Then: ``fusion="rrf"`` → reciprocal rank fusion (dedupe folds
        into the fusion); ``"legacy"`` → merge + content-hash dedupe.
        Finally: deterministic ranking. Returns ``(hits, deduped)``.
        """
        mode = fusion or self.default_fusion
        if mode not in ("rrf", "legacy"):
            raise ValueError(
                f"unknown fusion {mode!r}; expected 'rrf'/'legacy'")
        batches: list[list[SearchResult]] = []
        for hits in per_source_hits:
            batch = list(hits)
            if bm25:
                batch = self.bm25_rerank(batch, query)
            self.normalize(batch)
            batches.append(batch)
        if mode == "rrf":
            fused, dropped = reciprocal_rank_fusion(batches, k=self.rrf_k)
            return self.rank(fused, source_order), dropped
        merged: list[SearchResult] = []
        for batch in batches:
            merged.extend(batch)
        kept, dropped = self.dedupe(merged)
        return self.rank(kept, source_order), dropped


def search(
    query: str,
    *,
    fusion: str = "rrf",
    adapters: dict[str, SourceAdapter] | None = None,
    sources: list[str] | None = None,
    limit: int = 10,
    **kwargs: Any,
) -> SearchResponse:
    """Module-level canonical search: federated, ``fusion="rrf"``."""
    return SearchSpine().search(
        query, fusion=fusion, adapters=adapters,
        sources=sources, limit=limit, **kwargs)


# ── facades: original adapters + spine post-processing ──────────────────────

class _SpineFacadeAdapter(SourceAdapter):
    """One original adapter whose hits run through the spine pipeline.

    ``probe()`` is the original's verbatim. ``search()`` is the
    original's search, then BM25 rerank → normalize → content-hash
    dedupe → deterministic rank. The original adapter object is kept
    (``.inner``) so callers can always reach the unwrapped behaviour.
    """

    def __init__(
        self,
        inner: SourceAdapter,
        spine: SearchSpine | None = None,
    ) -> None:
        self.inner = inner
        self.spine = spine or SearchSpine()
        self.name = getattr(inner, "name", "")
        self.result_type = getattr(inner, "result_type", "")
        self.description = (
            f"spine facade over {type(inner).__name__}: "
            f"{getattr(inner, 'description', '')}"
        )

    def probe(self) -> str | None:
        return self.inner.probe()

    def search(
        self,
        query: str,
        *,
        limit: int,
        since: float | None = None,
        before: float | None = None,
    ) -> list[SearchResult]:
        hits = self.inner.search(
            query, limit=limit, since=since, before=before)
        batch = self.spine.bm25_rerank(list(hits), query)
        self.spine.normalize(batch)
        kept, _dropped = self.spine.dedupe(batch)
        return self.spine.rank(kept)


def memory_hybrid_adapter(
    context: Any = None,
    *,
    manager: Any | None = None,
    spine: SearchSpine | None = None,
) -> _SpineFacadeAdapter:
    """Spine facade over the memory adapter (replaces nothing).

    The memory subsystem's own ``memory/hybrid.py`` RRF stays in place
    for its vector+BM25 lane; this facade is the *federated* memory
    source routed through the spine, so federated memory hits get the
    same normalize/BM25/dedupe/rank treatment as every other source.
    """
    from .sources import MemoryAdapter
    return _SpineFacadeAdapter(
        MemoryAdapter(context, manager=manager), spine=spine)


def wisdom_hybrid_adapter(
    context: Any = None,
    *,
    keeper: Any | None = None,
    spine: SearchSpine | None = None,
) -> _SpineFacadeAdapter:
    """Spine facade over the wisdom adapter (replaces nothing).

    ``wisdom/hybrid.py`` keeps its own keyword+semantic RRF for
    in-corpus fusion; this facade routes federated wisdom hits through
    the spine.
    """
    from .sources import WisdomAdapter
    return _SpineFacadeAdapter(
        WisdomAdapter(context, keeper=keeper), spine=spine)


def books_library_adapter(
    context: Any = None,
    *,
    library: Any | None = None,
    spine: SearchSpine | None = None,
) -> _SpineFacadeAdapter:
    """Spine facade over the books adapter (replaces nothing).

    ``books.Library`` keeps its FTS5 search; this facade routes
    federated book hits through the spine (BM25 rerank + normalize +
    content-hash dedupe + deterministic rank).
    """
    from .sources import BooksAdapter
    return _SpineFacadeAdapter(
        BooksAdapter(context, library=library), spine=spine)


register_promotion()
