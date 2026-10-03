"""The federated query path: fan out, merge, dedupe, rank, filter.

Ranking scheme (the contract, kept in sync with ``model.py``):

``fusion="legacy"`` (default) —
1. Each source scores in its own native scale, so raw scores are
   min-max normalized *per source* into [0, 1] (a single-hit source
   normalizes to 1.0).
2. Hits sort by normalized score, descending.
3. Ties break on canonical source order (``memory, wisdom, books, docs,
   code, timeline, web_*`` — personal knowledge first), then newer
   timestamps first (undated hits sort last), then title. Deterministic.

``fusion="rrf"`` — reciprocal rank fusion across sources instead: each
source's native rank order contributes ``1 / (60 + rank)`` per hit,
summed across every source that returned it (see
``model.reciprocal_rank_fusion``). RRF scores are comparable *across*
sources without assuming anything about native scales, which is why it
is the right fusion for heterogeneous web backends mixed with local
indexes. Dedupe folds into the fusion; type/date filters and the
deterministic tie-break still apply afterwards.

Fan-out: sequential by default; ``parallel=True`` searches sources on a
thread pool (one worker per source, capped at 8). Probes always run
sequentially first (they are cheap and never touch the network), and
results/errors are collected in canonical source order — so failure
semantics and ``sources_searched`` order are identical either way.

Filters: ``--type`` keeps only matching result types; ``--since`` /
``--before`` drop hits whose timestamp falls outside the range. Hits
with no timestamp cannot be date-filtered and are *kept* — dropping
them would silently lose results the user asked for.

Failure semantics: empty query, unknown source, unknown type, or an
unparseable date raises immediately. A source that is unavailable (not
ingested, empty, unwired) is skipped with a note in
``SearchResponse.sources_skipped`` — never a crash. A source that
*errors* mid-search raises ``SearchError`` naming the source (fail
fast, with the real error chained). Zero hits is not silent: the
response records exactly which sources were searched.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from .errors import InvalidDateError, SearchError, UnknownSourceError, UnknownTypeError
from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .model import (
    SearchResponse,
    SearchResult,
    dedupe_results,
    normalize_scores,
    rank_results,
    reciprocal_rank_fusion,
)
from .sources import (
    SourceAdapter,
    build_adapters,
    list_sources,
    valid_source_names,
    valid_types,
)

__all__ = ["federated_search", "parse_date", "list_sources"]

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break a search (fail-open telemetry, fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


def parse_date(value: Any) -> float | None:
    """Parse a --since/--before value into epoch seconds.

    Accepts epoch seconds (int/float or numeric string) and ISO-8601
    strings. Raises :class:`InvalidDateError` on anything else.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidDateError(f"cannot parse date: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        epoch = float(text)
    except ValueError:
        epoch = None  # not epoch seconds — try ISO-8601 below
    if epoch is not None:
        return epoch
    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        raise InvalidDateError(
            f"cannot parse date {value!r}: use ISO-8601 or epoch seconds"
        ) from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _split_types(types: list[str] | None) -> list[str]:
    out: list[str] = []
    for t in types or []:
        out.extend(s.strip() for s in str(t).split(",") if s.strip())
    return out


def _search_one(
    name: str,
    adapter: SourceAdapter,
    query: str,
    per_source: int,
    since_ts: float | None,
    before_ts: float | None,
) -> list[SearchResult]:
    """Search one already-probed source. A mid-search error raises
    ``SearchError`` naming the source (fail fast, real cause chained)."""
    try:
        return adapter.search(
            query, limit=per_source, since=since_ts, before=before_ts
        )
    except Exception as exc:
        raise SearchError(f"source {name!r} failed: {exc}") from exc


def federated_search(
    query: str,
    *,
    context: Any = None,
    sources: list[str] | None = None,
    limit: int = 10,
    types: list[str] | None = None,
    since: Any = None,
    before: Any = None,
    doc_dir: str | None = None,
    doc_index_path: str | None = None,
    timeline: Any = None,
    adapters: dict[str, SourceAdapter] | None = None,
    fusion: str = "legacy",
    parallel: bool = False,
) -> SearchResponse:
    """Run ``query`` across the selected sources and return merged results.

    ``sources`` defaults to every known source; unknown names raise
    :class:`UnknownSourceError` listing the valid ones. ``adapters``
    (prebuilt, for tests/programmatic use) overrides adapter construction.
    ``fusion`` is ``"legacy"`` (per-source min-max normalize, the historic
    contract) or ``"rrf"`` (reciprocal rank fusion across sources).
    ``parallel=True`` fans the source searches out on a thread pool while
    preserving canonical order and failure semantics.
    """
    query = (query or "").strip()
    if not query:
        raise SearchError("search query must not be empty")

    if fusion not in ("legacy", "rrf"):
        raise SearchError(
            f"unknown fusion mode: {fusion!r}; valid: 'legacy', 'rrf'"
        )

    names = list(sources) if sources else valid_source_names()
    valid = valid_source_names()
    for name in names:
        if name not in valid:
            raise UnknownSourceError(name, valid)

    wanted_types = _split_types(types)
    known_types = valid_types()
    for t in wanted_types:
        if t not in known_types:
            raise UnknownTypeError(t, known_types)

    since_ts = parse_date(since)
    before_ts = parse_date(before)

    limit = int(limit)
    if limit <= 0:
        raise SearchError(f"limit must be positive, got {limit}")

    if adapters is None:
        adapters = build_adapters(
            context,
            doc_dir=doc_dir,
            doc_index_path=doc_index_path,
            timeline=timeline,
        )
    for name in names:
        if name not in adapters:
            raise SearchError(f"no adapter built for source {name!r}")

    response = SearchResponse(query=query)
    per_source = max(limit * 2, 10)

    # Probes run sequentially first: they are cheap, never touch the
    # network, and keep sources_skipped deterministic under parallel.
    searchable: list[str] = []
    for name in names:
        note = adapters[name].probe()
        if note is not None:
            response.sources_skipped[name] = note
        else:
            searchable.append(name)

    def _run(name: str) -> list[SearchResult]:
        return _search_one(
            name, adapters[name], query, per_source, since_ts, before_ts
        )

    per_source_hits: list[list[SearchResult]] = []
    if parallel and searchable:
        with ThreadPoolExecutor(
            max_workers=min(len(searchable), 8),
            thread_name_prefix="search",
        ) as pool:
            futures = {name: pool.submit(_run, name) for name in searchable}
            # Collect in canonical order: the first error in source order
            # is the one that surfaces, exactly like the sequential path.
            for name in searchable:
                try:
                    hits = futures[name].result()
                except Exception:
                    for f in futures.values():
                        f.cancel()
                    raise
                per_source_hits.append(hits)
                response.sources_searched.append(name)
    else:
        for name in searchable:
            per_source_hits.append(_run(name))
            response.sources_searched.append(name)

    if fusion == "rrf":
        merged, dropped = reciprocal_rank_fusion(per_source_hits)
        response.deduped = dropped
        kept = merged
    else:
        merged = []
        for hits in per_source_hits:
            normalize_scores(hits)
            merged.extend(hits)
        kept, dropped = dedupe_results(merged)
        response.deduped = dropped

    if wanted_types:
        wanted = set(wanted_types)
        kept = [h for h in kept if h.type in wanted]
    if since_ts is not None:
        kept = [h for h in kept if h.timestamp is None or h.timestamp >= since_ts]
    if before_ts is not None:
        kept = [h for h in kept if h.timestamp is None or h.timestamp <= before_ts]

    response.hits = rank_results(kept, valid_source_names())[:limit]
    _emit("search.performed", {
        "query": query,
        "sources": list(response.sources_searched),
        "sources_skipped": dict(response.sources_skipped),
        "result_count": len(response.hits),
        "deduped": response.deduped,
        "fusion": fusion,
        "parallel": parallel,
    })
    return response
