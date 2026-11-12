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
sequentially first (they are cheap and never touch the network), each
isolated on its own worker with a hard per-probe deadline (default
10s) and a phase budget (30s) — a probe that exceeds its deadline is
skipped with a timeout note, never awaited — and results/errors are
collected in canonical source order — so failure semantics and
``sources_searched`` order are identical either way. With
``deadline_s`` set, the whole call (probes + searches) is bounded by
it: slow sources are skipped with a timeout note and the call returns
the partial results instead of hanging past the deadline.

Filters: ``--type`` keeps only matching result types; ``--since`` /
``--before`` drop hits whose timestamp falls outside the range. Hits
with no timestamp cannot be date-filtered and are *kept* — dropping
them would silently lose results the user asked for.

Failure semantics: empty query, unknown source, unknown type, or an
unparseable date raises immediately. A source that is unavailable (not
ingested, empty, unwired) is skipped with a note in
``SearchResponse.sources_skipped`` — never a crash. A source that
*errors* mid-search is ALSO skipped with its real error recorded in
``sources_skipped`` — one bad source never kills the whole search; the
chain continues through ALL sources and returns what the healthy ones
found. Zero hits is not silent: the response records exactly which
sources were searched and which failed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import Any, TypeVar

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
    build_osint_adapter,
    list_sources,
    valid_source_names,
    valid_types,
)
from .web import (
    WEB_BACKEND_DOMAINS,
    WebSearchSource,
    ordered_web_backend_names,
)

__all__ = ["federated_search", "parse_date", "list_sources"]

_log = get_logger(__name__)

#: A probe is contractually cheap ("they are cheap, never touch the
#: network"). A probe that runs longer than this is the hang class
#: itself: it is skipped with a timeout note, never awaited. The
#: 2026-11-11 "motivation morning" kill died at the 900s wall inside
#: ``BooksAdapter.probe`` because one probe could block forever.
DEFAULT_PROBE_TIMEOUT_S = 10.0

#: The probe phase as a whole never exceeds this: probes are a gate,
#: not the work. Without an overall deadline the phase is still
#: capped, so slow probes cannot eat the call.
PROBE_PHASE_BUDGET_S = 30.0

_T = TypeVar("_T")


def _remaining(deadline_s: float | None, t_start: float) -> float | None:
    """Seconds left on the caller's overall deadline, or None (no
    deadline — the call runs unbounded, as before)."""
    if deadline_s is None:
        return None
    return (t_start + deadline_s) - time.perf_counter()


def _probe_timeout(deadline_s: float | None,
                   probe_timeout_s: float) -> float:
    """The per-probe cap for this call. Under an overall deadline a
    single probe may take at most a quarter of it — otherwise one hung
    probe eats the whole call before any source is searched."""
    cap = probe_timeout_s
    if deadline_s is not None:
        cap = min(cap, deadline_s / 4.0)
    return max(cap, 0.0)


def _run_isolated(fn: Callable[[], _T],
                  timeout_s: float) -> tuple[_T | None, bool]:
    """Run ``fn`` on its own worker thread with a hard deadline.

    Returns ``(value, timed_out)``. On timeout the worker is abandoned
    (never joined — joining would re-introduce the hang); the thread
    finishes its op in the background. An exception ``fn`` raises is
    re-raised here, except a ``TimeoutError`` the callable itself
    raised is also re-raised (it is the callable's own error, not our
    deadline — distinguished via ``future.done()``).
    """
    pool = ThreadPoolExecutor(max_workers=1,
                              thread_name_prefix="search-deadline")
    try:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=timeout_s), False
        except FuturesTimeoutError:
            if fut.done():
                raise
            return None, True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _collect(fut, timeout_s: float | None) -> tuple[Any, bool]:
    """``(result, timed_out)`` for one already-submitted future under
    a deadline. ``timeout_s=None`` waits forever (the no-deadline
    path). A future that finished by raising keeps its error
    (re-raised); a future still running past the deadline reports
    ``timed_out``."""
    try:
        return fut.result(timeout=timeout_s), False
    except FuturesTimeoutError:
        if fut.done():
            raise
        return None, True


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
    """Search one already-probed source.  Errors propagate to the
    caller, which records them per-source and keeps going (one bad
    source never kills the whole search)."""
    return adapter.search(
        query, limit=per_source, since=since_ts, before=before_ts
    )


def _resolve_weights(
    source_weights: dict[str, float] | list[float] | None,
    names: list[str],
) -> list[float] | None:
    """Align user-supplied RRF weights with the searched source order."""
    if source_weights is None:
        return None
    if isinstance(source_weights, dict):
        try:
            return [float(source_weights[name]) for name in names]
        except KeyError as exc:
            raise SearchError(
                f"source_weights missing source {exc}; searched: {names}"
            ) from exc
    if len(source_weights) != len(names):
        raise SearchError(
            f"source_weights length {len(source_weights)} != "
            f"sources searched {len(names)}"
        )
    return [float(w) for w in source_weights]


def _apply_web_ordering(
    names: list[str],
    adapters: dict[str, SourceAdapter],
    context: Any,
) -> list[str]:
    """Outcome-aware ordering of the WEB backends among themselves
    (Pass #90), via the ONE source registry the backends record
    into.  Only the web backends' own slots are permuted — local
    and OSINT sources keep their canonical positions, and ranking's
    source-order tie-break is untouched.  Day-one pin: with no
    learned state (or no registry — a db-less context, prebuilt
    test adapters) the names are returned byte-for-byte.  Never
    raises."""
    try:
        positions = [
            i for i, n in enumerate(names)
            if n in WEB_BACKEND_DOMAINS
            and isinstance(adapters.get(n), WebSearchSource)
        ]
        if len(positions) < 2:
            return names
        reg = None
        for i in positions:
            adapter = adapters[names[i]]
            resolve = getattr(adapter, "_registry", None)
            if callable(resolve):
                reg = resolve()
            if reg is not None:
                break
        if reg is None:
            from .web import web_registry_for_context

            reg = web_registry_for_context(context)
        if reg is None:
            return names
        subset = [names[i] for i in positions]
        ordered = ordered_web_backend_names(reg, subset)
        if ordered == subset:
            return names
        out = list(names)
        for i, n in zip(positions, ordered):
            out[i] = n
        return out
    except Exception:  # noqa: BLE001 - ordering never breaks a search
        return names


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
    source_weights: dict[str, float] | list[float] | None = None,
    deadline_s: float | None = None,
    probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
) -> SearchResponse:
    """Run ``query`` across the selected sources and return merged results.

    ``sources`` defaults to every known source; unknown names raise
    :class:`UnknownSourceError` listing the valid ones. ``adapters``
    (prebuilt, for tests/programmatic use) overrides adapter construction.
    ``fusion`` is ``"legacy"`` (per-source min-max normalize, the historic
    contract) or ``"rrf"`` (reciprocal rank fusion across sources).
    ``parallel=True`` fans the source searches out on a thread pool while
    preserving canonical order and failure semantics.
    ``deadline_s`` is the overall budget for the whole call (probes +
    searches): when it runs out, slow sources are skipped with a timeout
    note and the call returns what the healthy sources found — never a
    hang past the deadline. ``None`` (the default) keeps the historic
    unbounded search phase, but probes are ALWAYS capped at
    ``probe_timeout_s`` (default 10s): a probe is contractually cheap,
    and one that runs longer is the hang class itself.
    ``source_weights`` (RRF only) trusts some sources more than others —
    a dict keyed by source name or a list aligned with ``sources`` order;
    each weight scales that source's ``1/(k+rank)`` contributions (the
    Elasticsearch weighted-RRF extension).

    ``SearchResponse.timings`` records per-source wall-clock seconds and
    ``SearchResponse.elapsed`` the total — every metasearch reports
    per-engine latency, and operators need it to spot a slow backend.
    """
    t_start = time.perf_counter()
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

    if deadline_s is not None and float(deadline_s) <= 0:
        raise SearchError(f"deadline_s must be positive, got {deadline_s}")
    if float(probe_timeout_s) <= 0:
        raise SearchError(
            f"probe_timeout_s must be positive, got {probe_timeout_s}")

    if adapters is None:
        adapters = build_adapters(
            context,
            doc_dir=doc_dir,
            doc_index_path=doc_index_path,
            timeline=timeline,
        )
    for name in names:
        if name not in adapters:
            # OSINT adapters are context-free (no constructor arguments),
            # so they build on demand instead of living in the
            # context-bound adapter set — this is what makes osint_*
            # sources reachable through the standard query path.  The
            # browser-driven ones get the context's tool registry, so
            # their browser calls run on the production registry.
            lazy = build_osint_adapter(
                name, registry=getattr(context, "tools", None))
            if lazy is None:
                raise SearchError(f"no adapter built for source {name!r}")
            adapters[name] = lazy

    # Pass #90: with learned evidence, the web backends consult the
    # registry's scores/health for their relative order (day-one:
    # canonical order, byte-for-byte).
    names = _apply_web_ordering(names, adapters, context)

    response = SearchResponse(query=query)
    per_source = max(limit * 2, 10)

    # Probes run sequentially first: they are cheap, never touch the
    # network, and keep sources_skipped deterministic under parallel.
    # Each probe runs isolated on its own worker with a hard deadline —
    # the 2026-11-11 "motivation morning" kill: BooksAdapter.probe
    # stalled the whole job at the 900s wall because one probe could
    # block forever (a full FTS5 COUNT(*) scan, or a Library
    # construction doing a full seam import). A probe that exceeds its
    # deadline is skipped with a timeout note, never awaited, and the
    # phase as a whole is budgeted so slow probes cannot eat the
    # caller's deadline. A probe that raises is likewise skipped with
    # its error — one bad source never kills the whole search.
    searchable: list[str] = []
    probe_phase_deadline = t_start + PROBE_PHASE_BUDGET_S
    if deadline_s is not None:
        probe_phase_deadline = min(
            probe_phase_deadline, t_start + deadline_s)
    per_probe_cap = _probe_timeout(deadline_s, probe_timeout_s)
    for name in names:
        remaining_phase = probe_phase_deadline - time.perf_counter()
        if remaining_phase <= 0:
            response.sources_skipped[name] = (
                "probe timed out: probe phase budget exhausted")
            continue
        remaining_call = _remaining(deadline_s, t_start)
        timeout = min(per_probe_cap, remaining_phase)
        if remaining_call is not None:
            timeout = min(timeout, remaining_call)
        if timeout <= 0:
            response.sources_skipped[name] = (
                "probe timed out: deadline exceeded")
            continue
        try:
            note, timed_out = _run_isolated(
                lambda n=name: adapters[n].probe(), timeout)
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            response.sources_skipped[name] = f"probe error: {exc}"
            continue
        if timed_out:
            response.sources_skipped[name] = (
                f"probe timed out after {timeout:g}s")
        elif note is not None:
            response.sources_skipped[name] = note
        else:
            searchable.append(name)

    def _run(name: str) -> list[SearchResult]:
        return _search_one(
            name, adapters[name], query, per_source, since_ts, before_ts
        )

    per_source_hits: list[list[SearchResult]] = []
    hit_source_names: list[str] = []
    if parallel and searchable:
        pool = ThreadPoolExecutor(
            max_workers=min(len(searchable), 8),
            thread_name_prefix="search",
        )
        try:
            futures = {name: pool.submit(_run, name) for name in searchable}
            # Collect in canonical order.  A failing source is recorded
            # in sources_skipped and the rest still contribute — one
            # bad source never kills the whole search. Under a deadline
            # a source still running past it is skipped with a timeout
            # note and the call returns the partial results.
            for name in searchable:
                t0 = time.perf_counter()
                try:
                    hits, timed_out = _collect(
                        futures[name], _remaining(deadline_s, t_start))
                except Exception as exc:
                    response.sources_skipped[name] = (
                        f"error: {exc}")
                    response.timings[name] = time.perf_counter() - t0
                    _log.info("federated: source %r failed, continuing: %s",
                              name, exc)
                    continue
                response.timings[name] = time.perf_counter() - t0
                if timed_out:
                    response.sources_skipped[name] = (
                        f"search timed out after {deadline_s:g}s")
                    _log.info("federated: source %r timed out, continuing",
                              name)
                    continue
                per_source_hits.append(hits)
                hit_source_names.append(name)
                response.sources_searched.append(name)
        finally:
            # Never wait: a timed-out future's worker is abandoned,
            # not joined — joining would re-introduce the hang this
            # deadline exists to eliminate. With no deadline every
            # future is already resolved here, so this is a no-op.
            pool.shutdown(wait=False, cancel_futures=True)
    else:
        for name in searchable:
            t0 = time.perf_counter()
            remaining = _remaining(deadline_s, t_start)
            try:
                if remaining is not None:
                    hits, timed_out = _run_isolated(
                        lambda n=name: _run(n), remaining)
                    if timed_out:
                        response.sources_skipped[name] = (
                            f"search timed out after {deadline_s:g}s")
                        response.timings[name] = time.perf_counter() - t0
                        _log.info("federated: source %r timed out, continuing",
                                  name)
                        continue
                else:
                    hits = _run(name)
            except Exception as exc:
                response.sources_skipped[name] = f"error: {exc}"
                response.timings[name] = time.perf_counter() - t0
                _log.info("federated: source %r failed, continuing: %s",
                          name, exc)
                continue
            response.timings[name] = time.perf_counter() - t0
            per_source_hits.append(hits)
            hit_source_names.append(name)
            response.sources_searched.append(name)

    if fusion == "rrf":
        weights = _resolve_weights(source_weights, hit_source_names)
        merged, dropped = reciprocal_rank_fusion(per_source_hits, weights=weights)
        response.deduped = dropped
        kept = merged
    else:
        if source_weights is not None:
            raise SearchError(
                "source_weights only applies to fusion='rrf'"
            )
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
    response.elapsed = time.perf_counter() - t_start
    _emit("search.performed", {
        "query": query,
        "sources": list(response.sources_searched),
        "sources_skipped": dict(response.sources_skipped),
        "result_count": len(response.hits),
        "deduped": response.deduped,
        "fusion": fusion,
        "parallel": parallel,
        "elapsed": response.elapsed,
    })
    return response
