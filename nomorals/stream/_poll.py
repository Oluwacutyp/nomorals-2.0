"""Shared Timeline polling primitives for the stream package.

Both the per-connection emitter (:mod:`nomorals.stream.server`) and the
broadcast hub (:mod:`nomorals.stream.hub`) drain the Timeline through this
one code path, so the exactly-once cursor bookkeeping and the
deep-backlog drain exist exactly once.

``Timeline.query(since=)`` is inclusive (``ts >= cursor``) while the cursor
only advances on strictly newer timestamps — without identity tracking, an
event whose ``ts`` equals the cursor is re-emitted on every poll.
``seen_at_cursor`` holds the event_ids already emitted at this cursor, so a
stream is exactly-once per event while resume across reconnects stays
at-least-once (the boundary event is delivered again, once, to the new
connection).
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.logging_setup import get_logger

__all__ = [
    "PAGE_LIMIT",
    "POLL_INTERVAL",
    "advance_cursor",
    "classify_one",
    "drain_backlog",
    "query_page",
]

_log = get_logger(__name__)

#: Seconds between Timeline polls when a hub has live subscribers.
POLL_INTERVAL = 1.0

#: Events per backlog-drain page. A poll that finds a full page drains the
#: older slice behind it (see drain_backlog) instead of silently dropping
#: the tail the way a single fixed page would.
PAGE_LIMIT = 1000

#: Backlog-drain recursion cap: each level consumes at least one full
#: page, so this bounds the drain at ~50k events per poll — beyond that
#: the subscriber gets the newest slice and a loud warning, never a
#: RecursionError or a silent drop.
_MAX_DRAIN_DEPTH = 50


def _ts_of(ev: dict[str, Any]) -> float:
    return float(ev.get("ts", 0) or 0)


def _id_of(ev: dict[str, Any]) -> str:
    return str(ev.get("event_id") or "")


def query_page(
    timeline_factory: Callable[[], Any],
    topic: str | None,
    since: float,
    until: float | None,
    *,
    limit: int = PAGE_LIMIT,
) -> list[dict[str, Any]]:
    """One newest-first page; the timeline instance is never shared."""
    timeline = timeline_factory()
    try:
        kwargs: dict[str, Any] = {
            "since": since, "topic": topic, "limit": limit}
        if until is not None:
            kwargs["until"] = until
        return timeline.query(**kwargs)
    finally:
        close = getattr(timeline, "close", None)
        if callable(close):
            close()


def classify_one(
    ev: dict[str, Any],
    cursor: float,
    seen_at_cursor: set[str],
) -> tuple[bool, float, set[str]]:
    """Classify a single event against a cursor.

    Returns ``(is_new, cursor, seen_at_cursor)`` — the cursor bookkeeping
    for exactly-once delivery within one continuous stream.
    """
    ts = _ts_of(ev)
    eid = _id_of(ev)
    if ts > cursor:
        return True, ts, {eid} if eid else set()
    if ts == cursor and eid and eid not in seen_at_cursor:
        seen_at_cursor.add(eid)
        return True, cursor, seen_at_cursor
    return False, cursor, seen_at_cursor


def advance_cursor(
    events: list[dict[str, Any]],
    cursor: float,
    seen_at_cursor: set[str],
) -> tuple[list[dict[str, Any]], float, set[str]]:
    """Split a newest-first page into new events (oldest-first).

    Returns ``(new_events, cursor, seen_at_cursor)`` with the cursor
    advanced past everything emitted.
    """
    new: list[dict[str, Any]] = []
    for ev in reversed(events):  # query() is newest-first; emit oldest-first
        is_new, cursor, seen_at_cursor = classify_one(
            ev, cursor, seen_at_cursor)
        if is_new:
            new.append(ev)
    return new, cursor, seen_at_cursor


def drain_backlog(
    sink: Callable[[dict[str, Any]], None],
    timeline_factory: Callable[[], Any],
    topic: str | None,
    cursor: float,
    seen_at_cursor: set[str],
    *,
    limit: int = PAGE_LIMIT,
    _floor: float | None = None,
    _ceiling: float | None = None,
    _depth: int = 0,
) -> tuple[float, set[str]]:
    """Emit every event with ``_floor <= ts`` (``ts <= _ceiling`` when set).

    New events go to ``sink`` oldest-first. A full page may hide older
    events behind it (the query is newest-first): those are drained first
    via a narrowed ``until`` bound, so a burst bigger than one page cannot
    silently drop its tail. The inclusive-boundary overlap between the
    narrowed slice and its parent page is harmless — the event_id dedup in
    :func:`advance_cursor` skips re-emission.

    Recursion strictly narrows ``_ceiling`` each level, so the only
    non-shrinking shape is a full page of identical timestamps; that —
    and any backlog deeper than ``_MAX_DRAIN_DEPTH`` pages — is emitted
    once and logged loudly instead of looping forever or dying silently.
    """
    floor = cursor if _floor is None else _floor
    events = query_page(timeline_factory, topic, floor, _ceiling, limit=limit)
    if len(events) < limit:
        new, cursor, seen_at_cursor = advance_cursor(
            events, cursor, seen_at_cursor)
        for ev in new:
            sink(ev)
        return cursor, seen_at_cursor
    oldest = min(_ts_of(e) for e in events)
    if _depth >= _MAX_DRAIN_DEPTH or (
            _ceiling is not None and oldest >= _ceiling):
        _log.warning(
            "stream: backlog of >%d events at/above ts %r truncated for "
            "this poll (depth=%d)", limit, oldest, _depth)
        new, cursor, seen_at_cursor = advance_cursor(
            events, cursor, seen_at_cursor)
        for ev in new:
            sink(ev)
        return cursor, seen_at_cursor
    # Drain the older slice first, then this page.
    cursor, seen_at_cursor = drain_backlog(
        sink, timeline_factory, topic, cursor, seen_at_cursor,
        limit=limit, _floor=floor, _ceiling=oldest, _depth=_depth + 1)
    new, cursor, seen_at_cursor = advance_cursor(
        events, cursor, seen_at_cursor)
    for ev in new:
        sink(ev)
    return cursor, seen_at_cursor
