"""Organ events for the immune system.

Event kinds are exact — the bus (``organs.py``) and any consumer key
off these strings. This module does NOT import ``organs.py``: the
caller passes an emit function (the bus's ``emit``) as the sink, or
``None`` to just build the event dict without emitting.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

__all__ = ["EVENT_KINDS", "immune_event", "emit_immune_event"]

EVENT_KINDS: tuple[str, ...] = (
    "threat_detected",
    "threat_contained",
    "antibody_created",
    "false_positive_retired",
)


def immune_event(kind: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build an immune event dict: ``{src, kind, payload, ts}``."""
    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown immune event kind {kind!r}; expected one of {EVENT_KINDS}")
    return {"src": "immune", "kind": kind, "payload": dict(payload or {}), "ts": time.time()}


def emit_immune_event(
    db_or_sink: Callable[[str, Dict[str, Any]], Any] | Any | None,
    kind: str,
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build an immune event and emit it through an INJECTED sink.

    ``db_or_sink`` is a callable ``(kind, payload)``, or an object with
    an ``emit(event)`` method, or ``None`` (returns the event dict
    without emitting). Always returns the event dict.
    """
    event = immune_event(kind, payload)
    if db_or_sink is None:
        return event
    if callable(db_or_sink):
        db_or_sink(kind, dict(event["payload"]))
    elif hasattr(db_or_sink, "emit"):
        db_or_sink.emit(event)
    else:
        raise TypeError("db_or_sink must be a callable(kind, payload), an object with emit(event), or None")
    return event
