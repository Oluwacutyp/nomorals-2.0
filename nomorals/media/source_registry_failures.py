"""Failure ledger for the source registry engine.

``SourceRegistry`` is built on a never-raises contract: every method
that can fail falls back to a safe default instead of raising — a
binding that returns False, a registration that returns False, a
lookup that returns None, a score that returns the tier default, a
status that returns empty structures. That contract is sound for a
production engine — but until now the caught failures vanished without
a trace. A registration that fails with an exception looked identical
to a refused duplicate; a learned-rule lookup that fails looked like
"no rule on record"; a health status that fails looked like "no
sources configured"; a dropped outcome record silently corrupted the
success rates the ordering learns from. The degradation was honest in
return-value shape but silent about the underlying error — now it
leaves a trace.

The same contract covers ``PoliteFetch`` (Pass #129): the per-host
politeness gate — backoff, rate budget, robots.txt — also never
raises, and every one of its fallbacks leans toward allowing the
fetch: a crashed gate returns "allowed", a crashed robots check
returns "not disallowed", dropped bookkeeping silently un-enforces
the budget and the backoff. A fetch that should have been gated
looked identical to a fetch that was checked and allowed.

Stdlib only. ``record`` itself never raises — a ledger that can crash
its callers would defeat the purpose.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

__all__ = ["Failure", "record_failure", "recent_failures",
           "failure_count", "clear_failures", "format_failures"]

#: Ring size: enough to see a failing site's history, small enough to
#: never grow. Oldest entries drop off silently by design.
_MAX_FAILURES = 200


@dataclass(frozen=True)
class Failure:
    """One recorded failure: where, what, and when."""

    ts: float
    site: str
    exc_type: str
    message: str


_ledger: deque[Failure] = deque(maxlen=_MAX_FAILURES)
_lock = threading.Lock()
_total = 0


def record_failure(site: str, exc: BaseException) -> None:
    """Append ``exc`` to the ledger under ``site``. Never raises."""
    try:
        message = str(exc)
    except Exception as _err:  # noqa: BLE001 — pathological __str__; stay silent-proof
        message = f"<unprintable {type(_err).__name__}>"
    entry = Failure(ts=time.time(), site=site,
                    exc_type=type(exc).__name__, message=message)
    global _total
    with _lock:
        _ledger.append(entry)
        _total += 1


def recent_failures(limit: int = 20) -> list[Failure]:
    """Newest-first snapshot of the ledger (up to ``limit`` entries)."""
    with _lock:
        return list(reversed(_ledger))[:limit]


def failure_count() -> int:
    """Total failures recorded since process start (ring may have dropped old ones)."""
    with _lock:
        return _total


def clear_failures() -> None:
    """Empty the ring and reset the count (tests, mostly)."""
    global _total
    with _lock:
        _ledger.clear()
        _total = 0


def format_failures(limit: int = 20) -> str:
    """Human-readable render of the newest ledger entries, or an empty note."""
    entries = recent_failures(limit)
    if not entries:
        return "no recorded source registry failures"
    lines = []
    for entry in entries:
        lines.append(f"{entry.site}: {entry.exc_type}: {entry.message}")
    return "\n".join(lines)
