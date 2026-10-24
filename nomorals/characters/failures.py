"""Failure ledger for the characters subsystem.

The character engine is deliberately best-effort: prompt building, context
assembly, dialogue, matching, session processing, and store I/O all fall
back to safe defaults instead of raising. That contract is sound — a
character that can't load one memory should still answer — but until now
the caught failures vanished without a trace: a store save that failed,
a session-processing step that died, a match intro that never rendered,
all looked identical to a healthy no-op. This ledger fixes that half of
the contract without touching the other half: failures are recorded,
never re-raised, and readable via the ``characters`` tool's ``failures``
action and :func:`format_failures`.

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
    """Newest-first slice of the ledger (oldest drops off at the cap)."""
    with _lock:
        items = list(_ledger)
    return items[::-1][:max(0, limit)]


def failure_count() -> int:
    """Total failures ever recorded (ledger length is capped; this is not)."""
    with _lock:
        return _total


def clear_failures() -> None:
    """Empty the ledger and reset the counter. Tests only."""
    global _total
    with _lock:
        _ledger.clear()
        _total = 0


def format_failures(limit: int = 20) -> str:
    """One human-readable block for the ``failures`` tool action."""
    items = recent_failures(limit)
    if not items:
        return "no character failures recorded."
    lines = [f"⚠️ character failures ({failure_count()} total, "
             f"showing {len(items)} newest):"]
    for f in items:
        ts = time.strftime("%m-%d %H:%M", time.localtime(f.ts))
        lines.append(f"• {ts} {f.site}: {f.exc_type}: {f.message}")
    return "\n".join(lines)
