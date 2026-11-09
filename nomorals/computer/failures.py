"""Failure ledger for the computer organ.

The organ degrades honestly instead of raising where a surface
needs words (status reads, chat replies) — but a swallowed failure
must never vanish: a driver that died mid-session, a display that
would not start, a frame that would not save all land here, in the
shared durable failure store, readable from the package.

Stdlib only, plus the stdlib-only failure store. ``record`` itself
never raises — a ledger that can crash its callers would defeat
the purpose.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass

from ..core import failure_store as _failure_store

__all__ = ["Failure", "record_failure", "recent_failures",
           "failure_count", "clear_failures", "format_failures"]

#: Ring size: enough to see a failing site's history, small enough
#: to never grow. Oldest entries drop off silently by design.
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

_store = _failure_store.Store("computer")
_loaded = False


def _ensure_loaded() -> None:
    """Load persisted entries once per process. Caller holds ``_lock``."""
    global _loaded, _total
    if _loaded:
        return
    _loaded = True
    entries, total = _store.load()
    for row in entries:
        _ledger.append(Failure(ts=row["ts"], site=row["site"],
                               exc_type=row["exc_type"],
                               message=row["message"]))
    _total = max(_total, total, len(_ledger))


def record_failure(site: str, exc: BaseException) -> None:
    """Append ``exc`` to the ledger under ``site``. Never raises."""
    global _total
    try:
        message = str(exc)
    except Exception as _err:  # noqa: BLE001 — pathological __str__
        message = f"<unprintable {type(_err).__name__}>"
    entry = Failure(ts=time.time(), site=site,
                    exc_type=type(exc).__name__, message=message)
    try:
        with _lock:
            _ensure_loaded()
            _ledger.append(entry)
            _total += 1
            _store.save([{"ts": f.ts, "site": f.site,
                          "exc_type": f.exc_type, "message": f.message}
                         for f in _ledger], _total)
    except Exception:  # noqa: BLE001 — the ledger never raises
        pass


def recent_failures(limit: int = 20) -> list[Failure]:
    """Newest-first slice of the ring."""
    with _lock:
        _ensure_loaded()
        return list(reversed(_ledger))[: max(0, int(limit))]


def failure_count() -> int:
    """Total failures recorded (including ones aged out of the ring)."""
    with _lock:
        _ensure_loaded()
        return _total


def clear_failures() -> None:
    """Empty the ledger (in memory and on disk)."""
    global _loaded, _total
    with _lock:
        _loaded = True
        _ledger.clear()
        _total = 0
        with suppress(Exception):
            _store.save([], 0)


def format_failures(limit: int = 8) -> str:
    """Plain-words rendering for status surfaces."""
    rows = recent_failures(limit)
    if not rows:
        return "no computer failures recorded."
    lines = [f"computer failures (latest {len(rows)} of {failure_count()}):"]
    for f in rows:
        stamp = time.strftime("%H:%M:%S", time.localtime(f.ts))
        lines.append(f"• {stamp} {f.site}: {f.exc_type}: {f.message[:140]}")
    return "\n".join(lines)
