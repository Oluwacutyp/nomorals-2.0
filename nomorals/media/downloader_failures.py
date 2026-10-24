"""Failure ledger for the unified media download orchestrator.

``downloader`` is deliberately never-raises in its best-effort paths:
a profile probe that cannot resolve, a bookkeeping write that cannot
land, a cookie store that cannot open, a candidate parse that cannot
read, a progress report that cannot send, a hash that cannot compute
— all fall back to safe defaults instead of breaking the caller's
download. That contract is sound for a production engine — but until
now the caught failures vanished without a trace. A profile probe that
silently defaulted, or a bookkeeping write that silently never landed,
looked identical to a healthy no-op — now it leaves a trace.

Stdlib only. ``record`` itself never raises — a ledger that can crash
its callers would defeat the purpose.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from nomorals.core import failure_store as _failure_store

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

_store = _failure_store.Store('media.downloader')
_loaded = False


def _ensure_loaded() -> None:
    """Load persisted entries once per process. Caller must hold ``_lock``.

    Runs on first access only: the ring is empty at that point (every
    path through this module loads first), so no duplication is
    possible. ``clear_failures`` sets ``_loaded`` without loading.
    """
    global _loaded, _total
    if _loaded:
        return
    _loaded = True
    entries, total = _store.load()
    for row in entries:
        _ledger.append(Failure(ts=row["ts"], site=row["site"],
                               exc_type=row["exc_type"], message=row["message"]))
    _total = max(_total, total, len(_ledger))

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
        _ensure_loaded()
        _ledger.append(entry)
        _total += 1
        _store.save(
            [{"ts": f.ts, "site": f.site, "exc_type": f.exc_type,
              "message": f.message} for f in _ledger], _total)


def recent_failures(limit: int = 20) -> list[Failure]:
    """Newest-first snapshot of the ledger (up to ``limit`` entries)."""
    with _lock:
        _ensure_loaded()
        return list(reversed(_ledger))[:limit]


def failure_count() -> int:
    """Total failures recorded since process start (ring may have dropped old ones)."""
    with _lock:
        _ensure_loaded()
        return _total


def clear_failures() -> None:
    """Empty the ring and reset the count (tests, mostly)."""
    global _total, _loaded
    with _lock:
        _ledger.clear()
        _total = 0
        _loaded = True
        _store.clear()


def format_failures(limit: int = 20) -> str:
    """Human-readable render of the newest ledger entries, or an empty note."""
    entries = recent_failures(limit)
    if not entries:
        return "no recorded downloader failures"
    lines = []
    for entry in entries:
        lines.append(f"{entry.site}: {entry.exc_type}: {entry.message}")
    return "\n".join(lines)
