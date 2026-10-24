"""Failure ledger for the voice subsystem.

``voice`` is deliberately never-raises in its best-effort paths: a cache
read that fails, a telemetry write that dies, a progress callback that
throws — all fall back to safe defaults instead of breaking synthesis,
transcription, or a live session. That contract is sound for a voice
stack — but until now the caught failures vanished without a trace: a
segment cache that silently stopped caching, or telemetry that silently
stopped reporting, looked identical to a healthy no-op. This ledger
fixes that half of the contract without touching the other half:
failures are recorded, never re-raised, and readable from the package.

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

_store = _failure_store.Store('voice')
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
    """Newest-first slice of the ledger (oldest drops off at the cap)."""
    with _lock:
        _ensure_loaded()
        items = list(_ledger)
    return items[::-1][:max(0, limit)]


def failure_count() -> int:
    """Total failures ever recorded (ledger length is capped; this is not)."""
    with _lock:
        _ensure_loaded()
        return _total


def clear_failures() -> None:
    """Empty the ledger and reset the counter. Tests only."""
    global _total, _loaded
    with _lock:
        _ledger.clear()
        _total = 0
        _loaded = True
        _store.clear()


def format_failures(limit: int = 20) -> str:
    """One human-readable block for reading the ledger back."""
    items = recent_failures(limit)
    if not items:
        return "no voice failures recorded."
    lines = [f"⚠️ voice failures ({failure_count()} total, "
             f"showing {len(items)} newest):"]
    for f in items:
        ts = time.strftime("%m-%d %H:%M", time.localtime(f.ts))
        lines.append(f"• {ts} {f.site}: {f.exc_type}: {f.message}")
    return "\n".join(lines)
