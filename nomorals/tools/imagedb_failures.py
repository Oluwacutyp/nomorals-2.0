"""Failure ledger for the image library and reverse-image search.

``imagedb`` is deliberately never-raises in its best-effort paths: a
dimension read that cannot parse, an EXIF read that cannot open, a
field conversion that cannot coerce, a GPS block that cannot assemble,
an index write that cannot land, a near-duplicate scan that cannot
run, a library query that cannot execute, a clustering step that
cannot complete, a registry bind that cannot persist, an engine
descriptor that cannot resolve, an engine ordering that cannot sort,
an outcome record that cannot book, a URL dedupe that cannot decode,
a bot-signal vocabulary that cannot import — all fall back to safe
defaults instead of breaking the caller's lookup or search. That
contract is sound for a production image tool — but until now the
caught failures vanished without a trace. A failed index write that
silently looked like a first sighting, a dropped outcome record that
silently corrupted the success rates the ordering learns from, a
registry bind that silently left outcomes unpersisted, looked
identical to a healthy library — now each leaves a trace.

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
        return "no recorded image failures"
    lines = []
    for entry in entries:
        lines.append(f"{entry.site}: {entry.exc_type}: {entry.message}")
    return "\n".join(lines)
