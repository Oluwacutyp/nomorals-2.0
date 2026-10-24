"""Failure ledger for honest-degradation media rendering.

``fidelity`` and ``genedit`` are both built on an honest-degradation
contract: when the best path is unavailable, fall back to a safe
default instead of raising — a model route that returns "no candidate"
instead of inventing a model, a shot that drops into the timeline like
any footage. That contract is sound for a production engine — but until
now the caught failures vanished without a trace. A broker whose
``select`` raises for every task looked like a registry with no
``select``; a registry whose ``cards`` raises looked empty; a card
with a malformed quality attribute looked unrated; a broken prompt
engine looked like plain prompts; a broken imggen import looked like
passthrough-by-design; a failed keyframe restyle looked like a full
img2img shot. The degradation was honest in return-value shape but
silent about the underlying error — now it leaves a trace.

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
        return "no recorded render failures"
    lines = []
    for entry in entries:
        lines.append(f"{entry.site}: {entry.exc_type}: {entry.message}")
    return "\n".join(lines)
