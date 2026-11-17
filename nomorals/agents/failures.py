"""Failure ledger for the agents subsystem (morning pulse).

The morning pulse degrades instead of raising: a failed prefs read
returns defaults, a failed marker read returns empty, a failed stage
returns its degraded value — all by design, so one broken path never
silences the daily briefing. That contract is sound for an autonomous
flow — but until now the caught failures vanished without a trace: a
corrupt prefs sidecar silently reverting the pulse time, a dead
scheduler table read as "no job", a stale job that refuses removal
lingering past a disable, all looked identical to healthy behavior.
This ledger fixes that half of the contract without touching the
other half: failures are recorded, never re-raised, and readable from
the package.

Stdlib only. ``record`` itself never raises — a ledger that can crash
its callers would defeat the purpose.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from nomorals.core import failure_store as _failure_store

__all__ = ["Failure", "TriagedFailure", "record_failure", "recent_failures",
           "triaged_failures", "failure_count", "clear_failures",
           "format_failures"]

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


@dataclass(frozen=True)
class TriagedFailure:
    """One failure with its learned triage attached.

    ``verdict`` is "real" (needs attention) or "noise" when a trained
    classifier is loaded, and "ungraded" — with ``p_real`` 0.5 — when no
    classifier has ever been promoted. An ungraded row is waiting data,
    never an invented classification.
    """

    failure: Failure
    p_real: float
    verdict: str


_ledger: deque[Failure] = deque(maxlen=_MAX_FAILURES)
_lock = threading.Lock()
_total = 0

_store = _failure_store.Store('agents')
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


def triaged_failures(limit: int = 20, *, model: Any = None,
                      store: Any = None) -> list[TriagedFailure]:
    """Newest-first failures with the learned triage attached.

    ``model`` is a fitted classifier, or None to auto-load the latest
    promoted one via :func:`agents.failure_models.load_error_classifier`.
    With no model every row is "ungraded" (p_real 0.5) — the ledger
    surfaces it, exactly as before. Never raises: a bad store or a bad
    model degrades to ungraded, never to a crash in the read path.
    """
    items = recent_failures(limit)
    recent = [{"ts": f.ts, "site": f.site, "exc_type": f.exc_type,
               "message": f.message} for f in items]
    loaded = model
    if loaded is None:
        try:
            from .failure_models import load_error_classifier

            loaded = load_error_classifier(store)
        except Exception:  # noqa: BLE001 - a bad store never kills a read
            loaded = None
    out: list[TriagedFailure] = []
    for f in items:
        if loaded is None:
            out.append(TriagedFailure(failure=f, p_real=0.5,
                                      verdict="ungraded"))
            continue
        try:
            from .failure_models import classify_failure

            p = classify_failure(f, loaded, recent=recent)
        except Exception:  # noqa: BLE001 - a bad row never kills a read
            p = 0.5
        out.append(TriagedFailure(failure=f, p_real=p,
                                  verdict="real" if p >= 0.5 else "noise"))
    return out


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


def format_failures(limit: int = 20, *, triage: bool = True) -> str:
    """One human-readable block for reading the ledger back.

    With a trained classifier loaded, each line carries its triage
    (``[real 0.87]`` / ``[noise 0.12]``) and the header carries the
    split. With no classifier the output is byte-identical to the
    un-triaged format — an ungraded ledger is never reworded.
    ``triage=False`` forces the old format.
    """
    if triage:
        rows = triaged_failures(limit)
    else:
        rows = [TriagedFailure(failure=f, p_real=0.5, verdict="ungraded")
                for f in recent_failures(limit)]
    if not rows:
        return "no agents failures recorded."
    classified = [r for r in rows if r.verdict != "ungraded"]
    header = (f"⚠️ agents failures ({failure_count()} total, "
              f"showing {len(rows)} newest)")
    if classified:
        n_real = sum(1 for r in classified if r.verdict == "real")
        header += (f"; triaged: {n_real} real, "
                   f"{len(classified) - n_real} noise")
    lines = [header + ":"]
    for r in rows:
        f = r.failure
        ts = time.strftime("%m-%d %H:%M", time.localtime(f.ts))
        tag = (f" [{r.verdict} {r.p_real:.2f}]"
               if r.verdict != "ungraded" else "")
        lines.append(f"• {ts} {f.site}: {f.exc_type}: {f.message}{tag}")
    return "\n".join(lines)
