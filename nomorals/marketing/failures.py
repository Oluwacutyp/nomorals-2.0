"""Failure ledger for the marketing subsystem.

``guardrails`` and ``competitor`` are deliberately never-raises modules:
every public function is wrapped and returns a safe default on error.
``send``'s queue/delivery pipeline shares the contract: a failed enqueue,
rate gate, cost gate, or status/cost write returns its safe default.
``send``'s campaigns/variants region shares it too: a failed estimate,
variant write, A/B report, or campaign funnel returns its empty shape.
``send``'s suppression/warmup/blocklist region shares it as well: a
failed unsubscribe, warmup write, or blocklist write returns ``False``,
and a failed suppression/blocklist read returns ``False`` — the
fail-open shape, indistinguishable from "not suppressed / not
blocklisted / no warmup" on a healthy DB. The gates that refuse sends
(``enqueue``/``process`` consult ``is_suppressed``,
``is_blocklisted``, and ``_warmup_ok``) fail open on a dead DB, so a
suppressed or hard-bounced address keeps receiving until the record
says otherwise.
``send``'s templates/providers region shares it as well: a failed
template save returns ``False`` like an invalid name, a failed
template/provider list returns ``[]`` like "nothing configured yet",
and a failed template lookup returns ``None`` like "no such
template" — on a dead config store the engine looks unconfigured
rather than broken, and a failed provider registration silently
leaves the provider unregistered.
``send``'s module helpers share it too: a failed profile import
degrades every batch decision to "pc", a failed cost-tracker import
silently disables cost tracking, a failed default-DB path swaps the
persistent queue for an in-memory one, a failed warmup plan uncaps
the ramp (``warmup_status`` falls back to the full daily target
when the plan is empty), and failed spam/grade/DNS helpers return
their empty shapes — all without a trace until now. The deliberate
non-conversions in the region stay pinned in the pass #137 pins:
``parse_dsn`` (unparseable input is the documented "unknown"
verdict), the ``_dns_txt`` per-resolver fall-throughs (the other
resolver still answers), and the ``deliverability_check`` DNS
inner (the resolution failure is itself reported as a check with
``ok: False``, lowering the grade).
That contract is sound for ad-spend safety — but until now the caught
failures vanished without a trace: a guardrail that silently failed to
evaluate, or a store method silently returning ``False`` on a dead DB,
looked identical to a healthy no-op. This ledger fixes that half of the
contract without touching the other half: failures are recorded, never
re-raised, and readable via ``/guardrails failures`` /
``/competitor failures``.

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

#: Ring size: enough to see a failing rule's history, small enough to
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
    """One human-readable block for the ``failures`` chat subcommands."""
    items = recent_failures(limit)
    if not items:
        return "no marketing failures recorded."
    lines = [f"⚠️ marketing failures ({failure_count()} total, "
             f"showing {len(items)} newest):"]
    for f in items:
        ts = time.strftime("%m-%d %H:%M", time.localtime(f.ts))
        lines.append(f"• {ts} {f.site}: {f.exc_type}: {f.message}")
    return "\n".join(lines)
