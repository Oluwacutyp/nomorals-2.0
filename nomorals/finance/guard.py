"""Pre-transaction warning intervention (MoMo pattern, #49).

Before an unusual outgoing transaction executes, Devon warns with context:
"this doesn't match your pattern — sure?" The warning UX IS the product.

Warning is NEVER blocking: an explicit "send it anyway" (per the #43
override rule) executes. Every warning and every decision is logged to the
ledger — the audit trail is the safety net.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .ledger import Ledger, format_naira

_log = logging.getLogger(__name__)

__all__ = ["Warning", "check_outgoing", "known_recipients", "median_outgoing"]

#: Hour range considered "unusual" for money movement (11pm–5am local).
UNUSUAL_HOURS = frozenset({23, 0, 1, 2, 3, 4})

#: Amount multiplier over the median that counts as "far above normal".
LARGE_MULTIPLIER = 3.0

#: Minimum history before amount-based signals fire (avoid false alarms on
#: a fresh ledger).
MIN_HISTORY_FOR_BASELINE = 3


@dataclass
class Warning:
    """A pre-transaction warning. Advisory only — never blocking."""

    message: str
    reasons: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "reasons": self.reasons,
            "context": self.context,
        }


def _outgoing(ledger: Ledger, *, days: int = 90) -> list[Any]:
    """Outgoing (spend-kind) transfer-like transactions for the baseline."""
    since = time.time() - days * 86400
    txns = ledger.transactions(since=since, kind="spend")
    # Transfer-like: notes mentioning a recipient ("to Mama", "transfer").
    return [t for t in txns if t.amount_kobo > 0]


def median_outgoing(ledger: Ledger, *, days: int = 90) -> int | None:
    """Median outgoing amount in kobo, or None when history is too thin."""
    amounts = sorted(t.amount_kobo for t in _outgoing(ledger, days=days))
    if len(amounts) < MIN_HISTORY_FOR_BASELINE:
        return None
    return int(statistics.median(amounts))


def known_recipients(ledger: Ledger, *, days: int = 180) -> set[str]:
    """Recipient names seen in outgoing transfer notes (lowercased)."""
    names: set[str] = set()
    for t in _outgoing(ledger, days=days):
        note = (t.note or "").lower()
        # Convention: transfer notes look like "transfer to <name>" or
        # "sent to <name>". Extract the trailing name.
        for marker in ("transfer to ", "sent to ", "send to ", "to "):
            if marker in note:
                name = note.split(marker, 1)[1].strip().split(",")[0].strip()
                if name:
                    names.add(name)
                break
    return names


def _is_round_number(kobo: int) -> bool:
    naira = kobo // 100
    return kobo % 100 == 0 and naira > 0 and naira % 10000 == 0


def check_outgoing(
    to: str,
    amount_kobo: int,
    ledger: Ledger,
    *,
    now: float | None = None,
) -> Warning | None:
    """Inspect an outgoing transfer for anomalies.

    Returns a Warning when something doesn't match the owner's pattern,
    else None. Pure function of (to, amount, history) — never raises.
    """
    try:
        return _check(to, amount_kobo, ledger, now=now or time.time())
    except Exception:  # noqa: BLE001 - the guard must never break a send
        _log.debug("guard check failed", exc_info=True)
        return None


def _check(to: str, amount_kobo: int, ledger: Ledger, now: float) -> Warning | None:
    reasons: list[str] = []
    context: dict[str, Any] = {}
    name = (to or "").strip().lower()

    median = median_outgoing(ledger)
    if median is not None and amount_kobo >= median * LARGE_MULTIPLIER:
        reasons.append(
            f"₦{amount_kobo // 100:,} is far above your usual "
            f"(median {format_naira(median)})"
        )
        context["median_kobo"] = median

    recipients = known_recipients(ledger)
    is_new_recipient = bool(name) and name not in recipients
    if is_new_recipient:
        reasons.append(f"{to.strip()} is a new recipient — never sent to before")
        context["new_recipient"] = True

    hour = datetime.fromtimestamp(now).hour
    if hour in UNUSUAL_HOURS:
        reasons.append(
            f"unusual time — {hour:02d}:00 is outside your normal hours"
        )
        context["unusual_hour"] = hour

    if is_new_recipient and _is_round_number(amount_kobo) and amount_kobo >= 1_000_000:
        reasons.append(
            "round-number large amount to a new recipient "
            "(common scam pattern — double-check who asked for this)"
        )
        context["scam_pattern"] = True

    if not reasons:
        return None

    usual = (
        f"your usual is {format_naira(median)} to known contacts"
        if median is not None else
        "you don't have enough history for a baseline yet"
    )
    bits = "; ".join(reasons)
    message = (
        f"⚠️ This doesn't match your pattern — {format_naira(amount_kobo)} "
        f"to {to.strip()} ({bits}). {usual.capitalize()}. Sure?"
    )
    context.update({
        "to": to.strip(), "amount_kobo": amount_kobo,
        "checked_at": now,
    })
    return Warning(message=message, reasons=reasons, context=context)
