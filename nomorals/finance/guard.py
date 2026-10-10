"""Pre-transaction warning intervention (MoMo pattern, #49).

Before an unusual outgoing transaction executes, Devon warns with context:
"this doesn't match your pattern — sure?" The warning UX IS the product.

Warning is NEVER blocking: an explicit "send it anyway" (per the #43
override rule) executes. Every warning and every decision is logged to the
ledger — the audit trail is the safety net.

Scoring follows the fraud-literature pattern (PaySim/Z-score work): model
*normal* behavior from history, then score deviations as a weighted
0–100 risk score with per-signal contributions — "why is this 82/100?"
is always answerable.
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

__all__ = [
    "Warning",
    "check_outgoing",
    "known_recipients",
    "median_outgoing",
    "risk_score",
    "render_warning",
]

#: Hour range considered "unusual" for money movement (11pm–5am local).
UNUSUAL_HOURS = frozenset({23, 0, 1, 2, 3, 4})

#: Amount multiplier over the median that counts as "far above normal".
LARGE_MULTIPLIER = 3.0

#: Minimum history before amount-based signals fire (avoid false alarms on
#: a fresh ledger).
MIN_HISTORY_FOR_BASELINE = 3

#: Score bands.
WARN_SCORE = 60
STRONG_WARN_SCORE = 85

#: Signal weights for the 0–100 risk score.
_W_AMOUNT_Z = 35
_W_NEW_RECIPIENT = 25
_W_UNUSUAL_HOUR = 15
_W_VELOCITY = 15
_W_ROUND_SCAM = 10


@dataclass
class Warning:
    """A pre-transaction warning. Advisory only — never blocking."""

    message: str
    reasons: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    score: int = 0  # 0–100 weighted risk score

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "reasons": self.reasons,
            "context": self.context,
            "score": self.score,
        }


def _outgoing(ledger: Ledger, *, days: int = 90,
              now: float | None = None) -> list[Any]:
    """Outgoing (spend-kind) transfer-like transactions for the baseline."""
    now = time.time() if now is None else now
    since = now - days * 86400
    txns = ledger.transactions(since=since, until=now, kind="spend")
    # Transfer-like: notes mentioning a recipient ("to Mama", "transfer").
    return [t for t in txns if t.amount_kobo > 0]


def median_outgoing(ledger: Ledger, *, days: int = 90,
                    now: float | None = None) -> int | None:
    """Median outgoing amount in kobo, or None when history is too thin."""
    amounts = sorted(t.amount_kobo for t in _outgoing(ledger, days=days,
                                                      now=now))
    if len(amounts) < MIN_HISTORY_FOR_BASELINE:
        return None
    return int(statistics.median(amounts))


def known_recipients(ledger: Ledger, *, days: int = 180,
                     now: float | None = None) -> set[str]:
    """Recipient names seen in outgoing transfer notes (lowercased)."""
    names: set[str] = set()
    for t in _outgoing(ledger, days=days, now=now):
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


def _amount_zscore(amount_kobo: int, ledger: Ledger,
                   category: str = "", now: float | None = None) -> float | None:
    """Z-score of the amount vs history (category-scoped when possible).

    Models *normal* behavior, flags deviations — the MDPI/ARIMA intuition
    without the model: per-category when history allows, global fallback.
    """
    txns = [t for t in _outgoing(ledger, now=now)
            if not category or t.category == category]
    if len(txns) < MIN_HISTORY_FOR_BASELINE:
        txns = _outgoing(ledger, now=now)
    amounts = [t.amount_kobo for t in txns]
    if len(amounts) < MIN_HISTORY_FOR_BASELINE:
        return None
    mean = statistics.fmean(amounts)
    try:
        stdev = statistics.stdev(amounts)
    except statistics.StatisticsError:
        return None
    if stdev <= 0:
        return None
    return (amount_kobo - mean) / stdev


def _velocity_24h(ledger: Ledger, now: float) -> int:
    """Outgoing sends in the trailing 24h — the PaySim velocity signal."""
    return len([t for t in _outgoing(ledger, days=2, now=now)
                if t.ts >= now - 86400])


def risk_score(to: str, amount_kobo: int, ledger: Ledger, *,
               now: float | None = None) -> dict[str, Any]:
    """Weighted 0–100 risk score with per-signal contributions.

    Never raises — a broken signal contributes 0, not a crash.
    """
    now = now if now is not None else time.time()
    signals: dict[str, float] = {}
    try:
        z = _amount_zscore(amount_kobo, ledger, now=now)
        if z is not None and z > 1.0:
            # z=1 → ~12pts, z=3 → full 35pts.
            signals["amount_vs_history"] = min(
                _W_AMOUNT_Z, _W_AMOUNT_Z * (z - 1.0) / 2.0)
    except Exception:  # noqa: BLE001
        _log.debug("amount signal failed", exc_info=True)

    name = (to or "").strip().lower()
    try:
        if name and name not in known_recipients(ledger, now=now):
            signals["new_recipient"] = _W_NEW_RECIPIENT
    except Exception:  # noqa: BLE001
        _log.debug("recipient signal failed", exc_info=True)

    try:
        if datetime.fromtimestamp(now).hour in UNUSUAL_HOURS:
            signals["unusual_hour"] = _W_UNUSUAL_HOUR
    except Exception:  # noqa: BLE001
        pass

    try:
        n = _velocity_24h(ledger, now)
        if n >= 3:
            # 3 sends → half weight, 6+ → full.
            signals["velocity"] = min(_W_VELOCITY,
                                      _W_VELOCITY * (n - 2) / 4.0)
    except Exception:  # noqa: BLE001
        _log.debug("velocity signal failed", exc_info=True)

    try:
        new_recip = "new_recipient" in signals
        if new_recip and _is_round_number(amount_kobo) \
                and amount_kobo >= 1_000_000:
            signals["round_number_scam_pattern"] = _W_ROUND_SCAM
    except Exception:  # noqa: BLE001
        pass

    total = int(round(sum(signals.values())))
    return {
        "score": min(100, total),
        "level": ("high" if total >= STRONG_WARN_SCORE
                  else "elevated" if total >= WARN_SCORE else "normal"),
        "signals": {k: round(v, 1) for k, v in signals.items()},
    }


def _check(to: str, amount_kobo: int, ledger: Ledger, now: float) -> Warning | None:
    reasons: list[str] = []
    context: dict[str, Any] = {}
    name = (to or "").strip().lower()

    score = risk_score(to, amount_kobo, ledger, now=now)
    context["risk_score"] = score["score"]
    context["risk_signals"] = score["signals"]
    context["risk_level"] = score["level"]

    median = median_outgoing(ledger, now=now)
    if median is not None and amount_kobo >= median * LARGE_MULTIPLIER:
        reasons.append(
            f"₦{amount_kobo // 100:,} is far above your usual "
            f"(median {format_naira(median)})"
        )
        context["median_kobo"] = median

    recipients = known_recipients(ledger, now=now)
    is_new_recipient = bool(name) and name not in recipients
    if is_new_recipient:
        reasons.append(
            f"{to.strip()} is a new recipient — never sent to before")
        context["new_recipient"] = True

    hour = datetime.fromtimestamp(now).hour
    if hour in UNUSUAL_HOURS:
        reasons.append(
            f"unusual time — {hour:02d}:00 is outside your normal hours"
        )
        context["unusual_hour"] = hour

    if is_new_recipient and _is_round_number(amount_kobo) \
            and amount_kobo >= 1_000_000:
        reasons.append(
            "round-number large amount to a new recipient "
            "(common scam pattern — double-check who asked for this)"
        )
        context["scam_pattern"] = True

    velocity = _velocity_24h(ledger, now)
    if velocity >= 3:
        reasons.append(
            f"high send velocity — {velocity} outgoing transfers in 24h")
        context["velocity_24h"] = velocity

    z = _amount_zscore(amount_kobo, ledger, now=now)
    if z is not None and z >= 2.0:
        reasons.append(
            f"amount is {z:.1f}σ above your normal (z-score)")
        context["amount_zscore"] = round(z, 2)

    if score["score"] < WARN_SCORE and not reasons:
        return None

    usual = (
        f"your usual is {format_naira(median)} to known contacts"
        if median is not None else
        "you don't have enough history for a baseline yet"
    )
    bits = "; ".join(reasons) if reasons else "risk score elevated"
    strength = "🚨" if score["score"] >= STRONG_WARN_SCORE else "⚠️"
    message = (
        f"{strength} This doesn't match your pattern — "
        f"{format_naira(amount_kobo)} to {to.strip()} "
        f"(risk {score['score']}/100: {bits}). "
        f"{usual.capitalize()}. Sure?"
    )
    context.update({
        "to": to.strip(), "amount_kobo": amount_kobo,
        "checked_at": now,
    })
    return Warning(message=message, reasons=reasons, context=context,
                   score=score["score"])


def render_warning(w: Warning) -> str:
    """Readable breakdown of a warning — the "why, and the numbers" view."""
    lines = [w.message, "", "why this scored "
             f"{w.context.get('risk_score', w.score)}/100:"]
    for signal, pts in (w.context.get("risk_signals") or {}).items():
        lines.append(f"  • {signal.replace('_', ' ')}: +{pts:g} pts")
    if not (w.context.get("risk_signals") or {}):
        lines.append("  • (no strong signals — baseline thin)")
    return "\n".join(lines)
