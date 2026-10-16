"""Analytics: revenue from the ledger, plus the monthly client report.

Every number here comes from recorded data — the finance ledger
(``transactions()`` with ``kind="income"``) and, optionally, the
payment gateway's own payment list for channel splits. Nothing is
estimated, projected, or rounded into a nicer story: if the ledger is
empty, the report says so plainly.

``monthly_client_report`` answers "how well did I run your business?"
in markdown: revenue, transactions, refunds, fees, net, best day,
channel split, top categories, and an honest verdict line that states
exactly what the numbers do and do not prove.
"""

from __future__ import annotations

import calendar
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "revenue_by_day",
    "revenue_by_channel",
    "monthly_client_report",
    "period_summary",
]


def _txns(ledger: Any, since: float, until: float) -> list[Any]:
    try:
        txns = ledger.transactions()
    except Exception:  # noqa: BLE001 — ledger read failure → empty, stated
        return []
    out = []
    for t in txns or []:
        ts = float(getattr(t, "ts", 0) or 0)
        if since <= ts < until:
            out.append(t)
    return out


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _month_bounds(year: int, month: int) -> tuple[float, float]:
    start = datetime(year, month, 1, tzinfo=timezone.utc).timestamp()
    last = calendar.monthrange(year, month)[1]
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp()
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc).timestamp()
    return start, end


def period_summary(ledger: Any, *, since: float, until: float,
                   payments: list[Any] | None = None) -> dict[str, Any]:
    """Sum a period straight from the ledger. All figures minor units."""
    txns = _txns(ledger, since, until)
    income = [t for t in txns if str(getattr(t, "kind", "")) == "income"]
    spend = [t for t in txns if str(getattr(t, "kind", "")) == "spend"]
    fees = [t for t in spend
            if str(getattr(t, "category", "")).lower() == "fees"]
    refunds = [t for t in spend if "refund" in
               str(getattr(t, "category", "")).lower()
               or "refund" in str(getattr(t, "note", "")).lower()]
    gross = sum(int(getattr(t, "amount_kobo", 0) or 0) for t in income)
    fee_total = sum(int(getattr(t, "amount_kobo", 0) or 0) for t in fees)
    refund_total = sum(int(getattr(t, "amount_kobo", 0) or 0) for t in refunds)
    other_spend = sum(int(getattr(t, "amount_kobo", 0) or 0) for t in spend) \
        - fee_total - refund_total
    categories = Counter(
        str(getattr(t, "category", "other") or "other") for t in income)
    return {
        "gross_minor": gross,
        "fees_minor": fee_total,
        "refunds_minor": refund_total,
        "other_spend_minor": other_spend,
        "net_minor": gross - fee_total - refund_total,
        "income_txns": len(income),
        "spend_txns": len(spend),
        "total_txns": len(txns),
        "income_by_category": dict(categories),
        "payments_seen": len(payments or []),
    }


def revenue_by_day(ledger: Any, *, since: float, until: float) -> dict[str, int]:
    """Income per UTC day in the period. Minor units."""
    daily: dict[str, int] = defaultdict(int)
    for t in _txns(ledger, since, until):
        if str(getattr(t, "kind", "")) == "income":
            daily[_day(float(getattr(t, "ts", 0) or 0))] += \
                int(getattr(t, "amount_kobo", 0) or 0)
    return dict(sorted(daily.items()))


def revenue_by_channel(payments: list[Any], *, since: float = 0.0,
                       until: float = float("inf")) -> dict[str, int]:
    """Succeeded-payment revenue split by ``channel`` (whatsapp/online/...).

    Uses the payment records (they carry channel; ledger rows don't).
    Only ``status == "succeeded"`` counts. Refunds (negative amounts)
    are excluded here — they belong in the refunds line.
    """
    by_channel: dict[str, int] = defaultdict(int)
    for p in payments or []:
        if str(getattr(p, "status", "")) != "succeeded":
            continue
        ts = float(getattr(p, "created_at", 0) or 0)
        if not (since <= ts < until):
            continue
        by_channel[str(getattr(p, "channel", "online") or "online")] += \
            int(getattr(p, "amount_minor", 0) or 0)
    return dict(sorted(by_channel.items()))


def _fmt(minor: int, currency: str = "₦") -> str:
    return f"{currency}{minor / 100:,.2f}"


def monthly_client_report(ledger: Any, year: int, month: int, *,
                          store: str = "your business",
                          currency: str = "₦",
                          payments: list[Any] | None = None) -> str:
    """Monthly markdown report: 'how well did I run your business?'

    Built only from recorded ledger + payment data. An empty period
    reports zero with the words 'no recorded transactions' — never a
    fabricated upbeat summary.
    """
    start, end = _month_bounds(year, month)
    label = f"{calendar.month_name[month]} {year}"
    summary = period_summary(ledger, since=start, until=end,
                             payments=payments)
    daily = revenue_by_day(ledger, since=start, until=end)
    channels = revenue_by_channel(payments or [], since=start, until=end)
    lines = [
        f"# 📊 Monthly report — {store} ({label})",
        "",
        f"**How well did I run your business?** The numbers below are "
        f"from recorded payments and ledger entries for {label} — "
        f"nothing estimated.",
        "",
        "## Money in",
        "",
        f"- Gross revenue: **{_fmt(summary['gross_minor'], currency)}** "
        f"across {summary['income_txns']} payment(s)",
        f"- Refunds: {_fmt(summary['refunds_minor'], currency)}",
        f"- Gateway fees: {_fmt(summary['fees_minor'], currency)}",
        f"- **Net: {_fmt(summary['net_minor'], currency)}**",
        "",
    ]
    if daily:
        best_day, best_val = max(daily.items(), key=lambda kv: kv[1])
        lines += [
            "## Revenue by day",
            "",
            f"- Best day: **{best_day}** — {_fmt(best_val, currency)}",
            f"- Days with revenue: {len(daily)}",
            "",
        ]
    else:
        lines += ["_No recorded income transactions this month._", ""]
    if channels:
        lines += ["## Revenue by channel", ""]
        for ch, val in channels.items():
            lines.append(f"- {ch}: {_fmt(val, currency)}")
        lines.append("")
    if summary["income_by_category"]:
        lines += ["## Revenue by category", ""]
        for cat, n in sorted(summary["income_by_category"].items()):
            lines.append(f"- {cat}: {n} transaction(s)")
        lines.append("")
    if summary["total_txns"] == 0:
        lines.append(
            "_No recorded transactions this month — the business had no "
            "ledger activity, so there is nothing to rate._")
    else:
        verdict = (
            "✅ Revenue positive, refunds and fees tracked."
            if summary["net_minor"] > 0
            else "⚠️ Net is zero or negative this month — refunds/fees "
                 "ate the revenue. Review pricing and refund causes.")
        lines += ["## Verdict", "", verdict, "",
                  "_Verdict is arithmetic on recorded data only — it is "
                  "not a forecast._"]
    return "\n".join(lines)
