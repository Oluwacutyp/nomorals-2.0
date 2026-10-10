"""Native money analytics — burn rate, savings rate, recurring charges.

This is Devon's own analysis over the ledger: no API, no LLM needed.
Everything here is computed from the transaction history the owner
already has.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger
from .budgets import month_key
from .ledger import Ledger, format_naira

_log = get_logger("nomorals.finance")

__all__ = [
    "Insights",
    "compute_insights",
    "detect_recurring",
    "render_insights",
]


@dataclass
class RecurringCharge:
    """A charge that repeats on a roughly fixed cadence."""

    note_pattern: str
    category: str
    amount_kobo: int
    occurrences: int
    avg_interval_days: float
    last_ts: float


@dataclass
class Insights:
    """Computed analytics snapshot."""

    window_days: int
    total_spent_kobo: int
    total_income_kobo: int
    burn_rate_kobo_per_day: int
    savings_rate: float          # 0..1 (0 when no income)
    runway_days: float | None    # income==0 → None
    top_categories: list[tuple[str, int]]
    top_merchants: list[tuple[str, int]]
    recurring: list[RecurringCharge]
    trend_pct: float | None      # spend vs previous window (None = no baseline)


def _window(ledger: Ledger, days: int, now: float,
            offset_days: int = 0) -> list[Any]:
    end = now - offset_days * 86400
    start = end - days * 86400
    return ledger.transactions(since=start, until=end)


def detect_recurring(txns: list[Any], *, min_occurrences: int = 3) -> list[RecurringCharge]:
    """Find charges repeating on a fixed cadence (subscriptions etc.).

    Groups by normalized note + amount bucket; a group qualifies when it
    has >= min_occurrences and the intervals between hits are roughly
    regular (coefficient of variation < 0.35, or all within 25–35 days
    for the monthly special-case).
    """
    groups: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for t in txns:
        if t.kind != "spend":
            continue
        norm = re.sub(r"\s+", " ", (t.note or "").strip().lower())
        norm = re.sub(r"\d{4,}", "#", norm)  # mask refs/dates-ish numbers
        if not norm:
            continue
        bucket = int(round(t.amount_kobo / 1000.0))  # ₦10 buckets
        groups[(norm, bucket)].append(t)
    out: list[RecurringCharge] = []
    for (norm, _bucket), hits in groups.items():
        if len(hits) < min_occurrences:
            continue
        hits.sort(key=lambda t: t.ts)
        intervals = [b.ts - a.ts for a, b in zip(hits, hits[1:])]
        intervals = [i / 86400 for i in intervals if i > 0]
        if not intervals:
            continue
        avg = sum(intervals) / len(intervals)
        if avg <= 0:
            continue
        var = sum((i - avg) ** 2 for i in intervals) / len(intervals)
        cv = (var ** 0.5) / avg
        monthly = all(25 <= i <= 35 for i in intervals)
        if cv >= 0.35 and not monthly:
            continue
        out.append(RecurringCharge(
            note_pattern=hits[-1].note or norm,
            category=hits[-1].category,
            amount_kobo=hits[-1].amount_kobo,
            occurrences=len(hits),
            avg_interval_days=round(avg, 1),
            last_ts=hits[-1].ts,
        ))
    return sorted(out, key=lambda r: r.amount_kobo, reverse=True)


def compute_insights(
    ledger: Ledger,
    *,
    window_days: int = 30,
    now: float | None = None,
) -> Insights:
    """Compute the analytics snapshot over the trailing window."""
    now = time.time() if now is None else now
    cur = _window(ledger, window_days, now)
    prev = _window(ledger, window_days, now, offset_days=window_days)

    spent = sum(t.amount_kobo for t in cur if t.kind == "spend")
    income = sum(t.amount_kobo for t in cur if t.kind == "income")
    prev_spent = sum(t.amount_kobo for t in prev if t.kind == "spend")

    burn = spent / window_days if window_days else 0
    savings_rate = max(0.0, (income - spent) / income) if income > 0 else 0.0
    runway = (income / burn) * window_days / window_days if burn > 0 else None
    # runway: at current burn, days the window's income would last
    runway = (income / burn) if burn > 0 and income > 0 else None

    by_cat: dict[str, int] = defaultdict(int)
    by_merchant: dict[str, int] = defaultdict(int)
    for t in cur:
        if t.kind != "spend":
            continue
        by_cat[t.category] += t.amount_kobo
        note = (t.note or "").strip()
        if note:
            merchant = note.split(",")[0].split("-")[0].strip()[:40]
            if merchant:
                by_merchant[merchant] += t.amount_kobo

    trend = None
    if prev_spent > 0:
        trend = (spent - prev_spent) / prev_spent

    return Insights(
        window_days=window_days,
        total_spent_kobo=spent,
        total_income_kobo=income,
        burn_rate_kobo_per_day=int(burn),
        savings_rate=round(savings_rate, 3),
        runway_days=round(runway, 1) if runway is not None else None,
        top_categories=sorted(by_cat.items(), key=lambda kv: kv[1],
                              reverse=True)[:6],
        top_merchants=sorted(by_merchant.items(), key=lambda kv: kv[1],
                             reverse=True)[:6],
        recurring=detect_recurring(cur),
        trend_pct=round(trend, 3) if trend is not None else None,
    )


def render_insights(ins: Insights) -> str:
    """Human-readable analytics report."""
    lines = [f"📊 money insights — last {ins.window_days} days"]
    lines.append(f"spent: {format_naira(ins.total_spent_kobo)} · "
                 f"income: {format_naira(ins.total_income_kobo)}")
    lines.append(f"burn rate: {format_naira(ins.burn_rate_kobo_per_day)}/day")
    lines.append(f"savings rate: {ins.savings_rate:.0%}")
    if ins.runway_days is not None:
        lines.append(f"runway at this burn: {ins.runway_days:.0f} days")
    if ins.trend_pct is not None:
        direction = "up" if ins.trend_pct > 0 else "down"
        lines.append(f"vs previous {ins.window_days}d: {direction} "
                     f"{abs(ins.trend_pct):.0%}")
    if ins.top_categories:
        lines.append("top categories:")
        for cat, amt in ins.top_categories:
            lines.append(f"  • {cat}: {format_naira(amt)}")
    if ins.top_merchants:
        lines.append("top merchants:")
        for m, amt in ins.top_merchants:
            lines.append(f"  • {m}: {format_naira(amt)}")
    if ins.recurring:
        lines.append("recurring charges (subscriptions?):")
        for r in ins.recurring:
            cadence = ("monthly" if 25 <= r.avg_interval_days <= 35
                       else f"~every {r.avg_interval_days:.0f}d")
            lines.append(f"  • {r.note_pattern} — "
                         f"{format_naira(r.amount_kobo)} {cadence} "
                         f"({r.occurrences}×)")
    return "\n".join(lines)
