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
from datetime import datetime
from typing import Any

from ..core.logging_setup import get_logger
from .budgets import month_key
from .ledger import Ledger, extract_merchant, format_naira
from .style import current_theme, sparkline

_log = get_logger("nomorals.finance")

__all__ = [
    "Insights",
    "compute_insights",
    "detect_recurring",
    "render_insights",
    "upcoming_bills",
    "safe_to_spend",
    "forecast_month_end",
    "spend_series",
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
    # Rocket Money gold: did the price move? first vs last occurrence.
    first_amount_kobo: int = 0
    # Forecast: when the next charge is expected.
    next_due_ts: float = 0.0
    yearly_cost_kobo: int = 0

    @property
    def price_changed(self) -> bool:
        return bool(self.first_amount_kobo) and (
            self.first_amount_kobo != self.amount_kobo)

    @property
    def price_change_kobo(self) -> int:
        return self.amount_kobo - self.first_amount_kobo

    @property
    def monthly_cost_kobo(self) -> int:
        if self.avg_interval_days <= 0:
            return self.amount_kobo
        return int(self.amount_kobo * 30.0 / self.avg_interval_days)


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

    Groups by normalized note; a group qualifies when it has >=
    min_occurrences, the intervals between hits are roughly regular
    (coefficient of variation < 0.35, or all within 25–35 days for the
    monthly special-case), and the amounts are stable (all within 40% of
    the median — the same subscription, not coincidental same-merchant
    charges). Amount drift inside the band is KEPT (not bucketed away)
    so price hikes are detectable — Rocket Money's rate-increase flag.
    """
    groups: dict[str, list[Any]] = defaultdict(list)
    for t in txns:
        if t.kind != "spend":
            continue
        norm = re.sub(r"\s+", " ", (t.note or "").strip().lower())
        norm = re.sub(r"\d{4,}", "#", norm)  # mask refs/dates-ish numbers
        if not norm:
            continue
        groups[norm].append(t)
    out: list[RecurringCharge] = []
    for norm, hits in groups.items():
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
        # Amount stability: same subscription, not same-merchant noise.
        amounts = sorted(h.amount_kobo for h in hits)
        median_amt = amounts[len(amounts) // 2]
        if median_amt <= 0 or any(
                abs(a - median_amt) / median_amt > 0.4 for a in amounts):
            continue
        last_amount = hits[-1].amount_kobo
        first_amount = hits[0].amount_kobo
        next_due = hits[-1].ts + avg * 86400
        per_year = 365.0 / avg if avg > 0 else 0
        out.append(RecurringCharge(
            note_pattern=hits[-1].note or norm,
            category=hits[-1].category,
            amount_kobo=last_amount,
            occurrences=len(hits),
            avg_interval_days=round(avg, 1),
            last_ts=hits[-1].ts,
            first_amount_kobo=first_amount,
            next_due_ts=next_due,
            yearly_cost_kobo=int(last_amount * per_year),
        ))
    return sorted(out, key=lambda r: r.amount_kobo, reverse=True)


def upcoming_bills(ledger: Ledger, *, days: int = 14,
                   now: float | None = None) -> list[RecurringCharge]:
    """Recurring charges due within ``days``. Rocket Money's core view."""
    now = time.time() if now is None else now
    txns = _window(ledger, 120, now)
    return [r for r in detect_recurring(txns)
            if 0 < r.next_due_ts - now <= days * 86400]


def safe_to_spend(ledger: Ledger, *, now: float | None = None,
                  days: int = 30) -> dict[str, Any]:
    """Quicken Simplifi's killer number, natively.

    expected monthly income − committed monthly recurring − spent this
    month so far = what you can still spend without touching savings.
    """
    now = time.time() if now is None else now
    txns = _window(ledger, days * 3, now)
    income_txns = [t for t in txns if t.kind == "income"]
    avg_income = (sum(t.amount_kobo for t in income_txns)
                  / (days * 3 / 30)) if income_txns else 0
    recurring = detect_recurring(txns)
    committed = sum(r.monthly_cost_kobo for r in recurring)
    mk = month_key(now)
    year, mon = (int(x) for x in mk.split("-", 1))
    start = datetime(year, mon, 1).astimezone().timestamp()
    spent_mtd = ledger.total_spent(since=start, until=now)
    safe = int(avg_income - committed - spent_mtd)
    return {
        "safe_kobo": max(0, safe),
        "expected_income_kobo": int(avg_income),
        "committed_recurring_kobo": committed,
        "spent_this_month_kobo": spent_mtd,
        "recurring_count": len(recurring),
    }


def forecast_month_end(ledger: Ledger, *, now: float | None = None) -> int:
    """Projected total spend by month end at the current burn rate."""
    now = time.time() if now is None else now
    ins = compute_insights(ledger, window_days=30, now=now)
    mk = month_key(now)
    year, mon = (int(x) for x in mk.split("-", 1))
    start = datetime(year, mon, 1).astimezone().timestamp()
    if mon == 12:
        end = datetime(year + 1, 1, 1).astimezone().timestamp()
    else:
        end = datetime(year, mon + 1, 1).astimezone().timestamp()
    days_left = max(0.0, (end - now) / 86400)
    spent_mtd = ledger.total_spent(since=start, until=now)
    return spent_mtd + int(ins.burn_rate_kobo_per_day * days_left)


def spend_series(ledger: Ledger, *, months: int = 6,
                 now: float | None = None) -> list[tuple[str, int]]:
    """Monthly (YYYY-MM, spend_kobo) totals, oldest → newest. For sparklines."""
    now = time.time() if now is None else now
    out = []
    dt = datetime.fromtimestamp(now).astimezone().replace(day=1)
    for _ in range(max(1, months)):
        year, mon = dt.year, dt.month
        start = datetime(year, mon, 1).astimezone().timestamp()
        if mon == 12:
            end = datetime(year + 1, 1, 1).astimezone().timestamp()
        else:
            end = datetime(year, mon + 1, 1).astimezone().timestamp()
        out.append((f"{year}-{mon:02d}",
                    ledger.total_spent(since=start, until=min(end, now))))
        # step back one month
        if mon == 1:
            dt = dt.replace(year=year - 1, month=12)
        else:
            dt = dt.replace(month=mon - 1)
    return out[::-1]


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
        merchant = extract_merchant(t.note)
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


def render_insights(ins: Insights, theme: str | None = None) -> str:
    """Human-readable analytics report."""
    th = current_theme(theme)
    lines = [th.paint(f"{th.money_bag} money insights — last "
                      f"{ins.window_days} days", th.bold)]
    lines.append(f"spent: {format_naira(ins.total_spent_kobo)} · "
                 f"income: {format_naira(ins.total_income_kobo)}")
    lines.append(f"burn rate: {format_naira(ins.burn_rate_kobo_per_day)}/day")
    lines.append(f"savings rate: {ins.savings_rate:.0%}")
    if ins.runway_days is not None:
        lines.append(f"runway at this burn: {ins.runway_days:.0f} days")
    if ins.trend_pct is not None:
        direction = "up" if ins.trend_pct > 0 else "down"
        color = th.bad if ins.trend_pct > 0 else th.good
        lines.append(th.paint(
            f"vs previous {ins.window_days}d: {direction} "
            f"{abs(ins.trend_pct):.0%}", color))
    if ins.top_categories:
        lines.append("top categories:")
        for cat, amt in ins.top_categories:
            lines.append(f"  {th.bullet} {cat}: {format_naira(amt)}")
    if ins.top_merchants:
        lines.append("top merchants:")
        for m, amt in ins.top_merchants:
            lines.append(f"  {th.bullet} {m}: {format_naira(amt)}")
    if ins.recurring:
        lines.append("recurring charges (subscriptions?):")
        for r in ins.recurring:
            cadence = ("monthly" if 25 <= r.avg_interval_days <= 35
                       else f"~every {r.avg_interval_days:.0f}d")
            hike = ""
            if r.price_changed:
                arrow = "📈" if r.price_change_kobo > 0 else "📉"
                hike = th.paint(
                    f" {arrow} {format_naira(abs(r.price_change_kobo))} "
                    f"vs first charge", th.warn if r.price_change_kobo > 0
                    else th.good) if th.use_emoji else th.paint(
                    f" ({'+' if r.price_change_kobo > 0 else '-'}"
                    f"{format_naira(abs(r.price_change_kobo))} vs first)",
                    th.warn)
            lines.append(f"  {th.bullet} {r.note_pattern} — "
                         f"{format_naira(r.amount_kobo)} {cadence} "
                         f"({r.occurrences}×){hike}")
    return "\n".join(lines)


def render_recurring(recurring: list[RecurringCharge],
                     theme: str | None = None) -> str:
    """Standalone recurring-charge view: cost, cadence, hikes, next due."""
    th = current_theme(theme)
    if not recurring:
        return "no recurring charges detected yet — log a few months of " \
               "spending and I'll find your subscriptions."
    lines = [th.paint(f"{th.money_bag} recurring charges — "
                      f"{len(recurring)} found", th.bold)]
    total_year = sum(r.yearly_cost_kobo for r in recurring)
    lines.append(th.paint(
        f"committed: {format_naira(total_year)}/year", th.accent))
    for r in recurring:
        cadence = ("monthly" if 25 <= r.avg_interval_days <= 35
                   else f"~every {r.avg_interval_days:.0f}d")
        due = ""
        if r.next_due_ts:
            d = datetime.fromtimestamp(r.next_due_ts).astimezone()
            due = f" · next ~{d.strftime('%b %d')}"
        hike = ""
        if r.price_changed and r.price_change_kobo > 0:
            hike = th.paint(
                f"  {th.alert} price up "
                f"{format_naira(r.price_change_kobo)} since first charge",
                th.warn)
        lines.append(f"  {th.bullet} {r.note_pattern} — "
                     f"{format_naira(r.amount_kobo)} {cadence}{due}{hike}")
    return "\n".join(lines)
