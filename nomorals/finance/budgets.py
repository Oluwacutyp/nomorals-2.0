"""Monthly per-category budgets + status + weekly digest.

All money stays in integer kobo. Budgets are keyed by "YYYY-MM" so a fresh
month starts clean.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .ledger import CATEGORIES, Ledger, format_naira

_log = get_logger("nomorals.finance")

DEFAULT_CURRENCY = "₦"

#: At 80% of budget → warning; at 100% → over.
WARN_THRESHOLD = 0.8
OVER_THRESHOLD = 1.0


def month_key(ts: float | None = None) -> str:
    """'2026-10' for the given (or current) timestamp, local time."""
    dt = datetime.fromtimestamp(time.time() if ts is None else ts).astimezone()
    return dt.strftime("%Y-%m")


@dataclass
class BudgetStatus:
    category: str
    month: str
    budgeted_kobo: int
    spent_kobo: int

    @property
    def remaining_kobo(self) -> int:
        return self.budgeted_kobo - self.spent_kobo

    @property
    def pct_used(self) -> float:
        if self.budgeted_kobo <= 0:
            return 0.0
        return self.spent_kobo / self.budgeted_kobo

    @property
    def state(self) -> str:
        if self.pct_used >= OVER_THRESHOLD:
            return "over"
        if self.pct_used >= WARN_THRESHOLD:
            return "warning"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "month": self.month,
            "budgeted_kobo": self.budgeted_kobo,
            "spent_kobo": self.spent_kobo,
            "remaining_kobo": self.remaining_kobo,
            "pct_used": round(self.pct_used, 3),
            "state": self.state,
        }


class BudgetStore:
    """Monthly budgets per category, JSON file. Thread-safe."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".nomorals" / "finance" / "budgets.json"
        self.path = Path(path)
        self._lock = threading.RLock()

    def _load(self) -> dict[str, dict[str, int]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    str(m): {str(c): int(v) for c, v in cats.items()}
                    for m, cats in data.items()
                    if isinstance(cats, dict)
                }
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return {}

    def _save(self, data: dict[str, dict[str, int]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def set_budget(
        self, category: str, amount_kobo: int, month: str | None = None
    ) -> None:
        if amount_kobo <= 0:
            raise ValueError("budget amount must be positive")
        category = category.strip().lower()
        month = month or month_key()
        with self._lock:
            data = self._load()
            data.setdefault(month, {})[category] = int(amount_kobo)
            self._save(data)
        _log.debug("budget set: %s %s %s", month, category,
                   format_naira(int(amount_kobo)))

    def get_budget(self, category: str, month: str | None = None) -> int | None:
        month = month or month_key()
        with self._lock:
            return self._load().get(month, {}).get(category.strip().lower())

    def list_budgets(self, month: str | None = None) -> dict[str, int]:
        month = month or month_key()
        with self._lock:
            return dict(self._load().get(month, {}))


def _month_bounds(month: str) -> tuple[float, float]:
    """(start_ts, end_ts) for a 'YYYY-MM' key, local time."""
    year, mon = (int(x) for x in month.split("-", 1))
    start = datetime(year, mon, 1).astimezone()
    if mon == 12:
        end = datetime(year + 1, 1, 1).astimezone()
    else:
        end = datetime(year, mon + 1, 1).astimezone()
    return start.timestamp(), end.timestamp()


def budget_status(
    ledger: Ledger,
    budgets: BudgetStore,
    month: str | None = None,
    currency: str = DEFAULT_CURRENCY,
) -> list[BudgetStatus]:
    """Per-category status for the month: budgeted, spent, remaining, % used."""
    month = month or month_key()
    start, end = _month_bounds(month)
    statuses: list[BudgetStatus] = []
    for category, budgeted in budgets.list_budgets(month).items():
        spent = ledger.total_spent(since=start, until=end, category=category)
        statuses.append(BudgetStatus(
            category=category, month=month,
            budgeted_kobo=budgeted, spent_kobo=spent,
        ))
    statuses.sort(key=lambda s: s.pct_used, reverse=True)
    return statuses


def overspend_alerts(
    statuses: list[BudgetStatus],
    warn_threshold: float = WARN_THRESHOLD,
    over_threshold: float = OVER_THRESHOLD,
) -> list[dict[str, Any]]:
    """Categories at warning/over level, worst first."""
    alerts = []
    for s in statuses:
        if s.pct_used >= over_threshold:
            level = "over"
        elif s.pct_used >= warn_threshold:
            level = "warning"
        else:
            continue
        alerts.append({"category": s.category, "level": level,
                       "pct_used": round(s.pct_used, 2),
                       "spent_kobo": s.spent_kobo,
                       "budgeted_kobo": s.budgeted_kobo})
    alerts.sort(key=lambda a: a["pct_used"], reverse=True)
    return alerts


def weekly_digest(
    ledger: Ledger,
    budgets: BudgetStore,
    now: float | None = None,
    currency: str = DEFAULT_CURRENCY,
) -> str:
    """Human-readable weekly summary: totals, top categories, budget flags."""
    now = time.time() if now is None else now
    week_ago = now - 7 * 86400
    two_weeks_ago = now - 14 * 86400

    spent = ledger.total_spent(since=week_ago, until=now)
    prev = ledger.total_spent(since=two_weeks_ago, until=week_ago)
    income = ledger.total_income(since=week_ago, until=now)

    lines = ["💰 weekly money digest"]
    lines.append(f"spent this week: {format_naira(spent, currency)}")
    if prev > 0:
        delta = spent - prev
        pct = abs(delta) / prev * 100
        direction = "up" if delta > 0 else "down"
        lines.append(
            f"vs last week ({format_naira(prev, currency)}): "
            f"{direction} {pct:.0f}%")
    if income:
        lines.append(f"income this week: {format_naira(income, currency)}")

    # Top categories this week.
    by_cat: dict[str, int] = {}
    for t in ledger.transactions(since=week_ago, until=now, kind="spend"):
        by_cat[t.category] = by_cat.get(t.category, 0) + t.amount_kobo
    top = sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if top:
        lines.append("top categories:")
        for cat, amt in top:
            lines.append(f"  • {cat}: {format_naira(amt, currency)}")

    # Budget flags for the current month.
    statuses = budget_status(ledger, budgets, month_key(now), currency)
    alerts = overspend_alerts(statuses)
    if alerts:
        lines.append("budget flags:")
        for a in alerts:
            if a["level"] == "over":
                times = a["spent_kobo"] / a["budgeted_kobo"] if a["budgeted_kobo"] else 0
                lines.append(
                    f"  ⚠️ over on {a['category']}: "
                    f"{format_naira(a['spent_kobo'], currency)} "
                    f"of {format_naira(a['budgeted_kobo'], currency)} "
                    f"budget ({times:.1f}x)")
            else:
                lines.append(
                    f"  • {a['category']} at {a['pct_used']:.0%} of budget "
                    f"({format_naira(a['spent_kobo'], currency)} / "
                    f"{format_naira(a['budgeted_kobo'], currency)})")
    elif statuses:
        lines.append("all budgets on track ✅")

    return "\n".join(lines)


def finance_paths(settings: Any = None) -> tuple[Path, Path]:
    """(ledger_path, budgets_path) honoring the runtime settings home."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home is not None:
            base = Path(home) / "finance"
            return base / "ledger.jsonl", base / "budgets.json"
    base = Path.home() / ".nomorals" / "finance"
    return base / "ledger.jsonl", base / "budgets.json"
