"""Finance tools: log spends, summaries, budgets."""

from __future__ import annotations

from typing import Any

from .budgets import (
    BudgetStore,
    budget_status,
    finance_paths,
    month_key,
    weekly_digest,
)
from .ledger import Ledger, format_naira, parse_amount


def _ledger_budgets(context: Any) -> tuple[Ledger, BudgetStore]:
    settings = getattr(context, "settings", None)
    ledger_path, budgets_path = finance_paths(settings)
    return Ledger(ledger_path), BudgetStore(budgets_path)


def register(registry: Any) -> None:
    @registry.register(
        "finance_log",
        description=(
            "Log a spend or income. Args: amount (e.g. '5k', 2000, '₦1,500' — "
            "Nigerian shorthand ok), category (optional, auto-detected from "
            "note), note (optional text), kind (spend|income, default spend)."
        ),
        capability="finance",
        parameters={
            "amount": "str|int — amount, e.g. '5k' or 2000",
            "category": "str — optional category (auto-detected when omitted)",
            "note": "str — optional note",
            "kind": "str — spend|income (default spend)",
        },
    )
    def finance_log(
        context: Any,
        amount: Any,
        category: str = "",
        note: str = "",
        kind: str = "spend",
    ) -> dict[str, Any]:
        ledger, _ = _ledger_budgets(context)
        kobo = parse_amount(str(amount)) if not isinstance(amount, int) else int(amount) * 100
        if kobo is None or kobo <= 0:
            return {"ok": False, "error": f"couldn't parse amount {amount!r}"}
        txn = ledger.log(kobo, category=category or None, note=note, kind=kind)
        return {
            "ok": True,
            "amount": format_naira(txn.amount_kobo),
            "category": txn.category,
            "kind": txn.kind,
            "note": txn.note,
        }

    @registry.register(
        "finance_summary",
        description=(
            "Spending summary vs budgets. Args: period (week|month, default "
            "month), category (optional filter)."
        ),
        capability="finance",
        parameters={
            "period": "str — week|month (default month)",
            "category": "str — optional category filter",
        },
    )
    def finance_summary(
        context: Any, period: str = "month", category: str = ""
    ) -> dict[str, Any]:
        import time

        ledger, budgets = _ledger_budgets(context)
        period = (period or "month").lower()
        if period == "week":
            return {"ok": True, "period": "week",
                    "text": weekly_digest(ledger, budgets)}
        statuses = budget_status(ledger, budgets, month_key())
        cat = (category or "").strip().lower()
        if cat:
            statuses = [s for s in statuses if s.category == cat]
        lines = [f"spending — {month_key()}"]
        total = 0
        for s in statuses:
            total += s.spent_kobo
            lines.append(
                f"  • {s.category}: {format_naira(s.spent_kobo)} / "
                f"{format_naira(s.budgeted_kobo)} ({s.pct_used:.0%}) — {s.state}")
        if not statuses:
            lines.append("  no budgets set — /budget set <category> <amount>")
        lines.append(f"total tracked: {format_naira(total)}")
        return {"ok": True, "period": "month", "text": "\n".join(lines),
                "statuses": [s.to_dict() for s in statuses]}

    @registry.register(
        "finance_budget",
        description=(
            "Set or list monthly budgets. Args: action (set|list), category, "
            "amount (for set, e.g. '50k')."
        ),
        capability="finance",
        parameters={
            "action": "str — set|list",
            "category": "str — category (for set)",
            "amount": "str|int — amount (for set, e.g. '50k')",
        },
    )
    def finance_budget(
        context: Any, action: str = "list", category: str = "", amount: Any = ""
    ) -> dict[str, Any]:
        _, budgets = _ledger_budgets(context)
        action = (action or "list").lower()
        if action == "set":
            kobo = parse_amount(str(amount)) if not isinstance(amount, int) else int(amount) * 100
            if kobo is None or kobo <= 0:
                return {"ok": False, "error": f"couldn't parse amount {amount!r}"}
            if not category.strip():
                return {"ok": False, "error": "category is required for set"}
            budgets.set_budget(category, kobo)
            return {"ok": True, "category": category.strip().lower(),
                    "budget": format_naira(kobo), "month": month_key()}
        items = budgets.list_budgets()
        lines = [f"budgets — {month_key()}"]
        for cat, amt in sorted(items.items()):
            lines.append(f"  • {cat}: {format_naira(amt)}")
        if not items:
            lines.append("  none yet — /budget set <category> <amount>")
        return {"ok": True, "text": "\n".join(lines), "budgets": items}
