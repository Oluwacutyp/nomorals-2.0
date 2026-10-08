"""Chat command handlers: /spend, /budget, /spending."""

from __future__ import annotations

import re
from typing import Any

from .budgets import BudgetStore, budget_status, finance_paths, month_key
from .ledger import Ledger, format_naira, parse_amount

# /spend 5k on transport for lunch  → amount, category, note
_RE_SPEND_ON = re.compile(
    r"^\s*(?P<amount>\S+)\s+on\s+(?P<category>[a-zA-Z][\w-]*)\s*(?P<note>.*)$")
_RE_SPEND_BARE = re.compile(r"^\s*(?P<amount>\S+)\s*(?P<note>.*)$")


def _ledger_budgets(context: Any) -> tuple[Ledger, BudgetStore]:
    settings = getattr(context, "settings", None)
    ledger_path, budgets_path = finance_paths(settings)
    return Ledger(ledger_path), BudgetStore(budgets_path)


def parse_spend_tail(tail: str) -> dict[str, Any] | None:
    """/spend <amount> [on <category>] [note...] → dict or None.

    Returns {"amount_kobo", "category" (may be ""), "note"}.
    """
    tail = (tail or "").strip()
    m = _RE_SPEND_ON.match(tail)
    if m:
        kobo = parse_amount(m.group("amount"))
        if kobo is None:
            return None
        return {"amount_kobo": kobo,
                "category": m.group("category").strip().lower(),
                "note": m.group("note").strip()}
    m = _RE_SPEND_BARE.match(tail)
    if m:
        kobo = parse_amount(m.group("amount"))
        if kobo is None:
            return None
        return {"amount_kobo": kobo, "category": "",
                "note": m.group("note").strip()}
    return None


def control_spend(tail: str, context: Any) -> str:
    """/spend <amount> [on <category>] [note...] — log a spend."""
    parsed = parse_spend_tail(tail)
    if parsed is None:
        return ("usage: /spend <amount> [on <category>] [note...]\n"
                "e.g. /spend 5k on transport — or /spend 2000 lunch at mama put")
    ledger, _ = _ledger_budgets(context)
    txn = ledger.log(parsed["amount_kobo"],
                     category=parsed["category"] or None,
                     note=parsed["note"])
    bits = [f"logged {format_naira(txn.amount_kobo)} → {txn.category}"]
    if txn.note:
        bits.append(f"({txn.note})")
    return " ".join(bits)


def control_budget(tail: str, context: Any) -> str:
    """/budget set <category> <amount> | /budget list."""
    _, budgets = _ledger_budgets(context)
    parts = (tail or "").split()
    if not parts or parts[0].lower() == "list":
        items = budgets.list_budgets()
        lines = [f"budgets — {month_key()}"]
        for cat, amt in sorted(items.items()):
            lines.append(f"  • {cat}: {format_naira(amt)}")
        if not items:
            lines.append("  none yet — /budget set food 50k")
        return "\n".join(lines)
    if parts[0].lower() == "set" and len(parts) >= 3:
        category = parts[1].lower()
        kobo = parse_amount(parts[2])
        if kobo is None or kobo <= 0:
            return f"couldn't parse amount {parts[2]!r} — try /budget set food 50k"
        budgets.set_budget(category, kobo)
        return f"budget set: {category} → {format_naira(kobo)}/month"
    return ("usage: /budget set <category> <amount>  ·  /budget list\n"
            "e.g. /budget set food 50k")


def control_spending(tail: str, context: Any) -> str:
    """/spending [week|month] — summary vs budgets."""
    from .budgets import weekly_digest

    ledger, budgets = _ledger_budgets(context)
    period = (tail or "").strip().lower() or "month"
    if period == "week":
        return weekly_digest(ledger, budgets)
    if period != "month":
        return "usage: /spending [week|month]"
    statuses = budget_status(ledger, budgets, month_key())
    lines = [f"spending — {month_key()}"]
    total = 0
    for s in statuses:
        total += s.spent_kobo
        lines.append(
            f"  • {s.category}: {format_naira(s.spent_kobo)} / "
            f"{format_naira(s.budgeted_kobo)} ({s.pct_used:.0%}) — {s.state}")
    if not statuses:
        lines.append("  no budgets set — /budget set food 50k")
    lines.append(f"total tracked: {format_naira(total)}")
    return "\n".join(lines)
