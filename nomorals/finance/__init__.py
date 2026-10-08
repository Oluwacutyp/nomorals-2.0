"""Naira-first personal finance: expense tracking + conversational budgeting.

Integer kobo arithmetic everywhere — never floats for money. The core is
currency-agnostic (``currency`` defaults to ``₦``); Nigerian shorthand
("5k" → ₦5,000, "2.5m" → ₦2.5M) is parsed first.

Hook point for the future: bank auto-import (Mono connector) should write
through :meth:`Ledger.log` with ``source="mono"`` so imported transactions
flow through the same budgets and digest.
"""

from .ledger import (
    CATEGORIES,
    Transaction,
    Ledger,
    parse_amount,
    categorize,
    format_naira,
    naira_to_kobo,
)
from .budgets import (
    BudgetStore,
    BudgetStatus,
    budget_status,
    overspend_alerts,
    weekly_digest,
)

__all__ = [
    "CATEGORIES",
    "Transaction",
    "Ledger",
    "parse_amount",
    "categorize",
    "format_naira",
    "naira_to_kobo",
    "BudgetStore",
    "BudgetStatus",
    "budget_status",
    "overspend_alerts",
    "weekly_digest",
]
