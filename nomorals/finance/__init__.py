"""Naira-first personal finance: expense tracking + conversational budgeting.

Integer kobo arithmetic everywhere — never floats for money. The core is
currency-agnostic (``currency`` defaults to ``₦``); Nigerian shorthand
("5k" → ₦5,000, "2.5m" → ₦2.5M) is parsed first.

Hook point for the future: bank auto-import (Mono connector) should write
through :meth:`Ledger.log` with ``source="mono"`` so imported transactions
flow through the same budgets and digest.
"""

from .style import (
    Theme,
    THEMES,
    current_theme,
    bar,
    sparkline,
    table,
    header,
    status_dot,
)
from .ledger import (
    CATEGORIES,
    Transaction,
    Ledger,
    MerchantMemory,
    parse_amount,
    categorize,
    extract_merchant,
    format_naira,
    naira_to_kobo,
)
from .budgets import (
    BudgetStore,
    BudgetStatus,
    budget_status,
    budget_pace,
    overspend_alerts,
    weekly_digest,
    suggested_budgets,
    fifty_thirty_twenty,
    render_budget_grid,
)
from .alerts import (
    Alert,
    AlertStore,
    add_alert,
    evaluate_alerts,
    sync_bill_alerts,
    render_alerts,
    KIND_BILL_DUE,
)
from .goals import (
    Goal,
    GoalStore,
    create_goal,
    goal_progress,
    contribution_streak,
    monthly_need,
    render_goals,
    GOAL_TARGET,
    GOAL_MONTHLY,
    GOAL_BY_DATE,
)
from .guard import (
    Warning,
    check_outgoing,
    known_recipients,
    median_outgoing,
    risk_score,
    render_warning,
)
from .insights import (
    Insights,
    compute_insights,
    detect_recurring,
    render_insights,
    render_recurring,
    upcoming_bills,
    safe_to_spend,
    forecast_month_end,
    spend_series,
)
from .mandate import (
    PaymentMandate,
    MandateStore,
    MandateCheck,
    issue_mandate,
    require_mandate,
    check_mandate,
    daily_transfer_spend,
    weekly_transfer_spend,
    monthly_transfer_spend,
    mandate_remaining,
)
from .overview import (
    MoneyOverview,
    RailBalance,
    collect_balances,
    render_overview,
    BalanceCache,
    snapshot_net_worth,
    net_worth_series,
    render_net_worth_trend,
    asset_mix,
)
from .send import (
    RecipientStore,
    parse_send_request,
    resolve_recipient,
    send_money,
    confirm_send,
    confirm_send_otp,
    masked_account,
)
from .trading_desk import (
    RiskPolicy,
    PaperPosition,
    DeskError,
    TradingDesk,
    size_position,
    expected_value_r,
)

__all__ = [
    # style
    "Theme", "THEMES", "current_theme", "bar", "sparkline", "table",
    "header", "status_dot",
    # ledger
    "CATEGORIES", "Transaction", "Ledger", "MerchantMemory",
    "parse_amount", "categorize", "extract_merchant", "format_naira",
    "naira_to_kobo",
    # budgets
    "BudgetStore", "BudgetStatus", "budget_status", "budget_pace",
    "overspend_alerts", "weekly_digest", "suggested_budgets",
    "fifty_thirty_twenty", "render_budget_grid",
    # alerts
    "Alert", "AlertStore", "add_alert", "evaluate_alerts",
    "sync_bill_alerts", "render_alerts", "KIND_BILL_DUE",
    # goals
    "Goal", "GoalStore", "create_goal", "goal_progress",
    "contribution_streak", "monthly_need", "render_goals",
    "GOAL_TARGET", "GOAL_MONTHLY", "GOAL_BY_DATE",
    # guard
    "Warning", "check_outgoing", "known_recipients", "median_outgoing",
    "risk_score", "render_warning",
    # insights
    "Insights", "compute_insights", "detect_recurring", "render_insights",
    "render_recurring", "upcoming_bills", "safe_to_spend",
    "forecast_month_end", "spend_series",
    # mandate
    "PaymentMandate", "MandateStore", "MandateCheck", "issue_mandate",
    "require_mandate", "check_mandate", "daily_transfer_spend",
    "weekly_transfer_spend", "monthly_transfer_spend", "mandate_remaining",
    # overview
    "MoneyOverview", "RailBalance", "collect_balances", "render_overview",
    "BalanceCache", "snapshot_net_worth", "net_worth_series",
    "render_net_worth_trend", "asset_mix",
    # send
    "RecipientStore", "parse_send_request", "resolve_recipient",
    "send_money", "confirm_send", "confirm_send_otp", "masked_account",
    # trading desk
    "RiskPolicy", "PaperPosition", "DeskError", "TradingDesk",
    "size_position", "expected_value_r",
]
