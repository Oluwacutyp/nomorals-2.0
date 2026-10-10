"""Weekly finance digest scheduler job.

Follows the ``ensure_email_triage_job`` pattern: idempotent registration,
context captured in the closure, the overspend check rides along with the
digest and alerts via the Notifier.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from .budgets import (
    BudgetStore,
    budget_status,
    finance_paths,
    month_key,
    overspend_alerts,
    weekly_digest,
)
from .ledger import Ledger, format_naira

_log = get_logger("nomorals.finance")

FINANCE_WEEKLY_TASK_ID = "finance.weekly_digest"
FINANCE_WEEKLY_CRON = "0 9 * * SUN"  # Sunday 09:00, local
FINANCE_WEEKLY_ACTION = "finance.weekly_digest"


async def ensure_finance_digest_job(scheduler: Any, context: Any) -> bool:
    """Register the weekly finance digest. Idempotent. Profile-agnostic."""
    try:
        existing = await scheduler.list_cron_jobs()
    except Exception as exc:  # noqa: BLE001
        _log.warning("finance digest: could not list cron jobs: %s", exc)
        return False
    if any(getattr(j, "task_id", "") == FINANCE_WEEKLY_TASK_ID
           for j in existing):
        return True

    async def _weekly_run(**params: Any) -> None:
        try:
            settings = getattr(context, "settings", None)
            ledger_path, budgets_path = finance_paths(settings)
            ledger, budgets = Ledger(ledger_path), BudgetStore(budgets_path)
            digest = weekly_digest(ledger, budgets)

            statuses = budget_status(ledger, budgets, month_key())
            alerts = overspend_alerts(statuses)
            over = [a for a in alerts if a["level"] == "over"]
            if over:
                worst = over[0]
                digest += (
                    f"\n\n🚨 overspend alert: you've spent "
                    f"{format_naira(worst['spent_kobo'])} on "
                    f"{worst['category']} against a "
                    f"{format_naira(worst['budgeted_kobo'])} budget "
                    f"({worst['pct_used']:.0%} used)")

            # Safe-to-spend + upcoming bills ride along (Simplifi/Rocket
            # Money gold) — the digest should answer "can I buy this?"
            # and "what's about to hit my account?".
            try:
                from .insights import safe_to_spend, upcoming_bills
                safe = safe_to_spend(ledger)
                digest += (
                    f"\n\n💸 safe to spend: "
                    f"{format_naira(safe['safe_kobo'])}")
                bills = upcoming_bills(ledger, days=14)
                if bills:
                    digest += "\n\n🧾 upcoming bills:"
                    for b in bills[:8]:
                        digest += (f"\n  • {b.note_pattern} — "
                                   f"{format_naira(b.amount_kobo)}")
            except Exception as exc:  # noqa: BLE001
                _log.debug("digest extras failed: %s", exc)

            try:
                from ..agents.notifier import Notifier

                Notifier(context).publish(
                    "finance_digest", "weekly money digest", digest)
            except Exception as exc:  # noqa: BLE001
                _log.debug("finance digest notify failed: %s", exc)
            _log.info("finance weekly digest delivered")
        except Exception as exc:  # noqa: BLE001
            _log.warning("finance weekly digest failed: %s", exc)

    scheduler.register_action(FINANCE_WEEKLY_ACTION, _weekly_run)
    await scheduler.schedule_cron(
        FINANCE_WEEKLY_TASK_ID,
        FINANCE_WEEKLY_CRON,
        FINANCE_WEEKLY_ACTION,
        parameters={},
    )
    _log.info("finance weekly digest scheduled (%s)", FINANCE_WEEKLY_CRON)
    return True
