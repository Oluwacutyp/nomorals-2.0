"""Weekly finance digest — ONE composition, two callers.

AA-45 consolidation (2026-11-05): this module's legacy async-cron
registrar (``ensure_finance_digest_job`` below) carried its own
private copy of the digest composition (``_weekly_run``), while the
path that actually runs in production — the ``finance_digest`` tool,
registered by :mod:`nomorals.finance.jobs`' ``ensure_finance_jobs``
at partner boot — composed a richer digest in
:mod:`nomorals.finance.tools`.  Two compositions for one job is the
parallel-system defect: the legacy registrar was never called by any
runtime (grep evidence at consolidation time: zero production or
test references to ``ensure_finance_digest_job`` /
``FINANCE_WEEKLY_*``), and its composition had already drifted
behind the live one (bare safe-to-spend figure instead of the
income/recurring/spent breakdown, bills capped at 8, no savings-goal
pace flags).

The composition now lives here, once, as
:func:`compose_weekly_digest`.  The ``finance_digest`` tool delegates
to it, and the legacy cron registrar — kept for its
``ensure_email_triage_job``-pattern signature, with
:mod:`nomorals.finance.jobs` recorded as the registrar of record for
the partner runtime — runs the same function, so any surface that
ever schedules it produces the identical digest.
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


def compose_weekly_digest(context: Any) -> dict[str, Any]:
    """Compose the weekly money digest and publish it via Notifier.

    The one canonical composition (moved verbatim from the live
    ``finance_digest`` tool): weekly digest, the overspend alert,
    safe-to-spend with its breakdown, upcoming bills, and
    savings-goal pace flags.  Returns ``{"ok", "delivered", "text"}``
    — ``delivered`` is False when the Notifier publish failed, never
    silently True.
    """
    from .goals import GoalStore, goal_progress
    from .insights import safe_to_spend, upcoming_bills

    settings = getattr(context, "settings", None)
    ledger_path, budgets_path = finance_paths(settings)
    ledger, budgets = Ledger(ledger_path), BudgetStore(budgets_path)
    text = weekly_digest(ledger, budgets)
    statuses = budget_status(ledger, budgets, month_key())
    over = [a for a in overspend_alerts(statuses)
            if a["level"] == "over"]
    if over:
        worst = over[0]
        text += (
            f"\n\n🚨 overspend alert: you've spent "
            f"{format_naira(worst['spent_kobo'])} on "
            f"{worst['category']} against a "
            f"{format_naira(worst['budgeted_kobo'])} budget "
            f"({worst['pct_used']:.0%} used)")
    # Safe-to-spend (Simplifi's number) rides along.
    safe = safe_to_spend(ledger)
    text += (f"\n\n💸 safe to spend: {format_naira(safe['safe_kobo'])} "
             f"(expected {format_naira(safe['expected_income_kobo'])}/mo "
             f"− {format_naira(safe['committed_recurring_kobo'])} "
             f"recurring − {format_naira(safe['spent_this_month_kobo'])} "
             f"spent)")
    # Upcoming bills (Rocket Money's core view) ride along.
    bills = upcoming_bills(ledger, days=14)
    if bills:
        text += "\n\n🧾 upcoming bills:"
        for b in bills:
            text += (f"\n  • {b.note_pattern} — "
                     f"{format_naira(b.amount_kobo)}")
    # Savings-goal pace flags ride along with the digest.
    off_pace = []
    for g in GoalStore().list():
        if g.snoozed:
            continue
        p = goal_progress(g, ledger)
        if p.get("on_track") is False:
            off_pace.append(g.name)
    if off_pace:
        text += ("\n\n🎯 off pace on savings goals: "
                 + ", ".join(off_pace))
    try:
        from ..agents.notifier import Notifier

        Notifier(context).publish(
            "finance_digest", "weekly money digest", text)
        delivered = True
    except Exception:  # noqa: BLE001 — publish failure is reported, not hidden
        delivered = False
    return {"ok": True, "delivered": delivered, "text": text}


async def ensure_finance_digest_job(scheduler: Any, context: Any) -> bool:
    """Register the weekly finance digest on an async cron scheduler.

    Idempotent.  NOT the partner runtime's registrar — that is
    :func:`nomorals.finance.jobs.ensure_finance_jobs`, which
    schedules the ``finance_digest`` tool.  This registrar remains
    for async-scheduler surfaces; its run executes the same
    :func:`compose_weekly_digest`, so the digest is identical
    whichever surface schedules it.
    """
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
            result = compose_weekly_digest(context)
            if result.get("delivered"):
                _log.info("finance weekly digest delivered")
            else:
                _log.warning("finance weekly digest composed but the "
                             "Notifier publish failed")
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
