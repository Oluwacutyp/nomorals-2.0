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

    @registry.register(
        "finance_send",
        description=(
            "Send money as a chat primitive ('send 5k to Mama'). Args: to "
            "(recipient name), amount (e.g. '5k'), note (optional). Flow: "
            "resolves recipient (asks when unknown, never guesses) → "
            "mandate check (active payment mandate required, #69) → guard "
            "check (warns on anomalies, advisory only) → stages → returns "
            "a biometric prompt. Money moves ONLY via finance_confirm_send "
            "with a biometric token. override_warning=True executes after "
            "an explicit 'send it anyway'."
        ),
        capability="finance",
        parameters={
            "to": "str — recipient name, e.g. 'Mama'",
            "amount": "str|int — amount, e.g. '5k' or 5000",
            "note": "str — optional note",
            "override_warning": "bool — set true after explicit 'send it anyway'",
        },
    )
    def finance_send(
        context: Any, to: str = "", amount: Any = "", note: str = "",
        override_warning: bool = False,
    ) -> dict[str, Any]:
        from .send import send_money
        return send_money(to, amount, note=note, context=context,
                          override_warning=bool(override_warning))

    @registry.register(
        "finance_confirm_send",
        description=(
            "Execute a staged transfer after biometric approval. Args: "
            "staged_id (from finance_send), biometric_token (capability-bound "
            "token from policy.approve_with_biometric — no token, no movement). "
            "The active payment mandate (#69) is re-checked here: no mandate, "
            "expired/revoked, or over cap → money does not move."
        ),
        capability="finance",
        parameters={
            "staged_id": "str — staged transfer id",
            "biometric_token": "str — token from approve_with_biometric()",
        },
    )
    def finance_confirm_send(
        context: Any, staged_id: str = "", biometric_token: str = "",
    ) -> dict[str, Any]:
        from .send import RecipientStore, confirm_send
        paystack = getattr(context, "paystack", None)
        return confirm_send(staged_id, biometric_token or None,
                            context=context, paystack=paystack,
                            recipients=RecipientStore())

    @registry.register(
        "finance_balances",
        description=(
            "Unified money view: balances across every connected rail "
            "(Mono banks, Binance, Coinbase, Exness, Wise), converted to "
            "NGN. Rails that aren't connected are reported as such — "
            "never as zero. Args: rails (optional comma-separated subset, "
            "e.g. 'mono,binance')."
        ),
        capability="finance",
        parameters={
            "rails": "str — optional comma-separated rail subset",
        },
    )
    def finance_balances(context: Any, rails: str = "") -> dict[str, Any]:
        import os
        from .overview import collect_balances, render_overview
        try:
            from ..accounts.vault import CredentialVault
            passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
            if not passphrase:
                return {"ok": False,
                        "error": "vault locked: set NM_VAULT_PASSPHRASE"}
            vault = CredentialVault(getattr(context, "db", None),
                                    master_passphrase=passphrase)
        except Exception as exc:
            return {"ok": False, "error": f"vault unavailable: {exc}"}
        wanted = tuple(r.strip() for r in rails.split(",") if r.strip()) or None
        ov = collect_balances(vault, rails=wanted)
        return {
            "ok": True,
            "text": render_overview(ov),
            "total_ngn": round(ov.total_ngn, 2),
            "rails": [
                {"rail": r.rail, "label": r.label, "amount": r.amount,
                 "currency": r.currency, "amount_ngn": r.amount_ngn,
                 "status": r.status, "detail": r.detail}
                for r in ov.rails
            ],
        }

    @registry.register(
        "finance_alert",
        description=(
            "Price & money alerts (Devon's own watchtower, checked every "
            "15 min). Args: action (add|list|remove|check), kind "
            "(price_above|price_below|pct_change|budget_pct|fx_rate), "
            "target (symbol, category, or FX base like USD), threshold, "
            "market (crypto|forex|stock), alert_id (for remove)."
        ),
        capability="finance",
        parameters={
            "action": "str — add|list|remove|check",
            "kind": "str — price_above|price_below|pct_change|budget_pct|fx_rate",
            "target": "str — symbol, category, or FX base",
            "threshold": "float — price / % / budget fraction / rate",
            "market": "str — crypto|forex|stock (default crypto)",
            "alert_id": "str — for remove",
            "deliver": "bool — publish fired alerts via Notifier "
                       "(default false; the scheduler uses true)",
        },
    )
    def finance_alert(
        context: Any, action: str = "list", kind: str = "",
        target: str = "", threshold: float = 0.0, market: str = "crypto",
        alert_id: str = "", deliver: bool = False,
    ) -> dict[str, Any]:
        from .alerts import AlertStore, add_alert, evaluate_alerts
        action = (action or "list").lower()
        store = AlertStore()
        if action == "add":
            try:
                a = add_alert(kind, target, threshold, market=market)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "alert_id": a.id,
                    "description": a.describe()}
        if action == "remove":
            ok = store.remove(alert_id)
            return {"ok": ok, "alert_id": alert_id}
        if action == "check":
            fired_msgs: list[str] = []

            def _notify(title: str, body: str) -> None:
                fired_msgs.append(f"{title}\n{body}")
                if deliver:
                    try:
                        from ..agents.notifier import Notifier
                        Notifier(context).publish(
                            "price_alert", title, body)
                    except Exception:
                        pass

            result = evaluate_alerts(notify=_notify)
            return {"ok": True, "delivered": bool(deliver),
                    "messages": fired_msgs, **result}
        alerts = store.list()
        return {"ok": True, "count": len(alerts),
                "alerts": [
                    {"id": a.id, "kind": a.kind, "target": a.target,
                     "threshold": a.threshold, "enabled": a.enabled,
                     "description": a.describe(),
                     "fire_count": a.fire_count}
                    for a in alerts
                ]}

    @registry.register(
        "finance_insights",
        description=(
            "Native money analytics over the ledger: burn rate, savings "
            "rate, runway, top categories/merchants, recurring charges. "
            "Args: days (window, default 30)."
        ),
        capability="finance",
        parameters={"days": "int — analysis window in days (default 30)"},
    )
    def finance_insights(context: Any, days: int = 30) -> dict[str, Any]:
        from .insights import compute_insights, render_insights
        ledger, _ = _ledger_budgets(context)
        days = max(7, min(365, int(days or 30)))
        ins = compute_insights(ledger, window_days=days)
        return {
            "ok": True, "text": render_insights(ins),
            "burn_rate_kobo_per_day": ins.burn_rate_kobo_per_day,
            "savings_rate": ins.savings_rate,
            "runway_days": ins.runway_days,
            "trend_pct": ins.trend_pct,
            "top_categories": ins.top_categories,
            "top_merchants": ins.top_merchants,
            "recurring": [
                {"note": r.note_pattern, "category": r.category,
                 "amount_kobo": r.amount_kobo,
                 "occurrences": r.occurrences,
                 "avg_interval_days": r.avg_interval_days}
                for r in ins.recurring
            ],
        }

    @registry.register(
        "finance_goal",
        description=(
            "Savings goals. Args: action (add|list|done|remove), name, "
            "amount (e.g. '500k'), deadline ('2026-12-31', 'in 90d', or "
            "'dec'), goal_id (for done|remove)."
        ),
        capability="finance",
        parameters={
            "action": "str — add|list|done|remove",
            "name": "str — goal name (for add)",
            "amount": "str|int — target amount (for add)",
            "deadline": "str — deadline (for add)",
            "goal_id": "str — for done|remove",
        },
    )
    def finance_goal(
        context: Any, action: str = "list", name: str = "",
        amount: Any = "", deadline: str = "", goal_id: str = "",
    ) -> dict[str, Any]:
        from .goals import GoalStore, create_goal, goal_progress
        ledger, _ = _ledger_budgets(context)
        store = GoalStore()
        action = (action or "list").lower()
        if action == "add":
            kobo = (parse_amount(str(amount))
                    if not isinstance(amount, int) else int(amount) * 100)
            if kobo is None or kobo <= 0:
                return {"ok": False,
                        "error": f"couldn't parse amount {amount!r}"}
            try:
                g = create_goal(name, kobo, deadline=deadline)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "goal_id": g.id, "name": g.name,
                    "target": format_naira(g.target_kobo)}
        if action == "done":
            return {"ok": store.mark_done(goal_id), "goal_id": goal_id}
        if action == "remove":
            return {"ok": store.remove(goal_id), "goal_id": goal_id}
        goals = store.list()
        return {"ok": True, "count": len(goals),
                "goals": [goal_progress(g, ledger) for g in goals]}

    @registry.register(
        "finance_digest",
        description=(
            "Weekly money digest: spending vs last week, top categories, "
            "budget flags, savings-goal pace. Publishes through the "
            "Notifier (the scheduler runs this weekly). No args."
        ),
        capability="finance",
        parameters={},
    )
    def finance_digest(context: Any) -> dict[str, Any]:
        from .budgets import budget_status, month_key, overspend_alerts
        from .goals import GoalStore, goal_progress
        ledger, budgets = _ledger_budgets(context)
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
        # Savings-goal pace flags ride along with the digest.
        off_pace = []
        for g in GoalStore().list():
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
        except Exception:
            delivered = False
        return {"ok": True, "delivered": delivered, "text": text}

    @registry.register(
        "finance_mandate",
        description=(
            "Payment mandates: the agent's standing authority to move "
            "money (who, scope, per-txn and per-day caps, expiry). Money "
            "cannot move without an active mandate. Args: action "
            "(issue|list|revoke|revoke-all), scope "
            "(transfer|travel|all), per_txn (e.g. '50k'), per_day "
            "(e.g. '200k'), ttl_days, mandate_id (for revoke)."
        ),
        capability="finance",
        parameters={
            "action": "str — issue|list|revoke|revoke-all",
            "scope": "str — transfer|travel|all",
            "per_txn": "str|int — per-transaction cap",
            "per_day": "str|int — daily cap",
            "ttl_days": "float — mandate lifetime in days",
            "mandate_id": "str — for revoke",
        },
    )
    def finance_mandate(
        context: Any, action: str = "list", scope: str = "transfer",
        per_txn: Any = "", per_day: Any = "", ttl_days: float = 30.0,
        mandate_id: str = "",
    ) -> dict[str, Any]:
        from .mandate import MandateStore, issue_mandate
        store = MandateStore()
        action = (action or "list").lower()
        if action == "issue":
            txn_kobo = (parse_amount(str(per_txn))
                        if not isinstance(per_txn, int)
                        else int(per_txn) * 100)
            day_kobo = (parse_amount(str(per_day))
                        if not isinstance(per_day, int)
                        else int(per_day) * 100)
            if txn_kobo is None or day_kobo is None:
                return {"ok": False, "error": "couldn't parse caps"}
            try:
                m = issue_mandate(store, scope=scope, cap_per_txn=txn_kobo,
                                  cap_per_day=day_kobo,
                                  ttl_days=float(ttl_days or 30.0))
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "mandate_id": m.id, "scope": m.scope,
                    "cap_per_txn": format_naira(m.cap_per_txn),
                    "cap_per_day": format_naira(m.cap_per_day)}
        if action == "revoke":
            try:
                ok = store.revoke(mandate_id)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": ok, "mandate_id": mandate_id}
        if action == "revoke-all":
            n = store.revoke_all("owner")
            return {"ok": True, "revoked": n}
        mandates = store.list("owner")
        return {"ok": True, "count": len(mandates),
                "mandates": [m.to_dict() for m in mandates]}
