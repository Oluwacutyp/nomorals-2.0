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
        from .budgets import render_budget_grid
        statuses = budget_status(ledger, budgets, month_key())
        cat = (category or "").strip().lower()
        if cat:
            statuses = [s for s in statuses if s.category == cat]
        return {"ok": True, "period": "month",
                "text": render_budget_grid(statuses, month_key()),
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
        "finance_search",
        description=(
            "Full-text search over the ledger (notes, merchants, categories). "
            "Args: query (text), limit (default 20)."
        ),
        capability="finance",
        parameters={
            "query": "str — search text, e.g. 'netflix' or 'mama put'",
            "limit": "int — max results (default 20)",
        },
    )
    def finance_search(context: Any, query: str = "",
                       limit: int = 20) -> dict[str, Any]:
        ledger, _ = _ledger_budgets(context)
        hits = ledger.search(query, limit=max(1, min(100, int(limit or 20))))
        return {"ok": True, "count": len(hits), "query": query,
                "transactions": [
                    {"id": t.id, "amount": format_naira(t.amount_kobo),
                     "category": t.category, "note": t.note,
                     "merchant": t.merchant, "kind": t.kind,
                     "ts": t.ts}
                    for t in hits]}

    @registry.register(
        "finance_import",
        description=(
            "Import a bank-statement CSV into the ledger. Columns are "
            "sniffed (date/amount/narration); override with date_col, "
            "amount_col, note_col. Duplicates (same amount + merchant "
            "within 3 days) are skipped. Args: path, date_col, amount_col, "
            "note_col, kind (spend|income default)."
        ),
        capability="finance",
        parameters={
            "path": "str — path to the CSV file",
            "date_col": "str — optional explicit date column",
            "amount_col": "str — optional explicit amount column",
            "note_col": "str — optional explicit note column",
            "kind": "str — spend|income (default spend)",
        },
    )
    def finance_import(context: Any, path: str = "", date_col: str = "",
                       amount_col: str = "", note_col: str = "",
                       kind: str = "spend") -> dict[str, Any]:
        ledger, _ = _ledger_budgets(context)
        if not path:
            return {"ok": False, "error": "path is required"}
        try:
            report = ledger.import_csv(
                path, date_col=date_col, amount_col=amount_col,
                note_col=note_col, kind=kind or "spend")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **report}

    @registry.register(
        "finance_recurring",
        description=(
            "Recurring-charge intelligence: detected subscriptions, "
            "price hikes, upcoming bills, and committed monthly/yearly "
            "cost. Args: action (list|upcoming|sync_alerts), days "
            "(window for upcoming, default 14)."
        ),
        capability="finance",
        parameters={
            "action": "str — list|upcoming|sync_alerts (default list)",
            "days": "int — upcoming window in days (default 14)",
        },
    )
    def finance_recurring(context: Any, action: str = "list",
                          days: int = 14) -> dict[str, Any]:
        from .insights import (detect_recurring, render_recurring,
                               upcoming_bills)
        from .ledger import Ledger as _L
        import time as _t
        ledger, _ = _ledger_budgets(context)
        now = _t.time()
        action = (action or "list").lower()
        if action == "upcoming":
            bills = upcoming_bills(ledger, days=int(days or 14), now=now)
            return {"ok": True, "count": len(bills),
                    "text": render_recurring(bills),
                    "bills": [
                        {"note": b.note_pattern, "category": b.category,
                         "amount": format_naira(b.amount_kobo),
                         "amount_kobo": b.amount_kobo,
                         "next_due_ts": b.next_due_ts,
                         "price_changed": b.price_changed}
                        for b in bills]}
        if action == "sync_alerts":
            from .alerts import AlertStore, sync_bill_alerts
            result = sync_bill_alerts(AlertStore(), ledger, now=now)
            return {"ok": True, **result,
                    "text": f"watching {result['total']} recurring "
                            f"charge(s): {result['created']} new bill "
                            f"alert(s), {result['refreshed']} refreshed"}
        txns = ledger.transactions(since=now - 120 * 86400)
        recurring = detect_recurring(txns)
        return {"ok": True, "count": len(recurring),
                "text": render_recurring(recurring),
                "total_yearly_kobo": sum(r.yearly_cost_kobo
                                         for r in recurring),
                "charges": [
                    {"note": r.note_pattern, "category": r.category,
                     "amount_kobo": r.amount_kobo,
                     "occurrences": r.occurrences,
                     "avg_interval_days": r.avg_interval_days,
                     "price_changed": r.price_changed,
                     "price_change_kobo": r.price_change_kobo,
                     "yearly_cost_kobo": r.yearly_cost_kobo,
                     "next_due_ts": r.next_due_ts}
                    for r in recurring]}

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
        from .overview import (BalanceCache, collect_balances, render_overview,
                               snapshot_net_worth)
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
        cache = getattr(context, "_finance_balance_cache", None)
        if cache is None:
            cache = BalanceCache()
            try:
                context._finance_balance_cache = cache
            except Exception:
                pass
        ov = cache.get(vault, rails=wanted)
        snapshot_net_worth(ov)
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
            "15 min). Args: action (add|list|remove|check|snooze|"
            "sync_bills), kind (price_above|price_below|pct_change|"
            "budget_pct|fx_rate|bill_due), target (symbol, category, FX "
            "base like USD, or bill name), threshold, market "
            "(crypto|forex|stock), alert_id (for remove|snooze), "
            "snooze_days."
        ),
        capability="finance",
        parameters={
            "action": "str — add|list|remove|check|snooze|sync_bills",
            "kind": "str — price_above|price_below|pct_change|budget_pct|fx_rate|bill_due",
            "target": "str — symbol, category, FX base, or bill name",
            "threshold": "float — price / % / budget fraction / days-ahead",
            "market": "str — crypto|forex|stock (default crypto)",
            "alert_id": "str — for remove|snooze",
            "snooze_days": "float — snooze length (default 7)",
            "deliver": "bool — publish fired alerts via Notifier "
                       "(default false; the scheduler uses true)",
        },
    )
    def finance_alert(
        context: Any, action: str = "list", kind: str = "",
        target: str = "", threshold: float = 0.0, market: str = "crypto",
        alert_id: str = "", snooze_days: float = 7.0, deliver: bool = False,
    ) -> dict[str, Any]:
        from .alerts import (AlertStore, add_alert, evaluate_alerts,
                             render_alerts, sync_bill_alerts)
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
        if action == "snooze":
            ok = store.snooze(alert_id, days=float(snooze_days or 7.0))
            return {"ok": ok, "alert_id": alert_id,
                    "snoozed_days": float(snooze_days or 7.0)}
        if action == "sync_bills":
            ledger, _ = _ledger_budgets(context)
            result = sync_bill_alerts(store, ledger)
            return {"ok": True, **result}
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
                "text": render_alerts(alerts),
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
        from .insights import (compute_insights, render_insights,
                               safe_to_spend)
        ledger, _ = _ledger_budgets(context)
        days = max(7, min(365, int(days or 30)))
        ins = compute_insights(ledger, window_days=days)
        safe = safe_to_spend(ledger)
        return {
            "ok": True, "text": render_insights(ins),
            "burn_rate_kobo_per_day": ins.burn_rate_kobo_per_day,
            "savings_rate": ins.savings_rate,
            "runway_days": ins.runway_days,
            "trend_pct": ins.trend_pct,
            "safe_to_spend_kobo": safe["safe_kobo"],
            "safe_to_spend": format_naira(safe["safe_kobo"]),
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
            "Savings goals (YNAB-style: target|monthly|by_date). Args: "
            "action (add|list|done|remove|snooze), name, amount (e.g. "
            "'500k'), deadline ('2026-12-31', 'in 90d', or 'dec'), "
            "goal_type (target|monthly|by_date), goal_id (for done|"
            "remove|snooze), snooze_days."
        ),
        capability="finance",
        parameters={
            "action": "str — add|list|done|remove|snooze",
            "name": "str — goal name (for add)",
            "amount": "str|int — target amount (for add)",
            "deadline": "str — deadline (for add)",
            "goal_type": "str — target|monthly|by_date (for add)",
            "goal_id": "str — for done|remove|snooze",
            "snooze_days": "float — snooze length (default 30)",
        },
    )
    def finance_goal(
        context: Any, action: str = "list", name: str = "",
        amount: Any = "", deadline: str = "", goal_type: str = "target",
        goal_id: str = "", snooze_days: float = 30.0,
    ) -> dict[str, Any]:
        from .goals import (GoalStore, create_goal, goal_progress,
                            render_goals)
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
                g = create_goal(name, kobo, deadline=deadline,
                                goal_type=goal_type)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "goal_id": g.id, "name": g.name,
                    "target": format_naira(g.target_kobo),
                    "goal_type": g.goal_type}
        if action == "done":
            return {"ok": store.mark_done(goal_id), "goal_id": goal_id}
        if action == "remove":
            return {"ok": store.remove(goal_id), "goal_id": goal_id}
        if action == "snooze":
            ok = store.snooze(goal_id, days=float(snooze_days or 30.0))
            return {"ok": ok, "goal_id": goal_id,
                    "snoozed_days": float(snooze_days or 30.0)}
        goals = store.list()
        return {"ok": True, "count": len(goals),
                "text": render_goals(goals, ledger),
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
        from .insights import safe_to_spend, upcoming_bills
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
        except Exception:
            delivered = False
        return {"ok": True, "delivered": delivered, "text": text}

    @registry.register(
        "finance_mandate",
        description=(
            "Payment mandates: the agent's standing authority to move "
            "money (who, scope, per-txn / per-day / per-week / per-month "
            "caps, expiry). Money cannot move without an active mandate. "
            "Args: action (issue|list|revoke|revoke-all|describe), scope "
            "(transfer|travel|all), per_txn (e.g. '50k'), per_day "
            "(e.g. '200k'), per_week, per_month, ttl_days, mandate_id "
            "(for revoke|describe)."
        ),
        capability="finance",
        parameters={
            "action": "str — issue|list|revoke|revoke-all|describe",
            "scope": "str — transfer|travel|all",
            "per_txn": "str|int — per-transaction cap",
            "per_day": "str|int — daily cap",
            "per_week": "str|int — optional weekly cap",
            "per_month": "str|int — optional monthly cap",
            "ttl_days": "float — mandate lifetime in days",
            "mandate_id": "str — for revoke|describe",
        },
    )
    def finance_mandate(
        context: Any, action: str = "list", scope: str = "transfer",
        per_txn: Any = "", per_day: Any = "", per_week: Any = "",
        per_month: Any = "", ttl_days: float = 30.0,
        mandate_id: str = "",
    ) -> dict[str, Any]:
        from .mandate import (MandateStore, issue_mandate, mandate_remaining)

        def _kobo(v: Any) -> int | None:
            if v == "" or v is None:
                return None
            k = parse_amount(str(v)) if not isinstance(v, int) \
                else int(v) * 100
            return k if k and k > 0 else None

        store = MandateStore()
        action = (action or "list").lower()
        if action == "issue":
            txn_kobo = _kobo(per_txn)
            day_kobo = _kobo(per_day)
            if txn_kobo is None or day_kobo is None:
                return {"ok": False, "error": "couldn't parse caps"}
            try:
                m = issue_mandate(store, scope=scope, cap_per_txn=txn_kobo,
                                  cap_per_day=day_kobo,
                                  ttl_days=float(ttl_days or 30.0),
                                  cap_per_week=_kobo(per_week),
                                  cap_per_month=_kobo(per_month))
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "mandate_id": m.id, "scope": m.scope,
                    "cap_per_txn": format_naira(m.cap_per_txn),
                    "cap_per_day": format_naira(m.cap_per_day),
                    "describe": m.describe()}
        if action == "revoke":
            try:
                ok = store.revoke(mandate_id)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": ok, "mandate_id": mandate_id}
        if action == "revoke-all":
            n = store.revoke_all("owner")
            return {"ok": True, "revoked": n}
        if action == "describe":
            m = store.get(mandate_id)
            if m is None:
                return {"ok": False,
                        "error": f"no mandate {mandate_id!r}"}
            ledger, _ = _ledger_budgets(context)
            return {"ok": True, "describe": m.describe(),
                    "remaining_kobo": mandate_remaining(m, ledger),
                    "remaining": {
                        k: format_naira(v)
                        for k, v in mandate_remaining(m, ledger).items()}}
        mandates = store.list("owner")
        return {"ok": True, "count": len(mandates),
                "mandates": [m.to_dict() for m in mandates]}

    @registry.register(
        "finance_trade",
        description=(
            "Trading desk: risk-managed paper/live trading over Exness. "
            "Args: action (size|open|close|positions|stats|heat|kelly), "
            "instrument (e.g. XAUUSD), side (buy|sell), volume, entry, "
            "stop_loss, take_profit, position_id. Paper mode is default — "
            "live needs unlock_live + confirmation. Every market order "
            "requires a stop-loss."
        ),
        capability="finance",
        parameters={
            "action": "str — size|open|close|positions|stats|heat|kelly",
            "instrument": "str — e.g. XAUUSD",
            "side": "str — buy|sell",
            "volume": "float — lots",
            "entry": "float — optional entry price (default: latest close)",
            "stop_loss": "float — required for open",
            "take_profit": "float — optional",
            "position_id": "str — for close",
            "risk_pct": "float — for size (default: policy max)",
        },
    )
    def finance_trade(
        context: Any, action: str = "stats", instrument: str = "",
        side: str = "", volume: float = 0.0, entry: float = 0.0,
        stop_loss: float = 0.0, take_profit: float = 0.0,
        position_id: str = "", risk_pct: float = 0.0,
    ) -> dict[str, Any]:
        from .trading_desk import DeskError, TradingDesk, size_position
        connector = getattr(context, "exness", None)
        if connector is None:
            try:
                from ..connectors.exness import ExnessConnector
                from ..accounts.vault import CredentialVault
                import os as _os
                passphrase = _os.environ.get("NM_VAULT_PASSPHRASE", "")
                if passphrase:
                    vault = CredentialVault(getattr(context, "db", None),
                                            master_passphrase=passphrase)
                    connector = ExnessConnector(vault=vault)
            except Exception as exc:
                return {"ok": False,
                        "error": f"no Exness connector available: {exc}"}
            if connector is None:
                return {"ok": False,
                        "error": "no Exness connector available"}
        desk = TradingDesk(connector, mode="paper")
        action = (action or "stats").lower()
        try:
            if action == "size":
                eq = desk.equity()
                spec = desk.contract_specs(instrument)
                cv = desk._contract_value(instrument)
                sized = size_position(
                    equity=eq, entry=float(entry),
                    stop_loss=float(stop_loss),
                    risk_pct=float(risk_pct or desk.policy.max_risk_pct_per_trade),
                    contract_value=cv,
                    volume_min=float(spec.get("volume_min", 0.01)),
                    volume_max=float(spec.get("volume_max", 100.0)),
                    volume_step=float(spec.get("volume_step", 0.01)))
                return {"ok": True, "equity": eq, **sized}
            if action == "open":
                pos = desk.paper_open(
                    instrument, side, float(volume),
                    entry=float(entry) or None,
                    stop_loss=float(stop_loss) or None,
                    take_profit=float(take_profit) or None)
                return {"ok": True, "position_id": pos.id,
                        "instrument": pos.instrument, "side": pos.side,
                        "volume": pos.volume, "entry": pos.entry,
                        "stop_loss": pos.stop_loss,
                        "take_profit": pos.take_profit,
                        "risk_amount": pos.risk_amount}
            if action == "close":
                return {"ok": True, **desk.paper_close(
                    position_id, exit_price=float(entry) or None)}
            if action == "positions":
                return {"ok": True, "positions": [
                    {"id": p.id, "instrument": p.instrument,
                     "side": p.side, "volume": p.volume, "entry": p.entry,
                     "stop_loss": p.stop_loss, "take_profit": p.take_profit,
                     "risk_amount": p.risk_amount}
                    for p in desk.paper_positions()]}
            if action == "heat":
                return {"ok": True, **desk.portfolio_heat()}
            if action == "kelly":
                return desk.kelly_fraction()
            # stats (default)
            s = desk.stats()
            return {"ok": True, "text": desk.render_stats(), **s}
        except DeskError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"trade failed: {exc}"}
