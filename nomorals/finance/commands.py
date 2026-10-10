"""Chat command handlers: /spend, /budget, /spending, /mandate, /balances,
/alert, /goal, /insights, /recurring, /trade, /import."""

from __future__ import annotations

import re
import time
from typing import Any

from .budgets import (BudgetStore, budget_status, finance_paths, month_key,
                      render_budget_grid)
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
    return render_budget_grid(statuses, month_key())


# ── /mandate ───────────────────────────────────────────────────────────

def control_mandate(tail: str, context: Any) -> str:
    """/mandate issue <scope> <per-txn> <per-day> [ttl-days] [per-week] [per-month]
    /mandate list | /mandate describe <id> | /mandate revoke <id> | /mandate revoke-all"""
    from .mandate import (MandateStore, SCOPE_ALL, SCOPE_TRANSFER,
                          SCOPE_TRAVEL, mandate_remaining)

    _, budgets_path = finance_paths(getattr(context, "settings", None))
    store = MandateStore()
    parts = (tail or "").split()
    if not parts or parts[0].lower() == "list":
        mandates = store.list("owner")
        if not mandates:
            return ("no payment mandates. Money can't move until you issue "
                    "one:\n/mandate issue transfer 50k 200k  — ₦50k/txn, "
                    "₦200k/day, 30 days")
        lines = ["payment mandates:"]
        now = time.time()
        for m in mandates:
            state = ("revoked" if m.revoked else
                     "expired" if m.expires_at <= now else "ACTIVE")
            caps = (f"{format_naira(m.cap_per_txn)}/txn "
                    f"{format_naira(m.cap_per_day)}/day")
            if m.cap_per_week:
                caps += f" {format_naira(m.cap_per_week)}/wk"
            if m.cap_per_month:
                caps += f" {format_naira(m.cap_per_month)}/mo"
            lines.append(f"  • {m.id} [{state}] scope={m.scope} {caps}")
        return "\n".join(lines)
    verb = parts[0].lower()
    if verb == "describe" and len(parts) >= 2:
        m = store.get(parts[1])
        if m is None:
            return f"no mandate {parts[1]!r} found."
        settings = getattr(context, "settings", None)
        ledger_path, _ = finance_paths(settings)
        rem = mandate_remaining(m, Ledger(ledger_path))
        rem_txt = ", ".join(f"{k}: {format_naira(v)}"
                            for k, v in rem.items())
        return m.describe() + f"\nremaining → {rem_txt}"
    if verb == "issue" and len(parts) >= 4:
        scope = parts[1].lower()
        if scope not in (SCOPE_TRANSFER, SCOPE_TRAVEL, SCOPE_ALL):
            return (f"unknown scope {scope!r} — use transfer, travel, or all")
        txn_kobo = parse_amount(parts[2])
        day_kobo = parse_amount(parts[3])
        if txn_kobo is None or day_kobo is None:
            return ("couldn't parse caps — "
                    "e.g. /mandate issue transfer 50k 200k")
        ttl = 30.0
        week_kobo = month_kobo = None
        if len(parts) >= 5:
            try:
                ttl = float(parts[4])
            except ValueError:
                return f"couldn't parse ttl {parts[4]!r} (days)"
        if len(parts) >= 6:
            week_kobo = parse_amount(parts[5])
            if week_kobo is None:
                return f"couldn't parse weekly cap {parts[5]!r}"
        if len(parts) >= 7:
            month_kobo = parse_amount(parts[6])
            if month_kobo is None:
                return f"couldn't parse monthly cap {parts[6]!r}"
        try:
            m = store.issue(scope=scope, cap_per_txn=txn_kobo,
                            cap_per_day=day_kobo, ttl_days=ttl,
                            cap_per_week=week_kobo,
                            cap_per_month=month_kobo)
        except Exception as exc:
            return f"couldn't issue mandate: {exc}"
        return (f"mandate issued: {m.id}\n" + m.describe() +
                f"\nRevoke anytime: /mandate revoke {m.id} "
                f"or /mandate revoke-all")
    if verb == "revoke" and len(parts) >= 2:
        try:
            ok = store.revoke(parts[1])
        except Exception as exc:
            return f"couldn't revoke: {exc}"
        return (f"mandate {parts[1]} revoked — money stops moving on it "
                "immediately." if ok else f"no mandate {parts[1]!r} found.")
    if verb == "revoke-all":
        n = store.revoke_all("owner")
        return f"revoked {n} mandate(s) — all agent spending is now blocked."
    return ("usage:\n"
            "  /mandate issue <transfer|travel|all> <per-txn> <per-day> "
            "[days] [per-week] [per-month]\n"
            "  /mandate list | /mandate describe <id>\n"
            "  /mandate revoke <id> | /mandate revoke-all\n"
            "e.g. /mandate issue transfer 50k 200k 30 1m 3m")


# ── /balances ──────────────────────────────────────────────────────────

def control_balances(tail: str, context: Any) -> str:
    """/balances — unified money view across every connected rail."""
    from .overview import (BalanceCache, collect_balances, render_overview,
                           render_net_worth_trend, snapshot_net_worth)

    vault = _vault_from_context(context)
    if vault is None:
        return ("the credential vault is locked — set NM_VAULT_PASSPHRASE "
                "so I can read rail balances.")
    rails = None
    tail = (tail or "").strip().lower()
    if tail:
        rails = tuple(r.strip() for r in tail.split(",") if r.strip())
    try:
        ov = collect_balances(vault, rails=rails)
        snapshot_net_worth(ov)
    except Exception as exc:
        return f"couldn't collect balances: {exc}"
    return render_overview(ov) + "\n\n" + render_net_worth_trend()


def _vault_from_context(context: Any) -> Any | None:
    import os
    try:
        from ..accounts.vault import CredentialVault
        passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
        if not passphrase:
            return None
        return CredentialVault(getattr(context, "db", None),
                               master_passphrase=passphrase)
    except Exception:
        return None


# ── /alert ──────────────────────────────────────────────────────────────

_ALERT_USAGE = ("usage:\n"
                "  /alert add price_above BTC 90000\n"
                "  /alert add price_below BTC/USDT 60000 --market crypto\n"
                "  /alert add pct_change ETH 10\n"
                "  /alert add budget_pct food 0.8\n"
                "  /alert add fx_rate USD 1600\n"
                "  /alert add bill_due <bill name> 3   (3 days ahead)\n"
                "  /alert list\n"
                "  /alert remove <id> | /alert snooze <id> [days]\n"
                "  /alert check   (evaluate now)\n"
                "  /alert sync-bills   (auto-watch detected subscriptions)")


def control_alert(tail: str, context: Any) -> str:
    """/alert add|list|remove|check|snooze|sync-bills — price & money alerts."""
    from .alerts import (AlertStore, add_alert, evaluate_alerts,
                         render_alerts, sync_bill_alerts)

    parts = (tail or "").split()
    if not parts or parts[0].lower() == "list":
        alerts = AlertStore().list()
        if not alerts:
            return "no alerts set.\n" + _ALERT_USAGE
        return render_alerts(alerts)
    verb = parts[0].lower()
    if verb == "remove" and len(parts) >= 2:
        ok = AlertStore().remove(parts[1])
        return (f"alert {parts[1]} removed." if ok
                else f"no alert {parts[1]!r} found.")
    if verb == "snooze" and len(parts) >= 2:
        days = 7.0
        if len(parts) >= 3:
            try:
                days = float(parts[2])
            except ValueError:
                return f"couldn't parse days {parts[2]!r}"
        ok = AlertStore().snooze(parts[1], days=days)
        return (f"alert {parts[1]} snoozed for {days:g} day(s)." if ok
                else f"no alert {parts[1]!r} found.")
    if verb in ("sync-bills", "sync_bills"):
        settings = getattr(context, "settings", None)
        ledger_path, _ = finance_paths(settings)
        result = sync_bill_alerts(AlertStore(), Ledger(ledger_path))
        return (f"watching {result['total']} recurring charge(s): "
                f"{result['created']} new bill alert(s), "
                f"{result['refreshed']} refreshed.")
    if verb == "check":
        fired_msgs: list[str] = []

        def _notify(title: str, body: str) -> None:
            fired_msgs.append(f"{title}\n{body}")

        result = evaluate_alerts(notify=_notify)
        if result["fired"]:
            return ("checked "
                    f"{result['checked']} alert(s) — {result['fired']} fired:\n"
                    + "\n\n".join(fired_msgs))
        return (f"checked {result['checked']} alert(s) — none firing.")
    if verb == "add" and len(parts) >= 4:
        kind, target = parts[1].lower(), parts[2]
        try:
            threshold = float(parts[3])
        except ValueError:
            return f"threshold must be a number, got {parts[3]!r}"
        market = "crypto"
        for i, p in enumerate(parts):
            if p == "--market" and i + 1 < len(parts):
                market = parts[i + 1]
        try:
            a = add_alert(kind, target, threshold, market=market)
        except ValueError as exc:
            return f"couldn't add alert: {exc}\n{_ALERT_USAGE}"
        return (f"alert set: {a.id}\n{a.describe()}\n"
                "I'll check every 15 minutes and ping you when it fires.")
    return _ALERT_USAGE


# ── /goal ───────────────────────────────────────────────────────────────

def control_goal(tail: str, context: Any) -> str:
    """/goal add <name> <amount> [deadline] [type] | /goal list
    /goal done <id> | /goal snooze <id> [days] | /goal remove <id>
    types: target (have ₦X) | monthly (save ₦X/mo) | by_date (₦X by deadline)"""
    from .goals import GoalStore, create_goal, goal_progress, render_goals

    settings = getattr(context, "settings", None)
    ledger_path, _ = finance_paths(settings)
    ledger = Ledger(ledger_path)
    store = GoalStore()
    parts = (tail or "").split()
    if not parts or parts[0].lower() == "list":
        return render_goals(store.list(), ledger)
    verb = parts[0].lower()
    if verb == "add" and len(parts) >= 3:
        name = parts[1]
        kobo = parse_amount(parts[2])
        if kobo is None or kobo <= 0:
            return f"couldn't parse amount {parts[2]!r} — try /goal add name 500k"
        deadline = parts[3] if len(parts) >= 4 else ""
        goal_type = parts[4] if len(parts) >= 5 else "target"
        try:
            g = create_goal(name, kobo, deadline=deadline,
                            goal_type=goal_type)
        except ValueError as exc:
            return f"couldn't create goal: {exc}"
        return (f"goal set [{g.goal_type}]: {g.name} → "
                f"{format_naira(g.target_kobo)}"
                + (f" by {deadline}" if deadline else "")
                + "\nLog savings with /spend <amount> on savings <note>.")
    if verb == "done" and len(parts) >= 2:
        ok = store.mark_done(parts[1])
        return (f"goal {parts[1]} marked done. 🎉" if ok
                else f"no goal {parts[1]!r} found.")
    if verb == "snooze" and len(parts) >= 2:
        days = 30.0
        if len(parts) >= 3:
            try:
                days = float(parts[2])
            except ValueError:
                return f"couldn't parse days {parts[2]!r}"
        ok = store.snooze(parts[1], days=days)
        return (f"goal {parts[1]} snoozed for {days:g} day(s) — no pace "
                f"nagging till then." if ok
                else f"no goal {parts[1]!r} found.")
    if verb == "remove" and len(parts) >= 2:
        ok = store.remove(parts[1])
        return (f"goal {parts[1]} removed." if ok
                else f"no goal {parts[1]!r} found.")
    return ("usage:\n"
            "  /goal add <name> <amount> [deadline] [target|monthly|by_date]\n"
            "  /goal list\n"
            "  /goal done <id> | /goal snooze <id> [days] | /goal remove <id>\n"
            "e.g. /goal add emergency-fund 500k dec\n"
            "e.g. /goal add rent 120k 2026-12-31 by_date")


# ── /insights ────────────────────────────────────────────────────────────

def control_insights(tail: str, context: Any) -> str:
    """/insights [days] — native money analytics."""
    from .insights import compute_insights, render_insights

    settings = getattr(context, "settings", None)
    ledger_path, _ = finance_paths(settings)
    days = 30
    tail = (tail or "").strip()
    if tail:
        try:
            days = max(7, min(365, int(tail)))
        except ValueError:
            return "usage: /insights [days]  (7–365)"
    return render_insights(compute_insights(Ledger(ledger_path),
                                            window_days=days))


# ── /recurring ───────────────────────────────────────────────────────────

def control_recurring(tail: str, context: Any) -> str:
    """/recurring [list|upcoming|sync] — subscription intelligence."""
    from .insights import (detect_recurring, render_recurring,
                           upcoming_bills)
    from .alerts import AlertStore, sync_bill_alerts

    settings = getattr(context, "settings", None)
    ledger_path, _ = finance_paths(settings)
    ledger = Ledger(ledger_path)
    verb = (tail or "").strip().lower() or "list"
    if verb == "upcoming":
        return render_recurring(upcoming_bills(ledger, days=14))
    if verb in ("sync", "sync_alerts", "sync-bills"):
        result = sync_bill_alerts(AlertStore(), ledger)
        return (f"watching {result['total']} recurring charge(s): "
                f"{result['created']} new bill alert(s), "
                f"{result['refreshed']} refreshed. I'll ping you before "
                f"each one lands.")
    if verb != "list":
        return "usage: /recurring [list|upcoming|sync]"
    txns = ledger.transactions(since=time.time() - 120 * 86400)
    return render_recurring(detect_recurring(txns))


# ── /trade ───────────────────────────────────────────────────────────────

_TRADE_USAGE = ("usage:\n"
                "  /trade size XAUUSD <entry> <stop> [risk%]  — position size\n"
                "  /trade open XAUUSD buy 0.5 --sl 2600 --tp 2700\n"
                "  /trade close <position_id>\n"
                "  /trade positions\n"
                "  /trade stats   (win rate, profit factor, expectancy…)\n"
                "  /trade heat    (portfolio heat vs 6%)\n"
                "  /trade kelly   (Kelly sizing from your journal)")


def _trade_desk(context: Any) -> Any:
    from .trading_desk import TradingDesk
    connector = getattr(context, "exness", None)
    if connector is None:
        from ..connectors.exness import ExnessConnector
        from ..accounts.vault import CredentialVault
        import os
        passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
        if not passphrase:
            raise RuntimeError("vault locked — set NM_VAULT_PASSPHRASE")
        vault = CredentialVault(getattr(context, "db", None),
                                master_passphrase=passphrase)
        connector = ExnessConnector(vault=vault)
    return TradingDesk(connector, mode="paper")


def control_trade(tail: str, context: Any) -> str:
    """/trade … — the trading desk's chat surface (paper by default)."""
    from .trading_desk import DeskError, size_position

    parts = (tail or "").split()
    if not parts:
        return _TRADE_USAGE
    verb = parts[0].lower()
    try:
        desk = _trade_desk(context)
    except Exception as exc:
        return f"couldn't reach the trading desk: {exc}"
    try:
        if verb == "stats":
            return desk.render_stats()
        if verb == "positions":
            poss = desk.paper_positions()
            if not poss:
                return "no open paper positions."
            lines = ["paper positions:"]
            for p in poss:
                lines.append(
                    f"  • {p.id} {p.side} {p.volume} {p.instrument} @ "
                    f"{p.entry} (SL {p.stop_loss or '—'}, "
                    f"TP {p.take_profit or '—'}, "
                    f"risk {p.risk_amount:.2f})")
            return "\n".join(lines)
        if verb == "heat":
            h = desk.portfolio_heat()
            flag = ("✅ within the 6% rule" if h["within_limits"]
                    else "🚨 OVER 6% — reduce risk")
            return (f"portfolio heat: {h['heat_pct']:.1f}% "
                    f"({h['heat_amount']:.2f} risked on "
                    f"{h['equity']:.2f} equity, "
                    f"{h['open_positions']} open) — {flag}")
        if verb == "kelly":
            k = desk.kelly_fraction()
            if not k.get("ok"):
                return k["reason"]
            return (f"Kelly: {k['kelly_pct']:.1f}% full / "
                    f"{k['quarter_kelly_pct']:.1f}% quarter-Kelly. "
                    f"{k['note']}")
        if verb == "size" and len(parts) >= 4:
            instrument = parts[1].upper()
            entry, stop = float(parts[2]), float(parts[3])
            risk_pct = float(parts[4]) if len(parts) >= 5 else \
                desk.policy.max_risk_pct_per_trade
            spec = desk.contract_specs(instrument)
            cv = desk._contract_value(instrument)
            sized = size_position(
                equity=desk.equity(), entry=entry, stop_loss=stop,
                risk_pct=risk_pct, contract_value=cv,
                volume_min=float(spec.get("volume_min", 0.01)),
                volume_max=float(spec.get("volume_max", 100.0)),
                volume_step=float(spec.get("volume_step", 0.01)))
            return (f"size for {instrument}: {sized['volume']} lots — "
                    f"risking {sized['actual_risk']:.2f} "
                    f"({sized['actual_risk_pct']:.2f}%) on a "
                    f"{sized['stop_distance']} stop")
        if verb == "open" and len(parts) >= 4:
            instrument, side, volume = parts[1].upper(), parts[2].lower(), \
                float(parts[3])
            sl = tp = None
            for i, p in enumerate(parts):
                if p == "--sl" and i + 1 < len(parts):
                    sl = float(parts[i + 1])
                if p == "--tp" and i + 1 < len(parts):
                    tp = float(parts[i + 1])
            pos = desk.paper_open(instrument, side, volume,
                                  stop_loss=sl, take_profit=tp)
            return (f"paper opened: {pos.id} {side} {volume} "
                    f"{instrument} @ {pos.entry} "
                    f"(SL {sl or '—'}, TP {tp or '—'})")
        if verb == "close" and len(parts) >= 2:
            closed = desk.paper_close(parts[1])
            r = (f" ({closed['r_multiple']:+.2f}R)"
                 if closed.get("r_multiple") is not None else "")
            return (f"closed {closed['position_id']}: "
                    f"{closed['pnl']:+.2f}{r} — {closed['reason']}")
    except DeskError as exc:
        return f"desk refused: {exc}"
    except (ValueError, IndexError):
        return "couldn't parse that.\n" + _TRADE_USAGE
    return _TRADE_USAGE


# ── /import ──────────────────────────────────────────────────────────────

def control_import(tail: str, context: Any) -> str:
    """/import <csv path> — import a bank statement into the ledger."""
    settings = getattr(context, "settings", None)
    ledger_path, _ = finance_paths(settings)
    path = (tail or "").strip().strip("'\"")
    if not path:
        return ("usage: /import <csv path>\n"
                "imports a bank-statement CSV (columns auto-sniffed), "
                "skipping duplicates.")
    ledger = Ledger(ledger_path)
    try:
        report = ledger.import_csv(path)
    except Exception as exc:
        return f"import failed: {exc}"
    lines = [f"imported {report['imported']} transaction(s)"]
    if report["skipped_duplicates"]:
        lines.append(f"skipped {report['skipped_duplicates']} duplicate(s)")
    for err in report["errors"][:5]:
        lines.append(f"  ! {err}")
    if len(report["errors"]) > 5:
        lines.append(f"  …and {len(report['errors']) - 5} more errors")
    return "\n".join(lines)
