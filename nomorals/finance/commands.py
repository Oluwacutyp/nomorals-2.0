"""Chat command handlers: /spend, /budget, /spending, /mandate, /balances,
/alert, /goal, /insights."""

from __future__ import annotations

import re
import time
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


# ── /mandate ───────────────────────────────────────────────────────────

def control_mandate(tail: str, context: Any) -> str:
    """/mandate issue <scope> <per-txn> <per-day> [ttl-days]
    /mandate list | /mandate revoke <id> | /mandate revoke-all"""
    from .mandate import MandateStore, SCOPE_ALL, SCOPE_TRANSFER, SCOPE_TRAVEL

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
            lines.append(
                f"  • {m.id} [{state}] scope={m.scope} "
                f"{format_naira(m.cap_per_txn)}/txn "
                f"{format_naira(m.cap_per_day)}/day")
        return "\n".join(lines)
    verb = parts[0].lower()
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
        if len(parts) >= 5:
            try:
                ttl = float(parts[4])
            except ValueError:
                return f"couldn't parse ttl {parts[4]!r} (days)"
        try:
            m = store.issue(scope=scope, cap_per_txn=txn_kobo,
                            cap_per_day=day_kobo, ttl_days=ttl)
        except Exception as exc:
            return f"couldn't issue mandate: {exc}"
        return (f"mandate issued: {m.id}\nscope={m.scope} · "
                f"{format_naira(m.cap_per_txn)}/txn · "
                f"{format_naira(m.cap_per_day)}/day · {ttl:g} days\n"
                f"Revoke anytime: /mandate revoke {m.id} "
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
            "  /mandate issue <transfer|travel|all> <per-txn> <per-day> [days]\n"
            "  /mandate list\n"
            "  /mandate revoke <id> | /mandate revoke-all\n"
            "e.g. /mandate issue transfer 50k 200k")


# ── /balances ──────────────────────────────────────────────────────────

def control_balances(tail: str, context: Any) -> str:
    """/balances — unified money view across every connected rail."""
    from .overview import collect_balances, render_overview

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
    except Exception as exc:
        return f"couldn't collect balances: {exc}"
    return render_overview(ov)


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
                "  /alert list\n"
                "  /alert remove <id>\n"
                "  /alert check   (evaluate now)")


def control_alert(tail: str, context: Any) -> str:
    """/alert add|list|remove|check — price & money alerts."""
    from .alerts import AlertStore, add_alert, evaluate_alerts

    parts = (tail or "").split()
    if not parts or parts[0].lower() == "list":
        alerts = AlertStore().list()
        if not alerts:
            return "no alerts set.\n" + _ALERT_USAGE
        lines = ["alerts:"]
        for a in alerts:
            state = "on" if a.enabled else "off"
            extra = (f" · fired {a.fire_count}×" if a.fire_count else "")
            lines.append(f"  • {a.id} [{state}] {a.describe()}{extra}")
        return "\n".join(lines)
    verb = parts[0].lower()
    if verb == "remove" and len(parts) >= 2:
        ok = AlertStore().remove(parts[1])
        return (f"alert {parts[1]} removed." if ok
                else f"no alert {parts[1]!r} found.")
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
    """/goal add <name> <amount> [deadline] | /goal list | /goal done <id>"""
    from .goals import GoalStore, create_goal, goal_progress

    settings = getattr(context, "settings", None)
    ledger_path, _ = finance_paths(settings)
    ledger = Ledger(ledger_path)
    store = GoalStore()
    parts = (tail or "").split()
    if not parts or parts[0].lower() == "list":
        goals = store.list()
        if not goals:
            return ("no savings goals yet — "
                    "e.g. /goal add emergency-fund 500k dec")
        lines = ["savings goals:"]
        for g in goals:
            p = goal_progress(g, ledger)
            bar = _progress_bar(p["pct"])
            pace = ""
            if p.get("on_track") is False:
                pace = " ⚠️ off pace"
            elif p.get("on_track") is True:
                pace = " ✅ on pace"
            dl = f" by {p['deadline']}" if p["deadline"] else ""
            lines.append(f"  • {g.name}: {bar} {p['pct']:.0%} "
                         f"({p['contributed']} of {p['target']}){dl}{pace}")
        return "\n".join(lines)
    verb = parts[0].lower()
    if verb == "add" and len(parts) >= 3:
        name = parts[1]
        kobo = parse_amount(parts[2])
        if kobo is None or kobo <= 0:
            return f"couldn't parse amount {parts[2]!r} — try /goal add name 500k"
        deadline = parts[3] if len(parts) >= 4 else ""
        try:
            g = create_goal(name, kobo, deadline=deadline)
        except ValueError as exc:
            return f"couldn't create goal: {exc}"
        return (f"goal set: {g.name} → {format_naira(g.target_kobo)}"
                + (f" by {deadline}" if deadline else "")
                + "\nLog savings with /spend <amount> on savings <note>.")
    if verb == "done" and len(parts) >= 2:
        ok = store.mark_done(parts[1])
        return (f"goal {parts[1]} marked done. 🎉" if ok
                else f"no goal {parts[1]!r} found.")
    if verb == "remove" and len(parts) >= 2:
        ok = store.remove(parts[1])
        return (f"goal {parts[1]} removed." if ok
                else f"no goal {parts[1]!r} found.")
    return ("usage:\n"
            "  /goal add <name> <amount> [deadline]\n"
            "  /goal list\n"
            "  /goal done <id> | /goal remove <id>\n"
            "e.g. /goal add emergency-fund 500k dec")


def _progress_bar(pct: float, width: int = 10) -> str:
    filled = max(0, min(width, int(pct * width)))
    return "█" * filled + "░" * (width - filled)


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
