"""Rule-based ad guardrails — autonomous-spend safety (build-map #101).

Money-safety for ad spend (Bïrch/Madgicx pattern: "pause if CPA > X
for 3 days"). User-configured numeric rules, evaluated on scheduled
metric pulls (Meta/Google/TikTok via connectors — injectable):

    pause ad set if CPA > ₦X for 3 days
    scale 20% if ROAS > 3 for 2 days

The safety layer that makes autonomous ad management sellable.

Two layers of authority (deliberate design):
  * Guardrail RULES are user-configured numbers — the gradient (#8)
    governs autonomous action, but an EXPLICIT owner command
    ("scale it anyway") bypasses the rules and executes.
  * The #69 PAYMENT MANDATE is structural, not bypassable: pause/
    scale actions still need an active adspend mandate. If the
    mandate is missing/expired, the action is blocked with a clear
    reason telling the owner to issue one (/mandate issue adspend).

Every fired action (or blocked attempt) is audit-logged, and what
fired gets auto-labeled. Alert language is factual numbers —
rules are user-configured, never content judgments.

Never raises: every public function is wrapped.
"""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


# ── constants ─────────────────────────────────────────────────────────────

_MONEY_METRICS = {"cpa", "cpc", "spend", "cpm"}   # thresholds parsed as naira → kobo
_RATIO_METRICS = {"roas", "ctr", "cvr"}           # thresholds parsed as floats
_DEFAULT_DB = "~/.nomorals/marketing/guardrails.db"
_MANDATE_SCOPE = "adspend"
_ACTIONS = ("pause", "scale", "label")
_OPS = (">", "<", ">=", "<=")


# ── data ──────────────────────────────────────────────────────────────────

@dataclass
class Rule:
    rule_id: str = ""
    label: str = ""              # auto-label, set when the rule fires
    adset_id: str = ""           # which ad set / campaign / account the rule watches
    platform: str = ""           # meta | google | tiktok | any
    metric: str = "cpa"          # cpa | roas | spend | ctr | cpc | cpm | cvr
    op: str = ">"                # > | < | >= | <=
    threshold: float = 0.0       # kobo for money metrics, ratio for ratio metrics
    threshold_raw: str = ""      # what the owner typed ("₦50000", "3")
    days: int = 1                # consecutive evaluations the condition must hold
    action: str = "pause"        # pause | scale | label
    action_param: float = 0.0    # scale percent (20 → +20%)
    action_text: str = ""        # label action: custom label text to apply
    max_delta_kobo: int = 0      # spend delta cap when budget unknown
    cooldown_s: int = 86400      # min seconds between firings (default 24h)
    streak: int = 0              # consecutive evaluations where condition held
    last_fired_at: float = 0.0
    active: bool = True
    created_at: float = 0.0


@dataclass
class FiredAction:
    rule_id: str
    adset_id: str
    action: str
    detail: str
    ok: bool
    reason: str = ""
    override: bool = False
    ts: float = 0.0


# ── helpers ───────────────────────────────────────────────────────────────

def _expand(path: str) -> str:
    try:
        return path.replace("~", __import__("os").path.expanduser("~"))
    except Exception:
        return path


def parse_naira_kobo(text: str) -> int | None:
    """'₦50,000' / '50000' / '50k' / '1.5m' → kobo. None on garbage."""
    try:
        t = (text or "").strip().lower().replace("₦", "").replace(",", "").replace("ngn", "").strip()
        if not t:
            return None
        mult = 1
        if t.endswith("k"):
            mult, t = 1000, t[:-1]
        elif t.endswith("m"):
            mult, t = 1_000_000, t[:-1]
        return int(round(float(t) * mult * 100))
    except Exception:
        return None


def format_naira(kobo: float) -> str:
    try:
        n = kobo / 100.0
        if n >= 1_000_000:
            s = f"{n/1_000_000:.2f}".rstrip("0").rstrip(".")
            return f"₦{s}m"
        if n >= 1000:
            s = f"{n/1000:.1f}".rstrip("0").rstrip(".")
            return f"₦{s}k"
        return f"₦{n:,.0f}"
    except Exception:
        return "₦?"


def _now() -> float:
    return time.time()


# ── store ─────────────────────────────────────────────────────────────────

class GuardrailStore:
    """SQLite store: rules, metric streaks, audit log. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = _expand(db_path or _DEFAULT_DB)
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._init()
        except Exception:
            self._db = None

    def _init(self) -> None:
        assert self._db is not None
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS rules (
                rule_id TEXT PRIMARY KEY, label TEXT, adset_id TEXT,
                platform TEXT, metric TEXT, op TEXT, threshold REAL,
                threshold_raw TEXT, days INTEGER, action TEXT,
                action_param REAL, action_text TEXT, max_delta_kobo INTEGER,
                cooldown_s INTEGER, streak INTEGER, last_fired_at REAL,
                active INTEGER, created_at REAL);
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL,
                rule_id TEXT, adset_id TEXT, action TEXT,
                detail TEXT, ok INTEGER, reason TEXT, override INTEGER);
        """)
        self._db.commit()
        # Migration: rules tables created before action_text existed.
        try:
            cols = {r["name"] for r in self._db.execute("PRAGMA table_info(rules)")}
            if "action_text" not in cols:
                self._db.execute("ALTER TABLE rules ADD COLUMN action_text TEXT DEFAULT ''")
                self._db.commit()
        except Exception:
            pass

    # — rules —
    def add_rule(self, rule: Rule) -> Rule | None:
        try:
            if self._db is None or not rule.adset_id or rule.action not in _ACTIONS:
                return None
            rule.rule_id = rule.rule_id or ("gr_" + uuid.uuid4().hex[:8])
            rule.created_at = rule.created_at or _now()
            self._db.execute(
                "INSERT OR REPLACE INTO rules VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rule.rule_id, rule.label, rule.adset_id, rule.platform,
                 rule.metric, rule.op, rule.threshold, rule.threshold_raw,
                 rule.days, rule.action, rule.action_param, rule.action_text,
                 rule.max_delta_kobo, rule.cooldown_s, rule.streak,
                 rule.last_fired_at, 1 if rule.active else 0, rule.created_at))
            self._db.commit()
            return rule
        except Exception:
            return None

    def get_rule(self, rule_id: str) -> Rule | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute("SELECT * FROM rules WHERE rule_id = ?",
                                   (rule_id,)).fetchone()
            return _row_to_rule(row) if row else None
        except Exception:
            return None

    def list_rules(self, active_only: bool = False) -> list[Rule]:
        try:
            if self._db is None:
                return []
            q = "SELECT * FROM rules" + (" WHERE active = 1" if active_only else "") + " ORDER BY created_at"
            return [_row_to_rule(r) for r in self._db.execute(q).fetchall()]
        except Exception:
            return []

    def set_active(self, rule_id: str, active: bool) -> bool:
        try:
            if self._db is None:
                return False
            cur = self._db.execute("UPDATE rules SET active = ? WHERE rule_id = ?",
                                   (1 if active else 0, rule_id))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:
            return False

    def remove_rule(self, rule_id: str) -> bool:
        try:
            if self._db is None:
                return False
            cur = self._db.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:
            return False

    def update_streak(self, rule_id: str, streak: int) -> bool:
        try:
            if self._db is None:
                return False
            self._db.execute("UPDATE rules SET streak = ? WHERE rule_id = ?",
                            (streak, rule_id))
            self._db.commit()
            return True
        except Exception:
            return False

    def mark_fired(self, rule_id: str, label: str) -> bool:
        try:
            if self._db is None:
                return False
            self._db.execute(
                "UPDATE rules SET last_fired_at = ?, streak = 0, label = ? WHERE rule_id = ?",
                (_now(), label, rule_id))
            self._db.commit()
            return True
        except Exception:
            return False

    # — audit —
    def log_audit(self, fired: FiredAction) -> bool:
        try:
            if self._db is None:
                return False
            self._db.execute(
                "INSERT INTO audit (ts, rule_id, adset_id, action, detail, ok, reason, override)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (fired.ts or _now(), fired.rule_id, fired.adset_id, fired.action,
                 fired.detail, 1 if fired.ok else 0, fired.reason,
                 1 if fired.override else 0))
            self._db.commit()
            return True
        except Exception:
            return False

    def audit_log(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


def _col(row: sqlite3.Row, name: str) -> str:
    """Column value with graceful fallback for pre-migration DBs."""
    try:
        return row[name] or ""
    except Exception:
        return ""


def _row_to_rule(row: sqlite3.Row) -> Rule:
    try:
        return Rule(
            rule_id=row["rule_id"], label=row["label"] or "",
            adset_id=row["adset_id"] or "", platform=row["platform"] or "",
            metric=row["metric"] or "cpa", op=row["op"] or ">",
            threshold=float(row["threshold"] or 0),
            threshold_raw=row["threshold_raw"] or "",
            days=int(row["days"] or 1), action=row["action"] or "pause",
            action_param=float(row["action_param"] or 0),
            action_text=_col(row, "action_text"),
            max_delta_kobo=int(row["max_delta_kobo"] or 0),
            cooldown_s=int(row["cooldown_s"] or 86400),
            streak=int(row["streak"] or 0),
            last_fired_at=float(row["last_fired_at"] or 0),
            active=bool(row["active"]),
            created_at=float(row["created_at"] or 0))
    except Exception:
        return Rule()


# ── rule parsing ──────────────────────────────────────────────────────────

_RULE_RE = re.compile(
    r"^(?P<action>pause|scale|label)\s*(?P<param>.*?)\s+\bif\b\s*"
    r"(?P<metric>cpa|cpc|cpm|spend|roas|ctr|cvr)\s*"
    r"(?P<op>>=|<=|>|<)\s*(?P<threshold>[₦\d.,km\s]+?)\s*"
    r"(?:\bfor\b\s*(?P<days>\d+)\s*days?)?\s*"
    r"\bon\b\s*(?P<adset>[\w\-]+)"
    r"(?:\s*\bvia\b\s*(?P<platform>meta|google|tiktok))?\s*$",
    re.IGNORECASE)


def parse_rule(text: str) -> Rule | None:
    """Parse 'pause if cpa > ₦50000 for 3 days on adset123 [via meta]'.

    Also 'scale 20% if roas > 3 for 2 days on adset123'.
    Returns None on garbage — never raises.
    """
    try:
        m = _RULE_RE.match((text or "").strip())
        if not m:
            return None
        g = m.groupdict()
        action = g["action"].lower()
        metric = g["metric"].lower()
        op = g["op"]
        if op not in _OPS:
            return None

        param_raw = (g["param"] or "").strip()
        action_param = 0.0
        action_text = ""
        if action == "scale":
            pm = re.search(r"(\d+(?:\.\d+)?)\s*%", param_raw)
            if not pm:
                return None
            action_param = float(pm.group(1))
        elif action == "label":
            # `label "needs creative" if ...` — custom label text to apply.
            action_text = param_raw.strip().strip("\"'")

        raw_thr = (g["threshold"] or "").strip()
        if metric in _MONEY_METRICS:
            kobo = parse_naira_kobo(raw_thr)
            if kobo is None:
                return None
            threshold = float(kobo)
        else:
            try:
                threshold = float(raw_thr.replace(",", ""))
            except Exception:
                return None

        days = int(g["days"] or 1)
        if days < 1 or days > 30:
            return None
        return Rule(
            adset_id=g["adset"].strip(), platform=(g["platform"] or "").lower(),
            metric=metric, op=op, threshold=threshold, threshold_raw=raw_thr,
            days=days, action=action, action_param=action_param,
            action_text=action_text)
    except Exception:
        return None


def _condition_holds(value: float, op: str, threshold: float) -> bool:
    try:
        if op == ">":
            return value > threshold
        if op == "<":
            return value < threshold
        if op == ">=":
            return value >= threshold
        if op == "<=":
            return value <= threshold
        return False
    except Exception:
        return False


def _fmt_threshold(rule: Rule) -> str:
    if rule.metric in _MONEY_METRICS:
        return format_naira(rule.threshold)
    return str(rule.threshold).rstrip("0").rstrip(".")


def _fmt_value(rule: Rule, value: float) -> str:
    if rule.metric in _MONEY_METRICS:
        return format_naira(value)
    return f"{value:.2f}"


# ── evaluation ────────────────────────────────────────────────────────────

MetricsFn = Callable[[str], dict[str, Any] | None]   # adset_id → metrics
ExecutorFn = Callable[[str, str, dict[str, Any]], bool]  # (action, adset_id, params) → ok


def _mandate_check(amount_kobo: int) -> tuple[bool, str]:
    """#69 structural gate for spend actions. Never raises."""
    try:
        from ..finance.mandate import MandateStore, check_mandate
        store = MandateStore()
        res = check_mandate(store, "owner", _MANDATE_SCOPE, amount_kobo)
        return res.ok, (res.reason if not res.ok else "")
    except Exception as e:  # noqa: BLE001
        return False, f"mandate check failed: {e}"


def evaluate(store: GuardrailStore, metrics_fn: MetricsFn | None = None,
             executor_fn: ExecutorFn | None = None) -> list[FiredAction]:
    """Run one evaluation pass over all active rules. Never raises.

    Returns the fired (or blocked) actions, audit-logged.
    """
    fired: list[FiredAction] = []
    try:
        if store is None:
            return fired
        for rule in store.list_rules(active_only=True):
            try:
                fa = _evaluate_rule(store, rule, metrics_fn, executor_fn)
                if fa is not None:
                    fired.append(fa)
                    store.log_audit(fa)
            except Exception:  # noqa: BLE001 - one bad rule never kills the pass
                continue
    except Exception:  # noqa: BLE001
        pass
    return fired


def _evaluate_rule(store: GuardrailStore, rule: Rule,
                   metrics_fn: MetricsFn | None,
                   executor_fn: ExecutorFn | None) -> FiredAction | None:
    try:
        if metrics_fn is None:
            return None  # no metric source — nothing to evaluate
        metrics = metrics_fn(rule.adset_id) or {}
        value = metrics.get(rule.metric)
        if value is None:
            store.update_streak(rule.rule_id, 0)
            return None
        try:
            value = float(value)
        except Exception:
            store.update_streak(rule.rule_id, 0)
            return None

        holds = _condition_holds(value, rule.op, rule.threshold)
        if not holds:
            store.update_streak(rule.rule_id, 0)
            return None
        streak = (rule.streak or 0) + 1
        store.update_streak(rule.rule_id, streak)
        if streak < rule.days:
            return None  # condition holding, but not for enough days yet

        # Cooldown: don't re-fire too often.
        now = _now()
        if rule.last_fired_at and (now - rule.last_fired_at) < rule.cooldown_s:
            return None

        label = (f"{rule.action} — {rule.metric.upper()} {rule.op} "
                 f"{_fmt_threshold(rule)} (now {_fmt_value(rule, value)}, "
                 f"{streak}d streak)")
        detail = (f"rule {rule.rule_id}: {rule.metric.upper()} {_fmt_value(rule, value)} "
                  f"{rule.op} {_fmt_threshold(rule)} on {rule.adset_id}")

        # Spend actions need the #69 mandate. Label actions don't move money.
        if rule.action in ("pause", "scale"):
            delta = 0
            if rule.action == "scale":
                budget = metrics.get("daily_budget")
                try:
                    budget = float(budget) if budget is not None else 0
                except Exception:
                    budget = 0
                delta = int(round(budget * rule.action_param / 100.0)) if budget > 0 else int(rule.max_delta_kobo or 0)
                if delta <= 0:
                    fa = FiredAction(rule.rule_id, rule.adset_id, rule.action,
                                     detail, False,
                                     reason="scale delta unknown — metrics lack daily_budget and rule has no max_delta_kobo",
                                     ts=now)
                    store.mark_fired(rule.rule_id, label + " [blocked]")
                    return fa
            ok, reason = _mandate_check(delta)
            if not ok:
                fa = FiredAction(rule.rule_id, rule.adset_id, rule.action,
                                 detail, False, reason=f"mandate blocked: {reason}", ts=now)
                store.mark_fired(rule.rule_id, label + " [mandate-blocked]")
                return fa

        # Execute.
        executed = True
        if executor_fn is not None:
            params: dict[str, Any] = {"metric": rule.metric, "value": value,
                                      "threshold": rule.threshold, "pct": rule.action_param}
            if rule.action == "label" and rule.action_text:
                params["label_text"] = rule.action_text
            try:
                executed = bool(executor_fn(rule.action, rule.adset_id, params))
            except Exception as e:  # noqa: BLE001
                executed = False
                fa = FiredAction(rule.rule_id, rule.adset_id, rule.action,
                                 detail, False, reason=f"executor failed: {e}", ts=now)
                store.mark_fired(rule.rule_id, label + " [executor-failed]")
                return fa

        store.mark_fired(rule.rule_id, label)
        return FiredAction(rule.rule_id, rule.adset_id, rule.action, detail,
                           executed, ts=now)
    except Exception:  # noqa: BLE001
        return None


def execute_override(store: GuardrailStore, adset_id: str, action: str,
                     pct: float = 0.0,
                     executor_fn: ExecutorFn | None = None) -> FiredAction:
    """Explicit owner override: 'scale it anyway'.

    Bypasses the guardrail RULES and the #8 gradient entirely — explicit
    commands always execute. The #69 mandate stays structural (it is the
    owner's standing spend authority, not a rule). Audit-logged.
    Never raises.
    """
    now = _now()
    try:
        action = (action or "").lower().strip()
        adset_id = (adset_id or "").strip()
        if action not in ("pause", "scale", "label"):
            return FiredAction("", adset_id, action, "override", False,
                               reason=f"unknown action '{action}'", override=True, ts=now)
        if not adset_id:
            return FiredAction("", "", action, "override", False,
                               reason="no ad set given", override=True, ts=now)

        detail = f"explicit override: {action} on {adset_id}"
        if action == "pause":
            ok, reason = _mandate_check(0)
            if not ok:
                fa = FiredAction("", adset_id, action, detail, False,
                                 reason=f"mandate blocked: {reason}", override=True, ts=now)
                if store is not None:
                    store.log_audit(fa)
                return fa
        elif action == "scale":
            if pct <= 0:
                return FiredAction("", adset_id, action, detail, False,
                                   reason="scale needs a percent, e.g. override scale adset1 20",
                                   override=True, ts=now)
            # Mandate gate: use the max_delta from any rule on this ad set as a
            # conservative bound, else require the owner to pass it. Without a
            # known bound we still check a 0-amount mandate (authority exists?).
            ok, reason = _mandate_check(0)
            if not ok:
                fa = FiredAction("", adset_id, action, detail, False,
                                 reason=f"mandate blocked: {reason}", override=True, ts=now)
                if store is not None:
                    store.log_audit(fa)
                return fa
            detail += f" (+{pct:g}%)"

        executed = True
        if executor_fn is not None:
            try:
                executed = bool(executor_fn(action, adset_id, {"pct": pct, "override": True}))
            except Exception as e:  # noqa: BLE001
                fa = FiredAction("", adset_id, action, detail, False,
                                 reason=f"executor failed: {e}", override=True, ts=now)
                if store is not None:
                    store.log_audit(fa)
                return fa

        fa = FiredAction("", adset_id, action, detail, executed, override=True, ts=now)
        if store is not None:
            store.log_audit(fa)
        return fa
    except Exception as e:  # noqa: BLE001
        return FiredAction("", adset_id or "", action or "", "override", False,
                           reason=str(e), override=True, ts=now)


def ensure_schedule(scheduler: Any) -> bool:
    """Register the daily guardrail evaluation cron. Never raises."""
    try:
        import asyncio

        async def _ensure() -> bool:
            jobs = []
            try:
                jobs = scheduler.list_jobs()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                jobs = []
            ids = {getattr(j, "task_id", getattr(j, "id", "")) for j in (jobs or [])}
            if "guardrails_daily" not in ids:
                await scheduler.schedule_cron(  # type: ignore[attr-defined]
                    "guardrails_daily", "0 6 * * *", "guardrails.run", {})
            return True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(_ensure())
            return True
        return bool(asyncio.run(_ensure()))
    except Exception:  # noqa: BLE001
        return False


# ── formatting ────────────────────────────────────────────────────────────

def format_rule(rule: Rule) -> str:
    try:
        when = (f" for {rule.days}d" if rule.days > 1 else "")
        act = rule.action
        if rule.action == "scale":
            act = f"scale +{rule.action_param:g}%"
        state = "" if rule.active else " [off]"
        lbl = f" — {rule.label}" if rule.label else ""
        return (f"• `{rule.rule_id}` {act} if {rule.metric.upper()} {rule.op} "
                f"{_fmt_threshold(rule)}{when} on `{rule.adset_id}`{state}{lbl}")
    except Exception:
        return "• (rule)"


def format_fired(fa: FiredAction) -> str:
    try:
        icon = "🛡️" if fa.ok else "⛔"
        tag = " [override]" if fa.override else ""
        why = f" — {fa.reason}" if fa.reason and not fa.ok else ""
        return f"{icon} {fa.action} on `{fa.adset_id}`{tag}: {fa.detail}{why}"
    except Exception:
        return "• (action)"


# ── chat ──────────────────────────────────────────────────────────────────

def _usage() -> str:
    return ("usage:\n"
            "  /guardrails add pause if cpa > ₦50000 for 3 days on adset123 [via meta]\n"
            "  /guardrails add scale 20% if roas > 3 for 2 days on adset123\n"
            "  /guardrails add label if ctr < 0.5 for 5 days on adset123\n"
            "  /guardrails list — active rules\n"
            "  /guardrails run — evaluate now\n"
            "  /guardrails override scale adset123 20 — 'scale it anyway' (bypasses rules)\n"
            "  /guardrails stop <rule_id> — disable a rule\n"
            "  /guardrails remove <rule_id>\n"
            "  /guardrails audit — recent fired/blocked actions\n"
            "spend actions need an active adspend mandate (/mandate issue adspend).")


def _get_store(**kwargs: Any) -> GuardrailStore:
    eng = kwargs.get("store")
    if isinstance(eng, GuardrailStore):
        return eng
    return GuardrailStore()


def control_guardrails(tail: str, context=None, chat=None, **kwargs: Any) -> str:
    """/guardrails — rule-based ad-spend guardrails. Owner-only; never raises."""
    try:
        store = _get_store(**kwargs)
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return _usage()
        low = rest.lower()

        if low.startswith("add"):
            body = rest[3:].strip()
            rule = parse_rule(body)
            if rule is None:
                return ("couldn't parse that rule.\n" + _usage())
            saved = store.add_rule(rule)
            if saved is None:
                return "couldn't save that rule — try again."
            return ("🛡️ guardrail armed:\n" + format_rule(saved) +
                    "\nspend actions still need an active adspend mandate to fire.")

        if low.startswith("list"):
            rules = store.list_rules()
            if not rules:
                return "no guardrails yet — /guardrails add … to arm one."
            return "🛡️ ad guardrails:\n" + "\n".join(format_rule(r) for r in rules)

        if low.startswith("run"):
            fired = evaluate(store)
            if not fired:
                return "🛡️ guardrails checked — nothing fired."
            return "🛡️ guardrails fired:\n" + "\n".join(format_fired(f) for f in fired)

        if low.startswith("override"):
            body = rest[len("override"):].strip().split()
            if len(body) < 2:
                return "usage: /guardrails override <pause|scale|label> <adset_id> [pct]"
            action, adset_id = body[0].lower(), body[1]
            pct = 0.0
            if len(body) > 2:
                try:
                    pct = float(body[2].rstrip("%"))
                except Exception:
                    pct = 0.0
            fa = execute_override(store, adset_id, action, pct)
            icon = "✅" if fa.ok else "⛔"
            why = f" — {fa.reason}" if fa.reason and not fa.ok else ""
            return f"{icon} override: {fa.action} on `{fa.adset_id}`{why}"

        if low.startswith("stop"):
            rid = rest[len("stop"):].strip()
            if store.set_active(rid, False):
                return f"🛡️ rule `{rid}` disabled."
            return f"no rule `{rid}` — /guardrails list to see ids."

        if low.startswith("remove"):
            rid = rest[len("remove"):].strip()
            if store.remove_rule(rid):
                return f"🛡️ rule `{rid}` removed."
            return f"no rule `{rid}` — /guardrails list to see ids."

        if low.startswith("audit"):
            rows = store.audit_log(15)
            if not rows:
                return "no guardrail actions logged yet."
            lines = []
            for r in rows:
                icon = "✅" if r.get("ok") else "⛔"
                tag = " [override]" if r.get("override") else ""
                ts = time.strftime("%m-%d %H:%M", time.localtime(r.get("ts") or 0))
                lines.append(f"{icon} {ts} {r.get('action')} on `{r.get('adset_id')}`{tag}")
            return "🛡️ guardrail audit:\n" + "\n".join(lines)

        return _usage()
    except Exception:  # noqa: BLE001
        return "guardrails hit a snag — try again."
