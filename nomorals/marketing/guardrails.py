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
_PLAIN_METRICS = {"frequency", "impressions", "clicks"}  # plain floats
_ALL_METRICS = _MONEY_METRICS | _RATIO_METRICS | _PLAIN_METRICS
_DEFAULT_DB = "~/.nomorals/marketing/guardrails.db"
_MANDATE_SCOPE = "adspend"
_ACTIONS = ("pause", "scale", "label", "alert")
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
    action: str = "pause"        # pause | scale | label | alert
    action_param: float = 0.0    # scale percent (20 → +20%)
    action_text: str = ""        # label action: custom label text to apply
    max_delta_kobo: int = 0      # spend delta cap when budget unknown
    cooldown_s: int = 86400      # min seconds between firings (default 24h)
    streak: int = 0              # consecutive evaluations where condition held
    last_fired_at: float = 0.0
    active: bool = True
    created_at: float = 0.0
    # ── sweep upgrade: compound conditions + evidence gate ──
    conditions: list = field(default_factory=list)  # [{metric, op, threshold, threshold_raw}]
    cond_op: str = "and"         # how conditions combine: and | or
    min_spend_kobo: int = 0      # don't judge before this spend (learning window)


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
            if path != ":memory:":
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
        # Migrations for tables created before newer columns existed.
        try:
            cols = {r["name"] for r in self._db.execute("PRAGMA table_info(rules)")}
            if "action_text" not in cols:
                self._db.execute("ALTER TABLE rules ADD COLUMN action_text TEXT DEFAULT ''")
            if "conditions_json" not in cols:
                self._db.execute("ALTER TABLE rules ADD COLUMN conditions_json TEXT DEFAULT '[]'")
            if "cond_op" not in cols:
                self._db.execute("ALTER TABLE rules ADD COLUMN cond_op TEXT DEFAULT 'and'")
            if "min_spend_kobo" not in cols:
                self._db.execute("ALTER TABLE rules ADD COLUMN min_spend_kobo INTEGER DEFAULT 0")
            self._db.commit()
        except Exception:
            pass
        # Metric snapshots for creative-fatigue detection.
        try:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS metric_history ("
                "adset_id TEXT, ts REAL, metrics_json TEXT)")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_mh_adset_ts ON "
                "metric_history(adset_id, ts)")
            self._db.commit()
        except Exception:
            pass

    # — rules —
    def add_rule(self, rule: Rule) -> Rule | None:
        try:
            if self._db is None or not rule.adset_id or rule.action not in _ACTIONS:
                return None
            import json as _json
            rule.rule_id = rule.rule_id or ("gr_" + uuid.uuid4().hex[:8])
            rule.created_at = rule.created_at or _now()
            cols = {r["name"] for r in
                    self._db.execute("PRAGMA table_info(rules)")}
            has_new = {"conditions_json", "cond_op", "min_spend_kobo"} <= cols
            if has_new:
                self._db.execute(
                    "INSERT OR REPLACE INTO rules VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rule.rule_id, rule.label, rule.adset_id, rule.platform,
                     rule.metric, rule.op, rule.threshold, rule.threshold_raw,
                     rule.days, rule.action, rule.action_param, rule.action_text,
                     rule.max_delta_kobo, rule.cooldown_s, rule.streak,
                     rule.last_fired_at, 1 if rule.active else 0, rule.created_at,
                     _json.dumps(list(rule.conditions or [])),
                     rule.cond_op or "and", int(rule.min_spend_kobo or 0)))
            else:
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
        import json as _json
        conditions: list = []
        cond_op = "and"
        min_spend = 0
        try:
            conditions = list(_json.loads(row["conditions_json"] or "[]"))
        except Exception:
            conditions = []
        try:
            cond_op = (row["cond_op"] or "and").lower()
        except Exception:
            pass
        try:
            min_spend = int(row["min_spend_kobo"] or 0)
        except Exception:
            pass
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
            created_at=float(row["created_at"] or 0),
            conditions=conditions, cond_op=cond_op,
            min_spend_kobo=min_spend)
    except Exception:
        return Rule()


# ── rule parsing ──────────────────────────────────────────────────────────

_COND_RE = re.compile(
    r"(?P<metric>cpa|cpc|cpm|spend|roas|ctr|cvr|frequency|impressions|clicks)\s*"
    r"(?P<op>>=|<=|>|<)\s*(?P<threshold>[₦\d.,km\s]+)",
    re.IGNORECASE)
_HEAD_RE = re.compile(
    r"^(?P<action>pause|scale|label|alert)\s*(?P<param>.*?)\s+\bif\b\s+"
    r"(?P<conds>.+?)\s+\bon\b\s*(?P<adset>[\w\-]+)\s*$",
    re.IGNORECASE | re.DOTALL)
_CLAUSE_DAYS = re.compile(r"\bfor\s+(?P<days>\d+)\s*days?\b", re.IGNORECASE)
_CLAUSE_VIA = re.compile(r"\bvia\s+(?P<platform>meta|google|tiktok)\b", re.IGNORECASE)
_CLAUSE_MINSPEND = re.compile(r"\bmin\s*-?\s*spend\b\s*(?P<ms>[₦\d.,km\s]+?)(?=\s*$|\s+via\b|\s+for\b)",
                              re.IGNORECASE)


def _parse_condition(text: str) -> dict | None:
    """One 'metric op threshold' condition → dict. None on garbage."""
    try:
        m = _COND_RE.fullmatch((text or "").strip())
        if not m:
            return None
        metric = m.group("metric").lower()
        op = m.group("op")
        if op not in _OPS:
            return None
        raw_thr = (m.group("threshold") or "").strip()
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
        return {"metric": metric, "op": op, "threshold": threshold,
                "threshold_raw": raw_thr}
    except Exception:
        return None


def parse_rule(text: str) -> Rule | None:
    """Parse 'pause if cpa > ₦50000 for 3 days on adset123 [via meta]'.

    Also 'scale 20% if roas > 3 for 2 days on adset123', compound
    'pause if cpa > ₦50k and frequency > 4 for 3 days on adset1',
    and 'alert if ctr < 0.5 on adset1 minspend ₦20000'.
    Returns None on garbage — never raises.
    """
    try:
        raw = (text or "").strip()
        if not raw:
            return None
        # Pull optional trailing clauses (order-independent).
        days = 1
        m = _CLAUSE_DAYS.search(raw)
        if m:
            days = int(m.group("days"))
            raw = (raw[:m.start()] + raw[m.end():]).strip()
        platform = ""
        m = _CLAUSE_VIA.search(raw)
        if m:
            platform = m.group("platform").lower()
            raw = (raw[:m.start()] + raw[m.end():]).strip()
        min_spend_kobo = 0
        m = _CLAUSE_MINSPEND.search(raw)
        if m:
            kobo = parse_naira_kobo(m.group("ms"))
            if kobo:
                min_spend_kobo = kobo
            raw = (raw[:m.start()] + raw[m.end():]).strip()

        m = _HEAD_RE.match(raw)
        if not m:
            return None
        g = m.groupdict()
        action = g["action"].lower()
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
        elif action == "alert" and param_raw:
            action_text = param_raw.strip().strip("\"'")

        conds_raw = (g["conds"] or "").strip()
        cond_op = "and"
        if re.search(r"\bor\b", conds_raw, re.IGNORECASE):
            cond_op = "or"
        splitter = re.compile(r"\s+(?:and|or)\s+", re.IGNORECASE)
        pieces = [p for p in splitter.split(conds_raw) if p.strip()]
        # Mixed and/or is refused — keep semantics unambiguous.
        if (re.search(r"\band\b", conds_raw, re.IGNORECASE)
                and re.search(r"\bor\b", conds_raw, re.IGNORECASE)):
            return None
        conditions = []
        for piece in pieces:
            c = _parse_condition(piece)
            if c is None:
                return None
            conditions.append(c)
        if not conditions:
            return None

        if days < 1 or days > 30:
            return None
        first = conditions[0]
        return Rule(
            adset_id=g["adset"].strip(), platform=platform,
            metric=first["metric"], op=first["op"],
            threshold=first["threshold"], threshold_raw=first["threshold_raw"],
            days=days, action=action, action_param=action_param,
            action_text=action_text, conditions=conditions, cond_op=cond_op,
            min_spend_kobo=min_spend_kobo)
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


# ── presets (Bïrch-style ready-made automation strategies) ───────────────

RULE_PRESETS: dict[str, dict[str, str]] = {
    "cpa_kill": {
        "description": "Kill switch — pause when CPA blows past target 3 days running.",
        "template": "pause if cpa > {thr} for 3 days on {adset}",
        "default_thr": "₦50000",
    },
    "scale_winners": {
        "description": "Scale proven winners — +20% budget when ROAS > 3 for 2 days.",
        "template": "scale 20% if roas > 3 for 2 days on {adset}",
        "default_thr": "",
    },
    "fatigue_watch": {
        "description": "Early warning — alert when CTR sags 5 days running.",
        "template": "alert creative-fatigue if ctr < {thr} for 5 days on {adset}",
        "default_thr": "0.8",
    },
    "frequency_cap": {
        "description": "Saturation watch — alert when frequency crosses 4.",
        "template": "alert saturation if frequency > 4 for 2 days on {adset}",
        "default_thr": "",
    },
    "budget_guard": {
        "description": "Spend cap — pause when spend exceeds the daily cap.",
        "template": "pause if spend > {thr} for 1 days on {adset} minspend {thr}",
        "default_thr": "₦100000",
    },
}


def add_preset(store: GuardrailStore, name: str, adset_id: str,
               thr: str = "") -> Rule | None:
    """Arm a ready-made automation strategy. Never raises."""
    try:
        preset = RULE_PRESETS.get((name or "").strip().lower())
        if not preset or not (adset_id or "").strip() or store is None:
            return None
        text = preset["template"].format(
            adset=adset_id.strip(), thr=(thr or "").strip() or preset["default_thr"])
        rule = parse_rule(text)
        if rule is None:
            return None
        return store.add_rule(rule)
    except Exception:
        return None


# ── metric history + creative-fatigue detection ───────────────────────────

def record_metrics(store: GuardrailStore, adset_id: str,
                   metrics: dict | None) -> bool:
    """Snapshot an ad set's metrics for fatigue detection. Never raises."""
    try:
        if store is None or getattr(store, "_db", None) is None:
            return False
        adset_id = (adset_id or "").strip()
        if not adset_id:
            return False
        import json as _json
        store._db.execute(
            "INSERT INTO metric_history (adset_id, ts, metrics_json)"
            " VALUES (?, ?, ?)",
            (adset_id, _now(), _json.dumps(dict(metrics or {}))))
        store._db.execute("DELETE FROM metric_history WHERE ts < ?",
                          (_now() - 60 * 86400.0,))
        store._db.commit()
        return True
    except Exception:
        return False


def fatigue_signal(store: GuardrailStore, adset_id: str,
                   window_days: int = 7) -> dict:
    """Creative-fatigue check: CTR down ≥30% (1st→2nd half of window) with
    frequency rising. Never raises."""
    out: dict = {"fatigued": False, "ctr_drop_pct": 0.0,
                 "freq_trend": "flat", "reason": "not enough history"}
    try:
        if store is None or getattr(store, "_db", None) is None:
            return out
        rows = store._db.execute(
            "SELECT metrics_json FROM metric_history WHERE adset_id = ?"
            " AND ts >= ? ORDER BY ts",
            ((adset_id or "").strip(), _now() - max(2, int(window_days or 7)) * 86400.0)
        ).fetchall()
        import json as _json
        pts: list[tuple[float, float]] = []
        for r in rows:
            try:
                m = _json.loads(r["metrics_json"] or "{}")
                pts.append((float(m.get("ctr", 0) or 0),
                            float(m.get("frequency", 0) or 0)))
            except Exception:
                continue
        if len(pts) < 4:
            return out
        half = len(pts) // 2
        c1 = sum(p[0] for p in pts[:half]) / half
        c2 = sum(p[0] for p in pts[half:]) / (len(pts) - half)
        f1 = sum(p[1] for p in pts[:half]) / half
        f2 = sum(p[1] for p in pts[half:]) / (len(pts) - half)
        drop = ((c1 - c2) / c1 * 100.0) if c1 > 0 else 0.0
        out["ctr_drop_pct"] = round(drop, 1)
        out["freq_trend"] = ("rising" if f2 > f1 * 1.1
                             else "falling" if f2 < f1 * 0.9 else "flat")
        if drop >= 30 and out["freq_trend"] == "rising":
            out["fatigued"] = True
            out["reason"] = (f"CTR down {drop:.0f}% with frequency rising — "
                             "creative fatigue: pause or refresh creative")
        elif drop >= 30:
            out["reason"] = (f"CTR down {drop:.0f}% but frequency flat — "
                             "check tracking/landing before touching budgets")
        else:
            out["reason"] = "no fatigue signal"
        return out
    except Exception:
        return out


# ── dry-run preview ───────────────────────────────────────────────────────

def preview(store: GuardrailStore,
            metrics_fn: MetricsFn | None = None) -> list[dict]:
    """'What would fire right now' — no streak changes, no firing, no audit.
    Never raises."""
    out: list[dict] = []
    try:
        if store is None:
            return out
        for rule in store.list_rules(active_only=True):
            try:
                info: dict = {"rule_id": rule.rule_id,
                              "adset_id": rule.adset_id,
                              "action": rule.action,
                              "would_fire": False, "reason": "",
                              "streak": f"{rule.streak or 0}/{rule.days}d"}
                if metrics_fn is None:
                    info["reason"] = "no metric source"
                    out.append(info)
                    continue
                metrics = metrics_fn(rule.adset_id) or {}
                conds = _conditions_of(rule)
                bits: list[str] = []
                results: list[bool] = []
                for cond in conds:
                    v = metrics.get(cond.get("metric", ""))
                    try:
                        v = float(v) if v is not None else None
                    except Exception:
                        v = None
                    holds = (v is not None and _condition_holds(
                        v, cond.get("op", ">"), float(cond.get("threshold", 0))))
                    results.append(bool(holds))
                    bits.append(
                        f"{cond.get('metric', '').upper()} "
                        f"{_fmt_cond_value(cond, v) if v is not None else '?'} "
                        f"{cond.get('op', '')} {_fmt_cond(cond)} "
                        f"{'✓' if holds else '✗'}")
                holds_all = (any(results) if (rule.cond_op or "and") == "or"
                             else all(results))
                if not results:
                    info["reason"] = "no conditions"
                elif not holds_all:
                    info["reason"] = "conditions not met: " + f" {rule.cond_op} ".join(bits)
                elif (rule.streak or 0) + 1 < rule.days:
                    info["reason"] = ("holding — streak would be "
                                      f"{(rule.streak or 0) + 1}/{rule.days}d: "
                                      + f" {rule.cond_op} ".join(bits))
                elif rule.last_fired_at and (_now() - rule.last_fired_at) < rule.cooldown_s:
                    info["reason"] = "cooling down"
                else:
                    info["would_fire"] = True
                    info["reason"] = f"WOULD FIRE: {rule.action} — " + f" {rule.cond_op} ".join(bits)
                out.append(info)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return out

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


def _fmt_cond(cond: dict) -> str:
    try:
        metric = cond.get("metric", "")
        thr = float(cond.get("threshold", 0))
        if metric in _MONEY_METRICS:
            return format_naira(thr)
        return str(thr).rstrip("0").rstrip(".")
    except Exception:
        return "?"


def _fmt_cond_value(cond: dict, value: float) -> str:
    try:
        if cond.get("metric") in _MONEY_METRICS:
            return format_naira(value)
        return f"{value:.2f}"
    except Exception:
        return "?"


def _conditions_of(rule: Rule) -> list[dict]:
    """Compound conditions, falling back to the legacy single condition."""
    try:
        if rule.conditions:
            return [c for c in rule.conditions if isinstance(c, dict) and c.get("metric")]
        return [{"metric": rule.metric, "op": rule.op,
                 "threshold": rule.threshold, "threshold_raw": rule.threshold_raw}]
    except Exception:
        return []


def _evaluate_rule(store: GuardrailStore, rule: Rule,
                   metrics_fn: MetricsFn | None,
                   executor_fn: ExecutorFn | None) -> FiredAction | None:
    try:
        if metrics_fn is None:
            return None  # no metric source — nothing to evaluate
        metrics = metrics_fn(rule.adset_id) or {}

        # Minimum-evidence gate: don't judge a campaign still in learning.
        if rule.min_spend_kobo:
            try:
                spend_now = float(metrics.get("spend", 0) or 0)
            except Exception:
                spend_now = 0
            need = rule.min_spend_kobo / (100.0 if rule.min_spend_kobo > 1000 else 1.0)
            # metrics spend may be kobo or naira — accept either scale.
            if spend_now < rule.min_spend_kobo and spend_now < need:
                return None  # hold streak, no reset: evidence is accruing

        conds = _conditions_of(rule)
        results: list[tuple[dict, float | None, bool]] = []
        for cond in conds:
            value = metrics.get(cond.get("metric", ""))
            if value is None:
                results.append((cond, None, False))
                continue
            try:
                value = float(value)
            except Exception:
                results.append((cond, None, False))
                continue
            results.append((cond, value,
                            _condition_holds(value, cond.get("op", ">"),
                                             float(cond.get("threshold", 0)))))
        if any(v is None for _, v, _ in results):
            store.update_streak(rule.rule_id, 0)
            return None
        holds = (any(h for _, _, h in results) if (rule.cond_op or "and") == "or"
                 else all(h for _, _, h in results))
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

        cond_txt = (f" {rule.cond_op} ".join(
            f"{c.get('metric', '').upper()} {c.get('op', '')} {_fmt_cond(c)}"
            for c in conds))
        val_txt = ", ".join(
            f"{c.get('metric', '').upper()} {_fmt_cond_value(c, v)}"
            for c, v, _ in results)
        label = f"{rule.action} — {cond_txt} (now {val_txt}, {streak}d streak)"
        detail = (f"rule {rule.rule_id}: {val_txt} vs {cond_txt} "
                  f"on {rule.adset_id}")

        # Alert actions notify only — no money moves, no mandate needed.
        if rule.action == "alert":
            executed = True
            if executor_fn is not None:
                try:
                    executed = bool(executor_fn(
                        "alert", rule.adset_id,
                        {"conditions": conds, "values": {c.get("metric"): v
                                                        for c, v, _ in results}}))
                except Exception as e:  # noqa: BLE001
                    executed = False
                    fa = FiredAction(rule.rule_id, rule.adset_id, "alert",
                                     detail, False,
                                     reason=f"alert dispatch failed: {e}", ts=now)
                    store.mark_fired(rule.rule_id, label + " [alert-failed]")
                    return fa
            store.mark_fired(rule.rule_id, label)
            return FiredAction(rule.rule_id, rule.adset_id, "alert", detail,
                               executed,
                               reason="" if executed else "no alert channel",
                               ts=now)

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
            first = conds[0]
            params: dict[str, Any] = {"metric": first.get("metric"),
                                      "value": results[0][1],
                                      "threshold": first.get("threshold"),
                                      "pct": rule.action_param,
                                      "conditions": conds}
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
        if action not in ("pause", "scale", "label", "alert"):
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
        conds = _conditions_of(rule)
        cond_txt = f" {(rule.cond_op or 'and')} ".join(
            f"{c.get('metric', '').upper()} {c.get('op', '')} {_fmt_cond(c)}"
            for c in conds)
        when = (f" for {rule.days}d" if rule.days > 1 else "")
        act = rule.action
        if rule.action == "scale":
            act = f"scale +{rule.action_param:g}%"
        if rule.action in ("label", "alert") and rule.action_text:
            act = f"{rule.action} '{rule.action_text}'"
        ms = (f" · min-spend {format_naira(rule.min_spend_kobo)}"
              if rule.min_spend_kobo else "")
        streak = (f" · streak {rule.streak or 0}/{rule.days}d"
                  if rule.streak else "")
        state = "" if rule.active else " [off]"
        lbl = f" — {rule.label}" if rule.label else ""
        return (f"• `{rule.rule_id}` {act} if {cond_txt}{when} "
                f"on `{rule.adset_id}`{ms}{streak}{state}{lbl}")
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
            "  /guardrails add pause if cpa > ₦50k and frequency > 4 for 3 days on adset1\n"
            "  /guardrails add alert if ctr < 0.5 for 5 days on adset123 — notify only\n"
            "  /guardrails preset <name> on <adset> [threshold] — ready-made strategies\n"
            "    names: cpa_kill · scale_winners · fatigue_watch · frequency_cap · budget_guard\n"
            "  /guardrails list — active rules\n"
            "  /guardrails preview — dry run: what would fire right now\n"
            "  /guardrails run — evaluate now\n"
            "  /guardrails fatigue <adset> — creative-fatigue check\n"
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

        if low.startswith("preset"):
            # /guardrails preset <name> on <adset> [threshold]
            m = re.match(r"(?P<name>\w+)\s+on\s+(?P<adset>[\w\-]+)(?:\s+(?P<thr>\S+))?\s*$",
                         rest[6:].strip(), re.IGNORECASE)
            if not m:
                names = " · ".join(sorted(RULE_PRESETS))
                return (f"usage: /guardrails preset <name> on <adset> [threshold]\n"
                        f"names: {names}")
            rule = add_preset(store, m.group("name"), m.group("adset"),
                              m.group("thr") or "")
            if rule is None:
                return "couldn't arm that preset — check the name/adset."
            return ("🛡️ preset armed:\n" + format_rule(rule) +
                    "\nspend actions still need an active adspend mandate to fire.")

        if low.startswith("preview"):
            rows = preview(store)
            if not rows:
                return "no active rules to preview."
            lines = ["🔍 guardrail dry run — what would fire right now:"]
            for r in rows:
                icon = "🔥" if r["would_fire"] else "▫️"
                lines.append(f"{icon} `{r['rule_id']}` {r['action']} on "
                             f"`{r['adset_id']}` [{r['streak']}] — {r['reason'][:110]}")
            lines.append("nothing fired, no streaks touched, nothing audited.")
            return "\n".join(lines)

        if low.startswith("fatigue"):
            adset = rest[7:].strip()
            if not adset:
                return "usage: /guardrails fatigue <adset>"
            sig = fatigue_signal(store, adset)
            icon = "🚨" if sig["fatigued"] else "✅"
            return (f"{icon} fatigue check — `{adset}`\n"
                    f"CTR change: {sig['ctr_drop_pct']:+.0f}% · "
                    f"frequency: {sig['freq_trend']}\n"
                    f"{sig['reason']}")

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
