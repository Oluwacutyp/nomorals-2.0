"""Pattern study — plain-words market patterns, proven against history,
watched live.

A pattern is DATA, never a hardcoded list: a condition over named market
facts plus an expectation, both written as ``<weekday>.<field> <op>
<weekday>.<field>``. The organ parses the plain-words rule, backtests it
against the desk's daily-bar ledger (fetching fresh bars when the ledger
is thin), and watches it: when the condition triggers, the expectation
and the historical hit rate are noted and journaled; when the outcome
weekday lands, the outcome is journaled and the running score updates.

The first pattern is built in — the Friday/Thursday rule from the video
the owner shared (2026-11-21): a Friday high below Thursday's high means
Monday revisits Friday's low.

Patterns, scores and the trigger/outcome journal live in the Devon home
(``<home>/patterns/``), never in host process state and never under a
host-specific path — the thinking is portable data: copy the home, open
the store, and the watch resumes where it left off.

Analysis only — the organ studies and watches; it never trades, never
places orders, never touches GoldenSwing.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ...finance import trading_journal
from .collectors import SYMBOL_MARKETS, fetch_rows
from .failures import record_failure
from .store import Candle, CandleStore

__all__ = [
    "Pattern", "PatternStore", "parse_rule", "backtest", "study_rule",
    "list_patterns", "set_watching", "check_watches", "pattern_status",
    "remove_pattern", "render_study", "render_pattern_list",
    "render_check", "render_status", "patterns_dir",
    "BUILTIN_ID", "BUILTIN_RULE",
]

# ── rule language ────────────────────────────────────────────────────

WEEKDAYS: dict[str, int] = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
}
WEEKDAY_LABEL: dict[int, str] = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday",
}
FIELDS = ("open", "high", "low", "close")
OPS: dict[str, str] = {
    "<": "<", "<=": "<=", ">": ">", ">=": ">=",
    "==": "==", "=": "==",
}
_OP_WORD = {"<": "lt", "<=": "lte", ">": "gt", ">=": "gte", "==": "eq"}
_WEEKDAY_RE = (
    r"(?:monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thur|"
    r"thurs|friday|fri)"
)
_FIELD_RE = r"(?:open|high|low|close)"
_OP_RE = (
    r"(?:not as high as|not as low as|is not as high as|is not as low as"
    r"|<=|>=|==|=|<|>)"
)

BUILTIN_ID = "fri_thu_low_revisit"
BUILTIN_RULE = (
    "Friday high < Thursday high => Monday low <= Friday low for XAUUSD"
)
BUILTIN_NAME = "Friday high below Thursday high, Monday revisits Friday low"

RULE_SYNTAX = (
    "rule syntax: <Weekday> <high|low|open|close> < | <= | > | >= | == > "
    "<Weekday> <high|low|open|close> => <Weekday> <high|low|open|close> "
    "< | <= | > | >= | == > <Weekday> <high|low|open|close> "
    "[for <SYMBOL>] — e.g. 'Friday high < Thursday high => "
    "Monday low <= Friday low for XAUUSD'. "
    "Plain words work too: 'not as high as' means <, and 'will be visited "
    "on Monday' means the Monday low reaches the Friday low."
)


def _op_from_text(op_text: str, field: str = "") -> str:
    """Map an operator phrase to a canonical op."""
    t = (op_text or "").strip().lower()
    if t in ("not as high as", "is not as high as"):
        return "<"
    if t in ("not as low as", "is not as low as"):
        return ">"
    return OPS.get(t, "")


_FACT_A = re.compile(
    rf"(?P<f>{_FIELD_RE})\b.*?\b(?P<w>{_WEEKDAY_RE})\b")
_FACT_B = re.compile(
    rf"(?P<w>{_WEEKDAY_RE})\b.*?\b(?P<f>{_FIELD_RE})\b")


def _match_fact(t: str, pos: int = 0) -> dict[str, Any] | None:
    """The earliest-starting weekday+field fact at or after ``pos``.

    Both word orders are real: "Friday high" (compact syntax) and
    "the high during the Friday" (plain words). The earliest start
    wins, so a fact never spans across the operator into the next one.
    """
    best = None
    for rx in (_FACT_A, _FACT_B):
        m = rx.search(t, pos)
        if m and (best is None or m.start() < best.start()
                  or (m.start() == best.start()
                      and m.end() < best.end())):
            best = m
    if best is None:
        return None
    return {"weekday": WEEKDAYS[best.group("w")],
            "field": best.group("f"),
            "start": best.start(), "end": best.end()}


def _parse_comparison(text: str) -> dict[str, Any] | None:
    """Parse one ``<weekday> <field> <op> <weekday> <field>`` comparison.

    Tries the compact shape first, then the "visited" shape (``<field>
    ... <weekday> ... visited ... <weekday>``) used for expectations.
    Returns the comparison dict or None when the text carries no
    comparison the organ can check.
    """
    t = (text or "").lower()
    first = _match_fact(t)
    if first is not None:
        op_m = re.search(_OP_RE, t[first["end"]:])
        if op_m:
            op = _op_from_text(op_m.group(0), first["field"])
            second = _match_fact(
                t, first["end"] + op_m.end())
            if op and second is not None:
                return {
                    "weekday": first["weekday"],
                    "field": first["field"],
                    "op": op,
                    "other_weekday": second["weekday"],
                    "other_field": second["field"],
                }
    # "visited" shape: "<field> ... <weekday> ... visited ... <weekday>".
    # The field that matters is the one NEAREST the word "visited" —
    # the text before it may mention other fields ("the odds are
    # overwhelmingly HIGH that whatever LOW you made on Friday ...").
    vm = re.search(
        r"\b(?:visited|revisited|touched|reached)\b"
        rf".*?\b(?P<w_to>{_WEEKDAY_RE})\b",
        t)
    if vm:
        before = t[:vm.start()]
        field_ms = list(re.finditer(r"\b(?P<f>high|low)\b", before))
        if field_ms:
            f_m = field_ms[-1]
            w_m = re.search(
                rf"\b(?P<w>{_WEEKDAY_RE})\b", before[f_m.end():])
            if w_m:
                f = f_m.group("f")
                return {
                    "weekday": WEEKDAYS[vm.group("w_to")],
                    "field": f,
                    "op": "<=" if f == "low" else ">=",
                    "other_weekday": WEEKDAYS[w_m.group("w")],
                    "other_field": f,
                }
    return None


def _split_rule(text: str) -> tuple[str, str] | None:
    """Split a rule into (condition, expectation) on =>, -> or 'then'."""
    t = (text or "").strip()
    for sep in ("=>", "->"):
        if sep in t:
            cond, _, exp = t.partition(sep)
            return cond.strip(), exp.strip()
    m = re.search(r"\bthen\b", t, flags=re.IGNORECASE)
    if m:
        return t[:m.start()].strip(), t[m.end():].strip()
    return None


def parse_rule(text: str) -> dict[str, Any]:
    """Parse plain-words rule text into a structured pattern dict.

    Returns ``{"ok": True, ...pattern fields...}`` or
    ``{"ok": False, "error": ...}`` — an unparseable rule is refused
    with the syntax, never guessed.
    """
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "error": "empty rule — " + RULE_SYNTAX}
    symbol = "XAUUSD"
    m = re.search(r"\bfor\s+([A-Za-z]{2,12})\s*$", raw, flags=re.IGNORECASE)
    body = raw
    if m:
        symbol = m.group(1).upper()
        body = raw[:m.start()].strip()
    split = _split_rule(body)
    if split is None:
        return {"ok": False,
                "error": "no condition => expectation found — " + RULE_SYNTAX}
    cond_text, exp_text = split
    cond = _parse_comparison(cond_text)
    exp = _parse_comparison(exp_text)
    if cond is None or exp is None:
        return {"ok": False,
                "error": "could not read the condition or the expectation "
                         "as weekday highs/lows — " + RULE_SYNTAX}
    return {
        "ok": True,
        "symbol": symbol,
        "market": market_for_symbol(symbol),
        "condition": cond,
        "expectation": exp,
        "rule_text": raw,
    }


def market_for_symbol(symbol: str) -> str:
    """Price-class market for a symbol (same defaults as the desk)."""
    s = (symbol or "").strip().upper()
    if s in SYMBOL_MARKETS:
        return SYMBOL_MARKETS[s]
    if s.startswith(("XAU", "XAG", "XPT", "XPD")):
        return "forex"
    return "crypto"


def comparison_text(cmp: dict[str, Any]) -> str:
    """Render one comparison back into plain words."""
    return (
        f"{WEEKDAY_LABEL[cmp['weekday']]} {cmp['field']} "
        f"{cmp['op']} {WEEKDAY_LABEL[cmp['other_weekday']]} "
        f"{cmp['other_field']}"
    )


def pattern_slug(condition: dict[str, Any],
                 expectation: dict[str, Any]) -> str:
    """A stable id derived from the rule itself."""
    def _part(cmp: dict[str, Any]) -> str:
        return (
            f"{WEEKDAY_LABEL[cmp['weekday']][:3].lower()}_"
            f"{cmp['field']}_{_OP_WORD[cmp['op']]}_"
            f"{WEEKDAY_LABEL[cmp['other_weekday']][:3].lower()}_"
            f"{cmp['other_field']}"
        )
    return f"{_part(condition)}__{_part(expectation)}"


# ── durable store (portable: lives in the Devon home) ────────────────

def patterns_dir(settings: Any = None) -> str:
    """Patterns home: ``NOMORALS_PATTERNS_DIR`` or ``<home>/patterns``.

    The home comes from the runtime settings when present, else
    ``~/.nomorals`` — the same home the finance journal uses. No
    host-specific path is ever stored inside the files themselves.
    """
    override = os.environ.get("NOMORALS_PATTERNS_DIR", "").strip()
    if override:
        return override
    home = getattr(settings, "home_path", None) if settings is not None else None
    base = Path(home) if home else Path.home() / ".nomorals"
    return str(base / "patterns")


@dataclass
class Pattern:
    """One studied pattern: a rule, its backtest score, its watch state."""

    id: str
    name: str
    symbol: str
    market: str
    condition: dict[str, Any]
    expectation: dict[str, Any]
    rule_text: str
    created_at: float = 0.0
    watching: bool = True
    score: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["describe"] = self.describe()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pattern:
        known = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)  # type: ignore[arg-type]

    def describe(self) -> str:
        return (
            f"if {comparison_text(self.condition)}, expect "
            f"{comparison_text(self.expectation)}"
        )


class PatternStore:
    """Durable pattern state: definitions + trigger/outcome journal.

    ``patterns.json`` holds the definitions (atomic write); ``events.jsonl``
    is the append-only trigger/outcome journal. Both live under the Devon
    home, so the state moves with her and reloads on any box.
    """

    def __init__(self, settings: Any = None, patterns_dir_: str = ""):
        self.dir = patterns_dir_ or patterns_dir(settings)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "patterns.json")
        self.events_path = os.path.join(self.dir, "events.jsonl")

    # ── definitions ────────────────────────────────────────────────

    def patterns(self) -> list[Pattern]:
        if not os.path.exists(self.path):
            return []
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return []
        out: list[Pattern] = []
        for d in data if isinstance(data, list) else []:
            try:
                out.append(Pattern.from_dict(d))
            except (TypeError, ValueError):
                continue
        return out

    def get(self, pattern_id: str) -> Pattern | None:
        for p in self.patterns():
            if p.id == pattern_id:
                return p
        return None

    def save(self, pattern: Pattern) -> None:
        """Upsert one pattern; atomic write (tmp + replace)."""
        rows = [p.to_dict() for p in self.patterns()
                if p.id != pattern.id]
        rows.append(pattern.to_dict())
        tmp = self.path + ".tmp"
        Path(tmp).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def remove(self, pattern_id: str) -> bool:
        rows = [p.to_dict() for p in self.patterns()
                if p.id != pattern_id]
        if len(rows) == len(self.patterns()):
            return False
        tmp = self.path + ".tmp"
        Path(tmp).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        return True

    def seed_builtin(self) -> Pattern:
        """The video's Friday/Thursday rule, present from the first use."""
        existing = self.get(BUILTIN_ID)
        if existing is not None:
            return existing
        parsed = parse_rule(BUILTIN_RULE)
        pattern = Pattern(
            id=BUILTIN_ID, name=BUILTIN_NAME,
            symbol=parsed["symbol"], market=parsed["market"],
            condition=parsed["condition"],
            expectation=parsed["expectation"],
            rule_text=BUILTIN_RULE, created_at=time.time(), watching=True,
        )
        self.save(pattern)
        return pattern

    # ── events ─────────────────────────────────────────────────────

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        rec = {"ts": time.time()}
        rec.update(event)
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def events(self, pattern_id: str = "",
               kind: str = "") -> list[dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return []
        out: list[dict[str, Any]] = []
        for line in Path(self.events_path).read_text(
                encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if pattern_id and rec.get("pattern_id") != pattern_id:
                continue
            if kind and rec.get("event") != kind:
                continue
            out.append(rec)
        return out

    def pending_triggers(self, pattern_id: str = "") -> list[dict[str, Any]]:
        """Trigger events with no outcome yet, oldest first."""
        outcomes = {(r.get("pattern_id"), r.get("week"))
                    for r in self.events(kind="pattern_outcome")}
        return [r for r in self.events(pattern_id, kind="pattern_trigger")
                if (r.get("pattern_id"), r.get("week")) not in outcomes]


# ── backtest ─────────────────────────────────────────────────────────

def _week_monday(ts: float) -> Any:
    dt = datetime.fromtimestamp(ts, tz=UTC)
    return (dt - timedelta(days=dt.weekday())).date()


def _holds(week_bars: dict[int, Any], cmp: dict[str, Any]) -> bool | None:
    """Evaluate one comparison over one week's bars.

    None when a weekday bar is missing (a holiday week is not an
    instance — it is skipped, never counted against the pattern).
    """
    a = week_bars.get(cmp["weekday"])
    b = week_bars.get(cmp["other_weekday"])
    if a is None or b is None:
        return None
    va = a.get(cmp["field"])
    vb = b.get(cmp["field"])
    if va is None or vb is None:
        return None
    op = cmp["op"]
    if op == "<":
        return va < vb
    if op == "<=":
        return va <= vb
    if op == ">":
        return va > vb
    if op == ">=":
        return va >= vb
    return va == vb


def _outcome_monday(trigger_monday: Any, pattern: Pattern) -> Any:
    """Which week's bars carry the expectation's outcome.

    The outcome weekday is the expectation's ``weekday``. When it falls
    later in the week than every weekday the condition mentions, the
    expectation resolves in the trigger week itself (e.g. a Wednesday
    expectation after a Tuesday condition); otherwise it resolves in
    the NEXT week (the video rule: Monday follows Friday's week).
    """
    cond_days = {pattern.condition["weekday"],
                 pattern.condition["other_weekday"]}
    if pattern.expectation["weekday"] > max(cond_days):
        return trigger_monday
    return trigger_monday + timedelta(days=7)


def _holds_expectation(trigger_week: dict[int, Any],
                       outcome_week: dict[int, Any],
                       cmp: dict[str, Any]) -> bool | None:
    """Evaluate the expectation across the trigger and outcome weeks.

    The comparison's outcome-weekday side is read from the outcome
    week; the other side is read from the trigger week (the Friday low
    the Monday is expected to revisit). None when either bar is
    missing — a not-yet-landed outcome is pending, never a miss.
    """
    left = outcome_week.get(cmp["weekday"])
    other = trigger_week.get(cmp["other_weekday"])
    if left is None or other is None:
        return None
    va = left.get(cmp["field"])
    vb = other.get(cmp["other_field"])
    if va is None or vb is None:
        return None
    op = cmp["op"]
    if op == "<":
        return va < vb
    if op == "<=":
        return va <= vb
    if op == ">":
        return va > vb
    if op == ">=":
        return va >= vb
    return va == vb


def backtest(daily_bars: list[Candle], pattern: Pattern) -> dict[str, Any]:
    """Score one pattern against daily bars.

    An instance is a week where the condition's two weekday bars exist.
    The expectation is read off the NEXT week's outcome-weekday bar;
    a trigger whose outcome bar has not landed yet is pending, never a
    miss. Zero resolved instances → UNPROVEN, with the reason.
    """
    weeks: dict[Any, dict[int, dict[str, Any]]] = {}
    for bar in daily_bars or []:
        if bar.timeframe != "1d":
            continue
        dt = datetime.fromtimestamp(bar.ts, tz=UTC)
        if dt.weekday() > 4:
            continue  # weekend bars carry no weekday pattern
        monday = _week_monday(bar.ts)
        weeks.setdefault(monday, {})[dt.weekday()] = {
            "open": bar.open, "high": bar.high,
            "low": bar.low, "close": bar.close,
        }
    evaluated = triggered = resolved = hits = pending = 0
    span_start = span_end = None
    for monday in sorted(weeks):
        wk = weeks[monday]
        cond = _holds(wk, pattern.condition)
        if cond is None:
            continue
        evaluated += 1
        span_start = span_start or monday.isoformat()
        span_end = monday.isoformat()
        if not cond:
            continue
        triggered += 1
        outcome_monday = _outcome_monday(monday, pattern)
        outcome_week = weeks.get(outcome_monday)
        if outcome_week is None:
            pending += 1
            continue
        held = _holds_expectation(wk, outcome_week, pattern.expectation)
        if held is None:
            pending += 1
            continue
        resolved += 1
        if held:
            hits += 1
    rate = (hits / resolved) if resolved else None
    return {
        "pattern_id": pattern.id,
        "symbol": pattern.symbol,
        "evaluated": evaluated,
        "triggered": triggered,
        "resolved": resolved,
        "hits": hits,
        "pending": pending,
        "hit_rate": rate,
        "span_start": span_start,
        "span_end": span_end,
        "bars_used": len([b for b in daily_bars or []
                          if b.timeframe == "1d"]),
        "verdict": "proven" if resolved else "unproven",
    }


def ensure_bars(symbol: str, market: str, store: CandleStore, *,
                fetch: Any = None) -> dict[str, Any]:
    """Make sure the ledger holds enough daily bars to study.

    Returns {"bars": n, "added": m, "fetch": "ledger"|"live"|"failed",
    "error": ...}. The ledger is the first source; a thin ledger
    triggers one live fetch (appended, so the study is reproducible).
    A failed fetch is reported, never raised — the study then runs on
    whatever the ledger holds.
    """
    ledger = store.bars(symbol, "1d")
    if len(ledger) >= 20:
        return {"bars": len(ledger), "added": 0, "fetch": "ledger"}
    fetch = fetch or fetch_rows
    try:
        rows = fetch(symbol, market, "1d", 500)
    except Exception as exc:  # noqa: BLE001 - a dead feed is data
        record_failure("patterns.ensure_bars", exc)
        return {"bars": len(ledger), "added": 0, "fetch": "failed",
                "error": f"{type(exc).__name__}: {exc}"}
    candles = []
    for r in rows or []:
        try:
            candles.append(Candle(
                symbol=r.get("symbol", symbol), timeframe="1d",
                ts=float(r.get("ts") or 0.0),
                open=float(r.get("open") or 0.0),
                high=float(r.get("high") or 0.0),
                low=float(r.get("low") or 0.0),
                close=float(r.get("close") or 0.0),
                volume=float(r.get("volume") or 0.0),
                source="market_data.get_ohlcv"))
        except (TypeError, ValueError):
            continue
    landed = store.append(candles)
    return {"bars": len(store.bars(symbol, "1d")),
            "added": landed["added"], "fetch": "live"}


# ── organ operations (never raise — every failure is data) ───────────

def _organ(settings: Any = None, store: PatternStore | None = None,
           candle_store: CandleStore | None = None
           ) -> tuple[PatternStore, CandleStore]:
    pstore = store or PatternStore(settings=settings)
    cstore = candle_store or CandleStore()
    pstore.seed_builtin()
    return pstore, cstore


def study_rule(rule_text: str, *, settings: Any = None,
               store: PatternStore | None = None,
               candle_store: CandleStore | None = None,
               fetch: Any = None) -> dict[str, Any]:
    """Study one plain-words rule: parse it, backtest it, store it."""
    pstore, cstore = _organ(settings, store, candle_store)
    parsed = parse_rule(rule_text)
    if not parsed.get("ok"):
        return {"ok": False, "error": parsed["error"]}
    pid = pattern_slug(parsed["condition"], parsed["expectation"])
    builtin = parse_rule(BUILTIN_RULE)
    if (parsed["symbol"] == builtin["symbol"]
            and parsed["condition"] == builtin["condition"]
            and parsed["expectation"] == builtin["expectation"]):
        pid = BUILTIN_ID
    existing = pstore.get(pid)
    pattern = existing or Pattern(
        id=pid, name=comparison_text(parsed["condition"]) + " → "
        + comparison_text(parsed["expectation"]),
        symbol=parsed["symbol"], market=parsed["market"],
        condition=parsed["condition"], expectation=parsed["expectation"],
        rule_text=parsed["rule_text"], created_at=time.time(),
        watching=True)
    ensured = ensure_bars(pattern.symbol, pattern.market, cstore,
                          fetch=fetch)
    bars = cstore.bars(pattern.symbol, "1d")
    score = backtest(bars, pattern)
    score["source"] = ensured["fetch"]
    if ensured["fetch"] == "failed":
        score["fetch_error"] = ensured.get("error")
    pattern.score = score
    pstore.save(pattern)
    # The journal never kills a study.
    with contextlib.suppress(Exception):
        trading_journal.append_event(
            "pattern_study", settings=settings, pattern_id=pattern.id,
            symbol=pattern.symbol, rule=pattern.rule_text,
            verdict=score["verdict"], hit_rate=score["hit_rate"],
            resolved=score["resolved"], triggered=score["triggered"])
    return {"ok": True, "pattern": pattern.to_dict(), "score": score,
            "bars": ensured}


def list_patterns(*, settings: Any = None,
                  store: PatternStore | None = None) -> dict[str, Any]:
    """Every studied pattern with its last score and watch state."""
    pstore, _ = _organ(settings, store, None)
    return {"ok": True,
            "patterns": [p.to_dict() for p in pstore.patterns()]}


def set_watching(pattern_id: str, watching: bool, *,
                 settings: Any = None,
                 store: PatternStore | None = None) -> dict[str, Any]:
    """Arm or disarm a pattern's live watch."""
    pstore, _ = _organ(settings, store, None)
    pattern = pstore.get((pattern_id or "").strip())
    if pattern is None:
        return {"ok": False,
                "error": f"no studied pattern {pattern_id!r} — "
                         "see /pattern list"}
    pattern.watching = bool(watching)
    pstore.save(pattern)
    return {"ok": True, "pattern": pattern.to_dict()}


def remove_pattern(pattern_id: str, *, settings: Any = None,
                   store: PatternStore | None = None) -> dict[str, Any]:
    """Forget a studied pattern (its journal trail stays)."""
    pstore, _ = _organ(settings, store, None)
    pid = (pattern_id or "").strip()
    if pid == BUILTIN_ID:
        return {"ok": False,
                "error": "the built-in Friday/Thursday rule cannot be "
                         "removed — /pattern unwatch it to silence it"}
    if not pstore.remove(pid):
        return {"ok": False, "error": f"no studied pattern {pid!r}"}
    return {"ok": True, "removed": pid}


def pattern_status(pattern_id: str, *, settings: Any = None,
                   store: PatternStore | None = None,
                   candle_store: CandleStore | None = None,
                   fetch: Any = None) -> dict[str, Any]:
    """One pattern's stored score, refreshed against the ledger."""
    pstore, cstore = _organ(settings, store, candle_store)
    pattern = pstore.get((pattern_id or "").strip())
    if pattern is None:
        return {"ok": False,
                "error": f"no studied pattern {pattern_id!r} — "
                         "see /pattern list"}
    ensured = ensure_bars(pattern.symbol, pattern.market, cstore,
                          fetch=fetch)
    score = backtest(cstore.bars(pattern.symbol, "1d"), pattern)
    score["source"] = ensured["fetch"]
    pattern.score = score
    pstore.save(pattern)
    pending = pstore.pending_triggers(pattern.id)
    return {"ok": True, "pattern": pattern.to_dict(), "score": score,
            "pending": pending, "bars": ensured}


def check_watches(*, settings: Any = None,
                  store: PatternStore | None = None,
                  candle_store: CandleStore | None = None,
                  fetch: Any = None) -> dict[str, Any]:
    """One live-watch sweep over every armed pattern.

    For each pattern: fresh bars in, latest closed week evaluated. A
    condition that held on the latest Friday with no trigger for that
    week records a trigger (expectation + historical hit rate). A
    pending trigger whose outcome weekday has landed records the
    outcome and refreshes the running score. Idempotent per week.
    """
    pstore, cstore = _organ(settings, store, candle_store)
    notices: list[dict[str, Any]] = []
    errors: list[str] = []
    for pattern in pstore.patterns():
        if not pattern.watching:
            continue
        try:
            ensured = ensure_bars(pattern.symbol, pattern.market, cstore,
                                  fetch=fetch)
            bars = cstore.bars(pattern.symbol, "1d")
            score = backtest(bars, pattern)
            score["source"] = ensured["fetch"]
            pattern.score = score
            pstore.save(pattern)
            notices.extend(
                _check_one(pstore, pattern, bars, score, settings))
        except Exception as exc:  # noqa: BLE001 - one pattern never kills the sweep
            record_failure("patterns.check_watches", exc)
            errors.append(f"{pattern.id}: {type(exc).__name__}: {exc}")
    return {"ok": True, "notices": notices, "errors": errors}


def _check_one(pstore: PatternStore, pattern: Pattern,
               bars: list[Candle], score: dict[str, Any],
               settings: Any) -> list[dict[str, Any]]:
    notices: list[dict[str, Any]] = []
    weeks: dict[Any, dict[int, dict[str, Any]]] = {}
    for bar in bars:
        if bar.timeframe != "1d":
            continue
        dt = datetime.fromtimestamp(bar.ts, tz=UTC)
        if dt.weekday() > 4:
            continue
        weeks.setdefault(_week_monday(bar.ts), {})[dt.weekday()] = {
            "open": bar.open, "high": bar.high,
            "low": bar.low, "close": bar.close,
        }
    if not weeks:
        return notices
    # ── resolve pending triggers first (outcomes land before new triggers)
    for trig in pstore.pending_triggers(pattern.id):
        week_key = trig.get("week")
        try:
            trigger_monday = datetime.strptime(
                week_key, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        trigger_week = weeks.get(trigger_monday)
        if trigger_week is None:
            continue
        outcome_monday = _outcome_monday(trigger_monday, pattern)
        outcome_week = weeks.get(outcome_monday)
        if outcome_week is None:
            continue
        held = _holds_expectation(trigger_week, outcome_week,
                                  pattern.expectation)
        if held is None:
            continue
        rec = pstore.append_event({
            "event": "pattern_outcome", "pattern_id": pattern.id,
            "symbol": pattern.symbol, "week": week_key,
            "hit": bool(held),
            "expectation": comparison_text(pattern.expectation),
            "hit_rate": score["hit_rate"],
            "resolved": score["resolved"],
        })
        notices.append({"kind": "outcome", "pattern_id": pattern.id,
                        "symbol": pattern.symbol, "week": week_key,
                        "hit": bool(held), "hit_rate": score["hit_rate"],
                        "resolved": score["resolved"]})
        # The journal never kills a check.
        with contextlib.suppress(Exception):
            trading_journal.append_event(
                "pattern_outcome", settings=settings,
                pattern_id=pattern.id, symbol=pattern.symbol,
                week=week_key, hit=bool(held))
        _ = rec
    # ── new trigger on the latest evaluable week only ──────────────
    # (older weeks belong to the backtest, not the live watch — a first
    # check must not backfill two years of triggers at once. The latest
    # week may not be evaluable yet — a Tuesday check sees the new
    # week before it has a Thursday — so scan down to the latest week
    # whose condition actually resolves, and stop there.)
    seen = {(r.get("pattern_id"), r.get("week"))
            for r in pstore.events(pattern.id, kind="pattern_trigger")}
    for cand in sorted(weeks, reverse=True):
        wk = weeks[cand]
        held_cond = _holds(wk, pattern.condition)
        if held_cond is None:
            continue
        if held_cond is True:
            week_key = cand.isoformat()
            if (pattern.id, week_key) not in seen:
                pstore.append_event({
                    "event": "pattern_trigger", "pattern_id": pattern.id,
                    "symbol": pattern.symbol, "week": week_key,
                    "expectation": comparison_text(pattern.expectation),
                    "hit_rate": score["hit_rate"],
                    "resolved": score["resolved"],
                })
                notices.append({
                    "kind": "trigger", "pattern_id": pattern.id,
                    "symbol": pattern.symbol, "week": week_key,
                    "expectation": comparison_text(pattern.expectation),
                    "hit_rate": score["hit_rate"],
                    "resolved": score["resolved"]})
                # The journal never kills a check.
                with contextlib.suppress(Exception):
                    trading_journal.append_event(
                        "pattern_trigger", settings=settings,
                        pattern_id=pattern.id, symbol=pattern.symbol,
                        week=week_key,
                        expectation=comparison_text(pattern.expectation),
                        hit_rate=score["hit_rate"])
                # The outcome weekday may already have landed (a check
                # running on Tuesday sees Friday's trigger and Monday's
                # bar together) — resolve it in the same sweep.
                outcome_monday = _outcome_monday(cand, pattern)
                outcome_week = weeks.get(outcome_monday)
                if outcome_week is not None:
                    held = _holds_expectation(wk, outcome_week,
                                              pattern.expectation)
                    if held is not None:
                        pstore.append_event({
                            "event": "pattern_outcome",
                            "pattern_id": pattern.id,
                            "symbol": pattern.symbol, "week": week_key,
                            "hit": bool(held),
                            "expectation":
                                comparison_text(pattern.expectation),
                            "hit_rate": score["hit_rate"],
                            "resolved": score["resolved"],
                        })
                        notices.append({
                            "kind": "outcome", "pattern_id": pattern.id,
                            "symbol": pattern.symbol, "week": week_key,
                            "hit": bool(held),
                            "hit_rate": score["hit_rate"],
                            "resolved": score["resolved"]})
                        # The journal never kills a check.
                        with contextlib.suppress(Exception):
                            trading_journal.append_event(
                                "pattern_outcome", settings=settings,
                                pattern_id=pattern.id,
                                symbol=pattern.symbol,
                                week=week_key, hit=bool(held))
        break
    return notices


# ── rendering (chat-ready plain text) ────────────────────────────────

def _rate_line(score: dict[str, Any]) -> str:
    rate = score.get("hit_rate")
    if rate is None:
        return "UNPROVEN — no resolved instances yet"
    return (f"{rate:.0%} hit rate "
            f"({score['hits']}/{score['resolved']} resolved)")


def render_study(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"pattern study failed: {result.get('error')}"
    p = result["pattern"]
    s = result["score"]
    lines = [
        f"pattern {p['id']} — {p['name']}",
        f"  rule: {p['describe']} ({p['symbol']})",
        f"  verdict: {s['verdict'].upper()} — {_rate_line(s)}",
        f"  instances: {s['evaluated']} weeks evaluated, "
        f"{s['triggered']} triggered, {s['resolved']} resolved, "
        f"{s['pending']} pending",
    ]
    if s.get("span_start"):
        lines.append(f"  span: {s['span_start']} → {s['span_end']} "
                     f"({s['bars_used']} daily bars, source: "
                     f"{s.get('source')})")
    else:
        lines.append("  span: no history — " +
                     (f"fetch failed: {s.get('fetch_error')}"
                      if s.get("fetch_error")
                      else "no daily bars in the ledger and none fetched"))
    lines.append(f"  watching: {'on' if p['watching'] else 'off'}")
    return "\n".join(lines)


def render_pattern_list(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"pattern list failed: {result.get('error')}"
    patterns = result["patterns"]
    if not patterns:
        return "no patterns studied yet — /pattern study <rule>"
    lines = ["studied patterns:"]
    for p in patterns:
        s = p.get("score") or {}
        rate = s.get("hit_rate")
        rate_txt = (f"{rate:.0%} ({s.get('hits')}/{s.get('resolved')})"
                    if rate is not None else "UNPROVEN")
        lines.append(
            f"  {p['id']} — {p['name']} ({p['symbol']}) — {rate_txt} — "
            f"watching {'on' if p.get('watching') else 'off'}")
    return "\n".join(lines)


def render_check(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"pattern check failed: {result.get('error')}"
    notices = result.get("notices") or []
    errors = result.get("errors") or []
    if not notices and not errors:
        return "pattern watch: nothing triggered, no outcomes due"
    lines = ["pattern watch:"]
    for n in notices:
        if n["kind"] == "trigger":
            rate = n.get("hit_rate")
            rate_txt = (f"{rate:.0%} over {n.get('resolved')} resolved"
                        if rate is not None else "UNPROVEN")
            lines.append(
                f"  TRIGGER {n['pattern_id']} ({n['symbol']}, week of "
                f"{n['week']}): condition held — expect "
                f"{n['expectation']} — history says {rate_txt}")
        else:
            mark = "HIT" if n["hit"] else "MISS"
            rate = n.get("hit_rate")
            rate_txt = (f"{rate:.0%} over {n.get('resolved')} resolved"
                        if rate is not None else "UNPROVEN")
            lines.append(
                f"  OUTCOME {n['pattern_id']} ({n['symbol']}, week of "
                f"{n['week']}): {mark} — running rate {rate_txt}")
    for e in errors:
        lines.append(f"  error: {e}")
    return "\n".join(lines)


def render_status(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"pattern status failed: {result.get('error')}"
    p = result["pattern"]
    s = result["score"]
    lines = [
        f"pattern {p['id']} — {p['name']}",
        f"  rule: {p['describe']} ({p['symbol']})",
        f"  verdict: {s['verdict'].upper()} — {_rate_line(s)}",
        f"  instances: {s['evaluated']} evaluated, {s['triggered']} "
        f"triggered, {s['resolved']} resolved, {s['pending']} pending",
        f"  watching: {'on' if p['watching'] else 'off'}",
    ]
    pending = result.get("pending") or []
    for trig in pending:
        lines.append(
            f"  pending: week of {trig.get('week')} — awaiting "
            f"{comparison_text(p['expectation'])}")
    return "\n".join(lines)
