"""Transparent cost display — the Hercules steal (build-map extension #7).

"this answer cost $0.003" after every response.  The router already
meters every LLM call into ``~/.nomorals/llm/cost.jsonl`` (#18); this
module surfaces that metering in chat, toggleable per user, plus the
budgeted-research NL ("research this with a $0.50 budget").

Every function never raises.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "format_cost",
    "format_cost_table",
    "budget_alert_line",
    "parse_budget_nl",
    "CostDisplay",
    "get_display",
    "maybe_cost_footer",
    "control_cost",
]


# ── formatting ─────────────────────────────────────────────────────────────

def format_cost(usd: float) -> str:
    """Human cost: $0.003, $1.20, or 'free' for zero.  Never raises."""
    try:
        usd = float(usd or 0.0)
        if usd <= 0:
            return "free"
        if usd < 0.01:
            return f"${usd:.4f}".rstrip("0").rstrip(".")
        if usd < 10:
            return f"${usd:.2f}"
        return f"${usd:,.2f}"
    except Exception:  # noqa: BLE001
        return "free"


#: "with a $0.50 budget", "budget $2", "spend $0.50 researching this"
_BUDGET_RE = re.compile(
    r"(?:budget|spend)\s*(?:of\s*)?(?:a\s*)?\$?\s*(\d+(?:\.\d{1,4})?)",
    re.IGNORECASE,
)
_BUDGET_RE2 = re.compile(
    r"\$\s*(\d+(?:\.\d{1,4})?)\s*(?:budget|to\s+spend|spending)",
    re.IGNORECASE,
)


def parse_budget_nl(text: str) -> float | None:
    """Parse a spend budget from natural language.

    "research this with a $0.50 budget" → 0.50.  None when no budget
    is mentioned.  Never raises.
    """
    try:
        text = text or ""
        m = _BUDGET_RE.search(text) or _BUDGET_RE2.search(text)
        if not m:
            return None
        val = float(m.group(1))
        return val if val > 0 else None
    except Exception:  # noqa: BLE001
        return None


def format_cost_table(breakdown: dict[str, Any],
                      *, limit: int = 8) -> str:
    """God-tier spend table from :meth:`CostLog.breakdown`.

    Shows the total, then the top operations and providers by cost —
    the "which feature is burning the budget" view.  Never raises.
    """
    try:
        total = breakdown.get("total") or {}
        by_op = breakdown.get("by_operation") or {}
        by_prov = breakdown.get("by_provider") or {}
        lines = ["💰 LLM spend"]
        lines.append(
            f"  total {format_cost(total.get('cost_usd', 0.0))} · "
            f"{total.get('calls', 0)} calls · "
            f"{int(total.get('prompt_tokens', 0) + total.get('completion_tokens', 0)):,} tokens · "
            f"avg {total.get('avg_latency_ms', 0)}ms")
        cached = int(total.get("cached_tokens", 0) or 0)
        reasoning = int(total.get("reasoning_tokens", 0) or 0)
        if cached or reasoning:
            lines.append(
                f"  (cached {cached:,} in · reasoning {reasoning:,} — "
                "billed, never shown)")

        def _rows(bucket: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
            items = [(k, v) for k, v in bucket.items()
                     if isinstance(v, dict)]
            items.sort(key=lambda kv: -float(kv[1].get("cost_usd", 0.0)))
            return items[:max(1, limit)]

        if by_op:
            lines.append("  by operation:")
            for name, slot in _rows(by_op):
                lines.append(
                    f"    {name:<12} {format_cost(slot.get('cost_usd', 0.0)):>9} "
                    f"· {slot.get('calls', 0)} calls · "
                    f"avg {slot.get('avg_latency_ms', 0)}ms")
        if by_prov:
            lines.append("  by provider:")
            for name, slot in _rows(by_prov):
                lines.append(
                    f"    {name:<12} {format_cost(slot.get('cost_usd', 0.0)):>9} "
                    f"· {slot.get('calls', 0)} calls · "
                    f"avg {slot.get('avg_latency_ms', 0)}ms")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return "💰 spend unavailable"


def budget_alert_line(spent: float, budget: float) -> str:
    """One-line budget status with alert emoji.  Never raises."""
    try:
        spent = float(spent or 0.0)
        budget = float(budget or 0.0)
        if budget <= 0:
            return f"💰 {format_cost(spent)} spent (no budget set)"
        pct = spent / budget * 100.0
        bar_len = 12
        filled = max(0, min(bar_len, int(pct / 100 * bar_len)))
        bar = "█" * filled + "░" * (bar_len - filled)
        if pct >= 100:
            emoji, word = "🛑", "EXCEEDED"
        elif pct >= 80:
            emoji, word = "⚠️", "warning"
        elif pct >= 50:
            emoji, word = "👀", "watch"
        else:
            emoji, word = "💰", "ok"
        return (f"{emoji} budget {word}: {format_cost(spent)} / "
                f"{format_cost(budget)} [{bar}] {pct:.0f}%")
    except Exception:  # noqa: BLE001
        return "💰 budget status unavailable"


# ── per-user toggle + spend queries ─────────────────────────────────────────

def _default_db() -> str:
    d = Path.home() / ".nomorals" / "llm"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return str(d / "cost_display.db")


class CostDisplay:
    """Toggleable per-user cost display, backed by the router's CostLog.

    ``footer_for_turn(since_ts)`` sums the metered spend since the turn
    started: "this answer cost $0.003".  ``today_spend()`` is the rolling
    24h total.  SQLite, thread-safe enough for chat, never raises.
    """

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _default_db()
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cost_display ("
                " user_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 0,"
                " updated_at REAL NOT NULL)"
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("cost display db init failed", exc_info=True)
            self._db = None

    # -- toggle --

    def is_enabled(self, user_id: str = "owner") -> bool:
        try:
            if self._db is None:
                return False
            row = self._db.execute(
                "SELECT enabled FROM cost_display WHERE user_id = ?",
                (user_id or "owner",),
            ).fetchone()
            return bool(row and row["enabled"])
        except Exception:  # noqa: BLE001
            return False

    def set_enabled(self, on: bool, user_id: str = "owner") -> bool:
        try:
            if self._db is None:
                return False
            self._db.execute(
                "INSERT INTO cost_display (user_id, enabled, updated_at)"
                " VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET"
                " enabled = excluded.enabled, updated_at = excluded.updated_at",
                (user_id or "owner", 1 if on else 0, time.time()),
            )
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- spend --

    def turn_cost(self, since_ts: float, cost_path: Any = None) -> float:
        """Metered USD spend since ``since_ts``.  Never raises."""
        try:
            from .router import total_spend
            return float(total_spend(since_ts, cost_path) or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def today_spend(self, cost_path: Any = None) -> float:
        """Rolling 24h metered spend.  Never raises."""
        try:
            return self.turn_cost(time.time() - 86400, cost_path)
        except Exception:  # noqa: BLE001
            return 0.0

    def footer_for_turn(self, since_ts: float, user_id: str = "owner",
                        cost_path: Any = None) -> str:
        """The footer line, or '' when disabled.  Never raises."""
        try:
            if not self.is_enabled(user_id):
                return ""
            cost = self.turn_cost(since_ts, cost_path)
            return f"\n\n_this answer cost {format_cost(cost)}_"
        except Exception:  # noqa: BLE001
            return ""


_display: CostDisplay | None = None


def get_display(db_path: str = "") -> CostDisplay:
    """Process-global CostDisplay.  Never raises."""
    global _display
    if _display is None:
        _display = CostDisplay(db_path)
    return _display


def maybe_cost_footer(reply: str, user_id: str = "owner",
                      since_ts: float = 0.0,
                      display: CostDisplay | None = None) -> str:
    """Append the cost footer to a reply when the user opted in.

    This is the hook the chat reply path calls before sending.  Never
    raises; returns the reply unchanged when disabled or on any error.
    """
    try:
        disp = display or get_display()
        footer = disp.footer_for_turn(since_ts or time.time(), user_id)
        return (reply or "") + footer if footer else (reply or "")
    except Exception:  # noqa: BLE001
        return reply or ""


# ── chat ────────────────────────────────────────────────────────────────────

def _usage() -> str:
    return (
        "💰 /cost — transparent spend display\n"
        "  /cost on|off — show 'this answer cost $X' after every reply\n"
        "  /cost — today's metered spend\n"
        "  /cost budget $0.50 — set the research spend cap"
    )


def control_cost(tail: str, context: Any = None, chat: Any = None,
                 display: CostDisplay | None = None,
                 **kwargs: Any) -> str:
    """Chat handler for /cost.  Owner-only at dispatch.  Never raises."""
    try:
        disp = display or get_display()
        user_id = "owner"
        try:
            if context is not None:
                user_id = str(getattr(context, "user_id", "") or "owner")
        except Exception:  # noqa: BLE001
            pass
        parts = (tail or "").strip().split()
        if not parts:
            spent = disp.today_spend()
            state = "on" if disp.is_enabled(user_id) else "off"
            return (
                f"💰 spend today: {format_cost(spent)} (metered)\n"
                f"cost display: {state} — /cost on to see per-answer costs"
            )
        cmd = parts[0].lower()
        if cmd in ("on", "enable", "yes"):
            disp.set_enabled(True, user_id)
            return "💰 cost display on — you'll see 'this answer cost $X' after replies."
        if cmd in ("off", "disable", "no"):
            disp.set_enabled(False, user_id)
            return "💰 cost display off."
        if cmd in ("breakdown", "table", "detail", "details"):
            try:
                from .router import CostLog
                log = CostLog(cost_path) if (cost_path := kwargs.get("cost_path")) else CostLog()
                return format_cost_table(log.breakdown(time.time() - 86400))
            except Exception:  # noqa: BLE001
                return "💰 breakdown unavailable."
        if cmd == "budget" and len(parts) > 1:
            amount = parse_budget_nl(" ".join(parts[1:]))
            if amount is None:
                return "usage: /cost budget $0.50"
            try:
                if context is not None and hasattr(context, "set_research_budget"):
                    context.set_research_budget(amount)
            except Exception:  # noqa: BLE001
                pass
            return f"💰 research budget set: {format_cost(amount)} per run."
        return _usage()
    except Exception:  # noqa: BLE001
        return _usage()
