"""God-tier presentation for triggers: status lines, detail boxes, history.

Two themes: ``plain`` (no ANSI — logs, files, tests) and ``color``
(terminal).  Everything here is pure formatting over trigger/status
dicts — no I/O, no side effects.
"""

from __future__ import annotations

import time
from typing import Any

from .models import (
    OUTCOME_ERROR,
    OUTCOME_FIRED,
    OUTCOME_NO_MATCH,
    OUTCOME_SKIPPED,
)

#: outcome → glyph (changedetection.io / HA status-dot parity)
OUTCOME_GLYPHS = {
    OUTCOME_FIRED: "✅",
    OUTCOME_NO_MATCH: "⚪",
    OUTCOME_SKIPPED: "⏸️",
    OUTCOME_ERROR: "❌",
}

#: source → glyph
SOURCE_GLYPHS = {
    "schedule": "🕐",
    "file": "📝",
    "price": "💹",
    "message": "💬",
    "webhook": "🔔",
    "entity_state": "🏠",
    "bus": "🔀",
    "url": "🌐",
}

#: action → glyph
ACTION_GLYPHS = {
    "notify": "📣",
    "message": "✉️",
    "command": "⌨️",
    "mission": "🎯",
}

_COLORS = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def _c(text: str, color: str, theme: str) -> str:
    if theme != "color":
        return text
    return f"{_COLORS[color]}{text}{_COLORS['reset']}"


def outcome_glyph(outcome: str | None) -> str:
    """✅ fired / ⚪ no_match / ⏸️ skipped / ❌ error."""
    return OUTCOME_GLYPHS.get(outcome or "", "❔")


def _ago(ts: float | None, now: float | None = None) -> str:
    if not ts:
        return "never"
    now = now if now is not None else time.time()
    delta = max(0, now - ts)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _in_future(ts: float | None, now: float | None = None) -> str:
    if not ts:
        return "—"
    now = now if now is not None else time.time()
    delta = ts - now
    if delta <= 0:
        return "due"
    if delta < 3600:
        return f"in {int(delta // 60)}m"
    if delta < 86400:
        return f"in {int(delta // 3600)}h"
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def format_trigger_line(trigger: Any, *,
                        theme: str = "plain",
                        now: float | None = None,
                        next_run: float | None = None) -> str:
    """One rich line per trigger.

    ``🟢 abc123 [price→notify] BTC dip · ✅ fired · 12 fires · last 3m ago ·
    next in 5m``
    """
    d = trigger.to_dict() if hasattr(trigger, "to_dict") else dict(trigger)
    state = "🟢" if d.get("enabled") else "🔴"
    src = SOURCE_GLYPHS.get(d.get("source", ""), "•")
    act = ACTION_GLYPHS.get(d.get("action", ""), "•")
    glyph = outcome_glyph(d.get("last_outcome"))
    bits = [
        _c(state, "green" if d.get("enabled") else "red", theme),
        _c(str(d.get("id", ""))[:12], "dim", theme),
        f"[{src}{d.get('source')}→{act}{d.get('action')}]",
        _c(str(d.get("name", "")), "bold", theme),
        f"{glyph} {d.get('last_outcome') or 'never fired'}",
        f"{int(d.get('fire_count') or 0)} fires",
        f"last {_ago(d.get('last_fired'), now)}",
    ]
    if d.get("source") == "schedule" and next_run:
        bits.append(f"next {_in_future(next_run, now)}")
    if d.get("mode") not in (None, "parallel"):
        bits.append(f"mode={d.get('mode')}")
    if d.get("digest_pending"):
        bits.append(f"📦 {d['digest_pending']} buffered")
    return " · ".join(bits)


def format_history_row(row: dict[str, Any], *,
                       theme: str = "plain") -> str:
    """One history row: ``10-10 07:12:03 ✅ fired · price=61234.5``."""
    when = time.strftime("%m-%d %H:%M:%S",
                         time.localtime(float(row.get("at") or 0)))
    glyph = outcome_glyph(row.get("outcome"))
    head = f"{when} {glyph} {row.get('outcome')}"
    if row.get("trigger_id"):
        head += f" · {str(row['trigger_id'])[:12]}"
    detail = row.get("detail") or {}
    bits = []
    if isinstance(detail, dict):
        ev = detail.get("evidence") if "evidence" in detail else detail
        if isinstance(ev, dict):
            for key in ("price", "symbol", "event", "url", "path",
                        "entity_id", "topic", "matched", "reason",
                        "condition", "lines"):
                if ev.get(key) not in (None, ""):
                    bits.append(f"{key}={ev[key]}")
    if row.get("error"):
        bits.append(_c(f"error={row['error']}", "red", theme))
    return head + (" · " + " ".join(str(b)[:80] for b in bits[:6])
                   if bits else "")


def format_trigger_detail(status: dict[str, Any], *,
                          theme: str = "plain") -> str:
    """Box-style detail view for one trigger's full status."""
    d = dict(status)
    bar = "─" * 52
    lines = [
        f"╭{bar}╮",
        f"│ {_c(str(d.get('name', '')), 'bold', theme):<50} │",
        f"├{bar}┤",
    ]
    src = SOURCE_GLYPHS.get(d.get("source", ""), "•")
    act = ACTION_GLYPHS.get(d.get("action", ""), "•")
    lines.append(
        f"│ id      {str(d.get('id', '')):<42} │")
    lines.append(
        f"│ enabled {'🟢 yes' if d.get('enabled') else '🔴 no':<42} │")
    lines.append(
        f"│ flow    {src} {d.get('source')}  →  {act} {d.get('action')}"
        f"{'':<27} │")
    lines.append(
        f"│ mode    {d.get('mode', 'parallel'):<42} │")
    if d.get("cooldown_s"):
        lines.append(f"│ cooldown {d['cooldown_s']:g}s{'':<41} │")
    if d.get("poll_s"):
        lines.append(f"│ poll     every {d['poll_s']:g}s{'':<36} │")
    for cond in d.get("conditions") or []:
        ctype = cond.get("type", "?")
        if ctype == "time_window":
            desc = f"{cond.get('after')}–{cond.get('before')}"
        elif ctype == "rate":
            desc = (f"max {cond.get('max')} per "
                    f"{float(cond.get('window_s') or 0):g}s")
        elif ctype == "evidence":
            desc = f"match {cond.get('match')}"
        else:
            desc = str(cond)
        lines.append(f"│ gate    {ctype}: {desc:<35} │")
    if d.get("source") == "schedule":
        lines.append(
            f"│ next    {_in_future(d.get('next_run')):<42} │")
    stats = d.get("stats") or {}
    stat_bits = " ".join(
        f"{outcome_glyph(k)}{v}" for k, v in sorted(stats.items())
        if k != "last_event_at" and v)
    lines.append(f"│ stats   {stat_bits:<42} │")
    lines.append(
        f"│ fires   {int(d.get('fire_count') or 0)} "
        f"(last {_ago(d.get('last_fired'))})"
        f"{'':<27} │")
    if d.get("digest_pending"):
        lines.append(
            f"│ digest  📦 {d['digest_pending']} buffered{'':<35} │")
    if d.get("running"):
        lines.append(f"│         {_c('● running now', 'yellow', theme):<42} │")
    lines.append(f"╰{bar}╯")
    return "\n".join(lines)


def format_digest_preview(trigger_name: str,
                          items: list[dict[str, Any]]) -> str:
    """What a digest flush will send — for dry-run previews."""
    lines = [f"📦 {trigger_name}: {len(items)} alert(s)"]
    for it in items[:10]:
        head = f"{it['title']}: " if it.get("title") else ""
        lines.append(f"  • {head}{it['text']}"[:120])
    if len(items) > 10:
        lines.append(f"  … and {len(items) - 10} more")
    return "\n".join(lines)


def format_status_table(triggers: list[Any], *,
                        theme: str = "plain",
                        now: float | None = None,
                        next_runs: dict[str, float] | None = None) -> str:
    """All triggers, one rich line each — the ``nm trigger list`` view."""
    if not triggers:
        return "no triggers — add one with `nm trigger add`"
    next_runs = next_runs or {}
    lines = []
    for t in triggers:
        tid = t.id if hasattr(t, "id") else t.get("id", "")
        lines.append(format_trigger_line(
            t, theme=theme, now=now, next_run=next_runs.get(tid)))
    on = sum(1 for t in triggers
             if (t.enabled if hasattr(t, "enabled") else t.get("enabled")))
    lines.append(f"\n{on}/{len(triggers)} enabled")
    return "\n".join(lines)
