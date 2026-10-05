"""Live status dashboard renderer for the Devon console.

``render_dashboard`` is a pure function: it takes a snapshot dict and returns
the dashboard text. The snapshot is assembled by the entrypoint
(``scripts/run_chat_bot.py``) so this module stays testable without a
running bot.

Snapshot shape (all keys optional — missing sections render as "n/a")::

    {
        "uptime_s": 3723.5,
        "adapters": {"telegram": {"running": True, "received": 12, "sent": 9}, ...},
        "traffic": {"messages": 40, "replies": 32, "errors": 1, "controls": 0},
        "history": [3, 5, 2, ...],          # per-minute inbound counts (sparkline)
        "llm": {"active": "groq", "chain": [...], "health": {...}},
        "scheduler": {"running": True, "jobs": [
            {"name": "briefing", "enabled": True, "next_run": 1760000000.0}, ...]},
        "games": {"players": 7, "active_tables": 2},
        "extras": {"autonomy": "on", "arena": "off"},
        "theme": "ocean",
    }
"""

from __future__ import annotations

import datetime
import time
from typing import Any

from .palette import (
    ACCENT,
    BOLD,
    BRIGHT_WHITE,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RESET,
    SUBTLE,
    TITLE,
    WARN,
    paint,
    strip_ansi,
)
from .themes import Theme, get_theme
from .widgets import sparkline

_WIDTH = 58


def _fmt_uptime(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.datetime.fromtimestamp(float(ts))
        return dt.strftime("%H:%M")
    except (OSError, OverflowError, ValueError):
        return "—"


def _bar(label: str, value: str, value_color: str = BRIGHT_WHITE) -> str:
    label_p = paint(f"{label:<12}", CYAN)
    return f"  {label_p} {paint(value, value_color)}"


def _rule(title: str) -> str:
    t = paint(f" {title} ", TITLE + BOLD)
    fill = "─" * max(2, _WIDTH - len(strip_ansi(t)) - 2)
    return f"{paint('┌', SUBTLE)}{t}{paint(fill, SUBTLE)}{paint('┐', SUBTLE)}"


def _end() -> str:
    return paint("└" + "─" * (_WIDTH - 2) + "┘", SUBTLE)


def render_dashboard(
    snap: dict[str, Any] | None,
    *,
    color: bool | None = None,
    theme: Theme | None = None,
    bare: bool = False,
) -> str:
    """Render the full dashboard for ``snap``.

    With ``bare=True`` the outer frame (rule/end borders and hints) is
    skipped — the watch screen draws its own pane borders around it.
    """
    snap = snap or {}
    theme = theme or get_theme((snap.get("theme") or None))
    t = lambda s, role: paint(s, theme.get(role, ""), color=color)  # noqa: E731
    lines: list[str] = []
    if not bare:
        lines.append("")
        lines.append(_rule("DEVON · live status"))

    # ── system ──
    uptime = _fmt_uptime(float(snap.get("uptime_s", 0) or 0))
    lines.append(_bar("uptime", uptime, GREEN))
    extras = snap.get("extras") or {}
    for key in ("autonomy", "arena", "brain"):
        if key in extras:
            val = str(extras[key])
            vc = GREEN if val.lower() in {"on", "up", "ok", "ready"} else SUBTLE
            lines.append(_bar(key, val, vc))

    # ── brain (LLM) ──
    llm = snap.get("llm") or {}
    if llm:
        active = str(llm.get("active") or "—")
        chain = llm.get("chain") or []
        health = llm.get("health") or {}
        bad = [n for n, h in health.items() if isinstance(h, dict) and h.get("cooldown_until")]
        brain_color = GREEN if active and active != "—" else MAGENTA
        lines.append(_bar("brain", f"{active}  ({len(chain)} in chain)", brain_color))
        if bad:
            lines.append(f"    {t('cooling down:', 'warn')} {t(', '.join(bad[:4]), 'subtle')}")

    # ── adapters ──
    adapters = snap.get("adapters") or {}
    lines.append(_bar("adapters", f"{len(adapters)} configured"))
    for name in sorted(adapters):
        info = adapters[name] or {}
        running = bool(info.get("running"))
        dot = paint("●", GREEN) if running else paint("○", SUBTLE)
        recv = info.get("received", "—")
        sent = info.get("sent", "—")
        label = paint(f"{name:<14}", BRIGHT_WHITE)
        lines.append(
            f"    {dot} {label} "
            f"{paint('in', SUBTLE)} {recv} {paint('out', SUBTLE)} {sent}"
        )

    # ── traffic + sparkline ──
    traffic = snap.get("traffic") or {}
    msgs = traffic.get("messages", "—")
    replies = traffic.get("replies", "—")
    errors = traffic.get("errors", 0)
    ctrls = traffic.get("controls", "—")
    err_color = MAGENTA if errors else GREEN
    lines.append(
        _bar(
            "traffic",
            f"{msgs} msgs · {replies} replies · "
            f"{paint(str(errors), err_color + BOLD)} errors · {ctrls} controls",
        )
    )
    history = snap.get("history") or []
    if history:
        lines.append(f"    {paint('activity', CYAN):<21} {sparkline(history, color=color)}")

    # ── scheduler ──
    sched = snap.get("scheduler") or {}
    if sched:
        running = bool(sched.get("running"))
        jobs = sched.get("jobs") or []
        enabled = sum(1 for j in jobs if j.get("enabled", True))
        sdot = paint("●", GREEN) if running else paint("○", SUBTLE)
        lines.append(f"  {paint('scheduler', CYAN):<21} {sdot} {enabled}/{len(jobs)} jobs enabled")
        for job in jobs[:5]:
            name = str(job.get("name") or job.get("id", "?"))[:24]
            nxt = _fmt_ts(job.get("next_run"))
            en = "" if job.get("enabled", True) else paint(" (off)", SUBTLE)
            lines.append(f"    {paint('›', ACCENT)} {name:<24}{en} {paint('next ' + nxt, DIM)}")
        if len(jobs) > 5:
            lines.append(f"    {paint(f'… +{len(jobs) - 5} more', SUBTLE)}")
    else:
        lines.append(_bar("scheduler", "n/a", SUBTLE))

    # ── games ──
    games = snap.get("games") or {}
    if games:
        players = games.get("players", "—")
        tables = games.get("active_tables")
        extra = f" · {tables} active" if tables else ""
        lines.append(_bar("players", f"{players} game profiles{extra}", BRIGHT_WHITE))

    if not bare:
        lines.append(_end())
        lines.append(paint("  type 'dashboard --watch' for live mode · 'help' for commands", DIM))
        lines.append("")
    out = "\n".join(lines)
    if color is False:
        return strip_ansi(out)
    return out


def render_status_line(snap: dict[str, Any] | None) -> str:
    """One-line compact status for the ``status`` console command."""
    snap = snap or {}
    uptime = _fmt_uptime(float(snap.get("uptime_s", 0) or 0))
    adapters = snap.get("adapters") or {}
    up = sum(1 for a in adapters.values() if (a or {}).get("running"))
    traffic = snap.get("traffic") or {}
    sched = snap.get("scheduler") or {}
    jobs = sched.get("jobs") or []
    parts = [
        paint("devon", TITLE + BOLD),
        f"up {paint(uptime, GREEN)}",
        f"adapters {paint(f'{up}/{len(adapters)}', GREEN if up else WARN)}",
        f"msgs {paint(str(traffic.get('messages', '—')), CYAN)}",
        f"errors {paint(str(traffic.get('errors', 0)), MAGENTA if traffic.get('errors') else GREEN)}",
        f"jobs {paint(str(len(jobs)), CYAN)}",
    ]
    return " · ".join(parts)


# ── watch-mode views ──────────────────────────────────────────────────────

def render_view(
    snap: dict[str, Any] | None,
    view: str,
    *,
    color: bool | None = None,
    theme: Theme | None = None,
    bare: bool = False,
) -> str:
    """Render one watch-mode view: status | games | jobs | brain | debug.

    ``bare=True`` skips each view's own frame — the watch screen draws
    pane borders itself.
    """
    view = (view or "status").lower()
    if view == "games":
        return render_games_view(snap, color=color, theme=theme, bare=bare)
    if view == "jobs":
        return render_scheduler_view(snap, color=color, theme=theme, bare=bare)
    if view == "brain":
        return render_llm_view(snap, color=color, theme=theme, bare=bare)
    if view == "debug":
        return render_debug_view(snap, color=color, theme=theme, bare=bare)
    return render_dashboard(snap, color=color, theme=theme, bare=bare)


def render_games_view(
    snap: dict[str, Any] | None,
    *,
    color: bool | None = None,
    theme: Theme | None = None,
    bare: bool = False,
) -> str:
    """Games-focused watch view: tables, players, activity bars."""
    from .widgets import barchart, sparkline

    snap = snap or {}
    theme = theme or get_theme((snap.get("theme") or None))
    t = lambda s, role: paint(s, theme.get(role, ""), color=color)  # noqa: E731
    lines: list[str] = []
    if not bare:
        lines.append("")
        lines.append(_rule("games"))

    games = snap.get("games") or {}
    players = games.get("players", "—")
    tables = games.get("active_tables", 0)
    lines.append(_bar("players", str(players), BRIGHT_WHITE))
    lines.append(_bar("active tables", str(tables), GREEN if tables else SUBTLE))

    # Per-game activity bar chart.
    activity = games.get("activity") or {}
    if activity:
        try:
            items = sorted(
                ((str(k), float(v)) for k, v in activity.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )[:8]
            lines.append("")
            lines.append(f"  {paint('most played', CYAN)}")
            lines.extend(barchart(items, color=color))
        except (TypeError, ValueError):
            lines.append(f"  {paint('(activity data unreadable)', DIM)}")

    # Recent results / top players.
    top = games.get("top_players") or []
    if top:
        lines.append("")
        lines.append(f"  {paint('top players', CYAN)}")
        for i, p in enumerate(top[:5], 1):
            name = str(p.get("name") or p.get("id", "?"))[:20]
            score = p.get("score", p.get("wins", "—"))
            lines.append(
                f"    {paint(str(i), ACCENT)} {paint(name, BRIGHT_WHITE)} "
                f"{paint(str(score), DIM)}"
            )

    # Sparkline of game starts if provided.
    gstarts = games.get("starts_history") or []
    if gstarts:
        lines.append("")
        lines.append(
            f"  {paint('game starts', CYAN):<21} {sparkline(gstarts, color=color)}"
        )
    if not activity and not top and not gstarts:
        lines.append(f"  {paint('(no game activity yet — play a game to light this up)', DIM)}")
    if not bare:
        lines.append(_end())
    out = "\n".join(lines)
    return strip_ansi(out) if color is False else out


def render_scheduler_view(
    snap: dict[str, Any] | None,
    *,
    color: bool | None = None,
    theme: Theme | None = None,
    bare: bool = False,
) -> str:
    """Scheduler-focused watch view: every job, full detail."""
    snap = snap or {}
    theme = theme or get_theme((snap.get("theme") or None))
    lines: list[str] = []
    if not bare:
        lines.append("")
        lines.append(_rule("scheduler"))

    sched = snap.get("scheduler") or {}
    running = bool(sched.get("running"))
    jobs = sched.get("jobs") or []
    enabled = sum(1 for j in jobs if j.get("enabled", True))
    sdot = paint("●", GREEN) if running else paint("○", SUBTLE)
    lines.append(
        f"  {paint('state', CYAN):<12} {sdot} "
        f"{paint('running' if running else 'stopped', GREEN if running else SUBTLE)}"
    )
    lines.append(_bar("jobs", f"{enabled}/{len(jobs)} enabled"))
    if jobs:
        lines.append("")
        for job in jobs[:12]:
            name = str(job.get("name") or job.get("id", "?"))[:26]
            spec = str(job.get("spec") or job.get("schedule") or "")
            nxt = _fmt_ts(job.get("next_run"))
            en = job.get("enabled", True)
            dot = paint("●", GREEN) if en else paint("○", SUBTLE)
            retry = ""
            if job.get("retry_count"):
                retry = paint(f" retrying({job['retry_count']})", WARN)
            dep = ""
            if job.get("depends_on"):
                dep = paint(f" after:{job['depends_on']}", DIM)
            lines.append(
                f"    {dot} {paint(name, BRIGHT_WHITE):<28} "
                f"{paint(spec, DIM)}"
            )
            lines.append(
                f"      {paint('next ' + nxt, CYAN)}{retry}{dep}"
            )
        if len(jobs) > 12:
            lines.append(f"    {paint(f'… +{len(jobs) - 12} more', SUBTLE)}")
    else:
        lines.append(f"  {paint('(no jobs scheduled)', DIM)}")
    if not bare:
        lines.append(_end())
    out = "\n".join(lines)
    return strip_ansi(out) if color is False else out


def render_llm_view(
    snap: dict[str, Any] | None,
    *,
    color: bool | None = None,
    theme: Theme | None = None,
    bare: bool = False,
) -> str:
    """Brain-focused watch view: provider chain, health, active model."""
    from .widgets import barchart

    snap = snap or {}
    theme = theme or get_theme((snap.get("theme") or None))
    lines: list[str] = []
    if not bare:
        lines.append("")
        lines.append(_rule("brain · LLM providers"))

    llm = snap.get("llm") or {}
    active = str(llm.get("active") or "—")
    chain = llm.get("chain") or []
    health = llm.get("health") or {}
    stats = llm.get("stats") or {}

    brain_color = GREEN if active and active != "—" else MAGENTA
    lines.append(_bar("active", active, brain_color))
    lines.append("")
    lines.append(f"  {paint('provider chain', CYAN)}")
    for i, name in enumerate(chain):
        h = health.get(name) if isinstance(health, dict) else None
        cooling = bool(isinstance(h, dict) and h.get("cooldown_until"))
        if cooling:
            dot, state = paint("◌", WARN), paint("cooling down", WARN)
        elif str(name) == active:
            dot, state = paint("●", GREEN), paint("active", GREEN)
        else:
            dot, state = paint("●", SUBTLE), paint("standby", SUBTLE)
        lines.append(f"    {dot} {paint(str(name), BRIGHT_WHITE):<20} {state}")

    # Per-provider call counts as bars.
    calls = {}
    for name in chain:
        c = stats.get(name) if isinstance(stats, dict) else None
        if isinstance(c, dict) and c.get("calls"):
            calls[str(name)] = float(c["calls"])
        elif isinstance(c, (int, float)):
            calls[str(name)] = float(c)
    if calls:
        lines.append("")
        lines.append(f"  {paint('calls', CYAN)}")
        lines.extend(
            barchart(sorted(calls.items(), key=lambda kv: -kv[1]),
                     color=color, bar_color=GREEN)
        )
    if not chain:
        lines.append(f"  {paint('(no providers configured)', DIM)}")
    if not bare:
        lines.append(_end())
    out = "\n".join(lines)
    return strip_ansi(out) if color is False else out


def render_debug_view(
    snap: dict[str, Any] | None,
    *,
    color: bool | None = None,
    theme: Theme | None = None,
    bare: bool = False,
) -> str:
    """Debug-focused watch view: log tail, level counts, slow ops, LLM calls.

    Fed by :class:`nomorals.console.debug.DebugHub` — the watch screen
    installs it automatically, so the pane fills as soon as watch mode
    starts. With ``bare=True`` the outer frame is skipped (the watch
    screen draws its own pane borders).
    """
    from .debug import DebugHub
    from .widgets import barchart

    snap = snap or {}
    theme = theme or get_theme((snap.get("theme") or None))
    lines: list[str] = []
    if not bare:
        lines.append("")
        lines.append(_rule("debug · telemetry"))

    stats = DebugHub.stats()
    captured = stats["captured"]
    err = stats["errors"]
    warn = stats["warnings"]
    err_c = WARN + BOLD if err else SUBTLE
    warn_c = WARN if warn else SUBTLE
    level_legend = (
        f"{paint('ERRO', WARN + BOLD, color=color)} "
        f"{paint('WARN', WARN, color=color)} "
        f"{paint('INFO', SUBTLE, color=color)} "
        f"{paint('DEBU', DIM, color=color)}"
    )
    lines.append(
        _bar(
            "telemetry",
            f"{captured} logs · "
            f"{paint(str(err), err_c)} errors · "
            f"{paint(str(warn), warn_c)} warnings",
        )
    )
    lines.append(f"    {level_legend}")

    # ── noisiest loggers ──
    top = DebugHub.top_loggers(5)
    if top:
        lines.append("")
        lines.append(f"  {paint('noisiest loggers', CYAN)}")
        lines.extend(
            barchart([(name, float(cnt)) for name, cnt in top],
                     color=color, bar_color=MAGENTA, width=14)
        )

    # ── recent log lines ──
    lines.append("")
    lines.append(f"  {paint('recent logs', CYAN)}")
    recent = DebugHub.recent(10)
    if recent:
        for ts, level, logger_name, msg in recent[-10:]:
            tstr = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            if level in {"ERROR", "CRITICAL"}:
                lc = paint(level[:4], WARN + BOLD)
            elif level == "WARNING":
                lc = paint(level[:4], WARN)
            elif level == "DEBUG":
                lc = paint(level[:4], DIM)
            else:
                lc = paint(level[:4], SUBTLE)
            short_logger = logger_name.split(".")[-1][:16]
            text = " ".join(msg.split())
            if len(text) > 72:
                text = text[:69] + "…"
            lines.append(
                f"    {paint(tstr, DIM)} {lc} "
                f"{paint(short_logger, ACCENT)} {text}"
            )
    else:
        lines.append(
            f"    {paint('(capturing — logs appear here while watch mode runs)', DIM)}"
        )

    # ── slowest operations ──
    lines.append("")
    lines.append(f"  {paint('slowest ops', CYAN)}")
    slow = DebugHub.slow_ops()[:6]
    if slow:
        lines.extend(
            barchart(
                [(str(e["name"])[:30], float(e["duration_s"])) for e in slow],
                color=color, bar_color=WARN, width=14,
            )
        )
    else:
        lines.append(f"    {paint('(none recorded)', DIM)}")

    # ── LLM call traces ──
    lines.append("")
    lines.append(f"  {paint('LLM calls', CYAN)}")
    calls = DebugHub.llm_calls(8)
    if calls:
        for call in calls:
            tstr = datetime.datetime.fromtimestamp(call["ts"]).strftime("%H:%M:%S")
            ok = call["success"]
            mark = paint("✓", GREEN) if ok else paint("✗", WARN + BOLD)
            prov = str(call["provider"])[:16]
            op = str(call["operation"])[:12]
            lat = call["latency_s"]
            latc = WARN + BOLD if lat >= 10 else (WARN if lat >= 3 else SUBTLE)
            err_txt = ""
            if not ok and call.get("error"):
                err_txt = f" {paint(str(call['error'])[:36], DIM)}"
            lines.append(
                f"    {paint(tstr, DIM)} {mark} "
                f"{paint(prov, BRIGHT_WHITE):<18} {paint(op, CYAN):<14} "
                f"{paint(f'{lat:.2f}s', latc)}{err_txt}"
            )
    else:
        lines.append(f"    {paint('(no LLM calls recorded)', DIM)}")

    if not bare:
        lines.append(_end())
    out = "\n".join(lines)
    return strip_ansi(out) if color is False else out


def render_statusbar(
    snap: dict[str, Any] | None,
    view: str = "status",
    *,
    unread: int = 0,
    color: bool | None = None,
) -> str:
    """One-line bottom status bar for watch mode."""
    snap = snap or {}
    uptime = _fmt_uptime(float(snap.get("uptime_s", 0) or 0))
    traffic = snap.get("traffic") or {}
    msgs = traffic.get("messages", 0)
    errors = traffic.get("errors", 0)
    games = snap.get("games") or {}
    tables = games.get("active_tables", 0)
    llm = snap.get("llm") or {}
    active = str(llm.get("active") or "—")

    parts = [
        f"{paint('⏱', CYAN, color=color)} {paint(uptime, BRIGHT_WHITE, color=color)}",
        f"{paint('📨', CYAN, color=color)} {paint(str(msgs), BRIGHT_WHITE, color=color)}",
        f"{paint('🎮', CYAN, color=color)} {paint(str(tables), BRIGHT_WHITE, color=color)}",
        f"{paint('🧠', CYAN, color=color)} {paint(active, GREEN if active != '—' else MAGENTA, color=color)}",
    ]
    if errors:
        parts.append(
            f"{paint('⚠', WARN, color=color)} "
            f"{paint(str(errors), WARN + BOLD, color=color)}"
        )
    if unread:
        parts.append(
            f"{paint('📬', WARN, color=color)} "
            f"{paint(f'{unread} new', WARN + BOLD, color=color)}"
        )
    keys = paint("[1-4] views · d debug · q quit", DIM, color=color)
    return "  ".join(parts) + "    " + keys
