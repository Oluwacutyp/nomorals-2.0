"""Console-only commands for the local terminal adapter.

These are intercepted *before* a line reaches the brain, so typing
``dashboard`` in the console shows the dashboard instead of chatting about
dashboards. Anything unrecognized returns ``None`` and flows through to the
bot normally — console input is never broken by this.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from .banner import tip_of_the_day
from .dashboard import render_dashboard, render_status_line
from .palette import (
    ACCENT,
    BOLD,
    BRIGHT_WHITE,
    CYAN,
    DIM,
    GREEN,
    SUBTLE,
    TITLE,
    WARN,
    paint,
)
from .themes import get_theme, list_themes, theme_name

SnapshotProvider = Callable[[], dict[str, Any]]

_CLEAR_SEQ = "\033[2J\033[H"

#: All console commands + aliases, for completion and help.
_COMMANDS: dict[str, tuple[str, ...]] = {
    "dashboard": ("dashboard [--watch [secs]]", "full status · live split-pane with --watch"),
    "dash": ("dashboard [--watch [secs]]", "alias for dashboard"),
    "console": ("console", "god-tier fullscreen interactive console"),
    "status": ("status", "one-line compact status"),
    "jobs": ("jobs", "scheduled background jobs and next run times"),
    "theme": ("theme [name]", "show/switch palette"),
    "palette": ("palette [name]", "swatch card for a theme"),
    "banner": ("banner [style]", "re-show the startup banner"),
    "tip": ("tip", "a random tip of the day"),
    "uptime": ("uptime", "how long the bot has been running"),
    "errors": ("errors [n]", "recent errors from the debug telemetry"),
    "slow": ("slow [n]", "slowest recorded operations"),
    "llm": ("llm [n]", "recent LLM provider call traces"),
    "log": ("log [level] [n]", "tail captured log lines, e.g. 'log error 20'"),
    "export": ("export <file>", "save the dashboard snapshot to a file"),
    "history": ("history", "commands typed this session"),
    "clear": ("clear", "clear the terminal"),
    "help": ("help [command]", "this list · or help for one command"),
}

_HELP_TEXT = """console commands (local terminal only):
  console                    god-tier fullscreen interactive console
  dashboard [--watch [secs]]   full status · live split-pane with --watch
                               (in watch: 1=status 2=games 3=jobs 4=brain
                                5=adapters 6=health d=debug, :=cmd, /=filter,
                                ?=help, t=theme, j/k=scroll, q=quit)
  status          one-line compact status
  jobs            scheduled background jobs and next run times
  banner [style]  re-show the startup banner (block · slant · mini · pixel)
  theme [name]    show/switch palette
  palette [name]  swatch card for a theme
  uptime          how long the bot has been running
  errors [n]      recent errors from the debug telemetry
  slow [n]        slowest recorded operations
  llm [n]         recent LLM provider call traces
  log [level] [n] tail captured logs (e.g. 'log error 20')
  export <file>   save the dashboard snapshot to a file
  history         commands typed this session
  tip             a random tip of the day
  clear           clear the terminal
  help [command]  this list · or help for one command

anything else is sent to Devon as chat.
"""


def complete(prefix: str) -> list[str]:
    """Tab-completion candidates for a command prefix (REPL wiring)."""
    p = (prefix or "").lower()
    return sorted(n for n in _COMMANDS if n.startswith(p))


class ConsoleCommands:
    """Intercept console lines. ``handle`` returns text to print, or None."""

    def __init__(self, snapshot_provider: SnapshotProvider) -> None:
        self._snapshot = snapshot_provider
        self._history: list[str] = []  # commands typed this session

    def handle(self, text: str) -> str | None:
        """Return text to print for a console command, or None (→ brain)."""
        result = self._dispatch(text)
        if result is not None:
            self._record(text)
        return result

    def _dispatch(self, text: str) -> str | None:
        parts = (text or "").strip().split()
        if not parts:
            return None
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in {"dashboard", "dash", "/dashboard"}:
            if args and args[0] in {"--watch", "-w", "watch"}:
                secs = 2.0
                if len(args) > 1:
                    try:
                        secs = max(1.0, min(30.0, float(args[1])))
                    except ValueError:
                        secs = 2.0  # unparseable interval → default, not a crash
                return self._watch(secs)
            return render_dashboard(self._safe_snapshot())
        if cmd in {"console", "/console"}:
            return self._god_console()
        if cmd in {"status", "/status"}:
            return render_status_line(self._safe_snapshot())
        if cmd in {"jobs", "schedule", "/jobs"}:
            return self._render_jobs()
        if cmd == "theme":
            return self._handle_theme(args)
        if cmd == "palette":
            return self._handle_palette(args)
        if cmd == "banner":
            return self._handle_banner(args)
        if cmd == "tip":
            return paint(f"💡 {tip_of_the_day()}", CYAN)
        if cmd == "uptime":
            return self._handle_uptime()
        if cmd == "errors":
            return self._handle_errors(args)
        if cmd == "slow":
            return self._handle_slow(args)
        if cmd == "llm":
            return self._handle_llm(args)
        if cmd == "log":
            return self._handle_log(args)
        if cmd == "export":
            return self._handle_export(args)
        if cmd == "history":
            return self._handle_history()
        if cmd in {"clear", "cls"}:
            return _CLEAR_SEQ + paint("terminal cleared — type 'dashboard' for status", DIM)
        if cmd in {"help", "/help", "?"}:
            return self._handle_help(args)
        return None

    def _record(self, text: str) -> None:
        line = (text or "").strip()
        if line and (not self._history or self._history[-1] != line):
            self._history.append(line)
            del self._history[:-100]

    def _safe_snapshot(self) -> dict[str, Any]:
        try:
            snap = self._snapshot()
            return snap if isinstance(snap, dict) else {}
        except Exception:  # noqa: BLE001 - dashboard must never crash the console
            return {}

    def _watch(self, interval: float) -> str:
        """Blocking split-pane live dashboard. Keys: 1-4 views, q quits.

        Never raises: a failure returns a visible error + the static
        dashboard instead of silently falling through to the brain.
        """
        try:
            from .dashboard import render_dashboard
            from .widgets import GodScreen

            screen = GodScreen(interval=interval, snapshot=self._safe_snapshot)
            return screen.run()
        except Exception as exc:  # noqa: BLE001 - watch must never kill the console
            return (
                paint(f"watch mode failed ({exc}) — static snapshot instead", WARN)
                + "\n"
                + render_dashboard(self._safe_snapshot())
            )

    def _god_console(self) -> str:
        """Launch the god-tier fullscreen interactive console."""
        try:
            from .godconsole import GodConsole

            console = GodConsole(
                snapshot=self._safe_snapshot,
                on_command=self._handle_console_command,
                completer=complete,
                highlighter=self.highlight,
            )
            return console.run()
        except Exception as exc:  # noqa: BLE001
            return paint(f"console failed ({exc})", WARN)

    def _handle_console_command(self, cmd: str) -> str:
        """Handle a command typed in the god console."""
        result = self.handle(cmd)
        if result is not None:
            return result
        return f"(sent to Devon: {cmd})"

    def _handle_theme(self, args: list[str]) -> str:
        if not args:
            cur = theme_name()
            avail = " · ".join(list_themes())
            return (
                f"{paint('current theme:', ACCENT)} {paint(cur, BOLD)}\n"
                f"{paint('available:', ACCENT)} {paint(avail, CYAN)}\n"
                f"{paint('switch with: theme <name>', DIM)}"
            )
        want = args[0].strip().lower()
        if want not in list_themes():
            return paint(
                f"unknown theme '{want}' — try: {', '.join(list_themes())}", DIM
            )
        os.environ["NM_CONSOLE_THEME"] = want
        theme = get_theme(want)
        # Show a swatch so the user sees the new palette immediately.
        swatch = " ".join(
            paint("██", theme[role]) for role in ("title", "accent", "info", "ok", "warn", "err")
        )
        return f"{paint('theme →', GREEN)} {paint(want, BOLD)}  {swatch}"

    def _handle_palette(self, args: list[str]) -> str:
        """Swatch card for a theme (or the active one)."""
        from .themes import theme_preview

        name = args[0].strip().lower() if args else None
        if name and name not in list_themes():
            return paint(f"unknown theme '{name}' — try: {', '.join(list_themes())}", DIM)
        return theme_preview(name)

    def _handle_banner(self, args: list[str]) -> str:
        """Re-show the startup banner, optionally in another style."""
        from .banner import list_banner_styles, render_banner

        style = (args[0].strip().lower() if args else "block")
        if style not in list_banner_styles():
            return paint(
                f"unknown style '{style}' — try: {', '.join(list_banner_styles())}", DIM
            )
        snap = self._safe_snapshot()
        adapters = list((snap.get("adapters") or {}).keys())
        return render_banner(adapters, style=style)

    def _handle_uptime(self) -> str:
        from .dashboard import _fmt_uptime  # noqa: PLC2701 - internal reuse

        snap = self._safe_snapshot()
        up = _fmt_uptime(float(snap.get("uptime_s", 0) or 0))
        return f"{paint('uptime', ACCENT)} {paint(up, BOLD)}"

    def _handle_errors(self, args: list[str]) -> str:
        """Recent errors + tracebacks from DebugHub (lnav jump-to-error)."""
        from .debug import DebugHub

        n = self._parse_n(args, default=5)
        errs = DebugHub.recent(n * 3, level="ERROR")[-n:]
        crits = DebugHub.recent(n * 3, level="CRITICAL")[-n:]
        lines = [paint("recent errors", TITLE + BOLD), ""]
        entries = sorted(errs + crits, key=lambda e: e[0])[-n:]
        if not entries:
            lines.append(paint("  no errors captured — telemetry starts in watch mode", DIM))
        import datetime
        for ts, level, logger_name, msg in entries:
            tstr = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            lines.append(
                f"  {paint(tstr, DIM)} {paint(level[:4], WARN + BOLD)} "
                f"{paint(logger_name.split('.')[-1][:18], ACCENT)} "
                f"{' '.join(msg.split())[:80]}"
            )
        excs = DebugHub.exceptions(1)
        if excs:
            tb = excs[0]["traceback"].strip().split("\n")
            lines.append("")
            lines.append(paint("  latest traceback (tail):", WARN))
            for ln in tb[-6:]:
                lines.append(f"    {paint(ln[:88], DIM)}")
        return "\n".join(lines)

    def _handle_slow(self, args: list[str]) -> str:
        from .debug import DebugHub
        from .widgets import barchart

        n = self._parse_n(args, default=8)
        slow = DebugHub.slow_ops()[:n]
        lines = [paint("slowest operations", TITLE + BOLD), ""]
        if not slow:
            lines.append(paint("  none recorded yet", DIM))
        else:
            lines.extend(
                barchart([(str(e["name"])[:32], float(e["duration_s"])) for e in slow],
                         bar_color=WARN, width=16)
            )
        return "\n".join(lines)

    def _handle_llm(self, args: list[str]) -> str:
        from .debug import DebugHub

        n = self._parse_n(args, default=8)
        calls = DebugHub.llm_calls(n)
        lines = [paint("recent LLM calls", TITLE + BOLD), ""]
        if not calls:
            lines.append(paint("  none recorded — telemetry starts in watch mode", DIM))
        import datetime
        for call in calls:
            tstr = datetime.datetime.fromtimestamp(call["ts"]).strftime("%H:%M:%S")
            mark = paint("✓", GREEN) if call["success"] else paint("✗", WARN + BOLD)
            lines.append(
                f"  {paint(tstr, DIM)} {mark} "
                f"{paint(str(call['provider'])[:18], BRIGHT_WHITE)} "
                f"{paint(str(call['operation'])[:16], CYAN)} "
                f"{paint(f'{call['latency_s']:.2f}s', DIM)}"
            )
        return "\n".join(lines)

    def _handle_log(self, args: list[str]) -> str:
        """``log [level] [n]`` — lnav-lite tail over captured logs."""
        from .debug import DebugHub

        level = None
        n = 20
        for a in args:
            if a.isdigit():
                n = max(1, min(100, int(a)))
            else:
                level = a.upper()
        recs = DebugHub.recent(n, level=level)
        lines = [paint(f"logs{f' · {level}' if level else ''}", TITLE + BOLD), ""]
        if not recs:
            lines.append(paint("  nothing captured — telemetry starts in watch mode", DIM))
        import datetime
        for ts, lvl, logger_name, msg in recs:
            tstr = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            lc = WARN + BOLD if lvl in {"ERROR", "CRITICAL"} else (
                WARN if lvl == "WARNING" else (DIM if lvl == "DEBUG" else SUBTLE))
            lines.append(
                f"  {paint(tstr, DIM)} {paint(lvl[:4], lc)} "
                f"{paint(logger_name.split('.')[-1][:18], ACCENT)} "
                f"{' '.join(msg.split())[:88]}"
            )
        return "\n".join(lines)

    def _handle_export(self, args: list[str]) -> str:
        """Save the dashboard snapshot (rendered + raw) to a file."""
        from .dashboard import render_summary
        from .palette import strip_ansi

        if not args:
            return paint("usage: export <file>  (writes dashboard + snapshot)", DIM)
        path = os.path.expanduser(args[0])
        try:
            snap = self._safe_snapshot()
            text = strip_ansi(render_summary(snap, color=False))
            text += "\n\n── raw snapshot ──\n" + repr(snap) + "\n"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            return paint(f"export failed: {exc}", WARN)
        return f"{paint('exported →', GREEN)} {paint(path, BOLD)}"

    def _handle_history(self) -> str:
        lines = [paint("command history", TITLE + BOLD), ""]
        if not self._history:
            lines.append(paint("  (empty this session)", DIM))
        for i, cmd in enumerate(self._history[-20:], 1):
            lines.append(f"  {paint(str(i), DIM)} {cmd}")
        return "\n".join(lines)

    def _handle_help(self, args: list[str]) -> str:
        if args:
            want = args[0].lower()
            if want in _COMMANDS:
                usage, desc = _COMMANDS[want]
                return (
                    f"{paint(usage, CYAN + BOLD)}\n"
                    f"  {paint(desc, DIM)}"
                )
            return paint(f"unknown command '{want}'", DIM)
        return _HELP_TEXT

    @staticmethod
    def _parse_n(args: list[str], default: int) -> int:
        for a in args:
            if a.isdigit():
                return max(1, min(100, int(a)))
        return default

    def highlight(self, line: str) -> str:
        """Syntax-highlight an input line: command vs args (god console)."""
        parts = line.split(None, 1)
        if not parts:
            return line
        cmd = parts[0]
        color = CYAN + BOLD if cmd.lower().lstrip("/") in _COMMANDS else BRIGHT_WHITE
        out = paint(cmd, color)
        if len(parts) > 1:
            out += " " + paint(parts[1], DIM)
        return out

    def _render_jobs(self) -> str:
        snap = self._safe_snapshot()
        sched = snap.get("scheduler") or {}
        jobs = sched.get("jobs") or []
        lines = [paint("scheduled jobs", TITLE + BOLD), ""]
        if not jobs:
            lines.append(paint("  no jobs scheduled", DIM))
            lines.append(paint("  (Devon can add them: ask her to schedule a reminder)", DIM))
            return "\n".join(lines)
        from .dashboard import _fmt_ts  # noqa: PLC2701 - internal reuse

        for job in jobs:
            name = str(job.get("name") or job.get("id", "?"))
            spec = str(job.get("spec") or job.get("schedule") or "")
            enabled = job.get("enabled", True)
            nxt = _fmt_ts(job.get("next_run"))
            state = paint("on", CYAN) if enabled else paint("off", DIM)
            lines.append(f"  {paint('›', ACCENT)} {paint(name, BOLD)} [{state}]")
            detail = " · ".join(p for p in (spec, f"next {nxt}") if p)
            if detail:
                lines.append(f"    {paint(detail, DIM)}")
        return "\n".join(lines)
