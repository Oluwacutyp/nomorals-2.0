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
from .palette import ACCENT, BOLD, CYAN, DIM, GREEN, TITLE, paint
from .themes import get_theme, list_themes, theme_name

SnapshotProvider = Callable[[], dict[str, Any]]

_CLEAR_SEQ = "\033[2J\033[H"

_HELP_TEXT = """console commands (local terminal only):
  dashboard [--watch [secs]]   full status · live split-pane with --watch
                               (in watch: 1=status 2=games 3=jobs 4=brain, q=quit)
  status          one-line compact status
  jobs            scheduled background jobs and next run times
  theme [name]    show/switch palette (ocean · violet · sunrise)
  tip             a random tip of the day
  clear           clear the terminal
  help            this list

anything else is sent to Devon as chat.
"""


class ConsoleCommands:
    """Intercept console lines. ``handle`` returns text to print, or None."""

    def __init__(self, snapshot_provider: SnapshotProvider) -> None:
        self._snapshot = snapshot_provider

    def handle(self, text: str) -> str | None:
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
        if cmd in {"status", "/status"}:
            return render_status_line(self._safe_snapshot())
        if cmd in {"jobs", "schedule", "/jobs"}:
            return self._render_jobs()
        if cmd == "theme":
            return self._handle_theme(args)
        if cmd == "tip":
            return paint(f"💡 {tip_of_the_day()}", CYAN)
        if cmd in {"clear", "cls"}:
            return _CLEAR_SEQ + paint("terminal cleared — type 'dashboard' for status", DIM)
        if cmd in {"help", "/help", "?"}:
            return _HELP_TEXT
        return None

    def _safe_snapshot(self) -> dict[str, Any]:
        try:
            snap = self._snapshot()
            return snap if isinstance(snap, dict) else {}
        except Exception:  # noqa: BLE001 - dashboard must never crash the console
            return {}

    def _watch(self, interval: float) -> str:
        """Blocking split-pane live dashboard. Keys: 1-4 views, q quits."""
        from .widgets import GodScreen

        screen = GodScreen(interval=interval, snapshot=self._safe_snapshot)
        return screen.run()

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
