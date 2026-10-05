"""Console-only commands for the local terminal adapter.

These are intercepted *before* a line reaches the brain, so typing
``dashboard`` in the console shows the dashboard instead of chatting about
dashboards. Anything unrecognized returns ``None`` and flows through to the
bot normally — console input is never broken by this.
"""

from __future__ import annotations

from typing import Any, Callable

from .dashboard import render_dashboard, render_status_line
from .palette import ACCENT, BOLD, CYAN, DIM, TITLE, paint

SnapshotProvider = Callable[[], dict[str, Any]]

_CLEAR_SEQ = "\033[2J\033[H"

_HELP_TEXT = """console commands (local terminal only):
  dashboard   full live status dashboard
  status      one-line compact status
  jobs        scheduled background jobs and next run times
  clear       clear the terminal
  help        this list

anything else is sent to Devon as chat.
"""


class ConsoleCommands:
    """Intercept console lines. ``handle`` returns text to print, or None."""

    def __init__(self, snapshot_provider: SnapshotProvider) -> None:
        self._snapshot = snapshot_provider

    def handle(self, text: str) -> str | None:
        cmd = (text or "").strip().lower()
        if cmd in {"dashboard", "dash", "/dashboard"}:
            return render_dashboard(self._safe_snapshot())
        if cmd in {"status", "/status"}:
            return render_status_line(self._safe_snapshot())
        if cmd in {"jobs", "schedule", "/jobs"}:
            return self._render_jobs()
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
