"""Cockpit: one screen answering "what is she doing right now and why".

Shows, in a single render:
- missions currently running, each with its *reason* (why it exists)
- pause / kill controls for missions
- organ toggles (enable / disable)
- browser daemon status
- ledger view: last N autonomous actions
- budget meter

Cycle safety: this module imports nothing from missions / autonomy /
browser internals. All live data arrives through injected callables (or
plain lists, which are wrapped into callables). The cockpit never knows
*how* a mission runs — only what the provider tells it.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, Mapping, Sequence
from .failures import record_failure

MissionProvider = Callable[[], list[dict[str, Any]]]
OrganProvider = Callable[[], list[dict[str, Any]]]
BrowserProvider = Callable[[], dict[str, Any]]
LedgerProvider = Callable[[], list[dict[str, Any]]]
BudgetProvider = Callable[[], dict[str, Any]]


def _as_provider(value: Any) -> Callable[[], Any] | None:
    """Accept a callable, a list/dict (wrapped), or None."""
    if value is None:
        return None
    if callable(value):
        return value
    snapshot = value
    return lambda: snapshot


def _bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, float(frac)))
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac:.0%}"


class Cockpit:
    """One-screen view of what Devon is doing right now and why.

    All arguments are optional; anything not supplied renders as "n/a"
    instead of raising, so the cockpit is usable with a partial wiring.

    Args:
        missions: callable returning mission dicts, or a static list.
            Each mission: ``{id, name, status, goal, reason, progress}``.
            ``goal``/``reason`` is the "why" — at least one should be set.
        organs: callable returning organ dicts, or a static list.
            Each organ: ``{name, enabled, status}``.
        browser_status: callable returning ``{running, pages, last_action}``.
        ledger: callable returning action dicts, newest last:
            ``{ts, action, detail, mission_id}``.
        budget: callable returning ``{used, limit, unit}`` (fraction used
            may be given as ``used``/``limit`` or directly as ``fraction``).
        on_pause_mission / on_kill_mission: ``fn(mission_id) -> Any``.
        on_toggle_organ: ``fn(name, enabled) -> Any``.
        ledger_limit: how many ledger rows to show by default.
    """

    def __init__(
        self,
        *,
        missions: MissionProvider | Sequence[dict[str, Any]] | None = None,
        organs: OrganProvider | Sequence[dict[str, Any]] | None = None,
        browser_status: BrowserProvider | Mapping[str, Any] | None = None,
        ledger: LedgerProvider | Sequence[dict[str, Any]] | None = None,
        budget: BudgetProvider | Mapping[str, Any] | None = None,
        on_pause_mission: Callable[[str], Any] | None = None,
        on_kill_mission: Callable[[str], Any] | None = None,
        on_toggle_organ: Callable[[str, bool], Any] | None = None,
        ledger_limit: int = 10,
    ) -> None:
        self._missions = _as_provider(missions)
        self._organs = _as_provider(organs)
        self._browser = _as_provider(browser_status)
        self._ledger = _as_provider(ledger)
        self._budget = _as_provider(budget)
        self._on_pause = on_pause_mission
        self._on_kill = on_kill_mission
        self._on_toggle = on_toggle_organ
        self.ledger_limit = max(1, int(ledger_limit))
        self._lock = threading.RLock()
        self.last_action: str = ""

    # ── data access (providers, never internals) ──

    def missions(self) -> list[dict[str, Any]]:
        if self._missions is None:
            return []
        try:
            rows = self._missions() or []
            return [dict(r) for r in rows]
        except Exception as e:
            record_failure('cockpit.Cockpit.missions', e)
            return []

    def organs(self) -> list[dict[str, Any]]:
        if self._organs is None:
            return []
        try:
            rows = self._organs() or []
            return [dict(r) for r in rows]
        except Exception as e:
            record_failure('cockpit.Cockpit.organs', e)
            return []

    def browser(self) -> dict[str, Any]:
        if self._browser is None:
            return {}
        try:
            return dict(self._browser() or {})
        except Exception as e:
            record_failure('cockpit.Cockpit.browser', e)
            return {}

    def ledger_entries(self, n: int | None = None) -> list[dict[str, Any]]:
        if self._ledger is None:
            return []
        try:
            rows = self._ledger() or []
        except Exception as e:
            record_failure('cockpit.Cockpit.ledger_entries', e)
            return []
        rows = [dict(r) for r in rows]
        return rows[-max(1, n or self.ledger_limit):]

    def budget_state(self) -> dict[str, Any]:
        if self._budget is None:
            return {}
        try:
            return dict(self._budget() or {})
        except Exception as e:
            record_failure('cockpit.Cockpit.budget_state', e)
            return {}

    # ── "why" ──

    def why(self, mission_id: str) -> str:
        """Explain why a mission is running, from its own goal/reason."""
        for m in self.missions():
            if str(m.get("id")) == str(mission_id):
                reason = m.get("reason") or m.get("goal") or m.get("why") or ""
                name = m.get("name") or mission_id
                status = m.get("status") or "?"
                if reason:
                    return f"{name} [{status}]: {reason}"
                return f"{name} [{status}]: (no reason recorded)"
        return f"unknown mission '{mission_id}'"

    # ── controls (delegate to injected callbacks) ──

    def pause_mission(self, mission_id: str) -> Any:
        """Pause a mission via the injected callback. Returns its result."""
        if self._on_pause is None:
            raise RuntimeError("cockpit has no on_pause_mission callback wired")
        with self._lock:
            result = self._on_pause(str(mission_id))
            self.last_action = f"paused mission {mission_id}"
            return result

    def kill_mission(self, mission_id: str) -> Any:
        """Kill a mission via the injected callback. Returns its result."""
        if self._on_kill is None:
            raise RuntimeError("cockpit has no on_kill_mission callback wired")
        with self._lock:
            result = self._on_kill(str(mission_id))
            self.last_action = f"killed mission {mission_id}"
            return result

    def toggle_organ(self, name: str, enabled: bool | None = None) -> Any:
        """Enable/disable an organ. ``enabled=None`` flips its current state."""
        if self._on_toggle is None:
            raise RuntimeError("cockpit has no on_toggle_organ callback wired")
        if enabled is None:
            current = next(
                (o for o in self.organs() if str(o.get("name")) == str(name)),
                None,
            )
            enabled = not bool(current.get("enabled")) if current else True
        with self._lock:
            result = self._on_toggle(str(name), bool(enabled))
            self.last_action = f"organ {name} -> {'on' if enabled else 'off'}"
            return result

    # ── render ──

    def render(self, *, ledger_n: int | None = None) -> str:
        """One screen: what she is doing, why, and the controls' state."""
        lines: list[str] = ["== DEVON COCKPIT — what is she doing, and why =="]

        # Missions: the "what + why" core.
        lines.append("-- missions --")
        missions = self.missions()
        if not missions:
            lines.append("  (none)")
        for m in missions:
            mid = m.get("id", "?")
            name = m.get("name") or mid
            status = m.get("status") or "?"
            reason = m.get("reason") or m.get("goal") or ""
            prog = m.get("progress")
            prog_txt = f" {prog:.0%}" if isinstance(prog, (int, float)) else ""
            lines.append(f"  [{status}] {name} (id={mid}){prog_txt}")
            if reason:
                lines.append(f"      why: {reason}")

        # Organs.
        lines.append("-- organs --")
        organs = self.organs()
        if not organs:
            lines.append("  (none)")
        for o in organs:
            state = "ON " if o.get("enabled") else "OFF"
            extra = f" — {o.get('status')}" if o.get("status") else ""
            lines.append(f"  [{state}] {o.get('name', '?')}{extra}")

        # Browser daemon.
        lines.append("-- browser daemon --")
        b = self.browser()
        if not b:
            lines.append("  n/a")
        else:
            running = "running" if b.get("running") else "stopped"
            lines.append(f"  {running} · pages={b.get('pages', '?')}")
            if b.get("last_action"):
                lines.append(f"  last: {b['last_action']}")

        # Ledger: last N autonomous actions.
        lines.append(f"-- ledger (last {ledger_n or self.ledger_limit} autonomous actions) --")
        entries = self.ledger_entries(ledger_n)
        if not entries:
            lines.append("  (empty)")
        for e in entries:
            ts = e.get("ts", "")
            action = e.get("action", "?")
            detail = e.get("detail", "")
            mid = e.get("mission_id")
            suffix = f" [{mid}]" if mid else ""
            lines.append(f"  {ts} {action}{suffix} {detail}".rstrip())

        # Budget meter.
        lines.append("-- budget --")
        bg = self.budget_state()
        if not bg:
            lines.append("  n/a")
        else:
            if "fraction" in bg:
                frac = float(bg["fraction"])
            elif bg.get("limit"):
                frac = float(bg.get("used", 0)) / float(bg["limit"])
            else:
                frac = 0.0
            unit = bg.get("unit", "")
            used = bg.get("used", "")
            limit = bg.get("limit", "")
            lines.append(f"  {_bar(frac)} {used}/{limit} {unit}".rstrip())

        if self.last_action:
            lines.append(f"-- last cockpit action: {self.last_action} --")
        return "\n".join(lines)


__all__ = ["Cockpit"]
