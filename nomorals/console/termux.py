"""Termux / phone mode: narrow, plain-text, battery-aware rendering.

Constraints that make this different from the fullscreen GodConsole:
- max 80 columns, every line truncated (never wrapped into a flood);
- NO alternate screen, NO cursor hide/show, NO clear-screen — in fact
  NO ANSI escapes at all, so output is safe to scroll back through in
  Termux and to pipe into a file;
- minimal redraw: :meth:`TermuxScreen.render_changed` returns only the
  lines that changed since the last render, so a tick that changes one
  number does not repaint the whole screen;
- battery-aware: when ``battery_low`` is set the screen refreshes less
  often and drops the optional history/sparkline section, saving CPU
  and radio wakeups on a phone running on a low battery.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

MAX_COLS = 80
NORMAL_MIN_INTERVAL_S = 1.0
LOW_BATTERY_MIN_INTERVAL_S = 5.0


def _clip(line: str, width: int = MAX_COLS) -> str:
    return line[:width]


def _fmt_ts(ts: Any) -> str:
    if not ts:
        return ""
    try:
        return time.strftime("%H:%M", time.localtime(float(ts)))
    except (TypeError, ValueError, OSError, OverflowError):
        return str(ts)


def render(state: Mapping[str, Any] | None, *, battery_low: bool = False,
           width: int = MAX_COLS) -> str:
    """Render ``state`` as plain text, no ANSI escapes, max 80 cols.

    ``state`` keys (all optional): ``title``, ``missions`` (list of
    dicts with name/status), ``organs`` (list of dicts with
    name/enabled), ``ledger`` (list of dicts with ts/action/detail),
    ``budget`` (dict with used/limit/unit or fraction), ``history``
    (list of numbers, rendered as a sparkline — dropped when
    ``battery_low``), ``battery_low`` (bool, overrides the kwarg).
    """
    state = state or {}
    low = bool(state.get("battery_low", battery_low))
    w = max(20, min(MAX_COLS, int(width)))
    lines: list[str] = []
    title = str(state.get("title") or "DEVON")
    lines.append(_clip(f"== {title} ==", w))

    missions = state.get("missions") or []
    lines.append(_clip("-- now --", w))
    if not missions:
        lines.append(_clip("  idle", w))
    for m in missions[:5]:
        name = str(m.get("name") or m.get("id") or "?")
        status = str(m.get("status") or "?")
        lines.append(_clip(f"  [{status}] {name}", w))

    organs = state.get("organs") or []
    if organs:
        on = [str(o.get("name")) for o in organs if o.get("enabled")]
        off = [str(o.get("name")) for o in organs if not o.get("enabled")]
        lines.append(_clip(f"  organs on: {', '.join(on) or '-'}", w))
        if off:
            lines.append(_clip(f"  organs off: {', '.join(off)}", w))

    ledger = state.get("ledger") or []
    if ledger:
        lines.append(_clip("-- recent --", w))
        for e in list(ledger)[-3:]:
            lines.append(_clip(
                f"  {_fmt_ts(e.get('ts'))} {e.get('action', '?')} "
                f"{e.get('detail', '')}".rstrip(), w))

    budget = state.get("budget") or {}
    if budget:
        if "fraction" in budget:
            frac = float(budget["fraction"])
        elif budget.get("limit"):
            frac = float(budget.get("used", 0)) / float(budget["limit"])
        else:
            frac = 0.0
        bar_w = 16
        filled = int(round(max(0.0, min(1.0, frac)) * bar_w))
        lines.append(_clip(
            f"  budget [{'#' * filled}{'-' * (bar_w - filled)}] {frac:.0%}",
            w))

    if not low:
        history = state.get("history") or []
        if history:
            blocks = " ▁▂▃▄▅▆▇█"
            nums = [float(x) for x in history]
            lo, hi = min(nums), max(nums)
            span = (hi - lo) or 1.0
            spark = "".join(blocks[min(8, int((v - lo) / span * 8))] for v in nums)
            lines.append(_clip(f"  activity {spark}", w))

    if low:
        lines.append(_clip("  [battery low: slow refresh]", w))
    return "\n".join(lines)


class TermuxScreen:
    """Stateful phone screen: minimal redraw + battery-aware throttling.

    :meth:`render` returns the full plain-text frame.
    :meth:`render_changed` returns only ``(line_no, line)`` pairs whose
    content differs from the previous frame (``[]`` when nothing
    changed), so callers repaint just those lines.
    :meth:`should_render` gates ticks: at most one frame per
    ``min_interval`` (1s normal, 5s on low battery).
    """

    def __init__(self, *, clock: Callable[[], float] | None = None,
                 battery_low: bool = False) -> None:
        self._clock = clock or time.monotonic
        self.battery_low = battery_low
        self._prev_lines: list[str] = []
        self._last_render_at: float | None = None

    @property
    def min_interval(self) -> float:
        return LOW_BATTERY_MIN_INTERVAL_S if self.battery_low else NORMAL_MIN_INTERVAL_S

    def should_render(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        if self._last_render_at is None:
            return True
        return (now - self._last_render_at) >= self.min_interval

    def render(self, state: Mapping[str, Any] | None, *,
               now: float | None = None) -> str:
        low = bool((state or {}).get("battery_low", self.battery_low))
        self.battery_low = low
        out = render(state, battery_low=low)
        self._prev_lines = out.split("\n")
        self._last_render_at = self._clock() if now is None else now
        return out

    def render_changed(self, state: Mapping[str, Any] | None, *,
                       now: float | None = None) -> list[tuple[int, str]]:
        """Only the lines that changed since the last render."""
        out = render(state, battery_low=bool((state or {}).get("battery_low", self.battery_low)))
        new_lines = out.split("\n")
        changed: list[tuple[int, str]] = []
        for i, line in enumerate(new_lines):
            prev = self._prev_lines[i] if i < len(self._prev_lines) else None
            if line != prev:
                changed.append((i, line))
        self._prev_lines = new_lines
        self._last_render_at = self._clock() if now is None else now
        return changed


__all__ = ["MAX_COLS", "TermuxScreen", "render"]
