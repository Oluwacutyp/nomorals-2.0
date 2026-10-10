"""Shared presentation layer for core status/report rendering.

Every ``format_*`` / ``describe`` helper in this package renders through
here so chat surfaces, CLIs, and dashboards get one consistent visual
language instead of 41 ad-hoc formats.

Design rules (from the mining pass):
* Themes are named palettes, never inline escape codes at call sites.
* The owner hates red: errors render magenta, never red, in every theme.
* ``plain`` disables all color (logs, pipes, files).
* Helpers degrade gracefully: no tty → no color, narrow width → wrap.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "THEMES",
    "Theme",
    "active_theme",
    "bar",
    "header",
    "kv_lines",
    "paint",
    "set_theme",
    "sparkline",
    "status_dot",
    "styled_box",
    "styled_table",
    "supports_color",
    "theme_names",
]

#: Semantic roles — call sites name the *meaning*, themes pick the color.
_ROLES = (
    "ok", "warn", "error", "info", "muted", "accent",
    "header", "border", "value", "label",
)


@dataclass(frozen=True)
class Theme:
    """A named palette: role → ANSI escape (empty = no color)."""

    name: str
    palette: Mapping[str, str] = field(default_factory=dict)
    reset: str = "\033[0m"
    bold: str = "\033[1m"
    dim: str = "\033[2m"

    def color(self, role: str) -> str:
        return self.palette.get(role, "")


def _palette(**roles: str) -> dict[str, str]:
    return {role: roles.get(role, "") for role in _ROLES}


THEMES: dict[str, Theme] = {
    # The user's Termux theme: ninja "vrede peace" — electric blue on dark.
    "ninja": Theme("ninja", _palette(
        ok="\033[38;5;82m",       # neon green
        warn="\033[38;5;214m",    # amber
        error="\033[38;5;201m",   # magenta — never red
        info="\033[38;5;39m",     # electric blue
        muted="\033[38;5;240m",
        accent="\033[38;5;51m",   # cyan-blue
        header="\033[1;38;5;39m",
        border="\033[38;5;27m",
        value="\033[38;5;255m",
        label="\033[38;5;75m",
    )),
    # Warm dark-console default.
    "ember": Theme("ember", _palette(
        ok="\033[32m",
        warn="\033[33m",
        error="\033[35m",         # magenta — never red
        info="\033[36m",
        muted="\033[90m",
        accent="\033[96m",
        header="\033[1;37m",
        border="\033[90m",
        value="\033[97m",
        label="\033[36m",
    )),
    # Light-background friendly.
    "paper": Theme("paper", _palette(
        ok="\033[32m",
        warn="\033[33m",
        error="\033[35m",
        info="\033[34m",
        muted="\033[90m",
        accent="\033[34m",
        header="\033[1;30m",
        border="\033[90m",
        value="\033[30m",
        label="\033[34m",
    )),
    # No escape codes at all — for files, pipes, and log aggregators.
    "plain": Theme("plain", _palette(), reset="", bold="", dim=""),
}

_ACTIVE: dict[str, Theme] = {"theme": THEMES["ember"]}


def theme_names() -> list[str]:
    return sorted(THEMES)


def set_theme(name: str) -> Theme:
    """Switch the active theme process-wide. Returns the theme."""
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; choose from {theme_names()}")
    _ACTIVE["theme"] = THEMES[name]
    return THEMES[name]


def active_theme() -> Theme:
    return _ACTIVE["theme"]


def supports_color(stream: Any = None) -> bool:
    """True when color output is wanted on ``stream`` (default stderr)."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream if stream is not None else sys.stderr
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001 - exotic streams
        return False


def paint(text: str, role: str, theme: Theme | None = None,
          *, color: bool | None = None) -> str:
    """Color ``text`` by semantic role. No color when disabled."""
    theme = theme or active_theme()
    if color is None:
        color = theme is not THEMES["plain"] and supports_color()
    if not color:
        return text
    code = theme.color(role)
    if not code:
        return text
    return f"{code}{text}{theme.reset}"


def status_dot(status: str, theme: Theme | None = None, **kw: Any) -> str:
    """●/▲/■ glyph colored by health status."""
    mapping = {
        "ok": ("●", "ok"), "healthy": ("●", "ok"), "pass": ("●", "ok"),
        "warn": ("▲", "warn"), "warning": ("▲", "warn"),
        "degraded": ("▲", "warn"), "slow": ("▲", "warn"),
        "error": ("■", "error"), "fail": ("■", "error"),
        "failed": ("■", "error"), "critical": ("■", "error"),
        "down": ("■", "error"), "exhausted": ("■", "error"),
        "unknown": ("○", "muted"), "idle": ("○", "muted"),
        "frozen": ("❄", "info"), "frozen_ok": ("❄", "info"),
    }
    glyph, role = mapping.get(status.lower(), ("○", "muted"))
    return paint(glyph, role, theme, **kw)


def _width() -> int:
    try:
        return max(40, shutil.get_terminal_size().columns)
    except Exception:  # noqa: BLE001
        return 80


def header(title: str, theme: Theme | None = None, **kw: Any) -> str:
    """A section header: ── title ──…"""
    theme = theme or active_theme()
    width = _width()
    bar_len = max(3, width - len(title) - 5)
    rule = paint("─" * bar_len, "border", theme, **kw)
    return f"{paint(title, 'header', theme, **kw)} {rule}"


def styled_box(title: str, lines: Sequence[str], theme: Theme | None = None,
               **kw: Any) -> str:
    """A bordered box with a title."""
    theme = theme or active_theme()
    width = min(_width(), max([len(title)] + [len(l) for l in lines] + [20]))
    top = paint("╭" + "─" * (width - 2) + "╮", "border", theme, **kw)
    bottom = paint("╰" + "─" * (width - 2) + "╯", "border", theme, **kw)
    out = [top, paint(f"│ {title}", "header", theme, **kw).ljust(width + 9)[: width] + paint("│", "border", theme, **kw)]
    for line in lines:
        body = line[: width - 4]
        out.append(paint("│ ", "border", theme, **kw) + body.ljust(width - 4) + paint(" │", "border", theme, **kw))
    out.append(bottom)
    return "\n".join(out)


def styled_table(headers: Sequence[str], rows: Iterable[Sequence[Any]],
                  theme: Theme | None = None, **kw: Any) -> str:
    """A compact aligned table. Values are str()'d; keep cells short."""
    theme = theme or active_theme()
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], min(len(cell), 48))
    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(
            cell[: widths[i]].ljust(widths[i]) for i, cell in enumerate(cells))
    lines = [paint(fmt(list(headers)), "label", theme, **kw)]
    lines.append(paint("─" * min(sum(widths) + 2 * (len(widths) - 1), _width()),
                       "border", theme, **kw))
    for row in str_rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append(fmt(padded[: len(headers)]))
    return "\n".join(lines)


def kv_lines(mapping: Mapping[str, Any], theme: Theme | None = None,
             **kw: Any) -> list[str]:
    """Aligned ``key: value`` lines for status dicts."""
    theme = theme or active_theme()
    items = [(str(k), str(v)) for k, v in mapping.items()]
    width = max((len(k) for k, _ in items), default=0)
    return [f"{paint(k.ljust(width), 'label', theme, **kw)}  "
            f"{paint(v, 'value', theme, **kw)}" for k, v in items]


_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], theme: Theme | None = None,
              **kw: Any) -> str:
    """Tiny text sparkline for a series of numbers."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    glyphs = "".join(_SPARK[min(7, int((v - lo) / span * 7))] for v in values)
    role = "error" if values and values[-1] >= hi and len(values) > 3 else "info"
    return paint(glyphs, role, theme, **kw)


def bar(fraction: float, width: int = 20, theme: Theme | None = None,
        **kw: Any) -> str:
    """A small progress bar for a 0.0–1.0 fraction."""
    filled = max(0, min(width, int(round(fraction * width))))
    role = "ok" if fraction > 0.5 else "warn" if fraction > 0.2 else "error"
    body = paint("█" * filled, role, theme, **kw) + paint(
        "░" * (width - filled), "muted", theme, **kw)
    return f"[{body}] {fraction:.0%}"
