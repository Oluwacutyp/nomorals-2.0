"""Output themes for the finance module — god-tier presentation, not functional text.

Every render function in this module routes through here. Themes:

  ninja    (default) electric-blue accents, emoji markers — matches the
                      owner's Termux "vrede peace" theme
  plain              no ANSI, no emoji — for logs, pipes, tests
  minimal            text only, tight

Set with the ``FINANCE_THEME`` env var. ``style.bar`` / ``style.sparkline``
/ ``style.table`` are the shared primitives; keep per-module formatting
here, not scattered across render functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = [
    "Theme",
    "THEMES",
    "current_theme",
    "bar",
    "sparkline",
    "table",
    "header",
    "status_dot",
    "pct_color",
]

_SPARK = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class Theme:
    """Named visual style for finance output."""

    name: str
    use_color: bool = True
    use_emoji: bool = True
    # ANSI colors
    accent: str = "\033[96m"      # electric blue
    good: str = "\033[92m"        # green
    warn: str = "\033[93m"        # amber
    bad: str = "\033[91m"         # red
    dim: str = "\033[90m"         # grey
    bold: str = "\033[1m"
    reset: str = "\033[0m"
    # markers
    bullet: str = "•"
    check: str = "✅"
    alert: str = "⚠️"
    money_bag: str = "💰"

    def paint(self, text: str, color: str) -> str:
        if not self.use_color or not color:
            return text
        return f"{color}{text}{self.reset}"


THEMES: dict[str, Theme] = {
    "ninja": Theme(name="ninja"),
    "plain": Theme(
        name="plain", use_color=False, use_emoji=False,
        accent="", good="", warn="", bad="", dim="", bold="", reset="",
        bullet="-", check="[ok]", alert="[!]", money_bag="$",
    ),
    "minimal": Theme(
        name="minimal", use_color=False, use_emoji=False,
        accent="", good="", warn="", bad="", dim="", bold="", reset="",
        bullet="·", check="ok", alert="!!", money_bag="",
    ),
}


def current_theme(name: str | None = None) -> Theme:
    """Active theme — explicit name, else FINANCE_THEME env, else ninja."""
    key = (name or os.environ.get("FINANCE_THEME", "ninja")).strip().lower()
    return THEMES.get(key, THEMES["ninja"])


def bar(pct: float, width: int = 12, theme: Theme | None = None) -> str:
    """Progress bar: ████████░░░░ 67%. Clamps to [0, 1]."""
    theme = theme or current_theme()
    pct = max(0.0, min(1.0, pct))
    filled = int(round(pct * width))
    glyph = "█" * filled + "░" * (width - filled)
    color = theme.bad if pct >= 1.0 else theme.warn if pct >= 0.8 else theme.good
    return theme.paint(glyph, color)


def sparkline(values: list[float], theme: Theme | None = None) -> str:
    """Tiny trend sparkline: ▁▂▃▄▅▆▇. Empty → '—'."""
    theme = theme or current_theme()
    if not values:
        return "—"
    lo, hi = min(values), max(values)
    if hi <= lo:
        return _SPARK[3] * len(values)
    chars = [
        _SPARK[int((v - lo) / (hi - lo) * 7)] for v in values
    ]
    return theme.paint("".join(chars), theme.accent)


def table(
    rows: list[list[str]],
    headers: list[str] | None = None,
    theme: Theme | None = None,
) -> str:
    """Aligned plain-text table. No dependencies, pipe-safe."""
    theme = theme or current_theme()
    data = ([headers] if headers else []) + rows
    if not data:
        return ""
    widths = [0] * max(len(r) for r in data)
    for r in data:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    lines = []
    for ri, r in enumerate(data):
        padded = [c.ljust(widths[i]) for i, c in enumerate(r)]
        line = "  ".join(padded).rstrip()
        if headers is not None and ri == 0:
            line = theme.paint(line, theme.bold)
            lines.append(line)
            lines.append(theme.paint(
                "  ".join("─" * w for w in widths), theme.dim))
        else:
            lines.append(line)
    return "\n".join(lines)


def header(title: str, theme: Theme | None = None) -> str:
    """Section header with accent rule."""
    theme = theme or current_theme()
    return theme.paint(f"━━━ {title} ━━━", theme.accent)


def status_dot(state: str, theme: Theme | None = None) -> str:
    """Colored dot for ok/warning/over states."""
    theme = theme or current_theme()
    dot = "●"
    if state in ("over", "bad", "off"):
        return theme.paint(dot, theme.bad)
    if state in ("warning", "warn"):
        return theme.paint(dot, theme.warn)
    return theme.paint(dot, theme.good)


def pct_color(pct: float, theme: Theme | None = None) -> str:
    """'67%' painted by severity band."""
    theme = theme or current_theme()
    color = theme.bad if pct >= 1.0 else theme.warn if pct >= 0.8 else ""
    return theme.paint(f"{pct:.0%}", color)
