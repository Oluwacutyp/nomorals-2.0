"""Shared output styling for tool-facing text.

Stdlib-only by design: ``rich`` is used when importable, but nothing here
requires it. Every helper degrades gracefully when stdout is not a TTY
(ANSI stripped) and honors ``NO_COLOR`` / ``TERM=dumb``.

Themes: ``ninja`` (default — electric-blue ninja accents, the Devon
house style), ``plain`` (no color, for logs/pipes), ``mono`` (glyphs
only, no color codes).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from typing import Any, Iterable, Sequence

__all__ = [
    "Theme",
    "get_theme",
    "set_theme",
    "supports_color",
    "colorize",
    "strip_ansi",
    "status_line",
    "kv_block",
    "make_table",
    "panel",
    "progress_bar",
    "banner",
    "rich_console",
    "OK",
    "FAIL",
    "WARN",
    "INFO",
    "PENDING",
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

OK = "ok"
FAIL = "fail"
WARN = "warn"
INFO = "info"
PENDING = "pending"

_STATUS_GLYPH = {
    OK: "✓",
    FAIL: "✗",
    WARN: "⚠",
    INFO: "ℹ",
    PENDING: "…",
}

_STATUS_COLOR = {
    OK: "green",
    FAIL: "red",
    WARN: "yellow",
    INFO: "blue",
    PENDING: "cyan",
}

# ── themes ─────────────────────────────────────────────────────────────────

_THEMES: dict[str, dict[str, str]] = {
    "ninja": {  # electric-blue ninja — Devon house style
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
        "bright_blue": "94", "bright_cyan": "96", "bright_black": "90",
        "bold": "1", "dim": "2", "accent": "94", "heading": "1;94",
    },
    "plain": {},   # no colors at all
    "mono": {},    # glyphs, no color codes
}

_current_theme = "ninja"


def get_theme() -> str:
    return _current_theme


def set_theme(name: str) -> str:
    """Switch the active theme. Unknown names are ignored. Returns the
    theme actually in effect."""
    global _current_theme
    if name in _THEMES:
        _current_theme = name
    return _current_theme


class Theme:
    """Context manager that temporarily switches the theme."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._prev = _current_theme

    def __enter__(self) -> "Theme":
        set_theme(self.name)
        return self

    def __exit__(self, *exc: Any) -> None:
        set_theme(self._prev)


def supports_color() -> bool:
    """True when ANSI colors should be emitted."""
    if _current_theme in ("plain", "mono"):
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def colorize(text: str, *codes: str) -> str:
    """Wrap ``text`` in the theme's ANSI codes. No-op when color is off."""
    if not supports_color() or not codes:
        return text
    palette = _THEMES[_current_theme]
    seq = ";".join(palette.get(c, c) for c in codes)
    return f"\x1b[{seq}m{text}\x1b[0m"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _glyph(status: str) -> str:
    return _STATUS_GLYPH.get(status, "•")


def status_line(status: str, message: str) -> str:
    """One status line: ``✓ message`` / ``✗ message`` / …"""
    glyph = _glyph(status)
    if supports_color():
        glyph = colorize(glyph, _STATUS_COLOR.get(status, "white"))
    return f"{glyph} {message}"


def kv_block(pairs: Iterable[tuple[str, Any]], *, title: str = "",
             indent: int = 2) -> str:
    """Compact key: value block with aligned keys."""
    rows = [(str(k), str(v)) for k, v in pairs]
    if not rows:
        return title
    width = max(len(k) for k, _ in rows)
    pad = " " * indent
    lines = []
    if title:
        lines.append(colorize(title, "heading"))
    for key, value in rows:
        label = colorize(key.ljust(width), "bright_cyan")
        lines.append(f"{pad}{label}  {value}")
    return "\n".join(lines)


def make_table(headers: Sequence[str],
               rows: Iterable[Sequence[Any]],
               *,
               title: str = "",
               max_width: int | None = None) -> str:
    """Aligned plain-text table (no borders — scannable in chat)."""
    data = [[str(c) for c in row] for row in rows]
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in data:
        for i in range(cols):
            cell = row[i] if i < len(row) else ""
            widths[i] = max(widths[i], len(cell))
    total = sum(widths) + 2 * (cols - 1)
    width = max_width or shutil.get_terminal_size((100, 20)).columns
    if total > width and cols > 1:
        # shrink the widest column first
        widest = max(range(cols), key=lambda i: widths[i])
        widths[widest] = max(8, widths[widest] - (total - width))
    lines: list[str] = []
    if title:
        lines.append(colorize(title, "heading"))
    head = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(colorize(head, "bold"))
    lines.append(colorize("-" * min(len(head), width), "dim"))
    for row in data:
        cells = []
        for i in range(cols):
            cell = row[i] if i < len(row) else ""
            if len(cell) > widths[i]:
                cell = cell[: max(0, widths[i] - 1)] + "…"
            cells.append(cell.ljust(widths[i]))
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def panel(body: str, *, title: str = "", width: int | None = None) -> str:
    """Boxed content with a title bar."""
    w = width or min(shutil.get_terminal_size((80, 20)).columns, 100)
    inner = max(10, w - 4)
    lines = [title] if title else []
    lines.extend(body.splitlines() or [""])
    out: list[str] = []
    top = "╭" + "─" * inner + "╮"
    bottom = "╰" + "─" * inner + "╯"
    if title:
        label = f" {title} "
        top = "╭" + colorize(label, "heading") + "─" * max(0, inner - len(label)) + "╮"
    out.append(colorize(top, "accent") if supports_color() else top.replace(
        "╭", "+").replace("╮", "+").replace("─", "-"))
    for line in lines[1:] if title else lines:
        chunks = [line[i:i + inner] for i in range(0, len(line), inner)] or [""]
        for chunk in chunks:
            bar_l, bar_r = ("│", "│") if supports_color() else ("|", "|")
            out.append(
                (colorize(bar_l, "accent") if supports_color() else bar_l)
                + " " + chunk.ljust(inner - 1)
                + (colorize(bar_r, "accent") if supports_color() else bar_r))
    out.append(colorize(bottom, "accent") if supports_color()
               else bottom.replace("╰", "+").replace("╯", "+").replace("─", "-"))
    return "\n".join(out)


def progress_bar(frac: float, *, width: int = 24, label: str = "") -> str:
    """``██████░░░░░░ 62% label`` — clamps frac to [0, 1]."""
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    if supports_color():
        bar = colorize("█" * filled, "green") + colorize("░" * (width - filled), "dim")
    pct = f"{frac * 100:5.1f}%"
    return f"{bar} {pct} {label}".rstrip()


def banner(text: str, *, sub: str = "") -> str:
    """Section header for reports."""
    line = colorize("━" * max(8, len(text) + 4), "accent")
    head = colorize(f"  {text}", "heading")
    out = f"{line}\n{head}\n{line}"
    if sub:
        out += f"\n{colorize(sub, 'dim')}"
    return out


def rich_console() -> Any | None:
    """A ``rich.console.Console`` when rich is importable, else None.

    Callers that want rich Tables/Panels/Progress should go through this
    and fall back to the stdlib helpers above.
    """
    try:
        from rich.console import Console  # type: ignore

        return Console()
    except Exception:  # noqa: BLE001 — rich is optional
        return None
