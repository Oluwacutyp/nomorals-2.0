"""Terminal presentation for the builders module — god-tier, stdlib-only.

``rich`` is the terminal-rendering gold standard (spinners, tables,
panels, graceful TTY degradation), but it is not a dependency here, so
this module reimplements the *pattern*: ANSI themes, box drawing,
status glyphs, and a ``plain`` fallback for non-TTY / ``NO_COLOR``.

Themes:

* ``neon``    — electric accents, the Devon house style
* ``minimal`` — single-accent, quiet
* ``plain``   — no ANSI at all (pipes, logs, tests)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "Theme", "THEMES", "theme_names", "resolve_theme",
    "paint", "banner", "rule", "render_kv", "render_steps",
    "spinner_frames", "status_glyph",
]

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


@dataclass
class Theme:
    """One output theme: named ANSI colors for each semantic role."""

    name: str
    accent: str = ""
    ok: str = ""
    fail: str = ""
    warn: str = ""
    dim: str = ""
    title: str = ""
    border: str = ""
    glyph_ok: str = "✓"
    glyph_fail: str = "✗"
    glyph_skip: str = "○"
    glyph_run: str = "●"

    def paint(self, text: str, color: str) -> str:
        if not color or self.name == "plain":
            return text
        return f"{color}{text}{_RESET}"

    def bold(self, text: str) -> str:
        if self.name == "plain":
            return text
        return f"{_BOLD}{text}{_RESET}"


THEMES: dict[str, Theme] = {
    "neon": Theme(
        name="neon",
        accent="\033[38;5;39m",    # electric blue
        ok="\033[38;5;82m",        # neon green
        fail="\033[38;5;196m",     # hot red
        warn="\033[38;5;220m",     # amber
        dim="\033[38;5;240m",
        title="\033[38;5;45m\033[1m",
        border="\033[38;5;33m",
    ),
    "minimal": Theme(
        name="minimal",
        accent="\033[36m",
        ok="\033[32m",
        fail="\033[31m",
        warn="\033[33m",
        dim="\033[2m",
        title="\033[1m",
        border="\033[2m",
    ),
    "plain": Theme(name="plain"),
}


def theme_names() -> list[str]:
    return list(THEMES)


def _tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def resolve_theme(name: str | None = None) -> Theme:
    """Pick a theme: explicit name > NO_COLOR/BUILDERS_THEME env > TTY sniff."""
    if name:
        return THEMES.get(name, THEMES["neon"])
    if os.environ.get("NO_COLOR"):
        return THEMES["plain"]
    env = os.environ.get("BUILDERS_THEME", "").strip().lower()
    if env in THEMES:
        return THEMES[env]
    return THEMES["neon"] if _tty() else THEMES["plain"]


def paint(text: str, color: str, theme: Theme | None = None) -> str:
    """Paint ``text`` with a raw ANSI color under ``theme`` (no-op if plain)."""
    return (theme or resolve_theme()).paint(text, color)


def banner(title: str, subtitle: str = "",
           theme: Theme | None = None, width: int = 64) -> str:
    """A boxed banner header."""
    th = theme or resolve_theme()
    top = th.paint("┌" + "─" * (width - 2) + "┐", th.border)
    bot = th.paint("└" + "─" * (width - 2) + "┘", th.border)
    body = th.paint("│ ", th.border) + th.bold(th.paint(title, th.title))
    pad = width - 3 - len(title)
    body += " " * max(pad, 0) + th.paint("│", th.border)
    lines = [top, body]
    if subtitle:
        sub = th.paint("│ ", th.border) + th.paint(subtitle, th.dim)
        sub += " " * max(width - 3 - len(subtitle), 0) + th.paint("│", th.border)
        lines.append(sub)
    lines.append(bot)
    return "\n".join(lines)


def rule(label: str = "", theme: Theme | None = None, width: int = 64) -> str:
    th = theme or resolve_theme()
    if not label:
        return th.paint("─" * width, th.border)
    label = f" {label} "
    side = max((width - len(label)) // 2, 1)
    return (th.paint("─" * side, th.border) + th.paint(label, th.dim)
            + th.paint("─" * (width - side - len(label)), th.border))


def render_kv(pairs: Iterable[tuple[str, Any]],
              theme: Theme | None = None) -> str:
    """Aligned ``key: value`` block."""
    th = theme or resolve_theme()
    rows = [(str(k), str(v)) for k, v in pairs]
    width = max((len(k) for k, _ in rows), default=0)
    lines = []
    for k, v in rows:
        lines.append(f"  {th.paint(k.ljust(width), th.accent)}  {v}")
    return "\n".join(lines)


def status_glyph(ok: bool | None, theme: Theme | None = None) -> str:
    """ok=True → ✓, False → ✗, None → ○ (skipped)."""
    th = theme or resolve_theme()
    if ok is True:
        return th.paint(th.glyph_ok, th.ok)
    if ok is False:
        return th.paint(th.glyph_fail, th.fail)
    return th.paint(th.glyph_skip, th.dim)


def render_steps(steps: Iterable[Mapping[str, Any]],
                 theme: Theme | None = None) -> str:
    """Backstage-style step list: glyph + name + timing + first detail line."""
    th = theme or resolve_theme()
    lines = []
    for s in steps:
        ok = s.get("ok")
        name = str(s.get("name", "?"))
        elapsed = s.get("elapsed")
        detail = str(s.get("detail", "") or "").splitlines()
        first = detail[0].strip() if detail else ""
        if len(first) > 90:
            first = first[:87] + "..."
        timing = f"{float(elapsed):.1f}s" if elapsed is not None else ""
        glyph = status_glyph(bool(ok) if ok is not None else None, th)
        line = f"  {glyph} {th.bold(name)}"
        if timing:
            line += f"  {th.paint(timing, th.dim)}"
        if first:
            line += f"  {th.paint(first, th.dim)}"
        lines.append(line)
    return "\n".join(lines)


def spinner_frames(style: str = "dots") -> list[str]:
    """Spinner frames for live progress (dots / line / arc)."""
    return {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line": ["-", "\\", "|", "/"],
        "arc": ["◐", "◓", "◑", "◒"],
    }.get(style, ["-"])
