"""Color themes for the Devon console.

Hard rule from the owner: NO black backgrounds, NO red text — in every theme.
Each theme maps semantic roles to ANSI codes. The active theme is chosen via
``NM_CONSOLE_THEME`` (ocean | violet | sunrise); unknown values fall back to
``ocean``. Everything degrades gracefully when color is unsupported.
"""

from __future__ import annotations

import os

from . import palette as _p

# A theme is a flat mapping of semantic role -> ANSI code string.
Theme = dict[str, str]

_OCEAN: Theme = {
    "title": _p.BRIGHT_CYAN,
    "accent": _p.BRIGHT_BLUE,
    "info": _p.CYAN,
    "ok": _p.GREEN,
    "warn": _p.YELLOW,
    "err": _p.MAGENTA,          # magenta, never red
    "crit": _p.BRIGHT_MAGENTA,
    "subtle": _p.GRAY,
    "bright": _p.BRIGHT_WHITE,
    "dim": _p.DIM,
    "bold": _p.BOLD,
}

_VIOLET: Theme = {
    "title": _p.BRIGHT_MAGENTA,
    "accent": _p.MAGENTA,
    "info": _p.BRIGHT_BLUE,
    "ok": _p.BRIGHT_GREEN,
    "warn": _p.BRIGHT_YELLOW,
    "err": _p.MAGENTA,
    "crit": _p.BRIGHT_MAGENTA,
    "subtle": _p.GRAY,
    "bright": _p.BRIGHT_WHITE,
    "dim": _p.DIM,
    "bold": _p.BOLD,
}

_SUNRISE: Theme = {
    "title": _p.BRIGHT_YELLOW,
    "accent": _p.YELLOW,
    "info": _p.BRIGHT_CYAN,
    "ok": _p.BRIGHT_GREEN,
    "warn": _p.CYAN,
    "err": _p.BRIGHT_MAGENTA,   # magenta, never red
    "crit": _p.BRIGHT_MAGENTA,
    "subtle": _p.GRAY,
    "bright": _p.BRIGHT_WHITE,
    "dim": _p.DIM,
    "bold": _p.BOLD,
}

_THEMES: dict[str, Theme] = {
    "ocean": _OCEAN,
    "violet": _VIOLET,
    "sunrise": _SUNRISE,
}

DEFAULT_THEME = "ocean"


def list_themes() -> list[str]:
    """Names of all available themes."""
    return sorted(_THEMES)


def get_theme(name: str | None = None) -> Theme:
    """Resolve a theme by name (env override when ``name`` is None)."""
    key = (name or os.environ.get("NM_CONSOLE_THEME", DEFAULT_THEME)).strip().lower()
    return dict(_THEMES.get(key, _THEMES[DEFAULT_THEME]))


def theme_name() -> str:
    """The currently active theme name."""
    key = os.environ.get("NM_CONSOLE_THEME", DEFAULT_THEME).strip().lower()
    return key if key in _THEMES else DEFAULT_THEME


def paint_t(text: str, theme: Theme, role: str, *, color: bool | None = None) -> str:
    """Paint ``text`` with the theme's color for ``role``."""
    code = theme.get(role, "")
    return _p.paint(text, code, color=color)


def assert_no_banned_codes() -> list[str]:
    """Return any theme codes containing banned sequences (red/black)."""
    banned = ("31m", "40m", "41m")
    bad: list[str] = []
    for name, theme in _THEMES.items():
        for role, code in theme.items():
            if any(b in code for b in banned):
                bad.append(f"{name}.{role}")
    return bad
