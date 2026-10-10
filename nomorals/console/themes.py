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

#: Full semantic role contract. The original 11 roles keep their meaning;
#: the extra roles are filled for every theme (old themes derive them).
ROLES = (
    "title", "accent", "info", "ok", "warn", "err", "crit",
    "subtle", "bright", "dim", "bold",
    "border", "muted", "hl", "sel", "link", "code", "chart", "bar",
)

#: Hex palettes for the truecolor themes. Canonical upstreams:
#: tokyo-night (folke/tokyonight.nvim, night), nord (nordtheme/nord),
#: dracula (dracula/dracula-theme), catppuccin (catppuccin/catppuccin, mocha).
#: The owner's hard rule holds in every theme: NO red text, NO black
#: backgrounds — the upstream "red" slot is always remapped to the
#: magenta/pink family, and no background codes are emitted anywhere.
_HEX = {
    "tokyo-night": {
        "title": "#7dcfff", "accent": "#7aa2f7", "info": "#7dcfff",
        "ok": "#9ece6a", "warn": "#e0af68", "err": "#bb9af7",
        "crit": "#ff9e64", "subtle": "#565f89", "bright": "#c0caf5",
        "border": "#3b4261", "muted": "#9aa5ce", "hl": "#283457",
        "sel": "#7aa2f7", "link": "#7dcfff", "code": "#9ece6a",
        "chart": "#7aa2f7", "bar": "#9ece6a",
    },
    "nord": {
        "title": "#88c0d0", "accent": "#81a1c1", "info": "#88c0d0",
        "ok": "#a8c69a", "warn": "#ebcb8b", "err": "#b48ead",
        "crit": "#d08770", "subtle": "#4c566a", "bright": "#eceff4",
        "border": "#3b4252", "muted": "#8f9bb0", "hl": "#434c5e",
        "sel": "#81a1c1", "link": "#88c0d0", "code": "#a8c69a",
        "chart": "#81a1c1", "bar": "#a8c69a",
    },
    "dracula": {
        "title": "#8be9fd", "accent": "#bd93f9", "info": "#8be9fd",
        "ok": "#50fa7b", "warn": "#f2f0a1", "err": "#ff79c6",
        "crit": "#ffb86c", "subtle": "#6272a4", "bright": "#f8f8f2",
        "border": "#44475a", "muted": "#a8b0d0", "hl": "#44475a",
        "sel": "#bd93f9", "link": "#8be9fd", "code": "#50fa7b",
        "chart": "#bd93f9", "bar": "#50fa7b",
    },
    "catppuccin": {
        "title": "#89dceb", "accent": "#cba6f7", "info": "#89dceb",
        "ok": "#a6e3a1", "warn": "#f9e2af", "err": "#cba6f7",
        "crit": "#eba0ac", "subtle": "#6c7086", "bright": "#cdd6f4",
        "border": "#313244", "muted": "#a6adc8", "hl": "#313244",
        "sel": "#cba6f7", "link": "#89dceb", "code": "#a6e3a1",
        "chart": "#89b4fa", "bar": "#a6e3a1",
    },
}


def _hex_theme(name: str) -> Theme:
    """Build a Theme from the hex palette, adding dim/bold attrs."""
    spec = _HEX[name]
    theme: Theme = {role: _p.hex_color(spec[role]) for role in spec}
    theme["dim"] = _p.DIM
    theme["bold"] = _p.BOLD
    # "dim" needs a companion dimmed-text color for roles painted dim;
    # keep the plain DIM attribute (works on any terminal).
    return theme


def _extend_ansi(base: Theme) -> Theme:
    """Fill the extended roles for the legacy 16-color themes."""
    theme = dict(base)
    theme.setdefault("border", theme.get("subtle", ""))
    theme.setdefault("muted", theme.get("subtle", ""))
    theme.setdefault("hl", theme.get("accent", ""))
    theme.setdefault("sel", theme.get("accent", ""))
    theme.setdefault("link", theme.get("info", ""))
    theme.setdefault("code", theme.get("ok", ""))
    theme.setdefault("chart", theme.get("accent", ""))
    theme.setdefault("bar", theme.get("ok", ""))
    return theme


_THEMES: dict[str, Theme] = {
    "ocean": _extend_ansi(_OCEAN),
    "violet": _extend_ansi(_VIOLET),
    "sunrise": _extend_ansi(_SUNRISE),
    "tokyo-night": _hex_theme("tokyo-night"),
    "nord": _hex_theme("nord"),
    "dracula": _hex_theme("dracula"),
    "catppuccin": _hex_theme("catppuccin"),
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


def describe_theme(name: str | None = None) -> dict[str, str]:
    """Name, roles and raw codes for a theme (for docs/previews)."""
    key = (name or theme_name()).strip().lower()
    key = key if key in _THEMES else DEFAULT_THEME
    return {"name": key, **{f"role.{r}": c for r, c in _THEMES[key].items()}}


def validate_theme(theme: Theme) -> list[str]:
    """Return problems with a theme dict: missing roles, banned codes."""
    problems: list[str] = []
    for role in ROLES:
        if role not in theme:
            problems.append(f"missing role: {role}")
    for role, code in theme.items():
        if any(b in code for b in ("31m", "40m", "41m")):
            problems.append(f"banned code in {role}: {code!r}")
    return problems


def theme_preview(name: str | None = None, *, color: bool | None = None) -> str:
    """A swatch card showing every role of a theme, painted live.

    Like the ``theme`` command's swatch, but for the whole role contract:
    ``title ▓▓ · accent ▓▓ · …``. Used by the ``palette`` console command.
    """
    key = (name or theme_name()).strip().lower()
    key = key if key in _THEMES else DEFAULT_THEME
    theme = _THEMES[key]
    lines = [_p.paint(f"theme · {key}", theme["title"] + _p.BOLD, color=color), ""]
    row: list[str] = []
    for role in ROLES:
        code = theme.get(role, "")
        cell = f"{_p.paint(role, code, color=color)} {_p.paint('▓▓', code, color=color)}"
        row.append(cell)
        if len(row) == 3:
            lines.append("  ".join(f"{c:<34}" for c in row))
            row = []
    if row:
        lines.append("  ".join(f"{c:<34}" for c in row))
    out = "\n".join(lines)
    return _p.strip_ansi(out) if color is False else out


def assert_no_banned_codes() -> list[str]:
    """Return any theme codes containing banned sequences (red/black)."""
    banned = ("31m", "40m", "41m")
    bad: list[str] = []
    for name, theme in _THEMES.items():
        for role, code in theme.items():
            if any(b in code for b in banned):
                bad.append(f"{name}.{role}")
    return bad
