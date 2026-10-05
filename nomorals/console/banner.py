"""Startup banner for the Devon console: ASCII logo, version, adapters, tip.

Pure functions so the banner is testable without a running bot.
"""

from __future__ import annotations

import random

from .palette import (
    BOLD,
    BRIGHT_WHITE,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    TITLE,
    paint,
    strip_ansi,
)
from .themes import Theme, get_theme

_LOGO = [
    r"██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗",
    r"██╔══██╗██╔════╝██║   ██║██╔═══██╗████╗  ██║",
    r"██║  ██║█████╗  ██║   ██║██║   ██║██╔██╗ ██║",
    r"██║  ██║██╔══╝  ╚██╗ ██╔╝██║   ██║██║╚██╗██║",
    r"██████╔╝███████╗ ╚████╔╝ ╚██████╔╝██║ ╚████║",
    r"╚═════╝ ╚══════╝  ╚═══╝   ╚═════╝ ╚═╝  ╚═══╝",
]

_TIPS = [
    "type 'dashboard --watch' for a live auto-refreshing status screen.",
    "type 'jobs' to see scheduled background tasks and their next run.",
    "Devon learns your @usernames — /gift @name works across endpoints.",
    "set NM_CONSOLE_THEME=violet (or sunrise) for a different palette.",
    "quiet hours keep proactive messages polite — check the scheduler.",
    "type 'theme <name>' to switch palettes without restarting.",
    "/game delete soft-deletes a profile — 48h to undo, then it's gone.",
    "the console never sends your typing to the brain — it's local only.",
]


def tip_of_the_day(seed: int | None = None) -> str:
    """Deterministic-ish tip; ``seed`` pins it (tests), else day-based."""
    import datetime

    if seed is None:
        seed = int(datetime.date.today().strftime("%Y%m%d"))
    rng = random.Random(seed)
    return rng.choice(_TIPS)


def render_banner(
    adapters: list[str],
    *,
    version: str = "",
    theme: Theme | None = None,
    tip_seed: int | None = None,
    color: bool | None = None,
) -> str:
    """Render the full startup banner."""
    theme = theme or get_theme()
    t = lambda s, role: paint(s, theme[role], color=color)  # noqa: E731
    lines: list[str] = []
    lines.append("")
    for row in _LOGO:
        lines.append("  " + t(row, "title"))
    tag = "chat gateway live"
    if version:
        tag += f" · v{version}"
    lines.append("  " + t(tag, "subtle"))
    lines.append("")
    for name in adapters:
        lines.append(
            f"  {t('●', 'ok')} {t(name, 'bright')} {t('connected', 'subtle')}"
        )
    lines.append("")
    lines.append(
        f"  {t('console:', 'accent')} {t('dashboard · status · jobs · clear · help', 'info')}"
    )
    lines.append(f"  {t('theme:', 'accent')} {t(theme_name_label(theme), 'info')}")
    lines.append(f"  {t('tip:', 'accent')} {t(tip_of_the_day(tip_seed), 'subtle')}")
    lines.append("")
    out = "\n".join(lines)
    return strip_ansi(out) if color is False else out


def theme_name_label(theme: Theme) -> str:
    """Best-effort label for a theme dict (used in the banner)."""
    from . import themes as _t

    for name, cand in (("ocean", _t._OCEAN), ("violet", _t._VIOLET), ("sunrise", _t._SUNRISE)):
        if cand == theme:
            return name
    return "custom"


__all__ = ["render_banner", "tip_of_the_day"]
