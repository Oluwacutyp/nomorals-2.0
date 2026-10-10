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
from .themes import Theme, get_theme, theme_name

_LOGO = [
    r"██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗",
    r"██╔══██╗██╔════╝██║   ██║██╔═══██╗████╗  ██║",
    r"██║  ██║█████╗  ██║   ██║██║   ██║██╔██╗ ██║",
    r"██║  ██║██╔══╝  ╚██╗ ██╔╝██║   ██║██║╚██╗██║",
    r"██████╔╝███████╗ ╚████╔╝ ╚██████╔╝██║ ╚████║",
    r"╚═════╝ ╚══════╝  ╚═══╝   ╚═════╝ ╚═╝  ╚═══╝",
]

#: Hand-tuned slant-style "DEVON" (figlet slant idiom, stdlib-only).
_SLANT = [
    r"    ____  _______  ______  _   __",
    r"   / __ \/ ____/ | / / __ \/ | / /",
    r"  / / / / __/ /  |/ / / / /  |/ / ",
    r" / /_/ / /___/ /|  / /_/ / /|  /  ",
    r"/_____/_____/_/ |_/\____/_/ |_/   ",
]

#: Compact 2-row box-drawing "DEVON" for narrow terminals.
_MINI = [
    r"┌┬┐┌─┐┬  ┬┌─┐┌┐┌",
    r" ││├┤ └┐┌┘│ ││││",
    r"─┴┘└─┘ └┘ └─┘┘└┘",
]

#: Shaded pixel-gradient "DEVON" (░▒▓█ retro idiom).
_PIXEL = [
    r"▓▓▓▓▓▓ ▓▓▓▓▓▓▓ ▓▓   ▓▓ ▓▓▓▓▓▓ ▓▓▓   ▓▓",
    r"▓▓   ▓▓ ▓▓     ▓▓   ▓▓ ▓▓   ▓▓ ▓▓▓▓  ▓▓",
    r"▓▓   ▓▓ ▓▓▓▓▓  ▓▓   ▓▓ ▓▓   ▓▓ ▓▓ ▓▓ ▓▓",
    r"▓▓   ▓▓ ▓▓      ▓▓ ▓▓  ▓▓   ▓▓ ▓▓  ▓▓▓▓",
    r"▓▓▓▓▓▓ ▓▓▓▓▓▓▓  ▓▓▓▓▓   ▓▓▓▓▓▓  ▓▓   ▓▓",
]

_BANNER_FONTS: dict[str, list[str]] = {
    "block": _LOGO,
    "slant": _SLANT,
    "mini": _MINI,
    "pixel": _PIXEL,
}

#: All banner styles, in display order.
BANNER_STYLES = ("block", "slant", "mini", "pixel")

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


def list_banner_styles() -> list[str]:
    """Names of all banner styles."""
    return list(BANNER_STYLES)


def _gradient_logo(rows: list[str], theme: Theme, *, color: bool | None) -> list[str]:
    """Paint the logo with a title→accent→info gradient (per row)."""
    from .palette import paint as _paint

    def _grad_row(row: str, frac: float) -> str:
        # Blend title → accent → info across the banner height.
        c1, c2, c3 = theme["title"], theme["accent"], theme["info"]
        code = c1 if frac < 0.4 else (c2 if frac < 0.75 else c3)
        return _paint(row, code, color=color)

    n = max(1, len(rows))
    return [_grad_row(row, i / n) for i, row in enumerate(rows)]


def render_banner(
    adapters: list[str],
    *,
    version: str = "",
    theme: Theme | None = None,
    tip_seed: int | None = None,
    color: bool | None = None,
    style: str = "block",
    gradient: bool = False,
    frame: bool = False,
) -> str:
    """Render the full startup banner.

    ``style`` is one of ``list_banner_styles()`` (``block`` is the
    classic). ``gradient=True`` paints the logo as a title→accent→info
    gradient; ``frame=True`` draws a rounded box around the whole banner.
    """
    theme = theme or get_theme()
    t = lambda s, role: paint(s, theme[role], color=color)  # noqa: E731
    logo = _BANNER_FONTS.get((style or "block").lower(), _LOGO)
    lines: list[str] = []
    lines.append("")
    if gradient:
        lines.extend("  " + row for row in _gradient_logo(logo, theme, color=color))
    else:
        for row in logo:
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
    lines.append(f"  {t('theme:', 'accent')} {t(theme_name(), 'info')}")
    lines.append(f"  {t('tip:', 'accent')} {t(tip_of_the_day(tip_seed), 'subtle')}")
    lines.append("")
    out = "\n".join(lines)
    if frame:
        out = _frame_banner(out, theme, color=color)
    return strip_ansi(out) if color is False else out


def _frame_banner(text: str, theme: Theme, *, color: bool | None) -> str:
    """Draw a rounded box around banner text."""
    from .palette import BOX_ROUNDED, visible_width

    b = BOX_ROUNDED
    raw = text.split("\n")
    w = max((visible_width(ln) for ln in raw), default=0)
    top = b["tl"] + b["h"] * (w + 2) + b["tr"]
    bot = b["bl"] + b["h"] * (w + 2) + b["br"]
    border = theme.get("border", theme.get("subtle", ""))
    framed = [paint(top, border, color=color)]
    for ln in raw:
        pad = " " * max(0, w - visible_width(ln))
        framed.append(
            paint(b["v"] + " ", border, color=color)
            + ln + pad
            + paint(" " + b["v"], border, color=color)
        )
    framed.append(paint(bot, border, color=color))
    return "\n".join(framed)


def theme_name_label(theme: Theme) -> str:
    """Best-effort label for a theme dict (used in the banner)."""
    # The active theme is tracked by name in the environment; comparing
    # dicts is fragile now that themes carry an extended role contract.
    from .themes import theme_name as _tn

    return _tn()


__all__ = ["render_banner", "tip_of_the_day", "list_banner_styles", "BANNER_STYLES"]
