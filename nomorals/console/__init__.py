"""Console UI for the Devon chat bot: colors, dashboard, console commands.

The palette deliberately avoids black backgrounds and red text — the owner
hates both. Blues, greens, cyans, purples, whites and yellows only.
"""

from __future__ import annotations

from .banner import render_banner, tip_of_the_day
from .commands import ConsoleCommands
from .dashboard import render_dashboard, render_status_line
from .palette import paint, strip_ansi, supports_color
from .themes import get_theme, list_themes, theme_name
from .widgets import LiveScreen, ProgressBar, format_message_card, sparkline

__all__ = [
    "ConsoleCommands",
    "LiveScreen",
    "ProgressBar",
    "format_message_card",
    "get_theme",
    "list_themes",
    "paint",
    "render_banner",
    "render_dashboard",
    "render_status_line",
    "sparkline",
    "strip_ansi",
    "supports_color",
    "theme_name",
    "tip_of_the_day",
]
