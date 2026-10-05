"""Console UI for the Devon chat bot: colors, dashboard, console commands.

The palette deliberately avoids black backgrounds and red text — the owner
hates both. Blues, greens, cyans, purples, whites and yellows only.
"""

from __future__ import annotations

from .banner import render_banner, tip_of_the_day
from .commands import ConsoleCommands
from .debug import DebugHub
from .dashboard import (
    render_dashboard,
    render_debug_view,
    render_games_view,
    render_llm_view,
    render_scheduler_view,
    render_status_line,
    render_statusbar,
    render_view,
)
from .palette import paint, strip_ansi, supports_color, truncate_visible, visible_width
from .themes import get_theme, list_themes, theme_name
from .widgets import (
    AVATAR,
    GodScreen,
    LiveScreen,
    MessageEvent,
    MessageFeed,
    ProgressBar,
    WatchHub,
    barchart,
    format_feed_line,
    format_message_card,
    gradient_text,
    sparkline,
)

__all__ = [
    "AVATAR",
    "ConsoleCommands",
    "DebugHub",
    "GodScreen",
    "LiveScreen",
    "MessageEvent",
    "MessageFeed",
    "ProgressBar",
    "WatchHub",
    "barchart",
    "format_feed_line",
    "format_message_card",
    "get_theme",
    "gradient_text",
    "list_themes",
    "paint",
    "render_banner",
    "render_dashboard",
    "render_debug_view",
    "render_games_view",
    "render_llm_view",
    "render_scheduler_view",
    "render_status_line",
    "render_statusbar",
    "render_view",
    "sparkline",
    "strip_ansi",
    "supports_color",
    "theme_name",
    "tip_of_the_day",
    "truncate_visible",
    "visible_width",
]
