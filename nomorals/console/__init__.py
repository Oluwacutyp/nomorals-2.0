"""Console UI for the Devon chat bot: colors, dashboard, console commands.

The palette deliberately avoids black backgrounds and red text — the owner
hates both. Blues, greens, cyans, purples, whites and yellows only.
"""

from __future__ import annotations

from .avatar import (
    AVATAR_STYLES,
    avatar_size,
    list_avatar_styles,
    render_avatar,
    render_ninja,
    render_ninja_mini,
)
from .banner import BANNER_STYLES, list_banner_styles, render_banner, tip_of_the_day
from .alerts import AlertPolicy, AlertRecord, SeverityRule
from .cockpit import Cockpit
from .commands import ConsoleCommands, complete
from .costmeter import CostMeter, OrganCost
from .discovery import CommandEntry, CommandRegistry
from .recording import SessionRecorder
from .termux import MAX_COLS, TermuxScreen
from .dashboard import (
    render_adapters_view,
    render_dashboard,
    render_debug_view,
    render_games_view,
    render_health_view,
    render_llm_view,
    render_scheduler_view,
    render_status_line,
    render_statusbar,
    render_summary,
    render_top_view,
    render_view,
)
from .debug import DebugHub
# NOTE: ``godconsole`` is intentionally NOT imported eagerly here. It pulls
# in the fullscreen terminal UI (termios/tty on Unix); the chat bot imports
# this package just for ``ConsoleCommands`` and must not pay for — or crash
# on — the fullscreen console. ``GodConsole`` / ``EventBuffer`` /
# ``InputLine`` resolve lazily via ``__getattr__`` below (PEP 562), so
# ``from nomorals.console import GodConsole`` still works, and
# ``from nomorals.console import ConsoleCommands`` never touches it.
from .palette import (
    BOX_DOUBLE,
    BOX_HEAVY,
    BOX_ROUNDED,
    BOX_SINGLE,
    ITALIC,
    REVERSE,
    SEMANTIC,
    STRIKE,
    UNDERLINE,
    hex_color,
    pad_visible,
    paint,
    paint_sem,
    rgb,
    strip_ansi,
    supports_color,
    truncate_visible,
    visible_width,
    wrap_visible,
)
from .themes import (
    ROLES,
    describe_theme,
    get_theme,
    list_themes,
    theme_name,
    theme_preview,
    validate_theme,
)
from .widgets import (
    AVATAR,
    WATCH_VIEW_KEYS,
    WATCH_VIEWS,
    GodScreen,
    LiveScreen,
    MessageEvent,
    MessageFeed,
    ProgressBar,
    Spinner,
    WatchHub,
    barchart,
    braille_chart,
    columns_chart,
    format_feed_line,
    format_message_card,
    gauge,
    gradient_text,
    heatmap,
    panel,
    rule_caption,
    sparkline,
    status,
    table,
)

_LAZY_GODCONSOLE = {"EventBuffer", "GodConsole", "InputLine"}


def __getattr__(name: str):  # PEP 562 — lazy fullscreen console
    if name in _LAZY_GODCONSOLE:
        from .godconsole import EventBuffer as _EventBuffer
        from .godconsole import GodConsole as _GodConsole
        from .godconsole import InputLine as _InputLine

        _exports = {
            "EventBuffer": _EventBuffer,
            "GodConsole": _GodConsole,
            "InputLine": _InputLine,
        }
        value = _exports[name]
        globals()[name] = value  # cache: import only once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AVATAR",
    "AVATAR_STYLES",
    "AlertPolicy",
    "AlertRecord",
    "BANNER_STYLES",
    "BOX_DOUBLE",
    "BOX_HEAVY",
    "BOX_ROUNDED",
    "BOX_SINGLE",
    "ITALIC",
    "REVERSE",
    "ROLES",
    "SEMANTIC",
    "STRIKE",
    "UNDERLINE",
    "WATCH_VIEW_KEYS",
    "WATCH_VIEWS",
    "ConsoleCommands",
    "CommandEntry",
    "CommandRegistry",
    "Cockpit",
    "CostMeter",
    "DebugHub",
    "EventBuffer",
    "GodConsole",
    "GodScreen",
    "InputLine",
    "LiveScreen",
    "MAX_COLS",
    "MessageEvent",
    "MessageFeed",
    "OrganCost",
    "ProgressBar",
    "SessionRecorder",
    "SeverityRule",
    "Spinner",
    "TermuxScreen",
    "WatchHub",
    "avatar_size",
    "barchart",
    "braille_chart",
    "columns_chart",
    "complete",
    "describe_theme",
    "format_feed_line",
    "format_message_card",
    "gauge",
    "get_theme",
    "gradient_text",
    "heatmap",
    "hex_color",
    "list_avatar_styles",
    "list_banner_styles",
    "list_themes",
    "pad_visible",
    "paint",
    "paint_sem",
    "panel",
    "render_adapters_view",
    "render_avatar",
    "render_banner",
    "render_dashboard",
    "render_debug_view",
    "render_games_view",
    "render_health_view",
    "render_llm_view",
    "render_ninja",
    "render_ninja_mini",
    "render_scheduler_view",
    "render_status_line",
    "render_statusbar",
    "render_summary",
    "render_top_view",
    "render_view",
    "rgb",
    "rule_caption",
    "sparkline",
    "status",
    "strip_ansi",
    "supports_color",
    "table",
    "theme_name",
    "theme_preview",
    "tip_of_the_day",
    "truncate_visible",
    "validate_theme",
    "visible_width",
    "wrap_visible",
]
