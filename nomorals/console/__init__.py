"""Console UI for the Devon chat bot: colors, dashboard, console commands.

The palette deliberately avoids black backgrounds and red text — the owner
hates both. Blues, greens, cyans, purples, whites and yellows only.
"""

from __future__ import annotations

from .commands import ConsoleCommands
from .dashboard import render_dashboard
from .palette import paint, supports_color

__all__ = ["ConsoleCommands", "paint", "render_dashboard", "supports_color"]
