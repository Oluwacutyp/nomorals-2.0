"""L7 — terminal UI.

``model.py`` holds the state and layout and is pure, so it is unit tested.
``app.py`` is a thin curses driver that reads keys, calls the model, and draws
the result.

    from nomorals.tui import run
    run(context)
"""

from __future__ import annotations

from .app import TuiApp, run
from .model import (
    KEY_HELP,
    SLASH_HELP,
    KeyAction,
    Line,
    Panel,
    Rendered,
    TuiState,
    action_for,
    help_overlay,
    render,
)

__all__ = [
    "KEY_HELP", "SLASH_HELP", "KeyAction", "Line", "Panel", "Rendered",
    "TuiApp", "TuiState", "action_for", "help_overlay", "render", "run",
]
