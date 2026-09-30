"""The games registry — every playable game in one place."""
from __future__ import annotations

from .base import GAME_COMMANDS, MultiGame, Room, parse_command

__all__ = ["GAME_COMMANDS", "MultiGame", "Room", "parse_command"]
