"""Progression-gated commands: beating challenges unlocks new commands.

``COMMAND_UNLOCKS`` maps a chat command name to the ``unlock:command:<name>``
grant string a player must hold (earned via achievements — see
``nomorals/games/achievements.py::ACHIEVEMENT_UNLOCKS``). The partner
runtime gates these commands in ``handle_control`` before dispatching.

The owner is never gated: progression grants are additive for players and
the runtime bypasses the gate for operator chats.
"""
from __future__ import annotations

from typing import Any

__all__ = ["COMMAND_UNLOCKS", "command_unlock_required", "locked_reply"]

#: command name -> required unlock grant string.
COMMAND_UNLOCKS: dict[str, str] = {
    "predict": "unlock:command:predict",
}


def command_unlock_required(command: str) -> str | None:
    """The unlock grant string a command needs, or None if it's open."""
    return COMMAND_UNLOCKS.get((command or "").strip().lower())


def locked_reply(command: str, db: Any, player_key: str) -> str | None:
    """Reply text when a player lacks a command's unlock, else None.

    Names the achievement that grants the unlock so the player knows what
    to beat. Never raises — returns a generic locked message on any failure.
    """
    required = command_unlock_required(command)
    if required is None:
        return None
    try:
        from .achievements import UNLOCK_SOURCE, _achievement_map, has_unlock

        if has_unlock(db, player_key, required):
            return None
        achievement_id = UNLOCK_SOURCE.get(required)
        ach = _achievement_map().get(achievement_id) if achievement_id else None
        if ach is not None:
            return (
                f"\U0001f512 /{command} is locked — earn '{ach.name}' "
                f"({ach.description}) to unlock it."
            )
        return f"\U0001f512 /{command} is locked — keep playing to unlock it."
    except Exception:  # noqa: BLE001 - fail closed with a generic message
        return f"\U0001f512 /{command} is locked — keep playing to unlock it."
