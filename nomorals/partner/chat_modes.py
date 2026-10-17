"""Per-chat reply modes — games-only, normal.

One chat can be set to games-only: Devon sends NO reply to normal chat
messages there, but game commands (and game moves while a game is live)
still work, and owner control commands still work (so the owner can
unset it). The use case is a private chat (e.g. with the owner's
partner) where Devon should stay out of the conversation but the table
stays open.

Storage follows the voice-conversation pattern exactly: a JSON file
``{"chats": {chat_key: {"mode": ..., "set_at": ...}}}`` with tolerant
load/save. The store lives at ``~/.config/nomorals/chat_modes.json``
(env override ``NOMORALS_CHAT_MODES_PATH`` for tests). Unknown chats
default to ``normal``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

_log = logging.getLogger("nomorals.partner.chat_modes")

MODE_NORMAL = "normal"
MODE_GAMES_ONLY = "games_only"
MODES = (MODE_NORMAL, MODE_GAMES_ONLY)


def _store_path(path: str = "") -> str:
    if path:
        return os.path.expanduser(path)
    override = os.environ.get("NOMORALS_CHAT_MODES_PATH", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "chat_modes.json")


def _load(path: str = "") -> dict[str, Any]:
    try:
        with open(_store_path(path), encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(state: dict[str, Any], path: str = "") -> None:
    try:
        target = _store_path(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, target)
    except OSError as exc:
        _log.warning("chat modes save failed: %s", exc)


def set_chat_mode(chat_key: str, mode: str, path: str = "") -> str:
    """Set a chat's reply mode. Returns the stored mode.

    Raises :class:`ValueError` for an unknown mode or an empty chat key.
    Setting ``normal`` removes the entry (normal is the default).
    """
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown chat mode {mode!r} — want one of: {', '.join(MODES)}")
    chat_key = (chat_key or "").strip()
    if not chat_key:
        raise ValueError("chat key is required")
    state = _load(path)
    chats = state.get("chats", {})
    if not isinstance(chats, dict):
        chats = {}
    if mode == MODE_NORMAL:
        chats.pop(chat_key, None)
    else:
        chats[chat_key] = {"mode": mode, "set_at": time.time()}
    state["chats"] = chats
    _save(state, path)
    return mode


def get_chat_mode(chat_key: str, path: str = "") -> str:
    """A chat's reply mode; ``normal`` when never set."""
    chats = _load(path).get("chats", {})
    if not isinstance(chats, dict):
        return MODE_NORMAL
    entry = chats.get((chat_key or "").strip(), {})
    mode = entry.get("mode", MODE_NORMAL) if isinstance(entry, dict) else MODE_NORMAL
    return mode if mode in MODES else MODE_NORMAL


def is_games_only(chat_key: str, path: str = "") -> bool:
    return get_chat_mode(chat_key, path) == MODE_GAMES_ONLY


def list_chat_modes(path: str = "") -> dict[str, str]:
    """Every chat with a non-default mode, as {chat_key: mode}."""
    chats = _load(path).get("chats", {})
    if not isinstance(chats, dict):
        return {}
    out: dict[str, str] = {}
    for key, entry in chats.items():
        if isinstance(entry, dict) and entry.get("mode") in MODES:
            if entry["mode"] != MODE_NORMAL:
                out[key] = entry["mode"]
    return out
