"""Numbered button-fallback picks (register #27, 2026-11-05).

When a reply's buttons cannot ride the send (the platform takes
none), ``_send_control_reply`` rewrites the copy into a numbered
action list and registers the picks here. A later bare-number
message in the same chat executes the same command a tap would
have run — the fallback is a working control, not a description
of one. Picks are single-use and expire; a new registration for a
chat replaces the old list wholesale (the newest card on screen
owns the numbers).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

__all__ = ["ButtonPickStore", "TTL_S", "MAX_CHATS", "MAX_PICKS"]

TTL_S = 600.0
MAX_CHATS = 256
MAX_PICKS = 12


class ButtonPickStore:
    """Bounded, thread-safe chat -> ordered command list."""

    def __init__(self, *, ttl_s: float = TTL_S) -> None:
        self._ttl_s = float(ttl_s)
        self._lock = threading.Lock()
        self._by_chat: OrderedDict[
            str, tuple[float, list[str]]] = OrderedDict()

    def register(self, chat_key: str, commands: list[str]) -> None:
        key = str(chat_key or "")
        picks = [str(c) for c in (commands or []) if str(c).strip()]
        if not key or not picks:
            return
        with self._lock:
            self._by_chat[key] = (time.time(), picks[:MAX_PICKS])
            self._by_chat.move_to_end(key)
            while len(self._by_chat) > MAX_CHATS:
                self._by_chat.popitem(last=False)

    def take(self, chat_key: str, number: int, *,
             now: float | None = None) -> str:
        """The command for ``number`` in ``chat_key``, consuming it.

        Returns \"\" when there is no live pick list, the entry
        expired, or the number is out of range (an out-of-range
        number leaves the list intact — a typo must not burn the
        card's other actions).
        """
        key = str(chat_key or "")
        if not key:
            return ""
        ts = time.time() if now is None else float(now)
        with self._lock:
            entry = self._by_chat.get(key)
            if entry is None:
                return ""
            stamped, picks = entry
            if ts - stamped > self._ttl_s:
                self._by_chat.pop(key, None)
                return ""
            if number < 1 or number > len(picks):
                return ""
            command = picks[number - 1]
            self._by_chat.pop(key, None)
            return command

    def has(self, chat_key: str) -> bool:
        with self._lock:
            return str(chat_key or "") in self._by_chat
