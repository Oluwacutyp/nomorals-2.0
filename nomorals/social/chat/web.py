"""Web console adapter: the companion, in your browser.

Inbound never flows from this adapter (there is no network peer) — the
web console server pushes owner messages into the partner runtime
directly, and her replies land here as captured parts.  That split is
what makes the console a *view* of the same partner who lives on the
social platforms: same brain, same mood, same memory, same commands —
only the wire is different.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, MediaRef, SendResult

__all__ = ["WebAdapter"]


class WebAdapter(ChatAdapter):
    name = "web"
    supported_kinds = (ChatKind.DM,)

    def __init__(
        self,
        *,
        chat_id: str = "console",
        title: str = "Web Console",
        peer: str = "you",
    ) -> None:
        super().__init__()
        # chat_id "console" matters: PartnerRuntime._is_operator() treats
        # every "*:console" chat as the owner's, so slash commands work.
        self.chat = ChatRef(platform="web", chat_id=chat_id,
                            kind=ChatKind.DM, title=title, peer=peer)
        self._parts: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._last_send: dict[str, float] = {}
        #: Typing indicator state per chat key: unix timestamp until which
        #: "she is typing…" should render. The browser poll loop reads it
        #: via :meth:`typing_active`; ``typing()`` sets it for the full
        #: requested (length-scaled) window.
        self._typing_until: dict[str, float] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def run(self, handler: IncomingHandler) -> None:
        # No inbound feed: messages are pushed by the console server via
        # PartnerRuntime.on_message.  Block until stopped so the adapter
        # lifecycle stays uniform if a gateway ever starts us.
        self._stopped.wait()

    # ── sending (reply capture) ──────────────────────────────────────────────

    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             parse_mode: str = "") -> SendResult:
        with self._lock:
            self._parts.setdefault(chat.key, []).append(text)
            self._last_send[chat.key] = time.time()
        self.stats["sent"] += 1
        return SendResult(ok=True, platform=self.name,
                          message_id=f"web-{int(time.time() * 1000)}")

    def take_parts(self, key: str) -> list[str]:
        """Drain captured reply parts for one chat (oldest first)."""
        with self._lock:
            return self._parts.pop(key, [])

    def pending_parts(self, key: str) -> list[str]:
        """Peek without draining."""
        with self._lock:
            return list(self._parts.get(key, []))

    def last_send_age(self, key: str) -> float | None:
        """Seconds since the last reply part for this chat (None = never)."""
        with self._lock:
            ts = self._last_send.get(key)
            return None if ts is None else time.time() - ts

    def typing(self, chat: ChatRef, seconds: float = 3.0) -> bool:
        # The web console has no push channel for a typing event — the
        # browser renders "typing…" from the poll loop, which reads
        # typing_active(key). Record the full length-scaled window so the
        # indicator, when the frontend consumes it, lasts as long as the
        # message takes to type.
        if seconds <= 0:
            return False
        with self._lock:
            self._typing_until[chat.key] = time.time() + float(seconds)
        return True

    def typing_active(self, key: str) -> bool:
        """True while a typing indicator should render for this chat.

        Consumed by the web console's poll loop. False when no typing
        window is open or it has expired.
        """
        with self._lock:
            until = self._typing_until.get(key, 0.0)
        return time.time() < until

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "") -> SendResult:
        text = f"[media: {media.path}]" + (f" — {caption}" if caption else "")
        return self.send(chat, text, reply_to="")

    def health(self) -> dict[str, Any]:
        return {**super().health(),
                "note": "web console adapter: replies captured in-process"}
