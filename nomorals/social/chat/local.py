"""Local console adapter: the companion, in your terminal.

Real, not a stub — this is how you talk to her from Termux without any
platform account, and it is what the tests drive. Inbound lines come from
stdin; her replies are printed to stdout with a mood-tagged prefix so the
state is visible while you talk.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, MediaRef, SendResult

__all__ = ["LocalAdapter"]


class LocalAdapter(ChatAdapter):
    name = "local"
    supported_kinds = (ChatKind.DM,)

    def __init__(
        self,
        *,
        chat_id: str = "console",
        title: str = "Console",
        peer: str = "you",
        out: Any = None,
        mood_label_provider: Any = None,
    ) -> None:
        super().__init__()
        self.chat = ChatRef(platform="local", chat_id=chat_id, kind=ChatKind.DM,
                            title=title, peer=peer)
        self._out = out or sys.stdout
        self.mood_label_provider = mood_label_provider

    def run(self, handler: IncomingHandler) -> None:
        print("─" * 46, file=self._out)
        print(f"  {self.chat.title} — type to talk. 'exit' closes the console.", file=self._out)
        print("─" * 46, file=self._out)
        for line in sys.stdin:
            if self.stopped:
                return
            text = line.strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit", "/quit"}:
                return
            handler(
                ChatMessage(
                    chat=self.chat,
                    incoming=True,
                    text=text,
                    sender=self.chat.peer,
                    message_id=f"local-{int(time.time() * 1000)}",
                )
            )

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        label = ""
        if self.mood_label_provider is not None:
            try:
                label = f"  [{self.mood_label_provider()}]"
            except Exception:  # noqa: BLE001 - label is cosmetic
                label = ""
        print(f"her{label}: {text}", file=self._out)
        self.stats["sent"] += 1
        return SendResult(ok=True, platform=self.name, message_id=f"local-{int(time.time() * 1000)}")

    def typing(self, chat: ChatRef, seconds: float = 3.0) -> bool:
        # The console has no typing indicator; the latency itself is the signal.
        return False

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "") -> SendResult:
        print(f"her: [media: {media.path}]" + (f" — {caption}" if caption else ""), file=self._out)
        self.stats["sent"] += 1
        return SendResult(ok=True, platform=self.name)

    def health(self) -> dict[str, Any]:
        return {**super().health(), "note": "console adapter: no network"}
