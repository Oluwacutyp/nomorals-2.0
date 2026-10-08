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
        command_hook: Any = None,
    ) -> None:
        super().__init__()
        self.chat = ChatRef(platform="local", chat_id=chat_id, kind=ChatKind.DM,
                            title=title, peer=peer)
        self._out = out or sys.stdout
        self.mood_label_provider = mood_label_provider
        #: Optional callable(text) -> str | None. When set and it returns a
        #: string, that string is printed and the line is NOT dispatched to
        #: the brain. Used for console-only commands (dashboard, status…).
        self.command_hook = command_hook

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
            # Console-only commands (dashboard, status, …) are handled here
            # and never reach the brain.
            if self.command_hook is not None:
                try:
                    response = self.command_hook(text)
                except Exception:  # noqa: BLE001 - a broken hook must not kill input
                    response = None
                if response is not None:
                    print(response, file=self._out)
                    continue
            handler(
                ChatMessage(
                    chat=self.chat,
                    incoming=True,
                    text=text,
                    sender=self.chat.peer,
                    message_id=f"local-{int(time.time() * 1000)}",
                )
            )

    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             parse_mode: str = "") -> SendResult:
        label = ""
        if self.mood_label_provider is not None:
            try:
                label = f"  [{self.mood_label_provider()}]"
            except Exception:  # noqa: BLE001 - label is cosmetic
                label = ""
        # Rich reply line: timestamp + colored speaker, mood tag stays.
        try:
            from nomorals.console.palette import BOLD, DIM, MAGENTA, paint, supports_color

            ts = time.strftime("%H:%M:%S")
            if supports_color():
                who = paint("her", MAGENTA + BOLD)
                when = paint(ts, DIM)
                print(f"{when} {who}{label}: {text}", file=self._out)
            else:
                print(f"[{ts}] her{label}: {text}", file=self._out)
        except Exception:  # noqa: BLE001 - formatting is cosmetic
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
