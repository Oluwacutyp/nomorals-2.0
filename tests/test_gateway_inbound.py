"""R8: chat gateway inbound path (nomorals.social.chat.gateway).

Covers the previously untested ``_on_inbound`` owner/rate-limit logic —
the per-message gate every inbound message passes through: owner exemption,
per-chat (not per-platform) flood windows, console-always-owner, and
handler-error isolation. All offline with a fake adapter.
"""

from __future__ import annotations

import time
import unittest

from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway


class _FakeAdapter(ChatAdapter):
    name = "fake"

    def run(self, handler):
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        return SendResult(ok=True, platform=self.name)


def _msg(chat_id: str, platform: str = "tg",
         kind: str = ChatKind.DM, text: str = "hi") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=kind),
        incoming=True, text=text, sender="someone", media=[],
        reply_to="", mentioned=False, ts=time.time(), message_id="m1",
    )


def _gateway(**kw):
    kw.setdefault("adapters", {"fake": _FakeAdapter()})
    kw.setdefault("max_per_hour", 5)
    return ChatGateway(**kw)


class InboundOwnerTests(unittest.TestCase):
    def test_owner_verdict_stashed_on_message(self):
        gw = _gateway(owner_chats={"tg:1"})
        seen = []
        gw._inbound = seen.append
        gw._on_inbound(_msg("1"))
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].meta["is_owner"])

    def test_stranger_verdict_stashed(self):
        gw = _gateway(owner_chats={"tg:1"})
        seen = []
        gw._inbound = seen.append
        gw._on_inbound(_msg("2"))
        self.assertFalse(seen[0].meta["is_owner"])

    def test_console_chat_always_owner(self):
        gw = _gateway()  # no owner_chats configured at all
        seen = []
        gw._inbound = seen.append
        # key "local:console" endswith ":console" → always the owner's terminal
        gw._on_inbound(_msg("console", platform="local"))
        self.assertTrue(seen[0].meta["is_owner"])

    def test_owner_never_rate_limited(self):
        gw = _gateway(owner_chats={"tg:1"})
        seen = []
        gw._inbound = seen.append
        for _ in range(20):  # 4x the hourly cap
            gw._on_inbound(_msg("1"))
        self.assertEqual(len(seen), 20)
        self.assertEqual(gw.stats["dropped_rate_limited"], 0)


class InboundFloodTests(unittest.TestCase):
    def test_stranger_flood_dropped_after_cap(self):
        gw = _gateway()
        seen = []
        gw._inbound = seen.append
        for _ in range(12):
            gw._on_inbound(_msg("9"))
        self.assertEqual(len(seen), 5)
        self.assertEqual(gw.stats["dropped_rate_limited"], 7)
        self.assertEqual(gw.stats["inbound"], 12)

    def test_windows_are_per_chat(self):
        gw = _gateway()
        seen = []
        gw._inbound = seen.append
        for _ in range(5):
            gw._on_inbound(_msg("a"))
        for _ in range(5):
            gw._on_inbound(_msg("b"))
        self.assertEqual(len(seen), 10)
        self.assertEqual(gw.stats["dropped_rate_limited"], 0)

    def test_handler_error_does_not_kill_feed(self):
        gw = _gateway()

        def _boom(message):
            raise RuntimeError("brain exploded")

        gw._inbound = _boom
        gw._on_inbound(_msg("3"))  # must not raise
        self.assertEqual(gw.stats["inbound"], 1)

    def test_no_handler_configured_is_safe(self):
        gw = _gateway()
        gw._inbound = None
        gw._on_inbound(_msg("3"))  # must not raise
        self.assertEqual(gw.stats["inbound"], 1)


if __name__ == "__main__":
    unittest.main()
