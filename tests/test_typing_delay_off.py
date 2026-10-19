"""Typing delay OFF for the userbot owner DM and the personal bot.

Owner order: no typing delay / fake typing indicator on
(a) the userbot (MTProto, platform ``telegram``) in the owner's DM, and
(b) the personal bot (platform ``telegram-personal``, owner-only).

Everything else keeps the existing behavior — in particular the main
gated bot (``telegram-bot``) still types in non-owner chats AND in the
owner DM, and the userbot still types in non-owner chats.

Failing-first: before the fix, the userbot-owner and personal-bot cases
call ``gateway.typing`` (and sleep between parts); after the fix they
send immediately with zero typing calls and zero sleeps.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway

OWNER_ID = "5478650254"
OTHER_ID = "12345"


class _RecordingAdapter(ChatAdapter):
    """Records typing + send calls; typing returns instantly (no wait)."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.typing_calls: list[tuple[str, float, str]] = []
        self.sent: list[str] = []
        self.media_sent: list[object] = []

    def run(self, handler):  # noqa: D102 - test stub
        pass

    def typing(self, chat: ChatRef, seconds: float = 3.0, action: str = "typing") -> bool:
        self.typing_calls.append((chat.key, seconds, action))
        return True

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons=None, **kwargs) -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name, message_id=f"m{len(self.sent)}")

    def send_media(self, chat: ChatRef, media) -> SendResult:
        self.media_sent.append(media)
        return SendResult(ok=True, platform=self.name, message_id="media1")


def _runtime(tmp_home: str) -> tuple[PartnerRuntime, dict[str, _RecordingAdapter]]:
    settings = load_settings(overrides={
        "home": tmp_home,
        "partner.platforms": "telegram",
        "partner.owner_chats": OWNER_ID,
        "partner.owner_sender_ids": OWNER_ID,
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    adapters = {
        "telegram": _RecordingAdapter("telegram"),
        "telegram-bot": _RecordingAdapter("telegram-bot"),
        "telegram-personal": _RecordingAdapter("telegram-personal"),
    }
    gateway = ChatGateway(adapters, db=context.db)
    return PartnerRuntime(context, gateway=gateway), adapters


def _msg(platform: str, chat_id: str) -> ChatMessage:
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM)
    return ChatMessage(chat=chat, incoming=True, text="hi", sender="you")


class TestTypingDelayOff(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-typing-")
        self.rt, self.adapters = _runtime(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _sleep_recorder(self):
        calls: list[float] = []

        def _fake_sleep(seconds: float) -> None:
            calls.append(seconds)

        return calls, _fake_sleep

    def test_userbot_owner_dm_no_typing_no_sleep(self) -> None:
        calls, fake_sleep = self._sleep_recorder()
        with mock.patch("nomorals.agents.partner.runtime.time.sleep", fake_sleep):
            self.rt._send_reply(_msg("telegram", OWNER_ID), ["part one", "part two"])
        self.assertEqual(self.adapters["telegram"].typing_calls, [])
        self.assertEqual(calls, [])  # no between-parts breath either
        self.assertEqual(self.adapters["telegram"].sent, ["part one", "part two"])

    def test_personal_bot_no_typing_no_sleep(self) -> None:
        # Personal bot is owner-only regardless of chat id — any id is gated.
        calls, fake_sleep = self._sleep_recorder()
        with mock.patch("nomorals.agents.partner.runtime.time.sleep", fake_sleep):
            self.rt._send_reply(_msg("telegram-personal", "999"), ["part one", "part two"])
        self.assertEqual(self.adapters["telegram-personal"].typing_calls, [])
        self.assertEqual(calls, [])
        self.assertEqual(self.adapters["telegram-personal"].sent, ["part one", "part two"])

    def test_main_bot_nonowner_keeps_typing(self) -> None:
        # PIN: the main gated bot's non-owner chats are unchanged.
        self.rt._send_reply(_msg("telegram-bot", OTHER_ID), ["hello there"])
        self.assertGreaterEqual(len(self.adapters["telegram-bot"].typing_calls), 1)
        self.assertEqual(self.adapters["telegram-bot"].sent, ["hello there"])

    def test_main_bot_owner_dm_keeps_typing(self) -> None:
        # PIN: the main bot is untouched even in the owner DM — only the
        # userbot owner DM and the personal bot lose the typing delay.
        self.rt._send_reply(_msg("telegram-bot", OWNER_ID), ["hello there"])
        self.assertGreaterEqual(len(self.adapters["telegram-bot"].typing_calls), 1)
        self.assertEqual(self.adapters["telegram-bot"].sent, ["hello there"])

    def test_userbot_nonowner_keeps_typing(self) -> None:
        # PIN: the userbot still types in non-owner chats.
        self.rt._send_reply(_msg("telegram", OTHER_ID), ["hello there"])
        self.assertGreaterEqual(len(self.adapters["telegram"].typing_calls), 1)
        self.assertEqual(self.adapters["telegram"].sent, ["hello there"])

    def test_typing_for_gated(self) -> None:
        self.rt._typing_for(_msg("telegram-personal", "999").chat, "quick reply")
        self.assertEqual(self.adapters["telegram-personal"].typing_calls, [])
        self.rt._typing_for(_msg("telegram", OWNER_ID).chat, "quick reply")
        self.assertEqual(self.adapters["telegram"].typing_calls, [])
        self.rt._typing_for(_msg("telegram-bot", OTHER_ID).chat, "quick reply")
        self.assertEqual(len(self.adapters["telegram-bot"].typing_calls), 1)

    def test_send_long_gated(self) -> None:
        self.rt._send_long("telegram-personal",
                           _msg("telegram-personal", "999").chat, "hello")
        self.assertEqual(self.adapters["telegram-personal"].typing_calls, [])
        self.assertEqual(self.adapters["telegram-personal"].sent, ["hello"])
        self.rt._send_long("telegram-bot",
                           _msg("telegram-bot", OTHER_ID).chat, "hello")
        self.assertGreaterEqual(len(self.adapters["telegram-bot"].typing_calls), 1)

    def test_keepalive_loop_gated(self) -> None:
        # Keepalive runs ~0.1s (stop set from the test thread) — gated
        # chats show no typing indicator while the brain "works".
        stop = threading.Event()
        thread = threading.Thread(
            target=self.rt._typing_keepalive_loop,
            args=(_msg("telegram-personal", "999").chat, stop), daemon=True)
        thread.start()
        time.sleep(0.1)
        stop.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.adapters["telegram-personal"].typing_calls, [])

    def test_keepalive_loop_keeps_typing_on_main_bot(self) -> None:
        stop = threading.Event()
        thread = threading.Thread(
            target=self.rt._typing_keepalive_loop,
            args=(_msg("telegram-bot", OTHER_ID).chat, stop), daemon=True)
        thread.start()
        time.sleep(0.1)
        stop.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(len(self.adapters["telegram-bot"].typing_calls), 1)

    def test_voice_reply_gated(self) -> None:
        # Voice path: the "recording" indicator is the same simulation —
        # gated chats send the voice note with no indicator.
        with mock.patch(
            "nomorals.voice.pingpong.synthesize_voice_reply",
            return_value="/tmp/nm-test-reply.ogg"), mock.patch(
            "nomorals.voice.pingpong.ogg_duration", return_value=2.0):
            sent = self.rt._send_voice_reply(
                _msg("telegram-personal", "999"), ["hello voice"])
        self.assertTrue(sent)
        self.assertEqual(self.adapters["telegram-personal"].typing_calls, [])
        self.assertEqual(len(self.adapters["telegram-personal"].media_sent), 1)

        with mock.patch(
            "nomorals.voice.pingpong.synthesize_voice_reply",
            return_value="/tmp/nm-test-reply.ogg"), mock.patch(
            "nomorals.voice.pingpong.ogg_duration", return_value=2.0):
            sent = self.rt._send_voice_reply(
                _msg("telegram-bot", OTHER_ID), ["hello voice"])
        self.assertTrue(sent)
        self.assertGreaterEqual(len(self.adapters["telegram-bot"].typing_calls), 1)


if __name__ == "__main__":
    unittest.main()
