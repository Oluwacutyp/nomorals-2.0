"""Typing indicators are fire-and-forget (owner order, 2026-11-01).

The indicator must never delay the message it precedes: every
adapter's typing() dispatches ONE action and returns at once, no
matter how long a window the caller asked for. Sustained indication
is the caller's job (the runtime re-fires on its tick loop).
"""
from __future__ import annotations

import asyncio
import threading
import time
import unittest

from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.discord import DiscordAdapter
from nomorals.social.chat.telegram import TelegramAdapter, TelegramBotAdapter


def _loop_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop


class BotApiTypingTests(unittest.TestCase):
    def test_recording_maps_to_record_voice_single_call(self):
        adapter = TelegramBotAdapter(token="test-token")
        calls = []
        adapter._api = lambda method, **kw: calls.append(
            {"method": method, **kw}) or {"ok": True}  # noqa: SLF001
        chat = ChatRef(platform="telegram-bot", chat_id="123",
                       kind=ChatKind.DM)
        started = time.time()
        self.assertTrue(adapter.typing(chat, seconds=45.0,
                                       action="recording"))
        self.assertLess(time.time() - started, 2.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["action"], "record_voice")

    def test_api_failure_is_false_not_a_crash(self):
        adapter = TelegramBotAdapter(token="test-token")

        def boom(method, **kw):
            raise RuntimeError("api down")

        adapter._api = boom  # noqa: SLF001
        chat = ChatRef(platform="telegram-bot", chat_id="123",
                       kind=ChatKind.DM)
        self.assertFalse(adapter.typing(chat, seconds=3.0))


class TelethonTypingTests(unittest.TestCase):
    def test_schedules_one_action_and_returns_at_once(self):
        adapter = TelegramAdapter(api_id=1, api_hash="x")
        loop = _loop_thread()
        fired = []

        class SlowClient:
            def __call__(self, request):
                async def _slow():
                    await asyncio.sleep(5.0)
                    fired.append(request)
                return _slow()

        adapter._client = SlowClient()  # noqa: SLF001
        adapter._loop = loop  # noqa: SLF001
        adapter._input_entity_cache.set("123", object())  # noqa: SLF001
        chat = ChatRef(platform="telegram", chat_id="123", kind=ChatKind.DM)
        started = time.time()
        self.assertTrue(adapter.typing(chat, seconds=30.0))
        # the caller is NOT held for the action, let alone the window
        self.assertLess(time.time() - started, 2.0)
        loop.call_soon_threadsafe(loop.stop)

    def test_no_client_no_typing(self):
        adapter = TelegramAdapter(api_id=1, api_hash="x")
        chat = ChatRef(platform="telegram", chat_id="123", kind=ChatKind.DM)
        self.assertFalse(adapter.typing(chat, seconds=3.0))


class DiscordTypingTests(unittest.TestCase):
    def test_schedules_one_action_and_returns_at_once(self):
        adapter = DiscordAdapter(token="test-token")
        loop = _loop_thread()
        fired = []

        class SlowChannel:
            async def send_typing(self):
                await asyncio.sleep(5.0)
                fired.append(True)

        async def _channel(chat):
            return SlowChannel()

        adapter._channel = _channel  # noqa: SLF001
        adapter._loop = loop  # noqa: SLF001
        chat = ChatRef(platform="discord", chat_id="123", kind=ChatKind.DM)
        started = time.time()
        self.assertTrue(adapter.typing(chat, seconds=30.0))
        self.assertLess(time.time() - started, 2.0)
        loop.call_soon_threadsafe(loop.stop)

    def test_no_loop_no_typing(self):
        adapter = DiscordAdapter(token="test-token")
        chat = ChatRef(platform="discord", chat_id="123", kind=ChatKind.DM)
        self.assertFalse(adapter.typing(chat, seconds=3.0))


if __name__ == "__main__":
    unittest.main()
