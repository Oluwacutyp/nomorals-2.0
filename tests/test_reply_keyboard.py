"""Part 2c: reply_keyboard plumbing — base -> gateway -> every adapter.

* The Bot API adapter renders a persistent bottom keyboard
  (resize + one_time=False + is_persistent) and it wins over any
  inline keyboard on the same message.
* Non-Telegram adapters accept the param and ignore it.
* The gateway passes it through to the adapter.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.social.chat.base import ChatRef
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.telegram import TelegramBotAdapter

KEYBOARD = [["/help", "/search"], ["/game"]]


def _bot_adapter():
    adapter = TelegramBotAdapter.__new__(TelegramBotAdapter)
    adapter.token = "test-token"
    adapter.stats = {"sent": 0, "send_errors": 0}
    adapter.chat_allow = set()
    return adapter


class BotApiReplyKeyboardTests(unittest.TestCase):
    def test_reply_keyboard_rendered_persistent(self):
        adapter = _bot_adapter()
        seen = {}

        def fake_api(method, **params):
            seen.update(params)
            return {"message_id": 1}

        adapter._api = fake_api
        res = adapter.send(ChatRef(platform="telegram-bot", chat_id="123"),
                           "menu", reply_keyboard=KEYBOARD)
        self.assertTrue(res.ok)
        markup = seen.get("reply_markup")
        self.assertIsNotNone(markup)
        self.assertEqual(markup["keyboard"],
                         [[{"text": "/help"}, {"text": "/search"}],
                          [{"text": "/game"}]])
        self.assertTrue(markup["resize_keyboard"])
        self.assertFalse(markup["one_time_keyboard"])
        self.assertTrue(markup["is_persistent"])

    def test_reply_keyboard_wins_over_inline_buttons(self):
        adapter = _bot_adapter()
        seen = {}

        def fake_api(method, **params):
            seen.update(params)
            return {"message_id": 1}

        adapter._api = fake_api
        # This text would derive inline buttons via buttons_for_text
        # ("modes & mastery" + "/game <name>") — the reply keyboard wins.
        res = adapter.send(
            ChatRef(platform="telegram-bot", chat_id="123"),
            "modes & mastery\n/game <name> — start",
            buttons=[[("Join", "game join")]],
            reply_keyboard=KEYBOARD)
        self.assertTrue(res.ok)
        self.assertIn("keyboard", seen["reply_markup"])
        self.assertNotIn("inline_keyboard", seen["reply_markup"])

    def test_no_keyboard_when_none_passed(self):
        adapter = _bot_adapter()
        seen = {}

        def fake_api(method, **params):
            seen.update(params)
            return {"message_id": 1}

        adapter._api = fake_api
        adapter.send(ChatRef(platform="telegram-bot", chat_id="123"),
                     "plain text")
        self.assertNotIn("reply_markup", seen)


class GatewayPassthroughTests(unittest.TestCase):
    def test_gateway_forwards_reply_keyboard(self):
        gw = ChatGateway.__new__(ChatGateway)
        gw.dry_run = False
        gw._adapters = {}
        captured = {}

        class _FakeAdapter:
            name = "fake"

            def send(self, chat, text, *, reply_to="",
                     buttons=None, reply_keyboard=None, parse_mode=""):
                captured["reply_keyboard"] = reply_keyboard
                from nomorals.social.chat.base import SendResult
                return SendResult(ok=True, platform="fake")

        gw._adapters["fake"] = _FakeAdapter()
        # _adapter_for + touch + lock helpers: stub the minimal surface.
        gw._adapter_for = lambda platform: _FakeAdapter()
        gw.touch = lambda chat: None
        import threading
        gw._locks = {}
        gw._lock_for = lambda chat: threading.Lock()
        gw._bump = lambda *a, **k: None
        gw.send("fake", ChatRef(platform="fake", chat_id="1"), "hi",
                reply_keyboard=KEYBOARD)
        self.assertEqual(captured["reply_keyboard"], KEYBOARD)
