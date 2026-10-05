"""telegram-bot public gate: conversation + games only.

On the ``telegram-bot`` endpoint every known non-game slash command is
answered with a friendly redirect instead of running — owner/system
controls live on the userbot (``telegram`` endpoint). Game commands and
natural chat pass through untouched, and the userbot keeps full powers.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self._handler = None

    def run(self, handler) -> None:
        self._handler = handler

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_runtime(platform: str) -> tuple[PartnerRuntime, FakeAdapter,
                                          Any, "tempfile.TemporaryDirectory"]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-gate-")
    settings = load_settings(
        overrides={
            "home": tmp.name,
            f"partner.platforms": platform,
        }
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = FakeAdapter(platform)
    gateway = ChatGateway({platform: adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, context, tmp


def _msg(platform: str, text: str, *, owner: bool = False) -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id="c1",
                     kind=ChatKind.DM, peer="someone"),
        incoming=True,
        text=text,
        sender="someone",
        meta={"is_owner": True} if owner else {},
    )


class PublicBotGateTest(unittest.TestCase):
    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        self.context.close()
        self.tmp.cleanup()

    def _setup(self, platform: str) -> None:
        self.runtime, self.adapter, self.context, self.tmp = _make_runtime(platform)

    # ── telegram-bot: blocked ──────────────────────────────────────────
    def test_owner_model_blocked_on_telegram_bot(self) -> None:
        self._setup("telegram-bot")
        with mock.patch.object(self.runtime, "handle_control") as hc:
            self.runtime._process(_msg("telegram-bot", "/model", owner=True))
        hc.assert_not_called()
        self.assertEqual(len(self.adapter.sent), 1)
        self.assertIn("not something I can do here", self.adapter.sent[0])

    def test_owner_upgrade_blocked_on_telegram_bot(self) -> None:
        self._setup("telegram-bot")
        with mock.patch.object(self.runtime, "handle_control") as hc:
            self.runtime._process(_msg("telegram-bot", "/upgrade", owner=True))
        hc.assert_not_called()
        self.assertEqual(len(self.adapter.sent), 1)
        self.assertIn("conversation and games", self.adapter.sent[0])

    def test_stranger_model_blocked_on_telegram_bot(self) -> None:
        self._setup("telegram-bot")
        self.runtime._process(_msg("telegram-bot", "/model"))
        self.assertEqual(len(self.adapter.sent), 1)
        self.assertIn("not something I can do here", self.adapter.sent[0])

    def test_start_blocked_on_telegram_bot(self) -> None:
        # /start is a platform control (start_platform), not a greeting.
        self._setup("telegram-bot")
        with mock.patch.object(self.runtime, "handle_control") as hc:
            self.runtime._process(_msg("telegram-bot", "/start local",
                                       owner=True))
        hc.assert_not_called()
        self.assertEqual(len(self.adapter.sent), 1)

    # ── telegram-bot: allowed ──────────────────────────────────────────
    def test_game_command_passes_gate_on_telegram_bot(self) -> None:
        self._setup("telegram-bot")
        with mock.patch.object(self.runtime, "_control_game",
                               return_value="mocked game") as cg:
            self.runtime._process(_msg("telegram-bot", "/duel"))
        cg.assert_called_once()
        # the gate's redirect must NOT have been sent; the game reply was
        self.assertEqual(self.adapter.sent, ["mocked game"])

    def test_gift_passes_gate_on_telegram_bot(self) -> None:
        self._setup("telegram-bot")
        with mock.patch.object(self.runtime, "_control_gift",
                               return_value="mocked gift") as cg:
            self.runtime._process(_msg("telegram-bot", "/gift @x 10"))
        cg.assert_called_once()
        self.assertEqual(self.adapter.sent, ["mocked gift"])

    def test_unknown_slash_falls_through_to_chat(self) -> None:
        # unknown slashes are ordinary text, not blocked
        self._setup("telegram-bot")
        outcome = mock.Mock()
        outcome.parts = []
        outcome.presence.delay_seconds = 0
        with mock.patch.object(self.runtime, "handle_control") as hc:
            with mock.patch.object(self.runtime.brain, "handle_message",
                                   return_value=outcome) as bm:
                self.runtime._process(_msg("telegram-bot", "/shrug"))
        hc.assert_not_called()
        bm.assert_called_once()
        self.assertEqual(self.adapter.sent, [])

    # ── userbot untouched ──────────────────────────────────────────────
    def test_owner_model_runs_on_userbot(self) -> None:
        self._setup("telegram")
        with mock.patch.object(self.runtime, "handle_control",
                               return_value="mocked control") as hc:
            self.runtime._process(_msg("telegram", "/model", owner=True))
        hc.assert_called_once()
        self.assertEqual(self.adapter.sent, ["mocked control"])

    def test_game_command_still_works_on_userbot(self) -> None:
        self._setup("telegram")
        with mock.patch.object(self.runtime, "_control_game",
                               return_value="mocked game") as cg:
            self.runtime._process(_msg("telegram", "/duel", owner=True))
        cg.assert_called_once()
        self.assertEqual(self.adapter.sent, ["mocked game"])


if __name__ == "__main__":
    unittest.main()
