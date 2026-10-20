"""Slash commands must reach the static control dispatch on ALL adapters.

Regression: /help worked on the gated first bot (``telegram-bot``) but on
the personal bot (``telegram-personal``) and the userbot (``telegram``)
it fell through to the brain and came back as a lexicon fallback phrase
("brain's buffering, one sec"). Root cause was owner recognition, not
the LLM: ``parse_chat_keys`` silently dropped a bare owner id
(``NM_PARTNER_OWNER_CHATS=5478650254``), so ``_is_operator`` was False
and the control-command branch never ran. /help is static — it must
work with no LLM configured.

These tests drive /help through ``PartnerRuntime._process`` with a fake
gateway/adapter on each of the three platforms and assert the reply is
the static help catalog, never a fallback phrase.
"""

from __future__ import annotations

import tempfile
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
    is_owner_chat,
)
from nomorals.social.chat.gateway import ChatGateway, parse_chat_keys

OWNER_ID = "5478650254"
FALLBACK_PHRASES = (
    "brain's buffering, one sec",
    "say that again? i spaced",
    "i'm here, just glitching a little",
)


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self._handler = None

    def run(self, handler) -> None:
        self._handler = handler

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, **kwargs: Any) -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_runtime(platform: str):
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-help-routing-")
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": platform,
            # Owner configured the way the owner actually did: bare id,
            # no platform prefix — plus the sender-id allowlist.
            "partner.owner_chats": OWNER_ID,
            "partner.owner_sender_ids": OWNER_ID,
        }
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = FakeAdapter(platform)
    gateway = ChatGateway({platform: adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, context, tmp


def _owner_msg(platform: str, text: str) -> ChatMessage:
    # DM from the owner: chat id == sender id == the configured owner id.
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=OWNER_ID,
                     kind=ChatKind.DM, peer="owner"),
        incoming=True,
        text=text,
        sender="owner",
        sender_id=OWNER_ID,
        meta={},
    )


class BareOwnerIdTest(unittest.TestCase):
    def test_parse_chat_keys_keeps_bare_id(self) -> None:
        self.assertIn(OWNER_ID, parse_chat_keys(OWNER_ID))
        self.assertIn(OWNER_ID, parse_chat_keys(f" {OWNER_ID} , "))
        self.assertIn("telegram:123", parse_chat_keys("telegram:123"))

    def test_is_owner_chat_matches_bare_id_on_any_platform(self) -> None:
        for platform in ("telegram", "telegram-bot", "telegram-personal"):
            chat = ChatRef(platform=platform, chat_id=OWNER_ID,
                           kind=ChatKind.DM)
            self.assertTrue(
                is_owner_chat(chat, owner_chats={OWNER_ID}),
                f"bare owner id must match {platform}",
            )
        stranger = ChatRef(platform="telegram-personal", chat_id="999",
                           kind=ChatKind.DM)
        self.assertFalse(is_owner_chat(stranger, owner_chats={OWNER_ID}))


class HelpRoutingTest(unittest.TestCase):
    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        self.context.close()
        self.tmp.cleanup()

    def _setup(self, platform: str) -> None:
        self.runtime, self.adapter, self.context, self.tmp = _make_runtime(platform)

    def _help_reply(self, platform: str) -> str:
        self._setup(platform)
        # No LLM anywhere in this path: if /help reaches the brain,
        # the test fails on the fallback assertion, not on a model call.
        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=AssertionError("brain must not see /help")):
            self.runtime._process(_owner_msg(platform, "/help"))
        self.assertEqual(len(self.adapter.sent), 1,
                         f"expected exactly one reply on {platform}")
        return self.adapter.sent[0]

    def test_help_on_gated_first_bot_is_public_catalog(self) -> None:
        # Pass #55 (owner's model, 2026-10-20): the first bot is PUBLIC
        # and gated for everyone on it, owner included — /help there is
        # the PUBLIC catalog, never the full one. (This assertion used
        # to pin the full catalog on telegram-bot; that was the defect
        # the owner reported.) Routing is unchanged: static reply, the
        # brain never sees the command.
        reply = self._help_reply("telegram-bot")
        self.assertNotIn("full catalog", reply)
        self.assertIn("public catalog", reply)
        for kind in ("/update", "/exness", "/tempmail"):
            self.assertNotIn(kind, reply)
        for phrase in FALLBACK_PHRASES:
            self.assertNotIn(phrase, reply)

    def test_help_on_personal_bot(self) -> None:
        reply = self._help_reply("telegram-personal")
        self.assertIn("full catalog", reply)
        for phrase in FALLBACK_PHRASES:
            self.assertNotIn(phrase, reply)

    def test_help_on_userbot(self) -> None:
        reply = self._help_reply("telegram")
        self.assertIn("full catalog", reply)
        for phrase in FALLBACK_PHRASES:
            self.assertNotIn(phrase, reply)

    def test_stranger_is_not_operator_on_personal_bot(self) -> None:
        self._setup("telegram-personal")
        stranger = ChatMessage(
            chat=ChatRef(platform="telegram-personal", chat_id="999",
                         kind=ChatKind.DM, peer="stranger"),
            incoming=True, text="/help", sender="stranger", sender_id="999",
            meta={},
        )
        self.assertFalse(self.runtime._is_operator(stranger))


if __name__ == "__main__":
    unittest.main()
