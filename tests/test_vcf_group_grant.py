"""Public-surface fix 1 — /vcf in groups on the PUBLIC bot
(owner live report, 2026-11-10).

The defect: ``vcf`` is a control command, not a public one, and
on the gated public bot (``telegram-bot``) NOBODY is an operator
— the owner included — so the outer grant killed /vcf for every
group user before the handler's own gate could run. That handler
gate (operator, or the platform's word that the sender admins
the group) was designed for exactly this use; it was simply
unreachable behind the outer grant.

The fix is a group-scoped grant in the GROUP_DECISION_COMMANDS
family: ``GROUP_ADMIN_COMMANDS = {"vcf"}`` passes the outer
grant ONLY inside a group chat, on any surface; the handler's
admin rule stays the authority in the room. ``vcf`` is NOT in
PUBLIC_COMMANDS (a DM on the public bot is still refused,
exactly as before), ``caller_may_run`` itself is unchanged, and
no sibling control command gains anything.

These tests drive the REAL pipeline (``PartnerRuntime._process``
with fake adapters, the pass-55 harness shape): the claim gates,
``handle_control``'s grant, dispatch, and the shipped /vcf
handler compose exactly as in production. The fake adapters
answer admin/roster calls in the real shapes (the test_vcf_export
fixtures' shapes).
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import suppress
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
from nomorals.social.chat.control import (
    GROUP_ADMIN_COMMANDS,
    GROUP_DECISION_COMMANDS,
    PUBLIC_COMMANDS,
    caller_may_run,
)
from nomorals.social.chat.gateway import ChatGateway

OWNER_ID = "5478650254"
ADMIN_ID = "111"
MEMBER_ID = "222"
TG_GROUP_ID = "-100123"

PUBLIC_BOT_DENIAL = (
    "that's not something I can do here — this is my public chat")
HANDLER_REFUSAL = (
    "that's the owner's or a group admin's call, not a member's")

_TG_STATUSES = {OWNER_ID: "creator", ADMIN_ID: "administrator",
                MEMBER_ID: "member"}
_TG_MEMBERS = [
    {"id": "555", "name": "Boss Man", "username": "boss",
     "phone": "+2348055555555", "bot": False, "deleted": False},
    {"id": "556", "name": "Hidden Hana", "username": "hana",
     "phone": "", "bot": False, "deleted": False},
]


class _FakeTelegram(ChatAdapter):
    """A Telegram-family surface with the admin/roster verbs the
    /vcf handler consults, in the real adapters' shapes."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self.media: list[Any] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, **kwargs: Any) -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def send_media(self, chat: ChatRef, media: Any, *,
                   caption: str = "", **kwargs: Any) -> SendResult:
        self.media.append((media, caption))
        return SendResult(ok=True, platform=self.name,
                          message_id="f1")

    def typing(self, chat: ChatRef, seconds: float = 3.0,
               action: str = "typing") -> bool:
        return True

    def admin_member(self, ref: Any, user_id: Any) -> dict:
        return {"ok": True, "status": _TG_STATUSES.get(
            str(user_id), "member")}

    def admin_administrators(self, ref: Any) -> dict:
        return {"ok": True, "administrators": [
            {"id": OWNER_ID, "name": "Owner", "status": "creator"},
            {"id": ADMIN_ID, "name": "Admin A",
             "status": "administrator"}]}

    def admin_members_full(self, ref: Any,
                           limit: int = 20000) -> dict:
        return {"ok": True, "members": list(_TG_MEMBERS),
                "count": len(_TG_MEMBERS)}


class _FakeWhatsApp(ChatAdapter):
    def __init__(self) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = "whatsapp"
        self.sent: list[str] = []
        self.media: list[Any] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, **kwargs: Any) -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def send_media(self, chat: ChatRef, media: Any, *,
                   caption: str = "", **kwargs: Any) -> SendResult:
        self.media.append((media, caption))
        return SendResult(ok=True, platform=self.name,
                          message_id="f1")

    def typing(self, chat: ChatRef, seconds: float = 3.0,
               action: str = "typing") -> bool:
        return True

    def group_info(self, ref: Any) -> dict:
        return {"subject": "WA Group", "size": 2}

    def group_participants(self, ref: Any) -> list:
        return [
            {"jid": "2348011111111@s.whatsapp.net",
             "name": "Ada Lovelace", "role": ""},
            {"jid": "2348022222222@s.whatsapp.net",
             "name": "Bola", "role": "admin"},
        ]

    def admin_member(self, ref: Any, user_id: Any) -> dict:
        status = {"2348022222222": "admin"}.get(
            str(user_id), "member")
        return {"ok": True, "status": status}


class GrantTableTests(unittest.TestCase):
    def test_grant_set_shape(self):
        self.assertEqual(GROUP_ADMIN_COMMANDS,
                         frozenset({"vcf"}))

    def test_vcf_still_not_a_public_command(self):
        # The grant is group-scoped, NOT public membership: a DM
        # on the public bot must stay refused.
        self.assertNotIn("vcf", PUBLIC_COMMANDS)

    def test_caller_may_run_itself_unchanged(self):
        # The grant lives in the runtime beside caller_may_run
        # (the GROUP_DECISION plumbing), not inside the table.
        self.assertFalse(caller_may_run("vcf", False))
        self.assertTrue(caller_may_run("vcf", True))

    def test_decision_family_untouched(self):
        self.assertEqual(GROUP_DECISION_COMMANDS,
                         frozenset({"approve", "deny"}))

    def test_no_sibling_in_the_grant(self):
        for kind in ("db", "exec", "status", "quit", "power"):
            self.assertNotIn(kind, GROUP_ADMIN_COMMANDS, kind)
            self.assertNotIn(kind, GROUP_DECISION_COMMANDS, kind)


class _PipelineCase(unittest.TestCase):
    """One full runtime per row, fake adapter per surface."""

    def tearDown(self):
        with suppress(Exception):  # teardown hygiene only
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, platform: str, adapter: ChatAdapter,
             chat: ChatRef, sender_id: str,
             text: str = "/vcf") -> ChatAdapter:
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-test-vcf-grant-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": platform,
            "partner.owner_chats": OWNER_ID,
            "partner.owner_sender_ids": OWNER_ID,
        })
        self.context = build_context(
            settings, with_executor=False, with_tools=False)
        gateway = ChatGateway({platform: adapter},
                              db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=gateway)
        message = ChatMessage(
            chat=chat, incoming=True, text=text,
            sender=f"user{sender_id}", sender_id=sender_id, meta={})
        with mock.patch.object(
                self.runtime.brain, "handle_message",
                side_effect=lambda *a, **k: "BRAIN-REACHED"):
            self.runtime._process(message)
        for sent in adapter.sent:
            self.assertNotIn("BRAIN-REACHED", sent)
        return adapter

    # -- assertion helpers -------------------------------------

    def assert_cards_flowed(self, adapter: ChatAdapter) -> None:
        self.assertEqual(len(adapter.media), 1,
                         "exactly one contacts file must be sent")
        media, caption = adapter.media[0]
        self.assertTrue(
            str(getattr(media, "path", "")).endswith(".vcf"))
        self.assertIn("contact card", caption)
        joined = "\n".join(adapter.sent)
        self.assertIn("📇 sent", joined)
        self.assertNotIn(PUBLIC_BOT_DENIAL, joined)

    def assert_handler_refused(self, adapter: ChatAdapter) -> None:
        self.assertEqual(adapter.media, [])
        joined = "\n".join(adapter.sent)
        self.assertIn(HANDLER_REFUSAL, joined)
        self.assertNotIn(PUBLIC_BOT_DENIAL, joined)


def _tg_group(platform: str) -> ChatRef:
    return ChatRef(platform=platform, chat_id=TG_GROUP_ID,
                   kind=ChatKind.GROUP, title="Grant Group")


def _tg_dm(platform: str, chat_id: str) -> ChatRef:
    return ChatRef(platform=platform, chat_id=chat_id,
                   kind=ChatKind.DM, title="DM")


class PublicBotGroupTests(_PipelineCase):
    def test_group_admin_gets_the_cards(self):
        adapter = self._run(
            "telegram-bot", _FakeTelegram("telegram-bot"),
            _tg_group("telegram-bot"), ADMIN_ID)
        self.assert_cards_flowed(adapter)

    def test_group_owner_gets_the_cards(self):
        # The owner is not an operator on the gated surface; the
        # group grant + his creator standing carry the command.
        adapter = self._run(
            "telegram-bot", _FakeTelegram("telegram-bot"),
            _tg_group("telegram-bot"), OWNER_ID)
        self.assert_cards_flowed(adapter)

    def test_group_member_refused_by_the_handler(self):
        # The outer grant passes in-group; the HANDLER's admin
        # gate refuses, in the handler's own words — never the
        # public-bot gate text.
        adapter = self._run(
            "telegram-bot", _FakeTelegram("telegram-bot"),
            _tg_group("telegram-bot"), MEMBER_ID)
        self.assert_handler_refused(adapter)

    def test_sibling_control_command_still_gate_refused(self):
        adapter = self._run(
            "telegram-bot", _FakeTelegram("telegram-bot"),
            _tg_group("telegram-bot"), ADMIN_ID, text="/db tables")
        self.assertEqual(adapter.media, [])
        joined = "\n".join(adapter.sent)
        self.assertIn(PUBLIC_BOT_DENIAL, joined)
        self.assertNotIn("📇", joined)


class PublicBotDmTests(_PipelineCase):
    def test_dm_owner_still_refused_on_the_public_bot(self):
        adapter = self._run(
            "telegram-bot", _FakeTelegram("telegram-bot"),
            _tg_dm("telegram-bot", OWNER_ID), OWNER_ID)
        self.assertEqual(adapter.media, [])
        joined = "\n".join(adapter.sent)
        # The shipped owner-identity redirect (the gate stands;
        # he is told where the command lives) — unchanged.
        self.assertIn("doesn't run here", joined)
        self.assertNotIn("📇", joined)

    def test_dm_stranger_still_gets_the_public_denial(self):
        adapter = self._run(
            "telegram-bot", _FakeTelegram("telegram-bot"),
            _tg_dm("telegram-bot", "999"), "999")
        self.assertEqual(adapter.media, [])
        joined = "\n".join(adapter.sent)
        self.assertIn(PUBLIC_BOT_DENIAL, joined)


class PersonalSurfaceGroupTests(_PipelineCase):
    def test_group_owner_unchanged(self):
        adapter = self._run(
            "telegram-personal", _FakeTelegram("telegram-personal"),
            _tg_group("telegram-personal"), OWNER_ID)
        self.assert_cards_flowed(adapter)

    def test_group_admin_gets_the_cards(self):
        adapter = self._run(
            "telegram-personal", _FakeTelegram("telegram-personal"),
            _tg_group("telegram-personal"), ADMIN_ID)
        self.assert_cards_flowed(adapter)

    def test_group_member_refused_by_the_handler(self):
        adapter = self._run(
            "telegram-personal", _FakeTelegram("telegram-personal"),
            _tg_group("telegram-personal"), MEMBER_ID)
        self.assert_handler_refused(adapter)


class WhatsAppGroupTests(_PipelineCase):
    def _wa_group(self) -> ChatRef:
        return ChatRef(platform="whatsapp", chat_id="123@g.us",
                       kind=ChatKind.GROUP, title="WA Group")

    def test_group_admin_gets_the_cards(self):
        adapter = self._run(
            "whatsapp", _FakeWhatsApp(), self._wa_group(),
            "2348022222222")
        self.assert_cards_flowed(adapter)
        _media, caption = adapter.media[0]
        self.assertIn("2 contact cards", caption)

    def test_group_member_refused_by_the_handler(self):
        adapter = self._run(
            "whatsapp", _FakeWhatsApp(), self._wa_group(),
            "2348099999999")
        self.assert_handler_refused(adapter)


if __name__ == "__main__":
    unittest.main()
