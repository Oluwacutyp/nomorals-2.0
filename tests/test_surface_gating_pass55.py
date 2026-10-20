"""Pass #55 — surface gating on the Telegram bots (owner's model).

The owner's rule (2026-10-20): "the main bot is meant to be gated since
it's public. Only personal and userbot weren't supposed to be gated."
Gating is a property of the SURFACE, not the person:

* ``telegram-bot`` (the public bot) — GATED for everyone on it, owner
  included: restricted brain mode (no owner-private context), the
  PUBLIC command catalog only, owner-only commands neither listed nor
  runnable, owner-power NL hooks (money send, explicit post, ...) off.
* ``telegram-personal`` + ``telegram`` (userbot) — the owner's private
  surfaces: full owner treatment, unchanged.

These tests drive the real gating decision (``partner.gating``), the
real catalog renderers (``social.chat.control``), the real grant path
(``PartnerRuntime.handle_control`` / ``_is_operator``) and the real
inbound pipeline (``PartnerRuntime._process`` with a fake adapter).
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.partner.gating import (
    MODE_GROUP,
    MODE_OWNER,
    MODE_PRIVATE,
    effective_is_owner,
    gate_decision,
    is_gated_surface,
)
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
    is_owner_chat,
)
from nomorals.social.chat.control import detailed_help, list_catalog
from nomorals.social.chat.gateway import ChatGateway
from nomorals.storage.db import Database

OWNER_ID = "5478650254"
PUBLIC_BOT_DENIAL = (
    "that's not something I can do here — this is my public chat, "
    "so I keep it to conversation and games. "
    "try /game list to see what's on, or just talk to me."
)
OWNER_ONLY_KINDS = ("/update", "/pull", "/deploy", "/restart", "/stop",
                    "/exness", "/stripe", "/tempmail", "/db", "/exec",
                    "/status", "/quit")


def _chat(platform: str, kind: str = ChatKind.DM, chat_id: str = OWNER_ID):
    return SimpleNamespace(platform=platform, kind=kind, chat_id=chat_id,
                           key=f"{platform}:{chat_id}")


# ── 1. the gate decision is a surface property ────────────────────────

class GateDecisionSurfaceTests(unittest.TestCase):
    def test_owner_dm_on_public_bot_is_gated(self):
        mode = gate_decision(_chat("telegram-bot"), is_owner=True,
                             restricted_enabled=True)
        self.assertEqual(mode, MODE_PRIVATE)

    def test_owner_group_on_public_bot_is_gated(self):
        mode = gate_decision(_chat("telegram-bot", ChatKind.GROUP),
                             is_owner=True, restricted_enabled=True)
        self.assertEqual(mode, MODE_GROUP)

    def test_owner_dm_on_personal_bot_stays_owner(self):
        mode = gate_decision(_chat("telegram-personal"), is_owner=True,
                             restricted_enabled=True)
        self.assertEqual(mode, MODE_OWNER)

    def test_owner_dm_on_userbot_stays_owner(self):
        mode = gate_decision(_chat("telegram"), is_owner=True,
                             restricted_enabled=True)
        self.assertEqual(mode, MODE_OWNER)

    def test_stranger_modes_unchanged(self):
        self.assertEqual(
            gate_decision(_chat("telegram-bot", chat_id="999"),
                          is_owner=False, restricted_enabled=True),
            MODE_PRIVATE)
        self.assertEqual(
            gate_decision(_chat("telegram-personal", chat_id="999"),
                          is_owner=False, restricted_enabled=True),
            MODE_PRIVATE)

    def test_kill_switch_still_disables_all_gating(self):
        # Documented interaction: gate_restricted_chats=False is the
        # owner's global kill-switch — everything is owner mode,
        # surfaces included (preserved semantics, not a new hole).
        mode = gate_decision(_chat("telegram-bot"), is_owner=True,
                             restricted_enabled=False)
        self.assertEqual(mode, MODE_OWNER)

    def test_effective_is_owner(self):
        self.assertFalse(effective_is_owner(_chat("telegram-bot"), True))
        self.assertTrue(effective_is_owner(_chat("telegram-personal"), True))
        self.assertTrue(effective_is_owner(_chat("telegram"), True))
        self.assertFalse(effective_is_owner(_chat("telegram"), False))

    def test_is_gated_surface_accepts_ref_and_string(self):
        self.assertTrue(is_gated_surface(_chat("telegram-bot")))
        self.assertTrue(is_gated_surface("telegram-bot"))
        self.assertFalse(is_gated_surface(_chat("telegram-personal")))
        self.assertFalse(is_gated_surface("telegram"))
        self.assertFalse(is_gated_surface(""))

    def test_identity_test_itself_unchanged(self):
        # is_owner_chat (WHO the person is) is not the surface decision:
        # the bare owner id still matches on every platform.
        for platform in ("telegram-bot", "telegram-personal", "telegram"):
            chat = ChatRef(platform=platform, chat_id=OWNER_ID,
                           kind=ChatKind.DM)
            self.assertTrue(is_owner_chat(chat, owner_chats={OWNER_ID}))


# ── 2. the catalog on the gated surface is the PUBLIC list only ───────

class CatalogPublicOnlyTests(unittest.TestCase):
    def test_public_overview_excludes_owner_commands(self):
        text = detailed_help("", public_only=True)
        self.assertNotIn("full catalog", text)
        self.assertNotIn("platform control", text)
        for kind in OWNER_ONLY_KINDS:
            self.assertNotIn(kind, text, kind)

    def test_public_overview_includes_public_commands(self):
        text = detailed_help("", public_only=True)
        for kind in ("/search", "/play", "/wisdom", "/decode", "/news"):
            self.assertIn(kind, text, kind)

    def test_public_help_topic_for_owner_command_is_not_served(self):
        full = detailed_help("update")
        gated = detailed_help("update", public_only=True)
        self.assertNotEqual(gated, full)
        self.assertNotIn("self-update", gated)

    def test_public_help_topic_for_public_command_unchanged(self):
        self.assertEqual(detailed_help("search", public_only=True),
                         detailed_help("search"))

    def test_public_help_topic_pages_not_served(self):
        gated = detailed_help("budget", public_only=True)
        self.assertNotIn("model budget", gated)

    def test_public_list_catalog_excludes_owner_commands(self):
        text = list_catalog("", public_only=True)
        for kind in ("/update", "/exness", "/status", "/quit"):
            self.assertNotIn(kind, text, kind)
        self.assertIn("/search", text)

    def test_full_catalog_unchanged_without_flag(self):
        self.assertIn("full catalog", detailed_help(""))
        self.assertIn("/update", list_catalog(""))


# ── 3. operator power is a surface property too ───────────────────────

def _bare_runtime():
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(db=Database(":memory:"))
    rt._owner_chats = {OWNER_ID}
    rt._owner_sender_ids = {OWNER_ID}
    return rt


def _owner_msg(platform: str, text: str):
    chat = SimpleNamespace(key=f"{platform}:{OWNER_ID}", platform=platform,
                           kind=ChatKind.DM, chat_id=OWNER_ID)
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="owner",
                           sender_id=OWNER_ID)


class OperatorSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.rt = _bare_runtime()

    def test_owner_is_not_operator_on_public_bot(self):
        msg = _owner_msg("telegram-bot", "/status")
        self.assertFalse(self.rt._is_operator(msg))

    def test_owner_is_operator_on_private_surfaces(self):
        for platform in ("telegram-personal", "telegram"):
            msg = _owner_msg(platform, "/status")
            self.assertTrue(self.rt._is_operator(msg), platform)

    def test_owner_only_command_denied_on_public_bot(self):
        # /quit is confirmable: pre-fix the grant passed and the reply
        # was a confirm prompt (dispatch reached); the surface denial
        # must come back instead — the handler never runs.
        out = self.rt.handle_control(
            "/quit", f"telegram-bot:{OWNER_ID}",
            message=_owner_msg("telegram-bot", "/quit"))
        self.assertEqual(out, PUBLIC_BOT_DENIAL)

    def test_owner_only_command_still_dispatches_on_personal(self):
        out = self.rt.handle_control(
            "/quit", f"telegram-personal:{OWNER_ID}",
            message=_owner_msg("telegram-personal", "/quit"))
        self.assertNotEqual(out, PUBLIC_BOT_DENIAL)
        self.assertTrue(out.startswith("⚠️ confirm:"), out[:60])

    def test_stack_cannot_smuggle_owner_command_on_public_bot(self):
        out = self.rt.handle_control(
            "/help\n/mood", f"telegram-bot:{OWNER_ID}",
            message=_owner_msg("telegram-bot", "/help\n/mood"))
        self.assertIn("stack stopped before it started", out)
        self.assertIn("Nothing ran", out)
        self.assertNotIn("── 2/2", out)


# ── 4. the real inbound pipeline on the public bot ────────────────────

class _FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self.typing_calls: list[tuple[str, float, str]] = []
        self._handler = None

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, **kwargs: Any) -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def typing(self, chat: ChatRef, seconds: float = 3.0,
               action: str = "typing") -> bool:
        self.typing_calls.append((chat.key, seconds, action))
        return True


def _make_runtime(platform: str):
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-surface-gating-")
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": platform,
            "partner.owner_chats": OWNER_ID,
            "partner.owner_sender_ids": OWNER_ID,
        }
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = _FakeAdapter(platform)
    gateway = ChatGateway({platform: adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, context, tmp


def _inbound(platform: str, text: str) -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=OWNER_ID,
                     kind=ChatKind.DM, peer="owner"),
        incoming=True, text=text, sender="owner", sender_id=OWNER_ID,
        meta={},
    )


class PublicBotPipelineTests(unittest.TestCase):
    def tearDown(self):
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001
            pass
        self.context.close()
        self.tmp.cleanup()

    def _setup(self, platform: str) -> None:
        self.runtime, self.adapter, self.context, self.tmp = \
            _make_runtime(platform)

    def _run(self, platform: str, text: str) -> str:
        self._setup(platform)
        with mock.patch.object(
                self.runtime.brain, "handle_message",
                side_effect=AssertionError("brain must not see commands")):
            self.runtime._process(_inbound(platform, text))
        self.assertEqual(len(self.adapter.sent), 1,
                         f"expected exactly one reply on {platform}")
        return self.adapter.sent[0]

    def test_help_on_public_bot_is_public_catalog_with_typing(self):
        reply = self._run("telegram-bot", "/help")
        self.assertNotIn("full catalog", reply)
        for kind in ("/update", "/exness", "/tempmail"):
            self.assertNotIn(kind, reply, kind)
        self.assertIn("/search", reply)
        self.assertTrue(self.adapter.typing_calls,
                        "command replies on telegram-bot must show typing")

    def test_owner_command_on_public_bot_redirects_with_typing(self):
        reply = self._run("telegram-bot", "/update")
        self.assertEqual(reply, PUBLIC_BOT_DENIAL)
        self.assertTrue(self.adapter.typing_calls,
                        "the redirect is a reply too — typing must show")

    def test_public_command_on_public_bot_works_with_typing(self):
        reply = self._run("telegram-bot", "/decode")
        self.assertIn("usage:", reply)
        self.assertTrue(self.adapter.typing_calls)

    def test_help_on_personal_bot_stays_full_catalog(self):
        reply = self._run("telegram-personal", "/help")
        self.assertIn("full catalog", reply)
        self.assertIn("/update", reply)

    def test_money_hook_does_not_fire_on_public_bot(self):
        self._setup("telegram-bot")
        with mock.patch.object(
                self.runtime, "_money_send_hook",
                return_value="MONEY_HOOK_FIRED") as hook, \
                mock.patch.object(
                    self.runtime.brain, "handle_message",
                    side_effect=AssertionError("brain reached")):
            self.runtime._process(_inbound("telegram-bot", "send 5k to Mama"))
        hook.assert_not_called()
        self.assertEqual(self.adapter.sent, [])

    def test_money_hook_still_fires_on_personal_bot(self):
        self._setup("telegram-personal")
        with mock.patch.object(
                self.runtime, "_money_send_hook",
                return_value="MONEY_HOOK_FIRED") as hook:
            self.runtime._process(
                _inbound("telegram-personal", "send 5k to Mama"))
        hook.assert_called()
        self.assertEqual(self.adapter.sent, ["MONEY_HOOK_FIRED"])

    def test_brain_flags_identity_but_gate_mode_is_surface(self):
        # The brain still knows WHO the owner is (identity, style,
        # presence) — but the mode it must apply on the public bot is
        # the restricted one.
        self._setup("telegram-bot")
        chat = ChatRef(platform="telegram-bot", chat_id=OWNER_ID,
                       kind=ChatKind.DM)
        flags = self.runtime.brain._chat_flags(chat)
        self.assertTrue(flags["is_owner"])
        mode = gate_decision(chat, is_owner=flags["is_owner"],
                             restricted_enabled=True)
        self.assertEqual(mode, MODE_PRIVATE)


if __name__ == "__main__":
    unittest.main()
