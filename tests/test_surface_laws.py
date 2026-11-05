"""Personal-surface laws as executable gates (owner order 2026-11-05).

The owner's standing laws already forbade every defect in this
morning's WhatsApp self-chat screenshots — the failure class was
that the laws were PROSE, not gates. These tests pin the laws at
the personal-surface reply paths with faithful fakes:

(a) no reply text carrying internal narration (the router's
    ``(route: …)`` parenthetical, scaffold acknowledgments,
    "I'll report back" promises) crosses a personal-surface send
    boundary — register #33;
(b) control commands never answer empty on any surface —
    register #32 (``/sudo`` met total silence);
(c) when the owner himself already replied to a message/thread,
    Devon stays quiet on it, on every surface — register #28;
(d) reply copy may claim buttons only when buttons were attached,
    or the copy degrades to a working numbered fallback —
    register #27.

The leaked-string fixtures are assembled from fragments at import
time: the verbatim leak never appears as one literal in this file.
Context homes honour NM_TEST_TMPDIR (see test_partner_reply_guard
for why that matters on a small shared /tmp).
"""

from __future__ import annotations

import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner.failures import clear_failures
from nomorals.agents.partner.reply_guard import (
    narration_reason,
    scrub_narration,
)
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
from nomorals.social.chat.tgbot_buttons import actionable

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

_D = chr(8212)   # em dash
_RQ = chr(8217)  # right single quote
_LQ = chr(8220)  # left double quote
_RQQ = chr(8221)  # right double quote

#: The 2026-11-05 leak, fragment-assembled (see module docstring).
LEAKED = (
    "building " + _D + " " + _LQ
    + "a community called think peace think creativity" + _RQQ
    + ". I" + _RQ + "ll report back " + "here when it runs."
    + "\n(rou" + "te: cod" + "ing " + _D + " build verb, "
    + "no artifact noun " + _D + " model check next)")

CLEAN_ANSWER = "built it — the community page is live, 3 files written."
PIDGIN = "na wa o, the vibe for here na cruise, no wahala"


class RecordingAdapter(ChatAdapter):
    """Faithful fake: records text AND buttons per send."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, Any]] = []

    def run(self, handler) -> None:  # pragma: no cover - unused
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_context() -> tuple[Any, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-",
                                      dir=_TEST_TMPDIR)
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.enabled": "false",
        }
    )
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    _ws = Path(context.settings.workspace_dir)
    import json as _json
    for _marker in ("pulse_last_run.json",
                    "pulse_last_run_afternoon.json",
                    "pulse_last_run_evening.json",
                    "pulse_last_run_midnight.json"):
        (_ws / _marker).write_text(
            _json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    from nomorals.agents.morning_briefing import _today_str
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json,"
        " generated_at) VALUES (?, ?, '[]', ?)",
        ("test-seed", _today_str(context), time.time()),
    )
    return context, tmp


class _RuntimeCase(unittest.TestCase):
    PLATFORMS = ("whatsapp", "telegram", "telegram-personal",
                 "telegram-bot", "discord")

    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        self.adapters = {name: RecordingAdapter(name)
                         for name in self.PLATFORMS}
        self.gateway = ChatGateway(self.adapters, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.runtime.brain.presence_rng = random.Random(34)
        self.runtime._typing_enabled = lambda chat: False

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.context.close()
        self.tmp.cleanup()

    def _msg(self, platform: str, *, kind: ChatKind = ChatKind.GROUP,
             chat_id: str = "room1", text: str = "hello",
             incoming: bool = True, message_id: str = "",
             meta: dict | None = None) -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                       title="room", peer=chat_id)
        msg = ChatMessage(chat=chat, incoming=incoming, text=text,
                          sender="someone", message_id=message_id)
        if meta:
            msg.meta.update(meta)
        return msg


# ── (a) internal narration never crosses a personal send boundary ──

class NarrationDetectorTests(unittest.TestCase):
    def test_leaked_shape_is_narration(self) -> None:
        self.assertNotEqual(narration_reason(LEAKED), "")
        self.assertEqual(scrub_narration(LEAKED), "")

    def test_route_tail_stripped_from_real_answer(self) -> None:
        body = CLEAN_ANSWER + "\n(rou" + "te: coding " + _D + " x)"
        self.assertEqual(scrub_narration(body), CLEAN_ANSWER)

    def test_clean_text_byte_identical(self) -> None:
        for clean in (PIDGIN, CLEAN_ANSWER,
                      "the report back home was long, but fine"):
            self.assertEqual(narration_reason(clean), "")
            self.assertEqual(scrub_narration(clean), clean)


class NarrationBoundaryTests(_RuntimeCase):
    def test_mind_seam_holds_pure_narration_on_whatsapp(self) -> None:
        self.runtime._send_mind_reply(
            self._msg("whatsapp"), LEAKED)
        self.assertEqual(self.adapters["whatsapp"].sent, [])

    def test_mind_seam_holds_pure_narration_on_telegram_personal(
            self) -> None:
        self.runtime._send_mind_reply(
            self._msg("telegram"), LEAKED)
        self.assertEqual(self.adapters["telegram"].sent, [])

    def test_mind_seam_strips_route_from_real_answer(self) -> None:
        body = CLEAN_ANSWER + "\n(rou" + "te: coding " + _D + " x)"
        self.runtime._send_mind_reply(self._msg("whatsapp"), body)
        sent = self.adapters["whatsapp"].sent
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], CLEAN_ANSWER)
        self.assertNotIn("route", sent[0][0])

    def test_conversational_boundary_holds_narration(self) -> None:
        self.runtime._send_reply(self._msg("whatsapp"), [LEAKED])
        for text, _buttons in self.adapters["whatsapp"].sent:
            self.assertEqual(narration_reason(text), "")
            self.assertNotIn("report back", text)

    def test_coremind_started_notes_carry_no_route_line(self) -> None:
        import inspect

        from nomorals.agents import coremind
        src = inspect.getsource(coremind.CoreMind)
        self.assertNotIn("self._route_line(intent)", src)
        for phrase in ("findings land here",
                       "I'll post the result here",
                       "I'll confirm when it lands",
                       "I'll send it when"):
            self.assertNotIn(phrase, src)


# ── (b) control commands never answer empty ─────────────────────────

class ControlNeverEmptyTests(_RuntimeCase):
    def test_sudo_answers_on_whatsapp(self) -> None:
        reply = self.runtime.handle_control(
            "/sudo", "whatsapp:15551234567", message=None)
        self.assertIsInstance(reply, str)
        self.assertTrue(reply.strip())
        self.assertIn("sudo", reply.lower())

    def test_sudo_answers_on_telegram_personal(self) -> None:
        reply = self.runtime.handle_control(
            "/sudo", "telegram:4242", message=None)
        self.assertTrue(str(reply).strip())

    def test_sudo_add_delegates_to_owner_grammar(self) -> None:
        reply = self.runtime.handle_control(
            "/sudo add whatsapp 1555000111",
            "whatsapp:15551234567", message=None)
        self.assertTrue(str(reply).strip())
        state = self.runtime.handle_control(
            "/sudo", "whatsapp:15551234567", message=None)
        self.assertIn("1555000111", str(state))

    def test_representative_commands_answer(self) -> None:
        for cmd in ("/ping", "/help sudo", "/owner", "/status",
                    "/toolhub help", "/mode"):
            reply = self.runtime.handle_control(
                cmd, "whatsapp:15551234567", message=None)
            self.assertTrue(
                str(getattr(reply, "text", reply) or "").strip(),
                f"{cmd} answered empty")

    def test_empty_handler_result_becomes_honest_fallback(self) -> None:
        original = PartnerRuntime._control_ping
        PartnerRuntime._control_ping = (  # type: ignore[method-assign]
            lambda self, message=None: "")
        try:
            reply = self.runtime.handle_control(
                "/ping", "whatsapp:15551234567", message=None)
        finally:
            PartnerRuntime._control_ping = original  # type: ignore[method-assign]
        self.assertIn("produced no reply", str(reply))

    def test_non_command_text_keeps_empty_contract(self) -> None:
        reply = self.runtime.handle_control(
            "just chatting here", "whatsapp:15551234567",
            message=None)
        self.assertEqual(reply, "")


# ── (c) owner-replied suppression, per surface ──────────────────────

class OwnerReplySuppressionTests(_RuntimeCase):
    def _owner_echo(self, platform: str, *, chat_id: str = "room1",
                    reply_to: str = "") -> ChatMessage:
        return self._msg(
            platform, chat_id=chat_id, text="handled it myself",
            incoming=False, meta={"own_message": True,
                                  "from_owner": True,
                                  "reply_to": reply_to})

    def _inbound(self, platform: str, *, chat_id: str = "room1",
                 message_id: str = "") -> ChatMessage:
        return self._msg(platform, chat_id=chat_id,
                         text="what is the plan for today?",
                         message_id=message_id)

    def _assert_suppressed(self, platform: str) -> None:
        echo = self._owner_echo(platform)
        self.runtime._process_inner(echo)
        outcome = self.runtime.brain.handle_message(
            self._inbound(platform))
        self.assertFalse(outcome.parts,
                         f"{platform}: Devon answered over the owner")

    def test_whatsapp_owner_reply_suppresses(self) -> None:
        self._assert_suppressed("whatsapp")

    def test_telegram_userbot_owner_reply_suppresses(self) -> None:
        self._assert_suppressed("telegram")

    def test_telegram_bot_surface_owner_reply_suppresses(self) -> None:
        self._assert_suppressed("telegram-bot")

    def test_discord_owner_reply_suppresses(self) -> None:
        self._assert_suppressed("discord")

    def test_targeted_reply_suppresses_that_message(self) -> None:
        ledger = self.runtime.brain.owner_reply_ledger()
        ledger.record_owner_reply("whatsapp:room1", reply_to="m-42")
        self.assertEqual(
            ledger.owner_answered("whatsapp:room1", "m-42"),
            "owner-replied-target")

    def test_delayed_delivery_rechecks_suppression(self) -> None:
        msg = self._inbound("whatsapp")
        echo = self._owner_echo("whatsapp")
        self.runtime._process_inner(echo)
        outcome = self.runtime.brain.deliver_reply(msg)
        self.assertFalse(outcome.parts)

    def test_unrelated_chat_still_answered(self) -> None:
        echo = self._owner_echo("whatsapp", chat_id="room1")
        self.runtime._process_inner(echo)
        ledger = self.runtime.brain.owner_reply_ledger()
        self.assertEqual(
            ledger.owner_answered("whatsapp:other-room", ""), "")

    def test_commands_still_answer_after_owner_reply(self) -> None:
        echo = self._owner_echo("whatsapp")
        self.runtime._process_inner(echo)
        reply = self.runtime.handle_control(
            "/sudo", "whatsapp:room1", message=None)
        self.assertTrue(str(reply).strip())


class TelegramOutgoingEchoTests(unittest.TestCase):
    """The MTProto adapter delivers owner-sent group messages as
    own-echoes (previously dropped wholesale)."""

    def _adapter(self) -> Any:
        from nomorals.social.chat.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.name = "telegram"
        adapter._me_id = 777
        adapter._sent_ids = set()
        adapter.stats = {"received": 0}
        return adapter

    class _Event:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    def test_owner_group_message_becomes_echo(self) -> None:
        adapter = self._adapter()
        got: list[ChatMessage] = []
        reply_head = type("H", (), {"reply_to_msg_id": 55})()
        message = self._Event(
            id=99, action=None, message="sorted, check the file",
            reply_to=reply_head, date=None)
        event = self._Event(message=message, chat_id=-100123)
        adapter._deliver_owner_echo(event, got.append)
        self.assertEqual(len(got), 1)
        echo = got[0]
        self.assertFalse(echo.incoming)
        self.assertTrue(echo.meta.get("own_message"))
        self.assertEqual(echo.reply_to, "55")
        self.assertEqual(echo.chat.key, "telegram:-100123")

    def test_devon_own_send_is_not_an_echo(self) -> None:
        adapter = self._adapter()
        adapter._sent_ids = {99}
        got: list[ChatMessage] = []
        message = self._Event(id=99, action=None, message="mine",
                              reply_to=None, date=None)
        event = self._Event(message=message, chat_id=-100123)
        adapter._deliver_owner_echo(event, got.append)
        self.assertEqual(got, [])


# ── (d) button claim ⇒ buttons attached, or working fallback ────────

class ButtonHonestyTests(_RuntimeCase):
    WA_KEY = "whatsapp:15551234567"

    def _wa_msg(self) -> ChatMessage:
        return self._msg("whatsapp", kind=ChatKind.DM,
                         chat_id="15551234567")

    def test_toolhub_on_whatsapp_attaches_real_buttons(self) -> None:
        reply = self.runtime.handle_control(
            "/toolhub", self.WA_KEY, message=None)
        self.runtime._send_control_reply(self._wa_msg(), reply)
        sent = self.adapters["whatsapp"].sent
        self.assertEqual(len(sent), 1)
        text, buttons = sent[0]
        self.assertTrue(buttons, "claim copy sent without buttons")
        flat = [data for row in buttons for _label, data in row]
        self.assertIn("toolhub proxy", flat)
        self.assertIn("toolhub bin", flat)

    def test_no_button_platform_gets_numbered_fallback(self) -> None:
        msg = self._msg("discord", kind=ChatKind.DM, chat_id="u1")
        card = actionable(
            "Devon Hub — tap a tool below:\n\n"
            "Tap a button below to run it.",
            [[("🌐 Proxy", "toolhub proxy"),
              ("💳 BIN lookup", "toolhub bin")]])
        self.runtime._send_control_reply(msg, card)
        sent = self.adapters["discord"].sent
        self.assertEqual(len(sent), 1)
        text, buttons = sent[0]
        self.assertIsNone(buttons)
        self.assertNotIn("Tap a button below", text)
        self.assertIn("1 — 🌐 Proxy", text)
        self.assertIn("2 — 💳 BIN lookup", text)
        # The fallback executes: a bare "1" runs the pick's command.
        pick = self._msg("discord", kind=ChatKind.DM, chat_id="u1",
                         text="1")
        out = self.runtime._button_pick_hook(pick)
        self.assertIsNotNone(out)
        self.assertTrue(str(getattr(out, "text", out)).strip())

    def test_buttonless_claim_copy_is_rewritten(self) -> None:
        msg = self._msg("discord", kind=ChatKind.DM, chat_id="u1")
        self.runtime._send_control_reply(
            msg, "Devon Hub — tap a tool below:\n\n"
                 "Tap a button below to run it.")
        text, _buttons = self.adapters["discord"].sent[0]
        self.assertNotIn("tap a tool below", text.lower())
        self.assertNotIn("Tap a button below", text)

    def test_gateway_forwards_buttons_to_whatsapp(self) -> None:
        chat = ChatRef(platform="whatsapp", chat_id="15551234567",
                       kind=ChatKind.DM)
        self.gateway.send(
            "whatsapp", chat, "pick one",
            buttons=[[("🌐 Proxy", "toolhub proxy")]])
        _text, buttons = self.adapters["whatsapp"].sent[0]
        self.assertEqual(
            buttons, [[("🌐 Proxy", "toolhub proxy")]])


if __name__ == "__main__":
    unittest.main()
