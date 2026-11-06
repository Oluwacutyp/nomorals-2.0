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
import re
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

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
    MediaRef,
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


# ── (e) one command, one handling, across surfaces ─────────────────
#
# Owner screenshots, live, 2026-11-05: in his claimed Telegram group
# the owner's /rmdeleted@Nomorals_bot got TWO answers — the bot ran
# the scan while the as-him (userbot) surface posted the §33 gate
# text claiming the room "isn't switched on" (false: the bot is
# admin there and had just executed). /rminactive@Nomorals_bot did
# the same while the bot removed members. In the bot's DM, one
# /play minted TWO jobs (queue 10 and queue 11 — one per surface).
# The composed law: a recognized command is claimed by exactly one
# surface; every other surface that sees it stands down silently;
# the §33 gate and the executor read the SAME claim record.

GATE_MARKER = "isn't switched on for my group systems"


class _BotSurfaceAdapter(RecordingAdapter):
    """Bot API surface fake: carries the bot identity the real
    adapter learns from getMe (username + id) and the admin seams
    the cleanup roster reads. The bot is administrator in the
    claimed room (-1001) and a plain member in the other (-1002);
    the owner is an admin in both."""

    _bot_username = "Nomorals_bot"
    _bot_id = 777

    def admin_member(self, chat, user_id):  # type: ignore[no-untyped-def]
        uid = str(user_id)
        if uid == "777":
            status = ("administrator"
                      if str(chat.chat_id) == "-1001" else "member")
            return {"ok": True, "status": status}
        return {"ok": True, "status": "administrator"}

    def admin_administrators(self, chat):  # type: ignore[no-untyped-def]
        return {"ok": True, "administrators": [
            {"id": "4242", "status": "administrator"}]}

    def admin_members_full(self, chat):  # type: ignore[no-untyped-def]
        return {"ok": True, "members": [], "count": 0,
                "truncated": False}


class _PersonalSurfaceAdapter(RecordingAdapter):
    """Userbot surface fake: the owner's own account identity."""

    _me_username = "peacethefirst"
    _me_id = 4242


class OneCommandOneHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        self.adapters = {
            "whatsapp": RecordingAdapter("whatsapp"),
            "telegram": _PersonalSurfaceAdapter("telegram"),
            "telegram-personal": RecordingAdapter("telegram-personal"),
            "telegram-bot": _BotSurfaceAdapter("telegram-bot"),
            "discord": RecordingAdapter("discord"),
        }
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

    # -- faithful message shapes ------------------------------------
    def _owner_msg(self, platform: str, *, kind: ChatKind,
                   chat_id: str, text: str, incoming: bool,
                   message_id: str,
                   own: bool = False) -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                       title="Builders" if kind == ChatKind.GROUP
                       else "", peer=chat_id)
        meta: dict = {"is_owner": True}
        if own:
            meta["own_message"] = True
        msg = ChatMessage(chat=chat, incoming=incoming, text=text,
                          sender="death", message_id=message_id,
                          meta=meta)
        msg.sender_id = "4242"
        return msg

    def _group_pair(self, text: str, message_id: str,
                    chat_id: str = "-1001") -> tuple:
        """The same group message as Telegram delivers it twice:
        once to the bot (Bot API, incoming) and once to the
        owner's userbot (MTProto own echo)."""
        bot = self._owner_msg(
            "telegram-bot", kind=ChatKind.GROUP, chat_id=chat_id,
            text=text, incoming=True, message_id=message_id)
        personal = self._owner_msg(
            "telegram", kind=ChatKind.GROUP, chat_id=chat_id,
            text=text, incoming=False, message_id=message_id,
            own=True)
        return bot, personal

    def _claim_room_by_bot(self, chat_id: str = "-1001") -> None:
        surface = self.runtime._group_surface()
        surface.note_bot_admin(f"telegram-bot:{chat_id}", True)

    def _sent(self, platform: str) -> list:
        return [text for text, _b in self.adapters[platform].sent]

    # -- the incidents ----------------------------------------------
    def test_rmdeleted_bot_addressed_claimed_group_once(self) -> None:
        self._claim_room_by_bot()
        bot, personal = self._group_pair(
            "/rmdeleted@Nomorals_bot", "900")
        self.runtime._process_inner(bot)
        self.runtime._process_inner(personal)
        self.assertEqual(self._sent("telegram"), [],
                         "as-him surface answered a bot command")
        bot_sent = self._sent("telegram-bot")
        self.assertEqual(len(bot_sent), 1, bot_sent)
        self.assertNotIn(GATE_MARKER, bot_sent[0])

    def test_rminactive_bot_addressed_claimed_group_once(self) -> None:
        self._claim_room_by_bot()
        bot, personal = self._group_pair(
            "/rminactive@Nomorals_bot", "901")
        self.runtime._process_inner(bot)
        self.runtime._process_inner(personal)
        self.assertEqual(self._sent("telegram"), [],
                         "as-him surface answered a bot command")
        bot_sent = self._sent("telegram-bot")
        self.assertEqual(len(bot_sent), 1, bot_sent)
        self.assertNotIn(GATE_MARKER, bot_sent[0])

    def test_rmdeleted_personal_copy_first_still_bot_only(self) -> None:
        self._claim_room_by_bot()
        bot, personal = self._group_pair(
            "/rmdeleted@Nomorals_bot", "902")
        self.runtime._process_inner(personal)
        self.runtime._process_inner(bot)
        self.assertEqual(self._sent("telegram"), [])
        bot_sent = self._sent("telegram-bot")
        self.assertEqual(len(bot_sent), 1, bot_sent)
        self.assertNotIn(GATE_MARKER, bot_sent[0])

    def test_unaddressed_group_command_claimed_once(self) -> None:
        self._claim_room_by_bot()
        bot, personal = self._group_pair("/rmdeleted", "903")
        self.runtime._process_inner(bot)
        self.runtime._process_inner(personal)
        total = (len(self._sent("telegram"))
                 + len(self._sent("telegram-bot")))
        self.assertEqual(total, 1,
                         f"two handlings: {self._sent('telegram')} / "
                         f"{self._sent('telegram-bot')}")
        for text in (self._sent("telegram")
                     + self._sent("telegram-bot")):
            self.assertNotIn(GATE_MARKER, text)

    def test_unclaimed_group_gate_text_exactly_once(self) -> None:
        surface = self.runtime._group_surface()
        surface.note_bot_admin("telegram-bot:-1002", False)
        bot, personal = self._group_pair(
            "/rmdeleted@Nomorals_bot", "904", chat_id="-1002")
        self.runtime._process_inner(bot)
        self.runtime._process_inner(personal)
        all_sent = self._sent("telegram") + self._sent("telegram-bot")
        self.assertEqual(len(all_sent), 1, all_sent)
        self.assertIn(GATE_MARKER, all_sent[0])
        self.assertEqual(self._sent("telegram"), [])

    # -- the bot-DM double queue -------------------------------------
    def _patch_play(self) -> list:
        calls: list = []

        def fake_play(tail, chat_key=""):  # type: ignore[no-untyped-def]
            calls.append((tail, chat_key))
            n = len(calls)
            return (f"queued 1 (queue {n}):\n  - FEAR\n"
                    f"⏳ downloading in the background — I'll send "
                    f"it here when it's ready. [job job{n:04d}]")

        self.runtime._control_play = fake_play  # type: ignore[method-assign]
        return calls

    def _dm_pair(self, text: str, message_id: str) -> tuple:
        """The owner's message in the bot's DM, as both surfaces
        see it: the bot (incoming, chat = the owner) and the
        userbot (own echo, chat = the bot)."""
        bot = self._owner_msg(
            "telegram-bot", kind=ChatKind.DM, chat_id="4242",
            text=text, incoming=True, message_id=message_id)
        personal = self._owner_msg(
            "telegram", kind=ChatKind.DM, chat_id="777",
            text=text, incoming=False, message_id=message_id,
            own=True)
        return bot, personal

    def test_bot_dm_play_queues_exactly_one_job(self) -> None:
        calls = self._patch_play()
        bot, personal = self._dm_pair("/play fear by nf", "55")
        self.runtime._process_inner(bot)
        self.runtime._process_inner(personal)
        self.assertEqual(len(calls), 1,
                         f"/play ran {len(calls)} times (two jobs)")
        self.assertEqual(self._sent("telegram"), [])
        bot_sent = self._sent("telegram-bot")
        self.assertEqual(len(bot_sent), 1, bot_sent)
        self.assertIn("[job job0001]", bot_sent[0])

    def test_personal_self_chat_command_stays_personal(self) -> None:
        calls = self._patch_play()
        personal = self._owner_msg(
            "telegram", kind=ChatKind.DM, chat_id="4242",
            text="/play fear by nf", incoming=False, message_id="56",
            own=True)
        self.runtime._process_inner(personal)
        self.assertEqual(len(calls), 1)
        sent = self._sent("telegram")
        self.assertEqual(len(sent), 1, sent)
        self.assertIn("[job job0001]", sent[0])


# ── (e) presentation law: one message, edited in place ─────────────
#
# Owner rulings 2026-11-05 evening, from the live bot DM: a /play
# produced THREE pasted messages — a queue narration carrying an
# internal job id, the audio file, and a "sources tried … sent X
# to this chat" receipt.  The law: the ack is ONE message, edited
# in place as the job progresses; the artifact completes it; no
# receipts, no narration, no internal ids — on BOTH platforms.


class FlowAdapter(RecordingAdapter):
    """RecordingAdapter + the media/edit verbs the flow uses."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.files: list[tuple[str, str]] = []
        self.edits: list[tuple[str, str]] = []

    def send_file(self, chat: ChatRef, file_path: Any, *,
                  caption: str = "", filename: str = "") -> SendResult:
        self.files.append((str(file_path), caption))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"file{len(self.files)}")

    def send_media(self, chat: ChatRef, media: MediaRef, *,
                   caption: str = "",
                   parse_mode: str = "") -> SendResult:
        # The gateway's media verb (send_file routes through it).
        self.files.append((str(media.path), caption))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"file{len(self.files)}")

    def edit_message(self, chat: ChatRef, message_id: str,
                     text: str, **kwargs: Any) -> Any:
        self.edits.append((message_id, text))
        return {"ok": True}

    def edit_progress(self, chat: ChatRef, message_id: str,
                      text: str, **kwargs: Any) -> Any:
        self.edits.append((message_id, text))
        return {"ok": True}


class PresentationFlowTests(unittest.TestCase):
    """The REAL /play handler, end to end, on fake adapters for
    both platforms: ack → (progress edits) → one artifact → the ack
    edited to its final state, with zero receipts anywhere."""

    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        self.adapters = {name: FlowAdapter(name)
                         for name in ("telegram", "whatsapp")}
        self.gateway = ChatGateway(self.adapters, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.runtime._typing_enabled = lambda chat: False

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.context.close()
        self.tmp.cleanup()

    def _drive_play(self, platform: str) -> FlowAdapter:
        from nomorals.core.jobs import shared_registry
        from nomorals.media.picklist import PickCache
        from nomorals.media.resolver import ResolvedAudio

        PickCache.clear()
        adapter = self.adapters[platform]
        chat = ChatRef(platform=platform,
                       chat_id=f"flow-{platform}", kind=ChatKind.DM,
                       title="dm", peer=f"flow-{platform}")
        self.runtime._ref_from_key = lambda key: chat  # type: ignore[method-assign]
        yt = ResolvedAudio(
            ok=True, path_or_url="https://youtu.be/fear",
            title="FEAR", artist="NF", duration=213.0, kind="youtube",
            source_name="YouTube search (yt-dlp)")
        engine = mock.Mock()
        engine._enqueue.side_effect = lambda p, k, t, **kw: {
            "kind": k, "path": p, "title": t,
            "artist": kw.get("artist", ""),
            "duration": kw.get("duration", 0.0)}
        engine.queue.return_value = [{
            "kind": "youtube", "path": "https://youtu.be/fear",
            "title": "FEAR"}]
        engine.download.return_value = {
            "ok": True, "path": "/tmp/fear-full.mp3", "title": "FEAR"}
        with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
                mock.patch("nomorals.media.playback.PlaybackEngine",
                           return_value=engine):
            SR.return_value.resolve.return_value = yt
            SR.return_value.search_candidates_detailed.return_value = (
                [yt], ["youtube-search — ok (1)"])
            reply = self.runtime._control_play("fear by nf", chat.key)
        # The handler's own return carries nothing — the ack
        # message the flow sent IS the surface.
        self.assertEqual(reply, "")
        snaps = shared_registry().list(labels={
            "owner": "play", "chat": chat.key, "title": "FEAR"})
        self.assertTrue(snaps, "no background job was submitted")
        job = shared_registry().collect(snaps[-1]["id"], timeout=10)
        self.assertIsNotNone(job)
        self.assertTrue(job.done())
        PickCache.clear()
        return adapter

    def _assert_flow(self, platform: str) -> None:
        adapter = self._drive_play(platform)
        texts = [t for t, _b in adapter.sent]
        # The designed sequence: the ack first…
        self.assertTrue(texts and texts[0].startswith("⏳"),
                        f"{platform} sent: {texts}")
        # …exactly one audio artifact, captioned with the delivery
        # line (an expression GIF may ride along; it is not the
        # artifact and carries no caption)…
        audio = [f for f in adapter.files if f[0].endswith(".mp3")]
        self.assertEqual(len(audio), 1, adapter.files)
        self.assertEqual(audio[0][1], "🎵 FEAR — NF")
        # …and the ack EDITED to its final state — never left as a
        # stale "downloading…" above two more messages.
        self.assertTrue(adapter.edits,
                        f"{platform}: the ack was never edited")
        self.assertEqual(adapter.edits[-1][1], "🎵 FEAR — NF")
        # Zero receipts/narration/internal ids on ANY surface text.
        everything = "\n".join(
            texts + [e[1] for e in adapter.edits]
            + [c for _p, c in adapter.files])
        for banned in ("[job ", "sources tried", "to this chat",
                       "downloading in the background"):
            self.assertNotIn(banned, everything)
        self.assertIsNone(
            re.search(r"queued \d+ \(queue \d+\)", everything))

    def test_play_flow_telegram(self) -> None:
        self._assert_flow("telegram")

    def test_play_flow_whatsapp(self) -> None:
        self._assert_flow("whatsapp")


if __name__ == "__main__":
    unittest.main()
