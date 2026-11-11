"""Button-fix lane 1 — the /drafts review card, tap-tested BY EFFECT.

The button-effect audit (~/workspace/button-audit/AUDIT.md §1, §5)
found the draft card's organ and runtime route sound but the chain
broken ABOVE them: on WhatsApp the bridges cap native quick_reply
at 3 buttons and demote the overflow to inert text — Discard (#5)
and Edit (#4) were never tappable; on the Telegram Bot API the
chat_allow gate whitespace-split stamp payloads, so
``draft:discard:<ULID>`` classified as garbage; Schedule never got
an ``ask_schedule_time`` (state-free tap), Variants died silently
on seam-backed queues, and Edit was a prompt with no flow behind it.

This suite is the §5 harness for that card:

* **Harvest** — the card's real emitted payloads, collected from
  the emitters themselves.
* **Drive the real entry points** — the Bot API adapter's signed
  callback path (including a chat_allow-configured personal-bot
  variant), the MTProto userbot's callback-query path, and the
  WhatsApp adapter's bridge request (send-side form: a >3-action
  keyboard must go out in the list form, where every row is
  tappable — quick_reply would demote #4/#5 to text).
* **Assert STATE** — tap-path and typed-path end states are equal
  (discard discards, the schedule answer is stored, the edit
  answer replaces the content, variants persist on BOTH queue
  backends). Reply text alone never counts as proof.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social import drafts as drafts_mod
from nomorals.social.chat.control import callback_command_kind
from nomorals.social.chat.telegram import TelegramAdapter, TelegramBotAdapter
from nomorals.social.chat.tgbot_buttons import grid_layout
from nomorals.social.chat.whatsapp import WhatsAppAdapter
from nomorals.social.drafts import (
    DISCARDED,
    PENDING_REVIEW,
    SCHEDULED,
    DraftQueue,
    handle_draft_callback,
    propose_post,
    review_buttons,
)
from nomorals.storage.db import Database
from nomorals.storage.docstore import SQLiteDocumentStore

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402

_REAL_QUEUE = DraftQueue
_OWNER_ID = "5478650254"


class _FakeSocialManager:
    """The publisher seam: records publishes, posts to one platform."""

    instances: list[_FakeSocialManager] = []

    def __init__(self, context: Any) -> None:
        self.context = context
        self.published: list[tuple[str, Any]] = []

    def register_builtins(self) -> _FakeSocialManager:
        _FakeSocialManager.instances.append(self)
        return self

    def publish(self, content: str, platforms: Any = None,
                actor: str = "user", confirmation: str = "") -> Any:
        self.published.append((content, platforms))
        return SimpleNamespace(
            posted=["x"], failed=[], error="",
            results=[{"platform": "x", "ok": True}])


class _FakeScheduler:
    """The runtime scheduler's shape for schedule_post: an async
    ``schedule_once`` is NOT required — schedule_post accepts any
    return; what matters is the call and its timestamp."""

    def __init__(self) -> None:
        self.registered: list[str] = []
        self.once: list[tuple[str, float, str, dict]] = []

    def register_action(self, name: str, fn: Any) -> None:
        self.registered.append(name)

    def schedule_once(self, task_id: str, when_ts: float, action: str,
                      params: dict) -> None:
        self.once.append((task_id, when_ts, action, params))
        return None


class _FakeBotSession:
    """Records Bot API calls; never touches the network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, url, params, timeout=None):
        self.calls.append((url, dict(params)))
        return SimpleNamespace(
            json=lambda: {"ok": True, "result": True})

    def get(self, url, timeout=None):
        self.calls.append((url, {}))
        return SimpleNamespace(
            json=lambda: {"ok": True, "result": True})


class _RuntimeRig(unittest.TestCase):
    """PartnerRuntime over a tmp home (the ws3p2 harness shape),
    with the DraftQueue factory and SocialManager doubled — the
    dispatch, routing, queue semantics and callback router stay
    real."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix1-")
        self.db_path = str(Path(self.tmp.name) / "drafts.db")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": _OWNER_ID,
            "partner.owner_sender_ids": _OWNER_ID,
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        self.scheduler = _FakeScheduler()
        self.runtime._scheduler = self.scheduler
        _FakeSocialManager.instances = []
        self._patchers = [
            mock.patch.object(
                drafts_mod, "DraftQueue",
                lambda *a, **k: _REAL_QUEUE(db_path=self.db_path)),
            mock.patch("nomorals.social.SocialManager",
                       _FakeSocialManager),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in self._patchers:
            patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    # — seeding / reading the one real queue file ──────────────────
    def _seed(self, content: str, *, propose: bool = True) -> str:
        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            draft = queue.create_draft(content, ["x"])
            if propose:
                queue.propose(draft.id)
            return draft.id
        finally:
            queue.close()

    def _draft(self, draft_id: str) -> Any:
        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            return queue.get(draft_id)
        finally:
            queue.close()

    def _status(self, draft_id: str) -> str:
        draft = self._draft(draft_id)
        assert draft is not None
        return draft.status

    def _run(self, text: str, *, chat_key: str = "telegram:999",
             owner: bool = True) -> Any:
        return self.runtime.handle_control(
            text, chat_key,
            message=_msg(text, chat_key=chat_key, owner=owner))

    @staticmethod
    def _text(reply: Any) -> str:
        return reply.text if hasattr(reply, "text") else str(reply)


# ── harvest ────────────────────────────────────────────────────────


class HarvestTests(unittest.TestCase):
    def test_card_payloads_are_the_five_stamps(self) -> None:
        queue = DraftQueue(db_path=Path(
            tempfile.mkdtemp(prefix="nm-btnfix1-h-")) / "d.db")
        try:
            draft = queue.create_draft("harvest me", ["x"])
            queue.propose(draft.id)
            buttons = review_buttons(draft)
            self.assertEqual(
                [data for _label, data in buttons],
                [f"draft:post:{draft.id}",
                 f"draft:schedule:{draft.id}",
                 f"draft:variants:{draft.id}",
                 f"draft:edit:{draft.id}",
                 f"draft:discard:{draft.id}"])
            # Discard is #5, Edit #4 — the two the WhatsApp native
            # cap used to demote to inert text.
            self.assertTrue(buttons[4][1].endswith(":discard:" + draft.id)
                            or buttons[4][1] == f"draft:discard:{draft.id}")
            _text, standing = propose_post(draft)
            self.assertEqual(len(standing), 4)
            self.assertEqual(standing[-1][1], f"draft:discard:{draft.id}")
        finally:
            queue.close()


# ── the classifier (Break B) ───────────────────────────────────────


class ClassifierTests(unittest.TestCase):
    def test_stamp_payloads_classify_by_their_real_grammar(self) -> None:
        self.assertEqual(
            callback_command_kind("draft:discard:01JXYZABC"), "drafts")
        self.assertEqual(
            callback_command_kind("draft:post:01JXYZABC"), "drafts")
        self.assertEqual(
            callback_command_kind("/draft:schedule:01JXYZABC"), "drafts")
        self.assertEqual(
            callback_command_kind("cardtap ab12cd34 3 0"), "cardtap")

    def test_plain_commands_classify_by_first_word(self) -> None:
        self.assertEqual(
            callback_command_kind("research approve latest"), "research")
        self.assertEqual(callback_command_kind("/inventory"), "inventory")
        self.assertEqual(callback_command_kind("game trivia"), "game")
        self.assertEqual(
            callback_command_kind("proposals show upg_x"), "proposals")
        self.assertEqual(callback_command_kind(""), "")
        self.assertEqual(callback_command_kind("  "), "")


# ── Telegram Bot API surface ───────────────────────────────────────


def _personal_bot(chat_allow: str = "42") -> TelegramBotAdapter:
    """The personal bot's shape (owner_only, its own adapter name)
    with a chat_allow configured — the Break B regression rig."""
    return TelegramBotAdapter(
        token="<redacted>", session=_FakeBotSession(),
        chat_allow=chat_allow, adapter_name="telegram-personal",
        owner_only=True, owner_sender_ids=(_OWNER_ID,))


def _bot_query(adapter: TelegramBotAdapter, data: str,
               chat_id: str = "42") -> dict:
    return {
        "id": "q1",
        "from": {"id": int(_OWNER_ID), "is_bot": False,
                 "first_name": "Owner"},
        "message": {"chat": {"id": int(chat_id), "type": "private"},
                    "message_id": 5},
        "data": adapter._sign_callback(data),
    }


class BotApiTapTests(_RuntimeRig):
    """Signed Bot API callback → adapter → runtime → queue state."""

    def _tap(self, adapter: TelegramBotAdapter, data: str,
             chat_id: str = "42") -> tuple[list, list]:
        received: list = []
        replies: list = []

        def handler(msg: Any) -> None:
            received.append(msg)
            replies.append(self.runtime.handle_control(
                msg.text, msg.chat.key, message=msg))

        adapter._handle_callback(_bot_query(adapter, data, chat_id),
                                 handler)
        return received, replies

    def test_discard_tap_discards_like_the_typed_command(self) -> None:
        tapped = self._seed("botapi tapped draft")
        typed = self._seed("botapi typed draft")
        adapter = _personal_bot(chat_allow="42")
        received, replies = self._tap(adapter, f"draft:discard:{tapped}")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, f"/draft:discard:{tapped}")
        self.assertIn("discarded", self._text(replies[0]))
        self.assertEqual(self._status(tapped), DISCARDED)
        # The typed path on the twin draft lands in the same state.
        reply = self._run(f"/draft:discard:{typed}",
                          chat_key="telegram-personal:42")
        self.assertIn("discarded", self._text(reply))
        self.assertEqual(self._status(typed), DISCARDED)

    def test_non_allowlisted_chat_bounces_the_stamp_by_its_kind(self):
        draft_id = self._seed("foreign chat draft")
        adapter = _personal_bot(chat_allow="42")
        received, _replies = self._tap(
            adapter, f"draft:discard:{draft_id}", chat_id="99")
        # Owner-only kind ("drafts") in a chat outside the allowlist:
        # nothing is delivered and nothing moves — the verdict the
        # gate gives owner-only commands typed there too.
        self.assertEqual(received, [])
        self.assertEqual(self._status(draft_id), PENDING_REVIEW)
        toasts = [p.get("text", "") for u, p in adapter._session.calls
                  if u.endswith("answerCallbackQuery")]
        self.assertIn("Not authorized", toasts)

    def test_settled_card_late_tap_never_executes(self) -> None:
        draft_id = self._seed("settle me once")
        adapter = _personal_bot(chat_allow="42")
        self._tap(adapter, f"draft:discard:{draft_id}")
        self.assertEqual(self._status(draft_id), DISCARDED)
        # A stale client re-delivers the same tap: state must not
        # move again and nothing may post.
        self._tap(adapter, f"draft:discard:{draft_id}")
        self.assertEqual(self._status(draft_id), DISCARDED)
        self.assertEqual(_FakeSocialManager.instances, [])


# ── Telegram MTProto (userbot) surface ─────────────────────────────


class _FakeCallbackEvent:
    def __init__(self, data: bytes, chat_id: str = "42",
                 sender_id: str = _OWNER_ID) -> None:
        self.data = data
        self.chat_id = chat_id
        self.sender_id = sender_id
        self.message_id = 7

    async def answer(self) -> None:
        return None

    async def get_chat(self) -> None:
        return None

    async def get_sender(self) -> None:
        return None


class MtprotoTapTests(_RuntimeRig):
    def test_discard_tap_discards(self) -> None:
        draft_id = self._seed("userbot tapped draft")
        adapter = TelegramAdapter(api_id=123, api_hash="x",
                                  session_path=":memory:")
        received: list = []
        replies: list = []

        def handler(msg: Any) -> None:
            received.append(msg)
            replies.append(self.runtime.handle_control(
                msg.text, msg.chat.key, message=msg))

        event = _FakeCallbackEvent(f"draft:discard:{draft_id}".encode())
        asyncio.run(adapter._handle_callback_query(event, handler))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, f"/draft:discard:{draft_id}")
        self.assertIn("discarded", self._text(replies[0]))
        self.assertEqual(self._status(draft_id), DISCARDED)


# ── WhatsApp surface: the send-side form ───────────────────────────


def _wa_adapter(tmp: str, responder) -> tuple[WhatsAppAdapter, list]:
    adapter = WhatsAppAdapter(outbox_dir=str(Path(tmp) / "outbox"))
    adapter.connected.set()
    calls: list = []

    def fake_send_cmd(cmd, timeout=25.0):
        calls.append(dict(cmd))
        return responder(cmd)

    adapter._send_cmd = fake_send_cmd  # type: ignore[method-assign]
    return adapter, calls


class WhatsAppFormTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix1-wa-")
        queue = DraftQueue(db_path=Path(self.tmp.name) / "d.db")
        try:
            self.draft = queue.create_draft("whatsapp card draft", ["x"])
            queue.propose(self.draft.id)
        finally:
            queue.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_five_action_card_goes_out_in_list_form(self) -> None:
        adapter, calls = _wa_adapter(
            self.tmp.name, lambda cmd: {"ok": True, "message_id": "M1"})
        keyboard = grid_layout([review_buttons(self.draft)])
        out = adapter.send(
            _wa_chat(), "📝 draft review", buttons=keyboard)
        self.assertTrue(out.ok)
        send_cmd = [c for c in calls if c["cmd"] == "send"][0]
        # The list form is the only WhatsApp form where all five
        # rows stay tappable; quick_reply would demote #4/#5.
        self.assertEqual(send_cmd.get("buttons_style"), "list")
        flat = [data for row in send_cmd["buttons"] for _l, data in row]
        self.assertIn(f"draft:discard:{self.draft.id}", flat)
        self.assertIn(f"draft:edit:{self.draft.id}", flat)
        self.assertEqual(len(flat), 5)

    def test_four_action_standing_question_also_lists(self) -> None:
        adapter, calls = _wa_adapter(
            self.tmp.name, lambda cmd: {"ok": True, "message_id": "M1"})
        _text, standing = propose_post(self.draft)
        adapter.send(_wa_chat(), "📝 draft ready",
                     buttons=grid_layout([standing]))
        send_cmd = [c for c in calls if c["cmd"] == "send"][0]
        self.assertEqual(send_cmd.get("buttons_style"), "list")

    def test_three_button_keyboard_keeps_native_form(self) -> None:
        adapter, calls = _wa_adapter(
            self.tmp.name, lambda cmd: {"ok": True, "message_id": "M1"})
        adapter.send(_wa_chat(), "pick one", buttons=[
            [("A", "a"), ("B", "b"), ("C", "c")]])
        send_cmd = [c for c in calls if c["cmd"] == "send"][0]
        self.assertNotIn("buttons_style", send_cmd)

    def test_queued_card_flushes_in_list_form(self) -> None:
        adapter = WhatsAppAdapter(
            outbox_dir=str(Path(self.tmp.name) / "outbox"))
        keyboard = grid_layout([review_buttons(self.draft)])
        out = adapter.send(_wa_chat(), "📝 draft review",
                           buttons=keyboard)
        self.assertFalse(out.ok)  # bridge down — queued, not sent
        entries = adapter.outbox.pending()
        self.assertTrue(entries)
        calls: list = []
        adapter.connected.set()
        adapter._send_cmd = (  # type: ignore[method-assign]
            lambda cmd, timeout=25.0: (calls.append(dict(cmd)),
                                       {"ok": True,
                                        "message_id": "M9"})[1])
        self.assertTrue(adapter._deliver_entry(dict(entries[0])))
        send_cmd = [c for c in calls if c["cmd"] == "send"][0]
        self.assertEqual(send_cmd.get("buttons_style"), "list")


def _wa_chat():
    from nomorals.social.chat.base import ChatKind, ChatRef
    return ChatRef(platform="whatsapp",
                   chat_id="234801@s.whatsapp.net", kind=ChatKind.DM)


# ── Schedule: the tap asks, the answer is stored ───────────────────


class ScheduleFlowTests(_RuntimeRig):
    def test_tap_asks_then_the_answer_is_stored(self) -> None:
        draft_id = self._seed("schedule this one")
        reply = self._run(f"/draft:schedule:{draft_id}")
        self.assertIn("when should I post it", self._text(reply))
        # The tap alone changed nothing — it armed the ask.
        self.assertEqual(self._status(draft_id), PENDING_REVIEW)
        self.assertEqual(self.scheduler.once, [])
        ask = self.runtime._draft_ask_store().get("telegram:999")
        self.assertIsNotNone(ask)
        self.assertEqual(ask["draft_id"], draft_id)

        from nomorals.social.drafts import parse_post_time
        expected = parse_post_time("post at 6")
        answer = self.runtime._draft_ask_answer(
            _msg("post at 6", chat_key="telegram:999"))
        self.assertIsNotNone(answer)
        self.assertIn("scheduled for", answer)
        draft = self._draft(draft_id)
        self.assertEqual(draft.status, SCHEDULED)
        # The stored time is the answer's parse (within a couple of
        # seconds of this test's own parse — the two calls can
        # straddle a minute boundary), and the scheduler was armed
        # with exactly the stored instant.
        self.assertIsNotNone(draft.scheduled_time)
        assert draft.scheduled_time is not None
        self.assertAlmostEqual(draft.scheduled_time, expected, delta=2.0)
        self.assertEqual(len(self.scheduler.once), 1)
        self.assertEqual(self.scheduler.once[0][1], draft.scheduled_time)
        self.assertEqual(self.scheduler.once[0][3]["draft_id"], draft_id)
        # The ask is consumed: a later message is ordinary chat.
        self.assertNotIn("telegram:999", self.runtime._draft_ask_store())
        self.assertIsNone(self.runtime._draft_ask_answer(
            _msg("post at 7", chat_key="telegram:999")))
        self.assertEqual(self._draft(draft_id).scheduled_time,
                         draft.scheduled_time)

    def test_answer_flows_through_the_real_message_pipeline(self) -> None:
        # The _process_inner hook itself: an ordinary owner message
        # (not a command, not a tap) is captured as the armed ask's
        # answer before the brain or the explicit-post override can
        # see it, and the reply goes out through the gateway.
        draft_id = self._seed("pipeline answer draft")
        self._run(f"/draft:schedule:{draft_id}")
        self.runtime._process_inner(
            _msg("post at 6", chat_key="telegram:999"))
        sent = self.runtime.gateway.sent
        self.assertTrue(
            any("scheduled for" in str(entry) for entry in sent),
            f"gateway sends: {sent}")
        draft = self._draft(draft_id)
        self.assertEqual(draft.status, SCHEDULED)
        self.assertIsNotNone(draft.scheduled_time)
        self.assertEqual(len(self.scheduler.once), 1)

    def test_organ_route_with_answer_matches(self) -> None:
        # State after the runtime flow == state after the organ's
        # own ask_schedule_time seam driven directly.
        draft_id = self._seed("organ twin draft")
        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            from nomorals.social.drafts import parse_post_time
            when = parse_post_time("post at 6")
            reply = handle_draft_callback(
                queue, f"draft:schedule:{draft_id}",
                scheduler=self.scheduler,
                ask_schedule_time=lambda _id: when)
        finally:
            queue.close()
        self.assertIn("scheduled for", reply)
        draft = self._draft(draft_id)
        self.assertEqual(draft.status, SCHEDULED)
        self.assertEqual(draft.scheduled_time, when)

    def test_unparseable_answer_keeps_the_ask_armed(self) -> None:
        draft_id = self._seed("schedule eventually")
        self._run(f"/draft:schedule:{draft_id}")
        answer = self.runtime._draft_ask_answer(
            _msg("whenever you like", chat_key="telegram:999"))
        self.assertIsNotNone(answer)
        self.assertIn("couldn't read a time", answer)
        self.assertEqual(self._status(draft_id), PENDING_REVIEW)
        self.assertIn("telegram:999", self.runtime._draft_ask_store())
        answer = self.runtime._draft_ask_answer(
            _msg("cancel", chat_key="telegram:999"))
        self.assertIn("dropped", answer)
        self.assertNotIn("telegram:999", self.runtime._draft_ask_store())
        self.assertEqual(self._status(draft_id), PENDING_REVIEW)
        self.assertEqual(self.scheduler.once, [])


# ── Edit: the tap prompts, the answer replaces the content ─────────


class EditFlowTests(_RuntimeRig):
    def test_tap_prompts_then_the_answer_replaces_content(self) -> None:
        draft_id = self._seed("old wording that must go")
        reply = self._run(f"/draft:edit:{draft_id}")
        self.assertIn("send me the revised text", self._text(reply))
        self.assertEqual(self._draft(draft_id).content,
                         "old wording that must go")
        answer = self.runtime._draft_ask_answer(
            _msg("brand new wording", chat_key="telegram:999"))
        self.assertIsNotNone(answer)
        self.assertIn("draft updated", answer)
        self.assertEqual(self._draft(draft_id).content,
                         "brand new wording")
        self.assertNotIn("telegram:999", self.runtime._draft_ask_store())


# ── Variants: both queue backends, failures surface ────────────────


class VariantsBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix1-v-")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_variants(self, queue: DraftQueue) -> tuple[str, Any]:
        draft = queue.create_draft(
            "a perfectly ordinary draft about patience and timing",
            ["x"])
        queue.propose(draft.id)
        reply = handle_draft_callback(queue, f"draft:variants:{draft.id}")
        return reply, queue.get(draft.id)

    def test_variants_persist_on_native_queue(self) -> None:
        queue = DraftQueue(db_path=Path(self.tmp.name) / "n.db")
        try:
            reply, draft = self._run_variants(queue)
        finally:
            queue.close()
        self.assertIn("hook variants", reply)
        self.assertTrue((draft.metadata or {}).get("variants"))

    def test_variants_persist_on_seam_queue(self) -> None:
        # The audit's silent break: a seam-backed queue has _db
        # None; add_variant used to raise AttributeError into a
        # bare swallow and variants never landed.
        scratch = Database(":memory:")
        queue = DraftQueue(Path(self.tmp.name) / "s.db",
                           store=SQLiteDocumentStore(scratch))
        try:
            self.assertIsNone(queue._db)
            self.assertIsNotNone(queue._port)
            reply, draft = self._run_variants(queue)
        finally:
            queue.close()
            scratch.close()
        self.assertIn("hook variants", reply)
        self.assertTrue((draft.metadata or {}).get("variants"))

    def test_variant_failure_is_surfaced_not_swallowed(self) -> None:
        queue = DraftQueue(db_path=Path(self.tmp.name) / "f.db")
        try:
            draft = queue.create_draft("failure surface draft", ["x"])
            queue.propose(draft.id)
            with mock.patch.object(
                    drafts_mod, "add_variant",
                    side_effect=RuntimeError("store is down")):
                reply = handle_draft_callback(
                    queue, f"draft:variants:{draft.id}")
        finally:
            queue.close()
        self.assertIn("couldn't store hook variants", reply)
        self.assertIn("store is down", reply)


# ── the manifest knows this surface now ────────────────────────────


class ManifestTests(unittest.TestCase):
    def test_drafts_surface_registered(self) -> None:
        from nomorals.social.chat import surface_manifest as sm
        surface = sm.surface_by_id("drafts")
        self.assertIsNotNone(surface)
        assert surface is not None
        ids = {ref.action_id for ref in surface.actions}
        self.assertEqual(
            ids, {"drafts.open-card", "drafts.post", "drafts.schedule",
                  "drafts.variants", "drafts.edit", "drafts.discard"})


if __name__ == "__main__":
    unittest.main()
