"""Photo receipt honesty + conversational exactly-once (2026-11-06).

The incident (owner's live Telegram, 2026-11-06 04:32): a contact
sent a photo — an MT5/NFP screenshot, plainly visible in the chat.
The owner's account replied TWICE in the same minute with the
identical text "photo didn't come through", and the owner reports
the same false claim on WhatsApp. The photo HAD arrived; the
vision stack was degraded (a "model failed on iteration 1"
fragment was on his phone the same morning).

Root causes pinned here:

* the brain's media note for an unviewable photo read
  "photo (could not see it)" with NO degraded marking, so the
  model paraphrased a vision failure into a false RECEIPT claim
  and the competence gate — which only sees the responder's
  fallback provenance — passed the reply as healthy;
* adapters erased failed downloads: the Telegram userbot's
  download exception was debug-logged and the message proceeded
  with media=[]; the WhatsApp conversion filtered bridge media
  entries without a path out silently. The brain could not tell
  "no photo sent" from "photo sent, fetch failed";
* conversational replies had no exactly-once arbitration: the
  command ledger covers commands only, adapter seen-sets are
  in-memory/age-based, the runtime idempotency scope covers tool
  effects (not the reply), and a redelivery persists a FRESH
  inbound row so the brain's already-addressed guard structurally
  cannot fire — the same message generated and sent twice.

These tests pin the fixes end to end: a received photo is never
claimable as not-received; an unviewable photo on a personal
surface means silence (the observe-only law covers vision
degradation, not just responder fallback); and one platform
message identity gets exactly one conversational reply.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner.failures import clear_failures
from nomorals.agents.partner.reply_claims import (
    ReplyClaimLedger,
    reply_claim_key,
)
from nomorals.agents.partner_runtime import PartnerBrain, PartnerRuntime
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
from nomorals.social.chat.whatsapp import WhatsAppAdapter

#: See test_partner_brain_gate for why context homes honour
#: NM_TEST_TMPDIR (the /tmp tmpfs + swallowed-COMMIT trap).
_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

INCIDENT_TEXT = "photo didn't come through"


def _tools_stub(*, vision_ok: bool, description: str = "") -> Any:
    def _call(name: str, **kw: Any) -> Any:
        if name == "vision_describe" and vision_ok:
            return SimpleNamespace(
                ok=True, value={"description": description})
        return SimpleNamespace(ok=False, value=None)

    return SimpleNamespace(call=_call)


def _make_brain(tmp: str) -> PartnerBrain:
    settings = load_settings(
        overrides={"home": tmp, "partner.platforms": "local",
                   "chat.local_enabled": "true"}
    )
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    brain = PartnerBrain(context)
    brain.presence_rng = random.Random(34)
    return brain


def _stub_bundle(brain: PartnerBrain, parts: list[str]) -> None:
    bundle = SimpleNamespace(parts=parts, model="fake-7b",
                             degraded=False, degraded_note="")
    brain.responder.respond_bounded = mock.MagicMock(
        return_value=bundle)


def _photo_message(tmp: str, *, with_path: bool = True) -> ChatMessage:
    path = ""
    if with_path:
        p = Path(tmp) / "nfp-shot.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        path = str(p)
    chat = ChatRef(platform="local", chat_id="peer1",
                   kind=ChatKind.DM, peer="peer1")
    return ChatMessage(
        chat=chat, incoming=True, text="bro check this",
        sender="faheem", message_id="tg-1001",
        media=[MediaRef(path=path, mime="image/jpeg", kind="image")])


# ── the brain's media note + degraded fact ──────────────────────────


class PhotoNoteTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-photo-", dir=_TEST_TMPDIR)
        self.brain = _make_brain(self.tmp.name)

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.brain.context.close()
        self.tmp.cleanup()

    def test_vision_healthy_describes_the_photo(self) -> None:
        self.brain.context.tools = _tools_stub(
            vision_ok=True,
            description="an MT5 chart screenshot about NFP")
        msg = _photo_message(self.tmp.name)
        notes = self.brain._media_notes(msg)
        self.assertEqual(len(notes), 1)
        self.assertIn("an MT5 chart screenshot about NFP", notes[0])
        self.assertNotIn("DID arrive", notes[0])
        self.assertFalse((msg.meta or {}).get("photo_unseen"))

    def test_vision_down_note_states_receipt_and_forbids_the_claim(
            self) -> None:
        self.brain.context.tools = _tools_stub(vision_ok=False)
        msg = _photo_message(self.tmp.name)
        notes = self.brain._media_notes(msg)
        self.assertEqual(len(notes), 1)
        self.assertIn("DID arrive", notes[0])
        self.assertIn("never say it didn't", notes[0])
        self.assertTrue(msg.meta.get("photo_unseen"))

    def test_no_vision_tool_at_all_marks_unseen(self) -> None:
        self.brain.context.tools = None
        msg = _photo_message(self.tmp.name)
        notes = self.brain._media_notes(msg)
        self.assertEqual(len(notes), 1)
        self.assertIn("DID arrive", notes[0])
        self.assertTrue(msg.meta.get("photo_unseen"))

    def test_fetched_nothing_placeholder_path_marks_unseen(
            self) -> None:
        # The bot adapter's download-failure shape: a MediaRef
        # with an empty path. Vision must not even be attempted
        # against "" — the photo is unseen, not undescribed.
        self.brain.context.tools = _tools_stub(
            vision_ok=True, description="should never be used")
        msg = _photo_message(self.tmp.name, with_path=False)
        notes = self.brain._media_notes(msg)
        self.assertEqual(len(notes), 1)
        self.assertIn("DID arrive", notes[0])
        self.assertNotIn("should never be used", notes[0])
        self.assertTrue(msg.meta.get("photo_unseen"))

    def test_adapter_stamped_unavailable_photo_marks_unseen(
            self) -> None:
        # The adapters' shape when the download itself failed:
        # no MediaRef at all, the arrival stamped on meta.
        self.brain.context.tools = _tools_stub(vision_ok=True)
        chat = ChatRef(platform="local", chat_id="peer1",
                       kind=ChatKind.DM, peer="peer1")
        msg = ChatMessage(chat=chat, incoming=True,
                          text="bro check this", sender="faheem",
                          message_id="tg-1002")
        msg.meta["media_unavailable"] = [
            {"kind": "image", "reason": "download_failed"}]
        notes = self.brain._media_notes(msg)
        self.assertEqual(len(notes), 1)
        self.assertIn("DID arrive", notes[0])
        self.assertTrue(msg.meta.get("photo_unseen"))

    def test_unseen_photo_makes_the_turn_degraded(self) -> None:
        # A healthy TEXT model with a dead vision rung: the
        # outcome must read degraded so the competence gate holds
        # the reply on personal surfaces. Before the fix this
        # returned degraded=False and the incident text sent.
        self.brain.context.tools = _tools_stub(vision_ok=False)
        _stub_bundle(self.brain, [INCIDENT_TEXT])
        msg = _photo_message(self.tmp.name)
        parts, degraded, note = self.brain._generate_and_persist(
            msg, {"is_owner": False, "in_us": False})
        self.assertEqual(parts, [INCIDENT_TEXT])
        self.assertTrue(degraded)
        self.assertIn("vision", note)

    def test_seen_photo_turn_is_not_degraded(self) -> None:
        self.brain.context.tools = _tools_stub(
            vision_ok=True, description="an MT5 chart screenshot")
        _stub_bundle(self.brain, ["na the nfp print be that"])
        msg = _photo_message(self.tmp.name)
        _parts, degraded, note = self.brain._generate_and_persist(
            msg, {"is_owner": False, "in_us": False})
        self.assertFalse(degraded)
        self.assertEqual(note, "")


# ── the adapter stamps ──────────────────────────────────────────────


class WhatsAppMissedMediaTests(unittest.TestCase):
    def test_bridge_media_without_path_is_stamped_not_erased(
            self) -> None:
        adapter = WhatsAppAdapter()
        msgs: list = []
        adapter._dispatch({
            "type": "message",
            "id": "WA-9911",
            "chat": {"id": "2348012345678@c.us", "kind": "dm",
                     "title": "Faheem"},
            "from": {"id": "2348012345678@c.us", "name": "Faheem"},
            "text": "bro check this",
            "media": [{"mime": "image/jpeg", "kind": "photo"}],
            "ts": int(time.time() * 1000),
        }, msgs.append)
        self.assertEqual(len(msgs), 1)
        msg = msgs[0]
        self.assertEqual(msg.media, [])
        self.assertEqual(
            msg.meta.get("media_unavailable"),
            [{"kind": "photo", "reason": "download_failed"}])

    def test_bridge_media_with_path_is_untouched(self) -> None:
        adapter = WhatsAppAdapter()
        msgs: list = []
        adapter._dispatch({
            "type": "message",
            "id": "WA-9912",
            "chat": {"id": "2348012345678@c.us", "kind": "dm",
                     "title": "Faheem"},
            "from": {"id": "2348012345678@c.us", "name": "Faheem"},
            "text": "bro check this",
            "media": [{"path": "/media/a.jpg", "mime": "image/jpeg",
                       "kind": "photo"}],
            "ts": int(time.time() * 1000),
        }, msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(len(msgs[0].media), 1)
        self.assertNotIn("media_unavailable", msgs[0].meta)


# ── the reply-claim ledger ──────────────────────────────────────────


class ReplyClaimLedgerTests(unittest.TestCase):
    def _msg(self, platform: str = "telegram", chat_id: str = "42",
             message_id: str = "100", **meta: Any) -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id=chat_id,
                       kind=ChatKind.DM, peer=chat_id)
        msg = ChatMessage(chat=chat, incoming=True, text="hi",
                          sender="faheem", message_id=message_id)
        msg.meta.update(meta)
        return msg

    def test_first_claim_wins_second_stands_down(self) -> None:
        ledger = ReplyClaimLedger()
        key = reply_claim_key(self._msg())
        self.assertTrue(ledger.claim(key, now=1000.0))
        self.assertFalse(ledger.claim(key, now=1001.0))

    def test_claim_ages_out_of_the_window(self) -> None:
        ledger = ReplyClaimLedger()
        key = reply_claim_key(self._msg())
        self.assertTrue(ledger.claim(key, now=1000.0))
        self.assertTrue(ledger.claim(key, now=1000.0 + 601.0))

    def test_telegram_family_collapses_like_command_claims(
            self) -> None:
        a = reply_claim_key(self._msg(platform="telegram"))
        b = reply_claim_key(self._msg(platform="telegram-bot"))
        c = reply_claim_key(self._msg(platform="telegram-personal"))
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, "telegram:42:100")

    def test_unkeyable_messages_always_pass(self) -> None:
        ledger = ReplyClaimLedger()
        self.assertEqual(reply_claim_key(self._msg(message_id="")), "")
        self.assertTrue(ledger.claim(""))
        self.assertTrue(ledger.claim(""))

    def test_edits_are_not_redeliveries(self) -> None:
        self.assertEqual(
            reply_claim_key(self._msg(edited=True)), "")


# ── the runtime seam: full turns ────────────────────────────────────


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self._handler = None
        self.started_flag = False

    def run(self, handler) -> None:
        self._handler = handler
        self.started_flag = True
        while not self.stopped:
            time.sleep(0.01)

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def wait_started(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.started_flag:
                return True
            time.sleep(0.01)
        return False


def _wait(predicate, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _make_runtime_context() -> tuple[Any, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-photo-rt-",
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
    ws = Path(context.settings.workspace_dir)
    for marker in ("pulse_last_run.json",
                   "pulse_last_run_afternoon.json",
                   "pulse_last_run_evening.json",
                   "pulse_last_run_midnight.json"):
        (ws / marker).write_text(
            json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    from nomorals.agents.morning_briefing import _today_str
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json,"
        " generated_at) VALUES (?, ?, '[]', ?)",
        ("test-seed", _today_str(context), time.time()),
    )
    return context, tmp


class RuntimeSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_runtime_context()
        self.adapters = {
            name: FakeAdapter(name)
            for name in ("whatsapp", "telegram", "telegram-personal",
                         "telegram-bot")
        }
        self.gateway = ChatGateway(self.adapters, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.runtime.brain.presence_rng = random.Random(34)
        self.runtime._typing_enabled = lambda chat: False
        self.runtime.start()
        self.addCleanup(self._teardown)

    def _teardown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        with contextlib.suppress(Exception):
            self.context.close()
        self.tmp.cleanup()

    def _group_msg(self, platform: str, *, message_id: str,
                   text: str = "wren, did you see the nfp numbers",
                   media: list | None = None) -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id="grp-1",
                       kind=ChatKind.GROUP, title="Traders",
                       peer="grp-1")
        return ChatMessage(chat=chat, incoming=True, text=text,
                           sender="faheem", message_id=message_id,
                           media=media or [], ts=time.time())

    def test_redelivered_message_gets_exactly_one_reply(self) -> None:
        _stub_bundle(self.runtime.brain,
                     ["na the nfp print be that"])
        adapter = self.adapters["telegram"]
        self.assertTrue(adapter.wait_started())
        self.runtime.on_message(
            self._group_msg("telegram", message_id="100"))
        self.assertTrue(_wait(lambda: len(adapter.sent) == 1),
                        "first delivery never answered")
        # The same platform message delivered again (bridge
        # replay / reconnect redelivery): a fresh ChatMessage
        # object carrying the same identity.
        self.runtime.on_message(
            self._group_msg("telegram", message_id="100"))
        time.sleep(1.5)
        self.assertEqual(adapter.sent, ["na the nfp print be that"])
        stats = self.runtime._stats_snapshot()
        self.assertGreaterEqual(
            stats.get("reply_claim_held", 0), 1)

    def test_unseen_photo_on_personal_surface_is_silence(self) -> None:
        # Vision down, text model "healthy" and scripted to
        # produce the incident line: on the owner's own surfaces
        # nothing may be sent.
        self.context.tools = _tools_stub(vision_ok=False)
        _stub_bundle(self.runtime.brain, [INCIDENT_TEXT])
        adapter = self.adapters["telegram"]
        self.assertTrue(adapter.wait_started())
        photo = Path(self.tmp.name) / "nfp-shot.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        self.runtime.on_message(self._group_msg(
            "telegram", message_id="101",
            text="wren, check this photo",
            media=[MediaRef(path=str(photo), mime="image/jpeg",
                            kind="image")]))
        # The turn IS processed (the hold is journaled by the
        # competence gate) — and nothing is sent.
        self.assertTrue(
            _wait(lambda: self.runtime._brain_gate.episode_open),
            "the vision-degraded turn was never held")
        time.sleep(0.5)
        self.assertEqual(adapter.sent, [])

    def test_seen_photo_gets_a_normal_reply(self) -> None:
        self.context.tools = _tools_stub(
            vision_ok=True,
            description="an MT5 chart screenshot about NFP")
        _stub_bundle(self.runtime.brain,
                     ["na the nfp print be that"])
        adapter = self.adapters["telegram"]
        self.assertTrue(adapter.wait_started())
        photo = Path(self.tmp.name) / "nfp-shot.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        self.runtime.on_message(self._group_msg(
            "telegram", message_id="102",
            text="wren, check this photo",
            media=[MediaRef(path=str(photo), mime="image/jpeg",
                            kind="image")]))
        self.assertTrue(_wait(lambda: len(adapter.sent) == 1),
                        "healthy photo turn never answered")
        self.assertEqual(adapter.sent, ["na the nfp print be that"])
        self.assertNotIn(INCIDENT_TEXT, adapter.sent[0])

    def test_unseen_photo_on_bot_surface_still_sends(self) -> None:
        # The bot speaks in her OWN voice: the gate never held
        # bot surfaces, and the constrained media note (not
        # silence) is the honesty mechanism there.
        self.context.tools = _tools_stub(vision_ok=False)
        _stub_bundle(self.runtime.brain,
                     ["i can't view images right now o"])
        adapter = self.adapters["telegram-bot"]
        self.assertTrue(adapter.wait_started())
        photo = Path(self.tmp.name) / "nfp-shot.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        self.runtime.on_message(self._group_msg(
            "telegram-bot", message_id="103",
            text="wren, check this photo",
            media=[MediaRef(path=str(photo), mime="image/jpeg",
                            kind="image")]))
        self.assertTrue(_wait(lambda: len(adapter.sent) == 1),
                        "bot-surface photo turn never answered")
        self.assertEqual(adapter.sent,
                         ["i can't view images right now o"])


if __name__ == "__main__":
    unittest.main()
