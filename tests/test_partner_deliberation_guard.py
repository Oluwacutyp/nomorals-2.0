"""Deliberation hold on personal surfaces (2026-11-06).

The incident (owner's WhatsApp, 2026-11-05 ~23:32): mid-conversation
with a real contact ("good morning" → "how was your night" → "fine"
→ "ok"), the owner's account sent the model's raw deliberation
about the conversation — "the user seems to be continuing the
conversation, but the message is just … let me re-read… my previous
response was…" — and the contact quoted it back, confused.

Why it shipped, pinned by these tests:

* provider extraction took the content channel verbatim and the
  responder's post-processing is formatting-only, so reasoning
  inlined into a completion became the draft (the source half is
  pinned in test_llm_reasoning_split + the intake tests below);
* the outcome was NOT degraded — a real model rung answered — so
  the competence gate (brain_gate) passed it;
* the send-boundary scaffold guard keys on scaffold FURNITURE
  (analyze-headers, context:/vibe lines) that first-person
  deliberation doesn't carry, so it passed untouched.

The fix at this seam is the third check of the one personal-surface
gate: deliberation-shaped conversational text is HELD — journaled,
counted, never sent, and a fully-held turn resolves to silence (no
substitute reaction). Bot surfaces are untouched by it. The
detector pins the CLASS (paraphrased variants), never the incident
string, and the owner's own conversational shapes — one-word
replies, fragments, Pidgin, slang, emoji — are pinned clean.
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
from nomorals.agents.partner_runtime import PartnerBrain, PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import default_persona
from nomorals.partner.relationship import Relationship
from nomorals.partner.responder import PartnerResponder
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway

#: See test_partner_brain_gate for why context homes honour
#: NM_TEST_TMPDIR (the /tmp tmpfs + swallowed-COMMIT trap).
_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

# Paraphrased variants of the incident — the class, not the string.
DELIBERATION_V1 = (
    "the user seems to be keeping the conversation going, but the "
    "message is just \"did you eat\" followed by nothing else. "
    "wait, looking again, the user sent \"did you eat\" and then i "
    "answered small. let me re-read… actually, i see the user sent "
    "it right after my last one. looking at the messages: "
    "user: \"good afternoon\" me: \"afternoon\" (wait, that was…"
)
DELIBERATION_V2_STREAM_ONLY = (
    "hmm, looking again at this exchange, there's a new turn where "
    "the reply is just one word followed by nothing much, actually "
    "i see the pattern now — i responded too fast last time, and "
    "the conversation so far is mostly greetings, so how to "
    "respond without sounding dry is the thing"
)
DELIBERATION_V3_DIRECTIVE = (
    "let me review the conversation again before i answer this one"
)

#: The owner's real conversational shapes — all must pass untouched.
OWNER_CLEAN_SHAPES = (
    "k",
    "just chilling",
    "wetin u dey do",
    "fine",
    "ok",
    "😅",
    "lol 😂",
    "morning",
    "na so e be",
    "i dey come",
    "no wahala",
    "abeg no vex, i go reply you for night",
    # One deliberation-adjacent phrase in ordinary speech is a
    # story, not a stream:
    "i responded to the landlord yesterday, e don pay",
    "looking again at the price, e don go up o",
    # Reporting about a third party is not narration of THIS chat:
    "the user asked for a refund so i sent it back",
    # A longer story carrying two medium markers (below the stream
    # threshold) still passes:
    "yesterday i responded to two of my guys wey dey ask about "
    "the match, and looking again at my schedule i realise say i "
    "don promise mama call since morning, na wa o",
)


def _reason(text: str) -> str:
    # Lazy: on the pristine tree the detector does not exist yet,
    # and the runtime-seam teeth below must fail on their assertions
    # (the deliberation was SENT), not on this import.
    from nomorals.agents.partner.reply_guard import deliberation_reason

    return deliberation_reason(text)


def _guard_parts(parts, **kw):
    from nomorals.agents.partner.reply_guard import (
        guard_deliberation_parts,
    )

    return guard_deliberation_parts(parts, **kw)


class DeliberationReasonTests(unittest.TestCase):
    def test_incident_class_variants_are_deliberation(self) -> None:
        for text in (DELIBERATION_V1, DELIBERATION_V2_STREAM_ONLY,
                     DELIBERATION_V3_DIRECTIVE):
            self.assertTrue(_reason(text),
                            f"not held: {text[:60]}…")

    def test_owner_shapes_are_clean(self) -> None:
        for text in OWNER_CLEAN_SHAPES:
            self.assertEqual(_reason(text), "",
                             f"false positive: {text!r}")

    def test_blank_is_clean(self) -> None:
        self.assertEqual(_reason(""), "")
        self.assertEqual(_reason("   "), "")


class GuardPartsTests(unittest.TestCase):
    def test_fully_deliberative_reply_is_held_whole(self) -> None:
        kept, held = _guard_parts(
            [DELIBERATION_V1], platform="whatsapp", chat_key="wa:1")
        self.assertEqual(kept, [])
        self.assertEqual(held, 1)

    def test_split_deliberation_is_held_by_the_joined_judgment(
            self) -> None:
        # One draft chunked into two parts: neither chunk alone
        # carries a convicting stream (two markers, under the
        # length floor each), the joined body does.
        chunk1 = ("looking again at the exchange, i responded too "
                  "quickly before, that one na my side sha")
        chunk2 = ("actually i see the pattern, the conversation so "
                  "far is just greetings, nothing deep yet at all o")
        self.assertEqual(_reason(chunk1), "")
        self.assertEqual(_reason(chunk2), "")
        kept, held = _guard_parts(
            [chunk1, chunk2], platform="telegram", chat_key="tg:1")
        self.assertEqual(kept, [])
        self.assertEqual(held, 2)

    def test_deliberation_anywhere_holds_the_whole_draft(self) -> None:
        # Parts are chunks of ONE composed draft: a deliberation
        # segment convicts the whole body — the clean-looking front
        # of a deliberating draft is not a safe salvage.
        parts = ["just chilling", DELIBERATION_V3_DIRECTIVE, "k"]
        kept, held = _guard_parts(
            parts, platform="whatsapp", chat_key="wa:1")
        self.assertEqual(kept, [])
        self.assertEqual(held, 3)

    def test_clean_multi_part_reply_passes_untouched(self) -> None:
        parts = ["just chilling", "k"]
        kept, held = _guard_parts(
            parts, platform="whatsapp", chat_key="wa:1")
        self.assertEqual(kept, ["just chilling", "k"])
        self.assertEqual(held, 0)

    def test_blank_parts_are_not_the_guards_concern(self) -> None:
        kept, held = _guard_parts(
            ["", "k"], platform="whatsapp", chat_key="wa:1")
        self.assertEqual(kept, ["", "k"])
        self.assertEqual(held, 0)


# ── responder intake: the draft is the answer channel only ────────


class _ScriptedRouter:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def chat(self, messages, params=None, **kw):
        self.calls += 1
        text = self._texts.pop(0) if self._texts else ""
        return LLMResponse(text=text, model="fake-reasoner")


def _responder(router: Any) -> PartnerResponder:
    persona = default_persona()
    return PartnerResponder(
        router,
        persona,
        MoodEngine(persona.baselines),
        Relationship(id="default", stage="in_love", trust=90),
        None,
        None,
    )


class ResponderIntakeTests(unittest.TestCase):
    def test_think_wrapped_answer_yields_only_the_answer(self) -> None:
        responder = _responder(_ScriptedRouter([
            "<think>the user said good morning; keep it short and "
            "warm</think>morning"]))
        bundle = responder.respond(
            chat_platform="whatsapp", user_text="good morning",
            gate_mode="restricted")
        self.assertFalse(bundle.fallback)
        self.assertNotIn("think", bundle.text)
        self.assertNotIn("keep it short", bundle.text)
        self.assertIn("morning", bundle.text)

    def test_reasoning_only_generation_is_no_draft(self) -> None:
        # Every attempt deliberates and never answers: the bundle is
        # the honest no-model fallback (which personal surfaces
        # hold) — never the deliberation dressed as a reply.
        responder = _responder(_ScriptedRouter([
            "<think>the user seems to be continuing the conversation",
            "<think>still working through the turn exchange",
            "<think>and again, no answer channel at all",
        ]))
        bundle = responder.respond(
            chat_platform="whatsapp", user_text="ok",
            gate_mode="restricted")
        self.assertTrue(bundle.fallback)
        self.assertNotIn("user seems", bundle.text)
        self.assertNotIn("turn exchange", bundle.text)


# ── the runtime seam, end to end ──────────────────────────────────


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
    tmp = tempfile.TemporaryDirectory(prefix="nm-leak-rt-",
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


def _stub_bundle(brain: PartnerBrain, parts: list[str]) -> None:
    bundle = SimpleNamespace(parts=parts, model="fake-reasoner",
                             degraded=False, degraded_note="")
    brain.responder.respond_bounded = mock.MagicMock(
        return_value=bundle)


class RuntimeDeliberationSeamTests(unittest.TestCase):
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
                   text: str = "wren, how was your night") -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id="grp-1",
                       kind=ChatKind.GROUP, title="Traders",
                       peer="grp-1")
        return ChatMessage(chat=chat, incoming=True, text=text,
                           sender="faheem", message_id=message_id,
                           ts=time.time())

    def _dm_msg(self, platform: str) -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id="peer-1",
                       kind=ChatKind.DM, peer="peer-1")
        return ChatMessage(chat=chat, incoming=True, text="ok",
                           sender="peer", message_id="dm-1",
                           ts=time.time())

    def test_deliberation_held_on_whatsapp(self) -> None:
        _stub_bundle(self.runtime.brain, [DELIBERATION_V1])
        adapter = self.adapters["whatsapp"]
        self.assertTrue(adapter.wait_started())
        self.runtime.on_message(
            self._group_msg("whatsapp", message_id="200"))
        self.assertTrue(
            _wait(lambda: self.runtime._stats_snapshot().get(
                "deliberation_held", 0) >= 1),
            "deliberation was never held on WhatsApp")
        time.sleep(0.5)
        self.assertEqual(adapter.sent, [])

    def test_deliberation_held_on_telegram_userbot(self) -> None:
        _stub_bundle(self.runtime.brain, [DELIBERATION_V2_STREAM_ONLY])
        adapter = self.adapters["telegram"]
        self.assertTrue(adapter.wait_started())
        self.runtime.on_message(
            self._group_msg("telegram", message_id="201"))
        self.assertTrue(
            _wait(lambda: self.runtime._stats_snapshot().get(
                "deliberation_held", 0) >= 1),
            "deliberation was never held on Telegram")
        time.sleep(0.5)
        self.assertEqual(adapter.sent, [])

    def test_deliberation_held_at_the_dm_send_seam(self) -> None:
        # The incident's exact shape: a DM on the owner's WhatsApp.
        adapter = self.adapters["whatsapp"]
        self.runtime._send_reply(
            self._dm_msg("whatsapp"), [DELIBERATION_V1])
        self.assertEqual(adapter.sent, [])
        self.assertGreaterEqual(
            self.runtime._stats_snapshot().get(
                "deliberation_held", 0), 1)

    def test_clean_reply_still_sends_on_personal_surface(self) -> None:
        _stub_bundle(self.runtime.brain, ["just chilling"])
        adapter = self.adapters["whatsapp"]
        self.assertTrue(adapter.wait_started())
        self.runtime.on_message(
            self._group_msg("whatsapp", message_id="202",
                            text="wren, wetin u dey do"))
        self.assertTrue(_wait(lambda: len(adapter.sent) == 1),
                        "clean reply never sent")
        self.assertEqual(adapter.sent, ["just chilling"])

    def test_deliberation_on_bot_surface_is_not_held(self) -> None:
        # The bot speaks in her OWN voice: the deliberation hold is
        # a personal-surface law, and the scaffold guard (which this
        # text passes — no scaffold furniture) is the only boundary
        # there.
        _stub_bundle(self.runtime.brain, [DELIBERATION_V1])
        adapter = self.adapters["telegram-bot"]
        self.assertTrue(adapter.wait_started())
        self.runtime.on_message(
            self._group_msg("telegram-bot", message_id="203"))
        self.assertTrue(_wait(lambda: len(adapter.sent) == 1),
                        "bot-surface reply never sent")
        self.assertEqual(adapter.sent, [DELIBERATION_V1])
        self.assertEqual(
            self.runtime._stats_snapshot().get(
                "deliberation_held", 0), 0)

    def test_mind_reply_deliberation_held_clean_sent(self) -> None:
        adapter = self.adapters["whatsapp"]
        message = self._dm_msg("whatsapp")
        self.runtime._send_mind_reply(message, DELIBERATION_V2_STREAM_ONLY)
        self.assertEqual(adapter.sent, [])
        self.runtime._send_mind_reply(
            message, "i don find am — the file dey ready")
        self.assertEqual(adapter.sent,
                         ["i don find am — the file dey ready"])


if __name__ == "__main__":
    unittest.main()
