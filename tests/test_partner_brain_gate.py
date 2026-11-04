"""Competence gate for personal-surface conversational replies.

The harm (owner's WhatsApp, 2026-11-04): with no real model
answering, the responder serves its static template bank
(``ReplyBundle.fallback``) and Devon kept free-chatting AS THE
OWNER in his real groups — quality nonsense that cost him three
group memberships. The scaffold guard (test_partner_reply_guard)
only judges a reply's shape; this gate judges its provenance.

These tests pin:

* the brain-quality fact made explicit at the outcome level
  (responder fallback bundle → ``PresenceOutcome.degraded``);
* the send seam (``PartnerRuntime._send_reply``, the same seam the
  scaffold guard sits on): degraded personal-surface replies are
  held — whatsapp + telegram userbot + telegram-personal, groups
  and DMs — while bot surfaces and healthy replies pass untouched;
* episode discipline: one journal entry when the degraded episode
  opens, one when it closes, automatic recovery, and the
  /platforms surface showing the observe-only state meanwhile;
* composition with the scaffold guard: a degraded scaffold is held
  by this gate alone (one journal story), a healthy scaffold is
  still the scaffold guard's business.
"""

from __future__ import annotations

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
from nomorals.agents.partner.brain_gate import (
    BrainGate,
    is_personal_platform,
)
from nomorals.agents.partner.failures import (
    clear_failures,
    recent_failures,
)
from nomorals.agents.partner_runtime import PartnerBrain, PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway

#: Context homes in this module each carry a fully migrated
#: database. They honour NM_TEST_TMPDIR so a test runner can keep
#: them off the OS tmpdir: where /tmp is a small shared tmpfs, a
#: full tmpdir turns storage COMMITs into SQLITE_FULL, the failed
#: COMMIT is only logged (Database._safe), the migration sequence
#: silently half-applies, and a later test in the same process then
#: dies inside migrate() with "no such table: goals" — a failure
#: that lands on whichever test runs while the tmpfs is tightest.
#: Unset, tempfile's default placement is used, unchanged.
_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

INCIDENT = (
    "analyze the user's input:\n"
    "\"u dey smoke\" - this is a nigerian pidgin/english phrase "
    "meaning \"you are smoking,\" often used playfully\n"
    "context: group chat with sucidial and ekpo. the vibe is "
    "neutral, pace is buzzing"
)

TEMPLATE_LINE = "lol my head is just full today, give me a minute"


def _braingate_entries() -> list:
    return [f for f in recent_failures(200) if "braingate" in f.site]


def _scaffold_entries() -> list:
    return [f for f in recent_failures(200) if "scaffold_guard" in f.site]


# ── the episode bookkeeper ──────────────────────────────────────────


class BrainGateTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()

    def test_first_hold_opens_episode_and_journals_once(self) -> None:
        gate = BrainGate()
        self.assertTrue(gate.note_hold(
            platform="whatsapp", chat_key="whatsapp:room1"))
        self.assertTrue(gate.episode_open)
        entries = _braingate_entries()
        self.assertEqual(len(entries), 1)
        self.assertIn("BrainGateHeld", entries[0].exc_type)
        self.assertIn("whatsapp", entries[0].message)
        # The held reply's text is never journaled.
        self.assertNotIn(TEMPLATE_LINE, entries[0].message)

    def test_repeated_holds_journal_nothing_more(self) -> None:
        gate = BrainGate()
        gate.note_hold(platform="whatsapp", chat_key="whatsapp:room1")
        self.assertFalse(gate.note_hold(
            platform="whatsapp", chat_key="whatsapp:room2"))
        self.assertFalse(gate.note_hold(
            platform="telegram", chat_key="telegram:room9"))
        self.assertEqual(gate.held_count, 3)
        self.assertEqual(len(_braingate_entries()), 1)

    def test_healthy_closes_episode_and_journals_recovery(self) -> None:
        gate = BrainGate()
        gate.note_hold(platform="whatsapp", chat_key="whatsapp:room1")
        gate.note_hold(platform="whatsapp", chat_key="whatsapp:room1")
        self.assertTrue(gate.note_healthy())
        self.assertFalse(gate.episode_open)
        entries = _braingate_entries()
        self.assertEqual(len(entries), 2)
        # recent_failures is newest-first: the recovery leads.
        self.assertIn("BrainGateRecovered", entries[0].exc_type)
        self.assertIn("2 held", entries[0].message)
        self.assertIn("BrainGateHeld", entries[-1].exc_type)
        # A second healthy send journals nothing.
        self.assertFalse(gate.note_healthy())
        self.assertEqual(len(_braingate_entries()), 2)

    def test_new_episode_after_recovery_journals_again(self) -> None:
        gate = BrainGate()
        gate.note_hold(platform="whatsapp", chat_key="whatsapp:room1")
        gate.note_healthy()
        self.assertTrue(gate.note_hold(
            platform="telegram", chat_key="telegram:room3"))
        held = [f for f in _braingate_entries()
                if "BrainGateHeld" in f.exc_type]
        self.assertEqual(len(held), 2)

    def test_personal_platform_taxonomy(self) -> None:
        for name in ("whatsapp", "telegram", "telegram-personal"):
            self.assertTrue(is_personal_platform(name), name)
        for name in ("telegram-bot", "discord", "local", "sms", ""):
            self.assertFalse(is_personal_platform(name), name)


# ── the brain-quality flag at the outcome level ─────────────────────


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


def _stub_bundle(brain: PartnerBrain, **kw: Any) -> None:
    bundle = SimpleNamespace(
        parts=kw.get("parts", ["mhm, tell me more"]),
        model=kw.get("model", "fake-7b"),
        degraded=False, degraded_note="")
    if "fallback" in kw:
        bundle.fallback = kw["fallback"]
    brain.responder.respond_bounded = mock.MagicMock(
        return_value=bundle)


class OutcomeFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-braingate-", dir=_TEST_TMPDIR)
        self._brains: list[PartnerBrain] = []

    def tearDown(self) -> None:
        # Close each brain's context BEFORE removing its home: the
        # context's background threads (supervisor/self-heal) keep
        # writing into the home while it lives, and rmtree loses
        # that race with "Directory not empty".
        for brain in self._brains:
            try:
                brain.context.close()
            except Exception:  # noqa: E103 - best-effort teardown
                pass
        self.tmp.cleanup()

    def _new_brain(self) -> PartnerBrain:
        brain = _make_brain(self.tmp.name)
        self._brains.append(brain)
        return brain

    def _message(self) -> ChatMessage:
        chat = ChatRef(platform="local", chat_id="peer1",
                       kind=ChatKind.DM, peer="peer1")
        return ChatMessage(chat=chat, incoming=True,
                           text="you good?", sender="ada")

    def test_fallback_bundle_marks_generation_degraded(self) -> None:
        brain = self._new_brain()
        _stub_bundle(brain, parts=[TEMPLATE_LINE], model="fallback",
                     fallback=True)
        parts, degraded, note = brain._generate_and_persist(
            self._message(), {"is_owner": False, "in_us": False})
        self.assertEqual(parts, [TEMPLATE_LINE])
        self.assertTrue(degraded)
        self.assertTrue(note)

    def test_model_bundle_is_not_degraded(self) -> None:
        brain = self._new_brain()
        _stub_bundle(brain, fallback=False)
        _parts, degraded, note = brain._generate_and_persist(
            self._message(), {"is_owner": False, "in_us": False})
        self.assertFalse(degraded)
        self.assertEqual(note, "")

    def test_deliver_reply_carries_the_flag(self) -> None:
        brain = self._new_brain()
        _stub_bundle(brain, parts=[TEMPLATE_LINE], model="fallback",
                     fallback=True)
        outcome = brain.deliver_reply(self._message())
        self.assertEqual(outcome.parts, [TEMPLATE_LINE])
        self.assertTrue(outcome.degraded)


# ── the send boundary ───────────────────────────────────────────────


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


class DownRouter:
    """Every provider is down: real prompts raise, housekeeping
    prompts (memory distillation) still get their JSON."""

    def chat(self, messages: list[Message],
             params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        text_blob = " ".join(m.content for m in messages)
        if "memory-distillation" in text_blob:
            return LLMResponse(
                text=json.dumps({
                    "summary": "Banter in a group chat.",
                    "facts": [], "milestone": "", "mood_after": "happy",
                }),
                model="fake-curator",
            )
        raise RuntimeError("provider down: connection refused")


def _make_context() -> tuple[Any, "tempfile.TemporaryDirectory"]:
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
    for _marker in ("pulse_last_run.json",
                    "pulse_last_run_afternoon.json",
                    "pulse_last_run_evening.json",
                    "pulse_last_run_midnight.json"):
        (_ws / _marker).write_text(
            json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    from nomorals.agents.morning_briefing import _today_str
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json,"
        " generated_at) VALUES (?, ?, '[]', ?)",
        ("test-seed", _today_str(context), time.time()),
    )
    return context, tmp


def _wait(predicate, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class SendBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        # Scripted BEFORE the runtime builds its brain around the
        # router (the responder captures it at construction): every
        # provider is down, so any real generation falls back.
        self.context.router = DownRouter()
        self.adapters = {
            name: FakeAdapter(name)
            for name in ("whatsapp", "telegram", "telegram-personal",
                         "telegram-bot")
        }
        self.gateway = ChatGateway(self.adapters, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.runtime.brain.presence_rng = random.Random(34)
        self.runtime._typing_enabled = lambda chat: False

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: E103 - best-effort teardown
            pass
        self.context.close()
        self.tmp.cleanup()

    def _message(self, platform: str, kind: ChatKind,
                 chat_id: str = "room1") -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                       title="Pinnocent 7.2.0", peer=chat_id)
        return ChatMessage(chat=chat, incoming=True, text="u dey smoke",
                           sender="ekpo")

    # — holds: every personal surface, groups and DMs —

    def test_whatsapp_group_degraded_held(self) -> None:
        msg = self._message("whatsapp", ChatKind.GROUP)
        self.runtime._send_reply(msg, [TEMPLATE_LINE], degraded=True,
                                 degraded_note="provider down")
        self.assertEqual(self.adapters["whatsapp"].sent, [])
        self.assertTrue(self.runtime._brain_gate.episode_open)
        self.assertEqual(len(_braingate_entries()), 1)

    def test_telegram_userbot_group_degraded_held(self) -> None:
        msg = self._message("telegram", ChatKind.GROUP)
        self.runtime._send_reply(msg, [TEMPLATE_LINE], degraded=True)
        self.assertEqual(self.adapters["telegram"].sent, [])

    def test_telegram_personal_dm_degraded_held(self) -> None:
        msg = self._message("telegram-personal", ChatKind.DM,
                            chat_id="owner")
        self.runtime._send_reply(msg, [TEMPLATE_LINE], degraded=True)
        self.assertEqual(self.adapters["telegram-personal"].sent, [])

    def test_whatsapp_dm_degraded_held(self) -> None:
        msg = self._message("whatsapp", ChatKind.DM, chat_id="peer1")
        self.runtime._send_reply(msg, [TEMPLATE_LINE], degraded=True)
        self.assertEqual(self.adapters["whatsapp"].sent, [])

    def test_episode_journals_once_across_many_holds(self) -> None:
        for platform in ("whatsapp", "telegram", "telegram-personal"):
            self.runtime._send_reply(
                self._message(platform, ChatKind.GROUP),
                [TEMPLATE_LINE], degraded=True)
        self.assertEqual(len(_braingate_entries()), 1)
        self.assertEqual(self.runtime._brain_gate.held_count, 3)

    # — passes: healthy personal, and degraded bot surface —

    def test_healthy_personal_reply_byte_identical(self) -> None:
        msg = self._message("whatsapp", ChatKind.GROUP)
        clean = "u dey smoke ke? 😂 na play i dey play o"
        self.runtime._send_reply(msg, [clean])
        self.assertEqual(self.adapters["whatsapp"].sent, [clean])
        self.assertEqual(_braingate_entries(), [])
        self.assertFalse(self.runtime._brain_gate.episode_open)

    def test_degraded_bot_surface_unaffected(self) -> None:
        msg = self._message("telegram-bot", ChatKind.GROUP)
        self.runtime._send_reply(msg, [TEMPLATE_LINE], degraded=True)
        self.assertEqual(self.adapters["telegram-bot"].sent,
                         [TEMPLATE_LINE])
        self.assertFalse(self.runtime._brain_gate.episode_open)
        self.assertEqual(_braingate_entries(), [])

    # — recovery —

    def test_recovery_resumes_and_closes_episode(self) -> None:
        msg = self._message("whatsapp", ChatKind.GROUP)
        self.runtime._send_reply(msg, [TEMPLATE_LINE], degraded=True)
        self.assertEqual(self.adapters["whatsapp"].sent, [])
        self.runtime._send_reply(msg, ["i dey here o, wetin happen"])
        self.assertEqual(self.adapters["whatsapp"].sent,
                         ["i dey here o, wetin happen"])
        self.assertFalse(self.runtime._brain_gate.episode_open)
        kinds = [f.exc_type for f in _braingate_entries()]
        self.assertEqual(sorted(kinds),
                         ["BrainGateHeld", "BrainGateRecovered"])

    # — composition with the scaffold guard —

    def test_degraded_scaffold_held_by_braingate_alone(self) -> None:
        msg = self._message("whatsapp", ChatKind.GROUP)
        self.runtime._send_reply(msg, [INCIDENT], degraded=True)
        self.assertEqual(self.adapters["whatsapp"].sent, [])
        self.assertEqual(len(_braingate_entries()), 1)
        self.assertEqual(_scaffold_entries(), [])

    def test_healthy_scaffold_still_scaffold_guards_business(self) -> None:
        msg = self._message("whatsapp", ChatKind.GROUP)
        self.runtime._send_reply(msg, [INCIDENT])
        sent = self.adapters["whatsapp"].sent
        self.assertEqual(len(sent), 1)
        self.assertNotIn("analyze", sent[0])
        self.assertTrue(_scaffold_entries())
        self.assertEqual(_braingate_entries(), [])

    # — the full turn under a dead provider —

    def test_full_turn_with_dead_provider_sends_nothing(self) -> None:
        self.runtime.start()
        adapter = self.adapters["whatsapp"]
        self.assertTrue(adapter.wait_started())
        group = ChatRef(platform="whatsapp", chat_id="grp-wa",
                        kind=ChatKind.GROUP, title="Pinnocent 7.2.0",
                        peer="grp-wa")
        self.runtime.on_message(ChatMessage(
            chat=group, incoming=True,
            text="wren, wetin you think — u dey smoke?",
            sender="ekpo"))
        self.assertTrue(
            _wait(lambda: len(_braingate_entries()) >= 1),
            "the degraded turn was never held")
        self.assertEqual(adapter.sent, [])
        # The inbound turn itself was still observed + persisted —
        # only the reply is held.
        rows = self.context.db.query(
            "SELECT role FROM messages WHERE conversation_id = ?",
            ("whatsapp:grp-wa",))
        self.assertIn("user", [r["role"] for r in rows])


# ── /platforms shows the observe-only state ─────────────────────────


class PlatformsNoteTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.adapters = {
            name: FakeAdapter(name)
            for name in ("whatsapp", "telegram-bot")
        }
        self.rt = PartnerRuntime.__new__(PartnerRuntime)
        self.rt.gateway = ChatGateway(self.adapters, db=None)
        self.rt._brain_gate = BrainGate()

    def _platforms(self) -> str:
        return self.rt.handle_control("/platforms", "local:owner")

    def test_no_note_while_brain_healthy(self) -> None:
        reply = self._platforms()
        self.assertIn("whatsapp:", reply)
        self.assertNotIn("observe-only", reply)

    def test_note_on_personal_line_only_while_episode_open(self) -> None:
        self.rt._brain_gate.note_hold(platform="whatsapp",
                                      chat_key="whatsapp:room1")
        reply = self._platforms()
        wa_line = next(line for line in reply.splitlines()
                       if "whatsapp:" in line)
        bot_line = next(line for line in reply.splitlines()
                        if "telegram-bot:" in line)
        self.assertIn("observe-only (brain unavailable)", wa_line)
        self.assertNotIn("observe-only", bot_line)
        # Control replies keep working during the episode — this
        # very call is one — and recovery clears the note.
        self.rt._brain_gate.note_healthy()
        self.assertNotIn("observe-only", self._platforms())


if __name__ == "__main__":
    unittest.main()
