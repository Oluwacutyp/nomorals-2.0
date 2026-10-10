"""Tests for human presence: length-based typing pace and busy/distracted gaps.

The point of the module is ban-avoidance realism: a bot's tells are a
constant typing duration and a reply that lands at the same moment every
time. These tests pin the human shape of both — typing scales with message
length and mood, and the presence decision is a real (seeded) probability,
with the safety rule that substantive messages are delayed, never ignored.
"""

from __future__ import annotations

import random
import tempfile
import time
import unittest
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerBrain, PresenceOutcome, PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import default_persona
from nomorals.partner.presence import (
    Presence,
    decide_presence,
    human_typing_seconds,
    is_low_content,
)
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway


def _mood(label: str, **overrides: float) -> Any:
    """A mood-stand-in with the attribute shape decide_presence reads."""
    persona = default_persona()
    values = dict(persona.baselines)
    values.update(overrides)
    return _MoodState(label, values)


class _MoodState:
    def __init__(self, label: str, values: dict[str, float]) -> None:
        self.label = label
        self.values = values


# ── low-content detection ────────────────────────────────────────────────────


class LowContentTest(unittest.TestCase):
    def test_acknowledgments_are_low_content(self) -> None:
        for text in ("k", "ok", "okay", "lol", "lmao", "haha", "sure", "fine",
                     "yeah", "yep", "nope", "cool", "thx", "thanks", "ty", "👍"):
            self.assertTrue(is_low_content(text), text)
        self.assertTrue(is_low_content("OK!!"))
        self.assertTrue(is_low_content("lol."))

    def test_substantive_text_is_not_low_content(self) -> None:
        for text in ("can you come over tonight?", "i need to tell you something",
                     "what are we doing friday", "no way", "k k k",
                     "okay but hear me out", "a"):
            self.assertFalse(is_low_content(text), text)


# ── typing pace ──────────────────────────────────────────────────────────────


class TypingPaceTest(unittest.TestCase):
    def _avg(self, text: str, mood: dict, seed: int = 7, n: int = 30) -> float:
        rng = random.Random(seed)
        return sum(
            human_typing_seconds(text, mood=mood, rng=random.Random(seed + i))
            for i in range(n)
        ) / n

    def test_scales_with_length(self) -> None:
        short = self._avg("k", {"energy": 60, "happiness": 60})
        long = self._avg("x" * 400, {"energy": 60, "happiness": 60})
        self.assertGreater(long, 3 * short)
        self.assertLess(short, 6)

    def test_stays_in_bounds(self) -> None:
        rng = random.Random(1)
        for i in range(50):
            d = human_typing_seconds("hello there", mood={}, rng=rng)
            self.assertGreaterEqual(d, 2.0)
            self.assertLessEqual(d, 60.0)
        d = human_typing_seconds("x" * 5000, mood={}, rng=rng)
        self.assertLessEqual(d, 60.0)

    def test_tired_types_slower_than_excited(self) -> None:
        tired = self._avg("hey are you free tonight?", {"energy": 15, "happiness": 40})
        excited = self._avg("hey are you free tonight?", {"energy": 92, "happiness": 88})
        self.assertGreater(tired, excited)

    def test_minimum_floor_applies(self) -> None:
        rng = random.Random(3)
        d = human_typing_seconds("k", mood={}, rng=rng, minimum=5.0)
        self.assertGreaterEqual(d, 5.0)


# ── the presence decision ────────────────────────────────────────────────────


class DecidePresenceTest(unittest.TestCase):
    def test_substantive_messages_are_never_ignored(self) -> None:
        text = "can you come over tonight? i really need you"
        for seed in range(300):
            for label in ("tired", "annoyed", "cold", "distant", "calm", "happy", "angry"):
                mood = _mood(label)
                p = decide_presence(mood, text, rng=random.Random(seed))
                self.assertTrue(p.reply, f"seed {seed} label {label} ignored a question")

    def test_closed_off_can_be_delayed_and_within_bounds(self) -> None:
        mood = _mood("tired")
        p = decide_presence(mood, "are we still on for tomorrow?",
                           rng=random.Random(31))  # rolls 0.0123 < 0.15
        self.assertTrue(p.reply)
        self.assertGreater(p.delay_seconds, 0)
        self.assertLessEqual(p.delay_seconds, 1800.0)
        self.assertGreaterEqual(p.delay_seconds, 60.0)

    def test_low_content_can_be_left_on_read(self) -> None:
        mood = _mood("tired")
        p = decide_presence(mood, "k", rng=random.Random(31))  # rolls 0.0123 < 0.25
        self.assertFalse(p.reply)
        self.assertIn("low stakes", p.reason)

    def test_low_content_ignored_less_when_eager(self) -> None:
        texts = ["k", "lol", "ok"]
        closed = _mood("tired")
        eager = _mood("happy")
        ignored_closed = ignored_eager = 0
        for seed in range(400):
            for text in texts:
                if not decide_presence(closed, text, rng=random.Random(seed)).reply:
                    ignored_closed += 1
                if not decide_presence(eager, text, rng=random.Random(seed)).reply:
                    ignored_eager += 1
        self.assertGreater(ignored_closed, ignored_eager)
        self.assertGreater(ignored_eager, 0)  # eager still occasionally drifts

    def test_owner_gets_half_the_busy_chance(self) -> None:
        mood = _mood("calm")
        p_other = decide_presence(mood, "are you there?", is_owner=False,
                                  rng=random.Random(43))  # rolls 0.0386
        p_owner = decide_presence(mood, "are you there?", is_owner=True,
                                  rng=random.Random(43))
        # 0.0386 < calm busy (0.07) -> the other person gets the gap;
        # 0.0386 > owner busy (0.035) -> the owner gets an immediate reply.
        self.assertGreater(p_other.delay_seconds, 0)
        self.assertEqual(p_owner.delay_seconds, 0)

    def test_present_is_the_default(self) -> None:
        p = decide_presence(_mood("calm"), "hello", rng=random.Random(34))
        self.assertTrue(p.reply)
        self.assertEqual(p.delay_seconds, 0)
        self.assertEqual(p.reason, "present")


# ── brain integration ────────────────────────────────────────────────────────


class _FakeRouter:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[Message], params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text="mhm. yeah, i'm around", model="fake")


def _make_brain(tmp: str) -> tuple[PartnerBrain, _FakeRouter]:
    settings = load_settings(
        overrides={"home": tmp, "partner.platforms": "local", "chat.local_enabled": "true"}
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    router = _FakeRouter()
    context.router = router
    brain = PartnerBrain(context)
    brain.presence_rng = random.Random(34)
    return brain, router


class BrainPresenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-presence-")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _message(self, text: str) -> ChatMessage:
        chat = ChatRef(platform="local", chat_id="owner", kind=ChatKind.DM, peer="you")
        return ChatMessage(chat=chat, incoming=True, text=text, sender="you")

    def test_immediate_reply_returns_parts_and_no_delay(self) -> None:
        brain, router = _make_brain(self.tmp.name)
        outcome = brain.handle_message(self._message("hey, are you around?"))
        self.assertIsInstance(outcome, PresenceOutcome)
        self.assertTrue(outcome.parts)
        self.assertEqual(outcome.presence.delay_seconds, 0)
        self.assertEqual(router.calls, 1)
        # inbound message persisted alongside the reply
        rows = brain.context.db.query(
            "SELECT role FROM messages WHERE conversation_id = ?", ("local:owner",)
        )
        self.assertIn("user", [r["role"] for r in rows])
        self.assertIn("assistant", [r["role"] for r in rows])

    def test_busy_decision_defers_generation(self) -> None:
        brain, router = _make_brain(self.tmp.name)
        with mock.patch(
            "nomorals.agents.partner.brain.decide_presence",
            return_value=Presence(reply=True, delay_seconds=420.0, reason="busy / distracted"),
        ):
            outcome = brain.handle_message(self._message("are we still on for dinner?"))
        self.assertEqual(outcome.parts, [])
        self.assertEqual(outcome.presence.delay_seconds, 420.0)
        self.assertEqual(router.calls, 0, "a busy decision must not spend a model call now")
        # but she heard it: the inbound row exists for the later reply's history
        rows = brain.context.db.query(
            "SELECT role FROM messages WHERE conversation_id = ?", ("local:owner",)
        )
        self.assertIn("user", [r["role"] for r in rows])

    def test_read_and_left_sends_nothing_and_calls_no_model(self) -> None:
        brain, router = _make_brain(self.tmp.name)
        with mock.patch(
            "nomorals.agents.partner.brain.decide_presence",
            return_value=Presence(reply=False, reason="read, left — low stakes"),
        ):
            outcome = brain.handle_message(self._message("k"))
        self.assertEqual(outcome.parts, [])
        self.assertFalse(outcome.presence.reply)
        self.assertEqual(router.calls, 0)

    def test_deliver_reply_generates_without_re_ticking_signals(self) -> None:
        brain, router = _make_brain(self.tmp.name)
        # Pre-seed the mood so a fight signal is pending in the text:
        # deliver_reply must NOT re-detect it (no second escalation).
        with mock.patch(
            "nomorals.agents.partner.brain.decide_presence",
            return_value=Presence(reply=True, delay_seconds=1.0, reason="busy / distracted"),
        ):
            brain.handle_message(self._message("i'm so angry, you never listen"))
        self.assertIsNotNone(brain.mood.open_fight)
        frustration_after_first = brain.mood.value("frustration")
        # The delayed half: generate + persist only.
        parts = brain.deliver_reply(self._message("i'm so angry, you never listen"))
        self.assertTrue(parts)
        self.assertEqual(router.calls, 1, "exactly one generation for the whole turn")
        self.assertLessEqual(brain.mood.value("frustration"), frustration_after_first + 1e-9,
                             "deliver_reply re-detected the fight signal")


# ── runtime e2e: the delayed reply actually lands ───────────────────────────


class _RecordingAdapter(ChatAdapter):
    def __init__(self) -> None:
        super().__init__(media_dir="/tmp/nm-presence-media")
        self.name = "local"
        self.sent: list[str] = []
        self.typing_durations: list[float] = []
        self.started_flag = False

    def run(self, handler: Any) -> None:
        self.started_flag = True
        while not self.stopped:
            time.sleep(0.01)

    def typing(self, chat: ChatRef, seconds: float = 3.0, action: str = "typing") -> bool:
        # Record instead of blocking: the e2e test is about pacing shape,
        # not about actually waiting out the typing indicator.
        self.typing_durations.append(seconds)
        return True

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, parse_mode: str = "") -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name, message_id=f"m{len(self.sent)}")

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


class RuntimePresenceE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-presence-e2e-")
        settings = load_settings(
            overrides={
                "home": self.tmp.name,
                "partner.platforms": "local",
                "partner.typing_seconds": "2.0",
                "partner.part_delay_seconds": "0.2",
                "chat.local_enabled": "true",
            }
        )
        self.context = build_context(settings, with_executor=False, with_tools=False)
        # The responder holds this router for its whole life: it must be the
        # final router before PartnerRuntime builds the brain.
        self.router = _LongRouter()
        self.context.router = self.router
        self.adapter = _RecordingAdapter()
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.chat = ChatRef(platform="local", chat_id="owner", kind=ChatKind.DM, peer="you")
        from nomorals.agents.morning_briefing import _today_str
        self.context.db.execute(
            "INSERT OR REPLACE INTO briefings (id, date, sections_json, generated_at)"
            " VALUES (?, ?, '[]', ?)",
            ("test-seed", _today_str(self.context), time.time()),
        )

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_busy_reply_lands_later_and_typing_tracks_length(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())

        with mock.patch(
            "nomorals.agents.partner.brain.decide_presence",
            return_value=Presence(reply=True, delay_seconds=0.4, reason="busy / distracted"),
        ):
            self.runtime.on_message(ChatMessage(chat=self.chat, incoming=True,
                                                text="hey, what do you think of the new place?", sender="you"))
            time.sleep(0.15)
            self.assertEqual(self.adapter.sent, [], "nothing lands while she's busy")
            self.assertTrue(
                _wait(lambda: len(self.adapter.sent) >= 1, timeout=10),
                "delayed reply never landed",
            )

        # Typing now has two layers: a lead-in keep-alive while she "reads
        # and thinks" (may or may not land one tick before the delay ends),
        # then the per-message typing when the reply actually goes out. The
        # length-scaled one is always the LAST call.
        self.assertGreaterEqual(len(self.adapter.typing_durations), 1)
        duration = self.adapter.typing_durations[-1]
        self.assertGreater(duration, 2.0, "typing must exceed the minimum for a long message")
        self.assertLessEqual(duration, 60.0)
        self.assertEqual(self.context.router.calls, 1, "one generation for the whole turn")

    def test_immediate_reply_typing_is_human_paced(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.brain.presence_rng = random.Random(34)
        self.runtime.on_message(ChatMessage(chat=self.chat, incoming=True,
                                            text="hey, how's your day going", sender="you"))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1), "no reply")
        # The per-message typing (last call) is the length-scaled one; a
        # thinking lead-in tick may precede it.
        self.assertGreaterEqual(len(self.adapter.typing_durations), 1)
        self.assertGreaterEqual(self.adapter.typing_durations[-1], 2.0)


class _LongRouter(_FakeRouter):
    LONG = (
        "okay so honestly it sounds great and i keep picturing us there, "
        "like on a sunday morning with the windows open and nobody having "
        "to be anywhere, and yeah i think we should go see it this weekend "
        "before someone else beats us to it"
    )

    def chat(self, messages: list[Message], params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.LONG, model="fake")


if __name__ == "__main__":
    unittest.main()
