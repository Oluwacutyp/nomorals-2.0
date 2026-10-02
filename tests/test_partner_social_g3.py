"""Wave G3 partner/social: reliability under load.

Four workstreams, all about the partner + social layer holding up when the
conversation gets long and the chat is not the owner's:

* **Anti-echo under long threads** — the echo guard (``parrot_check`` /
  ``repair_echo``) used to see only the *latest* user message, so a model
  that lifts a phrase from turn 3 of a 60-turn thread sailed through.  Both
  functions now take ``history_texts``; the responder feeds them every user
  turn it was given, making the guard history-length-independent.
* **Gating under long threads** — ``is_owner_chat`` / ``gate_decision`` /
  ``classify_chat`` hold for every turn of a 60-turn thread: a stranger chat
  never drifts into owner treatment.
* **No AI-mode leakage** — a stranger/non-owner chat never gets agent-mode
  behavior: NL intent routing (CoreMind), the fast path, slash-command
  dispatch and ``/upgrade`` are all denied.  The enforcement points are the
  runtime's ``_is_operator`` (slash dispatch), CoreMind's ``_is_owner_dm``
  (NL routing), and ``_control_upgrade``'s own re-check (defense in depth).
* **Typing length-scaling** — the F2 length-scaled typing indicator contract
  (``human_typing_seconds``) still holds: longer parts type longer, mood
  modulates pace, and the runtime issues one scaled indicator per part.
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.agents.coremind import CoreMind
from nomorals.llm.base import LLMResponse, Message
from nomorals.partner import (
    PartnerResponder,
    default_persona,
    parrot_check,
    repair_echo,
)
from nomorals.partner.gating import (
    MODE_GROUP,
    MODE_OWNER,
    MODE_PRIVATE,
    classify_chat,
    gate_decision,
    is_owner_chat,
    is_restricted,
)
from nomorals.partner.mood import MoodEngine
from nomorals.partner.presence import human_typing_seconds
from nomorals.partner.relationship import Relationship
from nomorals.social.chat.base import (
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)

OWNER_CHATS = {"telegram:1"}

# An early user phrase, buried at turn 5 of a long thread.  Six content
# words so it trips the verbatim-4-gram check wherever it surfaces.
EARLY_PHRASE = "the northern lights over the frozen lake"
LATEST_USER_TEXT = "so anyway what time is it"


def _long_history(turns: int = 60, early_turn: int = 5) -> list[Message]:
    """A 60-turn conversation with the incriminating phrase at ``early_turn``."""
    msgs: list[Message] = []
    for i in range(turns):
        if i == early_turn:
            user = f"i saw {EARLY_PHRASE} last winter and i cannot stop thinking about it"
        else:
            user = f"turn {i}: just chatting about the day, nothing special"
        msgs.append(Message.user(user))
        msgs.append(Message.assistant(f"turn {i}: oh nice, tell me more"))
    return msgs


def _history_texts(msgs: list[Message]) -> list[str]:
    return [m.content for m in msgs if m.role == "user"]


class FakeRouter:
    """Scripted stand-in LLM router (mirrors the F2 test double)."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[list] = []

    def chat(self, messages, params=None, **kw):
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else "mhm, okay"
        return LLMResponse(text=text, model="fake")


def _responder(router: FakeRouter, *, retries: int = 0) -> PartnerResponder:
    persona = default_persona()
    mood = MoodEngine(persona.baselines, now=1_000_000.0)
    return PartnerResponder(
        router, persona, mood, Relationship(), None, None,
        rng=random.Random(0), lexicon=None, retries=retries,
    )


# ── 1. anti-echo under long threads ──────────────────────────────────────────

class EchoGuardLongHistoryTest(unittest.TestCase):
    def test_baseline_latest_turn_only_misses_early_phrase(self) -> None:
        # The pre-G3 guard: only the latest user message is checked, so an
        # echo of a turn-5 phrase in a 60-turn thread passes clean.  This is
        # the degradation the history-aware guard closes; it pins the old
        # signature's behaviour (no history_texts -> latest-turn only).
        history = _long_history()
        draft = f"oh that's beautiful, {EARLY_PHRASE} must have been amazing"
        verdict = parrot_check(LATEST_USER_TEXT, draft)
        self.assertTrue(
            verdict.ok,
            "baseline changed: the latest-turn-only check now flags this, "
            "which would make the history tests below vacuous",
        )
        self.assertEqual(_history_texts(history)[5].count(EARLY_PHRASE), 1)

    def test_history_aware_check_catches_early_phrase(self) -> None:
        history = _long_history()
        texts = _history_texts(history)
        draft = f"oh that's beautiful, {EARLY_PHRASE} must have been amazing"
        verdict = parrot_check(LATEST_USER_TEXT, draft, history_texts=texts)
        self.assertFalse(verdict.ok)
        self.assertIn("earlier user turn", verdict.reason)

    def test_history_aware_check_catches_early_phrase_case_insensitive(self) -> None:
        history = _long_history()
        texts = _history_texts(history)
        draft = "oh that's beautiful, The Northern Lights Over The Frozen Lake!!"
        verdict = parrot_check(LATEST_USER_TEXT, draft, history_texts=texts)
        self.assertFalse(verdict.ok)

    def test_repair_echo_strips_early_phrase(self) -> None:
        history = _long_history()
        texts = _history_texts(history)
        draft = f"oh that's beautiful, {EARLY_PHRASE} last winter must have been amazing"
        repaired = repair_echo(LATEST_USER_TEXT, draft, history_texts=texts)
        self.assertTrue(repaired)
        self.assertNotIn("northern lights", repaired.lower())
        # ...and the repaired draft passes the same history-aware check.
        self.assertTrue(
            parrot_check(LATEST_USER_TEXT, repaired, history_texts=texts).ok
        )

    def test_repair_echo_verbatim_early_phrase_drops(self) -> None:
        # A draft that IS the early user sentence has nothing left after the
        # strip -> "" so the caller serves the fallback line, never the echo.
        history = _long_history()
        texts = _history_texts(history)
        early = texts[5]
        self.assertEqual(repair_echo(LATEST_USER_TEXT, early, history_texts=texts), "")

    def test_repair_echo_leaves_clean_draft_alone_with_history(self) -> None:
        history = _long_history()
        texts = _history_texts(history)
        draft = "oh that's rough, want to talk about it?"
        self.assertEqual(
            repair_echo(LATEST_USER_TEXT, draft, history_texts=texts), draft
        )

    def test_responder_repairs_early_phrase_echo(self) -> None:
        history = _long_history()
        echo = (f"oh that's beautiful, {EARLY_PHRASE} last winter "
                "must have been amazing, tell me everything")
        router = FakeRouter(replies=[echo])
        responder = _responder(router, retries=0)
        bundle = responder.respond(
            chat_platform="local", user_text=LATEST_USER_TEXT, history=history
        )
        self.assertEqual(bundle.echo_action, "repaired")
        self.assertFalse(bundle.fallback)
        self.assertNotIn("northern lights", bundle.text.lower())
        self.assertTrue(
            parrot_check(
                LATEST_USER_TEXT, bundle.text,
                history_texts=_history_texts(history),
            ).ok
        )

    def test_responder_drops_unrepairable_early_echo(self) -> None:
        history = _long_history()
        early = _history_texts(history)[5]
        router = FakeRouter(replies=[early])  # pure verbatim echo of turn 5
        responder = _responder(router, retries=0)
        bundle = responder.respond(
            chat_platform="local", user_text=LATEST_USER_TEXT, history=history
        )
        # An echo must never ship: an in-character fallback line goes out
        # instead, and the bundle says so.
        self.assertEqual(bundle.echo_action, "dropped")
        self.assertTrue(bundle.fallback)
        self.assertNotIn(EARLY_PHRASE, bundle.text)

    def test_responder_clean_reply_with_long_history(self) -> None:
        # A long thread must not make the guard twitchy: a genuinely fresh
        # reply to the latest message ships untouched.
        history = _long_history()
        router = FakeRouter(replies=["it's almost three, why?"])
        responder = _responder(router, retries=0)
        bundle = responder.respond(
            chat_platform="local", user_text=LATEST_USER_TEXT, history=history
        )
        self.assertEqual(bundle.echo_action, "")
        self.assertFalse(bundle.fallback)
        self.assertIn("three", bundle.text)


# ── 2. gating under long threads ─────────────────────────────────────────────

class GatingLongThreadTest(unittest.TestCase):
    def _assert_stranger_turn(self, chat: ChatRef, turn: int) -> None:
        is_owner = is_owner_chat(chat, owner_chats=OWNER_CHATS)
        self.assertFalse(is_owner, f"turn {turn}: stranger chat tested as owner")
        mode = classify_chat(chat, is_owner=is_owner)
        self.assertIn(mode, (MODE_PRIVATE, MODE_GROUP), f"turn {turn}")
        decided = gate_decision(chat, is_owner=is_owner, restricted_enabled=True)
        self.assertTrue(is_restricted(decided), f"turn {turn}")
        self.assertEqual(decided, mode, f"turn {turn}")

    def test_stranger_dm_gated_every_turn(self) -> None:
        chat = ChatRef(platform="telegram", chat_id="999", kind=ChatKind.DM)
        for turn in range(60):
            self._assert_stranger_turn(chat, turn)
        self.assertEqual(
            classify_chat(chat, is_owner=False), MODE_PRIVATE
        )

    def test_stranger_group_gated_every_turn(self) -> None:
        chat = ChatRef(platform="telegram", chat_id="555", kind=ChatKind.GROUP,
                       title="some group")
        for turn in range(60):
            self._assert_stranger_turn(chat, turn)
            self.assertEqual(
                gate_decision(chat, is_owner=False, restricted_enabled=True),
                MODE_GROUP,
                f"turn {turn}",
            )

    def test_stranger_thread_gated_every_turn(self) -> None:
        # A thread inside a stranger group (key "telegram:555:42") is its own
        # surface — still never the owner.
        chat = ChatRef(platform="telegram", chat_id="555", kind=ChatKind.GROUP,
                       thread_id="42")
        self.assertEqual(chat.key, "telegram:555:42")
        for turn in range(60):
            self._assert_stranger_turn(chat, turn)

    def test_db_flag_alone_never_flips_mid_thread(self) -> None:
        # A registry row with is_owner=0 stays 0 no matter how long the
        # thread runs; the key set is not consulted differently per turn.
        chat = ChatRef(platform="whatsapp", chat_id="777", kind=ChatKind.DM)
        for turn in range(60):
            self.assertFalse(
                is_owner_chat(chat, owner_chats=OWNER_CHATS, db_is_owner=False),
                f"turn {turn}",
            )

    def test_owner_chats_set_not_mutated_by_repeated_checks(self) -> None:
        owner = {"telegram:1"}
        stranger = ChatRef(platform="telegram", chat_id="999", kind=ChatKind.DM)
        for _ in range(60):
            is_owner_chat(stranger, owner_chats=owner)
        self.assertEqual(owner, {"telegram:1"})

    def test_owner_dm_stays_owner_every_turn(self) -> None:
        chat = ChatRef(platform="telegram", chat_id="1", kind=ChatKind.DM)
        for turn in range(60):
            is_owner = is_owner_chat(chat, owner_chats=OWNER_CHATS)
            self.assertTrue(is_owner, f"turn {turn}")
            self.assertEqual(classify_chat(chat, is_owner=is_owner), MODE_OWNER)
            decided = gate_decision(chat, is_owner=is_owner, restricted_enabled=True)
            self.assertFalse(is_restricted(decided), f"turn {turn}")

    def test_console_stays_owner_every_turn(self) -> None:
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        for turn in range(60):
            self.assertTrue(is_owner_chat(chat, owner_chats=OWNER_CHATS),
                            f"turn {turn}")

    def test_kill_switch_contract_unchanged(self) -> None:
        # gate_decision's owner kill-switch: with restricted gating disabled
        # everything is owner mode (the owner's explicit choice).  Pinned so
        # the 60-turn tests above can't silently start passing vacuously.
        stranger = ChatRef(platform="telegram", chat_id="999", kind=ChatKind.DM)
        self.assertEqual(
            gate_decision(stranger, is_owner=False, restricted_enabled=False),
            MODE_OWNER,
        )


# ── 3. no AI-mode leakage into stranger chats ────────────────────────────────

class _ExplodingRouter:
    """Any model touch in a stranger chat is a test failure."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt, params=None, **kw):
        self.calls += 1
        raise AssertionError("model consulted for a stranger chat")


class _OperatorRuntime:
    """The production gate shape: PartnerRuntime._is_operator."""

    def __init__(self, owner_chats: set[str]) -> None:
        self._owner_chats = set(owner_chats)

    def _is_operator(self, message: Any) -> bool:
        return is_owner_chat(message.chat, owner_chats=self._owner_chats)


def _ctx(router: Any = None) -> SimpleNamespace:
    return SimpleNamespace(settings=None, extras={}, memory=None, router=router)


def _msg(chat: ChatRef, text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(chat=chat, text=text)


STRANGER_DM = ChatRef(platform="telegram", chat_id="999", kind=ChatKind.DM)
STRANGER_GROUP = ChatRef(platform="telegram", chat_id="555", kind=ChatKind.GROUP)
OWNER_DM = ChatRef(platform="telegram", chat_id="1", kind=ChatKind.DM)

#: NL inputs that would each launch an organ in the owner's DM.
AGENT_MODE_PROBES = [
    "research the price of bitcoin right now",   # research organ
    "start a mission to watch my portfolio",     # directives / missions
    "build me a small todo app",                 # builder organ
    "download that video for me",                # downloader organ
    "let's play hangman",                        # NL game launch (owner-DM only)
    "hello",                                     # fast path
    "what time is it",                           # fast path
    "/upgrade list",                             # slash: falls through as text
]


class NoAgentModeLeakageTest(unittest.TestCase):
    def _mind(self) -> CoreMind:
        return CoreMind(_ctx(router=_ExplodingRouter()),
                        runtime=_OperatorRuntime(OWNER_CHATS))

    def test_stranger_dm_nl_never_routes(self) -> None:
        mind = self._mind()
        for turn, probe in enumerate(AGENT_MODE_PROBES * 8):  # 64 turns
            reply = mind.handle(probe, message=_msg(STRANGER_DM, probe),
                                chat_key=STRANGER_DM.key)
            self.assertIsNone(
                reply,
                f"turn {turn}: stranger DM got agent-mode behaviour for {probe!r}",
            )
        router = mind.context.router
        self.assertEqual(router.calls, 0, "the model was consulted for a stranger")

    def test_stranger_group_nl_never_routes(self) -> None:
        mind = self._mind()
        for turn, probe in enumerate(AGENT_MODE_PROBES * 8):
            reply = mind.handle(probe, message=_msg(STRANGER_GROUP, probe),
                                chat_key=STRANGER_GROUP.key)
            self.assertIsNone(
                reply,
                f"turn {turn}: stranger group got agent-mode behaviour for {probe!r}",
            )

    def test_owner_dm_fast_path_still_works(self) -> None:
        # Positive control: the same mind answers the owner, so the Nones
        # above are the gate working — not the mind being broken.
        mind = self._mind()
        reply = mind.handle("hello", message=_msg(OWNER_DM, "hello"),
                            chat_key=OWNER_DM.key)
        self.assertIsNotNone(reply)

    def test_slash_commands_need_operator(self) -> None:
        # The exact condition PartnerRuntime._process uses before routing a
        # slash message into handle_control: a stranger's "/status" must fall
        # through to conversation, never to command dispatch.
        from nomorals.agents.partner_runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt._owner_chats = set(OWNER_CHATS)
        stranger_msg = _msg(STRANGER_DM, "/status")
        owner_msg = _msg(OWNER_DM, "/status")
        self.assertFalse(rt._is_operator(stranger_msg))
        self.assertTrue(rt._is_operator(owner_msg))
        # ...and on a group surface, even with a slash.
        self.assertFalse(rt._is_operator(_msg(STRANGER_GROUP, "/status")))

    def test_upgrade_denied_direct_call_stranger(self) -> None:
        # Defense in depth: _control_upgrade re-checks the owner gate itself,
        # so even a direct call with a stranger chat is denied fail-closed.
        from nomorals.agents.partner_runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = _ctx()
        rt._owner_chats = set(OWNER_CHATS)
        stranger_chat = SimpleNamespace(key=STRANGER_DM.key)
        reply = rt._control_upgrade("list", _chat=stranger_chat)
        self.assertEqual(reply, "owner-only: /upgrade is not available in this chat.")
        # And with no proven chat at all: denied, not defaulted.
        reply = rt._control_upgrade("list", _chat=None)
        self.assertEqual(reply, "owner-only: /upgrade is not available in this chat.")


# ── 4. typing length-scaling (F2 contract, still holding) ─────────────────────

class TypingScalingTest(unittest.TestCase):
    def test_longer_text_types_longer(self) -> None:
        short = human_typing_seconds("hey", rng=random.Random(7))
        long_ = human_typing_seconds("x" * 900, rng=random.Random(7))
        self.assertGreater(long_, short)
        self.assertGreater(long_, short * 5)

    def test_scaling_monotonic_across_lengths(self) -> None:
        durations = [
            human_typing_seconds("x" * n, rng=random.Random(42))
            for n in (10, 100, 400, 1200)
        ]
        for earlier, later in zip(durations, durations[1:]):
            self.assertGreater(later, earlier)

    def test_minimum_floor(self) -> None:
        self.assertGreaterEqual(
            human_typing_seconds("", rng=random.Random(0)), 2.0
        )
        self.assertGreaterEqual(
            human_typing_seconds("k", minimum=3.0, rng=random.Random(0)), 3.0
        )

    def test_cap_respected(self) -> None:
        capped = human_typing_seconds("x" * 10000, cap=10.0,
                                      rng=random.Random(0))
        self.assertLessEqual(capped, 10.0)

    def test_cap_zero_means_no_cap(self) -> None:
        # NM_PARTNER_TYPING_CAP_SECONDS=0 disables the cap — a long message
        # must not be clamped to ~minimum.
        uncapped = human_typing_seconds("x" * 10000, cap=0,
                                        rng=random.Random(0))
        self.assertGreater(uncapped, 45.0)

    def test_mood_modulates_pace(self) -> None:
        # Same seed, same text: a drained thumb is slower than an eager one.
        text = "x" * 300
        tired = human_typing_seconds(
            text, mood={"energy": 10, "happiness": 40}, rng=random.Random(3))
        eager = human_typing_seconds(
            text, mood={"energy": 90, "happiness": 90}, rng=random.Random(3))
        self.assertGreater(tired, eager)

    def test_runtime_types_each_part_with_scaled_seconds(self) -> None:
        # End to end at the runtime layer: _send_reply issues one typing
        # indicator per part, each scaled to THAT part's length.
        from nomorals.agents.partner_runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.settings = SimpleNamespace(partner=SimpleNamespace(
            typing_in_groups=False,
            part_delay_seconds=0.0,
            typing_seconds=2.0,
            typing_cap_seconds=45.0,
        ))
        rt.brain = SimpleNamespace(
            mood=SimpleNamespace(
                current=lambda: SimpleNamespace(values={"energy": 60,
                                                        "happiness": 60})),
            presence_rng=random.Random(11),
        )
        typing_calls: list[float] = []
        sent: list[str] = []

        class _Gateway:
            def typing(self, platform: str, chat: ChatRef, seconds: float = 3.0) -> bool:
                typing_calls.append(seconds)
                return True

            def send(self, platform: str, chat: ChatRef, text: str, **kw: Any):
                sent.append(text)
                return SendResult(ok=True, platform=platform)

        rt.gateway = _Gateway()
        chat = ChatRef(platform="telegram", chat_id="1", kind=ChatKind.DM)
        message = ChatMessage(chat=chat, incoming=True, text="hi")
        parts = ["ok " * 10, "here is the longer part of the reply " * 30]
        rt._send_reply(message, parts)

        self.assertEqual(len(sent), 2)
        self.assertEqual(len(typing_calls), 2,
                         "one length-scaled indicator per part")
        self.assertGreater(typing_calls[1], typing_calls[0],
                           "the longer part must hold the indicator longer")
        self.assertGreaterEqual(min(typing_calls), 2.0)


if __name__ == "__main__":
    unittest.main()
