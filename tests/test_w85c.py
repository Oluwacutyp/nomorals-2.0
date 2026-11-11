"""Wave 85c — character-scaled typing + ownership-aware chat gating.

Covers:

* the typing model: longer text types longer, floor/cap hold, mood shifts
  the pace, thinking lead-in stays short
* gating classification: owner DM = full, stranger DM / group / channel =
  restricted, owner messages anywhere = full
* prompt wiring: restricted chats get the gate block + neutral
  relationship line and NO owner-private context (memories, continuity)
* runtime: per-part typing in DMs AND groups, long-send chunk typing,
  typing while the brain is thinking, config switches respected
* Discord adapter: typing is fire-and-forget per call; the runtime
  keepalive's re-firing is what refreshes it
"""

from __future__ import annotations

import asyncio
import random
import statistics
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerBrain, PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.partner.context import PartnerContextBuilder
from nomorals.partner.gating import (
    MODE_GROUP,
    MODE_OWNER,
    MODE_PRIVATE,
    classify_chat,
    gate_block,
    is_restricted,
    relationship_block_for,
)
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import default_persona
from nomorals.partner.presence import human_typing_seconds, thinking_typing_seconds
from nomorals.partner.relationship import Relationship
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.discord import DiscordAdapter
from nomorals.social.chat.gateway import ChatGateway


def _median(fn, n=40, seed=7) -> float:
    rng = random.Random(seed)
    return statistics.median(fn(rng) for _ in range(n))


class TypingModelTests(unittest.TestCase):
    def test_longer_text_types_longer(self) -> None:
        fast = _median(lambda r: human_typing_seconds("k", rng=r, minimum=0.1))
        mid = _median(lambda r: human_typing_seconds("x" * 100, rng=r, minimum=0.1))
        long = _median(lambda r: human_typing_seconds("x" * 400, rng=r, minimum=0.1))
        self.assertLess(fast, mid)
        self.assertLess(mid, long)
        # a 400-char wall must actually feel like a wall of typing
        self.assertGreater(long, 10.0)

    def test_floor_and_cap_hold(self) -> None:
        self.assertGreaterEqual(human_typing_seconds("k", minimum=2.5), 2.5)
        for _ in range(25):
            self.assertLessEqual(human_typing_seconds("x" * 5000, cap=30.0), 30.0)

    def test_mood_shifts_pace(self) -> None:
        tired = _median(lambda r: human_typing_seconds(
            "x" * 200, mood={"energy": 10}, rng=r, minimum=0.1))
        excited = _median(lambda r: human_typing_seconds(
            "x" * 200, mood={"energy": 90, "happiness": 90}, rng=r, minimum=0.1))
        self.assertGreater(tired, excited)

    def test_thinking_lead_in_is_short_and_scaled(self) -> None:
        short = _median(lambda r: thinking_typing_seconds("hey", rng=r))
        long = _median(lambda r: thinking_typing_seconds("x" * 800, rng=r))
        self.assertLessEqual(long, 8.0)
        self.assertGreaterEqual(long, 1.5)
        self.assertGreater(long, short)


class GatingTests(unittest.TestCase):
    def _chat(self, kind: str, key_suffix: str = "") -> ChatRef:
        return ChatRef(platform="telegram", chat_id=key_suffix or "123", kind=kind)

    def test_classification(self) -> None:
        self.assertEqual(classify_chat(self._chat(ChatKind.DM), is_owner=True), MODE_OWNER)
        self.assertEqual(classify_chat(self._chat(ChatKind.GROUP), is_owner=True), MODE_OWNER)
        self.assertEqual(classify_chat(self._chat(ChatKind.DM), is_owner=False), MODE_PRIVATE)
        self.assertEqual(classify_chat(self._chat(ChatKind.GROUP), is_owner=False), MODE_GROUP)
        self.assertEqual(classify_chat(self._chat(ChatKind.CHANNEL), is_owner=False), MODE_GROUP)

    def test_owner_is_never_restricted(self) -> None:
        self.assertFalse(is_restricted(MODE_OWNER))
        self.assertTrue(is_restricted(MODE_PRIVATE))
        self.assertTrue(is_restricted(MODE_GROUP))

    def test_gate_blocks(self) -> None:
        self.assertEqual(gate_block(MODE_OWNER), "")
        private = gate_block(MODE_PRIVATE)
        self.assertIn("not the person you're with", private)
        self.assertIn("You are not their assistant", private)
        group = gate_block(MODE_GROUP)
        self.assertIn("group chat", group)
        self.assertIn("one voice in the room", group)

    def test_relationship_override(self) -> None:
        self.assertEqual(relationship_block_for(MODE_OWNER), "")
        neutral = relationship_block_for(MODE_PRIVATE)
        self.assertIn("not the one you're with", neutral)


class PromptWiringTests(unittest.TestCase):
    """Builder level: the restricted prompt must not carry owner context."""

    def _parts(self) -> SimpleNamespace:
        persona = default_persona()
        return SimpleNamespace(
            persona=persona,
            mood=MoodEngine(persona.baselines),
            relationship=Relationship(id="default", stage="in_love", trust=90),
            builder=PartnerContextBuilder(),
        )

    def test_owner_prompt_has_relationship_no_gate(self) -> None:
        p = self._parts()
        system = p.builder.build(persona=p.persona, mood=p.mood, relationship=p.relationship,
                                 memories=["memory-A"], continuity_lines=["thread-B"],
                                 platform="telegram")
        text = system.content
        self.assertIn("in_love", text)
        self.assertIn("You are in a relationship with the person you're talking to", text)
        self.assertNotIn("Who you're talking to RIGHT NOW", text)

    def test_restricted_builder_swaps_identity_and_relationship(self) -> None:
        p = self._parts()
        system = p.builder.build(
            persona=p.persona, mood=p.mood, relationship=p.relationship,
            platform="discord",
            gate_note=gate_block(MODE_PRIVATE),
            relationship_override=relationship_block_for(MODE_PRIVATE),
        )
        text = system.content
        self.assertIn("Who you're talking to RIGHT NOW", text)
        self.assertIn("not the one you're with", text)
        # the private relationship block AND the persona's partner paragraph
        # are both out of the prompt in restricted mode
        self.assertNotIn("in_love", text)
        self.assertNotIn("You are in a relationship with the person you're talking to", text)

    def test_owner_context_outranks_registry_blocks(self) -> None:
        """Regression pin (capabilities f04d658d + catalog ee8383e2):
        the two registry-rendered reference blocks were budget-PROTECTED,
        so as the live registries grew they silently evicted recalled
        memories and continuity from owner prompts. Per-turn owner
        context must spend BEFORE reference material at any budget —
        here a deliberately tight one, so the pin holds whatever size
        the registries reach next."""
        p = self._parts()
        tight = PartnerContextBuilder(total_budget=2500)
        system = tight.build(persona=p.persona, mood=p.mood, relationship=p.relationship,
                             memories=["memory-A"], continuity_lines=["thread-B"],
                             platform="telegram")
        self.assertIn("memory-A", system.content)
        self.assertIn("thread-B", system.content)


class _WindowRouter:
    """Router double whose active provider advertises a context window,
    in the shape Brain.context_window reads (active + get -> provider
    carrying context_len). No chat surface: construction + budget only."""

    def __init__(self, window: int) -> None:
        self.active = "fake-7b"
        self.broker = None
        self._provider = SimpleNamespace(context_len=window)

    def get(self, name: str) -> Any:
        return self._provider


class DerivedBudgetTests(unittest.TestCase):
    """Power mode (owner directive 2026-11-10): no fixed prompt budget.

    The builder's default used to be a hardcoded 4200-token ceiling;
    as the live registries grew, the shrink walk silently truncated
    owner prompts to fit it. The budget now derives from the serving
    rung's real context window minus the reply reserve, the shrink
    walk remains only as the genuine-overflow guard, and an unknown
    window means assemble-in-full.
    """

    def _fat_build(self, builder: PartnerContextBuilder):
        persona = default_persona()
        return builder.build(
            persona=persona,
            mood=MoodEngine(persona.baselines),
            relationship=Relationship(id="default", stage="in_love", trust=90),
            memories=[f"memory-line-{i} " + "x" * 120 for i in range(12)],
            continuity_lines=[f"thread-line-{i} " + "y" * 120 for i in range(6)],
            platform="telegram",
        )

    def test_default_builder_assembles_everything_in_full(self) -> None:
        builder = PartnerContextBuilder()
        # No ceiling is imposed by default construction.
        self.assertIsNone(builder.total_budget)
        system = self._fat_build(builder)
        # The assembled prompt EXCEEDS the old 4200 ceiling...
        self.assertGreater(PartnerContextBuilder.total_tokens(system), 4200)
        # ...and nothing was truncated to fit under it: the tail of
        # every block survives — memory, continuity, and both
        # registry-rendered reference blocks, whole.
        self.assertIn("memory-line-11", system.content)
        self.assertIn("thread-line-5", system.content)
        self.assertIn("never claim you can't", system.content)
        self.assertIn("self-description, not yours", system.content)

    def test_budget_tracks_the_rung_window(self) -> None:
        big = PartnerContextBuilder(budget_resolver=lambda: 131_072 - 512)
        full = self._fat_build(big)
        self.assertIn("self-description, not yours", full.content)
        # A small real window engages the shrink walk as a genuine-
        # overflow guard: reference material (catalog spends last) is
        # cut, while per-turn owner context still lands whole.
        small = PartnerContextBuilder(budget_resolver=lambda: 3000 - 512)
        shrunk = self._fat_build(small)
        self.assertLess(PartnerContextBuilder.total_tokens(shrunk),
                        PartnerContextBuilder.total_tokens(full))
        self.assertNotIn("self-description, not yours", shrunk.content)
        self.assertIn("memory-line-11", shrunk.content)
        self.assertIn("thread-line-5", shrunk.content)

    def test_unknown_window_means_no_budget(self) -> None:
        builder = PartnerContextBuilder(budget_resolver=lambda: None)
        system = self._fat_build(builder)
        self.assertIn("memory-line-11", system.content)
        self.assertIn("self-description, not yours", system.content)

    def test_responder_derives_budget_from_router_window(self) -> None:
        from nomorals.partner.responder import PartnerResponder

        persona = default_persona()

        def budget_for(window: int):
            responder = PartnerResponder(
                _WindowRouter(window), persona, MoodEngine(persona.baselines),
                Relationship(id="default", stage="in_love", trust=90),
                None, None)
            return responder._prompt_budget()

        # budget = window - reply reserve (the responder's max_tokens,
        # 512 by default) — two windows, two different budgets.
        self.assertEqual(budget_for(8192), 8192 - 512)
        self.assertEqual(budget_for(131_072), 131_072 - 512)

    def test_responder_unknown_window_is_unbounded(self) -> None:
        from nomorals.partner.responder import PartnerResponder

        persona = default_persona()
        responder = PartnerResponder(
            _RecordingRouter(), persona, MoodEngine(persona.baselines),
            Relationship(id="default", stage="in_love", trust=90),
            None, None)
        self.assertIsNone(responder._prompt_budget())
        self.assertIsNone(responder.builder._effective_budget())


class _RecordingRouter:
    def __init__(self) -> None:
        self.last_messages: list[Message] = []

    def chat(self, messages: list[Message], params: SamplingParams | None = None, **kw) -> LLMResponse:
        self.last_messages = list(messages)
        return LLMResponse(text="mhm, that tracks", model="fake-7b")


class ResponderGateTest(unittest.TestCase):
    """The actual guarantee: restricted mode strips owner context end-to-end
    through respond() — memory and continuity are dropped, the gate block
    and neutral relationship line are in, and the system prompt is the one
    the model actually receives."""

    def _responder(self, router) -> "PartnerResponder":
        from nomorals.partner.responder import PartnerResponder

        persona = default_persona()
        return PartnerResponder(
            router, persona, MoodEngine(persona.baselines),
            Relationship(id="default", stage="in_love", trust=90),
            None, None,
        )

    def test_restricted_respond_drops_owner_context(self) -> None:
        router = _RecordingRouter()
        self._responder(router).respond(
            chat_platform="telegram", user_text="hey",
            memories=["memory-A"], continuity_lines=["thread-B"],
            gate_mode=MODE_PRIVATE,
        )
        self.assertTrue(router.last_messages, "no model call happened")
        system = router.last_messages[0].content
        self.assertIn("Who you're talking to RIGHT NOW", system)
        self.assertIn("not the one you're with", system)
        self.assertNotIn("memory-A", system)
        self.assertNotIn("thread-B", system)
        self.assertNotIn("You are in a relationship with the person you're talking to", system)

    def test_owner_respond_keeps_everything(self) -> None:
        router = _RecordingRouter()
        self._responder(router).respond(
            chat_platform="telegram", user_text="hey",
            memories=["memory-A"], continuity_lines=["thread-B"],
            gate_mode=MODE_OWNER,
        )
        system = router.last_messages[0].content
        self.assertIn("memory-A", system)
        self.assertIn("thread-B", system)
        self.assertIn("You are in a relationship with the person you're talking to", system)
        self.assertNotIn("Who you're talking to RIGHT NOW", system)

    def test_system_prompt_is_actually_sent(self) -> None:
        """Regression: the assembled system prompt must reach the model."""
        router = _RecordingRouter()
        self._responder(router).respond(chat_platform="telegram", user_text="hey", gate_mode=MODE_OWNER)
        self.assertEqual(router.last_messages[0].role, "system")
        # The persona is Devon ("wren" is a nickname inside the prompt),
        # so the prompt opens with "You are Devon".
        self.assertIn("You are Devon", router.last_messages[0].content)


class _RuntimeFixture:
    """Shared scaffolding: real gateway + local fake adapter + fake router."""

    def _make(self, **partner_overrides) -> tuple[PartnerRuntime, "RecordingAdapter", SimpleNamespace]:
        import tempfile

        tmp = tempfile.TemporaryDirectory(prefix="nm-w85c-")
        settings = load_settings(overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            **partner_overrides,
        })
        context = build_context(settings, with_executor=False, with_tools=False)
        context.router = _ScriptedRouter()
        adapter = RecordingAdapter("local")
        gateway = ChatGateway({"local": adapter}, db=context.db)
        runtime = PartnerRuntime(context, gateway=gateway)
        runtime.brain.presence_rng = random.Random(34)  # seed 34: no busy gaps
        self._cleanups.append((context, tmp))
        return runtime, adapter, settings

    def __init__(self) -> None:
        self._cleanups = []

    def close(self) -> None:
        for context, tmp in self._cleanups:
            try:
                context.close()
            except Exception:
                pass
            tmp.cleanup()


class _ScriptedRouter:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], params: SamplingParams | None = None, **kw) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(text="mhm, that tracks", model="fake-7b")


class RecordingAdapter(ChatAdapter):
    """Local adapter that records typing calls (non-blocking)."""

    def __init__(self, name: str = "local") -> None:
        super().__init__(media_dir="/tmp/nm-w85c-media")
        self.name = name
        self.sent: list[str] = []
        self.typing_calls: list[tuple[str, float]] = []

    def run(self, handler) -> None:  # pragma: no cover - never started in tests
        raise NotImplementedError

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        # Full ChatAdapter.send signature: the gateway forwards buttons /
        # parse_mode when set, so a fake that narrows the signature breaks
        # every send with a TypeError.
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name, message_id=f"m{len(self.sent)}")

    def typing(self, chat: ChatRef, seconds: float = 3.0,
               action: str = "typing") -> bool:
        # Full ChatAdapter.typing signature: the gateway passes action=,
        # so a fake without it breaks every typing call with a TypeError.
        self.typing_calls.append((chat.key, seconds))
        return True


class BrainGatingTest(unittest.TestCase):
    """End-to-end: who sees what, per chat mode."""

    SECRET_MEMORY = "SECRET owner memory"
    CONTINUITY_LINE = "CONTINUITY open thread"

    def setUp(self) -> None:
        self.fx = _RuntimeFixture()
        self.runtime, self.adapter, self.settings = self.fx._make()
        self.brain: PartnerBrain = self.runtime.brain
        # Owner-private context, always available when the brain reaches for it.
        self.brain.responder.recall = lambda *a, **k: [self.SECRET_MEMORY]
        self.brain._continuity_lines = lambda *a, **k: [self.CONTINUITY_LINE]
        self.router = self.runtime.context.router

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.fx.close()

    def _last_system(self) -> str:
        self.assertTrue(self.router.calls, "no model call happened")
        return self.router.calls[-1][0].content

    def test_owner_dm_gets_full_context(self) -> None:
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        # deliver_reply: the generation half, no presence rolls — the gate
        # lives in the prompt assembly, which is exactly what's under test.
        self.brain.deliver_reply(ChatMessage(chat=chat, incoming=True, text="hey, how are you?", sender="you"))
        text = self._last_system()
        self.assertIn(self.SECRET_MEMORY, text)
        self.assertIn(self.CONTINUITY_LINE, text)
        self.assertNotIn("Who you're talking to RIGHT NOW", text)

    def test_stranger_dm_is_restricted(self) -> None:
        chat = ChatRef(platform="local", chat_id="stranger", kind=ChatKind.DM, peer="newbie")
        self.brain.deliver_reply(ChatMessage(chat=chat, incoming=True, text="hey, what are you up to?", sender="newbie"))
        text = self._last_system()
        self.assertIn("Who you're talking to RIGHT NOW", text)
        self.assertIn("not the person you're with", text)
        self.assertNotIn(self.SECRET_MEMORY, text)
        self.assertNotIn(self.CONTINUITY_LINE, text)

    def test_group_is_restricted(self) -> None:
        chat = ChatRef(platform="local", chat_id="g1", kind=ChatKind.GROUP, title="Game Night")
        message = ChatMessage(chat=chat, incoming=True, text=f"yo {self.brain.persona.name} what's up", sender="someone", mentioned=True)
        self.brain.deliver_reply(message)
        text = self._last_system()
        self.assertIn("group chat with other people", text)
        self.assertNotIn(self.SECRET_MEMORY, text)
        self.assertNotIn(self.CONTINUITY_LINE, text)

    def test_owner_message_in_group_still_full(self) -> None:
        chat = ChatRef(platform="local", chat_id="g1:console", kind=ChatKind.GROUP)
        self.brain.deliver_reply(ChatMessage(chat=chat, incoming=True, text="tell them about my day", sender="you"))
        text = self._last_system()
        self.assertIn(self.SECRET_MEMORY, text)
        self.assertNotIn("Who you're talking to RIGHT NOW", text)

    def test_gating_can_be_switched_off(self) -> None:
        self.settings.partner.gate_restricted_chats = False
        chat = ChatRef(platform="local", chat_id="stranger", kind=ChatKind.DM)
        self.brain.deliver_reply(ChatMessage(chat=chat, incoming=True, text="hey", sender="newbie"))
        text = self._last_system()
        self.assertIn(self.SECRET_MEMORY, text)
        self.assertNotIn("Who you're talking to RIGHT NOW", text)


class RuntimeTypingTest(unittest.TestCase):
    """Typing indicators: per part, in groups, per chunk, while thinking."""

    def setUp(self) -> None:
        self.fx = _RuntimeFixture()
        self.runtime, self.adapter, self.settings = self.fx._make()
        self.brain: PartnerBrain = self.runtime.brain
        self.brain.presence_rng = random.Random(34)  # seed 34: no busy gaps

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.fx.close()

    def test_per_part_typing_in_group_scales_with_length(self) -> None:
        chat = ChatRef(platform="local", chat_id="g1", kind=ChatKind.GROUP)
        message = ChatMessage(chat=chat, incoming=True, text="hi", sender="x")
        self.runtime._send_reply(message, ["short one", "y" * 300])
        self.assertEqual(len(self.adapter.sent), 2)
        self.assertEqual(len(self.adapter.typing_calls), 2)
        first, second = self.adapter.typing_calls
        self.assertGreater(second[1], first[1])  # longer part types longer
        self.assertGreaterEqual(first[1], self.settings.partner.typing_seconds)

    def test_dm_typing_still_works(self) -> None:
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        message = ChatMessage(chat=chat, incoming=True, text="hi", sender="you")
        self.runtime._send_reply(message, ["a" * 120])
        self.assertEqual(len(self.adapter.typing_calls), 1)

    def test_group_typing_config_off(self) -> None:
        self.settings.partner.typing_in_groups = False
        chat = ChatRef(platform="local", chat_id="g1", kind=ChatKind.GROUP)
        message = ChatMessage(chat=chat, incoming=True, text="hi", sender="x")
        self.runtime._send_reply(message, ["a" * 120])
        self.assertEqual(self.adapter.typing_calls, [])
        # ...but DMs are unaffected
        dm = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        self.runtime._send_reply(ChatMessage(chat=dm, incoming=True, text="hi", sender="you"), ["b" * 120])
        self.assertEqual(len(self.adapter.typing_calls), 1)

    def test_typing_cap_respected(self) -> None:
        self.settings.partner.typing_cap_seconds = 20.0
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        message = ChatMessage(chat=chat, incoming=True, text="hi", sender="you")
        self.runtime._send_reply(message, ["z" * 5000])
        self.assertLessEqual(self.adapter.typing_calls[0][1], 20.0)

    def test_long_send_types_per_chunk(self) -> None:
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        sent = self.runtime._send_long("local", chat, "x" * 9000, limit=3800)
        self.assertEqual(sent, 3)
        self.assertEqual(len(self.adapter.sent), 3)
        self.assertEqual(len(self.adapter.typing_calls), 3)
        for _, seconds in self.adapter.typing_calls:
            self.assertGreater(seconds, 0)

    def test_thinking_keepalive_holds_indicator(self) -> None:
        self.settings.partner.typing_keepalive_seconds = 0.1
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        stop = threading.Event()
        thread = threading.Thread(target=self.runtime._typing_keepalive, args=(chat, stop), daemon=True)
        thread.start()
        time.sleep(0.4)
        stop.set()
        thread.join(5)
        self.assertGreaterEqual(len(self.adapter.typing_calls), 2)

    def test_process_types_while_brain_thinks(self) -> None:
        self.settings.partner.typing_keepalive_seconds = 0.1
        real_handle = self.brain.handle_message

        def slow_handle(message):
            time.sleep(0.3)
            return real_handle(message)

        self.brain.handle_message = slow_handle  # type: ignore[method-assign]
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        message = ChatMessage(chat=chat, incoming=True, text="hey, you up?", sender="you")
        self.runtime._process(message)
        self.brain.handle_message = real_handle
        self.assertTrue(self.adapter.sent, "no reply was sent")
        self.assertTrue(self.adapter.typing_calls, "no typing indicator while thinking")

    def test_group_quiet_no_thinking_typing(self) -> None:
        """A group message she'll stay quiet on must never show typing.

        Regression: the thinking keepalive used to start before the brain's
        group-quiet decision, so every unmentioned group message flashed
        "typing..." with no reply after it.
        """
        self.settings.partner.typing_keepalive_seconds = 0.1
        real_handle = self.brain.handle_message

        def slow_handle(message):
            time.sleep(0.3)
            return real_handle(message)

        self.brain.handle_message = slow_handle  # type: ignore[method-assign]
        try:
            chat = ChatRef(platform="local", chat_id="g1", kind=ChatKind.GROUP)
            # no mention, no reply-to, persona name absent -> she stays quiet
            message = ChatMessage(chat=chat, incoming=True,
                                  text="anyone up for lunch?", sender="x")
            self.assertFalse(self.brain.should_speak_in_group(message))
            self.runtime._process(message)
            time.sleep(0.3)  # give a stray keepalive thread time to fire, if any
            self.assertEqual(self.adapter.typing_calls, [],
                             "typing indicator fired in a group she stayed quiet in")
            self.assertEqual(self.adapter.sent, [])
        finally:
            self.brain.handle_message = real_handle

    def test_group_mentioned_thinking_typing(self) -> None:
        """A group message addressing her still shows typing while thinking."""
        self.settings.partner.typing_keepalive_seconds = 0.1
        real_handle = self.brain.handle_message

        def slow_handle(message):
            time.sleep(0.3)
            return real_handle(message)

        self.brain.handle_message = slow_handle  # type: ignore[method-assign]
        try:
            chat = ChatRef(platform="local", chat_id="g1", kind=ChatKind.GROUP)
            message = ChatMessage(chat=chat, incoming=True, text="hey everyone",
                                  sender="x", mentioned=True)
            self.assertTrue(self.brain.should_speak_in_group(message))
            self.runtime._process(message)
            self.assertTrue(self.adapter.typing_calls,
                            "no typing indicator while thinking about a mentioned group message")
        finally:
            self.brain.handle_message = real_handle


class DiscordTypingRefreshTest(unittest.TestCase):
    """Discord clears typing after ~10s — somebody must re-send it.

    Since 9be2748a (owner order 2026-11-01) that somebody is the
    CALLER: adapter.typing is fire-and-forget (one call schedules one
    send_typing and returns at once — the indicator never delays the
    message it precedes), and the runtime's typing keepalive re-fires
    it (pinned against the local adapter in RuntimeTypingTest). This
    class pins the Discord half of that contract: every re-fire lands
    exactly one more send_typing, resolved through the adapter's DM
    channel path."""

    class _FakeChannel:
        def __init__(self) -> None:
            self.id = 777
            self.calls = 0

        async def send_typing(self) -> None:
            self.calls += 1

    class _FakeUser:
        id = 42

        def __init__(self, channel: _FakeChannel) -> None:
            self._channel = channel

        async def create_dm(self) -> _FakeChannel:
            return self._channel

    def test_refire_refreshes_the_indicator(self) -> None:
        channel = self._FakeChannel()
        user = self._FakeUser(channel)
        bot = SimpleNamespace(guilds=[], get_users=lambda: iter(()),
                              get_private_channels=lambda: [])

        async def fetch_user(uid: int):
            return user

        async def fetch_channel(cid: int):
            # DM chat ids are channel ids in this adapter; fetch_channel
            # is the primary resolution, fetch_user the fallback.
            return channel

        bot.fetch_user = fetch_user
        bot.fetch_channel = fetch_channel

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        time.sleep(0.05)
        try:
            adapter = DiscordAdapter(token="whatever.token.sig", media_dir="/tmp/nm-w85c-discord")
            adapter._bot = bot
            adapter._loop = loop
            chat = ChatRef(platform="discord", chat_id="777", kind=ChatKind.DM)
            # The keepalive's cadence, driven by hand: three re-fires.
            for _ in range(3):
                self.assertTrue(adapter.typing(chat, seconds=0.4))
            # Fire-and-forget: the events land on the adapter loop
            # asynchronously — wait for the scheduled coroutines.
            deadline = time.time() + 5
            while channel.calls < 3 and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(channel.calls, 3)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            loop.close()


if __name__ == "__main__":
    unittest.main()
