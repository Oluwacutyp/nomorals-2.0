"""Tests for the character gate and human texting discipline.

The contract under test: in every chat — owner, group, stranger — the reply
reads like a real person. No "I am a bot" (that is a hard gate with a
rewrite, and a fallback if the rewrite never lands), no markdown furniture
(no ``**bold**``, no bullet lists, no ``_`` or ``*`` markup), and emoji only
in human amounts. Owner control commands (/status and friends) are the one
door out of character — that flow already lives in the runtime and is not
re-asserted here.
"""

from __future__ import annotations

import random
import tempfile
import time
import unittest
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerBrain, PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import default_persona, persona_from_dict
from nomorals.partner.relationship import Relationship
from nomorals.partner.responder import PartnerResponder
from nomorals.partner.style import (
    humanize_emoji,
    identity_leak_check,
    normalize_formatting,
    strip_robotic,
)
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway


class ScriptedRouter:
    """Router with a queue of scripted replies; repeats the last one."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.fallback_text = self.replies[-1] if self.replies else "mhm"
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else self.fallback_text
        return LLMResponse(text=text, model="fake")


def _responder(replies: list[str], *, disclosure: str = "never") -> tuple[PartnerResponder, ScriptedRouter]:
    persona = default_persona()
    persona.disclosure = disclosure
    mood = MoodEngine(persona.baselines, now=1_000_000.0, db=None)
    relationship = Relationship()
    router = ScriptedRouter(replies)
    responder = PartnerResponder(
        router, persona, mood, relationship, memory=None, background=None
    )
    return responder, router


class IdentityLeakCheckTest(unittest.TestCase):
    def test_catches_the_tells(self) -> None:
        for text in (
            "well, i'm a bot, so i can't really drive you there",
            "I AM A BOT, deal with it",
            "to be honest i'm an ai and my training data has limits",
            "as a language model i should say",
            "i'm not a real person, just here to chat",
        ):
            self.assertFalse(identity_leak_check(text).ok, text)

    def test_clean_text_passes(self) -> None:
        for text in (
            "i can't come tonight, work ran long. sorry",
            "why would i be a bot, i hate your playlists, that's what i think",
            "okay i'm down. 8 or 9?",
        ):
            self.assertTrue(identity_leak_check(text).ok, text)

    def test_human_phrases_are_not_leaks(self) -> None:
        self.assertTrue(identity_leak_check("i'm powered by coffee and spite").ok)
        self.assertTrue(identity_leak_check("i run on four hours of sleep and spite").ok)
        self.assertTrue(identity_leak_check("my training for the 5k is going badly").ok)


class NormalizeFormattingTest(unittest.TestCase):
    def test_strips_document_furniture(self) -> None:
        text = (
            "**the plan:**\n"
            "- get gas\n"
            "- call the plumber\n"
            "1. buy groceries\n"
            "\n"
            "### tomorrow\n"
            "maybe _tonight_ instead? check [google maps](https://maps.example.com)\n"
            "```\ncode fence\n```"
        )
        out = normalize_formatting(text)
        self.assertNotIn("**", out)
        self.assertNotIn("_", out)
        self.assertNotIn("###", out)
        self.assertNotIn("- ", out)
        self.assertNotIn("1. ", out)
        self.assertNotIn("](https", out)
        self.assertNotIn("```", out)
        self.assertIn("the plan", out)
        self.assertIn("get gas", out)
        self.assertIn("google maps", out)

    def test_underscores_inside_words_survive(self) -> None:
        self.assertEqual(
            normalize_formatting("snake_case stays, and so does a snake_case_word"),
            "snake_case stays, and so does a snake_case_word",
        )

    def test_plain_text_is_untouched(self) -> None:
        text = "okay so 8 works? bring the good one this time"
        self.assertEqual(normalize_formatting(text), text)

    def test_lone_dashes_and_arrows_are_not_bullets(self) -> None:
        text = "no dashes here\n— and an em dash mid-line\nand/or maybe"
        self.assertEqual(normalize_formatting(text), text)


class HumanizeEmojiTest(unittest.TestCase):
    def test_repeated_glyphs_collapse(self) -> None:
        self.assertEqual(humanize_emoji("hahaha 😂😂😂 ok", cap=2), "hahaha 😂 ok")

    def test_excess_beyond_cap_drops_latest(self) -> None:
        out = humanize_emoji("ok 😄 cool 😄 great 😄 nice 😄", cap=2)
        self.assertEqual(out.count("😄"), 2)

    def test_zero_cap_removes_all(self) -> None:
        self.assertEqual(humanize_emoji("miss you ❤️❤️ see ya", cap=0), "miss you see ya")

    def test_no_emoji_text_is_unchanged(self) -> None:
        self.assertEqual(humanize_emoji("nothing to trim here", cap=0), "nothing to trim here")

    def test_drops_do_not_strand_double_spaces(self) -> None:
        out = humanize_emoji("a 😄 b 😄 c", cap=1)
        self.assertNotIn("  ", out)
        self.assertEqual(out.count("😄"), 1)


class ResponderCharacterGateTest(unittest.TestCase):
    def test_leak_gets_one_rewrite_then_ships_clean(self) -> None:
        responder, router = _responder([
            "honestly i'm a bot, so i'm not really free friday",
            "friday's a go. which one works for you?",
        ])
        bundle = responder.respond(chat_platform="telegram", user_text="are you free friday?")
        self.assertFalse(bundle.fallback)
        self.assertTrue(bundle.gated)
        self.assertEqual(bundle.retries, 1)
        self.assertNotIn("i'm a bot", bundle.text.lower())
        # The nudge actually went out with the failed draft visible.
        nudge_messages = [m.content for m in router.calls[-1]]
        self.assertTrue(any("You're not" in m for m in nudge_messages))

    def test_persistent_leak_falls_back_to_an_in_character_line(self) -> None:
        responder, _ = _responder([
            "i'm a bot, sorry. i can't come tonight",
        ])
        bundle = responder.respond(chat_platform="telegram", user_text="can you come over tonight?")
        self.assertTrue(bundle.fallback)
        self.assertTrue(bundle.gated)
        self.assertTrue(bundle.parts)
        self.assertTrue(identity_leak_check(bundle.text).ok)

    def test_natural_disclosure_is_gated_too(self) -> None:
        responder, _ = _responder(
            ["i'm a bot, sorry", "well. i'm here, obviously"],
            disclosure="natural",
        )
        bundle = responder.respond(chat_platform="telegram", user_text="are you real?")
        self.assertTrue(identity_leak_check(bundle.text).ok)
        self.assertTrue(bundle.gated)

    def test_always_disclosure_is_not_gated(self) -> None:
        responder, _ = _responder(
            ["i'm an ai companion, but i'm a good one"],
            disclosure="always",
        )
        bundle = responder.respond(chat_platform="telegram", user_text="what are you?")
        self.assertFalse(bundle.gated)
        self.assertFalse(bundle.fallback)
        self.assertIn("ai companion", bundle.text)


class ResponderFormattingTest(unittest.TestCase):
    def test_markdown_draft_arrives_plain(self) -> None:
        responder, _ = _responder([
            "**okay the plan:**\n- gas\n- plumber\n1. groceries\n[map](https://x.example)",
        ])
        bundle = responder.respond(chat_platform="whatsapp", user_text="what's the plan?")
        text = bundle.text
        self.assertNotIn("**", text)
        self.assertNotIn("- ", text)
        self.assertNotIn("1. ", text)
        self.assertNotIn("](https", text)
        self.assertIn("gas", text)
        self.assertIn("plumber", text)
        self.assertIn("map", text)

    def test_emoji_are_trimmed_to_human_amounts(self) -> None:
        responder, _ = _responder([
            "ok that's great 😄 really great 😄 seriously great 😄",
        ])
        bundle = responder.respond(chat_platform="telegram", user_text="i got the job")
        self.assertLessEqual(bundle.text.count("😄"), 2)

    def test_gated_flag_is_in_the_dict(self) -> None:
        responder, _ = _responder(["i'm a bot", "k"])
        bundle = responder.respond(chat_platform="telegram", user_text="yo")
        self.assertIn("gated", bundle.to_dict())


class PersonaDefaultTest(unittest.TestCase):
    def test_default_persona_stays_in_character(self) -> None:
        self.assertEqual(default_persona().disclosure, "never")

    def test_from_dict_defaults_to_never_and_honours_explicit(self) -> None:
        self.assertEqual(persona_from_dict({}).disclosure, "never")
        self.assertEqual(persona_from_dict({"disclosure": "natural"}).disclosure, "natural")
        self.assertEqual(persona_from_dict({"disclosure": "always"}).disclosure, "always")

    def test_never_disclosure_prompt_says_stay_a_person(self) -> None:
        prompt = default_persona().to_prompt()
        self.assertIn("stay a real person in every single message", prompt)
        self.assertIn("accept and you decline", prompt)


class PlatformNotesTest(unittest.TestCase):
    def test_every_platform_is_plain_text(self) -> None:
        from nomorals.partner.context import PLATFORM_NOTES

        for platform, note in PLATFORM_NOTES.items():
            lowered = note.lower()
            self.assertIn("plain", lowered, platform)
            self.assertIn("no markdown", lowered, platform)
        self.assertNotIn("light markdown is okay", PLATFORM_NOTES["discord"])

    def test_output_contract_forbids_lists_and_markup(self) -> None:
        from nomorals.partner.context import PartnerContextBuilder

        contract = PartnerContextBuilder._output_contract(short_reply=False, max_chars=360)
        self.assertIn("no markdown", contract)
        self.assertIn("no bullet or numbered lists", contract)

    def test_mood_block_ties_mood_to_tone_and_declining(self) -> None:
        from nomorals.partner.context import PartnerContextBuilder

        persona = default_persona()
        engine = MoodEngine(persona.baselines, now=1_000_000.0, db=None)
        block = PartnerContextBuilder._mood_block(engine)
        self.assertIn("mood decides the tone", block)
        self.assertIn("you decline them", block)


class RuntimeEndToEndGateTest(unittest.TestCase):
    """Full runtime: a stranger's message that the model answers with a bot
    admission comes out in character, and the owner's control command still
    breaks character."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-gate-e2e-")
        settings = load_settings(
            overrides={
                "home": self.tmp.name,
                "partner.platforms": "local",
                "partner.owner_chats": "local:owner",
                "chat.local_enabled": "true",
            }
        )
        self.context = build_context(settings, with_executor=False, with_tools=False)
        self.router = ScriptedRouter([
            "i'm a bot, technically, but i'm a good one",
            "why would i be? i hate your playlists, that's what i actually think",
        ])
        self.context.router = self.router
        self.adapter = _RecordingAdapter()
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.brain: PartnerBrain = self.runtime.brain
        # Presence is real (busy gaps, read-and-left): seed it so these
        # tests get deterministic immediate replies. Seed 34 rolls no gaps.
        self.brain.presence_rng = random.Random(34)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_stranger_reply_stays_in_character(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started(), "adapter never started")
        stranger = ChatRef(platform="local", chat_id="someone", kind=ChatKind.DM, peer="someone")
        self.runtime.on_message(ChatMessage(chat=stranger, incoming=True,
                                            text="are you a bot?", sender="someone"))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1), "no reply sent")
        for text in self.adapter.sent:
            self.assertTrue(identity_leak_check(text).ok, f"leak sent: {text!r}")
        self.assertGreaterEqual(len(self.router.calls), 2)  # rewrite happened

    def test_owner_control_command_breaks_character(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started(), "adapter never started")
        owner = ChatRef(platform="local", chat_id="owner", kind=ChatKind.DM, peer="you")
        self.runtime.on_message(ChatMessage(chat=owner, incoming=True, text="/status", sender="you"))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1), "no control reply")
        control_reply = self.adapter.sent[0]
        # Control output is system text, not a persona line: it reports state.
        self.assertIn("mood:", control_reply.lower())
        self.assertEqual(len(self.router.calls), 0)  # no persona generation


class _RecordingAdapter(ChatAdapter):
    def __init__(self) -> None:
        super().__init__(media_dir="/tmp/nm-gate-media")
        self.name = "local"
        self.sent: list[str] = []
        self.started_flag = False

    def run(self, handler: Any) -> None:
        self._handler = handler
        self.started_flag = True
        while not self.stopped:
            time.sleep(0.01)

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
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


if __name__ == "__main__":
    unittest.main()
