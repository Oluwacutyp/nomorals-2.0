"""Mined style system: dynamic per-chat style, no hardcoded tone presets.

Every assertion below cites its mining source (see STYLE_MINING.md):

* Markers / structure — Devon's OWN output code:
  ``nomorals/social/chat/style.py`` (emoji section headers, <code>
  commands, 🎯 CTA, ━━━ dividers), ``tgbot_buttons.py`` (✅/❌
  confirm/cancel, 🎮 games, 📊 stats, 📋 lists), ``action_footers.py``.
* Platform caps — ``nomorals/social/chat/platform_limits.py``
  (TG 4096, WA 4000, Discord 2000, SMS 160) + Telegram Bot API
  formatting docs (core.telegram.org/bots/api).
* Audience rules — ``nomorals/partner/gating.py`` (group: "no
  paragraphs, no lists, no explanations"; private: short + guarded).
* Observed measurements — the chat's own messages; the fixtures here
  are synthetic stand-ins proving the measurement math, NOT owner
  voice. No real owner messages exist in this repo, and none are
  claimed: with no observed sample the profile falls back to
  audience + platform defaults.

Honest gaps (also in STYLE_MINING.md): the first-edition repo
(Oluwacutyp/No-morals-ai) was unreachable read-only (GitHub API 404
unauthenticated), so no first-edition conventions were copied.
"""

from __future__ import annotations

import random
import re
import unittest
from pathlib import Path

from nomorals.llm.base import LLMResponse
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import default_persona
from nomorals.partner.relationship import Relationship
from nomorals.partner.responder import PartnerResponder
from nomorals.partner.style import (
    MINED_SECTION_MARKERS,
    VOICE_THEMES,
    apply_voice_theme,
    format_reply,
    measure_style,
    platform_hard_cap,
    purposeful_emoji,
    style_for_chat,
    style_prompt_lines,
)

_MOOD = {"energy": 60, "frustration": 10, "happiness": 60, "distance": 30}


class NoTonePresetsTest(unittest.TestCase):
    """The old VOICE_THEMES preset table is gone, not renamed."""

    def test_preset_table_is_empty(self) -> None:
        self.assertEqual(VOICE_THEMES, {})

    def test_apply_voice_theme_is_passthrough(self) -> None:
        speech = default_persona().speech
        for name in ("dry", "soft", "feral", "poetic", "sharp", "kind", "tough"):
            self.assertIs(apply_voice_theme(speech, name), speech)

    def test_no_preset_keys_in_style_module(self) -> None:
        src = Path(__file__).resolve().parent.parent.joinpath(
            "nomorals/partner/style.py"
        ).read_text()
        for key in ("dry", "soft", "feral", "poetic", "sharp", "kind", "tough"):
            self.assertNotRegex(src, rf'"{key}":\s*\{{', f"preset key {key!r} present")


class MeasureStyleTest(unittest.TestCase):
    """Measurement math over observed messages (synthetic fixtures)."""

    def test_empty_sample_is_zero(self) -> None:
        m = measure_style([])
        self.assertEqual(m.n, 0)
        self.assertEqual(m.emoji_per_msg, 0.0)

    def test_median_and_rates(self) -> None:
        texts = [
            "hi",  # 2 chars
            "ok cool, see you there?",  # question
            "- buy milk\n- call mum",  # list
            "wow! amazing! 🎉",  # exclaim + emoji
            "plain message here",
        ]
        m = measure_style(texts)
        self.assertEqual(m.n, 5)
        self.assertAlmostEqual(m.median_chars, 18.0)
        self.assertAlmostEqual(m.question_rate, 0.2)
        self.assertAlmostEqual(m.exclaim_rate, 0.2)
        self.assertAlmostEqual(m.list_rate, 0.2)
        self.assertAlmostEqual(m.emoji_per_msg, 0.2)

    def test_never_raises_on_junk(self) -> None:
        self.assertEqual(measure_style(None).n, 0)
        self.assertEqual(measure_style(["", "   ", None]).n, 0)  # type: ignore[list-item]


class PerChatStyleTest(unittest.TestCase):
    """Audience + platform + observed history -> different profiles."""

    def test_owner_vs_group_differs(self) -> None:
        owner = style_for_chat(audience="owner", platform="telegram", mood_values=_MOOD)
        group = style_for_chat(audience="group", platform="telegram", mood_values=_MOOD)
        self.assertTrue(owner.max_chars > group.max_chars)
        self.assertFalse(group.allow_sections)
        self.assertFalse(group.allow_lists)
        self.assertLessEqual(group.emoji_budget, 1)

    def test_private_is_between(self) -> None:
        private = style_for_chat(audience="private", platform="telegram", mood_values=_MOOD)
        group = style_for_chat(audience="group", platform="telegram", mood_values=_MOOD)
        self.assertFalse(private.allow_sections)
        self.assertTrue(private.allow_lists)
        self.assertGreaterEqual(private.max_chars, group.max_chars)

    def test_platform_caps_follow_platform_limits(self) -> None:
        self.assertEqual(platform_hard_cap("telegram"), 4096)
        self.assertEqual(platform_hard_cap("whatsapp"), 4000)
        self.assertEqual(platform_hard_cap("discord"), 2000)
        self.assertEqual(platform_hard_cap("sms"), 160)
        sms = style_for_chat(audience="owner", platform="sms", mood_values=_MOOD)
        self.assertLessEqual(sms.max_chars, 160)
        self.assertFalse(sms.allow_markup)

    def test_observed_history_bends_target(self) -> None:
        terse = ["ok", "yeah", "k", "lol", "brb", "np", "sure thing"]
        verbose = [
            "So I was thinking about this at length and there are really "
            "several dimensions to consider here, starting with the first "
            "one which is honestly the most important of the lot, and then "
            "moving on to the rest in turn. " * 3
            for _ in range(6)
        ]
        base = style_for_chat(audience="owner", platform="telegram", mood_values=_MOOD)
        terse_p = style_for_chat(
            audience="owner", platform="telegram", mood_values=_MOOD, observed=terse
        )
        verbose_p = style_for_chat(
            audience="owner", platform="telegram", mood_values=_MOOD, observed=verbose
        )
        # Same audience/platform/mood, different histories -> different targets.
        self.assertNotEqual(terse_p.target_chars, verbose_p.target_chars)
        self.assertLess(terse_p.target_chars, base.target_chars)
        self.assertGreater(verbose_p.target_chars, terse_p.target_chars)

    def test_unknown_audience_falls_back_to_owner(self) -> None:
        p = style_for_chat(audience="mystery", platform="telegram", mood_values=_MOOD)
        self.assertEqual(p.audience, "owner")

    def test_prompt_lines_are_numbers_not_tone_words(self) -> None:
        lines = style_prompt_lines(
            style_for_chat(audience="group", platform="telegram", mood_values=_MOOD)
        )
        blob = " ".join(lines).lower()
        self.assertIn("characters", blob)
        for tone_word in ("kind", "sharp", "tough", "warm", "feral", "poetic"):
            self.assertNotIn(tone_word, blob)


class PurposefulEmojiTest(unittest.TestCase):
    """Emoji = purposeful markers, never spam (mined marker vocabulary)."""

    def test_markers_are_the_mined_vocabulary(self) -> None:
        # Every marker in the table is an emoji actually used in Devon's
        # own output code (see STYLE_MINING.md counts).
        self.assertEqual(MINED_SECTION_MARKERS["games"], "🎮")
        self.assertEqual(MINED_SECTION_MARKERS["done"], "✅")
        self.assertEqual(MINED_SECTION_MARKERS["failed"], "❌")
        self.assertEqual(MINED_SECTION_MARKERS["music"], "🎵")
        self.assertEqual(MINED_SECTION_MARKERS["next"], "🎯")
        self.assertEqual(MINED_SECTION_MARKERS["stats"], "📊")

    def test_repeated_runs_collapse(self) -> None:
        out = purposeful_emoji("so funny 😂😂😂😂", budget=2)
        self.assertEqual(out.count("😂"), 1)

    def test_budget_is_a_hard_cap(self) -> None:
        out = purposeful_emoji("✅ done ❌ failed 🎮 games 📊 stats", budget=1)
        clusters = re.findall(
            r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+", out
        )
        self.assertLessEqual(len(clusters), 1)

    def test_zero_budget_strips_all(self) -> None:
        self.assertNotIn("🎮", purposeful_emoji("🎮 games tonight", budget=0))

    def test_decorative_run_keeps_one(self) -> None:
        out = purposeful_emoji("🎉🎉🎉🎉 party time", budget=3)
        self.assertEqual(out.count("🎉"), 1)


class FormatReplyTest(unittest.TestCase):
    def test_group_flattens_lists_and_drops_sections(self) -> None:
        profile = style_for_chat(audience="group", platform="telegram", mood_values=_MOOD)
        text = "🎮 GAMES\n━━━━━━━━━━━━━━━\n- play mafia\n- play trivia\n- play hangman"
        out = format_reply(text, profile)
        self.assertNotIn("━━━", out)
        self.assertNotIn("- play", out)
        self.assertIn("play mafia", out)

    def test_owner_structured_keeps_sections(self) -> None:
        profile = style_for_chat(
            audience="owner", platform="telegram", mood_values=_MOOD,
            content_kind="structured",
        )
        text = "🎮 GAMES\n━━━━━━━━━━━━━━━\n- play mafia\n- play trivia"
        out = format_reply(text, profile)
        self.assertIn("━━━", out)
        self.assertIn("🎮", out)

    def test_truncation_respects_max_and_sentence_boundary(self) -> None:
        profile = style_for_chat(audience="group", platform="telegram", mood_values=_MOOD)
        long_text = "First sentence here. " * 60
        out = format_reply(long_text, profile)
        self.assertLessEqual(len(out), profile.max_chars)
        self.assertTrue(out.endswith(".") or out.endswith("…"))

    def test_never_raises_and_passthrough_empty(self) -> None:
        profile = style_for_chat(audience="owner", platform="telegram", mood_values=_MOOD)
        self.assertEqual(format_reply("", profile), "")
        self.assertIsInstance(format_reply("hello", profile), str)


class FakeRouter:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def chat(self, messages, params=None, **kw):  # noqa: ANN001, ANN201
        text = self.replies.pop(0) if self.replies else "mhm, okay"
        return LLMResponse(text=text, model="fake")


class ResponderWiringTest(unittest.TestCase):
    """The reply path actually uses the style system (not a library)."""

    def _responder(self, replies: list[str]) -> PartnerResponder:
        persona = default_persona()
        return PartnerResponder(
            FakeRouter(replies), persona, MoodEngine(persona.baselines),
            Relationship(), None, None, rng=random.Random(0),
        )

    def test_group_reply_has_no_lists_or_emoji_spam(self) -> None:
        draft = (
            "🎮 GAMES\n- play mafia\n- play trivia\n"
            "🎉🎉🎉 so much fun 🎉🎉 " + ("This is a long group ramble. " * 30)
        )
        r = self._responder([draft])
        bundle = r.respond(
            chat_platform="telegram", user_text="what games?",
            gate_mode="group",
        )
        text = " ".join(bundle.parts)
        self.assertNotIn("- play", text)
        self.assertLessEqual(text.count("🎉"), 1)
        self.assertLessEqual(len(text), 280 + 360)  # group cap + split slack

    def test_owner_reply_keeps_content(self) -> None:
        r = self._responder(["Done — the report is ready."])
        bundle = r.respond(
            chat_platform="telegram", user_text="is it done?",
            gate_mode="owner",
        )
        self.assertIn("report", " ".join(bundle.parts))


if __name__ == "__main__":
    unittest.main()
