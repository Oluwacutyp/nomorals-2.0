"""Tests for nomorals.media.contentops.niches — niche plugin system.

All LLM calls are mocked; these tests exercise the templates, the
registry, the scaffold, and validation — never the real brain.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.media.contentops.niches import (
    NicheError,
    NichePlugin,
    ScriptResult,
    VisualPlan,
    VoiceSpec,
    all_plugins,
    get,
    get_niche,
    list_niches,
    register,
    scaffold,
    validate,
)
from nomorals.media.contentops.niches import registry as registry_mod

SAMPLE_TOPICS = {
    "motivation": "discipline on the days you feel nothing",
    "finance_facts": "why compound interest beats timing the market",
    "horror_stories": "the apartment with the extra room",
    "reddit_stories": "my neighbor built a fence through my garden",
    "did_you_know": "octopuses have three hearts",
    "sports_edits": "the boxer nobody wanted to train",
    "anime_edits": "the swordsman who swore an oath",
}

EXPECTED = sorted(SAMPLE_TOPICS)


class FakeLLMResponse:
    def __init__(self, text: str = "", error: str = ""):
        self.text = text
        self.error = error


class FakeBrain:
    """Mock brain: returns canned script text, records the prompt."""

    def __init__(self, text: str = "Hook line here. Beat two here. Payoff line here."):
        self.text = text
        self.last_prompt = ""

    def complete(self, prompt: str):
        self.last_prompt = prompt
        return FakeLLMResponse(text=self.text)


class BrokenBrain:
    def complete(self, prompt: str):
        raise RuntimeError("provider down")


# ── registry ─────────────────────────────────────────────────────────────

class RegistryTest(unittest.TestCase):
    def test_all_seven_shipped(self):
        self.assertEqual(list_niches(), EXPECTED)

    def test_get_round_trip(self):
        for name in EXPECTED:
            plugin = get(name)
            self.assertIsInstance(plugin, NichePlugin)
            self.assertEqual(plugin.name, name)

    def test_get_unknown_raises(self):
        with self.assertRaises(NicheError):
            get("no_such_niche_xyz")

    def test_get_niche_compat_known(self):
        plugin = get_niche("motivation")
        self.assertEqual(plugin.name, "motivation")

    def test_get_niche_compat_unknown_falls_back(self):
        # pipeline contract: unknown names get the generic shim, never a raise
        plugin = get_niche("quantum_gardening")
        self.assertEqual(plugin.name, "quantum_gardening")
        self.assertTrue(plugin.script_prompt("tomatoes"))

    def test_all_plugins_sorted(self):
        self.assertEqual([p.name for p in all_plugins()], EXPECTED)

    def test_duplicate_registration_rejected(self):
        from nomorals.media.contentops.niches.motivation import MotivationNiche
        with self.assertRaises(NicheError):
            register(MotivationNiche())  # same name, different instance

    def test_same_instance_reregister_is_idempotent(self):
        plugin = get("motivation")
        self.assertIs(register(plugin), plugin)

    def test_register_returns_plugin(self):
        class _Tmp(NichePlugin):
            name = "tmp_ok_niche"
            thesis = "t"
            cadence = 1.0
            title_template = "{topic} t"
            description_template = "{topic} d"
            hashtags = ["#t"]
            voice_spec = VoiceSpec()
            ypp_rationale = "r"
            def script_prompt(self, topic):
                return "x" * 300 + topic
            def visual_strategy(self, script):
                return VisualPlan(style_lock="s")
        try:
            out = register(_Tmp())
            self.assertEqual(out.name, "tmp_ok_niche")
            self.assertIn("tmp_ok_niche", list_niches())
        finally:
            registry_mod._REGISTRY.pop("tmp_ok_niche", None)


# ── every shipped niche ──────────────────────────────────────────────────

class ShippedNichesTest(unittest.TestCase):
    def test_every_niche_produces_full_package(self):
        for name, topic in SAMPLE_TOPICS.items():
            with self.subTest(niche=name):
                plugin = get(name)

                prompt = plugin.script_prompt(topic)
                self.assertGreaterEqual(len(prompt.strip()), 200,
                                        "script prompt must be substantive")
                self.assertIn(topic, prompt, "prompt must embed the topic")

                plan = plugin.visual_strategy(
                    "First beat of the script.\n\nSecond beat builds.\n\nPayoff beat lands.")
                self.assertIsInstance(plan, VisualPlan)
                self.assertGreaterEqual(len(plan.scenes), 1)
                self.assertTrue(plan.style_lock.strip(), "style lock required")
                prompts = plan.render_prompts()
                self.assertEqual(len(prompts), len(plan.scenes))
                for p in prompts:
                    self.assertTrue(p.strip())
                    self.assertIn(plan.style_lock, p,
                                  "every prompt must carry the style lock")

                title = plugin.title_for(topic)
                self.assertTrue(title.strip())
                desc = plugin.description_for(topic, script="sample script")
                self.assertTrue(desc.strip())
                self.assertGreaterEqual(len(plugin.tags_for("tiktok")), 1)
                self.assertGreaterEqual(len(plugin.tags_for("youtube")), 1)
                for tag in plugin.tags_for("tiktok"):
                    self.assertTrue(tag.startswith("#"))

                self.assertGreater(plugin.cadence, 0)
                self.assertEqual(plugin.cadence_spec()["posts_per_day"],
                                 float(plugin.cadence))
                self.assertTrue(plugin.ypp_rationale.strip())
                self.assertTrue(plugin.thesis.strip())

                # voice spec resolves without raising, even with no catalogue
                voice = plugin.voice_spec.resolve(catalogue=None)
                self.assertIsInstance(voice, str)
                self.assertTrue(str(plugin.voice_spec))

                lo, hi = plugin.word_budget()
                self.assertLess(lo, hi)
                self.assertGreater(lo, 0)

    def test_retention_grammar_present(self):
        """Each script prompt must demand hook + payoff beat + spoken-only."""
        payoff_words = ("payoff", "power-up", "climax", "glory", "twist", "reveal",
                        "cliffhanger")
        for name, topic in SAMPLE_TOPICS.items():
            with self.subTest(niche=name):
                prompt = plugin_script_prompt(name, topic).lower()
                self.assertIn("hook", prompt)
                self.assertTrue(any(w in prompt for w in payoff_words),
                                f"{name}: prompt has no payoff beat")
                self.assertIn("spoken", prompt)

    def test_original_visuals_only_for_edits_niches(self):
        for name in ("sports_edits", "anime_edits"):
            plugin = get(name)
            plan = plugin.visual_strategy("beat one.\n\nbeat two.")
            blob = " ".join(plan.render_prompts()).lower()
            self.assertIn("fictional" if name == "sports_edits" else "original",
                          blob)
            self.assertNotIn("broadcast", blob)


def plugin_script_prompt(name: str, topic: str) -> str:
    return get(name).script_prompt(topic)


# ── scaffold ─────────────────────────────────────────────────────────────

class ScaffoldTest(unittest.TestCase):
    def _import_scaffolded(self, tmpdir: str, name: str):
        path = scaffold(name, dest_dir=tmpdir)
        self.assertTrue(Path(path).exists())
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        return module

    def test_scaffold_creates_importable_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = self._import_scaffolded(tmpdir, "cooking_niche")
            cls = module.CookingNicheNiche
            plugin = cls()
            self.assertEqual(validate(plugin), [])
            self.assertTrue(plugin.script_prompt("eggs"))
            plan = plugin.visual_strategy("beat one.\n\nbeat two.")
            self.assertGreaterEqual(len(plan.scenes), 1)
            self.assertIn("cooking_niche", list_niches())
            registry_mod._REGISTRY.pop("cooking_niche", None)

    def test_scaffold_rejects_bad_name(self):
        with self.assertRaises(ValueError):
            scaffold("Bad Name!")

    def test_scaffold_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold("dup_niche", dest_dir=tmpdir)
            with self.assertRaises(FileExistsError):
                scaffold("dup_niche", dest_dir=tmpdir)


# ── validation ───────────────────────────────────────────────────────────

class ValidationTest(unittest.TestCase):
    def _base_kwargs(self, **over):
        kw = dict(
            name="valid_niche",
            thesis="t",
            cadence=1.0,
            title_template="{topic} title",
            description_template="{topic} desc",
            hashtags=["#tag"],
            voice_spec=VoiceSpec(),
            ypp_rationale="original content",
        )
        kw.update(over)
        return kw

    def _make(self, **over):
        kw = self._base_kwargs(**over)

        class _P(NichePlugin):
            def script_prompt(self, topic):
                return "x" * 300 + topic

            def visual_strategy(self, script):
                return VisualPlan(style_lock="s")

        for k, v in kw.items():
            setattr(_P, k, v)
        return _P()

    def test_valid_plugin_passes(self):
        self.assertEqual(validate(self._make()), [])

    def test_rejects_not_a_plugin(self):
        self.assertTrue(validate(object()))

    def test_rejects_bad_name(self):
        self.assertTrue(validate(self._make(name="")))
        self.assertTrue(validate(self._make(name="Bad Name")))

    def test_rejects_zero_cadence(self):
        self.assertTrue(validate(self._make(cadence=0)))

    def test_rejects_missing_topic_placeholder(self):
        self.assertTrue(validate(self._make(title_template="no placeholder")))

    def test_rejects_empty_hashtags(self):
        self.assertTrue(validate(self._make(hashtags=[])))

    def test_rejects_hashtag_without_hash(self):
        self.assertTrue(validate(self._make(hashtags=["notag"])))

    def test_rejects_empty_ypp(self):
        self.assertTrue(validate(self._make(ypp_rationale="  ")))

    def test_rejects_thin_prompt(self):
        class _Thin(NichePlugin):
            name = "thin"
            thesis = "t"
            cadence = 1.0
            title_template = "{topic}"
            description_template = "{topic}"
            hashtags = ["#t"]
            voice_spec = VoiceSpec()
            ypp_rationale = "r"

            def script_prompt(self, topic):
                return "too short"

            def visual_strategy(self, script):
                return VisualPlan(style_lock="s")

        self.assertTrue(validate(_Thin()))

    def test_rejects_raising_prompt(self):
        class _Boom(NichePlugin):
            name = "boom"
            thesis = "t"
            cadence = 1.0
            title_template = "{topic}"
            description_template = "{topic}"
            hashtags = ["#t"]
            voice_spec = VoiceSpec()
            ypp_rationale = "r"

            def script_prompt(self, topic):
                raise RuntimeError("llm exploded")

            def visual_strategy(self, script):
                return VisualPlan(style_lock="s")

        issues = validate(_Boom())
        self.assertTrue(any("script_prompt" in i for i in issues))

    def test_register_raises_on_malformed(self):
        with self.assertRaises(NicheError):
            register(self._make(name=""))


# ── generate_script with mocked brain ────────────────────────────────────

class GenerateScriptTest(unittest.TestCase):
    def test_mocked_brain_produces_result(self):
        brain = FakeBrain("one two three four five six")
        result = get("motivation").generate_script(brain, "discipline")
        self.assertIsInstance(result, ScriptResult)
        self.assertEqual(result.word_count, 6)
        self.assertGreater(result.est_seconds, 0)
        self.assertEqual(result.topic, "discipline")
        self.assertEqual(result.niche, "motivation")
        # the niche's retention prompt actually reached the brain
        self.assertIn("discipline", brain.last_prompt.lower())
        self.assertIn("hook", brain.last_prompt.lower())

    def test_brain_failure_never_raises(self):
        result = get("horror_stories").generate_script(BrokenBrain(), "topic")
        self.assertEqual(result.text, "")
        self.assertEqual(result.word_count, 0)

    def test_fence_stripped(self):
        brain = FakeBrain("```\nreal script words here\n```")
        result = get("did_you_know").generate_script(brain, "t")
        self.assertEqual(result.text, "real script words here")


# ── plan primitives ──────────────────────────────────────────────────────

class PlanPrimitivesTest(unittest.TestCase):
    def test_visual_plan_dict_and_total(self):
        plan = get("finance_facts").visual_strategy("a.\n\nb.\n\nc.")
        d = plan.to_dict()
        self.assertEqual(d["aspect"], "9:16")
        self.assertEqual(len(d["scenes"]), 3)
        self.assertGreater(plan.total_seconds(), 0)

    def test_beats_cycle_when_script_longer_than_briefs(self):
        plan = get("motivation").visual_strategy("\n\n".join(f"beat {i}" for i in range(10)))
        self.assertEqual(len(plan.scenes), 10)

    def test_empty_script_still_yields_scene(self):
        plan = get("anime_edits").visual_strategy("   ")
        self.assertGreaterEqual(len(plan.scenes), 1)

    def test_voice_spec_resolve_prefers_catalogue_hit(self):
        fake_catalogue = mock.Mock()
        fake_catalogue.voices = {"devon-deep": mock.Mock(), "default": mock.Mock()}
        spec = VoiceSpec(preferred_voice="devon-deep", fallback_voice="default")
        self.assertEqual(spec.resolve(fake_catalogue), "devon-deep")

    def test_voice_spec_resolve_falls_back_gracefully(self):
        spec = VoiceSpec(preferred_voice="missing", fallback_voice="also-missing")
        self.assertEqual(spec.resolve(mock.Mock(voices={})), "")
        self.assertEqual(spec.resolve(None), "")


if __name__ == "__main__":
    unittest.main()
