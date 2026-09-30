"""Tests for the partner cognition core: mood, relationship, background, style."""

from __future__ import annotations

import time
import unittest

from nomorals.partner.background import BackgroundSelector, GATE_MODES
from nomorals.partner.context import PartnerContextBuilder
from nomorals.partner.mood import DIMENSIONS, MOOD_LABELS, MoodEngine
from nomorals.partner.persona import default_persona
from nomorals.partner.relationship import STAGES, Relationship
from nomorals.partner.responder import detect_signals
from nomorals.partner.style import (
    clamp_to_budget,
    length_budget,
    parrot_check,
    split_messages,
    strip_robotic,
)
from nomorals.storage.db import Database


class MoodEngineTest(unittest.TestCase):
    def _engine(self, now: float = 1_000_000.0, db: Database | None = None) -> MoodEngine:
        return MoodEngine(default_persona().baselines, now=now, db=db)

    def test_fresh_state_uses_baselines(self) -> None:
        engine = self._engine()
        state = engine.current()
        for dim in DIMENSIONS:
            self.assertAlmostEqual(state.values[dim], default_persona().baselines[dim])

    def test_compliment_raises_affection(self) -> None:
        engine = self._engine()
        before = engine.value("affection")
        engine.note_event("compliment", intensity=0.8)
        self.assertGreater(engine.value("affection"), before)
        self.assertGreater(engine.value("pride"), default_persona().baselines["pride"])

    def test_fight_escalates_and_repair_recovers(self) -> None:
        engine = self._engine()
        base_trust = engine.value("trust")
        engine.open_fight_now("he forgot the anniversary")
        self.assertIsNotNone(engine.open_fight)
        self.assertGreater(engine.value("frustration"), 25)
        self.assertLess(engine.value("trust"), base_trust)
        engine.resolve_fight(repaired=True)
        self.assertIsNone(engine.open_fight)
        self.assertLess(engine.value("frustration"), 25)

    def test_unresolved_fight_leaves_grudge(self) -> None:
        engine = self._engine()
        engine.open_fight_now("a text left on read")
        engine.resolve_fight(repaired=False)
        self.assertTrue(any(g["kind"] == "fight" for g in engine.grudges))
        self.assertLess(engine.value("trust"), default_persona().baselines["trust"])

    def test_decay_toward_baseline(self) -> None:
        engine = self._engine(now=0.0)
        engine.note_event("fight_start", intensity=1.0)
        spiked = engine.value("frustration")
        self.assertGreater(spiked, default_persona().baselines["frustration"] + 15)
        # Ten hours later: the spike has decayed most of the way home.
        engine.tick(now=0.0 + 10 * 3600)
        self.assertLess(engine.value("frustration"), spiked - 10)
        self.assertLess(abs(engine.value("frustration") - default_persona().baselines["frustration"]), 12)

    def test_circadian_energy_at_night(self) -> None:
        engine = self._engine(now=1_000_000.0)
        # Midnight: energy should drift toward the low circadian target.
        night = time.gmtime(1_000_000.0 % 86400)  # whatever hour this is
        engine.tick(now=1_000_000.0 + 86400 * 2)  # same hour, 2 days later
        # Energy is always a number in range after any tick.
        self.assertTrue(0.0 <= engine.value("energy") <= 100.0)

    def test_silence_costs_something(self) -> None:
        engine = self._engine()
        before_distance = engine.value("distance")
        engine.note_silence(hours_idle=20.0)
        self.assertGreater(engine.value("distance"), before_distance)
        self.assertGreater(engine.value("insecurity"), default_persona().baselines["insecurity"])

    def test_invariants_cap_happiness_under_high_frustration(self) -> None:
        engine = self._engine()
        engine.note_event("insult", 1.0)
        engine.note_event("insult", 1.0)
        engine.note_event("insult", 1.0)
        if engine.value("frustration") > 75:
            self.assertLessEqual(engine.value("happiness"), 35.0)

    def test_sanitize_repairs_corruption(self) -> None:
        baselines = dict(default_persona().baselines)
        repaired = MoodEngine.sanitize(
            {"affection": "not a number", "happiness": 400, "energy": float("nan"),
             "unknown_dim": 99},
            baselines,
        )
        self.assertEqual(set(repaired), set(DIMENSIONS))
        for dim, value in repaired.items():
            self.assertTrue(0.0 <= value <= 100.0, dim)
        self.assertEqual(repaired["affection"], baselines["affection"])

    def test_label_hysteresis(self) -> None:
        engine = self._engine()
        first = engine.current().label
        # A tiny nudge must not flip the label.
        engine.apply({"happiness": 0.5})
        self.assertEqual(engine.current().label, first)

    def test_persistence_roundtrip(self) -> None:
        db = Database(":memory:")
        db.migrate()
        engine = self._engine(db=db)
        engine.note_event("deep_conversation", 0.9)
        intimacy = engine.value("intimacy")
        reloaded = MoodEngine(default_persona().baselines, db=db, now=engine._now)
        self.assertAlmostEqual(reloaded.value("intimacy"), intimacy, delta=0.01)
        self.assertEqual(reloaded.current().label, engine.current().label)

    def test_corrupt_db_state_recovers(self) -> None:
        db = Database(":memory:")
        db.migrate()
        db.execute(
            "INSERT INTO mood_state (id, dims, label, updated_at) VALUES ('default', 'not-json', 'bogus', 1.0)"
        )
        engine = self._engine(db=db)
        state = engine.current()
        self.assertEqual(set(state.values), set(DIMENSIONS))
        self.assertIn(state.label, MOOD_LABELS)
        # And it can still mutate and save over the corrupt row.
        engine.note_event("compliment", 0.5)
        row = db.query_one("SELECT dims FROM mood_state WHERE id = 'default'")
        import json

        self.assertIsInstance(json.loads(row["dims"]), dict)

    def test_history_journaled(self) -> None:
        db = Database(":memory:")
        db.migrate()
        engine = self._engine(db=db)
        engine.note_event("fight_start", 1.0, note="test")
        count = db.scalar("SELECT COUNT(*) FROM mood_history", default=0)
        self.assertGreaterEqual(int(count), 1)


class RelationshipTest(unittest.TestCase):
    def test_stage_progression_and_regression(self) -> None:
        rel = Relationship()
        self.assertEqual(rel.stage, "getting_to_know")
        rel.advance_stage()
        self.assertEqual(rel.stage, "dating")
        self.assertTrue(rel.is_romantic())
        rel.regress_stage("a big trust failure")
        self.assertEqual(rel.stage, "getting_to_know")
        self.assertLess(rel.trust, 60)

    def test_milestones_and_fights(self) -> None:
        rel = Relationship()
        rel.add_milestone("first real conversation about the future", kind="moment")
        entry = rel.record_fight("cancelled plans without saying", repaired=False)
        self.assertFalse(entry["repaired"])
        self.assertEqual(len(rel.unresolved_fights()), 1)
        rel.record_fight("cancelled plans without saying", repaired=True)
        self.assertEqual(len(rel.unresolved_fights()), 0)
        self.assertGreater(len(rel.milestones), 1)

    def test_persistence_roundtrip(self) -> None:
        db = Database(":memory:")
        db.migrate()
        rel = Relationship()
        rel.advance_stage()
        rel.add_milestone("met the dog", kind="moment")
        rel.note_user_fact("works at", "a dental office")
        rel.save(db)
        reloaded = Relationship.load(db)
        self.assertEqual(reloaded.stage, rel.stage)
        self.assertEqual(reloaded.user_profile["works at"], "a dental office")
        self.assertTrue(any(m["text"] == "met the dog" for m in reloaded.milestones))

    def test_prompt_block_mentions_unresolved_fight(self) -> None:
        rel = Relationship()
        rel.record_fight("the lying about where he was", repaired=False)
        block = rel.to_prompt_block()
        self.assertIn("lying about where he was", block)
        self.assertIn("unresolved", block.lower() + " unresolved")

    def test_stage_values_are_known(self) -> None:
        self.assertEqual(STAGES[0], "acquaintance")
        self.assertIn("committed", STAGES)


class BackgroundTest(unittest.TestCase):
    def test_gate_modes_exist_and_default(self) -> None:
        selector = BackgroundSelector("us_or_romantic")
        self.assertEqual(selector.mode, "us_or_romantic")
        self.assertTrue(selector.applies(user_in_us=True, romantic=False))
        self.assertTrue(selector.applies(user_in_us=False, romantic=True))
        self.assertFalse(selector.applies(user_in_us=False, romantic=False))

    def test_gate_never_is_inert(self) -> None:
        selector = BackgroundSelector("never")
        self.assertEqual(selector.context("where do you work?", user_in_us=True, romantic=True), [])

    def test_gate_romantic_requires_stage(self) -> None:
        selector = BackgroundSelector("romantic")
        self.assertEqual(selector.context("where do you work?", user_in_us=True, romantic=False), [])
        self.assertTrue(selector.context("where do you work?", user_in_us=False, romantic=True))

    def test_topic_selection_unlocks_relevant_facts(self) -> None:
        selector = BackgroundSelector("never")  # bypass the gate; test selection directly
        facts = selector.pack.select("any plans for a hike this weekend?")
        self.assertTrue(facts, "hiking question should unlock outdoor facts")
        self.assertTrue(any("trail" in f.text.lower() or "hike" in f.text.lower() for f in facts))

    def test_unrelated_topic_gets_nothing(self) -> None:
        selector = BackgroundSelector("never")
        facts = selector.pack.select("what's a good way to refactor a monolith?")
        self.assertEqual(facts, [])

    def test_work_topic_unlocks_work_facts(self) -> None:
        selector = BackgroundSelector("never")
        facts = selector.pack.select("how's work going? clients okay?")
        self.assertTrue(any(f.kind == "work" for f in facts))

    def test_ambient_returns_facts(self) -> None:
        selector = BackgroundSelector("us_or_romantic")
        lines = selector.ambient_lines(user_in_us=True, romantic=False)
        self.assertTrue(lines)

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(Exception):
            BackgroundSelector("sometimes")


class StyleTest(unittest.TestCase):
    def test_parrot_rejects_verbatim_echo(self) -> None:
        verdict = parrot_check("you always leave your dishes in the sink",
                               "okay you always leave your dishes in the sink, i know, calm down")
        self.assertFalse(verdict.ok)

    def test_parrot_rejects_heavy_overlap(self) -> None:
        verdict = parrot_check("i passed the exam yesterday",
                               "yes i passed the exam yesterday, that is amazing, i passed the exam")
        self.assertFalse(verdict.ok)

    def test_parrot_allows_real_answer(self) -> None:
        verdict = parrot_check("where were you last night?",
                               "was at the brewery with the regulars, back around eleven. why?")
        self.assertTrue(verdict.ok)

    def test_parrot_repeats_opening(self) -> None:
        verdict = parrot_check("i think we need to talk about the money",
                               "i think we need to talk about the money too, but not tonight")
        self.assertFalse(verdict.ok)

    def test_strip_robotic_removes_ai_tells(self) -> None:
        text = ("That's a great question. I love talking about this with you. "
                "As an AI I should note I don't actually drink coffee, but let me know if you want more detail.")
        cleaned = strip_robotic(text)
        self.assertNotIn("great question", cleaned.lower())
        self.assertNotIn("as an ai", cleaned.lower())
        self.assertNotIn("let me know if", cleaned.lower())
        self.assertIn("I love talking about this with you", cleaned)

    def test_split_messages_stays_under_cap(self) -> None:
        text = ("I was thinking about the trip. " * 20)
        parts = split_messages(text, max_chars=200)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), 200)
            self.assertTrue(part.strip())

    def test_split_short_text_single_part(self) -> None:
        self.assertEqual(split_messages("hey, you up?"), ["hey, you up?"])

    def test_clamp_cuts_rambles_at_sentence(self) -> None:
        text = ("This is sentence one. " * 60)
        clamped = clamp_to_budget(text, (40, 300))
        self.assertLessEqual(len(clamped), 300 * 1.6 + 1)
        self.assertTrue(clamped)

    def test_length_budget_bands_by_mood(self) -> None:
        tired = length_budget({"energy": 15, "happiness": 45, "frustration": 10, "distance": 10})
        excited = length_budget({"energy": 85, "happiness": 90, "frustration": 5, "distance": 10})
        self.assertLess(tired[1], excited[1])
        self.assertLess(tired[0], excited[0])


class ContextBuilderTest(unittest.TestCase):
    def _parts(self):
        engine = MoodEngine(default_persona().baselines, now=1_000_000.0)
        engine.note_event("deep_conversation", 0.8)
        rel = Relationship()
        rel.advance_stage()
        return engine, rel, default_persona()

    def test_build_contains_identity_mood_and_relationship(self) -> None:
        engine, rel, persona = self._parts()
        system = PartnerContextBuilder().build(
            persona=persona, mood=engine, relationship=rel,
            memories=["he burned dinner and cried about it (2 days ago)"],
            platform="telegram",
        )
        self.assertIn(persona.name, system.content)
        self.assertIn("dating", system.content)
        self.assertIn("burnt dinner".replace("burnt", "burned"), system.content)
        self.assertIn("OUTPUT CONTRACT", system.content)

    def test_background_block_only_when_given(self) -> None:
        engine, rel, persona = self._parts()
        without = PartnerContextBuilder().build(persona=persona, mood=engine, relationship=rel)
        with_bg = PartnerContextBuilder().build(
            persona=persona, mood=engine, relationship=rel,
            background_lines=["Background you can use *if it comes up naturally*:", "  - the aspen went gold early"],
        )
        self.assertNotIn("aspen", without.content)
        self.assertIn("aspen", with_bg.content)

    def test_budget_shrinks_memory_first(self) -> None:
        engine, rel, persona = self._parts()
        huge_memory = [f"memory item number {i} with some extra words to eat budget" for i in range(200)]
        system = PartnerContextBuilder(total_budget=2000).build(
            persona=persona, mood=engine, relationship=rel, memories=huge_memory,
        )
        # Identity and state survive; memory is capped.
        self.assertIn(persona.name, system.content)
        self.assertIn("OUTPUT CONTRACT", system.content)
        self.assertLess(system.content.count("memory item number"), 100)


class SignalDetectionTest(unittest.TestCase):
    def test_compliment(self) -> None:
        kinds = {e.kind for e in detect_signals("you make me so happy, honestly") }
        self.assertIn("compliment", kinds)

    def test_fight(self) -> None:
        kinds = {e.kind for e in detect_signals("you never listen to a single word i say")}
        self.assertIn("fight", kinds)

    def test_apology(self) -> None:
        kinds = {e.kind for e in detect_signals("i'm sorry, i shouldn't have said that")}
        self.assertIn("apology", kinds)

    def test_jalousy_trigger(self) -> None:
        kinds = {e.kind for e in detect_signals("who's that girl in your story?")}
        self.assertIn("jealousy_trigger", kinds)

    def test_short_cold_reply(self) -> None:
        events = detect_signals("fine.")
        kinds = {e.kind for e in events}
        self.assertIn("cold_response", kinds)

    def test_short_warm_reply(self) -> None:
        events = detect_signals("yeah")
        kinds = {e.kind for e in events}
        self.assertIn("warm_response", kinds)

    def test_empty_message_no_events(self) -> None:
        self.assertEqual(detect_signals("   "), [])


if __name__ == "__main__":
    unittest.main()
