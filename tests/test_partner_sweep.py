"""Sweep tests: partner module upgrades (mining-driven).

Covers the new behavior added in the partner sweep:
affect lexicon second pass, persona dialogue examples / greetings / Big Five /
continuity fingerprint, mood spikes + congruency + PAD, relationship axes +
repair attempts vs outcomes + activities + anniversaries, hard-layer texting
voice + voice themes, context post-history + shrink priorities, background
constant facts + recency, presence typing schedule + read delay, gating
first-contact frame, social-gate denial audit, Discord group roles, chat
profile vibe/activity, lexicon usage reinforcement + stale pruning, and the
responder wiring (bundle affect readout, proactive greetings).
"""

from __future__ import annotations

import random
import time
import unittest
from contextlib import nullcontext

from nomorals.partner import (
    AFFECT_EVENT_CAP,
    ARC_AXES,
    BIG_FIVE_TRAITS,
    DEFAULT_AXES,
    SHRINK_ORDER,
    TYPING_KEEPALIVE_S,
    VOICE_THEMES,
    AffectScorer,
    BackgroundSelector,
    DenialLog,
    DialogueExample,
    EmotionSpike,
    LexiconFeed,
    MoodEngine,
    PartnerContextBuilder,
    PartnerResponder,
    Persona,
    Relationship,
    TypingPlan,
    affect_to_events,
    apply_texting_voice,
    apply_voice_theme,
    build_context_lines,
    check_tool_call,
    default_persona,
    detect_signals,
    gate_block,
    grant_for,
    persona_from_dict,
    read_delay_seconds,
    refresh_profile,
    resolve_group_role,
    score_affect,
    typing_schedule,
)
from nomorals.partner.chat_profile import _activity_label, _vibe_label
from nomorals.partner.mood import MOOD_PAD


# ── affect ───────────────────────────────────────────────────────────────────

class AffectScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = AffectScorer()

    def test_joy_detection(self) -> None:
        r = self.scorer.score("i am so happy and thrilled today")
        self.assertIn(r.dominant, {"joy", "excitement"})
        self.assertGreater(r.valence, 0.5)

    def test_sadness_detection(self) -> None:
        r = self.scorer.score("i feel so sad and lonely tonight")
        self.assertIn(r.dominant, {"sadness", "loneliness"})
        self.assertLess(r.valence, -0.5)

    def test_negation_flips(self) -> None:
        r = self.scorer.score("i am not happy about this at all")
        self.assertNotEqual(r.dominant, "joy")
        self.assertLessEqual(r.valence, 0.0)

    def test_intensifier_boosts(self) -> None:
        plain = self.scorer.score("i am happy")
        strong = self.scorer.score("i am so very happy")
        self.assertGreaterEqual(strong.intensity, plain.intensity)

    def test_caps_and_bangs_boost(self) -> None:
        plain = self.scorer.score("i am angry")
        loud = self.scorer.score("I AM ANGRY!!!")
        self.assertGreaterEqual(loud.intensity, plain.intensity)

    def test_neutral_text(self) -> None:
        r = self.scorer.score("the meeting is at 3pm tomorrow")
        self.assertEqual(r.dominant, "neutral")
        self.assertEqual(r.intensity, 0.0)

    def test_empty(self) -> None:
        r = self.scorer.score("   ")
        self.assertEqual(r.dominant, "neutral")

    def test_deterministic(self) -> None:
        a = self.scorer.score("i miss you so much it hurts")
        b = self.scorer.score("i miss you so much it hurts")
        self.assertEqual(a.to_dict(), b.to_dict())


class AffectEventsTest(unittest.TestCase):
    def test_cap(self) -> None:
        r = score_affect("i am extremely happy thrilled delighted overjoyed")
        events = affect_to_events(r)
        for e in events:
            self.assertLessEqual(e.intensity, AFFECT_EVENT_CAP)

    def test_skip_kinds(self) -> None:
        r = score_affect("i am so sorry, i feel terrible and guilty")
        events = affect_to_events(r, skip_kinds=frozenset({"apology"}))
        self.assertNotIn("apology", {e.kind for e in events})

    def test_weak_reading_no_events(self) -> None:
        r = score_affect("the meeting is at 3pm")
        self.assertEqual(affect_to_events(r), [])


class DetectSignalsAffectTest(unittest.TestCase):
    def test_affect_second_pass_fires(self) -> None:
        # No regex matches this; the lexicon backstop should hear sadness.
        events = detect_signals("im so sad and exhausted today, everything hurts")
        kinds = {e.kind for e in events}
        self.assertIn("bad_news", kinds)
        for e in events:
            if e.note.startswith("affect:"):
                self.assertLessEqual(e.intensity, AFFECT_EVENT_CAP)

    def test_regex_kinds_not_duplicated(self) -> None:
        events = detect_signals("i'm sorry, i shouldn't have said that")
        kinds = [e.kind for e in events]
        self.assertEqual(kinds.count("apology"), 1)

    def test_use_affect_false(self) -> None:
        events = detect_signals("im so sad and exhausted today", use_affect=False)
        self.assertEqual(events, [])

    def test_event_cap_total(self) -> None:
        events = detect_signals(
            "i love you so much thank you you are amazing wonderful fantastic",
            use_affect=True,
        )
        self.assertLessEqual(len(events), 6)


# ── persona ──────────────────────────────────────────────────────────────────

class PersonaSweepTest(unittest.TestCase):
    def test_dialogue_block(self) -> None:
        p = default_persona()
        block = p.dialogue_block(mood_label="tired")
        self.assertIn("<START>", block)
        self.assertIn("them:", block)
        self.assertIn("you:", block)

    def test_dialogue_block_empty(self) -> None:
        p = persona_from_dict({"name": "X"})
        self.assertEqual(p.dialogue_block(), "")

    def test_greeting_for_mood(self) -> None:
        p = default_persona()
        rng = random.Random(7)
        g = p.greeting_for("excited", rng=rng)
        self.assertTrue(g)
        self.assertIn(g, list(p.mood_greetings["excited"]))

    def test_greeting_fallback(self) -> None:
        p = default_persona()
        g = p.greeting_for("suspicious", rng=random.Random(1))
        self.assertIn(g, list(p.greetings))

    def test_greeting_none(self) -> None:
        p = persona_from_dict({"name": "X"})
        self.assertEqual(p.greeting_for("happy"), "")

    def test_big_five_block_behavioral(self) -> None:
        p = default_persona()
        block = p.big_five_block()
        for trait in ("Openness", "Conscientiousness", "Extraversion",
                      "Agreeableness", "Neuroticism"):
            self.assertIn(trait, block)
        # Behavioral, not adjectives: no bare trait-value dump.
        self.assertIn("you ", block.lower())

    def test_big_five_validation(self) -> None:
        from nomorals.core.errors import ValidationError
        with self.assertRaises(ValidationError):
            persona_from_dict({"name": "X", "big_five": {"openness": 500}})

    def test_big_five_defaults_filled(self) -> None:
        p = persona_from_dict({"name": "X"})
        for t in BIG_FIVE_TRAITS:
            self.assertIn(t, p.big_five)

    def test_core_signature_stable(self) -> None:
        p = default_persona()
        self.assertEqual(p.core_signature(), p.core_signature())
        self.assertFalse(p.drift_from(p.core_signature()))

    def test_drift_detected(self) -> None:
        p = default_persona()
        sig = p.core_signature()
        p2 = persona_from_dict({**p.to_dict(), "name": "Someone Else"})
        self.assertTrue(p2.drift_from(sig))

    def test_to_dict_round_trip(self) -> None:
        p = default_persona()
        d = p.to_dict()
        p2 = persona_from_dict(d)
        self.assertEqual(p2.core_signature(), p.core_signature())
        self.assertEqual(len(p2.dialogue_examples), len(p.dialogue_examples))
        self.assertEqual(p2.nickname, p.nickname)
        self.assertEqual(p2.greetings, p.greetings)

    def test_prompt_has_big_five_and_nickname(self) -> None:
        p = default_persona()
        prompt = p.to_prompt()
        self.assertIn("wren", prompt)
        self.assertIn("How your personality actually shows", prompt)


# ── mood ─────────────────────────────────────────────────────────────────────

class MoodSweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MoodEngine(default_persona().baselines)

    def test_spike_registered(self) -> None:
        sp = self.engine.spike("touched", 0.7, cause="the compliment")
        self.assertIsInstance(sp, EmotionSpike)
        self.assertEqual(sp.label, "touched")
        self.assertAlmostEqual(sp.strength(), 0.7, places=2)

    def test_spike_decays(self) -> None:
        sp = self.engine.spike("stung", 0.8, half_life_s=60.0)
        self.assertLess(sp.strength(sp.ts + 600.0), 0.01)

    def test_spike_pruned_by_tick(self) -> None:
        self.engine.spike("stung", 0.9, half_life_s=60.0)
        self.engine.tick(now=self.engine._now + 3600.0)
        self.assertEqual(self.engine.spikes, [])

    def test_strong_spike_marks_mood(self) -> None:
        before = self.engine.value("happiness")
        self.engine.spike("delighted", 0.9, cause="great news")
        self.assertGreater(self.engine.value("happiness"), before)

    def test_note_event_math_is_exact(self) -> None:
        # The funnel stays exact: no appraisal inside note_event.
        self.engine.note_event("fight_start", intensity=1.0)
        self.assertEqual(self.engine.value("frustration"), 8 + 25)

    def test_appraise_dampens_negative_at_good_mood(self) -> None:
        # valence is positive at baselines
        self.assertGreater(self.engine.valence(), 0.3)
        self.engine.appraise("insult", intensity=1.0)
        # raw would be 8 + 20 = 28; appraised must be lower
        self.assertLess(self.engine.value("frustration"), 28.0)

    def test_appraise_amplifies_negative_at_bad_mood(self) -> None:
        self.engine.set_dimensions({
            "happiness": 15, "affection": 20, "trust": 20, "intimacy": 20,
            "pride": 20, "energy": 20, "frustration": 60, "jealousy": 60,
            "distance": 60, "insecurity": 60,
        })
        self.assertLess(self.engine.valence(), -0.3)
        self.engine.appraise("insult", intensity=1.0)
        # raw would be 60 + 20 = 80; appraised must exceed it
        self.assertGreater(self.engine.value("frustration"), 80.0)

    def test_appraise_dampens_positive_at_bad_mood(self) -> None:
        self.engine.set_dimensions({
            "happiness": 15, "affection": 20, "trust": 20, "intimacy": 20,
            "pride": 20, "energy": 20, "frustration": 60, "jealousy": 60,
            "distance": 60, "insecurity": 60,
        })
        self.engine.appraise("compliment", intensity=1.0)
        # raw would be 15 + 12 = 27; appraised must be lower
        self.assertLess(self.engine.value("happiness"), 27.0)

    def test_appraise_unknown_kind_noop(self) -> None:
        before = dict(self.engine.current().values)
        self.engine.appraise("not_a_kind", intensity=1.0)
        self.assertEqual(self.engine.current().values, before)

    def test_pad_point_shape(self) -> None:
        p, a, d = self.engine.pad_point()
        for v in (p, a, d):
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_mood_pad_covers_all_labels(self) -> None:
        from nomorals.partner.mood import MOOD_LABELS
        for label in MOOD_LABELS:
            self.assertIn(label, MOOD_PAD)
            self.assertEqual(len(MOOD_PAD[label]), 3)

    def test_active_influences(self) -> None:
        self.engine.spike("rattled", 0.6, cause="the loud message")
        influences = self.engine.active_influences()
        self.assertTrue(any(i["kind"] == "spike" and i["label"] == "rattled"
                            for i in influences))

    def test_describe_names_spike(self) -> None:
        self.engine.spike("touched", 0.7, cause="what they said")
        self.assertIn("touched", self.engine.describe())

    def test_reset_clears_spikes(self) -> None:
        self.engine.spike("stung", 0.7)
        self.engine.reset()
        self.assertEqual(self.engine.spikes, [])

    def test_valence_range(self) -> None:
        v = self.engine.valence()
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)


# ── relationship ─────────────────────────────────────────────────────────────

class FakeCursor:
    def __init__(self, rowcount: int = 0) -> None:
        self.rowcount = rowcount


class FakeDB:
    """Minimal stand-in for Database: relationship table only."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def query_one(self, sql: str, params: tuple = ()):  # noqa: ARG002
        return self.rows.get(params[0]) if params else None

    def execute(self, sql: str, params=None):  # noqa: ARG002
        if isinstance(params, dict):
            row = dict(params)
            self.rows[row["id"]] = row
        elif isinstance(params, (list, tuple)) and params:
            # Positional params (e.g. chat_profiles upsert): key on first.
            self.rows[str(params[0])] = {"_params": tuple(params)}
        # CREATE TABLE and friends: no-op.
        return FakeCursor()

    def transaction(self):
        return nullcontext()


class RelationshipSweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rel = Relationship()

    def test_axes_defaults(self) -> None:
        for axis in ARC_AXES:
            self.assertIn(axis, self.rel.axes)
        self.assertEqual(self.rel.axes["trust"], DEFAULT_AXES["trust"])

    def test_nudge_axis(self) -> None:
        new = self.rel.nudge_axis("warmth", 10, "good date")
        self.assertEqual(new, DEFAULT_AXES["warmth"] + 10)

    def test_nudge_axis_clamped(self) -> None:
        self.rel.nudge_axis("warmth", 1000)
        self.assertEqual(self.rel.axes["warmth"], 100)
        self.rel.nudge_axis("warmth", -1000)
        self.assertEqual(self.rel.axes["warmth"], 0)

    def test_nudge_unknown_axis_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.rel.nudge_axis("vibes", 5)

    def test_weakest_axis(self) -> None:
        self.rel.nudge_axis("respect", -50, "test")
        self.assertEqual(self.rel.weakest_axis(), "respect")

    def test_repair_attempt_vs_outcome(self) -> None:
        aid = self.rel.record_repair_attempt("brought coffee and apologized")
        self.assertEqual(len(self.rel.open_repair_attempts()), 1)
        # An attempt is not an outcome: still open.
        self.assertIsNone(self.rel.open_repair_attempts()[0]["outcome"])
        self.assertTrue(self.rel.record_repair_outcome(aid, True))
        self.assertEqual(self.rel.open_repair_attempts(), [])
        self.assertEqual(self.rel.repair_attempts[0]["outcome"], "landed")

    def test_repair_missed(self) -> None:
        aid = self.rel.record_repair_attempt("tried to explain")
        self.assertTrue(self.rel.record_repair_outcome(aid, False))
        self.assertEqual(self.rel.repair_attempts[0]["outcome"], "missed")

    def test_repair_unknown_id(self) -> None:
        self.assertFalse(self.rel.record_repair_outcome("nope", True))

    def test_log_activity(self) -> None:
        self.rel.log_activity("hiked the ridge trail together")
        acts = self.rel.recent_activities()
        self.assertEqual(len(acts), 1)
        self.assertIn("ridge", acts[0]["text"])
        # Shared reality grows with shared experience.
        self.assertGreater(self.rel.axes["shared_reality"], DEFAULT_AXES["shared_reality"])

    def test_anniversaries(self) -> None:
        self.assertEqual(self.rel.anniversaries(), [])
        self.rel.add_milestone("first met", kind="moment")
        self.rel.advance_stage("they asked")
        annivs = self.rel.anniversaries()
        self.assertTrue(any(a["name"] == "together since" for a in annivs))
        self.assertTrue(any(a["name"] == "dating" for a in annivs))

    def test_arc_persists_through_row(self) -> None:
        db = FakeDB()
        self.rel.nudge_axis("warmth", 7, "x")
        aid = self.rel.record_repair_attempt("trying")
        self.rel.log_activity("movie night")
        self.rel.note_user_fact("favorite_food", "ramen")
        self.rel.save(db)
        loaded = Relationship.load(db)
        self.assertEqual(loaded.axes["warmth"], self.rel.axes["warmth"])
        self.assertEqual(len(loaded.repair_attempts), 1)
        self.assertEqual(loaded.repair_attempts[0]["id"], aid)
        self.assertEqual(len(loaded.shared_activities), 1)
        # Real profile facts survive; the reserved arc key does not leak.
        self.assertEqual(loaded.user_profile.get("favorite_food"), "ramen")
        self.assertNotIn("__arc__", loaded.user_profile)

    def test_prompt_block_has_axes(self) -> None:
        block = self.rel.to_prompt_block()
        self.assertIn("warmth", block)
        self.rel.record_repair_attempt("said sorry properly")
        block = self.rel.to_prompt_block()
        self.assertIn("hasn't landed yet", block)


# ── style ────────────────────────────────────────────────────────────────────

class TextingVoiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.speech = default_persona().speech

    def test_lowercase_run(self) -> None:
        rng = random.Random(3)
        out = apply_texting_voice("I Am Really Tired Today", self.speech,
                                  {"energy": 60, "frustration": 10, "distance": 30},
                                  "calm", rng)
        # seeded: with lowercase_bias 0.5 this seed goes lowercase
        self.assertEqual(out, out.lower())

    def test_no_lowercase_sometimes(self) -> None:
        outs = {
            apply_texting_voice("Hello There", self.speech,
                                {"energy": 60, "frustration": 10, "distance": 30},
                                "calm", random.Random(i))
            for i in range(20)
        }
        self.assertGreater(len(outs), 1)  # probabilistic, not forced

    def test_tired_ellipsis(self) -> None:
        outs = [
            apply_texting_voice("I'm beat.", self.speech,
                                {"energy": 10, "frustration": 10, "distance": 30},
                                "tired", random.Random(i))
            for i in range(20)
        ]
        self.assertTrue(any("…" in o for o in outs))

    def test_frustration_flattens_bangs(self) -> None:
        out = apply_texting_voice("Wow! Amazing! Great!",
                                  self.speech,
                                  {"energy": 60, "frustration": 80, "distance": 30},
                                  "angry", random.Random(0))
        self.assertNotIn("!", out)

    def test_empty_passthrough(self) -> None:
        self.assertEqual(apply_texting_voice("", self.speech, {}, "", random.Random(0)), "")

    def test_never_raises(self) -> None:
        out = apply_texting_voice("hi", None, {}, "", random.Random(0))
        self.assertEqual(out, "hi")


class VoiceThemesTest(unittest.TestCase):
    def test_known_theme(self) -> None:
        speech = default_persona().speech
        feral = apply_voice_theme(speech, "feral")
        self.assertEqual(feral.emoji_rate, VOICE_THEMES["feral"]["emoji_rate"])
        # Original untouched.
        self.assertNotEqual(speech.emoji_rate, feral.emoji_rate)

    def test_unknown_theme_passthrough(self) -> None:
        speech = default_persona().speech
        self.assertIs(apply_voice_theme(speech, "nope"), speech)

    def test_all_themes_apply(self) -> None:
        speech = default_persona().speech
        for name in VOICE_THEMES:
            themed = apply_voice_theme(speech, name)
            self.assertTrue(hasattr(themed, "emoji_rate"))


# ── context ──────────────────────────────────────────────────────────────────

class ContextSweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.persona = default_persona()
        self.mood = MoodEngine(self.persona.baselines)
        self.rel = Relationship()
        self.builder = PartnerContextBuilder(total_budget=4200)

    def test_post_history_is_last(self) -> None:
        msg = self.builder.build(
            persona=self.persona, mood=self.mood, relationship=self.rel,
            post_history="tonight she is extra tired",
        )
        self.assertIn("tonight she is extra tired", msg.content)
        # Post-history block trails the output contract.
        self.assertGreater(msg.content.index("tonight she is extra tired"),
                           msg.content.index("OUTPUT CONTRACT"))

    def test_dialogue_block_included(self) -> None:
        msg = self.builder.build(
            persona=self.persona, mood=self.mood, relationship=self.rel)
        self.assertIn("<START>", msg.content)

    def test_shrink_order_respected(self) -> None:
        tight = PartnerContextBuilder(total_budget=1500)
        msg = tight.build(
            persona=self.persona, mood=self.mood, relationship=self.rel,
            memories=["memory one", "memory two"],
            background_lines=["bg line one", "bg line two"],
            continuity_lines=["elsewhere one"],
        )
        # Identity must survive the squeeze.
        self.assertIn(self.persona.name, msg.content)
        self.assertIn("OUTPUT CONTRACT", msg.content)
        # Shrink order: background goes before memory.
        self.assertEqual(SHRINK_ORDER[0], "background")

    def test_no_post_history_no_block(self) -> None:
        msg = self.builder.build(
            persona=self.persona, mood=self.mood, relationship=self.rel)
        self.assertNotIn("Before you reply", msg.content)


# ── background ───────────────────────────────────────────────────────────────

class BackgroundSweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sel = BackgroundSelector()

    def test_constant_facts_exist(self) -> None:
        constants = self.sel.pack.constant_facts()
        self.assertTrue(len(constants) >= 2)
        self.assertTrue(all(f.constant for f in constants))

    def test_constant_leads_context(self) -> None:
        # "dog" matches the dog fact; the constant cabin fact should lead.
        lines = self.sel.context("tell me about your dog", user_in_us=True,
                                 romantic=True)
        self.assertTrue(lines)
        joined = "\n".join(lines)
        self.assertIn("Juniper", joined)

    def test_recency_deprioritizes_repeat(self) -> None:
        # "work" matches several facts; after serving, the top one sinks.
        first = self.sel.pack.select("work", limit=3)
        self.assertGreater(len(first), 1)
        top_id = first[0].id
        self.sel.pack.mark_used([first[0]])
        second = self.sel.pack.select("work", limit=3)
        self.assertTrue(second)
        self.assertNotEqual(second[0].id, top_id)

    def test_ambient_line(self) -> None:
        line = self.sel.ambient_line(user_in_us=True, romantic=True, hour=8)
        self.assertTrue(line)
        self.assertIsInstance(line, str)

    def test_gate_closed(self) -> None:
        sel = BackgroundSelector(mode="never")
        self.assertEqual(sel.context("dog", user_in_us=True, romantic=True), [])
        self.assertEqual(sel.ambient_line(user_in_us=True, romantic=True), "")


# ── presence ─────────────────────────────────────────────────────────────────

class PresenceSweepTest(unittest.TestCase):
    def test_typing_schedule_keepalive(self) -> None:
        long_text = "word " * 200  # ~1000 chars -> long type
        plans = typing_schedule([long_text], mood={"energy": 60, "happiness": 60},
                                rng=random.Random(5))
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertIsInstance(plan, TypingPlan)
        self.assertGreater(len(plan.ticks), 1)
        # Ticks spaced by the keepalive interval.
        gaps = [b - a for a, b in zip(plan.ticks, plan.ticks[1:])]
        for g in gaps:
            self.assertAlmostEqual(g, TYPING_KEEPALIVE_S, places=1)

    def test_typing_schedule_short(self) -> None:
        plans = typing_schedule(["k"], rng=random.Random(1))
        self.assertEqual(plans[0].ticks[0], 0.0)
        self.assertGreaterEqual(plans[0].duration, 2.0)  # minimum typing

    def test_inter_bubble_gap(self) -> None:
        plans = typing_schedule(["one", "two"], rng=random.Random(1))
        self.assertEqual(len(plans), 2)
        # Second bubble's ticks are offset by the inter-bubble beat.
        self.assertGreater(plans[1].ticks[0], 0.0)

    def test_read_delay_range(self) -> None:
        rng = random.Random(2)
        short = read_delay_seconds("hi", rng=rng)
        long = read_delay_seconds("x" * 2000, rng=rng)
        self.assertGreaterEqual(short, 0.5)
        self.assertGreaterEqual(long, short)

    def test_plan_to_dict(self) -> None:
        plan = TypingPlan(ticks=(0.0, 4.0), duration=6.2)
        d = plan.to_dict()
        self.assertEqual(d["ticks"], [0.0, 4.0])


# ── gating / social gate / group roles ───────────────────────────────────────

class GatingSweepTest(unittest.TestCase):
    def test_first_contact_frame(self) -> None:
        block = gate_block("private", first_contact=True)
        self.assertIn("FIRST message", block)

    def test_no_first_contact_by_default(self) -> None:
        block = gate_block("private")
        self.assertNotIn("FIRST message", block)

    def test_owner_empty(self) -> None:
        self.assertEqual(gate_block("owner", first_contact=True), "")


class DenialLogTest(unittest.TestCase):
    def test_denial_recorded(self) -> None:
        log = DenialLog()
        grant = grant_for(is_owner=False, chat_kind="group")
        ok, _ = check_tool_call("memory_recall", grant=grant,
                                chat_kind="group", audit=log)
        self.assertFalse(ok)
        self.assertEqual(len(log), 1)
        rec = log.recent(1)[0]
        self.assertEqual(rec.tool_name, "memory_recall")
        self.assertEqual(rec.actor, "outsider")

    def test_allowed_not_recorded(self) -> None:
        log = DenialLog()
        grant = grant_for(is_owner=False, chat_kind="group")
        ok, _ = check_tool_call("games", grant=grant, chat_kind="group",
                                audit=log)
        self.assertTrue(ok)
        self.assertEqual(len(log), 0)

    def test_capacity_bound(self) -> None:
        log = DenialLog(capacity=10)
        grant = grant_for(is_owner=False, chat_kind="group")
        for i in range(25):
            check_tool_call(f"tool_{i}", grant=grant, audit=log)
        self.assertEqual(len(log), 10)
        # Newest first.
        self.assertEqual(log.recent(1)[0].tool_name, "tool_24")

    def test_recent_denials_shape(self) -> None:
        from nomorals.partner.social_gate import recent_denials
        recs = recent_denials(5)
        self.assertIsInstance(recs, list)


class GroupRolesDiscordTest(unittest.TestCase):
    def _adapter(self, roles):
        class A:
            def discord_member_roles(self, chat_key, user_id):
                return {"ok": True, "roles": roles}
        return A()

    def test_admin_by_permission_bit(self) -> None:
        adapter = self._adapter([{"name": "member", "permissions": 0x8}])
        self.assertEqual(resolve_group_role("discord", adapter, "c", "u"),
                         "admin")

    def test_admin_by_name(self) -> None:
        adapter = self._adapter([{"name": "Moderator", "permissions": 0}])
        self.assertEqual(resolve_group_role("discord", adapter, "c", "u"),
                         "admin")

    def test_member(self) -> None:
        adapter = self._adapter([{"name": "member", "permissions": 0}])
        self.assertEqual(resolve_group_role("discord", adapter, "c", "u"),
                         "member")

    def test_fail_closed_on_error(self) -> None:
        class Bad:
            def discord_member_roles(self, chat_key, user_id):
                raise RuntimeError("boom")
        from nomorals.partner.group_roles import clear_role_cache
        clear_role_cache()
        self.assertEqual(resolve_group_role("discord", Bad(), "c2", "u2"),
                         "unknown")

    def test_plain_name_list(self) -> None:
        class A:
            def discord_member_roles(self, chat_key, user_id):
                return ["admin"]
        from nomorals.partner.group_roles import clear_role_cache
        clear_role_cache()
        self.assertEqual(resolve_group_role("discord", A(), "c3", "u3"), "admin")


# ── chat profile ─────────────────────────────────────────────────────────────

class ChatProfileSweepTest(unittest.TestCase):
    def test_vibe_labels(self) -> None:
        self.assertEqual(_vibe_label(0.8), "warm")
        self.assertEqual(_vibe_label(0.2), "easy")
        self.assertEqual(_vibe_label(0.0), "neutral")
        self.assertEqual(_vibe_label(-0.2), "heavy")
        self.assertEqual(_vibe_label(-0.8), "tense")

    def test_activity_labels(self) -> None:
        self.assertEqual(_activity_label(0.5), "quiet lately")
        self.assertEqual(_activity_label(5), "steady")
        self.assertEqual(_activity_label(50), "buzzing")

    def test_context_lines_include_vibe(self) -> None:
        lines = build_context_lines({
            "participants": {"amy": 10},
            "topics": ["hiking"],
            "vibe_label": "warm",
            "activity_label": "buzzing",
        })
        joined = "\n".join(lines)
        self.assertIn("warm", joined)
        self.assertIn("buzzing", joined)


class FakeChatDB(FakeDB):
    def __init__(self, messages: list[dict]) -> None:
        super().__init__()
        self._messages = messages

    def query(self, sql: str, params: tuple = ()):
        if "FROM messages" in sql and "GROUP BY name" in sql:
            counts: dict[str, int] = {}
            for m in self._messages:
                if m["role"] == "user":
                    counts[m["name"]] = counts.get(m["name"], 0) + 1
            return [{"name": n, "n": c} for n, c in counts.items()]
        if "FROM messages" in sql:
            return [{"content": m["content"]} for m in self._messages
                    if m["role"] == "user"][: params[1] if len(params) > 1 else 999]
        return []


class ChatProfileRefreshTest(unittest.TestCase):
    def test_vibe_and_activity_computed(self) -> None:
        msgs = [
            {"role": "user", "name": "amy", "content": "i am so happy today wonderful news"},
            {"role": "user", "name": "amy", "content": "this is amazing i love it"},
            {"role": "user", "name": "bob", "content": "great stuff honestly"},
        ]
        db = FakeChatDB(msgs)
        profile = refresh_profile(db, "test:chat:1", force=True)
        self.assertEqual(profile["vibe_label"], "warm")
        self.assertIn("activity_label", profile)
        self.assertIn(profile["activity_label"],
                      {"quiet lately", "steady", "buzzing"})


# ── lexicon usage ────────────────────────────────────────────────────────────

class FakeLexiconCursor:
    def __init__(self, rowcount: int = 0) -> None:
        self.rowcount = rowcount


class FakeLexiconDB:
    """Enough of the DB for LexiconFeed usage + prune_stale."""

    def __init__(self, terms: list[dict]) -> None:
        self.terms_rows = terms  # {"term","category","created_at","status"}
        self.usage_rows: dict[tuple[str, str, str], dict] = {}

    def execute(self, sql: str, params: tuple = ()):
        if "lexicon_term_usage" in sql and "INSERT" in sql:
            key = (params[0], params[1], params[2])
            row = self.usage_rows.get(key, {"uses": 0})
            row["uses"] = row.get("uses", 0) + 1
            row["last_used"] = params[3]
            self.usage_rows[key] = row
            return FakeLexiconCursor(1)
        if sql.strip().startswith("UPDATE lexicon_terms"):
            changed = 0
            for r in self.terms_rows:
                if (r["term"] == params[0] and r.get("module", "partner") == params[1]
                        and r["status"] == "active"):
                    r["status"] = "retired"
                    changed += 1
            return FakeLexiconCursor(changed)
        return FakeLexiconCursor(0)

    def query(self, sql: str, params: tuple = ()):
        if "FROM lexicon_terms" in sql:
            cutoff = params[1]
            return [r for r in self.terms_rows
                    if r["status"] == "active" and r["created_at"] < cutoff]
        return []

    def query_one(self, sql: str, params: tuple = ()):
        if "FROM lexicon_term_usage" in sql:
            row = self.usage_rows.get((params[0], params[1], params[2]))
            return {"uses": row["uses"]} if row else None
        return None


class LexiconUsageTest(unittest.TestCase):
    def test_note_and_usage(self) -> None:
        feed = LexiconFeed(FakeLexiconDB([]))
        self.assertEqual(feed.usage("okay wait", "catchphrase"), 0)
        feed.note_used("okay wait", "catchphrase")
        feed.note_used("okay wait", "catchphrase")
        self.assertEqual(feed.usage("okay wait", "catchphrase"), 2)

    def test_note_used_no_db(self) -> None:
        feed = LexiconFeed(None)
        feed.note_used("okay wait", "catchphrase")  # must not raise
        self.assertEqual(feed.usage("okay wait", "catchphrase"), 1)

    def test_prune_stale(self) -> None:
        old = time.time() - 100 * 86400
        db = FakeLexiconDB([
            {"term": "dusty phrase", "category": "catchphrase",
             "created_at": old, "status": "active", "module": "partner"},
            {"term": "fresh phrase", "category": "catchphrase",
             "created_at": time.time(), "status": "active", "module": "partner"},
        ])
        feed = LexiconFeed(db)
        feed.note_used("dusty phrase", "catchphrase")  # used once...
        # ...but min_uses=2, so it still prunes.
        report = feed.prune_stale(max_age_days=90, min_uses=2)
        self.assertIn("catchphrase:dusty phrase", report["pruned"])
        self.assertEqual(db.terms_rows[0]["status"], "retired")
        self.assertEqual(db.terms_rows[1]["status"], "active")

    def test_prune_keeps_used(self) -> None:
        old = time.time() - 100 * 86400
        db = FakeLexiconDB([
            {"term": "beloved phrase", "category": "catchphrase",
             "created_at": old, "status": "active", "module": "partner"},
        ])
        feed = LexiconFeed(db)
        feed.note_used("beloved phrase", "catchphrase")
        feed.note_used("beloved phrase", "catchphrase")
        report = feed.prune_stale(max_age_days=90, min_uses=2)
        self.assertEqual(report["pruned"], [])
        self.assertEqual(report["kept"], 1)

    def test_prune_no_db(self) -> None:
        feed = LexiconFeed(None)
        report = feed.prune_stale()
        self.assertEqual(report["reason"], "no db")

    def test_terms_by_signal(self) -> None:
        feed = LexiconFeed(None)
        # No DB: dynamic_terms(None, ...) — must not raise.
        self.assertEqual(feed.terms_by_signal("catchphrase"), ())


# ── responder wiring ─────────────────────────────────────────────────────────

class ResponderSweepTest(unittest.TestCase):
    def _responder(self) -> PartnerResponder:
        persona = default_persona()
        mood = MoodEngine(persona.baselines)
        rel = Relationship()
        return PartnerResponder(
            router=None, persona=persona, mood=mood,
            relationship=rel, memory=None, background=BackgroundSelector(),
            rng=random.Random(11),
        )

    def test_proactive_greeting(self) -> None:
        r = self._responder()
        g = r.proactive_greeting()
        self.assertTrue(g)
        self.assertIsInstance(g, str)

    def test_bundle_affect_field(self) -> None:
        from nomorals.partner.responder import ReplyBundle
        b = ReplyBundle(parts=["hi"], affect={"dominant": "joy"})
        d = b.to_dict()
        self.assertEqual(d["affect"]["dominant"], "joy")

    def test_fallback_parts_still_work(self) -> None:
        r = self._responder()
        parts, used = r._fallback_parts("happy")
        self.assertTrue(parts)
        self.assertIsInstance(used, int)


if __name__ == "__main__":
    unittest.main()
