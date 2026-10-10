"""Sweep tests for the characters module upgrade.

Covers the new behavior added in the sweep: OCEAN conditioning,
reflection, three-factor recall, secret pressure, goal lifecycles,
persona-flavored fallbacks, relationship transitions + stories,
DramaDirector + scene formats, proactive pulse, match intros,
stylometry-grade voice fingerprints, the render layer, store queries,
and the processing reflection/drift hooks.
"""
import random
import time

import pytest

from nomorals.characters import (
    Character, derive_ocean, OCEAN_TRAITS,
    CharacterStore,
    Dialogue, converse, character_initiate, proactive_pulse, icebreakers,
    AgentSeat, MatchResult, run_agent_match, match_intro,
    RelationshipGraph, Edge, DIMS, EVENT_DELTAS,
    add_belief, challenge_belief, milestone, arc_summary, arc_story,
    cast_for, cast_preset, cast_against, cast_for_audience, chemistry,
    Scene, run_scene, podcast_episode,
    DramaDirector, Beat, default_beats, SCENE_FORMATS,
    process_session, SessionEvent, events_from_dialogue, decay_moods,
    CharacterMemory,
    VoiceFingerprint, fingerprint, consistency, distinctness,
    function_word_profile, style_report,
    CharacterContextBuilder,
    render_card, render_cast, render_scene, render_web, render_arc,
)
from nomorals.characters import processing as proc_mod


def make_char(name="Zara", **kw):
    base = dict(persona={"witty": 0.9, "energetic": 0.8},
                core_motive="To ask the question nobody else will ask.")
    base.update(kw)
    return Character(name=name, **base)


class FixedRng:
    """Deterministic rng: random() -> value, choice -> first."""
    def __init__(self, value=0.0):
        self.value = value
    def random(self):
        return self.value
    def choice(self, seq):
        return seq[0]


# ── OCEAN ─────────────────────────────────────────────────────────────

class TestOcean:
    def test_explicit_ocean_wins(self):
        c = make_char(ocean={"openness": 0.9, "conscientiousness": 0.1,
                             "extraversion": 0.9, "agreeableness": 0.5,
                             "neuroticism": 0.2})
        assert c.ocean["openness"] == 0.9
        assert c.ocean["conscientiousness"] == 0.1

    def test_derived_ocean_sane(self):
        c = make_char()  # witty + energetic
        for t in OCEAN_TRAITS:
            assert 0.0 <= c.ocean[t] <= 1.0
        assert c.ocean["extraversion"] > 0.6  # energetic persona

    def test_ocean_line_in_persona_block(self):
        c = make_char()
        block = c.persona_block()
        assert "Personality profile:" in block
        assert "extraversion" in block

    def test_derive_ocean_direct(self):
        o = derive_ocean({"calm": 0.9, "patient": 0.8})
        assert o["neuroticism"] < 0.5      # calm = steady
        assert o["conscientiousness"] > 0.5  # patient = disciplined


# ── recall: three-factor ──────────────────────────────────────────────

class TestRecall:
    def test_relevance_beats_stale_salience(self):
        c = make_char()
        c.memory = []  # bypass remember() clamping order
        old = time.time() - 30 * 86400
        c.memory.append({"ts": old, "text": "unrelated old note",
                         "salience": 1.0})
        c.memory.append({"ts": time.time(),
                         "text": "the podcast microphone broke today",
                         "salience": 0.6})
        out = c.recall("podcast microphone", limit=2)
        assert out and "microphone" in out[0]

    def test_reflections_surface_with_bonus(self):
        c = make_char()
        c.remember("went to the market", 0.5)
        c.reflections.append({"text": "I keep thinking about the market crowds",
                              "ts": time.time(), "source_count": 3})
        out = c.recall("market crowds thinking", limit=3)
        assert any(t.startswith("💭") for t in out)


# ── reflection ────────────────────────────────────────────────────────

class TestReflect:
    def test_model_free_reflection(self):
        c = make_char()
        for t in ["the podcast audience keeps growing every week",
                  "the podcast guests keep cancelling last minute",
                  "the podcast studio keeps flooding when it rains"]:
            c.remember(t, 0.7)
        insights = c.reflect(suggest=None)
        assert insights, "reflection should produce at least one insight"
        assert len(c.reflections) >= 1
        # reflection is remembered too
        assert any("Reflected" in m.get("text", "") for m in c.memory)

    def test_model_backed_reflection(self):
        c = make_char()
        c.remember("a big argument with Kilo about the show format", 0.8)
        c.remember("made up with Kilo over suya after the show", 0.8)
        ins = c.reflect(suggest=lambda p: "Kilo and I fight because we both care.\nWe always make up.")
        assert any("Kilo" in i for i in ins)

    def test_reflect_empty_memory(self):
        c = make_char()
        assert c.reflect() == []


# ── secrets ───────────────────────────────────────────────────────────

class TestSecrets:
    def test_normalized(self):
        c = make_char(secrets=["plain string",
                               {"text": "dict secret", "pressure": 0.9}])
        assert c.secrets[0] == {"text": "plain string", "pressure": 0.8}
        assert c.secret_texts() == ["plain string", "dict secret"]

    def test_no_slip_below_threshold(self):
        c = make_char(secrets=[{"text": "deep one", "pressure": 0.9}])
        assert c.slip_secret(pressure=0.5, rng=FixedRng(0.0)) is None

    def test_slip_at_crisis(self):
        c = make_char(secrets=[{"text": "deep one", "pressure": 0.9}])
        # pressure 1.0 -> always slips when a candidate exists
        assert c.slip_secret(pressure=1.0, rng=FixedRng(0.99)) == "deep one"

    def test_old_json_strings_still_load(self):
        c = Character.from_dict({"name": "Old", "secrets": ["a", "b"]})
        assert c.secret_texts() == ["a", "b"]


# ── goals lifecycle ───────────────────────────────────────────────────

class TestGoals:
    def test_goal_states_default(self):
        c = make_char(goals=["host the best podcast"])
        assert c.goal_states["host the best podcast"]["status"] == "pursuing"

    def test_achieve_goal(self):
        c = make_char(goals=["win the match"])
        assert c.set_goal_status("win the match", "achieved", progress=1.0)
        st = c.goal_states["win the match"]
        assert st["status"] == "achieved" and st["progress"] == 1.0
        assert "win the match" not in c.goal_agenda()
        assert any("Achieved a goal" in m.get("text", "") for m in c.memory)

    def test_agenda_shows_pursuing(self):
        c = make_char(goals=["a", "b"])
        c.set_goal_status("b", "paused")
        agenda = c.goal_agenda()
        assert "a" in agenda and "b" not in agenda


# ── fallback voice ────────────────────────────────────────────────────

class TestFallback:
    def test_no_brackets(self):
        c = make_char()
        line = c._fallback_line("the mic is live")
        assert "[" not in line and "]" not in line
        assert "mic is live" in line

    def test_deterministic(self):
        c = make_char()
        assert c._fallback_line("hello there") == c._fallback_line("hello there")

    def test_persona_flavored(self):
        dry = make_char(name="Kilo", persona={"dry": 0.95})
        line = dry._fallback_line("the show starts now")
        assert "show starts now" in line.lower()
        assert len(line) > len("the show starts now")

    def test_speak_fallback_when_no_model(self):
        c = make_char()
        line = c.speak("Devon asks about the podcast", suggest=None)
        assert line and "[" not in line


# ── relationships: transitions, stories, triads ───────────────────────

class TestRelationships:
    def test_transition_recorded_on_flip(self):
        g = RelationshipGraph()
        a, b = "aaa", "bbb"
        for _ in range(12):
            g.interact(a, b, "betrayed")
        e = g.edge(a, b)
        assert e.kind == "rival"
        assert e.transitions, "kind flip must be recorded"
        line = e.last_transition_line("Ama", "Bola")
        assert "Ama" in line and "Bola" in line

    def test_story_of(self):
        g = RelationshipGraph()
        g.interact("a", "b", "deep_conversation")
        story = g.story_of("a", "b", name_a="Zara", name_b="Kilo")
        assert "Zara" in story and "Kilo" in story
        assert "→" in story

    def test_triad(self):
        g = RelationshipGraph()
        g.interact("a", "c", "betrayed")
        g.interact("b", "c", "betrayed")
        t = g.triad("a", "b", "c")
        assert 0.0 <= t["tension"] <= 1.0
        assert t["note"]

    def test_new_events_exist(self):
        for ev in ("secret_shared", "defended", "abandoned", "celebrated",
                   "forgave", "opened_up", "stood_up_to", "missed_them"):
            assert ev in EVENT_DELTAS

    def test_edge_roundtrip_with_transitions(self):
        e = Edge(a="a", b="b")
        e.transitions.append({"from": "friend", "to": "rival", "ts": 1.0})
        back = Edge.from_dict(e.to_dict())
        assert back.transitions[0]["to"] == "rival"


# ── drama director + scene formats ────────────────────────────────────

class TestDramaDirector:
    def test_energy_drops_on_short_turns(self):
        d = DramaDirector()
        start = d.energy
        for _ in range(6):
            d.observe("Zara", "ok")
        assert d.energy < start

    def test_direct_fires_when_flat(self):
        d = DramaDirector(energy=0.3)
        sc = Scene(title="T", topic="podcasts", participants=["a", "b", "c"])
        steer = d.direct(sc, round_no=5)
        assert steer, "a flat scene should get a steer"
        assert d.interventions == 1
        assert d.fired

    def test_healthy_scene_no_intervention(self):
        d = DramaDirector(energy=0.7)
        sc = Scene(title="T", topic="x", participants=["a"])
        assert d.direct(sc, round_no=0) == ""

    def test_cooldown(self):
        d = DramaDirector(energy=0.2)
        sc = Scene(title="T", topic="x", participants=["a", "b", "c"])
        first = d.direct(sc, round_no=0)
        assert first
        # same round: the fired beat is cooling down, but another may fire;
        # at minimum the SAME beat must not fire twice in a row
        second = d.direct(sc, round_no=0)
        assert second != first or second == ""

    def test_run_scene_formats(self):
        chars = [make_char(name=n) for n in ("Zara", "Kilo", "Jax")]
        for fmt in SCENE_FORMATS:
            sc = run_scene(chars, "T", "music", suggest=None, rounds=2,
                           format=fmt, seed=7)
            assert sc.turns, fmt
            speakers = {t.speaker for t in sc.turns}
            assert speakers & {"Zara", "Kilo", "Jax"}

    def test_run_scene_with_director(self):
        chars = [make_char(name=n) for n in ("Zara", "Kilo")]
        d = DramaDirector(energy=0.25)
        sc = run_scene(chars, "T", "music", suggest=None, rounds=3,
                       director=d, seed=3)
        assert sc.turns
        assert isinstance(d.report()["energy"], float)

    def test_debate_and_roast_markers(self):
        chars = [make_char(name=n) for n in ("Zara", "Kilo", "Jax")]
        sc = run_scene(chars, "T", "jollof", suggest=None, rounds=1,
                       format="debate", seed=1)
        assert any(t.speaker == "Devon (director)" for t in sc.turns)
        sc2 = run_scene(chars, "T", "jollof", suggest=None, rounds=4,
                        format="roast", seed=1)
        # roast target (last char) gets rebuttal turns
        assert any(t.speaker == "Jax" for t in sc2.turns)

    def test_podcast_with_director(self):
        h = make_char(name="Zara", roles=["podcast_host"])
        g = make_char(name="Kilo")
        ep = podcast_episode(h, [g], "afrobeats", suggest=None, rounds=2,
                             director=DramaDirector(), seed=5)
        assert ep.turns and ep.turns[0].speaker == "Zara"


# ── proactive pulse ───────────────────────────────────────────────────

class TestProactive:
    def test_silent_when_recent(self):
        c = make_char()
        c.last_active = time.time()
        assert proactive_pulse(c, None, rng=FixedRng(0.0)) is None

    def test_reaches_out_when_idle(self):
        c = make_char()
        c.last_active = time.time() - 100 * 3600
        c.mood["trust"] = 0.9
        line = proactive_pulse(c, None, rng=FixedRng(0.0))
        assert line, "idle + willing rng should initiate"
        assert "[" not in line

    def test_icebreakers(self):
        c = make_char()
        sparks = icebreakers(c)
        assert sparks and all(isinstance(s, str) for s in sparks)


# ── match intro ───────────────────────────────────────────────────────

class TestMatchIntro:
    def test_intro_lines(self):
        seats = [AgentSeat(kind="brain", name="Devon"),
                 AgentSeat(kind="character", character=make_char(name="Zara")),
                 AgentSeat(kind="character",
                           character=make_char(name="Jax",
                                               persona={"competitive": 0.95}))]
        lines = match_intro("auction", seats, suggest=None,
                            rng=random.Random(1))
        assert lines[0].startswith("🎮 auction")
        assert any("Zara" in l for l in lines)
        assert any("Devon" in l for l in lines)

    def test_rivalry_callout(self):
        z = make_char(name="Zara")
        j = make_char(name="Jax")
        g = RelationshipGraph()
        for _ in range(12):
            g.interact(z.id, j.id, "betrayed")
            g.interact(j.id, z.id, "betrayed")
        seats = [AgentSeat(kind="character", character=z),
                 AgentSeat(kind="character", character=j)]
        lines = match_intro("chess", seats, suggest=None, graph=g,
                            rng=random.Random(1))
        assert any("⚔️" in l for l in lines)


# ── voice: stylometry ─────────────────────────────────────────────────

class TestVoice:
    UTTERANCES = [
        "Okay but really, what do you think about the new highlife revival?",
        "I interviewed three street vendors today and they all said the same thing.",
        "Say less, we're getting into it — the mic is hot and I'm ready.",
        "The best question is the one you're scared to ask, so ask it.",
        "Everyone's interesting if you shut up and listen for five minutes.",
        "We need to talk about Lagos traffic because it is a whole personality.",
    ]

    def test_fingerprint_has_stylometry_fields(self):
        fp = fingerprint(self.UTTERANCES, ["say less"])
        assert fp.n_samples == 6
        assert fp.function_words, "function-word profile must be populated"
        assert 0.0 < fp.ttr <= 1.0
        assert fp.mean_word_len > 0

    def test_function_word_profile(self):
        prof = function_word_profile(self.UTTERANCES)
        assert isinstance(prof, dict) and prof
        assert all(v >= 0 for v in prof.values())

    def test_style_report(self):
        fp = fingerprint(self.UTTERANCES)
        rep = style_report("Zara", fp)
        assert "Zara" in rep and "Rhythm" in rep

    def test_style_report_thin(self):
        rep = style_report("New", fingerprint(["hi"]))
        assert "not enough" in rep

    def test_distinctness_range(self):
        a = fingerprint(self.UTTERANCES)
        b = fingerprint(["Indeed. The epistemological framework suggests otherwise."] * 6)
        d = distinctness(a, b)
        assert 0.0 <= d <= 1.0

    def test_consistency_needs_data(self):
        assert consistency(fingerprint(["one"]), "hello") == 0.5


# ── render layer ──────────────────────────────────────────────────────

class TestRender:
    def test_card_rich(self):
        card = render_card(make_char(), theme="rich")
        assert "Zara" in card and "╭" in card and "OCEAN" in card

    def test_card_minimal(self):
        card = render_card(make_char(), theme="minimal")
        assert "Zara" in card and "╭" not in card

    def test_cast(self):
        c = make_char()
        out = render_cast([(c, 0.9, "natural host")], title="Tonight")
        assert "Zara" in out and "Tonight" in out

    def test_scene_render(self):
        sc = Scene(title="Late Night", topic="music")
        sc.add("Zara", "Welcome back to the show.")
        sc.add("Kilo", "Noted.")
        out = render_scene(sc)
        assert "Late Night" in out and "Zara" in out and "end scene" in out

    def test_web(self):
        g = RelationshipGraph()
        g.interact("a", "b", "deep_conversation")
        out = render_web(g, names={"a": "Zara", "b": "Kilo"})
        assert "Zara" in out and "Kilo" in out

    def test_arc_render(self):
        c = make_char()
        c.reflect(suggest=None)
        out = render_arc(c)
        assert "Zara" in out


# ── arcs ──────────────────────────────────────────────────────────────

class TestArcs:
    def test_arc_story(self):
        c = make_char()
        add_belief(c, "Real ones rematch.", 0.9)
        challenge_belief(c, "Real ones rematch.", 0.5, "lost badly")
        challenge_belief(c, "Real ones rematch.", 0.4, "lost again")
        milestone(c, "First live show", "nerves of steel")
        story = arc_story(c)
        assert "Zara" in story
        assert "revised 2" in story

    def test_arc_summary_still_works(self):
        c = make_char()
        assert "Zara" in arc_summary(c)


# ── casting additions ─────────────────────────────────────────────────

class TestCasting:
    def chars(self):
        return [make_char(name="Zara", roles=["podcast_host"],
                          skills={"interviewing": 0.9}),
                make_char(name="Jax", persona={"competitive": 0.95},
                          skills={"gaming": 0.9}),
                make_char(name="Elder", persona={"wise": 0.95, "calm": 0.9},
                          roles=["sage"])]

    def test_rotation_penalty(self):
        chars = self.chars()
        plain = {c.name: s for c, s, _ in
                 cast_for(chars, role="podcast_host", n=3)}
        rotated = {c.name: s for c, s, _ in
                   cast_for(chars, role="podcast_host", n=3,
                            recent={"Zara"})}
        assert rotated["Zara"] < plain["Zara"]

    def test_cast_against(self):
        chars = self.chars()
        bad = cast_against(chars, avoid_traits={"competitive": 1.0}, n=3)
        assert bad[0][0].name == "Jax"  # most competitive = worst pick
        assert bad[0][1] >= bad[-1][1]

    def test_cast_for_audience(self):
        chars = self.chars()
        picks = cast_for_audience(chars, {"energetic": 1.0}, n=2)
        assert len(picks) == 2
        assert picks[0][0].name == "Zara"  # most energetic


# ── store additions ───────────────────────────────────────────────────

class TestStore:
    def test_find(self, tmp_path):
        s = CharacterStore(tmp_path)
        s.save(make_char(name="Zara", roles=["podcast_host"],
                         skills={"interviewing": 0.9}))
        s.save(make_char(name="Jax", roles=["gamer"]))
        assert [c.name for c in s.find(role="podcast_host")] == ["Zara"]
        assert [c.name for c in s.find(skill="interviewing",
                                      min_value=0.8)] == ["Zara"]
        assert [c.name for c in s.find(name_contains="ax")] == ["Jax"]

    def test_export_import_roundtrip(self, tmp_path):
        s = CharacterStore(tmp_path)
        s.save(make_char(name="Zara"))
        data = s.export_all()
        s2 = CharacterStore(tmp_path / "other")
        stats = s2.import_all(data)
        assert stats["created"] == 1
        assert s2.get_by_name("Zara").ocean["extraversion"] > 0.5
        # idempotent re-import
        stats2 = s2.import_all(data)
        assert stats2["skipped"] == 1


# ── processing additions ──────────────────────────────────────────────

class TestProcessing:
    def test_reflection_hook_fires(self):
        z = make_char(name="Zara")
        z.remember("a huge fight with Kilo about the show", 0.9)
        z.remember("we made up and the show was better than ever", 0.9)
        g = RelationshipGraph()
        evs = [SessionEvent(z.id, "kilo", "argument", salience=0.9,
                            note="big fight about the show"),
               SessionEvent(z.id, "kilo", "made_up", salience=0.9,
                            note="made up after the show")]
        rep = process_session(evs, {z.id: z}, g, suggest=None)
        assert rep["processed"] == 2
        assert rep["reflections"] >= 1
        assert z.reflections

    def test_no_reflection_for_trivia(self):
        z = make_char(name="Zara")
        g = RelationshipGraph()
        evs = [SessionEvent(z.id, "x", "good_conversation", salience=0.3)]
        rep = process_session(evs, {z.id: z}, g)
        assert rep["reflections"] == 0

    def test_drift_flag(self, monkeypatch):
        z = make_char(name="Zara")
        z.voice_fingerprint = {"n_samples": 10, "top_words": [],
                               "mean_sentence_len": 12.0,
                               "sentence_len_spread": 2.0, "emoji_rate": 0.0,
                               "question_rate": 0.1, "exclaim_rate": 0.0,
                               "catchphrase_hits": 0.0, "function_words": [],
                               "ttr": 0.5, "mean_word_len": 4.5}
        monkeypatch.setattr("nomorals.characters.voice.consistency",
                            lambda fp, u, cp=(): 0.1)
        g = RelationshipGraph()
        evs = [SessionEvent(z.id, "x", "good_conversation", salience=0.5,
                            note="lol | haha ok")]
        rep = process_session(evs, {z.id: z}, g)
        assert rep["drift_flags"], "low consistency must flag drift"
        assert "drift_flag" in z.voice_fingerprint

    def test_context_builder_surfaces_drift(self):
        c = make_char()
        c.voice_fingerprint = {"drift_flag": "sounding off",
                               "n_samples": 10}
        built = CharacterContextBuilder().build(c)
        assert "sounding" in built or "off" in built

    def test_context_other_block(self):
        z, k = make_char(name="Zara"), make_char(name="Kilo")
        g = RelationshipGraph()
        g.interact(z.id, k.id, "deep_conversation")
        built = CharacterContextBuilder().build(z, graph=g, other=k)
        assert "Kilo" in built


# ── memory additions ──────────────────────────────────────────────────

class TestCharacterMemory:
    def test_consolidate_merges_dupes(self):
        c = make_char()
        c.remember("the show was great", 0.6)
        c.remember("the show was great", 0.8)
        c.remember("totally different thing", 0.5)
        mem = CharacterMemory.for_character(c, db=None)
        stats = mem.consolidate()
        assert stats["merged"] == 1
        assert len(c.memory) == 2
        # stronger salience kept
        assert c.memory[0]["salience"] == 0.8 or \
            c.memory[1]["salience"] == 0.8

    def test_flat_recall_uses_three_factor(self):
        c = make_char()
        c.memory = []
        c.memory.append({"ts": time.time() - 30 * 86400,
                         "text": "old irrelevant", "salience": 1.0})
        c.memory.append({"ts": time.time(), "text": "fresh podcast news",
                         "salience": 0.6})
        mem = CharacterMemory.for_character(c, db=None)
        out = mem.recall("podcast news")
        assert out and "podcast" in out[0]


# ── seeds enriched ────────────────────────────────────────────────────

class TestSeeds:
    def test_seeds_have_depth(self):
        from nomorals.characters.seeds import seed_characters
        chars = seed_characters()
        assert len(chars) == 6
        zara = next(c for c in chars if c.name == "Zara")
        assert zara.insecurities and zara.stage_of_life and zara.interests
        assert zara.ocean["extraversion"] > 0.7
        assert "Personality profile:" in zara.persona_block()
        assert "never admit" in zara.persona_block()

    def test_seed_bank_idempotent(self, tmp_path):
        from nomorals.characters.seeds import seed_bank
        store = CharacterStore(tmp_path)
        r1 = seed_bank(store)
        assert len(r1["created"]) == 6
        r2 = seed_bank(store)
        assert r2["created"] == [] and len(r2["skipped"]) == 6


# ── dialogue still sound ──────────────────────────────────────────────

class TestDialogue:
    def test_converse_no_model(self):
        c = make_char()
        dlg = converse("You are Devon.", c, "brain: how was the show?",
                       suggest=None, rounds=2)
        assert len(dlg.turns) >= 3
        assert dlg.turns[0].speaker == "Devon"

    def test_character_initiate_no_model(self):
        line = character_initiate(make_char(), None)
        assert line and "[" not in line
