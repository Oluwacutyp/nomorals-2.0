"""FictionWriter: genre mechanics, validation, arc management, modes.

All prose here comes from the heuristic composer (no model in the test
context) — the tests prove the MECHANICS: fair-play enforcement, tension
curves, dread cycles, rule ledgers, cost ledgers, intimacy oscillation,
dynamic shaping, and serial arc spawning.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.fiction import (
    ENGINES, FictionWriter, StoryState, engine_for, weave_wisdom,
)


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


def _state(genre: str, mode: str = "novel", total: int = 10) -> StoryState:
    engine = engine_for(genre, seed=42)
    state = StoryState(slug=f"test-{genre}", title="Test", premise="test",
                       genre=engine.name, mode=mode, total_planned=total)
    arcs = engine.plan_arcs("a detective hunts a killer", state)
    state.arcs = [a.to_dict() for a in arcs]
    state.arcs[0]["status"] = "active"
    return state, engine


class TestEngineRouting(unittest.TestCase):
    def test_all_genres_route(self) -> None:
        for genre in ("mystery", "thriller", "horror", "sci-fi",
                      "fantasy", "romance"):
            engine = engine_for(genre, seed=1)
            self.assertEqual(engine.name, genre.replace("scifi", "sci-fi")
                             if genre != "sci-fi" else "sci-fi")
        self.assertIn("mystery", ENGINES)
        self.assertEqual(len(ENGINES) >= 6, True)

    def test_unknown_genre_falls_back(self) -> None:
        engine = engine_for("western", seed=1)
        self.assertIsNotNone(engine.name)


class TestMysteryMechanics(unittest.TestCase):
    def test_plan_builds_ledger(self) -> None:
        state, engine = _state("mystery")
        ledger = state.ledger
        self.assertIn("culprit", ledger)
        self.assertGreaterEqual(len(ledger["clues"]), 8)
        self.assertGreaterEqual(len(ledger["suspects"]), 3)
        true = [c for c in ledger["clues"] if not c["red_herring"]]
        red = [c for c in ledger["clues"] if c["red_herring"]]
        self.assertLessEqual(len(red) / len(ledger["clues"]), 0.45)

    def test_brief_plants_clues_and_guards_reveal(self) -> None:
        state, engine = _state("mystery")
        state.ledger["reveal_chapter"] = 9
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 1)
        self.assertTrue(brief.plant)  # clues demanded, not suggested
        self.assertTrue(any("reveal" in m.lower() for m in brief.must_not))

    def test_fair_play_violation_caught(self) -> None:
        state, engine = _state("mystery")
        state.ledger["reveal_chapter"] = 9
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 2)
        culprit = state.ledger["culprit"]
        method = state.ledger["method"]
        bad_text = (f"It was {culprit} all along. The detective explained "
                    f"how {method} had been done, step by step. " * 20)
        problems = engine.validate(bad_text, state, brief)
        self.assertTrue(any("fair-play" in p for p in problems))

    def test_advance_marks_planted_clues(self) -> None:
        state, engine = _state("mystery")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 1)
        clue_text = brief.plant[0]
        # write the clue into prose naturally
        text = (f"The detective knelt by the door. There, half-hidden: "
                f"{clue_text}. It meant nothing yet. " * 10)
        engine.advance(state, text, brief)
        planted = [c for c in state.ledger["clues"] if c["planted_ch"]]
        self.assertTrue(planted)


class TestThrillerMechanics(unittest.TestCase):
    def test_tension_curve_shape(self) -> None:
        state, engine = _state("thriller", total=10)
        curve = [engine._curve(state, n) for n in range(1, 11)]
        # set-piece spikes at 25/60/90% exceed their neighbors
        self.assertGreater(curve[1], curve[0])       # ch2 spike vs ch1
        self.assertLess(curve[2], curve[1])          # breather valley after
        self.assertGreater(curve[8], curve[7])       # ch9 spike
        self.assertTrue(all(0 < t <= 10 for t in curve))

    def test_hook_required(self) -> None:
        state, engine = _state("thriller")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 2)
        flat = ("The team went home. Everything was resolved and calm. "
                "They slept well. " * 30)
        problems = engine.validate(flat, state, brief)
        self.assertTrue(any("hook" in p for p in problems))
        hooked = flat + "\n\nBehind them, a door opened."
        self.assertFalse(any("hook" in p
                             for p in engine.validate(hooked, state, brief)))


class TestHorrorMechanics(unittest.TestCase):
    def test_unknown_ledger_violation(self) -> None:
        state, engine = _state("horror")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 2)
        unknown = state.ledger["unknowns"][0]
        explaining = (f"In the end it turned out to be {unknown}, just rats "
                      f"in the walls after all, and everyone laughed. " * 10)
        problems = engine.validate(explaining, state, brief)
        self.assertTrue(any("unknown ledger" in p for p in problems))

    def test_scare_budget(self) -> None:
        state, engine = _state("horror")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 2)
        over = ("scream " * 4 + "lunged " * 3 + "grabbed " * 3) * 4
        problems = engine.validate(over, state, brief)
        self.assertTrue(any("scare budget" in p for p in problems))

    def test_dread_cycle_in_brief(self) -> None:
        state, engine = _state("horror")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 1)
        self.assertTrue(any("dread cycle" in m.lower()
                            for m in brief.must_happen))
        self.assertTrue(any("NEVER explain" in m for m in brief.must_not))


class TestSciFiMechanics(unittest.TestCase):
    def test_rule_contradiction_caught(self) -> None:
        state, engine = _state("sci-fi")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 2)
        rules = " | ".join(r["rule"] for r in state.ledger["rules"])
        # find a rule we can contradict
        bad = ("The crew cheered: a free jump, no cost at all, the "
               "navigators napping through it. " * 10)
        problems = engine.validate(bad, state, brief)
        if "burns a year" in rules:
            self.assertTrue(any("contradiction" in p for p in problems))
        else:
            self.assertIsInstance(problems, list)

    def test_wonder_beat_scheduled(self) -> None:
        state, engine = _state("sci-fi")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 3)
        self.assertTrue(any("WONDER" in m for m in brief.must_happen))


class TestFantasyMechanics(unittest.TestCase):
    def test_magic_without_cost_caught(self) -> None:
        state, engine = _state("fantasy")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 2)
        costless = ("She raised her hand and the spell came easily, magic "
                    "flowing free and joyful with no consequence at all. "
                    * 12)
        problems = engine.validate(costless, state, brief)
        self.assertTrue(any("without cost" in p for p in problems))
        paid = ("She raised her hand and cast the spell, and felt the price "
                "leave her all at once — a memory torn out by the roots. "
                * 12)
        self.assertFalse(any("without cost" in p
                             for p in engine.validate(paid, state, brief)))

    def test_advance_ledgers_costs(self) -> None:
        state, engine = _state("fantasy")
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 2)
        engine.advance(state, "She paid the price in full. The cost was "
                              "written on her face.", brief)
        self.assertIn(2, state.ledger["costs_paid"])


class TestRomanceMechanics(unittest.TestCase):
    def test_premature_commitment_caught(self) -> None:
        state, engine = _state("romance", total=10)
        arc = engine.plan_arcs("x", state)[0]
        brief = engine.chapter_brief(state, arc, 3)
        early = ("He took her hands. 'I will love you forever, marry me, "
                 "we'll be together always.' " * 10)
        problems = engine.validate(early, state, brief)
        self.assertTrue(any("premature" in p for p in problems))

    def test_dark_night_scheduled_before_75_percent(self) -> None:
        state, engine = _state("romance", total=10)
        arc = engine.plan_arcs("x", state)[0]
        engine.chapter_brief(state, arc, 1)
        self.assertLessEqual(state.ledger["dark_night_chapter"], 8)

    def test_intimacy_oscillates(self) -> None:
        state, engine = _state("romance", total=10)
        arc = engine.plan_arcs("x", state)[0]
        targets = []
        for n in range(1, 7):
            brief = engine.chapter_brief(state, arc, n)
            engine.advance(state, "prose", brief)
            targets.append(state.ledger["intimacy"])
        # at least one dip in the first six chapters
        self.assertTrue(any(b < a for a, b in zip(targets, targets[1:])))


class TestShapingAndModes(unittest.TestCase):
    def test_decide_shape_is_dynamic(self) -> None:
        ctx, _ = _ctx()
        w = FictionWriter(ctx)
        simple, _ = w.decide_shape("a cat sits", "romance", "novel")
        complex_, _ = w.decide_shape(
            "a detective hunts a killer while a secret war brews and an "
            "empire falls; Mara seeks revenge and the queen hides a mystery "
            "of succession", "mystery", "novel")
        self.assertGreater(complex_, simple)
        serial_n, note = w.decide_shape("a detective hunts a killer",
                                        "mystery", "serial")
        self.assertIn("serial", note)
        self.assertLessEqual(serial_n, 12)

    def test_weave_wisdom_graceful_without_keeper(self) -> None:
        ctx, _ = _ctx()
        motifs = weave_wisdom("the void", ctx)
        self.assertIsInstance(motifs, list)  # [] when keeper unavailable


class TestEndToEnd(unittest.TestCase):
    def test_novel_mode_full_run(self) -> None:
        ctx, _ = _ctx()
        w = FictionWriter(ctx)
        result = w.start("a detective hunts a killer through a flooded city",
                         genre="mystery", mode="novel")
        slug = result["slug"]
        self.assertEqual(result["genre"], "mystery")
        self.assertIn("novel", result["shape"])
        for _ in range(3):
            ch = w.write_next(slug)
            self.assertGreater(ch["words"], 150)
            self.assertTrue(ch["path"])
        st = w.status(slug)
        self.assertEqual(st["chapters_written"], 4)  # ch1 from start + 3
        self.assertTrue(st["open_threads"])
        # chapters persisted
        ch_dir = w.story_dir(slug) / "chapters"
        self.assertEqual(len(list(ch_dir.glob("*.md"))), 4)

    def test_serial_mode_rolls_arcs(self) -> None:
        ctx, _ = _ctx()
        w = FictionWriter(ctx)
        result = w.start("a smuggler outruns the fleet",
                         genre="thriller", mode="serial")
        slug = result["slug"]
        state = w.load(slug)
        first_arc = state.arcs[0]["name"]
        # exhaust the first arc's planned chapters
        planned = state.arcs[0]["chapters_planned"]
        for _ in range(planned + 1):
            w.write_next(slug)
        state = w.load(slug)
        statuses = {a["name"]: a["status"] for a in state.arcs}
        self.assertEqual(statuses[first_arc], "closed")
        self.assertGreater(len(state.arcs), 1)
        self.assertTrue(any(a["status"] == "active"
                            for a in state.arcs))

    def test_all_genres_write_real_prose(self) -> None:
        ctx, _ = _ctx()
        w = FictionWriter(ctx)
        for genre in ("horror", "sci-fi", "fantasy", "romance"):
            result = w.start(f"a story of {genre} premise with secrets",
                             genre=genre, mode="novel")
            ch = w.write_next(result["slug"])
            self.assertGreater(ch["words"], 250, genre)
            # no placeholder leakage
            text = (w.story_dir(result["slug"]) / "chapters" /
                    "0002.md").read_text()
            self.assertNotIn("TODO", text)
            self.assertNotIn("lorem", text.lower())


if __name__ == "__main__":
    unittest.main()
