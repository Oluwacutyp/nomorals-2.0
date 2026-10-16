"""Step 10 Game tests — one invariant test per new module."""

from __future__ import annotations

import random
import unittest


class TestDeclarative(unittest.TestCase):
    def test_catalog_has_12_affixes_and_validates(self):
        from nomorals.games.declarative import AFFIX_CATALOG, ROLE_POOL, validate_catalog
        self.assertEqual(len(AFFIX_CATALOG), 12)
        self.assertEqual(len(ROLE_POOL), 8)
        self.assertEqual(validate_catalog(), [])

    def test_solver_never_returns_incompatibles_or_repeats(self):
        from nomorals.games.declarative import AFFIX_CATALOG, solve
        rng = random.Random(7)
        picked = solve(list(AFFIX_CATALOG), 3, rng=rng, history=["bronze"],
                       level=80, slot="weapon")
        slugs = [d.slug for d in picked]
        self.assertEqual(len(set(slugs)), len(slugs))
        self.assertNotIn("bronze", slugs)  # anti-repeat: fresh pick exists
        for a in picked:
            for b in picked:
                if a is not b:
                    self.assertNotIn(b.slug, a.incompatible_with)

    def test_skill_descs_come_from_live_catalog(self):
        from nomorals.games.declarative import skill_descs_from_catalog
        from nomorals.games.skills import SKILL_CATALOG
        descs = skill_descs_from_catalog()
        self.assertEqual({d.slug for d in descs}, set(SKILL_CATALOG))


class TestTelemetry(unittest.TestCase):
    def test_tuning_derives_from_fights_not_vibes(self):
        from nomorals.games.telemetry import (
            FightTelemetry, TelemetryStore, difficulty_recommendation)
        store = TelemetryStore()
        self.assertEqual(difficulty_recommendation(store.stats())["action"], "hold")
        for i in range(12):
            store.record(FightTelemetry(f"f{i}", "brawler", True, 30.0, 100, 10,
                                        ("jab",)))
        rec = difficulty_recommendation(store.stats())
        self.assertEqual(rec["action"], "raise")
        store2 = TelemetryStore()
        for i in range(12):
            store2.record(FightTelemetry(f"f{i}", "brawler", i < 2, 300.0, 10, 100))
        self.assertEqual(difficulty_recommendation(store2.stats())["action"], "lower")


class TestTells(unittest.TestCase):
    def test_single_grammar_valid(self):
        from nomorals.games.tells import render_tell, tell_for_move, validate_grammar
        self.assertEqual(validate_grammar(), [])
        t1 = render_tell("charge", "charging!")
        t2 = tell_for_move("charge_attack", text="charging!")
        self.assertEqual(t1, t2)  # same move → same icon/color/motion everywhere
        self.assertEqual(t1["icon"], "⚡")
        with self.assertRaises(ValueError):
            render_tell("nope")


class TestCounters(unittest.TestCase):
    def test_every_move_has_a_counter_and_discovery_gates(self):
        from nomorals.games.counters import (
            ENEMY_MOVES, counter_for, discover_counter, known_counters,
            validate_codex)
        self.assertEqual(validate_codex(), [])
        for move in ENEMY_MOVES:
            self.assertTrue(counter_for(move))
        knowledge: set[str] = set()
        self.assertEqual(known_counters(knowledge, "heavy_strike"), [])
        discover_counter(knowledge, "sidestep")
        self.assertEqual([c.slug for c in known_counters(knowledge, "heavy_strike")],
                         ["sidestep"])
        self.assertIsNone(discover_counter(knowledge, "not-a-counter"))


class TestGhosts(unittest.TestCase):
    def test_rank_is_percentile_of_recorded_runs_not_elo(self):
        from nomorals.games.ghosts import GhostRun, rank_vs_ghosts, seed_brackets
        self.assertIsNone(rank_vs_ghosts(100, []))  # no population → no rating
        runs = [GhostRun("a", "b", 10, 60.0), GhostRun("b", "b", 20, 50.0),
                GhostRun("c", "b", 30, 40.0)]
        self.assertEqual(rank_vs_ghosts(25, runs), 66.7)
        brackets = seed_brackets(runs)
        self.assertEqual([b.name for b in brackets], ["bronze", "silver", "gold"])
        self.assertEqual(sum(len(b.runs) for b in brackets), 3)


class TestNovelty(unittest.TestCase):
    def test_novelty_pays_once(self):
        from nomorals.games.novelty import NoveltyLedger, compute_novelty_rewards
        ledger = NoveltyLedger()
        first = compute_novelty_rewards(ledger, build="brawler",
                                        skills_used=["jab", "uppercut"], won=True)
        self.assertEqual(first["coins"], 50 + 10 + 10)
        second = compute_novelty_rewards(ledger, build="brawler",
                                         skills_used=["jab"], won=True)
        self.assertEqual(second["coins"], 0)  # same build, same skill: no re-pay
        loss = compute_novelty_rewards(ledger, build="new-build",
                                       skills_used=["kick"], won=False)
        self.assertEqual(loss["coins"], 10)  # skill pays on first use even in loss
        self.assertNotIn("first win", " ".join(loss["reasons"]))


class TestPreemption(unittest.TestCase):
    def test_state_identical_across_preemption(self):
        from nomorals.games.preemption import PreemptibleRun, preempt
        run = PreemptibleRun({"turn": 3, "hp": 42, "hand": ["a", "b"]})
        before = dict(run.state)
        result = preempt(run, lambda: "real answer")
        self.assertEqual(result, "real answer")
        self.assertEqual(run.state, before)
        self.assertEqual(run.status, "running")

    def test_request_runs_while_paused(self):
        from nomorals.games.preemption import PreemptibleRun, preempt
        run = PreemptibleRun({"turn": 1})
        seen = []
        preempt(run, lambda: seen.append(run.status))
        self.assertEqual(seen, ["paused"])  # real request preempts before running


if __name__ == "__main__":
    unittest.main()
