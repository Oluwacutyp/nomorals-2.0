"""Case game god-tier pass: the 52-case library, anti-repeat dealing,
skill-adaptive clues, hints, timed mode, scoring, and achievements.

Companion to test_case_godtier.py, which covers the generator's
airtightness and the lighter content pools.
"""
from __future__ import annotations

import random
import unittest

from nomorals.games.achievements import get_achievements
from nomorals.games.engine import GameEngine
from nomorals.games.games.cases import (
    BANK_SIZE,
    CASES,
    HINT_COST,
    HISTORY_VERSION,
    TIME_BONUS_MAX,
    TIME_LIMITS,
    TIER_BASE_SCORE,
    adapt_case,
    blank_history,
    deal_case,
    eligible_tiers,
    generate_case,
    record_played,
    score_solve,
    solve_rate,
    unseen_bank_cases,
    verify_solvability,
)
from nomorals.games.players import Player
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db: Database) -> None:
        self.db = db


ADA = Player.from_sender("telegram", "456", "Ada")


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


def bank_cases(tier: str) -> list[dict]:
    return [c for c in CASES if c["tier"] == tier]


# ── the library ───────────────────────────────────────────────────────────

class LibraryTests(unittest.TestCase):
    def test_fifty_two_cases(self):
        self.assertEqual(BANK_SIZE, 52)
        self.assertEqual(len(CASES), 52)
        ids = [c["id"] for c in CASES]
        self.assertEqual(len(set(ids)), 52)

    def test_tier_spread(self):
        counts: dict[str, int] = {}
        for c in CASES:
            counts[c["tier"]] = counts.get(c["tier"], 0) + 1
        self.assertGreaterEqual(counts.get("easy", 0), 10)
        self.assertGreaterEqual(counts.get("medium", 0), 10)
        self.assertGreaterEqual(counts.get("hard", 0), 5)
        self.assertGreaterEqual(counts.get("expert", 0), 5)

    def test_every_case_fair_play_verified(self):
        for c in CASES:
            with self.subTest(case=c.get("id")):
                self.assertEqual(verify_solvability(c), [])

    def test_god_tier_fields_present(self):
        required = ("id", "title", "tier", "briefing", "story", "suspects",
                    "motives", "culprit", "clues", "red_herrings",
                    "herring_suspect", "solution", "statements", "difficulty")
        for c in CASES:
            with self.subTest(case=c.get("id")):
                for f in required:
                    self.assertIn(f, c, f)
                self.assertIn(c["tier"], ("easy", "medium", "hard", "expert"))
                self.assertIn(c["difficulty"], ("easy", "medium", "hard"))
                self.assertEqual(set(c["motives"]), set(c["suspects"]))
                self.assertTrue(c["red_herrings"])
                self.assertIn(c["herring_suspect"], c["suspects"])
                self.assertNotEqual(c["herring_suspect"], c["culprit"])
                self.assertTrue(c["solution"])
                self.assertTrue(c["title"])

    def test_expert_cases_map_to_hard_difficulty(self):
        # back-compat: the engine-facing difficulty stays 3-valued
        for c in bank_cases("expert"):
            self.assertEqual(c["difficulty"], "hard")

    def test_generated_case_full_schema(self):
        for tier in ("easy", "medium", "hard", "expert"):
            case = generate_case(random.Random(4242), tier)
            with self.subTest(tier=tier):
                self.assertEqual(case["tier"], tier)
                self.assertEqual(verify_solvability(case), [])
                self.assertTrue(case["title"])
                self.assertTrue(case["briefing"])
                self.assertEqual(set(case["motives"]), set(case["suspects"]))
                self.assertTrue(case["red_herrings"])
                self.assertIn(case["herring_suspect"], case["suspects"])
                self.assertNotEqual(case["herring_suspect"], case["culprit"])
                self.assertTrue(case["solution"])


# ── anti-repeat dealing ───────────────────────────────────────────────────

class AntiRepeatTests(unittest.TestCase):
    def test_no_bank_repeat_until_pool_exhausted(self):
        hist = blank_history()
        rng = random.Random(7)
        ids = []
        for _ in range(20):
            case, tier, reshuffled = deal_case(rng, hist)
            self.assertFalse(reshuffled)
            ids.append(case["id"])
            record_played(hist, case["id"], tier, True)
        bank_ids = [i for i in ids if i.startswith("bank:")]
        self.assertEqual(len(set(bank_ids)), len(bank_ids))

    def test_exhaustion_reshuffles(self):
        hist = blank_history()
        for tier in ("easy", "medium"):
            for c in bank_cases(tier):
                record_played(hist, c["id"], tier, True)
        rng = random.Random(1)
        case, tier, reshuffled = deal_case(rng, hist)
        self.assertTrue(reshuffled)
        self.assertTrue(case["id"])
        self.assertEqual(verify_solvability(case), [])

    def test_reshuffle_records_and_resets(self):
        hist = blank_history()
        easy = bank_cases("easy")
        for c in easy:
            record_played(hist, c["id"], "easy", True)
        record_played(hist, "bank:0", "easy", True, reshuffled=True)
        self.assertEqual(hist["reshuffles"], 1)
        self.assertEqual(hist["seen"]["easy"], ["bank:0"])

    def test_unseen_bank_cases(self):
        easy = bank_cases("easy")
        seen = [c["id"] for c in easy[:-1]]
        unseen = unseen_bank_cases("easy", seen)
        self.assertEqual([c["id"] for c in unseen], [easy[-1]["id"]])
        self.assertEqual(unseen_bank_cases("easy", seen + [easy[-1]["id"]]),
                         [])


# ── skill adaptation ──────────────────────────────────────────────────────

class SkillAdaptTests(unittest.TestCase):
    def test_new_player_gets_easy_and_medium_only(self):
        self.assertEqual(eligible_tiers(blank_history()), ("easy", "medium"))

    def test_tiers_unlock_with_record(self):
        hist = blank_history()
        hist["attempts"] = {"easy": 2, "medium": 2}
        hist["solved"] = {"easy": 1, "medium": 1}
        self.assertEqual(eligible_tiers(hist), ("easy", "medium", "hard"))
        hist["attempts"] = {"easy": 3, "medium": 3}
        hist["solved"] = {"easy": 2, "medium": 2}
        self.assertEqual(eligible_tiers(hist),
                         ("easy", "medium", "hard", "expert"))

    def test_struggling_player_gets_assist_clue(self):
        hist = blank_history()
        hist["attempts"] = {"medium": 4}
        hist["solved"] = {"medium": 0}
        case = bank_cases("medium")[0]
        out = adapt_case(case, random.Random(0), hist)
        self.assertTrue(out["assisted"])
        self.assertEqual(len(out["clues"]), 6)
        self.assertIn("background check", out["clues"][-2].lower())
        self.assertEqual(out["clues"][-1], case["clues"][-1])
        self.assertEqual(verify_solvability(out), [])

    def test_sharp_player_gets_extra_herrings(self):
        hist = blank_history()
        hist["attempts"] = {"medium": 6}
        hist["solved"] = {"medium": 6}
        case = bank_cases("medium")[0]
        out = adapt_case(case, random.Random(0), hist)
        self.assertTrue(out["sharpened"])
        self.assertEqual(len(out["clues"]), 7)
        self.assertEqual(len(out["red_herrings"]), 3)
        self.assertEqual(out["clues"][-1], case["clues"][-1])
        self.assertEqual(verify_solvability(out), [])

    def test_middle_player_gets_no_adapt(self):
        hist = blank_history()
        hist["attempts"] = {"medium": 4}
        hist["solved"] = {"medium": 3}
        case = bank_cases("medium")[0]
        out = adapt_case(case, random.Random(0), hist)
        self.assertFalse(out.get("assisted"))
        self.assertFalse(out.get("sharpened"))
        self.assertEqual(len(out["clues"]), 5)

    def test_deal_case_adapts_end_to_end(self):
        hist = blank_history()
        hist["attempts"] = {"medium": 5}
        hist["solved"] = {"medium": 1}
        rng = random.Random(11)
        seen_assisted = False
        for _ in range(30):
            case, tier, _ = deal_case(rng, hist)
            record_played(hist, case["id"], tier, False)
            if case.get("assisted"):
                seen_assisted = True
                self.assertEqual(verify_solvability(case), [])
                break
        self.assertTrue(seen_assisted)


# ── scoring ───────────────────────────────────────────────────────────────

class ScoringTests(unittest.TestCase):
    def test_base_by_tier(self):
        self.assertEqual(TIER_BASE_SCORE,
                         {"easy": 3, "medium": 5, "hard": 8, "expert": 12})

    def test_hint_cost(self):
        self.assertEqual(HINT_COST, 2)

    def test_strikes_and_hints_cost(self):
        self.assertEqual(score_solve("medium", 0, 0, 0, 0), 5)
        self.assertEqual(score_solve("medium", 1, 0, 0, 0), 4)
        self.assertEqual(score_solve("medium", 0, 1, 0, 0), 5 - HINT_COST)
        self.assertEqual(score_solve("medium", 3, 3, 0, 0), 1)  # floor

    def test_streak_multiplier(self):
        self.assertEqual(score_solve("easy", 0, 0, 3, 0), 4)   # 3 * 1.3
        self.assertEqual(score_solve("easy", 0, 0, 10, 0), 6)  # 3 * 2 cap
        self.assertEqual(score_solve("easy", 0, 0, 50, 0), 6)  # cap holds

    def test_time_bonus_added(self):
        self.assertEqual(score_solve("hard", 0, 0, 0, 7), 8 + 7)

    def test_solve_rate_neutral_prior(self):
        self.assertEqual(solve_rate(blank_history(), "medium"), 0.5)
        hist = blank_history()
        hist["attempts"] = {"medium": 4}
        hist["solved"] = {"medium": 3}
        self.assertAlmostEqual(solve_rate(hist, "medium"), 3 / 4)


# ── hints, through the engine ─────────────────────────────────────────────

class HintTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_hints_escalate_and_cap(self):
        self.engine.start("telegram:hints", "case", ADA)
        room = self.engine.live("telegram:hints")
        msgs = self.engine.move("telegram:hints", "hint", ADA)
        self.assertTrue(any("−2 pts" in m for m in msgs))
        self.assertEqual(room.state["hints_used"], 1)
        msgs = self.engine.move("telegram:hints", "hint", ADA)
        herring_suspect = room.state["case"]["herring_suspect"]
        self.assertTrue(any(herring_suspect in m for m in msgs))
        self.assertNotIn(room.state["case"]["culprit"], msgs[0])
        msgs = self.engine.move("telegram:hints", "hint", ADA)
        self.assertTrue(any("liar" in m for m in msgs))
        msgs = self.engine.move("telegram:hints", "hint", ADA)
        self.assertTrue(any("no more hints" in m for m in msgs))
        self.assertEqual(room.state["hints_used"], 3)

    def test_hint_costs_points(self):
        from nomorals.games.games.medium import InvestigationGame

        game = InvestigationGame()
        room = type("R", (), {})()
        base = {"case": {"tier": "medium"}, "tier": "medium",
                "strikes": 0, "streak_in": 0, "time_bonus": 0,
                "solved": True, "asked": ["x"]}
        room.state = dict(base, hints_used=0)
        no_hint = game.score(room, ADA)
        room.state = dict(base, hints_used=2)
        self.assertEqual(game.score(room, ADA), no_hint - 2 * HINT_COST)


# ── timed mode, through the engine ────────────────────────────────────────

class TimedTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_timed_mode_counts_down_and_bonuses(self):
        self.engine.start("telegram:timed", "case", ADA, timed=True)
        room = self.engine.live("telegram:timed")
        self.assertTrue(room.state["timed"])
        self.assertGreater(room.state["deadline"], 0)
        tier = room.state["tier"]
        self.assertEqual(room.state["time_limit"], TIME_LIMITS[tier])
        msgs = self.engine.move(
            "telegram:timed", "accuse " + room.state["case"]["culprit"], ADA)
        self.assertTrue(any("time bonus" in m for m in msgs))
        self.assertGreater(room.state["time_bonus"], 0)
        self.assertLessEqual(room.state["time_bonus"], TIME_BONUS_MAX[tier])

    def test_untimed_game_has_no_deadline(self):
        self.engine.start("telegram:plain", "case", ADA)
        room = self.engine.live("telegram:plain")
        self.assertFalse(room.state["timed"])
        self.assertNotIn("deadline", room.state)


# ── history + achievements, through the engine ────────────────────────────

class EngineHistoryTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _solve(self, chat: str) -> None:
        self.engine.start(chat, "case", ADA)
        room = self.engine.live(chat)
        for _ in range(3):
            self.engine.move(chat, "clue", ADA)
        self.engine.move(chat, "accuse " + room.state["case"]["culprit"],
                         ADA)

    def _history(self) -> dict:
        return (self.engine.store.get(ADA.key).per_game or {})["case"]

    def test_history_round_trip_through_engine(self):
        self._solve("telegram:hist1")
        hist = self._history()
        self.assertEqual(hist["v"], HISTORY_VERSION)
        self.assertEqual(sum(hist["attempts"].values()), 1)
        self.assertEqual(sum(hist["solved"].values()), 1)
        self.assertEqual(hist["streak"], 1)
        self.assertEqual(len(hist["seen"]), 1)

    def test_streak_builds_across_games(self):
        for n in range(3):
            self._solve(f"telegram:streak{n}")
        self.assertEqual(self._history()["streak"], 3)

    def test_streak_resets_on_failure(self):
        self._solve("telegram:ok1")
        self.engine.start("telegram:fail", "case", ADA)
        room = self.engine.live("telegram:fail")
        wrong = [s for s in room.state["case"]["suspects"]
                 if s != room.state["case"]["culprit"]][0]
        for _ in range(3):
            self.engine.move("telegram:fail", f"accuse {wrong}", ADA)
        self.assertEqual(self._history()["streak"], 0)

    def test_ten_games_never_repeat_a_bank_case(self):
        ids = []
        for n in range(10):
            chat = f"telegram:rep{n}"
            self.engine.start(chat, "case", ADA)
            room = self.engine.live(chat)
            ids.append(room.state["case"]["id"])
            self.engine.move(
                chat, "accuse " + room.state["case"]["culprit"], ADA)
        bank_ids = [i for i in ids if i.startswith("bank:")]
        self.assertEqual(len(set(bank_ids)), len(bank_ids))

    def test_case_achievements_unlock(self):
        self._solve("telegram:ach1")
        unlocked = {a["id"] for a in get_achievements(self.db, ADA.key)}
        self.assertIn("case_first", unlocked)
        # clean solve: no strikes, no hints
        self.assertIn("case_clean", unlocked)

    def test_streak_achievements_unlock(self):
        for n in range(5):
            self._solve(f"telegram:stk{n}")
        unlocked = {a["id"] for a in get_achievements(self.db, ADA.key)}
        self.assertIn("case_streak_3", unlocked)
        self.assertIn("case_streak_5", unlocked)

    def test_achievement_announced_once(self):
        self._solve("telegram:dup1")
        self._solve("telegram:dup2")
        rows = get_achievements(self.db, ADA.key)
        firsts = [a for a in rows if a["id"] == "case_first"]
        self.assertEqual(len(firsts), 1)


if __name__ == "__main__":
    unittest.main()
