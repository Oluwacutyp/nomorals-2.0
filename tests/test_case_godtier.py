"""God-tier games: the case bank + dynamic generator, and the
expanded content pools for the lighter games.

Covers:
- CASES: 30 hand-written cases, every one airtight (culprit in
  suspects, 5 clues, smoking gun names the culprit, every suspect has
  a statement).
- generate_case: seeded determinism, airtightness across 200 seeds x
  3 difficulties, no unfilled template slots, difficulty honored.
- random_case: bank/generator mix, id + difficulty on every case,
  anti-repeat window.
- InvestigationGame end-to-end through the engine (generated case can
  be closed; strikes still escape; scoring scales with difficulty).
- Content pools: wyrr 60, auction 36, spy 62, trivia 143.
"""
from __future__ import annotations

import random
import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.games.cases import (
    BANK_SIZE,
    CASES,
    generate_case,
    random_case,
)
from nomorals.games.games.easy import (
    AUCTION_ITEMS,
    SPY_WORDS,
    TRIVIA,
    WYRR_PAIRS,
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


def assert_airtight(test: unittest.TestCase, case: dict) -> None:
    test.assertIn(case["culprit"], case["suspects"])
    test.assertEqual(len(case["suspects"]), 4)
    test.assertEqual(len(set(case["suspects"])), 4)
    test.assertEqual(len(case["clues"]), 5)
    # the smoking gun names the culprit (bank cases sometimes use a
    # possessive or shortened reference: "The intern's build machine")
    words = [w for w in case["culprit"].lower().split()
             if w != "the" and len(w) > 3]
    test.assertTrue(any(w in case["clues"][4].lower() for w in words),
                    case["clues"][4])
    # everyone has a line; no unfilled template slots anywhere
    test.assertEqual(set(case["statements"]), set(case["suspects"]))
    for blob in ([case["story"]] + case["clues"]
                 + list(case["statements"].values())):
        test.assertNotIn("{", blob)
        test.assertNotIn("}", blob)
        test.assertTrue(blob.strip())


# ── the bank ──────────────────────────────────────────────────────────────

class BankTests(unittest.TestCase):
    def test_bank_holds_thirty_hand_written_cases(self):
        self.assertEqual(BANK_SIZE, 30)
        self.assertEqual(len(CASES), 30)

    def test_every_bank_case_is_airtight(self):
        for i, case in enumerate(CASES):
            with self.subTest(case=i):
                assert_airtight(self, case)

    def test_bank_cases_have_distinct_stories(self):
        stories = [c["story"] for c in CASES]
        self.assertEqual(len(set(stories)), len(stories))


# ── the generator ─────────────────────────────────────────────────────────

class GeneratorTests(unittest.TestCase):
    def test_seeded_generation_is_deterministic(self):
        a = generate_case(random.Random(99), "hard")
        b = generate_case(random.Random(99), "hard")
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        cases = {generate_case(random.Random(s), "medium")["id"]
                 for s in range(50)}
        self.assertEqual(len(cases), 50)

    def test_airtight_across_seeds_and_difficulties(self):
        for seed in range(200):
            for difficulty in ("easy", "medium", "hard"):
                with self.subTest(seed=seed, difficulty=difficulty):
                    case = generate_case(random.Random(seed), difficulty)
                    assert_airtight(self, case)
                    self.assertEqual(case["difficulty"], difficulty)
                    self.assertTrue(case["generated"])

    def test_culprit_statement_denies_the_gun(self):
        # the culprit never confesses: their line must not contain the
        # smoking-gun evidence verbatim as an admission
        for seed in range(50):
            case = generate_case(random.Random(seed), "medium")
            line = case["statements"][case["culprit"]]
            self.assertNotIn(case["clues"][4][:40], line)

    def test_hard_cases_use_subtle_templates(self):
        # hard herrings carry an exonerating tail after a worse look
        seen_subtle = 0
        for seed in range(60):
            case = generate_case(random.Random(seed), "hard")
            clue4 = case["clues"][3]
            if "—" in clue4 or "but" in clue4:
                seen_subtle += 1
        self.assertGreater(seen_subtle, 30)


# ── random_case: the mix ──────────────────────────────────────────────────

class RandomCaseTests(unittest.TestCase):
    def test_every_case_has_id_and_difficulty(self):
        for s in range(30):
            case = random_case(random.Random(1000 + s))
            self.assertTrue(case["id"])
            self.assertIn(case["difficulty"], ("easy", "medium", "hard"))
            assert_airtight(self, case)

    def test_mix_uses_both_sources(self):
        kinds = {random_case(random.Random(s)).get("generated")
                 for s in range(60)}
        self.assertEqual(kinds, {True, False})

    def test_no_repeat_within_recent_window(self):
        ids = [random_case(random.Random(s))["id"] for s in range(60)]
        for i in range(len(ids)):
            window = ids[max(0, i - 12):i]
            self.assertNotIn(ids[i], window)

    def test_explicit_difficulty_honored(self):
        for s in range(10):
            case = random_case(random.Random(s), difficulty="hard")
            self.assertEqual(case["difficulty"], "hard")


# ── the game, end to end ──────────────────────────────────────────────────

class InvestigationGameTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _close_with_culprit(self, chat: str) -> tuple[list[str], dict]:
        self.engine.start(chat, "case", ADA)
        room = self.engine.live(chat)
        state = room.state
        msgs: list[str] = []
        for _ in range(3):
            msgs.extend(self.engine.move(chat, "clue", ADA))
        msgs.extend(self.engine.move(
            chat, "accuse " + state["case"]["culprit"], ADA))
        return msgs, state

    def test_generated_or_bank_case_closes(self):
        for n in range(5):
            chat = f"telegram:case{n}"
            msgs, state = self._close_with_culprit(chat)
            self.assertTrue(any("closes the case" in m for m in msgs),
                            (n, state["case"].get("id")))
            self.assertIn(state["case"]["difficulty"],
                          ("easy", "medium", "hard"))

    def test_three_strikes_still_escapes(self):
        self.engine.start("telegram:strikes", "case", ADA)
        room = self.engine.live("telegram:strikes")
        wrong = [s for s in room.state["case"]["suspects"]
                 if s != room.state["case"]["culprit"]][0]
        msgs: list[str] = []
        for _ in range(3):
            msgs.extend(self.engine.move("telegram:strikes",
                                         f"accuse {wrong}", ADA))
        self.assertIsNone(self.engine.live("telegram:strikes"))
        self.assertTrue(any("walks" in m for m in msgs))

    def test_scoring_scales_with_difficulty(self):
        from nomorals.games.games.medium import InvestigationGame

        game = InvestigationGame()
        scores = {}
        for difficulty, base in (("easy", 3), ("medium", 5), ("hard", 8)):
            room = type("R", (), {})()
            room.state = {"case": {"difficulty": difficulty},
                          "strikes": 0, "asked": ["x"]}
            player = ADA
            scores[difficulty] = game.score(room, player)
            self.assertEqual(scores[difficulty], base + 1)
        self.assertLess(scores["easy"], scores["medium"])
        self.assertLess(scores["medium"], scores["hard"])

    def test_interview_then_accuse_flow(self):
        self.engine.start("telegram:flow", "case", ADA)
        room = self.engine.live("telegram:flow")
        suspect = room.state["case"]["suspects"][0]
        msgs = self.engine.move("telegram:flow", f"ask {suspect}", ADA)
        self.assertTrue(any(suspect in m for m in msgs))
        self.assertIn(suspect, room.state["asked"])


# ── content pools for the lighter games ───────────────────────────────────

class PoolTests(unittest.TestCase):
    def test_wyrr_pool_is_large(self):
        self.assertGreaterEqual(len(WYRR_PAIRS), 60)
        for a, b in WYRR_PAIRS:
            self.assertTrue(a and b and a != b)

    def test_auction_pool_is_large(self):
        self.assertGreaterEqual(len(AUCTION_ITEMS), 36)
        for name, low, high in AUCTION_ITEMS:
            self.assertTrue(name)
            self.assertLess(low, high)

    def test_spy_pool_is_large(self):
        self.assertGreaterEqual(len(SPY_WORDS), 60)  # 61 after expansion
        words = [w for w, _ in SPY_WORDS]
        self.assertEqual(len(set(words)), len(words))

    def test_trivia_pool_is_large(self):
        self.assertGreaterEqual(len(TRIVIA), 140)
        for q, a in TRIVIA:
            self.assertTrue(q and a)

    def test_wyrr_game_still_samples(self):
        from nomorals.games.games.easy import WyrrGame

        game = WyrrGame()
        room = type("R", (), {})()
        room.state = game.new_state(random.Random(7))
        self.assertEqual(len(room.state["pairs"]), 5)


if __name__ == "__main__":
    unittest.main()
