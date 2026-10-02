"""Wave H3: import-parity for the cases.py -> case_bank/ split.

``nomorals.games.games.cases`` is now a compatibility facade over the
``nomorals.games.games.case_bank`` package.  These tests pin that the old
import path still exposes the same public names, bound to the SAME
objects (not copies), and that the bank content is unchanged.
"""
from __future__ import annotations

import random
import unittest

import nomorals.games.games.case_bank as bank
import nomorals.games.games.case_bank.constants as constants
import nomorals.games.games.case_bank.logic as logic
import nomorals.games.games.cases as cases


PUBLIC_NAMES = [
    "BANK_SIZE", "CASES", "HINT_COST", "HISTORY_VERSION", "TIER_BASE_SCORE",
    "TIERS", "TIME_LIMITS", "TIME_BONUS_MAX",
    "generate_case", "random_case", "verify_solvability", "blank_history",
    "eligible_tiers", "solve_rate", "deal_case", "record_played",
    "adapt_case", "score_solve", "unseen_bank_cases",
]


class TestCasesFacadeParity(unittest.TestCase):
    def test_all_dunder_names_preserved(self):
        self.assertEqual(
            cases.__all__,
            ["CASES", "BANK_SIZE", "TIERS", "HINT_COST", "TIME_LIMITS",
             "TIME_BONUS_MAX", "HISTORY_VERSION",
             "generate_case", "random_case", "verify_solvability",
             "blank_history", "eligible_tiers", "solve_rate", "deal_case",
             "record_played", "adapt_case", "score_solve", "unseen_bank_cases"],
        )

    def test_public_names_are_identical_objects(self):
        for name in PUBLIC_NAMES:
            with self.subTest(name=name):
                self.assertTrue(hasattr(cases, name), name)
                self.assertIs(getattr(cases, name), getattr(bank, name), name)

    def test_bank_content_intact(self):
        self.assertEqual(cases.BANK_SIZE, 52)
        self.assertEqual(len(cases.CASES), 52)
        tiers = {c["tier"] for c in cases.CASES}
        self.assertEqual(tiers, {"easy", "medium", "hard", "expert"})
        self.assertIs(cases.CASES, logic.CASES)
        self.assertIs(cases.TIERS, constants.TIERS)

    def test_services_still_work_through_facade(self):
        rng = random.Random(1234)
        history = cases.blank_history()
        case, tier, _fresh = cases.deal_case(rng, history)
        self.assertIn("culprit", case)
        self.assertEqual(cases.verify_solvability(case), [])
        gen = cases.generate_case(random.Random(99), "hard")
        self.assertEqual(gen["tier"], "hard")
        self.assertGreater(cases.score_solve("medium"), 0)
        self.assertIn(tier, ("easy", "medium", "hard", "expert"))


if __name__ == "__main__":
    unittest.main()
