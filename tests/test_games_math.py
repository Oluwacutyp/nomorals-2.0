"""Tests for Math Class + difficulty-reward ladder (new modules only).

Invariants:
  * every reward reconstructible from the ledger;
  * difficulty maps to the reward TABLE as data (loaded, not if-chains);
  * coins are never described as an investment (the word appears only
    in negation in the new source files);
  * a problem's answer is verified before any reward (wrong = nothing);
  * the Impossible tier is actually hard: random guesses pass <5% over
    100 seeded problems, and every problem is solvable by its exact answer.
"""
import random
import re
import unittest
from pathlib import Path

from nomorals.games.math_class import (
    ATTEMPT_CAPS, COOLDOWN_SECONDS, TIERS, MathClass, check_answer,
)
from nomorals.games.math_commands import (
    MathCommandHandler, daily_math_pulse, daily_math_quest,
    digest_top_solvers, math_party_duel, math_season,
    math_tournament_payouts,
)
from nomorals.games.math_economy import (
    MATH_COINS_DISCLAIMER, MATH_EARN_CAPS, MathEconomy,
)
from nomorals.games.reward_ladder import (
    FIRST_CLEAR_CHEST, REWARD_TABLE, decay_factor, pool_split,
    reward_for, reward_for_game, streak_bonus,
)

NEW_FILES = [
    "nomorals/games/math_class.py",
    "nomorals/games/reward_ladder.py",
    "nomorals/games/math_economy.py",
    "nomorals/games/math_commands.py",
]
ROOT = Path(__file__).resolve().parent.parent


class LadderTableTest(unittest.TestCase):
    def test_table_holds_the_spec_values(self):
        self.assertEqual((REWARD_TABLE["easy"]["coins"], REWARD_TABLE["easy"]["xp"]), (1, 10))
        self.assertEqual((REWARD_TABLE["medium"]["coins"], REWARD_TABLE["medium"]["xp"]), (3, 25))
        self.assertEqual((REWARD_TABLE["hard"]["coins"], REWARD_TABLE["hard"]["xp"]), (10, 60))
        self.assertEqual((REWARD_TABLE["expert"]["coins"], REWARD_TABLE["expert"]["xp"]), (30, 150))
        self.assertEqual((REWARD_TABLE["master"]["coins"], REWARD_TABLE["master"]["xp"]), (100, 400))
        self.assertEqual((REWARD_TABLE["impossible"]["coins"], REWARD_TABLE["impossible"]["xp"]),
                         (1000, 1500))
        self.assertEqual(REWARD_TABLE["master"]["badge"], "Master of Numbers")
        self.assertEqual(REWARD_TABLE["impossible"]["title"], "Grandmaster of the Impossible")
        self.assertTrue(REWARD_TABLE["impossible"]["shoutout"])
        self.assertTrue(REWARD_TABLE["impossible"]["party_theme_vote"])

    def test_reward_for_reads_the_table(self):
        """reward_for must equal the table row for every difficulty —
        no hidden per-difficulty branches allowed to drift."""
        for tier, row in REWARD_TABLE.items():
            got = reward_for(tier)
            self.assertEqual(got["coins"], row["coins"], tier)
            self.assertEqual(got["xp"], row["xp"], tier)
            self.assertEqual(got["badge"], row["badge"], tier)
            self.assertEqual(got["title"], row["title"], tier)
            self.assertEqual(got["shoutout"], row["shoutout"], tier)
            self.assertEqual(got["party_theme_vote"], row["party_theme_vote"], tier)

    def test_same_ladder_for_every_game(self):
        for game in ("math", "trivia", "arena", "party", "quests"):
            got = reward_for_game(game, "hard")
            self.assertEqual((got["coins"], got["xp"]), (10, 60), game)

    def test_streak_bonuses(self):
        self.assertEqual(streak_bonus(0), (0.0, None))
        self.assertEqual(streak_bonus(3), (0.10, None))
        self.assertEqual(streak_bonus(7)[0], 0.25)
        bonus, badge = streak_bonus(30)
        self.assertEqual(bonus, 0.50)
        self.assertEqual(badge, "Month of Fire")
        # 3-day +10%: 100 -> 110
        self.assertEqual(reward_for("master", streak_days=3)["coins"], 110)
        self.assertEqual(reward_for("master", streak_days=7)["coins"], 125)
        self.assertEqual(reward_for("master", streak_days=30)["coins"], 150)

    def test_first_clear_chest(self):
        got = reward_for("easy", first_clear=True)
        self.assertEqual(got["coins"], 1 + FIRST_CLEAR_CHEST["easy"]["coins"])
        self.assertEqual(got["xp"], 10 + FIRST_CLEAR_CHEST["easy"]["xp"])

    def test_farming_decay(self):
        self.assertEqual(decay_factor(0), 1.0)
        self.assertLess(decay_factor(1), 1.0)
        self.assertGreaterEqual(decay_factor(100), 0.10)
        self.assertLess(reward_for("hard", repeat_count=2)["coins"],
                        reward_for("hard")["coins"])

    def test_pool_split_sums_to_pool(self):
        for n in (1, 2, 3, 5, 8):
            amounts = pool_split(1000, n)
            self.assertEqual(len(amounts), n)
            self.assertEqual(sum(amounts), 1000)
        self.assertEqual(pool_split(100, 5), [40, 25, 15, 12, 8])


class MathGenerationTest(unittest.TestCase):
    def test_every_tier_generates_and_exact_answer_passes(self):
        mc = MathClass()
        for tier in TIERS:
            p = mc.generate("tester", tier)
            self.assertEqual(p.tier, tier)
            self.assertTrue(p.question)
            self.assertTrue(p.solution)
            self.assertTrue(check_answer(p, p.answer_text),
                            f"exact answer failed for {tier}: {p.question} -> {p.answer_text}")

    def test_answer_forms_accepted(self):
        mc = MathClass()
        p = mc.generate("tester", "easy", topic="fractions")
        # fraction answer also accepted as decimal
        self.assertTrue(check_answer(p, p.answer_text))
        wrong = "999999"
        self.assertFalse(check_answer(p, wrong))
        self.assertFalse(check_answer(p, "not a number"))
        self.assertFalse(check_answer(p, ""))

    def test_no_repeat_per_player(self):
        mc = MathClass()
        seen = set()
        for _ in range(30):
            p = mc.generate("repeaty", "easy")
            self.assertNotIn(p.question, seen)
            seen.add(p.question)

    def test_public_view_hides_answer(self):
        mc = MathClass()
        p = mc.generate("tester", "hard")
        pub = mc.public_problem(p)
        self.assertNotIn("answer", pub)
        self.assertNotIn("answer_text", pub)
        self.assertNotIn("answer_value", pub)
        self.assertNotIn("solution", pub)
        self.assertIn("question", pub)

    def test_daily_challenge_same_for_all(self):
        mc = MathClass()
        a = mc.daily_challenge("2026-10-16")
        b = mc.daily_challenge("2026-10-16")
        self.assertEqual(a.question, b.question)
        self.assertEqual(a.answer_text, b.answer_text)

    def test_adaptive_tier_up_and_down(self):
        mc = MathClass()
        start = mc.adaptive_tier("ada")
        self.assertEqual(start, "medium")
        for _ in range(3):
            mc.record_result("ada", "medium", True)
        self.assertEqual(mc.adaptive_tier("ada"), "hard")
        mc.record_result("ada", "hard", False)
        mc.record_result("ada", "hard", False)
        self.assertEqual(mc.adaptive_tier("ada"), "medium")

    def test_player_may_attempt_higher_tier(self):
        mc = MathClass()
        p = mc.generate("newbie", "impossible")
        self.assertEqual(p.tier, "impossible")

    def test_cooldown_and_cap_enforced(self):
        mc = MathClass()
        now = 1_000_000.0
        p = mc.generate("cooly", "easy", now=now)
        res = mc.submit("cooly", p.problem_id, "wrong-wrong", now=now)
        self.assertTrue(res["ok"])
        self.assertFalse(res["correct"])
        allowed, reason, wait = mc.attempt_allowed("cooly", "easy", now + 1)
        self.assertFalse(allowed)
        self.assertEqual(reason, "cooldown")
        self.assertGreater(wait, 0)
        # After the cooldown, attempts work again.
        allowed, _, _ = mc.attempt_allowed(
            "cooly", "easy", now + COOLDOWN_SECONDS["easy"] + 1)
        self.assertTrue(allowed)
        # Cap: fill the day's attempts, then blocked.
        mc.store["attempts"][("cooly", "easy", "1970-01-12")] = ATTEMPT_CAPS["easy"]
        allowed, reason, _ = mc.attempt_allowed(
            "cooly", "easy", now + COOLDOWN_SECONDS["easy"] + 1)
        self.assertFalse(allowed)
        self.assertEqual(reason, "daily_cap")


class ImpossibleTierTest(unittest.TestCase):
    def test_random_guesses_pass_under_5_percent(self):
        mc = MathClass()
        rng = random.Random(1234)
        passed = 0
        total = 100
        for i in range(total):
            p = mc.generate(f"imp-{i}", "impossible")
            guess = rng.randint(0, 10_000)
            if check_answer(p, guess):
                passed += 1
        self.assertLess(passed / total, 0.05,
                        f"random guesses passed {passed}/{total} — impossible is too easy")

    def test_impossible_problems_solvable_by_exact_answer(self):
        mc = MathClass()
        solved = 0
        for i in range(100):
            p = mc.generate(f"imp-solve-{i}", "impossible")
            if check_answer(p, p.answer_text):
                solved += 1
        self.assertGreaterEqual(solved, 1)
        self.assertEqual(solved, 100, "every impossible problem must be exactly solvable")

    def test_impossible_is_multi_step(self):
        mc = MathClass()
        p = mc.generate("imp-x", "impossible")
        self.assertTrue(p.solution_steps, "impossible problems need worked multi-step solutions")


class MathEconomyTest(unittest.TestCase):
    def _verified(self):
        eco = MathEconomy()
        eco.verify_member("ada", fingerprint="device-1")
        return eco

    def test_unverified_earns_nothing(self):
        eco = MathEconomy()
        self.assertIsNone(eco.credit("ghost", 10, 10, "lesson"))
        self.assertEqual(eco.balance("ghost"), {"coins": 0, "xp": 0})

    def test_lesson_pass_pays_and_reconstructs(self):
        eco = self._verified()
        eco.register_lesson("add-1", "easy", "Adding up")
        res = eco.complete_lesson("ada", "add-1", passed=True)
        self.assertTrue(res["ok"])
        bal = eco.balance("ada")
        self.assertEqual(bal["coins"], res["reward"]["coins"])
        # Reconstruct from the statement: entries must sum to the balance.
        stmt = eco.statement("ada")
        self.assertEqual(sum(e["amount_coins"] for e in stmt["entries"]), bal["coins"])
        self.assertEqual(sum(e["amount_xp"] for e in stmt["entries"]), bal["xp"])

    def test_lesson_fail_pays_nothing(self):
        eco = self._verified()
        eco.register_lesson("add-1", "easy", "Adding up")
        res = eco.complete_lesson("ada", "add-1", passed=False)
        self.assertFalse(res["ok"])
        self.assertEqual(eco.balance("ada"), {"coins": 0, "xp": 0})

    def test_repeat_lesson_decays(self):
        eco = self._verified()
        eco.register_lesson("add-1", "easy", "Adding up")
        first = eco.complete_lesson("ada", "add-1", passed=True)
        second = eco.complete_lesson("ada", "add-1", passed=True)
        # Second clear: no first-clear chest, decayed base.
        self.assertLess(second["reward"]["coins"], first["reward"]["coins"])

    def test_reversal_never_deletes(self):
        eco = self._verified()
        entry = eco.credit("ada", 50, 20, "quiz")
        assert entry is not None
        rev = eco.reverse(entry.id, "test reversal")
        self.assertIsNotNone(rev)
        self.assertEqual(eco.balance("ada"), {"coins": 0, "xp": 0})
        # Original still in the ledger, marked reversed.
        ids = [e.id for e in eco._entries]
        self.assertIn(entry.id, ids)
        self.assertEqual(next(e for e in eco._entries if e.id == entry.id).reversed_by, rev.id)
        # Reversing twice / reversing a reversal is refused.
        self.assertIsNone(eco.reverse(entry.id, "again"))
        self.assertIsNone(eco.reverse(rev.id, "reverse the reversal"))

    def test_earn_caps_enforced(self):
        eco = self._verified()
        now = 1_700_000_000.0
        self.assertIsNone(eco.credit("ada", MATH_EARN_CAPS["daily"] + 1, 0, "too much", now=now))
        ok = eco.credit("ada", MATH_EARN_CAPS["daily"], 0, "at cap", now=now)
        self.assertIsNotNone(ok)

    def test_teaching_upvotes_pay_author(self):
        eco = self._verified()
        eco.verify_member("bob", fingerprint="device-2")
        expl = eco.explain_solution("ada", "math_1", "Because 2+2=4.")
        self.assertTrue(expl["ok"])
        res = eco.upvote_explanation(expl["explanation_id"], "bob")
        self.assertTrue(res.get("paid"))
        self.assertEqual(eco.balance("ada")["coins"], res["coins"])
        # Self-upvote and repeat vote pay nothing.
        self.assertNotEqual(eco.upvote_explanation(expl["explanation_id"], "ada").get("paid"), True)
        self.assertNotEqual(eco.upvote_explanation(expl["explanation_id"], "bob").get("paid"), True)

    def test_creator_bonus_on_solver_fail(self):
        eco = self._verified()
        eco.verify_member("bob", fingerprint="device-2")
        eco.register_created_problem("ada", "prob-1", "hard")
        res = eco.record_solver_outcome("prob-1", "bob", solved=False)
        self.assertTrue(res.get("paid"))
        self.assertGreater(eco.balance("ada")["coins"], 0)
        # A solve pays the creator nothing.
        res2 = eco.record_solver_outcome("prob-1", "bob", solved=True)
        self.assertNotEqual(res2.get("paid"), True)
        # Creator's own attempts pay nothing.
        self.assertNotEqual(eco.record_solver_outcome("prob-1", "ada", solved=False).get("paid"), True)

    def test_alt_accounts_flagged_record_only(self):
        eco = MathEconomy()
        eco.verify_member("a1", fingerprint="same-device")
        eco.verify_member("a2", fingerprint="same-device")
        eco.verify_member("a3", fingerprint="same-device")
        flags = eco.flags("a1")
        self.assertTrue(any(f["kind"] == "alt_accounts" for f in flags))
        self.assertTrue(all(f.get("action_taken", "").startswith("none") for f in flags))
        # No ban mechanism exists: no attribute that bans.
        self.assertFalse(hasattr(eco, "ban"))
        self.assertFalse(hasattr(eco, "auto_ban"))


class CommandHandlerTest(unittest.TestCase):
    def _handler(self):
        eco = MathEconomy()
        eco.verify_member("ada", fingerprint="dev-ada")
        eco.verify_member("bob", fingerprint="dev-bob")
        return MathCommandHandler(MathClass(), eco)

    def test_start_and_correct_answer_pays(self):
        h = self._handler()
        started = h.handle("/math easy", "ada")
        self.assertTrue(started["ok"])
        pid = started["problem"]["problem_id"]
        problem = h.math.store["active"][pid]
        res = h.handle(f"/math answer {problem.answer_text}", "ada")
        self.assertTrue(res["correct"])
        self.assertTrue(res["reward"]["paid"])
        self.assertEqual(res["reward"]["coins"],
                         h.economy.balance("ada")["coins"])

    def test_wrong_answer_pays_nothing(self):
        h = self._handler()
        started = h.handle("/math easy", "ada")
        self.assertTrue(started["ok"])
        res = h.handle("/math answer definitely-not-the-answer", "ada")
        self.assertTrue(res["ok"])
        self.assertFalse(res["correct"])
        self.assertEqual(res["reward"]["coins"], 0)
        self.assertEqual(h.economy.balance("ada")["coins"], 0)

    def test_leaderboard_streak_daily(self):
        h = self._handler()
        self.assertTrue(h.handle("/math leaderboard", "ada")["ok"])
        self.assertTrue(h.handle("/math streak", "ada")["ok"])
        daily = h.handle("/math daily", "ada")
        self.assertTrue(daily["ok"])
        self.assertTrue(daily["daily"])

    def test_duel_same_problems_first_correct_wins(self):
        h = self._handler()
        duel = h.handle("/math duel bob", "ada")
        self.assertTrue(duel["ok"])
        duel_id = duel["duel_id"]
        first = h.math.store["duels"][duel_id].problems[0]
        res = h.handle(f"/math duel_answer {duel_id} 0 {first.answer_text}", "ada")
        self.assertTrue(res["correct"])
        self.assertEqual(res["round_winner"], "ada")
        # Same round now closed for the opponent.
        res2 = h.handle(f"/math duel_answer {duel_id} 0 {first.answer_text}", "bob")
        self.assertFalse(res2["ok"])

    def test_handlers_return_dicts_never_send(self):
        h = self._handler()
        for text in ("/math", "/math hard", "/math leaderboard", "/math streak",
                     "/math daily", "/math balance"):
            out = h.handle(text, "ada")
            self.assertIsInstance(out, dict)

    def test_hooks(self):
        h = self._handler()
        quest = daily_math_quest(h, "ada")
        self.assertEqual(quest["period"], "daily")
        self.assertIn("problem", quest)
        self.assertTrue(math_season(h)["season"])
        self.assertEqual(daily_math_pulse(h)["kind"], "daily_math_pulse")
        party = math_party_duel(h, "ada", "bob")
        self.assertEqual(party["kind"], "party_math_duel")
        digest = digest_top_solvers(h)
        self.assertEqual(digest["kind"], "math_digest")
        payouts = math_tournament_payouts(1000, 5)
        self.assertEqual(sum(p["coins"] for p in payouts), 1000)
        self.assertEqual(payouts[0]["coins"], 400)


class NoInvestmentPromiseTest(unittest.TestCase):
    def test_no_coin_described_as_investment(self):
        """Every 'investment' mention in the new source files must be a
        negation (never / not / no) — never a promise."""
        for rel in NEW_FILES:
            text = (ROOT / rel).read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if "investment" in line.lower():
                    lowered = line.lower()
                    self.assertTrue(
                        ("never" in lowered) or ("not " in lowered) or ("no " in lowered),
                        f"{rel}:{i}: non-negated 'investment': {line.strip()}")
        self.assertIn("never", MATH_COINS_DISCLAIMER.lower())
        self.assertIn("never", MATH_COINS_DISCLAIMER.lower().split("investment")[0][-30:])


if __name__ == "__main__":
    unittest.main()
