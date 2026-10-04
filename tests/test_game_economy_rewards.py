"""Performance-based coin payouts: score scaling, difficulty multipliers,
win-streak bonuses, and the set-gear price sanity fix."""
import unittest

from nomorals.games.economy import GameEconomy
from nomorals.games.gear import GEAR_CATALOG


class CoinBreakdownTest(unittest.TestCase):
    def test_plain_win(self) -> None:
        total, why = GameEconomy.coin_breakdown(True)
        self.assertEqual(total, 40)
        self.assertIn("win 40", why)

    def test_score_bonus(self) -> None:
        total, _ = GameEconomy.coin_breakdown(True, score=20)
        self.assertEqual(total, 80)  # 40 + 20*2

    def test_score_bonus_capped(self) -> None:
        total, _ = GameEconomy.coin_breakdown(True, score=100)
        self.assertEqual(total, 120)  # 40 + min(100,40)*2

    def test_negative_score_ignored(self) -> None:
        total, _ = GameEconomy.coin_breakdown(True, score=-5)
        self.assertEqual(total, 40)

    def test_difficulty_multipliers(self) -> None:
        self.assertEqual(
            GameEconomy.coin_breakdown(True, score=20, difficulty="hard")[0],
            120)   # (40+40) * 1.5
        self.assertEqual(
            GameEconomy.coin_breakdown(True, difficulty="expert")[0],
            80)    # 40 * 2.0
        self.assertEqual(
            GameEconomy.coin_breakdown(True, score=20, difficulty="easy")[0],
            64)    # (40+40) * 0.8

    def test_unknown_difficulty_falls_back_to_normal(self) -> None:
        total, _ = GameEconomy.coin_breakdown(True, difficulty="lunatic")
        self.assertEqual(total, 40)

    def test_streak_bonus(self) -> None:
        total, why = GameEconomy.coin_breakdown(True, streak_after=4)
        self.assertEqual(total, 70)  # 40 + 3*10
        self.assertIn("streak +30", why)

    def test_streak_bonus_capped(self) -> None:
        total, _ = GameEconomy.coin_breakdown(True, streak_after=10)
        self.assertEqual(total, 90)  # 40 + 50 cap

    def test_first_win_no_streak_bonus(self) -> None:
        total, why = GameEconomy.coin_breakdown(True, streak_after=1)
        self.assertEqual(total, 40)
        self.assertNotIn("streak", why)

    def test_loss_and_draw_flat(self) -> None:
        self.assertEqual(GameEconomy.coin_breakdown(False)[0], 15)
        self.assertEqual(GameEconomy.coin_breakdown(None)[0], 25)

    def test_reward_coins_backward_compatible(self) -> None:
        # old single-argument callers keep working
        self.assertEqual(GameEconomy.reward_coins(True), 40)
        self.assertEqual(GameEconomy.reward_coins(False), 15)
        self.assertEqual(GameEconomy.reward_coins(None), 25)

    def test_reward_coins_matches_breakdown(self) -> None:
        total, _ = GameEconomy.coin_breakdown(
            True, score=25, difficulty="hard", streak_after=3)
        self.assertEqual(
            GameEconomy.reward_coins(
                True, score=25, difficulty="hard", streak_after=3),
            total)


class SetGearPricingTest(unittest.TestCase):
    def test_set_pieces_carry_two_x_plain_epic(self) -> None:
        # regression: set bases were written as final prices, then the
        # epic x4 multiplier applied on top (5600c+ for one piece)
        self.assertEqual(GEAR_CATALOG["katana_epic"].cost, 1200)
        self.assertEqual(GEAR_CATALOG["storm_katana"].cost, 2400)
        self.assertEqual(GEAR_CATALOG["storm_plate"].cost, 3600)
        self.assertEqual(GEAR_CATALOG["shadow_rapier"].cost, 1600)
        self.assertEqual(GEAR_CATALOG["shadow_mail"].cost, 2400)

    def test_full_set_is_reachable(self) -> None:
        full_storm = (GEAR_CATALOG["storm_katana"].cost
                      + GEAR_CATALOG["storm_plate"].cost)
        self.assertEqual(full_storm, 6000)


if __name__ == "__main__":
    unittest.main()
