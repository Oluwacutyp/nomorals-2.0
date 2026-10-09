"""Tests: intelligence now scales skill damage, so stat builds fight differently.

Covers the shared ``combat.skill_power_mult`` helper, the wiring into
both combat implementations (solo arena in ambitious.py, PvP duel in
pvp.py), and the never-raises contract.
"""
import random
import unittest

from nomorals.games.combat import (
    new_fighter, strike, skill_power_mult,
)
from nomorals.games.stats import (
    StatBlock, apply_stats_to_fighter, describe_stats,
)
from nomorals.games.games.ambitious import BattleArenaGame
from nomorals.games.games.pvp import _ArenaCombat


def _seeded_rng(seed=7):
    return random.Random(seed)


class SkillPowerMultTests(unittest.TestCase):
    def test_no_intelligence_no_bonus(self):
        self.assertEqual(skill_power_mult({}), 1.0)
        self.assertEqual(skill_power_mult(new_fighter()), 1.0)

    def test_two_percent_per_point(self):
        self.assertAlmostEqual(skill_power_mult({"intelligence": 10}), 1.2)
        self.assertAlmostEqual(skill_power_mult({"intelligence": 25}), 1.5)
        self.assertAlmostEqual(skill_power_mult({"intelligence": 1}), 1.02)

    def test_negative_clamped(self):
        self.assertEqual(skill_power_mult({"intelligence": -50}), 1.0)

    def test_never_raises(self):
        for bad in (None, {}, {"intelligence": None},
                    {"intelligence": "high"}, {"intelligence": [3]}):
            self.assertEqual(skill_power_mult(bad), 1.0)

    def test_strike_with_mult_still_scales(self):
        rng = _seeded_rng()
        a = new_fighter(atk=10)
        d = new_fighter(hp=1000, dfn=0)
        mult = skill_power_mult({"intelligence": 50})  # 2.0
        rep = strike(a, d, rng, mult=mult)
        # base raw: max(1, 10 - 0 + randint(-2,3)) then *2 (+ possible crit)
        self.assertGreaterEqual(rep["dmg"], 8)
        self.assertLessEqual(rep["dmg"], 26)


class StatWiringTests(unittest.TestCase):
    def test_intelligence_folds_into_fighter(self):
        f = new_fighter()
        apply_stats_to_fighter(f, StatBlock(intelligence=12))
        self.assertEqual(f["intelligence"], 12)
        self.assertAlmostEqual(skill_power_mult(f), 1.24)

    def test_sheet_shows_real_numbers(self):
        text = describe_stats(StatBlock(intelligence=10))
        self.assertIn("skill damage +20%", text)
        self.assertIn("combo luck +10%", text)

    def test_strength_vs_intelligence_differ(self):
        rng = _seeded_rng()
        # strength build: basics hit harder
        str_f = new_fighter()
        apply_stats_to_fighter(str_f, StatBlock(strength=10))
        # intelligence build: techniques hit harder
        int_f = new_fighter()
        apply_stats_to_fighter(int_f, StatBlock(intelligence=10))
        d1 = new_fighter(hp=1000, dfn=0)
        d2 = new_fighter(hp=1000, dfn=0)
        basic_str = strike(str_f, d1, rng)["dmg"]
        basic_int = strike(int_f, d2, rng)["dmg"]
        # strength build hits harder with basic attacks (10 more atk)
        self.assertGreaterEqual(basic_str, basic_int + 5)
        # but the intelligence build's techniques out-hit it at 1.2x
        self.assertGreater(skill_power_mult(int_f),
                           skill_power_mult(str_f))


class ArenaIntelligenceWiringTests(unittest.TestCase):
    """The solo arena's striking skills use skill_power_mult."""

    def test_skill_mult_wired(self):
        # sanity: the arena module's cast path references the helper
        import inspect
        from nomorals.games.games import ambitious
        src = inspect.getsource(ambitious.BattleArenaGame._cast_skill)
        self.assertIn("skill_power_mult", src)
        src = inspect.getsource(ambitious.BattleArenaGame._dual_cast)
        self.assertIn("skill_power_mult", src)

    def test_simulated_skill_cast_scales(self):
        # replicate the arena's technique formula: mult * skill_power_mult
        game = BattleArenaGame()
        me = new_fighter(atk=10)
        apply_stats_to_fighter(me, StatBlock(intelligence=20))
        from nomorals.games.skills import resolve_skill, effective_def
        defn = effective_def(resolve_skill("dragon_punch"), 1)
        base = float(defn.mult or 1.0)
        scaled = base * skill_power_mult(me)
        self.assertGreater(scaled, base)
        self.assertAlmostEqual(scaled, base * 1.4)


class PvPIntelligenceWiringTests(unittest.TestCase):
    def test_skill_mult_wired(self):
        import inspect
        src = inspect.getsource(_ArenaCombat._cast_skill)
        self.assertIn("skill_power_mult", src)
        src = inspect.getsource(_ArenaCombat._dual_cast)
        self.assertIn("skill_power_mult", src)


class DescribeStateVitalityTests(unittest.TestCase):
    """Turn summaries show the mana pool so players can plan spends."""

    def test_duel_describe_state_shows_mana(self):
        import inspect
        from nomorals.games.games.pvp import DuelGame
        src = inspect.getsource(DuelGame.describe_state)
        self.assertIn("mana", src)

    def test_arena_turn_tail_shows_mana(self):
        import inspect
        src = inspect.getsource(BattleArenaGame.on_move)
        self.assertIn("mana", src)


if __name__ == "__main__":
    unittest.main()
