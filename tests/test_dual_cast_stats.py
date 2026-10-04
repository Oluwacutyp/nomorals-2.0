"""Tests for dual-cast combos, RPG stats, and title battle effects."""
import random
import unittest

from nomorals.games.skills import (
    SKILL_CATALOG, COMBO_CATALOG, find_combo, combo_success_chance,
    resolve_skill, effective_def,
)
from nomorals.games.stats import (
    StatBlock, StatStore, apply_stats_to_fighter, gear_stat_bonuses,
    describe_stats, POINTS_PER_LEVEL,
)
from nomorals.games.titles import (
    TITLE_CATALOG, title_battle_effects, describe_title_effects,
)


class ComboCatalogTests(unittest.TestCase):
    def test_order_matters(self):
        self.assertIsNotNone(find_combo("war_cry", "dragon_punch"))
        self.assertIsNone(find_combo("dragon_punch", "war_cry"))

    def test_unknown_pair_returns_none(self):
        self.assertIsNone(find_combo("war_cry", "second_wind"))
        self.assertIsNone(find_combo("banana", "kamehameha"))

    def test_case_insensitive(self):
        self.assertIsNotNone(find_combo("War_Cry", "Dragon_Punch"))

    def test_combos_have_mechanics(self):
        for (a, b), combo in COMBO_CATALOG.items():
            self.assertIn(a, SKILL_CATALOG, combo.name)
            self.assertIn(b, SKILL_CATALOG, combo.name)
            self.assertGreater(combo.base_success, 0)
            self.assertLessEqual(combo.base_success, 1)
            self.assertGreaterEqual(combo.hp_cost_pct, 0)

    def test_cutyp_judgment_is_risky(self):
        combo = find_combo("war_cry", "slaying_force")
        self.assertIsNotNone(combo)
        self.assertLess(combo.base_success, 0.90)
        self.assertGreater(combo.hp_cost_pct, 0.10)

    def test_success_falls_with_tier(self):
        combo = find_combo("war_cry", "dragon_punch")
        t1 = combo_success_chance(combo, 1, 1)
        t3 = combo_success_chance(combo, 3, 3)
        self.assertGreater(t1, t3)

    def test_intelligence_helps(self):
        combo = find_combo("war_cry", "dragon_punch")
        low = combo_success_chance(combo, 2, 2, intelligence=0)
        high = combo_success_chance(combo, 2, 2, intelligence=20)
        self.assertGreater(high, low)

    def test_chance_clamped(self):
        combo = find_combo("war_cry", "dragon_punch")
        # absurd tiers + no intelligence → floor
        self.assertGreaterEqual(
            combo_success_chance(combo, 9, 9, 0), 0.05)
        # max intelligence → ceiling
        self.assertLessEqual(
            combo_success_chance(combo, 1, 1, 1000), 0.95)


class StatBlockTests(unittest.TestCase):
    def test_defaults(self):
        s = StatBlock()
        self.assertEqual(s.strength, 0)
        self.assertEqual(s.unspent, 0)

    def test_from_dict_roundtrip(self):
        s = StatBlock(strength=5, stamina=3, mana=2, intelligence=4,
                      unspent=1, level_applied=3)
        d = s.to_dict()
        s2 = StatBlock.from_dict(d)
        self.assertEqual(s2.strength, 5)
        self.assertEqual(s2.intelligence, 4)
        self.assertEqual(s2.level_applied, 3)

    def test_from_dict_none(self):
        s = StatBlock.from_dict(None)
        self.assertEqual(s.strength, 0)

    def test_total_with_gear(self):
        s = StatBlock(strength=5)
        self.assertEqual(s.total("strength", 3), 8)
        self.assertEqual(s.total("strength"), 5)

    def test_apply_to_fighter(self):
        fighter = {"hp": 50, "max_hp": 50, "atk": 10, "def": 5}
        stats = StatBlock(strength=6, stamina=4, mana=5, intelligence=8)
        apply_stats_to_fighter(fighter, stats)
        # strength 6 → +3 atk
        self.assertEqual(fighter["atk"], 13)
        # stamina 4 → +12 max HP, +1 def
        self.assertEqual(fighter["max_hp"], 62)
        self.assertEqual(fighter["hp"], 62)
        self.assertEqual(fighter["def"], 6)
        # mana 5 → 30 + 10 pool
        self.assertEqual(fighter["mana"], 40)
        self.assertEqual(fighter["max_mana"], 40)
        # intelligence stored for combo rolls
        self.assertEqual(fighter["intelligence"], 8)

    def test_gear_bonuses(self):
        loadout = {
            "weapon": {"stat_strength": 3},
            "armor": {"stat_stamina": 2},
            "trinket": {"stat_intelligence": 6, "stat_mana": 4},
        }
        bonus = gear_stat_bonuses(loadout)
        self.assertEqual(bonus["strength"], 3)
        self.assertEqual(bonus["stamina"], 2)
        self.assertEqual(bonus["intelligence"], 6)
        self.assertEqual(bonus["mana"], 4)

    def test_describe(self):
        s = StatBlock(strength=5, unspent=2)
        text = describe_stats(s, {"strength": 3})
        self.assertIn("Strength", text)
        self.assertIn("8", text)  # 5 + 3 gear
        self.assertIn("2", text)  # unspent


class StatStoreTests(unittest.TestCase):
    def _db(self):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        class Q:
            def __init__(self, conn):
                self.conn = conn
            def execute(self, sql, params=()):
                self.conn.execute(sql, params)
                self.conn.commit()
            def query(self, sql, params=()):
                return self.conn.execute(sql, params).fetchall()
        return Q(db)

    def test_grant_level_points(self):
        store = StatStore(self._db())
        # level 1 → no new points (starts at level_applied=1)
        self.assertEqual(store.grant_level_points("p1", 1), 0)
        # level 4 → 3 new levels × 3 points
        self.assertEqual(store.grant_level_points("p1", 4),
                         3 * POINTS_PER_LEVEL)
        # already applied → no double grant
        self.assertEqual(store.grant_level_points("p1", 4), 0)
        # level 5 → one more level
        self.assertEqual(store.grant_level_points("p1", 5),
                         POINTS_PER_LEVEL)

    def test_spend(self):
        store = StatStore(self._db())
        store.grant_level_points("p1", 3)  # 6 points
        ok, msg = store.spend("p1", "strength", 2)
        self.assertTrue(ok)
        self.assertIn("Strength +2", msg)
        stats = store.get("p1")
        self.assertEqual(stats.strength, 2)
        self.assertEqual(stats.unspent, 6 - 2)

    def test_spend_too_many(self):
        store = StatStore(self._db())
        ok, msg = store.spend("p1", "strength", 5)
        self.assertFalse(ok)
        self.assertIn("unspent", msg.lower())

    def test_spend_unknown_attr(self):
        store = StatStore(self._db())
        ok, msg = store.spend("p1", "charisma", 1)
        self.assertFalse(ok)


class TitleEffectTests(unittest.TestCase):
    def test_novice_has_no_effects(self):
        self.assertEqual(title_battle_effects("Novice"), {})

    def test_dragonslayer(self):
        fx = title_battle_effects("Dragonslayer")
        self.assertGreater(fx.get("atk", 0), 0)
        self.assertGreater(fx.get("boss_dmg_pct", 0), 0)

    def test_untouched_tradeoff(self):
        fx = title_battle_effects("Untouched")
        self.assertGreater(fx.get("def", 0), 0)
        self.assertLess(fx.get("atk", 0), 0)  # the tradeoff

    def test_unknown_title(self):
        self.assertEqual(title_battle_effects("No Such Title"), {})
        self.assertEqual(title_battle_effects(""), {})

    def test_describe(self):
        text = describe_title_effects("Dragonslayer")
        self.assertIn("atk", text)
        self.assertIn("boss", text.lower())

    def test_new_titles_exist(self):
        ids = {t.id for t in TITLE_CATALOG}
        for tid in ("phoenix", "comeback_king", "persistent", "underdog",
                    "pacifist", "speedster", "survivor", "dual_master",
                    "gambler"):
            self.assertIn(tid, ids, tid)

    def test_dual_master_boosts_combos(self):
        fx = title_battle_effects("Dual Master")
        self.assertGreater(fx.get("combo_pct", 0), 0)


if __name__ == "__main__":
    unittest.main()
