"""Tests for enemy-exclusive skills, titles, daily hunt, and new achievements."""
from __future__ import annotations

import random
import sqlite3
import unittest

from nomorals.games.skills import (
    ENEMY_SKILL_CATALOG, SKILL_CATALOG, is_enemy_skill, lookup_skill,
    resolve_skill,
)
from nomorals.games.enemies import roll_enemy_skills
from nomorals.games.titles import TITLE_CATALOG, TitleStore
from nomorals.games.daily import (
    complete_daily_hunt, daily_hunt_done, hunt_date,
)
from nomorals.games.achievements import ACHIEVEMENTS, unlock_achievement
from nomorals.games.combat import new_fighter, tick_fighter
from nomorals.games.power import skill_power_score


def _memdb():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE achievements (player_key TEXT NOT NULL, "
        "achievement_id TEXT NOT NULL, unlocked_at REAL NOT NULL DEFAULT 0, "
        "PRIMARY KEY (player_key, achievement_id))")

    class DB:
        def execute(self, sql, params=()):
            cur = db.execute(sql, params)
            db.commit()
            return cur

        def query(self, sql, params=()):
            return db.execute(sql, params).fetchall()

    return DB()


class EnemySkillCatalogTests(unittest.TestCase):
    def test_six_forbidden_techniques(self):
        self.assertEqual(len(ENEMY_SKILL_CATALOG), 6)
        for slug in ("soul_siphon", "venom_fang", "bone_crusher",
                     "blood_frenzy", "dread_aura", "executioner"):
            self.assertIn(slug, ENEMY_SKILL_CATALOG)

    def test_not_in_learnable_catalog(self):
        for slug in ENEMY_SKILL_CATALOG:
            self.assertNotIn(slug, SKILL_CATALOG)

    def test_players_cannot_resolve(self):
        for slug in ENEMY_SKILL_CATALOG:
            self.assertIsNone(resolve_skill(slug))
            self.assertIsNone(resolve_skill(
                ENEMY_SKILL_CATALOG[slug].name))

    def test_house_can_lookup(self):
        for slug in ENEMY_SKILL_CATALOG:
            defn = lookup_skill(slug)
            self.assertIsNotNone(defn)
            self.assertTrue(is_enemy_skill(slug))

    def test_player_skills_not_enemy(self):
        self.assertFalse(is_enemy_skill("dragon_punch"))
        self.assertIsNotNone(lookup_skill("dragon_punch"))

    def test_effect_knobs_present(self):
        self.assertGreater(
            ENEMY_SKILL_CATALOG["soul_siphon"].lifesteal_pct, 0)
        self.assertGreater(
            ENEMY_SKILL_CATALOG["venom_fang"].poison_turns, 0)
        self.assertGreater(
            ENEMY_SKILL_CATALOG["bone_crusher"].def_debuff, 0)
        self.assertTrue(ENEMY_SKILL_CATALOG["blood_frenzy"].frenzy)
        self.assertGreater(
            ENEMY_SKILL_CATALOG["dread_aura"].atk_debuff, 0)
        self.assertGreater(
            ENEMY_SKILL_CATALOG["executioner"].execute_mult, 0)
        # dread_aura is pure debuff — no strike
        self.assertEqual(ENEMY_SKILL_CATALOG["dread_aura"].mult, 0.0)

    def test_power_scores_sane(self):
        for slug, defn in ENEMY_SKILL_CATALOG.items():
            score = skill_power_score(defn)
            self.assertGreater(score, 0, slug)


class EnemyRollTests(unittest.TestCase):
    def test_low_ranks_fight_clean(self):
        rng = random.Random(7)
        for rank in (0, 1, 2):
            skills = roll_enemy_skills(rng, rank)
            for slug in skills:
                self.assertFalse(is_enemy_skill(slug),
                                 f"rank {rank} rolled forbidden {slug}")

    def test_high_ranks_fight_dirty(self):
        # S-rank always rolls forbidden techniques
        for seed in range(20):
            rng = random.Random(seed)
            skills = roll_enemy_skills(rng, 5)
            forbidden = [s for s in skills if is_enemy_skill(s)]
            self.assertGreaterEqual(len(forbidden), 1,
                                    f"seed {seed}: S-rank has no forbidden")

    def test_b_rank_rolls_one(self):
        for seed in range(20):
            rng = random.Random(100 + seed)
            skills = roll_enemy_skills(rng, 3)
            forbidden = [s for s in skills if is_enemy_skill(s)]
            self.assertEqual(len(forbidden), 1)


class PoisonDebuffTests(unittest.TestCase):
    def test_poison_ticks(self):
        f = new_fighter(hp=50)
        f["poison"] = {"turns": 3, "dmg": 6}
        notes = tick_fighter(f)
        self.assertEqual(f["hp"], 44)
        self.assertEqual(f["poison"]["turns"], 2)
        self.assertTrue(any("poison" in n for n in notes))

    def test_poison_expires(self):
        f = new_fighter(hp=50)
        f["poison"] = {"turns": 1, "dmg": 6}
        tick_fighter(f)
        self.assertNotIn("poison", f)
        self.assertEqual(f["hp"], 44)

    def test_poison_can_kill(self):
        f = new_fighter(hp=5)
        f["poison"] = {"turns": 2, "dmg": 6}
        tick_fighter(f)
        self.assertLessEqual(f["hp"], 0)

    def test_debuff_expires_and_restores(self):
        f = new_fighter(atk=10, dfn=5)
        f["atk"] = 6
        f["atk_debuff"] = {"turns": 1, "amt": 4}
        notes = tick_fighter(f)
        self.assertEqual(f["atk"], 10)
        self.assertNotIn("atk_debuff", f)
        self.assertTrue(any("wears off" in n for n in notes))

    def test_debuff_counts_down(self):
        f = new_fighter(atk=10, dfn=5)
        f["def"] = 1
        f["def_debuff"] = {"turns": 3, "amt": 4}
        tick_fighter(f)
        self.assertEqual(f["def"], 1)  # still drained
        self.assertEqual(f["def_debuff"]["turns"], 2)


class TitleTests(unittest.TestCase):
    def test_catalog_has_expected_titles(self):
        ids = {t.id for t in TITLE_CATALOG}
        for want in ("novice", "gladiator", "dragonslayer", "duelist",
                     "boss_hunter", "cutyps_heir", "unstoppable"):
            self.assertIn(want, ids)

    def test_novice_by_default(self):
        db = _memdb()
        store = TitleStore(db)
        self.assertIn("novice", store.unlocked("p1"))
        self.assertEqual(store.active("p1"), "Novice")

    def test_unlock_and_set_active(self):
        db = _memdb()
        store = TitleStore(db)
        self.assertTrue(store.unlock("p1", "gladiator"))
        self.assertFalse(store.unlock("p1", "gladiator"))  # idempotent
        self.assertTrue(store.set_active("p1", "gladiator"))
        self.assertEqual(store.active("p1"), "Gladiator")
        # can't wear what you haven't earned
        self.assertFalse(store.set_active("p1", "dragonslayer"))

    def test_check_unlocks_from_achievements(self):
        db = _memdb()
        unlock_achievement(db, "p1", "arena_s_rank")
        store = TitleStore(db)
        # the achievement hook already unlocked it
        self.assertIn("dragonslayer", store.unlocked("p1"))
        # a second check finds nothing new (idempotent)
        self.assertEqual(store.check_unlocks("p1"), [])

    def test_cutyp_heir_from_skill(self):
        db = _memdb()
        store = TitleStore(db)
        new = store.check_unlocks("p1", unlocked_achievements=set(),
                                  learned_skills={"slaying_force"})
        self.assertIn("Cutyp's Heir", new)

    def test_achievement_unlock_triggers_titles(self):
        db = _memdb()
        # unlock_achievement itself checks title unlocks
        self.assertTrue(unlock_achievement(db, "p2", "arena_pvp_win"))
        store = TitleStore(db)
        self.assertIn("duelist", store.unlocked("p2"))


class DailyHuntTests(unittest.TestCase):
    def test_complete_once(self):
        db = _memdb()
        self.assertFalse(daily_hunt_done(db, "p1"))
        self.assertTrue(complete_daily_hunt(db, "p1"))
        self.assertTrue(daily_hunt_done(db, "p1"))
        # second completion same day is a no-op
        self.assertFalse(complete_daily_hunt(db, "p1"))

    def test_per_player(self):
        db = _memdb()
        complete_daily_hunt(db, "p1")
        self.assertFalse(daily_hunt_done(db, "p2"))

    def test_hunt_date_format(self):
        d = hunt_date()
        self.assertEqual(len(d), 10)
        self.assertEqual(d[4], "-")


class NewAchievementTests(unittest.TestCase):
    def test_arena_achievements_exist(self):
        ids = {a.id for a in ACHIEVEMENTS}
        for want in ("arena_streak_5", "arena_streak_10", "arena_s_rank",
                     "arena_flawless", "arena_upset", "arena_skill_kill",
                     "arena_brutal", "arena_forbidden", "arena_pvp_win",
                     "arena_raid_win"):
            self.assertIn(want, ids)

    def test_unlock_new_achievements(self):
        db = _memdb()
        self.assertTrue(unlock_achievement(db, "p1", "arena_s_rank"))
        self.assertFalse(unlock_achievement(db, "p1", "arena_s_rank"))


if __name__ == "__main__":
    unittest.main()
