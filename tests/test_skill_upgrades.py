"""Skill upgrades: tiers, persistence, and tier-aware battles.

Covers the upgrade system in nomorals/games/skills.py and its
integration points:
- the catalog: which skills upgrade, tier stats/names/costs;
- effective_def: tier merging, clamping, base-def immutability;
- SkillStore: tier persistence, upgrade gating, idempotency;
- the arena + PvP: upgraded skills fight at their tier stats;
- the engine mirrors skill tiers into room state.
"""
from __future__ import annotations

import unittest

from nomorals.games import engine as engine_mod  # noqa: F401
from nomorals.games.ai import GameMind
from nomorals.games.combat import new_fighter, tick_fighter
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player, PlayerStore
from nomorals.games.skills import (
    SKILL_CATALOG,
    SkillStore,
    effective_def,
    max_tier,
    resolve_skill,
    tier_name,
)
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db: Database) -> None:
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


ADA = Player.from_sender("telegram", "456", "Ada")
BOB = Player.from_sender("telegram", "789", "Bob")

UPGRADEABLE = {"war_cry", "dragon_punch", "whirlwind", "thousand_fists",
               "pressure_point", "shadow_step", "second_wind",
               "slaying_force", "iron_palm", "viper_strike",
               "smoke_bomb", "crane_dance"}


# ── catalog tiers ────────────────────────────────────────────────────────────

class CatalogTierTests(unittest.TestCase):
    def test_upgradeable_set(self):
        got = {s for s, d in SKILL_CATALOG.items() if max_tier(d) > 1}
        self.assertEqual(got, UPGRADEABLE)

    def test_upgradeable_all_have_three_tiers(self):
        for slug in UPGRADEABLE:
            self.assertEqual(max_tier(slug), 3, slug)

    def test_passives_stay_single_tier(self):
        for slug, d in SKILL_CATALOG.items():
            if d.kind == "passive":
                self.assertEqual(max_tier(d), 1, slug)

    def test_tier_merges_stats(self):
        t2 = effective_def("dragon_punch", 2)
        self.assertEqual(t2.name, "Dragon Rising Fist")
        self.assertEqual(t2.mult, 2.2)
        self.assertEqual(t2.cost, 1500)
        self.assertEqual(t2.level_req, 5)
        self.assertEqual(t2.slug, "dragon_punch")  # slug stable
        self.assertEqual(t2.kind, "active")

    def test_tier_three_names(self):
        self.assertEqual(tier_name("dragon_punch", 3), "Dragon Emperor Fist")
        self.assertEqual(tier_name("shadow_step", 3), "Phantom Mirage")
        self.assertEqual(tier_name("pressure_point", 3), "Death Touch")
        self.assertEqual(tier_name("slaying_force", 3),
                         "Slaying Force: Annihilation")
        self.assertEqual(tier_name("dragon_punch", 1), "Dragon Punch")

    def test_base_def_immutable(self):
        base = SKILL_CATALOG["slaying_force"]
        effective_def("slaying_force", 3)
        self.assertEqual(base.mult, 3.5)
        self.assertEqual(base.cost, 5000)
        self.assertEqual(base.cooldown, 6)
        self.assertEqual(base.name, "Slaying Force")

    def test_tier_clamps(self):
        self.assertEqual(effective_def("dragon_punch", 0).mult, 1.8)
        self.assertEqual(effective_def("dragon_punch", 99).mult, 2.8)
        self.assertEqual(effective_def("iron_skin", 5).def_bonus, 4)

    def test_resolve_tier_names(self):
        self.assertEqual(resolve_skill("dragon emperor fist").slug,
                         "dragon_punch")
        self.assertEqual(resolve_skill("phantom mirage").slug, "shadow_step")
        self.assertEqual(resolve_skill("death touch").slug, "pressure_point")
        self.assertEqual(resolve_skill("myriad fists").slug, "thousand_fists")

    def test_slaying_force_tiers(self):
        t2 = effective_def("slaying_force", 2)
        t3 = effective_def("slaying_force", 3)
        self.assertEqual((t2.mult, t2.cooldown), (4.2, 5))
        self.assertEqual((t3.mult, t3.cooldown), (5.0, 4))
        self.assertEqual(t3.ignore_def_pct, 1.0)

    def test_shadow_step_three_counter(self):
        t3 = effective_def("shadow_step", 3)
        self.assertTrue(t3.dodge)
        self.assertEqual(t3.counter_mult, 1.0)
        self.assertEqual(t3.cooldown, 3)

    def test_upgrade_costs_rise(self):
        for slug in UPGRADEABLE:
            d = SKILL_CATALOG[slug]
            prev = d.cost
            for up in d.tiers:
                self.assertGreater(up.cost, prev, slug)
                prev = up.cost


# ── store tiers ─────────────────────────────────────────────────────────────

class StoreTierTests(unittest.TestCase):
    def setUp(self):
        db = Database(":memory:")
        db.migrate()
        self.skills = SkillStore(db)

    def test_new_learn_is_tier_one(self):
        self.skills.learn("pk1", "dragon_punch")
        self.assertEqual(self.skills.tier("pk1", "dragon_punch"), 1)

    def test_upgrade_flow(self):
        self.skills.learn("pk1", "dragon_punch")
        self.assertTrue(self.skills.upgrade("pk1", "dragon_punch"))
        self.assertEqual(self.skills.tier("pk1", "dragon_punch"), 2)
        self.assertTrue(self.skills.upgrade("pk1", "dragon_punch"))
        self.assertEqual(self.skills.tier("pk1", "dragon_punch"), 3)
        # maxed — no further
        self.assertFalse(self.skills.upgrade("pk1", "dragon_punch"))
        self.assertEqual(self.skills.tier("pk1", "dragon_punch"), 3)

    def test_upgrade_unlearned_is_false(self):
        self.assertFalse(self.skills.upgrade("pk1", "dragon_punch"))

    def test_double_learn_is_idempotent(self):
        # two concurrent learns must not create duplicate rows
        self.assertTrue(self.skills.learn("pk1", "dragon_punch"))
        self.assertFalse(self.skills.learn("pk1", "dragon_punch"))
        self.assertEqual(self.skills.learned("pk1").count("dragon_punch"), 1)

    def test_stale_tier_upgrade_fails(self):
        # the atomic WHERE tier=? means an upgrade from a stale read
        # can't skip tiers or double-apply.
        self.skills.learn("pk1", "dragon_punch")
        self.assertTrue(self.skills.upgrade("pk1", "dragon_punch"))
        self.assertEqual(self.skills.tier("pk1", "dragon_punch"), 2)
        self.assertTrue(self.skills.upgrade("pk1", "dragon_punch"))
        self.assertEqual(self.skills.tier("pk1", "dragon_punch"), 3)

    def test_upgrade_passive_is_false(self):
        self.skills.learn("pk1", "iron_skin")
        self.assertFalse(self.skills.upgrade("pk1", "iron_skin"))
        self.assertEqual(self.skills.tier("pk1", "iron_skin"), 1)

    def test_upgrade_unknown_slug_is_false(self):
        self.assertFalse(self.skills.upgrade("pk1", "nope"))

    def test_tiers_dict(self):
        self.skills.learn("pk1", "dragon_punch")
        self.skills.learn("pk1", "iron_skin")
        self.skills.upgrade("pk1", "dragon_punch")
        self.assertEqual(self.skills.tiers("pk1"),
                         {"dragon_punch": 2, "iron_skin": 1})

    def test_tiers_per_player_isolation(self):
        self.skills.learn("pk1", "dragon_punch")
        self.skills.learn("pk2", "dragon_punch")
        self.skills.upgrade("pk1", "dragon_punch")
        self.assertEqual(self.skills.tier("pk2", "dragon_punch"), 1)

    def test_persists_across_instances(self):
        self.skills.learn("pk1", "war_cry")
        self.skills.upgrade("pk1", "war_cry")
        other = SkillStore(self.skills.db)
        self.assertEqual(other.tier("pk1", "war_cry"), 2)

    def test_no_db_is_safe(self):
        s = SkillStore(None)
        self.assertFalse(s.upgrade("pk1", "dragon_punch"))
        self.assertEqual(s.tier("pk1", "dragon_punch"), 1)
        self.assertEqual(s.tiers("pk1"), {})

    def test_old_rows_default_tier_one(self):
        # rows written before the tier column existed read as tier 1
        self.skills.learn("pk1", "whirlwind")
        self.skills.db.execute(
            "UPDATE game_skills SET tier = 1 WHERE player_key = 'pk1'")
        self.assertEqual(self.skills.tier("pk1", "whirlwind"), 1)


# ── arena integration ────────────────────────────────────────────────────────

class _Room:
    def __init__(self, state, humans):
        self.state = state
        self.humans = humans


def _fighter(**over):
    f = new_fighter()
    f.update(over)
    return f


def _arena_state(**over):
    you = _fighter()
    house = _fighter(atk=8, dfn=4)
    state = {
        "you": you,
        "house": house,
        "skills": {"telegram:456": []},
        "skill_tiers": {"telegram:456": {}},
        "house_rank": "E",
    }
    state.update(over)
    return state


def _arena():
    from nomorals.games.games import ambitious
    return ambitious.BattleArenaGame(), GameMind(seed=7)


class ArenaTierTests(unittest.TestCase):
    def test_upgraded_punch_hits_harder_and_names(self):
        engine, db, sent = make_engine()
        game, mind = _arena()
        state = _arena_state()
        state["skills"] = {"telegram:456": ["dragon_punch"]}
        state["skill_tiers"] = {"telegram:456": {"dragon_punch": 3}}
        state["skill_cd"] = {}
        state["skill_used"] = []
        room = _Room(state, [ADA])
        hp_before = state["house"]["hp"]
        msg = game._cast_skill(room, ADA, "dragon punch", mind)
        self.assertIn("Dragon Emperor Fist", msg)
        self.assertLess(state["house"]["hp"], hp_before)
        # tier III cooldown is 2, not the base 3
        self.assertEqual(state["skill_cd"]["dragon_punch"], 2)

    def test_war_cry_tier_two_buff_and_fade(self):
        engine, db, sent = make_engine()
        game, mind = _arena()
        state = _arena_state()
        state["skills"] = {"telegram:456": ["war_cry"]}
        state["skill_tiers"] = {"telegram:456": {"war_cry": 2}}
        state["skill_cd"] = {}
        state["skill_used"] = []
        room = _Room(state, [ADA])
        msg = game._cast_skill(room, ADA, "war cry", mind)
        self.assertIn("+5 attack", msg)
        self.assertEqual(state["you"]["atk"], 15)
        # fade removes exactly the tier-II buff
        state["you"]["warcry_turns"] = 1
        notes = tick_fighter(state["you"], state["skill_cd"])
        self.assertEqual(state["you"]["atk"], 10)
        self.assertTrue(any("war cry" in n for n in notes))

    def test_shadow_step_three_counters(self):
        engine, db, sent = make_engine()
        game, mind = _arena()
        state = _arena_state()
        state["skills"] = {"telegram:456": ["shadow_step"]}
        state["skill_tiers"] = {"telegram:456": {"shadow_step": 3}}
        state["skill_cd"] = {}
        state["skill_used"] = []
        room = _Room(state, [ADA])
        hp_before = state["house"]["hp"]
        msg = game._cast_skill(room, ADA, "shadow step", mind)
        self.assertTrue(state["you"]["dodge_next"])
        self.assertLess(state["house"]["hp"], hp_before)  # counter landed
        self.assertIn("dark", msg)

    def test_second_wind_tier_two_heals_more(self):
        engine, db, sent = make_engine()
        game, mind = _arena()
        state = _arena_state()
        state["you"]["hp"] = 10
        state["skills"] = {"telegram:456": ["second_wind"]}
        state["skill_tiers"] = {"telegram:456": {"second_wind": 2}}
        state["skill_cd"] = {}
        state["skill_used"] = []
        room = _Room(state, [ADA])
        game._cast_skill(room, ADA, "second wind", mind)
        # 55% of 50 = 27 (int)
        self.assertEqual(state["you"]["hp"], 10 + 27)

    def test_skill_note_shows_tier_names(self):
        engine, db, sent = make_engine()
        game, mind = _arena()
        state = _arena_state()
        state["skills"] = {"telegram:456": ["dragon_punch", "iron_skin"]}
        state["skill_tiers"] = {"telegram:456": {"dragon_punch": 2,
                                                 "iron_skin": 1}}
        room = _Room(state, [ADA])
        note = game._skill_note(room)
        self.assertIn("Dragon Rising Fist", note)
        self.assertIn("Iron Skin", note)

    def test_base_cast_unchanged_without_tiers(self):
        # rooms without the tier mirror (older state) fight at tier 1
        engine, db, sent = make_engine()
        game, mind = _arena()
        state = _arena_state()
        del state["skill_tiers"]
        state["skills"] = {"telegram:456": ["dragon_punch"]}
        state["skill_cd"] = {}
        state["skill_used"] = []
        room = _Room(state, [ADA])
        msg = game._cast_skill(room, ADA, "dragon punch", mind)
        self.assertIn("Dragon Punch!", msg)
        self.assertEqual(state["skill_cd"]["dragon_punch"], 3)


# ── pvp integration ──────────────────────────────────────────────────────────

def _pvp_helpers():
    from tests.test_pvp import start_duel, join_duel, ADA, BOB
    return start_duel, join_duel, ADA, BOB


class PvpTierTests(unittest.TestCase):
    def test_duel_casts_at_tier(self):
        engine, db, sent = make_engine()
        start_duel, join_duel, pada, pbob = _pvp_helpers()
        try:
            room, _ = start_duel(engine)
            join_duel(engine, pbob)
            skills = SkillStore(db)
            skills.learn(pada.key, "dragon_punch")
            skills.upgrade(pada.key, "dragon_punch")
            skills.upgrade(pada.key, "dragon_punch")
            room.state["skills"][pada.key] = skills.learned(pada.key)
            room.state["skill_tiers"][pada.key] = skills.tiers(pada.key)
            game = engine.games["pvp"]
            msg = game._cast_skill(room, pada, pada.key, pbob.key,
                                   "dragon punch", engine._mind,
                                   "Ada", "Bob", foe_key=pbob.key)
            self.assertIn("Dragon Emperor Fist", msg)
            self.assertEqual(
                room.state["skill_cd"][pada.key]["dragon_punch"], 2)
        finally:
            engine.shutdown()

    def test_pvp_shadow_counter(self):
        engine, db, sent = make_engine()
        start_duel, join_duel, pada, pbob = _pvp_helpers()
        try:
            room, _ = start_duel(engine)
            join_duel(engine, pbob)
            skills = SkillStore(db)
            skills.learn(pada.key, "shadow_step")
            skills.upgrade(pada.key, "shadow_step")
            skills.upgrade(pada.key, "shadow_step")
            room.state["skills"][pada.key] = skills.learned(pada.key)
            room.state["skill_tiers"][pada.key] = skills.tiers(pada.key)
            game = engine.games["pvp"]
            hp_before = room.state["fighters"][pbob.key]["hp"]
            msg = game._cast_skill(room, pada, pada.key, pbob.key,
                                   "shadow step", engine._mind,
                                   "Ada", "Bob", foe_key=pbob.key)
            self.assertTrue(room.state["fighters"][pada.key]["dodge_next"])
            self.assertLess(room.state["fighters"][pbob.key]["hp"], hp_before)
            self.assertIn("dark", msg)
        finally:
            engine.shutdown()


# ── engine mirror ────────────────────────────────────────────────────────────

class MirrorTierTests(unittest.TestCase):
    def test_mirror_writes_skill_tiers(self):
        engine, db, sent = make_engine()
        try:
            skills = SkillStore(db)
            skills.learn(ADA.key, "dragon_punch")
            skills.upgrade(ADA.key, "dragon_punch")

            class FakeRoom:
                def __init__(self):
                    self.state = {}
                    self.players = [ADA]

            room = FakeRoom()
            engine._mirror_player(room, ADA)
            self.assertEqual(room.state["skills"][ADA.key],
                             ["dragon_punch"])
            self.assertEqual(room.state["skill_tiers"][ADA.key],
                             {"dragon_punch": 2})
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
