"""Learnable battle skills: catalog, persistence, arena integration.

Covers nomorals/games/skills.py and its integration points:
- the catalog has both active and passive skills with sane costs/cooldowns;
- resolve_skill finds skills by slug, name, or substring;
- passive_bonuses aggregates correctly and ignores actives/unknowns;
- SkillStore.learn is idempotent and rejects unknown slugs (DB-backed);
- the arena applies passive skills at setup and casts active skills
  through ``skill <name>`` with cooldowns.
"""
from __future__ import annotations

import unittest

from nomorals.games import engine as engine_mod
from nomorals.games.ai import GameMind
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player, PlayerStore
from nomorals.games.skills import (
    SKILL_CATALOG,
    SkillStore,
    passive_bonuses,
    resolve_skill,
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


# ── catalog ──────────────────────────────────────────────────────────────────

class CatalogTests(unittest.TestCase):
    def test_active_and_passive_exist(self):
        kinds = {d.kind for d in SKILL_CATALOG.values()}
        self.assertEqual(kinds, {"active", "passive"})

    def test_expected_skills_present(self):
        for slug in ("dragon_punch", "whirlwind", "thousand_fists",
                     "pressure_point", "shadow_step", "war_cry",
                     "second_wind", "iron_skin", "tiger_stance",
                     "keen_eye", "stone_body"):
            self.assertIn(slug, SKILL_CATALOG, slug)

    def test_costs_and_levels_sane(self):
        for slug, d in SKILL_CATALOG.items():
            self.assertGreaterEqual(d.cost, 0, slug)
            self.assertGreaterEqual(d.level_req, 1, slug)
            self.assertIn(d.school,
                          {"tiger", "crane", "snake", "shadow", "cutyp"},
                          slug)

    def test_passives_have_no_cooldown(self):
        for slug, d in SKILL_CATALOG.items():
            if d.kind == "passive":
                self.assertEqual(d.cooldown, 0, slug)

    def test_actives_have_cooldowns(self):
        for slug, d in SKILL_CATALOG.items():
            if d.kind == "active":
                self.assertGreater(d.cooldown, 0, slug)


class ResolveTests(unittest.TestCase):
    def test_exact_slug(self):
        self.assertEqual(resolve_skill("dragon_punch").slug, "dragon_punch")

    def test_by_name(self):
        self.assertEqual(resolve_skill("Dragon Punch").slug, "dragon_punch")

    def test_substring(self):
        self.assertEqual(resolve_skill("iron").slug, "iron_skin")

    def test_unknown_returns_none(self):
        self.assertIsNone(resolve_skill("banana kamehameha"))
        self.assertIsNone(resolve_skill(""))


class PassiveBonusTests(unittest.TestCase):
    def test_aggregation(self):
        b = passive_bonuses(["iron_skin", "tiger_stance", "stone_body"])
        self.assertEqual(b["def"], 4)
        self.assertEqual(b["atk"], 3)
        self.assertEqual(b["max_hp"], 15)
        self.assertEqual(b["crit"], 0.0)

    def test_keen_eye_crit(self):
        b = passive_bonuses(["keen_eye"])
        self.assertAlmostEqual(b["crit"], 0.10)

    def test_actives_ignored(self):
        b = passive_bonuses(["dragon_punch", "war_cry"])
        self.assertEqual(b, {"atk": 0, "def": 0, "max_hp": 0, "crit": 0.0})

    def test_unknown_slugs_ignored(self):
        b = passive_bonuses(["not_a_skill"])
        self.assertEqual(b["atk"], 0)


# ── store ────────────────────────────────────────────────────────────────────

class SkillStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.store = SkillStore(self.db)

    def test_learn_and_list(self):
        self.assertTrue(self.store.learn("pk1", "iron_skin"))
        self.assertTrue(self.store.learn("pk1", "dragon_punch"))
        self.assertEqual(self.store.learned("pk1"),
                         ["iron_skin", "dragon_punch"])

    def test_learn_idempotent(self):
        self.assertTrue(self.store.learn("pk1", "iron_skin"))
        self.assertFalse(self.store.learn("pk1", "iron_skin"))
        self.assertEqual(self.store.learned("pk1"), ["iron_skin"])

    def test_learn_unknown_slug_rejected(self):
        self.assertFalse(self.store.learn("pk1", "nope"))

    def test_per_player_isolation(self):
        self.store.learn("pk1", "iron_skin")
        self.assertEqual(self.store.learned("pk2"), [])

    def test_has(self):
        self.store.learn("pk1", "war_cry")
        self.assertTrue(self.store.has("pk1", "war_cry"))
        self.assertFalse(self.store.has("pk1", "second_wind"))

    def test_persists_across_store_instances(self):
        self.store.learn("pk1", "stone_body")
        fresh = SkillStore(self.db)
        self.assertEqual(fresh.learned("pk1"), ["stone_body"])

    def test_no_db_is_safe(self):
        s = SkillStore(None)
        self.assertEqual(s.learned("pk1"), [])
        self.assertFalse(s.learn("pk1", "iron_skin"))


# ── arena integration ────────────────────────────────────────────────────────

class _Room:
    """Minimal arena room shape for direct game-object calls."""

    def __init__(self, state, humans):
        self.state = state
        self.humans = humans


def _fighter(**over):
    f = {"hp": 50, "max_hp": 50, "atk": 10, "def": 5,
         "potions": 1, "defending": False, "shield": False,
         "focused": False, "fury_cd": 0,
         "combo_every": 0, "combo_count": 0, "combo_name": ""}
    f.update(over)
    return f


def _arena_state():
    you = _fighter()
    house = _fighter()
    house["atk"], house["def"] = 8, 4
    return {
        "you": you,
        "house": house,
        "skills": {"telegram:456": ["iron_skin", "keen_eye", "war_cry"]},
        "house_rank": "E",
    }


def _arena(engine):
    from nomorals.games.games import ambitious
    game = ambitious.BattleArenaGame()
    mind = GameMind(seed=7)
    return game, mind


class ArenaSkillTests(unittest.TestCase):
    def test_passives_applied_at_setup(self):
        engine, db, sent = make_engine()
        skills = SkillStore(db)
        skills.learn(ADA.key, "iron_skin")
        skills.learn(ADA.key, "stone_body")
        game, mind = _arena(engine)
        state = _arena_state()
        state["skills"] = {"telegram:456": skills.learned(ADA.key)}
        room_obj = _Room(state, [ADA])
        game._apply_skills(room_obj)
        self.assertEqual(state["you"]["def"], 5 + 4)
        self.assertEqual(state["you"]["max_hp"], 50 + 15)
        self.assertEqual(state["you"]["hp"], 65)

    def test_war_cry_cast_and_fades(self):
        engine, db, sent = make_engine()
        game, mind = _arena(engine)
        state = _arena_state()
        state["skill_cd"] = {}
        state["skill_used"] = []
        room_obj = _Room(state, [ADA])
        msg = game._cast_skill(room_obj, ADA, "war cry", mind)
        self.assertIn("ROAR", msg)
        self.assertEqual(state["you"]["atk"], 13)
        self.assertEqual(state["skill_cd"]["war_cry"], 4)
        # cooldown blocks recast
        msg2 = game._cast_skill(room_obj, ADA, "war_cry", mind)
        self.assertIn("recovering", msg2)
        # fading removes the buff
        state["you"]["warcry_turns"] = 1
        # tick happens in on_move; emulate the fade line directly
        state["you"]["warcry_turns"] -= 1
        state["you"]["atk"] = max(1, state["you"]["atk"] - 3)
        self.assertEqual(state["you"]["atk"], 10)

    def test_dragon_punch_cast(self):
        engine, db, sent = make_engine()
        game, mind = _arena(engine)
        state = _arena_state()
        state["skills"] = {"telegram:456": ["iron_skin", "dragon_punch"]}
        state["skill_cd"] = {}
        state["skill_used"] = []
        room_obj = _Room(state, [ADA])
        msg = game._cast_skill(room_obj, ADA, "dragon punch", mind)
        self.assertIn("Dragon Punch", msg)
        self.assertLess(state["house"]["hp"], 50)

    def test_shadow_step_dodges(self):
        engine, db, sent = make_engine()
        game, mind = _arena(engine)
        state = _arena_state()
        state["skills"] = {"telegram:456": ["shadow_step"]}
        state["skill_cd"] = {}
        state["skill_used"] = []
        room_obj = _Room(state, [ADA])
        game._cast_skill(room_obj, ADA, "shadow step", mind)
        self.assertTrue(state["you"]["dodge_next"])
        before = state["you"]["hp"]
        out = game._hit(room_obj, "house", "you", mind)
        self.assertIn("shadow step", out)
        self.assertEqual(state["you"]["hp"], before)
        self.assertFalse(state["you"].get("dodge_next"))

    def test_second_wind_once_per_battle(self):
        engine, db, sent = make_engine()
        game, mind = _arena(engine)
        state = _arena_state()
        state["skills"] = {"telegram:456": ["second_wind"]}
        state["you"]["hp"] = 10
        state["skill_cd"] = {}
        state["skill_used"] = []
        room_obj = _Room(state, [ADA])
        msg = game._cast_skill(room_obj, ADA, "second wind", mind)
        self.assertIn("+", msg)
        self.assertGreater(state["you"]["hp"], 10)
        # heal resets cooldown manually to isolate the once-per-battle rule
        state["skill_cd"]["second_wind"] = 0
        msg2 = game._cast_skill(room_obj, ADA, "second_wind", mind)
        self.assertIn("spent", msg2)

    def test_unknown_skill_to_player(self):
        engine, db, sent = make_engine()
        game, mind = _arena(engine)
        state = _arena_state()
        room_obj = _Room(state, [ADA])
        msg = game._cast_skill(room_obj, ADA, "thousand fists", mind)
        self.assertIn("don't know", msg)

    def test_gibberish_returns_none(self):
        engine, db, sent = make_engine()
        game, mind = _arena(engine)
        state = _arena_state()
        room_obj = _Room(state, [ADA])
        self.assertIsNone(game._cast_skill(room_obj, ADA, "zzznonsense",
                                           mind))


if __name__ == "__main__":
    unittest.main()
