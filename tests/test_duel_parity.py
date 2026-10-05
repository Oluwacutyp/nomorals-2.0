"""Game identity unification + duel/arena parity.

Covers:
- Player.from_sender folds telegram-bot -> telegram for the game key
  (same human, one profile) while keeping the real platform for routing.
- Migration 76 merges pre-existing telegram-bot: rows into telegram:.
- Duel fighters mirror level, gear, skills, RPG attributes and titles.
- Dual-cast (combo <a> + <b>) works in duels and raids.
- Skill mana costs are enforced in duels (arena parity).
"""
from __future__ import annotations

import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.games.pvp import DuelGame, RaidGame
from nomorals.games.players import (
    GAME_IDENTITY_ALIASES, Player, PlayerStore,
)
from nomorals.games.relay import GameRelay
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db):
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


ADA = Player.from_sender("telegram", "111", "Ada")
BOB = Player.from_sender("telegram", "222", "Bob")


class IdentityTests(unittest.TestCase):
    def test_alias_map_covers_bot_endpoint(self):
        self.assertEqual(GAME_IDENTITY_ALIASES.get("telegram-bot"),
                         "telegram")

    def test_from_sender_unifies_key(self):
        via_bot = Player.from_sender("telegram-bot", "999", "Me")
        via_ub = Player.from_sender("telegram", "999", "Me")
        self.assertEqual(via_bot.key, via_ub.key)
        self.assertEqual(via_bot.key, "telegram:999")

    def test_from_sender_keeps_routing_platform(self):
        via_bot = Player.from_sender("telegram-bot", "999", "Me")
        self.assertEqual(via_bot.platform, "telegram-bot")

    def test_unrelated_platforms_stay_split(self):
        tg = Player.from_sender("telegram", "999", "Me")
        dc = Player.from_sender("discord", "999", "Me")
        self.assertNotEqual(tg.key, dc.key)


class IdentityMergeMigrationTests(unittest.TestCase):
    """Migration 76: telegram-bot: rows merge into telegram:."""

    def _db_with(self, players):
        db = Database(":memory:")
        # migrate to 75 only, then seed legacy rows, then run 76
        from nomorals.storage.migrations import MIGRATIONS
        from nomorals.storage.schema import MigrationRunner
        runner = MigrationRunner(db)
        for m in MIGRATIONS:
            if m.version <= 75:
                runner.apply(m)
        for key, xp, coins in players:
            db.execute(
                "INSERT INTO game_players (player_key, platform, display, "
                "coins, xp) VALUES (?, ?, ?, ?, ?)",
                (key, "telegram-bot" if key.startswith("telegram-bot")
                 else "telegram", key, coins, xp))
        m76 = next(m for m in MIGRATIONS if m.version == 76)
        runner.apply(m76)
        return db

    def test_pure_rename_when_no_twin(self):
        db = self._db_with([("telegram-bot:111", 300, 50)])
        rows = db.query("SELECT player_key, xp, coins FROM game_players")
        keys = [r["player_key"] for r in rows]
        self.assertEqual(keys, ["telegram:111"])
        self.assertEqual(rows[0]["xp"], 300)
        self.assertEqual(rows[0]["coins"], 50)

    def test_merge_sums_counters_when_twin_exists(self):
        db = self._db_with([("telegram:111", 300, 50),
                            ("telegram-bot:111", 200, 25)])
        rows = db.query("SELECT player_key, xp, coins FROM game_players")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["player_key"], "telegram:111")
        self.assertEqual(rows[0]["xp"], 500)
        self.assertEqual(rows[0]["coins"], 75)

    def test_gear_rows_follow(self):
        db = Database(":memory:")
        from nomorals.storage.migrations import MIGRATIONS
        from nomorals.storage.schema import MigrationRunner
        runner = MigrationRunner(db)
        for m in MIGRATIONS:
            if m.version <= 75:
                runner.apply(m)
        db.execute(
            "INSERT INTO game_players (player_key) VALUES ('telegram:111')")
        db.execute(
            "INSERT INTO game_gear (id, player_key, slug, durability, "
            "max_durability) VALUES ('g1', 'telegram-bot:111', "
            "'katana_common', 10, 10)")
        m76 = next(m for m in MIGRATIONS if m.version == 76)
        runner.apply(m76)
        rows = db.query("SELECT player_key FROM game_gear")
        self.assertEqual([r["player_key"] for r in rows], ["telegram:111"])


def _kit_ada(engine, db):
    """Level 3 + gear + skills + RPG attributes for ADA."""
    from nomorals.games.progression import award_xp
    prof = engine.store.get(ADA.key)
    award_xp(engine.store, prof, 300, "test")
    from nomorals.games.gear import GearStore, GEAR_CATALOG
    gs = GearStore(db)
    for slug, defn in GEAR_CATALOG.items():
        if defn.slot == "weapon":
            w = gs.grant(ADA.key, slug)
            gs.equip(ADA.key, w.id)
            break
    for slug, defn in GEAR_CATALOG.items():
        if defn.slot == "armor":
            a = gs.grant(ADA.key, slug)
            gs.equip(ADA.key, a.id)
            break
    from nomorals.games.skills import SkillStore
    ss = SkillStore(db)
    ss.learn(ADA.key, "war_cry")
    ss.learn(ADA.key, "dragon_punch")
    from nomorals.games.stats import StatStore
    sts = StatStore(db)
    blk = sts.get(ADA.key)
    blk.strength += 10
    blk.stamina += 8
    blk.mana += 20
    blk.intelligence += 12
    sts._save(ADA.key, blk)
    return gs, ss, sts


class DuelParityTests(unittest.TestCase):
    def _duel(self):
        engine, db, _sent = make_engine()
        _kit_ada(engine, db)
        # challenge THROUGH the bot endpoint — the user's failing path
        challenger = Player.from_sender("telegram-bot", "111", "Ada")
        relay = GameRelay(engine)
        inv = relay.create_invite("telegram-bot:c1", challenger, "pvp",
                                  to_label="Bob")
        rr = relay.accept_invite(inv.code, "telegram:c2", BOB)
        room = engine._rooms.get(rr.virtual_chat)
        return engine, room, rr

    def test_duel_mirrors_level_gear_skills(self):
        _engine, room, _rr = self._duel()
        st = room.state
        f = st["fighters"][ADA.key]
        # level 3: +8 HP, +2 atk, +1 def over base
        self.assertGreaterEqual(f["max_hp"], 58)
        self.assertGreater(f["atk"], 12)   # gear folded in
        self.assertGreater(f["def"], 6)
        self.assertIn("war_cry", st["skills"][ADA.key])
        self.assertTrue(st["loadout"][ADA.key])

    def test_duel_applies_rpg_attributes(self):
        _engine, room, _rr = self._duel()
        f = room.state["fighters"][ADA.key]
        # stamina 8 -> +24 max HP ; strength 10 -> +5 atk
        self.assertEqual(f["max_hp"], 50 + 8 + 24)
        self.assertIn("max_mana", f)
        self.assertEqual(f["max_mana"], 30 + 20 * 2)
        self.assertEqual(f["intelligence"], 12)

    def test_duel_intro_shows_kit(self):
        _engine, room, _rr = self._duel()
        g = DuelGame()
        intro = g._intro(room, ADA, BOB)
        self.assertIn("level 3", intro)
        self.assertIn("wielding", intro)
        self.assertIn("🥋", intro)

    def test_skill_cast_burns_mana(self):
        engine, room, rr = self._duel()
        st = room.state
        before = st["fighters"][ADA.key]["mana"]
        challenger = Player.from_sender("telegram-bot", "111", "Ada")
        msgs = engine.move(rr.virtual_chat, "skill war cry", challenger)
        self.assertTrue(any("ROARS" in m for m in msgs))
        after = st["fighters"][ADA.key]["mana"]
        self.assertLess(after, before)

    def test_dual_cast_weaves_in_duel(self):
        engine, room, rr = self._duel()
        st = room.state
        room.turn = 0  # Ada moves first
        st["fighters"][ADA.key]["mana"] = st["fighters"][ADA.key]["max_mana"]
        challenger = Player.from_sender("telegram-bot", "111", "Ada")
        msgs = engine.move(rr.virtual_chat, "combo war cry + dragon punch",
                           challenger)
        body = " ".join(msgs)
        # either the weave lands or it unravels — both prove the path ran
        self.assertTrue("DUAL-CAST" in body or "unravels" in body, body)

    def test_dual_cast_rejects_unknown_pairing(self):
        engine, room, rr = self._duel()
        room.turn = 0
        challenger = Player.from_sender("telegram-bot", "111", "Ada")
        msgs = engine.move(rr.virtual_chat, "combo war cry + nosuchskill",
                           challenger)
        self.assertTrue(any("no such pairing" in m for m in msgs))

    def test_recache_does_not_stack_or_heal(self):
        _engine, room, _rr = self._duel()
        g = DuelGame()
        st = room.state
        f = st["fighters"][ADA.key]
        atk0, max0 = f["atk"], f["max_hp"]
        f["hp"] = 10  # battle damage
        g._recache_fighter(room, ADA.key, ADA.key)
        f2 = st["fighters"][ADA.key]
        self.assertEqual(f2["atk"], atk0)
        self.assertEqual(f2["max_hp"], max0)
        self.assertEqual(f2["hp"], 10)  # damage preserved, no heal

    def test_raid_fighter_gets_kit_too(self):
        engine, db, _sent = make_engine()
        _kit_ada(engine, db)
        room, _msgs = engine.start("test:raid", "raid", ADA, kind="group")
        f = room.state["fighters"][ADA.key]
        self.assertGreater(f["max_hp"], 50)
        self.assertIn("max_mana", f)


if __name__ == "__main__":
    unittest.main()
