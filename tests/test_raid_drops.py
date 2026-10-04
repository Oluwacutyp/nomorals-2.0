"""Enemy gear scaling by rank + difficulty, and raid-exclusive drops.

- difficulty (easy/normal/hard/expert) is a real dial on arena
  hunters: stats, gear grade pools, and technique counts all move.
- each hunter rank has its own gear "class" via GEAR_EFFECTIVENESS.
- raid-exclusive gear is never sold; it only drops from raid bosses
  (5-15% per player, scaled by damage share).
"""
from __future__ import annotations

import random
import unittest

from nomorals.games.economy import GameEconomy
from nomorals.games.enemies import (
    DIFFICULTY_MODS,
    GEAR_EFFECTIVENESS,
    roll_enemy,
    roll_enemy_gear,
    roll_enemy_skills,
)
from nomorals.games.engine import GameEngine
from nomorals.games.games.pvp import RaidGame
from nomorals.games.gear import (
    GEAR_CATALOG,
    RAID_EXCLUSIVE_GEAR,
    SET_BONUSES,
    GearStore,
    detect_set_bonus,
)
from nomorals.games.players import Player
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

FOE_BASE = {"max_hp": 74, "atk": 13, "def": 8}


def enemy_power(diff, seed, rank_idx=5):
    """Total rolled power incl. difficulty-scaled base + gear."""
    from nomorals.games.power import fighter_power
    e = roll_enemy(random.Random(seed), rank_idx, player_power=0,
                   foe_base=dict(FOE_BASE), difficulty=diff)
    mult = DIFFICULTY_MODS[diff]["stat_mult"]
    base = {"max_hp": int(FOE_BASE["max_hp"] * mult),
            "atk": int(FOE_BASE["atk"] * mult),
            "def": int(FOE_BASE["def"] * mult)}
    w = (e["gear"].get("weapon") or {}).get("atk", 0)
    a = (e["gear"].get("armor") or {}).get("def", 0)
    eff = GEAR_EFFECTIVENESS[rank_idx]
    base["atk"] += int(w * eff)
    base["def"] += int(a * eff)
    return fighter_power(base, tuple(e["skills"]), e["skills"])


class EnemyDifficultyTests(unittest.TestCase):
    def test_difficulty_orders_enemy_power(self):
        avgs = {}
        for diff in ("easy", "normal", "hard", "expert"):
            avgs[diff] = (sum(enemy_power(diff, s) for s in range(30))
                          / 30)
        self.assertLess(avgs["easy"], avgs["normal"])
        self.assertLess(avgs["normal"], avgs["hard"])
        self.assertLess(avgs["hard"], avgs["expert"])

    def test_difficulty_mods_shape(self):
        self.assertLess(DIFFICULTY_MODS["easy"]["stat_mult"], 1.0)
        self.assertEqual(DIFFICULTY_MODS["normal"]["stat_mult"], 1.0)
        self.assertGreater(DIFFICULTY_MODS["hard"]["stat_mult"], 1.0)
        self.assertGreater(DIFFICULTY_MODS["expert"]["stat_mult"],
                           DIFFICULTY_MODS["hard"]["stat_mult"])

    def test_hard_shifts_gear_pool_up(self):
        # SS-rank hard rolls the X pool (legendary/myth) — over many
        # rolls it must show myth pieces where normal never does.
        seen_myth = {"normal": False, "hard": False}
        for seed in range(60):
            for diff in ("normal", "hard"):
                g = roll_enemy_gear(random.Random(seed), 6,
                                    difficulty=diff)
                for piece in g.values():
                    if piece.get("grade") == "myth":
                        seen_myth[diff] = True
        self.assertTrue(seen_myth["hard"])
        self.assertFalse(seen_myth["normal"])

    def test_easy_rolls_fewer_skills_than_expert(self):
        easy_n = sum(len(roll_enemy_skills(random.Random(s), 5,
                                           difficulty="easy"))
                     for s in range(40))
        expert_n = sum(len(roll_enemy_skills(random.Random(s), 5,
                                             difficulty="expert"))
                       for s in range(40))
        self.assertLess(easy_n, expert_n)

    def test_expert_bonus_potion_at_high_rank(self):
        e = roll_enemy(random.Random(1), 5, difficulty="expert")
        n = roll_enemy(random.Random(1), 5, difficulty="normal")
        self.assertEqual(e["potions"], n["potions"] + 1)

    def test_no_bonus_potion_at_low_rank(self):
        e = roll_enemy(random.Random(1), 1, difficulty="expert")
        n = roll_enemy(random.Random(1), 1, difficulty="normal")
        self.assertEqual(e["potions"], n["potions"])

    def test_unknown_difficulty_defaults_to_normal(self):
        e = roll_enemy(random.Random(3), 5, player_power=0,
                       foe_base=dict(FOE_BASE), difficulty="bogus")
        self.assertEqual(e["difficulty"], "bogus")  # recorded as-is
        # but the mods fall back to normal's numbers
        n = roll_enemy(random.Random(3), 5, player_power=0,
                       foe_base=dict(FOE_BASE), difficulty="normal")
        self.assertEqual(sorted(e["skills"]), sorted(n["skills"]))

    def test_gear_effectiveness_strictly_increases(self):
        effs = [GEAR_EFFECTIVENESS[i] for i in range(8)]
        for a, b in zip(effs, effs[1:]):
            self.assertLess(a, b)

    def test_raid_gear_never_rolls_on_enemies(self):
        raid_slugs = set(RAID_EXCLUSIVE_GEAR)
        for seed in range(50):
            g = roll_enemy_gear(random.Random(seed), 7,
                                difficulty="expert")
            for piece in g.values():
                self.assertNotIn(piece.get("slug"), raid_slugs)


class RaidGearTests(unittest.TestCase):
    def test_raid_gear_exists_and_flagged(self):
        self.assertTrue(len(RAID_EXCLUSIVE_GEAR) >= 4)
        for slug in RAID_EXCLUSIVE_GEAR:
            defn = GEAR_CATALOG[slug]
            self.assertTrue(defn.raid_only)
            self.assertEqual(defn.cost, 0)

    def test_raid_gear_not_in_shop(self):
        econ = GameEconomy.__new__(GameEconomy)
        text = GameEconomy.gear_catalog_text(econ)
        for slug in RAID_EXCLUSIVE_GEAR:
            self.assertNotIn(slug, text)
            self.assertNotIn(GEAR_CATALOG[slug].name.split(" [")[0],
                             text)

    def test_raid_gear_purchase_blocked(self):
        engine, db, sent = make_engine()
        try:
            econ = GameEconomy(engine.store, gear_store=engine.gear)
            for slug in RAID_EXCLUSIVE_GEAR:
                ok, msg = econ.purchase(ADA, slug)
                self.assertFalse(ok)
                self.assertIn("raid", msg.lower())
        finally:
            engine.shutdown()

    def test_bossbane_set_bonus(self):
        bonus = detect_set_bonus({"weapon": "bossbane_cleaver",
                                  "armor": "bossbane_plate"})
        self.assertIsNotNone(bonus)
        self.assertEqual(bonus.set_name, "bossbane")
        self.assertEqual(SET_BONUSES["bossbane"].combo_name,
                         "bossbane rend")

    def test_raid_gear_grantable_and_equipable(self):
        db = Database(":memory:")
        db.migrate()
        store = GearStore(db)
        inst = store.grant("p1", RAID_EXCLUSIVE_GEAR[0])
        self.assertEqual(inst.slug, RAID_EXCLUSIVE_GEAR[0])
        ok, _msg = store.equip("p1", RAID_EXCLUSIVE_GEAR[0])
        self.assertTrue(ok)


class RaidDropTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()
        self.game = self.engine.games["raid"]

    def tearDown(self):
        self.engine.shutdown()

    def _won_raid(self):
        room, _ = self.engine.start("test:raid", "raid", ADA,
                                    kind="group")
        self.engine.join("test:raid", BOB)
        # Ada deals all the damage → MVP; Bob deals none
        room.state["fighters"]["boss"]["hp"] = 1
        room.state["dmg"][ADA.key] = 400
        room.state["dmg"][BOB.key] = 0
        out = self.engine.move("test:raid", "attack", ADA, kind="group")
        self.assertTrue(any("victorious" in m for m in out))
        return room

    def test_victory_loot_only_on_win(self):
        game = RaidGame()
        room = type("R", (), {"state": {"won": False},
                              "humans": []})()
        self.assertEqual(game.victory_loot(room), {})

    def test_victory_loot_rolls_raid_gear_only(self):
        room, _ = self.engine.start("test:raid2", "raid", ADA,
                                    kind="group")
        room.state["won"] = True
        room.state["dmg"] = {ADA.key: 100}
        seen = set()
        for _ in range(200):
            drops = self.game.victory_loot(room)
            for slugs in drops.values():
                seen.update(slugs)
        self.assertTrue(seen)
        self.assertTrue(seen <= set(RAID_EXCLUSIVE_GEAR))

    def test_mvp_gets_better_odds(self):
        room, _ = self.engine.start("test:raid3", "raid", ADA,
                                    kind="group")
        self.engine.join("test:raid3", BOB)
        room.state["won"] = True
        room.state["dmg"] = {ADA.key: 100, BOB.key: 0}  # Ada MVP
        mvp_hits = bystander_hits = 0
        for _ in range(400):
            drops = self.game.victory_loot(room)
            if ADA.key in drops:
                mvp_hits += 1
            if BOB.key in drops:
                bystander_hits += 1
        # MVP ~15%, bystander ~5% — MVP must clearly win
        self.assertGreater(mvp_hits, bystander_hits * 1.5)

    def test_engine_grants_loot_on_raid_win(self):
        room, _ = self.engine.start("test:raid", "raid", ADA,
                                    kind="group")
        self.engine.join("test:raid", BOB)
        room.state["fighters"]["boss"]["hp"] = 1
        room.state["dmg"][ADA.key] = 400
        room.state["dmg"][BOB.key] = 0
        out = self.engine.move("test:raid", "attack", ADA, kind="group")
        self.assertTrue(any("victorious" in m for m in out))
        # the loot roll ran as part of the finish flow
        self.assertIsNone(self.engine.live("test:raid"))

    def test_grant_victory_loot_persists_gear(self):
        room, _ = self.engine.start("test:raid4", "raid", ADA,
                                    kind="group")
        room.state["won"] = True
        game = self.game
        # force a deterministic drop
        orig = game.victory_loot
        game.victory_loot = lambda r: {ADA.key: [RAID_EXCLUSIVE_GEAR[0]]}
        try:
            lines = self.engine._grant_victory_loot(room, game)
        finally:
            game.victory_loot = orig
        self.assertEqual(len(lines), 1)
        self.assertIn("loots", lines[0])
        owned = [i.slug for i in self.engine.gear.list(ADA.key)]
        self.assertIn(RAID_EXCLUSIVE_GEAR[0], owned)

    def test_raid_boss_difficulty_scales(self):
        game = RaidGame()
        rng = random.Random(9)
        easy = game._boss_for(2, rng, difficulty="easy")
        normal = game._boss_for(2, rng, difficulty="normal")
        hard = game._boss_for(2, rng, difficulty="hard")
        expert = game._boss_for(2, rng, difficulty="expert")
        hps = [easy["max_hp"], normal["max_hp"],
               hard["max_hp"], expert["max_hp"]]
        self.assertEqual(hps, sorted(hps))
        self.assertLess(easy["atk"], normal["atk"])
        self.assertLess(normal["atk"], hard["atk"])
        self.assertLess(hard["atk"], expert["atk"])

    def test_raid_setup_stashes_difficulty(self):
        room, _ = self.engine.start("test:raid5", "raid", ADA,
                                    kind="group", difficulty="hard")
        self.assertEqual(room.state.get("difficulty"), "hard")
        self.assertEqual(room.state["fighters"]["boss"]["difficulty"],
                         "hard")


if __name__ == "__main__":
    unittest.main()
