"""The Cutyp legacy set: myth-tier unbreakable gear + Slaying Force skill.

Covers:
- cutyp_steel_katana / cutyp_robe exist in the catalog: myth grade,
  unbreakable, priced endgame, strongest stats in the game;
- GearStore.wear never reduces durability on unbreakable pieces;
- detect_set_bonus finds the cutyp set (+50%/+50%, combo every 2);
- Slaying Force is in the skill catalog with signature-grade knobs;
- the shop lists the legacy pieces with an ∞ unbreakable marker.
"""
from __future__ import annotations

import unittest

from nomorals.games.economy import GameEconomy
from nomorals.games.gear import (
    GEAR_CATALOG,
    SET_BONUSES,
    GearStore,
    detect_set_bonus,
    durability_display,
    effective_stats,
)
from nomorals.games.players import Player, PlayerStore
from nomorals.games.skills import SKILL_CATALOG, SkillStore, resolve_skill
from nomorals.storage.db import Database


ADA = Player.from_sender("telegram", "456", "Ada")


def fresh_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class LegacyGearCatalogTests(unittest.TestCase):
    def test_katana_and_robe_exist(self):
        for slug in ("cutyp_steel_katana", "cutyp_robe"):
            self.assertIn(slug, GEAR_CATALOG, slug)

    def test_myth_grade_unbreakable(self):
        for slug in ("cutyp_steel_katana", "cutyp_robe"):
            defn = GEAR_CATALOG[slug]
            self.assertEqual(defn.grade, "myth")
            self.assertTrue(defn.unbreakable)
            self.assertEqual(defn.set_name, "cutyp")

    def test_pricey_endgame_pricing(self):
        # endgame pricing: each piece costs more than any legendary
        legendary_costs = [d.cost for d in GEAR_CATALOG.values()
                           if d.grade == "legendary"]
        katana = GEAR_CATALOG["cutyp_steel_katana"]
        robe = GEAR_CATALOG["cutyp_robe"]
        self.assertGreater(katana.cost, max(legendary_costs))
        self.assertGreater(robe.cost, max(legendary_costs))

    def test_strongest_stats_in_game(self):
        katana_atk, _ = effective_stats(GEAR_CATALOG["cutyp_steel_katana"])
        _, robe_def = effective_stats(GEAR_CATALOG["cutyp_robe"])
        best_weapon_atk = max(
            effective_stats(d)[0]
            for d in GEAR_CATALOG.values()
            if d.slot == "weapon" and d.slug != "cutyp_steel_katana")
        best_armor_def = max(
            effective_stats(d)[1]
            for d in GEAR_CATALOG.values()
            if d.slot == "armor" and d.slug != "cutyp_robe")
        self.assertGreater(katana_atk, best_weapon_atk)
        self.assertGreater(robe_def, best_armor_def)

    def test_durability_display_shows_infinity(self):
        defn = GEAR_CATALOG["cutyp_steel_katana"]
        self.assertIn("unbreakable", durability_display(defn, 999, 999))
        plain = GEAR_CATALOG["katana_legendary"]
        self.assertNotIn("unbreakable",
                         durability_display(plain, 100, 100))


class CutypSetBonusTests(unittest.TestCase):
    def test_set_bonus_values(self):
        bonus = SET_BONUSES["cutyp"]
        self.assertEqual(bonus.needs, ("weapon", "armor"))
        self.assertEqual(bonus.atk_pct, 0.50)
        self.assertEqual(bonus.def_pct, 0.50)
        self.assertEqual(bonus.combo_name, "cutyp's fury")
        self.assertEqual(bonus.combo_every, 2)

    def test_detect_set_bonus(self):
        bonus = detect_set_bonus({"weapon": "cutyp_steel_katana",
                                  "armor": "cutyp_robe"})
        self.assertIsNotNone(bonus)
        self.assertEqual(bonus.set_name, "cutyp")

    def test_partial_set_not_detected(self):
        self.assertIsNone(detect_set_bonus({"weapon": "cutyp_steel_katana"}))


class UnbreakableWearTests(unittest.TestCase):
    def test_wear_is_noop_on_unbreakable(self):
        db = fresh_db()
        store = GearStore(db)
        inst = store.grant("ada", "cutyp_steel_katana")
        before = inst.durability
        for _ in range(50):
            ok, broke = store.wear(inst.id, 1)
            self.assertTrue(ok)
            self.assertFalse(broke)
        fresh = store.get(inst.id)
        self.assertIsNotNone(fresh)
        assert fresh is not None
        self.assertEqual(fresh.durability, before)
        self.assertFalse(fresh.broken)

    def test_ordinary_gear_still_wears(self):
        db = fresh_db()
        store = GearStore(db)
        inst = store.grant("ada", "katana_common")
        before = inst.durability
        ok, broke = store.wear(inst.id, 3)
        self.assertTrue(ok)
        fresh = store.get(inst.id)
        assert fresh is not None
        self.assertEqual(fresh.durability, before - 3)

    def test_unbreakable_stays_equippable_forever(self):
        db = fresh_db()
        store = GearStore(db)
        ok, msg = store.equip("ada", "cutyp_steel_katana")
        self.assertFalse(ok)  # not owned yet
        store.grant("ada", "cutyp_steel_katana")
        ok, msg = store.equip("ada", "cutyp_steel_katana")
        self.assertTrue(ok, msg)
        worn = store.equipped("ada")
        self.assertIn("weapon", worn)
        # 200 hits of wear: still equipped, still full
        for _ in range(200):
            store.wear(worn["weapon"].id, 1)
        worn = store.equipped("ada")
        self.assertIn("weapon", worn)
        self.assertEqual(worn["weapon"].slug, "cutyp_steel_katana")


class SlayingForceTests(unittest.TestCase):
    def test_in_catalog(self):
        self.assertIn("slaying_force", SKILL_CATALOG)

    def test_signature_knobs(self):
        defn = SKILL_CATALOG["slaying_force"]
        self.assertEqual(defn.name, "Slaying Force")
        self.assertEqual(defn.school, "cutyp")
        self.assertEqual(defn.kind, "active")
        self.assertEqual(defn.mult, 3.5)
        self.assertEqual(defn.ignore_def_pct, 1.0)
        self.assertEqual(defn.cost, 5000)
        self.assertGreaterEqual(defn.level_req, 10)

    def test_resolves_by_name(self):
        defn = resolve_skill("slaying force")
        self.assertIsNotNone(defn)
        assert defn is not None
        self.assertEqual(defn.slug, "slaying_force")

    def test_learn_persists(self):
        db = fresh_db()
        skills = SkillStore(db)
        self.assertTrue(skills.learn("ada", "slaying_force"))
        self.assertIn("slaying_force", skills.learned("ada"))
        # idempotent — second learn is a no-op
        self.assertFalse(skills.learn("ada", "slaying_force"))


class LegacyShopTests(unittest.TestCase):
    def test_shop_lists_legacy_with_unbreakable_marker(self):
        db = fresh_db()
        store = PlayerStore(db)
        gear = GearStore(db)
        econ = GameEconomy(store, gear_store=gear)
        text = econ.gear_catalog_text()
        self.assertIn("cutyp_steel_katana", text)
        self.assertIn("cutyp_robe", text)
        self.assertIn("unbreakable", text)

    def test_purchase_marks_unbreakable(self):
        db = fresh_db()
        store = PlayerStore(db)
        gear = GearStore(db)
        econ = GameEconomy(store, gear_store=gear)
        store.add_coins(ADA, 20000, "test grant")
        ok, msg = econ.purchase(ADA, "cutyp_steel_katana")
        self.assertTrue(ok, msg)
        self.assertIn("unbreakable", msg)
        insts = gear.list(ADA.key)
        self.assertEqual(len(insts), 1)
        self.assertEqual(insts[0].slug, "cutyp_steel_katana")


if __name__ == "__main__":
    unittest.main()


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


class CutypArenaTests(unittest.TestCase):
    """The full legacy set fires its combo in a real arena; Slaying
    Force casts through the generic striking path."""

    def test_legacy_set_combo_in_arena(self):
        from nomorals.games.ai import GameMind
        from nomorals.games.engine import GameEngine

        class Ctx:
            def __init__(self, db):
                self.db = db

        db = fresh_db()
        engine = GameEngine(Ctx(db), send=lambda c, t: None)
        try:
            engine.store.add_coins(ADA, 30000, "test-grant")
            self.assertTrue(engine.economy.purchase(
                ADA, "cutyp_steel_katana")[0])
            self.assertTrue(engine.economy.purchase(ADA, "cutyp_robe")[0])
            ok, _ = engine.gear.equip(ADA.key, "cutyp_steel_katana")
            self.assertTrue(ok)
            ok, _ = engine.gear.equip(ADA.key, "cutyp_robe")
            self.assertTrue(ok)

            room, _ = engine.start("t:cutyp1", "arena", ADA)
            try:
                y = room.state["you"]
                self.assertEqual(room.state["set_bonus"], "cutyp")
                self.assertEqual(y["combo_every"], 2)
                self.assertEqual(y["combo_name"], "cutyp's fury")
                # 105 atk +50% set → clearly above the base 10
                self.assertGreater(y["atk"], 100)

                # force the combo counter to the trigger point
                game = engine.games["arena"]
                room.state["you"]["combo_count"] = 1  # every=2 → fires
                room.state["house"]["hp"] = 500
                msgs = game.on_move(room, ADA, "attack", GameMind(seed=3))
                text = "\n".join(msgs)
                self.assertIn("cutyp's fury", text)

                # no wear recorded on the unbreakable pieces
                wear = room.state.get("gear_wear", {}).get(ADA.key, {})
                self.assertEqual(wear, {})
            finally:
                engine.quit("t:cutyp1")

            # DB durability untouched after the fight
            for inst in engine.gear.list(ADA.key):
                if inst.slug in ("cutyp_steel_katana", "cutyp_robe"):
                    self.assertEqual(inst.durability, inst.max_durability)
        finally:
            engine.shutdown()

    def test_slaying_force_cast(self):
        from nomorals.games.ai import GameMind
        from nomorals.games.games import ambitious

        game = ambitious.BattleArenaGame()
        mind = GameMind(seed=7)
        you = _fighter(atk=20)
        house = _fighter(hp=500)  # huge defense to prove the ignore
        house["atk"], house["def"] = 8, 50
        state = {
            "you": you, "house": house,
            "skills": {"telegram:456": ["slaying_force"]},
            "house_rank": "E", "skill_cd": {}, "skill_used": [],
        }
        room = _Room(state, [ADA])
        msg = game._cast_skill(room, ADA, "slaying force", mind)
        self.assertIn("Slaying Force", msg)
        # 3.5× strike ignoring 50 def — must have dealt real damage
        self.assertLess(house["hp"], 500 - 30)
        self.assertEqual(state["skill_cd"]["slaying_force"], 6)
