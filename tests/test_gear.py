"""Arena gear persistence: catalog, grades, durability, sets, shop wiring.

Covers nomorals/games/gear.py and its integration points:
- the catalog has every sword kind in every grade, with grade multipliers
  that actually scale stats;
- storm/shadow set detection + bonus values;
- GearStore CRUD: grant/list/equip/unequip/wear/repair are persistent
  across store instances (DB-backed, not in-memory);
- breakage zeroes durability but keeps the row for repair;
- legacy one-shot slugs (gear_sword/gear_armor) migrate to real gear;
- shop purchases forge persistent gear pieces and charge coins;
- the arena applies equipped stats, wears gear per exchange, breaks at 0,
  and fires set combo attacks.
"""
from __future__ import annotations

import unittest

from nomorals.games import engine as engine_mod
from nomorals.games.ai import GameMind
from nomorals.games.engine import GameEngine
from nomorals.games.gear import (
    GEAR_CATALOG,
    GRADES,
    LEGACY_GEAR_MAP,
    SET_BONUSES,
    GearStore,
    detect_set_bonus,
    durability_bar,
    effective_stats,
)
from nomorals.games.players import Player, PlayerStore
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
    def test_every_sword_kind_in_every_grade(self):
        kinds = {"katana", "broadsword", "rapier", "warhammer"}
        for kind in kinds:
            for grade in GRADES:
                slug = f"{kind}_{grade}"
                self.assertIn(slug, GEAR_CATALOG, slug)

    def test_grade_multipliers_scale(self):
        atks = [effective_stats(GEAR_CATALOG[f"katana_{g}"])[0]
                for g in ("common", "rare", "epic", "legendary")]
        self.assertEqual(atks, sorted(atks))
        self.assertGreater(atks[-1], atks[0] * 1.5)

    def test_armor_gives_defense_not_attack(self):
        atk, df = effective_stats(GEAR_CATALOG["plate_legendary"])
        self.assertEqual(atk, 0)
        self.assertGreater(df, 0)

    def test_durability_grows_with_grade(self):
        durs = [GEAR_CATALOG[f"katana_{g}"].max_durability
                for g in ("common", "rare", "epic", "legendary")]
        self.assertEqual(durs, sorted(durs))

    def test_set_pieces_exist(self):
        self.assertIn("storm_katana", GEAR_CATALOG)
        self.assertIn("storm_plate", GEAR_CATALOG)
        self.assertIn("shadow_rapier", GEAR_CATALOG)
        self.assertIn("shadow_mail", GEAR_CATALOG)


class SetBonusTests(unittest.TestCase):
    def test_full_storm_set_detected(self):
        bonus = detect_set_bonus({"weapon": "storm_katana",
                                  "armor": "storm_plate"})
        self.assertIsNotNone(bonus)
        self.assertEqual(bonus.set_name, "storm")
        self.assertGreater(bonus.atk_pct, 0)
        self.assertGreater(bonus.combo_every, 0)

    def test_partial_set_no_bonus(self):
        self.assertIsNone(
            detect_set_bonus({"weapon": "storm_katana"}))

    def test_mismatched_set_no_bonus(self):
        self.assertIsNone(detect_set_bonus(
            {"weapon": "storm_katana", "armor": "plate_common"}))

    def test_empty_no_bonus(self):
        self.assertIsNone(detect_set_bonus({}))


class DurabilityBarTests(unittest.TestCase):
    def test_full_bar(self):
        bar = durability_bar(100, 100)
        self.assertIn("█", bar)

    def test_empty_bar(self):
        bar = durability_bar(0, 100)
        self.assertNotIn("█", bar)


# ── GearStore persistence ────────────────────────────────────────────────────

class GearStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.store = GearStore(self.db)

    def test_grant_persists_across_instances(self):
        inst = self.store.grant("ada", "katana_rare")
        self.assertEqual(inst.durability, inst.max_durability)
        fresh = GearStore(self.db)  # new instance, same DB
        found = fresh.list("ada")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].id, inst.id)

    def test_grant_unknown_slug_fails_fast(self):
        with self.assertRaises(KeyError):
            self.store.grant("ada", "nope_not_real")

    def test_equip_and_unequip(self):
        inst = self.store.grant("ada", "katana_common")
        ok, msg = self.store.equip("ada", "katana_common")
        self.assertTrue(ok, msg)
        self.assertEqual(self.store.equipped("ada")["weapon"].id, inst.id)
        ok, msg = self.store.unequip("ada", "weapon")
        self.assertTrue(ok, msg)
        self.assertEqual(self.store.equipped("ada"), {})

    def test_equip_replaces_same_slot(self):
        self.store.grant("ada", "katana_common")
        self.store.grant("ada", "broadsword_common")
        self.store.equip("ada", "katana_common")
        self.store.equip("ada", "broadsword_common")
        worn = self.store.equipped("ada")
        self.assertEqual(len(worn), 1)
        self.assertEqual(worn["weapon"].slug, "broadsword_common")

    def test_equip_unknown_fails_clean(self):
        ok, msg = self.store.equip("ada", "katana_common")
        self.assertFalse(ok)
        self.assertIn("don't own", msg)

    def test_wear_decreases_and_breaks(self):
        inst = self.store.grant("ada", "rapier_common")
        ok, broke = self.store.wear(inst.id, 5)
        self.assertTrue(ok)
        self.assertFalse(broke)
        got = self.store.list("ada")[0]
        self.assertEqual(got.durability, got.max_durability - 5)
        self.assertFalse(got.broken)
        ok, broke = self.store.wear(inst.id, 10 ** 6)
        self.assertTrue(ok)
        self.assertTrue(broke)
        got = self.store.list("ada")[0]
        self.assertEqual(got.durability, 0)
        self.assertTrue(got.broken)

    def test_broken_cannot_equip(self):
        inst = self.store.grant("ada", "rapier_common")
        self.store.wear(inst.id, 10 ** 6)
        ok, msg = self.store.equip("ada", "rapier_common")
        self.assertFalse(ok)
        self.assertIn("broken", msg)

    def test_repair_restores(self):
        inst = self.store.grant("ada", "rapier_common")
        self.store.wear(inst.id, 10)
        ok, msg = self.store.repair("ada", "rapier_common")
        self.assertTrue(ok, msg)
        name, cost_s = msg.rsplit("|", 1)
        self.assertGreater(int(cost_s), 0)
        self.assertTrue(self.store.apply_repair(inst.id))
        fixed = self.store.list("ada")[0]
        self.assertEqual(fixed.durability, fixed.max_durability)
        self.assertFalse(fixed.broken)

    def test_repair_full_is_noop(self):
        self.store.grant("ada", "rapier_common")
        ok, msg = self.store.repair("ada", "rapier_common")
        self.assertFalse(ok)
        self.assertIn("full", msg)

    def test_display_name_and_durability_bar(self):
        inst = self.store.grant("ada", "katana_epic")
        self.assertIn("Katana", inst.display_name())
        bar = durability_bar(inst.durability, inst.max_durability)
        self.assertIn("█", bar)
        self.assertIn(str(inst.max_durability), bar)

    def test_legacy_migration(self):
        # old count-based items become real durable pieces on first list()
        pstore = PlayerStore(self.db)
        prof = pstore.get("ada")
        prof.items["gear_sword"] = 2
        pstore._upsert(prof)
        pieces = self.store.list("ada")
        slugs = [p.slug for p in pieces]
        self.assertEqual(slugs.count(LEGACY_GEAR_MAP["gear_sword"]), 2)
        # and the legacy counts are gone
        self.assertNotIn("gear_sword", pstore.get("ada").items)


# ── shop wiring ──────────────────────────────────────────────────────────────

class ShopGearTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()
        self.engine.store.add_coins(ADA, 5000, "test-grant")

    def tearDown(self):
        self.engine.shutdown()

    def test_buy_forges_persistent_gear(self):
        ok, msg = self.engine.economy.purchase(ADA, "katana_rare")
        self.assertTrue(ok, msg)
        pieces = self.engine.gear.list(ADA.key)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].slug, "katana_rare")
        # and it survives a fresh store instance (DB-backed)
        self.assertEqual(len(GearStore(self.db).list(ADA.key)), 1)

    def test_buy_charges_coins(self):
        before = self.engine.store.get(ADA.key).coins
        cost = GEAR_CATALOG["katana_rare"].cost
        self.engine.economy.purchase(ADA, "katana_rare")
        after = self.engine.store.get(ADA.key).coins
        self.assertEqual(before - after, cost)

    def test_buy_broke_fails_clean(self):
        poor = Player.from_sender("telegram", "999", "Poor")
        ok, msg = self.engine.economy.purchase(poor, "katana_legendary")
        self.assertFalse(ok)
        self.assertIn("costs", msg)

    def test_legacy_slug_buys_real_gear(self):
        ok, msg = self.engine.economy.purchase(ADA, "gear_sword")
        self.assertTrue(ok, msg)
        pieces = self.engine.gear.list(ADA.key)
        self.assertEqual(pieces[0].slug, LEGACY_GEAR_MAP["gear_sword"])

    def test_catalog_lists_gear_section(self):
        text = self.engine.economy.catalog_text("arena", ADA)
        self.assertIn("katana_common", text)
        self.assertIn("arena gear", text.lower())


# ── arena integration ────────────────────────────────────────────────────────

class ArenaGearTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()
        self.engine.store.add_coins(ADA, 20000, "test-grant")
        # full storm set, equipped
        self.engine.economy.purchase(ADA, "storm_katana")
        self.engine.economy.purchase(ADA, "storm_plate")
        self.engine.gear.equip(ADA.key, "storm_katana")
        self.engine.gear.equip(ADA.key, "storm_plate")

    def tearDown(self):
        self.engine.shutdown()

    def _start_arena(self):
        room, msgs = self.engine.start("t:arena1", "arena", ADA)
        return room

    def test_equipped_stats_apply_at_setup(self):
        room = self._start_arena()
        try:
            y = room.state["you"]
            # storm katana epic: 14 * 1.5 = 21, +25% set = ~26; base 10
            self.assertGreater(y["atk"], 10)
            self.assertGreater(y["def"], 5)
            self.assertEqual(room.state["set_bonus"], "storm")
            self.assertGreater(y["combo_every"], 0)
        finally:
            self.engine.quit("t:arena1")

    def test_no_gear_base_stats(self):
        engine2, _, _ = make_engine()
        try:
            room, _ = engine2.start("t:arena2", "arena", ADA)
            try:
                self.assertEqual(room.state["you"]["atk"], 10)
                self.assertEqual(room.state["you"]["def"], 5)
                self.assertIsNone(room.state["set_bonus"])
            finally:
                engine2.quit("t:arena2")
        finally:
            engine2.shutdown()

    def test_attack_wears_weapon(self):
        room = self._start_arena()
        try:
            game = self.engine.games["arena"]
            mind = GameMind(seed=1)
            wear_before = dict(room.state.get("gear_wear", {})
                               .get(ADA.key, {}))
            game.on_move(room, ADA, "attack", mind)
            wear_after = room.state.get("gear_wear", {}).get(ADA.key, {})
            self.assertGreater(sum(wear_after.values()),
                               sum(wear_before.values()))
        finally:
            self.engine.quit("t:arena1")

    def test_wear_reconciled_on_close(self):
        room = self._start_arena()
        game = self.engine.games["arena"]
        mind = GameMind(seed=1)
        try:
            game.on_move(room, ADA, "attack", mind)
        finally:
            self.engine.quit("t:arena1")
        # close() runs _reconcile_items — durability must have dropped
        pieces = {i.slug: i for i in self.engine.gear.list(ADA.key)}
        katana = pieces["storm_katana"]
        self.assertLess(katana.durability, katana.max_durability)

    def test_break_shatters_mid_battle(self):
        # wear the katana down to 1 durability, then attack
        katana = self.engine.gear.find(ADA.key, "storm_katana")
        self.engine.gear.wear(katana.id, katana.max_durability - 1)
        room = self._start_arena()
        game = self.engine.games["arena"]
        mind = GameMind(seed=7)
        try:
            msgs = game.on_move(room, ADA, "attack", mind)
            text = "\n".join(msgs)
            self.assertIn("SHATTERS", text)
            # stats reverted after the break
            self.assertEqual(room.state["you"]["atk"], 10)
        finally:
            self.engine.quit("t:arena1")

    def test_combo_fires_on_set(self):
        room = self._start_arena()
        game = self.engine.games["arena"]
        try:
            # force the combo counter to the trigger point
            room.state["you"]["combo_count"] = 2  # every=3 → next fires
            room.state["house"]["hp"] = 50
            mind = GameMind(seed=3)
            msgs = game.on_move(room, ADA, "attack", mind)
            text = "\n".join(msgs)
            self.assertIn("twin lightning", text)
        finally:
            self.engine.quit("t:arena1")

    def test_mid_battle_equip(self):
        # buy a second weapon mid-fight via the engine shop path
        room = self._start_arena()
        game = self.engine.games["arena"]
        mind = GameMind(seed=1)
        try:
            self.engine.economy.purchase(ADA, "katana_legendary")
            # refresh the mirror the way the engine does on shop buy
            room.state["gear_closet"][ADA.key] = [
                {"id": i.id, "slug": i.slug, "name": "x",
                 "slot": "weapon", "atk": 1, "def": 0,
                 "durability": i.durability,
                 "max_durability": i.max_durability,
                 "set": "", "equipped": False}
                for i in self.engine.gear.list(ADA.key)
                if i.slug == "katana_legendary"
            ] + room.state["gear_closet"][ADA.key]
            msg = game._equip(room, ADA, "katana_legendary")
            self.assertIsNotNone(msg)
            self.assertIn("equipped", msg)
            # set bonus lost (legendary katana isn't storm) — atk still up
            self.assertIsNone(room.state["set_bonus"])
        finally:
            self.engine.quit("t:arena1")


if __name__ == "__main__":
    unittest.main()
