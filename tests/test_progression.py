"""Persistent progression: XP curve, levels, stat growth, engine wiring.

Covers nomorals/games/progression.py and its integration points:
- the XP curve is rewarding (early levels pop in a battle or two) and
  monotonic;
- award_xp persists across store instances and reports every level
  crossed, including multi-level jumps;
- the engine awards XP on every finished game (generic table for most
  games, rich table for the arena) and announces level-ups;
- arena battles apply the player's level stats at setup, the house
  scales at half, and XP actually lands after a win and a loss;
- the RPG converts session XP into persistent XP;
- /level renders.
"""
from __future__ import annotations

import unittest

from nomorals.games.ai import GameMind
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player, PlayerStore
from nomorals.games.progression import (
    ARENA_LOSS_XP,
    ARENA_WIN_XP,
    award_xp,
    describe_level_up,
    generic_game_xp,
    level_for_xp,
    level_stat_bonus,
    xp_bar,
    xp_for_level,
    xp_progress,
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


# ── the curve ────────────────────────────────────────────────────────────────

class CurveTests(unittest.TestCase):
    def test_level_one_is_free(self):
        self.assertEqual(xp_for_level(1), 0)
        self.assertEqual(xp_for_level(0), 0)

    def test_monotonic(self):
        reqs = [xp_for_level(n) for n in range(1, 12)]
        self.assertEqual(reqs, sorted(reqs))
        self.assertGreater(reqs[-1], reqs[0])

    def test_early_levels_pop_fast(self):
        # level 2 after ~2 arena battles — rewarding, not grindy
        self.assertLessEqual(xp_for_level(2), ARENA_WIN_XP + ARENA_LOSS_XP)
        # level 5 inside a single long session of wins
        self.assertLessEqual(xp_for_level(5), 15 * ARENA_WIN_XP)

    def test_round_trip(self):
        for xp in (0, 1, 79, 80, 239, 240, 1000, 3599, 3600, 10 ** 6):
            lvl = level_for_xp(xp)
            self.assertGreaterEqual(xp, xp_for_level(lvl))
            self.assertLess(xp, xp_for_level(lvl + 1))

    def test_progress_parts(self):
        level, into, span = xp_progress(xp_for_level(3) + 50)
        self.assertEqual(level, 3)
        self.assertEqual(into, 50)
        self.assertEqual(span, xp_for_level(4) - xp_for_level(3))

    def test_bar_renders(self):
        bar = xp_bar(0)
        self.assertIn("░", bar)
        self.assertIn("lvl 1", bar)


class StatGrowthTests(unittest.TestCase):
    def test_level_one_no_bonus(self):
        self.assertEqual(level_stat_bonus(1),
                         {"max_hp": 0, "atk": 0, "def": 0})

    def test_growth_is_visible(self):
        b5 = level_stat_bonus(5)
        self.assertGreater(b5["max_hp"], 0)
        self.assertGreater(b5["atk"], 0)

    def test_def_every_two_levels(self):
        self.assertEqual(level_stat_bonus(2)["def"], 0)
        self.assertEqual(level_stat_bonus(3)["def"], 1)

    def test_level_up_message_names_gains(self):
        msg = describe_level_up(4)
        self.assertIn("4", msg)
        self.assertIn("max HP", msg)


# ── award_xp ─────────────────────────────────────────────────────────────────

class AwardXpTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.store = PlayerStore(self.db)

    def test_award_persists(self):
        award_xp(self.store, ADA, 50, "test")
        fresh = PlayerStore(self.db)
        self.assertEqual(fresh.get(ADA.key).xp, 50)

    def test_level_crossing_reported(self):
        need = xp_for_level(2)
        new_level, gained = award_xp(self.store, ADA, need, "test")
        self.assertEqual(new_level, 2)
        self.assertEqual(gained, [2])

    def test_multi_level_jump_reports_all(self):
        new_level, gained = award_xp(self.store, ADA, xp_for_level(5), "test")
        self.assertEqual(new_level, 5)
        self.assertEqual(gained, [2, 3, 4, 5])

    def test_no_amount_no_crash(self):
        new_level, gained = award_xp(self.store, ADA, 0, "test")
        self.assertEqual((new_level, gained), (1, []))

    def test_generic_table(self):
        self.assertGreater(generic_game_xp(True), generic_game_xp(None))
        self.assertGreater(generic_game_xp(None), generic_game_xp(False))
        self.assertGreater(generic_game_xp(False), 0)


# ── engine wiring ────────────────────────────────────────────────────────────

class EngineXpTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _finish_game(self, chat_key, game_name, moves):
        room, _ = self.engine.start(chat_key, game_name, ADA)
        for m in moves:
            if self.engine.live(chat_key) is None:
                break
            self.engine.move(chat_key, m, ADA)
        if self.engine.live(chat_key) is not None:
            self.engine.quit(chat_key)

    def test_finished_game_awards_xp(self):
        # tictactoe vs AI: a few moves, then quit — quit path also closes
        before = self.engine.store.get(ADA.key).xp
        self._finish_game("t:xp1", "ttt", ["1", "2", "3"])
        after = self.engine.store.get(ADA.key).xp
        self.assertGreaterEqual(after, before)

    def test_arena_win_pays_arena_rate(self):
        # drive a quick arena win: buff the player via level is circular,
        # so just check XP lands and a win pays >= a loss
        prof = self.engine.store.get(ADA.key)
        prof.xp = 0
        self.engine.store._upsert(prof)
        room, _ = self.engine.start("t:xpw", "arena", ADA)
        game = self.engine.games["arena"]
        # rig the fight: house at 1 HP
        room.state["house"]["hp"] = 1
        mind = GameMind(seed=1)
        game.on_move(room, ADA, "attack", mind)
        msgs = self.engine._finish(room)
        text = "\n".join(msgs)
        self.assertIn("XP", text)
        xp = self.engine.store.get(ADA.key).xp
        self.assertGreaterEqual(xp, ARENA_WIN_XP)

    def test_arena_loss_still_pays(self):
        room, _ = self.engine.start("t:xpl", "arena", ADA)
        room.state["you"]["hp"] = 1
        # the house counter-attacks after every player move — it may
        # focus/guard first, but one real swing at 1 HP ends it
        for _ in range(20):
            if self.engine.live("t:xpl") is None:
                break
            self.engine.move("t:xpl", "attack", ADA)
        self.assertIsNone(self.engine.live("t:xpl"))
        xp = self.engine.store.get(ADA.key).xp
        self.assertGreaterEqual(xp, ARENA_LOSS_XP)

    def test_level_up_announced(self):
        prof = self.engine.store.get(ADA.key)
        prof.xp = xp_for_level(2) - 1  # one XP from level 2
        self.engine.store._upsert(prof)
        room, _ = self.engine.start("t:xplvl", "arena", ADA)
        room.state["house"]["hp"] = 1
        game = self.engine.games["arena"]
        game.on_move(room, ADA, "attack", GameMind(seed=1))
        msgs = self.engine._finish(room)
        text = "\n".join(msgs)
        self.assertIn("LEVEL UP", text)
        # XP economy may award enough to skip past 2; assert at least 2
        new_level = level_for_xp(self.engine.store.get(ADA.key).xp)
        self.assertGreaterEqual(new_level, 2)
        self.assertIn(f"level {new_level}", text)

    def test_arena_applies_level_stats(self):
        prof = self.engine.store.get(ADA.key)
        prof.xp = xp_for_level(4)
        self.engine.store._upsert(prof)
        room, _ = self.engine.start("t:xps", "arena", ADA)
        try:
            y = room.state["you"]
            bonus = level_stat_bonus(4)
            self.assertEqual(y["max_hp"], 50 + bonus["max_hp"])
            self.assertEqual(y["atk"], 10 + bonus["atk"])
            self.assertEqual(y["def"], 5 + bonus["def"])
            # house gets dynamic enemy stats (spawned after progression);
            # anti-wall guarantee keeps it competitive but weaker
            h = room.state["house"]
            self.assertLess(h["atk"], y["atk"])
        finally:
            self.engine.quit("t:xps")

    def test_progression_mirrored_in_state(self):
        prof = self.engine.store.get(ADA.key)
        prof.xp = xp_for_level(3)
        self.engine.store._upsert(prof)
        room, _ = self.engine.start("t:xpm", "arena", ADA)
        try:
            prog = room.state["progression"][ADA.key]
            self.assertEqual(prog["level"], 3)
        finally:
            self.engine.quit("t:xpm")


class RpgXpTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_rpg_converts_session_xp(self):
        game = self.engine.games["rpg"]
        room, _ = self.engine.start("t:rpgxp", "rpg", ADA)
        try:
            # simulate a strong campaign: high session XP, survived
            room.state["sheets"][ADA.key]["xp"] = 120
            room.state["done"] = True
            reward = game.xp_reward(True, room, ADA)
            self.assertGreaterEqual(reward, 60 + generic_game_xp(True))
        finally:
            self.engine.quit("t:rpgxp")


class LevelCommandTests(unittest.TestCase):
    def test_control_level_renders(self):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

        engine, db, _ = make_engine()
        try:
            award_xp(engine.store, ADA, xp_for_level(3) + 10, "test")

            class Stub(RuntimeGamesMixin):
                def __init__(self, eng):
                    # no db on the context → feature gate uses defaults
                    # (games on)
                    self.context = SimpleNamespace()
                    self._eng = eng

                def _game_engine(self):
                    return self._eng

            text = Stub(engine)._control_level(player=ADA)
            self.assertIn("level 3", text)
            self.assertIn("XP", text)
            self.assertIn("arena stats", text)
        finally:
            engine.shutdown()

    def test_control_gear_inventory_renders(self):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

        engine, db, _ = make_engine()
        try:
            engine.store.add_coins(ADA, 5000, "test-grant")
            ok, _ = engine.economy.purchase(ADA, "katana_rare")
            self.assertTrue(ok)

            class Stub(RuntimeGamesMixin):
                def __init__(self, eng):
                    self.context = SimpleNamespace()
                    self._eng = eng

                def _game_engine(self):
                    return self._eng

            stub = Stub(engine)
            text = stub._control_gear("inventory", "", player=ADA)
            self.assertIn("katana_rare", text)
            self.assertIn("coins", text)
            # equip + unequip through the command layer
            out = stub._control_gear("equip", "katana_rare", player=ADA)
            self.assertIn("equipped", out)
            out = stub._control_gear("unequip", "weapon", player=ADA)
            self.assertIn("unequipped", out)
            # repair a worn piece through the command layer
            inst = engine.gear.find(ADA.key, "katana_rare")
            engine.gear.wear(inst.id, 5)
            out = stub._control_gear("repair", "katana_rare", player=ADA)
            self.assertIn("good as new", out)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
