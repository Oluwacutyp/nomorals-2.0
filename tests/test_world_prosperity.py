"""World/town prosperity overhaul: quitting a never-ending town settles it
as a completed run instead of a flat draw.

Covers:
  * the reported bug — a 360-day thriving town quitting paid 25 coins;
    it must now pay a score-based prosperity settlement (hundreds)
  * the engine hooks (finish_won / coin_payout) and that other games
    still use the standard winner()-based flow
  * score scaling with effort (days, population, buildings, stockpiles,
    rank, tech, quests, upgrades)
  * the deeper economy: upgrades, technologies, trade routes, happiness
    driving productivity
  * dilemmas with choices, disasters with mitigation, golden ages
  * rank progression hamlet → legend and old-save migration
"""
from __future__ import annotations

import re
import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.games.ambitious import (
    CHOICE_EVENTS,
    TECHS,
    TOWN_RANKS,
    WorldGame,
)
from nomorals.games.players import Player
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db: Database) -> None:
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, sent


ADA = Player.from_sender("telegram", "456", "Ada")

GAME = WorldGame()


def thriving_town() -> dict:
    """A 360-day town like the one in the bug report."""
    s = GAME.new_state(__import__("random").Random(1))
    s.update({
        "day": 360, "pop": 65, "food": 500, "gold": 2000, "tools": 10,
        "buildings": {"house": 28, "farm": 8, "shed": 4, "market": 4,
                      "workshop": 3, "wall": 2, "temple": 1, "granary": 2,
                      "tradepost": 2},
        "happiness": 80, "rank": 3,
        "upgrades": {"farm": 2, "market": 1},
        "tech": ["irrigation", "deep_mining", "guilds", "medicine"],
        "quests_done": 8,
    })
    return s


def coin_payout_from(msgs: list[str]) -> tuple[int, str]:
    for m in msgs:
        mm = re.search(r"\+(\d+) coins \((.*)\)", m)
        if mm:
            return int(mm.group(1)), mm.group(2)
    raise AssertionError(f"no coin payout in: {msgs!r}")


class WorldSettlementTests(unittest.TestCase):
    """The bug: 360 days → 25 coins. Now prosperity pays."""

    def setUp(self):
        self.engine, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_quit_360_day_town_pays_prosperity(self):
        self.engine.start("telegram:1", "world", ADA)
        room = self.engine.live("telegram:1")
        room.state.update(thriving_town())
        msgs = self.engine.quit("telegram:1")
        coins, why = coin_payout_from(msgs)
        score = GAME.score(room, ADA)
        self.assertGreaterEqual(score, 500,
                                f"360-day town should score in the hundreds+, got {score}")
        self.assertGreater(coins, 100,
                           f"360 days of effort must pay meaningfully, got {coins} ({why})")
        self.assertIn("prosperity", why)
        # the ledger actually received them
        prof = self.engine.store.get(ADA.key)
        self.assertGreaterEqual(prof.coins, coins)
        # a settlement counts as a completed run, not a draw
        self.assertEqual(prof.wins, 1)
        self.assertEqual(prof.draws, 0)

    def test_quit_new_town_pays_small_settlement(self):
        self.engine.start("telegram:2", "world", ADA)
        msgs = self.engine.quit("telegram:2")
        coins, why = coin_payout_from(msgs)
        # day-1 town: base 40 + tiny prosperity share — a completed run,
        # but there's nothing to settle yet
        self.assertGreaterEqual(coins, 40)
        self.assertLess(coins, 100)
        self.assertIn("town settled", why)

    def test_other_games_still_draw_on_quit(self):
        # the hooks must not change the standard flow: quitting rps
        # mid-game is still a draw paying the flat 25
        self.engine.start("telegram:3", "rps", ADA)
        msgs = self.engine.quit("telegram:3")
        coins, why = coin_payout_from(msgs)
        self.assertEqual((coins, why), (25, "draw"))

    def test_finish_won_default_is_winner_sentinel(self):
        from nomorals.games.games.base import MultiGame
        self.assertEqual(MultiGame().finish_won(None, None), "winner")
        self.assertIsNone(MultiGame().coin_payout(None, None, None, 0,
                                                  "normal", 0))

    def test_world_final_message_summarizes_town(self):
        self.engine.start("telegram:4", "world", ADA)
        room = self.engine.live("telegram:4")
        room.state.update(thriving_town())
        msgs = self.engine.quit("telegram:4")
        self.assertTrue(any("360 days" in m and "prosperity score" in m
                            for m in msgs),
                        f"expected settlement summary, got: {msgs!r}")


class WorldScoreTests(unittest.TestCase):
    def test_score_scales_with_effort(self):
        import random
        new = GAME.new_state(random.Random(1))
        mid = GAME.new_state(random.Random(1))
        mid.update({"day": 60, "pop": 25, "gold": 300, "food": 120,
                    "buildings": {"house": 8, "farm": 3, "market": 1},
                    "rank": 1, "tech": ["irrigation"], "quests_done": 2,
                    "upgrades": {"farm": 1}})
        old = thriving_town()
        s_new = GAME.score(type("R", (), {"state": new})(), ADA)
        s_mid = GAME.score(type("R", (), {"state": mid})(), ADA)
        s_old = GAME.score(type("R", (), {"state": old})(), ADA)
        self.assertLess(s_new, s_mid)
        self.assertLess(s_mid, s_old)
        self.assertGreaterEqual(s_old, 500)
        self.assertLess(s_new, 100)

    def test_score_counts_every_prosperity_axis(self):
        import random
        base = GAME.new_state(random.Random(1))
        rich = dict(base)
        rich.update({"day": 100, "pop": 40, "gold": 1000, "food": 400,
                     "buildings": {"house": 15, "farm": 5},
                     "rank": 2, "tech": ["irrigation", "guilds"],
                     "quests_done": 5, "upgrades": {"farm": 3}})
        room_b = type("R", (), {"state": base})()
        room_r = type("R", (), {"state": rich})()
        self.assertGreater(GAME.score(room_r, ADA), GAME.score(room_b, ADA))


class WorldEconomyTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.sent = make_engine()
        self.engine.start("telegram:1", "world", ADA)
        self.room = self.engine.live("telegram:1")

    def tearDown(self):
        self.engine.shutdown()

    def test_upgrade_improves_yield(self):
        s = self.room.state
        s["gold"], s["tools"], s["food"] = 500, 10, 500
        self.engine.move("telegram:1", "build farm", ADA)
        before = GAME._food_per_day(self.room)
        self.engine.move("telegram:1", "upgrade farm", ADA)
        after = GAME._food_per_day(self.room)
        self.assertEqual(s["upgrades"]["farm"], 1)
        self.assertGreater(after, before)

    def test_upgrade_cap_and_engineering(self):
        s = self.room.state
        s["gold"], s["tools"], s["food"] = 5000, 50, 500
        self.engine.move("telegram:1", "build farm", ADA)
        for _ in range(3):
            self.engine.move("telegram:1", "upgrade farm", ADA)
        self.assertEqual(s["upgrades"]["farm"], 3)
        out = self.engine.move("telegram:1", "upgrade farm", ADA)
        self.assertTrue(any("max level" in m for m in out))
        # engineering raises the cap to 5
        s["tech"].append("engineering")
        s["rank"] = 4
        self.engine.move("telegram:1", "upgrade farm", ADA)
        self.assertEqual(s["upgrades"]["farm"], 4)

    def test_research_gated_by_rank(self):
        s = self.room.state
        s["gold"] = 500
        out = self.engine.move("telegram:1", "research irrigation", ADA)
        self.assertTrue(any("needs a" in m for m in out),
                        f"rank gate expected, got: {out!r}")
        self.assertNotIn("irrigation", s["tech"])
        s["rank"] = 1
        before = GAME._food_per_day(self.room)
        out = self.engine.move("telegram:1", "research irrigation", ADA)
        self.assertIn("irrigation", s["tech"])
        self.assertTrue(any("researched" in m for m in out))
        # need a farm for the bonus to show
        s["buildings"]["farm"] = 2
        self.assertGreater(GAME._food_per_day(self.room), before)

    def test_trade_routes_sell_surplus(self):
        s = self.room.state
        s["buildings"]["tradepost"] = 2
        s["buildings"]["market"] = 1
        s["food"], s["gold"], s["pop"] = 200, 0, 10
        s["happiness"] = 60
        gold_before = s["gold"]
        food_before = s["food"]
        GAME._trade_routes(self.room, [])
        self.assertGreater(s["gold"], gold_before)
        self.assertLess(s["food"], food_before)
        # reserve is kept: pop*2+5 = 25
        self.assertGreaterEqual(s["food"], 25)

    def test_happiness_drives_productivity(self):
        s = self.room.state
        s["buildings"]["market"] = 3
        s["happiness"] = 0
        low = GAME._gold_per_day(self.room)
        s["happiness"] = 100
        high = GAME._gold_per_day(self.room)
        self.assertGreater(high, low)

    def test_feast_raises_happiness(self):
        s = self.room.state
        s["food"], s["happiness"] = 100, 40
        self.engine.move("telegram:1", "feast", ADA)
        # +8 from the feast; the tick's drift only adds, one bad event
        # can take at most 8 (plague) — net must still be up
        self.assertGreaterEqual(s["happiness"], 41)
        self.assertLess(s["food"], 100)  # the feast cost 10, tick ate more

    def test_golden_age_doubles_income(self):
        s = self.room.state
        s["buildings"]["market"] = 4
        s["happiness"] = 80
        s["pop"] = 35
        s["day"] = 100
        s["food"] = 1000
        events = GAME._tick(self.room)
        self.assertGreater(s["golden_age"], 0)
        self.assertTrue(any("GOLDEN AGE" in e for e in events))
        # the multiplier itself, deterministic (no tick randomness):
        # happiness 50 -> productivity exactly 1.0, no truncation drift
        s["happiness"] = 50
        s["golden_age"] = 0
        base_income = GAME._gold_per_day(self.room)
        s["golden_age"] = 5
        self.assertEqual(GAME._gold_per_day(self.room), base_income * 2)

    def test_wonder_requires_rank(self):
        s = self.room.state
        s["gold"], s["food"], s["tools"] = 5000, 5000, 50
        out = self.engine.move("telegram:1", "build wonder", ADA)
        self.assertTrue(any("needs a" in m for m in out))
        self.assertNotIn("wonder", s["buildings"])
        s["rank"] = 5
        out = self.engine.move("telegram:1", "build wonder", ADA)
        self.assertEqual(s["buildings"].get("wonder"), 1)
        self.assertTrue(any("WONDER" in m for m in out))


class WorldEventTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.sent = make_engine()
        self.engine.start("telegram:1", "world", ADA)
        self.room = self.engine.live("telegram:1")

    def tearDown(self):
        self.engine.shutdown()

    def _offer(self, name: str):
        s = self.room.state
        s["food"], s["gold"], s["pop"] = 500, 500, 20
        s["buildings"]["house"] = 10  # room for refugees
        GAME._offer_choice(self.room, name, [])
        return s["pending_choice"]

    def test_refugees_choice(self):
        pc = self._offer("refugees")
        self.assertIsNotNone(pc)
        s = self.room.state
        pop_before, food_before = s["pop"], s["food"]
        out = self.engine.move("telegram:1", "1", ADA)
        self.assertTrue(any("gates open" in m for m in out))
        # +3 refugees; the tick may add one more family moving in
        self.assertGreaterEqual(s["pop"], pop_before + 3)
        self.assertLess(s["food"], food_before)
        self.assertIsNone(s["pending_choice"])
        self.assertEqual(s["quests_done"], 1)

    def test_choice_unaffordable_is_refused(self):
        self._offer("mercenaries")
        s = self.room.state
        s["gold"] = 5  # can't afford the 25g hire
        out = self.engine.move("telegram:1", "1", ADA)
        self.assertTrue(any("can't" in m for m in out))
        self.assertIsNotNone(s["pending_choice"])  # still waiting

    def test_choice_does_not_block_other_moves(self):
        self._offer("scholar")
        out = self.engine.move("telegram:1", "farm", ADA)
        self.assertTrue(any("fields give" in m for m in out))
        self.assertIsNotNone(self.room.state["pending_choice"])

    def test_mercenaries_block_raid(self):
        s = self.room.state
        s["gold"], s["mercenary_days"] = 500, 20
        events: list[str] = []
        GAME._apply_simple_event(self.room, __import__("random").Random(1),
                                 "raid", "", events)
        self.assertTrue(any("hired swords" in e for e in events))
        self.assertEqual(s["gold"], 500)

    def test_warded_herbs_block_plague(self):
        s = self.room.state
        s["warded_days"] = 20
        pop_before = s["pop"]
        events: list[str] = []
        GAME._apply_simple_event(self.room, __import__("random").Random(1),
                                 "plague", "", events)
        self.assertEqual(s["pop"], pop_before)
        self.assertTrue(any("herbs" in e for e in events))

    def test_earthquake_mitigated_by_engineering(self):
        import random
        s = self.room.state
        s["buildings"]["farm"] = 2
        s["tech"].append("engineering")
        events: list[str] = []
        GAME._apply_simple_event(self.room, random.Random(1),
                                 "earthquake", "", events)
        self.assertEqual(s["buildings"]["farm"], 2)
        self.assertTrue(any("foundations hold" in e for e in events))

    def test_earthquake_destroys_without_mitigation(self):
        import random
        s = self.room.state
        s["buildings"] = {"farm": 1}
        events: list[str] = []
        GAME._apply_simple_event(self.room, random.Random(1),
                                 "earthquake", "", events)
        self.assertNotIn("farm", s["buildings"])
        self.assertTrue(any("collapses" in e for e in events))

    def test_dragon_fight_victory_with_wall(self):
        s = self.room.state
        s["buildings"]["wall"] = 2
        s["gold"] = 100
        self._offer("dragon")
        out = self.engine.move("telegram:1", "2", ADA)
        self.assertTrue(any("dragon falls" in m for m in out))
        # +80 hoard; the tick's own events may add a little more
        self.assertGreaterEqual(s["gold"], 100 + 80)
        self.assertEqual(s["quests_done"], 1)

    def test_dragon_tribute(self):
        s = self.room.state
        s["gold"] = 100
        self._offer("dragon")
        out = self.engine.move("telegram:1", "1", ADA)
        self.assertTrue(any("tribute" in m or "wheels away" in m
                            for m in out))
        # −40 tribute; the tick's own events may add a little back
        self.assertGreaterEqual(s["gold"], 60)

    def test_shrine_quest_tithe_completes(self):
        s = self.room.state
        s["food"] = 500
        self._offer("shrine")
        out = self.engine.move("telegram:1", "1", ADA)
        self.assertIsNotNone(s["quest"])
        self.engine.move("telegram:1", "tithe", ADA)
        out = self.engine.move("telegram:1", "tithe", ADA)
        self.assertIsNone(s["quest"])
        self.assertEqual(s["quests_done"], 1)
        self.assertTrue(any("shrine accepts" in m for m in out))

    def test_ranks_progress_with_fanfare(self):
        s = self.room.state
        s["pop"] = 25
        s["food"] = 5000
        s["happiness"] = 60
        events = GAME._tick(self.room)
        self.assertEqual(s["rank"], 1)
        self.assertEqual(TOWN_RANKS[1][1], "village")
        self.assertTrue(any("🏆" in e for e in events))

    def test_old_save_migrates(self):
        import random
        s = GAME.new_state(random.Random(1))
        # simulate a pre-prosperity save
        for key in ("happiness", "upgrades", "tech", "quests_done",
                    "golden_age", "mercenary_days", "warded_days",
                    "vein_days", "pending_choice", "quest",
                    "last_golden_age", "rank"):
            s.pop(key, None)
        s["milestone"] = 2
        room = type("R", (), {"state": s,
                              "rng": lambda self: random.Random(1)})()
        GAME._norm(s)
        self.assertEqual(s["rank"], 2)
        self.assertEqual(s["happiness"], 60)
        self.assertEqual(s["upgrades"], {})

    def test_choice_events_table_sane(self):
        names = [c[0] for c in CHOICE_EVENTS]
        self.assertEqual(len(names), len(set(names)))
        for name, weight, min_day, prompt, options in CHOICE_EVENTS:
            self.assertGreater(weight, 0)
            self.assertGreaterEqual(min_day, 1)
            self.assertTrue(prompt)
            self.assertEqual(len(options), 2)
            for label, effects, _result in options:
                self.assertTrue(label)
                self.assertIsInstance(effects, dict)

    def test_techs_gated_sanely(self):
        for name, spec in TECHS.items():
            self.assertLess(spec["rank"], len(TOWN_RANKS))
            self.assertGreater(spec["cost_g"], 0)


if __name__ == "__main__":
    unittest.main()
