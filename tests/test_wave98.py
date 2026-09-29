"""Wave 98 — casino games + game statistics.

Casino games:
* **blackjack** — 21 or bust, hit/stand/double/split, dealer AI.
* **roulette** — bet numbers/colors/dozens, spin wheel, payouts.
* **slots** — 3 reels, match symbols, jackpot.

Game statistics:
* per-player per-game aggregates (played, won, win rate, best score, total score);
* engine updates stats on game end;
* CLI command ``nm stats`` to view.

Everything hermetic: in-memory db, scripted moves, no network.
"""
from __future__ import annotations

import random
import unittest

from nomorals.games import Player
from nomorals.games.engine import GameEngine
from nomorals.games.games.casino import (
    CASINO_GAMES,
    BlackjackGame,
    RouletteGame,
    SlotsGame,
)
from nomorals.storage.db import Database

try:
    from tests.test_wave85_games import ADA, drive, make_engine
except ImportError:
    import sys
    from pathlib import Path
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_wave85_games import ADA, drive, make_engine


# ── 1. blackjack ─────────────────────────────────────────────────────────────

class BlackjackTests(unittest.TestCase):
    def test_registered(self):
        names = [g.name for g in CASINO_GAMES]
        self.assertIn("blackjack", names)

    def test_hand_value(self):
        game = BlackjackGame()
        self.assertEqual(game._hand_value([10, 7]), 17)
        self.assertEqual(game._hand_value([14, 7]), 18)  # Ace = 11
        self.assertEqual(game._hand_value([14, 14, 5]), 17)  # Ace + Ace + 5 = 27, -10 = 17
        self.assertEqual(game._hand_value([10, 10, 14]), 21)  # Ace = 1 when needed

    def test_card_value(self):
        game = BlackjackGame()
        self.assertEqual(game._card_value(2), 2)
        self.assertEqual(game._card_value(10), 10)
        self.assertEqual(game._card_value(11), 10)  # J
        self.assertEqual(game._card_value(14), 11)  # A = 11

    def test_hit(self):
        engine, _ = make_engine()
        room, _ = engine.start("bj-hit", "blackjack", ADA, kind="dm")
        old_len = len(room.state["player"])
        out = engine.move("bj-hit", "hit", ADA)
        self.assertEqual(len(room.state["player"]), old_len + 1)
        engine.shutdown()

    def test_bust(self):
        engine, _ = make_engine()
        room, _ = engine.start("bj-bust", "blackjack", ADA, kind="dm")
        room.state["player"] = [10, 10, 5]  # 25
        out = engine.move("bj-bust", "hit", ADA)
        self.assertTrue(room.state["done"])
        self.assertEqual(room.state["result"], "bust")
        engine.shutdown()

    def test_stand(self):
        engine, _ = make_engine()
        room, _ = engine.start("bj-stand", "blackjack", ADA, kind="dm")
        room.state["player"] = [10, 7]  # 17
        room.state["dealer"] = [10, 6]  # 16
        out = engine.move("bj-stand", "stand", ADA)
        self.assertTrue(room.state["done"])
        engine.shutdown()

    def test_double(self):
        engine, _ = make_engine()
        room, _ = engine.start("bj-double", "blackjack", ADA, kind="dm")
        room.state["player"] = [10, 5]
        old_bet = room.state["bet"]
        out = engine.move("bj-double", "double", ADA)
        self.assertEqual(room.state["bet"], old_bet * 2)
        self.assertEqual(len(room.state["player"]), 3)
        self.assertTrue(room.state["done"])
        engine.shutdown()

    def test_split(self):
        engine, _ = make_engine()
        room, _ = engine.start("bj-split", "blackjack", ADA, kind="dm")
        room.state["player"] = [10, 10]
        out = engine.move("bj-split", "split", ADA)
        self.assertIsNotNone(room.state["split"])
        self.assertEqual(len(room.state["player"]), 2)
        self.assertEqual(len(room.state["split"]), 2)
        engine.shutdown()

    def test_dealer_hits_on_16(self):
        game = BlackjackGame()
        room_state = {"dealer": [10, 6], "deck": [10, 10, 10],
                      "player": [10, 7], "bet": 10, "done": False,
                      "result": "", "split": None}
        from nomorals.games.games.base import Room
        room = Room(id="test", chat_key="test", game="blackjack",
                    kind="dm", platform="test", players=[ADA], state=room_state, seed=1)
        out = game._dealer_play(room)
        self.assertGreater(len(room_state["dealer"]), 2)
        self.assertGreaterEqual(game._hand_value(room_state["dealer"]), 17)

    def test_game_ends_on_dealer_play(self):
        engine, _ = make_engine()
        room, _ = engine.start("bj-end", "blackjack", ADA, kind="dm")
        room.state["player"] = [10, 8]
        out = engine.move("bj-end", "stand", ADA)
        self.assertTrue(engine.is_over(room))
        engine.shutdown()


# ── 2. roulette ──────────────────────────────────────────────────────────────

class RouletteTests(unittest.TestCase):
    def test_registered(self):
        names = [g.name for g in CASINO_GAMES]
        self.assertIn("roulette", names)

    def test_color(self):
        game = RouletteGame()
        self.assertEqual(game._color(0), "green")
        self.assertEqual(game._color(1), "red")
        self.assertEqual(game._color(2), "black")
        self.assertEqual(game._color(36), "red")

    def test_bet_number(self):
        engine, _ = make_engine()
        room, _ = engine.start("rou-num", "roulette", ADA, kind="dm")
        out = engine.move("rou-num", "bet number 17", ADA)
        self.assertEqual(room.state["bet_type"], "number")
        self.assertEqual(room.state["bet_value"], 17)
        engine.shutdown()

    def test_bet_color(self):
        engine, _ = make_engine()
        room, _ = engine.start("rou-col", "roulette", ADA, kind="dm")
        out = engine.move("rou-col", "bet red", ADA)
        self.assertEqual(room.state["bet_type"], "color")
        self.assertEqual(room.state["bet_value"], "red")
        engine.shutdown()

    def test_bet_dozen(self):
        engine, _ = make_engine()
        room, _ = engine.start("rou-doz", "roulette", ADA, kind="dm")
        out = engine.move("rou-doz", "bet dozen 2", ADA)
        self.assertEqual(room.state["bet_type"], "dozen")
        self.assertEqual(room.state["bet_value"], 2)
        engine.shutdown()

    def test_spin_wins(self):
        engine, _ = make_engine()
        room, _ = engine.start("rou-win", "roulette", ADA, kind="dm")
        room.state["bet_type"] = "number"
        room.state["bet_value"] = 5
        # Force the RNG to return 5
        room._rng_instance = random.Random(0)
        room._rng_instance.randint = lambda a, b: 5
        out = engine.move("rou-win", "spin", ADA)
        self.assertTrue(room.state["done"])
        self.assertEqual(room.state["result"], 5)
        self.assertEqual(room.state["payout"], 10 * 35)
        engine.shutdown()

    def test_spin_loses(self):
        engine, _ = make_engine()
        room, _ = engine.start("rou-lose", "roulette", ADA, kind="dm")
        room.state["bet_type"] = "number"
        room.state["bet_value"] = 5
        room._rng_instance = random.Random(0)
        room._rng_instance.randint = lambda a, b: 10
        out = engine.move("rou-lose", "spin", ADA)
        self.assertTrue(room.state["done"])
        self.assertEqual(room.state["payout"], -10)
        engine.shutdown()

    def test_game_ends_on_spin(self):
        engine, _ = make_engine()
        room, _ = engine.start("rou-end", "roulette", ADA, kind="dm")
        room.state["bet_type"] = "color"
        room.state["bet_value"] = "red"
        out = engine.move("rou-end", "spin", ADA)
        self.assertTrue(engine.is_over(room))
        engine.shutdown()


# ── 3. slots ─────────────────────────────────────────────────────────────────

class SlotsTests(unittest.TestCase):
    def test_registered(self):
        names = [g.name for g in CASINO_GAMES]
        self.assertIn("slots", names)

    def test_bet(self):
        engine, _ = make_engine()
        room, _ = engine.start("slots-bet", "slots", ADA, kind="dm")
        out = engine.move("slots-bet", "bet 50", ADA)
        self.assertEqual(room.state["bet"], 50)
        engine.shutdown()

    def test_spin_triple(self):
        engine, _ = make_engine()
        room, _ = engine.start("slots-tri", "slots", ADA, kind="dm")
        # Force RNG to return cherries (called 3 times, k=1 each)
        room._rng_instance = random.Random(0)
        room._rng_instance.choices = lambda syms, weights, k: ["🍒"]
        out = engine.move("slots-tri", "spin", ADA)
        self.assertTrue(room.state["done"])
        self.assertEqual(room.state["reels"], ["🍒", "🍒", "🍒"])
        self.assertEqual(room.state["payout"], 10 * 2)
        engine.shutdown()

    def test_spin_jackpot(self):
        engine, _ = make_engine()
        room, _ = engine.start("slots-jack", "slots", ADA, kind="dm")
        room._rng_instance = random.Random(0)
        room._rng_instance.choices = lambda syms, weights, k: ["💎"]
        out = engine.move("slots-jack", "spin", ADA)
        self.assertTrue(room.state["done"])
        self.assertEqual(room.state["reels"], ["💎", "💎", "💎"])
        self.assertEqual(room.state["payout"], 10 * 100)
        engine.shutdown()

    def test_spin_no_match(self):
        engine, _ = make_engine()
        room, _ = engine.start("slots-nom", "slots", ADA, kind="dm")
        # Mock choices to be called 3 times, returning one symbol each
        calls = [iter(["🍒", "🍋", "🍊"])]
        room._rng_instance = random.Random(0)
        room._rng_instance.choices = lambda syms, weights, k: [next(calls[0])]
        out = engine.move("slots-nom", "spin", ADA)
        self.assertTrue(room.state["done"])
        self.assertEqual(room.state["reels"], ["🍒", "🍋", "🍊"])
        self.assertEqual(room.state["payout"], -10)
        engine.shutdown()

    def test_game_ends_on_spin(self):
        engine, _ = make_engine()
        room, _ = engine.start("slots-end", "slots", ADA, kind="dm")
        out = engine.move("slots-end", "spin", ADA)
        self.assertTrue(engine.is_over(room))
        engine.shutdown()


# ── 4. game statistics ───────────────────────────────────────────────────────

class GameStatsTests(unittest.TestCase):
    def test_update_stat(self):
        from nomorals.games.achievements import update_game_stats, get_game_stats
        db = Database(":memory:")
        db.migrate()
        update_game_stats(db, "player:1", "2048", won=True, score=5000)
        stats = get_game_stats(db, "player:1")
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["game"], "2048")
        self.assertEqual(stats[0]["played"], 1)
        self.assertEqual(stats[0]["won"], 1)
        self.assertEqual(stats[0]["best_score"], 5000)

    def test_update_stat_aggregates(self):
        from nomorals.games.achievements import update_game_stats, get_game_stats
        db = Database(":memory:")
        db.migrate()
        update_game_stats(db, "player:1", "2048", won=True, score=3000)
        update_game_stats(db, "player:1", "2048", won=False, score=2000)
        update_game_stats(db, "player:1", "2048", won=True, score=6000)
        stats = get_game_stats(db, "player:1")
        self.assertEqual(stats[0]["played"], 3)
        self.assertEqual(stats[0]["won"], 2)
        self.assertEqual(stats[0]["win_rate"], 2/3)
        self.assertEqual(stats[0]["best_score"], 6000)
        self.assertEqual(stats[0]["total_score"], 11000)

    def test_multiple_games(self):
        from nomorals.games.achievements import update_game_stats, get_game_stats
        db = Database(":memory:")
        db.migrate()
        update_game_stats(db, "player:1", "2048", won=True, score=5000)
        update_game_stats(db, "player:1", "snake", won=False, score=100)
        stats = get_game_stats(db, "player:1")
        self.assertEqual(len(stats), 2)
        games = {s["game"] for s in stats}
        self.assertEqual(games, {"2048", "snake"})

    def test_engine_updates_stats(self):
        from nomorals.games.achievements import get_game_stats
        engine, _ = make_engine()
        room, _ = engine.start("2048-stat", "2048", ADA, kind="dm")
        room.state["score"] = 4000
        room.state["lost"] = True
        engine._finish(room)
        stats = get_game_stats(engine.db, ADA.key)
        self.assertGreater(len(stats), 0)
        stat = next((s for s in stats if s["game"] == "2048"), None)
        self.assertIsNotNone(stat)
        self.assertEqual(stat["played"], 1)
        engine.shutdown()


# ── 5. casino achievements ──────────────────────────────────────────────────

class CasinoAchievementsTests(unittest.TestCase):
    def test_blackjack_win_achievement(self):
        from nomorals.games.achievements import get_achievements
        engine, _ = make_engine()
        room, _ = engine.start("bj-ach", "blackjack", ADA, kind="dm")
        # Set up a winning state: player has 18, dealer has 16 and will bust
        room.state["player"] = [10, 8]  # 18
        room.state["dealer"] = [10, 6]  # 16, must hit
        room.state["deck"] = [10, 10, 10]  # dealer draws 10, busts with 26
        engine.move("bj-ach", "stand", ADA)
        achievements = get_achievements(engine.db, ADA.key)
        ids = [a["id"] for a in achievements]
        self.assertIn("blackjack_win", ids)
        engine.shutdown()

    def test_slots_jackpot_achievement(self):
        from nomorals.games.achievements import get_achievements
        engine, _ = make_engine()
        room, _ = engine.start("slots-ach", "slots", ADA, kind="dm")
        room.state["reels"] = ["💎", "💎", "💎"]
        room.state["payout"] = 1000
        room.state["done"] = True
        engine._finish(room)
        achievements = get_achievements(engine.db, ADA.key)
        ids = [a["id"] for a in achievements]
        self.assertIn("slots_jackpot", ids)
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
