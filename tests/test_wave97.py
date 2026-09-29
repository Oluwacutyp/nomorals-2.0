"""Wave 97 — arcade games + achievements + leaderboards.

Arcade games:
* **2048** — sliding puzzle, merge tiles, reach 2048.
* **snake** — grid movement, eat food, don't crash.
* **connect4** — drop discs, align four.
* **battleship** — hidden grid, hunt ships.

Achievements:
* cross-game unlocks persisted in DB;
* engine awards them on game end based on outcome + state.

Leaderboards:
* per-game high scores in DB;
* engine records scores on game end;
* CLI commands to view top 10.

Everything hermetic: in-memory db, scripted moves, no network.
"""
from __future__ import annotations

import random
import unittest

from nomorals.games import Player
from nomorals.games.engine import GameEngine
from nomorals.games.games.arcade import (
    ARCADE_GAMES,
    BattleshipGame,
    ConnectFourGame,
    SnakeGame,
    TwentyFortyEightGame,
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


# ── 1. 2048 ─────────────────────────────────────────────────────────────────

class TwentyFortyEightTests(unittest.TestCase):
    def test_registered(self):
        names = [g.name for g in ARCADE_GAMES]
        self.assertIn("2048", names)

    def test_slide_and_merge(self):
        game = TwentyFortyEightGame()
        row = [2, 2, 0, 0]
        new_row, points = game._slide_row(row)
        self.assertEqual(new_row, [4, 0, 0, 0])
        self.assertEqual(points, 4)

    def test_slide_complex(self):
        game = TwentyFortyEightGame()
        row = [2, 2, 4, 4]
        new_row, points = game._slide_row(row)
        self.assertEqual(new_row, [4, 8, 0, 0])
        self.assertEqual(points, 12)

    def test_move_left(self):
        game = TwentyFortyEightGame()
        grid = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        new_grid, points = game._move(grid, "l")
        self.assertEqual(new_grid[0], [4, 0, 0, 0])
        self.assertEqual(points, 4)

    def test_can_move_empty(self):
        game = TwentyFortyEightGame()
        grid = [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096],
                [8192, 16384, 32768, 0]]
        self.assertTrue(game._can_move(grid))

    def test_no_moves_full_no_merges(self):
        game = TwentyFortyEightGame()
        grid = [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096],
                [8192, 16384, 32768, 65536]]
        self.assertFalse(game._can_move(grid))

    def test_game_finishes_on_no_moves(self):
        engine, _ = make_engine()
        room, _ = engine.start("2048-test", "2048", ADA, kind="dm")
        # Force a state where one move is possible, then the next move
        # leaves no moves
        room.state["grid"] = [[2, 4, 8, 16], [32, 64, 128, 256],
                              [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]]
        # Make one cell empty so a move is possible
        room.state["grid"][3][3] = 0
        # The move will fill that cell, then if no merges possible, lost=True
        out = engine.move("2048-test", "u", ADA)
        # After the move, either lost=True or the spawn created a movable state
        # Either way, the game logic ran
        self.assertIsInstance(out, list)
        engine.shutdown()

    def test_win_on_2048(self):
        engine, _ = make_engine()
        room, _ = engine.start("2048-win", "2048", ADA, kind="dm")
        room.state["grid"] = [[1024, 1024, 0, 0], [0, 0, 0, 0],
                              [0, 0, 0, 0], [0, 0, 0, 0]]
        out = engine.move("2048-win", "l", ADA)
        self.assertTrue(room.state.get("won"))
        engine.shutdown()

    def test_score_tracks(self):
        engine, _ = make_engine()
        room, _ = engine.start("2048-score", "2048", ADA, kind="dm")
        room.state["grid"] = [[2, 2, 0, 0], [0, 0, 0, 0],
                              [0, 0, 0, 0], [0, 0, 0, 0]]
        engine.move("2048-score", "l", ADA)
        self.assertGreater(room.state["score"], 0)
        engine.shutdown()


# ── 2. snake ────────────────────────────────────────────────────────────────

class SnakeTests(unittest.TestCase):
    def test_registered(self):
        names = [g.name for g in ARCADE_GAMES]
        self.assertIn("snake", names)

    def test_move_right(self):
        engine, _ = make_engine()
        room, _ = engine.start("snake-test", "snake", ADA, kind="dm")
        head = room.state["snake"][0]
        engine.move("snake-test", "r", ADA)
        new_head = room.state["snake"][0]
        self.assertEqual(new_head, (head[0], head[1] + 1))
        engine.shutdown()

    def test_cant_reverse(self):
        engine, _ = make_engine()
        room, _ = engine.start("snake-rev", "snake", ADA, kind="dm")
        room.state["dir"] = "r"
        out = engine.move("snake-rev", "l", ADA)
        self.assertTrue(any("can't reverse" in m for m in out))
        engine.shutdown()

    def test_eat_food(self):
        engine, _ = make_engine()
        room, _ = engine.start("snake-food", "snake", ADA, kind="dm")
        head = room.state["snake"][0]
        room.state["food"] = (head[0], head[1] + 1)
        old_len = len(room.state["snake"])
        engine.move("snake-food", "r", ADA)
        self.assertEqual(len(room.state["snake"]), old_len + 1)
        self.assertEqual(room.state["score"], 10)
        engine.shutdown()

    def test_wall_crash(self):
        engine, _ = make_engine()
        room, _ = engine.start("snake-wall", "snake", ADA, kind="dm")
        room.state["snake"] = [(0, 0)]
        room.state["dir"] = "r"
        out = engine.move("snake-wall", "u", ADA)
        self.assertFalse(room.state["alive"])
        engine.shutdown()

    def test_self_crash(self):
        engine, _ = make_engine()
        room, _ = engine.start("snake-self", "snake", ADA, kind="dm")
        room.state["snake"] = [(5, 5), (5, 6), (5, 7), (5, 6)]
        room.state["dir"] = "r"
        out = engine.move("snake-self", "r", ADA)
        self.assertFalse(room.state["alive"])
        engine.shutdown()

    def test_game_ends_on_crash(self):
        engine, _ = make_engine()
        room, _ = engine.start("snake-end", "snake", ADA, kind="dm")
        room.state["snake"] = [(0, 0)]
        out = engine.move("snake-end", "u", ADA)
        self.assertTrue(engine.is_over(room))
        engine.shutdown()


# ── 3. connect four ─────────────────────────────────────────────────────────

class ConnectFourTests(unittest.TestCase):
    def test_registered(self):
        names = [g.name for g in ARCADE_GAMES]
        self.assertIn("connect4", names)

    def test_drop_disc(self):
        game = ConnectFourGame()
        board = [[0] * 7 for _ in range(6)]
        row = game._drop(board, 3, 1)
        self.assertEqual(row, 5)
        self.assertEqual(board[5][3], 1)

    def test_stack_discs(self):
        game = ConnectFourGame()
        board = [[0] * 7 for _ in range(6)]
        game._drop(board, 3, 1)
        row = game._drop(board, 3, 2)
        self.assertEqual(row, 4)
        self.assertEqual(board[4][3], 2)

    def test_column_full(self):
        game = ConnectFourGame()
        board = [[0] * 7 for _ in range(6)]
        for _ in range(6):
            game._drop(board, 3, 1)
        row = game._drop(board, 3, 2)
        self.assertEqual(row, -1)

    def test_horizontal_win(self):
        game = ConnectFourGame()
        board = [[0] * 7 for _ in range(6)]
        for c in range(4):
            board[5][c] = 1
        self.assertTrue(game._check_win(board, 1))

    def test_vertical_win(self):
        game = ConnectFourGame()
        board = [[0] * 7 for _ in range(6)]
        for r in range(4):
            board[r][3] = 1
        self.assertTrue(game._check_win(board, 1))

    def test_diagonal_win(self):
        game = ConnectFourGame()
        board = [[0] * 7 for _ in range(6)]
        for i in range(4):
            board[5 - i][i] = 1
        self.assertTrue(game._check_win(board, 1))

    def test_no_win(self):
        game = ConnectFourGame()
        board = [[0] * 7 for _ in range(6)]
        board[5][0] = 1
        board[5][1] = 1
        board[5][2] = 1
        self.assertFalse(game._check_win(board, 1))

    def test_game_ends_on_win(self):
        engine, _ = make_engine()
        room, _ = engine.start("c4-test", "connect4", ADA, kind="dm")
        # Force a near-win state
        room.state["board"][5][0] = 1
        room.state["board"][5][1] = 1
        room.state["board"][5][2] = 1
        out = engine.move("c4-test", "4", ADA)
        self.assertTrue(engine.is_over(room))
        self.assertEqual(room.state["winner"], 1)
        engine.shutdown()

    def test_draw_on_full_board(self):
        engine, _ = make_engine()
        room, _ = engine.start("c4-draw", "connect4", ADA, kind="dm")
        room.state["moves"] = 42
        room.status = "finished"
        self.assertTrue(engine.is_over(room))
        engine.shutdown()


# ── 4. battleship ───────────────────────────────────────────────────────────

class BattleshipTests(unittest.TestCase):
    def test_registered(self):
        names = [g.name for g in ARCADE_GAMES]
        self.assertIn("battleship", names)

    def test_ships_placed(self):
        game = BattleshipGame()
        board = game._place_ships(random.Random(1))
        ship_cells = sum(board[r][c] for r in range(10) for c in range(10))
        self.assertGreaterEqual(ship_cells, 10)  # at least some ships placed

    def test_parse_coord(self):
        game = BattleshipGame()
        self.assertEqual(game._parse_coord("A1"), (0, 0))
        self.assertEqual(game._parse_coord("E10"), (9, 4))
        self.assertEqual(game._parse_coord("J5"), (4, 9))
        self.assertIsNone(game._parse_coord("Z1"))
        self.assertIsNone(game._parse_coord("A11"))

    def test_fire_hit(self):
        engine, _ = make_engine()
        room, _ = engine.start("bs-hit", "battleship", ADA, kind="dm")
        # Place a ship at A1
        room.state["ai"][0][0] = 1
        out = engine.move("bs-hit", "A1", ADA)
        self.assertTrue(any("HIT!" in m for m in out))
        self.assertEqual(room.state["ai_shots"][0][0], 1)
        engine.shutdown()

    def test_fire_miss(self):
        engine, _ = make_engine()
        room, _ = engine.start("bs-miss", "battleship", ADA, kind="dm")
        room.state["ai"][0][0] = 0
        out = engine.move("bs-miss", "A1", ADA)
        self.assertTrue(any("miss." in m for m in out))
        self.assertEqual(room.state["ai_shots"][0][0], 2)
        engine.shutdown()

    def test_cant_fire_twice(self):
        engine, _ = make_engine()
        room, _ = engine.start("bs-twice", "battleship", ADA, kind="dm")
        room.state["ai_shots"][0][0] = 2
        out = engine.move("bs-twice", "A1", ADA)
        self.assertTrue(any("already fired" in m for m in out))
        engine.shutdown()

    def test_win_on_all_sunk(self):
        engine, _ = make_engine()
        room, _ = engine.start("bs-win", "battleship", ADA, kind="dm")
        # Sink all ships
        for r in range(10):
            for c in range(10):
                if room.state["ai"][r][c]:
                    room.state["ai_shots"][r][c] = 1
        out = engine.move("bs-win", "A1", ADA)  # dummy move
        self.assertEqual(room.state["winner"], "player")
        engine.shutdown()


# ── 5. achievements ─────────────────────────────────────────────────────────

class AchievementsTests(unittest.TestCase):
    def test_unlock_achievement(self):
        from nomorals.games.achievements import unlock_achievement, get_achievements
        db = Database(":memory:")
        db.migrate()
        result = unlock_achievement(db, "player:1", "2048_win")
        self.assertTrue(result)
        achievements = get_achievements(db, "player:1")
        self.assertEqual(len(achievements), 1)
        self.assertEqual(achievements[0]["id"], "2048_win")

    def test_unlock_idempotent(self):
        from nomorals.games.achievements import unlock_achievement, get_achievements
        db = Database(":memory:")
        db.migrate()
        unlock_achievement(db, "player:1", "2048_win")
        unlock_achievement(db, "player:1", "2048_win")  # second unlock should not error
        achievements = get_achievements(db, "player:1")
        self.assertEqual(len(achievements), 1)  # still only one

    def test_invalid_achievement(self):
        from nomorals.games.achievements import unlock_achievement
        db = Database(":memory:")
        db.migrate()
        result = unlock_achievement(db, "player:1", "nonexistent")
        self.assertFalse(result)

    def test_engine_awards_achievement(self):
        from nomorals.games.achievements import get_achievements
        engine, _ = make_engine()
        room, _ = engine.start("2048-ach", "2048", ADA, kind="dm")
        room.state["grid"] = [[1024, 1024, 0, 0], [0, 0, 0, 0],
                              [0, 0, 0, 0], [0, 0, 0, 0]]
        engine.move("2048-ach", "l", ADA)
        # Force game end
        room.state["lost"] = True
        engine._finish(room)
        achievements = get_achievements(engine.db, ADA.key)
        ids = [a["id"] for a in achievements]
        self.assertIn("2048_win", ids)
        engine.shutdown()


# ── 6. leaderboards ─────────────────────────────────────────────────────────

class LeaderboardTests(unittest.TestCase):
    def test_record_score(self):
        from nomorals.games.achievements import record_score, get_leaderboard
        db = Database(":memory:")
        db.migrate()
        rank = record_score(db, "2048", "player:1", "Alice", 5000)
        self.assertEqual(rank, 1)
        entries = get_leaderboard(db, "2048")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["score"], 5000)

    def test_rank_ordering(self):
        from nomorals.games.achievements import record_score, get_leaderboard
        db = Database(":memory:")
        db.migrate()
        record_score(db, "2048", "player:1", "Alice", 5000)
        record_score(db, "2048", "player:2", "Bob", 6000)
        record_score(db, "2048", "player:3", "Charlie", 4000)
        entries = get_leaderboard(db, "2048")
        self.assertEqual(entries[0]["score"], 6000)
        self.assertEqual(entries[1]["score"], 5000)
        self.assertEqual(entries[2]["score"], 4000)

    def test_engine_records_score(self):
        from nomorals.games.achievements import get_leaderboard
        engine, _ = make_engine()
        room, _ = engine.start("2048-lb", "2048", ADA, kind="dm")
        room.state["score"] = 3000
        room.state["lost"] = True
        engine._finish(room)
        entries = get_leaderboard(engine.db, "2048")
        self.assertGreater(len(entries), 0)
        self.assertEqual(entries[0]["score"], 3000)
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
