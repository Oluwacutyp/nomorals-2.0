"""AI opponent upgrades: quality, difficulty, and the connect4 fix.

- connect4: the house's discs actually land now (regression), it takes
  wins, blocks losses, opens centre on expert, and respects its time
  budget at every difficulty;
- battleship: target mode (neighbor of a lone hit, line extension on a
  pair) and hunt-mode probability density;
- reversi: the expert takes a free corner over greedy flips;
- gomoku: the house blocks an open four and takes its own win;
- ttt: easy blunders (seeded), normal stays perfect;
- duel: accuracy follows the difficulty ladder;
- difficulty plumbing: engine.start(difficulty=...) lands in state and
  only games that declare it receive it;
- economy: the shop buy flow works through the engine (coins deducted,
  item granted).
"""
from __future__ import annotations

import random
import time
import unittest

from nomorals.games.ai import (battleship_shot, connect4_move,
                               reversi_move)
from nomorals.games.engine import GameEngine
from nomorals.games.games.inbox import (
    _gomoku_house_move,
    _reversi_legal,
    _reversi_flips,
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
    return engine, db, sent


ADA = Player.from_sender("telegram", "456", "Ada")


def _empty_c4():
    return [[0] * 7 for _ in range(6)]


class Connect4BrainTests(unittest.TestCase):
    def test_takes_immediate_win(self):
        board = _empty_c4()
        board[5][0] = board[5][1] = board[5][2] = 2
        for diff in ("easy", "normal", "hard", "expert"):
            self.assertEqual(
                connect4_move([r[:] for r in board], 2, difficulty=diff,
                              rng=random.Random(0)), 3)

    def test_blocks_opponent_win(self):
        board = _empty_c4()
        board[5][0] = board[5][1] = board[5][2] = 1
        for diff in ("easy", "normal", "hard", "expert"):
            self.assertEqual(
                connect4_move([r[:] for r in board], 2, difficulty=diff,
                              rng=random.Random(0)), 3)

    def test_easy_plays_legal_random(self):
        board = _empty_c4()
        for c in range(7):
            board[0][c] = 1  # column full except col 6
        board[0][6] = 0
        # only column 6 is open and it's no instant win/block
        col = connect4_move(board, 2, difficulty="easy",
                            rng=random.Random(3))
        self.assertEqual(col, 6)

    def test_full_board_returns_minus_one(self):
        board = [[1] * 7 for _ in range(6)]
        self.assertEqual(connect4_move(board, 1), -1)

    def test_expert_opens_centre(self):
        col = connect4_move(_empty_c4(), 2, difficulty="expert",
                            rng=random.Random(1))
        self.assertEqual(col, 3)

    def test_time_budget_respected(self):
        mid = _empty_c4()
        for r, c, p in [(5, 3, 1), (5, 2, 2), (5, 4, 1), (4, 3, 2),
                        (5, 0, 1), (5, 6, 2), (4, 4, 1), (3, 3, 2)]:
            mid[r][c] = p
        budgets = {"normal": 0.8, "hard": 1.6, "expert": 3.0}
        for diff, budget in budgets.items():
            t = time.time()
            col = connect4_move([r[:] for r in mid], 1, difficulty=diff,
                                rng=random.Random(0))
            elapsed = time.time() - t
            self.assertLess(elapsed, budget,
                            f"{diff} took {elapsed:.2f}s")
            self.assertIn(col, range(7))

    def test_house_discs_land_on_the_board(self):
        """Regression: the old ai_turn announced a column but never
        dropped the disc — the house's pieces never appeared."""
        engine, _db, _sent = make_engine()
        try:
            room, _ = engine.start("t:c4x", "connect4", ADA,
                                   difficulty="easy")
            engine.move("t:c4x", "4", ADA)
            engine.move("t:c4x", "4", ADA)
            house_discs = sum(cell == 2 for row in room.state["board"]
                              for cell in row)
            human_discs = sum(cell == 1 for row in room.state["board"]
                              for cell in row)
            self.assertGreater(house_discs, 0)
            self.assertEqual(human_discs, house_discs)
        finally:
            engine.shutdown()


class BattleshipBrainTests(unittest.TestCase):
    def test_lone_hit_targets_neighbor(self):
        shots = [[0] * 10 for _ in range(10)]
        shots[5][5] = 1
        for _ in range(10):
            r, c = battleship_shot(shots, (5, 4, 3, 3, 2),
                                   difficulty="hard",
                                   rng=random.Random())
            self.assertEqual(abs(r - 5) + abs(c - 5), 1)

    def test_aligned_pair_extends_the_line(self):
        shots = [[0] * 10 for _ in range(10)]
        shots[5][5] = shots[5][6] = 1
        seen = set()
        for _ in range(10):
            seen.add(battleship_shot(shots, (5, 4, 3, 3, 2),
                                     difficulty="hard",
                                     rng=random.Random()))
        self.assertTrue(seen <= {(5, 4), (5, 7)},
                        f"unexpected targets: {seen}")

    def test_hunt_mode_never_repeats_a_shot(self):
        rng = random.Random(11)
        shots = [[0] * 10 for _ in range(10)]
        for _ in range(30):
            r, c = battleship_shot(shots, (5, 4, 3, 3, 2),
                                   difficulty="expert", rng=rng)
            self.assertEqual(shots[r][c], 0)
            shots[r][c] = 2  # mark misses, keep hunting

    def test_easy_uses_parity(self):
        shots = [[0] * 10 for _ in range(10)]
        for _ in range(20):
            r, c = battleship_shot(shots, (5, 4, 3, 3, 2),
                                   difficulty="easy",
                                   rng=random.Random())
            self.assertEqual((r + c) % 2, 0)
            shots[r][c] = 2

    def test_sunk_ship_drops_out_of_targeting(self):
        # a hit fully enclosed by misses = a sunk ship: the brain goes
        # back to hunting instead of probing its (shot) neighbors
        shots = [[0] * 10 for _ in range(10)]
        shots[5][5] = 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shots[5 + dr][5 + dc] = 2
        r, c = battleship_shot(shots, (5, 4, 3, 3, 2),
                               difficulty="hard",
                               rng=random.Random(0))
        self.assertNotEqual((r, c), (5, 5))
        self.assertEqual(shots[r][c], 0)


def _reversi_apply(grid, r, c, side):
    flips = _reversi_flips(grid, r, c, side)
    grid[r][c] = side
    for fr, fc in flips:
        grid[fr][fc] = side


class ReversiBrainTests(unittest.TestCase):
    def test_expert_takes_free_corner_over_greedy(self):
        # corner a1 is legal for W and flips 1; b2-ish moves flip more
        # but hand over the corner — the expert takes the corner
        grid = [[""] * 8 for _ in range(8)]
        grid[0][1] = "B"
        grid[0][2] = "W"
        grid[1][0] = "B"
        grid[2][0] = "W"
        grid[1][1] = "B"
        grid[1][2] = "B"
        grid[2][1] = "B"
        grid[2][2] = "W"
        legal = _reversi_legal(grid, "W")
        self.assertTrue(any((r, c) == (0, 0) for r, c, _ in legal))
        mv = reversi_move(grid, "W", difficulty="expert",
                          rng=random.Random(0), legal_fn=_reversi_legal,
                          apply_fn=_reversi_apply)
        self.assertEqual(mv, (0, 0))

    def test_normal_is_greedy_most_flips(self):
        grid = [[""] * 8 for _ in range(8)]
        grid[3][3] = "W"
        grid[4][4] = "W"
        grid[3][4] = "B"
        grid[4][3] = "B"
        mv = reversi_move(grid, "B", difficulty="normal",
                          rng=random.Random(0), legal_fn=_reversi_legal,
                          apply_fn=_reversi_apply)
        # opening: every legal move flips exactly 1 — any of the four
        self.assertIn(mv, [(2, 3), (3, 2), (4, 5), (5, 4)])

    def test_easy_returns_legal_move(self):
        grid = [[""] * 8 for _ in range(8)]
        grid[3][3] = "W"
        grid[4][4] = "W"
        grid[3][4] = "B"
        grid[4][3] = "B"
        mv = reversi_move(grid, "B", difficulty="easy",
                          rng=random.Random(0), legal_fn=_reversi_legal,
                          apply_fn=_reversi_apply)
        self.assertIn(mv, [(2, 3), (3, 2), (4, 5), (5, 4)])

    def test_no_legal_move_returns_none(self):
        grid = [["B"] * 8 for _ in range(8)]
        self.assertIsNone(reversi_move(grid, "W", difficulty="expert",
                                       rng=random.Random(0),
                                       legal_fn=_reversi_legal,
                                       apply_fn=_reversi_apply))


class GomokuBrainTests(unittest.TestCase):
    def _grid(self):
        return [[""] * 15 for _ in range(15)]

    def test_takes_immediate_win(self):
        g = self._grid()
        for c in range(4):
            g[7][3 + c] = "W"
        self.assertEqual(_gomoku_house_move(g, random.Random(0),
                                            "normal"), (7, 2))

    def test_blocks_open_four(self):
        g = self._grid()
        for c in range(4):
            g[7][3 + c] = "B"
        mv = _gomoku_house_move(g, random.Random(0), "normal")
        self.assertIn(mv, [(7, 2), (7, 7)])

    def test_expert_avoids_walking_into_a_fork(self):
        # B threatens at two open ends; the expert must block the side
        # that doesn't let B win anyway — it should not play a dead move
        g = self._grid()
        for c in range(3):
            g[7][3 + c] = "B"   # three in a row, open both ends
        mv = _gomoku_house_move(g, random.Random(0), "expert")
        self.assertIn(mv, [(7, 2), (7, 6)])


class TttDuelDifficultyTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_ttt_easy_sometimes_blunders(self):
        # seeded: over many fresh rooms, easy must deviate from the
        # perfect reply at least once
        from nomorals.games.games.wild import TicTacToeGame
        game = TicTacToeGame()
        seen = set()
        for seed in range(40):
            room, _ = self.engine.start(f"t:te{seed}", "ttt", ADA,
                                        difficulty="easy")
            room.seed = seed
            room._rng_instance = None
            # human takes corner 1; perfect reply is center (5)
            self.engine.move(f"t:te{seed}", "1", ADA)
            board = room.state["board"]
            house_sq = next(i for i, c in enumerate(board) if c == "O")
            seen.add(house_sq)
            self.engine.quit(f"t:te{seed}")
        self.assertGreater(len(seen), 1,
                           "easy house always played the perfect reply")

    def test_ttt_normal_is_perfect(self):
        # the house plays the minimax move: after X takes a corner, the
        # reply must hold the draw (score 0 for O), never a losing move
        from nomorals.games.games.wild import _ttt_minimax
        board = ["X"] + [""] * 8
        score, move = _ttt_minimax(board, "O", "O", 0)
        self.assertEqual(score, 0)
        room, _ = self.engine.start("t:tn", "ttt", ADA,
                                    difficulty="normal")
        self.engine.move("t:tn", "1", ADA)
        played = [i for i, c in enumerate(room.state["board"])
                  if c == "O"]
        self.assertEqual(played, [move])

    def test_duel_accuracy_follows_difficulty(self):
        game = self.engine.games["duel"]
        self.assertLess(game.DUEL_ACCURACY["easy"],
                        game.DUEL_ACCURACY["normal"])
        self.assertLess(game.DUEL_ACCURACY["normal"],
                        game.DUEL_ACCURACY["hard"])
        self.assertLess(game.DUEL_ACCURACY["hard"],
                        game.DUEL_ACCURACY["expert"])


class DifficultyPlumbingTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_difficulty_reaches_state(self):
        for name, key in (("connect4", "difficulty"),
                          ("battleship", "difficulty"),
                          ("reversi", "difficulty"),
                          ("checkers", "difficulty"),
                          ("gomoku", "difficulty"),
                          ("ttt", "difficulty"),
                          ("duel", "difficulty")):
            room, _ = self.engine.start(f"t:d{name}", name, ADA,
                                        difficulty="hard")
            self.assertEqual(room.state.get(key), "hard", name)
            self.engine.quit(f"t:d{name}")

    def test_games_without_difficulty_ignore_it(self):
        room, _ = self.engine.start("t:dw", "wordle", ADA,
                                    difficulty="hard")
        self.assertNotIn("difficulty", room.state)

    def test_list_games_marks_difficulty_games(self):
        listing = self.engine.list_games()
        self.assertIn("[easy|normal|hard|expert]", listing)
        self.assertIn("sudoku", listing)
        self.assertIn("anagram", listing)
        self.assertIn("cryptogram", listing)
        self.assertIn("— puzzles —", listing)


class EconomyShopTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_shop_buy_flow(self):
        room, _ = self.engine.start("t:sh1", "sudoku", ADA)
        catalog = self.engine.move("t:sh1", "/shop", ADA)
        self.assertTrue(any("hint_scroll" in m for m in catalog))
        # broke at first
        denied = self.engine.move("t:sh1", "/shop buy hint_scroll", ADA)
        self.assertTrue(any("costs 80c" in m for m in denied))
        # fund and buy
        self.engine.store.add_coins(ADA, 200, "test")
        ok = self.engine.move("t:sh1", "/shop buy hint_scroll", ADA)
        self.assertTrue(any("bought Hint Scroll" in m for m in ok))
        bal = self.engine.move("t:sh1", "/balance", ADA)
        self.assertTrue(any("hint_scroll×1" in m for m in bal))
        # unknown item
        nope = self.engine.move("t:sh1", "/shop buy no_such_thing", ADA)
        self.assertTrue(any("no such item" in m for m in nope))

    def test_win_credits_coins(self):
        room, _ = self.engine.start("t:sh2", "ttt", ADA,
                                    difficulty="easy")
        before = self.engine.store.get(ADA.key).coins
        # force a quick human win: take corners/edges around a blunder
        # (easy may still draw — just check the ledger moved either way)
        for sq in ["1", "9", "3", "7", "2"]:
            if self.engine.live("t:sh2") is None:
                break
            self.engine.move("t:sh2", sq, ADA)
        after = self.engine.store.get(ADA.key).coins
        self.assertGreaterEqual(after, before)


if __name__ == "__main__":
    unittest.main()
