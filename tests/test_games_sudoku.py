"""Sudoku: generator quality, rules, hints, difficulty, achievements.

Covers the new sudoku game end to end:
- the generator produces valid, uniquely-solvable puzzles at every
  difficulty (correct givens band);
- the engine flow: strikes, 3-strikes loss, solving to win;
- hint_scroll integration (shop item → consumed → revealed);
- difficulty plumbing from engine.start into room state;
- achievement unlocks on win (sudoku_win / sudoku_hard / sudoku_clean).
"""
from __future__ import annotations

import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.games.puzzles import (
    SudokuGame,
    _sudoku_count,
    _sudoku_generate,
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


def _valid_solution(grid: list[int]) -> bool:
    for r in range(9):
        if sorted(grid[r * 9:(r + 1) * 9]) != list(range(1, 10)):
            return False
    for c in range(9):
        if sorted(grid[r * 9 + c] for r in range(9)) != list(range(1, 10)):
            return False
    for br in range(3):
        for bc in range(3):
            cells = [grid[(br * 3 + dr) * 9 + bc * 3 + dc]
                     for dr in range(3) for dc in range(3)]
            if sorted(cells) != list(range(1, 10)):
                return False
    return True


class GeneratorTests(unittest.TestCase):
    def test_solution_is_valid_at_every_difficulty(self):
        import random
        # fixed seeds: the generator is randomized, so pin the seeds to
        # keep this deterministic (hash() varies per process)
        for difficulty, givens, seed in (("easy", 44, 101),
                                         ("normal", 36, 202),
                                         ("hard", 30, 303),
                                         ("expert", 26, 404)):
            rng = random.Random(seed)
            solution, puzzle = _sudoku_generate(rng, givens)
            self.assertTrue(_valid_solution(solution),
                            f"invalid solution at {difficulty}")
            n = sum(1 for v in puzzle if v)
            # symmetric digging stalls at a local minimum on some
            # layouts — the count is approximate, never wild
            self.assertLessEqual(n, givens + 6,
                                 f"too many givens at {difficulty}: {n}")
            self.assertGreaterEqual(n, givens - 14,
                                    f"too few givens at {difficulty}: {n}")
            self.assertEqual(_sudoku_count(puzzle[:]), 1,
                             f"not unique at {difficulty}")

    def test_uniqueness_holds_across_seeds(self):
        import random
        for seed in range(6):
            rng = random.Random(5000 + seed)
            solution, puzzle = _sudoku_generate(rng, 30)
            self.assertTrue(_valid_solution(solution), f"seed {seed}")
            self.assertEqual(_sudoku_count(puzzle[:]), 1, f"seed {seed}")

    def test_puzzle_cells_are_subset_of_solution(self):
        import random
        solution, puzzle = _sudoku_generate(random.Random(7), 36)
        for i in range(81):
            if puzzle[i]:
                self.assertEqual(puzzle[i], solution[i])


class SudokuEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_difficulty_plumbing(self):
        room, _ = self.engine.start("t:sd1", "sudoku", ADA,
                                    difficulty="hard")
        self.assertEqual(room.state["difficulty"], "hard")
        self.assertLessEqual(sum(room.state["givens"]), 32)
        room2, _ = self.engine.start("t:sd2", "sudoku", ADA)
        self.assertEqual(room2.state["difficulty"], "normal")
        room3, _ = self.engine.start("t:sd3", "sudoku", ADA,
                                     difficulty="bogus")
        self.assertEqual(room3.state["difficulty"], "normal")

    def test_wrong_digit_is_a_strike(self):
        room, _ = self.engine.start("t:sd4", "sudoku", ADA)
        s = room.state
        i = next(i for i in range(81) if s["board"][i] == 0)
        wrong = next(d for d in range(1, 10) if d != s["solution"][i])
        out = self.engine.move("t:sd4", f"{i // 9 + 1} {i % 9 + 1} {wrong}",
                               ADA)
        self.assertEqual(room.state["mistakes"], 1)
        self.assertEqual(room.state["board"][i], 0)
        self.assertTrue(any("strike 1/3" in m for m in out))

    def test_three_strikes_loses(self):
        room, _ = self.engine.start("t:sd5", "sudoku", ADA)
        s = room.state
        empties = [i for i in range(81) if s["board"][i] == 0][:3]
        for i in empties:
            wrong = next(d for d in range(1, 10) if d != s["solution"][i])
            self.engine.move("t:sd5",
                             f"{i // 9 + 1} {i % 9 + 1} {wrong}", ADA)
        self.assertTrue(room.state["over"])
        self.assertFalse(room.state["won"])
        # the room finished: no live table anymore
        self.assertIsNone(self.engine.live("t:sd5"))

    def test_solving_wins_and_unlocks(self):
        room, _ = self.engine.start("t:sd6", "sudoku", ADA,
                                    difficulty="expert")
        s = room.state
        for i in range(81):
            if self.engine.live("t:sd6") is None:
                break
            if s["board"][i] == 0:
                self.engine.move(
                    "t:sd6",
                    f"r{i // 9 + 1}c{i % 9 + 1} {s['solution'][i]}", ADA)
        self.assertTrue(s["won"])
        self.assertIsNone(self.engine.live("t:sd6"))
        from nomorals.games.achievements import get_achievements
        unlocked = {a["id"] for a in get_achievements(self.db, ADA.key)}
        self.assertIn("sudoku_win", unlocked)
        self.assertIn("sudoku_hard", unlocked)
        self.assertIn("sudoku_clean", unlocked)
        # and the finish announced at least one of them
        self.assertTrue(any("unlocked" in m for m in self.sent))

    def test_givens_cannot_be_touched(self):
        room, _ = self.engine.start("t:sd7", "sudoku", ADA)
        s = room.state
        i = next(i for i in range(81) if s["givens"][i])
        out = self.engine.move("t:sd7",
                               f"{i // 9 + 1} {i % 9 + 1} 9", ADA)
        self.assertTrue(any("given" in m for m in out))
        self.assertEqual(room.state["mistakes"], 0)

    def test_erase_clears_own_entry(self):
        room, _ = self.engine.start("t:sd8", "sudoku", ADA)
        s = room.state
        i = next(i for i in range(81) if s["board"][i] == 0)
        r, c = i // 9 + 1, i % 9 + 1
        self.engine.move("t:sd8", f"{r} {c} {s['solution'][i]}", ADA)
        self.assertNotEqual(room.state["board"][i], 0)
        self.engine.move("t:sd8", f"erase r{r}c{c}", ADA)
        self.assertEqual(room.state["board"][i], 0)

    def test_hint_needs_scroll_then_reveals(self):
        room, _ = self.engine.start("t:sd9", "sudoku", ADA)
        out = self.engine.move("t:sd9", "hint", ADA)
        self.assertTrue(any("Hint Scroll" in m for m in out))
        # buy one through the shop and hint again
        self.engine.store.add_coins(ADA, 500, "test")
        bought = self.engine.move("t:sd9", "/shop buy hint_scroll", ADA)
        self.assertTrue(any("bought" in m for m in bought))
        empties_before = sum(1 for v in room.state["board"] if v == 0)
        out = self.engine.move("t:sd9", "hint", ADA)
        empties_after = sum(1 for v in room.state["board"] if v == 0)
        self.assertEqual(empties_after, empties_before - 1)
        self.assertEqual(room.state["hints_used"], 1)
        self.assertTrue(any("reveals" in m for m in out))

    def test_score_scales_with_difficulty(self):
        game = SudokuGame()
        room, _ = self.engine.start("t:sd10", "sudoku", ADA,
                                    difficulty="easy")
        room.state["won"] = True
        easy_score = game.score(room, ADA)
        room2, _ = self.engine.start("t:sd11", "sudoku", ADA,
                                     difficulty="expert")
        room2.state["won"] = True
        expert_score = game.score(room2, ADA)
        self.assertGreater(expert_score, easy_score)


if __name__ == "__main__":
    unittest.main()
