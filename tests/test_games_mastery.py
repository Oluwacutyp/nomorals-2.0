"""Per-game mastery tiers, meaningful scores, and mastery-gated unlockables.

Covers the non-arena depth wave:
- mastery.py: tier math, per-game ladders, progress, unlocks, /mastery text
- sudoku: hard/expert gating, daily seeded board, timed speed bonus
- gomoku: big/huge board variants, size-aware helpers, scaled scores
- 2048: 5x5 marathon variant, reaching 2048 counts as a win
- hangman: score(), long-words variant
- trivia: sudden-death variant (1 life, double points)
- ttt: meaningful score (difficulty, draw, speed)
- wordchain / numberguess: real scores where there were none
- engine: mastery/variant plumbing, tier-up fanfare on finish
- /mastery command routing + rendering
"""
from __future__ import annotations

import random
import time
import unittest

from nomorals.games.achievements import update_game_stats
from nomorals.games.engine import GameEngine
from nomorals.games.games.arcade import TwentyFortyEightGame
from nomorals.games.games.base import Room
from nomorals.games.games.easy import (
    HangmanGame,
    NumberGuessGame,
    TriviaRoyaleGame,
    WordChainGame,
)
from nomorals.games.games.inbox import (
    GomokuGame,
    _gomoku_house_move,
    _gomoku_parse,
)
from nomorals.games.games.puzzles import SudokuGame
from nomorals.games.games.wild import TicTacToeGame
from nomorals.games.mastery import (
    THRESHOLDS,
    describe_game_mastery,
    describe_mastery,
    get_one_game_stats,
    mastery_line,
    mastery_points,
    mastery_progress,
    mastery_tier,
    new_unlocks,
    unlocks_for,
)
from nomorals.games.players import Player
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    GAME_COMMANDS,
    parse_control,
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


def make_room(game_name: str, state: dict, player: Player = ADA) -> Room:
    room = Room(id="test-room", game=game_name, chat_key="t:x",
                platform="telegram", kind="dm")
    room.state = state
    room.players = [player]
    return room


# ── mastery math ─────────────────────────────────────────────────────────────

class MasteryMathTests(unittest.TestCase):
    def test_no_stats_is_novice_zero(self):
        self.assertEqual(mastery_tier("sudoku", None), ("Novice", 0))
        self.assertEqual(mastery_tier("sudoku", {}), ("Novice", 0))
        self.assertEqual(mastery_points("sudoku", None), 0)

    def test_thresholds(self):
        stats = lambda pts: {"played": 0, "won": 0, "best_score": 0}  # noqa: E731
        # points() is wins*100 + played*10 + score-share; drive tiers
        # purely through wins: 2 wins = 200+ pts → tier 1
        self.assertEqual(
            mastery_tier("ttt", {"played": 2, "won": 2, "best_score": 0})[1], 1)
        self.assertEqual(
            mastery_tier("ttt", {"played": 1, "won": 1, "best_score": 0})[1], 0)
        # 55 wins clears the top threshold
        name, idx = mastery_tier(
            "ttt", {"played": 60, "won": 55, "best_score": 0})
        self.assertEqual(idx, 6)
        self.assertEqual(name, "Legend")
        self.assertEqual(len(THRESHOLDS), 7)

    def test_per_game_ladders(self):
        stats = {"played": 5, "won": 5, "best_score": 0}  # 550 pts → tier 2
        self.assertEqual(mastery_tier("sudoku", stats)[0], "Tactician")
        self.assertEqual(mastery_tier("ttt", stats)[0], "Tactician")
        self.assertEqual(mastery_tier("hangman", stats)[0], "Speller")
        self.assertEqual(mastery_tier("2048", stats)[0], "Merger")
        # unknown game falls back to the default ladder
        self.assertEqual(mastery_tier("zzz_unknown", stats)[0], "Adept")

    def test_score_share_normalizes_per_game(self):
        # a 900-pt sudoku best is worth more than a 900-pt… nothing —
        # the share is capped per game so score inflation can't dominate
        full = mastery_points("sudoku", {"played": 0, "won": 0,
                                         "best_score": 1400})
        half = mastery_points("sudoku", {"played": 0, "won": 0,
                                         "best_score": 700})
        self.assertEqual(full, 100)
        self.assertEqual(half, 50)
        # above the cap still only counts 100
        over = mastery_points("2048", {"played": 0, "won": 0,
                                       "best_score": 10 ** 9})
        self.assertEqual(over, 100)

    def test_progress(self):
        prog = mastery_progress(
            "sudoku", {"played": 10, "won": 8, "best_score": 900})
        self.assertEqual(prog["tier_index"], 2)
        self.assertEqual(prog["tier_name"], "Tactician")
        self.assertEqual(prog["next_name"], "Expert")
        self.assertGreater(prog["to_next"], 0)
        self.assertTrue(0 <= prog["pct_to_next"] <= 100)
        # maxed out: no next tier
        prog = mastery_progress(
            "ttt", {"played": 200, "won": 200, "best_score": 0})
        self.assertEqual(prog["tier_index"], 6)
        self.assertEqual(prog["next_name"], "")
        self.assertEqual(prog["to_next"], 0)

    def test_unlocks(self):
        self.assertEqual(unlocks_for("sudoku", 0), [unlocks_for("sudoku", 0)[0]])
        self.assertIn("hard boards", " ".join(unlocks_for("sudoku", 2)))
        self.assertNotIn("expert", " ".join(unlocks_for("sudoku", 2)))
        fresh = new_unlocks("gomoku", 1, 3)
        self.assertEqual(len(fresh), 1)
        self.assertIn("13×13", fresh[0])
        self.assertEqual(new_unlocks("gomoku", 3, 3), [])
        self.assertEqual(unlocks_for("nope", 6), [])

    def test_mastery_line(self):
        line = mastery_line("sudoku", {"played": 10, "won": 8,
                                       "best_score": 900})
        self.assertIn("Tactician", line)
        self.assertIn("964", line)

    def test_get_one_game_stats(self):
        db = Database(":memory:")
        db.migrate()
        self.assertEqual(get_one_game_stats(db, ADA.key, "sudoku"), {})
        update_game_stats(db, ADA.key, "sudoku", True, 700)
        update_game_stats(db, ADA.key, "sudoku", False, 100)
        row = get_one_game_stats(db, ADA.key, "sudoku")
        self.assertEqual(row["played"], 2)
        self.assertEqual(row["won"], 1)
        self.assertEqual(row["best_score"], 700)
        self.assertAlmostEqual(row["win_rate"], 0.5)

    def test_describe_mastery(self):
        db = Database(":memory:")
        db.migrate()
        empty = describe_mastery(db, ADA.key, "Ada")
        self.assertIn("no mastery yet", empty)
        update_game_stats(db, ADA.key, "sudoku", True, 700)
        update_game_stats(db, ADA.key, "ttt", True, 30)
        text = describe_mastery(db, ADA.key, "Ada")
        self.assertIn("Ada", text)
        self.assertIn("sudoku", text)
        self.assertIn("ttt", text)
        self.assertIn("/mastery <game>", text)

    def test_describe_game_mastery(self):
        db = Database(":memory:")
        db.migrate()
        update_game_stats(db, ADA.key, "sudoku", True, 700)
        text = describe_game_mastery(db, ADA.key, "sudoku")
        self.assertIn("sudoku", text)
        self.assertIn("Novice", text)
        self.assertIn("next:", text)
        self.assertIn("unlocks:", text)
        # fresh game: no-games copy
        fresh = describe_game_mastery(db, ADA.key, "gomoku")
        self.assertIn("no games yet", fresh)


# ── sudoku: gating, daily, timed ─────────────────────────────────────────────

class SudokuMasteryTests(unittest.TestCase):
    def test_expert_gated_without_mastery(self):
        game = SudokuGame()
        s = game.new_state(random.Random(11), difficulty="expert",
                           mastery=0)
        self.assertEqual(s["difficulty"], "normal")
        self.assertEqual(s["locked"], "expert")

    def test_expert_open_with_mastery(self):
        game = SudokuGame()
        s = game.new_state(random.Random(11), difficulty="expert",
                           mastery=3)
        self.assertEqual(s["difficulty"], "expert")
        self.assertEqual(s["locked"], "")

    def test_hard_gate(self):
        game = SudokuGame()
        s = game.new_state(random.Random(12), difficulty="hard", mastery=1)
        self.assertEqual(s["difficulty"], "normal")
        self.assertEqual(s["locked"], "hard")
        s = game.new_state(random.Random(12), difficulty="hard", mastery=2)
        self.assertEqual(s["difficulty"], "hard")
        self.assertEqual(s["locked"], "")

    def test_daily_board_is_deterministic(self):
        game = SudokuGame()
        a = game.new_state(random.Random(21), daily=True)
        b = game.new_state(random.Random(99), daily=True)
        self.assertTrue(a["daily"])
        self.assertEqual(a["puzzle"], b["puzzle"])
        self.assertEqual(a["solution"], b["solution"])

    def test_timed_score_has_speed_bonus(self):
        game = SudokuGame()
        fast = make_room("sudoku", {
            "won": True, "mistakes": 1, "hints_used": 0,
            "difficulty": "normal", "timed": True,
            "start": time.time()})
        slow = make_room("sudoku", {
            "won": True, "mistakes": 1, "hints_used": 0,
            "difficulty": "normal", "timed": True,
            "start": time.time() - 4000})
        plain = make_room("sudoku", {
            "won": True, "mistakes": 1, "hints_used": 0,
            "difficulty": "normal", "timed": False,
            "start": time.time()})
        fast_score = game.score(fast, ADA)
        slow_score = game.score(slow, ADA)
        plain_score = game.score(plain, ADA)
        # base 700 - 150 = 550; fast adds up to +180, slow adds 0
        self.assertEqual(plain_score, 550)
        self.assertEqual(slow_score, 550)
        self.assertGreater(fast_score, 550)
        self.assertLessEqual(fast_score, 730)

    def test_loss_scores_zero(self):
        game = SudokuGame()
        room = make_room("sudoku", {
            "won": False, "mistakes": 3, "hints_used": 0,
            "difficulty": "normal", "timed": False,
            "start": time.time()})
        self.assertEqual(game.score(room, ADA), 0)

    def test_final_message_shows_time(self):
        game = SudokuGame()
        room = make_room("sudoku", {
            "won": True, "mistakes": 0, "hints_used": 0,
            "difficulty": "hard", "timed": True,
            "start": time.time() - 125})
        msg = game.final_message(room, None)
        self.assertIn("2:05", msg)
        self.assertIn("hard", msg)

    def test_setup_shows_lock_note(self):
        engine, _db, _sent = make_engine()
        try:
            room, _ = engine.start("t:s1", "sudoku", ADA,
                                   difficulty="expert")
            text = engine.games["sudoku"].setup(room, engine._mind)
            self.assertIn("🔒", text)
        finally:
            engine.shutdown()


# ── gomoku: board variants ───────────────────────────────────────────────────

class GomokuVariantTests(unittest.TestCase):
    def test_parse_sized_boards(self):
        self.assertEqual(_gomoku_parse("h8", 15), (7, 7))
        self.assertEqual(_gomoku_parse("s19", 19), (18, 18))
        self.assertEqual(_gomoku_parse("a1", 13), (0, 0))
        self.assertIsNone(_gomoku_parse("t19", 19))   # beyond s
        self.assertIsNone(_gomoku_parse("p16", 15))   # beyond o
        self.assertIsNone(_gomoku_parse("a0", 15))
        self.assertEqual(_gomoku_parse("h8", 9), (7, 7))  # a–i on 9×9
        self.assertIsNone(_gomoku_parse("j8", 9))  # j beyond i
        self.assertEqual(_gomoku_parse("H8", 15), (7, 7))  # case-insensitive

    def test_big_gated_without_mastery(self):
        game = GomokuGame()
        s = game.new_state(random.Random(1), variant="big", mastery=0)
        self.assertEqual(s["size"], 15)
        self.assertEqual(s["locked"], "big")

    def test_big_open_with_mastery(self):
        game = GomokuGame()
        s = game.new_state(random.Random(1), variant="big", mastery=2)
        self.assertEqual(s["size"], 13)
        self.assertEqual(len(s["grid"]), 13)
        self.assertEqual(s["locked"], "")

    def test_huge_gate(self):
        game = GomokuGame()
        s = game.new_state(random.Random(1), variant="huge", mastery=3)
        self.assertEqual(s["size"], 15)
        self.assertEqual(s["locked"], "huge")
        s = game.new_state(random.Random(1), variant="huge", mastery=4)
        self.assertEqual(s["size"], 19)
        self.assertEqual(s["locked"], "")

    def test_house_plays_on_small_board(self):
        grid = [[""] * 13 for _ in range(13)]
        grid[6][6] = "B"
        mv = _gomoku_house_move(grid, random.Random(5), "normal")
        self.assertIsNotNone(mv)
        r, c = mv
        self.assertTrue(0 <= r < 13 and 0 <= c < 13)

    def test_score_scales_with_board(self):
        game = GomokuGame()
        big = make_room("gomoku", {"grid": [[""] * 19 for _ in range(19)],
                                   "winner": "B", "moves": 50})
        small = make_room("gomoku", {"grid": [[""] * 15 for _ in range(15)],
                                     "winner": "B", "moves": 50})
        self.assertEqual(game.score(big, ADA), 19 * 19 - 50)
        self.assertEqual(game.score(small, ADA), 15 * 15 - 50)
        draw = make_room("gomoku", {"grid": [[""] * 15 for _ in range(15)],
                                    "winner": "draw", "moves": 225})
        self.assertEqual(game.score(draw, ADA), 10)

    def test_engine_start_variant_plumbing(self):
        engine, db, _sent = make_engine()
        try:
            # fresh player: big is locked → 15×15
            room, _ = engine.start("t:g1", "gomoku", ADA, variant="big")
            self.assertEqual(room.state["size"], 15)
            self.assertEqual(room.state["locked"], "big")
            engine.quit("t:g1")
            # tier 2+ (Adept): 20 wins → 13×13
            for _ in range(20):
                update_game_stats(db, ADA.key, "gomoku", True, 100)
            room, _ = engine.start("t:g2", "gomoku", ADA, variant="big")
            self.assertEqual(room.state["size"], 13)
            self.assertEqual(room.state["locked"], "")
        finally:
            engine.shutdown()


# ── 2048: marathon board, reaching 2048 wins ─────────────────────────────────

class TwentyFortyEightMasteryTests(unittest.TestCase):
    def test_big_gated_without_mastery(self):
        game = TwentyFortyEightGame()
        s = game.new_state(random.Random(1), variant="big", mastery=0)
        self.assertEqual(s["size"], 4)
        self.assertEqual(s["locked"], "big")

    def test_big_open_with_mastery(self):
        game = TwentyFortyEightGame()
        s = game.new_state(random.Random(1), variant="big", mastery=3)
        self.assertEqual(s["size"], 5)
        self.assertEqual(len(s["grid"]), 5)
        self.assertEqual(len(s["grid"][0]), 5)

    def test_5x5_moves_and_renders(self):
        game = TwentyFortyEightGame()
        grid = [[0] * 5 for _ in range(5)]
        grid[0][0] = 2
        grid[0][1] = 2
        moved, pts = game._move(grid, "l")
        self.assertEqual(moved[0][0], 4)
        self.assertEqual(pts, 4)
        self.assertTrue(game._can_move(grid))
        text = game._render(grid, 4)
        self.assertIn("score: 4", text)

    def test_reaching_2048_is_a_win(self):
        game = TwentyFortyEightGame()
        room = make_room("2048", {"grid": [[2048, 0, 0, 0]] + [[0] * 4] * 3,
                                  "score": 20000, "won": True,
                                  "lost": True, "size": 4})
        w = game.winner(room)
        self.assertIsNotNone(w)
        self.assertEqual(w.key, ADA.key)

    def test_board_full_without_2048_is_not_a_win(self):
        game = TwentyFortyEightGame()
        grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        room = make_room("2048", {"grid": grid, "score": 100,
                                  "won": False, "lost": True, "size": 4})
        self.assertIsNone(game.winner(room))

    def test_final_message(self):
        game = TwentyFortyEightGame()
        room = make_room("2048", {"grid": [[2048, 0, 0, 0]] + [[0] * 4] * 3,
                                  "score": 20000, "won": True,
                                  "lost": True, "size": 5,
                                  "start": time.time() - 61})
        msg = game.final_message(room, None)
        self.assertIn("2048 reached", msg)
        self.assertIn("5×5", msg)


# ── hangman: score + long words ──────────────────────────────────────────────

class HangmanMasteryTests(unittest.TestCase):
    def test_score_win(self):
        game = HangmanGame()
        room = make_room("hangman", {"word": "dragon", "done": "table",
                                     "wrong": 2, "max_wrong": 6,
                                     "daily": False, "variant": ""})
        # 6 letters ×10 + (6-2)×10 = 100
        self.assertEqual(game.score(room, ADA), 100)

    def test_score_perfect_and_bonuses(self):
        game = HangmanGame()
        room = make_room("hangman", {"word": "dragon", "done": "table",
                                     "wrong": 0, "max_wrong": 6,
                                     "daily": True, "variant": "long"})
        # 60 + 60 + 25 perfect + 25 daily + 30 long = 200
        self.assertEqual(game.score(room, ADA), 200)

    def test_score_loss_is_zero(self):
        game = HangmanGame()
        room = make_room("hangman", {"word": "dragon", "done": "house",
                                     "wrong": 6, "max_wrong": 6,
                                     "daily": False, "variant": ""})
        self.assertEqual(game.score(room, ADA), 0)

    def test_long_words_gated(self):
        game = HangmanGame()
        s = game.new_state(random.Random(7), variant="long", mastery=0)
        self.assertEqual(s["locked"], "long")
        self.assertLess(len(s["word"]), 9)
        words = {game.new_state(random.Random(i), variant="long",
                                mastery=2)["word"]
                 for i in range(10)}
        self.assertTrue(all(len(w) >= 9 for w in words),
                        f"short words leaked: {words}")


# ── trivia: sudden death ────────────────────────────────────────────────────

class TriviaMasteryTests(unittest.TestCase):
    def test_sudden_gated(self):
        game = TriviaRoyaleGame()
        s = game.new_state(random.Random(3), variant="sudden", mastery=0)
        self.assertEqual(s["start_lives"], 3)
        self.assertEqual(s["locked"], "sudden")
        s = game.new_state(random.Random(3), variant="sudden", mastery=2)
        self.assertEqual(s["start_lives"], 1)
        self.assertEqual(s["locked"], "")

    def test_sudden_score_doubles(self):
        game = TriviaRoyaleGame()
        room = make_room("trivia", {"points": {ADA.key: 50},
                                    "variant": "sudden"})
        self.assertEqual(game.score(room, ADA), 100)
        room = make_room("trivia", {"points": {ADA.key: 50},
                                    "variant": ""})
        self.assertEqual(game.score(room, ADA), 50)

    def test_setup_seats_sudden_lives(self):
        engine, db, _sent = make_engine()
        try:
            for _ in range(20):
                update_game_stats(db, ADA.key, "trivia", True, 100)
            room, _ = engine.start("t:tr1", "trivia", ADA,
                                   variant="sudden")
            self.assertEqual(room.state["lives"][ADA.key], 1)
        finally:
            engine.shutdown()


# ── ttt: meaningful score ────────────────────────────────────────────────────

class TttScoreTests(unittest.TestCase):
    def test_easy_win_pays_with_speed(self):
        game = TicTacToeGame()
        room = make_room("ttt", {"winner": "X", "moves": 5,
                                 "difficulty": "easy"})
        self.assertEqual(game.score(room, ADA), 25 + 8)

    def test_perfect_house_win_is_legendary(self):
        game = TicTacToeGame()
        room = make_room("ttt", {"winner": "X", "moves": 9,
                                 "difficulty": "expert"})
        self.assertEqual(game.score(room, ADA), 100)

    def test_draw_pays(self):
        game = TicTacToeGame()
        room = make_room("ttt", {"winner": "draw", "moves": 9,
                                 "difficulty": "normal"})
        self.assertEqual(game.score(room, ADA), 15)

    def test_loss_pays_nothing(self):
        game = TicTacToeGame()
        room = make_room("ttt", {"winner": "O", "moves": 8,
                                 "difficulty": "easy"})
        self.assertEqual(game.score(room, ADA), 0)


# ── wordchain / numberguess: scores where there were none ────────────────────

class NewScoreTests(unittest.TestCase):
    def test_wordchain_scores_words_chained(self):
        game = WordChainGame()
        room = make_room("wordchain", {"words_by": {ADA.key: 7}})
        self.assertEqual(game.score(room, ADA), 70)
        room = make_room("wordchain", {})
        self.assertEqual(game.score(room, ADA), 0)

    def test_numberguess_fast_crack_pays(self):
        game = NumberGuessGame()
        room = make_room("numberguess", {"done": "player", "rounds": 4})
        self.assertEqual(game.score(room, ADA), 110)
        room = make_room("numberguess", {"done": "player", "rounds": 30})
        self.assertEqual(game.score(room, ADA), 10)
        room = make_room("numberguess", {"done": "house", "rounds": 4})
        self.assertEqual(game.score(room, ADA), 0)


# ── engine: tier-up fanfare on finish ────────────────────────────────────────

class EngineMasteryTests(unittest.TestCase):
    def test_tier_up_fanfare_on_finish(self):
        engine, db, _sent = make_engine()
        try:
            # one ttt win: 100 + 10 + 25 = 135 pts → tier 0
            update_game_stats(db, ADA.key, "ttt", True, 30)
            room, _ = engine.start("t:f1", "ttt", ADA, difficulty="easy")
            room.state["winner"] = "X"
            room.state["over"] = True
            room.state["moves"] = 5
            msgs = engine._finish(room)
            text = "\n".join(msgs)
            # second win: 200 + 20 + 27 = 247 → tier 1 → fanfare
            self.assertIn("mastery up", text)
            self.assertIn("Novice → Player", text)
        finally:
            engine.shutdown()

    def test_no_fanfare_without_tier_change(self):
        engine, db, _sent = make_engine()
        try:
            room, _ = engine.start("t:f2", "ttt", ADA, difficulty="easy")
            room.state["winner"] = "O"  # house wins
            room.state["over"] = True
            room.state["moves"] = 6
            msgs = engine._finish(room)
            self.assertNotIn("mastery up", "\n".join(msgs))
        finally:
            engine.shutdown()

    def test_mastery_line_at_start_uses_history(self):
        engine, db, _sent = make_engine()
        try:
            for _ in range(10):
                update_game_stats(db, ADA.key, "sudoku", True, 700)
            line = mastery_line(
                "sudoku", get_one_game_stats(db, ADA.key, "sudoku"))
            # 10 wins: 1000 + 100 played + 50 score-share = 1150 → Expert
            self.assertIn("Expert", line)
        finally:
            engine.shutdown()


# ── /mastery command ─────────────────────────────────────────────────────────

class MasteryCommandTests(unittest.TestCase):
    def test_command_registered(self):
        self.assertIn("mastery", GAME_COMMANDS)
        self.assertIn("mastery", CONTROL_COMMANDS)

    def test_parse_control_routes(self):
        cmd = parse_control("/mastery")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "mastery")
        cmd = parse_control("/mastery sudoku")
        self.assertEqual(cmd.kind, "mastery")
        self.assertEqual(cmd.tail, "sudoku")


if __name__ == "__main__":
    unittest.main()
