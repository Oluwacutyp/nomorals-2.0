"""House-winner settle contract: when the house beats the human, the
human LOSES — coins, points and record all say so.

The defect (found by the wager-integrity build, pre-existing): rps,
digits and 20q returned the bare string ``"house"`` from ``winner()``.
The engine's winner contract (games/base.py) understands exactly a
``Player``, ``"draw"``, ``"all"`` or ``None`` — every other game in the
catalog returns the house seat's Player when the house wins (same file:
hangman, wordchain, numberguess, spy construct ``Player("ai:<game>",
…, "The House", is_ai=True)``). The stray string fell through
``_settle_won`` to its default and settled as a DRAW: a house win paid
the human +25 draw coins and recorded a draw instead of +15
participation coins and a loss.

The fix lives in the three games (they now return the house Player,
like their siblings); the engine additionally journals any future
out-of-contract winner string instead of silently drawing. This file
pins the behavior and the contract itself — the census tests walk the
whole registered catalog so the next game that invents a winner value
fails here, at build time, instead of lying at settle time.
"""
from __future__ import annotations

import ast
import random
import unittest
from pathlib import Path

import nomorals.games.games as games_pkg
from nomorals.games.engine import GameEngine
from nomorals.games.games.base import Room
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


def coin_line(out: list[str]) -> str:
    lines = [m for m in out if "🪙" in m]
    assert lines, f"no coin line in settle: {out!r}"
    return lines[0]


def solve_seed(want: str) -> int:
    """A room seed whose first rng pick is ``want`` (rps house throw)."""
    return next(s for s in range(100_000)
                if random.Random(s).choice(
                    ("rock", "paper", "scissors")) == want)


class HouseWinSettlesAsLossTests(unittest.TestCase):
    """Full engine drives: the house wins rps / digits / 20q and the
    settle must read as a loss everywhere — coin line, points, record,
    and the card's winner label."""

    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def assert_loss_settle(self, out, game):
        line = coin_line(out)
        self.assertIn("+15 coins (participation)", line)
        self.assertNotIn("(draw)", line)
        self.assertNotIn("+25 coins", line)
        # the card names the winner truthfully — "The House", never a
        # raw internal string.
        self.assertTrue(any("The House wins" in m for m in out), out)
        self.assertTrue(any("👑 The House" in m for m in out), out)
        prof = self.engine.store.get_for(ADA)
        self.assertEqual(prof.wins, 0)
        self.assertEqual(prof.losses, 1)
        self.assertEqual(prof.draws, 0)
        self.assertEqual(prof.points, 10)   # loss crumb, not draw 20
        self.assertEqual(prof.coins, 15)

    def test_rps_house_win_is_a_loss(self):
        room, _ = self.engine.start("t:rps", "rps", ADA, kind="dm")
        room.state["me"] = 2  # house one round from taking it
        room._rng_instance = random.Random(solve_seed("paper"))
        out = self.engine.move("t:rps", "rock", ADA, kind="dm")
        self.assertIsNone(self.engine.live("t:rps"))
        self.assert_loss_settle(out, "rps")

    def test_digits_house_win_is_a_loss(self):
        room, _ = self.engine.start("t:dig", "digits", ADA, kind="dm")
        wrong = str(room.state["number"]) + "9"  # guaranteed wrong
        out = self.engine.move("t:dig", wrong, ADA, kind="dm")
        self.assertIsNone(self.engine.live("t:dig"))
        self.assert_loss_settle(out, "digits")

    def test_20q_house_win_is_a_loss(self):
        room, _ = self.engine.start("t:20q", "20q", ADA, kind="dm")
        room.state["asked"] = 19
        out = self.engine.move("t:20q", "is it blue?", ADA, kind="dm")
        self.assertIsNone(self.engine.live("t:20q"))
        self.assert_loss_settle(out, "20q")


class UnchangedOutcomesTests(unittest.TestCase):
    """The fix must not move the honest outcomes: human wins still pay
    as wins, true draws still pay as draws."""

    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_rps_human_win_still_a_win(self):
        room, _ = self.engine.start("t:rpsw", "rps", ADA, kind="dm")
        room.state["you"] = 2
        room._rng_instance = random.Random(solve_seed("scissors"))
        out = self.engine.move("t:rpsw", "rock", ADA, kind="dm")
        self.assertIsNone(self.engine.live("t:rpsw"))
        self.assertIn("win 40", coin_line(out))
        prof = self.engine.store.get_for(ADA)
        self.assertEqual((prof.wins, prof.losses, prof.draws), (1, 0, 0))

    def test_20q_human_win_still_a_win(self):
        room, _ = self.engine.start("t:20qw", "20q", ADA, kind="dm")
        target = room.state["target"]
        out = self.engine.move("t:20qw", f"guess: {target}", ADA,
                               kind="dm")
        self.assertIsNone(self.engine.live("t:20qw"))
        self.assertIn("win 40", coin_line(out))
        prof = self.engine.store.get_for(ADA)
        self.assertEqual((prof.wins, prof.losses, prof.draws), (1, 0, 0))

    def test_wyrr_true_draw_still_a_draw(self):
        self.engine.start("t:wy", "wyrr", ADA, kind="dm")
        out: list[str] = []
        for _ in range(15):
            if self.engine.live("t:wy") is None:
                break
            out = self.engine.move("t:wy", "1", ADA, kind="dm")
        self.assertIsNone(self.engine.live("t:wy"))
        self.assertIn("+25 coins (draw)", coin_line(out))
        prof = self.engine.store.get_for(ADA)
        self.assertEqual((prof.wins, prof.losses, prof.draws), (0, 0, 1))
        self.assertEqual(prof.coins, 25)


class WinnerValueUnitTests(unittest.TestCase):
    """The three games' winner() values, and _settle_won's reading of
    every contract shape."""

    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _room(self, game_name, state):
        return Room(id="u", game=game_name, chat_key="local:u",
                    platform="local", kind="dm", players=[ADA],
                    state=state, status="finished")

    def test_house_state_returns_house_player(self):
        for name in ("rps", "digits", "20q"):
            game = self.engine.games[name]
            w = game.winner(self._room(name, {"done": "house"}))
            self.assertIsInstance(w, Player, name)
            self.assertTrue(w.is_ai, name)
            self.assertEqual(w.name, "The House", name)

    def test_table_state_still_returns_the_human(self):
        for name in ("rps", "digits", "20q"):
            game = self.engine.games[name]
            w = game.winner(self._room(name, {"done": "table"}))
            self.assertIs(w, ADA, name)

    def test_unfinished_returns_none(self):
        for name in ("rps", "digits", "20q"):
            game = self.engine.games[name]
            self.assertIsNone(game.winner(self._room(name, {})), name)

    def test_settle_won_reads_every_contract_shape(self):
        game = self.engine.games["rps"]
        room = self._room("rps", {"done": "house"})
        house = Player(key="ai:rps", platform="ai", name="The House",
                       is_ai=True)
        other = Player.from_sender("telegram", "222", "Bob")
        settle = GameEngine._settle_won
        self.assertIs(settle(game, room, ADA, house), False)
        self.assertIs(settle(game, room, ADA, other), False)
        self.assertIs(settle(game, room, ADA, ADA), True)
        self.assertIsNone(settle(game, room, ADA, "draw"))
        self.assertIs(settle(game, room, ADA, "all"), True)
        self.assertIsNone(settle(game, room, ADA, None))


def _returned_string_leaves(expr) -> list[str]:
    """String constants a return expression can evaluate to, looking
    only through conditionals/boolean ops — strings nested inside a
    call (e.g. a Player's key/name) are not return values."""
    if isinstance(expr, ast.Constant):
        return [expr.value] if isinstance(expr.value, str) else []
    if isinstance(expr, ast.IfExp):
        return (_returned_string_leaves(expr.body)
                + _returned_string_leaves(expr.orelse))
    if isinstance(expr, ast.BoolOp):
        out: list[str] = []
        for v in expr.values:
            out += _returned_string_leaves(v)
        return out
    return []


class WinnerContractCensusTests(unittest.TestCase):
    """Every game's winner() outputs belong to the understood set:
    a Player, "draw", "all", or None. A new winner value must fail
    HERE, at build time — never settle wrong in a live room."""

    UNDERSTOOD_STRINGS = {"draw", "all"}

    def test_no_winner_returns_an_invented_string(self):
        games_dir = Path(games_pkg.__file__).parent
        seen = 0
        offenders = []
        for path in sorted(games_dir.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for stmt in node.body:
                    if not (isinstance(stmt, ast.FunctionDef)
                            and stmt.name == "winner"):
                        continue
                    seen += 1
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Return) and sub.value:
                            for s in _returned_string_leaves(sub.value):
                                if s not in self.UNDERSTOOD_STRINGS:
                                    offenders.append(
                                        f"{path.name}:{node.name} "
                                        f"returns {s!r}")
        # guard against the pin silently scanning nothing
        self.assertGreaterEqual(seen, 40)
        self.assertEqual(offenders, [])

    def test_every_registered_game_winner_is_understood(self):
        engine, db, sent = make_engine()
        try:
            probed = 0
            for name, game in sorted(engine.games.items()):
                room = Room(
                    id="census", game=name, chat_key="local:census",
                    platform="local", kind="dm", players=[ADA],
                    state=game.new_state(random.Random(7)),
                    status="finished")
                game.setup(room, engine._mind)
                w = game.winner(room)
                probed += 1
                self.assertTrue(
                    w is None or w in self.UNDERSTOOD_STRINGS
                    or isinstance(w, Player),
                    f"{name}.winner() returned {w!r} — outside the "
                    f"winner contract (Player | 'draw' | 'all' | None)")
            self.assertEqual(probed, len(engine.games))
            self.assertGreaterEqual(probed, 40)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
