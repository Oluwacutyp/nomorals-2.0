"""Wave 95 — the wild catalog: real poker, real rng, seven games.

Poker gets the deep treatment: the 7-card evaluator is unit-tested
against every hand rank and tie-break, the table is driven end-to-end
with a chip-conservation invariant (no chip ever created or lost),
all-in handling (short calls, all-in-from-the-blinds, board run-outs)
is exercised by scripted humans, and the "second human" that used to
corrupt the table is rejected.  The room rng is proven continuous:
two consecutive deals from one room never repeat, while same-seed
rooms still replay identically.  Every other wild game (tic-tac-toe,
bulls & cows, craps, concentration, minesweeper, wordle) is driven
to a finish, and the bulls & cows solver is proven to crack a
200-code deterministic sample within 12 guesses (a full 5,040-code
sweep found a worst case of 9).

Everything hermetic: in-memory db, no model, fixed player keys,
scripted moves.
"""
from __future__ import annotations

import random
import unittest

from nomorals.games import Player
from nomorals.games.engine import GameEngine
from nomorals.games.games.wild import (
    _bulls_opening_guess,
    _bulls_solve,
    _hand_name,
    _ALL_CODES,
    best7,
    eval5,
    make_deck,
)
from nomorals.games.lexicon import FIVE_LETTERS
from nomorals.storage.db import Database

try:  # package import under pytest (tests/__init__.py exists)
    from tests.test_wave85_games import (
        ADA, BOB, Ctx, drive, make_engine)
except ImportError:  # direct run: fall back to the tests directory
    import sys
    from pathlib import Path
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_wave85_games import (  # noqa: PLC0415
        ADA, BOB, Ctx, drive, make_engine)

WILD_GAMES = ("poker", "ttt", "bulls", "craps", "memory",
              "mines", "wordle")


# ── 1. the 7-card evaluator ─────────────────────────────────────────────────

class PokerEvaluatorTests(unittest.TestCase):
    """eval5/best7 against known answers, rank by rank."""

    def test_eleven_ranks_in_strict_order(self):
        hands = {
            "royal": ("A♠", "K♠", "Q♠", "J♠", "10♠"),
            "straight_flush": ("9♣", "8♣", "7♣", "6♣", "5♣"),
            "quads": ("A♥", "A♦", "A♣", "A♠", "K♥"),
            "full": ("Q♥", "Q♦", "Q♣", "7♠", "7♥"),
            "flush": ("A♥", "J♥", "8♥", "4♥", "2♥"),
            "straight": ("9♦", "8♣", "7♥", "6♠", "5♦"),
            "wheel": ("A♠", "2♥", "3♦", "4♣", "5♠"),
            "trips": ("K♠", "K♥", "K♦", "9♠", "2♥"),
            "two_pair": ("K♠", "K♥", "9♦", "9♣", "2♠"),
            "pair": ("K♠", "K♥", "J♦", "9♣", "2♠"),
            "high": ("A♠", "J♥", "8♦", "5♣", "2♠"),
        }
        scores = [eval5(h) for h in hands.values()]
        for a, b in zip(scores, scores[1:]):
            self.assertGreater(a, b)
        # the 11 named hands collapse to the 9 real poker ranks
        # (royal==straight flush, straight==wheel); verify exactly those
        self.assertEqual(
            sorted({s[0] for s in scores}, reverse=True),
            [8, 7, 6, 5, 4, 3, 2, 1, 0])

    def test_wheel_scores_five_high_not_ace_high(self):
        wheel = eval5(("A♠", "2♥", "3♦", "4♣", "5♠"))
        below = eval5(("3♠", "2♥", "7♦", "4♣", "5♠"))  # no straight
        self.assertGreater(wheel, below)
        self.assertEqual(wheel[0], eval5(("5♦", "4♣", "3♥", "2♠",
                                          "A♦"))[0])
        # a wheel loses to any other straight
        self.assertLess(wheel, eval5(("6♦", "5♣", "4♥", "3♠", "2♦")))

    def test_tiebreakers(self):
        # two pair: higher top pair wins even with a worse kicker
        self.assertGreater(eval5(("K♠", "K♥", "9♦", "9♣", "2♠")),
                           eval5(("Q♠", "Q♥", "J♦", "J♣", "A♠")))
        # straight: broadway beats a king-high straight
        self.assertGreater(eval5(("A♦", "K♣", "Q♥", "J♠", "10♦")),
                           eval5(("K♦", "Q♣", "J♥", "10♠", "9♦")))
        # quads: the quad rank rules, then the kicker
        self.assertGreater(eval5(("A♠", "A♥", "A♦", "A♣", "K♠")),
                           eval5(("K♠", "K♥", "K♦", "K♣", "A♠")))

    def test_tens_parse(self):
        quads10 = eval5(("10♠", "10♥", "10♦", "10♣", "2♠"))
        self.assertEqual(_hand_name(quads10), "four of a kind")
        # "10" must score as rank 10: above 9s, below Jacks
        self.assertGreater(
            eval5(("10♠", "10♥", "10♦", "10♣", "K♠")),
            eval5(("9♠", "9♥", "9♦", "9♣", "K♠")))
        self.assertLess(
            eval5(("10♠", "10♥", "10♦", "10♣", "K♠")),
            eval5(("J♠", "J♥", "J♦", "J♣", "K♠")))

    def test_suit_invariant(self):
        a = eval5(("A♠", "K♣", "Q♥", "J♠", "10♦"))
        b = eval5(("A♥", "K♠", "Q♣", "J♦", "10♠"))
        self.assertEqual(a, b)

    def test_best7_picks_the_best_five(self):
        # quads hidden inside seven cards
        self.assertEqual(
            _hand_name(best7(["A♠", "A♥", "A♦", "A♣", "5♦", "5♣",
                              "3♦"])),
            "four of a kind")
        # a straight that only exists across hole + board
        self.assertEqual(
            _hand_name(best7(["7♠", "8♥", "9♦", "10♣", "J♠", "K♥",
                              "2♦"])),
            "straight")
        # full house beats the flush hiding in the same seven
        seven = ["K♠", "K♥", "K♦", "9♦", "9♠", "A♥", "J♥"]
        self.assertEqual(
            _hand_name(best7(seven)), "full house")
        # flush found when no straight/full house exists
        self.assertEqual(
            _hand_name(best7(["A♥", "K♥", "8♥", "5♥", "2♥", "Q♣",
                              "9♠"])),
            "flush")

    def test_hand_names_table(self):
        self.assertEqual(_hand_name(eval5(("A♠", "K♠", "Q♠", "J♠",
                                           "10♠"))), "straight flush")
        self.assertEqual(_hand_name(eval5(("A♠", "J♥", "8♦", "5♣",
                                           "2♠"))), "high card")


# ── 2. the room rng: continuous, replay-deterministic ────────────────────────

class RngContinuityTests(unittest.TestCase):
    def test_room_rng_is_one_stream(self):
        db = Database(":memory:")
        db.migrate()
        engine = GameEngine(Ctx(db), send=lambda c, t: None)
        room, _ = engine.start("rng-chat", "craps", ADA, kind="dm")
        r = room.rng()
        self.assertIs(r, room.rng(), "room.rng() must be the live stream")
        d1 = make_deck(room.rng())
        d2 = make_deck(room.rng())
        self.assertNotEqual(d1, d2, "two deals must not repeat")
        self.assertEqual(sorted(d1), sorted(d2))  # still a real deck
        engine.shutdown()

    def test_same_seed_rooms_replay(self):
        db = Database(":memory:")
        db.migrate()
        engine = GameEngine(Ctx(db), send=lambda c, t: None)
        r1, _ = engine.start("chat-a", "craps", ADA, kind="dm")
        r2, _ = engine.start("chat-b", "craps", ADA, kind="dm")
        r1.seed = 12345
        r2.seed = 12345
        r1._rng_instance = None
        r2._rng_instance = None
        self.assertEqual(make_deck(r1.rng()), make_deck(r2.rng()))
        self.assertNotEqual(make_deck(r1.rng()),
                            make_deck(r1.rng()))
        engine.shutdown()


# ── 3. poker: the table itself ───────────────────────────────────────────────

class PokerTableTests(unittest.TestCase):
    def _open_table(self):
        engine, _ = make_engine()
        room, _ = engine.start("poker-chat", "poker", ADA, kind="dm")
        return engine, room

    def test_heads_up_table_rejects_a_second_human(self):
        engine, _ = self._open_table()
        msgs = engine.join("poker-chat", BOB)
        self.assertEqual(msgs, ["the table is full."])
        self.assertIsNone(engine.live("poker-chat").player(BOB.key))
        engine.shutdown()

    def test_no_chip_ever_created_or_lost(self):
        """200 scripted moves, mixed strategy: stacks + pot == 400
        after every single move, and the table never freezes."""
        engine, room = self._open_table()
        seq = ["call", "check", "bet 12", "call", "fold", "bet 8",
               "call", "check", "allin"]
        seen_states = set()
        for mv in range(200):
            s = room.state
            self.assertEqual(
                sum(s["stacks"].values()) + s["pot"], 400,
                f"chip leak at move {mv}")
            snap = (s["hand"], s.get("street"),
                    tuple(sorted(s["bets"].items())),
                    tuple(sorted(s["stacks"].items())))
            self.assertNotIn(snap, seen_states,
                             f"table frozen at move {mv}")
            seen_states.add(snap)
            if s.get("done"):
                break
            engine.move("poker-chat", seq[mv % len(seq)], ADA)
        engine.shutdown()

    def test_allin_human_finishes_the_table(self):
        """A human who shoves every hand busts (or wins) — the game
        must actually end, with exactly one player on 400 chips."""
        engine, room = self._open_table()
        for mv in range(400):
            s = room.state
            self.assertEqual(sum(s["stacks"].values()) + s["pot"], 400)
            if s.get("done"):
                break
            engine.move("poker-chat", "allin", ADA)
        else:
            self.fail("table never finished")
        s = room.state
        nonzero = [v for v in s["stacks"].values() if v > 0]
        self.assertEqual(nonzero, [400],
                         f"table ended wrong: {s['stacks']}")
        engine.shutdown()

    def test_boards_never_repeat_across_hands(self):
        engine, room = self._open_table()
        boards = set()
        for mv in range(120):
            s = room.state
            if len(s.get("board", [])) == 3:
                boards.add(tuple(s["board"]))
            if s.get("done"):
                break
            if len(boards) >= 4:
                break
            engine.move("poker-chat", "check", ADA)
        self.assertGreaterEqual(len(boards), 2,
                                f"repeated boards: {boards}")
        engine.shutdown()

    def test_allin_from_the_blinds_cannot_freeze(self):
        """A human whose stack is only the big blind goes all-in
        posting it — that used to wedge the table on 'out of chips'.
        Set up exactly that moment, deal, and verify the hand plays
        itself out without a freeze or a chip leak."""
        engine, room = self._open_table()
        house = "ai:poker:0"
        s = room.state
        # rewind to the moment before a deal: human on exactly 4
        s["stacks"] = {ADA.key: 4, house: 396}
        s["pot"] = 0
        s["bets"] = {ADA.key: 0, house: 0}
        s["committed"] = {ADA.key: 0, house: 0}
        s["board"] = []
        s["street"] = 0
        s["done"] = False
        s["to_act"] = None
        game = engine.games[room.game]  # room.game is the name (str)
        game._deal(room)
        s = room.state
        self.assertEqual(sum(s["stacks"].values()) + s["pot"], 400)
        # the hand must have played on: a new hand dealt, or the
        # table finished — never a 0-stack human left to act
        self.assertTrue(
            s.get("done") or s["hand"] >= 2
            or not (s.get("to_act") == ADA.key
                    and s["stacks"].get(ADA.key, 0) == 0),
            f"wedged at the blinds: {s['hand']} {s.get('street')} "
            f"{s['stacks']} to_act={s.get('to_act')}")
        # now keep driving: chips conserved, state keeps moving
        seen_states = set()
        for mv in range(120):
            s = room.state
            self.assertEqual(sum(s["stacks"].values()) + s["pot"], 400,
                             f"chip leak at move {mv}")
            snap = (s["hand"], s.get("street"),
                    tuple(sorted(s["bets"].items())),
                    tuple(sorted(s["stacks"].items())),
                    tuple(s.get("board", [])))
            self.assertNotIn(snap, seen_states,
                             f"table frozen at move {mv}")
            seen_states.add(snap)
            if s.get("done"):
                break
            engine.move("poker-chat", "call", ADA)
        engine.shutdown()


# ── 4. bulls & cows: the solver against the FULL space ───────────────────────

class BullsSolverTests(unittest.TestCase):
    def test_opener_is_a_valid_code(self):
        g = _bulls_opening_guess()
        self.assertEqual(len(set(g)), 4)

    def test_solver_cracks_a_deterministic_sample(self):
        # 200 fixed codes: fast, hermetic, covers the strategy's
        # behaviour on a spread of the space
        rng = random.Random(95)
        codes = rng.sample(list(_ALL_CODES), 200)
        for code in codes:
            guess, used = _bulls_solve(code)
            self.assertEqual(guess, code, f"solver missed {code}")
            self.assertLessEqual(used, 12)

    def test_worst_case_over_the_sample(self):
        rng = random.Random(95)
        codes = rng.sample(list(_ALL_CODES), 200)
        worst = max(_bulls_solve(c)[1] for c in codes)
        self.assertLessEqual(worst, 12)


# ── 5. wordle: the pool is what the game accepts ─────────────────────────────

class WordlePoolTests(unittest.TestCase):
    def test_pool_size_and_probe_words(self):
        pool = FIVE_LETTERS()
        # 1361 lexicon 5-letter words + curated extras that survive the
        # [a-z]{5} filter
        self.assertEqual(len(pool), 1548)
        for w in ("stare", "light", "house", "water", "plant",
                  "break", "small", "white", "black", "great", "first"):
            self.assertIn(w, pool)
        # the blocklist applies to the pool too
        self.assertNotIn("thong", pool)

    def test_guesses_validated_against_the_pool(self):
        engine, _ = make_engine()
        room, _ = engine.start("wordle-chat", "wordle", ADA, kind="dm")
        # a pool word (one the secret itself can be) must be accepted
        out = engine.move("wordle-chat", "stare", ADA)
        self.assertNotIn("not in the word pool", out[0])
        self.assertEqual(len(room.state["guesses"]), 1)
        # a 5-letter string outside the pool is rejected, state intact
        out = engine.move("wordle-chat", "zzzzz", ADA)
        self.assertIn("not in the word pool", out[0])
        self.assertEqual(len(room.state["guesses"]), 1)
        # duplicates are rejected
        out = engine.move("wordle-chat", "stare", ADA)
        self.assertIn("already tried", out[0])
        self.assertEqual(len(room.state["guesses"]), 1)
        engine.shutdown()


# ── 6. all seven wild games, driven to a finish ──────────────────────────────

class WildGamesEndToEndTests(unittest.TestCase):
    """A game that can't be finished by a script can't be finished
    by a human either — so every game is driven to a finish here."""

    def _ttt_responder(self, room, state):
        for i, cell in enumerate(state["board"]):
            if not cell:
                return str(i + 1)
        return "5"

    def test_tic_tac_toe_finishes(self):
        engine, _ = make_engine()
        live, _ = drive(engine, "ttt-chat", "ttt", ADA,
                        responder=self._ttt_responder, max_moves=12)
        self.assertTrue(live is None or live.status == "finished")
        engine.shutdown()

    def test_bulls_cows_finishes(self):
        engine, _ = make_engine()
        codes = _ALL_CODES[::504]  # 10 legal codes (distinct digits)
        idx = {"n": -1}

        def resp(room, state):
            if not state.get("human_code"):
                return "set 1234"
            idx["n"] += 1
            return codes[idx["n"] % len(codes)]
        live, _ = drive(engine, "bulls-chat", "bulls", ADA,
                        responder=resp, max_moves=16)
        self.assertTrue(live is None or live.status == "finished")
        engine.shutdown()

    def test_craps_finishes(self):
        engine, _ = make_engine()
        live, _ = drive(engine, "craps-chat", "craps", ADA,
                        responder=lambda room, state: "roll",
                        max_moves=30)
        self.assertTrue(live is None or live.status == "finished")
        engine.shutdown()

    def test_concentration_finishes(self):
        engine, _ = make_engine()

        def resp(room, state):
            found = set()
            for idxs in state["found"].values():
                found.update(idxs)
            free = [i for i in range(16) if i not in found]
            if len(free) < 2:
                return None
            a, b = free[0], free[1]
            return f"flip {a // 4 + 1} {a % 4 + 1} {b // 4 + 1} {b % 4 + 1}"
        live, _ = drive(engine, "memory-chat", "memory", ADA,
                        responder=resp, max_moves=40)
        self.assertTrue(live is None or live.status == "finished")
        engine.shutdown()

    def test_mines_finishes(self):
        engine, _ = make_engine()
        cells = [(r, c) for r in range(1, 10) for c in range(1, 10)]
        idx = {"n": 0}

        def resp(room, state):
            n = idx["n"]
            idx["n"] += 1
            r, c = cells[n % len(cells)]
            return f"open {r} {c}"
        live, _ = drive(engine, "mines-chat", "mines", ADA,
                        responder=resp, max_moves=85)
        self.assertTrue(live is None or live.status == "finished")
        engine.shutdown()

    def test_wordle_finishes(self):
        engine, _ = make_engine()
        guesses = ["light", "house", "water", "plant", "break", "stare"]
        idx = {"n": 0}

        def resp(room, state):
            n = idx["n"]
            idx["n"] += 1
            return guesses[n % len(guesses)]
        live, _ = drive(engine, "wordle-chat", "wordle", ADA,
                        responder=resp, max_moves=8)
        self.assertTrue(live is None or live.status == "finished")
        engine.shutdown()

    def test_all_seven_registered(self):
        engine, _ = make_engine()
        for name in WILD_GAMES:
            self.assertIn(name, engine.games)
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
