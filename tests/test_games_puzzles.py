"""Anagram & cryptogram: race mechanics, house AI, multiplayer, achievements.

- anagram: round flow, scoring, the house answering on its turn,
  group races with several humans, full-match win + achievements;
- cryptogram: letter guesses (right/wrong), full solves, hint_scroll,
  the house's frequency-analysis play, full-match win + achievements;
- both games drive through engine.start/move — the same path CLI and
  chat use.
"""
from __future__ import annotations

import unittest

from nomorals.games.achievements import get_achievements
from nomorals.games.engine import GameEngine
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
BOB = Player.from_sender("telegram", "789", "Bob")


class AnagramTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_round_flow_and_scoring(self):
        from nomorals.games.ai import GameMind
        room, _ = self.engine.start("t:an1", "anagram", ADA)
        s = room.state
        game = self.engine.games["anagram"]
        self.assertEqual(s["rounds"], 6)
        self.assertEqual(len(s["words"][0]), len(s["scrambles"][0]))
        self.assertNotEqual(s["words"][0], s["scrambles"][0])
        # wrong guess, direct on_move (no house turn): round holds,
        # attempts tick up
        out = game.on_move(room, ADA, "zzzzz", GameMind())
        self.assertEqual(room.state["round"], 0)
        self.assertGreater(room.state["attempts"], 0)
        self.assertEqual(out, ["nope."])
        # right answer through the engine takes the round for Ada —
        # the house may steal *later* rounds on its own turns, so only
        # assert Ada scored and the round advanced
        word = s["words"][0]
        before_round = room.state["round"]
        out = self.engine.move("t:an1", word, ADA)
        self.assertGreater(room.state["round"], before_round)
        self.assertGreater(room.state["scores"].get(ADA.key, 0), 0)
        self.assertGreaterEqual(room.state["round_wins"].get(ADA.key), 1)
        self.assertTrue(any("unscrambles" in m for m in out))

    def test_house_answers_on_its_turn(self):
        # expert house should take a round within a few of its turns
        room, _ = self.engine.start("t:an2", "anagram", ADA,
                                    difficulty="expert")
        house_rounds = 0
        for _ in range(30):
            if room.state.get("over"):
                break
            before = dict(room.state["round_wins"])
            self.engine.move("t:an2", "zzzzz", ADA)
            after = room.state["round_wins"]
            # any round win not by Ada is the house's
            for k, v in after.items():
                if k != ADA.key and v > before.get(k, 0):
                    house_rounds += 1
        self.assertGreater(house_rounds, 0,
                           "expert house never won a round in 30 turns")

    def test_group_race_two_humans(self):
        room, _ = self.engine.start("t:an3", "anagram", ADA, kind="group")
        self.engine.join("t:an3", BOB)
        word = room.state["words"][0]
        # Bob is not the current seat necessarily — force turn order via
        # direct game call is not needed; the engine routes by seat.
        # Just have whoever is current answer; check someone scores.
        cur = room.current
        out = self.engine.move("t:an3", word, cur)
        # the current seat's answer took round 1; the house may have
        # answered round 2 on its own turn right after — both are fine
        self.assertGreaterEqual(room.state["round"], 1)
        total = sum(room.state["scores"].values())
        self.assertGreater(total, 0)

    def test_full_match_win_and_achievements(self):
        room, _ = self.engine.start("t:an4", "anagram", ADA,
                                    difficulty="easy")
        # answer every round correctly on Ada's turns; the easy house
        # rarely steals one — loop until the match closes
        for _ in range(60):
            if room.state.get("over"):
                break
            s = room.state
            cur = room.current
            if cur is not None and not cur.is_ai:
                self.engine.move("t:an4", s["words"][s["round"]], cur)
            else:
                # pump the house by having Ada pass her turn
                self.engine.move("t:an4", "zzzzz", ADA)
        self.assertTrue(room.state.get("over"))
        self.assertIsNone(self.engine.live("t:an4"))
        unlocked = {a["id"] for a in get_achievements(self.db, ADA.key)}
        # Ada may or may not have won every round (easy house can steal),
        # but she should have the win if she took the match
        game = self.engine.games["anagram"]
        if game.winner(room) is not None and \
                getattr(game.winner(room), "key", "") == ADA.key:
            self.assertIn("anagram_win", unlocked)

    def test_difficulty_plumbing(self):
        room, _ = self.engine.start("t:an5", "anagram", ADA,
                                    difficulty="expert")
        self.assertEqual(room.state["difficulty"], "expert")


class CryptogramTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _truth(self, room):
        rnd = room.state["rounds"][room.state["round"]]
        return rnd, {v: k for k, v in rnd["cipher"].items()}

    def test_letter_guess_right_and_wrong(self):
        from nomorals.games.ai import GameMind
        room, _ = self.engine.start("t:cg1", "cryptogram", ADA)
        game = self.engine.games["cryptogram"]
        rnd, truth = self._truth(room)
        ch = next(c for c in rnd["enciphered"] if c.isalpha())
        out = self.engine.move("t:cg1", f"{ch}={truth[ch]}", ADA)
        self.assertIn(ch, room.state["rounds"][0]["revealed"])
        self.assertTrue(any("✅" in m for m in out))
        self.assertEqual(room.state["scores"].get(ADA.key), 5)
        # wrong guess, direct on_move (no house turn to interfere)
        ch2 = next(c for c in rnd["enciphered"]
                   if c.isalpha() and c != ch
                   and c not in room.state["rounds"][0]["revealed"])
        out = game.on_move(room, ADA, f"{ch2}={truth[ch]}", GameMind())
        self.assertTrue(any("❌" in m for m in out))
        self.assertNotIn(ch2, room.state["rounds"][0]["revealed"])

    def test_full_solve_advances_round(self):
        room, _ = self.engine.start("t:cg1b", "cryptogram", ADA)
        rnd, _truth = self._truth(room)
        out = self.engine.move("t:cg1b", "solve " + rnd["quote"], ADA)
        self.assertTrue(any("cracks it" in m for m in out))
        self.assertEqual(room.state["round"], 1)
        self.assertEqual(room.state["scores"].get(ADA.key), 50)

    def test_wrong_solve_keeps_round(self):
        room, _ = self.engine.start("t:cg2", "cryptogram", ADA)
        out = self.engine.move("t:cg2", "solve totally wrong guess", ADA)
        self.assertEqual(room.state["round"], 0)
        self.assertTrue(any("not quite" in m for m in out))

    def test_hint_reveals_with_scroll(self):
        room, _ = self.engine.start("t:cg3", "cryptogram", ADA)
        out = self.engine.move("t:cg3", "hint", ADA)
        self.assertTrue(any("Hint Scroll" in m for m in out))
        self.engine.store.add_coins(ADA, 500, "test")
        self.engine.move("t:cg3", "/shop buy hint_scroll", ADA)
        before = len(room.state["rounds"][0]["revealed"])
        out = self.engine.move("t:cg3", "hint", ADA)
        after = len(room.state["rounds"][0]["revealed"])
        # the hint reveals one letter; the house may crack another on
        # its own turn right after — so at least one new letter
        self.assertGreater(after, before)
        self.assertTrue(any("hint:" in m for m in out))

    def test_house_plays_frequency_analysis(self):
        room, _ = self.engine.start("t:cg4", "cryptogram", ADA,
                                    difficulty="normal")
        # let the house take several turns (Ada passes each time)
        cracked = 0
        for _ in range(12):
            if room.state.get("over"):
                break
            before = len(room.state["rounds"][room.state["round"]]
                         ["revealed"])
            self.engine.move("t:cg4", "/pass", ADA)
            after = len(room.state["rounds"][room.state["round"]]
                        ["revealed"])
            cracked = max(cracked, after)
        self.assertGreater(cracked, 0,
                           "house never cracked a letter in 12 turns")

    def test_full_match_win_and_achievements(self):
        room, _ = self.engine.start("t:cg5", "cryptogram", ADA,
                                    difficulty="easy")
        for _ in range(3):
            if room.state.get("over"):
                break
            rnd = room.state["rounds"][room.state["round"]]
            # Ada solves immediately, before the house can interfere
            cur = room.current
            if cur is not None and not cur.is_ai:
                self.engine.move("t:cg5", "solve " + rnd["quote"], cur)
            else:
                self.engine.move("t:cg5", "/pass", ADA)
                if not room.state.get("over"):
                    rnd = room.state["rounds"][room.state["round"]]
                    self.engine.move("t:cg5", "solve " + rnd["quote"], ADA)
        self.assertTrue(room.state.get("over"))
        unlocked = {a["id"] for a in get_achievements(self.db, ADA.key)}
        self.assertIn("cryptogram_win", unlocked)
        # Ada solved with zero wrong guesses of any kind
        wrong = sum(int((r.get("wrong") or {}).get(ADA.key, 0))
                   for r in room.state["rounds"])
        if wrong == 0:
            self.assertIn("cryptogram_perfect", unlocked)

    def test_derangement_has_no_fixed_points(self):
        import random
        from nomorals.games.games.puzzles import _derangement
        for seed in range(20):
            cipher = _derangement(random.Random(seed))
            self.assertTrue(all(a != b for a, b in cipher.items()))
            self.assertEqual(sorted(cipher.values()),
                             sorted(cipher.keys()))


if __name__ == "__main__":
    unittest.main()
