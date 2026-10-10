"""Provably-fair RNG: commit-reveal, deterministic draws, casino wiring."""
from __future__ import annotations

import unittest

from nomorals.games import fairness
from nomorals.games.engine import GameEngine
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


class CommitRevealTests(unittest.TestCase):
    def test_commit_then_verify(self):
        fair = fairness.init_table("ada-seed")
        self.assertIn("commit", fair)
        self.assertFalse(fairness.verify(fair["commit"], "00" * 32))
        self.assertTrue(fairness.verify(fair["commit"], fair["seed"]))

    def test_verify_rejects_tampered_commit(self):
        fair = fairness.init_table("x")
        bad = ("0" if fair["commit"][0] != "0" else "1") + fair["commit"][1:]
        self.assertFalse(fairness.verify(bad, fair["seed"]))

    def test_verify_rejects_garbage_seed(self):
        fair = fairness.init_table("x")
        self.assertFalse(fairness.verify(fair["commit"], "not-hex"))
        self.assertFalse(fairness.verify("", fair["seed"]))


class DeterminismTests(unittest.TestCase):
    def _two_tables(self):
        # same seeds → identical draw sequences (the audit guarantee)
        a = fairness.init_table("client")
        b = dict(a)  # same server seed + client seed, fresh counter
        b["counter"] = 0
        return a, b

    def test_draw_int_deterministic(self):
        a, b = self._two_tables()
        for i in range(20):
            self.assertEqual(
                fairness.draw_int(a, f"tag:{i}", 0, 36),
                fairness.draw_int(b, f"tag:{i}", 0, 36))

    def test_draw_int_range(self):
        fair = fairness.init_table("c")
        for _ in range(200):
            v = fairness.draw_int(fair, "r", 1, 6)
            self.assertGreaterEqual(v, 1)
            self.assertLessEqual(v, 6)

    def test_shuffle_deterministic_and_complete(self):
        a, b = self._two_tables()
        d1 = list(range(52))
        d2 = list(range(52))
        fairness.shuffle(a, "deck", d1)
        fairness.shuffle(b, "deck", d2)
        self.assertEqual(d1, d2)
        self.assertEqual(sorted(d1), list(range(52)))

    def test_weighted_draw(self):
        fair = fairness.init_table("c")
        counts = [0] * 3
        for i in range(600):
            counts[fairness.draw_weighted(fair, f"w:{i}", (70, 20, 10))] += 1
        # heavy weight wins most, light weight wins some
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])
        self.assertGreater(counts[2], 0)

    def test_client_seed_changes_sequence(self):
        a = fairness.init_table("alice")
        b = fairness.init_table("bob")
        # same server seed, different client seed → different draws
        b["seed"] = a["seed"]
        draws_a = [fairness.draw_int(a, "x", 0, 100) for _ in range(5)]
        draws_b = [fairness.draw_int(b, "x", 0, 100) for _ in range(5)]
        self.assertNotEqual(draws_a, draws_b)

    def test_counter_advances(self):
        fair = fairness.init_table("c")
        fairness.draw_int(fair, "t", 0, 9)
        self.assertEqual(fair["counter"], 1)
        self.assertEqual(fair["draws"], 1)


class ClientSeedTests(unittest.TestCase):
    def test_minted_and_persistent(self):
        db = Database(":memory:")
        db.migrate()
        s1 = fairness.get_client_seed(db, "p:1")
        s2 = fairness.get_client_seed(db, "p:1")
        self.assertEqual(s1, s2)
        self.assertTrue(s1)

    def test_set_and_get(self):
        db = Database(":memory:")
        db.migrate()
        fairness.set_client_seed(db, "p:1", "banana")
        self.assertEqual(fairness.get_client_seed(db, "p:1"), "banana")

    def test_set_rejects_empty(self):
        db = Database(":memory:")
        db.migrate()
        with self.assertRaises(ValueError):
            fairness.set_client_seed(db, "p:1", "   ")


class CasinoWiringTests(unittest.TestCase):
    def test_blackjack_commit_before_deal_and_reveal_at_finish(self):
        engine, db, sent = make_engine()
        room, msgs = engine.start("test:bj", "blackjack", ADA, kind="dm")
        # commitment published at table open, before any card is known
        self.assertIn("fair", room.state)
        commit = room.state["fair"]["commit"]
        self.assertTrue(any("provably fair" in m for m in msgs))
        # the deck was dealt from the committed sequence
        self.assertTrue(room.state["dealt"])
        self.assertEqual(len(room.state["player"]), 2)
        # finish → the seed is revealed and verifies
        engine.quit("test:bj")
        joined = "\n".join(sent)
        self.assertIn("fairness reveal", joined)
        self.assertIn(room.state["fair"]["seed"], joined)
        self.assertTrue(
            fairness.verify(commit, room.state["fair"]["seed"]))

    def test_roulette_spin_uses_fair_stream(self):
        engine, db, sent = make_engine()
        room, msgs = engine.start("test:rl", "roulette", ADA, kind="dm")
        before = room.state["fair"]["draws"]
        engine.move("test:rl", "bet red", ADA, kind="dm")
        engine.move("test:rl", "spin", ADA, kind="dm")
        self.assertGreater(room.state["fair"]["draws"], before)
        self.assertTrue(room.state["done"])
        self.assertIn(room.state["result"], range(0, 37))

    def test_slots_reels_use_fair_stream(self):
        engine, db, sent = make_engine()
        room, msgs = engine.start("test:sl", "slots", ADA, kind="dm")
        before = room.state["fair"]["draws"]
        engine.move("test:sl", "spin", ADA, kind="dm")
        self.assertGreater(room.state["fair"]["draws"], before)
        self.assertEqual(len(room.state["reels"]), 3)
        game = engine.games["slots"]
        for r in room.state["reels"]:
            self.assertIn(r, game.SYMBOLS)

    def test_fair_games_listed(self):
        for name in ("blackjack", "roulette", "slots", "poker", "craps"):
            self.assertIn(name, fairness.FAIR_GAMES)

    def test_reveal_block_format(self):
        fair = fairness.init_table("c")
        block = fairness.reveal_block(fair)
        self.assertIn(fair["commit"], block)
        self.assertIn(fair["seed"], block)
        self.assertIn("✓", block)
        self.assertTrue(fair["revealed"])


if __name__ == "__main__":
    unittest.main()
