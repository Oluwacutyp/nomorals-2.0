"""Matchmaking queue + ELO ratings: pairing, tolerance, ranked finishes."""
from __future__ import annotations

import time
import unittest

from nomorals.games import matchmaking
from nomorals.games.engine import GameEngine
from nomorals.games.matchmaking import (
    QUEUE_GAMES,
    Matchmaker,
    get_rating,
    match_status,
    record_elo,
    render_ratings,
)
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
BOB = Player.from_sender("telegram", "222", "Bob")
CAT = Player.from_sender("telegram", "333", "Cat")


class EloTests(unittest.TestCase):
    def test_unrated_is_1000(self):
        db = Database(":memory:")
        db.migrate()
        self.assertEqual(get_rating(db, "nobody", "pvp"), 1000)

    def test_win_moves_points(self):
        db = Database(":memory:")
        db.migrate()
        ra, da, rb, db_ = record_elo(db, "pvp", "a", "b", 1.0,
                                     a_name="A", b_name="B")
        self.assertEqual(da, 16)
        self.assertEqual(db_, -16)
        self.assertEqual(ra, 1016)
        self.assertEqual(rb, 984)
        # zero-sum
        self.assertEqual(da + db_, 0)

    def test_draw_moves_less(self):
        db = Database(":memory:")
        db.migrate()
        ra, da, rb, db_ = record_elo(db, "pvp", "a", "b", 0.5)
        self.assertEqual(da, 0)
        self.assertEqual(db_, 0)

    def test_upset_pays_more(self):
        db = Database(":memory:")
        db.migrate()
        # b is 400 points stronger; a winning is an upset
        record_elo(db, "ttt", "b", "a", 1.0)  # b beats a once... reset below
        db.execute("DELETE FROM game_elo")
        db.execute(
            "INSERT INTO game_elo (player_key, game, rating, games) "
            "VALUES ('a', 'ttt', 1000, 5), ('b', 'ttt', 1400, 5)")
        ra, da, rb, db_ = record_elo(db, "ttt", "a", "b", 1.0)
        self.assertGreater(da, 16)  # upset bonus over the even 16
        self.assertEqual(da + db_, 0)

    def test_ratings_render(self):
        db = Database(":memory:")
        db.migrate()
        record_elo(db, "pvp", "a", "b", 1.0, a_name="Ada", b_name="Bob")
        text = render_ratings(db, "pvp")
        self.assertIn("Ada", text)
        self.assertIn("1016", text)

    def test_ratings_empty(self):
        db = Database(":memory:")
        db.migrate()
        self.assertIn("no rated", render_ratings(db, "pvp"))


class QueueTests(unittest.TestCase):
    def test_enqueue_and_status_and_dequeue(self):
        engine, db, sent = make_engine()
        mm = engine.matchmaker
        reply = mm.enqueue(ADA, "test:a", "pvp")
        self.assertIn("queued", reply)
        status = match_status(db, ADA.key)
        self.assertIsNotNone(status)
        self.assertIn("pvp", status)
        self.assertIn("left the queue", mm.dequeue(ADA.key))
        self.assertIsNone(match_status(db, ADA.key))

    def test_double_enqueue_refused(self):
        engine, db, sent = make_engine()
        mm = engine.matchmaker
        mm.enqueue(ADA, "test:a", "pvp")
        self.assertIn("already queued", mm.enqueue(ADA, "test:a", "ttt"))

    def test_unknown_game_refused(self):
        engine, db, sent = make_engine()
        mm = engine.matchmaker
        self.assertIn("queueable", mm.enqueue(ADA, "test:a", "slots"))

    def test_sweep_pairs_two_waiters(self):
        engine, db, sent = make_engine()
        mm = engine.matchmaker
        mm.enqueue(ADA, "test:a", "pvp")
        mm.enqueue(BOB, "test:b", "pvp")
        made = mm.sweep()
        self.assertEqual(made, 1)
        # both left the queue, both got told
        self.assertIsNone(match_status(db, ADA.key))
        self.assertIsNone(match_status(db, BOB.key))
        joined = "\n".join(sent)
        self.assertIn("match found", joined)
        # a relay duel now connects the two chats
        relay = engine.relay
        self.assertIsNotNone(relay.get_relay_for_chat("test:a"))
        self.assertIsNotNone(relay.get_relay_for_chat("test:b"))

    def test_sweep_respects_elo_tolerance(self):
        engine, db, sent = make_engine()
        mm = engine.matchmaker
        # Ada is a 1600-rated shark, Bob a fresh 1000 — no match yet
        get_rating(db, ADA.key, "pvp")  # ensure tables exist
        db.execute(
            "INSERT INTO game_elo (player_key, game, rating, games) "
            "VALUES (?, 'pvp', 1600, 20)", (ADA.key,))
        mm.enqueue(ADA, "test:a", "pvp")
        mm.enqueue(BOB, "test:b", "pvp")
        self.assertEqual(mm.sweep(), 0)
        self.assertIsNotNone(match_status(db, ADA.key))
        # ...but after waiting long enough the tolerance widens and
        # they pair (nobody waits forever)
        db.execute(
            "UPDATE match_queue SET enqueued_at = ?",
            (time.time() - 600,))
        self.assertEqual(mm.sweep(), 1)

    def test_third_wheel_waits(self):
        engine, db, sent = make_engine()
        mm = engine.matchmaker
        mm.enqueue(ADA, "test:a", "pvp")
        mm.enqueue(BOB, "test:b", "pvp")
        mm.enqueue(CAT, "test:c", "pvp")
        self.assertEqual(mm.sweep(), 1)
        # Cat is still waiting for the next opponent
        self.assertIsNotNone(match_status(db, CAT.key))

    def test_queue_games_are_two_player(self):
        engine, db, sent = make_engine()
        for name in QUEUE_GAMES:
            game = engine.games[name]
            self.assertEqual(game.max_players, 2, name)


class RankedFinishTests(unittest.TestCase):
    def _two_human_room(self, engine, game_name="pvp"):
        room, _ = engine.start("test:rank", game_name, ADA, kind="group")
        engine.join("test:rank", BOB)
        return room

    def test_win_moves_elo(self):
        from nomorals.games.matchmaking import maybe_record_ranked
        engine, db, sent = make_engine()
        room = self._two_human_room(engine)
        game = engine.games["pvp"]
        lines = maybe_record_ranked(db, room, game, ADA)
        self.assertTrue(any("rated pvp" in l for l in lines))
        self.assertEqual(get_rating(db, ADA.key, "pvp"), 1016)
        self.assertEqual(get_rating(db, BOB.key, "pvp"), 984)
        engine.quit("test:rank")

    def test_draw_moves_nothing_at_even_elo(self):
        from nomorals.games.matchmaking import maybe_record_ranked
        engine, db, sent = make_engine()
        room = self._two_human_room(engine)
        game = engine.games["pvp"]
        lines = maybe_record_ranked(db, room, game, "draw")
        self.assertTrue(any("rated pvp" in l for l in lines))
        self.assertEqual(get_rating(db, ADA.key, "pvp"), 1000)
        engine.quit("test:rank")

    def test_second_human_missing_means_no_rating(self):
        from nomorals.games.matchmaking import maybe_record_ranked
        engine, db, sent = make_engine()
        room, _ = engine.start("test:ttt2", "ttt", ADA, kind="group")
        game = engine.games["ttt"]
        self.assertEqual(maybe_record_ranked(db, room, game, "draw"), [])
        engine.quit("test:ttt2")

    def test_solo_games_ignored(self):
        from nomorals.games.matchmaking import maybe_record_ranked
        engine, db, sent = make_engine()
        room, _ = engine.start("test:sl", "slots", ADA, kind="dm")
        game = engine.games["slots"]
        self.assertEqual(maybe_record_ranked(db, room, game, ADA), [])
        engine.quit("test:sl")

    def test_engine_finish_announces_rating(self):
        # end-to-end: a real finish with 2 humans posts the rated line
        from nomorals.games.matchmaking import maybe_record_ranked
        engine, db, sent = make_engine()
        room = self._two_human_room(engine, "pvp")
        game = engine.games["pvp"]
        # simulate the engine's finish path for a decided duel
        room.state["done"] = True
        room.state["winner_key"] = ADA.key
        lines = maybe_record_ranked(db, room, game, game.winner(room))
        self.assertTrue(any("Ada" in l and "1016" in l for l in lines))
        engine.quit("test:rank")


if __name__ == "__main__":
    unittest.main()
