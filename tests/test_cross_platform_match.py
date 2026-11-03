"""Cross-platform matchmaking proof (group build §13).

DESIGN §13: the relay invite is platform-neutral, the Matchmaker
stores each queued player's platform, and a made match creates a
relay duel — but pairing a Telegram player with a Discord or
WhatsApp player was never proven end to end. This file is that
proof: queue across platforms, sweep, both chats notified on their
own keys, moves relayed both ways through the virtual room, and the
ranked finish recording ELO for both profiles.
"""
from __future__ import annotations

import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.matchmaking import get_rating, match_status
from nomorals.games.players import Player
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db):
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[tuple[str, str]] = []
    engine = GameEngine(
        Ctx(db), send=lambda chat, text: sent.append((chat, text)))
    return engine, db, sent


TG = Player.from_sender("telegram", "111", "Ada")
DC = Player.from_sender("discord", "222", "Bob")
WA = Player.from_sender("whatsapp", "234", "Cat")

TG_CHAT = "telegram:111"
DC_CHAT = "discord:222"
WA_CHAT = "whatsapp:234"


def _pair(engine, mm, a, a_chat, b, b_chat, game="connect4"):
    mm.enqueue(a, a_chat, game)
    mm.enqueue(b, b_chat, game)
    return mm.sweep()


class CrossPlatformPairingTests(unittest.TestCase):
    def test_telegram_pairs_with_discord(self):
        engine, db, sent = make_engine()
        made = _pair(engine, engine.matchmaker, TG, TG_CHAT, DC, DC_CHAT)
        self.assertEqual(made, 1)
        self.assertIsNone(match_status(db, TG.key))
        self.assertIsNone(match_status(db, DC.key))
        relay = engine.relay
        room_a = relay.get_relay_for_chat(TG_CHAT)
        room_b = relay.get_relay_for_chat(DC_CHAT)
        self.assertIsNotNone(room_a)
        self.assertIs(room_a, room_b)
        # each player was notified on their OWN chat key
        chats_told = {chat for chat, _text in sent}
        self.assertIn(TG_CHAT, chats_told)
        self.assertIn(DC_CHAT, chats_told)
        told = dict()
        for chat, text in sent:
            told.setdefault(chat, []).append(text)
        self.assertTrue(any("match found" in t for t in told[TG_CHAT]))
        self.assertTrue(any("match found" in t for t in told[DC_CHAT]))

    def test_telegram_pairs_with_whatsapp(self):
        engine, db, sent = make_engine()
        made = _pair(engine, engine.matchmaker, TG, TG_CHAT, WA, WA_CHAT)
        self.assertEqual(made, 1)
        relay = engine.relay
        self.assertIsNotNone(relay.get_relay_for_chat(WA_CHAT))
        self.assertIsNotNone(relay.get_relay_for_chat(TG_CHAT))

    def test_moves_relay_both_ways(self):
        engine, db, sent = make_engine()
        _pair(engine, engine.matchmaker, TG, TG_CHAT, DC, DC_CHAT)
        relay = engine.relay
        # player A (telegram) opens in column 4
        out_a = relay.relay_move(TG_CHAT, "4", TG)
        self.assertTrue(out_a)
        room = engine.live(relay.get_relay_for_chat(TG_CHAT).virtual_chat)
        self.assertIsNotNone(room)
        board = room.state["board"]
        self.assertEqual(board[5][3], 1)
        # player B (discord) answers in column 5
        out_b = relay.relay_move(DC_CHAT, "5", DC)
        self.assertTrue(out_b)
        self.assertEqual(board[5][4], 2)

    def test_wrong_turn_does_not_move(self):
        engine, db, sent = make_engine()
        _pair(engine, engine.matchmaker, TG, TG_CHAT, DC, DC_CHAT)
        relay = engine.relay
        # discord tries to move first — telegram (seat 1) opens
        relay.relay_move(DC_CHAT, "4", DC)
        room = engine.live(relay.get_relay_for_chat(TG_CHAT).virtual_chat)
        self.assertEqual(room.state["board"][5][3], 0)

    def test_cross_platform_finish_records_both_ratings(self):
        engine, db, sent = make_engine()
        _pair(engine, engine.matchmaker, TG, TG_CHAT, DC, DC_CHAT)
        relay = engine.relay
        # A stacks column 1 four high while B fills column 2: A wins.
        for a_move, b_move in (("1", "2"), ("1", "2"), ("1", "2")):
            relay.relay_move(TG_CHAT, a_move, TG)
            relay.relay_move(DC_CHAT, b_move, DC)
        relay.relay_move(TG_CHAT, "1", TG)
        self.assertEqual(get_rating(db, TG.key, "connect4"), 1016)
        self.assertEqual(get_rating(db, DC.key, "connect4"), 984)
        # the relay retires when the game ends
        self.assertIsNone(relay.get_relay_for_chat(TG_CHAT))
        self.assertIsNone(relay.get_relay_for_chat(DC_CHAT))


class AdmittedGamesDuelTests(unittest.TestCase):
    """Every QUEUE_GAMES admission proven the same way (§13): paired
    across platforms, one legal move each way, the house never plays
    a third move."""

    def _duel(self, game):
        engine, db, sent = make_engine()
        made = _pair(engine, engine.matchmaker, TG, TG_CHAT, DC,
                     DC_CHAT, game=game)
        self.assertEqual(made, 1, game)
        relay = engine.relay
        room_ref = relay.get_relay_for_chat(TG_CHAT)
        self.assertIsNotNone(room_ref, game)
        room = engine.live(room_ref.virtual_chat)
        self.assertIsNotNone(room, game)
        # exactly two humans at the table — no House seat survives
        self.assertEqual(len(room.humans), 2, game)
        self.assertEqual(len(room.players), 2, game)
        return engine, db, relay, room

    def test_ttt_duel_marks_and_win(self):
        engine, db, relay, room = self._duel("ttt")
        relay.relay_move(TG_CHAT, "1", TG)   # X corner
        self.assertEqual(room.state["board"][0], "X")
        relay.relay_move(DC_CHAT, "5", DC)   # O center — no house reply
        self.assertEqual(room.state["board"][4], "O")
        self.assertEqual(room.state["moves"], 2)
        relay.relay_move(TG_CHAT, "2", TG)
        relay.relay_move(DC_CHAT, "9", DC)
        relay.relay_move(TG_CHAT, "3", TG)   # X completes the top row
        self.assertEqual(room.state["winner"], "X")
        self.assertEqual(get_rating(db, TG.key, "ttt"), 1016)
        self.assertEqual(get_rating(db, DC.key, "ttt"), 984)

    def test_inbox_duels_not_admitted_yet(self):
        # reversi/gomoku/checkers were duel-tested 2026-11-02 and are
        # human-side vs house-side inside on_move (the second human's
        # move is rejected or the house replies for them). They stay
        # OUT of the queue until each grows a two-human mode and the
        # duel tests move here from the design notes.
        from nomorals.games.matchmaking import QUEUE_GAMES

        for game in ("reversi", "gomoku", "checkers"):
            self.assertNotIn(game, QUEUE_GAMES)


if __name__ == "__main__":
    unittest.main()
