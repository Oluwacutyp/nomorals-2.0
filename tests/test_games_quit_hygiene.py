"""Quit hygiene for every game: full session clear on quit.

For EACH registered game this suite:
  1. starts a game,
  2. plays a mid-game move,
  3. quits,
  4. asserts clean state — no session, no room, no id-index entry, no
     active DB row, no relay residue, no scheduler output afterwards —
  5. asserts the zombie test: a move after quit is rejected (never
     accepted, never resurrects the room),
  6. asserts a fresh start works with a brand-new room.

Plus: relay-duel quit (both sides), pending-invite cancellation, the
start-failure zombie-row regression, restart-restore quit, and the
finished-row prune.
"""
from __future__ import annotations

import time
import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.players import Player
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db: Database) -> None:
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[tuple[str, str]] = []
    engine = GameEngine(Ctx(db), send=lambda c, t: sent.append((c, t)))
    return engine, db, sent


ADA = Player.from_sender("telegram", "456", "Ada")
BOB = Player.from_sender("telegram", "789", "Bob")

#: one real mid-game move per game (game, chat kind, move text)
GAME_MOVES = [
    ("2048", "dm", "up"),
    ("20q", "dm", "yes"),
    ("arena", "dm", "attack"),
    ("auction", "dm", "bid 10"),
    ("battleship", "dm", "b2"),
    ("blackjack", "dm", "hit"),
    ("bulls", "dm", "1234"),
    ("case", "dm", "clue"),
    ("connect4", "dm", "3"),
    ("craps", "dm", "roll"),
    ("digits", "dm", "7"),
    ("duel", "dm", "a"),
    ("escape", "dm", "look"),
    ("hangman", "dm", "e"),
    ("king", "dm", "attack"),
    ("mafia", "group", "vote bob"),
    ("memory", "dm", "1 2"),
    ("mines", "dm", "b2"),
    ("numberguess", "dm", "50"),
    ("poker", "dm", "call"),
    ("political", "group", "tax"),
    ("roulette", "dm", "red 10"),
    ("rpg", "dm", "north"),
    ("rps", "dm", "rock"),
    ("shop", "dm", "buy torch"),
    ("slots", "dm", "spin"),
    ("snake", "dm", "up"),
    ("spy", "dm", "bob"),
    ("story", "dm", "the dragon woke"),
    ("trivia", "dm", "b"),
    ("ttt", "dm", "5"),
    ("two_truths", "dm", "1"),
    ("wordchain", "dm", "elephant"),
    ("wordle", "dm", "arise"),
    ("world", "dm", "build farm"),
    ("wyrr", "dm", "1"),
]


def active_rows(db: Database, chat_key: str) -> list:
    return db.query(
        "SELECT id FROM game_rooms WHERE chat_key = ? AND status = 'active'",
        (chat_key,)) or []


class PerGameQuitHygieneTests(unittest.TestCase):
    """Every game: start → mid-game move → quit → clean; then the
    zombie test (move after quit is rejected) and a fresh start."""

    def test_all_games_registered(self):
        engine, _db, _sent = make_engine()
        try:
            registered = set(engine.games)
            for game, _kind, _move in GAME_MOVES:
                self.assertIn(game, registered, f"{game} missing from engine")
        finally:
            engine.shutdown()

    def test_quit_clears_every_game(self):
        for game, kind, move in GAME_MOVES:
            with self.subTest(game=game):
                self._check_one(game, kind, move)

    def _check_one(self, game: str, kind: str, move: str) -> None:
        engine, db, sent = make_engine()
        try:
            chat = f"t:quit-{game}"
            room, _msgs = engine.start(chat, game, ADA, kind=kind)
            room_id = room.id
            # mid-game move (some one-shot games finish here — both
            # paths must end clean)
            engine.move(chat, move, ADA)
            engine.quit(chat)

            # no session
            self.assertIsNone(engine.live(chat),
                              f"{game}: live() still returns a room")
            # no room
            self.assertNotIn(chat, engine._rooms,
                             f"{game}: room still in _rooms")
            self.assertNotIn(room_id, engine._by_id,
                             f"{game}: room still in _by_id")
            # no active DB row
            self.assertEqual(active_rows(db, chat), [],
                             f"{game}: active DB row leaked")
            # no relay residue for this chat
            self.assertIsNone(engine.relay.get_relay_for_chat(chat),
                              f"{game}: relay mapping stuck")

            # ── zombie test: a move after quit must be rejected ──
            n_sent = len(sent)
            out = engine.move(chat, move, ADA)
            self.assertEqual(out, [],
                             f"{game}: post-quit move accepted: {out!r}")
            self.assertEqual(len(sent), n_sent,
                             f"{game}: post-quit move emitted output")
            self.assertIsNone(engine.live(chat),
                              f"{game}: post-quit move resurrected the room")
            self.assertEqual(active_rows(db, chat), [],
                             f"{game}: post-quit move leaked a DB row")

            # ── no dead scheduler ticks for the finished game ──
            sent.clear()
            engine._sweep_timeouts()
            for c, _t in sent:
                self.assertNotEqual(c, chat,
                                    f"{game}: scheduler ticked a dead room")

            # ── fresh start works, brand-new table ──
            room2, _ = engine.start(chat, game, ADA, kind=kind)
            self.assertNotEqual(room2.id, room_id,
                                f"{game}: fresh start reused the old room")
            self.assertEqual(room2.status, "active")
            self.assertIs(engine.live(chat), room2)
            engine.quit(chat)
            self.assertIsNone(engine.live(chat))
        finally:
            engine.shutdown()


class RelayQuitHygieneTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _duel(self):
        inv = self.engine.relay.create_invite("t:duelA", ADA, "ttt")
        relay = self.engine.relay.accept_invite(inv.code, "t:duelB", BOB)
        self.engine.relay.relay_move("t:duelA", "1", ADA)
        return relay

    def _assert_relay_clean(self):
        r = self.engine.relay
        self.assertEqual(r.relays, {})
        self.assertEqual(r.chat_to_relay, {})
        self.assertEqual(r.invites, {})
        self.assertEqual(self.db.query("SELECT * FROM game_relays") or [], [])
        self.assertEqual(self.db.query("SELECT * FROM game_invites") or [], [])

    def test_quit_from_inviter_side(self):
        relay = self._duel()
        vchat = relay.virtual_chat
        self.engine.quit("t:duelA")
        self._assert_relay_clean()
        self.assertIsNone(self.engine.live(vchat))
        self.assertNotIn(vchat, self.engine._rooms)
        self.assertEqual(active_rows(self.db, vchat), [])
        # opponent's move is rejected, not accepted
        n = len(self.sent)
        self.assertEqual(self.engine.relay.relay_move("t:duelB", "2", BOB), [])
        self.assertEqual(len(self.sent), n)
        self.assertIsNone(self.engine.live(vchat))

    def test_quit_from_accepter_side(self):
        relay = self._duel()
        vchat = relay.virtual_chat
        self.engine.quit("t:duelB")
        self._assert_relay_clean()
        self.assertIsNone(self.engine.live(vchat))
        n = len(self.sent)
        self.assertEqual(self.engine.relay.relay_move("t:duelA", "2", ADA), [])
        self.assertEqual(len(self.sent), n)

    def test_quit_cancels_pending_invite(self):
        inv = self.engine.relay.create_invite("t:invA", ADA, "ttt")
        out = self.engine.quit("t:invA")
        self.assertIn("cancelled", "\n".join(out).lower())
        self.assertNotIn(inv.code, self.engine.relay.invites)
        self.assertEqual(
            self.db.query("SELECT * FROM game_invites WHERE code = ?",
                          (inv.code,)) or [], [])
        # the code is dead — accepting now fails
        with self.assertRaises(ValueError):
            self.engine.relay.accept_invite(inv.code, "t:invB", BOB)

    def test_quit_after_natural_finish_clears_relay(self):
        inv = self.engine.relay.create_invite("t:finA", ADA, "ttt")
        relay = self.engine.relay.accept_invite(inv.code, "t:finB", BOB)
        # play until the engine finishes the game
        for sq in ("1", "2", "4", "3", "7"):
            self.engine.relay.relay_move("t:finA", sq, ADA)
            if self.engine.live(relay.virtual_chat) is None:
                break
        self._assert_relay_clean()
        self.assertIsNone(self.engine.live(relay.virtual_chat))


class StartFailureHygieneTests(unittest.TestCase):
    def test_failed_start_leaves_no_zombie_row(self):
        engine, db, sent = make_engine()
        try:
            orig = engine._pump_ai

            def boom(room, out):
                raise RuntimeError("ai exploded")

            engine._pump_ai = boom
            with self.assertRaises(RuntimeError):
                engine.start("t:boom", "ttt", ADA)
            engine._pump_ai = orig
            self.assertEqual(active_rows(db, "t:boom"), [])
            self.assertIsNone(engine.live("t:boom"))
            self.assertNotIn("t:boom", engine._rooms)
            # a fresh start works afterwards
            room, _ = engine.start("t:boom", "ttt", ADA)
            self.assertEqual(room.status, "active")
            engine.quit("t:boom")
        finally:
            engine.shutdown()


class RestartQuitHygieneTests(unittest.TestCase):
    def test_quit_kills_db_restored_room(self):
        db = Database(":memory:")
        db.migrate()
        eng1 = GameEngine(Ctx(db), send=lambda c, t: None)
        eng1.start("t:rest", "ttt", ADA)
        eng1.shutdown()
        # "restart": a brand-new engine over the same DB
        sent: list = []
        eng2 = GameEngine(Ctx(db), send=lambda c, t: sent.append((c, t)))
        try:
            out = eng2.quit("t:rest")
            self.assertTrue(any("table closed" in m for m in out),
                            f"quit did not close the restored room: {out!r}")
            self.assertEqual(active_rows(db, "t:rest"), [])
            self.assertIsNone(eng2.live("t:rest"))
        finally:
            eng2.shutdown()

    def test_restored_room_keeps_ai_seats(self):
        db = Database(":memory:")
        db.migrate()
        eng1 = GameEngine(Ctx(db), send=lambda c, t: None)
        eng1.start("t:ai", "ttt", ADA)
        eng1.shutdown()
        eng2 = GameEngine(Ctx(db), send=lambda c, t: None)
        try:
            room = eng2.live("t:ai")
            self.assertIsNotNone(room)
            ai = [p for p in room.players if p.is_ai]
            self.assertTrue(ai, "AI seats came back as humans after restore")
            eng2.quit("t:ai")
        finally:
            eng2.shutdown()


class HistoryPruneTests(unittest.TestCase):
    def test_old_finished_rows_pruned(self):
        engine, db, sent = make_engine()
        try:
            old = time.time() - 31 * 86400
            db.execute(
                "INSERT INTO game_rooms (id, game, chat_key, platform, kind,"
                " players, turn, state, status, started_at, seed, updated_at)"
                " VALUES ('old1','ttt','t:old','t','dm','[]',0,'{}',"
                "'finished',0,1,?)", (old,))
            db.execute(
                "INSERT INTO game_rooms (id, game, chat_key, platform, kind,"
                " players, turn, state, status, started_at, seed, updated_at)"
                " VALUES ('new1','ttt','t:new','t','dm','[]',0,'{}',"
                "'finished',0,1,?)", (time.time(),))
            engine._last_prune = 0.0
            engine._maybe_prune_history(time.time())
            rows = {r["id"] for r in
                    db.query("SELECT id FROM game_rooms") or []}
            self.assertNotIn("old1", rows)
            self.assertIn("new1", rows)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
