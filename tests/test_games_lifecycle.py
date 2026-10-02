"""Session lifecycle hardening: quit teardown, sticky sessions, idle expiry.

Regression coverage for the games session lifecycle:
- quitting deterministically tears a session down (rooms, id index,
  timers, per-session memory) — nothing sticks around;
- after quit, further input never resurrects or continues the old
  session (no zombie replies, no zombie ticks from the scheduler);
- idle/abandoned tables expire and free the chat;
- one scheduler thread per engine no matter how many sessions run;
- game errors surface as real errors, not vague hiccups or silence.
"""
from __future__ import annotations

import threading
import time
import unittest

from nomorals.games import engine as engine_mod
from nomorals.games.ai import GameMind
from nomorals.games.engine import GameEngine
from nomorals.games.games.base import MultiGame
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


def scheduler_threads() -> int:
    return sum(1 for t in threading.enumerate()
               if t.name == "nm-game-turns" and t.is_alive())


# ── quit teardown ────────────────────────────────────────────────────────────

class QuitTeardownTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_quit_pops_room_and_id_index(self):
        room, _ = self.engine.start("t:q1", "ttt", ADA)
        room_id = room.id
        self.engine.quit("t:q1")
        self.assertNotIn("t:q1", self.engine._rooms)
        self.assertNotIn(room_id, self.engine._by_id)

    def test_quit_persists_finished_status(self):
        self.engine.start("t:q2", "ttt", ADA)
        self.engine.quit("t:q2")
        row = self.db.query_one(
            "SELECT status FROM game_rooms WHERE chat_key = ?", ("t:q2",))
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "finished")

    def test_move_after_quit_is_silent_and_does_not_resurrect(self):
        # the sticky-session regression: quit → input must not continue
        # the old session or spin a zombie one back up
        self.engine.start("t:q3", "ttt", ADA)
        self.engine.quit("t:q3")
        self.sent.clear()
        out = self.engine.move("t:q3", "0", ADA)
        self.assertEqual(out, [])  # no zombie reply
        self.assertIsNone(self.engine.live("t:q3"))
        self.assertNotIn("t:q3", self.engine._rooms)  # no DB resurrection
        # ... even when asked twice (the _load_live fallback path)
        self.assertIsNone(self.engine.live("t:q3"))
        self.assertNotIn("t:q3", self.engine._rooms)

    def test_quit_then_new_start_works(self):
        self.engine.start("t:q4", "ttt", ADA)
        self.engine.quit("t:q4")
        room, _ = self.engine.start("t:q4", "hangman", ADA)
        self.assertEqual(room.game, "hangman")
        self.assertEqual(room.status, "active")

    def test_double_quit_is_clean(self):
        self.engine.start("t:q5", "ttt", ADA)
        self.engine.quit("t:q5")
        self.assertEqual(self.engine.quit("t:q5"), ["no game is live here."])

    def test_finish_is_idempotent_no_double_ledger(self):
        room, _ = self.engine.start("t:q6", "ttt", ADA)
        before = self.engine.store.get(ADA.key).games_played
        self.engine.quit("t:q6")
        after = self.engine.store.get(ADA.key).games_played
        self.assertEqual(after, before + 1)
        # a second finish must not credit the ledger again
        self.assertEqual(self.engine._finish(room), [])
        self.assertEqual(self.engine.store.get(ADA.key).games_played, after)

    def test_last_game_pruned_on_new_start(self):
        self.engine.start("t:q7", "ttt", ADA)
        self.engine.quit("t:q7")
        self.assertIn("t:q7", self.engine._last_game)
        self.engine.start("t:q7", "ttt", ADA)
        self.assertNotIn("t:q7", self.engine._last_game)

    def test_quit_closes_relay_duel(self):
        # relay players quit from their own DM; the room lives at the
        # virtual key — quitting must tear the relay down too
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        accepted = relay.accept_invite(inv.code, "telegram:chatB", BOB)
        out = self.engine.quit("telegram:chatA")
        self.assertTrue(any("duel closed" in m for m in out))
        self.assertIsNone(relay.get_relay_for_chat("telegram:chatA"))
        self.assertIsNone(relay.get_relay_for_chat("telegram:chatB"))
        self.assertIsNone(self.engine.live(accepted.virtual_chat))
        # and the dead relay answers nothing (no zombie)
        self.assertEqual(relay.relay_move("telegram:chatA", "0", ADA), [])

    def test_quit_unknown_chat_with_relay_instantiated(self):
        # relay object exists but this chat has no relay — clean answer
        _ = self.engine.relay
        self.assertEqual(self.engine.quit("t:nobody"),
                         ["no game is live here."])


# ── idle expiry + dead-session ticks ─────────────────────────────────────────

class IdleExpiryTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        engine_mod.IDLE_ROOM_TTL = 3600.0
        self.engine.shutdown()

    def test_idle_room_expires_and_frees_table(self):
        engine_mod.IDLE_ROOM_TTL = 3600.0
        room, _ = self.engine.start("t:i1", "blackjack", ADA)  # move_timeout=0
        room.last_activity = time.time() - 7200
        self.engine._sweep_timeouts()
        self.assertIsNone(self.engine.live("t:i1"))
        self.assertNotIn("t:i1", self.engine._rooms)
        self.assertTrue(any("idle" in m for m in self.sent))
        # the table is free again
        room2, _ = self.engine.start("t:i1", "ttt", ADA)
        self.assertEqual(room2.status, "active")

    def test_active_room_survives_sweep(self):
        self.engine.start("t:i2", "blackjack", ADA)
        self.engine._sweep_timeouts()
        live = self.engine.live("t:i2")
        self.assertIsNotNone(live)
        self.assertEqual(live.status, "active")

    def test_human_move_resets_idle_clock(self):
        room, _ = self.engine.start("t:i3", "ttt", ADA)
        room.last_activity = time.time() - 7200
        self.engine.move("t:i3", "0", ADA)  # a real move, game continues
        self.assertGreater(room.last_activity, time.time() - 7200)
        self.engine._sweep_timeouts()
        self.assertIsNotNone(self.engine.live("t:i3"))

    def test_abandoned_timed_game_eventually_expires(self):
        # per-turn timeouts alone ping-pong forever on an abandoned
        # table; the idle sweep must still reclaim it
        engine_mod.IDLE_ROOM_TTL = 3600.0
        room, _ = self.engine.start("t:i4", "ttt", ADA)
        room.last_activity = time.time() - 7200
        self.engine._sweep_timeouts()
        self.assertIsNone(self.engine.live("t:i4"))

    def test_scheduler_ignores_dead_session(self):
        # the race: sweep selects the room, quit() lands, then the tick
        # runs — a dead room must never receive the timeout tick
        room, _ = self.engine.start("t:i5", "ttt", ADA)
        game = self.engine.games["ttt"]
        old, game.move_timeout = game.move_timeout, 0.05
        try:
            room.turn_started = time.time() - 10
            self.engine.quit("t:i5")
            self.sent.clear()
            self.engine._handle_timeout(room)  # the raced tick
            self.engine._sweep_timeouts()
            self.assertFalse(any("⏰" in m for m in self.sent))
            self.assertIsNone(self.engine.live("t:i5"))
        finally:
            game.move_timeout = old

    def test_timeout_still_fires_for_live_room(self):
        room, _ = self.engine.start("t:i6", "ttt", ADA)
        game = self.engine.games["ttt"]
        old, game.move_timeout = game.move_timeout, 0.05
        try:
            room.turn_started = time.time() - 10
            self.engine._sweep_timeouts()
            self.assertTrue(any("⏰" in m for m in self.sent))
        finally:
            game.move_timeout = old


# ── threads ──────────────────────────────────────────────────────────────────

class ThreadLeakTests(unittest.TestCase):
    def test_no_leaked_threads_after_many_sessions(self):
        engine, _db, _sent = make_engine()
        try:
            self.assertEqual(scheduler_threads(), 1)
            before = threading.active_count()
            for i in range(25):
                engine.start(f"t:th{i}", "ttt", ADA)
                engine.quit(f"t:th{i}")
            self.assertEqual(scheduler_threads(), 1)
            self.assertLessEqual(threading.active_count(), before + 1)
            self.assertEqual(len(engine._rooms), 0)
            self.assertEqual(len(engine._by_id), 0)
        finally:
            engine.shutdown()

    def test_shutdown_stops_ticker(self):
        engine, _db, _sent = make_engine()
        self.assertEqual(scheduler_threads(), 1)
        engine.shutdown()
        # give the ticker a beat to observe the stop flag
        for _ in range(20):
            if scheduler_threads() == 0:
                break
            time.sleep(0.05)
        self.assertEqual(scheduler_threads(), 0)


# ── error surfacing ──────────────────────────────────────────────────────────

class BrokenMoveGame(MultiGame):
    name = "brokentest-move"
    description = "test game whose moves explode"
    min_players = 1
    max_players = 1
    ai_seats = 0

    def on_move(self, room, player, text, mind):
        raise RuntimeError("kaboom-42")


class BrokenAIGame(MultiGame):
    name = "brokentest-ai"
    description = "test game whose house turn explodes"
    min_players = 1
    max_players = 2
    ai_seats = 1

    def on_move(self, room, player, text, mind):
        room.state["moved"] = True
        return ["ok"]

    def is_over(self, room):
        return False

    def ai_turn(self, room, mind):
        raise RuntimeError("ai-broke-7")


class BrokenSetupGame(MultiGame):
    name = "brokentest-setup"
    description = "test game whose setup explodes"
    min_players = 1
    max_players = 1
    ai_seats = 0

    def new_state(self, rng, **kw):
        raise RuntimeError("setup-blew-up-9")


class ErrorSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()
        self.engine.register(BrokenMoveGame())
        self.engine.register(BrokenAIGame())
        self.engine.register(BrokenSetupGame())

    def tearDown(self):
        self.engine.shutdown()

    def test_broken_move_surfaces_real_error(self):
        self.engine.start("t:e1", "brokentest-move", ADA)
        out = self.engine.move("t:e1", "hello", ADA)
        self.assertTrue(out)
        self.assertIn("kaboom-42", out[0])
        self.assertIn("RuntimeError", out[0])
        # the table is still open — the player can quit or retry
        live = self.engine.live("t:e1")
        self.assertIsNotNone(live)
        self.assertEqual(live.status, "active")

    def test_broken_ai_turn_surfaces_real_error(self):
        room, _ = self.engine.start("t:e2", "brokentest-ai", ADA)
        ai_idx = next(i for i, p in enumerate(room.players) if p.is_ai)
        room.turn = ai_idx
        out: list[str] = []
        self.engine._pump_ai(room, out)  # must not hang or die silently
        self.assertTrue(any("errored" in m for m in out))

    def test_broken_setup_raises_and_leaves_no_orphan(self):
        with self.assertRaisesRegex(RuntimeError, "setup-blew-up-9"):
            self.engine.start("t:e3", "brokentest-setup", ADA)
        self.assertNotIn("t:e3", self.engine._rooms)
        self.assertIsNone(self.engine.live("t:e3"))


# ── bounded caches ───────────────────────────────────────────────────────────

class CacheBoundTests(unittest.TestCase):
    def test_mind_cache_is_bounded(self):
        mind = GameMind(suggest=lambda prompt: "flavor")
        for i in range(600):
            mind.ask(f"prompt {i}", cache_key=f"key-{i}")
        self.assertLessEqual(len(mind._cache), 512)

    def test_mind_cache_still_serves_fresh_hits(self):
        mind = GameMind(suggest=lambda prompt: "flavor")
        mind.ask("p", cache_key="k")
        self.assertEqual(mind.ask("p2", cache_key="k"), "flavor")


# ── case anti-repeat wiring ──────────────────────────────────────────────────

class CaseAntiRepeatTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_finished_case_is_recorded_seen(self):
        room, _ = self.engine.start("t:c1", "case", ADA)
        case_id = room.state["case"]["id"]
        tier = room.state["tier"]
        self.engine.quit("t:c1")  # finish → save_history
        hist = self.engine.games["case"].load_history(self.engine.store, ADA)
        seen = hist.get("seen", {}).get(tier, [])
        self.assertIn(case_id, seen)

    def test_unseen_filter_excludes_seen_bank_cases(self):
        from nomorals.games.games.cases import CASES, unseen_bank_cases
        easy_ids = [c["id"] for c in CASES if c.get("tier") == "easy"]
        self.assertTrue(easy_ids)
        remaining = unseen_bank_cases("easy", easy_ids[:-1])
        self.assertEqual([c["id"] for c in remaining], easy_ids[-1:])


if __name__ == "__main__":
    unittest.main()
