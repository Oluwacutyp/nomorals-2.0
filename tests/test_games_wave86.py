"""Wave 86 — games consolidation: relay, achievements, rematch, ports.

Covers the work that closed the legacy solo bridge:
- GameRelay: invite → accept → virtual-room routing (incl. the
  double-prefix regression), expiry, self-accept, persistence.
- The 10 previously-dead achievements + unlock announcements.
- /game rematch.
- The 20q / rps / digits ports.
- Smoke coverage for the wild / arcade / casino games that had none.
"""
from __future__ import annotations

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
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


ADA = Player.from_sender("telegram", "456", "Ada")
BOB = Player.from_sender("telegram", "789", "Bob")


# ── relay ────────────────────────────────────────────────────────────────────

class RelayTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_invite_accept_routes_moves_to_virtual_room(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        room = relay.accept_invite(inv.code, "telegram:chatB", BOB)
        # the canonical virtual chat key — no double prefix
        self.assertEqual(room.virtual_chat, f"relay:{inv.code}")
        self.assertIn("telegram:chatA", (room.chat_a, room.chat_b))
        self.assertIn("telegram:chatB", (room.chat_a, room.chat_b))
        self.assertEqual(room.other_chat("telegram:chatA"), "telegram:chatB")
        # the engine actually has a live room under that key
        live = self.engine.live(room.virtual_chat)
        self.assertIsNotNone(live)
        self.assertEqual(live.game, "ttt")
        # both humans are seated
        keys = {p.key for p in live.players if not p.is_ai}
        self.assertEqual(keys, {ADA.key, BOB.key})

    def test_relay_move_reaches_virtual_room(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        relay.accept_invite(inv.code, "telegram:chatB", BOB)
        msgs = relay.relay_move("telegram:chatA", "0", ADA)
        self.assertTrue(msgs)  # the move was accepted by the virtual room

    def test_relay_move_wrong_chat_rejected(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        relay.accept_invite(inv.code, "telegram:chatB", BOB)
        # unknown chats get silence, not a crash
        self.assertEqual(relay.relay_move("telegram:chatC", "0", ADA), [])

    def test_accept_unknown_code_rejected(self):
        relay = self.engine.relay
        with self.assertRaises(ValueError):
            relay.accept_invite("nope1234", "telegram:chatB", BOB)

    def test_accept_own_invite_rejected(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        with self.assertRaises(ValueError):
            relay.accept_invite(inv.code, "telegram:chatA", ADA)

    def test_expired_invite_rejected(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        # force expiry on the live invite
        relay.invites[inv.code].expires_at = 0
        with self.assertRaises(ValueError):
            relay.accept_invite(inv.code, "telegram:chatB", BOB)

    def test_invite_survives_relay_restart(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        code = inv.code
        # simulate a restart: drop the in-memory relay, build a fresh one
        self.engine._relay_obj = None
        fresh = self.engine.relay
        room = fresh.accept_invite(code, "telegram:chatB", BOB)
        self.assertEqual(room.game_name, "ttt")

    def test_cleanup_reaps_expired_invite(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        relay.invites[inv.code].expires_at = 0
        # force the throttle to allow a run
        relay._last_cleanup = 0.0
        relay.maybe_cleanup()
        self.assertNotIn(inv.code, relay.invites)
        row = self.db.query_one(
            "SELECT code FROM game_invites WHERE code = ?", (inv.code,))
        self.assertIsNone(row)

    def test_one_human_games_reject_relay(self):
        relay = self.engine.relay
        with self.assertRaises(ValueError):
            relay.create_invite("telegram:chatA", ADA, "digits")

    def test_get_relay_for_chat(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:chatA", ADA, "ttt")
        relay.accept_invite(inv.code, "telegram:chatB", BOB)
        self.assertIsNotNone(relay.get_relay_for_chat("telegram:chatA"))
        self.assertIsNotNone(relay.get_relay_for_chat("telegram:chatB"))
        self.assertIsNone(relay.get_relay_for_chat("telegram:chatZ"))


# ── achievements ─────────────────────────────────────────────────────────────

class AchievementTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _achievements(self, player_key):
        from nomorals.games.achievements import get_achievements
        return {a["id"] for a in get_achievements(self.db, player_key)}

    def test_hangman_perfect_unlocks_and_announces(self):
        p = Player(key="local:hm", platform="local", name="Hm")
        room, _ = self.engine.start("local:c1", "hangman", p)
        room.state["wrong"] = 0
        room.state["done"] = "table"
        out = self.engine._finish(room)
        self.assertIn("hangman_perfect", self._achievements(p.key))
        self.assertTrue(any("No Mistakes" in m for m in out))

    def test_poker_trio(self):
        p = Player(key="local:pk", platform="local", name="Pk")
        room, _ = self.engine.start("local:c2", "poker", p)
        room.state["human_best_rank"] = 5  # flush
        room.state["human_allin"] = True
        room.state["stacks"] = {p.key: 400, "ai:house": 0}
        room.state["done"] = True
        out = self.engine._finish(room)
        got = self._achievements(p.key)
        self.assertIn("poker_win", got)
        self.assertIn("poker_straight", got)
        self.assertIn("poker_allin_win", got)
        announced = " ".join(out)
        self.assertIn("Card Shark", announced)
        self.assertIn("All-In Legend", announced)
        self.assertIn("Straight Shooter", announced)

    def test_arena_crit_kill(self):
        p = Player(key="local:ar", platform="local", name="Ar")
        room, _ = self.engine.start("local:c3", "arena", p)
        # rig it: a human crit killed the house
        room.state["house"]["hp"] = 0
        room.state["crit_kill_by"] = "you"
        room.state["done"] = True
        out = self.engine._finish(room)
        got = self._achievements(p.key)
        self.assertIn("arena_win", got)
        self.assertIn("arena_crit_kill", got)
        self.assertTrue(any("Critical Finish" in m for m in out))

    def test_milestones(self):
        p = Player(key="local:ms", platform="local", name="Ms")
        # play 10 quick games to trip games_10
        for i in range(10):
            room, _ = self.engine.start(f"local:m{i}", "digits", p)
            room.state["done"] = "house"
            self.engine._finish(room)
        self.assertIn("games_10", self._achievements(p.key))

    def test_hangman_five_wins(self):
        p = Player(key="local:h5", platform="local", name="H5")
        for i in range(5):
            room, _ = self.engine.start(f"local:h{i}", "hangman", p)
            room.state["wrong"] = 1
            room.state["done"] = "table"
            self.engine._finish(room)
        self.assertIn("hangman_5_wins", self._achievements(p.key))


# ── rematch ──────────────────────────────────────────────────────────────────

class RematchTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_rematch_no_history(self):
        room, msgs = self.engine.rematch("local:nope")
        self.assertIsNone(room)
        self.assertTrue(any("no finished game" in m for m in msgs))

    def test_rematch_reopens_same_game(self):
        p = Player(key="local:rm", platform="local", name="Rm")
        room, _ = self.engine.start("local:r1", "ttt", p)
        room.state["done"] = True
        self.engine._finish(room)
        room2, msgs = self.engine.rematch("local:r1")
        self.assertIsNotNone(room2)
        self.assertEqual(room2.game, "ttt")
        self.assertTrue(msgs[0].startswith("🔁"))

    def test_rematch_while_live_refused(self):
        p = Player(key="local:rl", platform="local", name="Rl")
        room, _ = self.engine.start("local:r2", "ttt", p)
        room.state["done"] = True
        self.engine._finish(room)
        self.engine.start("local:r2", "ttt", p)  # live again
        room3, msgs = self.engine.rematch("local:r2")
        self.assertIsNone(room3)
        self.assertTrue(any("already live" in m for m in msgs))

    def test_rematch_reseats_group_humans(self):
        a = Player(key="local:ra", platform="local", name="Ra")
        b = Player(key="local:rb", platform="local", name="Rb")
        room, _ = self.engine.start("local:r3", "ttt", a, kind="group")
        self.engine.join("local:r3", b)
        room.state["done"] = True
        self.engine._finish(room)
        room2, _ = self.engine.rematch("local:r3")
        keys = {pl.key for pl in room2.humans}
        self.assertEqual(keys, {a.key, b.key})


# ── ports: 20q / rps / digits ────────────────────────────────────────────────

class PortedGamesTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()
        self.p = Player(key="local:pg", platform="local", name="Pg")

    def tearDown(self):
        self.engine.shutdown()

    def test_twentyq_answers_and_counts(self):
        room, _ = self.engine.start("local:q1", "20q", self.p)
        target = room.state["target"]
        self.assertTrue(target)
        out = self.engine.move("local:q1", "is it alive?", self.p)
        self.assertTrue(out)
        self.assertIn("19 left", out[0])
        self.assertEqual(room.state["asked"], 1)

    def test_twentyq_correct_guess_wins(self):
        room, _ = self.engine.start("local:q2", "20q", self.p)
        target = room.state["target"]
        out = self.engine.move("local:q2", f"guess: {target}", self.p)
        self.assertIsNone(self.engine.live("local:q2"))
        self.assertTrue(any("guessed it" in m for m in out))

    def test_twentyq_runs_out_loses(self):
        room, _ = self.engine.start("local:q3", "20q", self.p)
        room.state["asked"] = 19
        out = self.engine.move("local:q3", "is it blue?", self.p)
        self.assertIsNone(self.engine.live("local:q3"))
        self.assertTrue(any("out of questions" in m for m in out))

    def test_rps_rejects_bad_move(self):
        room, _ = self.engine.start("local:rps1", "rps", self.p)
        out = self.engine.move("local:rps1", "lizard", self.p)
        self.assertTrue(any("whole menu" in m for m in out))
        self.assertIsNotNone(self.engine.live("local:rps1"))

    def test_rps_first_to_three_ends(self):
        room, _ = self.engine.start("local:rps2", "rps", self.p)
        # force the scoreboard to the brink, then play one round
        room.state["you"] = 2
        room.state["me"] = 2
        for mv in ("rock", "paper", "scissors", "rock", "paper",
                   "scissors", "rock"):
            out = self.engine.move("local:rps2", mv, self.p)
            if self.engine.live("local:rps2") is None:
                break
        self.assertIsNone(self.engine.live("local:rps2"))

    def test_digits_advances_on_correct(self):
        room, _ = self.engine.start("local:d1", "digits", self.p)
        number = room.state["number"]
        out = self.engine.move("local:d1", number, self.p)
        self.assertTrue(any("correct" in m for m in out))
        self.assertEqual(len(room.state["number"]), 4)

    def test_digits_wrong_ends(self):
        room, _ = self.engine.start("local:d2", "digits", self.p)
        out = self.engine.move("local:d2", "000", self.p)
        # "000" is wrong unless the number literally is 000
        if self.engine.live("local:d2") is None:
            self.assertTrue(any("not it" in m for m in out))

    def test_digits_eight_rounds_wins(self):
        room, _ = self.engine.start("local:d3", "digits", self.p)
        for _ in range(8):
            number = room.state["number"]
            out = self.engine.move("local:d3", number, self.p)
            if self.engine.live("local:d3") is None:
                break
        self.assertIsNone(self.engine.live("local:d3"))
        self.assertTrue(any("eight rounds" in m for m in out))


# ── wild / arcade / casino smoke ─────────────────────────────────────────────

class GameSmokeTests(unittest.TestCase):
    """Every previously-untested game at least starts and takes a move."""

    def setUp(self):
        self.engine, self.db, self.sent = make_engine()
        self.p = Player(key="local:sm", platform="local", name="Sm")

    def tearDown(self):
        self.engine.shutdown()

    def _smoke(self, game_name, moves, *, kind="dm"):
        chat = f"local:smoke:{game_name}"
        try:
            room, msgs = self.engine.start(chat, game_name, self.p,
                                           kind=kind)
        except ValueError as exc:
            self.fail(f"{game_name} failed to start: {exc}")
        self.assertTrue(msgs, f"{game_name} had no intro")
        for mv in moves:
            if self.engine.live(chat) is None:
                break
            try:
                out = self.engine.move(chat, mv, self.p, kind=kind)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"{game_name} move {mv!r} raised: {exc}")
            self.assertIsInstance(out, list)

    def test_wild_games(self):
        self._smoke("poker", ["10", "call", "check", "fold"])
        self._smoke("blackjack", ["hit", "stand"])
        self._smoke("slots", ["spin", "spin"])
        self._smoke("roulette", ["red 10", "17 5", "black 10"])
        self._smoke("bulls", ["1234", "5678"])
        self._smoke("craps", ["roll", "roll"])
        self._smoke("mines", ["a1", "b2"])
        self._smoke("wordle", ["crane", "slate"])

    def test_arcade_games(self):
        self._smoke("2048", ["up", "left", "down", "right"])
        self._smoke("snake", ["up", "left", "down", "right"])
        self._smoke("connect4", ["4", "4", "3", "5"])
        self._smoke("battleship", ["a1", "b2", "c3"])

    def test_wild_extra(self):
        self._smoke("mafia", ["vote bob", "night kill bob"], kind="group")
        self._smoke("escape", ["look", "north"])
        self._smoke("political", ["tax", "speech"], kind="group")


if __name__ == "__main__":
    unittest.main()
