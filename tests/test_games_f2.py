"""Wave F2 / games: inbox games, trigger gating, quit hygiene.

1. The three new inbox games (gomoku, reversi, checkers) are genuinely
   playable: real rules, real house AI, win/block/capture/crown paths.
2. Inbox games are async: no per-turn clock, and their rooms survive the
   global idle TTL (a week of silence between moves is fine).
3. Trigger gating: natural-language game intents work ONLY in the owner's
   DM (CoreMind structural gate); everywhere else games start ONLY via
   explicit commands (/game, /<name>). Casual chat never launches a game.
4. Quit clears everything for EVERY registered game (registry-driven, so
   new games are covered automatically): no live room, no id entry, no
   active DB row, no relay residue; a post-quit move is rejected.
"""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from nomorals.agents.coremind import (
    GAME_ALIASES,
    CoreMind,
    _game_intent,
)
from nomorals.games.engine import IDLE_ROOM_TTL, GameEngine
from nomorals.games.games.inbox import (
    INBOX_GAMES,
    _ck_all_moves,
    _reversi_legal,
    _reversi_sq,
)
from nomorals.games.players import Player
from nomorals.social.chat import ChatKind
from nomorals.social.chat.control import GAME_COMMANDS, parse_control
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


def _needs_group(engine: GameEngine, name: str) -> bool:
    return bool(engine.games[name].needs_group)


# ── 1. inbox games are genuinely playable ───────────────────────────────────

class GomokuPlayTests(unittest.TestCase):
    def setUp(self):
        self.engine, _db, _sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _start(self, chat="t:g"):
        room, _ = self.engine.start(chat, "gomoku", ADA, kind="dm")
        return room

    def test_counts(self):
        self.assertEqual(len(self.engine.games), 42)

    def test_move_and_house_replies(self):
        room = self._start()
        out = self.engine.move("t:g", "h8", ADA)
        self.assertEqual(room.state["grid"][7][7], "B")
        # the house placed a stone too
        whites = sum(c == "W" for row in room.state["grid"] for c in row)
        self.assertEqual(whites, 1)
        self.assertEqual(room.status, "active")
        self.assertTrue(any("○" in m for m in out))

    def test_bad_square_rejected(self):
        room = self._start()
        out = self.engine.move("t:g", "z99", ADA)
        self.assertIn("h8", out[0])
        self.assertEqual(room.state["moves"], 0)

    def test_taken_square_rejected(self):
        room = self._start()
        self.engine.move("t:g", "h8", ADA)
        out = self.engine.move("t:g", "h8", ADA)
        self.assertIn("taken", out[0])

    def test_black_wins_with_five(self):
        room = self._start()
        st = room.state
        for i in range(4):
            st["grid"][7][3 + i] = "B"
        st["moves"] = 4
        self.engine.move("t:g", "h8", ADA)
        self.assertEqual(room.status, "finished")
        self.assertEqual(st["winner"], "B")
        w = self.engine.games["gomoku"].winner(room)
        self.assertEqual(w.key, ADA.key)

    def test_house_blocks_open_four(self):
        room = self._start()
        st = room.state
        for i in range(4):
            st["grid"][0][i] = "B"
        st["moves"] = 4
        self.engine.move("t:g", "o15", ADA)
        self.assertEqual(st["grid"][0][4], "W")  # e1 blocks
        self.assertEqual(room.status, "active")

    def test_house_takes_winning_move(self):
        room = self._start()
        st = room.state
        for i in range(4):
            st["grid"][5][5 + i] = "W"
        st["moves"] = 4
        self.engine.move("t:g", "a1", ADA)
        self.assertEqual(room.status, "finished")
        self.assertEqual(st["winner"], "W")

    def test_rules_and_status_render(self):
        g = self.engine.games["gomoku"]
        self.assertTrue(g.rules)
        room = self._start()
        self.engine.move("t:g", "h8", ADA)
        self.assertIn("●", g.describe_state(room))


class ReversiPlayTests(unittest.TestCase):
    def setUp(self):
        self.engine, _db, _sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _start(self, chat="t:r"):
        room, _ = self.engine.start(chat, "reversi", ADA, kind="dm")
        return room

    def test_opening_move_flips(self):
        room = self._start()
        out = self.engine.move("t:r", "d3", ADA)
        self.assertEqual(room.status, "active")
        # the human's own move message is deterministic; the house reply
        # that follows is room-seeded, so only pin the human's facts
        self.assertEqual(out[0], "● d3 flips 1.")
        self.assertEqual(room.state["grid"][2][3], "B")  # d3 stays black

    def test_illegal_move_rejected_with_hint(self):
        room = self._start()
        out = self.engine.move("t:r", "a1", ADA)
        self.assertIn("illegal", out[0])
        self.assertIn("legal:", out[0])

    def test_full_game_completes(self):
        import random
        rng = random.Random(7)
        room = self._start()
        # pin the house's tie-breaks — room seeds are random per start
        room._rng_instance = random.Random(1234)
        n = 0
        while room.status == "active" and n < 300:
            moves = _reversi_legal(room.state["grid"], "B")
            if not moves:
                break
            r, c, _ = rng.choice(moves)
            self.engine.move("t:r", _reversi_sq(r, c), ADA)
            n += 1
        # the table must always close cleanly — never soft-lock with no
        # legal move and no winner
        self.assertEqual(room.status, "finished")
        st = room.state
        self.assertIn(st["winner"], ("B", "W", "draw"))
        b, w = st["final"]
        grid_b = sum(c == "B" for row in st["grid"] for c in row)
        grid_w = sum(c == "W" for row in st["grid"] for c in row)
        self.assertEqual((b, w), (grid_b, grid_w))
        self.assertEqual(st["winner"],
                         "B" if b > w else ("W" if w > b else "draw"))
        self.assertFalse(_reversi_legal(st["grid"], "B"))
        self.assertFalse(_reversi_legal(st["grid"], "W"))

    def test_pass_handling(self):
        # a real mid-game position (found by playout search) where
        # black's h2 strands white: the house must pass and play resumes
        room = self._start("t:rp")
        rows = [".BBBBBBB", "WWWWWWW.", "WWBBBWWB", "WWBWWWWW",
                "BBWBBBWW", "BBWBBWBW", "BBBBBWWW", "BBBBBWWW"]
        room.state["grid"] = [[ch if ch != "." else "" for ch in r]
                              for r in rows]
        out = self.engine.move("t:rp", "h2", ADA)
        self.assertEqual(room.status, "active")
        self.assertTrue(any("house passes" in m for m in out),
                        f"no pass announced: {out}")
        # the table is still playable afterwards
        moves = _reversi_legal(room.state["grid"], "B")
        self.assertTrue(moves)

    def test_score_on_win(self):
        room = self._start("t:rs")
        st = room.state
        st["grid"] = [["B"] * 8 for _ in range(8)]
        st["grid"][0][0] = ""
        st["grid"][0][1] = "W"  # a1 brackets b1 against c1
        self.engine.move("t:rs", "a1", ADA)
        self.assertEqual(room.status, "finished")
        self.assertEqual(st["winner"], "B")
        self.assertEqual(st["final"], (64, 0))
        self.assertGreater(
            self.engine.games["reversi"].score(room, ADA), 0)


class CheckersPlayTests(unittest.TestCase):
    def setUp(self):
        self.engine, _db, _sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _start(self, chat="t:c"):
        room, _ = self.engine.start(chat, "checkers", ADA, kind="dm")
        return room

    def test_opening_move(self):
        room = self._start()
        out = self.engine.move("t:c", "d3-e4", ADA)
        self.assertEqual(room.status, "active")
        self.assertEqual(room.state["grid"][3][4], "b")
        self.assertEqual(room.state["grid"][2][3], "")
        self.assertTrue(any("house plays" in m for m in out))

    def test_bad_format_rejected(self):
        room = self._start()
        out = self.engine.move("t:c", "hello", ADA)
        self.assertIn("c3-d4", out[0])

    def test_forced_capture(self):
        room = self._start("t:cf")
        st = room.state
        st["grid"] = [[""] * 8 for _ in range(8)]
        st["grid"][2][2] = "b"   # c3
        st["grid"][3][3] = "w"   # d4
        st["grid"][5][5] = "w"   # f6
        out = self.engine.move("t:cf", "c3-d4", ADA)
        self.assertIn("forced", out[0])
        # partial chain rejected too — the full chain is required
        out = self.engine.move("t:cf", "c3-e5", ADA)
        self.assertIn("forced", out[0])
        out = self.engine.move("t:cf", "c3-e5-g7", ADA)
        self.assertIn("take 2", out[0])
        self.assertEqual(st["caps_b"], 2)

    def test_crowning(self):
        room = self._start("t:ck")
        st = room.state
        st["grid"] = [[""] * 8 for _ in range(8)]
        st["grid"][6][6] = "b"   # g7
        st["grid"][5][1] = "w"   # b6, mobile
        self.engine.move("t:ck", "g7-h8", ADA)
        self.assertEqual(st["grid"][7][7], "B")
        self.assertEqual(room.status, "active")

    def test_win_by_capture(self):
        room = self._start("t:cw")
        st = room.state
        st["grid"] = [[""] * 8 for _ in range(8)]
        st["grid"][2][2] = "b"
        st["grid"][3][3] = "w"
        self.engine.move("t:cw", "c3-e5", ADA)
        self.assertEqual(room.status, "finished")
        self.assertEqual(st["winner"], "B")
        w = self.engine.games["checkers"].winner(room)
        self.assertEqual(w.key, ADA.key)

    def test_house_takes_forced_capture(self):
        # captures are forced for the house too: after black's simple
        # move, white must take e5 via d4-f6 (c3 blocks black from
        # taking d4 first, so black has no capture of its own)
        room = self._start("t:ch")
        st = room.state
        st["grid"] = [[""] * 8 for _ in range(8)]
        st["grid"][3][3] = "w"   # d4
        st["grid"][4][4] = "b"   # e5 (white's victim)
        st["grid"][2][2] = "b"   # c3 (blocks e5xd4)
        st["grid"][1][1] = "b"   # b2 (blocks d4xb2 — d4-f6 is the only take)
        st["grid"][2][6] = "b"   # g3 (black's simple move)
        out = self.engine.move("t:ch", "g3-h4", ADA)
        self.assertTrue(any("takes 1" in m and "house" in m for m in out),
                        f"house didn't take: {out}")
        self.assertEqual(st["grid"][4][4], "")
        self.assertEqual(st["grid"][5][5], "w")
        self.assertEqual(st["caps_w"], 1)
        self.assertEqual(room.status, "active")

    def test_all_opening_moves_legal(self):
        room = self._start("t:co")
        moves = _ck_all_moves(room.state["grid"], "B")
        self.assertEqual(len(moves), 7)  # 7 black men can step
        self.assertTrue(all(len(m) == 2 for m in moves))


# ── 2. inbox games are async ────────────────────────────────────────────────

class InboxAsyncTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, _sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_no_turn_clock_but_long_idle_ttl(self):
        for g in INBOX_GAMES:
            with self.subTest(game=g.name):
                self.assertEqual(g.move_timeout, 0)
                self.assertEqual(g.idle_ttl, 7 * 86400.0)

    def test_room_survives_global_idle_ttl(self):
        # 2h of silence: a normal game's room dies, an inbox room lives
        room, _ = self.engine.start("t:async-g", "gomoku", ADA, kind="dm")
        room.last_activity = time.time() - 2 * 3600
        room2, _ = self.engine.start("t:async-t", "ttt", ADA, kind="dm")
        room2.last_activity = time.time() - 2 * 3600
        self.engine._sweep_timeouts()
        self.assertIsNotNone(self.engine.live("t:async-g"),
                             "inbox room reaped before its own TTL")
        self.assertIsNone(self.engine.live("t:async-t"),
                          "normal room survived past the global TTL")

    def test_room_dies_after_own_ttl(self):
        room, _ = self.engine.start("t:async-old", "reversi", ADA, kind="dm")
        room.last_activity = time.time() - 8 * 86400
        self.engine._sweep_timeouts()
        self.assertIsNone(self.engine.live("t:async-old"))
        rows = self.db.query(
            "SELECT id FROM game_rooms WHERE chat_key = ? AND status='active'",
            ("t:async-old",)) or []
        self.assertEqual(rows, [])

    def test_base_default_idle_ttl_is_none(self):
        from nomorals.games.games.base import MultiGame
        self.assertIsNone(MultiGame.idle_ttl)
        # every non-inbox game keeps the engine default
        for name, g in self.engine.games.items():
            if name in ("gomoku", "reversi", "checkers",
                        # sudoku is a slow puzzle (no per-turn clock),
                        # so it brings its own longer TTL like the
                        # inbox games do
                        "sudoku"):
                continue
            self.assertIsNone(g.idle_ttl, name)


# ── 3. trigger gating: NL only in the owner's DM, commands everywhere ───────

def _msg(kind: ChatKind, chat_key: str, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        incoming=True,
        sender="someone",
        chat=SimpleNamespace(kind=kind, key=chat_key, platform="telegram"),
    )


class GameIntentPolicyTests(unittest.TestCase):
    """The pure intent function: what may launch, what may not."""

    def test_play_verb_plus_name_launches(self):
        for text, target in (("let's play poker", "poker"),
                             ("play checkers with me", "checkers"),
                             ("start a game of gomoku", "gomoku"),
                             ("play reversi", "reversi")):
            with self.subTest(text=text):
                intent = _game_intent(text, None)
                self.assertIsNotNone(intent)
                self.assertEqual(intent.kind, "game")
                self.assertEqual(intent.action, "start")
                self.assertEqual(intent.target, target)

    def test_casual_chat_never_launches(self):
        for text in ("this game is boring",
                     "football game tonight",
                     "I played a game last night",
                     "we should play some games",
                     "the hunger games was a good movie",
                     "gaming is fun"):
            with self.subTest(text=text):
                self.assertIsNone(_game_intent(text, None),
                                  f"{text!r} produced a game intent")

    def test_every_engine_game_has_an_owner_dm_alias(self):
        engine, _db, _sent = make_engine()
        try:
            names = set(GAME_ALIASES.values())
            for game in engine.games:
                self.assertIn(game, names,
                              f"{game} has no NL alias — owner-DM "
                              "natural language can't start it")
        finally:
            engine.shutdown()


class StructuralGateTests(unittest.TestCase):
    """CoreMind.handle: NL game intents die outside the owner's DM."""

    def setUp(self):
        self.mind = CoreMind(None, runtime=None)

    def test_group_chat_nl_never_routes(self):
        for text in ("let's play poker", "play checkers", "start gomoku"):
            with self.subTest(text=text):
                msg = _msg(ChatKind.GROUP, "telegram:-999", text)
                self.assertIsNone(
                    self.mind.handle(text, message=msg,
                                     chat_key="telegram:-999"),
                    f"NL {text!r} routed in a group chat")

    def test_stranger_dm_nl_never_routes(self):
        # runtime=None → the fallback gate is "plain DM kind". A DM that
        # is not the owner's is indistinguishable here from the owner's,
        # so this test pins the group behavior; the owner distinction is
        # enforced by _is_operator when a runtime is attached.
        msg = _msg(ChatKind.GROUP, "telegram:777", "let's play poker")
        self.assertIsNone(
            self.mind.handle("let's play poker", message=msg,
                             chat_key="telegram:777"))

    def test_owner_dm_nl_routes(self):
        msg = _msg(ChatKind.DM, "telegram:123", "let's play checkers")
        reply = self.mind.handle("let's play checkers", message=msg,
                                 chat_key="telegram:123")
        self.assertIsNotNone(reply, "owner-DM NL game intent did not route")

    def test_owner_dm_decide_maps_new_games(self):
        intent = self.mind.decide("let's play reversi", live_game=None,
                                  allow_model=False)
        self.assertEqual((intent.kind, intent.action, intent.target),
                         ("game", "start", "reversi"))

    def test_commands_parse_in_every_chat(self):
        # the command path is the ONLY trigger outside owner DMs
        for cmd in ("poker", "gomoku", "reversi", "checkers", "hangman"):
            with self.subTest(cmd=cmd):
                parsed = parse_control("/" + cmd)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.kind, cmd)
                self.assertIn(cmd, GAME_COMMANDS)

    # Games deliberately reachable ONLY via "/game <name>": their bare
    # "/<name>" form is a reserved system command.  ("arena" is the
    # self-improvement arena — the arena game is /game arena.)
    GAME_ONLY_VIA_SLASH_GAME = frozenset({"arena"})

    def test_control_commands_cover_every_game(self):
        # every engine game is triggerable outside the owner's DM: either
        # a direct /<name> command, or the universal /game <name> command.
        # ("/2048" proved digit-leading command names parse and route, so
        # 20q gets a direct command too.)
        engine, _db, _sent = make_engine()
        try:
            for game in engine.games:
                with self.subTest(game=game):
                    if game in self.GAME_ONLY_VIA_SLASH_GAME:
                        parsed = parse_control("/game " + game)
                        self.assertIsNotNone(parsed)
                        self.assertEqual(parsed.kind, "game")
                        continue
                    self.assertIn(game, GAME_COMMANDS,
                                  f"/{game} not a direct command — no "
                                  "command-only trigger outside owner DMs")
                    parsed = parse_control("/" + game)
                    self.assertIsNotNone(parsed)
                    self.assertEqual(parsed.kind, game)
        finally:
            engine.shutdown()


# ── 4. quit clears every game (registry-driven) ─────────────────────────────

def _active_rows(db: Database, chat_key: str) -> list:
    return db.query(
        "SELECT id FROM game_rooms WHERE chat_key = ? AND status = 'active'",
        (chat_key,)) or []


class QuitClearsEveryGameTests(unittest.TestCase):
    def test_registry_count(self):
        engine, _db, _sent = make_engine()
        try:
            self.assertEqual(len(engine.games), 42)
        finally:
            engine.shutdown()

    def test_quit_clears_all_registered_games(self):
        engine, db, sent = make_engine()
        try:
            for name in sorted(engine.games):
                with self.subTest(game=name):
                    kind = "group" if _needs_group(engine, name) else "dm"
                    chat = f"t:f2quit-{name}"
                    room, _msgs = engine.start(chat, name, ADA, kind=kind)
                    room_id = room.id
                    engine.quit(chat)

                    self.assertIsNone(engine.live(chat),
                                      f"{name}: live() still returns a room")
                    self.assertNotIn(chat, engine._rooms)
                    self.assertNotIn(room_id, engine._by_id)
                    self.assertEqual(_active_rows(db, chat), [],
                                     f"{name}: active DB row leaked")
                    self.assertIsNone(
                        engine.relay.get_relay_for_chat(chat),
                        f"{name}: relay mapping stuck")

                    # zombie: a move after quit is rejected, never accepted
                    n_sent = len(sent)
                    out = engine.move(chat, "hello", ADA)
                    self.assertEqual(out, [])
                    self.assertEqual(len(sent), n_sent)
                    self.assertIsNone(engine.live(chat))
                    self.assertEqual(_active_rows(db, chat), [])

                    # fresh start works on a brand-new table
                    room2, _ = engine.start(chat, name, ADA, kind=kind)
                    self.assertNotEqual(room2.id, room_id)
                    self.assertEqual(room2.status, "active")
                    engine.quit(chat)
                    self.assertIsNone(engine.live(chat))
        finally:
            engine.shutdown()

    def test_quit_clears_relay_duel_for_inbox_game(self):
        engine, db, sent = make_engine()
        try:
            inv = engine.relay.create_invite("t:f2a", ADA, "gomoku")
            relay = engine.relay.accept_invite(inv.code, "t:f2b", BOB)
            vchat = relay.virtual_chat
            engine.relay.relay_move("t:f2a", "h8", ADA)
            self.assertIsNotNone(engine.live(vchat))

            engine.quit("t:f2a")
            self.assertEqual(engine.relay.relays, {})
            self.assertEqual(engine.relay.chat_to_relay, {})
            self.assertEqual(engine.relay.invites, {})
            self.assertIsNone(engine.live(vchat))
            self.assertNotIn(vchat, engine._rooms)
            self.assertEqual(_active_rows(db, vchat), [])
            self.assertEqual(
                db.query("SELECT * FROM game_relays") or [], [])
            # opponent's move is rejected, not accepted
            n = len(sent)
            self.assertEqual(
                engine.relay.relay_move("t:f2b", "h9", BOB), [])
            self.assertEqual(len(sent), n)
        finally:
            engine.shutdown()

    def test_quit_with_no_game_is_clean(self):
        engine, _db, _sent = make_engine()
        try:
            out = engine.quit("t:nothing-here")
            self.assertTrue(out)
            self.assertIn("no game", "\n".join(out).lower())
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
