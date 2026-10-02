"""Wave G1: direct /20q, /digits, /rps commands.

Three engine games (TwentyQuestionsGame "20q", RpsGame "rps",
DigitMemoryGame "digits") lived in the engine and were NL-startable in the
owner's DM and /game-startable anywhere, but had no direct /<name> command.
The old F2 carve-out ("Telegram commands must start with a letter, so these
stay /game-only") is stale: /2048 shipped in wave 97 and digit-leading
command names parse fine (parse_control is a plain dict lookup, no regex).

This file pins the wave-87 contract for the three new commands:

1. `/20q`, `/digits`, `/rps` parse to their own kinds and are in
   GAME_COMMANDS — so the any-chat dispatch in PartnerRuntime._process
   routes them to _control_game in EVERY chat, including non-owner chats.
2. Each one actually starts its engine game (the verb the dispatcher builds
   — kind + tail, exactly as _process does — reaches engine.start via
   PartnerRuntime._control_game with a real GameEngine).
3. `/game 20q` / `/game digits` / `/game rps` still work (the universal path).
4. Natural language still only starts them in the owner's DM: NL in a group
   chat never routes; NL in the owner DM still maps to the right game.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.coremind import CoreMind
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player
from nomorals.social.chat import ChatKind
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    GAME_COMMANDS,
    parse_control,
)
from nomorals.storage.db import Database


COMMANDS = ("20q", "digits", "rps")
ADA = Player.from_sender("telegram", "456", "Ada")


def make_engine() -> GameEngine:
    db = Database(":memory:")
    db.migrate()
    return GameEngine(SimpleNamespace(db=db), send=lambda c, t: None)


def _msg(kind: ChatKind, chat_key: str, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        incoming=True,
        sender="someone",
        chat=SimpleNamespace(kind=kind, key=chat_key, platform="telegram"),
    )


def _dispatch_verb(command) -> str:
    """Replicate the any-chat verb building in PartnerRuntime._process."""
    return (command.tail or command.arg) if command.kind == "game" \
        else command.kind + ((" " + command.tail) if command.tail else "")


class ParseTests(unittest.TestCase):
    def test_direct_commands_parse_to_their_kind(self):
        for cmd in COMMANDS:
            with self.subTest(cmd=cmd):
                parsed = parse_control("/" + cmd)
                self.assertIsNotNone(parsed, f"/{cmd} did not parse")
                self.assertEqual(parsed.kind, cmd)

    def test_digit_leading_command_parses(self):
        # the naming decision this wave hinged on: parse_control must accept
        # a command whose name starts with a digit
        parsed = parse_control("/20q")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kind, "20q")

    def test_commands_in_game_dispatch_tables(self):
        for cmd in COMMANDS:
            with self.subTest(cmd=cmd):
                self.assertIn(cmd, GAME_COMMANDS)
                self.assertIn(cmd, CONTROL_COMMANDS)
                self.assertIn(cmd, COMMAND_DETAILS)

    def test_game_command_path_in_process_gate(self):
        # the exact gating condition in PartnerRuntime._process: a parsed
        # command is game-routed iff kind == "game" or kind in GAME_COMMANDS
        for cmd in COMMANDS:
            with self.subTest(cmd=cmd):
                parsed = parse_control("/" + cmd)
                self.assertIsNotNone(parsed)
                self.assertTrue(
                    parsed.kind == "game" or parsed.kind in GAME_COMMANDS,
                    f"/{cmd} would not reach _control_game")


class DirectCommandStartsGameTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _start_via_dispatch(self, text: str, chat_key: str = "telegram:-999") -> str:
        """Drive /<name> exactly the way _process does: parse, gate on
        GAME_COMMANDS, build the verb, hand it to _control_game."""
        parsed = parse_control(text)
        self.assertIsNotNone(parsed, f"{text} did not parse")
        self.assertIn(parsed.kind, GAME_COMMANDS,
                      f"{text} not routed to the any-chat game path")
        verb = _dispatch_verb(parsed)
        engine = self.engine
        stub = SimpleNamespace(context=SimpleNamespace(),
                               _game_engine=lambda: engine)
        return PartnerRuntime._control_game(
            stub, verb, chat_key=chat_key, player=ADA, kind="group")

    def test_20q_starts_its_game(self):
        reply = self._start_via_dispatch("/20q")
        room = self.engine.live("telegram:-999")
        self.assertIsNotNone(room, "no game room after /20q")
        self.assertEqual(room.game, "20q")
        self.assertIn("🎮", reply)

    def test_digits_starts_its_game(self):
        reply = self._start_via_dispatch("/digits")
        room = self.engine.live("telegram:-999")
        self.assertIsNotNone(room, "no game room after /digits")
        self.assertEqual(room.game, "digits")
        self.assertIn("🎮", reply)

    def test_rps_starts_its_game(self):
        reply = self._start_via_dispatch("/rps")
        room = self.engine.live("telegram:-999")
        self.assertIsNotNone(room, "no game room after /rps")
        self.assertEqual(room.game, "rps")
        self.assertIn("🎮", reply)


class GameCommandPathStillWorksTests(unittest.TestCase):
    """/game <name> must keep working alongside the new direct commands."""

    def setUp(self):
        self.engine = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_game_slash_name_still_starts(self):
        for cmd in COMMANDS:
            with self.subTest(cmd=cmd):
                chat_key = f"telegram:{cmd}"
                parsed = parse_control("/game " + cmd)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.kind, "game")
                verb = _dispatch_verb(parsed)
                self.assertEqual(verb, cmd)
                engine = self.engine
                stub = SimpleNamespace(context=SimpleNamespace(),
                                       _game_engine=lambda: engine)
                reply = PartnerRuntime._control_game(
                    stub, verb, chat_key=chat_key, player=ADA, kind="dm")
                room = self.engine.live(chat_key)
                self.assertIsNotNone(room, f"no game room after /game {cmd}")
                self.assertEqual(room.game, cmd)
                self.assertIn("🎮", reply)


class NLGatingTests(unittest.TestCase):
    """Wave-87 contract: natural language never starts these outside the
    owner's DM; inside the owner's DM it still maps to the right game."""

    def setUp(self):
        self.mind = CoreMind(None, runtime=None)

    def test_group_nl_never_routes(self):
        for text in ("let's play 20q", "play digits", "start rps"):
            with self.subTest(text=text):
                msg = _msg(ChatKind.GROUP, "telegram:-999", text)
                self.assertIsNone(
                    self.mind.handle(text, message=msg,
                                     chat_key="telegram:-999"),
                    f"NL {text!r} routed in a group chat")

    def test_owner_dm_nl_maps_to_game(self):
        for text, target in (("let's play 20q", "20q"),
                             ("play digit memory", "digits"),
                             ("let's play rock paper scissors", "rps")):
            with self.subTest(text=text):
                intent = self.mind.decide(text, live_game=None,
                                          allow_model=False)
                self.assertEqual((intent.kind, intent.action, intent.target),
                                 ("game", "start", target))

    def test_owner_dm_nl_still_routes(self):
        msg = _msg(ChatKind.DM, "telegram:123", "let's play 20q")
        reply = self.mind.handle("let's play 20q", message=msg,
                                 chat_key="telegram:123")
        self.assertIsNotNone(reply, "owner-DM NL game intent did not route")


if __name__ == "__main__":
    unittest.main()
