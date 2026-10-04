"""Wave G1: regression tests for the command routing table.

The ``/arena`` routing collision (fixed by coordinator): ``"arena"`` was a
member of ``GAME_COMMANDS``, so the any-chat game dispatch in
``PartnerRuntime`` intercepted ``/arena`` into ``_control_game`` before the
owner control dispatch could ever reach ``_control_arena`` — the
self-improvement arena was unreachable from chat. The fix removed
``"arena"`` from ``GAME_COMMANDS``. The routing contract is now:

  * ``/arena ...``      → parse kind ``"arena"`` → owner dispatch → ``_control_arena``
  * ``/game arena``     → parse kind ``"game"``  → any-chat dispatch → ``_control_game``
    (verb ``"arena"``, i.e. the arena *game* in ``nomorals/games``)

Every test reads the REAL routing tables (``parse_control``,
``GAME_COMMANDS``, ``CONTROL_COMMANDS``, the real ``handle_control``
dispatch chain) — never a copy. If ``"arena"`` is re-added to
``GAME_COMMANDS``, the collision tests fail. The table-driven game tests
iterate the live ``GAME_COMMANDS`` tuple, so new games (``/20q``,
``/digits``, ``/rps``, …) are covered automatically the moment they land.
"""

from __future__ import annotations

import unittest

from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    GAME_COMMANDS,
    parse_control,
)

#: Reserved system commands that must never be swallowed by the any-chat
#: game dispatch.  They are owner-level controls (or catalogued kinds), not
#: game starters.
RESERVED_SYSTEM_COMMANDS = (
    "arena", "mission", "upgrade", "deliver", "proposals",
    "approve", "deny", "features", "trial", "book",
)


def any_chat_game_route(text: str) -> str:
    """Mirror the any-chat game dispatch decision in
    ``nomorals/agents/partner_runtime.py`` (the
    ``if message.incoming and message.text.strip().startswith("/")`` block):
    a parsed command is intercepted into ``_control_game`` when its kind is
    ``"game"`` or a direct game-start command in ``GAME_COMMANDS``.

    Returns ``"control_game"`` or ``"not_intercepted"``.

    Keep the boolean expression in sync with the source.  It reads the live
    ``GAME_COMMANDS`` on purpose: the routing must move with the table, so
    the collision tests fail if the table regresses.
    """
    command = parse_control(text)
    if command is not None and (command.kind == "game"
                                or command.kind in GAME_COMMANDS):
        return "control_game"
    return "not_intercepted"


def any_chat_game_verb(text: str) -> str:
    """Mirror the verb construction used by the any-chat game dispatch
    (``partner_runtime.py``): ``/game <verb>`` forwards the tail, a direct
    command forwards its own kind.
    """
    command = parse_control(text)
    assert command is not None, f"{text!r} did not parse"
    if command.kind == "game":
        return command.tail or command.arg
    return command.kind + ((" " + command.tail) if command.tail else "")


class _RuntimeStub:
    """Minimal stand-in that lets a test call the REAL
    ``PartnerRuntime.handle_control`` dispatch chain while recording which
    leaf handler it chose.

    ``handle_control``'s kind checks are pure comparisons until the chosen
    branch fires, so for kinds ``"arena"`` and ``"game"`` only the two leaf
    handlers (plus the player helper used by the ``"game"`` branch) need
    to exist on the stub.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def _control_arena(self, tail: str, chat_key: str = "", **kwargs) -> str:
        self.calls.append(("arena", tail, chat_key))
        return "arena-ok"

    def _control_game(self, tail: str, chat_key: str = "", **kwargs) -> str:
        self.calls.append(("game", tail, chat_key))
        return "game-ok"

    def _game_player_for_key(self, chat_key: str) -> str:
        return "stub-player"


def owner_dispatch(text: str, chat_key: str = "test-chat"):
    """Run the real ``PartnerRuntime.handle_control`` dispatch chain on a
    stub runtime.  Returns ``(reply, calls)`` where each call is
    ``(handler, tail, chat_key)`` with handler in ``{"arena", "game"}``.
    """
    from nomorals.agents.partner_runtime import PartnerRuntime

    stub = _RuntimeStub()
    reply = PartnerRuntime.handle_control(stub, text, chat_key)
    return reply, stub.calls


def engine_game_names() -> set[str]:
    """Every game name the engine registry knows, by the same route the
    engine itself uses (``GameEngine._register_builtins``)."""
    from nomorals.games.games import (  # noqa: PLC0415
        ambitious, arcade, casino, easy, inbox, medium, wild,
    )

    registries = (
        easy.EASY_GAMES, medium.MEDIUM_GAMES, ambitious.AMBITIOUS_GAMES,
        wild.WILD_GAMES, arcade.ARCADE_GAMES, casino.CASINO_GAMES,
        inbox.INBOX_GAMES,
    )
    return {game.name for registry in registries for game in registry}


class ArenaCollisionTests(unittest.TestCase):
    """The /arena routing collision must never come back."""

    def test_arena_parses_to_kind_arena(self):
        for text in ("/arena", "/arena status", "/arena run escape",
                     "/arena topics", "/arena approve x1"):
            with self.subTest(text=text):
                command = parse_control(text)
                self.assertIsNotNone(command)
                self.assertEqual(command.kind, "arena")

    def test_arena_not_in_game_commands(self):
        # The one-line fix.  Fails the moment someone re-adds "arena".
        self.assertNotIn("arena", GAME_COMMANDS)

    def test_arena_not_intercepted_by_any_chat_game_dispatch(self):
        # THE regression test: with "arena" back in GAME_COMMANDS the
        # any-chat path intercepts "/arena" into _control_game (it runs
        # before the owner dispatch) and the self-improvement arena is
        # unreachable again.  This mirrors the exact dispatch condition,
        # so it fails on the regression.
        for text in ("/arena", "/arena status", "/arena run",
                     "/arena topics", "/arena stream 5"):
            with self.subTest(text=text):
                self.assertEqual(any_chat_game_route(text), "not_intercepted")

    def test_arena_owner_dispatch_reaches_control_arena(self):
        # The real handle_control chain: kind "arena" → _control_arena.
        reply, calls = owner_dispatch("/arena status")
        self.assertEqual(reply, "arena-ok")
        self.assertEqual(calls, [("arena", "status", "test-chat")])

    def test_arena_bare_owner_dispatch_reaches_control_arena(self):
        reply, calls = owner_dispatch("/arena")
        self.assertEqual(reply, "arena-ok")
        self.assertEqual(calls, [("arena", "", "test-chat")])
        self.assertFalse(any(handler == "game" for handler, _, _ in calls),
                         "_control_game must never fire for /arena")


class GameArenaRoutingTests(unittest.TestCase):
    """``/game arena`` is the arena GAME — it must keep working."""

    def test_game_arena_parses_to_kind_game(self):
        command = parse_control("/game arena")
        self.assertIsNotNone(command)
        self.assertEqual(command.kind, "game")
        self.assertEqual(command.tail, "arena")

    def test_game_arena_hits_any_chat_game_dispatch(self):
        self.assertEqual(any_chat_game_route("/game arena"), "control_game")

    def test_game_arena_verb_is_the_arena_game(self):
        # The any-chat dispatch forwards the tail as the verb; "arena"
        # must be a real engine game name (see test_arena_game_registered).
        self.assertEqual(any_chat_game_verb("/game arena"), "arena")

    def test_arena_game_registered_in_engine(self):
        self.assertIn("arena", engine_game_names())

    def test_game_arena_owner_dispatch_reaches_control_game(self):
        reply, calls = owner_dispatch("/game arena")
        self.assertEqual(reply, "game-ok")
        self.assertEqual(calls, [("game", "arena", "test-chat")])


class GameCommandsRoutingTests(unittest.TestCase):
    """Every direct game command must parse and hit the any-chat game path.

    Table-driven over the live GAME_COMMANDS tuple: new games (/20q,
    /digits, /rps, …) are covered automatically the moment they are added
    to the tuple.
    """

    def test_every_game_command_parses_to_itself(self):
        for name in GAME_COMMANDS:
            with self.subTest(command=name):
                command = parse_control("/" + name)
                self.assertIsNotNone(command, f"/{name} must parse")
                self.assertNotEqual(command.kind, "error",
                                    f"/{name} must not be an arg error")
                self.assertEqual(command.kind, name)

    def test_every_game_command_hits_any_chat_game_dispatch(self):
        for name in GAME_COMMANDS:
            with self.subTest(command=name):
                self.assertEqual(any_chat_game_route("/" + name),
                                 "control_game",
                                 f"/{name} must reach _control_game in any chat")

    def test_game_commands_all_in_control_catalog(self):
        # parse_control returns None for unknown kinds; a game command
        # missing from CONTROL_COMMANDS could never trigger the any-chat
        # dispatch.
        for name in GAME_COMMANDS:
            with self.subTest(command=name):
                self.assertIn(name, CONTROL_COMMANDS)

    def test_direct_game_command_verb_is_its_kind(self):
        self.assertEqual(any_chat_game_verb("/hangman"), "hangman")
        self.assertEqual(any_chat_game_verb("/mafia"), "mafia")

    def test_game_itself_not_in_game_commands(self):
        # "game" routes via kind == "game", not via tuple membership.
        self.assertNotIn("game", GAME_COMMANDS)
        self.assertEqual(any_chat_game_route("/game list"), "control_game")


class ReservedSystemCommandsTests(unittest.TestCase):
    """Reserved system commands must never be intercepted by the game
    dispatch — they are owner-level controls, not game starters."""

    def test_reserved_commands_not_intercepted_by_game_dispatch(self):
        for name in RESERVED_SYSTEM_COMMANDS:
            with self.subTest(command=name):
                command = parse_control("/" + name)
                self.assertIsNotNone(command,
                                      f"/{name} must be a known control command")
                self.assertNotIn(command.kind, GAME_COMMANDS)
                self.assertEqual(any_chat_game_route("/" + name),
                                 "not_intercepted",
                                 f"/{name} must not be swallowed by game dispatch")

    def test_reserved_commands_in_control_catalog(self):
        for name in RESERVED_SYSTEM_COMMANDS:
            with self.subTest(command=name):
                self.assertIn(name, CONTROL_COMMANDS)


if __name__ == "__main__":
    unittest.main()
