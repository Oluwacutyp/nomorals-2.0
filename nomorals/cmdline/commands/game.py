"""``nm game`` — play the engine games from the terminal.

Thin CLI over :class:`nomorals.games.engine.GameEngine` on the local
chat key: the same rooms the chat games play, so a game started here
can be continued in chat and vice versa. Rooms persist in the context
database and are restored across invocations.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from ...games.engine import GameEngine
from ...games.players import Player
from ..emit import _emit

#: The CLI's table — one room, local play.
_CLI_CHAT = "local:cli"


def _engine(context: Any) -> GameEngine:
    return GameEngine(context, send=lambda chat, text: None)


def _player() -> Player:
    return Player.from_sender("local", "cli", "cli")


def _cmd_game(args: argparse.Namespace, context: Any) -> int:
    """Route `nm game` subcommands."""
    action = getattr(args, "game_action", "") or "list"
    eng = _engine(context)
    try:
        if action == "list":
            text = eng.list_games()
            _emit(args, {"games": sorted(eng.games)}, text)
            return 0

        if action == "start":
            name = (getattr(args, "game_name", "") or "").strip().lower()
            if not name:
                print("nm game start needs a game — nm game start laser",
                      file=sys.stderr)
                return 2
            try:
                _room, msgs = eng.start(_CLI_CHAT, name, _player(),
                                        kind="dm")
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            _emit(args, {"game": name, "messages": msgs},
                  "\n".join(msgs))
            return 0

        if action == "move":
            text = " ".join(getattr(args, "game_move", []) or []).strip()
            if not text:
                print("nm game move needs a move — nm game move fire",
                      file=sys.stderr)
                return 2
            if eng.live(_CLI_CHAT) is None:
                print("no game live here — nm game start <name> first",
                      file=sys.stderr)
                return 1
            out = eng.move(_CLI_CHAT, text, _player())
            _emit(args, {"move": text, "messages": out}, "\n".join(out))
            return 0

        if action == "state":
            room = eng.live(_CLI_CHAT)
            if room is None:
                print("no game live here — nm game start <name> first",
                      file=sys.stderr)
                return 1
            game = eng.games.get(room.game)
            lines = []
            if game is not None:
                lines.append(game.status_line(room))
                extra = game.describe_state(room)
                if extra:
                    lines.append(extra)
            _emit(args, {"game": room.game, "state": room.state},
                  "\n".join(lines))
            return 0

        if action == "quit":
            out = eng.quit(_CLI_CHAT)
            _emit(args, {"messages": out}, "\n".join(out) or "no game live")
            return 0

        if action == "render":
            room = eng.live(_CLI_CHAT)
            if room is None:
                print("no game live here — nm game start <name> first",
                      file=sys.stderr)
                return 1
            game = eng.games.get(room.game)
            if game is None:
                print(f"unknown game {room.game!r}", file=sys.stderr)
                return 1
            png = (getattr(args, "png", "") or "").strip()
            lines = []
            if room.game == "laser":
                from ...games.laser import LaserGame
                laser = LaserGame.from_dict(room.state)
                if png:
                    laser.render_png(png)
                    lines.append(f"frame saved to {png}")
                lines.append(laser.render_text())
            else:
                describe = game.describe_state(room)
                if describe:
                    lines.append(describe)
            _emit(args, {"game": room.game, "png": png or None},
                  "\n".join(lines))
            return 0

        print(f"game: unknown action {action}", file=sys.stderr)
        return 2
    finally:
        eng.shutdown()
