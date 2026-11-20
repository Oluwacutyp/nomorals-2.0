"""Laser Asteroids as a Devon engine game.

The arcade game from echelon/laser-asteroids, wired into the social game
engine as a single-player table: ``/game laser`` or ``/laser`` starts it,
then every message is a move — fly the ship, aim, shoot, watch the rocks
break. State is plain JSON-able dicts in ``room.state`` (the organ's
:class:`LaserGame` rehydrates per move), so the table survives restarts
like every other game.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any

from ..laser.game import COMMANDS, LaserGame
from ..players import Player
from .base import GAME_COMMANDS, MultiGame, Room

__all__ = ["LaserAsteroidsGame"]

#: Exact strings the brain may reply with as moves.
MOVE_OPTIONS = ("up", "down", "left", "right", "aim 90", "fire",
                "tick 10", "render")


class LaserAsteroidsGame(MultiGame):
    name = "laser"
    description = "laser asteroids — fly, shoot rocks, don't die"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 0
    family = "arcade"
    rules = (
        "Fly the ship across the laser field and blast the rocks. "
        "Move: up / down / left / right. Aim: aim <degrees> (0 = east, "
        "90 = north). Fire: fire. Let the world run: tick [n]. "
        "Destroying a rock scores by size — small rocks pay 100, medium "
        "50, large 25 — and big rocks split into two smaller ones. "
        "A rock hit costs 10 health; at 0 the game is over. "
        "render shows the field again; png saves one frame as a picture."
    )

    # ── state ────────────────────────────────────────────────────────────
    def new_state(self, rng: random.Random, **kw: Any) -> dict[str, Any]:
        return LaserGame(rng).to_dict()

    def _load(self, room: Room) -> LaserGame:
        return LaserGame.from_dict(room.state)

    def _save(self, room: Room, game: LaserGame) -> None:
        # Merge, don't replace: the engine stashes its own keys
        # (_difficulty, …) in the same blob and they must survive.
        room.state = {**room.state, **game.to_dict()}

    # ── lifecycle ────────────────────────────────────────────────────────
    def setup(self, room: Room, mind: Any) -> str:
        game = self._load(room)
        return ("🔫 Laser Asteroids — the field is live.\n"
                + game.render_text())

    def on_move(self, room: Room, player: Player, text: str,
                mind: Any) -> list[str]:
        game = self._load(room)
        if game.game_over:
            return ["game over — start a new game to play again",
                    f"final score: {game.score}"]
        t = (text or "").strip().lower()
        if not t:
            return [game.render_text()]
        verb = t.split(None, 1)[0]
        if verb == "render":
            return [game.render_text()]
        if verb == "png":
            path = Path(tempfile.gettempdir()) / f"laser-{room.id}.png"
            game.render_png(str(path))
            return [f"frame saved to {path}", game.render_text()]
        if verb == "tick":
            parts = t.split()
            count = 1
            if len(parts) > 1:
                try:
                    count = int(parts[1])
                except ValueError:
                    return ["tick needs a number — tick 10"]
            events = game.tick_steps(count)
            self._save(room, game)
            return events + [game.render_text()]
        if verb in COMMANDS:
            events = game.step([t])
            self._save(room, game)
            return events + [game.render_text()]
        return ["move: up / down / left / right / aim <deg> / fire / "
                "tick [n] / render / png"]

    def ai_turn(self, room: Room, mind: Any) -> list[str]:
        return []

    def is_move_text(self, text: str) -> bool:
        """Only exact moves reach the table — casual chat like "fire
        that up" must not swallow a turn while a game is live. A move
        is one bare word (up, fire, render…), ``aim <deg>``,
        ``tick [n]``, or ``rotate left|right``."""
        words = (text or "").strip().lower().split()
        if not words:
            return False
        if words[0] in GAME_COMMANDS:
            return len(words) == 1
        if words[0] in ("up", "down", "left", "right", "fire", "render",
                        "png"):
            return len(words) == 1
        if words[0] == "aim":
            if len(words) != 2:
                return False
            try:
                float(words[1])
            except ValueError:
                return False
            return True
        if words[0] == "tick":
            if len(words) > 2:
                return False
            if len(words) == 2:
                try:
                    int(words[1])
                except ValueError:
                    return False
            return True
        if words[0] == "rotate":
            return len(words) == 2 and words[1] in ("left", "right")
        return False

    # ── outcome ──────────────────────────────────────────────────────────
    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("game_over", False))

    def score(self, room: Room, player: Player) -> int:
        return int(room.state.get("score", 0))

    # ── display ──────────────────────────────────────────────────────────
    def describe_options(self, room: Room) -> list[str]:
        return list(MOVE_OPTIONS)

    def describe_state(self, room: Room) -> str:
        return self._load(room).describe()

    def status_line(self, room: Room) -> str:
        game = self._load(room)
        return (f"🔫 laser — score {game.score}, "
                f"health {game.healthbar.health}, "
                f"{len(game.asteroids)} rocks")
