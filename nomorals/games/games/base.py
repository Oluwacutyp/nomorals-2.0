"""The multiplayer game contract.

A :class:`MultiGame` is one playable game.  The engine owns *when* a
message reaches a game and *who* is allowed to move; the game owns *what
a move does*.  The split is strict:

* game code is pure — no I/O, no I/O-adjacent state outside
  ``room.state`` (a plain JSON-able dict), so every game is unit-testable
  with a fixed RNG and a scripted list of moves;
* the engine is platform-agnostic — it never parses game rules, it only
  routes turns, runs timers, persists rooms, and credits outcomes.

Rooms live in a chat (``platform:chat_id[:thread]``).  A DM room has one
human and its AI seats; a group room has every human who joined plus
enough AI seats for the game to work; a channel room is spectator mode —
the house plays all seats and the channel watches (channels are
"supported" where the platform lets sends through at all).
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..ai import GameMind
from ..players import AI_PLAYER, Player

__all__ = ["Room", "MultiGame", "parse_command", "GAME_COMMANDS",
           "DIFFICULTY_LEVELS", "DEFAULT_DIFFICULTY"]

#: in-game commands understood by the engine (not the game).
GAME_COMMANDS = {
    "pass": "pass your turn",
    "skip": "same as pass",
    "leave": "quit the game (your seat is dropped)",
    "status": "show the current table",
    "help": "show the rules again",
    "shop": "open the shop (catalog | buy <slug>)",
    "balance": "your coins and items",
}

#: AI/puzzle difficulty ladder. Games opt in by declaring a named
#: ``difficulty`` parameter on ``new_state`` — the engine only hands it
#: to games that ask for it, so everyone else plays as always.
#: Players pick it at the table: ``/game connect4 hard``.
DIFFICULTY_LEVELS = ("easy", "normal", "hard", "expert")
DEFAULT_DIFFICULTY = "normal"


def normalize_difficulty(value: str | None) -> str:
    """Coerce a difficulty word to the ladder; unknown → normal."""
    v = (value or "").strip().lower()
    return v if v in DIFFICULTY_LEVELS else DEFAULT_DIFFICULTY


@dataclass
class Room:
    """One live game at one chat table."""

    id: str
    game: str                       # game name
    chat_key: str                   # "platform:chat_id[:thread]"
    platform: str                   # telegram|whatsapp|discord|local
    kind: str                       # dm|group|channel
    players: list[Player] = field(default_factory=list)
    turn: int = 0                   # index into players whose turn it is
    state: dict[str, Any] = field(default_factory=dict)
    status: str = "waiting"         # waiting|active|finished
    started_at: float = field(default_factory=time.time)
    turn_started: float = field(default_factory=time.time)
    messages: list[str] = field(default_factory=list)  # transcript tail
    seed: int = field(default_factory=lambda: random.randrange(2 ** 31))
    # transient (never persisted): the room's live random stream
    _rng_instance: random.Random | None = field(
        default=None, repr=False, compare=False)
    #: Last human-initiated activity (start/move/join/leave). Turn
    #: timeouts and AI pumps deliberately do NOT touch this — it drives
    #: idle expiry, so only a real player keeps the table alive.
    last_activity: float = field(default_factory=time.time)

    #: Per-room re-entrant guard. The engine's global lock protects the
    #: room *tables*; this guard serializes everything that touches one
    #: room's mutable state (moves, AI pumps, quits, timeouts) so two
    #: inbound messages for the same chat can't interleave a move with
    #: a quit or double-apply a turn. Lock order is always engine →
    #: room; the guard is never held while taking the engine lock.
    guard: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False)

    # ── players ─────────────────────────────────────────────────────────────
    def rng(self) -> random.Random:
        """The room's continuous random stream, seeded once from ``seed``.

        The instance lives on the room, so every call keeps drawing from
        the same stream instead of restarting it — deals, dice, and house
        plays are never repeated across a game, while replays of a room
        with the same seed still start from the same stream.
        """
        if self._rng_instance is None:
            self._rng_instance = random.Random(self.seed)
        return self._rng_instance

    @property
    def humans(self) -> list[Player]:
        return [p for p in self.players if not p.is_ai]

    @property
    def ai_seats(self) -> list[Player]:
        return [p for p in self.players if p.is_ai]

    def player(self, key: str) -> Player | None:
        for p in self.players:
            if p.key == key:
                return p
        return None

    @property
    def current(self) -> Player | None:
        if 0 <= self.turn < len(self.players):
            return self.players[self.turn]
        return None

    @property
    def current_human(self) -> Player | None:
        c = self.current
        return c if (c is not None and not c.is_ai) else None

    def advance_turn(self) -> None:
        """Move to the next seat, skipping finished AI seats is the
        game's job (it may remove players from room.players first)."""
        if self.players:
            self.turn = (self.turn + 1) % len(self.players)
        self.turn_started = time.time()

    def transcript(self, limit: int = 12) -> str:
        return "\n".join(self.messages[-limit:])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "game": self.game, "chat_key": self.chat_key,
            "platform": self.platform, "kind": self.kind,
            "players": [p.key for p in self.players],
            "turn": self.turn, "state": self.state, "status": self.status,
            "started_at": self.started_at, "seed": self.seed,
            "last_activity": self.last_activity,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any], game_players: dict) -> "Room":
        try:
            state = json.loads(row.get("state") or "{}")
        except Exception:  # noqa: BLE001
            state = {}
        try:
            keys = json.loads(row.get("players") or "[]")
        except Exception:  # noqa: BLE001
            keys = []
        return cls(
            id=row["id"], game=row["game"], chat_key=row["chat_key"],
            platform=row.get("platform") or "local",
            kind=row.get("kind") or "dm",
            players=[Player(key=k, platform=k.split(":", 1)[0],
                            name=k.split(":", 1)[-1],
                            is_ai=(k == AI_PLAYER
                                   or k.startswith(AI_PLAYER + ":")))
                     for k in keys],
            turn=int(row.get("turn") or 0), state=state,
            status=row.get("status") or "active",
            started_at=float(row.get("started_at") or time.time()),
            seed=int(row.get("seed") or 1),
        )


def parse_command(text: str) -> tuple[str, str]:
    """/pass, /shop buy x → (command, rest). ('', text) when not a
    command. Only game commands are parsed here; control commands
    (/game …) never reach the engine."""
    t = (text or "").strip()
    if not t.startswith("/"):
        return "", t
    parts = t[1:].split(None, 1)
    cmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if cmd in GAME_COMMANDS or cmd == "game":
        return cmd, rest
    return "", t


class MultiGame:
    """One game's rules. Subclass, set the class attributes, implement
    the hooks. ``mind`` is the house brain (model-optional); ``rng``
    comes from the room's seed so replays are deterministic."""

    name: str = ""
    description: str = ""
    min_players: int = 1        # humans needed to start
    max_players: int = 8
    ai_seats: int = 1           # AI players to fill in
    needs_group: bool = False   # refuse DM starts
    channel_mode: str = "house"  # house|block — channel play style
    turn_based: bool = True
    move_timeout: float = 90.0  # seconds; 0 disables the turn clock
    #: how long a room may sit with no human move before the scheduler
    #: closes it. None = the engine default (IDLE_ROOM_TTL). Inbox games
    #: (one message per turn, days between moves) set this to days.
    idle_ttl: float | None = None
    rules: str = ""             # shown by /help and in the intro
    #: short how-to-play blurb (3-4 lines) posted when a multiplayer
    #: game starts in a group: how to join and the core loop. Empty =
    #: no blurb (solo/DM games don't need one).
    howto: str = ""
    #: difficulty ladder this game actually honors (subset of
    #: DIFFICULTY_LEVELS). Empty = the game doesn't take a difficulty.
    difficulties: tuple[str, ...] = ()
    #: game-specific mode words the player can pass at start
    #: (``/game gomoku big``). word → one-line description shown in
    #: help. The word arrives as ``variant`` in ``new_state``; the game
    #: validates it against the player's mastery tier itself.
    variants: dict[str, str] = {}

    def rng(self, room: Room) -> random.Random:
        return room.rng()

    def difficulty(self, room: Room) -> str:
        """This room's difficulty, or ``DEFAULT_DIFFICULTY`` when the
        game doesn't take one."""
        return normalize_difficulty(room.state.get("difficulty"))

    def is_move_text(self, text: str) -> bool:
        """Does this plain-text message look like a game move?

        Used by the relay/router to decide whether non-command text
        should be swallowed by the game engine or fall through to
        normal chat. Default is True (every message is a potential
        move — the historical behavior). Games with a fixed move
        vocabulary (combat games, etc.) should override this so
        casual chat like "say that again?" doesn't trigger
        "waiting on X" spam while it's not your turn.
        """
        return True

    # ── lifecycle ───────────────────────────────────────────────────────────
    def new_state(self, rng: random.Random, **kw: Any) -> dict[str, Any]:
        # kw carries engine-level options (e.g. daily=True); games that
        # don't know them simply ignore them.
        return {}

    def setup(self, room: Room, mind: GameMind) -> str:
        """Broadcast when the game starts. Also the place to finalize
        the player list (remove extra AI seats, etc.)."""
        return f"{self.name} — ready. {self.description}"

    def on_join(self, room: Room, player: Player, mind: GameMind) -> str | None:
        """A human joins mid-lobby. None = silent accept."""
        return None

    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        """One human move. Returns the messages to send (may be empty).
        Must be side-effect-free apart from ``room.state`` /
        ``room.players`` / ``room.turn``."""
        raise NotImplementedError

    def ai_turn(self, room: Room, mind: GameMind) -> list[str]:
        """Play one move for an AI seat. Return the messages to send."""
        return []

    def describe_options(self, room: Room) -> list[str]:
        """Legal moves for the current seat, as exact strings the agent
        may reply with. Used by character/brain seats to pick valid moves.
        Default: no structured options (free-text move)."""
        return []

    def on_timeout(self, room: Room, player: Player,
                   mind: GameMind) -> list[str]:
        """The current human's clock ran out. Default: pass the turn."""
        room.advance_turn()
        return [f"⏰ {player.name} took too long — turn passes."]

    def on_leave(self, room: Room, player: Player,
                 mind: GameMind) -> str | None:
        """A human left mid-game. Return a notice (or None)."""
        return None

    # ── outcome ─────────────────────────────────────────────────────────────
    def is_over(self, room: Room) -> bool:
        return room.status == "finished"

    def winner(self, room: Room) -> Player | str | None:
        """The winning player, "draw", or None (no winner)."""
        return None

    def finish_won(self, room: Room, player: Player) -> bool | None | str:
        """Decide this player's win/loss/draw outcome for finish payout.

        Return True (win), False (loss), or None (draw) to settle the
        outcome directly.  Return the string ``"winner"`` (the default)
        to fall back to the standard ``winner()``-based derivation.

        Never-ending games (``world``) settle a prosperous run as a win
        instead of a draw; competitive games use ``winner()``.
        """
        return "winner"

    def coin_payout(self, room: Room, player: Player, won: bool | None,
                    score: int, difficulty: str, streak_after: int
                    ) -> tuple[int, str] | None:
        """Custom coin payout for this finish.

        Return ``(coins, reason)`` to bypass
        ``GameEconomy.coin_breakdown`` — e.g. a score-based settlement
        that outgrows the standard win caps — or None (the default) for
        the standard performance-based breakdown.
        """
        return None

    def score(self, room: Room, player: Player) -> int:
        """Numeric score for the ledger (per-game 'best')."""
        return 0

    def xp_reward(self, won: bool | None, room: Room, player: Player) -> int:
        """Persistent XP for finishing this game. Games with a richer
        story (arena) override this; everyone else gets the generic
        win/draw/loss table from progression.generic_game_xp."""
        from ..progression import generic_game_xp
        return generic_game_xp(won)

    def final_message(self, room: Room, mind: GameMind) -> str:
        w = self.winner(room)
        if w == "draw":
            return "it's a draw."
        if w is not None:
            return f"🏁 {w.name} wins."
        return "game over."

    # ── display ─────────────────────────────────────────────────────────────
    def describe_state(self, room: Room) -> str:
        """Extra lines for /status — the game's own picture of the table."""
        return ""

    def status_line(self, room: Room) -> str:
        cur = room.current
        who = "house" if (cur and cur.is_ai) else (cur.name if cur else "?")
        return (f"🎮 {self.name} — {len(room.players)} player(s), "
                f"turn: {who}")
