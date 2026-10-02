"""Game relay: connect two separate DM chats for multiplayer.

User A (in a DM with the bot, on any platform) invites user B (in their
own DM with the bot, on any platform) to play a game together. The bot
relays moves between the two chats through one virtual engine room.

Usage:
  User A: /game invite <game_name> [who]
  User B: /game accept <invite_code>      (from any chat, any platform)

The invite code is the source of truth — the optional ``who`` argument
only triggers a best-effort direct DM on the inviter's own platform.
Cross-platform invites work by sharing the code.

The virtual engine room lives at chat key ``relay:<code>``. Invites and
relay rooms persist in SQLite so a restart doesn't strand players;
``cleanup_expired`` (called from the engine ticker) reaps dead ones.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .engine import GameEngine, Player, Room

__all__ = ["GameInvite", "RelayRoom", "GameRelay"]


@dataclass
class GameInvite:
    """A pending game invitation between two users."""

    code: str
    game_name: str
    from_chat: str  # chat_key of inviter
    from_player: Player
    to_label: str = ""  # who it was addressed to (a label, not a key)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.created_at + 3600  # 1 hour

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class RelayRoom:
    """A virtual room connecting two separate DM chats."""

    room_id: str  # the invite code (NOT prefixed — see virtual_chat)
    game_name: str
    chat_a: str  # chat_key of player A
    chat_b: str  # chat_key of player B
    player_a: Player
    player_b: Player
    active: bool = True
    last_activity: float = field(default_factory=time.time)

    @property
    def virtual_chat(self) -> str:
        """The engine room key both chats play through. Single source of
        truth — every engine call for this relay must use this."""
        return f"relay:{self.room_id}"

    def other_chat(self, chat_key: str) -> str:
        return self.chat_b if chat_key == self.chat_a else self.chat_a


class GameRelay:
    """Manages game invites and relay rooms between separate DM chats."""

    _CLEANUP_INTERVAL = 60.0
    _RELAY_TTL = 6 * 3600.0  # drop relays idle 6h

    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.db = getattr(engine, "db", None)
        self.invites: dict[str, GameInvite] = {}  # code -> invite
        self.relays: dict[str, RelayRoom] = {}  # code -> relay
        self.chat_to_relay: dict[str, str] = {}  # chat_key -> code
        self._lock = threading.RLock()
        self._last_cleanup = 0.0
        self._ensure_tables()
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _ensure_tables(self) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS game_invites ("
                "code TEXT PRIMARY KEY, game_name TEXT NOT NULL, "
                "from_chat TEXT NOT NULL, from_player_key TEXT NOT NULL, "
                "from_player_name TEXT NOT NULL, "
                "from_player_platform TEXT NOT NULL, "
                "to_label TEXT NOT NULL DEFAULT '', "
                "created_at REAL NOT NULL, expires_at REAL NOT NULL)")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS game_relays ("
                "room_id TEXT PRIMARY KEY, game_name TEXT NOT NULL, "
                "chat_a TEXT NOT NULL, chat_b TEXT NOT NULL, "
                "player_a_key TEXT NOT NULL, player_a_name TEXT NOT NULL, "
                "player_a_platform TEXT NOT NULL, "
                "player_b_key TEXT NOT NULL, player_b_name TEXT NOT NULL, "
                "player_b_platform TEXT NOT NULL, "
                "active INTEGER NOT NULL DEFAULT 1, "
                "last_activity REAL NOT NULL)")
        except Exception:  # noqa: BLE001 - relay stays memory-only
            pass

    def _load(self) -> None:
        if self.db is None:
            return
        try:
            for row in self.db.query("SELECT * FROM game_invites") or []:
                inv = GameInvite(
                    code=row["code"], game_name=row["game_name"],
                    from_chat=row["from_chat"],
                    from_player=Player(
                        key=row["from_player_key"],
                        platform=row["from_player_platform"],
                        name=row["from_player_name"]),
                    to_label=row.get("to_label") or "",
                    created_at=float(row["created_at"]),
                    expires_at=float(row["expires_at"]))
                if not inv.is_expired():
                    self.invites[inv.code] = inv
            for row in self.db.query(
                    "SELECT * FROM game_relays WHERE active = 1") or []:
                relay = RelayRoom(
                    room_id=row["room_id"], game_name=row["game_name"],
                    chat_a=row["chat_a"], chat_b=row["chat_b"],
                    player_a=Player(
                        key=row["player_a_key"],
                        platform=row["player_a_platform"],
                        name=row["player_a_name"]),
                    player_b=Player(
                        key=row["player_b_key"],
                        platform=row["player_b_platform"],
                        name=row["player_b_name"]),
                    active=True,
                    last_activity=float(row["last_activity"]))
                self.relays[relay.room_id] = relay
                self.chat_to_relay[relay.chat_a] = relay.room_id
                self.chat_to_relay[relay.chat_b] = relay.room_id
        except Exception:  # noqa: BLE001
            pass

    def _save_invite(self, inv: GameInvite) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO game_invites (code, game_name, "
                "from_chat, from_player_key, from_player_name, "
                "from_player_platform, to_label, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (inv.code, inv.game_name, inv.from_chat,
                 inv.from_player.key, inv.from_player.name,
                 inv.from_player.platform, inv.to_label,
                 inv.created_at, inv.expires_at))
        except Exception:  # noqa: BLE001
            pass

    def _delete_invite(self, code: str) -> None:
        if self.db is None:
            return
        try:
            self.db.execute("DELETE FROM game_invites WHERE code = ?",
                            (code,))
        except Exception:  # noqa: BLE001
            pass

    def _save_relay(self, relay: RelayRoom) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO game_relays (room_id, game_name, "
                "chat_a, chat_b, player_a_key, player_a_name, "
                "player_a_platform, player_b_key, player_b_name, "
                "player_b_platform, active, last_activity) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (relay.room_id, relay.game_name, relay.chat_a, relay.chat_b,
                 relay.player_a.key, relay.player_a.name,
                 relay.player_a.platform, relay.player_b.key,
                 relay.player_b.name, relay.player_b.platform,
                 1 if relay.active else 0, relay.last_activity))
        except Exception:  # noqa: BLE001
            pass

    def _delete_relay(self, code: str) -> None:
        if self.db is None:
            return
        try:
            self.db.execute("DELETE FROM game_relays WHERE room_id = ?",
                            (code,))
        except Exception:  # noqa: BLE001
            pass

    # ── invites ──────────────────────────────────────────────────────────
    def create_invite(self, from_chat: str, from_player: Player,
                      game_name: str, to_label: str = "") -> GameInvite:
        """Create a game invite. The code is the source of truth — the
        invitee accepts from any chat on any platform."""
        game_name = (game_name or "").lower()
        if game_name not in self.engine.games:
            raise ValueError(f"unknown game {game_name!r} — /game list")
        game = self.engine.games[game_name]
        if game.max_players < 2:
            raise ValueError(
                f"{game_name} is solo — relays need a 2+ player game")
        with self._lock:
            code = secrets.token_urlsafe(8)
            while code in self.invites:
                code = secrets.token_urlsafe(8)
            invite = GameInvite(
                code=code, game_name=game_name, from_chat=from_chat,
                from_player=from_player, to_label=(to_label or "")[:60])
            self.invites[code] = invite
            self._save_invite(invite)
            return invite

    def accept_invite(self, code: str, to_chat: str,
                      to_player: Player) -> RelayRoom:
        """Accept a game invite and create a relay room."""
        code = (code or "").strip()
        with self._lock:
            invite = self.invites.get(code)
            if invite is None:
                raise ValueError("invalid or expired invite code")
            if invite.is_expired():
                del self.invites[code]
                self._delete_invite(code)
                raise ValueError("invite has expired — ask for a fresh one")
            if to_chat == invite.from_chat:
                raise ValueError(
                    "that's your own invite — share the code with a friend, "
                    "they accept from their own chat")

            room_id = code
            relay = RelayRoom(
                room_id=room_id, game_name=invite.game_name,
                chat_a=invite.from_chat, chat_b=to_chat,
                player_a=invite.from_player, player_b=to_player)

            # start the game in the virtual room both chats play through
            room, _msgs = self.engine.start(
                relay.virtual_chat, invite.game_name, invite.from_player,
                kind="dm")
            # seat the second player alongside the inviter
            if room.player(to_player.key) is None:
                room.players.append(to_player)

            self.relays[room_id] = relay
            self.chat_to_relay[relay.chat_a] = room_id
            self.chat_to_relay[relay.chat_b] = room_id
            self._save_relay(relay)

            del self.invites[code]
            self._delete_invite(code)
            return relay

    # ── moves ────────────────────────────────────────────────────────────
    def relay_move(self, from_chat: str, text: str,
                   player: Player) -> list[str]:
        """Relay a game move from one DM to the shared virtual room."""
        with self._lock:
            code = self.chat_to_relay.get(from_chat)
            relay = self.relays.get(code) if code else None
            if relay is None or not relay.active:
                return []
            # the virtual room may have finished or been lost on restart —
            # never strand the players on a dead relay
            if self.engine.live(relay.virtual_chat) is None:
                self._close_locked(relay.room_id, reason="game over")
                return []
            msgs = self.engine.move(relay.virtual_chat, text, player)
            relay.last_activity = time.time()
            self._save_relay(relay)
            # the engine finished the game → retire the relay now
            if self.engine.live(relay.virtual_chat) is None:
                self._close_locked(relay.room_id, reason="game over")
            return msgs

    def get_relay_for_chat(self, chat_key: str) -> RelayRoom | None:
        with self._lock:
            code = self.chat_to_relay.get(chat_key)
            relay = self.relays.get(code) if code else None
            if relay is not None and relay.active:
                return relay
            return None

    def close_relay(self, room_id: str, reason: str = "") -> None:
        with self._lock:
            self._close_locked(room_id, reason=reason)

    def cancel_invites_from_chat(self, chat_key: str) -> int:
        """Cancel every pending invite sent from this chat.

        Quit cleanup: a player who quits mid-invite leaves nobody to
        play with, so the invite must die here rather than linger for
        its full hour and ambush an acceptor later.
        """
        n = 0
        with self._lock:
            for code in [c for c, inv in self.invites.items()
                         if inv.from_chat == chat_key]:
                del self.invites[code]
                self._delete_invite(code)
                n += 1
        return n

    def _close_locked(self, room_id: str, reason: str = "") -> None:
        relay = self.relays.get(room_id)
        if relay is None:
            return
        try:
            self.engine.quit(relay.virtual_chat)
        except Exception:  # noqa: BLE001
            pass
        self.chat_to_relay.pop(relay.chat_a, None)
        self.chat_to_relay.pop(relay.chat_b, None)
        self.relays.pop(room_id, None)
        self._delete_relay(room_id)

    # ── housekeeping ─────────────────────────────────────────────────────
    def maybe_cleanup(self) -> None:
        """Throttled cleanup — safe to call from a hot ticker."""
        now = time.time()
        if now - self._last_cleanup < self._CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        try:
            self.cleanup_expired()
        except Exception:  # noqa: BLE001
            pass

    def cleanup_expired(self) -> dict[str, int]:
        """Remove expired invites and long-idle relays. Returns counts."""
        now = time.time()
        removed = {"invites": 0, "relays": 0}
        with self._lock:
            for code in [c for c, inv in self.invites.items()
                         if inv.is_expired()]:
                del self.invites[code]
                self._delete_invite(code)
                removed["invites"] += 1
            for code, relay in list(self.relays.items()):
                idle = now - relay.last_activity
                dead_game = self.engine.live(relay.virtual_chat) is None
                if idle > self._RELAY_TTL or (dead_game and idle > 300):
                    self._close_locked(code, reason="expired")
                    removed["relays"] += 1
        return removed
