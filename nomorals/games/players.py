"""Player identity, persistent profiles, and the leaderboard.

A *player* is one human (or the AI, ``ai``) identified per platform by
``platform:sender``.  The same person on Telegram and Discord is two
player keys — platforms don't share identity, and we don't pretend they
do — but the *profile* system is global: every game played in any chat
on any platform updates one per-player ledger of wins, losses, points,
streaks and per-game stats, which is what the leaderboard ranks.

Everything persists to ``game_players`` so a player's record survives
restarts and accumulates across sessions.  The store is deliberately
dependency-free (just the shared ``Database``), so it works in CLI,
chat, tests and the engine all at once.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = ["AI_PLAYER", "Player", "PlayerStore", "Leaderboard"]

_log = get_logger(__name__)

#: The house seat. The AI joins, plays, and is ranked like anyone else.
AI_PLAYER = "ai"


@dataclass(frozen=True)
class Player:
    """One seat at the table: who they are and where they're playing."""

    key: str            # "platform:sender" (or "ai")
    platform: str       # telegram|whatsapp|discord|local|ai
    name: str           # display name
    is_ai: bool = False

    @classmethod
    def from_sender(cls, platform: str, sender: str, name: str = "") -> "Player":
        sender = (sender or "").strip() or "unknown"
        return cls(
            key=f"{platform}:{sender}",
            platform=platform,
            name=(name or sender)[:40],
            is_ai=False,
        )

    @classmethod
    def house(cls) -> "Player":
        return cls(key=AI_PLAYER, platform="ai", name="The House", is_ai=True)


@dataclass
class Profile:
    """One player's durable competitive record."""

    key: str
    name: str = ""
    platform: str = ""
    coins: int = 0
    points: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    streak: int = 0                 # >0 winning streak, <0 losing streak
    best_streak: int = 0
    games_played: int = 0
    xp: int = 0                      # persistent progression (see progression.py)
    per_game: dict[str, dict[str, Any]] = field(default_factory=dict)
    items: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "platform": self.platform,
            "coins": self.coins, "points": self.points, "wins": self.wins,
            "losses": self.losses, "draws": self.draws, "streak": self.streak,
            "best_streak": self.best_streak, "games_played": self.games_played,
            "per_game": self.per_game, "items": self.items,
            "xp": self.xp,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Profile":
        def _load(key: str) -> dict[str, Any]:
            try:
                return json.loads(row.get(key) or "{}")
            except Exception:  # noqa: BLE001
                return {}
        return cls(
            key=row["player_key"], name=row.get("display") or "",
            platform=row.get("platform") or "",
            coins=int(row.get("coins") or 0), points=int(row.get("points") or 0),
            wins=int(row.get("wins") or 0), losses=int(row.get("losses") or 0),
            draws=int(row.get("draws") or 0), streak=int(row.get("streak") or 0),
            best_streak=int(row.get("best_streak") or 0),
            games_played=int(row.get("games_played") or 0),
            xp=int(row.get("xp") or 0),
            per_game=_load("per_game"), items=_load("items"),
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
        )


class PlayerStore:
    """Persistent per-player ledgers.

    All mutations are transactional; readers never block writers (SQLite
    read while the writer holds the write lock is fine on the busy
    timeout the shared Database already configures).

    Read-modify-write sequences (balance checks, item grants) run under
    an in-process lock so two concurrent purchases for the same player
    can't both pass the affordability check on a stale read — the
    classic double-spend.  SQLite serializes the writes anyway, but the
    Python-side check has to be atomic with the write.
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self._lock = threading.RLock()

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, key: str, *, name: str = "", platform: str = "") -> Profile:
        """Fetch a profile, creating a fresh one on first sight."""
        row = None
        try:
            row = self.db.query_one(
                "SELECT * FROM game_players WHERE player_key = ?", (key,)
            )
        except Exception:  # noqa: BLE001
            _log.debug("game_players read failed", exc_info=True)
        if row is not None:
            return Profile.from_row(row)
        prof = Profile(key=key, name=name, platform=platform)
        if key != AI_PLAYER:
            self._upsert(prof)
        return prof

    def all(self, limit: int = 500) -> list[Profile]:
        try:
            rows = self.db.query(
                "SELECT * FROM game_players ORDER BY points DESC LIMIT ?",
                (limit,),
            )
        except Exception:  # noqa: BLE001
            return []
        return [Profile.from_row(r) for r in rows]

    # ── writes ───────────────────────────────────────────────────────────────
    def record_outcome(self, player: Player, *, won: bool | None,
                       game: str, points: int = 0, coins: int = 0,
                       score: int = 0) -> Profile:
        """Update one player's ledger after a finished game.

        ``won=None`` is a draw / participation-only outcome.  Streaks and
        best-streaks are maintained here so no game can forget to.
        """
        with self._lock:
            prof = self.get(player.key, name=player.name,
                            platform=player.platform)
            prof.games_played += 1
            prof.points += max(0, points)
            prof.coins = max(0, prof.coins + coins)
            stats = prof.per_game.setdefault(
                game, {"played": 0, "wins": 0, "points": 0, "best": 0}
            )
            stats["played"] = int(stats.get("played") or 0) + 1
            stats["points"] = int(stats.get("points") or 0) + max(0, points)
            stats["best"] = max(int(stats.get("best") or 0), score)
            if won is True:
                prof.wins += 1
                stats["wins"] = int(stats.get("wins") or 0) + 1
                prof.streak = prof.streak + 1 if prof.streak > 0 else 1
                prof.best_streak = max(prof.best_streak, prof.streak)
            elif won is False:
                prof.losses += 1
                prof.streak = prof.streak - 1 if prof.streak < 0 else -1
            else:
                prof.draws += 1
            prof.updated_at = time.time()
            self._write(prof)
            return prof

    def add_coins(self, player: Player, amount: int, reason: str = "") -> int:
        with self._lock:
            prof = self.get(player.key, name=player.name,
                            platform=player.platform)
            prof.coins = max(0, prof.coins + amount)
            prof.updated_at = time.time()
            self._write(prof, ledger=[(amount, reason)] if reason else [])
            return prof.coins

    def spend_coins(self, player: Player, amount: int,
                    reason: str = "") -> int | None:
        """Atomically spend coins. Returns the new balance, or None when
        the player can't afford it (balance untouched).

        Check and deduction happen under one lock in one transaction, so
        two concurrent purchases can't both spend the same coins.
        """
        if amount < 0:
            raise ValueError("spend_coins amount must be >= 0")
        with self._lock:
            prof = self.get(player.key, name=player.name,
                            platform=player.platform)
            if prof.coins < amount:
                return None
            prof.coins -= amount
            prof.updated_at = time.time()
            self._write(prof, ledger=[(-amount, reason)] if reason else [])
            return prof.coins

    def grant_item(self, player: Player, item: str, count: int = 1) -> dict[str, int]:
        with self._lock:
            prof = self.get(player.key, name=player.name,
                            platform=player.platform)
            prof.items[item] = int(prof.items.get(item) or 0) + count
            prof.updated_at = time.time()
            self._write(prof)
            return prof.items

    def consume_item(self, player: Player, item: str) -> bool:
        """Spend one owned item. False if they don't have one."""
        with self._lock:
            prof = self.get(player.key, name=player.name,
                            platform=player.platform)
            if int(prof.items.get(item) or 0) <= 0:
                return False
            prof.items[item] = int(prof.items.get(item) or 0) - 1
            prof.updated_at = time.time()
            self._write(prof)
            return True

    def _upsert(self, p: Profile) -> None:
        """Persist a profile (kept for callers that mutate a Profile
        directly, e.g. the economy). Takes the store lock."""
        with self._lock:
            self._write(p)

    def _write(self, p: Profile,
               ledger: list[tuple[int, str]] | None = None) -> None:
        """Write one profile plus optional wallet rows in a single DB
        transaction. Caller must hold ``self._lock`` (``_upsert`` takes
        it; the write methods above already hold it)."""
        if self.db is None:
            return
        try:
            with self.db.transaction():
                self.db.execute(
                    "INSERT INTO game_players (player_key, platform, display, coins, "
                    "points, wins, losses, draws, streak, best_streak, games_played, "
                    "xp, per_game, items, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(player_key) DO UPDATE SET platform = excluded.platform, "
                    "display = excluded.display, coins = excluded.coins, "
                    "points = excluded.points, wins = excluded.wins, "
                    "losses = excluded.losses, draws = excluded.draws, "
                    "streak = excluded.streak, best_streak = excluded.best_streak, "
                    "games_played = excluded.games_played, xp = excluded.xp, "
                    "per_game = excluded.per_game, "
                    "items = excluded.items, updated_at = excluded.updated_at",
                    (
                        p.key, p.platform, p.name, p.coins, p.points, p.wins,
                        p.losses, p.draws, p.streak, p.best_streak,
                        p.games_played, p.xp, json.dumps(p.per_game),
                        json.dumps(p.items), p.created_at, p.updated_at,
                    ),
                )
                for amount, reason in ledger or ():
                    if not reason:
                        continue
                    self.db.execute(
                        "INSERT INTO game_wallet (player_key, amount, reason, at) "
                        "VALUES (?, ?, ?, ?)",
                        (p.key, amount, reason[:80], time.time()),
                    )
        except Exception:  # noqa: BLE001
            _log.debug("game_players upsert failed", exc_info=True)


class Leaderboard:
    """Rankings over the player ledger — global and per game."""

    def __init__(self, store: PlayerStore) -> None:
        self.store = store

    def top(self, limit: int = 10, *, game: str = "") -> list[dict[str, Any]]:
        profiles = self.store.all(limit=500)
        if game:
            rows = [
                {
                    "name": p.name or p.key.split(":", 1)[-1],
                    "platform": p.platform,
                    "wins": p.per_game.get(game, {}).get("wins", 0),
                    "points": p.per_game.get(game, {}).get("points", 0),
                    "played": p.per_game.get(game, {}).get("played", 0),
                    "best": p.per_game.get(game, {}).get("best", 0),
                }
                for p in profiles if p.per_game.get(game)
            ]
            rows.sort(key=lambda r: (r["points"], r["wins"]), reverse=True)
            return rows[:limit]
        rows = [
            {
                "name": p.name or p.key.split(":", 1)[-1],
                "platform": p.platform,
                "points": p.points,
                "wins": p.wins,
                "losses": p.losses,
                "win_rate": round(p.win_rate, 3),
                "streak": p.streak,
                "games": p.games_played,
            }
            for p in profiles if p.games_played
        ]
        rows.sort(key=lambda r: (r["points"], r["wins"], -abs(r["streak"])),
                  reverse=True)
        return rows[:limit]

    def render(self, limit: int = 10, *, game: str = "") -> str:
        rows = self.top(limit, game=game)
        if not rows:
            return "the board is empty — play a game first."
        title = f"🏆 leaderboard — {game}" if game else "🏆 leaderboard"
        lines = [title]
        medals = ("🥇", "🥈", "🥉")
        for i, row in enumerate(rows):
            mark = medals[i] if i < 3 else f"{i + 1}."
            if game:
                lines.append(
                    f" {mark} {row['name']} — {row['points']} pts · "
                    f"{row['wins']}W/{row['played']} games"
                )
            else:
                streak = ""
                if row["streak"] > 1:
                    streak = f" 🔥{row['streak']}"
                elif row["streak"] < -1:
                    streak = f" 🧊{-row['streak']} down"
                lines.append(
                    f" {mark} {row['name']} — {row['points']} pts · "
                    f"{row['wins']}W-{row['losses']}L · "
                    f"{row['win_rate']:.0%}{streak}"
                )
        return "\n".join(lines)
