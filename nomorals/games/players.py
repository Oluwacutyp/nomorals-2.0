"""Player identity, persistent profiles, and the leaderboard.

A *player* is one human (or the AI, ``ai``) identified per platform by
``platform:sender``.  The same person on Telegram and Discord is two
player keys — genuinely different networks don't share identity — but
endpoints on the *same* network do: ``telegram`` (userbot) and
``telegram-bot`` (BotFather bot) see the same Telegram user IDs, so
``GAME_IDENTITY_ALIASES`` folds them into one game key.  The *profile*
system is global: every game played in any chat on any platform updates
one per-player ledger of wins, losses, points, streaks and per-game
stats, which is what the leaderboard ranks.

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

__all__ = ["AI_PLAYER", "Player", "PlayerStore", "Leaderboard",
           "GAME_IDENTITY_ALIASES"]

_log = get_logger(__name__)

#: The house seat. The AI joins, plays, and is ranked like anyone else.
AI_PLAYER = "ai"

#: Platforms that share one underlying network identity. ``telegram``
#: (the userbot) and ``telegram-bot`` (the BotFather bot) both see the
#: same Telegram user IDs, so a human's XP, gear, skills and attributes
#: must follow them across the two endpoints — otherwise a duel started
#: from one side sees a fresh level-1 profile from the other.
#: The alias applies to the *game key* only; ``Player.platform`` keeps
#: the real endpoint so message routing is untouched.
GAME_IDENTITY_ALIASES = {
    "telegram-bot": "telegram",
}


def _caller_sets_display_name(stored_platform: str,
                              caller_platform: str) -> bool:
    """True when a sighting from ``caller_platform`` may overwrite the
    stored display name.

    Endpoints on the same network share one profile (see
    ``GAME_IDENTITY_ALIASES``), but they don't see the same name: the
    userbot reports the account display name (``Mary``) while the
    BotFather bot reports the username (``chfjdhx``).  The canonical
    endpoint — the userbot (``telegram``) — is the name authority: its
    sightings always refresh the stored name, so a rename takes effect
    everywhere.  A sighting from an aliased endpoint (``telegram-bot``)
    may only fill in a name when no canonical name exists yet; it must
    never clobber one.  Sightings from unrelated platforms always
    refresh (each network owns its own profile).
    """
    caller = (caller_platform or "").strip().lower()
    stored = (stored_platform or "").strip().lower()
    if not caller:
        return False
    canonical = GAME_IDENTITY_ALIASES.get(caller)
    if canonical is not None:
        # Aliased endpoint: only set when the canonical name isn't set.
        return stored != canonical
    return True


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
        # One human, one game identity: aliased platforms (telegram-bot →
        # telegram) share the same underlying network IDs, so the key
        # uses the canonical platform. ``platform`` itself is kept as
        # the real endpoint for message routing.
        key_platform = GAME_IDENTITY_ALIASES.get(
            (platform or "").lower(), platform)
        return cls(
            key=f"{key_platform}:{sender}",
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

    The lock is class-level (shared by every PlayerStore instance in
    this process) because stores are constructed ad-hoc in several
    places (engine, gear migration, handlers) — a per-instance lock
    would not serialize two stores wrapping the same database.
    """

    _LOCK = threading.RLock()

    def __init__(self, db: Any) -> None:
        self.db = db
        self._lock = PlayerStore._LOCK

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, key: str, *, name: str = "", platform: str = "") -> Profile:
        """Fetch a profile, creating a fresh one on first sight.

        Row creation is atomic (INSERT OR IGNORE under the store lock)
        so two threads racing to create the same player can't clobber
        each other's subsequent writes with a stale xp=0 profile.

        When the key is a stable-ID key (``telegram:<numeric_id>``) and a
        display name is supplied, any legacy display-name-keyed profile
        (``telegram:Mary``) for the same human is folded into the ID key
        first — see :meth:`_merge_legacy_name_key`.

        The display name is refreshed when the caller is authoritative:
        the userbot (``telegram``) is the name authority for the aliased
        Telegram endpoints, so its sightings always update the name (a
        rename takes effect); a ``telegram-bot`` sighting never
        overwrites a userbot-set name.
        """
        with self._lock:
            if name:
                # ID-keyed lookup carrying the current display name: fold
                # any legacy name-keyed profile for the same human.
                self._merge_legacy_name_key(key, name, platform)
            row = None
            try:
                row = self.db.query_one(
                    "SELECT * FROM game_players WHERE player_key = ?", (key,)
                )
            except Exception:  # noqa: BLE001
                _log.debug("game_players read failed", exc_info=True)
            if row is not None:
                prof = Profile.from_row(row)
                if (name and name != prof.name
                        and _caller_sets_display_name(prof.platform,
                                                      platform)):
                    self._refresh_display(key, name, platform)
                    prof.name = name
                    if platform:
                        prof.platform = platform
                    prof.updated_at = time.time()
                return prof
            prof = Profile(key=key, name=name, platform=platform)
            if key != AI_PLAYER and self.db is not None:
                try:
                    with self.db.transaction():
                        self.db.execute(
                            "INSERT OR IGNORE INTO game_players (player_key, "
                            "platform, display, coins, points, wins, losses, "
                            "draws, streak, best_streak, games_played, xp, "
                            "per_game, items, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                prof.key, prof.platform, prof.name, prof.coins,
                                prof.points, prof.wins, prof.losses, prof.draws,
                                prof.streak, prof.best_streak, prof.games_played,
                                prof.xp, json.dumps(prof.per_game),
                                json.dumps(prof.items), prof.created_at,
                                prof.updated_at,
                            ),
                        )
                except Exception:  # noqa: BLE001
                    _log.debug("game_players create failed", exc_info=True)
                # Re-read: another thread may have created it first.
                try:
                    row = self.db.query_one(
                        "SELECT * FROM game_players WHERE player_key = ?", (key,)
                    )
                except Exception:  # noqa: BLE001
                    row = None
                if row is not None:
                    return Profile.from_row(row)
            return prof

    def set_display_name(self, key: str, name: str,
                         *, platform: str = "") -> Profile:
        """Explicitly rename a profile (e.g. the user changed their name).

        Unlike the guarded refresh inside :meth:`get`, this is
        authoritative — it always applies.  Prefer userbot-sourced
        names: a name set here from the ``telegram-bot`` endpoint will
        be overwritten again on the next userbot sighting.
        """
        name = (name or "").strip()[:40]
        with self._lock:
            prof = self.get(key)  # ensure the row exists; no name touch
            if not name or name == prof.name:
                return prof
            self._refresh_display(key, name, platform or prof.platform)
            prof.name = name
            if platform:
                prof.platform = platform
            prof.updated_at = time.time()
            return prof

    def _refresh_display(self, key: str, name: str, platform: str) -> None:
        """Targeted display/platform UPDATE. Caller must hold ``self._lock``."""
        if self.db is None:
            return
        try:
            with self.db.transaction():
                if platform:
                    self.db.execute(
                        "UPDATE game_players SET display = ?, platform = ?, "
                        "updated_at = ? WHERE player_key = ?",
                        (name, platform, time.time(), key),
                    )
                else:
                    self.db.execute(
                        "UPDATE game_players SET display = ?, "
                        "updated_at = ? WHERE player_key = ?",
                        (name, time.time(), key),
                    )
        except Exception:  # noqa: BLE001
            _log.debug("game_players display refresh failed", exc_info=True)

    def _merge_legacy_name_key(self, id_key: str, name: str,
                               platform: str) -> None:
        """Fold a legacy display-name-keyed profile into an ID-keyed one.

        Before stable ``sender_id`` keys, profiles were keyed by display
        name (``telegram:Mary``), so one human could own several profiles
        — ``telegram:Mary`` (userbot sighting) vs ``telegram:chfjdhx``
        (BotFather sighting).  When an ID-keyed lookup arrives carrying
        the current display name, any legacy profile under
        ``<canon_platform>:<name>`` belongs to the same human: merge it
        into the ID key.  Nothing is dropped:

        - xp: keep the HIGHER (same human — don't double-count levels)
        - coins/points/wins/losses/draws/games_played: summed
        - streak: larger absolute value wins; best_streak: max
        - per_game/items JSON ledgers: union, ID side wins conflicts
        - gear: renamed (instance ids are globally unique, no clashes)
        - skills: union by slug
        - attributes: max per column
        - titles: union by title_id, preserving an active title
        - game_stats: summed counters, max best_score

        Caller must hold ``self._lock``.  No-op when the key isn't an
        ID key, when no legacy row exists, or when the DB is down.
        After the merge the legacy row is gone, so repeat calls are a
        single cheap SELECT.
        """
        if self.db is None:
            return
        name = (name or "").strip()
        if not name:
            return
        plat_part, _, sender_part = (id_key or "").partition(":")
        if not sender_part.isdigit():
            return  # not an ID key — nothing to merge into
        canon = GAME_IDENTITY_ALIASES.get(plat_part.strip().lower(),
                                          plat_part.strip())
        legacy_key = f"{canon}:{name}"
        if legacy_key == id_key:
            return

        def _q1(sql: str, args: tuple = ()) -> Any:
            try:
                return self.db.query_one(sql, args)
            except Exception:  # noqa: BLE001
                return None

        def _qall(sql: str, args: tuple = ()) -> list:
            try:
                return self.db.query(sql, args) or []
            except Exception:  # noqa: BLE001
                return []

        def _cols(table: str) -> list[str]:
            try:
                rows = self.db.query(f"PRAGMA table_info({table})") or []
                return [str(r["name"]) for r in rows]
            except Exception:  # noqa: BLE001
                return []

        def _tables() -> set[str]:
            try:
                rows = self.db.query(
                    "SELECT name FROM sqlite_master WHERE type = 'table'") or []
                return {str(r["name"]) for r in rows}
            except Exception:  # noqa: BLE001
                return set()

        have = _tables()
        if "game_players" not in have:
            return
        legacy_row = _q1("SELECT * FROM game_players WHERE player_key = ?",
                         (legacy_key,))
        if legacy_row is None:
            return

        player_cols = _cols("game_players")
        col_list = player_cols or []
        if isinstance(legacy_row, dict):
            leg = dict(legacy_row)
        else:
            leg = dict(zip(col_list, legacy_row)) if col_list else {}
        if not leg:
            return

        # Ensure the ID-keyed row exists before merging into it.
        id_row = _q1("SELECT * FROM game_players WHERE player_key = ?",
                     (id_key,))
        if id_row is None:
            # Fresh ID key, legacy exists: rename the legacy row outright
            # across every per-player table — nothing to combine.
            try:
                with self.db.transaction():
                    for t in ("game_players", "game_gear", "game_skills",
                              "game_attributes", "game_titles", "game_stats"):
                        if t in have:
                            self.db.execute(
                                f"UPDATE {t} SET player_key = ? "
                                "WHERE player_key = ?",
                                (id_key, legacy_key))
            except Exception:  # noqa: BLE001
                _log.debug("id-identity merge: legacy rename failed",
                           exc_info=True)
            return
        if isinstance(id_row, dict):
            cur = dict(id_row)
        else:
            cur = dict(zip(col_list, id_row)) if col_list else {}

        def _num(d: dict, k: str) -> int:
            try:
                return int(d.get(k, 0) or 0)
            except Exception:  # noqa: BLE001
                return 0

        def _fnum(d: dict, k: str) -> float:
            try:
                return float(d.get(k, 0) or 0)
            except Exception:  # noqa: BLE001
                return 0.0

        def _jmerge(k: str) -> str:
            def _load(d: dict) -> dict:
                try:
                    v = json.loads(d.get(k) or "{}")
                    return v if isinstance(v, dict) else {}
                except Exception:  # noqa: BLE001
                    return {}
            merged = _load(leg)
            merged.update(_load(cur))  # ID side wins conflicts
            return json.dumps(merged)

        a_streak, c_streak = _num(leg, "streak"), _num(cur, "streak")
        streak = a_streak if abs(a_streak) > abs(c_streak) else c_streak
        sets: dict[str, Any] = {
            "xp": max(_num(leg, "xp"), _num(cur, "xp")),
            "coins": _num(leg, "coins") + _num(cur, "coins"),
            "points": _num(leg, "points") + _num(cur, "points"),
            "wins": _num(leg, "wins") + _num(cur, "wins"),
            "losses": _num(leg, "losses") + _num(cur, "losses"),
            "draws": _num(leg, "draws") + _num(cur, "draws"),
            "games_played": _num(leg, "games_played") + _num(cur, "games_played"),
            "streak": streak,
            "best_streak": max(_num(leg, "best_streak"),
                               _num(cur, "best_streak")),
            # The ID row's display was just set from the authoritative
            # sighting; keep it, fall back to the legacy name.
            "display": cur.get("display") or leg.get("display") or "",
            "created_at": min(_fnum(leg, "created_at"),
                              _fnum(cur, "created_at")),
            "updated_at": max(_fnum(leg, "updated_at"),
                              _fnum(cur, "updated_at")),
            "per_game": _jmerge("per_game"),
            "items": _jmerge("items"),
        }
        # Only write columns that actually exist.
        pcols = set(col_list)
        sets = {k: v for k, v in sets.items() if k in pcols}
        try:
            with self.db.transaction():
                if sets:
                    set_sql = ", ".join(f"{k} = ?" for k in sets)
                    self.db.execute(
                        f"UPDATE game_players SET {set_sql} "
                        "WHERE player_key = ?",
                        (*sets.values(), id_key))
                self.db.execute(
                    "DELETE FROM game_players WHERE player_key = ?",
                    (legacy_key,))
        except Exception:  # noqa: BLE001
            _log.debug("id-identity merge: game_players merge failed",
                       exc_info=True)
            return

        def _rename(table: str) -> None:
            if table in have:
                try:
                    self.db.execute(
                        f"UPDATE {table} SET player_key = ? "
                        "WHERE player_key = ?",
                        (id_key, legacy_key))
                except Exception:  # noqa: BLE001
                    _log.debug("id-identity merge: rename %s failed", table,
                               exc_info=True)

        # Gear: instance ids are globally unique — plain rename.
        _rename("game_gear")

        # Skills: drop legacy dupes by slug, then rename.
        if "game_skills" in have:
            try:
                with self.db.transaction():
                    canon_slugs = {str(r["slug"]) for r in _qall(
                        "SELECT slug FROM game_skills WHERE player_key = ?",
                        (id_key,))}
                    for r in _qall(
                            "SELECT slug FROM game_skills WHERE player_key = ?",
                            (legacy_key,)):
                        slug = str(r["slug"])
                        if slug in canon_slugs:
                            self.db.execute(
                                "DELETE FROM game_skills WHERE player_key = ? "
                                "AND slug = ?", (legacy_key, slug))
                    self.db.execute(
                        "UPDATE game_skills SET player_key = ? "
                        "WHERE player_key = ?", (id_key, legacy_key))
            except Exception:  # noqa: BLE001
                _log.debug("id-identity merge: game_skills merge failed",
                           exc_info=True)

        # Attributes: UNIQUE(player_key) — merge, keep max per column.
        if "game_attributes" in have:
            try:
                with self.db.transaction():
                    lr = _q1("SELECT strength, stamina, mana, intelligence, "
                             "unspent, level_applied FROM game_attributes "
                             "WHERE player_key = ?", (legacy_key,))
                    cr = _q1("SELECT strength, stamina, mana, intelligence, "
                             "unspent, level_applied FROM game_attributes "
                             "WHERE player_key = ?", (id_key,))
                    if lr and cr:
                        cols6 = ("strength", "stamina", "mana",
                                 "intelligence", "unspent", "level_applied")
                        merged = [max(int(lr.get(c, 0) or 0),
                                      int(cr.get(c, 0) or 0)) for c in cols6]
                        self.db.execute(
                            "UPDATE game_attributes SET strength = ?, "
                            "stamina = ?, mana = ?, intelligence = ?, "
                            "unspent = ?, level_applied = ? "
                            "WHERE player_key = ?", (*merged, id_key))
                        self.db.execute(
                            "DELETE FROM game_attributes WHERE player_key = ?",
                            (legacy_key,))
                    else:
                        self.db.execute(
                            "UPDATE game_attributes SET player_key = ? "
                            "WHERE player_key = ?", (id_key, legacy_key))
            except Exception:  # noqa: BLE001
                _log.debug("id-identity merge: game_attributes failed",
                           exc_info=True)

        # Titles: UNIQUE(player_key, title_id) — dedupe, rename, keep active.
        if "game_titles" in have:
            try:
                with self.db.transaction():
                    id_titles = {str(r["title_id"]) for r in _qall(
                        "SELECT title_id FROM game_titles WHERE player_key = ?",
                        (id_key,))}
                    legacy_active = [str(r["title_id"]) for r in _qall(
                        "SELECT title_id FROM game_titles WHERE player_key = ? "
                        "AND active = 1", (legacy_key,))]
                    id_has_active = bool(_q1(
                        "SELECT 1 FROM game_titles WHERE player_key = ? "
                        "AND active = 1", (id_key,)))
                    for r in _qall(
                            "SELECT title_id FROM game_titles WHERE player_key = ?",
                            (legacy_key,)):
                        tid = str(r["title_id"])
                        if tid in id_titles:
                            self.db.execute(
                                "DELETE FROM game_titles WHERE player_key = ? "
                                "AND title_id = ?", (legacy_key, tid))
                    self.db.execute(
                        "UPDATE game_titles SET player_key = ? "
                        "WHERE player_key = ?", (id_key, legacy_key))
                    if legacy_active and not id_has_active:
                        self.db.execute(
                            "UPDATE game_titles SET active = 1 "
                            "WHERE player_key = ? AND title_id = ?",
                            (id_key, legacy_active[0]))
                        self.db.execute(
                            "UPDATE game_titles SET active = 0 "
                            "WHERE player_key = ? AND title_id != ?",
                            (id_key, legacy_active[0]))
            except Exception:  # noqa: BLE001
                _log.debug("id-identity merge: game_titles merge failed",
                           exc_info=True)

        # Per-game stats: PK(player_key, game_name) — sum, max best.
        if "game_stats" in have:
            try:
                with self.db.transaction():
                    for row in _qall(
                            "SELECT game_name, games_played, games_won, "
                            "total_score, best_score, total_time, created_at "
                            "FROM game_stats WHERE player_key = ?",
                            (legacy_key,)):
                        gname = str(row["game_name"])
                        cr2 = _q1(
                            "SELECT games_played, games_won, total_score, "
                            "best_score, total_time, created_at "
                            "FROM game_stats WHERE player_key = ? AND "
                            "game_name = ?", (id_key, gname))
                        if cr2:
                            self.db.execute(
                                "UPDATE game_stats SET games_played = ?, "
                                "games_won = ?, total_score = ?, "
                                "best_score = ?, total_time = ?, "
                                "created_at = ? "
                                "WHERE player_key = ? AND game_name = ?",
                                (int(row["games_played"] or 0)
                                 + int(cr2["games_played"] or 0),
                                 int(row["games_won"] or 0)
                                 + int(cr2["games_won"] or 0),
                                 int(row["total_score"] or 0)
                                 + int(cr2["total_score"] or 0),
                                 max(int(row["best_score"] or 0),
                                     int(cr2["best_score"] or 0)),
                                 float(row["total_time"] or 0)
                                 + float(cr2["total_time"] or 0),
                                 min(float(row["created_at"] or 0),
                                     float(cr2["created_at"] or 0)),
                                 id_key, gname))
                            self.db.execute(
                                "DELETE FROM game_stats WHERE player_key = ? "
                                "AND game_name = ?", (legacy_key, gname))
                    self.db.execute(
                        "UPDATE game_stats SET player_key = ? "
                        "WHERE player_key = ?", (id_key, legacy_key))
            except Exception:  # noqa: BLE001
                _log.debug("id-identity merge: game_stats merge failed",
                           exc_info=True)

        _log.info("id-identity merge: folded %s into %s", legacy_key, id_key)

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
    def add_xp(self, player_key: str, amount: int) -> int:
        """Atomically add XP. Returns the new XP total.

        Single UPDATE — two concurrent awards can't lose one.  Creates
        the row first if the player is new.
        """
        amount = max(0, int(amount))
        with self._lock:
            self.get(player_key)  # ensure the row exists
            try:
                with self.db.transaction():
                    self.db.execute(
                        "UPDATE game_players SET xp = xp + ?, "
                        "updated_at = ? WHERE player_key = ?",
                        (amount, time.time(), player_key),
                    )
            except Exception:  # noqa: BLE001
                _log.debug("game_players add_xp failed", exc_info=True)
                return 0
            try:
                row = self.db.query_one(
                    "SELECT xp FROM game_players WHERE player_key = ?",
                    (player_key,),
                )
                return int(row["xp"]) if row else 0
            except Exception:  # noqa: BLE001
                return 0

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
