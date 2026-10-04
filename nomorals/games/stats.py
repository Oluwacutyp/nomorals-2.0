"""RPG attributes: strength, stamina, mana, intelligence.

The fourth pillar next to gear, levels, and skills.  Every player has
four attributes that shape how they fight:

* **Strength** — raw muscle.  Each point adds to physical attack.
* **Stamina** — endurance.  Each point adds max HP and a little defense.
* **Mana** — the fuel for techniques.  Active skills cost mana to cast;
  it regenerates a little every turn.  Running dry means fighting
  bare-handed until it recovers.
* **Intelligence** — battle sense.  Each point sharpens skill power and
  raises dual-cast combo success rates.

Attributes grow two ways:

1. **Level-ups** grant attribute points (3 per level) — the player
   spends them with ``/stats``.  Unspent points never expire.
2. **Gear** can carry attribute bonuses (a warlord's belt grants
   strength, a sage's robe grants intelligence).

Storage: ``game_attributes`` holds base attributes + unspent points per
player.  Gear bonuses are folded in at battle setup, never stored.
"""
from __future__ import annotations

import time
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = [
    "STAT_NAMES", "StatBlock", "StatStore",
    "POINTS_PER_LEVEL", "MANA_REGEN_PER_TURN",
    "apply_stats_to_fighter", "describe_stats",
]

_log = get_logger(__name__)

#: The four attributes, in display order.
STAT_NAMES = ("strength", "stamina", "mana", "intelligence")

#: Attribute points granted per player level (level 1 starts with 0).
POINTS_PER_LEVEL = 3

#: Mana restored at the start of each of the player's turns.
MANA_REGEN_PER_TURN = 8

#: Base mana pool before bonuses.
BASE_MANA = 30


class StatBlock:
    """One player's attributes: base values + unspent points."""

    __slots__ = ("strength", "stamina", "mana", "intelligence",
                 "unspent", "level_applied")

    def __init__(self, strength: int = 0, stamina: int = 0,
                 mana: int = 0, intelligence: int = 0,
                 unspent: int = 0, level_applied: int = 1) -> None:
        self.strength = max(0, int(strength))
        self.stamina = max(0, int(stamina))
        self.mana = max(0, int(mana))
        self.intelligence = max(0, int(intelligence))
        self.unspent = max(0, int(unspent))
        # highest player level whose points have been granted
        self.level_applied = max(1, int(level_applied))

    def to_dict(self) -> dict[str, int]:
        return {"strength": self.strength, "stamina": self.stamina,
                "mana": self.mana, "intelligence": self.intelligence,
                "unspent": self.unspent, "level_applied": self.level_applied}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "StatBlock":
        d = d or {}
        return cls(strength=d.get("strength", 0),
                   stamina=d.get("stamina", 0),
                   mana=d.get("mana", 0),
                   intelligence=d.get("intelligence", 0),
                   unspent=d.get("unspent", 0),
                   level_applied=d.get("level_applied", 1))

    def total(self, name: str, gear_bonus: int = 0) -> int:
        """Base + gear bonus for one attribute."""
        return max(0, int(getattr(self, name, 0)) + int(gear_bonus or 0))


def apply_stats_to_fighter(fighter: dict[str, Any], stats: StatBlock,
                           gear_bonus: dict[str, int] | None = None) -> None:
    """Fold RPG attributes into a battle-ready fighter dict.

    Mutates the fighter in place:

    * strength → +1 attack per 2 points
    * stamina → +3 max HP per point, +1 defense per 4 points
    * mana → sets the fighter's mana pool (base 30 + 2 per mana point)
    * intelligence → stored for skill-power and combo rolls

    Gear bonuses (from ``gear_stat_bonuses``) stack on top.
    """
    gear_bonus = gear_bonus or {}
    str_ = stats.total("strength", gear_bonus.get("strength", 0))
    sta = stats.total("stamina", gear_bonus.get("stamina", 0))
    man = stats.total("mana", gear_bonus.get("mana", 0))
    int_ = stats.total("intelligence", gear_bonus.get("intelligence", 0))

    fighter["atk"] = int(fighter.get("atk", 10)) + str_ // 2
    hp_gain = sta * 3
    fighter["max_hp"] = int(fighter.get("max_hp", 50)) + hp_gain
    fighter["hp"] = int(fighter.get("hp", 50)) + hp_gain
    fighter["def"] = int(fighter.get("def", 5)) + sta // 4
    fighter["mana"] = BASE_MANA + man * 2
    fighter["max_mana"] = BASE_MANA + man * 2
    fighter["intelligence"] = int_
    fighter["stat_strength"] = str_
    fighter["stat_stamina"] = sta


def gear_stat_bonuses(loadout: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Sum the RPG stat bonuses of equipped gear pieces."""
    out = {name: 0 for name in STAT_NAMES}
    for piece in (loadout or {}).values():
        for name in STAT_NAMES:
            out[name] += int((piece or {}).get(f"stat_{name}", 0))
    return out


def describe_stats(stats: StatBlock,
                   gear_bonus: dict[str, int] | None = None) -> str:
    """Human-readable stat sheet."""
    gear_bonus = gear_bonus or {}
    lines = []
    descs = {
        "strength": "physical attack",
        "stamina": "max HP + defense",
        "mana": "skill fuel",
        "intelligence": "skill power + combo luck",
    }
    for name in STAT_NAMES:
        base = int(getattr(stats, name))
        bonus = int(gear_bonus.get(name, 0))
        total = base + bonus
        extra = f" (+{bonus} gear)" if bonus else ""
        lines.append(f"  {name.capitalize():<12} {total:<3} "
                     f"— {descs[name]}{extra}")
    lines.append(f"  Unspent points: {stats.unspent}")
    return "\n".join(lines)


class StatStore:
    """Persistent per-player RPG attributes. Backed by ``game_attributes``.

    Level-up points are granted at the actual level-up event in the
    game engine (see ``engine.py`` award path). The grant is idempotent:
    the atomic UPDATE only fires when ``level_applied`` still matches,
    so points for a level can never be granted twice.

    Display paths (``/stats``) must NEVER grant — they only read.
    A battle-setup mirror in the engine tops up any points missed due
    to transient failures, as a safety net.
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    # ── schema ─────────────────────────────────────────────────────────────
    def _ensure(self) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS game_attributes ("
                "id TEXT PRIMARY KEY, player_key TEXT NOT NULL UNIQUE, "
                "strength INTEGER NOT NULL DEFAULT 0, "
                "stamina INTEGER NOT NULL DEFAULT 0, "
                "mana INTEGER NOT NULL DEFAULT 0, "
                "intelligence INTEGER NOT NULL DEFAULT 0, "
                "unspent INTEGER NOT NULL DEFAULT 0, "
                "level_applied INTEGER NOT NULL DEFAULT 1, "
                "updated_at REAL NOT NULL DEFAULT 0)")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_attributes_player "
                "ON game_attributes(player_key)")
        except Exception:  # noqa: BLE001
            _log.debug("game_attributes ensure failed", exc_info=True)

    # ── reads ──────────────────────────────────────────────────────────────
    def get(self, player_key: str) -> StatBlock:
        """This player's attributes (fresh block if never seen)."""
        self._ensure()
        if self.db is None:
            return StatBlock()
        try:
            rows = self.db.query(
                "SELECT strength, stamina, mana, intelligence, unspent, "
                "level_applied FROM game_attributes WHERE player_key = ?",
                (player_key,))
            if rows:
                r = rows[0]
                return StatBlock(
                    strength=r["strength"], stamina=r["stamina"],
                    mana=r["mana"], intelligence=r["intelligence"],
                    unspent=r["unspent"],
                    level_applied=r["level_applied"])
        except Exception:  # noqa: BLE001
            _log.debug("game_attributes read failed", exc_info=True)
        return StatBlock()

    # ── level-up grants ────────────────────────────────────────────────────
    def grant_level_points(self, player_key: str,
                           level: int) -> tuple[int, "StatBlock"]:
        """Top up unspent points for levels gained.

        Returns (points_added, updated_stats). The stats block is the
        freshly-saved state — callers should use it directly instead of
        doing a separate read that might hit a transient failure.

        Atomic: the UPDATE only applies when level_applied still matches
        what we read, so two concurrent grants can't double-count the
        same levels.
        """
        level = max(1, int(level))
        self._ensure()
        if self.db is None:
            return 0, StatBlock()
        # ensure a row exists (idempotent)
        try:
            self.db.execute(
                "INSERT OR IGNORE INTO game_attributes (id, player_key, "
                "level_applied) VALUES (?, ?, 1)",
                (new_id(), player_key))
        except Exception:  # noqa: BLE001
            pass
        stats = self.get(player_key)
        if level <= stats.level_applied:
            return 0, stats
        new_levels = level - stats.level_applied
        points = new_levels * POINTS_PER_LEVEL
        try:
            cursor = self.db.execute(
                "UPDATE game_attributes SET unspent = unspent + ?, "
                "level_applied = ?, updated_at = ? "
                "WHERE player_key = ? AND level_applied = ?",
                (points, level, time.time(), player_key,
                 stats.level_applied))
            # rowcount 0 = someone else applied these levels first.
            # (None cursor = mock DB in tests — trust the write.)
            if cursor is not None and getattr(cursor, "rowcount", 1) == 0:
                # someone else applied these levels first — re-read
                return 0, self.get(player_key)
        except Exception:  # noqa: BLE001
            _log.warning("grant_level_points failed", exc_info=True)
            return 0, stats
        return points, self.get(player_key)

    # ── spending ───────────────────────────────────────────────────────────
    def spend(self, player_key: str, name: str,
              points: int = 1, stats: "StatBlock | None" = None) -> tuple[bool, str]:
        """Spend unspent points on one attribute.

        Atomic: the UPDATE only fires when unspent still covers the
        spend, so two concurrent spends can't double-spend the same
        points. The optional ``stats`` is used for the display message
        only, never trusted for the balance check.
        """
        name = (name or "").strip().lower()
        if name not in STAT_NAMES:
            return False, (f"unknown attribute — choose from "
                           f"{', '.join(STAT_NAMES)}.")
        points = max(1, int(points))
        self._ensure()
        if self.db is None:
            return False, "stats storage is unavailable."
        try:
            before = self.get(player_key)
            cursor = self.db.execute(
                f"UPDATE game_attributes SET {name} = {name} + ?, "
                f"unspent = unspent - ?, updated_at = ? "
                f"WHERE player_key = ? AND unspent >= ?",
                (points, points, time.time(), player_key, points))
            # rowcount 0 = insufficient funds (or a concurrent spend won).
            if cursor is not None and getattr(cursor, "rowcount", 1) == 0:
                cur = self.get(player_key)
                return False, (f"only {cur.unspent} unspent point(s) — "
                               f"level up to earn more.")
            fresh = self.get(player_key)
            # None cursor = mock DB: verify the spend actually landed.
            if fresh.unspent >= before.unspent:
                return False, (f"only {fresh.unspent} unspent point(s) — "
                               f"level up to earn more.")
            total = int(getattr(fresh, name))
            return True, (f"{name.capitalize()} +{points} → {total} "
                          f"({fresh.unspent} unspent left).")
        except Exception:  # noqa: BLE001
            _log.warning("stat spend failed", exc_info=True)
            return False, "couldn't spend the points — try again."

    def _save(self, player_key: str, stats: StatBlock) -> None:
        self._ensure()
        if self.db is None:
            return
        try:
            self.db.execute(
                "INSERT INTO game_attributes (id, player_key, strength, stamina, "
                "mana, intelligence, unspent, level_applied, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(player_key) DO UPDATE SET "
                "strength=excluded.strength, stamina=excluded.stamina, "
                "mana=excluded.mana, intelligence=excluded.intelligence, "
                "unspent=excluded.unspent, "
                "level_applied=excluded.level_applied, "
                "updated_at=excluded.updated_at",
                (new_id(), player_key, stats.strength, stats.stamina,
                 stats.mana, stats.intelligence, stats.unspent,
                 stats.level_applied, time.time()))
        except Exception:  # noqa: BLE001
            _log.warning("game_attributes save failed", exc_info=True)
