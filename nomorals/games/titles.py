"""Earnable titles: flair the player wears next to their name.

Titles unlock through arena achievements and milestones — a Dragonslayer
is someone who actually slew an S-rank, not someone who typed it.  The
player picks their active title with ``/title``; the arena shows it in
the intro so every fight opens with who you are.

Unlock conditions are achievement IDs, ``skill:<slug>`` for learning a
skill, or ``default`` for titles everyone starts with.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = [
    "Title", "TITLE_CATALOG", "TitleStore",
    "active_title", "display_name",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class Title:
    """One earnable title."""

    id: str
    name: str
    desc: str
    unlock: str  # achievement id | "skill:<slug>" | "default"


TITLE_CATALOG: tuple[Title, ...] = (
    Title("novice", "Novice", "Where every legend starts.", "default"),
    Title("gladiator", "Gladiator", "Win an arena battle.", "arena_win"),
    Title("on_fire", "On Fire", "Reach a 5-win streak.", "arena_streak_5"),
    Title("unstoppable", "Unstoppable", "Reach a 10-win streak.",
          "arena_streak_10"),
    Title("relentless", "Relentless", "Reach a 15-win streak.",
          "arena_streak_15"),
    Title("war_machine", "War Machine", "Reach a 20-win streak.",
          "arena_streak_20"),
    Title("immortal", "Immortal", "Reach a 25-win streak.",
          "arena_streak_25"),
    Title("dragonslayer", "Dragonslayer", "Defeat an S-rank hunter.",
          "arena_s_rank"),
    Title("stormbreaker", "Stormbreaker", "Defeat an SS-rank hunter.",
          "arena_ss_rank"),
    Title("legend_killer", "Legend Killer", "Defeat an X-rank hunter.",
          "arena_x_rank"),
    Title("mythslayer", "Mythslayer", "Defeat a myth-foe hunter.",
          "arena_myth_foe"),
    Title("untouched", "Untouched", "Win without taking damage.",
          "arena_flawless"),
    Title("giant_slayer", "Giant Slayer", "Defeat a stronger foe.",
          "arena_upset"),
    Title("brutal", "Brutal", "Land a brutal finish.", "arena_brutal"),
    Title("duelist", "Duelist", "Win a PvP duel.", "arena_pvp_win"),
    Title("boss_hunter", "Boss Hunter", "Defeat a raid boss.",
          "arena_raid_win"),
    Title("raid_mvp", "Raid MVP", "Deal the most damage in a raid.",
          "arena_raid_mvp"),
    Title("centurion", "Centurion", "Win 100 arena battles.",
          "arena_100_wins"),
    Title("master_of_arts", "Master of Arts", "Upgrade a skill to tier III.",
          "arena_tier3"),
    Title("myth_forged", "Myth Forged", "Equip a full myth-tier set.",
          "arena_myth_set"),
    Title("purist", "Purist", "Win without using a potion.",
          "arena_no_potion"),
    Title("cutyps_heir", "Cutyp's Heir", "Learn the Slaying Force.",
          "skill:slaying_force"),
)


def _title_map() -> dict[str, Title]:
    return {t.id: t for t in TITLE_CATALOG}


class TitleStore:
    """Persistent per-player titles. Backed by ``game_titles``.

    Unlocking is idempotent; exactly one title is active at a time.
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    def _ensure(self) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS game_titles ("
                "id TEXT PRIMARY KEY, player_key TEXT NOT NULL, "
                "title_id TEXT NOT NULL, unlocked_at REAL NOT NULL DEFAULT 0, "
                "active INTEGER NOT NULL DEFAULT 0, "
                "UNIQUE(player_key, title_id))")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_titles_player "
                "ON game_titles(player_key)")
        except Exception:  # noqa: BLE001
            _log.debug("game_titles ensure failed", exc_info=True)

    def _grant_default(self, player_key: str) -> None:
        """Everyone starts as a Novice."""
        try:
            self.db.execute(
                "INSERT OR IGNORE INTO game_titles "
                "(id, player_key, title_id, unlocked_at, active) "
                "VALUES (?, ?, 'novice', ?, 1)",
                (new_id(), player_key, time.time()))
        except Exception:  # noqa: BLE001
            _log.debug("novice grant failed", exc_info=True)

    def unlocked(self, player_key: str) -> list[str]:
        """Title IDs this player has unlocked, in unlock order."""
        self._ensure()
        self._grant_default(player_key)
        try:
            rows = self.db.query(
                "SELECT title_id FROM game_titles WHERE player_key = ? "
                "ORDER BY unlocked_at", (player_key,))
            return [str(r["title_id"]) for r in rows
                    if r["title_id"] in _title_map()]
        except Exception:  # noqa: BLE001
            _log.debug("game_titles list failed", exc_info=True)
            return ["novice"]

    def unlock(self, player_key: str, title_id: str) -> bool:
        """Unlock a title. True if newly unlocked."""
        if title_id not in _title_map():
            return False
        self._ensure()
        self._grant_default(player_key)
        try:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO game_titles "
                "(id, player_key, title_id, unlocked_at, active) "
                "VALUES (?, ?, ?, ?, 0)",
                (new_id(), player_key, title_id, time.time()))
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            _log.debug("game_titles unlock failed", exc_info=True)
            return False

    def check_unlocks(self, player_key: str,
                      unlocked_achievements: set[str] | None = None,
                      learned_skills: set[str] | None = None) -> list[str]:
        """Unlock every title whose condition is met. Returns the names
        of newly unlocked titles."""
        self._ensure()
        self._grant_default(player_key)
        if unlocked_achievements is None:
            try:
                from .achievements import get_achievements
                unlocked_achievements = {
                    a["id"] for a in get_achievements(self.db, player_key)}
            except Exception:  # noqa: BLE001
                unlocked_achievements = set()
        if learned_skills is None:
            try:
                from .skills import SkillStore
                learned_skills = set(
                    SkillStore(self.db).learned(player_key))
            except Exception:  # noqa: BLE001
                learned_skills = set()
        new: list[str] = []
        cmap = _title_map()
        for title in TITLE_CATALOG:
            cond = title.unlock
            met = (cond == "default"
                   or cond in (unlocked_achievements or set())
                   or (cond.startswith("skill:")
                       and cond[6:] in (learned_skills or set())))
            if met and self.unlock(player_key, title.id):
                new.append(cmap[title.id].name)
        return new

    def set_active(self, player_key: str, title_id: str) -> bool:
        """Make an unlocked title the active one."""
        if title_id not in self.unlocked(player_key):
            return False
        try:
            self.db.execute(
                "UPDATE game_titles SET active = 0 WHERE player_key = ?",
                (player_key,))
            self.db.execute(
                "UPDATE game_titles SET active = 1 "
                "WHERE player_key = ? AND title_id = ?",
                (player_key, title_id))
            return True
        except Exception:  # noqa: BLE001
            _log.debug("game_titles set_active failed", exc_info=True)
            return False

    def active(self, player_key: str) -> str:
        """The active title's display name, or ''."""
        self._ensure()
        self._grant_default(player_key)
        try:
            rows = self.db.query(
                "SELECT title_id FROM game_titles "
                "WHERE player_key = ? AND active = 1 LIMIT 1",
                (player_key,))
            if rows:
                title = _title_map().get(str(rows[0]["title_id"]))
                if title:
                    return title.name
        except Exception:  # noqa: BLE001
            _log.debug("game_titles active read failed", exc_info=True)
        return ""


def active_title(db: Any, player_key: str) -> str:
    """The player's active title name, or ''."""
    try:
        return TitleStore(db).active(player_key)
    except Exception:  # noqa: BLE001
        return ""


def display_name(db: Any, player_key: str, name: str) -> str:
    """``name`` with the active title prepended, if any."""
    title = active_title(db, player_key)
    return f"{title} {name}" if title else name
