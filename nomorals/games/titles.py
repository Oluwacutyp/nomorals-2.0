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
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = [
    "Title", "TITLE_CATALOG", "TitleStore",
    "active_title", "display_name",
    "title_battle_effects", "describe_title_effects",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class Title:
    """One earnable title."""

    id: str
    name: str
    desc: str
    unlock: str  # achievement id | "skill:<slug>" | "default"
    # ── battle effects: meaningful choices, not just cosmetic ──
    # Keys: "atk", "def", "max_hp" (flat bonuses), "boss_dmg_pct"
    # (extra damage vs raid bosses), "xp_pct" (bonus XP),
    # "combo_pct" (dual-cast success bonus, 0-1).
    effects: dict = field(default_factory=dict)


#: Title id → human-readable effect description (shown in /title).
def _fx(**kw: Any) -> dict:
    return dict(kw)


TITLE_CATALOG: tuple[Title, ...] = (
    Title("novice", "Novice", "Where every legend starts.", "default"),
    Title("gladiator", "Gladiator", "Win an arena battle.", "arena_win",
          _fx(atk=2)),
    Title("on_fire", "On Fire", "Reach a 5-win streak.", "arena_streak_5",
          _fx(atk=3)),
    Title("unstoppable", "Unstoppable", "Reach a 10-win streak.",
          "arena_streak_10", _fx(atk=5)),
    Title("relentless", "Relentless", "Reach a 15-win streak.",
          "arena_streak_15", _fx(atk=7)),
    Title("war_machine", "War Machine", "Reach a 20-win streak.",
          "arena_streak_20", _fx(atk=10)),
    Title("immortal", "Immortal", "Reach a 25-win streak.",
          "arena_streak_25", _fx(atk=12, max_hp=20)),
    Title("dragonslayer", "Dragonslayer", "Defeat an S-rank hunter.",
          "arena_s_rank", _fx(atk=5, boss_dmg_pct=0.05)),
    Title("stormbreaker", "Stormbreaker", "Defeat an SS-rank hunter.",
          "arena_ss_rank", _fx(atk=8, boss_dmg_pct=0.08)),
    Title("legend_killer", "Legend Killer", "Defeat an X-rank hunter.",
          "arena_x_rank", _fx(atk=12, boss_dmg_pct=0.12)),
    Title("mythslayer", "Mythslayer", "Defeat a myth-foe hunter.",
          "arena_myth_foe", _fx(atk=15, boss_dmg_pct=0.15)),
    Title("untouched", "Untouched", "Win without taking damage.",
          "arena_flawless", _fx(def_=8, atk=-3)),
    Title("giant_slayer", "Giant Slayer", "Defeat a stronger foe.",
          "arena_upset", _fx(atk=4, xp_pct=0.10)),
    Title("brutal", "Brutal", "Land a brutal finish.", "arena_brutal",
          _fx(atk=6)),
    Title("duelist", "Duelist", "Win a PvP duel.", "arena_pvp_win",
          _fx(atk=4, def_=4)),
    Title("boss_hunter", "Boss Hunter", "Defeat a raid boss.",
          "arena_raid_win", _fx(boss_dmg_pct=0.10)),
    Title("raid_mvp", "Raid MVP", "Deal the most damage in a raid.",
          "arena_raid_mvp", _fx(atk=6, boss_dmg_pct=0.05)),
    Title("centurion", "Centurion", "Win 100 arena battles.",
          "arena_100_wins", _fx(atk=8, max_hp=30)),
    Title("master_of_arts", "Master of Arts", "Upgrade a skill to tier III.",
          "arena_tier3", _fx(combo_pct=0.10)),
    Title("myth_forged", "Myth Forged", "Equip a full myth-tier set.",
          "arena_myth_set", _fx(atk=10, def_=10)),
    Title("purist", "Purist", "Win without using a potion.",
          "arena_no_potion", _fx(def_=6)),
    Title("cutyps_heir", "Cutyp's Heir", "Learn the Slaying Force.",
          "skill:slaying_force", _fx(atk=8, combo_pct=0.05)),
    # ── unlikely scenarios: the strange glories ──
    Title("phoenix", "Phoenix", "Win with exactly 1 HP remaining.",
          "arena_1hp_win", _fx(max_hp=25)),
    Title("comeback_king", "Comeback King",
          "Win after falling below 20% HP.", "arena_comeback",
          _fx(atk=5, def_=5)),
    Title("persistent", "Persistent", "Lose 10 battles in a row — and keep "
          "fighting.", "arena_lose_10", _fx(xp_pct=0.15)),
    Title("underdog", "Underdog", "Win as the weaker fighter 5 times.",
          "arena_underdog_5", _fx(atk=6)),
    Title("pacifist", "Pacifist", "Win using only skills, never a basic "
          "attack.", "arena_skills_only", _fx(combo_pct=0.08)),
    Title("speedster", "Speedster", "Win in 3 turns or fewer.",
          "arena_fast_win", _fx(atk=7)),
    Title("survivor", "Survivor", "Survive 20 turns in one battle.",
          "arena_marathon", _fx(max_hp=40, def_=5)),
    Title("dual_master", "Dual Master", "Land 10 successful dual-casts.",
          "arena_dual_10", _fx(combo_pct=0.15)),
    Title("gambler", "Gambler", "Win a dual-cast with under 30% odds.",
          "arena_lucky_dual", _fx(combo_pct=0.10)),
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
            # Both UPDATEs must land together — otherwise two concurrent
            # set_active calls can leave two titles active.
            txn = getattr(self.db, "transaction", None)
            if txn is None:  # mock DBs in tests
                self.db.execute(
                    "UPDATE game_titles SET active = 0 WHERE player_key = ?",
                    (player_key,))
                self.db.execute(
                    "UPDATE game_titles SET active = 1 "
                    "WHERE player_key = ? AND title_id = ?",
                    (player_key, title_id))
            else:
                with txn():
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


def title_battle_effects(title_name: str) -> dict[str, Any]:
    """Battle bonuses granted by an equipped title.

    Returns flat stat bonuses (``atk``, ``def``, ``max_hp``) plus
    special keys: ``boss_dmg_pct``, ``xp_pct``, ``combo_pct``.
    Unknown titles grant nothing.
    """
    if not title_name:
        return {}
    name = title_name.strip().lower()
    for title in TITLE_CATALOG:
        if title.name.lower() == name or title.id == name:
            out: dict[str, Any] = {}
            for key, val in (title.effects or {}).items():
                # ``def_`` avoids the Python keyword in the catalog
                out["def" if key == "def_" else key] = val
            return out
    return {}


def describe_title_effects(title_name: str) -> str:
    """Human-readable effect line for a title, or ''."""
    fx = title_battle_effects(title_name)
    if not fx:
        return ""
    bits = []
    if fx.get("atk"):
        bits.append(f"{fx['atk']:+d} atk")
    if fx.get("def"):
        bits.append(f"{fx['def']:+d} def")
    if fx.get("max_hp"):
        bits.append(f"{fx['max_hp']:+d} max HP")
    if fx.get("boss_dmg_pct"):
        bits.append(f"+{int(fx['boss_dmg_pct'] * 100)}% vs bosses")
    if fx.get("xp_pct"):
        bits.append(f"+{int(fx['xp_pct'] * 100)}% XP")
    if fx.get("combo_pct"):
        bits.append(f"+{int(fx['combo_pct'] * 100)}% dual-cast")
    return ", ".join(bits)


def display_name(db: Any, player_key: str, name: str) -> str:
    """``name`` with the active title prepended, if any."""
    title = active_title(db, player_key)
    return f"{title} {name}" if title else name
