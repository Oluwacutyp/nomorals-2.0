"""Learnable battle skills: martial arts for the arena.

Skills are the third pillar next to gear and levels.  Each skill is
learned once (``/skill learn <name>``), costs coins, and may require a
player level.  They persist in ``game_skills`` — learned skills are
yours forever.

* **Active** skills are battle moves, cast in the arena with
  ``skill <name>``.  They have cooldowns measured in turns.
* **Passive** skills are always on once learned — the arena folds
  them into your fighter at setup.

Schools are flavor with a mechanical identity:
tiger hits hard, crane endures, snake is precise, shadow evades.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = [
    "SkillDef", "SKILL_CATALOG", "SkillStore",
    "passive_bonuses", "resolve_skill",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class SkillDef:
    """The blueprint for one learnable skill."""

    slug: str
    name: str
    school: str          # tiger | crane | snake | shadow
    kind: str            # "active" | "passive"
    desc: str
    level_req: int
    cost: int            # coins to learn
    cooldown: int = 0    # active: turns before it can be cast again
    # effect knobs, interpreted by the arena
    mult: float = 0.0    # damage multiplier (active strikes)
    hits: int = 1        # number of strikes
    heal_pct: float = 0.0   # fraction of max HP restored
    atk_buff: int = 0    # temporary attack bonus
    buff_turns: int = 0
    def_bonus: int = 0   # passive defense
    atk_bonus: int = 0   # passive attack
    hp_bonus: int = 0    # passive max HP
    crit_bonus: float = 0.0  # passive crit chance
    dodge: bool = False  # active: dodge the next incoming attack
    ignore_def_pct: float = 0.0  # strike ignores this much enemy def
    once_per_battle: bool = False


def _catalog() -> dict[str, SkillDef]:
    defs: dict[str, SkillDef] = {}

    def add(slug: str, name: str, school: str, kind: str, desc: str,
            level_req: int, cost: int, **kw: Any) -> None:
        defs[slug] = SkillDef(slug=slug, name=name, school=school,
                              kind=kind, desc=desc, level_req=level_req,
                              cost=cost, **kw)

    # ── active: tiger hits hard ──
    add("war_cry", "War Cry", "tiger", "active",
        "+3 attack for 3 turns. Roar first, hit harder after.",
        2, 350, cooldown=4, atk_buff=3, buff_turns=3)
    add("dragon_punch", "Dragon Punch", "tiger", "active",
        "A devastating 1.8× strike. Slow to recover.",
        3, 500, cooldown=3, mult=1.8)
    add("whirlwind", "Whirlwind Kick", "tiger", "active",
        "Two sweeping 0.7× kicks in one turn.",
        5, 900, cooldown=3, mult=0.7, hits=2)
    add("thousand_fists", "Thousand Fists", "tiger", "active",
        "Four blurring 0.5× strikes. The ultimate flurry.",
        8, 1500, cooldown=5, mult=0.5, hits=4)
    # ── active: snake is precise ──
    add("pressure_point", "Pressure Point", "snake", "active",
        "Strike a nerve cluster — ignores half the enemy's defense.",
        7, 1200, cooldown=4, mult=1.0, ignore_def_pct=0.5)
    # ── active: shadow evades ──
    add("shadow_step", "Shadow Step", "shadow", "active",
        "Vanish — the house's next attack misses completely.",
        4, 700, cooldown=4, dodge=True)
    # ── active: crane endures ──
    add("second_wind", "Second Wind", "crane", "active",
        "Catch your breath mid-fight: restore 40% max HP. Once per battle.",
        6, 1000, cooldown=99, heal_pct=0.4, once_per_battle=True)
    # ── passive ──
    add("iron_skin", "Iron Skin", "crane", "passive",
        "Hardened body: +4 defense in every battle.",
        2, 400, def_bonus=4)
    add("tiger_stance", "Tiger Stance", "tiger", "passive",
        "Rooted strikes: +3 attack in every battle.",
        4, 800, atk_bonus=3)
    add("keen_eye", "Keen Eye", "snake", "passive",
        "See the opening: +10% crit chance in every battle.",
        5, 900, crit_bonus=0.10)
    add("stone_body", "Stone Body", "crane", "passive",
        "Deep reserves: +15 max HP in every battle.",
        6, 1000, hp_bonus=15)
    return defs


#: slug → SkillDef, the full learnable catalog.
SKILL_CATALOG: dict[str, SkillDef] = _catalog()


def resolve_skill(ref: str) -> SkillDef | None:
    """Fuzzy resolve a skill by slug, name, or substring."""
    ref = (ref or "").strip().lower()
    if not ref:
        return None
    if ref in SKILL_CATALOG:
        return SKILL_CATALOG[ref]
    for slug, defn in SKILL_CATALOG.items():
        if ref == slug or ref in slug or ref in defn.name.lower():
            return defn
    return None


def passive_bonuses(slugs: list[str]) -> dict[str, Any]:
    """Aggregate passive skill effects for a list of learned slugs."""
    out: dict[str, Any] = {"atk": 0, "def": 0, "max_hp": 0,
                           "crit": 0.0}
    for slug in slugs or []:
        defn = SKILL_CATALOG.get(slug)
        if defn is None or defn.kind != "passive":
            continue
        out["atk"] += defn.atk_bonus
        out["def"] += defn.def_bonus
        out["max_hp"] += defn.hp_bonus
        out["crit"] += defn.crit_bonus
    return out


class SkillStore:
    """Persistent per-player learned skills. Backed by ``game_skills``.

    All mutations are transactional; learning is idempotent — learning
    a skill you already know is a no-op, never a double charge (callers
    must still check coins *before* calling ``learn``).
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    # ── schema ─────────────────────────────────────────────────────────────
    def _ensure(self) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS game_skills ("
                "id TEXT PRIMARY KEY, player_key TEXT NOT NULL, "
                "slug TEXT NOT NULL, learned_at REAL NOT NULL DEFAULT 0)")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_skills_player "
                "ON game_skills(player_key)")
        except Exception:  # noqa: BLE001 - table may already exist
            _log.debug("game_skills ensure failed", exc_info=True)

    # ── reads ──────────────────────────────────────────────────────────────
    def learned(self, player_key: str) -> list[str]:
        """Slugs of every skill this player has learned, in learn order."""
        self._ensure()
        try:
            rows = self.db.query(
                "SELECT slug FROM game_skills WHERE player_key = ? "
                "ORDER BY learned_at", (player_key,))
            return [str(r["slug"]) for r in rows
                    if r["slug"] in SKILL_CATALOG]
        except Exception:  # noqa: BLE001
            _log.debug("game_skills list failed", exc_info=True)
            return []

    def has(self, player_key: str, slug: str) -> bool:
        return slug in self.learned(player_key)

    # ── writes ─────────────────────────────────────────────────────────────
    def learn(self, player_key: str, slug: str) -> bool:
        """Record a learned skill. Idempotent — False if already known."""
        if slug not in SKILL_CATALOG:
            return False
        self._ensure()
        if self.has(player_key, slug):
            return False
        try:
            self.db.execute(
                "INSERT INTO game_skills (id, player_key, slug, learned_at) "
                "VALUES (?, ?, ?, ?)",
                (new_id(), player_key, slug, time.time()))
            return True
        except Exception:  # noqa: BLE001
            _log.warning("game_skills learn failed", exc_info=True)
            return False
