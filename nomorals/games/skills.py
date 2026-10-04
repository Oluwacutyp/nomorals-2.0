"""Learnable battle skills: martial arts for the arena.

Skills are the third pillar next to gear and levels.  Each skill is
learned once (``/skill learn <name>``), costs coins, and may require a
player level.  They persist in ``game_skills`` — learned skills are
yours forever.

* **Active** skills are battle moves, cast in the arena with
  ``skill <name>``.  They have cooldowns measured in turns.
* **Passive** skills are always on once learned — the arena folds
  them into your fighter at setup.

The most exciting actives are **upgradeable**: ``/skill upgrade
<name>`` raises them through tiers (II, III), each stronger, cheaper
on cooldown, or with new effects.  Passives stay single-tier — they
are the quiet foundation, not the spectacle.

Schools are flavor with a mechanical identity:
tiger hits hard, crane endures, snake is precise, shadow evades —
and cutyp is the owner's signature school, devastating and absolute.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = [
    "SkillDef", "SkillTier", "SKILL_CATALOG", "SkillStore",
    "passive_bonuses", "resolve_skill",
    "max_tier", "effective_def", "tier_name",
]

_log = get_logger(__name__)

#: Roman numerals for skill tiers (tier 1 is the base, unnamed).
TIER_ROMAN = {1: "", 2: "II", 3: "III", 4: "IV", 5: "V"}


def _roman(tier: int) -> str:
    if tier in TIER_ROMAN:
        return TIER_ROMAN[tier]
    # fall back to additive notation past V
    return "V" + "I" * (tier - 5)


def tier_name(defn: SkillDef | str, tier: int) -> str:
    """Display name of a skill at a tier (base name for tier 1)."""
    return effective_def(defn, tier).name


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
    counter_mult: float = 0.0  # dodge-style skills: strike back at mult
    ignore_def_pct: float = 0.0  # strike ignores this much enemy def
    once_per_battle: bool = False
    tiers: tuple = field(default=())  # SkillTier steps above the base


@dataclass(frozen=True)
class SkillTier:
    """One upgrade step above a skill's base (tier 2, 3, ...).

    Only the fields set (non-``None``) override the base skill;
    ``cost``/``level_req`` are the price and gate for *this* upgrade.
    """

    cost: int
    level_req: int
    desc: str
    name: str = ""       # "" → base name + roman numeral
    cooldown: int | None = None
    mult: float | None = None
    hits: int | None = None
    heal_pct: float | None = None
    atk_buff: int | None = None
    buff_turns: int | None = None
    def_bonus: int | None = None
    atk_bonus: int | None = None
    hp_bonus: int | None = None
    crit_bonus: float | None = None
    dodge: bool | None = None
    counter_mult: float | None = None
    ignore_def_pct: float | None = None
    once_per_battle: bool | None = None


def _catalog() -> dict[str, SkillDef]:
    defs: dict[str, SkillDef] = {}

    def T(cost: int, level_req: int, desc: str, name: str = "",
          **kw: Any) -> SkillTier:
        return SkillTier(cost=cost, level_req=level_req, desc=desc,
                         name=name, **kw)

    def add(slug: str, name: str, school: str, kind: str, desc: str,
            level_req: int, cost: int, tiers: tuple = (),
            **kw: Any) -> None:
        defs[slug] = SkillDef(slug=slug, name=name, school=school,
                              kind=kind, desc=desc, level_req=level_req,
                              cost=cost, tiers=tiers, **kw)

    # ── active: tiger hits hard ──
    add("war_cry", "War Cry", "tiger", "active",
        "+3 attack for 3 turns. Roar first, hit harder after.",
        2, 350, cooldown=4, atk_buff=3, buff_turns=3,
        tiers=(
            T(1200, 5, "+5 attack for 4 turns — the ground trembles.",
              "Battle Roar", atk_buff=5, buff_turns=4),
            T(2800, 8, "+8 attack for 5 turns, recovers fast. "
              "The roar of a warlord.", "Primal Roar",
              atk_buff=8, buff_turns=5, cooldown=3),
        ))
    add("dragon_punch", "Dragon Punch", "tiger", "active",
        "A devastating 1.8× strike. Slow to recover.",
        3, 500, cooldown=3, mult=1.8,
        tiers=(
            T(1500, 5, "A rising 2.2× uppercut — the dragon takes wing.",
              "Dragon Rising Fist", mult=2.2),
            T(3800, 9, "A 2.8× imperial strike that recovers in 2 turns. "
              "Bow.", "Dragon Emperor Fist", mult=2.8, cooldown=2),
        ))
    add("whirlwind", "Whirlwind Kick", "tiger", "active",
        "Two sweeping 0.7× kicks in one turn.",
        5, 900, cooldown=3, mult=0.7, hits=2,
        tiers=(
            T(2200, 7, "Three 0.8× kicks — a storm in a circle.",
              "Tempest Kick", mult=0.8, hits=3),
            T(4500, 11, "Four 0.9× kicks, recovered in 2 turns. "
              "You are the weather.", "Hurricane Rend",
              mult=0.9, hits=4, cooldown=2),
        ))
    add("thousand_fists", "Thousand Fists", "tiger", "active",
        "Four blurring 0.5× strikes. The ultimate flurry.",
        8, 1500, cooldown=5, mult=0.5, hits=4,
        tiers=(
            T(3200, 10, "Five 0.55× strikes — the barrage thickens.",
              "Fivefold Barrage", mult=0.55, hits=5, cooldown=4),
            T(6500, 14, "Six 0.65× strikes. The room runs out of air.",
              "Myriad Fists", mult=0.65, hits=6, cooldown=4),
        ))
    # ── active: snake is precise ──
    add("pressure_point", "Pressure Point", "snake", "active",
        "Strike a nerve cluster — ignores half the enemy's defense.",
        7, 1200, cooldown=4, mult=1.0, ignore_def_pct=0.5,
        tiers=(
            T(2600, 9, "1.2× through the guard — ignores 75% defense.",
              "Vital Strike", mult=1.2, ignore_def_pct=0.75, cooldown=3),
            T(5200, 12, "1.5× that ignores ALL defense. One touch, "
              "one ending.", "Death Touch",
              mult=1.5, ignore_def_pct=1.0, cooldown=3),
        ))
    # ── active: shadow evades ──
    add("shadow_step", "Shadow Step", "shadow", "active",
        "Vanish — the house's next attack misses completely.",
        4, 700, cooldown=4, dodge=True,
        tiers=(
            T(1600, 6, "The vanish comes quicker — 3-turn cooldown.",
              "Shadow Evasion", cooldown=3),
            T(3600, 10, "Vanish, then strike from the dark for 1.0× "
              "as you reappear.", "Phantom Mirage",
              counter_mult=1.0, cooldown=3),
        ))
    # ── active: crane endures ──
    add("second_wind", "Second Wind", "crane", "active",
        "Catch your breath mid-fight: restore 40% max HP. Once per battle.",
        6, 1000, cooldown=99, heal_pct=0.4, once_per_battle=True,
        tiers=(
            T(2200, 8, "Deeper breath — restore 55% max HP.",
              "Deep Breath", heal_pct=0.55),
            T(4500, 11, "Restore 75% max HP. Death will have to wait.",
              "Phoenix Renewal", heal_pct=0.75),
        ))
    # ── signature: the Cutyp school ──
    add("slaying_force", "Slaying Force", "cutyp", "active",
        "The Cutyp signature technique — a cleaving 3.5× strike that "
        "ignores ALL enemy defense. The legacy made technique.",
        10, 5000, cooldown=6, mult=3.5, ignore_def_pct=1.0,
        tiers=(
            T(9000, 13, "4.2×, recovers in 5 turns. The air splits first.",
              "Slaying Force: Sever", mult=4.2, cooldown=5),
            T(15000, 16, "5.0× in 4 turns. Nothing stands after.",
              "Slaying Force: Annihilation", mult=5.0, cooldown=4),
        ))
    # ── passive (single-tier: the quiet foundation) ──
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

#: effect knobs a SkillTier may override on the base SkillDef.
_TIER_KNOBS = ("cooldown", "mult", "hits", "heal_pct", "atk_buff",
               "buff_turns", "def_bonus", "atk_bonus", "hp_bonus",
               "crit_bonus", "dodge", "counter_mult", "ignore_def_pct",
               "once_per_battle")


def max_tier(defn: SkillDef | str) -> int:
    """Highest tier for a skill (1 = single-tier, no upgrades)."""
    if isinstance(defn, str):
        defn = SKILL_CATALOG.get(defn)
        if defn is None:
            return 1
    return 1 + len(defn.tiers)


_effective_cache: dict[tuple[str, int], SkillDef] = {}


def effective_def(defn: SkillDef | str, tier: int) -> SkillDef:
    """The skill blueprint as it fights at ``tier`` (1-based).

    Tier 1 is the base.  Higher tiers merge the SkillTier overrides and
    take the tier's display name/desc/cost/level_req.  Unknown slugs or
    out-of-range tiers fall back to the base def.
    """
    base = SKILL_CATALOG.get(defn) if isinstance(defn, str) else defn
    if base is None:
        raise KeyError(f"unknown skill {defn!r}")
    tier = max(1, min(int(tier or 1), max_tier(base)))
    key = (base.slug, tier)
    hit = _effective_cache.get(key)
    if hit is not None:
        return hit
    if tier == 1:
        _effective_cache[key] = base
        return base
    up = base.tiers[tier - 2]
    overrides = {k: getattr(up, k) for k in _TIER_KNOBS
                 if getattr(up, k) is not None}
    name = up.name or f"{base.name} {_roman(tier)}"
    merged = replace(base, name=name, desc=up.desc,
                     cost=up.cost, level_req=up.level_req, **overrides)
    _effective_cache[key] = merged
    return merged


def _tier_aliases() -> dict[str, str]:
    """Display name (lower) of every tier → slug, for fuzzy resolving."""
    out: dict[str, str] = {}
    for slug, defn in SKILL_CATALOG.items():
        for tier in range(2, max_tier(defn) + 1):
            out[effective_def(defn, tier).name.lower()] = slug
    return out


_TIER_ALIASES = _tier_aliases()


def resolve_skill(ref: str) -> SkillDef | None:
    """Fuzzy resolve a skill by slug, name, tier name, or substring.

    Returns the *base* def — callers look up the player's tier and use
    ``effective_def`` for the fighting stats.
    """
    ref = (ref or "").strip().lower()
    if not ref:
        return None
    if ref in SKILL_CATALOG:
        return SKILL_CATALOG[ref]
    if ref in _TIER_ALIASES:
        return SKILL_CATALOG[_TIER_ALIASES[ref]]
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
            # tier column for skill upgrades (older rows default to 1)
            try:
                self.db.execute(
                    "ALTER TABLE game_skills ADD COLUMN tier "
                    "INTEGER NOT NULL DEFAULT 1")
            except Exception:  # noqa: BLE001 - already migrated
                pass
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

    # ── tiers ──────────────────────────────────────────────────────────────
    def tier(self, player_key: str, slug: str) -> int:
        """Current tier (1-based) of a learned skill; 1 if unknown."""
        self._ensure()
        if self.db is None:
            return 1
        try:
            rows = self.db.query(
                "SELECT tier FROM game_skills "
                "WHERE player_key = ? AND slug = ?",
                (player_key, slug))
            if rows:
                return max(1, int(rows[0]["tier"] or 1))
        except Exception:  # noqa: BLE001
            _log.debug("game_skills tier read failed", exc_info=True)
        return 1

    def tiers(self, player_key: str) -> dict[str, int]:
        """slug → tier for every learned skill (mirror for battles)."""
        self._ensure()
        if self.db is None:
            return {}
        try:
            rows = self.db.query(
                "SELECT slug, tier FROM game_skills WHERE player_key = ?",
                (player_key,))
            return {str(r["slug"]): max(1, int(r["tier"] or 1))
                    for r in rows if r["slug"] in SKILL_CATALOG}
        except Exception:  # noqa: BLE001
            _log.debug("game_skills tiers read failed", exc_info=True)
            return {}

    def upgrade(self, player_key: str, slug: str) -> bool:
        """Raise a learned skill one tier. False if not learned, maxed,
        or the skill has no upgrades."""
        if slug not in SKILL_CATALOG:
            return False
        defn = SKILL_CATALOG[slug]
        if max_tier(defn) < 2:
            return False
        self._ensure()
        if self.db is None or not self.has(player_key, slug):
            return False
        cur = self.tier(player_key, slug)
        if cur >= max_tier(defn):
            return False
        try:
            self.db.execute(
                "UPDATE game_skills SET tier = ? "
                "WHERE player_key = ? AND slug = ?",
                (cur + 1, player_key, slug))
            return True
        except Exception:  # noqa: BLE001
            _log.warning("game_skills upgrade failed", exc_info=True)
            return False

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
