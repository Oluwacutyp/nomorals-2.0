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
    "ENEMY_SKILL_CATALOG", "lookup_skill", "is_enemy_skill",
    "ComboDef", "COMBO_CATALOG", "find_combo",
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
    # ── mana: the fuel for techniques ──
    mana_cost: int = 5    # mana burned to cast (actives only)
    # ── forbidden techniques (enemy-only) ──
    lifesteal_pct: float = 0.0  # heal this fraction of damage dealt
    poison_turns: int = 0       # venom: damage-over-time duration
    poison_dmg: int = 0         # venom: damage per turn
    atk_debuff: int = 0         # dread: reduce target attack
    def_debuff: int = 0         # crusher: reduce target defense
    debuff_turns: int = 0       # how long debuffs last
    frenzy: bool = False        # +50% damage when caster below half HP
    execute_mult: float = 0.0   # bonus mult when target is near death
    execute_below: float = 0.0  # ...below this HP fraction
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
    mana_cost: int | None = None
    atk_debuff: int | None = None
    def_debuff: int | None = None
    debuff_turns: int | None = None


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
        2, 350, cooldown=4, atk_buff=3, buff_turns=3, mana_cost=6,
        tiers=(
            T(1200, 5, "+5 attack for 4 turns — the ground trembles.",
              "Battle Roar", atk_buff=5, buff_turns=4),
            T(2800, 8, "+8 attack for 5 turns, recovers fast. "
              "The roar of a warlord.", "Primal Roar",
              atk_buff=8, buff_turns=5, cooldown=3),
        ))
    add("dragon_punch", "Dragon Punch", "tiger", "active",
        "A devastating 1.8× strike. Slow to recover.",
        3, 500, cooldown=3, mult=1.8, mana_cost=8,
        tiers=(
            T(1500, 5, "A rising 2.2× uppercut — the dragon takes wing.",
              "Dragon Rising Fist", mult=2.2),
            T(3800, 9, "A 2.8× imperial strike that recovers in 2 turns. "
              "Bow.", "Dragon Emperor Fist", mult=2.8, cooldown=2),
        ))
    add("whirlwind", "Whirlwind Kick", "tiger", "active",
        "Two sweeping 0.7× kicks in one turn.",
        5, 900, cooldown=3, mult=0.7, hits=2, mana_cost=8,
        tiers=(
            T(2200, 7, "Three 0.8× kicks — a storm in a circle.",
              "Tempest Kick", mult=0.8, hits=3),
            T(4500, 11, "Four 0.9× kicks, recovered in 2 turns. "
              "You are the weather.", "Hurricane Rend",
              mult=0.9, hits=4, cooldown=2),
        ))
    add("thousand_fists", "Thousand Fists", "tiger", "active",
        "Four blurring 0.5× strikes. The ultimate flurry.",
        8, 1500, cooldown=5, mult=0.5, hits=4, mana_cost=12,
        tiers=(
            T(3200, 10, "Five 0.55× strikes — the barrage thickens.",
              "Fivefold Barrage", mult=0.55, hits=5, cooldown=4),
            T(6500, 14, "Six 0.65× strikes. The room runs out of air.",
              "Myriad Fists", mult=0.65, hits=6, cooldown=4),
        ))
    add("iron_palm", "Iron Palm", "tiger", "active",
        "One palm, all your weight behind it — a 2.4× crushing blow. "
        "Slow, but nothing blocks a mountain.",
        9, 1800, cooldown=5, mult=2.4, mana_cost=12,
        tiers=(
            T(3600, 11, "2.8× — the air cracks before the palm lands.",
              "Mountain Palm", mult=2.8, cooldown=4),
            T(7000, 15, "3.4× in 4 turns. Mountains move.",
              "Titan's Palm", mult=3.4, cooldown=4),
        ))
    # ── active: snake is precise ──
    add("pressure_point", "Pressure Point", "snake", "active",
        "Strike a nerve cluster — ignores half the enemy's defense.",
        7, 1200, cooldown=4, mult=1.0, ignore_def_pct=0.5, mana_cost=10,
        tiers=(
            T(2600, 9, "1.2× through the guard — ignores 75% defense.",
              "Vital Strike", mult=1.2, ignore_def_pct=0.75, cooldown=3),
            T(5200, 12, "1.5× that ignores ALL defense. One touch, "
              "one ending.", "Death Touch",
              mult=1.5, ignore_def_pct=1.0, cooldown=3),
        ))
    add("viper_strike", "Viper Strike", "snake", "active",
        "Three lightning 0.6× bites — fast, precise, venomous rhythm.",
        6, 1100, cooldown=3, mult=0.6, hits=3, mana_cost=8,
        tiers=(
            T(2400, 8, "Four 0.7× strikes — the viper quickens.",
              "Cobra Barrage", mult=0.7, hits=4),
            T(5000, 12, "Five 0.8× strikes in 2 turns. Nowhere to hide.",
              "Serpent's Judgment", mult=0.8, hits=5, cooldown=2),
        ))
    # ── active: shadow evades ──
    add("shadow_step", "Shadow Step", "shadow", "active",
        "Vanish — the house's next attack misses completely.",
        4, 700, cooldown=4, dodge=True, mana_cost=6,
        tiers=(
            T(1600, 6, "The vanish comes quicker — 3-turn cooldown.",
              "Shadow Evasion", cooldown=3),
            T(3600, 10, "Vanish, then strike from the dark for 1.0× "
              "as you reappear.", "Phantom Mirage",
              counter_mult=1.0, cooldown=3),
        ))
    add("smoke_bomb", "Smoke Bomb", "shadow", "active",
        "Vanish in smoke — dodge the next attack AND −3 enemy attack "
        "for 2 turns. The dark fights for you.",
        5, 900, cooldown=5, dodge=True, atk_debuff=3, debuff_turns=2,
        mana_cost=8,
        tiers=(
            T(2000, 7, "Thicker smoke — −4 attack for 3 turns.",
              "Blinding Smoke", atk_debuff=4, debuff_turns=3, cooldown=4),
            T(4200, 11, "The smoke strikes back — 1.2× counter as you "
              "reappear, −5 attack for 3 turns.", "Nightmare Veil",
              counter_mult=1.2, atk_debuff=5, debuff_turns=3, cooldown=4),
        ))
    # ── active: crane endures ──
    add("second_wind", "Second Wind", "crane", "active",
        "Catch your breath mid-fight: restore 40% max HP. Once per battle.",
        6, 1000, cooldown=99, heal_pct=0.4, once_per_battle=True,
        mana_cost=10,
        tiers=(
            T(2200, 8, "Deeper breath — restore 55% max HP.",
              "Deep Breath", heal_pct=0.55),
            T(4500, 11, "Restore 75% max HP. Death will have to wait.",
              "Phoenix Renewal", heal_pct=0.75),
        ))
    add("crane_dance", "Crane Dance", "crane", "active",
        "Flow like water — dodge the next attack and restore 15% max HP "
        "as you move. Grace is armor.",
        7, 1300, cooldown=5, dodge=True, heal_pct=0.15, mana_cost=10,
        tiers=(
            T(2600, 9, "Restore 25% — the dance deepens.",
              "Heronsong", heal_pct=0.25, cooldown=4),
            T(5200, 13, "Restore 35% and strike back for 1.0× — beauty "
              "with teeth.", "Crane's Vengeance",
              heal_pct=0.35, counter_mult=1.0, cooldown=4),
        ))
    # ── signature: the Cutyp school ──
    add("slaying_force", "Slaying Force", "cutyp", "active",
        "The Cutyp signature technique — a cleaving 3.5× strike that "
        "ignores ALL enemy defense. The legacy made technique.",
        10, 5000, cooldown=6, mult=3.5, ignore_def_pct=1.0, mana_cost=18,
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


def _enemy_catalog() -> dict[str, SkillDef]:
    """Forbidden techniques — the house's own martial arts.

    These are NEVER learnable.  They exist so high-rank hunters feel
    genuinely alien: life drain, venom, debuffs, frenzy, executions.
    The arena resolves them for the house only; ``resolve_skill`` will
    not find them, ``SkillStore.learn`` rejects them, and ``/skill``
    never lists them.
    """
    defs: dict[str, SkillDef] = {}

    def add(slug: str, name: str, desc: str, cooldown: int,
            **kw: Any) -> None:
        defs[slug] = SkillDef(slug=slug, name=name, school="forbidden",
                              kind="active", desc=desc, level_req=99,
                              cost=0, cooldown=cooldown, **kw)

    add("soul_siphon", "Soul Siphon",
        "A 1.2× strike that drinks 60% of the damage as HP. "
        "The hunter feeds on you.",
        4, mult=1.2, lifesteal_pct=0.6)
    add("venom_fang", "Venom Fang",
        "A 0.8× bite that poisons — 6 damage a turn for 3 turns. "
        "The wound keeps bleeding.",
        5, mult=0.8, poison_turns=3, poison_dmg=6)
    add("bone_crusher", "Bone Crusher",
        "A 1.0× smash that cracks armor — −4 defense for 3 turns. "
        "Your guard means less and less.",
        4, mult=1.0, def_debuff=4, debuff_turns=3)
    add("blood_frenzy", "Blood Frenzy",
        "Five wild 0.45× strikes — and the bloodied hunter hits 50% "
        "harder below half HP. Do not let it bleed.",
        5, mult=0.45, hits=5, frenzy=True)
    add("dread_aura", "Dread Aura",
        "No strike — pure malice. −4 attack for 3 turns. "
        "Your arms feel like lead.",
        5, atk_debuff=4, debuff_turns=3)
    add("executioner", "Executioner's Mercy",
        "A 1.0× axe-fall — 2.5× when you are below 30% HP. "
        "It can smell the end.",
        4, mult=1.0, execute_mult=2.5, execute_below=0.30)
    return defs


#: slug → SkillDef, the forbidden catalog.  House-only.
ENEMY_SKILL_CATALOG: dict[str, SkillDef] = _enemy_catalog()


def lookup_skill(slug: str) -> SkillDef | None:
    """Find a skill in either catalog (learnable or forbidden).

    The house uses this; players go through ``resolve_skill``, which
    only sees the learnable catalog.
    """
    if not slug:
        return None
    return SKILL_CATALOG.get(slug) or ENEMY_SKILL_CATALOG.get(slug)


def is_enemy_skill(slug: str) -> bool:
    """True if this is a forbidden technique players can never learn."""
    return slug in ENEMY_SKILL_CATALOG

#: effect knobs a SkillTier may override on the base SkillDef.
_TIER_KNOBS = ("cooldown", "mult", "hits", "heal_pct", "atk_buff",
               "buff_turns", "def_bonus", "atk_bonus", "hp_bonus",
               "crit_bonus", "dodge", "counter_mult", "ignore_def_pct",
               "once_per_battle", "mana_cost",
               "atk_debuff", "def_debuff", "debuff_turns")


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


# ── dual-cast combos ───────────────────────────────────────────────────────
# Only complementary skills chain together, and only in the right order:
# a setup move first (buff, dodge, heal), then the payoff (strike,
# precision).  Reversed or unrelated pairs simply don't combo.
#
# Fields:
#   first, second — the ordered skill slugs (base slugs; tiers apply)
#   name         — the combo's display name
#   desc         — flavor text shown on a successful dual-cast
#   hp_cost_pct  — fraction of the caster's max HP sacrificed
#   extra_cd     — bonus cooldown turns added to BOTH skills
#   mana_cost    — mana burned on top of each skill's own cost
#   base_success — success chance at tier-1 skills (0-1); each tier
#                  above 1 on either skill makes it harder


@dataclass(frozen=True)
class ComboDef:
    """One dual-cast pairing: two skills, one devastating turn."""

    first: str
    second: str
    name: str
    desc: str
    # ── the combo's own strike (the payoff) ──
    mult: float = 2.0       # damage multiplier of the combined strike
    hits: int = 1           # number of strikes
    ignore_def_pct: float = 0.0  # defense ignored by the strike
    # ── costs ──
    hp_cost_pct: float = 0.10   # fraction of max HP sacrificed
    extra_cd: int = 2           # bonus cooldown on BOTH skills
    mana_cost: int = 10         # mana burned on top of skill costs
    base_success: float = 0.90  # success at tier-1 skills


def _combo_catalog() -> dict[tuple[str, str], ComboDef]:
    defs: dict[tuple[str, str], ComboDef] = {}

    def add(first: str, second: str, name: str, desc: str,
            **kw: Any) -> None:
        defs[(first, second)] = ComboDef(first=first, second=second,
                                         name=name, desc=desc, **kw)

    # ── buff → strike: roar, then hit while the echo lasts ──
    add("war_cry", "dragon_punch", "Dragon's Roar",
        "the roar still rings as the dragon descends — a 2.4× "
        "rising strike wrapped in fury.",
        mult=2.4)
    add("war_cry", "whirlwind", "Storm's Fury",
        "the war cry becomes wind — three 1.0× kicks in a screaming "
        "circle.",
        mult=1.0, hits=3)
    add("war_cry", "thousand_fists", "War God's Barrage",
        "every fist carries the roar — six 0.7× strikes, no mercy.",
        mult=0.7, hits=6)
    add("war_cry", "pressure_point", "Surgical Strike",
        "fury focuses to a needle's point — 1.6× through 75% of "
        "their guard.",
        mult=1.6, ignore_def_pct=0.75)
    add("war_cry", "slaying_force", "Cutyp's Judgment",
        "the signature, crowned in fury — 4.5× that ignores all "
        "defense. The arena holds its breath.",
        mult=4.5, ignore_def_pct=1.0,
        hp_cost_pct=0.15, extra_cd=3, mana_cost=20, base_success=0.75)
    add("war_cry", "iron_palm", "Mountain Breaker",
        "fury behind a mountain — 3.0× that shakes the arena floor.",
        mult=3.0, hp_cost_pct=0.12)
    # ── dodge → precision: vanish, then strike where it hurts ──
    add("shadow_step", "pressure_point", "Assassin's Touch",
        "from nowhere, to the nerve — 1.4×, ignoring 75% defense, "
        "and they never saw you move.",
        mult=1.4, ignore_def_pct=0.75)
    add("shadow_step", "dragon_punch", "Phantom Strike",
        "the dark itself throws the punch — 2.2× from behind.",
        mult=2.2)
    add("shadow_step", "whirlwind", "Night Cyclone",
        "a storm with no center — three 0.9× kicks out of the dark.",
        mult=0.9, hits=3)
    add("shadow_step", "viper_strike", "Serpent's Ambush",
        "four 0.8× bites from the dark — the viper was never there.",
        mult=0.8, hits=4)
    add("smoke_bomb", "iron_palm", "Crushing Dark",
        "the smoke hides a mountain — 2.8× through the blinded guard.",
        mult=2.8, hp_cost_pct=0.12)
    # ── heal → buff: recover, then rise stronger ──
    add("second_wind", "war_cry", "Phoenix Rising",
        "breath returns, and with it rage — heal, then +5 attack "
        "for 4 turns. Death will have to wait.",
        mult=0.0, hp_cost_pct=0.05, mana_cost=5)
    add("second_wind", "dragon_punch", "Reborn Fang",
        "the healed body strikes harder — 2.0× with fresh blood.",
        mult=2.0, hp_cost_pct=0.08)
    # ── grace → precision: the crane's path ──
    add("crane_dance", "pressure_point", "Flowing Needle",
        "grace becomes precision — 1.4×, ignoring 75% defense, "
        "while the dance still shields you.",
        mult=1.4, ignore_def_pct=0.75)
    return defs


#: (first_slug, second_slug) → ComboDef.  Order matters.
COMBO_CATALOG: dict[tuple[str, str], ComboDef] = _combo_catalog()


def find_combo(first: str, second: str) -> ComboDef | None:
    """The combo for this ordered pair, or None if they don't chain."""
    return COMBO_CATALOG.get(((first or "").strip().lower(),
                              (second or "").strip().lower()))


def combo_success_chance(combo: ComboDef, tier1: int, tier2: int,
                         intelligence: int = 0) -> float:
    """Success chance for a dual-cast.

    Base chance minus 8% per tier above 1 on either skill (higher
    mastery = harder weave), plus 1% per intelligence point, clamped to
    5%–95%.
    """
    chance = float(combo.base_success)
    chance -= 0.08 * (max(1, int(tier1)) - 1)
    chance -= 0.08 * (max(1, int(tier2)) - 1)
    chance += 0.01 * max(0, int(intelligence))
    return max(0.05, min(0.95, chance))


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
            # one row per player+skill — guards against double-learn races
            # (two concurrent /skill learn must not create duplicates or
            # double-charge; the second INSERT becomes a no-op).
            try:
                self.db.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "uq_game_skills_player_slug "
                    "ON game_skills(player_key, slug)")
            except Exception:  # noqa: BLE001 - e.g. pre-existing dupes
                _log.debug("game_skills unique index failed",
                           exc_info=True)
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
        or the skill has no upgrades.

        Atomic: the UPDATE only fires when the row is still at the tier
        we read, so two concurrent upgrades can't both succeed on the
        same tier (the loser sees rowcount 0 and the caller refunds).
        """
        if slug not in SKILL_CATALOG:
            return False
        defn = SKILL_CATALOG[slug]
        if max_tier(defn) < 2:
            return False
        self._ensure()
        if self.db is None:
            return False
        cur = self.tier(player_key, slug)
        if cur >= max_tier(defn):
            return False
        # learned check folded into the UPDATE's rowcount: if the player
        # never learned it, no row matches and rowcount is 0.
        try:
            cursor = self.db.execute(
                "UPDATE game_skills SET tier = ? "
                "WHERE player_key = ? AND slug = ? AND tier = ?",
                (cur + 1, player_key, slug, cur))
            # None cursor = mock DB in tests — verify via re-read.
            if cursor is None:
                return self.tier(player_key, slug) == cur + 1
            if cursor.rowcount == 0:
                return False
            # tier III mastery earns an achievement
            if cur + 1 >= 3:
                try:
                    from .achievements import unlock_achievement
                    unlock_achievement(self.db, player_key, "arena_tier3")
                except Exception:  # noqa: BLE001
                    pass
            return True
        except Exception:  # noqa: BLE001
            _log.warning("game_skills upgrade failed", exc_info=True)
            return False

    # ── writes ─────────────────────────────────────────────────────────────
    def learn(self, player_key: str, slug: str) -> bool:
        """Record a learned skill. Idempotent — False if already known.

        Atomic: INSERT OR IGNORE on the (player_key, slug) unique index,
        so two concurrent learns can't double-insert (or double-charge —
        the caller refunds when this returns False).
        """
        if slug not in SKILL_CATALOG:
            return False
        self._ensure()
        try:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO game_skills (id, player_key, slug, learned_at) "
                "VALUES (?, ?, ?, ?)",
                (new_id(), player_key, slug, time.time()))
            # None cursor = mock DB in tests — verify via has().
            if cursor is None:
                return self.has(player_key, slug)
            return cursor.rowcount > 0
        except Exception:  # noqa: BLE001
            _log.warning("game_skills learn failed", exc_info=True)
            return False
