"""Dynamic enemy generation for the battle arena.

Every arena fight spawns a fresh opponent: a named hunter with their own
rolled gear (weapon + armor, grades gated by hunter rank), their own
learned skills, and a potion count that grows with rank.  Higher ranks
roll better grades and more skills — and a power-target backstop bumps
the gear up when the rolled enemy would otherwise be trivial next to
the player, so climbing never feels free.

The house's ``slaying_force`` is never rolled: the Cutyp signature
technique stays the player's legacy.
"""

from __future__ import annotations

import random
from typing import Any

#: gear grades each hunter rank may roll.  Myth is reserved for the
#: named Cutyp legacy set and never spawns on enemies.
GRADE_POOLS: dict[int, tuple[str, ...]] = {
    0: ("common", "rare"),          # E
    1: ("rare",),                   # D
    2: ("rare", "epic"),            # C
    3: ("epic",),                   # B
    4: ("epic", "legendary"),       # A
    5: ("legendary",),              # S
}

#: how many active skills each rank rolls (0 = bare-knuckle brawlers).
SKILL_COUNTS: dict[int, int] = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
}

#: potion count by rank — veterans pack more.
POTIONS: dict[int, int] = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3,
}

#: chance a C+-rank enemy rolls a matched set (storm/shadow) instead
#: of random pieces — set combos aren't the player's toy alone.
SET_CHANCE: dict[int, float] = {
    0: 0.0, 1: 0.0, 2: 0.25, 3: 0.40, 4: 0.50, 5: 0.60,
}

ENEMY_SKILL_POOL = (
    "war_cry", "dragon_punch", "whirlwind", "thousand_fists",
    "pressure_point", "shadow_step", "second_wind",
)

_ENEMY_FIRST = (
    "Gore", "Mara", "Vex", "Rusk", "Dain", "Sable", "Korr", "Juno",
    "Pike", "Tarn", "Vessa", "Odo", "Rin", "Kess", "Bran", "Zev",
)

_ENEMY_TITLES: dict[int, tuple[str, ...]] = {
    0: ("the Cutthroat", "the Brawler", "the Desperate"),
    1: ("the Duelist", "the Blade", "the Scarred"),
    2: ("the Reaver", "the Iron Fang", "the Stormcrow"),
    3: ("the Warlord", "the Crimson Edge", "the Nightmare"),
    4: ("the Executioner", "the Hollow Blade", "the Dreadnought"),
    5: ("the Render", "the World-Eater", "the Unbroken"),
}


def roll_enemy_name(rng: random.Random, rank_idx: int) -> str:
    """A flavorful name, e.g. ``Vex the Render``."""
    rank_idx = max(0, min(5, int(rank_idx)))
    return (f"{rng.choice(_ENEMY_FIRST)} "
            f"{rng.choice(_ENEMY_TITLES[rank_idx])}")


def _gear_pool(rank_idx: int) -> tuple[str, ...]:
    return GRADE_POOLS[max(0, min(5, int(rank_idx)))]


def roll_enemy_gear(rng: random.Random,
                    rank_idx: int) -> dict[str, dict[str, Any]]:
    """Roll ``{"weapon": gear_dict, "armor": gear_dict}`` for an enemy.

    Gear dicts match the shop ``to_dict()`` shape (slug, name, atk,
    def, grade, ...).  C+-rank enemies sometimes roll a matched
    storm/shadow set for the combo attack.
    """
    from .gear import GEAR_CATALOG
    rank_idx = max(0, min(5, int(rank_idx)))
    if rng.random() < SET_CHANCE[rank_idx]:
        set_name = rng.choice(("storm", "shadow"))
        weapon = GEAR_CATALOG[f"{set_name}_{'katana' if set_name == 'storm' else 'rapier'}"].to_dict()
        armor = GEAR_CATALOG[f"{set_name}_{'plate' if set_name == 'storm' else 'mail'}"].to_dict()
        return {"weapon": weapon, "armor": armor}
    pool = _gear_pool(rank_idx)
    weapons = [d for d in GEAR_CATALOG.values()
               if d.slot == "weapon" and d.grade in pool
               and not d.set_name and not d.unbreakable]
    armors = [d for d in GEAR_CATALOG.values()
              if d.slot == "armor" and d.grade in pool
              and not d.set_name and not d.unbreakable]
    weapon = rng.choice(weapons).to_dict() if weapons else None
    armor = rng.choice(armors).to_dict() if armors else None
    out: dict[str, dict[str, Any]] = {}
    if weapon:
        out["weapon"] = weapon
    if armor:
        out["armor"] = armor
    return out


def roll_enemy_skills(rng: random.Random,
                      rank_idx: int) -> dict[str, int]:
    """Roll ``{slug: tier}`` active skills for an enemy of this rank."""
    rank_idx = max(0, min(5, int(rank_idx)))
    count = SKILL_COUNTS[rank_idx]
    if count <= 0:
        return {}
    picks = rng.sample(ENEMY_SKILL_POOL, k=min(count, len(ENEMY_SKILL_POOL)))
    skills: dict[str, int] = {}
    for slug in picks:
        if rank_idx >= 4:
            tier = rng.choice((2, 2, 3))
        elif rank_idx >= 2:
            tier = rng.choice((1, 1, 2))
        else:
            tier = 1
        skills[slug] = tier
    return skills


def _gear_power(gear: dict[str, Any]) -> tuple[int, int]:
    """(atk, def) a gear set contributes at 50% battle-worn effectiveness."""
    weapon = gear.get("weapon") or {}
    armor = gear.get("armor") or {}
    return (int(weapon.get("atk", 0)) // 2, int(armor.get("def", 0)) // 2)


def roll_enemy(rng: random.Random, rank_idx: int,
               player_power: int = 0,
               foe_base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Roll a complete enemy: name, gear, skills, potions.

    ``player_power`` feeds the anti-triviality backstop, measured with
    the enemy's real pre-gear stats (``foe_base``) plus 50%-effective
    gear: if the enemy would land below 85% of the player's power the
    gear is re-rolled one grade pool up; above 125% it's re-rolled one
    pool down.  One adjustment each way, then accept whatever lands —
    the fight stays competitive but never a foregone conclusion.
    """
    rank_idx = max(0, min(5, int(rank_idx)))
    enemy: dict[str, Any] = {
        "name": roll_enemy_name(rng, rank_idx),
        "rank_idx": rank_idx,
        "gear": roll_enemy_gear(rng, rank_idx),
        "skills": roll_enemy_skills(rng, rank_idx),
        "potions": POTIONS[rank_idx],
    }
    if player_power > 0:
        from .power import fighter_power
        base = dict(foe_base) if foe_base else {"max_hp": 50,
                                               "atk": 10, "def": 5}

        def _with_gear(gear: dict[str, Any]) -> int:
            gatk, gdef = _gear_power(gear)
            stats = dict(base)
            stats["atk"] = int(stats.get("atk", 10)) + gatk
            stats["def"] = int(stats.get("def", 5)) + gdef
            return fighter_power(stats, tuple(enemy["skills"]),
                                 enemy["skills"])

        power = _with_gear(enemy["gear"])
        if power < player_power * 0.85:
            enemy["gear"] = roll_enemy_gear(rng, min(5, rank_idx + 1))
        elif power > player_power * 1.25:
            enemy["gear"] = roll_enemy_gear(rng, max(0, rank_idx - 1))
    return enemy
