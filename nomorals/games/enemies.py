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
#: named Cutyp legacy set and never spawns on enemies — unless the
#: player brings myth gear themselves (``myth_foe=True``), in which case
#: S-rank hunters answer with their own myth-forged kit.
GRADE_POOLS: dict[int, tuple[str, ...]] = {
    0: ("common", "rare"),          # E
    1: ("rare",),                   # D
    2: ("rare", "epic"),            # C
    3: ("epic",),                   # B
    4: ("epic", "legendary"),       # A
    5: ("legendary",),              # S
    6: ("legendary",),              # SS
    7: ("legendary", "myth"),       # X — legends forge their own myth
}

#: enemy-only myth gear — myth-forged, but never the Cutyp legacy set.
#: These pieces exist only as dicts (never in the shop catalog), with
#: the same shape as ``GearDef.to_dict()``.  S-rank hunters wield them
#: when facing a myth-equipped player.
MYTH_FOE_WEAPONS: tuple[dict[str, Any], ...] = (
    {"slug": "myth_foe_blade", "name": "Mythril Reaver [myth]",
     "cost": 0, "slot": "weapon", "kind": "mythril_blade",
     "grade": "myth", "set": "", "atk": 90, "def": 0,
     "durability": 999},
    {"slug": "myth_foe_fang", "name": "Void-Touched Katana [myth]",
     "cost": 0, "slot": "weapon", "kind": "void_katana",
     "grade": "myth", "set": "", "atk": 84, "def": 0,
     "durability": 999},
    {"slug": "myth_foe_maul", "name": "Worldbreaker Maul [myth]",
     "cost": 0, "slot": "weapon", "kind": "worldbreaker",
     "grade": "myth", "set": "", "atk": 96, "def": 0,
     "durability": 999},
)

MYTH_FOE_ARMORS: tuple[dict[str, Any], ...] = (
    {"slug": "myth_foe_plate", "name": "Dreadplate of the Void [myth]",
     "cost": 0, "slot": "armor", "kind": "dreadplate",
     "grade": "myth", "set": "", "atk": 0, "def": 72,
     "durability": 999},
    {"slug": "myth_foe_mail", "name": "Abyssal Dragonscale [myth]",
     "cost": 0, "slot": "armor", "kind": "abyssal_scale",
     "grade": "myth", "set": "", "atk": 0, "def": 66,
     "durability": 999},
    {"slug": "myth_foe_shroud", "name": "Nightmare Shroud [myth]",
     "cost": 0, "slot": "armor", "kind": "nightmare_shroud",
     "grade": "myth", "set": "", "atk": 0, "def": 78,
     "durability": 999},
)

#: how many active skills each rank rolls (0 = bare-knuckle brawlers).
SKILL_COUNTS: dict[int, int] = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,   # SS
    7: 4,   # X
}

#: gear effectiveness by hunter rank — the fraction of a rolled piece's
#: stats the house actually fights with.  Higher ranks maintain their
#: kit better; myth-foe hunters fight at full power (handled in
#: _apply_house_gear).  This is the "class" of each rank: an S-rank's
#: legendary blade bites far harder than an E-rank's common one, even
#: before grades differ.
GEAR_EFFECTIVENESS: dict[int, float] = {
    0: 0.40,  # E — rusty, notched
    1: 0.45,  # D
    2: 0.50,  # C
    3: 0.55,  # B — veterans keep their kit
    4: 0.60,  # A
    5: 0.65,  # S — near shop-fresh
    6: 0.70,  # SS
    7: 0.75,  # X — legends maintain their arms
}

#: difficulty → enemy adjustments.  The arena's /game arena <difficulty>
#: flag isn't a label: harder modes field genuinely stronger hunters.
#: stat_mult scales the hunter's base stats; pool_shift moves the gear
#: grade pool up/down a rank; skill_shift / xskill_shift adjust how
#: many techniques (and forbidden arts) they bring.
DIFFICULTY_MODS: dict[str, dict[str, float]] = {
    "easy":   {"stat_mult": 0.80, "pool_shift": -1,
               "skill_shift": -1, "xskill_shift": -1},
    "normal": {"stat_mult": 1.00, "pool_shift": 0,
               "skill_shift": 0, "xskill_shift": 0},
    "hard":   {"stat_mult": 1.15, "pool_shift": 1,
               "skill_shift": 0, "xskill_shift": 0},
    "expert": {"stat_mult": 1.30, "pool_shift": 1,
               "skill_shift": 1, "xskill_shift": 1},
}

#: human one-liner per difficulty for the arena intro.
DIFFICULTY_BLURB: dict[str, str] = {
    "easy": "the hunter looks green — an easy mark.",
    "normal": "",
    "hard": "the hunter fights at 115% — careful.",
    "expert": "the hunter fights at 130% with extra techniques — good luck.",
}

#: potion count by rank — veterans pack more.
POTIONS: dict[int, int] = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3,
    6: 4, 7: 5,
}

#: chance a C+-rank enemy rolls a matched set (storm/shadow) instead
#: of random pieces — set combos aren't the player's toy alone.
SET_CHANCE: dict[int, float] = {
    0: 0.0, 1: 0.0, 2: 0.25, 3: 0.40, 4: 0.50, 5: 0.60,
    6: 0.70, 7: 0.80,
}

ENEMY_SKILL_POOL = (
    "war_cry", "dragon_punch", "whirlwind", "thousand_fists",
    "pressure_point", "shadow_step", "second_wind",
)

#: forbidden techniques each rank may roll — the house's own martial
#: arts, never learnable.  Only veterans know them.
ENEMY_EXCLUSIVE_POOL = (
    "soul_siphon", "venom_fang", "bone_crusher",
    "blood_frenzy", "dread_aura", "executioner",
)

#: how many forbidden techniques each rank rolls on top of the normal
#: skill count.  E–C hunters fight clean; B+ fight dirty.
ENEMY_EXCLUSIVE_COUNTS: dict[int, int] = {
    0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2,
    6: 2, 7: 3,
}

_ENEMY_FIRST = (
    "Gore", "Mara", "Vex", "Rusk", "Dain", "Sable", "Korr", "Juno",
    "Pike", "Tarn", "Vessa", "Odo", "Rin", "Kess", "Bran", "Zev",
    "Hale", "Yrsa", "Drok", "Fenn", "Garr", "Hush", "Ivo", "Jex",
    "Karn", "Lira", "Moss", "Nyx", "Orr", "Pell", "Quill", "Rho",
    "Sarn", "Tove", "Ulric", "Vann", "Wren", "Xan", "Ysol", "Zara",
)

_ENEMY_TITLES: dict[int, tuple[str, ...]] = {
    0: ("the Cutthroat", "the Brawler", "the Desperate",
        "the Rat", "the Hungry", "the Cornered"),
    1: ("the Duelist", "the Blade", "the Scarred",
        "the Quick", "the Patient", "the Oathbound"),
    2: ("the Reaver", "the Iron Fang", "the Stormcrow",
        "the Relentless", "the Ashen", "the Gatekeeper"),
    3: ("the Warlord", "the Crimson Edge", "the Nightmare",
        "the Butcher", "the Ironclad", "the Howling Dark"),
    4: ("the Executioner", "the Hollow Blade", "the Dreadnought",
        "the Reaper", "the Kingslayer", "the Maw"),
    5: ("the Render", "the World-Eater", "the Unbroken",
        "the Godslayer", "the Everburning", "the Thronebreaker"),
    6: ("the Calamity", "the Stormborn", "the Twice-Crowned",
        "the Harbinger", "the Doom of Armies", "the Unmade"),
    7: ("the Annihilator", "the End of Legends", "the Absolute",
        "the Final Argument", "the Death of Hope", "the Omega"),
}

# ---------------------------------------------------------------------------
# Enemy archetypes — not every foe is a hunter.  Each archetype reshapes
# the stat profile and draws from its own name pool, so fights feel
# different beyond the numbers.
# ---------------------------------------------------------------------------

#: archetype → stat multipliers (hp, atk, def) + flavor.
ARCHETYPES: dict[str, dict[str, Any]] = {
    "hunter": {"hp": 1.0, "atk": 1.0, "def": 1.0,
               "blurb": "a hunter",
               "names": _ENEMY_FIRST},
    "beast": {"hp": 1.45, "atk": 1.18, "def": 0.78,
              "blurb": "a beast",
              "names": (
                  "Ravage", "Gnash", "Howler", "Mauler", "Fang",
                  "Claw", "Rip", "Snarl", "Bloodmaw", "Thornback",
                  "Gorehorn", "Nightpelt", "Razorback", "Deathroll",
              )},
    "machine": {"hp": 1.12, "atk": 0.95, "def": 1.45,
                "blurb": "a war machine",
                "names": (
                  "Ironclad", "Siegebreaker", "Rustjaw", "Geargrinder",
                  "Piston", "Anvil", "Bulwark", "Cogsworth", "Furnace",
                  "Hammerfall", "Steelrain", "Ironmonger",
                )},
    "shade": {"hp": 0.78, "atk": 1.22, "def": 0.85,
              "blurb": "a shade",
              "names": (
                  "Whisper", "Gloom", "Umbra", "Wraith", "Hollow",
                  "Dusk", "Murk", "Eclipse", "Nightfall", "Specter",
                  "Vapor", "Shroud",
              )},
}

#: elite affixes — ~15% of enemies roll one.  The affix prefixes the
#: name and bends the stats; elites also drop better rewards.
ELITE_AFFIXES: dict[str, dict[str, Any]] = {
    "swift": {"hp": 0.9, "atk": 1.0, "def": 1.0, "dodge": True,
              "blurb": "moves like smoke"},
    "armored": {"hp": 1.1, "atk": 0.95, "def": 1.35,
                "blurb": "plated head to toe"},
    "vampiric": {"hp": 1.0, "atk": 1.1, "def": 0.9, "lifesteal": True,
                 "blurb": "drinks your strength"},
    "frenzied": {"hp": 0.85, "atk": 1.35, "def": 0.8,
                 "blurb": "frothing, wild-eyed"},
    "titanic": {"hp": 1.6, "atk": 1.1, "def": 1.1,
                "blurb": "simply enormous"},
}
ELITE_CHANCE = 0.15

#: boss names — one per rank tier, unique fights.  Bosses get phased
#: combat: below 30% HP they enrage (attack surge, new message).
BOSS_NAMES: dict[int, tuple[str, ...]] = {
    0: ("Rat-King Skree", "Mudfang the Desperate"),
    1: ("Duelist Corvus", "The Scarred Captain"),
    2: ("Ironjaw Grull", "Stormcaller Vex"),
    3: ("Warlord Kargath", "The Crimson Matriarch"),
    4: ("Executioner Morvain", "The Hollow King"),
    5: ("World-Eater Ythra", "The Unbroken Throne"),
    6: ("Calamity Zero", "The Twice-Crowned Tyrant"),
    7: ("The Absolute", "Omega Prime"),
}

#: titles reserved for myth-foe hunters — S-rank killers who rose to
#: meet a myth-equipped player.
_MYTH_FOE_TITLES: tuple[str, ...] = (
    "the Mythslayer", "the God-Eater", "the Unmaking",
    "the Final Verdict",
)


def roll_enemy_name(rng: random.Random, rank_idx: int,
                    myth_foe: bool = False,
                    archetype: str = "hunter") -> str:
    """A flavorful name, e.g. ``Vex the Render``.

    Myth-foe hunters (spawned against myth-equipped players) take a
    darker title — the player knows this one is different.  Non-hunter
    archetypes draw from their own name pools.
    """
    rank_idx = max(0, min(7, int(rank_idx)))
    arch = ARCHETYPES.get(archetype, ARCHETYPES["hunter"])
    if archetype != "hunter":
        # beasts/machines/shades get a raw name, no hunter title
        return str(rng.choice(arch["names"]))
    titles = (_MYTH_FOE_TITLES if myth_foe and rank_idx >= 5
              else _ENEMY_TITLES[rank_idx])
    return (f"{rng.choice(_ENEMY_FIRST)} "
            f"{rng.choice(titles)}")


def roll_elite(rng: random.Random) -> Optional[str]:
    """Maybe roll an elite affix (~15%).  Returns the affix slug or None."""
    if rng.random() < ELITE_CHANCE:
        return str(rng.choice(tuple(ELITE_AFFIXES)))
    return None


def _gear_pool(rank_idx: int, difficulty: str = "normal") -> tuple[str, ...]:
    """Grade pool for a rank, shifted by difficulty.

    Hard/expert hunters roll a rank up; easy hunters roll a rank down —
    difficulty genuinely changes what the house brings.
    """
    shift = int(DIFFICULTY_MODS.get(difficulty,
                                    DIFFICULTY_MODS["normal"])["pool_shift"])
    return GRADE_POOLS[max(0, min(7, int(rank_idx) + shift))]


def roll_enemy_gear(rng: random.Random,
                    rank_idx: int,
                    myth_foe: bool = False,
                    difficulty: str = "normal") -> dict[str, dict[str, Any]]:
    """Roll ``{"weapon": gear_dict, "armor": gear_dict}`` for an enemy.

    Gear dicts match the shop ``to_dict()`` shape (slug, name, atk,
    def, grade, ...).  C+-rank enemies sometimes roll a matched
    storm/shadow set for the combo attack.  B-rank and up always bring
    both a weapon and armor — veterans don't show up empty-handed.

    When ``myth_foe`` is true (the player brought myth gear), S-rank
    hunters roll from the enemy-only myth-forged kit instead — the
    Cutyp legacy set itself is never wielded by the house.

    ``difficulty`` shifts the grade pool: hard/expert hunters roll one
    rank higher, easy hunters one rank lower.
    """
    from .gear import GEAR_CATALOG
    rank_idx = max(0, min(7, int(rank_idx)))
    if myth_foe and rank_idx >= 5:
        return {
            "weapon": dict(rng.choice(MYTH_FOE_WEAPONS)),
            "armor": dict(rng.choice(MYTH_FOE_ARMORS)),
        }
    # X-rank legends forge their own myth — half the time they skip
    # the set roll and go straight for myth-forged kit.
    if rank_idx == 7 and rng.random() < 0.5:
        return {
            "weapon": dict(rng.choice(MYTH_FOE_WEAPONS)),
            "armor": dict(rng.choice(MYTH_FOE_ARMORS)),
        }
    if rng.random() < SET_CHANCE[rank_idx]:
        set_name = rng.choice(("storm", "shadow"))
        weapon = GEAR_CATALOG[f"{set_name}_{'katana' if set_name == 'storm' else 'rapier'}"].to_dict()
        armor = GEAR_CATALOG[f"{set_name}_{'plate' if set_name == 'storm' else 'mail'}"].to_dict()
        return {"weapon": weapon, "armor": armor}
    pool = _gear_pool(rank_idx, difficulty)
    weapons = [d for d in GEAR_CATALOG.values()
               if d.slot == "weapon" and d.grade in pool
               and not d.set_name and not d.unbreakable
               and not d.raid_only]
    armors = [d for d in GEAR_CATALOG.values()
              if d.slot == "armor" and d.grade in pool
              and not d.set_name and not d.unbreakable
              and not d.raid_only]
    # X-rank forges its own myth: supplement the catalog pool with the
    # enemy-only myth kit when the grade pool includes myth.
    if "myth" in pool:
        myth_weapons = [dict(w) for w in MYTH_FOE_WEAPONS]
        myth_armors = [dict(a) for a in MYTH_FOE_ARMORS]
        weapon_pool = ([d.to_dict() for d in weapons] + myth_weapons
                       if weapons else myth_weapons)
        armor_pool = ([d.to_dict() for d in armors] + myth_armors
                      if armors else myth_armors)
        weapon = rng.choice(weapon_pool) if weapon_pool else None
        armor = rng.choice(armor_pool) if armor_pool else None
    else:
        weapon = rng.choice(weapons).to_dict() if weapons else None
        armor = rng.choice(armors).to_dict() if armors else None
    out: dict[str, dict[str, Any]] = {}
    if weapon:
        out["weapon"] = weapon
    if armor:
        out["armor"] = armor
    return out


def roll_enemy_skills(rng: random.Random,
                      rank_idx: int,
                      myth_foe: bool = False,
                      difficulty: str = "normal") -> dict[str, int]:
    """Roll ``{slug: tier}`` active skills for an enemy of this rank.

    Higher ranks also roll forbidden techniques — the house's own
    martial arts that players can never learn.  Myth-foe hunters
    (S-rank vs a myth-equipped player) roll a third forbidden art:
    they came to kill a legend, and they brought everything.

    ``difficulty`` shifts the counts: easy hunters bring fewer
    techniques, expert hunters bring more.
    """
    rank_idx = max(0, min(7, int(rank_idx)))
    mods = DIFFICULTY_MODS.get(difficulty, DIFFICULTY_MODS["normal"])
    skills: dict[str, int] = {}
    count = max(0, SKILL_COUNTS[rank_idx]
                + int(mods["skill_shift"]))
    if count > 0:
        picks = rng.sample(ENEMY_SKILL_POOL,
                           k=min(count, len(ENEMY_SKILL_POOL)))
        for slug in picks:
            if rank_idx >= 4:
                tier = rng.choice((2, 2, 3))
            elif rank_idx >= 2:
                tier = rng.choice((1, 1, 2))
            else:
                tier = 1
            skills[slug] = tier
    # forbidden techniques: B+ hunters fight dirty
    xcount = max(0, ENEMY_EXCLUSIVE_COUNTS[rank_idx]
                 + int(mods["xskill_shift"]))
    if myth_foe and rank_idx >= 5:
        xcount = 3
    if xcount > 0:
        xpicks = rng.sample(ENEMY_EXCLUSIVE_POOL,
                            k=min(xcount, len(ENEMY_EXCLUSIVE_POOL)))
        for slug in xpicks:
            skills[slug] = 1  # forbidden arts have no tiers
    return skills


def _gear_power(gear: dict[str, Any],
                rank_idx: int = 2) -> tuple[int, int]:
    """(atk, def) a gear set contributes at the rank's effectiveness.

    Higher ranks maintain their kit better (see GEAR_EFFECTIVENESS) —
    an S-rank's legendary blade bites harder than an E-rank's common
    one even before grades differ.
    """
    eff = GEAR_EFFECTIVENESS.get(max(0, min(7, int(rank_idx))), 0.5)
    weapon = gear.get("weapon") or {}
    armor = gear.get("armor") or {}
    return (int(int(weapon.get("atk", 0)) * eff),
            int(int(armor.get("def", 0)) * eff))


def roll_enemy(rng: random.Random, rank_idx: int,
               player_power: int = 0,
               foe_base: dict[str, Any] | None = None,
               myth_foe: bool = False,
               difficulty: str = "normal",
               archetype: str = "hunter",
               force_elite: bool = False) -> dict[str, Any]:
    """Roll a complete enemy: name, gear, skills, potions.

    ``player_power`` feeds the anti-triviality backstop, measured with
    the enemy's real pre-gear stats (``foe_base``) plus rank-scaled gear
    effectiveness: if the enemy would land below 85% of the player's
    power the gear is re-rolled one grade pool up; above 125% it's
    re-rolled one pool down.  One adjustment each way, then accept
    whatever lands — the fight stays competitive but never a foregone
    conclusion.

    ``myth_foe`` marks a hunter spawned against a myth-equipped
    player: S-rank rolls myth-forged gear, a darker title, and an
    extra forbidden technique.  The Cutyp legacy set itself is never
    wielded by the house.

    ``difficulty`` (easy/normal/hard/expert) scales the hunter's base
    stats, shifts their gear grade pool, and adjusts their technique
    counts — the flag is a real dial, not a label.

    ``archetype`` (hunter/beast/machine/shade) reshapes the stat
    profile and name pool.  Beasts hit hard with big HP; machines
    shrug off damage; shades are fragile but lethal.

    ``force_elite`` (or the 15% roll) applies an elite affix — a
    named modifier with real stat bends and better rewards.
    """
    rank_idx = max(0, min(7, int(rank_idx)))
    mods = DIFFICULTY_MODS.get(difficulty, DIFFICULTY_MODS["normal"])
    mult = float(mods["stat_mult"])
    arch = ARCHETYPES.get(archetype, ARCHETYPES["hunter"])
    elite = roll_elite(rng) if not force_elite else None
    if force_elite and elite is None:
        elite = rng.choice(tuple(ELITE_AFFIXES))
    affix = ELITE_AFFIXES.get(elite, {}) if elite else {}

    base_name = roll_enemy_name(rng, rank_idx, myth_foe=myth_foe,
                               archetype=archetype)
    if elite:
        base_name = f"{elite.title()} {base_name}"

    # archetype + elite stat shaping, applied to the foe base
    hp_m = float(arch.get("hp", 1.0)) * float(affix.get("hp", 1.0))
    atk_m = float(arch.get("atk", 1.0)) * float(affix.get("atk", 1.0))
    dfn_m = float(arch.get("def", 1.0)) * float(affix.get("def", 1.0))

    enemy: dict[str, Any] = {
        "name": base_name,
        "rank_idx": rank_idx,
        "difficulty": difficulty,
        "archetype": archetype,
        "elite": elite,
        "elite_blurb": affix.get("blurb", ""),
        "lifesteal": bool(affix.get("lifesteal", False)),
        "dodge_bonus": bool(affix.get("dodge", False)),
        "gear": roll_enemy_gear(rng, rank_idx, myth_foe=myth_foe,
                                difficulty=difficulty),
        "skills": roll_enemy_skills(rng, rank_idx, myth_foe=myth_foe,
                                    difficulty=difficulty),
        "potions": POTIONS[rank_idx] + (1 if difficulty == "expert"
                                        and rank_idx >= 3 else 0),
        "myth_foe": myth_foe and rank_idx >= 5,
        "stat_mults": {"hp": hp_m, "atk": atk_m, "def": dfn_m},
    }
    if player_power > 0:
        from .power import fighter_power
        raw = dict(foe_base) if foe_base else {"max_hp": 50,
                                              "atk": 10, "def": 5}
        base = {k: (int(round(v * mult)) if k in ("max_hp", "atk", "def")
                   else v)
                for k, v in raw.items()}
        # archetype + elite shaping lands on the base stats
        sm = enemy["stat_mults"]
        base["max_hp"] = int(base.get("max_hp", 50) * sm["hp"])
        base["atk"] = int(base.get("atk", 10) * sm["atk"])
        base["def"] = int(base.get("def", 5) * sm["def"])
        enemy["shaped_base"] = dict(base)

        def _with_gear(gear: dict[str, Any]) -> int:
            gatk, gdef = _gear_power(gear, rank_idx)
            stats = dict(base)
            stats["atk"] = int(stats.get("atk", 10)) + gatk
            stats["def"] = int(stats.get("def", 5)) + gdef
            return fighter_power(stats, tuple(enemy["skills"]),
                                 enemy["skills"])

        power = _with_gear(enemy["gear"])
        if power < player_power * 0.85:
            enemy["gear"] = roll_enemy_gear(
                rng, min(7, rank_idx + 1), myth_foe=myth_foe,
                difficulty=difficulty)
        elif power > player_power * 1.25:
            enemy["gear"] = roll_enemy_gear(
                rng, max(0, rank_idx - 1), myth_foe=myth_foe,
                difficulty=difficulty)
    else:
        # no backstop: still expose the shaped base for the caller
        raw = dict(foe_base) if foe_base else {"max_hp": 50,
                                              "atk": 10, "def": 5}
        sm = enemy["stat_mults"]
        enemy["shaped_base"] = {
            "max_hp": int(raw.get("max_hp", 50) * mult * sm["hp"]),
            "atk": int(raw.get("atk", 10) * mult * sm["atk"]),
            "def": int(raw.get("def", 5) * mult * sm["def"]),
        }
    return enemy


def roll_boss(rng: random.Random, rank_idx: int,
              player_power: int = 0,
              foe_base: dict[str, Any] | None = None,
              difficulty: str = "normal") -> dict[str, Any]:
    """Roll a boss: a named, phased, elite-tier threat.

    Bosses are always elite, always at least A-rank presence, and carry
    an enrage phase: below 30% HP they surge (+35% atk, new message).
    The caller checks ``enemy["enrage_at"]`` against current HP and
    flips ``enemy["enraged"]`` once.
    """
    rank_idx = max(4, min(7, int(rank_idx)))
    names = BOSS_NAMES.get(rank_idx, BOSS_NAMES[7])
    enemy = roll_enemy(rng, rank_idx, player_power,
                       foe_base=foe_base, difficulty=difficulty,
                       archetype="hunter", force_elite=True)
    enemy["name"] = str(rng.choice(names))
    enemy["is_boss"] = True
    enemy["enrage_at"] = 0.30
    enemy["enraged"] = False
    enemy["enrage_mult"] = 1.35
    # bosses bring an extra potion and hit harder by nature
    enemy["potions"] = enemy.get("potions", 2) + 1
    return enemy
