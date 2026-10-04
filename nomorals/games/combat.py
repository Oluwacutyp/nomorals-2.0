"""Shared combat math for the arena family (arena, duel, raid).

Fighters are plain dicts with the battle-arena shape::

    {"hp", "max_hp", "atk", "def", "potions", "defending", "shield",
     "focused", "fury_cd", "dodge_next", "warcry_turns", "warcry_amt", ...}

Everything here is pure (no I/O, no store): games own their messages,
gear wear, and state layout; this module owns the numbers so the three
combat games can't drift apart.
"""
from __future__ import annotations

import random
from typing import Any

__all__ = ["new_fighter", "strike", "tick_fighter"]


def new_fighter(hp: int = 50, atk: int = 10, dfn: int = 5) -> dict[str, Any]:
    """A fresh fighter dict with the battle-arena shape."""
    return {"hp": hp, "max_hp": hp, "atk": atk, "def": dfn,
            "potions": 1, "defending": False, "shield": False,
            "focused": False, "fury_cd": 0, "dodge_next": False,
            "warcry_turns": 0, "warcry_amt": 0,
            "combo_every": 0, "combo_count": 0, "combo_name": ""}


def strike(attacker: dict[str, Any], defender: dict[str, Any],
           rng: random.Random, *, mult: float = 1.0,
           ignore_def: float = 0.0,
           crit_bonus: float = 0.0) -> dict[str, Any]:
    """Resolve one hit. Mutates the fighter dicts, returns a report::

        {"dmg", "crit", "dodged", "focused", "shielded"}

    Semantics match the battle arena exactly: shadow-step dodges before
    anything else, focus is spent on the swing, crits double, guard
    halves, and a shield converts a killing blow to 1 HP.
    """
    report = {"dmg": 0, "crit": False, "dodged": False,
              "focused": False, "shielded": False}
    # shadow step: the fighter simply isn't there
    if defender.get("dodge_next"):
        defender["dodge_next"] = False
        report["dodged"] = True
        return report
    # a focused fighter spends its focus on this hit
    focused = bool(attacker.get("focused", False))
    attacker["focused"] = False
    report["focused"] = focused
    eff_def = int(defender["def"] * (1.0 - ignore_def))
    raw = max(1, attacker["atk"] - eff_def // 2 + rng.randint(-2, 3))
    if mult != 1.0:
        raw = max(1, int(raw * mult))
    if focused:
        raw = max(1, int(raw * 1.5))
    crit = rng.random() < (0.10 + float(crit_bonus or 0.0))
    if crit:
        raw *= 2
    if defender.get("defending"):
        raw = max(1, raw // 2)
        defender["defending"] = False
    defender["hp"] -= raw
    report["dmg"] = raw
    report["crit"] = crit
    if defender["hp"] <= 0 and defender.get("shield"):
        defender["shield"] = False
        defender["hp"] = 1
        report["shielded"] = True
    return report


def tick_fighter(fighter: dict[str, Any],
                 skill_cd: dict[str, int] | None = None) -> list[str]:
    """Start-of-turn decay: fury cooldown, skill cooldowns, war-cry
    expiry. Returns narrative notes (worn-off buffs)."""
    notes: list[str] = []
    if int(fighter.get("fury_cd", 0)) > 0:
        fighter["fury_cd"] = int(fighter["fury_cd"]) - 1
    if skill_cd:
        for slug in list(skill_cd):
            if int(skill_cd[slug]) > 0:
                skill_cd[slug] = int(skill_cd[slug]) - 1
    if int(fighter.get("warcry_turns", 0)) > 0:
        fighter["warcry_turns"] = int(fighter["warcry_turns"]) - 1
        if int(fighter["warcry_turns"]) <= 0:
            # fade the buff that was actually applied (3 for base War Cry;
            # older saves / hand-built fighters carry no amount)
            amt = int(fighter.pop("warcry_amt", 0) or 3)
            fighter["atk"] = max(1, int(fighter["atk"]) - amt)
            notes.append("the war cry fades — your attack settles.")
    return notes
