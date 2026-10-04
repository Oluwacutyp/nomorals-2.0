"""Visible power ratings for arena fighters.

Power is one number that answers "how strong is this fighter?" so the
player can gauge a fight at a glance.  The formula is deliberately
transparent — it's shown in the arena intro, not hidden math::

    power = max_hp + atk * 10 + def * 10 + skill_power

``atk``/``def`` already include equipped gear and passive skills (the
arena folds those into the fighter before calling here), so gear and
passives count through the stats.  Learned *active* skills add their
own term: a fighter who can cast is more dangerous than one who can't.
"""

from __future__ import annotations

from typing import Any

ATK_WEIGHT = 10
DEF_WEIGHT = 10


def skill_power_score(defn: Any, tier: int = 1) -> int:
    """Power contribution of one learned active skill at ``tier``.

    Roughly "how much extra damage/utility a fight gains": strikes scale
    with their damage shape, heals with the heal fraction, buffs with
    the buff, dodges are flat value, defense-ignore adds a kicker.
    Higher tiers pay a 25%-per-tier premium.
    """
    score = 0.0
    mult = float(getattr(defn, "mult", 0.0) or 0.0)
    hits = int(getattr(defn, "hits", 1) or 1)
    if mult:
        score += mult * hits * 20.0
    heal = float(getattr(defn, "heal_pct", 0.0) or 0.0)
    if heal:
        score += heal * 80.0
    buff = int(getattr(defn, "atk_buff", 0) or 0)
    turns = int(getattr(defn, "buff_turns", 0) or 0)
    if buff:
        score += buff * max(turns, 1) * 6.0
    if getattr(defn, "dodge", False):
        score += 45.0
        score += float(getattr(defn, "counter_mult", 0.0) or 0.0) * 30.0
    ignore = float(getattr(defn, "ignore_def_pct", 0.0) or 0.0)
    if ignore:
        score += ignore * 50.0
    score *= 1.0 + 0.25 * (max(1, int(tier)) - 1)
    return int(round(score))


def skills_power(slugs: list[str] | tuple[str, ...],
                 tiers: dict[str, int] | None = None) -> int:
    """Total power of a fighter's learned active skills."""
    try:
        from .skills import SKILL_CATALOG, effective_def
    except Exception:  # noqa: BLE001
        return 0
    tiers = tiers or {}
    total = 0
    for slug in slugs or ():
        defn = SKILL_CATALOG.get(slug)
        if defn is None or getattr(defn, "kind", "") != "active":
            continue
        # effective_def already folds the tier's knobs into the
        # blueprint, so score the folded def at tier 1 (no double
        # counting of the tier premium).
        total += skill_power_score(
            effective_def(defn, int(tiers.get(slug, 1))))
    return total


def fighter_power(fighter: dict[str, Any],
                  slugs: list[str] | tuple[str, ...] = (),
                  tiers: dict[str, int] | None = None) -> int:
    """One-number strength of a battle-ready fighter dict.

    The fighter's ``atk``/``def``/``max_hp`` are expected to already
    include gear and passive bonuses (the arena applies those first).
    """
    base = (int(fighter.get("max_hp", 50))
            + int(fighter.get("atk", 10)) * ATK_WEIGHT
            + int(fighter.get("def", 5)) * DEF_WEIGHT)
    return base + skills_power(slugs, tiers)


def power_bar(power: int, foe_power: int, width: int = 10) -> str:
    """Tiny visual gauge of your power relative to the foe's."""
    if foe_power <= 0:
        return "▰" * width
    ratio = max(0.0, min(1.0, power / (foe_power * 1.5)))
    filled = int(round(ratio * width))
    return "▰" * filled + "▱" * (width - filled)
