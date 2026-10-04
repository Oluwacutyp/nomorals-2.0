"""Persistent player progression: XP, levels, stat growth.

One shared level across every game (arena, RPG, all 40+ others).  The
curve is deliberately rewarding, not grindy: the first levels pop after
a battle or two, and every level visibly moves arena stats.

XP curve
--------
``xp_for_level(n)`` is the *cumulative* XP needed to reach level ``n``::

    xp_for_level(n) = 40 * n * (n - 1)

So level 2 needs 80 XP, level 3 needs 240, level 5 needs 800, level 10
needs 3,600.  An arena win pays 60 XP — level 2 lands after about two
battles; later levels take a session each, never a week.

Stat growth (arena)
-------------------
Each level above 1 grants: +4 max HP, +1 attack, +1 defense every two
levels.  The house scales at half the player's bonus so fights stay
competitive while progression always feels powerful.

Storage
-------
XP lives on the player profile (``game_players.xp``, migration 70).
Level is *derived* from XP — single source of truth, no drift.
"""
from __future__ import annotations

import time
from typing import Any

# ── the curve ────────────────────────────────────────────────────────────────

#: XP for one arena win / loss. Wins pay more than double — winning
#: should always feel like the fastest way up.
ARENA_WIN_XP = 60
ARENA_LOSS_XP = 25

#: generic games (everything that isn't the arena): small but nonzero,
#: so every table moves the bar.
GAME_WIN_XP = 20
GAME_DRAW_XP = 12
GAME_LOSS_XP = 8


def xp_for_level(level: int) -> int:
    """Cumulative XP required to *reach* ``level`` (level 1 = 0)."""
    level = max(1, int(level))
    return 40 * level * (level - 1)


def level_for_xp(xp: int) -> int:
    """Invert the curve: what level does this XP total earn?"""
    xp = max(0, int(xp))
    level = 1
    # quadratic inversion would be cute; the loop is clearer and
    # levels stay small (a million XP is still only level ~160).
    while xp_for_level(level + 1) <= xp:
        level += 1
    return level


def xp_progress(xp: int) -> tuple[int, int, int]:
    """(level, xp into current level, xp needed for next level)."""
    level = level_for_xp(xp)
    base = xp_for_level(level)
    span = xp_for_level(level + 1) - base
    return level, max(0, int(xp) - base), span


def xp_bar(xp: int, width: int = 10) -> str:
    """``██████░░░░ 145/240`` style progress bar to the next level."""
    level, into, span = xp_progress(xp)
    filled = int(round(width * into / span)) if span else width
    return ("█" * filled + "░" * (width - filled)
            + f" {into}/{span} (lvl {level})")


# ── stat growth ──────────────────────────────────────────────────────────────

def level_stat_bonus(level: int) -> dict[str, int]:
    """Arena stat bonus for a player level. Level 1 = no bonus."""
    steps = max(0, int(level) - 1)
    return {
        "max_hp": 4 * steps,
        "atk": 1 * steps,
        "def": steps // 2,
    }


def describe_level_up(new_level: int) -> str:
    """The level-up fanfare with the concrete gains."""
    bonus = level_stat_bonus(new_level)
    bits = [f"+{bonus['max_hp']} max HP", f"+{bonus['atk']} atk"]
    if bonus["def"]:
        bits.append(f"+{bonus['def']} def")
    return (f"⭐ LEVEL UP — you're level {new_level}! "
            f"({', '.join(bits)} in the arena)")


# ── awarding ─────────────────────────────────────────────────────────────────

def award_xp(store: Any, player: Any, amount: int,
             reason: str = "") -> tuple[int, list[int]]:
    """Add XP to a player's profile. Returns (new_level, levels_gained).

    ``levels_gained`` holds every level crossed (usually 0-1 entries;
    several on a big award).  Never raises — progression must never
    break a game finish.

    The increment is a single atomic UPDATE (``PlayerStore.add_xp``),
    so two games finishing at once can't lose one award — the old
    read-modify-write-upsert could.
    """
    amount = max(0, int(amount))
    try:
        prof = store.get(player.key, name=player.name,
                         platform=player.platform)
    except Exception:  # noqa: BLE001
        return 1, []
    old_level = level_for_xp(prof.xp)
    add_xp = getattr(store, "add_xp", None)
    try:
        if callable(add_xp):
            new_xp = add_xp(player.key, amount)
        else:  # pragma: no cover — legacy stores without add_xp
            prof.xp = max(0, int(prof.xp) + amount)
            prof.updated_at = time.time()
            store._upsert(prof)
            new_xp = prof.xp
    except Exception:  # noqa: BLE001
        return old_level, []
    new_level = level_for_xp(new_xp)
    gained = list(range(old_level + 1, new_level + 1))
    return new_level, gained


def generic_game_xp(won: bool | None) -> int:
    """Default XP for any non-arena game outcome."""
    if won is True:
        return GAME_WIN_XP
    if won is False:
        return GAME_LOSS_XP
    return GAME_DRAW_XP
