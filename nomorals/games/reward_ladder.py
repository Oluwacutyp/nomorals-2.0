"""Difficulty -> reward ladder, shared by EVERY game.

One table, one rule: the same difficulty pays the same reward whether
the win came from Math Class, trivia, arena, a party game, or a quest.
The table is DATA (:data:`REWARD_TABLE`), never a chain of ``if``
branches — adding a game or retuning a tier means editing the table,
not the code paths that read it.

Rewards grow exponentially because difficulty grows exponentially: the
jump from Master to Impossible is a cliff, not a step, and the payout
reflects that.

Coins here are game currency only: they buy status inside the group,
never cash, never value outside the game. See
:mod:`nomorals.games.math_economy` for the full disclaimer.

Streaks, first clears, and group pools all read from tables in this
module too, so every number a player sees can be traced back to one
place.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "DIFFICULTIES", "REWARD_TABLE", "STREAK_BONUS_TABLE",
    "FIRST_CLEAR_CHEST", "PLACEMENT_SHARES", "FARMING_DECAY",
    "FARMING_DECAY_FLOOR", "normalize_difficulty", "reward_for",
    "reward_for_game", "streak_bonus", "pool_split", "decay_factor",
    "describe_reward",
]

#: Canonical difficulty order, easiest first. Every game uses these
#: exact keys — no per-game synonyms.
DIFFICULTIES: tuple[str, ...] = (
    "easy", "medium", "hard", "expert", "master", "impossible",
)

#: The ladder. ``badge`` / ``title`` are names (or None); ``shoutout``
#: means a pinned shout-out in the group; ``party_theme_vote`` means a
#: party-theme vote is granted with the reward.
REWARD_TABLE: dict[str, dict[str, Any]] = {
    "easy": {
        "coins": 1, "xp": 10,
        "badge": None, "title": None,
        "shoutout": False, "party_theme_vote": False,
    },
    "medium": {
        "coins": 3, "xp": 25,
        "badge": None, "title": None,
        "shoutout": False, "party_theme_vote": False,
    },
    "hard": {
        "coins": 10, "xp": 60,
        "badge": None, "title": None,
        "shoutout": False, "party_theme_vote": False,
    },
    "expert": {
        "coins": 30, "xp": 150,
        "badge": None, "title": None,
        "shoutout": False, "party_theme_vote": False,
    },
    "master": {
        "coins": 100, "xp": 400,
        "badge": "Master of Numbers", "title": None,
        "shoutout": False, "party_theme_vote": False,
    },
    "impossible": {
        "coins": 1000, "xp": 1500,
        "badge": "Impossible Mind",
        "title": "Grandmaster of the Impossible",
        "shoutout": True, "party_theme_vote": True,
    },
}

#: Streak bonus tiers, as data: (min consecutive days, coin/xp
#: multiplier bonus, badge earned at that tier). Highest qualifying
#: tier wins; tiers are NOT stacked. Read top-down, first match wins.
STREAK_BONUS_TABLE: tuple[tuple[int, float, str | None], ...] = (
    (30, 0.50, "Month of Fire"),
    (7, 0.25, "Week Warrior"),
    (3, 0.10, None),
)

#: First-time clear per tier = bonus chest, on top of the base reward.
#: Clearing a tier for the first time is a one-off achievement, so the
#: chest pays once per player per tier, ever.
FIRST_CLEAR_CHEST: dict[str, dict[str, int]] = {
    "easy": {"coins": 5, "xp": 20},
    "medium": {"coins": 15, "xp": 50},
    "hard": {"coins": 50, "xp": 120},
    "expert": {"coins": 150, "xp": 300},
    "master": {"coins": 500, "xp": 800},
    "impossible": {"coins": 5000, "xp": 3000},
}

#: Group-tournament pool split by placement, as data. Shares for 1st..5th;
#: anything beyond 5th splits the tail evenly. Shares sum to 1.0.
PLACEMENT_SHARES: tuple[float, ...] = (0.40, 0.25, 0.15, 0.12, 0.08)

#: Repeat-farming decay: each repeat of the same rewardable action in
#: the same period pays ``FARMING_DECAY ** repeats`` of base, floored.
FARMING_DECAY = 0.75
FARMING_DECAY_FLOOR = 0.10


def normalize_difficulty(difficulty: str) -> str:
    """Lowercase/strip a difficulty key; unknown keys fall back to easy."""
    key = (difficulty or "").strip().lower()
    return key if key in REWARD_TABLE else "easy"


def streak_bonus(streak_days: int) -> tuple[float, str | None]:
    """(bonus multiplier, badge) for a streak length, from the table.

    Data-driven: first table row whose threshold the streak meets wins.
    """
    days = max(0, int(streak_days or 0))
    for threshold, bonus, badge in STREAK_BONUS_TABLE:
        if days >= threshold:
            return bonus, badge
    return 0.0, None


def decay_factor(repeat_count: int) -> float:
    """Farming decay for ``repeat_count`` repeats of the same action."""
    repeats = max(0, int(repeat_count or 0))
    return max(FARMING_DECAY_FLOOR, FARMING_DECAY ** repeats)


def reward_for(
    difficulty: str,
    streak_days: int = 0,
    first_clear: bool = False,
    repeat_count: int = 0,
) -> dict[str, Any]:
    """Compute a reward from the table.

    ``difficulty`` selects the row; ``streak_days`` picks the streak
    multiplier from :data:`STREAK_BONUS_TABLE`; ``first_clear`` adds
    the tier's bonus chest; ``repeat_count`` applies farming decay to
    the coin/xp portion (badges/titles/shout-outs never decay — they
    are earned once or not at all).

    Returns a dict with ``coins``, ``xp``, ``badge``, ``title``,
    ``shoutout``, ``party_theme_vote``, and a ``breakdown`` string
    showing exactly how the numbers were derived.
    """
    key = normalize_difficulty(difficulty)
    row = REWARD_TABLE[key]
    bonus, streak_badge = streak_bonus(streak_days)
    decay = decay_factor(repeat_count)

    coins = int(round(row["coins"] * (1.0 + bonus) * decay))
    xp = int(round(row["xp"] * (1.0 + bonus) * decay))
    chest_coins = chest_xp = 0
    if first_clear:
        chest = FIRST_CLEAR_CHEST[key]
        chest_coins, chest_xp = int(chest["coins"]), int(chest["xp"])
        coins += chest_coins
        xp += chest_xp

    # The streak badge and the tier badge are both earned; report both.
    badges = [b for b in (row["badge"], streak_badge) if b]
    parts = [f"base {row['coins']}c + {row['xp']} XP ({key})"]
    if bonus:
        parts.append(f"streak x{1.0 + bonus:g} ({streak_days}d)")
    if repeat_count:
        parts.append(f"repeat decay x{decay:g}")
    if first_clear:
        parts.append(f"first-clear chest +{chest_coins}c +{chest_xp} XP")
    return {
        "difficulty": key,
        "coins": coins,
        "xp": xp,
        "badge": row["badge"],
        "badges": badges,
        "title": row["title"],
        "shoutout": bool(row["shoutout"]),
        "party_theme_vote": bool(row["party_theme_vote"]),
        "streak_days": int(streak_days),
        "streak_bonus": bonus,
        "first_clear": bool(first_clear),
        "breakdown": " · ".join(parts),
    }


def reward_for_game(game: str, difficulty: str, **kwargs: Any) -> dict[str, Any]:
    """Same ladder for every game: ``game`` is recorded, never priced.

    Math, trivia, arena, party, quests — all read the same table, so
    no game can quietly pay more than another at the same difficulty.
    """
    out = reward_for(difficulty, **kwargs)
    out["game"] = (game or "math").strip().lower() or "math"
    return out


def pool_split(pool: int, placements: int) -> list[int]:
    """Split a group-tournament pool by placement.

    First five places take :data:`PLACEMENT_SHARES`; any remaining
    places split whatever is left evenly. The split always sums to
    exactly ``pool`` — the rounding remainder goes to 1st place.
    Returns a list of coin amounts, index 0 = 1st place.
    """
    pool = max(0, int(pool))
    n = max(1, int(placements))
    if pool == 0:
        return [0] * n
    shares: list[float] = list(PLACEMENT_SHARES[:n])
    if n > len(PLACEMENT_SHARES):
        tail = 1.0 - sum(PLACEMENT_SHARES)
        extra = n - len(PLACEMENT_SHARES)
        shares.extend([tail / extra] * extra)
    elif n < len(PLACEMENT_SHARES):
        # Fewer places than the table: renormalise the used shares.
        total = sum(shares)
        shares = [s / total for s in shares]
    amounts = [int(pool * s) for s in shares]
    amounts[0] += pool - sum(amounts)  # rounding remainder to the winner
    return amounts


def describe_reward(reward: dict[str, Any]) -> str:
    """One-line human rendering of a computed reward."""
    bits = [f"+{reward['coins']} coins", f"+{reward['xp']} XP"]
    if reward.get("badge"):
        bits.append(f"🏅 {reward['badge']}")
    if reward.get("title"):
        bits.append(f"👑 {reward['title']}")
    if reward.get("shoutout"):
        bits.append("📣 pinned shout-out")
    if reward.get("party_theme_vote"):
        bits.append("🎉 party-theme vote")
    return ", ".join(bits)
