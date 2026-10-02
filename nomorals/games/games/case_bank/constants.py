"""Case-bank constants (split from cases.py, Wave H3)."""
from __future__ import annotations

#: difficulty tiers, easiest first
TIERS = ("easy", "medium", "hard", "expert")

#: base score paid for closing a case, by tier
TIER_BASE_SCORE = {"easy": 3, "medium": 5, "hard": 8, "expert": 12}

#: score points one hint costs
HINT_COST = 2

#: timed-mode countdown per tier, in seconds
TIME_LIMITS = {"easy": 300, "medium": 240, "hard": 180, "expert": 150}

#: maximum time bonus (full speed) per tier, in score points
TIME_BONUS_MAX = {"easy": 4, "medium": 6, "hard": 10, "expert": 16}

#: bump when the persisted history schema changes
HISTORY_VERSION = 1

#: how many seen case ids are kept per tier (anti-repeat memory)
_SEEN_CAP = 400
