"""Case bank for the ``case`` game (InvestigationGame) — Wave H3 split.

The bank used to live in a single 4,069-line ``cases.py``.  It is now
organized as:

* :mod:`constants`  — tiers, scoring, time limits, history version
* :mod:`raw_cases`  — the 30 original hand-written cases (bank:0–29)
* :mod:`bank_meta`  — god-tier metadata for the 30 originals
* :mod:`new_cases`  — newer full-schema cases (bank:30 …)
* :mod:`logic`      — enrichment (``CASES``) + deal/score/verify services

``nomorals.games.games.cases`` remains as a compatibility façade that
re-exports everything below, so existing imports keep working.
"""
from __future__ import annotations

from .constants import (
    HINT_COST,
    HISTORY_VERSION,
    TIER_BASE_SCORE,
    TIERS,
    TIME_BONUS_MAX,
    TIME_LIMITS,
    _SEEN_CAP,
)
from .logic import (
    BANK_SIZE,
    CASES,
    adapt_case,
    blank_history,
    deal_case,
    eligible_tiers,
    generate_case,
    random_case,
    record_played,
    score_solve,
    solve_rate,
    unseen_bank_cases,
    verify_solvability,
)

__all__ = [
    "CASES", "BANK_SIZE", "TIERS", "HINT_COST", "TIME_LIMITS",
    "TIME_BONUS_MAX", "HISTORY_VERSION",
    "generate_case", "random_case", "verify_solvability",
    "blank_history", "eligible_tiers", "solve_rate", "deal_case",
    "record_played", "adapt_case", "score_solve", "unseen_bank_cases",
]
