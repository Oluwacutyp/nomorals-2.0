"""Mentoring / learning surface adapter for matching (build-map #83, surface #46).

Binds matching to mentoring: daily learning picks, mentor ↔ mentee
matching — including capacity-aware matching (one mentor takes N
mentees, Hospital–Resident) and max-total-score assignment for
learning slots. Every function delegates to :mod:`nomorals.matching`.
"""

from __future__ import annotations

from typing import Any

from .batches import Candidate, DailyBatchStore, curate_daily, explain_pick
from .questionnaire import Questionnaire
from .stable import (
    CapacityMatchResult,
    optimal_assignment,
    stable_match_capacities,
    top_trading_cycles,
)

__all__ = [
    "SURFACE",
    "daily_picks",
    "todays_batch",
    "explain",
    "capacity_match",
    "assign",
    "trade_cycles",
    "new_questionnaire",
    "filter_dealbreakers",
]

#: Documented surface name (matching/__init__.py: "#46 mentoring";
#: the chat surface flag is ``learning``).
SURFACE = "learning"


def daily_picks(candidates: list[Candidate],
                preferences: dict[str, float] | None = None,
                n: int = 5, **kwargs: Any) -> list[Candidate]:
    """Curated daily learning picks — :func:`curate_daily` (learning)."""
    return curate_daily(candidates, preferences, n, **kwargs)


def todays_batch(store: DailyBatchStore, owner: str = "owner",
                 n: int = 5) -> list[Candidate]:
    """The stored stable-all-day learning batch."""
    return store.today(SURFACE, owner, n=n)


def explain(candidate: Candidate,
            preferences: dict[str, float] | None = None) -> str:
    """Why this learning pick was picked — :func:`explain_pick`."""
    return explain_pick(candidate, preferences)


def capacity_match(proposers: list[str], reviewers: list[str],
                   proposer_prefs: dict[str, list[str]],
                   reviewer_prefs: dict[str, list[str]],
                   capacities: dict[str, int] | None = None
                   ) -> CapacityMatchResult:
    """Mentor ↔ mentees, one mentor taking N — Hospital–Resident."""
    return stable_match_capacities(proposers, reviewers, proposer_prefs,
                                   reviewer_prefs, capacities)


def assign(scores: dict[str, dict[str, float]]) -> dict[str, str | None]:
    """Max-total-score assignment of learning slots — Hungarian."""
    return optimal_assignment(scores)


def trade_cycles(agents: list[str], prefs: dict[str, list[str]],
                 endowments: dict[str, str]) -> dict[str, str | None]:
    """Top Trading Cycles — allocate indivisible goods (who gets which slot)."""
    return top_trading_cycles(agents, prefs, endowments)


def new_questionnaire(owner: str = "owner", db_path: str = "") -> Questionnaire:
    """Learning questionnaire (depth/practical/time starters)."""
    q = Questionnaire(owner=owner, db_path=db_path)
    q.ensure_starters(SURFACE)
    return q


def filter_dealbreakers(questionnaire: Questionnaire,
                        candidates: list[dict]) -> list[dict]:
    """Front-loaded deal-breakers — before matching."""
    return questionnaire.filter_dealbreakers(candidates)
