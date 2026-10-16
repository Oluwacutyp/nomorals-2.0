"""Gig-marketplace surface adapter for matching (build-map #83, surface #1).

The matching package is portable to every surface that matches people
or picks; this adapter binds it to the gig marketplace: daily gig
picks, freelancer ↔ client two-sided matching, and the gig
questionnaire. Every function delegates to :mod:`nomorals.matching`
— nothing is re-implemented here.
"""

from __future__ import annotations

from typing import Any

from .batches import Candidate, DailyBatchStore, curate_daily, explain_pick
from .questionnaire import Questionnaire
from .stable import MatchResult, stable_match, verify_stable

__all__ = [
    "SURFACE",
    "daily_picks",
    "todays_batch",
    "explain",
    "stable_match_for",
    "verify",
    "new_questionnaire",
    "filter_dealbreakers",
]

#: Documented surface name (matching/__init__.py: "#1 gig marketplace").
SURFACE = "gig"


def daily_picks(candidates: list[Candidate],
                preferences: dict[str, float] | None = None,
                n: int = 5, **kwargs: Any) -> list[Candidate]:
    """Curated daily gig picks — :func:`curate_daily` with surface ``gig``."""
    return curate_daily(candidates, preferences, n, **kwargs)


def todays_batch(store: DailyBatchStore, owner: str = "owner",
                 n: int = 5) -> list[Candidate]:
    """The stored stable-all-day gig batch (curates + stores on first call)."""
    return store.today(SURFACE, owner, n=n)


def explain(candidate: Candidate,
            preferences: dict[str, float] | None = None) -> str:
    """Why this gig candidate was picked — :func:`explain_pick`."""
    return explain_pick(candidate, preferences)


def stable_match_for(proposers: list[str], reviewers: list[str],
                     proposer_prefs: dict[str, list[str]],
                     reviewer_prefs: dict[str, list[str]]) -> MatchResult:
    """Gale–Shapley freelancer ↔ client matching — :func:`stable_match`."""
    return stable_match(proposers, reviewers, proposer_prefs, reviewer_prefs)


def verify(result: MatchResult, proposer_prefs: dict[str, list[str]],
           reviewer_prefs: dict[str, list[str]]) -> tuple[bool, list]:
    """Blocking-pair check — :func:`verify_stable`."""
    return verify_stable(result, proposer_prefs, reviewer_prefs)


def new_questionnaire(owner: str = "owner", db_path: str = "") -> Questionnaire:
    """Gig questionnaire (remote/budget/speed/experience starters)."""
    q = Questionnaire(owner=owner, db_path=db_path)
    q.ensure_starters(SURFACE)
    return q


def filter_dealbreakers(questionnaire: Questionnaire,
                        candidates: list[dict]) -> list[dict]:
    """Front-loaded deal-breakers (budget, availability, intent) — before matching."""
    return questionnaire.filter_dealbreakers(candidates)
