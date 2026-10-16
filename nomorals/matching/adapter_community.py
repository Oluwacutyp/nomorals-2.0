"""Community surface adapter for matching (build-map #83, surface #12).

Binds matching to the community: daily member highlights, member ↔
member matching, and the community questionnaire. Single-pool pairing
uses Irving's algorithm — some instances admit NO stable pairing and
the result says so honestly instead of fabricating one. Every
function delegates to :mod:`nomorals.matching`.
"""

from __future__ import annotations

from typing import Any

from .batches import Candidate, DailyBatchStore, curate_daily, explain_pick
from .questionnaire import Questionnaire
from .stable import RoommateResult, roommate_match, verify_roommate_stable

__all__ = [
    "SURFACE",
    "daily_picks",
    "todays_batch",
    "explain",
    "pair_members",
    "verify_pairing",
    "new_questionnaire",
    "filter_dealbreakers",
]

#: Documented surface name (matching/__init__.py: "#12 community").
SURFACE = "community"


def daily_picks(candidates: list[Candidate],
                preferences: dict[str, float] | None = None,
                n: int = 5, **kwargs: Any) -> list[Candidate]:
    """Curated daily member highlights — :func:`curate_daily` (community)."""
    return curate_daily(candidates, preferences, n, **kwargs)


def todays_batch(store: DailyBatchStore, owner: str = "owner",
                 n: int = 5) -> list[Candidate]:
    """The stored stable-all-day community batch."""
    return store.today(SURFACE, owner, n=n)


def explain(candidate: Candidate,
            preferences: dict[str, float] | None = None) -> str:
    """Why this member highlight was picked — :func:`explain_pick`."""
    return explain_pick(candidate, preferences)


def pair_members(people: list[str], prefs: dict[str, list[str]],
                 *, allow_odd: bool = True) -> RoommateResult:
    """Single-pool member ↔ member pairing — Irving (1985)."""
    return roommate_match(people, prefs, allow_odd=allow_odd)


def verify_pairing(pairs: dict[str, str | None],
                   prefs: dict[str, list[str]]) -> tuple[bool, list]:
    """Blocking-pair check — :func:`verify_roommate_stable`."""
    return verify_roommate_stable(pairs, prefs)


def new_questionnaire(owner: str = "owner", db_path: str = "") -> Questionnaire:
    """Community questionnaire (shared-interest/availability starters)."""
    q = Questionnaire(owner=owner, db_path=db_path)
    q.ensure_starters(SURFACE)
    return q


def filter_dealbreakers(questionnaire: Questionnaire,
                        candidates: list[dict]) -> list[dict]:
    """Front-loaded deal-breakers — before matching."""
    return questionnaire.filter_dealbreakers(candidates)
