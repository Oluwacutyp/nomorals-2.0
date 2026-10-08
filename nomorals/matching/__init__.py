"""Curated daily batches + mutual-preference matching (build-map #83).

Scarcity-as-ritual + two-sided matching, portable to every surface
that matches people or picks:

- #1 gig marketplace — daily gig picks, freelancer ↔ client matching
- #12 community — daily member highlights, member ↔ member matching
- #46 mentoring — daily learning picks, mentor ↔ mentee matching

Three small modules, one honest rule set:

1. **batches** — a small daily set of high-quality picks instead of
   infinite scroll. The day's batch is stable (same picks all day,
   new picks tomorrow). Scarcity creates ritual.
2. **stable** — Gale-Shapley two-sided stable matching. "One best
   pairing per day" beats infinite browsing for quality perception.
3. **questionnaire** — importance-weighted preference capture
   ("how important is remote work? 1–5") with an answer-more →
   better-results loop, plus front-loaded deal-breakers
   (budget, availability, intent) filtered BEFORE matching —
   the app absorbs the awkward conversation.

Everything is pure algorithms + SQLite storage, never raises, and
fully offline-testable.
"""

from __future__ import annotations

from .batches import Candidate, DailyBatchStore, curate_daily
from .questionnaire import DealBreaker, Question, Questionnaire
from .stable import MatchResult, stable_match, verify_stable

__all__ = [
    "Candidate",
    "DailyBatchStore",
    "curate_daily",
    "DealBreaker",
    "Question",
    "Questionnaire",
    "MatchResult",
    "stable_match",
    "verify_stable",
]
