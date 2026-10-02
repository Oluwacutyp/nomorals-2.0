"""The case bank for the ``case`` game (InvestigationGame).

Fifty-two full cases across four tiers (easy/medium/hard/expert), each
with airtight internal logic:

* ``id``          — "bank:N", stable across runs.
* ``title``       — the case's name.
* ``tier``        — easy | medium | hard | expert.
* ``briefing``    — the setup, two to three sentences.
* ``story``       — the crime, one or two lines (shown at case open).
* ``suspects``    — four named suspects.
* ``motives``     — one motive per suspect (everyone has a reason).
* ``culprit``     — one of the suspects.
* ``clues``       — five, sharpest last, revealed in order.  The pattern
  every case follows:
  clue 1  = the mechanism (how it was done)
  clue 2  = an alibi that clears one innocent
  clue 3  = an alibi that clears a second innocent
  clue 4  = a red herring (looks suspicious, is innocent)
  clue 5  = the smoking gun that names the culprit
* ``red_herrings``— the misleading trails, spelled out (each names an
  innocent — never the culprit).
* ``herring_suspect`` — the suspect the red herring points at.
* ``solution``    — the fair-play deduction walkthrough: how the clues
  identify the culprit.
* ``statements``  — one line per suspect, what they say when asked.
  The culprit's statement contradicts clue 5 (the smoking gun); the
  innocents' statements are consistent with their alibis.  Interviewing
  (``ask <name>``) is how you find the contradiction before you accuse.
* ``difficulty``  — back-compat grade (expert maps to "hard").

``verify_solvability`` runs the fair-play check: the true culprit must
be deducible from the clues alone.  ``deal_case`` serves cases with
per-player anti-repeat tracking, and ``adapt_case`` tunes clue counts
to the player's solve rate.

Wave H3: the bank implementation moved to
:mod:`nomorals.games.games.case_bank`; this module is a
compatibility facade re-exporting the same public names.
"""
from __future__ import annotations

from .case_bank import (
    BANK_SIZE,
    CASES,
    HINT_COST,
    HISTORY_VERSION,
    TIER_BASE_SCORE,
    TIERS,
    TIME_BONUS_MAX,
    TIME_LIMITS,
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
