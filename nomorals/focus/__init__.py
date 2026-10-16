"""Focus sessions + weekly review — the week layer.

See :mod:`nomorals.focus.sessions` for the design rules: sessions are
a ledger (pauses subtracted, nothing estimated), the review aggregates
and never invents, next week is derived from this week.
"""

from __future__ import annotations

from .sessions import (
    FocusSession,
    FocusTimer,
    WeeklyReview,
    build_weekly_review,
    format_duration,
    week_bounds,
)

__all__ = [
    "FocusSession",
    "FocusTimer",
    "WeeklyReview",
    "build_weekly_review",
    "format_duration",
    "week_bounds",
]
