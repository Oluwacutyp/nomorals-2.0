"""Pass tail-117 pins: nomorals/reviews/review.py — 3 silent handlers.

- 100 E104 ``_run_summarizer``: a crashing summarizer -> "{Name}
  summary unavailable (summarizer failed)." — the failure is NAMED
  in-band (mirrors focus.sessions honesty); healthy/empty/None
  shapes pinned alongside.
- 116 E104 ``_focus_weekly_review`` import guard: focus.sessions
  unavailable -> None, and the caller composes the fallback review
  from the supplied rows only ("no invented data", per the fallback's
  own docstring).
- 641 E104 ``_format_duration``: the focus formatter import crashes
  -> the local formatter, byte-identical to focus.sessions'
  own implementation (control compares both directly).

All exclusions: refusal-shaped or documented fallback rungs.
"""

import sys

from nomorals.focus.sessions import format_duration as real_format_duration
from nomorals.reviews import review as rv


# ── 100: _run_summarizer ─────────────────────────────────────────────

def test_summarizer_not_connected():
    assert rv._run_summarizer("focus", None) == (
        "No data — focus summary not connected.")


def test_summarizer_crash_is_named_not_faked():
    def boom():
        raise RuntimeError("summarizer boom")

    assert rv._run_summarizer("focus", boom) == (
        "Focus summary unavailable (summarizer failed).")


def test_summarizer_value_passes_through():
    assert rv._run_summarizer("focus", lambda: "did things") == "did things"


def test_summarizer_empty_is_honest_no_data():
    assert rv._run_summarizer("focus", lambda: None) == (
        "No data — no focus activity recorded this week.")


def test_summarizer_dict_is_joined():
    assert rv._run_summarizer("focus", lambda: {"a": 1, "b": 2}) == "a: 1; b: 2"


# ── 116: _focus_weekly_review import guard ───────────────────────────

def test_focus_review_import_crash_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.focus.sessions", None)
    assert rv._focus_weekly_review(
        commitments=None, decisions=None, focus=None,
        summarizers=None, now=0.0, week_start=None) is None


# ── 641: _format_duration ────────────────────────────────────────────

def test_format_duration_healthy_matches_focus_impl():
    assert rv._format_duration(7380) == real_format_duration(7380) == "2h 03m"


def test_format_duration_import_crash_uses_local_copy(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.focus.sessions", None)
    assert rv._format_duration(7380) == "2h 03m"
    assert rv._format_duration(900) == "15m"
    assert rv._format_duration(45) == "45s"
