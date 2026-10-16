"""Tests for nomorals.reviews — periodic synthesis on top of focus."""

import datetime as _dt

import pytest

from nomorals.commitments.ledger import CommitmentLedger
from nomorals.decisions.journal import DecisionJournal
from nomorals.focus.sessions import FocusTimer
from nomorals.reviews import (
    MonthlyReview,
    WeeklyReview,
    build_monthly_review,
    build_weekly_review,
    render_markdown,
)


def _ts(year, month, day, hour=12):
    return _dt.datetime(year, month, day, hour, tzinfo=_dt.timezone.utc).timestamp()


NOW = _ts(2026, 10, 14)  # Wednesday of the week starting Mon 2026-10-12
MON = _ts(2026, 10, 12, 0)


@pytest.fixture()
def fixtures():
    ledger = CommitmentLedger()
    overdue = ledger.add(promisee="Chidi", what="send the deck",
                         deadline=NOW - 86400, ts=MON + 3600)
    future = ledger.add(promisee="Ada", what="review the draft",
                        deadline=NOW + 5 * 86400, ts=MON + 3600)
    kept = ledger.add(promisee="Ada", what="send notes",
                      deadline=NOW + 86400, ts=MON + 3600)
    ledger.mark_kept(kept.id, "sent", ts=MON + 7200)
    broken = ledger.add(promisee="Ada", what="call back",
                        deadline=NOW + 86400, ts=MON + 3600)
    ledger.mark_broken(broken.id, "missed", ts=MON + 7200)

    journal = DecisionJournal()
    open_decision = journal.record(decision="Ship the bot", chosen="",
                                   ts=MON + 3600)
    resolved = journal.record(decision="Paper-trade a month",
                              chosen="Paper-trade a month",
                              confidence=0.7, ts=MON + 3600)
    journal.resolve(resolved.id, outcome="Found two bugs.", success=True,
                    ts=MON + 7200)

    timer = FocusTimer()
    session = timer.start("codebeast training", ts=MON + 3600)
    timer.interrupt(session.id, note="phone call")
    timer.stop(session.id, note="smoke test green", ts=MON + 7200)
    return {"ledger": ledger, "journal": journal, "timer": timer,
            "overdue": overdue, "future": future,
            "open_decision": open_decision}


def _summarizers():
    def _boom():
        raise RuntimeError("learn down")

    return {"health": None, "learn": _boom,
            "finance": lambda: "spent $12 on coffee"}


def test_weekly_composes_focus_sections(fixtures):
    review = build_weekly_review(
        commitments=fixtures["ledger"], decisions=fixtures["journal"],
        focus=fixtures["timer"], summarizers=_summarizers(), now=NOW)
    assert isinstance(review, WeeklyReview)
    # Focus totals come from the focus builder, not recomputed here.
    assert review.focus_totals["sessions"] == 1
    assert review.focus_totals["total_seconds"] == pytest.approx(3600.0)
    assert review.focus_totals["by_label"][0]["label"] == "codebeast training"
    assert review.focus_totals["by_label"][0]["interruptions"] == 1
    # Happened carries focus wins + summarizer lines.
    assert any("Kept promise" in line for line in review.happened)
    assert any("Decision paid off" in line for line in review.happened)
    assert any("spent $12" in line for line in review.happened)
    # Reliability: 1 kept / (1 kept + 1 broken) = 0.5, never invented.
    assert review.kept == 1 and review.broken == 1
    assert review.reliability == pytest.approx(0.5)


def test_weekly_slipped_only_overdue(fixtures):
    review = build_weekly_review(
        commitments=fixtures["ledger"], decisions=fixtures["journal"],
        focus=fixtures["timer"], summarizers=_summarizers(), now=NOW)
    slipped_whats = [row["what"] for row in review.slipped]
    assert slipped_whats == ["send the deck"]
    # The future commitment is open but not slipped.
    assert "review the draft" not in slipped_whats
    # Next week carries the slipped item as a carry-over.
    assert any("send the deck" in item for item in review.next_week)


def test_weekly_blocked_open_decisions_and_interruptions(fixtures):
    review = build_weekly_review(
        commitments=fixtures["ledger"], decisions=fixtures["journal"],
        focus=fixtures["timer"], summarizers=_summarizers(), now=NOW)
    kinds = [entry["kind"] for entry in review.blocked]
    assert "decision" in kinds and "focus" in kinds
    decisions = [e for e in review.blocked if e["kind"] == "decision"]
    assert [d["decision"] for d in decisions] == ["Ship the bot"]
    focus_blocks = [e for e in review.blocked if e["kind"] == "focus"]
    assert focus_blocks[0]["interruptions"] == 1
    assert focus_blocks[0]["label"] == "codebeast training"


def test_weekly_missing_and_failing_summarizer_honest(fixtures):
    review = build_weekly_review(
        commitments=fixtures["ledger"], decisions=fixtures["journal"],
        focus=fixtures["timer"], summarizers=_summarizers(), now=NOW)
    assert "not connected" in review.summaries["health"]
    assert "unavailable" in review.summaries["learn"]
    # No summarizers at all -> all three honest, nothing invented.
    empty = build_weekly_review(now=NOW)
    assert all("not connected" in empty.summaries[name]
               for name in ("health", "learn", "finance"))
    assert empty.slipped == [] and empty.blocked == []
    assert empty.reliability is None


def test_render_markdown_sections(fixtures):
    review = build_weekly_review(
        commitments=fixtures["ledger"], decisions=fixtures["journal"],
        focus=fixtures["timer"], summarizers=_summarizers(), now=NOW)
    text = render_markdown(review)
    for heading in ("## What happened", "## What slipped",
                    "## What's blocked", "## Focus", "## Next week"):
        assert heading in text
    assert "send the deck" in text
    assert "Ship the bot" in text
    assert "not connected" in text
    assert "unavailable" in text
    # Method form agrees with the function form.
    assert review.render_markdown() == text


def test_monthly_aggregates_weekly_reviews(fixtures):
    week1 = build_weekly_review(
        commitments=fixtures["ledger"], decisions=fixtures["journal"],
        focus=fixtures["timer"], summarizers=_summarizers(), now=NOW)
    week2 = build_weekly_review(
        commitments=fixtures["ledger"], decisions=fixtures["journal"],
        focus=fixtures["timer"], summarizers=_summarizers(),
        now=_ts(2026, 10, 21))
    monthly = build_monthly_review(weeks=[week1, week2])
    assert isinstance(monthly, MonthlyReview)
    assert monthly.week_count == 2
    assert len(monthly.reliability_trend) == 2
    # Totals are sums of the weekly reviews, not invented.
    assert monthly.totals["wins"] == len(week1.wins) + len(week2.wins)
    assert monthly.totals["slipped"] == len(week1.slipped) + len(week2.slipped)
    assert monthly.totals["blocked"] == len(week1.blocked) + len(week2.blocked)
    assert monthly.totals["focus_sessions"] == (
        week1.focus_totals["sessions"] + week2.focus_totals["sessions"])
    # The same overdue item in both weeks is the top slipped item (x2).
    assert monthly.top_slipped[0]["item"] == "send the deck"
    assert monthly.top_slipped[0]["count"] == 2
    # The same open decision in both weeks is the top blocked item.
    assert monthly.top_blocked[0]["item"] == "Ship the bot"
    assert monthly.top_blocked[0]["count"] == 2
    text = render_markdown(monthly)
    for heading in ("## What happened", "## What slipped",
                    "## What's blocked", "## Focus",
                    "## Reliability trend", "## Next month"):
        assert heading in text


def test_monthly_empty_weeks_builds_one_honest_week():
    monthly = build_monthly_review(now=NOW)
    assert monthly.week_count == 1
    assert monthly.totals["wins"] == 0
    assert monthly.reliability_trend[0]["reliability"] is None
