"""Tail-73 pins for nomorals/health/patterns.py —
0 converts / 6 exclusions (115, 122, 242, 448, 585, 590).

6 tests for 6 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED. The file contains no exception handlers, so it
adds zero census sites.

patterns.py calls no failures-ledger record function anywhere
(grep-proven; its recording sibling handlers log via _log), so
NOT-recording is pinned against the health package ledger
(nomorals/health/failures.py): recent_failures() snapshot
before/after every trigger.

Verdict notes (full grounds in the tail-73 SURVEY.md):
- 115: _factor_days sleep-short float guard failure -> the
  day is not counted; the FACTOR_DEFS regex guarantees a
  numeric group, and claims need >= MIN_EVIDENCE days anyway.
  Fail-closed (fewer pattern claims).
- 122: _factor_days good-sleep float guard — mirror of 115.
- 242: _sleep_hours failure -> None, the documented
  "unparseable" value; consumers treat None as absence.
- 448: MoodStats.format failure -> "couldn't build mood
  stats.", an honest string; also the _assert_safe escape
  hatch that keeps banned phrasing from shipping.
- 585: provider collect BriefingSection import failure ->
  None = "no section" in the briefing protocol; lazy-import
  degradation ladder, test-pinned by the existing suite.
- 590: provider collect HealthTimeline() failure -> None;
  the timeline raises BY DESIGN in community contexts
  (owner-scoping) and the provider must not crash a briefing.
"""

import re
import sys
import time
from types import SimpleNamespace
from unittest import mock

import nomorals.health.patterns as P
import nomorals.health.timeline as tl_mod
from nomorals.health.failures import recent_failures


def _snap():
    return list(recent_failures(200))


class _Ev:
    def __init__(self, text, ts=None, event_type="note", severity=None):
        self.text = text
        self.ts = ts if ts is not None else time.time()
        self.event_type = event_type
        self.severity = severity


def test_115_sleep_short_guard_failure_skips_the_day(monkeypatch):
    before = _snap()
    day = P._day(time.time())
    healthy = P._factor_days([_Ev("slept 5h")])
    assert day in healthy.get("sleep-short", set())  # control
    monkeypatch.setattr(P, "FACTOR_DEFS", ((
        "sleep-short",
        re.compile(r"\b(slept)\s+(\w+)"), "short night"),))
    broken = P._factor_days([_Ev("slept abc")])
    assert broken.get("sleep-short", set()) == set()
    assert _snap() == before


def test_122_good_sleep_guard_failure_skips_the_day(monkeypatch):
    before = _snap()
    day = P._day(time.time())
    healthy = P._factor_days([_Ev("slept 8h")])
    assert day in healthy.get("good-sleep", set())  # control
    monkeypatch.setattr(P, "FACTOR_DEFS", ((
        "good-sleep",
        re.compile(r"\b(slept)\s+(\w+)"), "solid night"),))
    broken = P._factor_days([_Ev("slept abc")])
    assert broken.get("good-sleep", set()) == set()
    assert _snap() == before


def test_242_sleep_hours_failure_is_the_unparseable_value():
    before = _snap()
    assert P._sleep_hours(_Ev("slept 6.5h")) == 6.5
    assert P._sleep_hours(_Ev("no data")) is None
    # non-str text: re.search raises TypeError inside the guard
    assert P._sleep_hours(_Ev(12345)) is None
    assert _snap() == before


def test_448_mood_stats_format_failure_string_is_honest():
    before = _snap()
    empty = P.MoodStats(days=30).format()
    assert empty.startswith("no mood entries")
    banned = P.MoodStats(days=30, entries=1, avg_mood=3.0,
                         best_day="a clinical day").format()
    assert banned == "couldn't build mood stats."
    assert _snap() == before


def test_585_collect_import_failure_is_no_section(monkeypatch):
    before = _snap()
    prov = P.PatternsBriefingProvider()
    timeline_cls = mock.Mock(
        return_value=SimpleNamespace(timeline=lambda **k: []))
    monkeypatch.setattr(tl_mod, "HealthTimeline", timeline_cls)
    # control: import works, timeline is constructed, no patterns
    assert prov.collect(object(), 0) is None
    assert timeline_cls.call_count == 1
    # blocked import: collect returns None BEFORE the timeline rung
    monkeypatch.setitem(sys.modules,
                        "nomorals.agents.morning_briefing", None)
    assert prov.collect(object(), 0) is None
    assert timeline_cls.call_count == 1  # never reached again
    assert _snap() == before


def test_590_collect_timeline_failure_is_no_section(monkeypatch):
    before = _snap()
    prov = P.PatternsBriefingProvider()
    broken_cls = mock.Mock(side_effect=RuntimeError("community context"))
    monkeypatch.setattr(tl_mod, "HealthTimeline", broken_cls)
    assert prov.collect(object(), 0) is None
    assert broken_cls.call_count == 1  # construction was attempted
    ok_cls = mock.Mock(
        return_value=SimpleNamespace(timeline=lambda **k: []))
    monkeypatch.setattr(tl_mod, "HealthTimeline", ok_cls)
    assert prov.collect(object(), 0) is None  # no patterns -> no section
    assert ok_cls.call_count == 1
    assert _snap() == before
