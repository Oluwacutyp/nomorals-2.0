"""Mood-pattern detection (build-map #55). Wellness, never therapy."""
import time

import pytest

from nomorals.health.patterns import (
    BANNED_PHRASES,
    CAUSATION_PHRASES,
    PatternsBriefingProvider,
    detect_patterns,
    format_patterns,
)
from nomorals.health.timeline import HealthTimeline


@pytest.fixture()
def tl(tmp_path):
    t = HealthTimeline(db_path=str(tmp_path / "h.db"))
    yield t
    t.close()


def _log_mood_sleep(tl, moods, sleeps):
    """moods: [(severity, day_offset)], sleeps: [(hours, day_offset)]."""
    now = time.time()
    for sev, off in moods:
        tl.log("mood", f"mood {sev}/5", severity=sev,
               ts=now - off * 86400)
    for hours, off in sleeps:
        tl.log("sleep", f"slept {hours} hours",
               ts=now - off * 86400 - 8 * 3600)


def test_sleep_mood_correlation(tl):
    # 4 low-mood days, 3 after short nights
    _log_mood_sleep(
        tl,
        moods=[(2, 1), (1, 3), (2, 5), (2, 7)],
        sleeps=[(5.0, 1), (4.5, 3), (5.5, 5), (8.0, 7)],
    )
    pats = detect_patterns(tl, days=30)
    sm = [p for p in pats if p.kind == "sleep-mood"]
    assert sm, "expected a sleep-mood pattern"
    assert sm[0].strength == pytest.approx(0.75)
    assert "3 of your last 4" in sm[0].description
    assert sm[0].involves_low_mood


def test_minimum_evidence_no_claim(tl):
    # only 2 low-mood days — not enough to claim
    _log_mood_sleep(
        tl,
        moods=[(2, 1), (1, 3)],
        sleeps=[(5.0, 1), (4.5, 3)],
    )
    pats = detect_patterns(tl, days=30)
    assert not [p for p in pats if p.kind == "sleep-mood"]


def test_no_low_mood_no_pattern(tl):
    _log_mood_sleep(
        tl,
        moods=[(4, 1), (5, 3), (4, 5)],
        sleeps=[(8.0, 1), (7.5, 3), (8.0, 5)],
    )
    pats = detect_patterns(tl, days=30)
    assert not [p for p in pats if p.kind == "sleep-mood"]


def test_weak_pattern_not_surfaced(tl):
    # 4 low days, only 1 after a short night → strength 0.25 < 0.6
    _log_mood_sleep(
        tl,
        moods=[(2, 1), (2, 3), (1, 5), (2, 7)],
        sleeps=[(5.0, 1), (8.0, 3), (7.5, 5), (8.0, 7)],
    )
    pats = detect_patterns(tl, days=30)
    assert not [p for p in pats if p.kind == "sleep-mood"]


def test_activity_mood_pattern(tl):
    now = time.time()
    for sev, off in [(5, 1), (4, 3), (5, 5), (4, 7)]:
        tl.log("mood", f"mood {sev}/5", severity=sev,
               ts=now - off * 86400)
    for off in [2, 4, 6]:  # active the day before 3 of 4 good days
        tl.log("note", "morning workout at the gym",
               ts=now - off * 86400)
    pats = detect_patterns(tl, days=30)
    am = [p for p in pats if p.kind == "activity-mood"]
    assert am
    assert "tend to" in am[0].description or "came after" in am[0].description


def test_correlation_not_causation_language(tl):
    _log_mood_sleep(
        tl,
        moods=[(2, 1), (1, 3), (2, 5)],
        sleeps=[(5.0, 1), (4.5, 3), (5.5, 5)],
    )
    text = format_patterns(detect_patterns(tl, days=30))
    lowered = text.lower()
    for phrase in CAUSATION_PHRASES:
        assert phrase not in lowered, f"causation language: {phrase!r}"


def test_no_diagnostic_phrases(tl):
    _log_mood_sleep(
        tl,
        moods=[(2, 1), (1, 3), (2, 5)],
        sleeps=[(5.0, 1), (4.5, 3), (5.5, 5)],
    )
    text = format_patterns(detect_patterns(tl, days=30))
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        assert phrase not in lowered, f"banned phrase: {phrase!r}"


def test_crisis_resources_on_low_mood(tl):
    _log_mood_sleep(
        tl,
        moods=[(2, 1), (1, 3), (2, 5)],
        sleeps=[(5.0, 1), (4.5, 3), (5.5, 5)],
    )
    text = format_patterns(detect_patterns(tl, days=30))
    assert "112" in text  # national emergency line from CRISIS_RESOURCES
    assert "Lagos Lifeline" in text


def test_no_crisis_resources_without_low_mood():
    from nomorals.health.patterns import Pattern
    p = Pattern(description="2 of 3 good days came after active days.",
                strength=0.67, involves_low_mood=False,
                kind="activity-mood")
    text = format_patterns([p])
    assert "112" not in text


def test_empty_patterns_message():
    text = format_patterns([])
    assert "no strong patterns" in text.lower()


def test_briefing_provider(tl):
    _log_mood_sleep(
        tl,
        moods=[(2, 1), (1, 3), (2, 5)],
        sleeps=[(5.0, 1), (4.5, 3), (5.5, 5)],
    )
    # provider reads the default HealthTimeline; patch via monkeypatch
    import nomorals.health.patterns as pat_mod
    orig = pat_mod.HealthTimeline if hasattr(pat_mod, "HealthTimeline") else None
    try:
        # collect() constructs its own HealthTimeline — instead test the
        # section-building path with a stubbed timeline class
        class FakeTL:
            def __init__(self, *a, **k):
                self._tl = tl

            def __getattr__(self, name):
                return getattr(self._tl, name)
        import nomorals.health.timeline as tl_mod
        real = tl_mod.HealthTimeline
        tl_mod.HealthTimeline = FakeTL  # type: ignore[assignment]
        # patterns.collect imports HealthTimeline from .timeline lazily
        import importlib
        prov = PatternsBriefingProvider()
        sec = prov.collect(object(), time.time() - 86400)
        assert sec is not None
        assert any("low-mood" in line for line in sec.lines)
        assert any("112" in line for line in sec.lines)
    finally:
        tl_mod.HealthTimeline = real  # type: ignore[assignment]


def test_briefing_provider_no_data(tmp_path):
    empty = HealthTimeline(db_path=str(tmp_path / "e.db"))
    import nomorals.health.timeline as tl_mod
    real = tl_mod.HealthTimeline
    tl_mod.HealthTimeline = lambda *a, **k: empty  # type: ignore[assignment]
    try:
        prov = PatternsBriefingProvider()
        assert prov.collect(object(), 0) is None
    finally:
        tl_mod.HealthTimeline = real  # type: ignore[assignment]
        empty.close()


def test_detect_never_raises():
    assert detect_patterns(object(), days=30) == []
    assert format_patterns([]) != ""
