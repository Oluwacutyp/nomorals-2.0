"""Ask-your-data coaching — build-map #53. All offline (mocked source).

The critical invariant: coaching, NOT diagnosis. Every public output must
pass guard_coaching(); banned diagnostic phrases raise.
"""
import re
from datetime import date, timedelta

import pytest

from nomorals.health.coach import (
    Answer,
    COACH_BANNED_PHRASES,
    HealthCoach,
    HealthDataSource,
    Readiness,
    ReadinessBriefingProvider,
    guard_coaching,
    parse_intent,
)


# ── fakes ────────────────────────────────────────────────────────────────

class FakeSource(HealthDataSource):
    """Canned health-cli data. No subprocesses, ever."""

    def __init__(self, *, metrics=None, sleep=None, workouts=None,
                 status_counts=None):
        super().__init__(binary="/nonexistent/health-cli")
        self._metrics = metrics or []
        self._sleep = sleep or []
        self._workouts = workouts or []
        self._status = status_counts or {}

    def _run(self, *args):
        raise AssertionError("FakeSource must not shell out")

    def status(self, provider):
        # Exercise the REAL resolve_provider()/has_data() fallback logic.
        n = self._status.get(provider, 0)
        return {"categories": [{"name": "daily-metrics", "record_count": n}]}

    def metrics(self, start, end):
        return self._metrics

    def sleep_sessions(self, start, end):
        return self._sleep

    def workouts(self, start, end):
        return self._workouts


def _sleep(bed_h, awake_min=20, eff=0.92, awaken=1, day_offset=0):
    d = date.today() - timedelta(days=day_offset)
    return {
        "sleep_in_bed_duration_sec": bed_h * 3600,
        "sleep_awake_duration_sec": awake_min * 60,
        "sleep_efficiency": eff,
        "number_of_awakenings": awaken,
        "start_datetime": f"{d}T22:45:00-04:00",
        "end_datetime": f"{d + timedelta(days=1)}T06:30:00-04:00",
    }


def _metric(day_offset, steps, hrv):
    d = date.today() - timedelta(days=day_offset)
    return {"date": d.isoformat(), "step_count": steps,
            "heart_rate_variability_ms": hrv}


def _good_source():
    """Healthy user: 8h sleep, steady HRV, active, no recent workouts."""
    metrics = [_metric(i, 9000 + i * 100, 52.0) for i in range(30)]
    sleep = [_sleep(8.2, day_offset=i) for i in range(7)]
    return FakeSource(metrics=metrics, sleep=sleep, workouts=[],
                      status_counts={"healthconnect": 120})


def _bad_source():
    """Struggling user: short sleep, crashed HRV, 2 recent workouts."""
    metrics = ([_metric(i, 3000, 56.0) for i in range(7, 30)]
               + [_metric(i, 2500, 40.0) for i in range(7)])
    sleep = [_sleep(5.3, eff=0.78, awaken=4, day_offset=i) for i in range(7)]
    w = {"workout_type": "running", "active_duration_sec": 1800,
         "start_datetime": "2026-10-07T18:00:00-04:00",
         "end_datetime": "2026-10-07T18:30:00-04:00"}
    return FakeSource(metrics=metrics, sleep=sleep,
                      workouts=[dict(w), dict(w)],
                      status_counts={"healthconnect": 120})


def _coach(source):
    import tempfile
    from nomorals.health.timeline import HealthTimeline
    tl = HealthTimeline(db_path=tempfile.mktemp(suffix=".db"))
    return HealthCoach(source=source, timeline=tl)


# ── intent parsing ───────────────────────────────────────────────────────

def test_parse_intent_sleep():
    assert parse_intent("how did I sleep this week?") == ("sleep", 7)
    assert parse_intent("how did I sleep last night") == ("sleep", 1)


def test_parse_intent_recovery():
    assert parse_intent("am I recovering?")[0] == "recovery"
    assert parse_intent("what is my HRV doing?")[0] == "recovery"


def test_parse_intent_activity():
    assert parse_intent("how active was I yesterday?") == ("activity", 1)
    assert parse_intent("show my workouts this month")[0] == "activity"


def test_parse_intent_overview():
    assert parse_intent("how am I doing?")[0] == "overview"


# ── answers ──────────────────────────────────────────────────────────────

def test_ask_sleep_reports_numbers():
    a = _coach(_good_source()).ask("how did I sleep this week?")
    assert a.intent == "sleep" and a.has_data
    guard_coaching(a.text)
    assert "7h" in a.text or "8h" in a.text  # last-night duration present
    assert a.numbers["sessions"] == 7


def test_ask_activity_reports_numbers():
    a = _coach(_good_source()).ask("how active was I this week?")
    assert a.intent == "activity" and a.has_data
    guard_coaching(a.text)
    assert "steps" in a.text
    assert a.numbers["avg_steps"] > 8000


def test_ask_recovery_routes_to_readiness():
    a = _coach(_good_source()).ask("am I recovering well?")
    assert a.intent == "recovery"
    guard_coaching(a.text)
    assert "readiness" in a.text.lower()


def test_ask_overview():
    a = _coach(_good_source()).ask("how am I doing?")
    assert a.intent == "overview" and a.has_data
    guard_coaching(a.text)


def test_no_data_is_honest():
    src = FakeSource()  # status_counts=None → no data
    a = _coach(src).ask("how did I sleep?")
    assert not a.has_data
    assert "no health data synced" in a.text
    # no fabricated numbers anywhere
    assert not re.search(r"\d+h \d+m|\d{2,}ms|[\d,]+ steps", a.text)
    r = _coach(src).readiness()
    assert not r.has_data
    w = _coach(src).weekly_recap()
    assert not w.has_data and "no health data synced" in w.text


# ── readiness ────────────────────────────────────────────────────────────

def test_readiness_high():
    r = _coach(_good_source()).readiness(log=False)
    assert r.has_data and r.level == "high", r.reasons
    assert r.score >= 70
    text = r.format()
    guard_coaching(text)
    assert "green light" in text


def test_readiness_low():
    r = _coach(_bad_source()).readiness(log=False)
    assert r.has_data and r.level == "low", (r.level, r.score, r.reasons)
    assert r.score < 45
    text = r.format()
    guard_coaching(text)
    assert "light day" in text
    # reasons carry real numbers
    assert any("5h" in x or "5.3" in x or "sleep" in x for x in r.reasons)


def test_readiness_logs_to_timeline():
    import tempfile
    from nomorals.health.timeline import HealthTimeline
    path = tempfile.mktemp(suffix=".db")
    tl = HealthTimeline(db_path=path)
    coach = HealthCoach(source=_good_source(), timeline=tl)
    coach.readiness(log=True)
    notes = tl.timeline(event_type="note")
    assert any("readiness check" in e.text for e in notes)
    tl.close()


# ── coaching guard ───────────────────────────────────────────────────────

def test_guard_rejects_diagnosis():
    for bad in ("you might have sleep apnea", "this sounds like overtraining",
                "you have a vitamin deficiency", "I diagnos this as"):
        with pytest.raises(ValueError):
            guard_coaching(bad)


def test_all_public_outputs_pass_guard():
    coach = _coach(_bad_source())
    texts = [
        coach.ask("how did I sleep this week?").text,
        coach.ask("how active was I?").text,
        coach.readiness(log=False).format(),
        coach.weekly_recap(log=False).text,
        coach.ask("how am I doing?").text,
    ]
    for t in texts:
        guard_coaching(t)  # raises on any banned phrase


# ── provider fallback ────────────────────────────────────────────────────

def test_provider_fallback_picks_data():
    src = FakeSource(status_counts={"healthkit": 0, "healthconnect": 87})
    assert src.resolve_provider() == "healthconnect"
    assert src.has_data()


def test_no_provider_anywhere():
    src = FakeSource(status_counts={"healthkit": 0, "healthconnect": 0})
    assert src.resolve_provider() is None
    assert not src.has_data()


# ── briefing hook ────────────────────────────────────────────────────────

def test_briefing_low_readiness_emits_section():
    coach = _coach(_bad_source())
    prov = ReadinessBriefingProvider(coach=coach)
    section = prov.collect(object(), 0.0)
    assert section is not None
    assert section.name == "readiness"
    assert "low" in "\n".join(section.lines)
    guard_coaching("\n".join(section.lines))


def test_briefing_high_readiness_silent():
    coach = _coach(_good_source())
    prov = ReadinessBriefingProvider(coach=coach)
    assert prov.collect(object(), 0.0) is None


def test_briefing_never_raises():
    class Boom(HealthDataSource):
        def has_data(self):
            raise RuntimeError("boom")
    prov = ReadinessBriefingProvider(
        coach=HealthCoach(source=Boom(), timeline=None))
    assert prov.collect(object(), 0.0) is None


# ── scoping ──────────────────────────────────────────────────────────────

def test_community_refused():
    with pytest.raises(PermissionError):
        HealthCoach(community=True)


def test_empty_question_usage():
    a = _coach(_good_source()).ask("   ")
    assert "ask me about" in a.text
