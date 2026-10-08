"""Pre-visit layer tests (build-map #52): navigation, NOT diagnosis.

Hard rules under test:
- red flags → emergency, immediately
- uncertain → bumped UP, never down
- crisis resources on EVERY route output
- no condition-naming anywhere (banned-phrase scan)
- disclaimer on every user-facing surface
"""
import re

import pytest

from nomorals.health.previsit import (
    BANNED_PHRASES,
    CRISIS_RESOURCES,
    NAVIGATION_DISCLAIMER,
    PREVISIT_BANNED_PHRASES,
    Route,
    consultation_costs,
    format_costs,
    format_recap,
    format_route,
    prepare_visit,
    summarize_visit,
    triage_route,
    _check_banned,
)
from nomorals.health.timeline import HealthTimeline


@pytest.fixture()
def tl(tmp_path):
    t = HealthTimeline(db_path=tmp_path / "t.db")
    yield t
    t.close()


# ── red flags → emergency ───────────────────────────────────────────────

@pytest.mark.parametrize("symptom", [
    "chest pain",
    "pressure in my chest",
    "I can't breathe",
    "difficulty breathing",
    "shortness of breath",
    "bleeding won't stop",
    "severe bleeding from the cut",
    "I want to kill myself",
    "suicidal thoughts",
    "thinking of ending my life",
    "face drooping on one side",
    "slurred speech suddenly",
    "sudden weakness in my arm",
    "I fainted",
    "collapsed at work",
    "seizure",
    "worst headache of my life",
    "sudden severe headache",
    "throat swelling shut",
    "baby has fever",
    "my infant has a fever",
    "sudden severe abdominal pain",
])
def test_red_flags_route_emergency(symptom):
    route = triage_route([symptom])
    assert route.level == "emergency", f"{symptom!r} should be emergency"
    assert route.reasons, "red flag reason must be named"


def test_red_flag_mid_sentence():
    route = triage_route(["mild headache", "chest pain started an hour ago"])
    assert route.level == "emergency"


def test_multiple_red_flags_all_named():
    route = triage_route(["chest pain and difficulty breathing"])
    assert route.level == "emergency"
    assert len(route.reasons) >= 2


# ── conservative: uncertain bumps UP ────────────────────────────────────

def test_vague_symptom_bumps_to_routine_not_self_care():
    route = triage_route(["weird feeling in my leg"])
    assert route.level == "routine_care"
    assert any("conservative" in r or "routing up" in r
               for r in route.reasons)


def test_empty_symptoms_default_routine():
    route = triage_route([])
    assert route.level == "routine_care"


def test_mild_headache_stays_self_care():
    route = triage_route(["mild headache"])
    assert route.level == "self_care"


def test_high_severity_history_bumps_up():
    class Ev:
        severity = 5
    route = triage_route(["headache"], history=[Ev()])
    assert route.level == "urgent"


def test_history_read_failure_does_not_crash():
    def boom():
        raise RuntimeError("db gone")
    route = triage_route(["headache"], history=boom)
    assert route.level in ("self_care", "routine_care", "urgent", "emergency")


def test_urgent_pattern():
    route = triage_route(["high fever for two days"])
    assert route.level == "urgent"


# ── crisis resources + disclaimer always present ────────────────────────

@pytest.mark.parametrize("symptoms", [
    ["mild headache"],
    ["weird leg feeling"],
    ["high fever"],
    ["chest pain"],
])
def test_crisis_resources_on_every_route(symptoms):
    route = triage_route(symptoms)
    assert route.crisis_resources == CRISIS_RESOURCES
    out = format_route(route)
    for resource in CRISIS_RESOURCES:
        assert resource in out
    assert "112" in out and "199" in out


def test_disclaimer_on_every_surface():
    route = triage_route(["headache"])
    assert NAVIGATION_DISCLAIMER in format_route(route)
    assert NAVIGATION_DISCLAIMER in format_costs()
    assert "navigation, not diagnosis" in prepare_visit(["headache"]).lower() \
        or "navigation, not diagnosis" in prepare_visit(["headache"])


# ── no diagnosis anywhere ───────────────────────────────────────────────

def test_no_banned_phrases_in_all_outputs(tl):
    route = triage_route(["chest pain"])
    outputs = [
        format_route(route),
        format_route(triage_route(["mild headache"])),
        format_route(triage_route(["weird thing"])),
        format_costs(),
        prepare_visit(["headache"], timeline=tl),
        format_recap(summarize_visit(
            "doctor said I have malaria, prescribed coartem, come back in 3 days",
            timeline=tl)),
    ]
    for out in outputs:
        hits = _check_banned(out)
        assert not hits, f"banned phrases in output: {hits}\n{out[:300]}"


def test_route_never_names_conditions():
    # condition names must never appear — only care levels and actions
    route = triage_route(["chest pain", "sweating", "nausea"])
    out = format_route(route).lower()
    for word in ("heart attack", "stroke", "malaria", "typhoid",
                 "appendicitis", "pneumonia", "asthma"):
        assert word not in out, f"condition named: {word}"


# ── price transparency ──────────────────────────────────────────────────

def test_costs_format():
    costs = consultation_costs()
    assert len(costs) >= 4
    out = format_costs()
    assert "₦" in out
    assert "approximate" in out.lower()
    assert "verify" in out.lower()
    for c in costs:
        assert c.low_naira < c.high_naira
        assert c.low_naira > 0


# ── visit prep ──────────────────────────────────────────────────────────

def test_prepare_visit_uses_timeline(tl):
    tl.log("symptom", "headache since morning", severity=3)
    tl.log("measurement", "temperature", value="38.5", unit="°C")
    out = prepare_visit(["headache"], timeline=tl)
    assert "headache since morning" in out
    assert "38.5" in out
    assert "bring" in out.lower()
    assert "questions to ask" in out.lower()


def test_prepare_visit_without_timeline():
    out = prepare_visit(["knee pain"])
    assert "knee pain" in out
    assert "bring" in out.lower()


# ── post-visit recap: quotes the doctor, never diagnoses ────────────────

def test_summarize_visit_quotes_doctor(tl):
    notes = ("The doctor said I have malaria. She prescribed coartem "
             "twice daily. Come back in 3 days for review.")
    recap = summarize_visit(notes, timeline=tl)
    assert recap.doctor_said, "doctor statements should be extracted"
    assert any("malaria" in d for d in recap.doctor_said)
    assert any("coartem" in m for m in recap.meds_reported)
    assert recap.follow_ups, "follow-up should be extracted"
    assert recap.logged_event_id, "must log a visit event"
    # the recap quotes — the words are the USER's report, framed as such
    out = format_recap(recap)
    assert "your words" in out or "as you reported" in out or \
        "what you told me" in out
    hits = _check_banned(out)
    assert not hits, f"banned phrases in recap: {hits}"


def test_summarize_visit_logs_to_timeline(tl):
    recap = summarize_visit("doctor said rest, come back next week",
                            timeline=tl)
    events = tl.timeline(event_type="visit")
    assert any(e.id == recap.logged_event_id for e in events)


def test_summarize_empty_notes():
    recap = summarize_visit("")
    assert recap.doctor_said == []
    out = format_recap(recap)
    assert "usage" in out.lower()


def test_recap_never_adds_medical_advice():
    # even with loaded notes, no advice may appear
    recap = summarize_visit.__wrapped__ if hasattr(
        summarize_visit, "__wrapped__") else summarize_visit
    out = format_recap(summarize_visit(
        "doctor said take paracetamol", timeline=None))
    assert "you should take" not in out.lower()
