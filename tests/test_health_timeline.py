"""Patient-side health timeline (build-map #51).

TRACKING ONLY. These tests also enforce the no-diagnosis rule:
no banned phrase may appear in any module output.
"""
import re
import time

import pytest

from nomorals.health import (
    BANNED_PHRASES,
    EVENT_TYPES,
    HealthTimeline,
    health_db_path,
    parse_health_note,
)
from nomorals.health import timeline as tl_module


@pytest.fixture()
def tl(tmp_path):
    t = HealthTimeline(db_path=str(tmp_path / "health.db"))
    yield t
    t.close()


# ── event types ──────────────────────────────────────────────────────

def test_all_event_types_log(tl):
    tl.log("symptom", "knee hurt today", severity=3)
    tl.log("visit", "saw Dr Ada, annual checkup")
    tl.log("medication", "started vitamin D 1000IU")
    tl.log("measurement", "morning BP", value="120/80", unit="mmHg")
    tl.log("mood", "feeling okay", severity=3)
    tl.log("sleep", "slept 6 hours", severity=3)
    tl.log("note", "drank more water today")
    types = {e.event_type for e in tl.timeline()}
    assert types == set(EVENT_TYPES)


def test_unknown_event_type_rejected(tl):
    with pytest.raises(ValueError, match="unknown health event type"):
        tl.log("diagnosis", "you have the flu")


def test_empty_text_rejected(tl):
    with pytest.raises(ValueError, match="must not be empty"):
        tl.log("symptom", "   ")


def test_severity_validation(tl):
    tl.log("symptom", "mild headache", severity=1)
    tl.log("symptom", "worst pain", severity=5)
    with pytest.raises(ValueError, match="severity must be 1-5"):
        tl.log("symptom", "bad", severity=0)
    with pytest.raises(ValueError, match="severity must be 1-5"):
        tl.log("symptom", "bad", severity=6)


def test_timeline_ordering(tl):
    now = time.time()
    tl.log("note", "third", ts=now + 30)
    tl.log("note", "first", ts=now)
    tl.log("note", "second", ts=now + 10)
    texts = [e.text for e in tl.timeline()]
    assert texts == ["first", "second", "third"]


def test_timeline_filtering(tl):
    tl.log("symptom", "headache")
    tl.log("mood", "good day", severity=4)
    tl.log("symptom", "back pain", severity=2)
    syms = tl.timeline(event_type="symptom")
    assert len(syms) == 2
    assert all(e.event_type == "symptom" for e in syms)
    since = time.time() - 1
    assert len(tl.timeline(since=since)) == 3
    assert tl.timeline(since=time.time() + 100) == []


def test_user_words_preserved_verbatim(tl):
    raw = "my knee hurt today, kinda throbbing lol"
    ev = tl.log("symptom", raw, severity=3)
    assert ev.text == raw
    assert tl.timeline()[0].text == raw


def test_source_recorded(tl):
    ev = tl.log("note", "from voice memo", source="voice")
    assert ev.source == "voice"


# ── summary ──────────────────────────────────────────────────────────

def test_summary_format(tl):
    tl.log("symptom", "headache", severity=3)
    tl.log("measurement", "weight check", value="72", unit="kg")
    s = tl.summary(days=30)
    assert "headache" in s
    assert "[3/5]" in s
    assert "72 kg" in s
    assert "show it to your doctor" in s


def test_summary_empty(tl):
    assert "nothing logged" in tl.summary(days=7)


# ── natural parsing ──────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("log: headache, 3pm", "symptom"),
    ("my knee hurt today", "symptom"),
    ("headache 4/5 since morning", "symptom"),
    ("saw dr Ada about the knee", "visit"),
    ("took ibuprofen 200mg", "medication"),
    ("stopped the antibiotics", "medication"),
    ("BP 120/80 this morning", "measurement"),
    ("weigh 72kg", "measurement"),
    ("slept 5 hours, restless", "sleep"),
    ("feeling anxious today", "mood"),
    ("drank more water", "note"),
    ("random thought about dinner", "note"),
])
def test_parse_health_note_types(text, expected):
    assert parse_health_note(text)["event_type"] == expected


def test_parse_preserves_verbatim():
    raw = "log: headache, 3pm, ugh"
    assert parse_health_note(raw)["text"] == raw


def test_parse_severity():
    assert parse_health_note("headache 4/5")["severity"] == 4
    assert parse_health_note("pain severity 2")["severity"] == 2
    assert parse_health_note("just a headache")["severity"] is None


def test_parse_log_prefix_stripped_for_classification():
    # "log:" prefix must not confuse classification
    assert parse_health_note("log: took my vitamins")["event_type"] == \
        "medication"


# ── privacy: owner-scoped ────────────────────────────────────────────

def test_community_context_refused():
    with pytest.raises(PermissionError, match="owner-scoped"):
        HealthTimeline(community=True)


def test_db_path_owner_scoped():
    p = health_db_path()
    assert ".nomorals" in str(p) and "health" in str(p)
    assert "community" not in str(p).lower()


def test_no_community_imports_health():
    import pathlib
    repo = pathlib.Path(tl_module.__file__).resolve().parents[2]
    offenders = []
    for py in (repo / "nomorals" / "community").rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        src = py.read_text(errors="ignore")
        if "nomorals.health" in src or "from ..health" in src \
                or "from ...health" in src:
            offenders.append(str(py.relative_to(repo)))
    assert offenders == [], f"community code touches health: {offenders}"


# ── no medical interpretation, ever ──────────────────────────────────

def _all_module_text():
    """Health-module source with the BANNED_PHRASES *definition* stripped.

    (The definition itself contains the phrases by construction — what
    matters is they never appear in prose, replies, or summaries.)
    Also scoped to the health handler in runtime_memory, not the tutor's
    ``turn.diagnosis`` (different domain).
    """
    src = open(tl_module.__file__).read()
    # strip the BANNED_PHRASES tuple literal
    src = re.sub(r"BANNED_PHRASES = \(.*?\n\)", "", src, flags=re.DOTALL)
    import pathlib
    rm = (pathlib.Path(tl_module.__file__).resolve().parents[1]
          / "agents" / "partner" / "runtime_memory.py")
    full = rm.read_text()
    # scope to the _control_health method only
    m = re.search(r"def _control_health\(.*?(?=\n    def |\Z)",
                  full, flags=re.DOTALL)
    if m:
        src += "\n" + m.group(0)
    return src.lower()


def test_no_banned_phrases_in_source():
    src = _all_module_text()
    for phrase in BANNED_PHRASES:
        assert phrase.lower() not in src, \
            f"banned medical-interpretation phrase in source: {phrase!r}"


def test_no_banned_phrases_in_outputs(tl):
    tl.log("symptom", "chest pain, 4/5")
    tl.log("medication", "took aspirin")
    outputs = [
        tl.summary(days=30),
        parse_health_note("my head hurts, what is this")["text"],
    ]
    for out in outputs:
        for phrase in BANNED_PHRASES:
            assert phrase.lower() not in out.lower(), \
                f"banned phrase in output: {phrase!r}"


def test_summary_gives_no_advice(tl):
    tl.log("symptom", "fever 38.5", severity=4)
    s = tl.summary(days=30).lower()
    assert "should" not in s or "show it to your doctor" in s
    # the only guidance allowed: see a doctor
    assert "doctor" in s
