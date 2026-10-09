"""Tests for SongSpec, ABC melody parsing, and the LLM composer."""

from nomorals.media.abc_melody import parse_abc_melody, validate_abc
from nomorals.media.composer_llm import (
    _extract_json,
    algorithmic_spec,
    brain_available,
)
from nomorals.media.songspec import GrooveSpec, SectionSpec, SongSpec


# ── SongSpec ──────────────────────────────────────────────────────────────

def _make_spec():
    return SongSpec(
        title="Test Song", style="afrobeats", key="A", mode="minor",
        tempo=100,
        groove=GrooveSpec(feel="bouncy", swing=0.15),
        sections=[
            SectionSpec(name="verse", bars=8,
                        chords=["i", "VI", "III", "VII"], energy=0.5,
                        lyrics=["line one", "line two"],
                        melody_abc="C2 D2 | E2 z2 |"),
            SectionSpec(name="chorus", bars=8,
                        chords=["i", "VII", "VI", "VII"], energy=0.85),
        ],
    )


def test_spec_roundtrip():
    s = _make_spec()
    d = s.to_dict()
    s2 = SongSpec.from_dict(d)
    assert s2.title == s.title
    assert s2.tempo == s.tempo
    assert len(s2.sections) == 2
    assert s2.sections[0].melody_abc == "C2 D2 | E2 z2 |"
    assert s2.groove.swing == 0.15


def test_spec_validation_ok():
    assert _make_spec().validate() == []


def test_spec_validation_catches_problems():
    s = SongSpec(title="", sections=[])
    errs = s.validate()
    assert any("title" in e for e in errs)
    assert any("sections" in e for e in errs)


def test_spec_validation_flat_energy():
    s = SongSpec(title="x", sections=[
        SectionSpec(name="verse", bars=8, chords=["i"], energy=0.5),
        SectionSpec(name="chorus", bars=8, chords=["i"], energy=0.52),
        SectionSpec(name="bridge", bars=8, chords=["i"], energy=0.5),
    ])
    assert any("flat" in e for e in s.validate())


def test_spec_validation_bad_mode():
    s = _make_spec()
    s.mode = "dorian"
    assert any("mode" in e for e in s.validate())


def test_spec_clamps():
    s = SongSpec.from_dict({"tempo": 9999, "sections": [
        {"name": "verse", "bars": 999, "energy": 5.0}]})
    assert s.tempo == 220
    assert s.sections[0].bars == 64
    assert s.sections[0].energy == 1.0


# ── ABC melody ────────────────────────────────────────────────────────────

def test_abc_basic():
    notes = parse_abc_melody("C D E F")
    assert len(notes) == 4
    assert notes[0] == (60, 0.0, 0.5)  # C4, beat 0, eighth note
    assert notes[1][0] == 62  # D


def test_abc_octaves():
    notes = parse_abc_melody("C c C, c'")
    assert notes[0][0] == 60   # C4
    assert notes[1][0] == 72   # c5
    assert notes[2][0] == 48   # C3
    assert notes[3][0] == 84   # c6


def test_abc_accidentals():
    notes = parse_abc_melody("^C _D =E")
    assert notes[0][0] == 61  # C#
    assert notes[1][0] == 61  # Db
    assert notes[2][0] == 64  # E


def test_abc_durations():
    notes = parse_abc_melody("C2 D/ E//")
    assert notes[0][2] == 1.0   # 2 * 0.5
    assert notes[1][2] == 0.25  # half of 0.5
    assert notes[2][2] == 0.125


def test_abc_rests_advance_time():
    notes = parse_abc_melody("C z2 D")
    assert len(notes) == 2
    assert notes[1][1] == 1.5  # 0.5 (C) + 1.0 (z2)


def test_abc_bar_lines_ignored():
    a = parse_abc_melody("C D | E F")
    b = parse_abc_melody("C D E F")
    assert a == b


def test_abc_chords():
    notes = parse_abc_melody("[CEG]2")
    assert len(notes) == 3
    assert {n[0] for n in notes} == {60, 64, 67}
    assert all(n[1] == 0.0 for n in notes)  # simultaneous


def test_abc_empty():
    assert parse_abc_melody("") == []
    assert parse_abc_melody("   ") == []


def test_abc_never_raises():
    assert parse_abc_melody("!!! garbage {{{") == [] or True


def test_abc_validate_catches_leaps():
    errs = validate_abc("C,,,, c''''")
    assert any("leap" in e or "span" in e for e in errs)


def test_abc_validate_ok():
    assert validate_abc("C D E F | G2 z2 |") == []


def test_abc_validate_duration_mismatch():
    errs = validate_abc("C D", expected_beats=16.0)
    assert any("beats" in e for e in errs)


# ── composer ──────────────────────────────────────────────────────────────

def test_brain_unavailable_without_router():
    class Ctx:
        pass
    assert not brain_available(Ctx())


def test_brain_unavailable_mock():
    class Router:
        def stats_snapshot(self):
            return {"active": "mock"}
    class Ctx:
        router = Router()
    assert not brain_available(Ctx())


def test_extract_json_plain():
    d = _extract_json('{"title": "x", "tempo": 100}')
    assert d == {"title": "x", "tempo": 100}


def test_extract_json_fenced():
    d = _extract_json('```json\n{"a": 1}\n```')
    assert d == {"a": 1}


def test_extract_json_garbage():
    assert _extract_json("no json here") is None
    assert _extract_json("") is None


def test_algorithmic_spec_deterministic():
    s1 = algorithmic_spec("test song", seed=42)
    s2 = algorithmic_spec("test song", seed=42)
    assert s1.to_dict() == s2.to_dict()


def test_algorithmic_spec_valid():
    s = algorithmic_spec("a sad afrobeats song about Lagos traffic", seed=7)
    assert s.source == "algorithmic"
    assert s.sections, "must have sections"
    assert s.validate() == [], f"spec invalid: {s.validate()}"
    assert s.groove.feel, "groove feel should be described"
    assert s.bass_approach, "bass approach should be described"


def test_algorithmic_spec_genre_groove():
    s = algorithmic_spec("dark trap banger", seed=7)
    assert s.sections
    # trap or high-energy should be reflected in the groove text
    text = (s.groove.feel + s.groove.kick_style).lower()
    assert any(w in text for w in
               ("trap", "808", "driving", "four", "heavy"))
