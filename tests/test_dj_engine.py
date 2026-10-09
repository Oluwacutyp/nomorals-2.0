"""Tests for nomorals.media.dj_engine — real DJ DSP."""

import math
from array import array

import pytest

from nomorals.media import dj_engine as eng
from nomorals.media.dj_engine import (
    TrackAnalysis,
    analyze_track,
    camelot_code,
    detect_bpm,
    detect_key,
    harmonic_score,
    plan_energy_arc,
    plan_transition,
    track_energy,
    HeuristicTaste,
    TASTE_FEATURE_SCHEMA,
)

SR = 22050


def _click_track(bpm: float, seconds: float = 12.0) -> array:
    """Metronome clicks with drum-like decay — what BPM detection eats."""
    n = int(SR * seconds)
    out = array("d", [0.0]) * n
    beat_n = int(SR * 60.0 / bpm)
    for start in range(0, n, beat_n):
        for i in range(min(2205, n - start)):  # 100ms click
            t = i / SR
            out[start + i] += math.exp(-t * 60.0) * math.sin(
                2 * math.pi * 180.0 * t)
    return out


def _chord_tones(freqs: list[float], seconds: float = 20.0) -> array:
    n = int(SR * seconds)
    out = array("d", [0.0]) * n
    for f in freqs:
        for i in range(n):
            t = i / SR
            out[i] += 0.3 * math.sin(2 * math.pi * f * t)
    # gentle amplitude wobble so it's not a dead drone
    for i in range(n):
        out[i] *= 0.7 + 0.3 * math.sin(2 * math.pi * 2.0 * i / SR)
    return out


def _write_wav(path, samples: array) -> str:
    import wave, struct
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(struct.pack("<%dh" % len(samples),
                                  *[max(-32768, min(32767, int(s * 32767)))
                                    for s in samples]))
    return str(path)


# ── BPM detection ────────────────────────────────────────────────────────────

def test_detect_bpm_click_120():
    bpm = detect_bpm(_click_track(120.0))
    assert bpm is not None
    assert 114.0 <= bpm <= 126.0, f"got {bpm}"


def test_detect_bpm_click_98():
    bpm = detect_bpm(_click_track(98.0))
    assert bpm is not None
    assert 92.0 <= bpm <= 104.0, f"got {bpm}"


def test_detect_bpm_silence_is_none():
    assert detect_bpm(array("d", [0.0]) * SR * 6) is None


def test_detect_bpm_too_short_is_none():
    assert detect_bpm(_click_track(120.0, seconds=2.0)) is None


# ── key detection ────────────────────────────────────────────────────────────

def test_detect_key_c_major():
    # C4 E4 G4 — unambiguous C major triad
    key, mode = detect_key(_chord_tones([261.63, 329.63, 392.00]))
    assert (key, mode) == ("C", "major"), f"got {(key, mode)}"


def test_detect_key_a_minor():
    # A3 C4 E4 — A minor triad
    key, mode = detect_key(_chord_tones([220.00, 261.63, 329.63]))
    assert (key, mode) == ("A", "minor"), f"got {(key, mode)}"


def test_detect_key_silence_unknown():
    assert detect_key(array("d", [0.0]) * SR * 10) == ("", "")


# ── energy ───────────────────────────────────────────────────────────────────

def test_energy_loud_brighter_than_quiet():
    loud = track_energy(_click_track(120.0))
    quiet = array("d", [s * 0.05 for s in _click_track(120.0)])
    assert track_energy(quiet) < loud
    assert 0.0 <= loud <= 1.0


# ── Camelot ──────────────────────────────────────────────────────────────────

def test_camelot_table():
    assert camelot_code("C", "major") == "8B"
    assert camelot_code("A", "minor") == "8A"
    assert camelot_code("F#", "major") == "2B"
    assert camelot_code("", "major") == ""


def test_harmonic_score():
    a = TrackAnalysis(camelot="8A")            # A minor
    assert harmonic_score(a, TrackAnalysis(camelot="8A")) == 1.0
    assert harmonic_score(a, TrackAnalysis(camelot="9A")) == 0.8   # +1 step
    assert harmonic_score(a, TrackAnalysis(camelot="8B")) == 0.7   # relative
    assert harmonic_score(a, TrackAnalysis(camelot="3B")) == 0.0   # clash
    # unknown keys are neutral, not punished
    assert harmonic_score(a, TrackAnalysis(camelot="")) == 0.5


# ── transition planning ──────────────────────────────────────────────────────

def _trk(bpm=None, camelot="", energy=0.5, dur=60.0):
    return TrackAnalysis(bpm=bpm, camelot=camelot, energy=energy,
                         duration_s=dur,
                         bpm_source="ground-truth" if bpm else "unknown")


def test_plan_blend_when_tempo_close():
    out = _trk(bpm=120.0, camelot="8A", dur=120.0)
    inn = _trk(bpm=122.0, camelot="9A")
    plan = plan_transition(out, inn)
    assert plan.kind == "blend"
    assert plan.blend_beats == 16
    assert plan.start_beat % 4 == 0  # phrase-aligned
    assert abs(plan.sync_ratio - 120.0 / 122.0) < 1e-6


def test_plan_echo_drop_when_tempo_far_but_harmonic():
    out = _trk(bpm=120.0, camelot="8A")
    inn = _trk(bpm=140.0, camelot="9A")  # too far to sync, harmonically fine
    plan = plan_transition(out, inn)
    assert plan.kind == "echo-drop"


def test_plan_break_when_clash():
    out = _trk(bpm=120.0, camelot="8A")
    inn = _trk(bpm=150.0, camelot="3B")  # far tempo + key clash
    plan = plan_transition(out, inn)
    assert plan.kind == "break"


def test_plan_no_fake_blend_without_bpm():
    out = _trk(bpm=None, camelot="8A")
    inn = _trk(bpm=None, camelot="8A")
    plan = plan_transition(out, inn)
    # harmonic 1.0 >= 0.7 → echo-drop, never a claimed beatmatch
    assert plan.kind == "echo-drop"


# ── energy arc ───────────────────────────────────────────────────────────────

def test_energy_arc_peaks_middle():
    tracks = [_trk(energy=e) for e in (0.9, 0.2, 0.7, 0.4, 0.6)]
    arc = plan_energy_arc(tracks)
    energies = [t.energy for t in arc]
    peak_pos = energies.index(max(energies))
    assert peak_pos == len(arc) // 2
    # ramps up to peak
    assert energies[0] <= energies[peak_pos]


# ── taste interface ──────────────────────────────────────────────────────────

def test_heuristic_taste_scores():
    taste = HeuristicTaste()
    out = _trk(bpm=120.0, camelot="8A", energy=0.5, dur=120.0)
    inn = _trk(bpm=121.0, camelot="8A", energy=0.6)
    plan = plan_transition(out, inn)
    s = taste.score_transition(out, inn, plan)
    assert 0.0 <= s <= 1.0
    assert s > 0.7  # perfect harmonic + blend should score high
    seq = taste.score_sequence([out, inn])
    assert 0.0 <= seq <= 1.0


def test_taste_schema_documented():
    assert "bpm_delta_pct" in TASTE_FEATURE_SCHEMA
    assert "label" in TASTE_FEATURE_SCHEMA


# ── end-to-end analysis ──────────────────────────────────────────────────────

def test_analyze_track_with_ground_truth(tmp_path):
    wav = _write_wav(tmp_path / "t.wav", _click_track(128.0, seconds=10.0))
    a = analyze_track(wav, title="test", known_bpm=128.0,
                      known_key="G", known_mode="major")
    assert a.bpm == 128.0
    assert a.bpm_source == "ground-truth"
    assert a.key == "G" and a.camelot == "9B"
    assert a.key_source == "ground-truth"
    assert a.duration_s > 9.0


def test_analyze_track_detects(tmp_path):
    wav = _write_wav(tmp_path / "t.wav", _click_track(120.0, seconds=12.0))
    a = analyze_track(wav, title="test")
    assert a.bpm is not None and 114.0 <= a.bpm <= 126.0
    assert a.bpm_source == "detected"
