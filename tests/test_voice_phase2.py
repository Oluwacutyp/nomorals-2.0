"""Phase 2 voice god-tier tests — real behavior, no stubs."""
import math
import os
import sys
import wave
from array import array

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice.ambience import (
    AMBIENCES,
    apply_room,
    generate_ambience,
    mix_under,
    with_ambience,
)
from nomorals.voice.accent import normalize_accent, ACCENTS
from nomorals.voice.longform import split_sentences, _is_degenerate, _crossfade, _rms
from nomorals.voice.mastering import remove_dc, deess, normalize, soft_limit, master
from nomorals.voice.nl_director import parse_direction
from nomorals.voice.rvc_bridge import detect_rvc, RVCUnavailable
from nomorals.voice.singing import parse_melody, midi_to_hz, Note


def _tone(sr=24000, secs=1.0, freq=220.0, amp=8000):
    n = int(sr * secs)
    return array("h", [int(amp * math.sin(2 * math.pi * freq * i / sr))
                       for i in range(n)]), sr


# ── ambience ──────────────────────────────────────────────────────────────

def test_ambience_kinds_generate():
    for kind in AMBIENCES:
        samples, sr = generate_ambience(kind, 0.5, seed=1)
        assert sr == 24000
        assert len(samples) == 12000, kind
        # not digital silence
        assert max(abs(s) for s in samples) > 100, kind


def test_ambience_deterministic():
    a, _ = generate_ambience("rain", 0.5, seed=42)
    b, _ = generate_ambience("rain", 0.5, seed=42)
    assert a == b


def test_mix_under_ducks():
    voice, sr = _tone(secs=1.0, amp=12000)
    amb, _ = generate_ambience("rain", 1.0, seed=1)
    mixed = mix_under(voice, amb, ambience_level=0.3)
    assert len(mixed) == len(voice)
    # voice dominates: mixed close to voice where voice is loud
    assert max(abs(s) for s in mixed) > 9000


def test_apply_room_adds_tail():
    voice, sr = _tone(secs=0.3)
    wet = apply_room(voice, sr, "hall")
    assert len(wet) == len(voice)
    # reverb changes the signal (wet mix is audible)
    assert wet != voice


def test_with_ambience_parses_description():
    voice, sr = _tone(secs=0.5)
    out = with_ambience(voice, sr, "light rain", room="small")
    assert len(out) == len(voice)
    assert out != voice


# ── accent ────────────────────────────────────────────────────────────────

def test_normalize_accent():
    assert normalize_accent("british") == "british"
    assert normalize_accent("British English") == "british"
    assert normalize_accent("nigerian") == "nigerian"
    assert normalize_accent("klingon") == ""


def test_nl_director_parses_accent():
    d = parse_direction("[said angrily in British accent]")
    assert d.emotion == "angry"
    assert d.accent == "british"


def test_nl_director_parses_ambient():
    d = parse_direction("[light rain]")
    assert d.ambient == "light rain"


# ── singing ───────────────────────────────────────────────────────────────

def test_parse_melody_note_names():
    notes = parse_melody("C4:0.5:hello D4:0.5:world")
    assert len(notes) == 2
    assert notes[0].midi == 60  # middle C
    assert notes[1].midi == 62
    assert notes[0].lyric == "hello"


def test_parse_melody_midi_numbers():
    notes = parse_melody("60:0.5:la 64:1:laaa")
    assert notes[0].midi == 60
    assert notes[1].duration_s == 1.0


def test_parse_melody_sharps():
    notes = parse_melody("C#4:0.5:x")
    assert notes[0].midi == 61


def test_midi_to_hz():
    assert abs(midi_to_hz(69) - 440.0) < 0.01
    assert abs(midi_to_hz(60) - 261.63) < 0.1


def test_singing_backend_unavailable_honest():
    from nomorals.voice.singing import DiffSingerBackend
    # no voicebank in a fresh env → honest RuntimeError, never a fake
    import unittest.mock as mock
    with mock.patch("nomorals.voice.singing.SVS_DIR", return_value="/nonexistent"):
        with pytest.raises(RuntimeError, match="voicebank"):
            DiffSingerBackend()


# ── longform ──────────────────────────────────────────────────────────────

def test_split_sentences_packs():
    text = "Hello world. This is a test. " * 50
    chunks = split_sentences(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 1200


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_is_degenerate_detects_silence():
    sr = 24000
    silent = array("h", [0] * sr)
    assert _is_degenerate(silent, sr) == "silence"
    voice, _ = _tone(secs=1.0)
    assert _is_degenerate(voice, sr) == ""


def test_crossfade_no_click():
    a, sr = _tone(secs=0.5, freq=220.0)
    b, _ = _tone(secs=0.5, freq=440.0)
    out = _crossfade(a, b, sr, fade_ms=50.0)
    # length = a + b - fade overlap region handled internally
    assert len(out) > len(a)
    assert max(abs(s) for s in out) <= 32767


def test_longform_synthesizer_with_fake_tts(tmp_path):
    from nomorals.voice.longform import LongFormSynthesizer

    class FakeTTS:
        def perform(self, text, voice_name=None, mood="neutral"):
            samples, sr = _tone(secs=0.4)
            p = str(tmp_path / f"chunk_{abs(hash(text))}.wav")
            with wave.open(p, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(samples.tobytes())
            return {"path": p}

    synth = LongFormSynthesizer(FakeTTS(), mood="neutral")
    text = "First sentence here. Second sentence here. " * 20
    result = synth.synthesize(text, str(tmp_path / "out.wav"))
    assert result["ok"]
    assert result["chunks"] >= 2
    assert result["failed_chunks"] == []
    assert os.path.exists(result["path"])
    # output longer than a single chunk
    assert result["seconds"] > 0.5


# ── mastering ──────────────────────────────────────────────────────────────

def test_master_chain():
    voice, sr = _tone(secs=0.5, amp=20000)
    out, stages = master(voice, sr)
    assert stages == ["dc", "deess", "normalize", "limit"]
    assert max(abs(s) for s in out) <= 32767
    # normalized near target
    assert max(abs(s) for s in out) > 25000


def test_soft_limit_no_clip():
    hot = array("h", [30000] * 1000)
    out = soft_limit(hot)
    assert max(abs(s) for s in out) <= 32767


def test_normalize_silence_safe():
    silent = array("h", [0] * 100)
    assert normalize(silent) == silent


# ── rvc bridge ────────────────────────────────────────────────────────────

def test_rvc_detect_never_raises():
    probe = detect_rvc()
    assert "ok" in probe and "method" in probe
    # In this sandbox RVC is not installed → honest False
    assert probe["ok"] is False


def test_rvc_convert_raises_unavailable():
    with pytest.raises(RVCUnavailable):
        from nomorals.voice.rvc_bridge import convert
        convert("/tmp/nonexistent.wav", "nosuchmodel")


# ── neural emotion tiers ──────────────────────────────────────────────────

def test_emotion_tiers_honest_without_rvc(tmp_path):
    from nomorals.voice.neural_emotion import render_emotional

    class FakeTTS:
        def perform(self, text, voice_name=None, mood="neutral",
                    intensity=3):
            samples, sr = _tone(secs=0.3)
            p = str(tmp_path / "emo.wav")
            with wave.open(p, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(samples.tobytes())
            return {"path": p, "backend": "fake"}

        def speak(self, text, voice_name=None):
            return self.perform(text)

    # No RVC in sandbox → must fall to native or dsp, and SAY so
    result = render_emotional("I am so angry!", FakeTTS(),
                              direction="[said angrily]")
    assert result["ok"]
    assert result["tier"] in ("native", "dsp")
    assert result["tier"] != "rvc"
    assert os.path.exists(result["path"])


# ── voice package imports ─────────────────────────────────────────────────

def test_voice_package_exports():
    import nomorals.voice as v
    for name in ("sing", "synthesize_long", "render_emotional",
                 "convert_accent", "generate_ambience", "master",
                 "detect_rvc", "parse_melody"):
        assert hasattr(v, name), name


def test_tts_has_godtier_methods():
    from nomorals.voice.tts import UniversalTTS
    for name in ("speak_rich", "sing", "speak_long", "_post_process"):
        assert hasattr(UniversalTTS, name), name


def test_diffsinger_registered():
    from nomorals.voice.tts import _BACKENDS
    assert "diffsinger" in _BACKENDS
