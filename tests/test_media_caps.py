"""Tests for media write-size caps (rex-disk mitigation).

All offline: oversized writes are simulated by shrinking the cap via
``monkeypatch`` instead of generating hundreds of megabytes of audio.
"""

from __future__ import annotations

import os
import wave

import pytest

from nomorals.media import caps
from nomorals.media import synth


# ── constants exist ────────────────────────────────────────────────────


def test_cap_constants_exist():
    assert caps.MAX_WRITE_BYTES > 0
    assert caps.MAX_AUDIO_WRITE_BYTES == 500 * 1024 * 1024
    assert caps.MAX_VIDEO_WRITE_BYTES > 0
    assert caps.MAX_IMAGE_WRITE_BYTES > 0


def test_audio_cap_generous_for_audiobooks():
    # 500 MiB holds ~99 min of 16-bit mono WAV @44.1 kHz — full audiobook
    # chapters and long mixes fit; raw WAV itself tops out at 4 GiB anyway.
    mono_minutes = caps.MAX_AUDIO_WRITE_BYTES / (44100 * 2) / 60
    assert mono_minutes > 90


# ── check_write_size ───────────────────────────────────────────────────


def test_check_write_size_ok():
    ok, reason = caps.check_write_size(1024)
    assert ok is True
    assert reason == ""


def test_check_write_size_refused_has_clear_reason():
    ok, reason = caps.check_write_size(caps.MAX_AUDIO_WRITE_BYTES + 1)
    assert ok is False
    assert isinstance(reason, str) and len(reason) > 10
    assert "cap" in reason.lower()
    # never raises, even for garbage input
    ok2, reason2 = caps.check_write_size(None)  # type: ignore[arg-type]
    assert ok2 is False
    assert reason2


# ── synth.write_wav ────────────────────────────────────────────────────


def test_write_wav_refuses_oversized_never_raises(tmp_path, monkeypatch):
    out = str(tmp_path / "huge.wav")
    monkeypatch.setattr(caps, "MAX_AUDIO_WRITE_BYTES", 100)
    monkeypatch.setattr(synth, "render_wav", lambda *a, **k: b"x" * 200)
    result = synth.write_wav(out, {}, 120.0)
    assert result is None
    assert not os.path.exists(out)


def test_write_wav_normal_size_succeeds(tmp_path):
    from nomorals.core.midi import NoteEvent

    out = str(tmp_path / "ok.wav")
    parts = {"melody": [NoteEvent(note=69, start=0.0, duration=1.0)]}
    result = synth.write_wav(out, parts, 120.0)
    assert result == out
    assert os.path.isfile(out)
    with wave.open(out, "rb") as wf:
        assert wf.getnframes() > 0


# ── per-module wav helpers ─────────────────────────────────────────────


def _shrink_audio_cap(monkeypatch, to_bytes: int = 100):
    monkeypatch.setattr(caps, "MAX_AUDIO_WRITE_BYTES", to_bytes)


def test_stems_helper_refuses_oversized(tmp_path, monkeypatch):
    from nomorals.media import stems

    _shrink_audio_cap(monkeypatch)
    out = str(tmp_path / "stem.wav")
    assert stems._write_wav_mono(out, [0.5] * 1000, 16000) is False
    assert not os.path.exists(out)


def test_stems_helper_normal_succeeds(tmp_path):
    from nomorals.media import stems

    out = str(tmp_path / "stem.wav")
    assert stems._write_wav_mono(out, [0.5] * 100, 16000) is True
    assert os.path.isfile(out)


def test_vocals_helper_refuses_oversized(tmp_path, monkeypatch):
    from nomorals.media import vocals

    _shrink_audio_cap(monkeypatch)
    out = str(tmp_path / "mix.wav")
    assert vocals._write_wav_stereo(out, [0.5] * 1000, [0.5] * 1000, 16000) is False
    assert not os.path.exists(out)


def test_vocals_helper_normal_succeeds(tmp_path):
    from nomorals.media import vocals

    out = str(tmp_path / "mix.wav")
    assert vocals._write_wav_stereo(out, [0.5] * 100, [0.5] * 100, 16000) is True
    assert os.path.isfile(out)


def test_ace_step_helper_refuses_oversized(tmp_path, monkeypatch):
    from nomorals.media import ace_step

    _shrink_audio_cap(monkeypatch)
    out = str(tmp_path / "bed.wav")
    import struct

    pcm = struct.pack("<1000f", *([0.1] * 1000))
    assert ace_step._write_wav_float32_stereo(out, 44100, pcm) is False
    assert not os.path.exists(out)


def test_ace_step_helper_normal_succeeds(tmp_path):
    from nomorals.media import ace_step

    out = str(tmp_path / "bed.wav")
    import struct

    pcm = struct.pack("<100f", *([0.1] * 100))
    assert ace_step._write_wav_float32_stereo(out, 44100, pcm) is True
    assert os.path.isfile(out)
