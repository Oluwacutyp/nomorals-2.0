"""Tests for the voice rebuild: design, morphing, biometrics, conversation."""

import json
import os
import struct
import tempfile
import wave

import pytest

from nomorals.voice.design import describe_to_params, VoiceDesign


def _make_wav(path, seconds=2.0, freq=220.0, sr=22050):
    n = int(seconds * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(16000 * (
                0.6 * __import__("math").sin(2 * 3.14159 * freq * i / sr)
            ))) for i in range(n))
        wf.writeframes(frames)


# ── describe_to_params ────────────────────────────────────────────────

class TestDescribeToParams:
    def test_deep(self):
        d = describe_to_params("deep warm narrator")
        assert d.pitch_semitones < 0
        assert d.warmth > 0
        assert d.matched

    def test_bright_young(self):
        d = describe_to_params("bright young female")
        assert d.pitch_semitones > 0
        assert d.brightness > 0

    def test_stacking(self):
        d = describe_to_params("deep dark slow")
        assert d.pitch_semitones <= -4.0
        assert d.speed < 1.0

    def test_unknown_description(self):
        d = describe_to_params("xyzzy plugh")
        assert not d.matched
        assert d.pitch_semitones == 0.0

    def test_clamping(self):
        d = describe_to_params("deep deep deep deep deep low bass dark")
        assert d.pitch_semitones >= -12.0

    def test_to_dict(self):
        d = describe_to_params("warm")
        dd = d.to_dict()
        assert set(dd) == {"pitch_semitones", "speed", "brightness",
                           "warmth", "matched"}


# ── shape_voice ───────────────────────────────────────────────────────

class TestShapeVoice:
    def test_shape_roundtrip(self):
        from nomorals.voice.design import shape_voice
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.wav")
            out = os.path.join(td, "out.wav")
            _make_wav(src)
            res = shape_voice(src, out, VoiceDesign(pitch_semitones=2.0))
            assert res["ok"], res
            assert os.path.exists(out)
            # output is a valid wav of similar length
            with wave.open(out, "rb") as wf:
                assert wf.getnframes() > 0

    def test_shape_unreadable(self):
        from nomorals.voice.design import shape_voice
        res = shape_voice("/nonexistent.wav", "/tmp/x.wav", VoiceDesign())
        assert not res["ok"]

    def test_tone_shaping(self):
        from nomorals.voice.design import shape_voice, _read_mono_wav
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.wav")
            out = os.path.join(td, "out.wav")
            _make_wav(src)
            before, _ = _read_mono_wav(src)
            res = shape_voice(src, out,
                              VoiceDesign(brightness=0.8, warmth=0.8))
            assert res["ok"], res
            after, _ = _read_mono_wav(out)
            # shaping changed the samples
            assert any(abs(a - b) > 1e-6
                       for a, b in zip(before[:1000], after[:1000]))


# ── biometrics ────────────────────────────────────────────────────────

class TestBiometrics:
    def test_enroll_and_verify_same(self):
        from nomorals.voice import biometrics as bio
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "owner.wav")
            _make_wav(clip, freq=180.0)
            vd = os.path.join(td, "voices")
            os.makedirs(vd)
            res = bio.register_owner_voice(clip, voices_dir=vd)
            assert res["ok"], res
            assert bio.owner_enrolled(vd)
            # same clip should match itself
            v = bio.verify_voice(clip, voices_dir=vd)
            assert v["ok"]
            assert v["match"] is True

    def test_verify_different(self):
        from nomorals.voice import biometrics as bio
        with tempfile.TemporaryDirectory() as td:
            owner = os.path.join(td, "owner.wav")
            other = os.path.join(td, "other.wav")
            _make_wav(owner, freq=150.0)
            _make_wav(other, freq=400.0)
            vd = os.path.join(td, "voices")
            os.makedirs(vd)
            assert bio.register_owner_voice(owner, voices_dir=vd)["ok"]
            v = bio.verify_voice(other, voices_dir=vd)
            assert v["ok"]
            # very different pitch → should not match
            assert v["match"] is not True

    def test_not_enrolled(self):
        from nomorals.voice import biometrics as bio
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "x.wav")
            _make_wav(clip)
            v = bio.verify_voice(clip, voices_dir=td)
            assert v["ok"]
            assert v["match"] is None


# ── conversation mode state ───────────────────────────────────────────

class TestConversationMode:
    def test_toggle(self):
        from nomorals.voice import conversation as conv
        with tempfile.TemporaryDirectory() as td:
            assert not conv.voice_mode_on("chat1", td)
            conv.set_voice_mode("chat1", True, td)
            assert conv.voice_mode_on("chat1", td)
            assert conv.list_voice_chats(td) == ["chat1"]
            conv.set_voice_mode("chat1", False, td)
            assert not conv.voice_mode_on("chat1", td)

    def test_voice_turn_no_audio(self):
        from nomorals.voice import conversation as conv
        res = conv.voice_turn("/nonexistent.wav", chat_key="c")
        assert not res["ok"]
