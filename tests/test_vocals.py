"""Offline tests for the #58 vocal pipeline (stems + vocals)."""

from __future__ import annotations

import os
import wave

import numpy as np
import pytest

from nomorals.media import stems as stems_mod
from nomorals.media import vocals as vocals_mod
from nomorals.media.stems import (
    DemucsUnavailable,
    Stems,
    probe_demucs,
    separate,
)
from nomorals.media.vocals import (
    FullSongResult,
    VocalChain,
    VocalModelUnavailable,
    VocalResult,
    VoiceInfo,
    parse_full_request,
    probe_diffsinger,
    probe_rvc,
    select_voice,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _write_test_wav(path: str, seconds: float = 1.0, sr: int = 44100,
                    freq: float = 440.0) -> str:
    t = np.arange(int(sr * seconds)) / sr
    pcm = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype("<i2")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


class FakeSeparator:
    """Mimics _RealSeparator: 4 constant stems."""
    name = "fake-demucs"

    def load(self):
        pass

    def unload(self):
        pass

    def separate(self, audio_path):
        mono, sr, _ = vocals_mod._read_wav(audio_path)
        return {n: np.full_like(mono, 0.1 * (i + 1))
                for i, n in enumerate(("vocals", "drums", "bass", "other"))}, sr


class FakeDiffSinger:
    name = "fake-diffsinger"
    loaded = False

    def load(self):
        type(self).loaded = True

    def unload(self):
        type(self).loaded = False

    def render(self, lyrics, melody, out_path):
        _write_test_wav(out_path, seconds=2.0, freq=523.0)
        return out_path


class FakeRVC:
    name = "fake-rvc"

    def load(self):
        pass

    def unload(self):
        pass

    def convert(self, vocal_wav, model_path, out_path):
        # "convert": pitch-shift the dry vocal down a touch, write it out
        mono, sr, _ = vocals_mod._read_wav(vocal_wav)
        _write_test_wav(out_path, seconds=len(mono) / sr, freq=392.0)
        return out_path


VOICES = {
    "owner-xtts": VoiceInfo("owner-xtts", source="xtts",
                            license="noncommercial",
                            model_path="/tmp/owner_xtts.pth"),
    "narrator-cb": VoiceInfo("narrator-cb", source="chatterbox",
                             license="mit",
                             model_path="/tmp/narrator_cb.pth"),
}


def _touch_models(tmp_path):
    for v in VOICES.values():
        p = tmp_path / os.path.basename(v.model_path)
        p.write_bytes(b"fake")
        v.model_path = str(p)


# ── Demucs ───────────────────────────────────────────────────────────────

def test_separate_mocked(tmp_path):
    src = _write_test_wav(str(tmp_path / "mix.wav"))
    s = separate(src, out_dir=str(tmp_path / "stems"), separator=FakeSeparator())
    assert isinstance(s, Stems) and s.ok
    assert not s.missing()
    for p in s.paths().values():
        assert os.path.isfile(p)


def test_separate_missing_file():
    with pytest.raises(DemucsUnavailable):
        separate("/nope/not-here.wav", separator=FakeSeparator())


def test_probe_demucs_never_raises():
    p = probe_demucs(profile="workstation")
    assert isinstance(p.available, bool)


def test_probe_demucs_termux_refuses():
    p = probe_demucs(profile="termux")
    assert not p.available
    assert p.reason


# ── voice selection / audience rule ──────────────────────────────────────

def test_select_voice_private_xtts_ok():
    v = select_voice("owner-xtts", "private", VOICES)
    assert v.voice_id == "owner-xtts"


def test_select_voice_public_xtts_blocked():
    with pytest.raises(VocalModelUnavailable):
        select_voice("owner-xtts", "public", VOICES)


def test_select_voice_public_chatterbox_ok():
    v = select_voice("narrator-cb", "public", VOICES)
    assert v.voice_id == "narrator-cb"


def test_select_voice_unknown():
    with pytest.raises(VocalModelUnavailable):
        select_voice("ghost", "private", VOICES)


def test_select_voice_bad_audience():
    with pytest.raises(VocalModelUnavailable):
        select_voice("narrator-cb", "everyone", VOICES)


# ── DiffSinger render ────────────────────────────────────────────────────

def test_render_mocked(tmp_path):
    chain = VocalChain(profile="workstation", diffsinger=FakeDiffSinger())
    melody = [(60, 0.0, 1.0), (62, 1.0, 1.0)]
    r = chain.render("[Verse]\nhello world", melody,
                     workdir=str(tmp_path))
    assert isinstance(r, VocalResult) and r.ok
    assert os.path.isfile(r.audio_path)
    assert not FakeDiffSinger.loaded  # VRAM hygiene: unloaded


def test_render_no_lyrics():
    chain = VocalChain(profile="workstation", diffsinger=FakeDiffSinger())
    with pytest.raises(VocalModelUnavailable):
        chain.render("", [(60, 0.0, 1.0)])


def test_render_no_melody():
    chain = VocalChain(profile="workstation", diffsinger=FakeDiffSinger())
    with pytest.raises(VocalModelUnavailable):
        chain.render("hello", [])


def test_render_termux_refuses(tmp_path):
    chain = VocalChain(profile="termux", diffsinger=FakeDiffSinger())
    with pytest.raises(VocalModelUnavailable):
        chain.render("hello", [(60, 0.0, 1.0)],
                     workdir=str(tmp_path))


def test_render_no_model_fails_closed(tmp_path):
    # no adapter, no diffsinger installed here → honest failure
    chain = VocalChain(profile="workstation")
    with pytest.raises(VocalModelUnavailable):
        chain.render("hello", [(60, 0.0, 1.0)],
                     workdir=str(tmp_path))


# ── RVC convert ──────────────────────────────────────────────────────────

def test_convert_mocked(tmp_path):
    _touch_models(tmp_path)
    chain = VocalChain(profile="workstation", rvc=FakeRVC(), voices=VOICES)
    dry = _write_test_wav(str(tmp_path / "dry.wav"), seconds=1.0)
    r = chain.convert(dry, "owner-xtts", audience="private",
                      workdir=str(tmp_path))
    assert r.ok and os.path.isfile(r.audio_path)
    assert r.voice_id == "owner-xtts"


def test_convert_public_xtts_blocked(tmp_path):
    _touch_models(tmp_path)
    chain = VocalChain(profile="workstation", rvc=FakeRVC(), voices=VOICES)
    dry = _write_test_wav(str(tmp_path / "dry.wav"))
    with pytest.raises(VocalModelUnavailable):
        chain.convert(dry, "owner-xtts", audience="public",
                      workdir=str(tmp_path))


def test_convert_missing_model_file(tmp_path):
    chain = VocalChain(profile="workstation", rvc=FakeRVC(), voices={
        "ghost": VoiceInfo("ghost", source="rvc", license="mit",
                           model_path="/nope/ghost.pth")})
    dry = _write_test_wav(str(tmp_path / "dry.wav"))
    with pytest.raises(VocalModelUnavailable):
        chain.convert(dry, "ghost", workdir=str(tmp_path))


# ── full chain ───────────────────────────────────────────────────────────

def test_full_song_mocked(tmp_path):
    _touch_models(tmp_path)
    chain = VocalChain(profile="workstation", diffsinger=FakeDiffSinger(),
                       rvc=FakeRVC(), voices=VOICES)
    bed = _write_test_wav(str(tmp_path / "bed.wav"), seconds=2.0,
                          freq=220.0)
    melody = [(60, 0.0, 1.0), (64, 1.0, 1.0)]
    r = chain.full_song(bed, "[Verse]\nlagos nights", melody, "owner-xtts",
                        audience="private", title="test song",
                        workdir=str(tmp_path))
    assert isinstance(r, FullSongResult) and r.ok
    assert os.path.isfile(r.master_path)
    assert "AI-generated" in r.note


def test_mix_pure_dsp(tmp_path):
    chain = VocalChain(profile="workstation")
    bed = _write_test_wav(str(tmp_path / "bed.wav"), seconds=1.0, sr=48000)
    voc = _write_test_wav(str(tmp_path / "voc.wav"), seconds=1.0,
                          sr=44100, freq=660.0)
    out = str(tmp_path / "master.wav")
    path = chain.mix(bed, voc, out_path=out)
    assert os.path.isfile(path)
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getframerate() == 48000  # bed rate wins


# ── chat parsing ─────────────────────────────────────────────────────────

def test_parse_full_request_natural():
    r = parse_full_request("make me an afrobeats song about Lagos",
                           styles={"afrobeats", "pop"})
    assert r == {"topic": "Lagos", "style": "afrobeats"}


def test_parse_full_request_slash():
    r = parse_full_request("/music full lagos nights afrobeats",
                           styles={"afrobeats", "pop"})
    assert r == {"topic": "lagos nights", "style": "afrobeats"}


def test_parse_full_request_not_a_request():
    assert parse_full_request("what's the weather") is None
    assert parse_full_request("") is None


def test_probes_never_raise():
    assert isinstance(probe_diffsinger("workstation").available, bool)
    assert isinstance(probe_rvc("workstation").available, bool)


def test_separate_no_demucs_fails_closed(tmp_path):
    src = _write_test_wav(str(tmp_path / "mix.wav"))
    if stems_mod._importable("demucs"):
        pytest.skip("demucs actually installed here")
    with pytest.raises(DemucsUnavailable) as exc:
        separate(src, out_dir=str(tmp_path / "stems"))
    assert "pip install demucs" in str(exc.value)


# ── RVC voice registry ───────────────────────────────────────────────────

def test_registry_add_and_license(tmp_path):
    from nomorals.media.vocals import RVCVoiceRegistry
    reg = RVCVoiceRegistry(path=str(tmp_path / "voices.json"))
    model = tmp_path / "v.pth"
    model.write_bytes(b"x")
    v = reg.add("mine", str(model), source="xtts")
    assert v.license == "noncommercial"
    v2 = reg.add("cb", str(model), source="chatterbox")
    assert v2.license == "mit"
    # persisted
    reg2 = RVCVoiceRegistry(path=str(tmp_path / "voices.json"))
    assert set(reg2.all()) == {"mine", "cb"}


def test_registry_add_missing_model(tmp_path):
    from nomorals.media.vocals import RVCVoiceRegistry
    reg = RVCVoiceRegistry(path=str(tmp_path / "voices.json"))
    with pytest.raises(VocalModelUnavailable):
        reg.add("ghost", "/nope/ghost.pth")


def test_registry_default_single(tmp_path):
    from nomorals.media.vocals import RVCVoiceRegistry
    reg = RVCVoiceRegistry(path=str(tmp_path / "voices.json"))
    model = tmp_path / "v.pth"
    model.write_bytes(b"x")
    reg.add("only", str(model))
    assert reg.default().voice_id == "only"


def test_registry_default_none_when_many(tmp_path):
    from nomorals.media.vocals import RVCVoiceRegistry
    reg = RVCVoiceRegistry(path=str(tmp_path / "voices.json"))
    model = tmp_path / "v.pth"
    model.write_bytes(b"x")
    reg.add("a", str(model))
    reg.add("b", str(model))
    assert reg.default() is None
