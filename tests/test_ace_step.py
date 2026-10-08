"""Offline tests for ACE-Step 1.5 song-bed generation (build-map #57)."""

import os
import wave

import numpy as np
import pytest

from nomorals.media.ace_step import (
    ACEModelUnavailable,
    ACEStepBackend,
    BedRequest,
    format_lyrics_for_acestep,
    make_bed,
    parse_bed_request,
    probe_acestep,
    tags_for_style,
)
from nomorals.media.music import STYLES, resolve_style


class FakeAdapter:
    """Stands in for the real DiT — renders 2s of real PCM, tracks load."""

    name = "fake"

    def __init__(self) -> None:
        self.loaded = False
        self.load_count = 0
        self.unload_count = 0
        self.fail_render = False

    def load(self) -> None:
        self.loaded = True
        self.load_count += 1

    def unload(self) -> None:
        self.loaded = False
        self.unload_count += 1

    def render(self, caption, lyrics, duration_s, seed):
        assert self.loaded, "rendered without load"
        if self.fail_render:
            raise RuntimeError("boom")
        n = 48000 * 2
        t = np.arange(n, dtype=np.float32) / 48000.0
        tone = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        stereo = np.stack([tone, tone], axis=0)
        return 48000, stereo.tobytes()


@pytest.fixture()
def backend(tmp_path):
    return ACEStepBackend(profile="workstation",
                          adapter=FakeAdapter(),
                          checkpoint_dir=str(tmp_path))


# ── style → tags ─────────────────────────────────────────────────────────────

def test_afrobeats_tags():
    tags = tags_for_style(STYLES["afrobeats"])
    assert "afrobeats" in tags
    assert "105bpm" in tags          # midpoint of (100, 110)
    assert "major" in tags
    assert "log drums" in tags
    assert "shakers" in tags


def test_tags_deduped_and_ordered():
    tags = tags_for_style(STYLES["lofi"]).split(", ")
    assert len(tags) == len({t.lower() for t in tags})


def test_tags_include_drum_pattern():
    assert "lofi drums" in tags_for_style(STYLES["lofi"])


# ── profile gating ───────────────────────────────────────────────────────────

def test_termux_refuses_honestly():
    p = probe_acestep(profile="termux")
    assert not p.available
    assert "workstation" in p.reason.lower() or "gpu" in p.reason.lower()


def test_mobile_refuses_honestly():
    assert not probe_acestep(profile="mobile").available


def test_workstation_needs_model_here():
    # acestep isn't installed in this env → honest fail-closed
    p = probe_acestep(profile="workstation")
    assert not p.available
    assert p.reason  # a real reason, not ""


def test_generate_fails_closed_without_model():
    be = ACEStepBackend(profile="workstation")  # real probing, no adapter
    with pytest.raises(ACEModelUnavailable):
        be.generate("la la", "pop")


def test_generate_fails_closed_on_termux():
    be = ACEStepBackend(profile="termux", adapter=FakeAdapter())
    with pytest.raises(ACEModelUnavailable):
        be.generate("la la", "pop")


# ── generate with mocked model ───────────────────────────────────────────────

def test_generate_returns_real_shaped_bed(backend, tmp_path):
    res = backend.generate("[Verse]\nhello lagos", STYLES["afrobeats"],
                           duration_s=60, title="Lagos Nights",
                           workdir=str(tmp_path))
    assert res.ok
    assert res.audio_path.endswith("-bed.wav")
    assert os.path.getsize(res.audio_path) > 1000
    assert "afrobeats" in res.tags
    assert "vocals come next" in res.note.lower()
    assert res.variant == "full"
    # valid WAV: 48kHz stereo
    with wave.open(res.audio_path, "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnchannels() == 2


def test_vram_hygiene_unload_on_success(backend, tmp_path):
    adapter = backend._adapter
    backend.generate("la", "pop", workdir=str(tmp_path))
    assert adapter.load_count == 1
    assert adapter.unload_count == 1
    assert not adapter.loaded  # never held idle


def test_vram_hygiene_unload_on_failure(backend, tmp_path):
    adapter = backend._adapter
    adapter.fail_render = True
    with pytest.raises(RuntimeError):
        backend.generate("la", "pop", workdir=str(tmp_path))
    assert adapter.unload_count == 1  # finally still ran
    assert not adapter.loaded


def test_laptop_duration_cap(tmp_path):
    adapter = FakeAdapter()
    be = ACEStepBackend(profile="pc", adapter=adapter,
                        checkpoint_dir=str(tmp_path))
    res = be.generate("la", "pop", duration_s=200, workdir=str(tmp_path))
    assert res.duration_s == 90  # capped for laptop-class
    assert res.variant == "turbo"


def test_empty_lyrics_rejected(backend, tmp_path):
    with pytest.raises(ACEModelUnavailable):
        backend.generate("   ", "pop", workdir=str(tmp_path))


# ── lyrics formatting ────────────────────────────────────────────────────────

def test_format_lyrics_section_tags():
    from nomorals.media.music import MusicCreator
    song = MusicCreator(None).compose("rain", style="lofi", with_midi=False,
                                      with_audio=False, with_score=False)
    text = format_lyrics_for_acestep(song)
    assert "[Verse]" in text or "[Chorus]" in text
    assert len(text) > 50


# ── chat parsing ─────────────────────────────────────────────────────────────

def test_parse_explicit_bed():
    req = parse_bed_request("bed lagos nights afrobeats", STYLES)
    assert isinstance(req, BedRequest)
    assert req.topic == "lagos nights"
    assert req.style == "afrobeats"


def test_parse_natural_make_me():
    req = parse_bed_request("make me an afrobeats song about Lagos", STYLES)
    assert req is not None
    assert req.style == "afrobeats"
    assert req.topic == "Lagos"


def test_parse_natural_generate():
    req = parse_bed_request("generate a lofi song about rain", STYLES)
    assert req is not None
    assert req.style == "lofi"
    assert req.topic == "rain"


def test_parse_not_a_bed_request():
    assert parse_bed_request("what's the weather like", STYLES) is None
    assert parse_bed_request("", STYLES) is None


def test_parse_bed_no_topic():
    assert parse_bed_request("bed", STYLES) is None


# ── high-level flow ──────────────────────────────────────────────────────────

def test_make_bed_end_to_end(tmp_path):
    be = ACEStepBackend(profile="workstation", adapter=FakeAdapter(),
                        checkpoint_dir=str(tmp_path))
    res = make_bed("lagos nights", style="afrobeats", context=None,
                   backend=be, workdir=str(tmp_path))
    assert res.ok
    assert res.lyrics  # real lyrics from the template engine
    assert "[Verse]" in res.lyrics or "[Chorus]" in res.lyrics
    assert os.path.exists(res.audio_path)


def test_make_bed_no_model_fails_honestly(tmp_path):
    with pytest.raises(ACEModelUnavailable):
        make_bed("lagos", style="pop", context=None,
                 backend=ACEStepBackend(profile="workstation"),
                 workdir=str(tmp_path))


# ── fallback intact ──────────────────────────────────────────────────────────

def test_builtin_music_compose_untouched():
    """ACE-Step is additive: /music compose still works without it."""
    from nomorals.media.music import MusicCreator
    song = MusicCreator(None).compose("test", style="pop", with_midi=False,
                                      with_audio=False, with_score=False)
    assert song.title
    assert song.sections


def test_never_fake_audio(backend, tmp_path):
    """No path writes silence and calls it a bed."""
    res = backend.generate("[Verse]\nhey", "pop", workdir=str(tmp_path))
    with wave.open(res.audio_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    assert np.abs(pcm).max() > 100  # real signal, not silence
