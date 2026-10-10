"""Tests for nomorals.media.motion_studio — the CPU-only procedural video suite.

All fixtures are synthetic (generated PNG frames, numpy-synthesized WAV
audio), so the suite runs offline in seconds. Motion-studio renders are
REAL short clips (ffmpeg pipe), not mocks: every test asserts a valid
mp4 on disk with nonzero duration.

Neural backends are NOT tested here (no GPU in CI) — see
tests/test_videogen.py for the capability-gating mocks.
"""
from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from nomorals.media.motion_studio import MotionStudioError
from nomorals.media.motion_studio import kenburns as KB
from nomorals.media.motion_studio import visualizer as VIZ
from nomorals.media.motion_studio import typography as TYPO
from nomorals.media.motion_studio import montage as MON
from nomorals.media.motion_studio import grading as GR
from nomorals.media.motion_studio import studio as ST
from nomorals.media.motion_studio._core import (
    EASINGS,
    ease,
    probe_duration,
    profile_defaults,
    read_ledger,
)

TINY = (160, 120)
FPS = 8


@pytest.fixture()
def cover_png(tmp_path):
    p = tmp_path / "cover.png"
    img = Image.new("RGB", (320, 240), (30, 20, 60))
    d = ImageDraw.Draw(img)
    for i in range(6):
        d.ellipse([40 + i * 30, 60 + i * 20, 120 + i * 30, 140 + i * 20],
                  fill=(100 + i * 20, 50, 150))
    img.save(p)
    return str(p)


@pytest.fixture()
def song_wav(tmp_path):
    """4s synthetic track: 120bpm kick thumps + bassline."""
    sr = 22050
    t = np.arange(sr * 4) / sr
    sig = np.zeros_like(t)
    for b in range(8):
        s = int(b * 0.5 * sr)
        n = min(2000, len(t) - s)
        sig[s:s + n] += np.sin(2 * np.pi * 55 * t[:n]) * np.exp(-np.arange(n) / 400)
    sig += 0.25 * np.sin(2 * np.pi * 110 * t)
    sig = (sig / (np.abs(sig).max() + 1e-9) * 28000).astype(np.int16)
    p = tmp_path / "song.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(sig.tobytes())
    return str(p)


def _assert_valid_mp4(path: str, min_duration: float = 0.5):
    assert os.path.exists(path), f"missing output {path}"
    assert os.path.getsize(path) > 1000, f"suspiciously small {path}"
    dur = probe_duration(path)
    assert dur >= min_duration, f"duration {dur} < {min_duration} for {path}"


# -- core ------------------------------------------------------------------

def test_easings_bounded():
    for name, fn in EASINGS.items():
        assert fn(0.0) == pytest.approx(0.0), name
        assert fn(1.0) == pytest.approx(1.0), name
        assert 0.0 <= fn(0.5) <= 1.0, name
    assert ease("nope", 0.5) == ease("smooth", 0.5)  # unknown → default


def test_profile_defaults_have_all_profiles():
    for kind in ("termux", "laptop", "workstation"):
        d = profile_defaults(kind)
        assert d["size"] and d["fps"] and d["crf"]


# -- kenburns ---------------------------------------------------------------

def test_kenburns_renders(cover_png, tmp_path):
    out = str(tmp_path / "kb.mp4")
    p = KB.kenburns(cover_png, out, duration=1.0, move="zoom_in",
                    size=TINY, fps=FPS, grain=False, vignette=False)
    _assert_valid_mp4(p)


def test_kenburns_all_moves(cover_png, tmp_path):
    for mv in KB.MOVES:
        out = str(tmp_path / f"kb-{mv}.mp4")
        KB.kenburns(cover_png, out, duration=0.5, move=mv,
                    size=TINY, fps=FPS, grain=False, vignette=False)
        assert os.path.getsize(out) > 500, mv


def test_kenburns_bad_move(cover_png, tmp_path):
    with pytest.raises(MotionStudioError):
        KB.kenburns(cover_png, str(tmp_path / "x.mp4"), move="barrel_roll",
                    size=TINY, fps=FPS)


def test_kenburns_missing_image(tmp_path):
    with pytest.raises(MotionStudioError):
        KB.kenburns("/nope/missing.png", str(tmp_path / "x.mp4"),
                    size=TINY, fps=FPS)


def test_multilayer_drift(cover_png, tmp_path):
    out = str(tmp_path / "drift.mp4")
    p = KB.multilayer_drift(cover_png, out, duration=1.0,
                            size=TINY, fps=FPS)
    _assert_valid_mp4(p)


# -- visualizer --------------------------------------------------------------

def test_visualizer_bars(song_wav, tmp_path):
    out = str(tmp_path / "viz.mp4")
    p = VIZ.render_visualizer(song_wav, out, style="bars", palette="phonk",
                              size=TINY, fps=FPS)
    _assert_valid_mp4(p, min_duration=3.0)


def test_visualizer_all_styles(song_wav, tmp_path, cover_png):
    for style in VIZ.VISUALIZER_STYLES:
        out = str(tmp_path / f"viz-{style}.mp4")
        VIZ.render_visualizer(song_wav, out, style=style, palette="neon",
                              images=[cover_png], duration=1.0,
                              size=TINY, fps=FPS)
        assert os.path.getsize(out) > 500, style


def test_visualizer_preset_override(song_wav, tmp_path):
    out = str(tmp_path / "viz-preset.mp4")
    p = VIZ.render_visualizer(song_wav, out, preset="lofi",
                              duration=1.0, size=TINY, fps=FPS)
    _assert_valid_mp4(p, min_duration=0.8)


def test_visualizer_bad_inputs(song_wav, tmp_path):
    with pytest.raises(MotionStudioError):
        VIZ.render_visualizer(song_wav, str(tmp_path / "x.mp4"),
                              style="hologram", size=TINY, fps=FPS)
    with pytest.raises(MotionStudioError):
        VIZ.render_visualizer("/nope/missing.wav", str(tmp_path / "x.mp4"),
                              size=TINY, fps=FPS)
    with pytest.raises(MotionStudioError):
        VIZ.render_visualizer(song_wav, str(tmp_path / "x.mp4"),
                              preset="nope", size=TINY, fps=FPS)


def test_analyze_audio_shape(song_wav):
    a = VIZ.analyze_audio(song_wav, fps=FPS, duration=2.0)
    assert a.n_frames == 16
    assert a.bands.shape == (16, 28)
    assert a.energy.shape == (16,)
    assert 0.0 <= a.energy.max() <= 1.0 + 1e-6
    assert isinstance(a.beats, list)  # may be [] — never crashes


# -- typography ---------------------------------------------------------------

def test_to_words_timings():
    words = TYPO.to_words([("hello", 0.0, 0.5), ("world", 0.6, 1.0)])
    assert [(w.text, w.start) for w in words] == [("hello", 0.0), ("world", 0.6)]
    words2 = TYPO.to_words("one two three")
    assert len(words2) == 3 and words2[0].start < words2[1].start
    with pytest.raises(MotionStudioError):
        TYPO.to_words("   ")


def test_render_lyrics_timed(tmp_path):
    out = str(tmp_path / "lyric.mp4")
    words = [("we", 0.0, 0.4), ("ride", 0.5, 0.9), ("at", 1.0, 1.2),
             ("midnight", 1.3, 1.9)]
    p = TYPO.render_lyrics(words, out, duration=2.5, preset="neon_pop",
                           size=TINY, fps=FPS)
    _assert_valid_mp4(p, min_duration=2.0)


def test_render_lyrics_all_presets(tmp_path):
    for preset in TYPO.STYLE_PRESETS:
        out = str(tmp_path / f"lyric-{preset}.mp4")
        TYPO.render_lyrics("one two three four", out, duration=1.5,
                           preset=preset, size=TINY, fps=FPS)
        assert os.path.getsize(out) > 500, preset


def test_render_quote_card(tmp_path):
    out = str(tmp_path / "quote.mp4")
    p = TYPO.render_quote_card("THE DROP IS COMING", out, duration=1.5,
                               size=TINY, fps=FPS)
    _assert_valid_mp4(p, min_duration=1.0)


# -- grading ------------------------------------------------------------------

def test_grades(cover_png, tmp_path):
    src = KB.kenburns(cover_png, str(tmp_path / "g-src.mp4"), duration=1.0,
                      size=TINY, fps=FPS, grain=False, vignette=False)
    for name in GR.GRADE_PRESETS:
        out = str(tmp_path / f"graded-{name}.mp4")
        GR.grade(src, out, preset=name)
        assert os.path.getsize(out) > 500, name
    with pytest.raises(MotionStudioError):
        GR.grade(src, str(tmp_path / "x.mp4"), preset="nope")
    with pytest.raises(MotionStudioError):
        GR.grade("/nope/missing.mp4", str(tmp_path / "x.mp4"))


def test_export_formats(cover_png, tmp_path):
    src = KB.kenburns(cover_png, str(tmp_path / "e-src.mp4"), duration=1.0,
                      size=TINY, fps=FPS, grain=False, vignette=False)
    for fmt in ("9:16", "16:9", "1:1"):
        out = str(tmp_path / f"exp-{fmt.replace(':', 'x')}.mp4")
        p = GR.export(src, out, format=fmt)
        _assert_valid_mp4(p, min_duration=0.5)


# -- montage -------------------------------------------------------------------

def test_assemble_mixed_timeline(cover_png, tmp_path, song_wav):
    clip = KB.kenburns(cover_png, str(tmp_path / "m-clip.mp4"), duration=1.0,
                       size=TINY, fps=FPS, grain=False, vignette=False)
    out = str(tmp_path / "montage.mp4")
    p = MON.assemble([
        {"kind": "video", "src": clip, "duration": 1.0, "transition": "cut"},
        {"kind": "text", "text": "THE DROP", "duration": 1.0,
         "transition": "crossfade"},
        {"kind": "image", "src": cover_png, "duration": 1.0, "move": "pan_right"},
    ], out, audio=song_wav, size=TINY, fps=FPS)
    # 3x1.0s segments minus two 0.6s crossfades ≈ 1.75s
    _assert_valid_mp4(p, min_duration=1.5)


def test_assemble_empty():
    with pytest.raises(MotionStudioError):
        MON.assemble([])


# -- studio --------------------------------------------------------------------

def test_studio_slideshow(cover_png, tmp_path, song_wav):
    p = ST.make_slideshow([cover_png, cover_png],
                          out=str(tmp_path / "show.mp4"),
                          audio=song_wav, per_image=1.0,
                          grade_preset="", format="",
                          size=TINY, fps=FPS)
    # 2x1.0s images minus one 0.6s crossfade ≈ 1.4s
    _assert_valid_mp4(p, min_duration=1.2)


def test_studio_lyric_video(tmp_path, song_wav):
    p = ST.make_lyric_video(song_wav, "one two three four five six",
                            out=str(tmp_path / "s-lyric.mp4"),
                            preset="minimal", grade_preset="", format="",
                            size=TINY, fps=FPS)
    _assert_valid_mp4(p, min_duration=3.0)


def test_studio_visualizer(tmp_path, song_wav, cover_png):
    p = ST.make_music_visualizer(song_wav, out=str(tmp_path / "s-viz.mp4"),
                                 images=[cover_png], preset="minimal",
                                 duration=2.0, grade_preset="", format="",
                                 size=TINY, fps=FPS)
    _assert_valid_mp4(p, min_duration=1.5)


def test_studio_trailer(cover_png, tmp_path):
    clip = KB.kenburns(cover_png, str(tmp_path / "tr-clip.mp4"), duration=2.0,
                       size=TINY, fps=FPS, grain=False, vignette=False)
    p = ST.make_trailer([clip, clip], out=str(tmp_path / "trailer.mp4"),
                        title="MIDNIGHT RUN", clip_len=1.0,
                        grade_preset="", format="",
                        size=TINY, fps=FPS)
    _assert_valid_mp4(p, min_duration=2.0)


def test_studio_errors(tmp_path):
    with pytest.raises(MotionStudioError):
        ST.make_slideshow([])
    with pytest.raises(MotionStudioError):
        ST.make_lyric_video("/nope/missing.mp3", "hello")
    with pytest.raises(MotionStudioError):
        ST.make_trailer([])


def test_ledger_records_something(cover_png, tmp_path):
    before = len(read_ledger())
    KB.kenburns(cover_png, str(tmp_path / "led.mp4"), duration=0.5,
                size=TINY, fps=FPS, grain=False, vignette=False)
    assert len(read_ledger()) >= before + 1
