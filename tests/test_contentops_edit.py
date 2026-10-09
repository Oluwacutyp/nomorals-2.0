"""Stream 1 — edit engine tests. Synthetic fixtures only.

- Audio: numpy-generated kick-burst grids (known BPM) and sine beds.
- Video: solid / two-tone PNGs via PIL → tiny mp4s via ffmpeg.
- No network, no real media, no ripped content.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nomorals.media.contentops import (
    AudioSpec,
    CaptionSpec,
    Clip,
    EditSpec,
    Effect,
    apply_effect,
    assemble_cuts,
    beat_sync_words,
    build_captions,
    burn_beat_captions,
    detect_beats,
    estimate_word_timings,
    make_music_bed,
    mix_audio,
    render,
    render_report,
    render_vertical,
    voice_segments,
    words_to_beat_ass,
)
from nomorals.media.contentops.beats import _beats_numpy, detect_beats_full
from nomorals.media.contentops.edit import EFFECTS, _effect_filter
from nomorals.media_edit.captions import Word
from nomorals.media_edit.videos import MediaEditError, ffmpeg_path, video_probe

SR = 22050


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _grid_wav(path: Path, bpm: float = 120.0, seconds: float = 8.0) -> float:
    """Kick-like bursts on an exact grid. Returns the period."""
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    per = 60.0 / bpm
    n = 0
    while n * per < seconds:
        c = int(n * per * SR)
        L = int(0.09 * SR)
        tt = np.arange(L) / SR
        burst = np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 45)
        y[c:c + L] += burst[:max(0, min(L, len(y) - c))]
        n += 1
    y += 0.002 * np.random.RandomState(7).randn(len(y))
    y = np.clip(y, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    return per


def _sine_wav(path: Path, freq: float = 440.0, seconds: float = 3.0,
              decay: bool = False) -> None:
    t = np.arange(int(seconds * SR)) / SR
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    if decay:
        y *= np.exp(-t * 1.2)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


def _vo_wav(path: Path, seconds: float = 3.0) -> None:
    """Voice-ish: two speech bursts with silence between/around."""
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for s0, s1 in ((0.3, 1.2), (1.7, 2.7)):
        m = (t >= s0) & (t < s1)
        y[m] = 0.4 * np.sin(2 * np.pi * 180 * t[m]) * np.sin(
            2 * np.pi * 7 * t[m])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


def _png(path: Path, left=(255, 0, 0), right=(0, 0, 255),
         size=(320, 240)) -> None:
    from PIL import Image
    im = Image.new("RGB", size)
    px = im.load()
    for x in range(size[0]):
        for y in range(size[1]):
            px[x, y] = left if x < size[0] // 2 else right
    im.save(path)


def _clip(path: Path, png: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-y", "-v", "error",
         "-loop", "1", "-framerate", "30", "-t", str(seconds),
         "-i", str(png),
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=120)
    return path


def _scene_times(video: Path, thresh: float = 0.35) -> list[float]:
    """Timestamps where the picture actually cuts (scene detection)."""
    proc = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", str(video),
         "-vf", f"select='gt(scene,{thresh})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=120)
    times = [float(m.group(1)) for m in
             re.finditer(r"pts_time:([0-9.]+)", proc.stderr or "")]
    return [t for t in times if t > 0.05]


@pytest.fixture()
def workdir():
    with tempfile.TemporaryDirectory(prefix="edit-test-") as d:
        yield Path(d)


@pytest.fixture()
def two_clips(workdir):
    red = workdir / "red.png"
    blu = workdir / "blu.png"
    _png(red, left=(255, 0, 0), right=(200, 0, 0))
    _png(blu, left=(0, 0, 255), right=(0, 0, 200))
    c1 = _clip(workdir / "c1.mp4", red)
    c2 = _clip(workdir / "c2.mp4", blu)
    return c1, c2


# ---------------------------------------------------------------------------
# beat detection
# ---------------------------------------------------------------------------

class TestDetectBeats:
    def test_finds_synthetic_grid(self, workdir):
        wav = workdir / "grid.wav"
        per = _grid_wav(wav, bpm=120.0, seconds=8.0)
        beats = detect_beats(wav)
        assert len(beats) == 16, beats
        expected = [i * per for i in range(16)]
        errs = [min(abs(b - e) for b in beats) for e in expected]
        assert max(errs) < 0.06, errs

    def test_other_tempos(self, workdir):
        for bpm, n_exp in ((90.0, 12), (140.0, 19), (175.0, 24)):
            wav = workdir / f"grid{bpm}.wav"
            per = _grid_wav(wav, bpm=bpm, seconds=8.0)
            beats = detect_beats(wav)
            expected = [i * per for i in range(n_exp)]
            errs = [min(abs(b - e) for b in beats) for e in expected]
            assert len(beats) == n_exp, (bpm, beats)
            assert max(errs) < 0.06, (bpm, errs)

    def test_silence_returns_empty(self, workdir):
        wav = workdir / "sil.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(np.zeros(SR * 3, dtype=np.int16).tobytes())
        assert detect_beats(wav) == []

    def test_missing_file_raises_honest_error(self):
        with pytest.raises(MediaEditError):
            detect_beats("/tmp/does-not-exist-xyz.wav")

    def test_numpy_engine_directly(self, workdir):
        wav = workdir / "grid.wav"
        _grid_wav(wav, bpm=100.0, seconds=4.0)
        from nomorals.media.contentops.beats import _decode_mono
        y = _decode_mono(wav)
        bpm, beats = _beats_numpy(y, SR)
        assert abs(bpm - 100.0) < 2.0
        assert len(beats) == 7  # 0..3.6 at 100bpm over 4s

    def test_librosa_path_uses_beat_track(self, workdir):
        """Fake librosa module: detect_beats must call beat_track and
        convert frames → seconds (verified librosa 0.10 API shape)."""
        import types
        wav = workdir / "grid.wav"
        _grid_wav(wav, bpm=120.0, seconds=2.0)
        fake = types.ModuleType("librosa")
        fake_beat = types.ModuleType("librosa.beat")
        fake_beat.beat_track = lambda y=None, sr=22050: (
            120.0, np.array([10, 20, 30]))
        fake.beat = fake_beat
        fake.frames_to_time = lambda frames, sr=22050: (
            np.asarray(frames) * 512.0 / sr)
        with patch.dict(sys.modules, {"librosa": fake}):
            beats = detect_beats(wav, backend="librosa")
        # detect_beats rounds to 4 decimals
        assert beats == pytest.approx(
            [round(10 * 512 / SR, 4), round(20 * 512 / SR, 4),
             round(30 * 512 / SR, 4)])

    def test_librosa_failure_falls_back_to_numpy(self, workdir):
        import types
        wav = workdir / "grid.wav"
        _grid_wav(wav, bpm=120.0, seconds=4.0)
        fake = types.ModuleType("librosa")
        fake_beat = types.ModuleType("librosa.beat")

        def boom(*a, **k):
            raise RuntimeError("fake librosa exploded")

        fake_beat.beat_track = boom
        fake.beat = fake_beat
        fake.frames_to_time = boom
        with patch.dict(sys.modules, {"librosa": fake}):
            beats = detect_beats(wav, backend="auto")
        assert len(beats) == 8  # numpy fallback still finds the grid

    def test_never_raises_without_librosa(self, workdir):
        """librosa unimportable → honest numpy degrade, no exception."""
        wav = workdir / "grid.wav"
        _grid_wav(wav, bpm=120.0, seconds=2.0)
        with patch.dict(sys.modules, {"librosa": None}):
            with patch("importlib.util.find_spec", return_value=None):
                info = detect_beats_full(wav, backend="auto")
        assert info.backend == "numpy"
        assert len(info.beats) == 4


# ---------------------------------------------------------------------------
# assemble_cuts
# ---------------------------------------------------------------------------

class TestAssembleCuts:
    def test_cuts_land_on_beats(self, workdir, two_clips):
        c1, c2 = two_clips
        beats = [0.0, 0.5, 1.0, 1.5, 2.0]
        out = assemble_cuts(
            [Clip(str(c1)), Clip(str(c2))], beats,
            out=workdir / "cut.mp4", profile="termux")
        info = video_probe(out["output"])
        assert info["width"] == 1080 and info["height"] == 1920
        assert abs(info["fps"] - 30) < 1
        assert abs(info["duration"] - 2.0) < 0.15
        cuts = _scene_times(Path(out["output"]))
        assert len(cuts) == 3, cuts
        for got, want in zip(cuts, [0.5, 1.0, 1.5]):
            assert abs(got - want) < 0.1, (cuts, beats)

    def test_sequential_without_beats(self, workdir, two_clips):
        c1, c2 = two_clips
        out = assemble_cuts([Clip(str(c1)), Clip(str(c2))], None,
                            out=workdir / "seq.mp4", profile="termux")
        info = video_probe(out["output"])
        assert abs(info["duration"] - 4.0) < 0.2
        assert out["segments"] == 2

    def test_effects_and_flash_transition(self, workdir, two_clips):
        c1, c2 = two_clips
        clips = [Clip(str(c1), effects=(Effect("shake"),)),
                 Clip(str(c2), effects=(Effect("punch_zoom"),))]
        out = assemble_cuts(clips, [0.0, 1.0, 2.0],
                            out=workdir / "fx.mp4", transition="flash",
                            seed=11, profile="termux")
        assert Path(out["output"]).stat().st_size > 0
        assert abs(video_probe(out["output"])["duration"] - 2.0) < 0.15

    def test_deterministic_seed(self, workdir, two_clips):
        c1, _ = two_clips
        a = assemble_cuts([Clip(str(c1), effects=(Effect("shake"),))],
                          [0.0, 1.0], out=workdir / "a.mp4",
                          profile="termux", seed=99)["output"]
        b = assemble_cuts([Clip(str(c1), effects=(Effect("shake"),))],
                          [0.0, 1.0], out=workdir / "b.mp4",
                          profile="termux", seed=99)["output"]
        ha = subprocess.run([ffmpeg_path(), "-v", "error", "-i", a,
                             "-f", "md5", "-"],
                            capture_output=True, text=True).stdout
        hb = subprocess.run([ffmpeg_path(), "-v", "error", "-i", b,
                             "-f", "md5", "-"],
                            capture_output=True, text=True).stdout
        assert ha == hb and ha.strip()

    def test_bad_inputs_raise(self, workdir):
        with pytest.raises(MediaEditError):
            assemble_cuts([], [0.0, 1.0])
        with pytest.raises(MediaEditError):
            assemble_cuts([Clip("/tmp/nope.mp4")], [0.0, 1.0])
        with pytest.raises(MediaEditError):
            assemble_cuts([Clip("/tmp/nope.mp4")], [0.5])

    def test_keep_audio(self, workdir):
        """keep_audio=True cuts + speed-matches the audio twin."""
        png = workdir / "a.png"
        _png(png)
        wav = workdir / "tone.wav"
        _sine_wav(wav, 440.0, 2.0)
        clip = workdir / "av.mp4"
        subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-y", "-v", "error",
             "-loop", "1", "-framerate", "30", "-t", "2", "-i", str(png),
             "-i", str(wav),
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip)],
            check=True, timeout=120)
        out = assemble_cuts([Clip(str(clip), speed=2.0)], [0.0, 0.5, 1.0],
                            out=workdir / "avc.mp4", keep_audio=True,
                            profile="termux")
        info = video_probe(out["output"])
        assert any(s.get("type") == "audio" for s in info["streams"])
        assert abs(info["duration"] - 1.0) < 0.2  # 2s of footage @2x

    def test_beat_pulse_in_assemble(self, workdir, two_clips):
        c1, c2 = two_clips
        clips = [Clip(str(c1), effects=(Effect("beat_pulse"),)),
                 Clip(str(c2), effects=(Effect("beat_pulse",
                                               params={"strength": 0.15}),))]
        out = assemble_cuts(clips, [0.0, 0.5, 1.0, 1.5, 2.0],
                            out=workdir / "pulse.mp4", profile="termux")
        assert abs(video_probe(out["output"])["duration"] - 2.0) < 0.15


# ---------------------------------------------------------------------------
# apply_effect
# ---------------------------------------------------------------------------

def _frame_md5(video: Path) -> str:
    return subprocess.run(
        [ffmpeg_path(), "-v", "error", "-i", str(video), "-f", "md5", "-"],
        capture_output=True, text=True, timeout=60).stdout.strip()


class TestApplyEffect:
    @pytest.mark.parametrize("effect", sorted(EFFECTS))
    def test_each_effect_runs(self, workdir, two_clips, effect):
        c1, _ = two_clips
        params: dict = {}
        kw: dict = {}
        if effect == "beat_pulse":
            params["beats"] = [0.5]
        if effect == "flash":
            params["time"] = 0.5
        if effect == "ken_burns":
            png = workdir / "still.png"
            _png(png)
            out = apply_effect(png, effect, out=workdir / f"{effect}.mp4",
                               still_duration=1.0, profile="termux",
                               **params)["output"]
        else:
            out = apply_effect(c1, effect, out=workdir / f"{effect}.mp4",
                               profile="termux", **params, **kw)["output"]
        assert Path(out).stat().st_size > 0
        assert video_probe(out)["width"] > 0

    def test_effects_change_pixels(self, workdir, two_clips):
        """shake/rgb_split/grain must actually alter the picture."""
        c1, _ = two_clips
        before = _frame_md5(c1)
        for effect in ("shake", "rgb_split", "grain"):
            out = apply_effect(c1, effect, out=workdir / f"chg-{effect}.mp4",
                               profile="termux")["output"]
            assert _frame_md5(Path(out)) != before, effect

    def test_velocity_ramp_changes_duration(self, workdir, two_clips):
        c1, _ = two_clips
        out = apply_effect(c1, "velocity_ramp",
                           out=workdir / "vel.mp4", profile="termux",
                           speed=2.0)["output"]
        assert abs(video_probe(out)["duration"] - 1.0) < 0.15

    def test_windowed_effect(self, workdir, two_clips):
        c1, _ = two_clips
        out = apply_effect(c1, "shake", out=workdir / "win.mp4",
                           at=(0.2, 0.6), profile="termux")["output"]
        assert abs(video_probe(out)["duration"] - 2.0) < 0.15

    def test_unknown_effect_raises(self, workdir, two_clips):
        c1, _ = two_clips
        with pytest.raises(MediaEditError):
            apply_effect(c1, "explode", out=workdir / "x.mp4")

    def test_render_vertical(self, workdir, two_clips):
        c1, _ = two_clips
        out = render_vertical(c1, out=workdir / "vert.mp4",
                              profile="termux")["output"]
        info = video_probe(out)
        assert (info["width"], info["height"]) == (1080, 1920)
        assert abs(info["fps"] - 30) < 1


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------

def _words():
    return [Word(0.10, 0.30, "they"), Word(0.35, 0.55, "counted"),
            Word(0.60, 0.80, "him"), Word(0.85, 1.05, "out")]


class TestCaptions:
    def test_beat_sync_words(self):
        synced = beat_sync_words(_words(), [0.0, 0.5, 1.0])
        starts = [w.start for w in synced]
        assert starts[0] == pytest.approx(0.0)
        assert starts[1] == pytest.approx(0.51)
        # monotonic, non-overlapping
        for a, b in zip(synced, synced[1:]):
            assert b.start >= a.end

    def test_beat_sync_no_beats_passthrough(self):
        assert [w.start for w in beat_sync_words(_words(), [])] == \
               [w.start for w in _words()]

    def test_words_to_beat_ass(self):
        ass = words_to_beat_ass(_words(), [0.0, 0.5, 1.0], style="karaoke")
        assert "PlayResX: 1080" in ass
        assert "PlayResY: 1920" in ass
        assert r"{\kf" in ass  # word-highlight sweep tags
        with pytest.raises(MediaEditError):
            words_to_beat_ass(_words(), style="nope")

    def test_estimate_word_timings(self):
        words = estimate_word_timings("one two three", wpm=60.0)
        assert [w.text for w in words] == ["one", "two", "three"]
        assert words[0].start == pytest.approx(0.0)
        assert words[1].start == pytest.approx(1.0)
        snapped = estimate_word_timings("one two three", wpm=60.0,
                                        beats=[0.0, 1.0, 2.0])
        assert snapped[1].start == pytest.approx(1.0)
        # monotonicity nudge: a beat-snapped start can't overlap the
        # previous word's end
        tight = estimate_word_timings("one two", wpm=60.0, beats=[0.0, 0.5])
        assert tight[1].start == pytest.approx(0.93)

    def test_burn_beat_captions(self, workdir, two_clips):
        c1, _ = two_clips
        vert = render_vertical(c1, out=workdir / "cap.mp4",
                               profile="termux")["output"]
        out = burn_beat_captions(vert, _words(), [0.0, 0.5, 1.0],
                                 style="karaoke",
                                 out_dir=workdir)["output"]
        assert Path(out).stat().st_size > 0
        info = video_probe(out)
        assert (info["width"], info["height"]) == (1080, 1920)
        # sidecars kept (decoupled-transcript pattern)
        assert Path(vert).with_suffix(".srt").exists()

    def test_build_captions_srt(self, workdir):
        srt = build_captions([("hello world", 0.0, 1.2),
                              ("second line", 1.3, 2.0)],
                             workdir / "c.srt")
        text = Path(srt).read_text()
        assert "00:00:00,000 --> 00:00:01,200" in text
        assert "hello world" in text


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

class TestAudio:
    def test_mix_sidechain(self, workdir):
        music = workdir / "music.wav"
        vo = workdir / "vo.wav"
        _sine_wav(music, 220.0, 3.0)
        _vo_wav(vo, 3.0)
        out = mix_audio(music, vo, out=workdir / "mix.m4a",
                        ducking="sidechain")["output"]
        info = video_probe(out)
        assert any(s.get("type") == "audio" for s in info["streams"])
        assert abs(info["duration"] - 3.0) < 0.3

    def test_mix_volume_ducking(self, workdir):
        music = workdir / "music.wav"
        vo = workdir / "vo.wav"
        _sine_wav(music, 220.0, 3.0)
        _vo_wav(vo, 3.0)
        out = mix_audio(music, vo, out=workdir / "mixv.m4a",
                        ducking="volume")["output"]
        assert Path(out).stat().st_size > 0

    def test_mix_music_only(self, workdir):
        music = workdir / "music.wav"
        _sine_wav(music, 220.0, 2.0)
        out = mix_audio(music, out=workdir / "m.m4a")["output"]
        assert abs(video_probe(out)["duration"] - 2.0) < 0.3

    def test_voice_segments(self, workdir):
        vo = workdir / "vo.wav"
        _vo_wav(vo, 3.0)
        segs = voice_segments(vo)
        assert len(segs) == 2, segs
        assert segs[0][0] < 0.5 and segs[1][1] > 2.5

    def test_make_music_bed(self, workdir):
        bed = make_music_bed(4.0, workdir / "bed.wav", seed=7)
        info = video_probe(bed)
        assert abs(info["duration"] - 4.0) < 0.2

    def test_bad_ducking_raises(self, workdir):
        music = workdir / "music.wav"
        _sine_wav(music)
        with pytest.raises(MediaEditError):
            mix_audio(music, out=workdir / "x.m4a", ducking="squash")


# ---------------------------------------------------------------------------
# EditSpec round-trip + full render
# ---------------------------------------------------------------------------

def _spec(workdir, two_clips) -> EditSpec:
    c1, c2 = two_clips
    music = workdir / "music.wav"
    vo = workdir / "vo.wav"
    _sine_wav(music, 110.0, 4.0)
    _vo_wav(vo, 2.5)
    return EditSpec(
        clips=[Clip(str(c1), effects=(Effect("shake"),)),
               Clip(str(c2), effects=(Effect("punch_zoom"),))],
        beats=[0.0, 0.5, 1.0, 1.5, 2.0],
        captions=CaptionSpec(text="they counted him out", style="karaoke"),
        audio=AudioSpec(music=str(music), voiceover=str(vo),
                        ducking="sidechain"),
        fps=30, seed=7, transition="flash", profile="termux",
        out=str(workdir / "final.mp4"),
    )


class TestEditSpec:
    def test_round_trip(self, workdir, two_clips):
        spec = _spec(workdir, two_clips)
        data = json.loads(json.dumps(spec.to_dict()))
        back = EditSpec.from_dict(data)
        assert back.to_dict() == spec.to_dict()
        assert back.clips[0].effects[0].name == "shake"
        assert back.audio.ducking == "sidechain"
        assert back.captions.style == "karaoke"

    def test_pipeline_payload_shape(self, workdir):
        """The pipeline's loose payload (scenes/beat_times/output/...) is
        accepted, never rejected."""
        spec = EditSpec(
            scenes=[{"image": "/tmp/a.png", "duration": 2.0,
                     "effect": "kenburns", "zoom_direction": "in"}],
            audio="/tmp/vo.wav", captions="hello", music="",
            beat_times=[0.0, 1.0, 2.0], output="/tmp/draft.mp4",
            width=1080, height=1920, fps=30)
        assert spec.clips[0].effects[0].name == "ken_burns"
        assert spec.beats == [0.0, 1.0, 2.0]
        assert spec.out == "/tmp/draft.mp4"

    def test_render_returns_path(self, workdir, two_clips):
        spec = _spec(workdir, two_clips)
        out = render(spec)
        assert isinstance(out, str)
        assert Path(out).stat().st_size > 0
        info = video_probe(out)
        assert (info["width"], info["height"]) == (1080, 1920)
        assert abs(info["fps"] - 30) < 1
        assert abs(info["duration"] - 2.0) < 0.3
        assert any(s.get("type") == "audio" for s in info["streams"])

    def test_render_report(self, workdir, two_clips):
        spec = _spec(workdir, two_clips)
        rep = render_report(spec)
        assert rep["n_beats"] == 5
        assert rep["profile"] == "termux"
        assert rep["captions"]["synced_to_beats"] is True
        assert rep["audio"]["ducking"] == "sidechain"

    def test_render_with_still_scenes(self, workdir):
        """Pipeline shape: image scenes → stills materialised, kenburns."""
        png = workdir / "scene.png"
        _png(png)
        spec = EditSpec(
            scenes=[{"image": str(png), "duration": 1.5,
                     "effect": "kenburns", "zoom_direction": "in"},
                    {"image": str(png), "duration": 1.5,
                     "effect": "kenburns", "zoom_direction": "out"}],
            beat_times=[0.0, 1.5, 3.0], output=str(workdir / "still.mp4"),
            profile="termux")
        out = render(spec)
        info = video_probe(out)
        assert abs(info["duration"] - 3.0) < 0.3
        assert (info["width"], info["height"]) == (1080, 1920)

    def test_render_beat_audio(self, workdir, two_clips):
        c1, c2 = two_clips
        grid = workdir / "grid.wav"
        _grid_wav(grid, bpm=120.0, seconds=4.0)
        spec = EditSpec(clips=[Clip(str(c1)), Clip(str(c2))],
                        beat_audio=str(grid), profile="termux",
                        out=str(workdir / "ba.mp4"))
        rep = render_report(spec)
        assert rep["n_beats"] == 8, rep["n_beats"]
        assert abs(rep["duration"] - 3.5) < 0.3  # 8 beats → 7 intervals

    def test_no_beats_no_beat_source_sequential(self, workdir, two_clips):
        c1, c2 = two_clips
        spec = EditSpec(clips=[Clip(str(c1)), Clip(str(c2))],
                        profile="termux",
                        out=str(workdir / "seqr.mp4"))
        rep = render_report(spec)
        assert rep["n_beats"] == 0
        assert abs(rep["duration"] - 4.0) < 0.3

    def test_empty_spec_raises(self):
        with pytest.raises(MediaEditError):
            render(EditSpec())
