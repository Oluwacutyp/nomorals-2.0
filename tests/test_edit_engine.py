"""Tests for the style-agnostic edit engine + style presets.

Proves: timeline math, spec JSON round-trips, every effect and every
transition renders a real clip via ffmpeg, presets produce valid
engine specs, and the legacy shim keeps the old surface working.
"""
from __future__ import annotations

import json
import math
import os
import struct
import wave
from pathlib import Path

import pytest

from nomorals.media.edit_engine import (
    AudioLayer,
    AudioMix,
    Clip,
    EditSpec,
    Effect,
    TextLayer,
    Transition,
    Timeline,
    Track,
    EFFECTS,
    apply_effect,
    assemble_segments,
    build_captions,
    estimate_word_timings,
    list_transitions,
    make_music_bed,
    mix_layers,
    plan_from_cut_points,
    render,
    render_vertical,
    validate_transition,
)
from nomorals.media.edit_engine.text import layer_to_ass
from nomorals.media_edit.videos import MediaEditError, video_probe

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "eng"
    d.mkdir()
    return d


def _mp4(workdir, name="c.mp4", color="red", seconds=2.0, w=320, h=480):
    from nomorals.media_edit.videos import run_ffmpeg
    p = workdir / name
    run_ffmpeg(["-f", "lavfi", "-i",
                f"color=c={color}:s={w}x{h}:d={seconds}:r=30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
               duration=seconds)
    return p


def _wav(workdir, name="a.wav", freq=440.0, seconds=2.0):
    p = workdir / name
    n = int(44100 * seconds)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"".join(
            struct.pack("<h", int(12000 * math.sin(
                2 * math.pi * freq * i / 44100)))
            for i in range(n)))
    return p


# ---------------------------------------------------------------------------
# timeline math
# ---------------------------------------------------------------------------

class TestTimelineMath:
    def test_clip_play_duration(self):
        c = Clip(path="a.mp4", start=1.0, end=3.0, speed=2.0)
        assert c.play_duration() == pytest.approx(1.0)

    def test_clip_explicit_duration_wins(self):
        c = Clip(path="a.mp4", start=0.0, end=10.0, duration=2.5)
        assert c.play_duration() == pytest.approx(2.5)

    def test_clip_bad_window_raises(self):
        with pytest.raises(ValueError):
            Clip(path="a.mp4", start=5.0, end=2.0)
        with pytest.raises(ValueError):
            Clip(path="a.mp4", speed=0)

    def test_effect_window_validated(self):
        with pytest.raises(ValueError):
            Effect(name="shake", at=(2.0, 1.0))

    def test_effect_alias_canonicalized(self):
        assert Effect(name="kenburns").name == "ken_burns"
        assert Effect(name="Glitch").name == "rgb_split"

    def test_effect_round_trip(self):
        e = Effect(name="shake", params={"amplitude": 9.0}, at=(0.5, 1.5))
        assert Effect.from_dict(json.loads(json.dumps(e.to_dict()))) == e

    def test_clip_round_trip(self):
        c = Clip(path="a.mp4", start=0.5, end=4.0, speed=1.5,
                 effects=(Effect("grain", {"strength": 5.0}),))
        assert Clip.from_dict(json.loads(json.dumps(c.to_dict()))) == c

    def test_transition_validation(self):
        assert validate_transition("hardcut") == "cut"
        assert validate_transition("whip") == "whip_pan"
        with pytest.raises(MediaEditError):
            validate_transition("explode")
        assert set(list_transitions()) >= {
            "cut", "fade", "dissolve", "flash", "dip",
            "whip_pan", "glitch_cut"}

    def test_transition_round_trip(self):
        t = Transition(kind="whip_pan", duration=0.4,
                       params={"direction": "up"})
        assert Transition.from_dict(
            json.loads(json.dumps(t.to_dict()))) == t

    def test_timeline_duration_sums_clips(self):
        tl = Timeline(video=Track(clips=[
            Clip(path="a.mp4", start=0.0, end=2.0),
            Clip(path="b.mp4", start=0.0, end=3.0, speed=2.0)]))
        assert tl.duration == pytest.approx(3.5)

    def test_plan_from_cut_points(self):
        plan = plan_from_cut_points(2, [0.0, 1.0, 2.0, 3.0], fps=30)
        assert plan == [(0, 1.0), (1, 1.0), (0, 1.0)]
        with pytest.raises(MediaEditError):
            plan_from_cut_points(2, [1.0], fps=30)


# ---------------------------------------------------------------------------
# spec serialization
# ---------------------------------------------------------------------------

class TestSpecSerialization:
    def test_full_round_trip(self):
        s = EditSpec(
            clips=[Clip(path="a.mp4",
                        effects=(Effect("shake", {"amplitude": 9.0}),))],
            cut_points=[0.0, 1.0, 2.0],
            transition=[Transition(kind="dissolve", duration=0.4), "cut"],
            text_layers=[TextLayer(text="hi", position="top")],
            audio=AudioMix(layers=[AudioLayer(path="m.mp3", loop=True)],
                           ducking="none"),
            width=640, height=960, fps=30, seed=11)
        back = EditSpec.from_dict(json.loads(json.dumps(s.to_dict())))
        assert back.to_dict() == s.to_dict()

    def test_beats_alias(self):
        s = EditSpec(clips=[Clip(path="a.mp4")], beats=[0.0, 1.0])
        assert s.cut_points == [0.0, 1.0]

    def test_scenes_alias(self):
        s = EditSpec(scenes=[{"image": "x.png", "duration": 2.0,
                              "effect": "kenburns"}])
        assert s.clips[0].effects[0].name == "ken_burns"

    def test_single_mapping_arg(self):
        s = EditSpec({"clips": [{"path": "a.mp4"}], "fps": 24})
        assert s.fps == 24 and len(s.clips) == 1


# ---------------------------------------------------------------------------
# effects — every one renders a real clip
# ---------------------------------------------------------------------------

class TestEffectsRender:
    @NEEDS_FFMPEG
    @pytest.mark.parametrize("effect", sorted(EFFECTS))
    def test_each_effect_renders(self, workdir, effect):
        src = _mp4(workdir, f"fx-{effect}.mp4")
        params: dict = {}
        if effect == "beat_pulse":
            params = {"beats": [0.0, 0.5, 1.0]}
        out = apply_effect(src, effect,
                           out=workdir / f"{effect}.out.mp4",
                           seed=5, **params)["output"]
        info = video_probe(out)
        assert info["width"] == 320 and info["duration"] > 1.5

    @NEEDS_FFMPEG
    def test_velocity_ramp_renders(self, workdir):
        src = _mp4(workdir, seconds=4.0)
        out = apply_effect(src, "velocity_ramp",
                           out=workdir / "vr.mp4",
                           speed=2.0, at=(1.0, 3.0))["output"]
        info = video_probe(out)
        assert abs(info["duration"] - 3.0) < 0.3  # 4s → 1+1+1

    @NEEDS_FFMPEG
    def test_rgb_split_animated_renders(self, workdir):
        # animated rgbashift expressions are rejected by this ffmpeg
        # build — the engine discretizes instead (must still render).
        src = _mp4(workdir)
        out = apply_effect(src, "rgb_split",
                           out=workdir / "rgbani.mp4",
                           animate=True, shift=12.0,
                           period=0.4)["output"]
        assert video_probe(out)["duration"] > 1.5

    @NEEDS_FFMPEG
    def test_color_grade_presets(self, workdir):
        from nomorals.media.edit_engine.effects import _GRADES
        src = _mp4(workdir)
        for grade in _GRADES:
            out = apply_effect(src, "color_grade",
                               out=workdir / f"g-{grade}.mp4",
                               grade=grade)["output"]
            assert Path(out).stat().st_size > 0
        with pytest.raises(MediaEditError):
            apply_effect(src, "color_grade",
                         out=workdir / "g-bad.mp4", grade="nope")

    @NEEDS_FFMPEG
    def test_windowed_effect(self, workdir):
        src = _mp4(workdir, seconds=4.0)
        out = apply_effect(src, "shake",
                           out=workdir / "win.mp4",
                           at=(1.0, 2.0))["output"]
        assert abs(video_probe(out)["duration"] - 4.0) < 0.3

    def test_unknown_effect_raises(self, workdir):
        src = _mp4(workdir)
        with pytest.raises(MediaEditError):
            apply_effect(src, "explode", out=workdir / "x.mp4")


# ---------------------------------------------------------------------------
# transitions — every one renders
# ---------------------------------------------------------------------------

class TestTransitionsRender:
    @NEEDS_FFMPEG
    @pytest.mark.parametrize("kind", list_transitions())
    def test_each_transition_renders(self, workdir, kind):
        c0 = _mp4(workdir, "t0.mp4", color="red")
        c1 = _mp4(workdir, "t1.mp4", color="blue")
        tr = kind
        if kind == "whip_pan":
            tr = Transition(kind="whip_pan", duration=0.4,
                            params={"direction": "right"})
        rep = assemble_segments(
            [Clip(path=str(c0)), Clip(path=str(c1))],
            segments=[(0, 1.0), (1, 1.0)],
            transitions=tr,
            out=workdir / f"tr-{kind}.mp4",
            width=320, height=480, fps=30, seed=1)
        assert Path(rep["output"]).stat().st_size > 0
        # overlap kinds shorten the total; blink/cut kinds don't
        if kind in ("fade", "dissolve", "whip_pan"):
            assert rep["duration"] < 2.0
        else:
            assert abs(rep["duration"] - 2.0) < 0.15

    @NEEDS_FFMPEG
    def test_cut_grid_round_robin(self, workdir):
        c0 = _mp4(workdir, "g0.mp4", color="red", seconds=4.0)
        c1 = _mp4(workdir, "g1.mp4", color="blue", seconds=4.0)
        rep = assemble_segments(
            [Clip(path=str(c0)), Clip(path=str(c1))],
            cut_points=[0.0, 1.0, 2.0, 3.0],
            out=workdir / "grid.mp4",
            width=320, height=480, fps=30, seed=1)
        assert rep["segments"] == 3
        assert abs(rep["duration"] - 3.0) < 0.2

    def test_bad_transition_raises(self):
        with pytest.raises(MediaEditError):
            validate_transition("explode")


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

class TestText:
    def test_layer_validation(self):
        with pytest.raises(MediaEditError):
            TextLayer()  # neither text nor words
        with pytest.raises(MediaEditError):
            TextLayer(text="x", position="sideways")
        with pytest.raises(MediaEditError):
            TextLayer(text="x", start=2.0, end=1.0)

    def test_layer_to_ass_positions(self):
        for pos in ("top", "center", "bottom", "bottom-left"):
            ass = layer_to_ass(TextLayer(text="hi", position=pos),
                               width=320, height=480)
            assert "PlayResX: 320" in ass and "Dialogue" in ass

    def test_estimate_timings(self):
        words = estimate_word_timings("one two three", wpm=60.0)
        assert [w.text for w in words] == ["one", "two", "three"]
        assert words[1].start == pytest.approx(1.0)

    def test_build_captions_srt(self, workdir):
        p = build_captions([("hi", 0.0, 1.0), ("yo", 1.0, 2.0)],
                           workdir / "c.srt")
        text = Path(p).read_text()
        assert "hi" in text and "00:00:01" in text

    def test_textlayer_round_trip(self):
        tl = TextLayer(text="hi", position="top-left", size=64,
                       color="#ff0000", box=True)
        assert TextLayer.from_dict(
            json.loads(json.dumps(tl.to_dict()))) == tl


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

class TestAudio:
    def test_layer_validation(self):
        with pytest.raises(MediaEditError):
            AudioLayer(path="")
        with pytest.raises(MediaEditError):
            AudioMix(layers=[AudioLayer(path="a.wav")], duck_key=5)
        with pytest.raises(MediaEditError):
            AudioMix(layers=[AudioLayer(path="a.wav")],
                     ducking="sidechain-ish")

    def test_round_trip(self):
        m = AudioMix(
            layers=[AudioLayer(path="m.mp3", loop=True, volume=0.8),
                    AudioLayer(path="v.wav", start=0.5, fade_in=0.2)],
            ducking="sidechain", duck_key=1, duck_level=0.4)
        assert AudioMix.from_dict(
            json.loads(json.dumps(m.to_dict()))) == m

    @NEEDS_FFMPEG
    def test_mix_layers_renders(self, workdir):
        a1 = _wav(workdir, "m1.wav", 440.0)
        a2 = _wav(workdir, "m2.wav", 660.0)
        mix = AudioMix(
            layers=[AudioLayer(path=str(a1), volume=0.8),
                    AudioLayer(path=str(a2), start=0.5, fade_in=0.2)],
            ducking="sidechain", duck_key=1, target_lufs=-16.0)
        rep = mix_layers(mix, 2.5, workdir / "mix.m4a")
        assert rep["bytes"] > 0 and rep["ducking"] == "sidechain"

    @NEEDS_FFMPEG
    def test_music_bed(self, workdir):
        p = make_music_bed(3.0, workdir / "bed.wav", seed=7)
        assert Path(p).stat().st_size > 0


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

class TestPresets:
    def test_all_styles_build_valid_specs(self):
        from nomorals.media.contentops.styles import (
            get_style, list_styles)
        assert set(list_styles()) >= {
            "phonk", "documentary", "vlog", "minimal"}
        for name in list_styles():
            spec = get_style(name).build_spec(
                scenes=[{"image": "x.png", "duration": 2.0}],
                captions="one two three four",
                voiceover="v.wav", music="m.mp3")
            assert isinstance(spec, EditSpec)
            assert len(spec.clips) == 1
            assert len(spec.text_layers) == 1
            assert spec.audio is not None
            # JSON-serializable
            EditSpec.from_dict(
                json.loads(json.dumps(spec.to_dict())))

    def test_unknown_style_raises(self):
        from nomorals.media.contentops.styles import get_style
        with pytest.raises(MediaEditError):
            get_style("nope")

    def test_phonk_applies_effect_cycle(self):
        from nomorals.media.contentops.styles import get_style
        spec = get_style("phonk").build_spec(
            clips=[Clip(path="a.mp4"), Clip(path="b.mp4"),
                   Clip(path="c.mp4")],
            beats=[0.0, 1.0, 2.0, 3.0])
        names = [c.effects[0].name for c in spec.clips]
        assert names == ["shake", "punch_zoom", "beat_pulse"]

    def test_phonk_keeps_explicit_effects(self):
        from nomorals.media.contentops.styles import get_style
        spec = get_style("phonk").build_spec(
            clips=[Clip(path="a.mp4",
                        effects=(Effect("grain"),))],
            beats=[0.0, 1.0])
        assert spec.clips[0].effects[0].name == "grain"

    def test_documentary_uses_dissolve(self):
        from nomorals.media.contentops.styles import get_style
        spec = get_style("documentary").build_spec(
            clips=[Clip(path="a.mp4")], title="The Story")
        assert spec.transition.kind == "dissolve"
        assert any(l.position == "bottom-left"
                   for l in spec.text_layers)


# ---------------------------------------------------------------------------
# engine render end-to-end
# ---------------------------------------------------------------------------

class TestEngineRender:
    @NEEDS_FFMPEG
    def test_full_render(self, workdir):
        c0 = _mp4(workdir, "e0.mp4", color="red")
        c1 = _mp4(workdir, "e1.mp4", color="green")
        vo = _wav(workdir, "vo.wav", 520.0, seconds=3.0)
        spec = EditSpec(
            clips=[Clip(path=str(c0),
                        effects=(Effect("color_grade",
                                        {"grade": "warm"}),)),
                   Clip(path=str(c1),
                        effects=(Effect("shake",
                                        {"amplitude": 6.0}),))],
            cut_points=[0.0, 1.5, 3.0],
            transition="dissolve",
            text_layers=[TextLayer(text="engine", position="bottom")],
            audio=AudioMix(layers=[AudioLayer(path=str(vo))],
                           target_lufs=-16.0),
            width=320, height=480, fps=30, seed=7,
            out=str(workdir / "full.mp4"))
        out = render(spec)
        info = video_probe(out)
        assert (info["width"], info["height"]) == (320, 480)
        assert any(s.get("type") == "audio"
                   for s in info["streams"])

    @NEEDS_FFMPEG
    def test_render_vertical(self, workdir):
        src = _mp4(workdir, "wide.mp4", w=640, h=360)
        rep = render_vertical(src, out=workdir / "vert.mp4",
                              width=320, height=480)
        info = video_probe(rep["output"])
        assert (info["width"], info["height"]) == (320, 480)

    def test_empty_spec_raises(self):
        with pytest.raises(MediaEditError):
            render(EditSpec())
