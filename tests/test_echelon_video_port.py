"""Echelon Wave 3 — the video port (plan §2.2 items 1.2a–1.2e,
§2.4 captions, 1.1b(b) render-cache honesty, Wave-1.2 renderer
convergence).

Proves: rational-time exactness at the spec layer; the renderer
composites the canonical Document/Timeline multi-track with
keyframes (and keyframes provably change pixels); per-rung honesty
in render results; pure_render manifest followed/worked-out fields
accurate on hit AND miss; host plans per profile with degradations
stated; caption timing labelled measured vs estimated.
"""
from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from nomorals.media.edit_engine import (
    Clip,
    DocTrack,
    Document,
    Effect,
    FrameGrid,
    Keyframe,
    KeyframeTrack,
    OverlayTrack,
    TextLayer,
    Timeline,
    Track,
)
from nomorals.media.edit_engine.host import HostCapabilities, LocalHost
from nomorals.media.edit_engine.pure_render import render_pure
from nomorals.media.edit_engine.render import (
    assemble_segments,
    plan_from_cut_points,
    render_document,
    render_timeline,
)
from nomorals.media.edit_engine.spec import EditSpec
from nomorals.media.edit_engine.text import (
    captions_from_audio,
    karaoke_layer,
)
from nomorals.media_edit.videos import MediaEditError, run_ffmpeg, video_probe

from tests.edit_engine_golden import color_source, frame_stats

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)

W, H, FPS = 320, 180, 30


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "vp"
    d.mkdir()
    return d


def _clip(workdir, name="c.mp4", color="red", seconds=2.0, audio=True):
    p = workdir / name
    args = ["-f", "lavfi", "-i",
            f"color=c={color}:s={W}x{H}:r={FPS}:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i",
                 f"sine=frequency=440:duration={seconds}"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    args.append(str(p))
    run_ffmpeg(args)
    return str(p)


# ---------------------------------------------------------------------------
# 1.2a rational time
# ---------------------------------------------------------------------------

class TestRationalTime:
    @pytest.mark.parametrize("fps", [24, 25, 30, 60])
    def test_round_trip_exact(self, fps):
        g = FrameGrid(fps)
        for n in (0, 1, 7, 30, 90, 441, 3600, 86_400):
            assert g.frames(g.seconds(n)) == n
            assert g.seconds(g.frames(g.seconds(n))) == g.seconds(n)

    def test_ntsc_rates_are_rational(self):
        assert FrameGrid(29.97).fps == Fraction(30000, 1001)
        assert FrameGrid(23.976).fps == Fraction(24000, 1001)
        assert FrameGrid(59.94).fps == Fraction(60000, 1001)
        g = FrameGrid(29.97)
        # one second of 29.97 is exactly 30 frames of 30000/1001
        assert g.frames(1.0) == 30
        assert g.seconds(30) == Fraction(1001, 1000)

    def test_ten_thousand_cut_plan_total_is_frame_sum(self):
        # cut grid: 10,000 boundaries a frame apart (plus head/tail)
        cuts = [i / 30.0 for i in range(10_001)]
        plan = plan_from_cut_points(3, cuts, fps=30)
        total = sum(d for _i, d in plan)
        # every segment is exactly one frame; the float total must be
        # the frame sum to within one frame — no drift accumulation
        assert abs(total - (10_000 / 30.0)) < (1.0 / 30.0)

    def test_timeline_frame_counts(self):
        tl = Timeline(
            video=Track(kind="video", clips=[
                Clip(path="a.mp4", start=0.0, end=1.5),
                Clip(path="b.mp4", start=0.25, end=2.25, speed=2.0),
            ]), fps=30)
        assert tl.frame_counts() == [45, 30]
        assert tl.total_frames() == 75
        q = tl.quantized()
        assert q.frame_counts() == [45, 30]


# ---------------------------------------------------------------------------
# 1.2b / convergence — multi-track renders
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestMultiTrackRender:
    def test_document_two_picture_tracks_render(self, workdir):
        base = _clip(workdir, "base.mp4", "red")
        over_png = workdir / "over.png"
        run_ffmpeg(["-f", "lavfi", "-i", f"color=c=blue:s={W}x{H}",
                    "-frames:v", "1", str(over_png)])
        doc = Document(
            tracks=[
                DocTrack(kind="video",
                         clips=[Clip(path=base, id="b0")]),
                DocTrack(kind="image", label="wash", blend="screen",
                         clips=[Clip(path=str(over_png), id="o0",
                                     duration=2.0)]),
            ],
            output={"width": W, "height": H, "fps": FPS, "crf": 23})
        r = render_document(doc, out=workdir / "doc.mp4",
                            workdir=workdir / "rw")
        assert r["renderer"] == "edit_engine.render.render_document"
        assert r["backend"] == "edit_engine.ffmpeg"
        assert any(t.get("blend") == "screen" for t in r["tracks"])
        stats = frame_stats(r["output"], 1.0, workdir=workdir)
        # screen(red, blue) = magenta: R and B high, G near zero
        assert stats["mean"][0] > 200 and stats["mean"][2] > 200
        assert stats["mean"][1] < 40

    def test_muted_overlay_track_is_reported_not_rendered(self, workdir):
        base = _clip(workdir, "base.mp4", "red")
        over = _clip(workdir, "over.mp4", "blue")
        tl = Timeline(
            video=Track(kind="video", clips=[Clip(path=base, id="b0")]),
            overlays=[OverlayTrack(
                kind="video", label="off",
                clips=[Clip(path=over, id="o0", duration=1.0)],
                muted=True)],
            width=W, height=H, fps=FPS)
        r = render_timeline(tl, out=workdir / "m.mp4",
                            workdir=workdir / "rw")
        assert r["tracks"] == [{"label": "off", "blend": "normal",
                                "rendered": False, "reason": "muted"}]
        stats = frame_stats(r["output"], 0.5, workdir=workdir)
        assert stats["mean"][0] > 200 and stats["mean"][2] < 40

    def test_position_keyframes_move_the_overlay(self, workdir):
        base = _clip(workdir, "base.mp4", "black", audio=False)
        white = _clip(workdir, "white.mp4", "white", seconds=1.0,
                      audio=False)
        kf = [KeyframeTrack(clip_id="sq", effect="position", param="x",
                            keys=[Keyframe(0.0, 0.0, "linear"),
                                  Keyframe(1.0, 240.0, "linear")])]
        tl = Timeline(
            video=Track(kind="video", clips=[Clip(path=base, id="b0")]),
            overlays=[OverlayTrack(
                kind="video", label="sq", width=80, height=80, y=50,
                clips=[Clip(path=white, id="sq", at=0.0, duration=1.0)])],
            keyframes=kf, width=W, height=H, fps=FPS)
        r = render_timeline(tl, out=workdir / "pos.mp4",
                            workdir=workdir / "rw")
        assert any(k["family"] == "position"
                   for k in r["keyframes_applied"])

        def halves(t):
            from PIL import Image
            png = workdir / f"pos_{t}.png"
            run_ffmpeg(["-ss", str(t), "-i", r["output"],
                        "-frames:v", "1", "-y", str(png)])
            img = Image.open(png).convert("L")
            w, h = img.size
            left = img.crop((0, 0, w // 3, h))
            right = img.crop((2 * w // 3, 0, w, h))
            import statistics
            return (statistics.fmean(left.getdata()),
                    statistics.fmean(right.getdata()))

        l0, r0 = halves(0.15)
        l1, r1 = halves(0.85)
        assert l0 > r0 + 30  # square starts left
        assert r1 > l1 + 30  # and ends right


# ---------------------------------------------------------------------------
# 1.2c keyframes in the render
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestKeyframesInRender:
    def test_zoom_keyframes_change_output(self, workdir):
        src = _clip(workdir, "pat.mp4", "green", audio=False)
        plain = Timeline(
            video=Track(kind="video", clips=[Clip(path=src, id="z0")]),
            width=W, height=H, fps=FPS)
        a = render_timeline(plain, out=workdir / "plain.mp4",
                            workdir=workdir / "ra")
        zoomed = Timeline(
            video=Track(kind="video", clips=[Clip(path=src, id="z0")]),
            keyframes=[KeyframeTrack(
                clip_id="z0", effect="zoom", param="z",
                keys=[Keyframe(0.0, 1.0), Keyframe(2.0, 2.5)])],
            width=W, height=H, fps=FPS)
        b = render_timeline(zoomed, out=workdir / "zoom.mp4",
                            workdir=workdir / "rb")
        assert b["keyframes_applied"]
        # solid green zooms to itself — use a pattern for pixel proof
        from tests.edit_engine_golden import pattern_source
        pat = pattern_source(workdir / "pat2.mp4", w=W, h=H, seconds=2.0)
        plain2 = Timeline(
            video=Track(kind="video", clips=[Clip(path=pat, id="z0")]),
            width=W, height=H, fps=FPS)
        a2 = render_timeline(plain2, out=workdir / "plain2.mp4",
                             workdir=workdir / "rc")
        zoomed2 = Timeline(
            video=Track(kind="video", clips=[Clip(path=pat, id="z0")]),
            keyframes=[KeyframeTrack(
                clip_id="z0", effect="zoom", param="z",
                keys=[Keyframe(0.0, 1.0), Keyframe(2.0, 2.5)])],
            width=W, height=H, fps=FPS)
        b2 = render_timeline(zoomed2, out=workdir / "zoom2.mp4",
                             workdir=workdir / "rd")
        sa = frame_stats(a2["output"], 1.8, workdir=workdir)
        sb = frame_stats(b2["output"], 1.8, workdir=workdir)
        delta = max(abs(x - y) for x, y in zip(sa["mean"], sb["mean"]))
        assert delta > 10, "zoom keyframes did not change the picture"

    def test_opacity_fade_in_darkens_first_frames(self, workdir):
        src = _clip(workdir, "w.mp4", "white", audio=False)
        tl = Timeline(
            video=Track(kind="video", clips=[Clip(path=src, id="f0")]),
            keyframes=[KeyframeTrack(
                clip_id="f0", effect="opacity", param="alpha",
                keys=[Keyframe(0.0, 0.0), Keyframe(2.0, 1.0)])],
            width=W, height=H, fps=FPS)
        r = render_timeline(tl, out=workdir / "fade.mp4",
                            workdir=workdir / "rw")
        assert any(k["family"] == "opacity"
                   for k in r["keyframes_applied"])
        early = frame_stats(r["output"], 0.1, workdir=workdir)
        late = frame_stats(r["output"], 1.9, workdir=workdir)
        assert early["mean"][0] < late["mean"][0] - 100

    def test_position_track_in_assemble_raises(self, workdir):
        src = _clip(workdir, "c.mp4", "red", audio=False)
        with pytest.raises(MediaEditError, match="position"):
            assemble_segments(
                [Clip(path=src, id="p0")],
                keyframes=[KeyframeTrack(
                    clip_id="p0", effect="position", param="x",
                    keys=[Keyframe(0.0, 0.0), Keyframe(1.0, 10.0)])],
                out=workdir / "x.mp4", width=W, height=H, fps=FPS,
                workdir=workdir / "rw")

    def test_undeclared_keyframe_target_raises(self, workdir):
        src = _clip(workdir, "c.mp4", "red", audio=False)
        tl = Timeline(
            video=Track(kind="video", clips=[Clip(path=src, id="u0")]),
            keyframes=[KeyframeTrack(
                clip_id="u0", effect="blur", param="amount",
                keys=[Keyframe(0.0, 0.0), Keyframe(1.0, 5.0)])],
            width=W, height=H, fps=FPS)
        with pytest.raises(MediaEditError):
            render_timeline(tl, out=workdir / "x.mp4",
                            workdir=workdir / "rw")

    def test_keyframe_for_missing_clip_raises(self, workdir):
        src = _clip(workdir, "c.mp4", "red", audio=False)
        tl = Timeline(
            video=Track(kind="video", clips=[Clip(path=src, id="u0")]),
            keyframes=[KeyframeTrack(
                clip_id="ghost", effect="zoom", param="z",
                keys=[Keyframe(0.0, 1.0), Keyframe(1.0, 2.0)])],
            width=W, height=H, fps=FPS)
        with pytest.raises(MediaEditError, match="ghost"):
            render_timeline(tl, out=workdir / "x.mp4",
                            workdir=workdir / "rw")


# ---------------------------------------------------------------------------
# 1.2d host seam
# ---------------------------------------------------------------------------

class TestHostPlans:
    def _host(self, profile, ffmpeg=True):
        caps = HostCapabilities(ffmpeg=ffmpeg, profile=profile,
                                accel_note="note")
        return LocalHost(profile, capabilities=caps)

    def test_termux_plan_degrades_and_says_why(self):
        plan = self._host("termux").plan(width=320, height=1080, fps=30)
        assert plan["degraded"] is True
        assert plan["height"] == 480
        assert plan["preset"] == "ultrafast"
        assert any("full-res" in r for r in plan["reasons"])

    def test_mobile_plan_degrades(self):
        plan = self._host("mobile").plan(width=320, height=720, fps=30)
        assert plan["degraded"] is True
        assert plan["height"] == 480

    def test_pc_plan_full_res_no_degradation(self):
        plan = self._host("pc").plan(width=1920, height=1080, fps=30)
        assert plan["degraded"] is False
        assert plan["height"] == 1080
        assert plan["reasons"] == []

    def test_missing_ffmpeg_is_a_stated_degradation(self):
        plan = self._host("pc", ffmpeg=False).plan(
            width=320, height=240, fps=30)
        assert plan["degraded"] is True
        assert plan["ffmpeg"] is False


@NEEDS_FFMPEG
class TestHostThroughRender:
    def test_assemble_with_host_reports_plan(self, workdir):
        src = _clip(workdir, "c.mp4", "red", audio=False)
        caps = HostCapabilities(ffmpeg=True, profile="termux",
                                accel_note="note")
        host = LocalHost("termux", capabilities=caps)
        # a 1080p request on a termux plan: the gate degrades it and
        # the render report carries the plan that says so.
        r = assemble_segments([Clip(path=src, id="h0")], host=host,
                              out=workdir / "h.mp4", width=W, height=1080,
                              fps=FPS, workdir=workdir / "rw")
        assert r["host"]["profile"] == "termux"
        assert r["host"]["degraded"] is True
        assert r["host"]["preset"] == "ultrafast"
        assert r["backend"] == "edit_engine.ffmpeg"
        assert r["rungs"][0]["stage"] == "assemble"


# ---------------------------------------------------------------------------
# 1.1b(b) pure_render manifest honesty
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestPureRenderHonesty:
    def _spec(self, workdir):
        src = _clip(workdir, "c.mp4", "red", seconds=1.0)
        return src, EditSpec(clips=[Clip(path=src)], width=W, height=H,
                             fps=FPS, out=str(workdir / "o.mp4"))

    def test_miss_then_hit(self, workdir):
        src, spec = self._spec(workdir)
        cache = workdir / "cache"
        miss = render_pure([src], spec, cache_dir=cache)
        assert miss["cached"] is False
        assert miss["followed"] is None
        assert miss["worked_out"]["renderer"] == "edit_engine.spec.render"
        assert miss["worked_out"]["via"] == "render_pure"
        stored = json.loads(
            (cache / f"{miss['cache_key']}.json").read_text())
        assert stored["worked_out"] == miss["worked_out"]
        assert stored["followed"] is None

        hit = render_pure([src], spec, cache_dir=cache)
        assert hit["cached"] is True
        assert hit["worked_out"] is None
        assert hit["followed"]["cache_key"] == miss["cache_key"]
        assert hit["followed"]["worked_out_at"] == \
            miss["worked_out"]["at"]
        assert hit["followed"]["renderer"] == "edit_engine.spec.render"

    def test_legacy_manifest_reports_unknown_provenance(self, workdir):
        src, spec = self._spec(workdir)
        cache = workdir / "cache"
        miss = render_pure([src], spec, cache_dir=cache)
        man = cache / f"{miss['cache_key']}.json"
        stored = json.loads(man.read_text())
        stored.pop("worked_out")
        stored.pop("followed")
        man.write_text(json.dumps(stored))
        hit = render_pure([src], spec, cache_dir=cache)
        assert hit["cached"] is True
        assert hit["followed"]["worked_out_at"] is None
        assert hit["followed"]["renderer"] == \
            "unknown (pre-honesty manifest)"


# ---------------------------------------------------------------------------
# §2.4 captions — measured vs estimated, labelled
# ---------------------------------------------------------------------------

class _FakeTranscriber:
    def __init__(self, words):
        self._words = words

    def transcribe_words(self, path):
        return list(self._words)


@NEEDS_FFMPEG
class TestCaptionTiming:
    WORDS = [{"start": 0.0, "end": 0.4, "text": "hello",
              "speaker": "Ada"},
             {"start": 0.5, "end": 0.9, "text": "world",
              "speaker": "Ada"},
             {"start": 1.0, "end": 1.5, "text": "again",
              "speaker": "Ben"}]

    def test_measured_words_verbatim_with_speakers(self, workdir):
        src = _clip(workdir, "c.mp4", "red", seconds=1.0)
        out = captions_from_audio(src, transcriber=_FakeTranscriber(
            self.WORDS))
        assert out["timing_source"] == "measured"
        assert out["transcriber"] == "_FakeTranscriber"
        assert [w["text"] for w in out["words"]] == \
            ["hello", "world", "again"]
        assert out["speakers"] == ["Ada", "Ben"]
        # word-timing error against the reference: zero (verbatim)
        mae = sum(abs(w["start"] - r["start"])
                  for w, r in zip(out["words"], self.WORDS)) / 3
        assert mae == 0.0

    def test_estimate_is_labelled_estimated(self, workdir):
        src = _clip(workdir, "c.mp4", "red", seconds=1.0)
        out = captions_from_audio(
            src, text="one two three",
            transcriber=_FakeTranscriber([]))
        assert out["timing_source"] == "estimated"
        assert out["transcriber"] is None
        assert len(out["words"]) == 3

    def test_neither_measured_nor_text_raises(self, workdir):
        src = _clip(workdir, "c.mp4", "red", seconds=1.0)
        with pytest.raises(MediaEditError):
            captions_from_audio(src, transcriber=_FakeTranscriber([]))

    def test_sidecar_writes_srt_with_provenance(self, workdir):
        from nomorals.media.edit_engine.text import captions_sidecar
        src = _clip(workdir, "c.mp4", "red", seconds=1.0)
        out = captions_sidecar(src, workdir / "caps.srt",
                               transcriber=_FakeTranscriber(self.WORDS))
        assert out["timing_source"] == "measured"
        assert out["speakers"] == ["Ada", "Ben"]
        srt = Path(out["output"]).read_text(encoding="utf-8")
        assert "hello" in srt and "again" in srt

    def test_karaoke_layer_carries_the_label(self):
        layer = karaoke_layer(self.WORDS, timing_source="measured")
        assert layer.timing_source == "measured"
        assert layer.start == 0.0 and layer.end == 1.5
        # speaker keys are stripped for the caption Word model
        assert all("speaker" not in w for w in layer.words)
        d = layer.to_dict()
        assert d["timing_source"] == "measured"
        assert TextLayer.from_dict(d).timing_source == "measured"


# ---------------------------------------------------------------------------
# convergence — session Documents through render_document
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestSessionDocumentRender:
    def test_title_fade_speed_chapter_end_to_end(self, workdir):
        a = _clip(workdir, "a.mp4", "red")
        b = _clip(workdir, "b.mp4", "blue")
        fx = [Effect(name="v_title",
                     params={"text": "Hello", "duration": 1.0}),
              Effect(name="v_transition",
                     params={"type": "fade", "duration": 0.5}),
              Effect(name="v_speed",
                     params={"factor": 2.0, "start": 1.5, "end": 3.0}),
              Effect(name="v_chapter",
                     params={"name": "Intro", "at": 0.0}),
              Effect(name="v_export",
                     params={"preset": "web-optimized"})]
        doc = Document(
            tracks=[DocTrack(kind="video", clips=[
                Clip(path=a, effects=tuple(fx)), Clip(path=b)])],
            output={"width": W, "height": H, "fps": FPS, "crf": 23})
        r = render_document(doc, out=workdir / "doc.mp4",
                            workdir=workdir / "rw")
        assert r["renderer"] == "edit_engine.render.render_document"
        # 1.0 title + 2 + 2 − 2×0.5 fades − 0.75 ramp saving = 3.25
        assert 3.0 < r["duration"] < 3.6
        assert video_probe(r["output"])["duration"] == \
            pytest.approx(r["duration"], abs=0.35)
        names = [d["directive"] for d in r["session_directives"]]
        assert names == ["v_title", "v_transition", "v_speed",
                         "v_chapter", "v_export"]
        import subprocess
        meta = subprocess.run(
            ["ffprobe", "-v", "error", "-show_chapters", r["output"]],
            capture_output=True, text=True).stdout
        assert "Intro" in meta
        # audio survived the speed ramp (video+audio retimed together)
        info = video_probe(r["output"])
        assert any(s["type"] == "audio" for s in info["streams"])

    def test_duck_uses_directive_bgm(self, workdir):
        a = _clip(workdir, "a.mp4", "red")
        bgm = _clip(workdir, "bgm.mp4", "black")
        fx = [Effect(name="v_duck",
                     params={"bgm": bgm, "amount_db": 12.0,
                             "bgm_gain": 0.5})]
        doc = Document(
            tracks=[DocTrack(kind="video",
                             clips=[Clip(path=a, effects=tuple(fx))])],
            output={"width": W, "height": H, "fps": FPS, "crf": 23})
        r = render_document(doc, out=workdir / "duck.mp4",
                            workdir=workdir / "rw")
        assert any(x["stage"] == "session:v_duck" for x in r["rungs"])
        info = video_probe(r["output"])
        assert any(s["type"] == "audio" for s in info["streams"])

    def test_unknown_session_transition_raises(self, workdir):
        a = _clip(workdir, "a.mp4", "red")
        b = _clip(workdir, "b.mp4", "blue")
        fx = [Effect(name="v_transition",
                     params={"type": "circleopen", "duration": 0.5})]
        doc = Document(
            tracks=[DocTrack(kind="video", clips=[
                Clip(path=a, effects=tuple(fx)), Clip(path=b)])],
            output={"width": W, "height": H, "fps": FPS, "crf": 23})
        with pytest.raises(MediaEditError, match="no engine rung"):
            render_document(doc, out=workdir / "x.mp4",
                            workdir=workdir / "rw")


# ---------------------------------------------------------------------------
# tools emit timeline keyframes (1.2c surface)
# ---------------------------------------------------------------------------

class TestToolsEmitTimelineKeyframes:
    def _reg(self, tmp_path):
        from nomorals.tools.registry import ToolRegistry

        class FakeContext:
            workspace = str(tmp_path)

        return ToolRegistry(context=FakeContext()).register_builtins()

    def test_keyframe_tools_carry_timeline_keyframes(self, tmp_path):
        reg = self._reg(tmp_path)
        out = reg.call("studio_zoom_ramp", clip_id="c1", start=0.0,
                       end=1.0)
        assert out.value["timeline_keyframes"][0]["effect"] == "zoom"
        out = reg.call("studio_fade", clip_id="c1", direction="in",
                       duration=1.0)
        assert out.value["timeline_keyframes"][0]["effect"] == "opacity"
        out = reg.call("studio_keyframe", clip_id="c1", effect="zoom",
                       param="z", keys=["0:1.0", "1:1.5"])
        assert out.value["timeline_keyframes"][0]["param"] == "z"
        # legacy keys stay for compat
        assert out.value["track"]["clip_id"] == "c1"


# ---------------------------------------------------------------------------
# LayerStack → Document convergence (1.2b, stills side)
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
class TestLayerStackToDocument:
    def _png(self, workdir, name, color):
        p = workdir / name
        run_ffmpeg(["-f", "lavfi", "-i", f"color=c={color}:s={W}x{H}",
                    "-frames:v", "1", str(p)])
        return str(p)

    def test_stack_converts(self, workdir):
        from nomorals.media_edit.layers import LayerStack
        base = self._png(workdir, "base.png", "red")
        over = self._png(workdir, "over.png", "blue")
        stack = LayerStack(base)
        stack.add_image(over, blend="screen", opacity=0.8)
        stack.add_text("Caption", font_size=32)
        doc = stack.to_document()
        assert [t.kind for t in doc.tracks] == ["image", "image"]
        assert doc.tracks[1].blend == "screen"
        assert doc.tracks[1].opacity == 0.8
        assert len(doc.text_layers) == 1
        assert doc.output["width"] == W and doc.output["height"] == H

    def test_shape_layer_refuses_loudly(self, workdir):
        from nomorals.media_edit.layers import LayerStack
        base = self._png(workdir, "base.png", "red")
        stack = LayerStack(base)
        stack.add_shape("rect", box=(0, 0, 50, 50))
        with pytest.raises(MediaEditError, match="raster ops"):
            stack.to_document()
