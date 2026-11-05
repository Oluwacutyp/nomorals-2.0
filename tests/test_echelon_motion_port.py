"""Echelon Wave 4 — the motion port (plan §2.3 items 1.3a/1.3b,
§2.4 typography timing consumption, 1.1d motion-brush carrier).

Proves: MotionProject is a document in the D1 idiom (round-trip,
snapshot identity, save/load); property expressions evaluate
exactly at FrameGrid frames; compilation is total — engine paths
become grid-exact Document keyframes (foreign AND name-colliding
easings baked, never re-eased), procedural paths materialize
through the CPU tier, every refusal names its reason; typography
consumes MEASURED word timings with provenance; renders flow
through render_document deterministically with the project hash
in the ledger.
"""
from __future__ import annotations

import json
import math
import os
import wave
from fractions import Fraction
from pathlib import Path

import pytest

from nomorals.media.edit_engine.frametime import FrameGrid
from nomorals.media.edit_engine.keyframes import KeyframeTrack
from nomorals.media.motion_studio import MotionStudioError
from nomorals.media.motion_studio.project import (
    MOTION_PROPERTIES,
    MotionProject,
    MotionSegment,
    PropKey,
    PropertyTrack,
    lyric_project,
    slideshow_project,
    trailer_project,
)
from nomorals.media.motion_studio.typography import (
    timing_sources,
    to_words,
)

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed")
def _pattern_image(path: Path, *, w: int = 320, h: int = 180) -> str:
    """A two-tone pattern with a centred square — zoom/pan move it."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), (30, 30, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([w // 2, 0, w, h], fill=(220, 220, 230))
    d.rectangle([w // 2 - 30, h // 2 - 30, w // 2 + 30, h // 2 + 30],
                fill=(230, 60, 60))
    img.save(path)
    return str(path)


def _solid_image(path: Path, color=(120, 80, 200)) -> str:
    from PIL import Image
    Image.new("RGB", (320, 180), color).save(path)
    return str(path)


def _sine_wav(path: Path, seconds: float = 2.5) -> str:
    sr = 22050
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            int(12000 * math.sin(2 * math.pi * 440 * i / sr))
            .to_bytes(2, "little", signed=True)
            for i in range(int(sr * seconds)))
        w.writeframes(frames)
    return str(path)


# ---------------------------------------------------------------------------
# 1.3a — expressions evaluate exactly on the frame grid
# ---------------------------------------------------------------------------

def test_property_sampling_exact_at_frames():
    tr = PropertyTrack("transform/zoom", [
        PropKey(0, 1.0), PropKey(48, 2.0, "ease_in"),
        PropKey(72, 0.5, "linear")])
    for fps in (24, 30, 60):
        grid = FrameGrid(fps)
        assert tr.sample_at(grid, 0) == 1.0
        assert tr.sample_at(grid, 48) == 2.0
        assert tr.sample_at(grid, 72) == 0.5
    # ease_in in the expression layer is CUBIC (motion_score), and
    # the sampler must mean it: u=0.5 -> 0.125 of the span
    assert tr.sample(24) == pytest.approx(1.0 + 0.125)
    # holds outside the key range
    assert tr.sample(-5) == 1.0 and tr.sample(10_000) == 0.5
    # linear span midpoint is exact
    assert tr.sample(60) == pytest.approx(1.25)


def test_property_validation_is_a_build_error():
    with pytest.raises(MotionStudioError):
        PropertyTrack("transform/nope", [PropKey(0, 1.0)])
    with pytest.raises(MotionStudioError):
        PropertyTrack("transform/zoom", [PropKey(0, 100.0)])
    with pytest.raises(MotionStudioError):
        PropKey(0, 1.0, "wobble")
    with pytest.raises(MotionStudioError):
        PropKey(1.5, 1.0)  # frames are integers on the grid
    with pytest.raises(MotionStudioError):
        PropertyTrack("opacity", [])


def test_segment_sample_defaults_and_deferred():
    seg = MotionSegment(uid="a", kind="image", src="x.png",
                        duration_s=1.0)
    assert seg.sample("transform/cx", 0) == 0.5  # registry default
    seg2 = MotionSegment(uid="b", kind="image", src="x.png",
                         duration_s=1.0,
                         properties={"camera3d/fov": 35.0})
    assert seg2.properties["camera3d/fov"] == 35.0  # carried
    with pytest.raises(MotionStudioError):
        seg2.sample("camera3d/fov", 0)  # never sampled


# ---------------------------------------------------------------------------
# 1.3a — the document idiom: round-trip, snapshot identity, save/load
# ---------------------------------------------------------------------------

def _sample_project(tmp_path) -> MotionProject:
    img = _solid_image(tmp_path / "a.png")
    return MotionProject(
        title="demo", seed=3,
        output={"width": 320, "height": 180, "fps": 30},
        segments=[
            MotionSegment(uid="one", kind="image", src=img,
                          duration_s=2.0, transition="crossfade",
                          properties={
                              "transform/zoom": {"keys": [
                                  {"frame": 0, "value": 1.0},
                                  {"frame": 60, "value": 1.4,
                                   "easing": "spring"}]},
                              "opacity": 0.9}),
            MotionSegment(uid="two", kind="video", src="clip.mp4",
                          duration_s=3.0),
        ])


def test_project_round_trip_and_snapshot_identity(tmp_path):
    p = _sample_project(tmp_path)
    d = p.to_dict()
    assert json.loads(json.dumps(d)) == d  # pure JSON
    p2 = MotionProject.from_dict(d)
    assert p2.to_dict() == d
    assert p.compute_hash() == p2.compute_hash()
    assert not p.is_dirty()               # unsealed: nothing to differ from
    p.seal()
    assert not p.is_dirty()
    p.set_property("one", "opacity", 0.5)
    assert p.is_dirty()                   # snapshot identity = modified-ness
    p.seal()
    assert not p.is_dirty()


def test_project_save_load_like_editstudio(tmp_path):
    p = _sample_project(tmp_path)
    path = p.save(tmp_path / "proj.json")
    assert p.doc_hash  # save seals: the file carries its snapshot
    loaded = MotionProject.load(path)
    assert loaded.to_dict() == p.to_dict()
    assert not loaded.is_dirty()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "something-else"}))
    with pytest.raises(MotionStudioError):
        MotionProject.load(bad)
    dup = _sample_project(tmp_path)
    dup.segments.append(dup.segments[0])
    with pytest.raises(MotionStudioError):
        MotionProject.from_dict(dup.to_dict())  # duplicate @uid


def test_property_tree_api(tmp_path):
    p = _sample_project(tmp_path)
    paths = p.property_paths()
    assert "segments/one/transform/zoom" in paths
    assert "segments/one/opacity" in paths
    assert isinstance(p.get_property("one", "transform/zoom"),
                      PropertyTrack)
    with pytest.raises(MotionStudioError):
        p.set_property("one", "transform/zoom", 99.0)   # out of range
    with pytest.raises(MotionStudioError):
        p.set_property("ghost", "opacity", 0.5)         # no such @uid
    with pytest.raises(MotionStudioError):
        p.get_property("one", "effects/nope")           # not registered


# ---------------------------------------------------------------------------
# §2.4 — typography consumes measured timings, provenance labelled
# ---------------------------------------------------------------------------

def test_to_words_provenance():
    measured = [{"text": "hello", "start": 0.50, "end": 0.90},
                {"text": "world", "start": 1.00, "end": 1.60}]
    words = to_words(measured, timing_source="measured")
    assert [w.timing_source for w in words] == ["measured", "measured"]
    assert words[0].start == 0.50 and words[1].end == 1.60  # untouched
    # a per-word stamp wins over the caller's blanket label
    stamped = [{"text": "a", "start": 0.0, "end": 0.3,
                "timing_source": "measured"}]
    assert to_words(stamped)[0].timing_source == "measured"
    # unlabelled timed input is authored, never upgraded to measured
    assert to_words(measured)[0].timing_source == "authored"
    # plain text is estimated, and says so
    est = to_words("one two three")
    assert {w.timing_source for w in est} == {"estimated"}
    assert timing_sources(words) == {"measured": 2}


# ---------------------------------------------------------------------------
# 1.3a — compilation: total, grid-exact, honest about tiers
# ---------------------------------------------------------------------------

def _grid_exact(track_dict, grid: FrameGrid) -> None:
    for key in track_dict["keys"]:
        t = Fraction(key["time"]).limit_denominator(10 ** 6)
        assert t == grid.seconds(grid.frames(key["time"])), key


def test_compile_engine_tier_grid_exact(tmp_path):
    img = _pattern_image(tmp_path / "a.png")
    p = slideshow_project([img], per_image=2.0, move="zoom_in",
                           size=(320, 180), fps=30)
    doc, rec = p.to_document(workdir=tmp_path / "c")
    (seg_rec,) = rec["segments"]
    assert seg_rec["tier"] == "engine"
    assert seg_rec["engine_paths"] == ["transform/zoom"]
    assert seg_rec["procedural_paths"] == []
    (kf,) = [k for k in doc.keyframes if k["clip_id"] == "img1"]
    assert (kf["effect"], kf["param"]) == ("zoom", "z")
    _grid_exact(kf, p.grid)
    track = p.segment("img1").properties["transform/zoom"]
    # two paths, one result: the engine sampler over the compiled
    # dict reproduces the expression layer's values at grid frames
    engine_track = KeyframeTrack.from_dict(kf)
    for frame in (0, 15, 30, 45, 60):
        t = float(p.grid.seconds(frame))
        assert engine_track.sample(t) == pytest.approx(
            track.sample(frame), abs=1e-6)


def test_compile_bakes_name_colliding_and_foreign_easings(tmp_path):
    img = _solid_image(tmp_path / "a.png")
    for easing in ("ease_in", "spring", "sine"):
        p = MotionProject(
            output={"width": 320, "height": 180, "fps": 30},
            segments=[MotionSegment(
                uid="v", kind="image", src=img, duration_s=2.0,
                properties={"transform/zoom": {"keys": [
                    {"frame": 0, "value": 1.0},
                    {"frame": 60, "value": 2.0, "easing": easing}]}})])
        doc, _ = p.to_document(workdir=tmp_path / f"c-{easing}")
        (kf,) = doc.keyframes
        track = p.segment("v").properties["transform/zoom"]
        # nothing passes through but linear subdivisions, and their
        # values sit ON the authored (cubic/spring/sine) curve —
        # ease_in here must be the cubic one, not the engine's
        # quadratic namesake
        assert all(k["easing"] == "linear" for k in kf["keys"])
        assert len(kf["keys"]) == 9
        for key in kf["keys"]:
            f = float(Fraction(key["time"]).limit_denominator(10 ** 6)
                      * 30)
            assert key["value"] == pytest.approx(track.sample(f),
                                                 abs=1e-9)
        mid = [k for k in kf["keys"]
               if abs(k["time"] - 1.0) < 1e-9][0]
        if easing == "ease_in":
            assert mid["value"] == pytest.approx(1.125)  # cubic, not 1.25


def test_compile_procedural_tier_for_pan(tmp_path):
    img = _pattern_image(tmp_path / "a.png")
    p = slideshow_project([img], per_image=2.0, move="pan_left",
                           size=(320, 180), fps=30)
    doc, rec = p.to_document(workdir=tmp_path / "c")
    (seg_rec,) = rec["segments"]
    assert seg_rec["tier"] == "procedural"
    assert "transform/cx" in seg_rec["procedural_paths"]
    assert "transform/zoom" in seg_rec["procedural_paths"]
    assert seg_rec["frames"] == 60
    clip = doc.tracks[0].clips[0]
    assert Path(clip.path).exists() and clip.path.endswith(".mp4")
    # zoom was sampled by the materializer — it must NOT also ride
    # the engine (one property, one application)
    assert doc.keyframes == []


def test_compile_refusals_name_their_reasons(tmp_path):
    img = _solid_image(tmp_path / "a.png")
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"not a real video")  # compile checks paths only
    # procedural path on a video segment
    p = MotionProject(segments=[MotionSegment(
        uid="v", kind="video", src=str(vid), duration_s=1.0,
        properties={"transform/cx": 0.4})])
    with pytest.raises(MotionStudioError, match="no rung on video"):
        p.to_document(workdir=tmp_path / "c1")
    # procedural path on a text segment
    p = MotionProject(segments=[MotionSegment(
        uid="t", kind="text", text="hi", duration_s=1.0,
        properties={"effects/blur": 4.0})])
    with pytest.raises(MotionStudioError, match="no rung on text"):
        p.to_document(workdir=tmp_path / "c2")
    # unbuilt procedural engine names its deferral
    p = MotionProject(segments=[MotionSegment(
        uid="x", kind="procedural", duration_s=1.0,
        design={"engine": "explainer"})])
    with pytest.raises(MotionStudioError, match="1.5d"):
        p.to_document(workdir=tmp_path / "c3")
    # brush without its mask anchor
    p = MotionProject(segments=[MotionSegment(
        uid="b", kind="image", src=img, duration_s=1.0,
        properties={"brush/zoom": 2.0})])
    with pytest.raises(MotionStudioError, match="needs a mask"):
        p.to_document(workdir=tmp_path / "c4")
    # a transition kind with no session rung
    p = MotionProject(segments=[
        MotionSegment(uid="a", kind="image", src=img, duration_s=1.0),
        MotionSegment(uid="b", kind="image", src=img, duration_s=1.0,
                      transition="wipeleft")])
    with pytest.raises(MotionStudioError, match="no engine rung"):
        p.to_document(workdir=tmp_path / "c5")
    # camera3d: carried by the document, refused at compile
    p = MotionProject(segments=[MotionSegment(
        uid="d", kind="image", src=img, duration_s=1.0,
        properties={"camera3d/fov": 35.0})])
    assert p.to_dict()["segments"][0]["properties"]["camera3d/fov"] == 35.0
    with pytest.raises(MotionStudioError, match="3D rung is not chosen"):
        p.to_document(workdir=tmp_path / "c6")


def test_compile_transitions_ride_session_directives(tmp_path):
    a = _solid_image(tmp_path / "a.png", (200, 0, 0))
    b = _solid_image(tmp_path / "b.png", (0, 0, 200))
    p = slideshow_project([a, b], per_image=1.0, move="zoom_in",
                           transition="crossfade",
                           size=(320, 180), fps=30)
    doc, rec = p.to_document(workdir=tmp_path / "c")
    clip2 = doc.tracks[0].clips[1]
    stamps = [e for e in clip2.effects if e.name == "v_transition"]
    assert len(stamps) == 1
    assert stamps[0].params["type"] == "fade"
    assert rec["segments"][1]["transition"].startswith("crossfade")


def test_brush_mask_compiles_and_names_its_rung(tmp_path):
    img = _pattern_image(tmp_path / "a.png")
    p = MotionProject(
        output={"width": 320, "height": 180, "fps": 30},
        segments=[MotionSegment(
            uid="m", kind="image", src=img, duration_s=1.0,
            mask={"kind": "box", "box": [80, 40, 240, 140],
                  "feather": 6},
            properties={"brush/zoom": {"keys": [
                {"frame": 0, "value": 1.0},
                {"frame": 30, "value": 2.2, "easing": "smooth"}]}})])
    doc, rec = p.to_document(workdir=tmp_path / "c")
    (seg_rec,) = rec["segments"]
    assert seg_rec["tier"] == "procedural"
    assert "box+feather" in seg_rec["mask"]
    # the ref on the segment is stamped with the producing rung
    assert p.segment("m").mask["rung"] == "box+feather"


# ---------------------------------------------------------------------------
# renders (ffmpeg): end-to-end, determinism, provenance
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
def test_render_slideshow_project_end_to_end(tmp_path):
    from tests.edit_engine_golden import frame_stats
    a = _pattern_image(tmp_path / "a.png")
    b = _solid_image(tmp_path / "b.png", (40, 160, 60))
    p = slideshow_project([a, b], per_image=2.0, move="zoom_in",
                           transition="crossfade",
                           size=(320, 180), fps=30)
    out1 = tmp_path / "r1.mp4"
    res = p.render(out=out1, workdir=tmp_path / "w1")
    assert Path(res["output"]).exists()
    assert 3.0 <= res["duration"] <= 4.2  # 2+2 s, crossfade overlaps
    rec = res["motion_project"]
    assert rec["project_hash"] == (p.doc_hash or p.compute_hash())
    stages = [r["stage"] for r in res["rungs"]]
    assert "motion:img1" in stages and "motion:img2" in stages
    # the ledger gained the project hash (plan §2.3a acceptance)
    from nomorals.media.motion_studio._core import read_ledger
    entries = [e for e in read_ledger()
               if e.get("kind") == "motion_project"]
    assert entries and entries[-1]["project_hash"] == rec["project_hash"]
    # determinism: same project -> identical sampled frames
    out2 = tmp_path / "r2.mp4"
    p.render(out=out2, workdir=tmp_path / "w2")

    def _stats(video, t, tag):
        d = tmp_path / tag
        d.mkdir(exist_ok=True)
        return frame_stats(video, t, workdir=d)

    for t in (0.5, 1.5, 3.2):
        s1 = _stats(out1, t, "f1")
        s2 = _stats(out2, t, "f2")
        assert s1 == s2, f"frame at {t}s diverged: {s1} vs {s2}"
    # the zoom is real: segment 1's centre changes as it pushes in
    early = _stats(out1, 0.2, "f3")
    late = _stats(out1, 1.8, "f4")
    assert early["center"] != late["center"]


@NEEDS_FFMPEG
def test_render_lyric_project_with_measured_words(tmp_path):
    audio = _sine_wav(tmp_path / "song.wav", seconds=2.5)
    words = [{"text": "HELLO", "start": 0.5, "end": 1.1},
             {"text": "WORLD", "start": 1.2, "end": 1.9}]
    p = lyric_project(audio, "", words=words, timing_source="measured",
                      size=(320, 180), fps=15)
    res = p.render(out=tmp_path / "lyric.mp4", workdir=tmp_path / "w")
    assert Path(res["output"]).exists()
    (seg_rec,) = res["motion_project"]["segments"]
    # measured words land as measured — the provenance reaches the
    # render record, not just the input dict
    assert seg_rec["timing_sources"] == {"measured": 2}
    note = [r for r in res["rungs"] if r["stage"] == "motion:lyrics"][0]
    assert "measured×2" in note["note"]


@NEEDS_FFMPEG
def test_render_lyric_project_estimated_is_labelled(tmp_path):
    audio = _sine_wav(tmp_path / "song.wav", seconds=2.5)
    p = lyric_project(audio, "one two three four",
                      size=(320, 180), fps=15)
    res = p.render(out=tmp_path / "lyric.mp4", workdir=tmp_path / "w")
    (seg_rec,) = res["motion_project"]["segments"]
    assert seg_rec["timing_sources"] == {"estimated": 4}


@NEEDS_FFMPEG
def test_render_trailer_project(tmp_path):
    from nomorals.media_edit.videos import run_ffmpeg
    clip = tmp_path / "c.mp4"
    run_ffmpeg(["-f", "lavfi", "-i",
                "color=c=teal:s=320x180:r=30:d=3",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)])
    p = trailer_project([str(clip)], title="MIDNIGHT RUN",
                        clip_len=1.5, size=(320, 180), fps=30)
    res = p.render(out=tmp_path / "trailer.mp4", workdir=tmp_path / "w")
    assert Path(res["output"]).exists()
    kinds = [s["kind"] for s in res["motion_project"]["segments"]]
    assert kinds == ["text", "video"]


# ---------------------------------------------------------------------------
# 1.3b — the decision row is recorded where the plan says it lives
# ---------------------------------------------------------------------------

def test_parity_records_the_1_3b_decision():
    parity = (Path(__file__).parent.parent
              / "nomorals" / "media" / "PARITY.md").read_text()
    assert "1.3b" in parity
    assert "deliberately not ported" in parity
    assert "crates/expr" in parity and "crates/script" in parity
    assert "camera3d" in parity  # the deferred rung is named too


def test_registry_shape():
    # every registered property declares its rung(s); a property with
    # neither an engine mapping nor a procedural sampler would be a
    # stored-but-never-applied value — the V5 sin, pinned shut here
    for path, spec in MOTION_PROPERTIES.items():
        assert spec.engine is not None or spec.procedural, path
        assert spec.lo < spec.hi, path
