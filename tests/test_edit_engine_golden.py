"""Golden renders for the edit engine (Echelon Wave 1.2e).

Fixed synthetic fixtures rendered through the real engine, compared
against recorded frame statistics in
``tests/golden/edit_engine_goldens.json``. A deliberately introduced
regression (wrong blend mode) is caught by the diff — that catch is
itself under test.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from nomorals.media.edit_engine import TextLayer
from nomorals.media.edit_engine.keyframes import (
    Keyframe,
    KeyframeTrack,
    keyframe_filter,
)
from nomorals.media.edit_engine.render import render_timeline
from nomorals.media.edit_engine.timeline import Clip, OverlayTrack, Timeline, Track
from nomorals.media_edit.videos import run_ffmpeg

from tests.edit_engine_golden import (
    color_source,
    diff_stats,
    frame_stats,
    load_goldens,
    pattern_source,
)

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed",
)

W, H, FPS = 320, 180, 30


def build_three_track(workdir: Path, *, blend: str = "screen") -> Path:
    """base (red) + screen overlay (blue) + text — the 1.2b fixture."""
    workdir.mkdir(parents=True, exist_ok=True)
    base = color_source(workdir / "base.mp4", "red", w=W, h=H)
    over = color_source(workdir / "over.mp4", "blue", w=W, h=H)
    tl = Timeline(
        video=Track(kind="video", clips=[Clip(path=base, id="b0")]),
        overlays=[OverlayTrack(
            kind="video", label="wash",
            clips=[Clip(path=over, id="o0", at=0.0, duration=1.0)],
            blend=blend, opacity=1.0)],
        text_layers=[TextLayer(text="GOLD", start=0.0, end=1.0,
                               position="bottom", size=28, margin_v=8,
                               color="white")],
        width=W, height=H, fps=FPS)
    out = workdir / "three_track.mp4"
    render_timeline(tl, out=out, workdir=workdir / "render")
    return out


def build_zoom(workdir: Path) -> Path:
    """Pattern clip with a 1.0 → 2.0 zoom keyframe track (1.2c)."""
    workdir.mkdir(parents=True, exist_ok=True)
    src = pattern_source(workdir / "pat.mp4", w=W, h=H, seconds=2.0)
    track = KeyframeTrack(clip_id="z0", effect="zoom", param="z",
                          keys=[Keyframe(0.0, 1.0), Keyframe(2.0, 2.0)])
    tl = Timeline(
        video=Track(kind="video", clips=[Clip(path=src, id="z0")]),
        keyframes=[track], width=W, height=H, fps=FPS)
    out = workdir / "zoom.mp4"
    render_timeline(tl, out=out, workdir=workdir / "render")
    return out


@NEEDS_FFMPEG
def test_three_track_matches_golden(tmp_path):
    out = build_three_track(tmp_path)
    stats = frame_stats(out, 0.5, workdir=tmp_path)
    golden = load_goldens()["three_track"]["stats"]
    assert diff_stats(stats, golden) == []


@NEEDS_FFMPEG
def test_zoom_keyframes_match_golden(tmp_path):
    out = build_zoom(tmp_path)
    goldens = load_goldens()["zoom"]
    for key, t in (("early", 0.2), ("late", 1.6)):
        stats = frame_stats(out, t, workdir=tmp_path)
        assert diff_stats(stats, goldens[key]["stats"]) == [], key


@NEEDS_FFMPEG
def test_zoom_matches_standalone_filter_oracle(tmp_path):
    """Two paths, one result (1.2c acceptance): the renderer's
    keyframed zoom vs the standalone keyframe_filter() — which is
    track_to_filter(track, ZOOM_TEMPLATE) with the canvas
    substituted — applied directly. Sampled frames must agree."""
    src = pattern_source(tmp_path / "pat.mp4", w=W, h=H, seconds=2.0)
    track = KeyframeTrack(clip_id="z0", effect="zoom", param="z",
                          keys=[Keyframe(0.0, 1.0), Keyframe(2.0, 2.0)])
    filt = keyframe_filter(track, width=W, height=H)
    oracle = tmp_path / "oracle.mp4"
    run_ffmpeg(["-i", src, "-vf", filt, "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(oracle)])
    via_engine = build_zoom(tmp_path / "engine")
    for t in (0.2, 1.0, 1.6):
        a = frame_stats(via_engine, t, workdir=tmp_path)
        b = frame_stats(oracle, t, workdir=tmp_path)
        assert diff_stats(a, b, tol=8.0) == [], f"t={t}"


@NEEDS_FFMPEG
def test_deliberate_regression_is_caught(tmp_path):
    """The documented regression: same fixture, wrong blend mode.

    screen(red, blue) is magenta; multiply(red, blue) is black. The
    golden diff must see it."""
    out = build_three_track(tmp_path, blend="multiply")
    stats = frame_stats(out, 0.5, workdir=tmp_path)
    golden = load_goldens()["three_track"]["stats"]
    diffs = diff_stats(stats, golden)
    assert diffs, "golden diff failed to catch a blend regression"


@NEEDS_FFMPEG
def test_render_determinism(tmp_path):
    """Same fixture twice → identical sampled frames (and, on the
    same machine/encoder, identical bytes — the Wave-2 shape)."""
    a = build_three_track(tmp_path / "run_a")
    b = build_three_track(tmp_path / "run_b")
    sa = frame_stats(a, 0.5, workdir=tmp_path)
    sb = frame_stats(b, 0.5, workdir=tmp_path)
    assert diff_stats(sa, sb, tol=0.5) == []
    ha = hashlib.sha256(Path(a).read_bytes()).hexdigest()
    hb = hashlib.sha256(Path(b).read_bytes()).hexdigest()
    assert ha == hb
