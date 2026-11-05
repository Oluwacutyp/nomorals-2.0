"""Echelon Wave 1 · B0 acceptance: genedit.video_to_shot honesty.

B0 (INTEGRATION_PLAN §2.0): video_to_shot's neural probe pointed at a
non-existent ``imggen.pipeline.img2img``, so every v2v silently
degraded to unstyled passthrough copies — and per-keyframe failures
still reported ``backend="img2img"``. The fix resolves the styler
through the real imggen path (Studio → pipeline → edit.img2img) and
makes the return shape honest:

* keyframes actually styled when a styler exists;
* ``backend="passthrough"`` + a user-visible ``note`` naming the
  reason when no pipeline exists, or when every keyframe fails;
* partial styling keeps ``backend="img2img"`` with the unstyled
  keyframe count named in the note.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from nomorals.media import genedit

SRC_COLOR = (12, 34, 56)
STYLED_COLOR = (200, 100, 50)


def _fake_ffmpeg(monkeypatch):
    """Fake ffmpeg: extraction writes 8 REAL PNG frames; the
    reassembly call is a no-op success."""

    def fake_run(args, **kwargs):
        if "-vf" in args:
            out_pat = [a for a in args if a.endswith("f_%04d.png")][0]
            frames_dir = Path(out_pat).parent
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in range(8):
                Image.new("RGB", (16, 16), SRC_COLOR).save(
                    frames_dir / f"f_{i:04d}.png")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)


@pytest.fixture
def video(tmp_path, monkeypatch):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"fake-video")
    monkeypatch.setattr(genedit, "_ffmpeg", lambda: "/usr/bin/ffmpeg")
    _fake_ffmpeg(monkeypatch)
    return v


def _pixel(path):
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((0, 0))


def _write_styled(src, prompt, strength, dst):
    Image.new("RGB", (16, 16), STYLED_COLOR).save(dst)


# ── the neural path really styles ─────────────────────────────────────

def test_keyframes_actually_styled_with_styler(video, tmp_path,
                                               monkeypatch):
    calls = []

    def styler(src, prompt, strength, dst):
        calls.append((src, prompt, strength))
        _write_styled(src, prompt, strength, dst)

    monkeypatch.setattr(genedit, "_v2v_styler", lambda: (styler, ""))
    shot = genedit.video_to_shot(str(video), "make it anime",
                                 strength=0.7,
                                 workdir=str(tmp_path / "wd"))
    assert shot.backend == "img2img"
    assert shot.note == ""
    # 8 frames → keyframes are indices 0 and 4, styled with our args
    assert len(calls) == 2
    assert all(c[1] == "make it anime" and c[2] == 0.7 for c in calls)
    styled = tmp_path / "wd" / "styled"
    assert _pixel(styled / "s_0000.png") == STYLED_COLOR
    assert _pixel(styled / "s_0004.png") == STYLED_COLOR
    # non-keyframes inherit the nearest styled keyframe
    assert _pixel(styled / "s_0001.png") == STYLED_COLOR
    assert _pixel(styled / "s_0007.png") == STYLED_COLOR


# ── no pipeline: labelled passthrough, frames untouched ───────────────

def test_passthrough_labelled_when_no_styler(video, tmp_path,
                                             monkeypatch):
    monkeypatch.setattr(
        genedit, "_v2v_styler",
        lambda: (None, "ImgGenError: no loadable checkpoint"))
    shot = genedit.video_to_shot(str(video), "make it anime",
                                 workdir=str(tmp_path / "wd"))
    assert shot.backend == "passthrough"
    assert "NOT restyled" in shot.note
    assert "no loadable checkpoint" in shot.note
    styled = tmp_path / "wd" / "styled"
    for i in range(8):
        assert _pixel(styled / f"s_{i:04d}.png") == SRC_COLOR


def test_all_keyframes_failed_is_passthrough_not_img2img(
        video, tmp_path, monkeypatch):
    def boom(src, prompt, strength, dst):
        raise RuntimeError("cuda exploded")

    monkeypatch.setattr(genedit, "_v2v_styler", lambda: (boom, ""))
    shot = genedit.video_to_shot(str(video), "make it anime",
                                 workdir=str(tmp_path / "wd"))
    # never dressed as styled output
    assert shot.backend == "passthrough"
    assert "all 2 keyframes failed" in shot.note
    assert "cuda exploded" in shot.note
    styled = tmp_path / "wd" / "styled"
    assert _pixel(styled / "s_0000.png") == SRC_COLOR


# ── the real resolver wires Studio → edit.img2img ─────────────────────

def test_real_styler_wires_studio_pipeline_to_edit_img2img(
        tmp_path, monkeypatch):
    sentinel_pipe = object()
    seen = {}

    studio_mod = types.ModuleType("nomorals.media.imggen.studio")

    class _Studio:
        def pipeline(self):
            return sentinel_pipe

    studio_mod.Studio = _Studio

    edit_mod = types.ModuleType("nomorals.media.imggen.edit")

    def _fake_img2img(pipe, image, prompt, strength=0.6, **kw):
        seen["pipe"] = pipe
        seen["prompt"] = prompt
        seen["strength"] = strength
        seen["size"] = (kw.get("width"), kw.get("height"))
        return [Image.new("RGB", image.size, STYLED_COLOR)]

    edit_mod.img2img = _fake_img2img
    monkeypatch.setitem(sys.modules, "nomorals.media.imggen.studio",
                        studio_mod)
    monkeypatch.setitem(sys.modules, "nomorals.media.imggen.edit",
                        edit_mod)

    styler, reason = genedit._v2v_styler()
    assert styler is not None and reason == ""
    src = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), SRC_COLOR).save(src)
    dst = tmp_path / "styled.png"
    styler(str(src), "anime", 0.5, str(dst))
    # the Studio-built pipeline is what edit.img2img received
    assert seen["pipe"] is sentinel_pipe
    assert seen["prompt"] == "anime" and seen["strength"] == 0.5
    assert seen["size"] == (16, 16)
    assert _pixel(dst) == STYLED_COLOR


def test_real_styler_unavailable_reason_is_named(monkeypatch):
    studio_mod = types.ModuleType("nomorals.media.imggen.studio")

    class _Studio:
        def pipeline(self):
            raise RuntimeError("native image generation needs torch")

    studio_mod.Studio = _Studio
    monkeypatch.setitem(sys.modules, "nomorals.media.imggen.studio",
                        studio_mod)
    styler, reason = genedit._v2v_styler()
    assert styler is None
    assert "needs torch" in reason
