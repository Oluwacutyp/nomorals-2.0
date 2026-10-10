"""Tests for nomorals.media.videogen — neural pipeline orchestration.

No GPU in CI: every neural test exercises the interface and the
capability-gating mocks. What IS asserted for real:

- capability probing never raises and is honest about missing hardware
- the pipeline routes to the motion studio with a plain-language note
  when neural is unavailable (and says why)
- backends fail with VideogenError (install instructions), never with a
  raw ImportError/traceback
- request_hero_clip declines honestly instead of faking a neural clip
- chaining works end-to-end in motion mode with image/text scenes
"""
from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nomorals.media.videogen import (
    VideogenError,
    capability_report,
    cuda_vram_gb,
    diffusers_available,
    generate,
    neural_capability,
    request_hero_clip,
)
from nomorals.media.videogen.capabilities import (
    LTX_MIN_VRAM_GB,
    WAN_MIN_VRAM_GB,
)
from nomorals.media.videogen.ltx_backend import LTXBackend, _snap_frames
from nomorals.media.videogen.wan_backend import WanBackend
from nomorals.media.videogen.chaining import chain_scenes
from nomorals.media.motion_studio._core import probe_duration


@pytest.fixture()
def cover_png(tmp_path):
    p = tmp_path / "cover.png"
    Image.new("RGB", (320, 240), (40, 30, 80)).save(p)
    return str(p)


# -- capability probing --------------------------------------------------------

def test_neural_capability_never_raises():
    cap = neural_capability()
    assert cap.backend in ("ltx", "wan", "none")
    assert isinstance(cap.reason, str) and cap.reason
    assert isinstance(cap.explain(), str)


def test_capability_report_is_honest():
    report = capability_report()
    assert isinstance(report, str) and len(report) > 10
    # on this CI box there is no CUDA — the report must say so plainly
    if cuda_vram_gb() is None:
        assert "motion studio" in report.lower()


def test_vram_floors_sane():
    assert LTX_MIN_VRAM_GB == 8.0
    assert WAN_MIN_VRAM_GB == 16.0
    assert diffusers_available() in (True, False)


def test_prefer_ltx_reports_floor(monkeypatch):
    import nomorals.media.videogen.capabilities as C
    monkeypatch.setattr(C, "torch_cuda_available", lambda: True)
    monkeypatch.setattr(C, "cuda_vram_gb", lambda: 4.0)
    monkeypatch.setattr(C, "diffusers_available", lambda: True)
    cap = neural_capability(prefer="ltx")
    assert not cap.available
    assert "8GB" in cap.reason


def test_termux_never_neural(monkeypatch):
    import nomorals.media.videogen.capabilities as C
    monkeypatch.setattr(C, "get_profile_kind", lambda: "termux")
    cap = neural_capability()
    assert not cap.available
    assert "termux" in cap.reason.lower()


# -- backends: interface + honest failure ---------------------------------------

def test_ltx_check_honest_without_gpu():
    info = LTXBackend().check()
    assert info["backend"] == "ltx"
    assert info["available"] in (True, False)
    assert info["reason"]
    if not info["available"]:
        with pytest.raises(VideogenError):
            LTXBackend().generate("a neon alley")


def test_wan_check_honest_without_gpu():
    info = WanBackend().check()
    assert info["backend"] == "wan"
    assert info["available"] in (True, False)
    if not info["available"]:
        with pytest.raises(VideogenError):
            WanBackend().generate("a neon alley")


def test_ltx_frame_snapping():
    # LTX VAE needs num_frames = 8k+1
    for dur in (1.0, 2.5, 5.0, 10.0):
        n = _snap_frames(dur, 24)
        assert (n - 1) % 8 == 0, (dur, n)
        assert 9 <= n <= 257


def test_backends_reject_empty_prompt():
    # force "available" past the gate to reach prompt validation —
    # done via monkeypatched check(), no GPU touched
    for cls in (LTXBackend, WanBackend):
        b = cls()
        b.check = lambda: {"available": True, "reason": "mocked"}
        b.require = lambda: None
        with pytest.raises(VideogenError):
            b.generate("   ")


# -- pipeline routing ------------------------------------------------------------

def test_generate_motion_forced(cover_png, tmp_path):
    res = generate("a neon alley in rain", backend="motion",
                   image=cover_png, duration_s=1.0,
                   out=str(tmp_path / "m.mp4"),
                   size=(160, 120), fps=8)
    assert res.routed == "motion"
    assert res.engine == "motion"
    assert os.path.getsize(res.path) > 500
    assert "motion" in res.note.lower()


def test_generate_falls_back_honestly_without_gpu(tmp_path):
    if cuda_vram_gb() is not None:
        pytest.skip("this box has CUDA — fallback path not exercised")
    res = generate("a neon alley in rain", backend="auto", duration_s=1.0,
                   out=str(tmp_path / "f.mp4"), size=(160, 120), fps=8)
    assert res.routed == "motion"
    # the note must say WHY neural didn't run — never silent
    assert "cuda" in res.note.lower() or "neural" in res.note.lower()
    assert os.path.getsize(res.path) > 500


def test_generate_no_fallback_raises_without_gpu():
    if cuda_vram_gb() is not None:
        pytest.skip("this box has CUDA — gate not exercised")
    with pytest.raises(VideogenError):
        generate("a neon alley", backend="auto", allow_fallback=False)


def test_generate_rejects_empty():
    with pytest.raises(VideogenError):
        generate("   ")


def test_request_hero_clip_declines_without_gpu():
    if cuda_vram_gb() is not None:
        pytest.skip("this box has CUDA — decline path not exercised")
    res = request_hero_clip("a neon alley")
    assert res["ok"] is False
    assert res["reason"]
    assert "suggestion" in res  # caller gets a plan-B hint, not silence


# -- chaining (motion mode) -------------------------------------------------------

def test_chain_scenes_motion_mode(cover_png, tmp_path):
    report = chain_scenes(
        [
            {"prompt": "neon alley", "mode": "image", "image": cover_png,
             "duration_s": 1.0},
            {"prompt": "THE CITY", "mode": "text", "text": "THE CITY",
             "duration_s": 1.0},
        ],
        out=str(tmp_path / "film.mp4"),
        backend="motion",
        grade_preset="",
        format="",
        size=(160, 120),
        fps=8,
    )
    assert report.motion_scenes == 2
    assert report.neural_scenes == 0
    assert os.path.getsize(report.final_path) > 1000
    # 2x1.0s scenes minus one 0.7s crossfade ≈ 1.3s
    assert probe_duration(report.final_path) >= 1.1
    assert "motion" in report.summary().lower()


def test_chain_scenes_empty():
    with pytest.raises(VideogenError):
        chain_scenes([], backend="motion")


def test_chain_neural_unavailable_says_so(cover_png, tmp_path):
    if cuda_vram_gb() is not None:
        pytest.skip("this box has CUDA — decline path not exercised")
    report = chain_scenes(
        [{"prompt": "neon alley", "mode": "image", "image": cover_png,
          "duration_s": 1.0}],
        out=str(tmp_path / "film2.mp4"),
        backend="auto",
        grade_preset="",
        format="",
        size=(160, 120),
        fps=8,
    )
    assert report.neural_scenes == 0
    assert "neural" in report.note.lower()


# -- temporal consistency ------------------------------------------------------

@pytest.fixture()
def _clip_pair(tmp_path):
    """Two 1s clips with deliberately different color casts."""
    import subprocess
    from shutil import which
    ff = which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg not available")
    a = str(tmp_path / "a.mp4")
    b = str(tmp_path / "b.mp4")
    for path, color in ((a, "0xB06030"), (b, "0x3060B0")):
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i",
             f"color=c={color}:s=160x120:d=1:r=8",
             "-pix_fmt", "yuv420p", path],
            check=True, capture_output=True, timeout=60)
    return a, b


def test_boundary_metric_identical_frames():
    from nomorals.media.videogen.consistency import boundary_metric
    img = Image.new("RGB", (160, 160), (120, 80, 200))
    m = boundary_metric(img, img)
    assert m["score"] > 0.99
    assert m["color_shift"] < 0.01


def test_boundary_metric_mismatched_frames():
    from nomorals.media.videogen.consistency import boundary_metric
    warm = Image.new("RGB", (160, 160), (200, 120, 60))
    cool = Image.new("RGB", (160, 160), (60, 120, 200))
    m = boundary_metric(warm, cool)
    assert m["score"] < 0.7
    assert m["color_shift"] > 0.2


def test_reinhard_match_moves_palette():
    import numpy as np
    from nomorals.media.videogen.consistency import reinhard_match
    src = Image.new("RGB", (64, 64), (100, 100, 100))
    ref = Image.new("RGB", (64, 64), (180, 90, 60))
    out = reinhard_match(src, ref, strength=1.0)
    arr = np.asarray(out).mean(axis=(0, 1))
    assert arr[0] > 150 and arr[2] < 90  # warm cast adopted
    untouched = reinhard_match(src, ref, strength=0.0)
    assert np.asarray(untouched).mean() == 100.0


def test_consistency_pass_reports_and_improves(_clip_pair):
    from nomorals.media.videogen.consistency import (
        consistency_pass, boundary_metric, boundary_frames)
    a, b = _clip_pair
    before_a, before_b = boundary_frames(a, b)
    score_before = boundary_metric(before_a, before_b)["score"]
    paths, report = consistency_pass([a, b], head_frames=4)
    assert len(paths) == 2 and len(report["boundaries"]) == 1
    entry = report["boundaries"][0]
    assert "score_before" in entry and "score_after" in entry
    assert abs(entry["score_before"] - score_before) < 0.05
    assert entry["score_after"] >= entry["score_before"] - 0.05
    assert os.path.getsize(paths[1]) > 500


def test_consistency_pass_single_clip_noop(tmp_path):
    from nomorals.media.videogen.consistency import consistency_pass
    paths, report = consistency_pass(["only.mp4"])
    assert paths == ["only.mp4"] and report["boundaries"] == []


def test_chain_scenes_consistency_wired(cover_png, tmp_path):
    """chain_scenes runs the boundary pass and reports scores."""
    report = chain_scenes(
        [
            {"prompt": "warm desert", "mode": "image", "image": cover_png,
             "duration_s": 1.0},
            {"prompt": "cool ocean", "mode": "image", "image": cover_png,
             "duration_s": 1.0},
        ],
        out=str(tmp_path / "film2.mp4"),
        backend="motion",
        grade_preset="",
        format="",
        size=(160, 120),
        fps=8,
        consistency=True,
    )
    bnds = report.consistency.get("boundaries") or []
    assert len(bnds) == 1
    assert "score_after" in bnds[0]
    assert "consistency" in report.summary()
    # opting out keeps the old behavior
    report2 = chain_scenes(
        [{"prompt": "one", "mode": "image", "image": cover_png,
          "duration_s": 1.0},
         {"prompt": "two", "mode": "image", "image": cover_png,
          "duration_s": 1.0}],
        out=str(tmp_path / "film3.mp4"),
        backend="motion",
        grade_preset="",
        format="",
        size=(160, 120),
        fps=8,
        consistency=False,
    )
    assert report2.consistency == {}
