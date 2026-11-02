"""Pass tail-115 pins for nomorals/media/videogen/consistency.py — 0 converts / 3 exclusions.

107 (boundary_frames) and 289 (consistency_pass): the ffprobe fps
probe is best-effort (in-code note); a crash falls back to 24.0 fps,
which only positions the boundary frame grab a frame or two inside
the clip end. Frames are still extracted and the pass still grades.

186 (boundary_metric): an _edge_sim crash substitutes the histogram
correlation for the edge term (in-code note: "degrade the metric, not
the pass"). The composite score stays honestly derived from real
signals; the one wart is presentational — the returned dict's
"edge_sim" field then carries the hist value. No consumer reads that
field (chaining and edit_engine read only "score"; verified by grep),
and grading itself never consumes the metric.

No except handlers appear in this file (AST-checked).
"""
from pathlib import Path

from PIL import Image

import nomorals.media.motion_studio._core as core_mod
import nomorals.media.videogen.consistency as cons
import nomorals.media_edit.videos as videos_mod


def _boom(*a, **k):
    raise RuntimeError("probe boom")


def _fake_grab(path, t, width=320, out=None):
    Image.new("RGB", (16, 16), (120, 30, 30)).save(out)
    return out


def test_107_fps_probe_crash_defaults_to_24(monkeypatch):
    monkeypatch.setattr(videos_mod, "video_probe", _boom)
    monkeypatch.setattr(core_mod, "probe_duration", lambda p: 2.0)
    monkeypatch.setattr(cons, "_grab_frame", _fake_grab)
    fa, fb = cons.boundary_frames("a.mp4", "b.mp4")
    assert fa is not None and fb is not None


def test_107_healthy_fps_control(monkeypatch):
    monkeypatch.setattr(videos_mod, "video_probe", lambda p: {"fps": 30})
    monkeypatch.setattr(core_mod, "probe_duration", lambda p: 2.0)
    monkeypatch.setattr(cons, "_grab_frame", _fake_grab)
    fa, fb = cons.boundary_frames("a.mp4", "b.mp4")
    assert fa is not None and fb is not None


def test_186_edge_sim_crash_substitutes_hist_corr(monkeypatch):
    monkeypatch.setattr(cons, "_edge_sim", _boom)
    a = Image.new("RGB", (32, 32), (200, 40, 40))
    b = Image.new("RGB", (32, 32), (40, 40, 200))
    met = cons.boundary_metric(a, b)
    assert met["edge_sim"] == met["hist_corr"]
    assert 0.0 <= met["score"] <= 1.0


def test_186_healthy_metric_control():
    a = Image.new("RGB", (32, 32), (200, 40, 40))
    b = Image.new("RGB", (32, 32), (40, 40, 200))
    met = cons.boundary_metric(a, b)
    assert set(met) == {"color_shift", "hist_corr", "edge_sim", "score"}


def test_289_fps_probe_crash_boundary_still_graded(monkeypatch, tmp_path):
    img = Image.new("RGB", (32, 32), (90, 90, 90))
    monkeypatch.setattr(cons, "boundary_frames", lambda a, b: (img, img))
    monkeypatch.setattr(cons, "boundary_metric",
                        lambda a, b: {"color_shift": 0.1, "hist_corr": 0.9,
                                      "edge_sim": 0.9, "score": 0.9})

    def fake_grade(clip, ref, head_frames=12, out_path=None):
        Path(out_path).write_bytes(b"x")

    monkeypatch.setattr(cons, "_grade_head", fake_grade)
    monkeypatch.setattr(videos_mod, "video_probe", _boom)
    monkeypatch.setattr(core_mod, "probe_duration", lambda p: 2.0)
    monkeypatch.setattr(cons, "_grab_frame", _fake_grab)
    adjusted, report = cons.consistency_pass(["c0.mp4", "c1.mp4"],
                                             out_dir=str(tmp_path / "out"))
    assert len(adjusted) == 2
    entry = report["boundaries"][0]
    assert "score_before" in entry and "score_after" in entry
