"""Render-critique loop: score a FINISHED render, feed it back.

Covers ``nomorals/media/directed/render_critique.py`` (the dream-loop
resolution rung) and the ``compile_score`` boundary-handoff fix it
required: phases used to commit one frame late, so the first frame
after a phase boundary rendered the next phase's move from the
uncommitted state (a visible snap-back in the repo's own shadow-box
fixture). All renders here are REAL: headless PIL skeleton frames +
ffmpeg mp4 (no Blender binary on this box), and the critique reads the
rendered pixels back — a blacked-out frame scores worse, which is the
check that the measurement reads the render, not the track.
"""

import json

import numpy as np
import pytest

from nomorals.media.directed import render_critique as rc
from nomorals.media.directed.motion_score import compile_score
from nomorals.media.directed.pose_rig import BODY_NAMES

#: root scale 1.6 shoves the raised wrist + head off the top of the
#: frame on pass 1. Valid score (scale <= 2.0, travel <= 0.35).
SCALE_SCORE = {
    "action": "raise arm high, camera dollies in",
    "phases": [{"name": "raise", "t": [0.0, 1.0], "easing": "ease_out",
                "moves": [
                    {"root": {"to": [0.0, 0.0], "scale": 1.6}},
                    {"joint": "r_wrist", "to": [0.43, 0.20]},
                    {"joint": "r_elbow", "to": [0.38, 0.31]},
                ]}],
}

#: one phase asks r_ankle for 0.61 travel (> 0.45 limit), so the
#: compiler clamps it and the render falls short. The knee moves with
#: it so the final pose stays inside the stretch guard — the shortfall
#: is the clamp alone, which the split must recover.
CLAMP_SCORE = {
    "action": "lift the right foot high",
    "phases": [{"name": "lift", "t": [0.0, 1.0], "easing": "ease_out",
                "moves": [{"joint": "r_knee", "to": [0.48, 0.55]},
                          {"joint": "r_ankle", "to": [0.42, 0.35]}]}],
}

_SHADOW_BOX = (
    '{"action": "shadow box", "implausible": false, "phases": ['
    '{"name": "guard", "t": [0.0, 0.3], "easing": "ease_out", "moves": ['
    '{"joint": "r_elbow", "to": [0.40, 0.38]},'
    '{"joint": "r_wrist", "to": [0.44, 0.30]},'
    '{"hand": "right", "pose": "fist"}]},'
    '{"name": "jab", "t": [0.3, 0.6], "easing": "ease_out", "repeat": 2,'
    ' "moves": [{"joint": "r_wrist", "to": [0.58, 0.28]}]},'
    '{"name": "recover", "t": [0.6, 1.0], "easing": "ease_in_out",'
    ' "moves": []}]}'
)


@pytest.fixture(autouse=True)
def _isolated_journal(tmp_path, monkeypatch):
    """Keep the critique journal out of the real ~/.nomorals."""
    monkeypatch.setenv("NM_HOME", str(tmp_path / "nmhome"))
    rc.clear_critiques()
    yield
    rc.clear_critiques()


def _scores(result):
    return [p["score"] for p in result["passes"]]


def test_loop_improves_scale_render(tmp_path):
    result = rc.critique_loop(score=SCALE_SCORE, threshold=0.8,
                              max_iterations=3,
                              workdir=str(tmp_path / "scale"))
    assert result["ok"] is True
    assert result["met"] is True
    scores = _scores(result)
    assert len(scores) >= 2
    assert scores[-1] > scores[0] + 0.05
    assert result["video"] is not None
    assert (tmp_path / "scale" / "final.mp4").exists()
    assert (tmp_path / "scale" / "final.mp4").stat().st_size > 1000
    frames = list((tmp_path / "scale"
                   / f"pass{len(scores)}").glob("*.png"))
    assert len(frames) >= 8


def test_loop_recovers_clamped_travel(tmp_path):
    result = rc.critique_loop(score=CLAMP_SCORE, threshold=0.95,
                              max_iterations=3,
                              workdir=str(tmp_path / "clamp"))
    assert result["ok"] is True
    assert result["met"] is True
    scores = _scores(result)
    assert len(scores) == 2  # split once, then the render reaches it
    assert scores[-1] > scores[0] + 0.05
    assert any("split" in a for p in result["passes"]
               for a in p["adjustments"])
    assert len(result["final_score"]["phases"]) == 2
    assert result["final_score"]["phases"][0]["name"] == "lift (part 1/2)"


def test_budget_stop_is_bounded(tmp_path):
    result = rc.critique_loop(score=SCALE_SCORE, threshold=0.8,
                              max_iterations=1,
                              workdir=str(tmp_path / "budget"))
    assert result["ok"] is True
    assert result["met"] is False
    assert len(result["passes"]) == 1


def test_mimicmotion_renderer_refused_by_name(tmp_path):
    result = rc.critique_loop(score=SCALE_SCORE, renderer="mimicmotion",
                              workdir=str(tmp_path / "refusal"))
    assert result["ok"] is False
    assert "MimicMotion" in result["error"]


def test_unknown_renderer_refused_by_name(tmp_path):
    result = rc.critique_loop(score=SCALE_SCORE, renderer="cycles",
                              workdir=str(tmp_path / "refusal2"))
    assert result["ok"] is False
    assert "cycles" in result["error"]


def test_no_pillow_refusal_names_pillow(tmp_path, monkeypatch):
    import nomorals.media.directed.pose_rig as pose_rig
    monkeypatch.setattr(pose_rig, "Image", None)
    result = rc.critique_loop(score=SCALE_SCORE,
                              workdir=str(tmp_path / "nopil"))
    assert result["ok"] is False
    assert "Pillow" in result["error"]


def test_invalid_score_refused(tmp_path):
    result = rc.critique_loop(score={"action": "nothing", "phases": []},
                              workdir=str(tmp_path / "invalid"))
    assert result["ok"] is False
    assert "validation" in result["error"]


def test_preset_description_refused_by_name(tmp_path):
    result = rc.critique_loop(description="wave",
                              workdir=str(tmp_path / "preset"))
    assert result["ok"] is False
    assert "preset" in result["error"]


def test_description_with_model_runs(tmp_path):
    def _suggest(prompt: str) -> str:
        return _SHADOW_BOX

    result = rc.critique_loop(description="shadow box", suggest=_suggest,
                              threshold=0.8, max_iterations=3,
                              workdir=str(tmp_path / "desc"))
    assert result["ok"] is True
    assert len(result["passes"]) >= 1
    assert rc.critique_count() >= len(result["passes"])


def test_journal_records_every_pass(tmp_path):
    result = rc.critique_loop(score=SCALE_SCORE, threshold=0.8,
                              max_iterations=3,
                              workdir=str(tmp_path / "journal"))
    entries = rc.recent_critiques(limit=10)
    assert len(entries) >= len(result["passes"])
    assert all("score" in e and "issues" in e for e in entries)
    assert rc.critique_count() >= len(result["passes"])
    assert "render critiques" in rc.format_critiques()


def test_critique_reads_pixels_not_track(tmp_path):
    from PIL import Image

    track, _notes = compile_score(SCALE_SCORE, n_frames=12)
    frames = rc.render_pose_frames(track, str(tmp_path / "pixels"))
    clean = rc.critique_render(frames, track, SCALE_SCORE)
    first = frames[0]
    Image.new("RGB", (track.width, track.height), (0, 0, 0)).save(first)
    dirty = rc.critique_render(frames, track, SCALE_SCORE)
    assert dirty.score < clean.score


def test_compile_boundary_handoff_has_no_snap():
    """The repo's own shadow-box fixture used to snap the wrist back
    ~0.19 at the guard->jab boundary (frame 5 -> 6); the commit now
    lands before the boundary frame renders."""
    score = json.loads(_SHADOW_BOX)
    track, _notes = compile_score(score, n_frames=20)
    w = BODY_NAMES.index("r_wrist")
    step56 = float(np.linalg.norm(track.frames[6, w] - track.frames[5, w]))
    assert step56 < 0.10
    # still rising toward the jab, not snapped back down
    assert track.frames[6, w, 1] < track.frames[5, w, 1] + 0.02
    assert track.frames[-1, w, 1] < track.frames[0, w, 1] - 0.1


def test_split_phase_stays_valid(tmp_path):
    """A split score still passes structural validation and compiles
    to a track whose ankle reaches the target."""
    result = rc.critique_loop(score=CLAMP_SCORE, threshold=0.95,
                              max_iterations=3,
                              workdir=str(tmp_path / "split"))
    assert result["met"] is True
    track, _notes = compile_score(result["final_score"], n_frames=24)
    ankle = BODY_NAMES.index("r_ankle")
    assert abs(track.frames[-1, ankle, 0] - 0.42) < 0.01
    assert abs(track.frames[-1, ankle, 1] - 0.35) < 0.01
