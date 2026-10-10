"""Weakness cures: fine fingers, root motion + planting, validator."""

import numpy as np
import pytest

from nomorals.media.directed.motion_score import (
    compile_score,
    generate_track,
)
from nomorals.media.directed.pose_rig import (
    FINGERS,
    N_KP,
    R_HAND,
    expand_flex,
    finger_state_from_pose,
    render_fingers,
    rest_pose,
)
from nomorals.media.directed.validate import (
    validate_score,
    validate_track,
)


def _wrist():
    return np.array([0.5, 0.5])


# ── Weakness 1: fine finger choreography ─────────────────────────────
def test_finger_fk_extended_vs_curled():
    ext = render_fingers(_wrist(), 0.05,
                         {f: (0.0, 0.0, 0.0, 0.0) for f in FINGERS})
    curl = render_fingers(_wrist(), 0.05,
                          {f: (1.0, 1.0, 0.65, 0.0) for f in FINGERS})
    # index tip (8): extended points up, curled folds back down
    assert ext[8, 1] < curl[8, 1]  # extended tip higher (smaller y)
    assert float(np.linalg.norm(ext[8] - curl[8])) > 0.02


def test_count_to_three():
    # index/middle/ring extended, pinky curled, thumb tucked
    state = {"index": (0, 0, 0, 0), "middle": (0, 0, 0, 0),
             "ring": (0, 0, 0, 0), "pinky": (1, 1, 0.65, 0),
             "thumb": (0.9, 0.9, 0.0, 0.2)}
    pts = render_fingers(_wrist(), 0.05, state)
    # three extended tips above wrist, pinky tip near/below wrist level
    assert pts[8, 1] < _wrist()[1]    # index tip up
    assert pts[12, 1] < _wrist()[1]   # middle tip up
    assert pts[16, 1] < _wrist()[1]   # ring tip up
    assert pts[20, 1] > pts[8, 1]     # pinky tip lower than index tip


def test_expand_flex_coupling():
    # uniform flex applies DIP tendon coupling
    mcp, pip, dip = expand_flex(0.8)
    assert (mcp, pip) == (0.8, 0.8)
    assert abs(dip - 0.8 * 0.65) < 1e-9
    # explicit triple respected
    assert expand_flex([0.1, 0.4, 0.3]) == (0.1, 0.4, 0.3)


def test_finger_choreography_in_score():
    score = {"action": "count", "phases": [
        {"name": "three", "t": [0.0, 0.6], "easing": "ease_out",
         "moves": [{"hand": "right",
                    "fingers": {"index": 0.0, "middle": 0.0, "ring": 0.0,
                                "pinky": 1.0, "thumb": 0.9}}]},
        {"name": "fist", "t": [0.6, 1.0], "easing": "ease_in_out",
         "moves": [{"hand": "right", "pose": "fist"}]},
    ]}
    track, _ = compile_score(score, n_frames=12)
    assert track.frames.shape == (12, N_KP, 2)
    # mid-track: index extended (tip above wrist), pinky curled
    mid = track.frames[6]
    rwrist = mid[4]
    rhand = mid[R_HAND]
    assert rhand[8, 1] < rwrist[1]  # index tip above wrist


def test_single_finger_detail():
    score = {"action": "tap", "phases": [
        {"name": "tap", "t": [0.0, 1.0], "easing": "linear", "repeat": 3,
         "moves": [{"hand": "right", "finger": "index",
                    "flex": [0.1, 0.5, 0.3], "abduct": 0.2}]},
    ]}
    track, _ = compile_score(score, n_frames=18)
    assert track.frames.shape[0] == 18
    assert np.all(np.isfinite(track.frames))


# ── Weakness 2: root motion + foot planting ──────────────────────────
def test_root_channel_recorded():
    score = {"action": "step", "phases": [
        {"name": "step", "t": [0.0, 1.0], "easing": "linear",
         "moves": [{"root": {"to": [0.10, 0.0]}}]},
    ]}
    track, _ = compile_score(score, n_frames=10)
    assert track.root is not None
    assert track.root.shape == (10, 3)
    # root x grows across the track
    assert track.root[-1, 0] > track.root[0, 0] + 0.05
    # body actually translated: nose moved right
    assert track.frames[-1, 0, 0] > track.frames[0, 0, 0] + 0.03


def test_planted_foot_does_not_skate():
    score = {"action": "step", "phases": [
        {"name": "step", "t": [0.0, 1.0], "easing": "linear", "plant": "left",
         "moves": [{"root": {"to": [0.12, 0.0]}}]},
    ]}
    track, _ = compile_score(score, n_frames=16)
    l_ankle = 13  # BODY_NAMES index
    seg = track.frames[:, l_ankle]
    drift = float(np.max(np.linalg.norm(seg - seg[0], axis=1)))
    assert drift < 0.02, f"planted foot skated {drift:.3f}"


def test_walking_alternating_plants():
    score = {"action": "walk", "phases": [
        {"name": "stepL", "t": [0.0, 0.5], "easing": "ease_in_out",
         "plant": "left", "moves": [{"root": {"to": [0.06, 0.0]}}]},
        {"name": "stepR", "t": [0.5, 1.0], "easing": "ease_in_out",
         "plant": "right", "moves": [{"root": {"to": [0.06, 0.0]}}]},
    ]}
    track, notes = compile_score(score, n_frames=20)
    assert track.root[-1, 0] > 0.08  # traveled right across both steps
    rep = validate_track(track, score)
    assert rep.ok, rep.errors


def test_root_teleport_clamped():
    score = {"action": "jump", "phases": [
        {"name": "leap", "t": [0.0, 1.0], "easing": "linear",
         "moves": [{"root": {"to": [0.9, 0.0]}}]},
    ]}
    _, notes = compile_score(score, n_frames=8)
    assert any("root travel" in n for n in notes)


# ── Weakness 3: validator ────────────────────────────────────────────
def test_validator_rejects_unknown_joint():
    rep = validate_score({"action": "x", "phases": [
        {"name": "p", "t": [0, 1], "easing": "linear",
         "moves": [{"joint": "tentacle", "to": [0.5, 0.5]}]}]})
    assert not rep.ok
    assert any("tentacle" in e for e in rep.errors)


def test_validator_rejects_teleport_speed():
    rep = validate_score({"action": "x", "phases": [
        {"name": "snap", "t": [0.0, 0.05], "easing": "linear",
         "moves": [{"joint": "r_wrist", "to": [0.9, 0.1]},
                   {"joint": "r_wrist", "to": [0.1, 0.9]}]}]})
    # second move: huge travel in tiny window
    assert not rep.ok or rep.score < 100


def test_validator_rejects_bad_finger():
    rep = validate_score({"action": "x", "phases": [
        {"name": "p", "t": [0, 1], "easing": "linear",
         "moves": [{"hand": "right", "finger": "tentacle", "flex": 0.5}]}]})
    assert not rep.ok


def test_validator_rejects_bad_root():
    rep = validate_score({"action": "x", "phases": [
        {"name": "p", "t": [0, 1], "easing": "linear",
         "moves": [{"root": {"to": [0.8, 0.0]}}]}]})
    assert not rep.ok
    assert any("root travel" in e for e in rep.errors)


def test_validator_accepts_good_score():
    rep = validate_score({"action": "wave hello", "phases": [
        {"name": "raise", "t": [0.0, 0.4], "easing": "ease_out",
         "moves": [{"joint": "r_wrist", "to": [0.55, 0.30]},
                   {"hand": "right", "pose": "open"}]},
        {"name": "wave", "t": [0.4, 1.0], "easing": "ease_in_out",
         "repeat": 3,
         "moves": [{"hand": "right", "fingers": {"index": 0.0}}]},
    ]})
    assert rep.ok, rep.errors
    assert rep.score == 100.0


def test_generate_track_validates_before_compile():
    def bad(prompt: str) -> str:
        return ('{"action": "x", "phases": [{"name": "p", "t": [0, 1], '
                '"easing": "linear", "moves": '
                '[{"joint": "r_wrist", "to": [9.9, 9.9]}]}]}')
    with pytest.raises(ValueError, match="validation"):
        generate_track("bad move", suggest=bad, n_frames=8)


def test_plausibility_in_meta():
    def ok(prompt: str) -> str:
        return ('{"action": "nod", "phases": [{"name": "n", "t": [0.1, 0.9], '
                '"easing": "ease_in_out", "repeat": 2, "moves": '
                '[{"joint": "nose", "to": [0.50, 0.19]}]}]}')
    _, meta = generate_track("nod", suggest=ok, n_frames=10)
    assert meta["plausibility"] == 100.0
