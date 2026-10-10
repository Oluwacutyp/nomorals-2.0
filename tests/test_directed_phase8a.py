"""Phase 8A: easing library, finger IK, camera pack/language, visemes,
depth warp, beat alignment, consolidation. New tests; test_directed.py
stays untouched."""

import math

import numpy as np
import pytest

from nomorals.media.directed import motion_score as ms


# ── easing library ───────────────────────────────────────────────────

def test_easing_endpoints_exact():
    for n in ms.list_easings():
        assert ms.ease_value(n, 0.0) == 0.0, n
        assert abs(ms.ease_value(n, 1.0) - 1.0) < 1e-9, n


def test_easing_legacy_curves_unchanged():
    assert ms.ease_value("ease_in", 0.5) == 0.125
    assert ms.ease_value("ease_out", 0.5) == 0.875
    assert ms.ease_value("linear", 0.5) == 0.5
    assert ms.ease_value("ease_in_out", 0.25) == 4 * 0.25 ** 3


def test_easing_unknown_falls_back():
    assert ms.ease_value("nope", 0.5) == ms.ease_value("smooth", 0.5)


def test_spring_physics():
    sv = [ms.ease_value("spring", i / 400) for i in range(401)]
    assert max(sv) > 1.0                      # overshoot
    crossings = sum(1 for a, b in zip(sv, sv[1:])
                    if (a - 1) * (b - 1) < 0)
    assert crossings >= 3                     # oscillates, then settles
    assert min(sv) >= 0.0


def test_overshoot_and_anticipation_shapes():
    ov = [ms.ease_value("overshoot", i / 100) for i in range(101)]
    assert max(ov) > 1.0
    av = [ms.ease_value("anticipation", i / 100) for i in range(101)]
    assert min(av) < -0.05                    # wind-up dip
    assert max(av) > 1.0                       # then overshoot
    ib = [ms.ease_value("ease_in_back", i / 100) for i in range(101)]
    assert min(ib) < -0.05 and max(ib) <= 1.0 + 1e-9


def test_cubic_bezier_solver():
    # identity control points == linear
    for u in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert abs(ms.cubic_bezier_ease(u, 0, 0, 1, 1) - u) < 1e-6
    # CSS ease is front-loaded: ahead of linear at midpoint
    assert ms.cubic_bezier_ease(0.5) > 0.5
    # degenerate flat-x control points still converge (bisection fallback)
    # true solution: bezier_x(t)=0.5 at t≈0.235, bezier_y(t)≈0.168
    assert abs(ms.cubic_bezier_ease(0.5, 0.9, 0.1, 0.9, 0.9) - 0.168) < 0.02


def test_new_easings_compile_into_score():
    score = {"action": "test", "phases": [{
        "name": "p", "t": [0.0, 1.0], "easing": "spring",
        "moves": [{"joint": "r_wrist", "to": [0.5, 0.4]}]}]}
    track, notes = ms.compile_score(score, n_frames=10)
    assert track.frames.shape[0] == 10


# ── finger IK ────────────────────────────────────────────────────────
from nomorals.media.directed import pose_rig as pr


def _tip_of(state_dict, finger, wrist=None, size=0.045):
    wrist = pr.rest_pose()[4] if wrist is None else wrist
    pts = pr.render_fingers(wrist, size, state_dict)
    return pts[{"thumb": 4, "index": 8, "middle": 12,
                "ring": 16, "pinky": 20}[finger]]


def test_finger_ik_roundtrip_all_fingers():
    rng = np.random.RandomState(1)
    wrist = pr.rest_pose()[4]
    worst = 0.0
    for f in pr.FINGERS:
        for _ in range(20):
            q = tuple(rng.uniform(0.05, 0.9, 3))
            q = (q[0], q[1], min(q[2], q[1] * pr._DIP_COUPLING))
            tgt = pr._tip_fk(wrist, f, 0.045, q, 1.0, 0.0)
            sol = pr.finger_ik(wrist, f, tgt, size=0.045)
            assert sol.reached, (f, q)
            st = {ff: (1.0, 1.0, 0.65, 0.0) for ff in pr.FINGERS}
            st[f] = (*sol.flex, sol.abduct)
            err = float(np.linalg.norm(_tip_of(st, f) - tgt))
            worst = max(worst, err)
            assert err < 5e-4, (f, q, err)
    assert worst < 5e-4


def test_finger_ik_reach_clamp():
    wrist = pr.rest_pose()[4]
    # far beyond reach: clamps to the reach circle, honest flag
    sol = pr.finger_ik(wrist, "index", wrist + np.array([0.0, -0.5]),
                       size=0.045)
    assert not sol.reached
    k, _ = pr._knuckle(wrist, "index", 0.045, 1.0, 0.0)
    rmax = pr.finger_reach("index", 0.045)[1]
    assert abs(float(np.linalg.norm(np.array(sol.tip) - k)) - rmax) < 1e-3
    # tendon coupling: dip follows pip
    assert abs(sol.flex[2] - min(1.0, sol.flex[1] * pr._DIP_COUPLING)) < 1e-9
    # flexions stay in joint limits
    assert all(0.0 <= v <= 1.0 for v in sol.flex)
    # unknown finger raises
    with pytest.raises(ValueError):
        pr.finger_ik(wrist, "extra", (0.5, 0.5))


def test_fingertip_move_compiles_and_lands():
    score = {"action": "press key", "phases": [
        {"name": "reach", "t": [0.0, 0.6], "easing": "ease_out", "moves": [
            {"joint": "r_wrist", "to": [0.55, 0.45]}]},
        {"name": "press", "t": [0.6, 1.0], "easing": "ease_in_out",
         "moves": [{"hand": "right",
                    "fingertips": {"index": [0.58, 0.42]}}]},
    ]}
    track, notes = ms.compile_score(score, n_frames=20)
    tip = track.frames[-1][pr.N_BODY + 8]
    assert float(np.linalg.norm(tip - np.array([0.58, 0.42]))) < 1e-3
    assert np.isfinite(track.frames).all()
    # unreachable target -> honest note, no NaN
    score["phases"][1]["moves"] = [
        {"hand": "right", "fingertips": {"index": [0.95, 0.05]}}]
    track, notes = ms.compile_score(score, n_frames=20)
    assert any("beyond reach" in n for n in notes)
    assert np.isfinite(track.frames).all()
