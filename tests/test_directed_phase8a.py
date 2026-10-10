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
