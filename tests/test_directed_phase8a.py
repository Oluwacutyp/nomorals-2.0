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


# ── camera pack + camera language ────────────────────────────────────
from PIL import Image as _PILImage

from nomorals.media.directed import camera as cam


def _test_frames(n=8, w=160, h=120):
    rng = np.random.RandomState(0)
    base = _PILImage.fromarray(
        rng.randint(0, 255, (h, w, 3)).astype(np.uint8))
    return [base.copy() for _ in range(n)]


def test_new_looks_registered_and_stable():
    for look in ("drone", "bodycam", "webcam", "vintage_film",
                 "anamorphic", "gimbal"):
        assert look in cam.LOOKS, look
        assert look in cam.LOOK_INFO, look
    frames = _test_frames()
    for look in ("drone", "bodycam", "webcam", "vintage_film",
                 "anamorphic", "gimbal"):
        out = cam.apply_look(frames, look, fps=24, seed=3)
        assert len(out) == len(frames)
        assert all(f.size == frames[0].size for f in out)
        a = np.array(out[0])
        assert a.dtype == np.uint8 and a.shape == (120, 160, 3)


def test_new_looks_deterministic():
    frames = _test_frames()
    a1 = np.array(cam.apply_look(frames, "vintage_film", seed=5)[3])
    a2 = np.array(cam.apply_look(frames, "vintage_film", seed=5)[3])
    assert (a1 == a2).all()


def test_anamorphic_flare_fires():
    bright = _PILImage.new("RGB", (160, 120), (10, 10, 10))
    from PIL import ImageDraw as _ID
    _ID.Draw(bright).ellipse([70, 50, 90, 70], fill=(255, 255, 255))
    fl = np.array(cam.anamorphic_look(bright)).astype(float)
    assert fl[:, :, 2].max() > 60  # blue streak off the highlight


def test_bodycam_bob_in_path():
    path = cam.camera_path(48, 24.0, "bodycam", seed=1, W=320, H=240)
    # step bounce: vertical range well above the noise floor of tripod
    still = cam.camera_path(48, 24.0, "tripod", seed=1, W=320, H=240)
    assert path[:, 1].ptp() > still[:, 1].ptp() + 2.0


def test_parse_camera_language():
    p = cam.parse_camera_language("dolly in slowly")
    assert [(m.verb, m.direction, m.speed) for m in p.moves] == [
        ("dolly", "in", "slow")]
    p = cam.parse_camera_language("orbit left, then crane up")
    assert [(m.verb, m.direction) for m in p.moves] == [
        ("orbit", "left"), ("crane", "up")]
    assert p.describe() == "orbit left, then crane up"
    p = cam.parse_camera_language("push in and tilt up quickly")
    assert [(m.verb, m.direction, m.speed) for m in p.moves] == [
        ("dolly", "in", "normal"), ("tilt", "up", "fast")]
    # no camera language -> empty, never raises
    p = cam.parse_camera_language("a woman waves and smiles")
    assert p.empty and p.describe() == ""
    p = cam.parse_camera_language("")
    assert p.empty


def test_camera_program_path_continuous():
    prog = cam.parse_camera_language("dolly in, then orbit left")
    path = cam.camera_program_path(prog, 40, 24.0, 320, 240)
    assert path.shape == (40, 4)
    assert path[0, 3] == 1.0
    assert path[-1, 3] > 1.1          # dolly in ends zoomed in
    assert np.abs(np.diff(path[:, 0])).max() < 20   # no jumps at handoff
    assert np.abs(np.diff(path[:, 3])).max() < 0.1


def test_apply_camera_program_roundtrip():
    frames = _test_frames(n=12)
    out = cam.apply_camera_program(
        frames, cam.parse_camera_language("crane up then pan right"), fps=24)
    assert len(out) == 12 and all(f.size == (160, 120) for f in out)
    # empty program -> passthrough
    out2 = cam.apply_camera_program(
        frames, cam.parse_camera_language("nothing camera-ish"))
    assert all(np.array(a).tobytes() == np.array(b).tobytes()
               for a, b in zip(frames, out2))


def test_prompt_engine_uses_camera_language():
    from nomorals.media.directed.prompt_engine import _extract_camera
    _, movement, _ = _extract_camera("dolly in slowly, then orbit left")
    assert movement == "slow dolly in, then orbit left"
    # legacy single keywords still work
    _, movement, _ = _extract_camera("a cat sits")
    assert movement == ""


# ── CPU lipsync visemes ──────────────────────────────────────────────
from nomorals.media.directed import lipsync as ls


def test_phoneme_viseme_coverage():
    arpabet = ("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW B CH D DH "
               "F G HH JH K L M N NG P R S SH T TH V W Y Z ZH").split()
    for p in arpabet:
        v = ls.phoneme_to_viseme(p)
        assert v in ls.VISEME_SHAPES, (p, v)
        shape = ls.mouth_shape_for_viseme(v)
        assert len(shape) == 3
        jaw, width, rnd = shape
        assert 0.0 <= jaw <= 1.0 and 0.5 <= width <= 1.25 and 0.0 <= rnd <= 1.0
    assert ls.phoneme_to_viseme("nonsense") == "sil"


def test_g2p_common_words():
    ph = ls.text_to_phonemes("Hello, how are you?")
    assert ph[:4] == ["HH", "EH", "L", "OW"]
    assert ph[-1] == "sil"  # sentence punctuation -> pause
    vs = ls.phonemes_to_visemes(ph)
    assert all(v in ls.VISEME_SHAPES for v in vs)
    # unknown words still produce phonemes via rules
    assert ls.text_to_phonemes("xylophone") != []


def test_viseme_track_timing():
    ph = ["HH", "EH", "L", "OW"]
    tr = ls.viseme_track(40, ph, 2.0, 20.0)
    assert len(tr) == 40
    assert all(len(s) == 3 for s in tr)
    # uniform timing: first frame ~ HH shape, last ~ OW shape
    assert tr[0][0] < 0.2      # HH -> sil-ish closed
    assert tr[-1][0] > 0.4     # OW -> rounded open
    # empty phonemes -> closed track
    tr = ls.viseme_track(8, [], 1.0, 8.0)
    assert all(s == ls.VISEME_SHAPES["sil"] for s in tr)


def test_acoustic_viseme_heuristic():
    assert ls.acoustic_viseme(0.0, 0.5) == "sil"
    assert ls.acoustic_viseme(0.05, 0.9) == "sil"
    assert ls.acoustic_viseme(0.8, 0.2) == "AH"
    assert ls.acoustic_viseme(0.3, 0.9) == "IY"


def test_warp_mouth_shapes_differ():
    rng = np.random.RandomState(0)
    frame = rng.randint(0, 255, (120, 160, 3)).astype(np.uint8)
    box = (60, 70, 100, 95)
    open_out = ls._warp_mouth(frame, box, ls.mouth_shape_for_viseme("AH"))
    shut_out = ls._warp_mouth(frame, box, ls.mouth_shape_for_viseme("sil"))
    assert open_out.shape == frame.shape
    # open jaw changes pixels below the mouth line; closed is identity
    assert (shut_out == frame).all()
    assert not (open_out == frame).all()
    # rounded W is narrower than wide IY
    w_out = ls._warp_mouth(frame, box, ls.mouth_shape_for_viseme("W"))
    iy_out = ls._warp_mouth(frame, box, ls.mouth_shape_for_viseme("IY"))
    assert not (w_out == iy_out).all()
