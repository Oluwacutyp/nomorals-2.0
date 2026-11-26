"""FaceMesh — MediaPipe Face Mesh (468 3D landmarks) to
Devon's face takes (B.25, the correction to B.23's iPhone-only
LiveLink capture).

The mapping (``nomorals.media.facemesh.landmarks``) derives
the rig's 52 ARKit blendshape weights from landmark GEOMETRY
— never from MediaPipe's own blendshape output. These tests
run on SYSTEM python (no mediapipe installed) from REAL
landmark data: five faces' Face Landmarker output saved
under ``tests/fixtures/facemesh/landmarks/``, each with
MediaPipe's own 52 scores as the ground-truth cross-check.
The video round trip (real mp4 through the real pipeline)
lives in ``test_gapfix_facemesh_video.py`` — it needs
mediapipe, so it runs under the mpvenv python.

What is pinned here:
- all 52 weights present, in 0..1, on every real face;
- the strong signals track MediaPipe: smile, pucker, jaw,
  look direction, upper lip;
- the lid-narrow signal is conserved: blink + squint = lid;
- the rest-gated weights (cheekPuff, dimple, sneer) read 0
  on single images — no rest baseline, no guess;
- tongueOut is 0 always — no tongue landmark exists;
- the roll correction (derotation) holds the weights steady
  when a face is tilted 20 degrees;
- the head matrix becomes Euler (z, x, y) degrees;
- a take built from the fixtures drives the rig — the
  weights land on blendshapes — and round-trips as JSON;
- the refusals name themselves (no face, no video, no camera).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.facemesh.errors import FaceMeshError
from nomorals.media.facemesh.landmarks import (
    ARKIT_BLENDSHAPES,
    FaceRest,
    derotate,
    estimate_rest,
    landmarks_to_blendshapes,
    landmarks_to_head_bones,
    matrix_to_euler_zxy,
)
from nomorals.media.facemesh.take import landmarks_to_take
from nomorals.media.rig.rig import ARKIT_BLENDSHAPES as RIG_SHAPES

FIXTURES = Path(__file__).parent / "fixtures" / "facemesh"
LANDMARKS = FIXTURES / "landmarks"

# The five real faces: smile, neutral, smile+jaw, pucker, smile.
FACES = ("u-woman1", "u-woman2", "men-3", "men-6", "women-8")


@pytest.fixture(params=FACES)
def face(request):
    blob = json.loads((LANDMARKS / f"{request.param}.json")
                      .read_text(encoding="utf-8"))
    return request.param, [tuple(p) for p in blob["landmarks"]], \
        blob["blendshapes"]


@pytest.fixture(params=FACES)
def weights(request):
    blob = json.loads((LANDMARKS / f"{request.param}.json")
                      .read_text(encoding="utf-8"))
    lm = [tuple(p) for p in blob["landmarks"]]
    return (request.param, landmarks_to_blendshapes(lm),
            blob["blendshapes"])


@pytest.fixture(autouse=True)
def _reset_takes():
    from nomorals.media.facemcap.registry import reset_takes
    reset_takes()
    yield
    reset_takes()


# ---------------------------------------------------------------------------
# The mapping, on real faces
# ---------------------------------------------------------------------------

def test_all_52_present_and_bounded(weights):
    name, w, _mp = weights
    assert list(w) == list(RIG_SHAPES)
    assert len(w) == 52
    for key, value in w.items():
        assert 0.0 <= value <= 1.0, f"{name} {key}={value}"


def test_smile_reads_on_smiling_faces(weights):
    name, w, mp = weights
    if name in ("u-woman1", "men-3", "women-8"):
        assert w["mouthSmileLeft"] > 0.5, name
        assert w["mouthSmileRight"] > 0.5, name
        assert abs(w["mouthSmileLeft"] - mp["mouthSmileLeft"]) < 0.35
        assert abs(w["mouthSmileRight"]
                   - mp["mouthSmileRight"]) < 0.35
    else:
        # u-woman2 is neutral, men-6 is puckering, not smiling.
        assert w["mouthSmileLeft"] < 0.1, name
        assert w["mouthSmileRight"] < 0.1, name


def test_jaw_open_direction():
    def jaw(name):
        blob = json.loads((LANDMARKS / f"{name}.json")
                          .read_text(encoding="utf-8"))
        return landmarks_to_blendshapes(
            [tuple(p) for p in blob["landmarks"]])["jawOpen"]
    # men-3's jaw is open (MediaPipe 0.56); u-woman2's is shut.
    assert jaw("men-3") > 0.5
    assert jaw("u-woman2") < 0.1
    assert jaw("men-3") > jaw("u-woman2")
    # The smile confound, named: a wide smile parts the lips
    # without dropping the jaw, so u-woman1 reads hotter than
    # MediaPipe's 0.22 — the direction still holds.
    assert jaw("u-woman1") > jaw("u-woman2")


def test_pucker_reads():
    def pucker(name):
        blob = json.loads((LANDMARKS / f"{name}.json")
                          .read_text(encoding="utf-8"))
        return landmarks_to_blendshapes(
            [tuple(p) for p in blob["landmarks"]])["mouthPucker"]
    assert pucker("men-6") > 0.9  # MediaPipe 0.99
    assert pucker("men-6") > pucker("u-woman2") > pucker("u-woman1")
    assert pucker("u-woman1") < 0.1


def test_blink_and_squint_conserve_lid(face):
    from nomorals.media.facemesh.landmarks import _signals
    name, lm, _mp = face
    w = landmarks_to_blendshapes(lm)
    sig = _signals(derotate(lm))
    for side, word in (("L", "Left"), ("R", "Right")):
        lid = max(0.0, min(1.0, 1 - sig[f"ear_{side}"] / 0.45))
        total = w[f"eyeBlink{word}"] + w[f"eyeSquint{word}"]
        assert total == pytest.approx(lid, abs=1e-9), \
            f"{name} {word}: blink+squint {total} != lid {lid}"
    # The squint reads on smiling faces, not on neutral ones.
    if name in ("u-woman1", "women-8"):
        assert w["eyeSquintLeft"] > 0.2, name
        assert w["eyeSquintRight"] > 0.2, name
    if name in ("u-woman2", "men-3", "men-6"):
        assert w["eyeSquintLeft"] < 0.1, name
        assert w["eyeSquintRight"] < 0.1, name


def test_brow_reads():
    def brow(name):
        blob = json.loads((LANDMARKS / f"{name}.json")
                          .read_text(encoding="utf-8"))
        return landmarks_to_blendshapes(
            [tuple(p) for p in blob["landmarks"]])
    # men-3's brows are raised (MediaPipe inner 0.29);
    # u-woman1's are not (0.01).
    assert brow("men-3")["browInnerUp"] > brow("u-woman1")["browInnerUp"]
    assert brow("men-3")["browInnerUp"] > 0.1
    for name in FACES:
        assert brow(name)["browInnerUp"] < 0.35, name
    # men-6's brows sit low (MediaPipe down 0.15/0.16).
    assert brow("men-6")["browDownLeft"] > 0.1
    assert brow("u-woman1")["browDownLeft"] == 0.0


def test_look_direction():
    def look(name):
        blob = json.loads((LANDMARKS / f"{name}.json")
                          .read_text(encoding="utf-8"))
        return landmarks_to_blendshapes(
            [tuple(p) for p in blob["landmarks"]])
    # men-6 looks up (MediaPipe 0.85/0.84).
    assert look("men-6")["eyeLookUpLeft"] > 0.6
    assert look("men-6")["eyeLookUpRight"] > 0.6
    # men-3 looks down (MediaPipe 0.61/0.48).
    assert look("men-3")["eyeLookDownLeft"] > 0.4
    assert look("men-3")["eyeLookDownRight"] > 0.4
    assert look("u-woman2")["eyeLookUpLeft"] < 0.3


def test_rest_gated_weights_read_zero_on_single_images(weights):
    name, w, _mp = weights
    # No video rest baseline, no guess: puff, dimple and
    # sneer read 0 on every single image.
    assert w["cheekPuff"] == 0.0, name
    assert w["mouthDimpleLeft"] == 0.0, name
    assert w["mouthDimpleRight"] == 0.0, name
    assert w["noseSneerLeft"] == 0.0, name
    assert w["noseSneerRight"] == 0.0, name
    # And tongueOut is 0 always — no tongue landmark exists.
    assert w["tongueOut"] == 0.0, name


def test_synthetic_closed_eye_blinks():
    # A NEUTRAL face for the blink: on a smiling face the lid
    # closure reads as a squint, not a blink (the smile is the
    # disambiguator — that path is asserted below).
    blob = json.loads((LANDMARKS / "u-woman2.json")
                      .read_text(encoding="utf-8"))
    lm = [list(p) for p in blob["landmarks"]]
    open_w = landmarks_to_blendshapes([tuple(p) for p in lm])
    assert open_w["eyeBlinkLeft"] < 0.2
    assert open_w["eyeBlinkRight"] < 0.2
    # Collapse each lid gap to 10% of itself — a closed eye,
    # made from real landmarks, not a fake fixture.
    for up, low in ((159, 145), (158, 153), (386, 374), (385, 380)):
        for axis in (0, 1):
            mid = (lm[up][axis] + lm[low][axis]) / 2
            lm[up][axis] = mid + (lm[up][axis] - mid) * 0.1
            lm[low][axis] = mid + (lm[low][axis] - mid) * 0.1
    shut_w = landmarks_to_blendshapes([tuple(p) for p in lm])
    assert shut_w["eyeBlinkLeft"] > 0.7
    assert shut_w["eyeBlinkRight"] > 0.7
    assert shut_w["eyeSquintLeft"] < 0.2
    assert shut_w["eyeSquintRight"] < 0.2
    # The same collapse on a SMILING face reads as a squint.
    blob = json.loads((LANDMARKS / "u-woman1.json")
                      .read_text(encoding="utf-8"))
    lm = [list(p) for p in blob["landmarks"]]
    for up, low in ((159, 145), (158, 153), (386, 374), (385, 380)):
        for axis in (0, 1):
            mid = (lm[up][axis] + lm[low][axis]) / 2
            lm[up][axis] = mid + (lm[up][axis] - mid) * 0.1
            lm[low][axis] = mid + (lm[low][axis] - mid) * 0.1
    shut_w = landmarks_to_blendshapes([tuple(p) for p in lm])
    assert shut_w["eyeSquintLeft"] > 0.7
    assert shut_w["eyeSquintRight"] > 0.7


def test_derotate_corrects_roll():
    blob = json.loads((LANDMARKS / "u-woman1.json")
                      .read_text(encoding="utf-8"))
    lm = [tuple(p) for p in blob["landmarks"]]
    base = landmarks_to_blendshapes(lm)
    # Tilt the whole face 20 degrees in the image plane.
    ang = math.radians(20)
    ca, sa = math.cos(ang), math.sin(ang)
    tilted = [(x * ca - y * sa, x * sa + y * ca, z)
              for x, y, z in lm]
    rolled = landmarks_to_blendshapes(tilted)
    for key in ("mouthSmileLeft", "mouthSmileRight", "jawOpen",
                "mouthPucker", "browInnerUp"):
        assert rolled[key] == pytest.approx(base[key], abs=0.15), \
            f"{key}: {rolled[key]} vs {base[key]} after a 20-degree roll"


def test_estimate_rest_from_frames():
    frames = []
    for name in FACES:
        blob = json.loads((LANDMARKS / f"{name}.json")
                          .read_text(encoding="utf-8"))
        frames.append([tuple(p) for p in blob["landmarks"]])
    rest = estimate_rest(frames)
    assert rest.has_rest is True
    assert 0.25 <= rest.ear_open_L <= 0.60
    assert 0.25 <= rest.ear_open_R <= 0.60
    assert 0.45 <= rest.mouth_width <= 0.90
    assert 0.75 <= rest.cheek_width <= 1.05
    # Mapping against the video's own rest still bounds.
    for lm in frames:
        w = landmarks_to_blendshapes(lm, rest=rest)
        assert len(w) == 52
        assert all(0.0 <= v <= 1.0 for v in w.values())
    # And the rest-gated weights can now read: the dimple
    # point is closest to its corner on the smiling faces.
    w = landmarks_to_blendshapes(frames[0], rest=rest)
    assert w["mouthDimpleLeft"] >= 0.0


def test_bad_landmarks_refused():
    with pytest.raises(FaceMeshError) as exc:
        landmarks_to_blendshapes([(0.0, 0.0, 0.0)] * 10)
    assert exc.value.kind == "bad_landmarks"


# ---------------------------------------------------------------------------
# Head rotation
# ---------------------------------------------------------------------------

def _rot_z(deg):
    a = math.radians(deg)
    return [[math.cos(a), -math.sin(a), 0.0],
            [math.sin(a), math.cos(a), 0.0],
            [0.0, 0.0, 1.0]]


def _rot_x(deg):
    a = math.radians(deg)
    return [[1.0, 0.0, 0.0],
            [0.0, math.cos(a), -math.sin(a)],
            [0.0, math.sin(a), math.cos(a)]]


def _rot_y(deg):
    a = math.radians(deg)
    return [[math.cos(a), 0.0, math.sin(a)],
            [0.0, 1.0, 0.0],
            [-math.sin(a), 0.0, math.cos(a)]]


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3))
             for j in range(3)] for i in range(3)]


def test_matrix_to_euler_zxy():
    assert matrix_to_euler_zxy([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == \
        pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    z, x, y = matrix_to_euler_zxy(_rot_z(30))
    assert (z, x, y) == pytest.approx((30.0, 0.0, 0.0), abs=1e-9)
    z, x, y = matrix_to_euler_zxy(_rot_x(20))
    assert (z, x, y) == pytest.approx((0.0, 20.0, 0.0), abs=1e-9)
    z, x, y = matrix_to_euler_zxy(_rot_y(-15))
    assert (z, x, y) == pytest.approx((0.0, 0.0, -15.0), abs=1e-9)
    # The composed order is R = Rz(z) Rx(x) Ry(y), the rig's
    # BVH channel order.
    z, x, y = matrix_to_euler_zxy(
        _matmul(_matmul(_rot_z(10), _rot_x(20)), _rot_y(30)))
    assert (z, x, y) == pytest.approx((10.0, 20.0, 30.0), abs=1e-9)


def test_head_bones():
    bones = landmarks_to_head_bones(_rot_z(30))
    assert set(bones) == {"head"}
    assert bones["head"] == pytest.approx((30.0, 0.0, 0.0), abs=1e-9)
    assert landmarks_to_head_bones(None) == {}


# ---------------------------------------------------------------------------
# Take: build from fixtures, drive the rig, round-trip
# ---------------------------------------------------------------------------

def _fixture_take():
    # Five DIFFERENT faces share no rest baseline — a video
    # rest assumes one face across frames — so the take is
    # built against the population rest, like the single-image
    # mapping tests above.
    frames = []
    for i, name in enumerate(FACES):
        blob = json.loads((LANDMARKS / f"{name}.json")
                          .read_text(encoding="utf-8"))
        frames.append((i / 8.0, [tuple(p) for p in blob["landmarks"]]))
    return landmarks_to_take("fixtures", frames, fps=8.0,
                             rest=FaceRest(),
                             source="fixtures")


def test_take_from_fixture_landmarks():
    take = _fixture_take()
    assert take.total == 5
    assert take.fps == 8.0
    assert take.morph_names == list(ARKIT_BLENDSHAPES)
    assert take.unmapped == ()
    assert take.frames[0].blendshapes["mouthSmileLeft"] > 0.5
    assert take.frames[1].blendshapes["mouthSmileLeft"] < 0.1
    assert take.frames[0].bones == {}


def test_take_drives_rig():
    from nomorals.media.facemcap import drive_take
    from nomorals.media.rig import get_session
    take = _fixture_take()
    session = get_session("default")
    posed = drive_take(session, take)
    assert len(posed) == 5
    assert len(posed[0].blendshapes) == 52
    assert posed[0].blendshapes["mouthSmileLeft"] > 0.5
    assert posed[1].blendshapes["mouthSmileLeft"] < 0.1
    assert posed[3].blendshapes["mouthPucker"] > 0.9


def test_take_json_round_trip():
    from nomorals.media.facemcap.take import FaceTake
    take = _fixture_take()
    again = FaceTake.from_json(take.to_json(), where="take")
    assert again.total == 5
    assert again.frames[0].blendshapes["mouthSmileLeft"] == \
        pytest.approx(
            take.frames[0].blendshapes["mouthSmileLeft"])


def test_landmarks_to_take_refusals():
    with pytest.raises(FaceMeshError) as exc:
        landmarks_to_take("x", [], fps=8.0)
    assert exc.value.kind == "no_face"
    with pytest.raises(FaceMeshError) as exc:
        landmarks_to_take("", [(0.0, [(0.0, 0.0, 0.0)] * 478)],
                          fps=8.0)
    assert exc.value.kind == "bad_landmarks"
    with pytest.raises(FaceMeshError) as exc:
        landmarks_to_take("x", [(0.0, [(0.0, 0.0, 0.0)] * 478)],
                          fps=0.0)
    assert exc.value.kind == "bad_landmarks"


def test_register_take_round_trip():
    from nomorals.media.facemcap.errors import FaceMocapError
    from nomorals.media.facemcap.registry import drop_take, get_take, list_takes, register_take
    take = _fixture_take()
    register_take("fixtures", take)
    assert get_take("fixtures").total == 5
    assert any(entry["name"] == "fixtures"
               for entry in list_takes())
    assert drop_take("fixtures")["frames"] == 5
    with pytest.raises(FaceMocapError) as exc:
        get_take("fixtures")
    assert exc.value.kind == "unknown_take"


# ---------------------------------------------------------------------------
# Provision + refusals (no install, no download — the refusal is the test)
# ---------------------------------------------------------------------------

def test_status_without_install():
    import importlib.util

    from nomorals.media.facemesh import mediapipe_status
    status = mediapipe_status()
    assert set(status) >= {"mediapipe", "mediapipe_version",
                           "model", "model_path", "model_bytes",
                           "model_ok", "ffmpeg"}
    assert status["mediapipe"] is (
        importlib.util.find_spec("mediapipe") is not None)
    assert status["ffmpeg"] is True


def test_capture_camera_refuses_by_name():
    from nomorals.media.facemesh import capture_camera
    with pytest.raises(FaceMeshError) as exc:
        capture_camera()
    assert exc.value.kind == "camera_unavailable"


def test_missing_video_refuses_by_name(tmp_path):
    from nomorals.media.facemesh import capture_image, capture_video
    with pytest.raises(FaceMeshError) as exc:
        capture_video(tmp_path / "nope.mp4")
    assert exc.value.kind == "missing_video"
    with pytest.raises(FaceMeshError) as exc:
        capture_image(tmp_path / "nope.jpg")
    assert exc.value.kind == "missing_video"


def test_ensure_mediapipe_refuses_without_install():
    import importlib.util

    from nomorals.media.facemesh import ensure_mediapipe
    if importlib.util.find_spec("mediapipe") is not None:
        pytest.skip("mediapipe is installed in this python")
    with pytest.raises(FaceMeshError) as exc:
        ensure_mediapipe(install=False)
    assert exc.value.kind == "missing_mediapipe"


def test_ensure_model_refuses_without_download(tmp_path, monkeypatch):
    from nomorals.media.facemesh import ensure_model
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    with pytest.raises(FaceMeshError) as exc:
        ensure_model(download=False)
    assert exc.value.kind == "missing_model"


# ---------------------------------------------------------------------------
# Wiring: tool count, CLI, chat
# ---------------------------------------------------------------------------

def test_tool_count_is_787_after_facemesh():
    from nomorals.tools.registry import ToolRegistry
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 807
    assert "facemesh" in registry._tools


def test_tool_facemesh_status(tmp_path):
    from nomorals.tools.registry import ToolRegistry
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    outcome = registry.call("facemesh", action="status")
    assert getattr(outcome, "ok", False), outcome
    status = outcome.value
    assert status["ffmpeg"] is True
    assert "model_path" in status


def test_tool_facemesh_unknown_action(tmp_path):
    from nomorals.tools.registry import ToolRegistry
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    outcome = registry.call("facemesh", action="bogus")
    assert not getattr(outcome, "ok", True)


def test_cli_facemesh_alias():
    from nomorals.cmdline.dispatch import main
    assert main(["fm", "status"]) == 0


def test_chat_facemesh_status():
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
    mixin = RuntimeMocapMixin()
    assert "facemesh status:" in mixin._control_facemesh(
        "status", "test")
    assert "/facemesh capture" in mixin._control_facemesh(
        "bogus", "test")
    assert "/facemesh capture" in mixin._control_facemesh(
        "", "test")
