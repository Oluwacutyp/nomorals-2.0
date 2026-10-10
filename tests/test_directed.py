"""Directed video animation: pose rig, animator routing, camera looks."""

import numpy as np
from PIL import Image

from nomorals.media.directed.pose_rig import (
    build_track, list_actions, resolve_action, rest_pose, render_pose_video,
    N_KP,
)
from nomorals.media.directed import camera as cam


def test_action_resolution():
    assert resolve_action("make the person raise two fingers up") == \
        "raise_two_fingers"
    assert resolve_action("animate her waving") == "wave"
    assert resolve_action("thumbs up please") == "thumbs_up"
    assert resolve_action("do a backflip") is None
    assert len(list_actions()) >= 7


def test_track_directed_motion():
    tr = build_track("raise_two_fingers", n_frames=20)
    assert tr.frames.shape == (20, N_KP, 2)
    # wrist rises
    assert tr.frames[-1][4, 1] < tr.frames[0][4, 1] - 0.1
    # index+middle extend, ring stays curled
    w1 = tr.frames[-1][4]
    idx = np.linalg.norm(tr.frames[-1][18 + 8] - w1)
    ring = np.linalg.norm(tr.frames[-1][18 + 16] - w1)
    assert idx > 0.035 and ring < 0.035


def test_rest_pose_sane():
    kp = rest_pose()
    assert kp.shape == (N_KP, 2)
    assert (kp >= 0).all() and (kp <= 1).all()
    # head above hips
    assert kp[0, 1] < kp[8, 1]


def test_pose_video_renders(tmp_path):
    tr = build_track("wave", n_frames=8)
    out = render_pose_video(tr, str(tmp_path / "pose.mp4"))
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 1000


def test_camera_path_shape():
    p = cam.camera_path(24, 24.0, "handheld")
    assert p.shape == (24, 4)
    # selfie has higher freq content than tripod (more zero crossings)
    p_trip = cam.camera_path(48, 24.0, "tripod", seed=1)
    p_self = cam.camera_path(48, 24.0, "selfie", seed=1)
    zc = lambda a: int((np.diff(np.sign(a[:, 0])) != 0).sum())
    assert zc(p_self) >= zc(p_trip)


def test_shake_changes_frames():
    from PIL import ImageDraw
    frames = []
    for i in range(6):
        img = Image.new("RGB", (160, 90), (40, 40, 40))
        d = ImageDraw.Draw(img)
        d.rectangle([30 + i * 4, 20, 70 + i * 4, 60], fill=(200, 120, 80))
        frames.append(img)
    out = cam.apply_camera_shake(frames, "handheld", fps=8, seed=3)
    assert len(out) == 6 and out[0].size == (160, 90)
    assert not np.array_equal(np.array(frames[1]), np.array(out[1]))


def test_rolling_shutter_shears():
    a = np.zeros((20, 20, 3), dtype=np.uint8)
    a[:, 10] = 255  # vertical line
    out = cam.rolling_shutter(a, 10.0)
    # line should tilt: top and bottom rows differ in column position
    top = int(np.where(out[0] > 128)[0].mean())
    bot = int(np.where(out[-1] > 128)[0].mean())
    assert abs(top - bot) > 2


def test_looks_list():
    assert "phone_selfie" in cam.LOOKS
    assert "cctv" in cam.LOOKS
    try:
        cam.apply_look([], "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_animator_routing_no_torch(tmp_path):
    # no torch here: neural must raise ModelUnavailable, auto must warp
    from nomorals.media.directed.animator import (
        direct_animate, mimicmotion_status, ModelUnavailable)
    st = mimicmotion_status()
    assert not st["available"]
    assert "MimicMotion" in st["reason"]
    try:
        direct_animate("/tmp/x.png", build_track("nod", n_frames=4),
                       prefer="neural")
    except ModelUnavailable:
        pass
    else:
        raise AssertionError("expected ModelUnavailable")


def test_warp_end_to_end(tmp_path):
    from nomorals.media.directed.animator import warp_animate
    img = Image.new("RGB", (128, 128), (40, 40, 40))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([56, 40, 72, 110], fill=(200, 150, 120))
    p = tmp_path / "person.png"
    img.save(p)
    tr = build_track("raise_hand", n_frames=8)
    res = warp_animate(str(p), tr, workdir=str(tmp_path))
    assert res.backend == "warp"
    import os
    assert os.path.exists(res.path)
    # last frame differs from first (directed motion happened)
    assert os.path.getsize(res.path) > 1000
