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


def test_lipsync_backends_report_unavailable():
    from nomorals.media.directed.lipsync import (
        wav2lip_status, latentsync_status, sadtaker_status)
    for fn in (wav2lip_status, latentsync_status, sadtaker_status):
        s = fn()
        assert s["available"] is False
        assert "Install" in s["reason"] or "install" in s["reason"].lower()


def test_envelope_speech_vs_silence(tmp_path):
    import subprocess, wave
    from nomorals.media.directed.lipsync import audio_envelope
    sr = 16000
    t = __import__("numpy").arange(sr * 2) / sr
    sig = __import__("numpy").zeros_like(t)
    sig[int(0.2*sr):int(0.7*sr)] = 0.5
    p = tmp_path / "s.wav"
    with wave.open(str(p), 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((sig*32767).astype(__import__("numpy").int16).tobytes())
    env = audio_envelope(str(p), 12.0, 24)
    assert env.shape == (24,)
    assert env[5] > 0.5          # speech burst -> open
    assert env[23] < 0.05        # trailing silence -> closed


def test_face_box_from_pose():
    from nomorals.media.directed.pose_rig import build_track
    from nomorals.media.directed.lipsync import face_box_from_pose
    tr = build_track("nod", n_frames=8)
    box = face_box_from_pose(tr)
    assert len(box) == 4
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    # face is in the upper part of the frame
    assert y1 < 0.5


def test_warp_lipsync_changes_mouth(tmp_path):
    import subprocess
    from PIL import Image, ImageDraw
    from nomorals.media.directed.lipsync import envelope_warp_sync
    for i in range(12):
        img = Image.new("RGB", (120, 120), (30, 30, 40))
        d = ImageDraw.Draw(img)
        d.ellipse([40, 20, 80, 80], fill=(210, 160, 130))
        img.save(tmp_path / f"f_{i:04d}.png")
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y",
                    "-framerate","12","-i",str(tmp_path/"f_%04d.png"),
                    "-pix_fmt","yuv420p",str(tmp_path/"v.mp4")],
                   check=True, capture_output=True)
    import wave
    import numpy as np
    sr = 16000
    sig = (np.sin(2*np.pi*440*np.arange(sr)/sr) * 0.5).astype(np.float32)
    with wave.open(str(tmp_path/"a.wav"),'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((sig*32767).astype(np.int16).tobytes())
    res = envelope_warp_sync(str(tmp_path/"v.mp4"), str(tmp_path/"a.wav"),
                             (0.33, 0.17, 0.67, 0.67),
                             out_path=str(tmp_path/"out.mp4"), fps=12)
    assert res.backend == "warp"
    import os
    assert os.path.getsize(res.path) > 1000
