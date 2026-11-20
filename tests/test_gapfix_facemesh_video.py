"""FaceMesh video round trip — a REAL mp4 through the REAL
pipeline (B.25).

Runs ONLY under the mpvenv python (mediapipe 0.10.35): the
module-level ``importorskip`` skips the whole file on system
python, and a skip is never counted as a pass — prove.py
asserts these tests PASSED under the mpvenv python.

The fixture video (``tests/fixtures/facemesh/face-video.mp4``)
is 640x480, 8 fps, 48 frames, 6 seconds: segment A is a big
smile, segment B is neutral. The tests assert the take that
lands: 48 frames, the smile in A and not in B, the take
drives the rig, and the tool + CLI + chat verb capture it.
The model file is the one already downloaded for the
calibration (``~/workspace/facemesh-verify/face_landmarker.task``),
copied into a temp NM_HOME so the tests never touch the real
home — except ``test_ensure_model_downloads``, which exercises
the real download once.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

mediapipe = pytest.importorskip("mediapipe")

from nomorals.media.facemesh import capture_image, capture_video, ensure_model  # noqa: E402
from nomorals.media.facemesh.provision import MODEL_MIN_BYTES  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "facemesh"
VIDEO = FIXTURES / "face-video.mp4"
VERIFY_MODEL = (Path.home() / "workspace" / "facemesh-verify"
                / "face_landmarker.task")


@pytest.fixture(autouse=True)
def _nm_home(tmp_path, monkeypatch):
    """A temp NM_HOME with the model already in place, so
    capture works without a download on every test."""
    home = tmp_path / "home"
    models = home / "media" / "facemesh" / "models"
    models.mkdir(parents=True)
    shutil.copyfile(VERIFY_MODEL, models / "face_landmarker.task")
    monkeypatch.setenv("NM_HOME", str(home))
    from nomorals.media.facemcap.registry import reset_takes
    reset_takes()
    yield
    reset_takes()


def _mean(frames, key, before=None, after=None):
    rows = [f for f in frames
            if (before is None or f.time_s < before)
            and (after is None or f.time_s >= after)]
    return sum(f.blendshapes[key] for f in rows) / len(rows)


def test_video_to_take_round_trip():
    result = capture_video(VIDEO)
    take = result.take
    assert result.frames_read == 48
    # Frame 24 (t=3.0, the hard cut to segment B) has a face
    # in IMAGE mode but VIDEO mode's tracker loses it for
    # exactly that one frame, then re-acquires — the skip is
    # the organ working as designed, not a miss.
    assert result.frames_with_face == 47
    assert result.frames_skipped == 1
    assert take.total == 47
    assert take.fps == 8.0
    assert take.source.endswith("face-video.mp4")
    assert len(take.morph_names) == 52
    # Segment A (t < 3.0) smiles, segment B (t >= 3.0) does not.
    assert _mean(take.frames, "mouthSmileLeft", before=3.0) > 0.5
    assert _mean(take.frames, "mouthSmileRight", before=3.0) > 0.5
    assert _mean(take.frames, "mouthSmileLeft", after=3.0) < 0.3
    assert _mean(take.frames, "mouthSmileRight", after=3.0) < 0.3
    # Segment A's jaw opens further than B's (the smile
    # confound is named in the mapping — A still reads higher).
    assert _mean(take.frames, "jawOpen", before=3.0) > _mean(
        take.frames, "jawOpen", after=3.0)
    # The head bone is present on every frame.
    assert all("head" in f.bones for f in take.frames)
    assert all(len(f.bones["head"]) == 3 for f in take.frames)
    assert all(0.0 <= v <= 1.0
               for f in take.frames
               for v in f.blendshapes.values())


def test_video_take_json_round_trip():
    from nomorals.media.facemcap.take import FaceTake
    take = capture_video(VIDEO).take
    again = FaceTake.from_json(take.to_json(), where="take")
    assert again.total == 47
    assert again.frames[0].blendshapes["mouthSmileLeft"] == \
        pytest.approx(take.frames[0].blendshapes["mouthSmileLeft"])
    assert again.frames[-1].blendshapes["mouthSmileLeft"] == \
        pytest.approx(take.frames[-1].blendshapes["mouthSmileLeft"])


def test_video_take_drives_rig():
    from nomorals.media.facemcap import drive_take
    from nomorals.media.rig import get_session
    take = capture_video(VIDEO).take
    posed = drive_take(get_session("default"), take)
    assert len(posed) == 47
    assert posed[0].blendshapes["mouthSmileLeft"] > 0.5
    assert posed[-1].blendshapes["mouthSmileLeft"] < 0.3


def test_capture_image_from_video_frame(tmp_path):
    frame = tmp_path / "frame.png"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-ss", "1", "-i", str(VIDEO), "-frames:v", "1",
         str(frame)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    result = capture_image(frame)
    assert result.frames_with_face == 1
    assert result.take.total == 1
    # One second in is segment A — the smile.
    assert (result.take.frames[0].blendshapes["mouthSmileLeft"]
            > 0.5)


def test_capture_video_max_frames():
    result = capture_video(VIDEO, max_frames=6)
    assert result.frames_read == 6
    assert result.take.total == 6


def test_ensure_model_downloads(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("NM_HOME", str(home))
    path = ensure_model(download=True)
    assert path.is_file()
    assert path.stat().st_size >= MODEL_MIN_BYTES


def test_tool_facemesh_capture(tmp_path):
    from nomorals.media.facemcap.registry import list_takes
    from nomorals.tools.registry import ToolRegistry
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    outcome = registry.call(
        "facemesh", action="capture", node="video",
        sources=[str(VIDEO)])
    assert getattr(outcome, "ok", False), outcome
    value = outcome.value
    assert value["ok"] is True
    assert value["frames"] == 47
    assert value["frames_with_face"] == 47
    assert value["frames_skipped"] == 1
    assert any(entry["name"] == "video" for entry in list_takes())
    landmarks = registry.call(
        "facemesh", action="landmarks", sources=[str(VIDEO)])
    assert getattr(landmarks, "ok", False), landmarks
    assert len(landmarks.value["landmarks"]) == 478


def test_cli_facemesh_capture():
    from nomorals.cmdline.dispatch import main
    assert main(["fm", "capture", "--path", str(VIDEO),
                 "--session", "video"]) == 0
    assert main(["fc", "status", "--session", "video"]) == 0


def test_chat_facemesh_capture():
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
    mixin = RuntimeMocapMixin()
    captured = mixin._control_facemesh(
        f"capture {VIDEO} video", "test")
    assert "captured: 47 frames" in captured
    status = mixin._control_facemesh("status", "test")
    assert "facemesh status:" in status
    assert "mediapipe: 0.10.35" in status
