"""Video to landmarks — decode with ffmpeg, detect with
MediaPipe Face Landmarker.

The decode is the repo's own shape (an ffmpeg rawvideo pipe,
the same as ``media_edit.cv_video._iter_raw_frames`` — stdlib
only, no cv2). The detection is the MediaPipe Tasks API
(``FaceLandmarker`` in VIDEO mode): the legacy
``mp.solutions`` API is gone from mediapipe 0.10.35, so
nothing here touches it.

A frame with no face is SKIPPED, not zero-filled — a missing
face is not a neutral face, and a take must not claim a
neutral expression for a frame nobody was in. The capture
reports how many frames were skipped; a video with no face
at all is refused (kind ``no_face``), never an empty take.

Live camera is NOT here: no live-camera capture exists
anywhere in the repo (``VideoCapture`` is used only on file
paths), so there is no path to wire. It is the named next
rung — ``capture_camera`` raises ``camera_unavailable``
until one is built, rather than a stub that pretends.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..facemcap.take import FaceTake
from .errors import FaceMeshError
from .provision import ensure_mediapipe, ensure_model
from .take import Landmarks, landmarks_to_take

__all__ = [
    "CaptureResult",
    "VideoInfo",
    "capture_camera",
    "capture_image",
    "capture_video",
    "iter_video_frames",
    "probe_video",
]

#: Detection runs on frames scaled to at most this width —
#: 640 wide is plenty for 468 landmarks, and it keeps the
#: CPU pass fast.
MAX_DETECT_WIDTH = 640

_FFMPEG_INSTALL = (
    "ffmpeg is not installed. Install it — Debian/Ubuntu: "
    "`sudo apt install ffmpeg`; macOS: `brew install ffmpeg`; "
    "Windows: `winget install ffmpeg`; or "
    "https://ffmpeg.org/download.html.")


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _ffprobe_bin() -> str | None:
    return shutil.which("ffprobe")


@dataclass
class VideoInfo:
    """What ffprobe says about a video."""
    path: str
    width: int
    height: int
    fps: float


def probe_video(path: str | Path) -> VideoInfo:
    """Width, height, fps for a video file."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FaceMeshError(f"no video file at {target}",
                            kind="missing_video")
    ffprobe = _ffprobe_bin()
    if ffprobe is None:
        raise FaceMeshError(_FFMPEG_INSTALL,
                            kind="missing_ffmpeg")
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=width,height,r_frame_rate",
             "-of", "json", str(target)],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FaceMeshError(f"ffprobe failed on {target}: {exc}",
                            kind="bad_video") from exc
    try:
        blob = json.loads(proc.stdout or "{}")
        stream = blob["streams"][0]
        num, den = (int(part) for part in
                    stream["r_frame_rate"].split("/"))
        fps = num / den if den else 0.0
        return VideoInfo(path=str(target),
                         width=int(stream["width"]),
                         height=int(stream["height"]),
                         fps=fps)
    except (KeyError, IndexError, ValueError) as exc:
        raise FaceMeshError(
            f"ffprobe could not read {target} as a video",
            kind="bad_video") from exc


def iter_video_frames(path: str | Path, *, max_width: int = MAX_DETECT_WIDTH):
    """Yield ``(index, time_s, rgb, width, height)`` — raw
    RGB frames through an ffmpeg pipe, scaled to at most
    ``max_width`` wide. The fps comes from ffprobe; the time
    is the frame index over it."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FaceMeshError(f"no video file at {target}",
                            kind="missing_video")
    exe = _ffmpeg_bin()
    if not exe:
        raise FaceMeshError(_FFMPEG_INSTALL,
                            kind="missing_ffmpeg")
    info = probe_video(target)
    scale = min(1.0, max_width / info.width) if info.width else 1.0
    w = max(1, round(info.width * scale))
    h = max(1, round(info.height * scale))
    size = w * h * 3
    cmd = [exe, "-hide_banner", "-loglevel", "error",
           "-i", str(target),
           "-vf", f"scale={w}:{h}",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    index = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            time_s = index / info.fps if info.fps else 0.0
            yield index, time_s, bytes(buf), w, h
            index += 1
    finally:
        with contextlib.suppress(OSError):
            proc.stdout.close()
        proc.wait()
    if proc.returncode not in (0, None):
        raise FaceMeshError(
            f"ffmpeg failed decoding {target} "
            f"(rc={proc.returncode})",
            kind="bad_video")


@dataclass
class CaptureResult:
    """A capture run: the take plus what it cost."""
    take: FaceTake
    frames_read: int
    frames_with_face: int
    frames_skipped: int
    #: The first detected frame's raw landmarks, so a caller
    #: can inspect the geometry the take was built from.
    first_landmarks: Landmarks | None = None


def _landmarker(*, model_path: str, mode: str, install: bool = True):
    """A FaceLandmarker in IMAGE or VIDEO mode, or the reason
    it could not be built. The import failure on Linux is
    usually a missing native lib (libGLESv2/libEGL) — the
    message names it, because a bare ImportError hides it."""
    ensure_mediapipe(install=install)
    try:
        from mediapipe.tasks.python import BaseOptions  # noqa: PLC0415
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode  # noqa: PLC0415
    except ImportError as exc:
        raise FaceMeshError(
            "mediapipe imported but its Tasks vision API did "
            f"not: {exc} — on Linux this is usually a missing "
            "system library (libGLESv2.so.2 / libEGL.so.1); "
            "install libgles2 + libegl1",
            kind="missing_system_lib") from exc
    running = (RunningMode.VIDEO if mode == "VIDEO"
               else RunningMode.IMAGE)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True)
    try:
        return FaceLandmarker.create_from_options(options)
    except Exception as exc:  # noqa: BLE001 - mediapipe's
        # native errors are opaque; the message carries them.
        raise FaceMeshError(
            f"FaceLandmarker would not start: {exc}",
            kind="missing_model") from exc


def _mp_image(rgb: bytes, width: int, height: int):
    """Raw RGB bytes to a MediaPipe image."""
    import numpy as np  # noqa: PLC0415
    from mediapipe import Image as MpImage  # noqa: PLC0415
    from mediapipe import ImageFormat  # noqa: PLC0415
    arr = np.frombuffer(rgb, dtype=np.uint8).reshape(
        height, width, 3)
    return MpImage(image_format=ImageFormat.SRGB, data=arr)


def _matrix_of(result, index: int = 0):
    """The head matrix for face ``index``: in mediapipe
    0.10.35 it arrives as a plain 4x4 numpy array, not an
    object with ``.matrix_rows`` — take the array when that
    is what it is, and the rows when it is an object."""
    matrices = getattr(result, "facial_transformation_matrixes",
                       None)
    if not matrices or index >= len(matrices):
        return None
    entry = matrices[index]
    rows = getattr(entry, "matrix_rows", None)
    if rows is not None:
        return [[float(v) for v in row] for row in rows]
    return entry


def _landmarks_of(result, index: int = 0) -> Landmarks | None:
    faces = getattr(result, "face_landmarks", None)
    if not faces or index >= len(faces):
        return None
    return [(float(pt.x), float(pt.y), float(pt.z))
            for pt in faces[index]]


def capture_video(path: str | Path, *, name: str | None = None,
                  max_frames: int | None = None,
                  max_width: int = MAX_DETECT_WIDTH,
                  install: bool = True,
                  download: bool = True) -> CaptureResult:
    """A video file to a face take: decode, detect every
    frame, map the landmarks, register nothing — the caller
    names the take and keeps it. Frames with no face are
    skipped and counted; a video with no face at all is
    refused."""
    target = Path(path).expanduser()
    info = probe_video(target)
    model = ensure_model(download=download)
    landmarker = _landmarker(model_path=str(model), mode="VIDEO",
                            install=install)
    frames: list[tuple[float, Landmarks]] = []
    matrices: list[object | None] = []
    read = 0
    skipped = 0
    try:
        for _index, time_s, rgb, w, h in iter_video_frames(
                target, max_width=max_width):
            if max_frames is not None and read >= max_frames:
                break
            read += 1
            image = _mp_image(rgb, w, h)
            result = landmarker.detect_for_video(
                image, int(time_s * 1000))
            lm = _landmarks_of(result)
            if lm is None:
                skipped += 1
                continue
            frames.append((time_s, lm))
            matrices.append(_matrix_of(result))
    finally:
        # A landmarker left unclosed dies noisily at GC time
        # (the native dispatcher is already gone) — close it.
        landmarker.close()
    if not frames:
        raise FaceMeshError(
            f"no face found in any of {read} frames of "
            f"{target} — nothing to capture",
            kind="no_face")
    take = landmarks_to_take(
        name or target.stem, frames, matrices,
        fps=info.fps, source=str(target))
    return CaptureResult(take=take, frames_read=read,
                         frames_with_face=len(frames),
                         frames_skipped=skipped,
                         first_landmarks=frames[0][1])


def capture_image(path: str | Path, *, name: str | None = None,
                  install: bool = True,
                  download: bool = True) -> CaptureResult:
    """One still image to a one-frame face take (IMAGE mode).
    The rest baseline is the population default — a single
    frame has no video to calibrate from."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FaceMeshError(f"no image file at {target}",
                            kind="missing_video")
    from PIL import Image as PilImage  # noqa: PLC0415
    model = ensure_model(download=download)
    landmarker = _landmarker(model_path=str(model), mode="IMAGE",
                            install=install)
    with PilImage.open(target) as pil:
        rgb = pil.convert("RGB")
        w, h = rgb.size
        raw = rgb.tobytes()
    try:
        result = landmarker.detect(_mp_image(raw, w, h))
    finally:
        landmarker.close()
    lm = _landmarks_of(result)
    if lm is None:
        raise FaceMeshError(f"no face found in {target}",
                            kind="no_face")
    take = landmarks_to_take(
        name or target.stem, [(0.0, lm)],
        [_matrix_of(result)], fps=1.0, source=str(target))
    return CaptureResult(take=take, frames_read=1,
                         frames_with_face=1, frames_skipped=0,
                         first_landmarks=lm)


def capture_camera(*args, **kwargs):
    """Live camera capture — NOT BUILT. No live-camera path
    exists anywhere in the repo, so there is nothing to wire;
    this refuses by name (kind ``camera_unavailable``) rather
    than a stub that pretends to open a camera."""
    raise FaceMeshError(
        "live camera capture is not built — no camera path "
        "exists in the repo to wire it to; video files work "
        "now (facemesh capture <video>)",
        kind="camera_unavailable")
