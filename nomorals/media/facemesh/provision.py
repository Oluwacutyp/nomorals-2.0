"""Self-provision — mediapipe + the face-landmarker model.

Devon installs her own runtimes: a missing mediapipe is
installed on first use, and the model file is downloaded on
first use. Either step can fail — network down, no disk, pip
refused — and then the organ refuses BY NAME with the reason
(``FaceMeshError`` kind ``missing_mediapipe`` /
``provision_failed`` / ``missing_model``), never a silent
pass and never a stub. A caller that passes
``install=False`` / ``download=False`` gets the refusal
without an install attempt.

Model: MediaPipe Face Landmarker, float16, from Google's
public model storage — free, Apache-2.0. The install is
``mediapipe>=0.10.14,<0.11``: the Tasks API (FaceLandmarker)
is the supported path; the legacy ``mp.solutions`` API is
gone from 0.10.35, so nothing here touches it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .errors import FaceMeshError

__all__ = [
    "MODEL_FILE_NAME",
    "MODEL_MIN_BYTES",
    "MODEL_URL",
    "ensure_mediapipe",
    "ensure_model",
    "facemesh_home",
    "mediapipe_status",
    "model_path",
]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/"
             "face_landmarker.task")
MODEL_FILE_NAME = "face_landmarker.task"
#: The real model is 3,758,596 bytes; anything far under that
#: is a truncated download, not a model.
MODEL_MIN_BYTES = 3_000_000
MEDIAPIPE_SPEC = "mediapipe>=0.10.14,<0.11"
INSTALL_TIMEOUT_S = 600
DOWNLOAD_TIMEOUT_S = 300


def facemesh_home() -> Path:
    """``{NM_HOME or ~/.nomorals}/media/facemesh``."""
    home = os.path.expanduser(
        os.environ.get("NM_HOME", "~/.nomorals"))
    return Path(home) / "media" / "facemesh"


def model_path() -> Path:
    """Where the face-landmarker model file lives."""
    return facemesh_home() / "models" / MODEL_FILE_NAME


def ensure_mediapipe(*, install: bool = True):
    """Return the imported mediapipe module, installing it
    first when it is missing and installs are allowed.

    Raises FaceMeshError (kind ``missing_mediapipe`` when
    installs are refused, ``provision_failed`` when the
    install itself fails)."""
    try:
        import mediapipe  # noqa: PLC0415
    except ImportError:
        if not install:
            raise FaceMeshError(
                "mediapipe is not installed and installs are "
                "refused — run `nm facemesh status` to see "
                "the install command",
                kind="missing_mediapipe") from None
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 MEDIAPIPE_SPEC],
                capture_output=True, text=True,
                timeout=INSTALL_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FaceMeshError(
                f"mediapipe install failed: {exc}",
                kind="provision_failed") from exc
        if proc.returncode != 0:
            raise FaceMeshError(
                "mediapipe install failed: "
                f"{proc.stderr.strip()[-500:]}",
                kind="provision_failed") from None
        try:
            import mediapipe  # noqa: PLC0415
        except ImportError as exc:
            raise FaceMeshError(
                "mediapipe installed but still not importable: "
                f"{exc}",
                kind="provision_failed") from exc
    return mediapipe


def ensure_model(*, download: bool = True) -> Path:
    """Return the face-landmarker model path, downloading it
    first when it is missing and downloads are allowed.

    Raises FaceMeshError (kind ``missing_model`` when
    downloads are refused or the download is bad)."""
    path = model_path()
    if path.is_file() and path.stat().st_size >= MODEL_MIN_BYTES:
        return path
    if not download:
        raise FaceMeshError(
            f"face-landmarker model is not at {path} and "
            "downloads are refused — run `nm facemesh status`",
            kind="missing_model")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".task.tmp")
    try:
        with urllib.request.urlopen(
                MODEL_URL, timeout=DOWNLOAD_TIMEOUT_S) as resp, \
                open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except OSError as exc:
        raise FaceMeshError(
            f"face-landmarker model download failed: {exc}",
            kind="missing_model") from exc
    if tmp.stat().st_size < MODEL_MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise FaceMeshError(
            "face-landmarker model download is truncated "
            f"({tmp.stat().st_size} bytes) — not a model",
            kind="missing_model")
    tmp.replace(path)
    return path


def mediapipe_status() -> dict[str, object]:
    """Availability, without installing or downloading
    anything: mediapipe importable + version, model present +
    size, ffmpeg present."""
    try:
        import mediapipe  # noqa: PLC0415
        mediapipe_version = getattr(mediapipe, "__version__",
                                    "unknown")
    except ImportError:
        mediapipe_version = None
    path = model_path()
    return {
        "mediapipe": mediapipe_version is not None,
        "mediapipe_version": mediapipe_version,
        "model": path.is_file(),
        "model_path": str(path),
        "model_bytes": (path.stat().st_size
                        if path.is_file() else 0),
        "model_ok": (path.is_file()
                     and path.stat().st_size >= MODEL_MIN_BYTES),
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
