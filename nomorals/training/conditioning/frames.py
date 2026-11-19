"""Video frame decoding and preprocessing — the pixel half of the dataset.

Upstream decodes with ``decord`` and resizes with ``torchvision``;
neither is installed on this box (and decord is a frequent source of
the segfaults upstream's own comment warns about). This module does the
same job with tools Devon already has: ``ffprobe``/``ffmpeg`` for
decode, Pillow for resize, numpy for the tensor. The math is upstream's:

* frame sampling — ``sample_frame_indices`` (see
  :mod:`nomorals.training.conditioning.buckets`);
* resize to the nearest resolution bucket (upstream:
  ``torchvision.transforms.functional.resize``, bilinear);
* scale ``x / 255`` then normalise ``(x - 0.5) / 0.5`` — i.e. the
  tensor ends in ``[-1, 1]``, exactly upstream's
  ``scale_transform`` + ``Normalize(mean=[0.5]*3, std=[0.5]*3)``.

The result is a float32 ``[F, C, H, W]`` array — the same shape
upstream's ``ImageOrVideoDataset.__getitem__`` returns — ready for a
VAE. No torch is needed for any of this; torch only enters when the
array is handed to an encoder (see
:class:`nomorals.training.conditioning.encoders.RealEncoders`).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.logging_setup import get_logger
from .buckets import find_nearest_resolution, frame_bucket_for, sample_frame_indices
from .constants import DEFAULT_VIDEO_RESOLUTION_BUCKETS
from .errors import ConditioningError, ConditioningUnavailable

__all__ = [
    "decode_video_frames",
    "preprocess_video",
    "probe_video",
]

_log = get_logger(__name__)


def _run(command: list[str], *, what: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command, capture_output=True, check=False, timeout=120
        )
    except FileNotFoundError as exc:
        raise ConditioningUnavailable(
            f"{what} needs the '{command[0]}' binary, which is not on "
            f"PATH on this machine."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ConditioningError(
            f"{what} timed out after 120s: {' '.join(command)}"
        ) from exc


def probe_video(path: str | Path) -> dict[str, int | float]:
    """Width, height, frame count, and duration of a video file.

    Uses ``ffprobe``; the frame count falls back to duration × frame
    rate when the container does not store ``nb_frames`` (common for
    mp4). Raises :class:`ConditioningError` naming the file when the
    probe fails.
    """
    path = Path(path)
    if not path.is_file():
        raise ConditioningError(f"video file does not exist: {path}")
    result = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of", "json",
            str(path),
        ],
        what=f"probing {path}",
    )
    if result.returncode != 0:
        raise ConditioningError(
            f"ffprobe failed on {path}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    try:
        info = json.loads(result.stdout.decode("utf-8"))
        stream = info["streams"][0]
    except (ValueError, KeyError, IndexError) as exc:
        raise ConditioningError(
            f"ffprobe returned no video stream for {path}"
        ) from exc
    width = int(stream["width"])
    height = int(stream["height"])
    frames = stream.get("nb_frames")
    if frames not in (None, "N/A"):
        num_frames = int(frames)
    else:
        num, den = (int(part) for part in stream["r_frame_rate"].split("/"))
        duration = float(stream.get("duration", 0) or 0)
        num_frames = max(1, round(duration * num / den))
    return {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "duration": float(stream.get("duration", 0) or 0),
    }


def decode_video_frames(
    path: str | Path, *, indices: list[int] | None = None
) -> np.ndarray:
    """Decode a video to an RGB ``[F, H, W, C]`` uint8 array.

    ``ffmpeg`` decodes the whole clip to raw RGB24; ``indices`` (from
    :func:`buckets.sample_frame_indices`) selects the frames to keep.
    """
    path = Path(path)
    info = probe_video(path)
    result = _run(
        [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        what=f"decoding {path}",
    )
    if result.returncode != 0:
        raise ConditioningError(
            f"ffmpeg failed on {path}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    frame_bytes = info["width"] * info["height"] * 3
    raw = result.stdout
    if len(raw) < frame_bytes:
        raise ConditioningError(
            f"ffmpeg decoded no frames from {path} "
            f"({len(raw)} bytes, expected at least {frame_bytes})"
        )
    count = len(raw) // frame_bytes
    frames = np.frombuffer(raw[: count * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(count, info["height"], info["width"], 3)
    if indices is not None:
        frames = frames[indices]
    return frames


def preprocess_video(
    path: str | Path,
    *,
    resolution_buckets: list[tuple[int, int, int]] | None = None,
    max_num_frames: int | None = None,
) -> np.ndarray:
    """Decode + sample + resize + normalise one video, upstream's way.

    Returns float32 ``[F, C, H, W]`` in ``[-1, 1]``. The frame bucket is
    picked from the clip's real frame count
    (:func:`buckets.frame_bucket_for`); frames are resized to the
    nearest bucket resolution (:func:`buckets.find_nearest_resolution`,
    Pillow bilinear — upstream's ``torchvision`` default). ``max_num_frames``
    defaults to the largest frame bucket, exactly as upstream's resizing
    subclass derives it.
    """
    buckets = resolution_buckets or list(DEFAULT_VIDEO_RESOLUTION_BUCKETS)
    max_frames = max_num_frames or max(bucket[0] for bucket in buckets)
    info = probe_video(path)
    frame_bucket = frame_bucket_for(info["num_frames"], buckets, max_frames)
    indices = sample_frame_indices(info["num_frames"], frame_bucket)
    raw = decode_video_frames(path, indices=indices)
    target_h, target_w = find_nearest_resolution(
        buckets, info["height"], info["width"]
    )
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(
                    (target_w, target_h), Image.BILINEAR
                )
            )
            for frame in raw
        ],
        axis=0,
    )
    # Upstream: scale x/255 then Normalize(mean=0.5, std=0.5) => x*2-1.
    video = resized.astype(np.float32) / 255.0
    video = (video - 0.5) / 0.5
    # [F, H, W, C] -> [F, C, H, W]
    return np.ascontiguousarray(video.transpose(0, 3, 1, 2))
