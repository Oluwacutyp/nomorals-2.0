"""The ``nomorals.media_edit`` package: god-tier image + video editing.

Pure Pillow image ops (:mod:`.images`), an ffmpeg video engine
(:mod:`.videos`), OpenCV frame-level video processing (:mod:`.cv_video`),
background jobs (:mod:`.jobs`), a deterministic
natural-language intent parser (:mod:`.intent`), the pro session layer
(:mod:`.studio` — EditStudio), and pluggable generative/AI instruction
edits (:mod:`.generate`).
"""

from __future__ import annotations

from .images import (
    MediaEditError,
    WatermarkSpec,
    edit_image,
    batch_edit,
    image_probe,
    load_image,
    save_image,
    apply_chain,
    validate_ops,
    locate_object,
    register_object_locator,
    clear_object_locators,
    OP_ALLOWLIST,
)
from .studio import EditStudio, build_template, list_templates, studio_presets
from .layers import LayerStack
# Registers the Pillow advanced image ops into the engine.
try:
    from . import cv_ops  # noqa: F401
    _cv_ops_error = None
except ImportError as _exc:  # zero-deps default: package imports fine
    _cv_ops_error = _exc
from .generate import (
    GenerativeBackend,
    GenerativeEditError,
    get_backend,
    backend_status,
)

_CV_VIDEO_NAMES = (
    "cv2_available",
    "cv_extract_frames",
    "grab_frame_at",
    "apply_filter_to_video",
    "create_timelapse",
    "stabilize_basic",
    "frame_diff_highlights",
    "slow_motion",
    "reverse_video",
    "boomerang",
    "find_blurry_frames",
    "motion_heatmap",
    "split_on_scenes",
    "kenburns",
    "slideshow",
    "chroma_key",
    "pip",
    "freeze_frame",
    "denoise_video",
    "CV_FILTERS",
    "CV_FILTER_BACKENDS",
)

try:
    from .cv_video import (  # noqa: F401
        cv2_available,
        extract_frames as cv_extract_frames,
        grab_frame_at,
        apply_filter_to_video,
        create_timelapse,
        stabilize_basic,
        frame_diff_highlights,
        slow_motion,
        reverse_video,
        boomerang,
        find_blurry_frames,
        motion_heatmap,
        split_on_scenes,
        kenburns,
        slideshow,
        chroma_key,
        pip,
        freeze_frame,
        denoise_video,
        FILTERS as CV_FILTERS,
        FILTER_BACKENDS as CV_FILTER_BACKENDS,
    )
    _cv_video_error = None
except ImportError as _exc:  # zero-deps default: package imports fine
    _cv_video_error = _exc


def __getattr__(name: str):
    # PEP 562: accessing a missing optional dependency raises the original
    # helpful ImportError instead of AttributeError.
    if name == "cv_ops" and _cv_ops_error is not None:
        raise _cv_ops_error
    if name in _CV_VIDEO_NAMES and _cv_video_error is not None:
        raise _cv_video_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MediaEditError",
    "WatermarkSpec",
    "edit_image",
    "batch_edit",
    "image_probe",
    "load_image",
    "save_image",
    "apply_chain",
    "validate_ops",
    "locate_object",
    "register_object_locator",
    "clear_object_locators",
    "OP_ALLOWLIST",
    "EditStudio",
    "build_template",
    "list_templates",
    "studio_presets",
    "LayerStack",
    "cv_ops",
    "GenerativeBackend",
    "GenerativeEditError",
    "get_backend",
    "backend_status",
    "cv2_available",
    "cv_extract_frames",
    "grab_frame_at",
    "apply_filter_to_video",
    "create_timelapse",
    "stabilize_basic",
    "frame_diff_highlights",
    "slow_motion",
    "reverse_video",
    "boomerang",
    "find_blurry_frames",
    "motion_heatmap",
    "split_on_scenes",
    "kenburns",
    "slideshow",
    "chroma_key",
    "pip",
    "freeze_frame",
    "denoise_video",
    "CV_FILTERS",
    "CV_FILTER_BACKENDS",
]
