"""Face swap via INSwapper (insightface ONNX) + GFPGAN restoration.

``swap_face(source_img, target_img)`` puts the face from ``source_img``
onto the face in ``target_img``. Both libraries are optional and lazy:
missing them raises a clear pip hint, never a silent skip.

Safety contract (not policy theater — engineering correctness):
- Faces are DETECTED in both images. No face in either → clear error.
  The swap never runs on a guessed region.
- Multiple faces in the target → the largest face is swapped (documented),
  or pass ``target_index=`` to choose.
- Profile-gate: laptop/workstation only. On termux this raises a clear
  "needs a laptop/workstation" error — INSwapper wants real RAM/CPU.

Weights (via ``_model_cache``, first run downloads):
- inswapper_128.onnx ~530 MB
- buffalo_l face-analysis pack (insightface downloads itself)
- GFPGANv1.4.pth ~350 MB (only when restore=True)
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind
from . import _model_cache as mc

_log = get_logger(__name__)

INSWAPPER_URL = ("https://github.com/facefusion/facefusion-assets/"
                 "releases/download/models-3.0.0/inswapper_128.onnx")
INSWAPPER_FILE = "inswapper_128.onnx"
INSWAPPER_SIZE_MB = 530.0

GFPGAN_URL = ("https://github.com/TencentARC/GFPGAN/releases/download/"
              "v1.3.0/GFPGANv1.4.pth")
GFPGAN_FILE = "GFPGANv1.4.pth"
GFPGAN_SIZE_MB = 350.0

_ANALYZER: Any = None
_SWAPPER: Any = None
_RESTORER: Any = None


class FaceSwapError(Exception):
    """Face swap unavailable or failed — the real reason."""


def _gate_profile() -> None:
    if get_profile_kind() == "termux":
        raise FaceSwapError(
            "face swap needs a laptop/workstation (insightface + ONNX "
            "want real RAM/CPU); termux can't run it honestly")


def _analyzer() -> Any:
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise FaceSwapError(
            "face swap needs insightface: pip install insightface "
            "onnxruntime") from exc
    try:
        import torch  # noqa: F401
    except ImportError:
        pass  # insightface works CPU-only via onnxruntime
    _gate_profile()
    _log.info("loading insightface buffalo_l analyzer")
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as exc:  # noqa: BLE001 - wrap with context
        raise FaceSwapError(f"face analyzer failed to load: {exc}") from exc
    _ANALYZER = app
    return app


def _swapper() -> Any:
    global _SWAPPER
    if _SWAPPER is not None:
        return _SWAPPER
    try:
        from insightface.model_zoo import get_model
    except ImportError as exc:
        raise FaceSwapError(
            "face swap needs insightface: pip install insightface "
            "onnxruntime") from exc
    _gate_profile()
    path = mc.model_path(INSWAPPER_FILE, url=INSWAPPER_URL,
                         size_mb=INSWAPPER_SIZE_MB)
    _log.info("loading INSwapper from %s", path)
    try:
        _SWAPPER = get_model(str(path),
                             providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - wrap with context
        raise FaceSwapError(f"INSWapper failed to load: {exc}") from exc
    return _SWAPPER


def _restorer() -> Any:
    global _RESTORER
    if _RESTORER is not None:
        return _RESTORER
    try:
        from gfpgan import GFPGANer
    except ImportError as exc:
        raise FaceSwapError(
            "face restoration needs gfpgan: pip install gfpgan "
            "(or pass restore=False to skip restoration)") from exc
    _gate_profile()
    path = mc.model_path(GFPGAN_FILE, url=GFPGAN_URL,
                         size_mb=GFPGAN_SIZE_MB)
    _log.info("loading GFPGAN from %s", path)
    try:
        _RESTORER = GFPGANer(model_path=str(path), upscale=1,
                             arch="clean", channel_multiplier=2)
    except Exception as exc:  # noqa: BLE001 - wrap with context
        raise FaceSwapError(f"GFPGAN failed to load: {exc}") from exc
    return _RESTORER


def _faces(img: Any) -> list[Any]:
    """Detect faces (BGR numpy). Raises when none are found."""
    import numpy as np
    app = _analyzer()
    bgr = np.asarray(img.convert("RGB"))[:, :, ::-1]
    try:
        faces = app.get(bgr)
    except Exception as exc:  # noqa: BLE001 - wrap with context
        raise FaceSwapError(f"face detection failed: {exc}") from exc
    if not faces:
        raise FaceSwapError(
            "no face detected in the image — face swap needs a visible "
            "face in both the source and the target")
    return faces


def _biggest(faces: list[Any]) -> Any:
    def area(f: Any) -> float:
        x1, y1, x2, y2 = f.bbox
        return float((x2 - x1) * (y2 - y1))
    return max(faces, key=area)


def swap_face(source_img: Any, target_img: Any, *,
              restore: bool = True, target_index: int = -1) -> Any:
    """Swap the face from ``source_img`` onto ``target_img`` → PIL RGB.

    ``restore=True`` runs GFPGAN on the result for a clean finish.
    ``target_index`` picks which target face when several are detected
    (-1 = the largest). Clear errors when insightface/gfpgan are missing
    or when either image has no detectable face.
    """
    from .images import _require_pillow
    Image = _require_pillow()
    _gate_profile()
    import numpy as np

    src_faces = _faces(source_img)
    tgt_faces = _faces(target_img)
    src_face = _biggest(src_faces)
    if target_index == -1:
        tgt_face = _biggest(tgt_faces)
    elif 0 <= target_index < len(tgt_faces):
        tgt_face = tgt_faces[target_index]
    else:
        raise FaceSwapError(
            f"target_index {target_index} out of range "
            f"({len(tgt_faces)} face(s) detected)")

    swapper = _swapper()
    tgt_bgr = np.asarray(target_img.convert("RGB"))[:, :, ::-1].copy()
    try:
        swapped = swapper.get(tgt_bgr, tgt_face, src_face, paste_back=True)
    except Exception as exc:  # noqa: BLE001 - wrap with context
        raise FaceSwapError(f"INSWapper failed: {exc}") from exc

    if restore:
        try:
            restorer = _restorer()
            _, _, restored = restorer.enhance(
                swapped, has_aligned=False, only_center_face=False,
                paste_back=True)
            swapped = restored
        except FaceSwapError:
            raise
        except Exception as exc:  # noqa: BLE001 - restoration is best-effort
            _log.warning("GFPGAN restoration failed (%s); returning the "
                         "unrestored swap", exc)
    return Image.fromarray(swapped[:, :, ::-1])


def op_faceswap(img: Any, *, source: Any, restore: bool = True) -> Any:
    """Chain op: swap the face from ``source`` onto ``img``.

    Registered as ``faceswap``. ``source``: PIL image or image path.
    """
    from .images import load_image
    from pathlib import Path
    src = load_image(source) if isinstance(source, (str, Path)) else source
    return swap_face(src, img, restore=restore)
