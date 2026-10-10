"""Background removal via rembg — the dedicated segmentation module.

This is the single home for AI background removal. The older inline
rembg path in ``generate.op_bg_remove`` delegates here (see
``op_bg_remove_v2`` below); the chroma-key mode stays in generate.py
as the zero-dependency fallback.

Model selection is profile-aware:

- ``termux`` → ``u2netp`` (4.7 MB, runs on a phone)
- ``laptop`` / ``workstation`` → ``birefnet-general`` (quality default)

Pass ``model=`` explicitly to override. rembg is an optional dependency;
when it is missing the error names the pip install — never a silent
quality downgrade to chroma keying.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind

_log = get_logger(__name__)

#: model → (rembg model name, profile suitability, rough size)
MODELS = {
    "u2netp": {"size_mb": 4.7, "profiles": ("termux", "laptop", "workstation"),
               "note": "tiny, phone-friendly"},
    "u2net": {"size_mb": 176.0, "profiles": ("laptop", "workstation"),
              "note": "classic general-purpose"},
    "birefnet-general": {"size_mb": 800.0,
                         "profiles": ("laptop", "workstation"),
                         "note": "quality default, high detail"},
    "isnet-general-use": {"size_mb": 170.0,
                          "profiles": ("laptop", "workstation"),
                          "note": "DIS high-resolution variant"},
    "u2net_human_seg": {"size_mb": 176.0,
                        "profiles": ("laptop", "workstation"),
                        "note": "people — trained for human segmentation"},
    "birefnet-portrait": {"size_mb": 800.0,
                          "profiles": ("workstation",),
                          "note": "portraits — best hair/edge detail"},
    "isnet-anime": {"size_mb": 170.0,
                    "profiles": ("laptop", "workstation"),
                    "note": "anime/illustration line art"},
}

_SESSIONS: dict[str, Any] = {}


class SegmentationError(Exception):
    """Background removal unavailable or failed — the real reason."""


def list_models() -> list[dict[str, Any]]:
    """Registered rembg models with sizes and profile suitability."""
    return [{"name": name, **info} for name, info in MODELS.items()]


def default_model(kind: str = "") -> str:
    """Profile-aware default: tiny on termux, quality everywhere else."""
    kind = (kind or get_profile_kind()).strip().lower()
    return "u2netp" if kind == "termux" else "birefnet-general"


def _require_rembg() -> Any:
    try:
        import rembg  # noqa: F401
    except ImportError as exc:
        raise SegmentationError(
            "background removal needs rembg: pip install rembg "
            "(u2netp model is only ~4.7 MB)") from exc
    from rembg import new_session, remove
    return new_session, remove


def _session(model: str) -> tuple[Any, Any]:
    if model not in _SESSIONS:
        new_session, remove = _require_rembg()
        _log.info("loading rembg session: %s", model)
        _SESSIONS[model] = (new_session(model), remove)
    return _SESSIONS[model]


def remove_background(img: Any, *, model: str = "") -> Any:
    """Remove the background → RGBA image with transparent background.

    ``model``: a key of :data:`MODELS`; empty picks the profile default
    (u2netp on termux, birefnet-general elsewhere). Raises
    :class:`SegmentationError` with a pip hint when rembg is missing —
    never silently degrades.
    """
    from .images import _require_pillow
    Image = _require_pillow()
    model = (model or default_model()).strip().lower()
    if model not in MODELS:
        raise SegmentationError(
            f"unknown segmentation model {model!r}; use: {sorted(MODELS)}")
    info = MODELS[model]
    if get_profile_kind() not in info["profiles"]:
        _log.warning("model %s is heavy for this profile; continuing anyway",
                     model)
    session, remove = _session(model)
    try:
        out = remove(img.convert("RGB"), session=session)
    except Exception as exc:  # noqa: BLE001 - wrap with context
        raise SegmentationError(
            f"rembg ({model}) failed: {exc}") from exc
    if isinstance(out, Image.Image):
        return out.convert("RGBA")
    return Image.fromarray(out).convert("RGBA")


def op_bg_remove_v2(img: Any, *, model: str = "") -> Any:
    """Chain op: AI background removal (rembg, profile-aware default).

    Registered as ``bg_remove_v2``; the legacy ``bg_remove`` op in
    generate.py keeps its modes and delegates its ``rembg``/``auto``
    path here.
    """
    return remove_background(img, model=model)
