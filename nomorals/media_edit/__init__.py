"""The ``nomorals.media_edit`` package: god-tier image + video editing.

Pure Pillow image ops (:mod:`.images`), an ffmpeg video engine
(:mod:`.videos`), background jobs (:mod:`.jobs`), a deterministic
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
from .generate import (
    GenerativeBackend,
    GenerativeEditError,
    get_backend,
    backend_status,
)

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
    "GenerativeBackend",
    "GenerativeEditError",
    "get_backend",
    "backend_status",
]
