"""AniCraft port — a 2D animation editor organ in Devon's Python.

Ports storytold/anicraft (a clean-room Adobe Animate reimplementation
in pure Rust, MIT OR Apache-2.0): the document model (scenes, layers,
keyframes, classic/shape/motion tweens), the Penner easing families,
the pure timeline evaluator, the PIL frame renderer, the publish
formats (GIF/APNG/WebP/MP4/PNG sequence/spritesheet/SVG/HTML), the
XFL interchange subset, the preset library, and the ai-client seam
(ported from craft-ai-integrations' ai-client shape).
"""

from __future__ import annotations

from .ai import AiClient, ConsentLedger, Job, Role
from .attribution import verify_attribution
from .easing import Easing, list_easings
from .edit import AnimationEditor
from .errors import AnimationError
from .eval import DisplayItem, DisplayList, evaluate
from .library import build_preset, list_presets, search_presets
from .model import (
    AnimationDocument,
    Element,
    Keyframe,
    Layer,
    MotionTrack,
    MotionTween,
    Transform,
    Tween,
)
from .parity import run_parity
from .player import Player
from .publish import PUBLISH_FORMATS, publish
from .render import render_frame

__all__ = [
    "AiClient",
    "AnimationDocument",
    "AnimationEditor",
    "AnimationError",
    "ConsentLedger",
    "DisplayItem",
    "DisplayList",
    "Easing",
    "Element",
    "Job",
    "Keyframe",
    "Layer",
    "MotionTrack",
    "MotionTween",
    "PUBLISH_FORMATS",
    "Player",
    "Role",
    "Transform",
    "Tween",
    "build_preset",
    "evaluate",
    "list_easings",
    "list_presets",
    "publish",
    "render_frame",
    "run_parity",
    "search_presets",
    "verify_attribution",
]
