"""EffectCraft port — Devon's motion graphics compositor.

Compositions, layers, keyframes, raster effects, safe value
expressions, and a 3D plate camera (the scene3d rung that completes
the motion_studio ``camera3d`` group). The upstream dual JS
surfaces (expr value expressions as a JS runtime, script project
scripting) were deliberately NOT ported — see PARITY.md §1.3b:
expressions live here as a safe AST language, and scripting is
the brain driving ToolSpecs.
"""

from __future__ import annotations

from .camera import Camera3D, PerspectiveCameraRenderer, register_camera_rung
from .composition import Composition, Layer, render_composition
from .effects import apply_effect, list_effects
from .keyframes import PropertyTrack

__all__ = [
    "Camera3D",
    "Composition",
    "Layer",
    "PerspectiveCameraRenderer",
    "PropertyTrack",
    "apply_effect",
    "list_effects",
    "register_camera_rung",
    "render_composition",
]
