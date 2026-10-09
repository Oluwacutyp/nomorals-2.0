"""Niche plugin package — registry, base classes, scaffold, shipped niches."""

from .base import NichePlugin, SceneVisual, ScriptResult, VisualPlan, VoiceSpec
from .registry import (
    NicheError,
    all_plugins,
    get,
    get_niche,
    list_niches,
    register,
    validate,
)
from .scaffold import scaffold

__all__ = [
    "NichePlugin",
    "SceneVisual",
    "ScriptResult",
    "VisualPlan",
    "VoiceSpec",
    "NicheError",
    "all_plugins",
    "get",
    "get_niche",
    "list_niches",
    "register",
    "scaffold",
    "validate",
]
