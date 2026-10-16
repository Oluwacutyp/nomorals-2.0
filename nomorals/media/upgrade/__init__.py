"""MEDIA upgrade tiers — Tier 0/1/2 + project-room extras.

Additive package: nothing here overwrites Step 7 Media files or any
existing organ. Everything is deterministic and offline unless a caller
supplies a real model hook (generation, embedding, face recognition).
Where a measurement needs a model that is not present, the function
computes a real, documented image/audio feature proxy and says so in its
docstring — it never invents a model score.

One invariant per capability (see tests/test_media_upgrade.py):
no output below the quality floor, consistency measured not assumed,
text composited never generated, render reproducible from recipe.
"""

from .bible import (
    CONSISTENCY_TARGET,
    ConsistencyReport,
    SceneBible,
    drift_report,
    face_similarity,
    palette_drift,
    segment_needs_rerender,
)
from .camera import CameraMove, MotionBrush, camera_plan, keyframe_plan
from .conditioning import Reference, ReferenceConditioning, conditioning_pack
from .critique import (
    QUALITY_FLOOR,
    Claim,
    Critique,
    brief_to_checklist,
    critique_render,
    passes_floor,
)
from .failures import FAILURE_MODES, FailureFinding, detect_failures, remedy_for
from .masks import Mask, derive_mask, geometric_mask, mask_from_prompt
from .nigerian import NIGERIAN_PACKS, nigerian_pack, nigerian_prompt
from .opstack import EditOp, OpStack, apply_stack_to_spec
from .pipeline import QUALITY_LADDER, PipelineResult, contact_sheet, run_pipeline
from .projects import (
    Asset,
    AssetLibrary,
    BudgetLedger,
    ProjectRoom,
    ReviewRoom,
    auto_highlights,
    beat_times,
    silence_spans,
    thumbnail_ab,
)
from .recipes import Recipe, RecipeHistory, recipe_from_spec
from .regional import AdjustmentStack, inpaint_spec, outpaint_spec, regional_prompt
from .routing import ModelRoute, RouteTable, route_for_task
from .taste_model import TasteModel, keep_or_discard
from .typography import TextComposite, composite_text

__all__ = [
    "QUALITY_FLOOR", "Claim", "Critique", "brief_to_checklist",
    "critique_render", "passes_floor",
    "Mask", "derive_mask", "geometric_mask", "mask_from_prompt",
    "Reference", "ReferenceConditioning", "conditioning_pack",
    "EditOp", "OpStack", "apply_stack_to_spec",
    "QUALITY_LADDER", "PipelineResult", "contact_sheet", "run_pipeline",
    "CONSISTENCY_TARGET", "ConsistencyReport", "SceneBible",
    "drift_report", "face_similarity", "palette_drift",
    "segment_needs_rerender",
    "CameraMove", "MotionBrush", "camera_plan", "keyframe_plan",
    "FAILURE_MODES", "FailureFinding", "detect_failures", "remedy_for",
    "NIGERIAN_PACKS", "nigerian_pack", "nigerian_prompt",
    "Recipe", "RecipeHistory", "recipe_from_spec",
    "AdjustmentStack", "inpaint_spec", "outpaint_spec", "regional_prompt",
    "ModelRoute", "RouteTable", "route_for_task",
    "TasteModel", "keep_or_discard",
    "TextComposite", "composite_text",
    "AssetLibrary", "Asset", "BudgetLedger", "ProjectRoom", "ReviewRoom",
    "auto_highlights", "beat_times", "silence_spans", "thumbnail_ab",
]
