"""Devon the Director — coverage → breakdown → design → shoot → QA.

The director layer above :mod:`nomorals.media.videogen` (the shipped
V0/V1 registry + pipeline): a script goes in and a *production*
comes out — professional coverage, a scene/shot breakdown, lighting
and camera design per shot composed from primitives (style-agnostic;
style packs are data), generation through the existing videogen
registry machinery, and a frame/Seer QA loop with one flagged
regeneration per failed shot.

Stages are separate, persisted calls (``Director.cover`` /
``plan`` / ``shoot`` / ``regen``) so the owner gates each one
conversationally. This folds into videogen V3 (story→movie) as its
front half; the temporal WatchVideo organ remains a later build —
QA here is still-image judgment over sampled frames, and says so.
"""

from .errors import DirectorError
from .screenplay import Element, Scene, Screenplay, parse_screenplay
from .coverage import Assessment, Coverage, cover_script
from .breakdown import Shot, break_down, break_down_prose
from .design import (
    DesignSpec, CameraSpec, LightingSpec, STYLE_PACKS,
    design_shot, get_style_pack, refine_design,
)
from .prompts import assemble_prompt
from .qa import qa_shot
from .production import Director, Production

__all__ = [
    "DirectorError",
    "Element", "Scene", "Screenplay", "parse_screenplay",
    "Assessment", "Coverage", "cover_script",
    "Shot", "break_down", "break_down_prose",
    "DesignSpec", "CameraSpec", "LightingSpec", "STYLE_PACKS",
    "design_shot", "get_style_pack", "refine_design",
    "assemble_prompt", "qa_shot",
    "Director", "Production",
]
