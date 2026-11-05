"""Devon Studio — motion studio: CPU-only procedural video.

Zero models, zero GPU: Ken Burns stills animation, audio-reactive
visualizers, kinetic typography, code-driven explainers (diagrams,
equations-as-text, chart builds — plan §1.5d), timeline montage,
color grading, and format exports. Runs on every profile including
termux.

Neural video generation (LTX-Video / Wan, workstation-gated) lives in
:mod:`nomorals.media.videogen` — this package is the always-works tier
and the honest fallback when the GPU isn't there.
"""

from ._core import MotionStudioError, profile_defaults
from . import kenburns, visualizer, typography, montage, grading, studio
from . import explainers, project
from .project import MotionProject, MotionSegment, PropertyTrack

__all__ = [
    "MotionStudioError",
    "profile_defaults",
    "kenburns",
    "visualizer",
    "typography",
    "montage",
    "grading",
    "studio",
    "explainers",
    "project",
    "MotionProject",
    "MotionSegment",
    "PropertyTrack",
]
