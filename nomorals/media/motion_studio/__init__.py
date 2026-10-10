"""Devon Studio — motion studio: CPU-only procedural video.

Zero models, zero GPU: Ken Burns stills animation, audio-reactive
visualizers, kinetic typography, timeline montage, color grading, and
format exports. Runs on every profile including termux.

Neural video generation (LTX-Video / Wan, workstation-gated) lives in
:mod:`nomorals.media.videogen` — this package is the always-works tier
and the honest fallback when the GPU isn't there.
"""

from ._core import MotionStudioError, profile_defaults
from . import kenburns, visualizer, typography, montage, grading, studio

__all__ = [
    "MotionStudioError",
    "profile_defaults",
    "kenburns",
    "visualizer",
    "typography",
    "montage",
    "grading",
    "studio",
]
