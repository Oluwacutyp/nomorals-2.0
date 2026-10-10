"""Directed video animation: pose/motion control + camera emulation.

Pipeline (from DIRECTED_MINING.md):

    action text -> pose program (parametric rig) -> pose video (skeleton)
                                                       |
    reference image --------------------------> MimicMotion / warp animator -> raw clip
                                                       |
                                                 camera emulation
                                                       |
                                                 edit timeline

- pose_rig.py   : parametric keypoint trajectories for directed actions.
- animator.py   : neural (MimicMotion) + CPU mesh-warp render backends.
- camera.py     : handheld shake, phone/cctv/dashcam/cinema looks (post).
"""
from __future__ import annotations

__all__ = ["pose_rig", "animator", "camera"]
