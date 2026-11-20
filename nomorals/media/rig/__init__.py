"""VTuber rig — Devon's own live avatar rig (HyperJail pattern).

The study (storytold/hyperjail, pattern only, no upstream code):
a UE5 VTubing experiment whose project is an empty shell — the rig
IS the configuration: body mocap in (Xsens MVN / Rokoko via
LiveLink), face capture in (ARKit blendshapes), voice in (OVR
LipSync visemes), a named bone-remap table between the tracking
skeleton and the avatar skeleton, and the driven model out.

Devon's rig is the same shape in Python, wired into the organs
that already produce those inputs: the mocap organ's BVH
skeletons and state store, the directed lip-sync organ's visemes,
and the LiveKit video session's frame sources. A rig is data
(:mod:`rig`), a remap is data (:mod:`remap`), the driver poses
one frame at a time (:mod:`driver`), the renderer draws the posed
rig as RGB frames (:mod:`render`), and a session drives whole
takes and hands them to the sinks (:mod:`session`).
"""

from __future__ import annotations

from .driver import FaceInput, PosedFrame, RigDriver
from .errors import RigError
from .registry import (
    get_session,
    list_sessions,
    load_session,
    reset_sessions,
)
from .remap import BoneRemap, RemapResult, default_remap
from .render import render_pose
from .rig import (
    ARKIT_BLENDSHAPES,
    Bone,
    RigDefinition,
    default_rig,
)
from .session import (
    RigSession,
    load_take_bvh,
    load_take_json,
    take_from_store,
)

__all__ = [
    "ARKIT_BLENDSHAPES",
    "Bone",
    "BoneRemap",
    "FaceInput",
    "PosedFrame",
    "RemapResult",
    "RigDefinition",
    "RigDriver",
    "RigError",
    "RigSession",
    "default_remap",
    "default_rig",
    "get_session",
    "list_sessions",
    "load_session",
    "load_take_bvh",
    "load_take_json",
    "render_pose",
    "reset_sessions",
    "take_from_store",
]
