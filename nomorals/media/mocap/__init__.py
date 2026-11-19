"""Cross-engine mocap state for Devon.

One shared store every engine reads and writes (``state``), a router
that picks the engine for a category and fails honestly when it is
not configured (``router``), BVH read/write for the mocap
interchange format game engines read (``bvh``), and bridges from
Devon's engines into the store (``kinect_bridge``).

The pattern is artcraftx's cross-engine tasks table + provider
router, re-expressed for live mocap state — pattern only, no
upstream code (see ``MOCAP_STATE_SURVEY.md`` in
``~/workspace/mocap-state-verify/``).
"""

from __future__ import annotations

from .bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    hierarchy_joints,
    parse_bvh,
    skeleton_from_dict,
    skeleton_to_dict,
    write_bvh,
)
from .errors import (
    ChannelOwnershipError,
    EngineNotConfigured,
    MocapError,
    StaleChannelError,
    UnknownChannelError,
    UnknownEngineError,
    UnknownSubscriberError,
)
from .kinect_bridge import (
    BvhExporter,
    BvhPlaybackProducer,
    KinectProducer,
    load_bvh_json,
)
from .router import MocapRouter, default_router
from .state import MocapStateStore

__all__ = [
    "BvhExporter",
    "BvhPlaybackProducer",
    "ChannelOwnershipError",
    "EngineNotConfigured",
    "Joint",
    "KinectProducer",
    "MocapError",
    "MocapRouter",
    "MocapStateStore",
    "Skeleton",
    "SkeletonFrame",
    "StaleChannelError",
    "UnknownChannelError",
    "UnknownEngineError",
    "UnknownSubscriberError",
    "default_router",
    "hierarchy_joints",
    "load_bvh_json",
    "parse_bvh",
    "skeleton_from_dict",
    "skeleton_to_dict",
    "write_bvh",
]
