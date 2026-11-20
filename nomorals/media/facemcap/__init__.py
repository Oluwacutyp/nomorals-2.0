"""Face mocap — ARKit LiveLink CSV facial capture, as face takes.

The bevy-facial-mocap pattern (storytold/bevy-mocap, pattern
only, no upstream code): Unreal's LiveLink Face app writes one
CSV row per frame — a SMPTE timecode, a blendshape count, one
float column per ARKit morph. Devon's organ parses that CSV,
applies a mapping (morph rename, bone rotations), and hands
back a face take: per-frame rig blendshape weights plus bone
rotations in ``(z, x, y)`` degrees, playable through a rig
session frame by frame, stepped into the show's schedule by
the face plugin, or exported as a face-take JSON (and, once
driven, as BVH through the rig's export).

One take is one recording, direct: it holds its own frames
and its own player. There is no hub and no shared queue.
"""

from __future__ import annotations

from .csv import CSVAnimation, CSVFrame, Timestamp
from .errors import FaceMocapError
from .loader import load_csv, load_manifest, load_take_json
from .mapping import (
    BoneMapping,
    FaceMapping,
    MorphMapping,
    default_arkit_mapping,
    default_mapping_path,
)
from .registry import (
    drop_take,
    get_take,
    list_takes,
    load_take,
    reset_takes,
)
from .take import (
    FaceFrame,
    FaceTake,
    FaceTakePlayer,
    drive_take,
    render_frame,
)

__all__ = [
    "BoneMapping",
    "CSVAnimation",
    "CSVFrame",
    "FaceFrame",
    "FaceMocapError",
    "FaceMapping",
    "FaceTake",
    "FaceTakePlayer",
    "MorphMapping",
    "Timestamp",
    "default_arkit_mapping",
    "default_mapping_path",
    "drive_take",
    "drop_take",
    "get_take",
    "list_takes",
    "load_csv",
    "load_manifest",
    "load_take",
    "load_take_json",
    "render_frame",
    "reset_takes",
]
