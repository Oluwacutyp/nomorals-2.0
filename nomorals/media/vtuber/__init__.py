"""The VTuber show — a scripted character performance on a stage
(AnimationTest pattern: UE5 VTuber show, pattern only).

Devon already has the inputs a VTuber show is driven by — the rig
(B.18), the show coordinator (B.22), face takes (B.23), MVN live
capture (B.24) — but no performance: no scripted act, no
character, no camera direction, no audience loop, and no owned
live-capture performance. This organ is the performance: a
character (named expressions + gestures over the rig), an act
(timed beats: say / expression / gesture / camera / take /
face_take), cameras (shots as crops of the rendered frame), an act
player that performs the act through the show's own phases, and a
live-MVN attach that holds the recorded take while the character
is driven live. One performance per show, no hub, no shared
queue — a second show is a second, independent performance.
"""

from __future__ import annotations

from .act import Act, Beat
from .camera import SHOTS, apply_shot, shot_names
from .character import (
    BUILTIN_EXPRESSIONS,
    BUILTIN_GESTURES,
    Character,
)
from .errors import VtuberError
from .player import ActPlayer
from .plugin import (
    VtuberPlugin,
    attach_live,
    attach_performance,
    detach_live,
    detach_performance,
)
from .registry import (
    default_name,
    drop_act,
    drop_character,
    get_act,
    get_character,
    list_acts,
    list_characters,
    load_act,
    load_character,
    reset_all,
)

__all__ = [
    "Act",
    "ActPlayer",
    "BUILTIN_EXPRESSIONS",
    "BUILTIN_GESTURES",
    "Beat",
    "Character",
    "SHOTS",
    "VtuberError",
    "VtuberPlugin",
    "apply_shot",
    "attach_live",
    "attach_performance",
    "default_name",
    "detach_live",
    "detach_performance",
    "drop_act",
    "drop_character",
    "get_act",
    "get_character",
    "list_acts",
    "list_characters",
    "load_act",
    "load_character",
    "reset_all",
    "shot_names",
]
