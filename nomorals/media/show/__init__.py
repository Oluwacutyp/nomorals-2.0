"""The live show — one run of mocap + rig + cloudcam, through
named states (storyteller-bevy patterns, pattern only, no
upstream code).

The study (storytold/storyteller-bevy): a Bevy 0.13.2 studio
monorepo — a Bevy app (`studio/`) plus a BVH mocap crate
(`studio-bvh/`), compiled to wasm32. The patterns taken are
Bevy's own: an App plugins extend, named States with
OnEnter/OnExit hooks, a Schedule of ordered phases, typed
Resources, and Events drained at a schedule boundary.

Devon's show is the same shape in Python, over the organs that
already produce its inputs: the mocap organ's state store, the
VTuber rig's sessions, and the cloudcam organ's sessions. A
show is a state machine (:mod:`states`) driving a schedule of
phases (:mod:`show`, ``capture → drive → render → publish``)
that plugins (:mod:`plugins`) register systems into, over
named resources, with a bounded event journal. A take player
(:mod:`player`) steps a BVH take into the store one frame per
tick. One show is one run, direct — no hub, no shared queue.
A show also exports as a hosted demo (:mod:`web`): a static
site — Start button + canvas — that plays the show's rendered
frames in a browser, with the act's lines and camera cuts.
"""

from __future__ import annotations

from .errors import ShowError
from .player import TakePlayer
from .plugins import (
    CloudCamPlugin,
    FaceMocapPlugin,
    MocapPlugin,
    RigPlugin,
    ShowPlugin,
)
from .registry import (
    drop_show,
    get_show,
    list_shows,
    load_show,
    reset_shows,
)
from .show import MAX_EVENTS, MAX_POSED, PHASES, Show
from .states import TRANSITIONS, ShowState, legal
from .web import (
    DemoServer,
    export_web,
    list_demos,
    serve_demo,
    stop_demo,
)

__all__ = [
    "MAX_EVENTS",
    "MAX_POSED",
    "PHASES",
    "CloudCamPlugin",
    "DemoServer",
    "FaceMocapPlugin",
    "MocapPlugin",
    "RigPlugin",
    "Show",
    "ShowError",
    "ShowPlugin",
    "ShowState",
    "TakePlayer",
    "TRANSITIONS",
    "drop_show",
    "export_web",
    "get_show",
    "legal",
    "list_demos",
    "list_shows",
    "load_show",
    "reset_shows",
    "serve_demo",
    "stop_demo",
]
