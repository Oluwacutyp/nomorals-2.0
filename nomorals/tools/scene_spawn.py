"""Scene spawning: put actors into a Scene3D document and manage
their lives (actorspawningplugin pattern, pattern only).

A thin surface over :mod:`nomorals.media.edit_engine.spawn` — the
organ owns placement, lifecycle and the per-scene budget; the tool
only routes arguments and shapes the reply. Every call loads the
scene document fresh and saves it sealed: there is no process-level
shared state and no central registry — spawning is per-scene,
direct, with the budget enforced at the scene. Writes scene files,
so FS_WRITE.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.edit_engine.spawn import (
    Placement,
    SpawnError,
    SpawnRequest,
    actors_in_file,
    despawn_from_file,
    spawn_into_file,
    tick_file,
)

__all__ = ["register"]

_ACTIONS = ("spawn", "despawn", "list", "status", "tick")


def _home_of(context: Any) -> str:
    """Devon's home from a tool-registry context.

    The established idiom (tools/security.py ``_home_of``): settings
    first, a bare ``.home`` (the test harness shape) second, ``"."``
    only when neither exists. AgentContext carries ``.settings``
    and no bare ``.home`` — reading ``getattr(context, "home",
    ".")`` would land scene paths in the caller's cwd.
    """
    settings = getattr(context, "settings", None)
    if settings is not None:
        home = (getattr(settings, "home_path", None)
                or getattr(settings, "home", None))
        if home:
            return str(Path(os.path.expanduser(str(home))).resolve())
    home = getattr(context, "home", None)
    if home:
        return str(Path(os.path.expanduser(str(home))).resolve())
    return "."


def _resolve_scene(context: Any, scene: str) -> str:
    if not scene:
        raise SpawnError(
            "scene_spawn needs a scene — a scene3d document path "
            "(absolute, or relative to Devon's home)")
    path = Path(os.path.expanduser(str(scene)))
    if not path.is_absolute():
        path = Path(_home_of(context)) / path
    return str(path)


def _json_arg(value: Any, *, what: str) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except ValueError as exc:
        raise ValueError(f"{what} must be JSON: {exc}") from exc


def _placement_arg(value: Any) -> Placement:
    data = _json_arg(value, what="placement")
    if data is None:
        return Placement()
    if not isinstance(data, dict):
        raise ValueError(
            f"placement must be a dict, got {data!r}")
    return Placement.from_dict(data)


def register(registry: Any) -> None:
    """Attach the scene-spawn family to a registry."""
    context = registry.context

    @registry.register(
        "scene_spawn",
        description=(
            "Spawn actors into a 3D scene document (Scene3D) and "
            "manage their lives — props, characters, sounds and "
            "lights, placed deliberately (at a point, offset from a "
            "named anchor, in front of the camera on the camera -> "
            "anchor arc, or scattered with stated bounds), with a "
            "lifetime and a per-scene budget. Spawning is per-scene, "
            "direct — no central queue. Actions: spawn=<put an actor "
            "in the scene, kind=<prop|character|sound|light>, "
            "placement=<{mode, position, anchor, distance, bounds, "
            "seed}>>; despawn=<remove one, name=<actor>>; list=<the "
            "scene's spawned actors>; status=<one actor's state, "
            "name=<actor>>; tick=<activate new actors and cull "
            "expired ones>. scene=<scene3d document path>."
        ),
        capability=Capability.FS_WRITE,
    )
    def scene_spawn(action: str = "list", scene: str = "",
                    kind: str = "prop", name: str = "",
                    placement: Any = None, lifetime_s: float | None = None,
                    parent: str = "", rotation: Any = None,
                    look_at: Any = None, scale: Any = None,
                    primitive: str = "cube", params: Any = None,
                    material: str = "", bvh: str = "",
                    skeleton: Any = None, sound_clip: str = "",
                    volume: float = 1.0, destroy_after_play: bool = False,
                    duration_s: float | None = None,
                    light_kind: str = "point", color: Any = None,
                    intensity: float = 1.0) -> dict[str, Any]:
        action = (action or "list").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown scene_spawn action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        path = _resolve_scene(context, scene)
        if action == "list":
            return actors_in_file(path)
        if action == "status":
            if not name:
                raise ValueError("status needs name=<actor>")
            found = actors_in_file(path)
            for row in found["actors"]:
                if row["name"] == name:
                    return {"actor": row, "scene": path,
                            "budget": found["budget"]}
            raise SpawnError(
                f"no spawned actor {name!r} in scene {path}")
        if action == "tick":
            return tick_file(path)
        if action == "despawn":
            if not name:
                raise ValueError("despawn needs name=<actor>")
            return despawn_from_file(path, name)
        # spawn
        request = SpawnRequest(
            kind=kind,
            placement=_placement_arg(placement),
            name=name or None,
            lifetime_s=lifetime_s,
            parent=parent or None,
            rotation=_json_arg(rotation, what="rotation"),
            look_at=_json_arg(look_at, what="look_at"),
            scale=_json_arg(scale, what="scale") or (1.0, 1.0, 1.0),
            primitive=primitive,
            params=_json_arg(params, what="params") or {},
            material=material or None,
            bvh_path=bvh or None,
            skeleton=_json_arg(skeleton, what="skeleton"),
            sound_clip=sound_clip or None,
            volume=volume,
            destroy_after_play=destroy_after_play,
            duration_s=duration_s,
            light_kind=light_kind,
            color=_json_arg(color, what="color") or (1.0, 1.0, 1.0),
            intensity=intensity)
        return spawn_into_file(path, request)
