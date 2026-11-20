"""`nm scene` — spawn actors into a Scene3D document and manage
their lives (actorspawningplugin pattern, pattern only).

Operates on a scene3d JSON document: spawn a prop, character, sound
or light at a deliberate placement, despawn it (freeing its scene
slot), list the scene's spawned actors, or query one actor's state.
Spawning is per-scene, direct — no central queue.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...media.edit_engine.spawn import (
    Placement,
    SpawnError,
    SpawnRequest,
)


def _parse_vec3(text: str, what: str) -> tuple[float, float, float]:
    try:
        parts = [float(p) for p in str(text).split(",")]
    except ValueError:
        raise ValueError(
            f"{what} must be three numbers like 1,2,3, got {text!r}"
        ) from None
    if len(parts) != 3:
        raise ValueError(
            f"{what} must be three numbers like 1,2,3, got {text!r}")
    return (parts[0], parts[1], parts[2])


def _parse_bounds(text: str) -> dict[str, list[float]]:
    try:
        parts = [float(p) for p in str(text).split(",")]
    except ValueError:
        raise ValueError(
            "bounds must be six numbers x0,x1,y0,y1,z0,z1, "
            f"got {text!r}"
        ) from None
    if len(parts) != 6:
        raise ValueError(
            "bounds must be six numbers x0,x1,y0,y1,z0,z1, "
            f"got {text!r}")
    return {"x": [parts[0], parts[1]],
            "y": [parts[2], parts[3]],
            "z": [parts[4], parts[5]]}


def _placement(args: argparse.Namespace) -> Placement:
    if args.at is not None:
        return Placement.at(_parse_vec3(args.at, "--at"))
    if args.offset_from is not None:
        if args.offset is None:
            raise ValueError(
                "--offset-from needs --offset x,y,z")
        return Placement.offset_from(
            args.offset_from, _parse_vec3(args.offset, "--offset"))
    if args.in_front_of is not None:
        return Placement.in_front_of(args.in_front_of,
                                     distance=args.distance)
    if args.random_around is not None or args.random_bounds is not None:
        bounds = (_parse_bounds(args.random_bounds)
                  if args.random_bounds is not None else None)
        return Placement.random_around(args.random_around,
                                       bounds=bounds, seed=args.seed)
    return Placement()


def _emit(args: argparse.Namespace, payload: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    elif isinstance(payload, dict) and "error" in payload:
        print(f"error: {payload['error']}", file=sys.stderr)
        return 1
    else:
        print(json.dumps(payload, indent=2, default=str))
    return 0


def _show_actors(args: argparse.Namespace,
                 payload: dict[str, Any]) -> int:
    if getattr(args, "json", False):
        return _emit(args, payload)
    actors = payload.get("actors", [])
    budget = payload.get("budget", {})
    if not actors:
        print(f"no spawned actors in {payload.get('scene')} "
              f"(budget {budget.get('alive', 0)}/"
              f"{budget.get('max_actors', '?')})")
        return 0
    for row in actors:
        remaining = row.get("remaining_s")
        remaining = (f"{remaining}s left" if remaining is not None
                     else "no timer")
        print(f"{row['name']} [{row['kind']}] {row['status']} "
              f"at {row['position']} age {row['age_s']}s, {remaining}")
    print(f"budget {budget.get('alive', 0)}/"
          f"{budget.get('max_actors', '?')} actors alive")
    return 0


def cmd_scene(args: argparse.Namespace, context: Any) -> int:
    """Route `nm scene` subcommands (actorspawn §3B.19)."""
    from ...media.edit_engine.spawn import (
        actors_in_file,
        despawn_from_file,
        spawn_into_file,
        tick_file,
    )

    action = args.scene_action
    try:
        if action == "list":
            return _show_actors(
                args, actors_in_file(args.scene))
        if action == "status":
            found = actors_in_file(args.scene)
            for row in found["actors"]:
                if row["name"] == args.name:
                    if getattr(args, "json", False):
                        return _emit(args, {"actor": row,
                                            "scene": args.scene,
                                            "budget": found["budget"]})
                    print(f"{row['name']} [{row['kind']}] "
                          f"{row['status']} at {row['position']} "
                          f"age {row['age_s']}s")
                    return 0
            print(f"error: no spawned actor {args.name!r} in "
                  f"{args.scene}", file=sys.stderr)
            return 1
        if action == "tick":
            result = tick_file(args.scene)
            if getattr(args, "json", False):
                return _emit(args, result)
            if result["culled"]:
                print(f"culled: {', '.join(result['culled'])}")
            else:
                print("no actors expired")
            return 0
        if action == "despawn":
            result = despawn_from_file(args.scene, args.name)
            if getattr(args, "json", False):
                return _emit(args, result)
            print(f"despawned {args.name} from {args.scene}")
            return 0
        # spawn
        placement = _placement(args)
        request = SpawnRequest(
            kind=args.kind, placement=placement, name=args.name or None,
            lifetime_s=args.lifetime,
            parent=args.parent or None,
            rotation=(_parse_vec3(args.rotation, "--rotation")
                      if args.rotation is not None else None),
            look_at=(_parse_vec3(args.look_at, "--look-at")
                     if args.look_at is not None else None),
            scale=_parse_vec3(args.scale, "--scale")
            if args.scale is not None else (1.0, 1.0, 1.0),
            primitive=args.primitive,
            params=(json.loads(args.params) if args.params else {}),
            material=args.material or None,
            bvh_path=args.bvh or None,
            sound_clip=args.sound_clip or None,
            volume=args.volume,
            destroy_after_play=args.destroy_after_play,
            duration_s=args.duration,
            light_kind=args.light_kind,
            color=(_parse_vec3(args.color, "--color")
                   if args.color is not None else (1.0, 1.0, 1.0)),
            intensity=args.intensity)
        result = spawn_into_file(args.scene, request)
        if getattr(args, "json", False):
            return _emit(args, result)
        actor = result["actor"]
        print(f"spawned {actor['name']} [{actor['kind']}] "
              f"at {actor['position']} in {args.scene}")
        return 0
    except (SpawnError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
