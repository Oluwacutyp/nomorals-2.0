"""FaceMocap: ARKit LiveLink CSV facial capture, as face takes
(bevy-facial-mocap pattern port).

A thin surface over :mod:`nomorals.media.facemcap` — the organ
owns the CSV parse, the mapping, the take and the player; the
tool only routes arguments and shapes the reply.

One take is one recording, direct, no hub: a LiveLink CSV
(SMPTE timecode, blendshape count, one float column per ARKit
morph) plus a mapping (morph rename, bone rotations) becomes a
face take — per-frame rig blendshape weights and bone
rotations — drivable through a rig session, attachable to a
show, exportable as a face-take JSON or a driven BVH. Writes
output files, so FS_WRITE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.facemcap import (
    FaceMocapError,
    drive_take,
    drop_take,
    get_take,
    list_takes,
    load_take,
    render_frame,
)
from ..media.facemcap.take import FaceTake

__all__ = ["register"]

_ACTIONS = ("load", "list", "status", "drive", "frame", "export",
            "render", "drop", "attach")


def _params(params: dict | None) -> dict:
    return dict(params or {})


def _rig_session(name: str) -> Any:
    from ..media.rig import get_session as get_rig_session
    return get_rig_session(name)


def _drive(take: FaceTake, rig_name: str,
           args: dict) -> dict[str, Any]:
    """Drive a whole take through a rig session, rendering and
    exporting when asked."""
    session = _rig_session(rig_name)
    posed = drive_take(session, take)
    result: dict[str, Any] = {
        "take": take.name,
        "frames": len(posed),
        "fps": take.fps,
        "duration_s": round(take.duration_s, 3),
        "rig_session": rig_name,
        "rig": session.rig.name,
    }
    render_dir = args.get("render_dir")
    if render_dir:
        result["rendered"] = session.render_take(
            posed, out_dir=str(render_dir))
    export = args.get("export")
    if export:
        result["export"] = session.export_bvh(posed, str(export))
    return result


def _render_frame(take: FaceTake, rig_name: str, index: int,
                  output: str) -> dict[str, Any]:
    """Render one frame of a take as a PNG, through the organ's
    shared helper."""
    session = _rig_session(rig_name)
    return render_frame(session, take, index, output)


def register(registry: Any) -> None:
    """Attach the facemcap family to a registry."""
    @registry.register(
        "facemcap",
        description=(
            "ARKit LiveLink CSV facial capture — a recording "
            "becomes a face take (bevy-facial-mocap pattern, "
            "pattern only). A LiveLink CSV (SMPTE timecode, "
            "blendshape count, one float column per ARKit morph) "
            "plus a mapping (morph rename, bone rotations with "
            "axis/offset/scale) loads as a take: per-frame rig "
            "blendshape weights and bone rotations in (z, x, y) "
            "degrees, drivable through a rig session, attachable "
            "to a show, exportable as a face-take JSON or a "
            "driven BVH. Facemcap actions: load=<load a take "
            "from a .morph_anim.json manifest, a face-take .json "
            "or a .csv, node=<name>, sources=[path], params=<{"
            "mapping: path|'arkit', root_name, fps}>>; list=<"
            "loaded takes>; status=<one take's state, node=<"
            "name>>; drive=<drive the whole take through a rig "
            "session, node=<take>, value=<rig session>, params=<{"
            "render_dir, export: path.bvh}>>; frame=<one frame "
            "as data, node=<take>, params=<{index}>>; export=<"
            "write the take as a face-take JSON, node=<take>, "
            "output=<path.json>>; render=<render one frame as a "
            "PNG, node=<take>, value=<rig session>, output=<"
            "path.png>, params=<{index}>>; drop=<drop a take, "
            "node=<name>>; attach=<attach a take to a show, node="
            "<take>, value=<show>>. action=<load|list|status|"
            "drive|frame|export|render|drop|attach>; sources=<"
            "list of paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def facemcap_tool(action: str = "status",
                      sources: list[str] | None = None,
                      output: str = "", kind: str = "",
                      node: str = "", value: Any = None,
                      command: str = "",
                      params: dict | None = None,
                      title: str = "",
                      ) -> dict[str, Any]:
        action = (action or "status").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown facemcap action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        args = _params(params)
        sources = list(sources or [])
        name = (node or "").strip() or "default"
        if action == "list":
            return {"takes": list_takes()}
        if action == "load":
            if not sources:
                raise ValueError(
                    "load needs sources=[path.morph_anim.json|"
                    "path.json|path.csv]")
            mapping = args.get("mapping")
            fps = args.get("fps")
            take = load_take(
                name, sources[0],
                mapping=(str(mapping) if mapping else None),
                root_name=(str(args["root_name"])
                           if args.get("root_name") else None),
                fps=(float(fps) if fps is not None else None))
            return dict(take.to_dict(), ok=True)
        take = get_take(name)
        if action == "status":
            return take.to_dict()
        if action == "drive":
            rig_name = str(value or "default").strip() or "default"
            return _drive(take, rig_name, args)
        if action == "frame":
            index = int(args.get("index") or 0)
            if index < 0 or index >= take.total:
                raise FaceMocapError(
                    f"take {take.name!r} has frames "
                    f"0..{take.total - 1}, got {index}",
                    kind="bad_frame")
            return dict(take.frames[index].to_dict(),
                        take=take.name, index=index,
                        total=take.total)
        if action == "export":
            if not output:
                raise ValueError("export needs output=<path.json>")
            target = Path(output).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(take.to_json(), indent=1),
                encoding="utf-8")
            return {"take": take.name, "path": str(target),
                    "frames": take.total}
        if action == "render":
            if not output:
                raise ValueError("render needs output=<path.png>")
            rig_name = str(value or "default").strip() or "default"
            return _render_frame(
                take, rig_name, int(args.get("index") or 0), output)
        if action == "drop":
            return drop_take(name)
        if action == "attach":
            show_name = str(value or "").strip()
            if not show_name:
                raise ValueError(
                    "attach needs value=<show>")
            from ..media.show import get_show
            show = get_show(show_name)
            return show.attach_face_take(take.player())
        raise ValueError(f"unhandled facemcap action {action!r}")
