"""``nm mocap`` — motion-capture state and the VTuber rig.

Thin CLI over the mocap organ and the rig organ: the same
sessions the ``mocap`` brain tool drives, from the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...media.rig import FaceInput


def _out(args: argparse.Namespace, obj: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=2))
        return 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: {value}")
    else:
        print(obj)
    return 0


def _cmd_mocap(args: argparse.Namespace, context: Any) -> int:
    """Route `nm mocap` subcommands."""
    action = args.mocap_action
    if action == "rig":
        return _cmd_mocap_rig(args, context)
    if action == "status":
        from ...tools.mocap import _STORE
        return _out(args, {
            "channels": _STORE.channels(),
            "engines": [e["name"] for e in _STORE.engines()],
        })
    print(f"unknown mocap action: {action}", file=sys.stderr)
    return 2


def _cmd_mocap_rig(args: argparse.Namespace, context: Any) -> int:
    """Route `nm mocap rig` verbs."""
    from ...media.mocap.bvh import SkeletonFrame
    from ...media.rig import (
        get_session,
        list_sessions,
        load_session,
        load_take_bvh,
    )
    from ...media.rig.errors import RigError

    verb = args.mocap_rig_verb
    name = (getattr(args, "session", "") or "").strip() or "default"
    try:
        if verb == "list":
            return _out(args, {"sessions": list_sessions()})
        if verb == "load":
            session = load_session(
                name, rig_path=(args.rig or None),
                remap_dialect=args.dialect,
                remap_path=(args.remap or None),
                width=args.width, height=args.height,
                fps=args.fps, smooth=args.smooth)
            return _out(args, dict({"name": name, "ok": True},
                                   **session.to_dict()))
        if verb == "status":
            session = get_session(name)
            return _out(args, dict({"name": name},
                                   **session.to_dict()))
        session = get_session(name)
        if verb == "pose":
            face = _face_arg(args)
            if args.from_store:
                from ...tools.mocap import _STORE
                pose = session.drive_from_store(
                    _STORE, require_fresh=args.require_fresh,
                    face=face)
            else:
                frame = None
                if args.frame:
                    try:
                        raw = json.loads(args.frame)
                    except ValueError as exc:
                        print(f"--frame is not valid JSON: {exc}",
                              file=sys.stderr)
                        return 2
                    if not isinstance(raw, dict):
                        print("--frame must be {joint: [values]}",
                              file=sys.stderr)
                        return 2
                    frame = SkeletonFrame(values={
                        str(k): tuple(float(v) for v in vals)
                        for k, vals in raw.items()})
                pose = session.drive_frame(frame, face)
            return _out(args, dict({"session": name},
                                   **pose.to_dict()))
        if verb == "render":
            if not args.out:
                print("rig render needs --out <path.png>",
                      file=sys.stderr)
                return 2
            pose = session.drive_frame(None)
            rgb = session.render_frame(pose)
            target = Path(args.out).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                from PIL import Image
            except ImportError as exc:
                print(f"rig rendering needs Pillow: {exc}",
                      file=sys.stderr)
                return 1
            Image.frombytes("RGB", (session.width, session.height),
                            rgb).save(target)
            return _out(args, {"path": str(target),
                               "session": name,
                               "viseme": pose.viseme})
        if verb in ("drive", "export"):
            if not args.bvh:
                print(f"rig {verb} needs --bvh <path>",
                      file=sys.stderr)
                return 2
            take = load_take_bvh(args.bvh)
            posed = session.drive_take(
                take["skeleton"], take["frames"],
                text=(args.text or "") or None)
            if verb == "export":
                if not args.out:
                    print("rig export needs --out <path.bvh>",
                          file=sys.stderr)
                    return 2
                from ...tools.mocap import _STORE
                result = session.export_bvh(posed, args.out,
                                            store=_STORE)
                return _out(args, dict(result, session=name))
            if not args.out:
                return _out(args, {"session": name,
                                   "frames": len(posed),
                                   "visemes": [p.viseme
                                               for p in posed]})
            paths = session.render_take(posed, out_dir=args.out)
            return _out(args, {"session": name,
                               "frames": len(posed),
                               "visemes": [p.viseme for p in posed],
                               "rendered": paths})
    except RigError as exc:
        print(f"rig: {exc}", file=sys.stderr)
        return 1
    print(f"unknown mocap rig verb: {verb}", file=sys.stderr)
    return 2


def _face_arg(args: argparse.Namespace) -> FaceInput | None:
    """--face JSON -> FaceInput. Unknown keys are refused."""
    if not (args.face or "").strip():
        if (args.text or "").strip():
            return FaceInput(text=args.text)
        return None
    try:
        raw = json.loads(args.face)
    except ValueError as exc:
        print(f"--face is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if not isinstance(raw, dict):
        print("--face must be an object", file=sys.stderr)
        raise SystemExit(2)
    known = {"blendshapes", "viseme", "text", "openness",
             "centroid", "laughter"}
    unknown = sorted(set(raw) - known)
    if unknown:
        print(f"--face has unknown keys: {', '.join(unknown)}",
              file=sys.stderr)
        raise SystemExit(2)
    return FaceInput(
        blendshapes=raw.get("blendshapes"),
        viseme=raw.get("viseme"),
        text=raw.get("text"),
        openness=raw.get("openness"),
        centroid=raw.get("centroid"),
        laughter=float(raw.get("laughter") or 0.0))
