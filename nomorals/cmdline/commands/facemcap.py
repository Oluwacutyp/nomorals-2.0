"""``nm facemcap`` — ARKit LiveLink CSV facial capture, as face
takes, live from the terminal.

Thin CLI over the facemcap organ: the same takes the
``facemcap`` brain tool drives, from the terminal. A take is
loaded from a ``.morph_anim.json`` manifest, a face-take
``.json`` or a ``.csv``; driving it steps it through a rig
session frame by frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...media.facemcap import (
    FaceMocapError,
    drive_take,
    drop_take,
    get_take,
    list_takes,
    load_take,
)


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


def _rig_session(name: str) -> Any:
    from ...media.rig import get_session as get_rig_session
    return get_rig_session(name)


def _cmd_facemcap(args: argparse.Namespace, context: Any) -> int:
    """Route `nm facemcap` subcommands."""
    action = args.facemcap_action
    name = (getattr(args, "session", "") or "").strip() or "default"
    try:
        if action == "list":
            return _out(args, {"takes": list_takes()})
        if action == "load":
            take = load_take(
                name, args.path,
                mapping=(args.mapping or None),
                root_name=(args.root or None),
                fps=(args.fps if args.fps is not None else None))
            return _out(args, dict(take.to_dict(), ok=True))
        take = get_take(name)
        if action == "status":
            return _out(args, take.to_dict())
        if action == "frame":
            index = int(args.index)
            if index < 0 or index >= take.total:
                print(f"take {take.name!r} has frames "
                      f"0..{take.total - 1}, got {index}",
                      file=sys.stderr)
                return 2
            return _out(args, dict(take.frames[index].to_dict(),
                                   take=take.name, index=index,
                                   total=take.total))
        if action == "drive":
            session = _rig_session(args.rig)
            posed = drive_take(session, take)
            result: dict[str, Any] = {
                "take": take.name,
                "frames": len(posed),
                "fps": take.fps,
                "duration_s": round(take.duration_s, 3),
                "rig_session": args.rig,
                "rig": session.rig.name,
            }
            if getattr(args, "render_dir", ""):
                result["rendered"] = session.render_take(
                    posed, out_dir=args.render_dir)
            if getattr(args, "export", ""):
                result["export"] = session.export_bvh(
                    posed, args.export)
            return _out(args, result)
        if action == "export":
            if not args.out:
                print("facemcap export needs --out <path.json>",
                      file=sys.stderr)
                return 2
            target = args.out
            from pathlib import Path
            path = Path(target).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(take.to_json(), indent=1),
                            encoding="utf-8")
            return _out(args, {"take": take.name,
                               "path": str(path),
                               "frames": take.total})
        if action == "render":
            if not args.out:
                print("facemcap render needs --out <path.png>",
                      file=sys.stderr)
                return 2
            from ...media.facemcap.take import render_frame
            session = _rig_session(args.rig)
            return _out(args, render_frame(
                session, take, int(args.index), args.out))
        if action == "drop":
            return _out(args, drop_take(name))
        if action == "attach":
            from ...media.show import get_show
            show = get_show(args.show)
            return _out(args, show.attach_face_take(take.player()))
    except FaceMocapError as exc:
        print(f"facemcap: {exc}", file=sys.stderr)
        return 1
    print(f"unknown facemcap action: {action}", file=sys.stderr)
    return 2
