"""``nm cloudcam`` — streamed point clouds, live from the terminal.

Thin CLI over the cloudcam organ: the same sessions the
``cloudcam`` brain tool drives, from the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...media.cloudcam.errors import CloudCamError


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


def _cmd_cloudcam(args: argparse.Namespace, context: Any) -> int:
    """Route `nm cloudcam` subcommands."""
    from ...media.cloudcam import (
        get_session,
        list_sessions,
        render_projection,
        write_ply,
    )

    action = args.cloudcam_action
    name = (getattr(args, "session", "") or "").strip() or "default"
    try:
        if action == "list":
            return _out(args, {"sessions": list_sessions()})
        if action == "status":
            return _out(args, dict(get_session(name).status()))
        if action == "start":
            from ...media.cloudcam import load_session
            source = (args.source or "idle").strip().lower()
            store = None
            if source == "store":
                from ...tools.mocap import _STORE
                store = _STORE
            session = load_session(
                name, source=source,
                endpoint=(args.endpoint or None),
                path=(args.path or None),
                store=store,
                width=args.width, height=args.height,
                capacity=args.capacity)
            return _out(args, dict(session.start(), ok=True))
        if action == "stop":
            return _out(args, dict(get_session(name).stop(), ok=True))
        if action in ("tick", "stage"):
            return _out(args, get_session(name).tick())
        if action in ("export", "render"):
            session = get_session(name)
            points = session.latest_points()
            if not points:
                print(f"cloudcam session {name!r} has staged no frame "
                      f"yet — start a source and tick first",
                      file=sys.stderr)
                return 1
            if not args.out:
                need = "path.ply" if action == "export" else "path.png"
                print(f"cloudcam {action} needs --out <{need}>",
                      file=sys.stderr)
                return 2
            if action == "export":
                result = write_ply(points, args.out)
            else:
                result = render_projection(
                    points, args.out, width=args.width_png,
                    height=args.height_png, view=args.view)
            return _out(args, dict(result, session=name))
    except CloudCamError as exc:
        print(f"cloudcam: {exc}", file=sys.stderr)
        return 1
    print(f"unknown cloudcam action: {action}", file=sys.stderr)
    return 2
