"""``nm show`` — one live run of mocap + rig + cloudcam, through
named states, live from the terminal.

Thin CLI over the show organ: the same shows the ``show`` brain
tool drives, from the terminal. Starting a show with no rig
attached auto-attaches the default rig and an idle cloudcam —
and says so, by name.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...media.show.errors import ShowError
from ...tools.show import ensure_defaults


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


def _cmd_show(args: argparse.Namespace, context: Any) -> int:
    """Route `nm show` subcommands."""
    from ...media.show import (
        drop_show,
        get_show,
        list_shows,
        load_show,
    )
    from ...tools.mocap import _STORE

    action = args.show_action
    name = (getattr(args, "session", "") or "").strip() or "default"
    try:
        if action == "list":
            return _out(args, {"shows": list_shows()})
        if action == "create":
            from ...media.cloudcam import (
                load_session as load_cc_session,
            )
            from ...media.rig import (
                get_session as get_rig_session,
            )
            rig = None
            if getattr(args, "rig", ""):
                rig = get_rig_session(args.rig)
            cam = None
            if getattr(args, "cloudcam", ""):
                from ...media.cloudcam import (
                    get_session as get_cc_session,
                )
                cam = get_cc_session(args.cloudcam)
            elif getattr(args, "cloudcam_source", ""):
                source = args.cloudcam_source.strip().lower()
                cam = load_cc_session(
                    f"{name}-cam", source=source,
                    endpoint=(args.endpoint or None),
                    path=(args.path or None),
                    store=_STORE if source == "store" else None)
            show = load_show(
                name, _STORE, rig=rig, cloudcam=cam,
                fps=args.fps,
                render_dir=(args.render_dir or None),
                export_path=(args.export_path or None))
            if getattr(args, "take", ""):
                show.load_take(args.take)
            return _out(args, show.status())
        if action == "serve":
            # Serving needs no show — it serves an exported
            # directory, not a live run.
            if not args.out:
                print("show serve needs --out <dir>",
                      file=sys.stderr)
                return 2
            from ...media.show.web import serve_demo, stop_demo
            try:
                status = serve_demo(
                    args.out, host=args.host, port=args.port)
            except ShowError as exc:
                print(f"show: {exc}", file=sys.stderr)
                return 1
            print(f"serving {status['dir']} at {status['url']} "
                  f"(Ctrl-C to stop)")
            if status.get("network"):
                print("listening on all interfaces (host 0.0.0.0) — "
                      "the demo is reachable on the network",
                      file=sys.stderr)
            import threading
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                print("stopping the demo server", file=sys.stderr)
            stop_demo(args.out)
            return 0
        if action == "unserve":
            if not args.out:
                print("show unserve needs --out <dir>",
                      file=sys.stderr)
                return 2
            from ...media.show.web import stop_demo
            return _out(args, stop_demo(args.out))
        show = get_show(name)
        if action == "status":
            return _out(args, show.status())
        if action == "start":
            attached = ensure_defaults(show)
            return _out(args, dict(show.start(),
                                   auto_attached=attached))
        if action == "pause":
            return _out(args, show.pause())
        if action == "resume":
            return _out(args, show.resume())
        if action == "stop":
            return _out(args, show.stop())
        if action == "tick":
            return _out(args, show.step())
        if action == "run":
            attached = ensure_defaults(show)
            if getattr(args, "export", ""):
                show.export_path = args.export
            result = show.run(args.ticks, args.interval)
            result = dict(result, auto_attached=attached)
            if getattr(args, "export", ""):
                result = dict(result,
                              export=show.export(args.export))
            return _out(args, result)
        if action == "events":
            return _out(args, {
                "show": name,
                "events": show.events(args.limit),
                "events_dropped": show.events_dropped})
        if action == "export":
            if not args.out:
                print("show export needs --out <path.bvh>",
                      file=sys.stderr)
                return 2
            return _out(args, dict(show.export(args.out), show=name))
        if action == "web":
            if not args.out:
                print("show web needs --out <dir>",
                      file=sys.stderr)
                return 2
            from ...media.show.web import export_web
            attached = ensure_defaults(show)
            result = export_web(
                show, args.out, ticks=args.ticks,
                max_frames=args.max_frames,
                width=(args.width or None),
                height=(args.height or None),
                title=(args.title or None),
                overwrite=args.overwrite)
            return _out(args, dict(result, auto_attached=attached))
        if action == "attach-rig":
            from ...media.rig import get_session as get_rig_session
            return _out(args, show.attach_rig(
                get_rig_session(args.value)))
        if action == "attach-cloudcam":
            from ...media.cloudcam import (
                get_session as get_cc_session,
            )
            return _out(args, show.attach_cloudcam(
                get_cc_session(args.value)))
        if action == "load-take":
            return _out(args, dict(show.load_take(args.take),
                                   show=name))
        if action == "drop":
            return _out(args, drop_show(name))
    except ShowError as exc:
        print(f"show: {exc}", file=sys.stderr)
        return 1
    print(f"unknown show action: {action}", file=sys.stderr)
    return 2
