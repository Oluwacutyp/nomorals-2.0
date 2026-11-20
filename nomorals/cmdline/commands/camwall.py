"""``nm camwall`` — the cameras wall, live from the terminal.

Thin CLI over the ``camwall`` brain tool: the same walls the
tool loads, from the terminal. Loading a config, starting a
wall, and exporting it are open; the fetches go only to the
cameras in the owner's config file, never to a URL typed here
except through an explicit ``add``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...tools.camwall import camwall_sync


def _out(args: argparse.Namespace, obj: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=2))
        return 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: {value}")
    elif isinstance(obj, list):
        for item in obj:
            print(item)
    else:
        print(obj)
    return 0


def _wall_text(status: dict[str, Any]) -> str:
    lines = [f"wall {status.get('wall')}: "
             f"{'running' if status.get('running') else 'stopped'} "
             f"({status.get('server')} server, "
             f"{status.get('rows')} per row)"]
    for cam in status.get("camera_list", []):
        state = "paused" if cam.get("paused") else "live"
        frame = "frame" if cam.get("has_frame") else "no frame"
        error = (f" — {cam.get('last_error')}"
                 if cam.get("last_error") else "")
        lines.append(
            f"  [{cam.get('id')}] {cam.get('name')} — "
            f"{cam.get('location')} — {state} — {frame} — "
            f"{cam.get('fps')} fps — "
            f"{cam.get('counts', {}).get('load', 0)} loads, "
            f"{cam.get('counts', {}).get('fail', 0)} fails{error}")
    return "\n".join(lines)


def _cmd_camwall(args: argparse.Namespace, context: Any) -> int:
    """Route `nm camwall` subcommands."""
    action = args.camwall_action
    wall = (getattr(args, "wall", "") or "").strip() or "default"
    try:
        if action == "load":
            result = camwall_sync(
                action="load", node=wall,
                params={"config": getattr(args, "config", None),
                        "rows": getattr(args, "rows", 2),
                        "server": getattr(args, "server", "remote")})
            if not result.get("ok"):
                print(f"load failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(_wall_text(result))
            return 0
        if action == "list":
            result = camwall_sync(action="list")
            if getattr(args, "json", False):
                return _out(args, result)
            walls = result.get("walls", [])
            if not walls:
                print("no walls loaded "
                      "(run `nm camwall load <config>` first)")
                return 0
            for entry in walls:
                print(f"{entry.get('wall')}: "
                      f"{entry.get('cameras')} cameras, "
                      f"{'running' if entry.get('running') else 'stopped'}")
            return 0
        if action == "status":
            result = camwall_sync(action="status", node=wall)
            if getattr(args, "json", False):
                return _out(args, result)
            print(_wall_text(result))
            return 0
        if action in ("start", "stop", "drop", "tick", "wall"):
            result = camwall_sync(
                action=action, node=wall,
                params={"wait": bool(getattr(args, "wait", False))})
            if not result.get("ok"):
                print(f"{action} failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            if action == "tick":
                print(f"ticked {len(result.get('ticked', []))} camera(s)")
            else:
                print(f"wall {wall}: {action} ok")
            return 0
        if action in ("pause", "resume", "focus", "snapshot"):
            camera = (getattr(args, "camera", "") or "").strip()
            if not camera:
                print(f"{action} needs a camera name or id",
                      file=sys.stderr)
                return 2
            result = camwall_sync(
                action=action, node=wall, value=camera,
                output=getattr(args, "output", "") or "")
            if not result.get("ok"):
                print(f"{action} failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            if action == "snapshot":
                print(f"saved {result.get('bytes')} bytes to "
                      f"{result.get('output')}")
            else:
                print(f"camera {camera}: {action} ok")
            return 0
        if action == "server":
            value = (getattr(args, "value", "") or "").strip()
            result = camwall_sync(action="server", node=wall, value=value)
            if not result.get("ok"):
                print(f"server failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            print(f"wall {wall}: server {result.get('server')}")
            return 0
        if action == "rows":
            value = getattr(args, "value", 2)
            result = camwall_sync(action="rows", node=wall, value=value)
            if not result.get("ok"):
                print(f"rows failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            print(f"wall {wall}: {result.get('rows')} per row")
            return 0
        if action == "add":
            entry = json.loads(getattr(args, "camera_json", "{}"))
            result = camwall_sync(
                action="add", node=wall, params={"camera": entry})
            if not result.get("ok"):
                print(f"add failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(f"added camera {result.get('name')} "
                  f"(id {result.get('id')}) to wall {wall}")
            return 0
        if action == "export":
            output = (getattr(args, "output", "") or "").strip()
            if not output:
                print("export needs an output directory",
                      file=sys.stderr)
                return 2
            result = camwall_sync(
                action="export", node=wall, output=output,
                params={"title": getattr(args, "title", "") or None,
                        "overwrite": bool(
                            getattr(args, "overwrite", False))})
            if not result.get("ok"):
                print(f"export failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(f"wall {wall} exported to {result.get('dir')} "
                  f"({result.get('cameras')} cameras, "
                  f"{result.get('frames')} frames)")
            return 0
        if action == "serve":
            output = (getattr(args, "output", "") or "").strip()
            if not output:
                print("serve needs an output directory",
                      file=sys.stderr)
                return 2
            result = camwall_sync(
                action="serve", output=output,
                params={"host": getattr(args, "host", "127.0.0.1"),
                        "port": getattr(args, "port", 0)})
            if not result.get("ok"):
                print(f"serve failed: {result.get('error')}",
                      file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(f"serving {result.get('dir')} at {result.get('url')}")
            if result.get("network"):
                print("network: this wall is exposed on 0.0.0.0")
            return 0
        if action == "unserve":
            output = (getattr(args, "output", "") or "").strip()
            if not output:
                print("unserve needs an output directory",
                      file=sys.stderr)
                return 2
            result = camwall_sync(action="unserve", output=output)
            print(f"stopped serving {result.get('dir')}")
            return 0
        if action == "served":
            result = camwall_sync(action="served")
            if getattr(args, "json", False):
                return _out(args, result)
            servers = result.get("servers", [])
            if not servers:
                print("no walls served")
                return 0
            for server in servers:
                print(f"{server.get('dir')} — {server.get('url')}")
            return 0
    except Exception as exc:  # noqa: BLE001 - CLI prints, never traces
        print(f"camwall {action} failed: {exc}", file=sys.stderr)
        return 1
    print(f"unknown camwall action {action!r}", file=sys.stderr)
    return 2
