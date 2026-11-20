"""``nm robot`` — the robot daemon, live from the terminal.

Thin CLI over the ``robot`` brain tool: the same robots the tool
drives, from the terminal. Driving a robot with no hardware
attached drives its model and says so — the status always names
whether the pose is simulated.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...tools.robot import robot_sync


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


def _status_text(status: dict[str, Any]) -> str:
    pose = status.get("pose", {})
    motors = status.get("motors", {})
    lines = [
        f"robot {status.get('name')}: "
        f"{'running' if status.get('running') else 'stopped'} — "
        f"backend {status.get('backend')} "
        f"({'simulated' if status.get('simulated') else 'hardware'})",
        f"  pose x={pose.get('x', 0):.2f} y={pose.get('y', 0):.2f} "
        f"heading={pose.get('heading_deg', 0):.1f}°",
        f"  motors m1={motors.get('m1')} m2={motors.get('m2')} "
        f"(written: {'yes' if status.get('motors_written') else 'no'})",
        f"  battery: {status.get('battery')}",
        f"  map: {status.get('map', {})}",
    ]
    return "\n".join(lines)


def _cmd_robot(args: argparse.Namespace, context: Any) -> int:
    """Route `nm robot` subcommands."""
    action = args.robot_action
    robot = (getattr(args, "robot", "") or "").strip() or "default"
    try:
        if action == "create":
            return _out(args, robot_sync(action="create", node=robot))
        if action == "list":
            result = robot_sync(action="list")
            if getattr(args, "json", False):
                return _out(args, result)
            for entry in result.get("robots", []):
                print(_status_text(entry))
            return 0
        if action == "status":
            result = robot_sync(action="status", node=robot)
            if not result.get("ok", True):
                print(f"status failed: {result.get('error')}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(_status_text(result))
            return 0
        if action == "drive":
            # A bare number in the robot slot is seconds, not a name:
            # `nm robot drive 100 100 2` means two seconds, not a
            # robot called "2".
            robot_arg = (getattr(args, "robot", "") or "").strip()
            seconds = getattr(args, "seconds", None)
            if seconds is None and robot_arg:
                try:
                    seconds = float(robot_arg)
                    robot_arg = ""
                except ValueError:
                    pass
            result = robot_sync(
                action="drive", node=robot_arg or robot,
                value=args.left,
                params={"m2": args.right,
                        "seconds": seconds})
            if not result.get("ok", True):
                print(f"drive failed: {result.get('error')}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(_status_text(result))
            return 0
        if action in ("stop", "start", "tick", "detach", "drop"):
            result = robot_sync(
                action=action, node=robot,
                params={"dt": getattr(args, "dt", None)} if action == "tick" else None)
            if not result.get("ok", True):
                print(f"{action} failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "scan":
            params: dict[str, Any] = {}
            if getattr(args, "ranges", None):
                params["ranges"] = json.loads(args.ranges)
            if getattr(args, "source", None):
                params["source"] = args.source
            if getattr(args, "session", None):
                params["session"] = args.session
            if getattr(args, "depths", None):
                params["depths"] = json.loads(args.depths)
            if getattr(args, "fov", None):
                params["fov"] = args.fov
            result = robot_sync(action="scan", node=robot, params=params)
            if not result.get("ok", True):
                print(f"scan failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "map":
            result = robot_sync(action="map", node=robot,
                                params={"width": getattr(args, "width", 80)})
            if not result.get("ok", True):
                print(f"map failed: {result.get('error')}", file=sys.stderr)
                return 1
            if getattr(args, "json", False):
                return _out(args, result)
            print(result.get("ascii", ""))
            print(f"cells={result.get('cells')} observed="
                  f"{result.get('observed')} occupied="
                  f"{result.get('occupied')} free={result.get('free')}")
            return 0
        if action == "localize":
            params = {}
            if getattr(args, "ranges", None):
                params["ranges"] = json.loads(args.ranges)
            if getattr(args, "source", None):
                params["source"] = args.source
            if getattr(args, "session", None):
                params["session"] = args.session
            result = robot_sync(action="localize", node=robot, params=params)
            if not result.get("ok", True):
                print(f"localize failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "plan":
            result = robot_sync(action="plan", node=robot,
                                params={"steps": args.steps})
            if not result.get("ok", True):
                print(f"plan failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "serve":
            result = robot_sync(
                action="serve", node=robot,
                params={"host": getattr(args, "host", "127.0.0.1"),
                        "port": getattr(args, "port", 0)})
            if not result.get("ok", True):
                print(f"serve failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "unserve":
            result = robot_sync(action="unserve", node=robot)
            if not result.get("ok", True):
                print(f"unserve failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "served":
            return _out(args, robot_sync(action="served"))
        if action == "blink":
            result = robot_sync(
                action="blink", node=robot,
                params={"l1": getattr(args, "l1", 1),
                        "l2": getattr(args, "l2", 0)})
            if not result.get("ok", True):
                print(f"blink failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "battery":
            result = robot_sync(action="battery", node=robot)
            if not result.get("ok", True):
                print(f"battery failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "attach":
            result = robot_sync(
                action="attach", node=robot,
                params={"kind": args.kind,
                        "port": getattr(args, "port", ""),
                        "baud": getattr(args, "baud", 9600)})
            if not result.get("ok", True):
                print(f"attach failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        if action == "export":
            result = robot_sync(action="export", node=robot,
                                output=args.output)
            if not result.get("ok", True):
                print(f"export failed: {result.get('error')}", file=sys.stderr)
                return 1
            return _out(args, result)
        print(f"unknown robot action {action!r}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — the CLI reports, never traces
        print(f"robot {action} failed: {exc}", file=sys.stderr)
        return 1
