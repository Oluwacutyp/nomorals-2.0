"""RuntimeRobotMixin: PartnerRuntime command group (robot daemon)."""

from __future__ import annotations

import json
from typing import Any

from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)


class RuntimeRobotMixin:
    """RuntimeRobotMixin for :class:`PartnerRuntime`."""

    # ── robot daemon: /robot ────────────────────────────────────────
    def _control_robot(self, tail: str, chat_key: str,
                       message: Any = None) -> str:
        """Robot daemon — drive a robot, map where it is, find it again.

        /robot list                        robots now
        /robot create [name]               make a robot
        /robot status [name]               one robot's state + pose
        /robot drive <left> [right] [seconds] [name]
                                           mogo (-100..100 each)
        /robot stop [name]                 stop the motors + plan
        /robot start [name]                start the 20 Hz tick loop
        /robot tick [name]                 one manual step
        /robot scan <ranges-json> [name]   fold a scan into the map
        /robot scan cloudcam [session] [name]
                                           scan from a cloudcam session
        /robot map [name]                  counts + ascii map
        /robot localize <ranges-json> [name]
                                           re-find the robot in its map
        /robot plan <steps-json> [name]    scripted drive (loop running)
        /robot serve [name]                loopback TCP control
        /robot unserve [name]              stop serving
        /robot served                      servers running now
        /robot blink [l1] [l2] [name]      LEDs
        /robot battery [name]              attached hardware's battery
        /robot attach loopback|serial <port> [name]
                                           attach hardware
        /robot detach [name]               drop the hardware
        /robot drop [name]                 forget the robot
        /robot export <dir> [name]         map.pgm + map.json + map.txt
        """
        from ...tools.robot import robot_sync

        parts = (tail or "").split()
        if not parts:
            return self._control_robot_help()
        verb = parts[0]
        rest = parts[1:]
        try:
            if verb == "list":
                result = robot_sync(action="list")
                robots = result.get("robots", [])
                if not robots:
                    return ("robot: no robots yet "
                            "(/robot create [name] first)")
                lines = ["robot robots:"]
                for entry in robots:
                    pose = entry.get("pose", {})
                    lines.append(
                        f"  {entry.get('name')}: "
                        f"{'running' if entry.get('running') else 'stopped'} — "
                        f"backend {entry.get('backend')} "
                        f"({'simulated' if entry.get('simulated') else 'hardware'}) — "
                        f"pose x={pose.get('x', 0):.2f} y={pose.get('y', 0):.2f} "
                        f"heading={pose.get('heading_deg', 0):.1f}°")
                return "\n".join(lines)
            if verb == "served":
                result = robot_sync(action="served")
                servers = result.get("robots", [])
                if not servers:
                    return "robot: no robots served"
                lines = ["robot served:"]
                for server in servers:
                    lines.append(f"  {server.get('name')} — "
                                 f"port {server.get('port')}")
                return "\n".join(lines)
            if verb == "create":
                name = rest[0] if rest else "default"
                result = robot_sync(action="create", node=name)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return self._robot_status_text(result)
            if verb == "status":
                name = rest[0] if rest else "default"
                result = robot_sync(action="status", node=name)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return self._robot_status_text(result)
            if verb == "drive":
                if not rest:
                    return self._control_robot_help()
                try:
                    left = int(rest[0])
                    right = int(rest[1]) if len(rest) > 1 else left
                    seconds = (float(rest[2]) if len(rest) > 2 else None)
                    name = rest[3] if len(rest) > 3 else "default"
                except ValueError:
                    return self._control_robot_help()
                result = robot_sync(action="drive", node=name, value=left,
                                    params={"m2": right, "seconds": seconds})
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return self._robot_status_text(result)
            if verb in ("stop", "start", "tick", "detach", "drop"):
                name = rest[0] if rest else "default"
                result = robot_sync(action=verb, node=name)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                if verb == "stop":
                    return f"robot: {name} stopped"
                if verb == "start":
                    return f"robot: {name} tick loop running"
                if verb == "tick":
                    return self._robot_status_text(result)
                if verb == "detach":
                    return f"robot: {name} hardware detached (simulated now)"
                return f"robot: {name} dropped"
            if verb in ("scan", "localize"):
                if not rest:
                    return self._control_robot_help()
                params: dict[str, Any] = {}
                name = "default"
                if rest[0] == "cloudcam":
                    params["source"] = "cloudcam"
                    if len(rest) > 1:
                        params["session"] = rest[1]
                    if len(rest) > 2:
                        name = rest[2]
                else:
                    try:
                        params["ranges"] = json.loads(rest[0])
                    except (ValueError, TypeError):
                        return self._control_robot_help()
                    if len(rest) > 1:
                        name = rest[1]
                result = robot_sync(action=verb, node=name, params=params)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                if verb == "scan":
                    return (f"robot: {name} scan folded in — "
                            f"{result.get('hits')} hits, "
                            f"{result.get('free')} free cells, "
                            f"{result.get('hit')} occupied")
                pose = result.get("pose", {})
                return (f"robot: {name} localized — x={pose.get('x', 0):.2f} "
                        f"y={pose.get('y', 0):.2f} "
                        f"heading={pose.get('heading_deg', 0):.1f}° "
                        f"(score {result.get('score', 0):.1f})")
            if verb == "map":
                name = rest[0] if rest else "default"
                result = robot_sync(action="map", node=name)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return (f"robot: {name} map — {result.get('observed')} "
                        f"observed cells, {result.get('occupied')} occupied, "
                        f"{result.get('free')} free\n"
                        f"{result.get('ascii', '')}")
            if verb == "plan":
                if not rest:
                    return self._control_robot_help()
                try:
                    steps = json.loads(rest[0])
                except (ValueError, TypeError):
                    return self._control_robot_help()
                name = rest[1] if len(rest) > 1 else "default"
                result = robot_sync(action="plan", node=name,
                                    params={"steps": steps})
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return (f"robot: {name} plan queued — "
                        f"{result.get('plan', {}).get('steps')} steps")
            if verb == "serve":
                name = rest[0] if rest else "default"
                result = robot_sync(action="serve", node=name)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return (f"robot: {name} serving control on 127.0.0.1:"
                        f"{result.get('port')}")
            if verb == "unserve":
                name = rest[0] if rest else "default"
                result = robot_sync(action="unserve", node=name)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return f"robot: {name} stopped serving"
            if verb == "blink":
                try:
                    l1 = int(rest[0]) if rest else 1
                    l2 = int(rest[1]) if len(rest) > 1 else 0
                    name = rest[2] if len(rest) > 2 else "default"
                except ValueError:
                    return self._control_robot_help()
                result = robot_sync(action="blink", node=name,
                                    params={"l1": l1, "l2": l2})
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return f"robot: {name} LEDs l1={l1} l2={l2}"
            if verb == "battery":
                name = rest[0] if rest else "default"
                result = robot_sync(action="battery", node=name)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return f"robot: {name} battery {result.get('battery')}%"
            if verb == "attach":
                if not rest:
                    return self._control_robot_help()
                kind = rest[0]
                port = rest[1] if len(rest) > 1 else ""
                name = rest[2] if len(rest) > 2 else "default"
                result = robot_sync(action="attach", node=name,
                                    params={"kind": kind, "port": port})
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return (f"robot: {name} attached ({result.get('backend')}) — "
                        f"firmware {result.get('firmware')}, "
                        f"battery {result.get('battery')}%")
            if verb == "export":
                if not rest:
                    return self._control_robot_help()
                output = rest[0]
                name = rest[1] if len(rest) > 1 else "default"
                result = robot_sync(action="export", node=name, output=output)
                if not result.get("ok"):
                    return f"robot: {result.get('error')}"
                return (f"robot: {name} map exported — "
                        f"{result.get('pgm')}, {result.get('json')}")
            return self._control_robot_help()
        except Exception as exc:  # noqa: BLE001 - chat replies, never traces
            record_failure(self, exc, context="robot")
            _log.exception("robot command failed: %s", exc)
            return f"robot: failed ({exc})"

    def _control_robot_help(self) -> str:
        return (
            "robot — robot daemon:\n"
            "/robot list | create [name] | status [name]\n"
            "/robot drive <left> [right] [seconds] [name]\n"
            "/robot stop | start | tick [name]\n"
            "/robot scan <ranges-json> | scan cloudcam [session]\n"
            "/robot map | localize <ranges-json> [name]\n"
            "/robot plan <steps-json> [name]\n"
            "/robot serve | unserve | served [name]\n"
            "/robot blink [l1] [l2] | battery [name]\n"
            "/robot attach loopback|serial <port> | detach [name]\n"
            "/robot drop [name] | export <dir> [name]")

    def _robot_status_text(self, status: dict[str, Any]) -> str:
        pose = status.get("pose", {})
        motors = status.get("motors", {})
        return (
            f"robot {status.get('name')}: "
            f"{'running' if status.get('running') else 'stopped'} — "
            f"backend {status.get('backend')} "
            f"({'simulated' if status.get('simulated') else 'hardware'})\n"
            f"  pose x={pose.get('x', 0):.2f} y={pose.get('y', 0):.2f} "
            f"heading={pose.get('heading_deg', 0):.1f}°\n"
            f"  motors m1={motors.get('m1')} m2={motors.get('m2')} "
            f"(written: {'yes' if status.get('motors_written') else 'no'})\n"
            f"  battery: {status.get('battery')}\n"
            f"  map: {status.get('map', {})}")
