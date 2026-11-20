"""Robot: a robot daemon, as a brain tool.

A thin surface over :mod:`nomorals.hardware.robot` — the organ owns
the state, the drive model, the map and the daemon loop; the tool
only routes arguments and shapes the reply. One robot = one named
daemon with its own pose, map and backend. A slow or missing
device never stalls the rest: with no hardware attached the daemon
drives its own model and says so (``backend: none``,
``simulated: true``).

Scan sources (the ``scan`` action): explicit JSON ranges, a live
cloudcam point-cloud session (points → ground-plane scan), or one
depth row. The organ itself stays stdlib-only — the adapters live
here, at the tool layer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..hardware.robot import (
    LoopbackBackend,
    RobotError,
    Scan,
    create_robot,
    drop_robot,
    get_robot,
    list_robots,
    open_serial,
)

__all__ = ["register", "robot", "robot_sync"]

_ACTIONS = ("status", "list", "create", "drive", "stop", "start", "tick",
            "scan", "map", "localize", "plan", "serve", "unserve", "served",
            "blink", "battery", "attach", "detach", "drop", "export")


def _params(params: dict | None) -> dict:
    return dict(params or {})


def _name(node: str) -> str:
    return (node or "").strip() or "default"


def _scan_from_params(args: dict) -> Scan:
    """Build a scan from the caller's params. Three shapes:
    * ``ranges``: {"angles": [deg...], "ranges": [m...]} (or a bare
      list of ranges + ``fov`` in degrees);
    * ``source``: "cloudcam" (+ ``session``) — the latest frame of
      a cloudcam session, projected onto the ground plane;
    * ``source``: "depth" (+ ``depths`` + ``fov``) — one depth row.
    Anything else is a named refusal, never a guessed scan."""
    source = str(args.get("source") or "").strip().lower()
    max_range = float(args.get("max_range") or 5.0)
    if source in ("", "ranges"):
        raw = args.get("ranges")
        if raw is None:
            raise RobotError("no_scan",
                             "scan needs ranges=<{angles, ranges}> or "
                             "source=cloudcam|depth")
        if isinstance(raw, dict):
            angles = [math.radians(float(a))
                      for a in raw.get("angles", [])]
            ranges: list[float | None] = [
                (None if v is None else float(v))
                for v in raw.get("ranges", [])]
            return Scan(angles=angles, ranges=ranges, max_range=max_range)
        if isinstance(raw, list):
            fov = float(args.get("fov") or 60.0)
            depths: list[float | None] = [
                (None if v is None else float(v)) for v in raw]
            return Scan.from_depth_row(depths, fov_deg=fov,
                                       max_range=max_range)
        raise RobotError("bad_scan",
                         f"ranges must be a list or {{angles, ranges}}, got {raw!r}")
    if source == "cloudcam":
        from ..media.cloudcam.errors import CloudCamError
        from ..media.cloudcam.registry import get_session

        try:
            session = get_session(str(args.get("session") or "default"))
            points = session.latest_points()
        except CloudCamError as exc:
            raise RobotError(
                "no_depth",
                f"no cloudcam depth to scan from: {exc}") from exc
        if not points:
            raise RobotError(
                "no_depth",
                f"cloudcam session {session.name!r} has no frame yet — "
                "start the session and let a frame arrive first")
        # CloudCamPoint is Kinect camera space (mm): +x right, +y
        # down, +z forward. The scan frame is +x ahead, +y left,
        # z up — so (ahead, left, up) = (z, -x, -y), in metres.
        # The height band keeps the robot's own eye-line and drops
        # floor and ceiling.
        converted = [(p.z / 1000.0, -p.x / 1000.0, -p.y / 1000.0)
                     for p in points]
        return Scan.from_points(converted, max_range=max_range,
                                beams=int(args.get("beams") or 72))
    if source == "depth":
        depths = args.get("depths")
        if not isinstance(depths, list) or not depths:
            raise RobotError("bad_scan",
                             "source=depth needs depths=<[metres...]>")
        fov = float(args.get("fov") or 60.0)
        values: list[float | None] = [
            (None if v is None else float(v)) for v in depths]
        return Scan.from_depth_row(values, fov_deg=fov, max_range=max_range)
    raise RobotError("no_scan", f"unknown scan source {source!r}")


def robot(action: str = "status", **kwargs: Any) -> dict[str, Any]:
    """The robot family, as a plain function (sync — the organ is
    thread-based, not async)."""
    action = (action or "status").lower()
    if action not in _ACTIONS:
        raise ValueError(f"unknown robot action {action!r}; "
                         f"use one of {', '.join(_ACTIONS)}")
    args = _params(kwargs.get("params"))
    name = _name(str(kwargs.get("node") or ""))
    if action == "list":
        return {"robots": list_robots()}
    if action == "served":
        return {"robots": [
            {"name": r["name"], "port": r["serving"]}
            for r in list_robots() if r.get("serving")]}
    if action == "create":
        return create_robot(name).status()
    daemon = get_robot(name)
    if action == "status":
        return daemon.status()
    if action == "drive":
        m1 = int(kwargs.get("value") if kwargs.get("value") is not None
                 else args.get("m1", 0))
        m2 = int(args.get("m2", m1))
        seconds = args.get("seconds")
        return daemon.drive(m1, m2,
                            seconds=(None if seconds is None else float(seconds)))
    if action == "stop":
        return daemon.stop()
    if action == "start":
        return daemon.start()
    if action == "tick":
        return daemon.tick(dt=float(args.get("dt") or 0.05))
    if action == "scan":
        return daemon.scan(_scan_from_params(args))
    if action == "map":
        grid = daemon.grid
        return {"ok": True, "robot": name, "pose": daemon.state.pose.as_dict(),
                **grid.counts(),
                "ascii": grid.render_ascii(
                    width=int(args.get("width") or 80))}
    if action == "localize":
        return daemon.localize(_scan_from_params(args))
    if action == "plan":
        steps = args.get("steps")
        if isinstance(steps, str):
            steps = json.loads(steps)
        return daemon.plan(steps)
    if action == "serve":
        return daemon.serve(host=str(args.get("host") or "127.0.0.1"),
                            port=int(args.get("port") or 0))
    if action == "unserve":
        return daemon.unserve()
    if action == "blink":
        l1 = int(args.get("l1", 1))
        l2 = int(args.get("l2", 0))
        return daemon.blink(l1, l2)
    if action == "battery":
        backend = daemon.backend
        if backend is None:
            raise RobotError(
                "no_serial",
                f"{name} has no hardware attached — nothing to read a "
                "battery from (status shows the last known value)")
        daemon.state.battery = backend.battery()
        return {"ok": True, "robot": name, "battery": daemon.state.battery}
    if action == "attach":
        kind = str(args.get("kind") or "").strip().lower()
        if kind == "loopback":
            return daemon.attach(LoopbackBackend())
        if kind == "serial":
            port = str(args.get("port") or "")
            if not port:
                raise RobotError("no_serial",
                                 "attach kind=serial needs port=<device path>")
            return daemon.attach(open_serial(
                port, baud=int(args.get("baud") or 9600)))
        raise RobotError(
            "no_serial",
            f"unknown attach kind {kind!r} (loopback | serial)")
    if action == "detach":
        return daemon.detach()
    if action == "drop":
        return drop_robot(name)
    if action == "export":
        output = str(kwargs.get("output") or "")
        if not output:
            raise ValueError("export needs output=<dir>")
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        pgm = out / "map.pgm"
        pgm.write_bytes(daemon.grid.to_pgm())
        saved = daemon.grid.save(out / "map.json")
        ascii_path = out / "map.txt"
        ascii_path.write_text(daemon.grid.render_ascii(width=80),
                              encoding="utf-8")
        return {"ok": True, "robot": name,
                "pgm": str(pgm), "json": saved["path"],
                "ascii": str(ascii_path), **daemon.grid.counts()}
    raise ValueError(f"unhandled robot action {action!r}")


def robot_sync(action: str = "status", **kwargs: Any) -> dict[str, Any]:
    """Sync entry — the registry, the chat handler and the CLI call
    this directly; the organ is thread-based, so there is no event
    loop to bridge."""
    try:
        return robot(action, **kwargs)
    except RobotError as exc:
        return {"ok": False, "error": str(exc), "kind": exc.kind}


def register(registry: Any) -> None:
    """Attach the robot family to a registry."""
    @registry.register(
        "robot",
        description=(
            "A robot daemon — drive a robot, map where it is, find "
            "it again (echelon/robot pattern, AGPL, pattern only): "
            "one named robot holds its pose, its motors and its "
            "occupancy-grid map. Actions: status=<one robot's state, "
            "node=<robot>>; list=<robots now>; create=<make a robot, "
            "node=<robot>>; drive=<mogo, value=<left -100..100>, "
            "params=<{m2, seconds}>>; stop=<stop the motors + plan, "
            "node=<robot>>; start=<start the 20 Hz tick loop, "
            "node=<robot>>; tick=<one manual step, params=<{dt}>>; "
            "scan=<fold a scan into the map, params=<{ranges | "
            "source=cloudcam|depth, session, depths, fov, "
            "max_range}>>; map=<counts + ascii map, node=<robot>>; "
            "localize=<re-find the robot in its map from a scan, "
            "params=<same as scan>>; plan=<scripted drive, params=<"
            "{steps: [{m1, m2, seconds}...]}>> (needs the loop "
            "running); serve=<loopback TCP control, params=<{host, "
            "port}>>; unserve=<stop serving>; served=<servers now>; "
            "blink=<LEDs, params=<{l1, l2}>>; battery=<read the "
            "attached hardware's battery>; attach=<kind=loopback|"
            "serial, params=<{port, baud}>>; detach=<drop the "
            "hardware>; drop=<forget the robot>; export=<write "
            "map.pgm + map.json + map.txt, output=<dir>>. With no "
            "hardware attached the daemon drives its own model and "
            "says so (backend: none, simulated: true) — a simulated "
            "drive is never a real one. action=<status|list|create|"
            "drive|stop|start|tick|scan|map|localize|plan|serve|"
            "unserve|served|blink|battery|attach|detach|drop|export>; "
            "node=<robot name>; value=<left motor -100..100>; output="
            "<dir to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def robot_tool(action: str = "status",
                   sources: list[str] | None = None,
                   output: str = "", kind: str = "",
                   node: str = "", value: Any = None,
                   command: str = "",
                   params: dict | None = None,
                   title: str = "",
                   ) -> dict[str, Any]:
        return robot(action, sources=list(sources or []),
                     output=output, kind=kind, node=node, value=value,
                     command=command, params=params, title=title)
