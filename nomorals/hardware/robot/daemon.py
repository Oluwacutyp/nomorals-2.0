"""RobotDaemon — one robot, owned by a daemon, driven from sources.

Pattern from echelon/robot (AGPL-3.0, pattern only, no code taken):
the original is a daemon that owns the robot state and takes drive
commands from keyboard, joystick and a remote PC over TCP, keeping
hardware apart from the AI code that drives it. Devon's daemon
keeps the same shape with Devon's own sources: chat/CLI/tool (the
keyboard), scripted plans (the joystick's recorded moves), and a
loopback TCP server (the remote PC).

One daemon = one named robot: its state, its drive model, its map,
its backend (if any). The pose is the drive model's integration —
with no backend attached the daemon still drives, and its status
says so plainly (``backend: none``, ``simulated: true``), so a
simulated drive is never mistaken for a real one.
"""

from __future__ import annotations

import contextlib
import json
import socketserver
import threading
from dataclasses import dataclass

from .drive import Geometry, step_pose, velocities
from .errors import RobotError
from .grid import OccupancyGrid
from .sensors import Scan
from .serial import RobotBackend
from .state import Pose, RobotState

TICK_DT = 0.05  # 20 Hz tick


@dataclass
class PlanStep:
    """One leg of a scripted drive: motor speeds for N seconds."""

    m1: int
    m2: int
    seconds: float


class RobotDaemon:
    """A robot and the daemon that runs it."""

    def __init__(self, name: str,
                 geometry: Geometry | None = None,
                 grid: OccupancyGrid | None = None) -> None:
        if not name or not str(name).strip():
            raise RobotError("bad_name", "a robot needs a non-empty name")
        self.name = str(name).strip()
        self.geometry = geometry or Geometry()
        self.state = RobotState(name=self.name)
        self.grid = grid or OccupancyGrid()
        self.backend: RobotBackend | None = None
        self._plan: list[PlanStep] = []
        self._plan_index = 0
        self._plan_elapsed = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._server: socketserver.ThreadingTCPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._server_port: int | None = None
        self.firmware: str | None = None

    # ── backend ────────────────────────────────────────────────────
    def attach(self, backend: RobotBackend) -> dict:
        """Attach a hardware backend. Queries firmware + battery once
        so the status can name them; refuses when the backend does
        not answer."""
        with self._lock:
            if self.backend is not None:
                self.backend.close()
            self.backend = backend
            self.firmware = backend.fw()
            self.state.battery = backend.battery()
            return self.status()

    def detach(self) -> dict:
        """Drop the backend (the robot keeps driving simulated)."""
        with self._lock:
            if self.backend is not None:
                with contextlib.suppress(Exception):
                    self.backend.stop()
                self.backend.close()
                self.backend = None
                self.firmware = None
                self.state.battery = None
            return self.status()

    # ── driving ────────────────────────────────────────────────────
    def drive(self, m1: int, m2: int, seconds: float | None = None) -> dict:
        """Command motor speeds. With ``seconds``, integrate for that
        long in small steps and then stop the motors (a synchronous
        drive — the chat/CLI path). Without it, set the speeds and
        return; the tick loop (if running) keeps integrating."""
        if not isinstance(m1, int) or isinstance(m1, bool) \
                or not isinstance(m2, int) or isinstance(m2, bool):
            raise RobotError("bad_drive",
                             f"motor speeds must be integers, got {m1!r}/{m2!r}")
        if m1 < -100 or m1 > 100 or m2 < -100 or m2 > 100:
            raise RobotError(
                "bad_drive",
                f"motor speeds must be -100..100, got {m1}/{m2}")
        if seconds is not None and seconds <= 0:
            raise RobotError("bad_drive",
                             f"drive seconds must be positive, got {seconds}")
        self.state.set_motors(m1, m2)
        if seconds is not None:
            remaining = float(seconds)
            while remaining > 1e-9:
                step = min(TICK_DT, remaining)
                self.tick(step)
                remaining -= step
            self.state.stop_motors()
            self._push_motors()
        return self.status()

    def stop(self) -> dict:
        """Stop the motors and clear any running plan."""
        with self._lock:
            self._plan = []
            self._plan_index = 0
            self._plan_elapsed = 0.0
        self.state.stop_motors()
        self._push_motors()
        return self.status()

    def tick(self, dt: float = TICK_DT) -> dict:
        """One integration step: advance the pose from the current
        motor commands, run the plan, and push the motors to the
        backend (written/applied tracking). Safe to call directly —
        this is how a manual step works without the loop."""
        with self._lock:
            self._run_plan(dt)
            pose = step_pose(self.state.pose, self.state.m1, self.state.m2,
                             dt, self.geometry)
            self.state.pose = pose
            self.state.linear, self.state.angular = velocities(
                self.state.m1, self.state.m2, self.geometry)
        self._push_motors()
        return self.status()

    def _run_plan(self, dt: float) -> None:
        """Advance the scripted plan (called with the lock held)."""
        if not self._plan:
            return
        step = self._plan[self._plan_index]
        self.state.set_motors(step.m1, step.m2)
        self._plan_elapsed += dt
        if self._plan_elapsed >= step.seconds:
            self._plan_index += 1
            self._plan_elapsed = 0.0
            if self._plan_index >= len(self._plan):
                self._plan = []
                self._plan_index = 0
                self.state.stop_motors()

    def _push_motors(self) -> None:
        """Push the commanded motors/LEDs to the backend, if one is
        attached, and record what it confirmed applied. A channel
        that already matches what was written is not re-sent — the
        loopback test caught this: the first version pushed the
        motors on every tick and never pushed the LEDs at all."""
        backend = self.backend
        if backend is None:
            # Nothing to push to: the model is the state.
            self.state.set_written(self.state.m1, self.state.m2,
                                   self.state.l1, self.state.l2)
            return
        try:
            if not self.state.is_motor_written() and backend.mogo(
                    self.state.m1, self.state.m2):
                self.state.set_written(
                    self.state.m1, self.state.m2,
                    self.state.written_l1, self.state.written_l2)
            if not self.state.is_blink_written() and backend.blink(
                    self.state.l1, self.state.l2):
                self.state.set_written(
                    self.state.written_m1, self.state.written_m2,
                    self.state.l1, self.state.l2)
        except RobotError:
            raise
        except Exception as exc:  # noqa: BLE001 — a dead device is no_serial
            raise RobotError("no_serial",
                             f"robot backend stopped answering: {exc}") from exc

    def blink(self, l1: int, l2: int) -> dict:
        """Command the two LEDs."""
        self.state.set_blink(l1, l2)
        self._push_motors()
        return self.status()

    def plan(self, steps: list[dict]) -> dict:
        """Queue a scripted drive: a list of {m1, m2, seconds} legs.
        The tick loop executes them in order; a leg's motors are set
        when the leg starts, and the motors stop when the plan ends.
        The loop must be running (start it first) — a plan with no
        loop to run it is refused, not silently queued."""
        parsed: list[PlanStep] = []
        if not isinstance(steps, list) or not steps:
            raise RobotError("bad_plan",
                             "a plan is a non-empty list of steps")
        for raw in steps:
            if not isinstance(raw, dict):
                raise RobotError("bad_plan", f"plan step not an object: {raw!r}")
            try:
                m1 = int(raw["m1"])
                m2 = int(raw["m2"])
                seconds = float(raw["seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RobotError(
                    "bad_plan",
                    f"plan step needs m1, m2, seconds: {raw!r}") from exc
            if m1 < -100 or m1 > 100 or m2 < -100 or m2 > 100:
                raise RobotError("bad_plan",
                                 f"plan motor speeds must be -100..100: {raw!r}")
            if seconds <= 0:
                raise RobotError("bad_plan",
                                 f"plan step seconds must be positive: {raw!r}")
            parsed.append(PlanStep(m1, m2, seconds))
        with self._lock:
            self._plan = parsed
            self._plan_index = 0
            self._plan_elapsed = 0.0
        if not self.state.running:
            raise RobotError(
                "drive_refused",
                "start the robot first (`robot start`) — a plan needs "
                "the tick loop running to execute it")
        return self.status()

    # ── tick loop ──────────────────────────────────────────────────
    def start(self) -> dict:
        """Start the tick loop (20 Hz). Idempotent."""
        with self._lock:
            if self.state.running:
                return self.status()
            self.state.running = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name=f"robot-{self.name}")
            self._thread.start()
            return self.status()

    def stop_loop(self) -> dict:
        """Stop the tick loop (the motors stop with it)."""
        self.state.stop_motors()
        self._push_motors()
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            self.state.running = False
            self._thread = None
        return self.status()

    def _loop(self) -> None:
        while not self._stop_event.wait(TICK_DT):
            try:
                self.tick(TICK_DT)
            except RobotError:
                # A dead backend stops the loop rather than spinning
                # on refusals; the state still says what happened.
                self.state.stop_motors()
                self._stop_event.set()

    # ── perception ─────────────────────────────────────────────────
    def scan(self, scan: Scan) -> dict:
        """Fold a scan into the map from the robot's current pose."""
        scan.origin = Pose(self.state.pose.x, self.state.pose.y,
                           self.state.pose.theta)
        counts = self.grid.update(self.state.pose, scan)
        return {"ok": True, "robot": self.name, "pose": self.state.pose.as_dict(),
                "beams": scan.beams, "hits": scan.hits, **counts}

    def localize(self, scan: Scan) -> dict:
        """Re-find the robot in its own map from a fresh scan. The
        pose is updated to the match (the robot moves there)."""
        result = self.grid.localize(self.state.pose, scan)
        pose = result["pose"]
        self.state.pose = Pose(pose["x"], pose["y"], pose["theta"])
        return {"ok": True, "robot": self.name, **result}

    # ── TCP control (the remote PC) ────────────────────────────────
    def serve(self, host: str = "127.0.0.1", port: int = 0) -> dict:
        """Serve one-line drive commands over TCP — the original's
        remote PC, on loopback only. Commands: DRIVE <m1> <m2>,
        STOP, STATUS, POSE, QUIT. Each command gets one JSON line
        back. A non-loopback host is refused: remote control of a
        robot from off the machine is an explicit opt-in the daemon
        does not offer."""
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise RobotError(
                "drive_refused",
                f"robot control serves loopback only, not {host!r}")
        with self._lock:
            if self._server is not None:
                raise RobotError(
                    "already_serving",
                    f"{self.name} already serves on port {self._server_port}")
            daemon = self

            class Handler(socketserver.StreamRequestHandler):
                def handle(self) -> None:
                    for raw in self.rfile:
                        line = raw.decode("utf-8", "replace").strip()
                        reply = daemon._tcp_command(line)
                        self.wfile.write(
                            (json.dumps(reply) + "\n").encode("utf-8"))
                        if line.upper() == "QUIT":
                            return

            server = socketserver.ThreadingTCPServer(
                (host, port), Handler)
            server.daemon_threads = True
            self._server = server
            self._server_port = int(server.server_address[1])
            self._server_thread = threading.Thread(
                target=server.serve_forever, daemon=True,
                name=f"robot-tcp-{self.name}")
            self._server_thread.start()
            return {"ok": True, "robot": self.name, "host": host,
                    "port": self._server_port}

    def _tcp_command(self, line: str) -> dict:
        """One remote command line -> one reply dict."""
        parts = line.split()
        if not parts:
            return {"ok": False, "error": "empty command"}
        verb = parts[0].upper()
        try:
            if verb == "DRIVE" and len(parts) == 3:
                return self.drive(int(parts[1]), int(parts[2]))
            if verb == "STOP":
                return self.stop()
            if verb == "STATUS":
                return self.status()
            if verb == "POSE":
                return {"ok": True, "pose": self.state.pose.as_dict()}
            if verb == "QUIT":
                return {"ok": True, "bye": True}
            return {"ok": False,
                    "error": f"unknown command {verb!r} "
                             "(DRIVE <m1> <m2> | STOP | STATUS | POSE | QUIT)"}
        except (ValueError, RobotError) as exc:
            if isinstance(exc, RobotError):
                return {"ok": False, "error": str(exc), "kind": exc.kind}
            return {"ok": False, "error": f"bad numbers in {line!r}"}

    def unserve(self) -> dict:
        """Stop the TCP control server."""
        with self._lock:
            if self._server is None:
                raise RobotError(
                    "not_serving",
                    f"{self.name} is not serving a control port")
            self._server.shutdown()
            self._server.server_close()
            port = self._server_port
            self._server = None
            self._server_port = None
            self._server_thread = None
            return {"ok": True, "robot": self.name, "port": port}

    # ── views ──────────────────────────────────────────────────────
    def status(self) -> dict:
        """The whole robot, as one dict."""
        snap = self.state.snapshot()
        snap["backend"] = self.backend.kind if self.backend else "none"
        snap["simulated"] = self.backend is None
        snap["firmware"] = self.firmware
        snap["plan"] = {"steps": len(self._plan),
                        "index": self._plan_index,
                        "running": bool(self._plan)}
        snap["serving"] = self._server_port
        snap["map"] = self.grid.counts()
        snap["ok"] = True
        return snap


#: The daemon registry: one RobotDaemon per name, like camwall's
#: walls and cloudcam's sessions. Robots are per-name, never a
#: shared central queue — each daemon owns its own robot.
_ROBOTS: dict[str, RobotDaemon] = {}
_ROBOTS_LOCK = threading.Lock()


def get_robot(name: str = "default") -> RobotDaemon:
    """The named robot, or a ``no_robot`` refusal."""
    with _ROBOTS_LOCK:
        robot = _ROBOTS.get((name or "").strip() or "default")
    if robot is None:
        raise RobotError("no_robot",
                         f"no robot named {name!r} — create it first")
    return robot


def create_robot(name: str = "default",
                 geometry: Geometry | None = None) -> RobotDaemon:
    """Create (or return, if it already exists) the named robot."""
    name = (name or "").strip() or "default"
    with _ROBOTS_LOCK:
        robot = _ROBOTS.get(name)
        if robot is None:
            robot = RobotDaemon(name, geometry=geometry)
            _ROBOTS[name] = robot
        return robot


def list_robots() -> list[dict]:
    """One status dict per robot."""
    with _ROBOTS_LOCK:
        robots = list(_ROBOTS.values())
    return [robot.status() for robot in robots]


def drop_robot(name: str = "default") -> dict:
    """Stop a robot's loop + server and forget it."""
    robot = get_robot(name)
    if robot.state.running:
        robot.stop_loop()
    if robot._server is not None:
        robot.unserve()
    robot.detach()
    with _ROBOTS_LOCK:
        _ROBOTS.pop(robot.name, None)
    return {"ok": True, "dropped": robot.name}
