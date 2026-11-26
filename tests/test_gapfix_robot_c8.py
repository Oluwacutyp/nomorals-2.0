"""§3C.8 — robot daemon: echelon/robot pattern (AGPL, pattern only).

Tests the organ (kinematics, state, scans, grid, daemon), the brain
tool, the CLI, and the chat wiring. The organ is stdlib-only and
hardware-free by design: every live-hardware path refuses by name
when nothing is attached, and the daemon's simulated pose is always
labelled simulated.

Tool-count pin: 784 after this lane (C.4) added camwall; 785 after
this lane (C.8) added robot; 786 after this lane (goal-class) added goal_class.
"""

from __future__ import annotations

import contextlib
import json
import math
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from nomorals.hardware.robot import (
    Geometry,
    LoopbackBackend,
    OccupancyGrid,
    Pose,
    RobotError,
    Scan,
    create_robot,
    drop_robot,
    get_robot,
)
from nomorals.hardware.robot.drive import step_pose, velocities
from nomorals.hardware.robot.state import RobotState


def _wall_scan(distance: float = 2.0, beams: int = 9,
               span_deg: float = 80.0) -> Scan:
    """A flat wall straight ahead: every beam hits at ``distance``."""
    span = math.radians(span_deg)
    angles = [-span / 2.0 + (i + 0.5) * span / beams for i in range(beams)]
    return Scan(angles=angles, ranges=[distance] * beams, max_range=5.0)


def _corner_scan() -> Scan:
    """An L-shaped obstacle: a wall ahead + a wall to the left, so
    x, y and heading are all observable for localization."""
    angles = [math.radians(a) for a in (-80, -60, -40, -20, 0, 20, 40, 60, 80)]
    # Beams left of -45 see the left wall at ~1.5 m; the rest see
    # the front wall at 2 m.
    ranges = [1.5 if a < -45 else 2.0 for a in (-80, -60, -40, -20, 0, 20, 40, 60, 80)]
    return Scan(angles=angles, ranges=ranges, max_range=5.0)


class KinematicsTests(unittest.TestCase):
    def test_straight_drive_lands_exact(self) -> None:
        pose = step_pose(Pose(), 100, 100, 2.0, Geometry())
        self.assertAlmostEqual(pose.x, 1.0, places=9)
        self.assertAlmostEqual(pose.y, 0.0, places=9)
        self.assertAlmostEqual(pose.theta, 0.0, places=9)

    def test_spin_in_place(self) -> None:
        pose = step_pose(Pose(), 100, -100, 1.0, Geometry())
        self.assertAlmostEqual(pose.x, 0.0, places=9)
        self.assertAlmostEqual(pose.y, 0.0, places=9)
        # w = (vr - vl) / B = (-0.5 - 0.5) / 0.3 = -3.333 rad/s;
        # the pose normalises the heading into (-pi, pi].
        self.assertAlmostEqual(pose.theta, 2 * math.pi - 10.0 / 3.0,
                               places=6)

    def test_arc_is_exact_not_euler(self) -> None:
        # A quarter-circle arc: v = 0.25 m/s, w = 0.5 rad/s for pi s
        # turns 90 degrees on a 0.5 m radius. Euler stepping would
        # drift off the circle; the exact arc lands on it.
        geometry = Geometry(wheelbase=0.5, max_wheel_speed=0.5)
        # m1/m2 for v=0.25, w=0.5: vr = v + w*B/2 = 0.375 -> 75,
        # vl = v - w*B/2 = 0.125 -> 25.
        pose = step_pose(Pose(), 25, 75, math.pi, geometry)
        self.assertAlmostEqual(pose.x, 0.5, places=6)
        self.assertAlmostEqual(pose.y, 0.5, places=6)
        self.assertAlmostEqual(pose.theta, math.pi / 2.0, places=6)

    def test_velocities(self) -> None:
        v, w = velocities(100, 100, Geometry())
        self.assertAlmostEqual(v, 0.5)
        self.assertAlmostEqual(w, 0.0)
        v, w = velocities(-100, 100, Geometry())
        self.assertAlmostEqual(v, 0.0)
        self.assertAlmostEqual(w, 1.0 / 0.3, places=6)

    def test_bad_drive_refused(self) -> None:
        for m1, m2 in ((101, 0), (0, -101), (50.5, 0), ("50", 0)):
            with self.assertRaises(RobotError) as ctx:
                velocities(m1, m2, Geometry())  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.kind, "bad_drive")

    def test_zero_dt_is_noop(self) -> None:
        pose = step_pose(Pose(1.0, 2.0, 0.5), 100, 100, 0.0, Geometry())
        self.assertEqual((pose.x, pose.y, pose.theta), (1.0, 2.0, 0.5))


class StateTests(unittest.TestCase):
    def test_commanded_vs_written(self) -> None:
        state = RobotState(name="t")
        self.assertTrue(state.is_motor_written())
        state.set_motors(50, -50)
        self.assertFalse(state.is_motor_written())
        state.set_written(50, -50, 0, 0)
        self.assertTrue(state.is_motor_written())
        self.assertTrue(state.is_blink_written())

    def test_motor_clamp(self) -> None:
        state = RobotState(name="t")
        state.set_motors(500, -500)
        self.assertEqual((state.m1, state.m2), (100, -100))

    def test_stop(self) -> None:
        state = RobotState(name="t")
        state.set_motors(80, 80)
        state.set_blink(1, 1)
        state.stop_motors()
        state.stop_blink()
        self.assertEqual((state.m1, state.m2, state.l1, state.l2),
                         (0, 0, 0, 0))


class ScanTests(unittest.TestCase):
    def test_validation(self) -> None:
        with self.assertRaises(RobotError) as ctx:
            Scan(angles=[0.0, 0.1], ranges=[1.0], max_range=5.0)
        self.assertEqual(ctx.exception.kind, "bad_scan")
        with self.assertRaises(RobotError) as ctx:
            Scan(angles=[0.1, 0.0], ranges=[1.0, 1.0], max_range=5.0)
        self.assertEqual(ctx.exception.kind, "bad_scan")
        with self.assertRaises(RobotError) as ctx:
            Scan(angles=[0.0], ranges=[6.0], max_range=5.0)
        self.assertEqual(ctx.exception.kind, "bad_scan")
        with self.assertRaises(RobotError) as ctx:
            Scan(angles=[], ranges=[], max_range=5.0)
        self.assertEqual(ctx.exception.kind, "bad_scan")

    def test_to_points(self) -> None:
        scan = Scan(angles=[0.0], ranges=[2.0], max_range=5.0,
                    origin=Pose(1.0, 1.0, 0.0))
        self.assertEqual(scan.to_points(), [(3.0, 1.0)])
        self.assertEqual(scan.hits, 1)
        self.assertEqual(scan.beams, 1)

    def test_from_points_nearest_wins(self) -> None:
        # Two points on the same beam: the nearer one is the hit.
        scan = Scan.from_points([(2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                 (0.0, 3.0, 0.0)],
                                max_range=5.0, beams=4)
        self.assertEqual(scan.hits, 2)
        forward = scan.ranges[2]  # beam centred on 0 radians
        self.assertIsNotNone(forward)
        self.assertAlmostEqual(forward, 1.0, places=6)

    def test_from_points_height_band(self) -> None:
        # A point at the floor and one at the ceiling are ignored.
        scan = Scan.from_points([(1.0, 0.0, -2.0), (1.0, 0.0, 3.0),
                                 (1.0, 0.0, 0.0)],
                                max_range=5.0, beams=4)
        self.assertEqual(scan.hits, 1)

    def test_from_depth_row(self) -> None:
        scan = Scan.from_depth_row([1.0, 2.0, None], fov_deg=60.0)
        self.assertEqual(scan.beams, 3)
        self.assertEqual(scan.hits, 2)
        self.assertAlmostEqual(scan.angles[1], 0.0, places=9)
        with self.assertRaises(RobotError) as ctx:
            Scan.from_depth_row([], fov_deg=60.0)
        self.assertEqual(ctx.exception.kind, "bad_scan")


class GridTests(unittest.TestCase):
    def test_update_marks_hit_and_carves_free(self) -> None:
        grid = OccupancyGrid()
        scan = _wall_scan()
        grid.update(Pose(), scan)
        # The hit cell (2 m ahead) is occupied; the midpoint is free.
        self.assertGreater(grid.probability_at(2.0, 0.0), 0.65)
        self.assertLess(grid.probability_at(1.0, 0.0), 0.5)
        self.assertEqual(grid.counts()["scans"], 1)

    def test_probability_unknown_is_half(self) -> None:
        grid = OccupancyGrid()
        self.assertEqual(grid.probability_at(0.0, 0.0), 0.5)

    def test_ascii_shows_rays_and_hit(self) -> None:
        grid = OccupancyGrid()
        grid.update(Pose(), _wall_scan())
        art = grid.render_ascii(width=40)
        self.assertIn("#", art)
        self.assertIn(".", art)

    def test_pgm_header(self) -> None:
        grid = OccupancyGrid()
        blob = grid.to_pgm()
        self.assertTrue(blob.startswith(b"P5\n400 400\n255\n"))

    def test_save_load_round_trip(self) -> None:
        grid = OccupancyGrid()
        grid.update(Pose(), _wall_scan())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            grid.save(path)
            loaded = OccupancyGrid.load(path)
        self.assertEqual(loaded.counts(), grid.counts())
        self.assertAlmostEqual(loaded.probability_at(2.0, 0.0),
                               grid.probability_at(2.0, 0.0))

    def test_localize_refuses_empty_map(self) -> None:
        grid = OccupancyGrid()
        with self.assertRaises(RobotError) as ctx:
            grid.localize(Pose(), _wall_scan())
        self.assertEqual(ctx.exception.kind, "pose_unknown")

    def test_localize_recovers_kidnapped_pose(self) -> None:
        grid = OccupancyGrid()
        truth = Pose(1.0, 0.5, 0.2)
        grid.update(truth, _corner_scan())
        grid.update(truth, _corner_scan())
        kidnapped = Pose(1.2, 0.35, 0.35)
        found = grid.localize(kidnapped, _corner_scan())
        pose = found["pose"]
        before = math.hypot(kidnapped.x - truth.x, kidnapped.y - truth.y)
        after = math.hypot(pose["x"] - truth.x, pose["y"] - truth.y)
        self.assertLess(after, before)
        self.assertLess(abs(pose["theta"] - truth.theta),
                        abs(kidnapped.theta - truth.theta))


class DaemonTests(unittest.TestCase):
    def tearDown(self) -> None:
        for name in ("t-drive", "t-plan", "t-tcp", "t-attach", "t-loop"):
            with contextlib.suppress(RobotError):
                drop_robot(name)

    def test_drive_seconds_moves_pose(self) -> None:
        robot = create_robot("t-drive")
        robot.drive(100, 100, seconds=2.0)
        self.assertAlmostEqual(robot.state.pose.x, 1.0, places=6)
        self.assertEqual((robot.state.m1, robot.state.m2), (0, 0))
        self.assertTrue(robot.state.is_motor_written())
        status = robot.status()
        self.assertEqual(status["backend"], "none")
        self.assertTrue(status["simulated"])

    def test_drive_without_seconds_sets_motors_only(self) -> None:
        robot = create_robot("t-drive")
        robot.drive(50, 50)
        self.assertEqual((robot.state.m1, robot.state.m2), (50, 50))
        self.assertAlmostEqual(robot.state.pose.x, 0.0, places=9)

    def test_manual_tick(self) -> None:
        robot = create_robot("t-drive")
        robot.drive(100, 100)
        robot.tick(dt=1.0)
        self.assertAlmostEqual(robot.state.pose.x, 0.5, places=9)

    def test_plan_needs_loop(self) -> None:
        robot = create_robot("t-plan")
        with self.assertRaises(RobotError) as ctx:
            robot.plan([{"m1": 100, "m2": 100, "seconds": 0.2}])
        self.assertEqual(ctx.exception.kind, "drive_refused")

    def test_plan_executes_on_loop(self) -> None:
        robot = create_robot("t-plan")
        robot.start()
        try:
            robot.plan([{"m1": 100, "m2": 100, "seconds": 0.2},
                        {"m1": 0, "m2": 0, "seconds": 0.2}])
            # Wait for the plan to drain (0.4 s of driving + margin).
            deadline = threading.Event()
            deadline.wait(1.5)
            self.assertAlmostEqual(robot.state.pose.x, 0.1, places=1)
            self.assertEqual((robot.state.m1, robot.state.m2), (0, 0))
        finally:
            robot.stop_loop()

    def test_loop_start_stop(self) -> None:
        robot = create_robot("t-loop")
        robot.drive(100, 100)
        robot.start()
        self.assertTrue(robot.state.running)
        threading.Event().wait(0.3)
        self.assertGreater(robot.state.pose.x, 0.05)
        robot.stop_loop()
        self.assertFalse(robot.state.running)
        self.assertEqual((robot.state.m1, robot.state.m2), (0, 0))

    def test_attach_loopback(self) -> None:
        robot = create_robot("t-attach")
        backend = LoopbackBackend()
        status = robot.attach(backend)
        self.assertEqual(status["backend"], "loopback")
        self.assertFalse(status["simulated"])
        self.assertEqual(status["firmware"], "loopback-1.0")
        self.assertEqual(status["battery"], 100.0)
        robot.drive(60, 60, seconds=0.1)
        # One mogo per command, not one per tick: the second tick
        # must not re-send motors the hardware already has.
        self.assertEqual(backend.commands, [("mogo", 60, 60),
                                            ("mogo", 0, 0)])
        self.assertTrue(robot.state.is_motor_written())
        robot.blink(1, 0)
        self.assertEqual(backend.commands[-1], ("blink", 1, 0))
        self.assertTrue(robot.state.is_blink_written())
        robot.detach()
        self.assertEqual(robot.status()["backend"], "none")

    def test_attach_serial_missing_port_refused(self) -> None:
        from nomorals.hardware.robot.serial import open_serial

        robot = create_robot("t-attach")
        with self.assertRaises(RobotError) as ctx:
            robot.attach(open_serial("/dev/nonexistent-robot-xyz"))
        self.assertEqual(ctx.exception.kind, "no_serial")

    def test_tcp_control(self) -> None:
        robot = create_robot("t-tcp")
        served = robot.serve()
        port = served["port"]
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                handle = sock.makefile("rw")
                handle.write("STATUS\n")
                handle.flush()
                reply = json.loads(handle.readline())
                self.assertTrue(reply["ok"])
                self.assertEqual(reply["name"], "t-tcp")
                handle.write("DRIVE 50 50\n")
                handle.flush()
                reply = json.loads(handle.readline())
                self.assertEqual(reply["motors"], {"m1": 50, "m2": 50})
                handle.write("BOGUS\n")
                handle.flush()
                reply = json.loads(handle.readline())
                self.assertFalse(reply["ok"])
                handle.write("QUIT\n")
                handle.flush()
                reply = json.loads(handle.readline())
                self.assertTrue(reply.get("bye"))
        finally:
            robot.unserve()
        with self.assertRaises(RobotError) as ctx:
            robot.unserve()
        self.assertEqual(ctx.exception.kind, "not_serving")

    def test_tcp_non_loopback_refused(self) -> None:
        robot = create_robot("t-tcp")
        with self.assertRaises(RobotError) as ctx:
            robot.serve(host="0.0.0.0")
        self.assertEqual(ctx.exception.kind, "drive_refused")

    def test_scan_and_map(self) -> None:
        robot = create_robot("t-drive")
        result = robot.scan(_wall_scan())
        self.assertTrue(result["ok"])
        self.assertEqual(result["hits"], 9)
        self.assertGreater(result["free"], 0)
        self.assertEqual(robot.grid.counts()["scans"], 1)

    def test_get_unknown_robot_refused(self) -> None:
        with self.assertRaises(RobotError) as ctx:
            get_robot("no-such-robot")
        self.assertEqual(ctx.exception.kind, "no_robot")


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.tools.robot import robot_sync

        self.sync = robot_sync

    def tearDown(self) -> None:
        for name in ("tool-robot",):
            with contextlib.suppress(RobotError):
                drop_robot(name)

    def test_tool_count_is_787_after_c8_added_robot(self) -> None:
        # 784 after this lane (C.4) added camwall; 785 after this lane
        # (C.8) added robot; 786 after this lane (goal-class) added goal_class.
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 804)
        self.assertIn("robot", registry._tools)

    def test_tool_status_drive_scan_map(self) -> None:
        self.assertEqual(
            self.sync(action="create", node="tool-robot")["name"], "tool-robot")
        drove = self.sync(action="drive", node="tool-robot", value=100,
                          params={"m2": 100, "seconds": 1.0})
        self.assertAlmostEqual(drove["pose"]["x"], 0.5, places=6)
        scanned = self.sync(
            action="scan", node="tool-robot",
            params={"ranges": {"angles": [-20, 0, 20],
                               "ranges": [2.0, 2.0, 2.0]}})
        self.assertTrue(scanned["ok"])
        self.assertEqual(scanned["hits"], 3)
        mapped = self.sync(action="map", node="tool-robot")
        self.assertIn("#", mapped["ascii"])
        self.assertIn(".", mapped["ascii"])

    def test_tool_unknown_action(self) -> None:
        with self.assertRaises(ValueError):
            self.sync(action="fly", node="tool-robot")

    def test_tool_scan_no_data_refused(self) -> None:
        self.sync(action="create", node="tool-robot")
        result = self.sync(action="scan", node="tool-robot", params={})
        self.assertFalse(result["ok"])
        self.assertEqual(result["kind"], "no_scan")

    def test_tool_scan_cloudcam_no_frame_refused(self) -> None:
        from nomorals.media.cloudcam.registry import load_session

        self.sync(action="create", node="tool-robot")
        # An idle session exists but has no frame yet.
        load_session("robot-idle", source="idle")
        result = self.sync(action="scan", node="tool-robot",
                           params={"source": "cloudcam",
                                   "session": "robot-idle"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["kind"], "no_depth")
        # An unknown session is the same refusal, not a traceback.
        result = self.sync(action="scan", node="tool-robot",
                           params={"source": "cloudcam",
                                   "session": "no-such-session-xyz"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["kind"], "no_depth")

    def test_tool_scan_cloudcam_frame(self) -> None:
        import time

        from nomorals.media.cloudcam.registry import load_session

        self.sync(action="create", node="tool-robot")
        # A wall 2 m ahead, as a cloudcam frames file (mm, camera
        # space: +x right, +y down, +z forward).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.json"
            wall = [[x, 0, 2000] for x in range(-500, 501, 100)]
            path.write_text(json.dumps({"frames": [wall]}),
                            encoding="utf-8")
            session = load_session("robot-frames", source="file", path=path)
            session.start()
            try:
                deadline = time.time() + 5
                while time.time() < deadline:
                    receipt = session.tick()
                    if receipt["point_count"] > 0:
                        break
                    time.sleep(0.02)
                self.assertGreater(receipt["point_count"], 0)
                result = self.sync(
                    action="scan", node="tool-robot",
                    params={"source": "cloudcam", "session": "robot-frames"})
                self.assertTrue(result["ok"])
                self.assertGreater(result["hits"], 0)
            finally:
                session.stop()

    def test_tool_export_writes_files(self) -> None:
        self.sync(action="create", node="tool-robot")
        self.sync(action="scan", node="tool-robot",
                  params={"ranges": {"angles": [0], "ranges": [2.0]}})
        with tempfile.TemporaryDirectory() as tmp:
            result = self.sync(action="export", node="tool-robot",
                               output=tmp)
            self.assertTrue(result["ok"])
            self.assertTrue(Path(result["pgm"]).exists())
            self.assertTrue(Path(result["json"]).exists())
            self.assertTrue(Path(result["ascii"]).exists())
            loaded = OccupancyGrid.load(result["json"])
            self.assertEqual(loaded.counts()["scans"], 1)

    def test_tool_battery_without_hardware_refused(self) -> None:
        self.sync(action="create", node="tool-robot")
        result = self.sync(action="battery", node="tool-robot")
        self.assertFalse(result["ok"])
        self.assertEqual(result["kind"], "no_serial")


class WiringTests(unittest.TestCase):
    def test_robot_in_exactly_one_list_group(self) -> None:
        from nomorals.social.chat.control import CONTROL_COMMANDS, LIST_GROUPS

        groups = [title for title, kinds in LIST_GROUPS if "robot" in kinds]
        self.assertEqual(groups, ["media system — music · playback · video · podcast"])
        self.assertIn("robot", CONTROL_COMMANDS)

    def test_robot_has_real_help_page(self) -> None:
        from nomorals.social.chat.control import COMMAND_DETAILS, LIST_ONELINERS

        page = COMMAND_DETAILS["robot"]
        self.assertTrue(page["what"].startswith("Robot daemon"))
        self.assertIn("drive <left>", page["usage"])
        self.assertNotEqual(page["what"], LIST_ONELINERS["robot"])
        self.assertIn("robot", LIST_ONELINERS)

    def test_robot_not_on_public_menu(self) -> None:
        from nomorals.social.chat.control import PUBLIC_COMMANDS

        self.assertNotIn("robot", PUBLIC_COMMANDS)
        self.assertEqual(len(PUBLIC_COMMANDS), 279)

    def test_cli_alias_and_parser(self) -> None:
        from nomorals.cmdline.parser import CLI_ALIASES, _parser

        self.assertEqual(CLI_ALIASES["robot"], ["rb"])
        parser = _parser()
        args = parser.parse_args(["robot", "status", "mybot"])
        self.assertEqual(args.robot_action, "status")
        self.assertEqual(args.robot, "mybot")
        args = parser.parse_args(["rb", "drive", "100", "100",
                                  "--seconds", "2"])
        self.assertEqual(args.robot_action, "drive")
        self.assertEqual(args.left, 100)
        self.assertEqual(args.right, 100)
        self.assertEqual(args.seconds, 2.0)

    def test_chat_handler_routes(self) -> None:
        from nomorals.agents.partner.runtime_robot import RuntimeRobotMixin

        mixin = RuntimeRobotMixin()
        reply = mixin._control_robot("create chat-robot", "test")
        self.assertIn("chat-robot", reply)
        reply = mixin._control_robot("drive 100 100 1 chat-robot", "test")
        self.assertIn("chat-robot", reply)
        self.assertIn("x=0.50", reply)
        reply = mixin._control_robot("status chat-robot", "test")
        self.assertIn("simulated", reply)
        reply = mixin._control_robot("drop chat-robot", "test")
        self.assertIn("dropped", reply)


if __name__ == "__main__":
    unittest.main()
