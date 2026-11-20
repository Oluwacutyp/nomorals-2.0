"""Differential-drive kinematics — the original never had a model.

The echelon/robot daemon pushes motor speeds to a Serializer and
never asks where the robot ends up. Devon integrates the pose, so
a drive command has a verifiable consequence: after `mogo` + a
tick, the pose moved exactly where the model says it did.

Model: a unicycle. Motor commands -100..100 map linearly to wheel
speeds (max_wheel_speed m/s at 100). With wheelbase B (metres):

    v = (vl + vr) / 2            forward speed (m/s)
    w = (vr - vl) / B            yaw rate (rad/s)

Integration is the exact arc (instantaneous centre of curvature),
not Euler stepping, so a long turn lands on the true circle and a
straight drive lands on the true line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import RobotError
from .state import Pose


@dataclass(frozen=True)
class Geometry:
    """The robot's drive geometry. Defaults fit a small rover."""

    wheelbase: float = 0.30      # metres between the drive wheels
    max_wheel_speed: float = 0.5  # m/s at motor command 100


def wheel_speeds(m1: int, m2: int, geometry: Geometry) -> tuple[float, float]:
    """Motor commands (-100..100) -> left/right wheel speeds (m/s)."""
    for value in (m1, m2):
        if not isinstance(value, int) or isinstance(value, bool):
            raise RobotError("bad_drive",
                             f"motor speeds must be integers, got {m1!r}/{m2!r}")
        if value < -100 or value > 100:
            raise RobotError(
                "bad_drive",
                f"motor speeds must be -100..100, got {m1}/{m2}")
    scale = geometry.max_wheel_speed / 100.0
    return m1 * scale, m2 * scale


def velocities(m1: int, m2: int,
               geometry: Geometry) -> tuple[float, float]:
    """Motor commands -> (forward m/s, yaw rad/s)."""
    vl, vr = wheel_speeds(m1, m2, geometry)
    return (vl + vr) / 2.0, (vr - vl) / geometry.wheelbase


def step_pose(pose: Pose, m1: int, m2: int, dt: float,
              geometry: Geometry) -> Pose:
    """Integrate one tick exactly. Returns the new pose (the input
    pose is not mutated). dt <= 0 is a no-op that returns a copy."""
    if dt <= 0:
        return Pose(pose.x, pose.y, pose.theta)
    v, w = velocities(m1, m2, geometry)
    if abs(w) < 1e-9:
        # Straight: the exact arc degenerates to a line.
        return Pose(pose.x + v * dt * math.cos(pose.theta),
                    pose.y + v * dt * math.sin(pose.theta),
                    pose.theta)
    # Exact arc about the instantaneous centre of curvature.
    radius = v / w
    dtheta = w * dt
    theta2 = pose.theta + dtheta
    return Pose(
        pose.x + radius * (math.sin(theta2) - math.sin(pose.theta)),
        pose.y + radius * (math.cos(pose.theta) - math.cos(theta2)),
        _wrap_angle(theta2),
    )


def _wrap_angle(theta: float) -> float:
    """Wrap to (-pi, pi]."""
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    while theta > math.pi:
        theta -= 2.0 * math.pi
    return theta
