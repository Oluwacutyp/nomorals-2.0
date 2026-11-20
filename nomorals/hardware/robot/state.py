"""RobotState — the robot's live state, commanded vs applied.

Pattern from echelon/robot (AGPL-3.0, pattern only, no code taken):
the original's ``RobotState`` holds motor speeds (m1/m2) and LEDs
(l1/l2) plus what was actually written to the hardware, with
``isMotorWritten()`` telling the daemon whether the hardware has
caught up with the command. One thread sets state, another pushes
it down the serial link; a mutex guards the struct.

Devon's state adds what the original never had: the pose (x, y,
heading) the drive model integrates, the velocities, and the
battery — because a daemon that cannot say where the robot is and
how much charge it has cannot drive it.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field


@dataclass
class Pose:
    """Where the robot is, in the map frame (metres, radians)."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # heading, 0 = +x, counter-clockwise positive

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta,
                "heading_deg": math.degrees(self.theta)}


@dataclass
class RobotState:
    """One robot's live state. Thread-safe; the daemon's tick thread
    and the drive sources share it through the lock."""

    name: str
    # Commanded motor speeds, -100..100 (percent of max wheel speed).
    m1: int = 0
    m2: int = 0
    # What the hardware last confirmed applied.
    written_m1: int = 0
    written_m2: int = 0
    # LEDs, 0 = off, 1 = on.
    l1: int = 0
    l2: int = 0
    written_l1: int = 0
    written_l2: int = 0
    # Battery in percent, or None when the hardware never reported it.
    battery: float | None = None
    pose: Pose = field(default_factory=Pose)
    # Current velocities (m/s, rad/s) from the last integration step.
    linear: float = 0.0
    angular: float = 0.0
    # True while the daemon's tick loop is running.
    running: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock,
                                  repr=False, compare=False)

    def set_motors(self, m1: int, m2: int) -> None:
        """Command new motor speeds (clamped to -100..100)."""
        with self._lock:
            self.m1 = max(-100, min(100, int(m1)))
            self.m2 = max(-100, min(100, int(m2)))

    def set_blink(self, l1: int, l2: int) -> None:
        """Command the two LEDs (0/1 each)."""
        with self._lock:
            self.l1 = 1 if l1 else 0
            self.l2 = 1 if l2 else 0

    def stop_motors(self) -> None:
        """Command both motors to zero."""
        self.set_motors(0, 0)

    def stop_blink(self) -> None:
        """Command both LEDs off."""
        self.set_blink(0, 0)

    def set_written(self, m1: int, m2: int, l1: int, l2: int) -> None:
        """Record what the hardware confirmed applied."""
        with self._lock:
            self.written_m1 = int(m1)
            self.written_m2 = int(m2)
            self.written_l1 = 1 if l1 else 0
            self.written_l2 = 1 if l2 else 0

    def is_motor_written(self) -> bool:
        """True when the hardware has caught up with the command."""
        with self._lock:
            return self.m1 == self.written_m1 and self.m2 == self.written_m2

    def is_blink_written(self) -> bool:
        """True when the LEDs have caught up with the command."""
        with self._lock:
            return self.l1 == self.written_l1 and self.l2 == self.written_l2

    def snapshot(self) -> dict:
        """A plain-dict copy of the state (safe to hand to callers)."""
        with self._lock:
            return {
                "name": self.name,
                "motors": {"m1": self.m1, "m2": self.m2},
                "written": {"m1": self.written_m1, "m2": self.written_m2},
                "motors_written": self.m1 == self.written_m1
                and self.m2 == self.written_m2,
                "leds": {"l1": self.l1, "l2": self.l2},
                "battery": self.battery,
                "pose": self.pose.as_dict(),
                "velocity": {"linear": self.linear,
                             "angular": self.angular},
                "running": self.running,
            }
