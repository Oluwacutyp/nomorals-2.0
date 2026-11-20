"""Range scans — what a robot's distance sensors report.

A scan is the one shape every depth source reduces to: a list of
beam angles (radians, 0 = straight ahead, counter-clockwise
positive) with a range per beam (metres), plus the sensor's max
range. A range of None means "no return" on that beam — open
space, not zero distance.

Sources, all plain numbers by the time they arrive here:
* explicit JSON ranges from the caller;
* cloudcam point clouds (x/y/z in mm, camera space) projected
  onto the ground plane — see :func:`Scan.from_points`;
* depth rows (one row of a depth frame) — see
  :func:`Scan.from_depth_row`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import RobotError
from .state import Pose


@dataclass
class Scan:
    """One sweep of range readings."""

    angles: list[float]
    ranges: list[float | None]
    max_range: float = 5.0
    # Where the sensor sat when the scan was taken (map frame).
    origin: Pose = field(default_factory=Pose)

    def __post_init__(self) -> None:
        if len(self.angles) != len(self.ranges):
            raise RobotError(
                "bad_scan",
                f"{len(self.angles)} angles but {len(self.ranges)} ranges")
        if not self.angles:
            raise RobotError("bad_scan", "a scan needs at least one beam")
        if self.max_range <= 0:
            raise RobotError("bad_scan",
                             f"max_range must be positive, got {self.max_range}")
        for angle in self.angles:
            if not isinstance(angle, (int, float)) or isinstance(angle, bool):
                raise RobotError("bad_scan", f"angle not a number: {angle!r}")
        for index, angle in enumerate(self.angles[1:], start=1):
            if angle <= self.angles[index - 1]:
                raise RobotError(
                    "bad_scan",
                    "scan angles must be strictly increasing")
        for value in self.ranges:
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RobotError("bad_scan", f"range not a number: {value!r}")
            if value < 0 or value > self.max_range:
                raise RobotError(
                    "bad_scan",
                    f"range {value} outside 0..{self.max_range}")

    @property
    def beams(self) -> int:
        """Number of beams in the scan."""
        return len(self.angles)

    @property
    def hits(self) -> int:
        """Beams that returned a range (vs open space)."""
        return sum(1 for value in self.ranges if value is not None)

    def to_points(self) -> list[tuple[float, float]]:
        """Hit endpoints as (x, y) points in the map frame."""
        points = []
        for angle, value in zip(self.angles, self.ranges, strict=True):
            if value is None:
                continue
            world = self.origin.theta + angle
            points.append((self.origin.x + value * math.cos(world),
                           self.origin.y + value * math.sin(world)))
        return points

    @classmethod
    def from_points(cls, points: list[tuple[float, float, float]],
                    *, origin: Pose | None = None,
                    max_range: float = 5.0,
                    beams: int = 72,
                    min_height: float = -0.5,
                    max_height: float = 1.5) -> Scan:
        """Build a scan from 3D points (x, y, z in metres, robot
        frame: +x ahead, +y left, z up). Points outside the height
        band are ignored (floor and ceiling are not obstacles); each
        beam keeps the NEAREST point, because the nearest surface is
        what the sensor sees. No point on a beam = no return."""
        if beams < 1:
            raise RobotError("bad_scan", f"beams must be >= 1, got {beams}")
        nearest: dict[int, float] = {}
        for point in points:
            if len(point) != 3:
                raise RobotError("bad_scan",
                                 f"point must be (x, y, z), got {point!r}")
            x, y, z = (float(v) for v in point)
            if z < min_height or z > max_height:
                continue
            dist = math.hypot(x, y)
            if dist <= 0 or dist > max_range:
                continue
            angle = math.atan2(y, x)
            beam = int((angle + math.pi) / (2.0 * math.pi) * beams)
            beam = max(0, min(beams - 1, beam))
            if beam not in nearest or dist < nearest[beam]:
                nearest[beam] = dist
        angles = [-math.pi + (i + 0.5) * 2.0 * math.pi / beams
                  for i in range(beams)]
        ranges: list[float | None] = [nearest.get(i) for i in range(beams)]
        return cls(angles=angles, ranges=ranges, max_range=max_range,
                   origin=origin or Pose())

    @classmethod
    def from_depth_row(cls, depths: list[float | None], *,
                       fov_deg: float = 60.0,
                       max_range: float = 5.0,
                       origin: Pose | None = None) -> Scan:
        """Build a scan from one row of a depth frame: ``depths`` is
        one distance per column, left to right, in metres; the row
        spans ``fov_deg`` centred on straight ahead."""
        if not depths:
            raise RobotError("bad_scan", "a depth row needs at least one value")
        if fov_deg <= 0 or fov_deg > 360:
            raise RobotError("bad_scan",
                             f"fov must be 0..360 degrees, got {fov_deg}")
        n = len(depths)
        fov = math.radians(fov_deg)
        angles = [-fov / 2.0 + (i + 0.5) * fov / n for i in range(n)]
        ranges: list[float | None] = []
        for value in depths:
            if value is None:
                ranges.append(None)
                continue
            value = float(value)
            if value < 0 or value > max_range:
                raise RobotError(
                    "bad_scan",
                    f"depth {value} outside 0..{max_range}")
            ranges.append(value)
        return cls(angles=angles, ranges=ranges, max_range=max_range,
                   origin=origin or Pose())
