"""Robot perception organ — a robot daemon, from the design of
echelon/robot (AGPL-3.0, Brandon Thomas 2008-2010, pattern only:
no code from that project is reproduced here).

The organ is stdlib-only (hardware/ is L4: stdlib + itself).
Depth sources reach it as plain numbers — the tool layer adapts
cloudcam frames and depth rows into scans.
"""

from __future__ import annotations

from .daemon import RobotDaemon, create_robot, drop_robot, get_robot, list_robots
from .drive import Geometry
from .errors import CODES, RobotError
from .grid import OccupancyGrid
from .sensors import Scan
from .serial import LoopbackBackend, RobotBackend, open_serial
from .state import Pose, RobotState

__all__ = [
    "CODES",
    "Geometry",
    "LoopbackBackend",
    "OccupancyGrid",
    "Pose",
    "RobotBackend",
    "RobotDaemon",
    "RobotError",
    "RobotState",
    "Scan",
    "create_robot",
    "drop_robot",
    "get_robot",
    "list_robots",
    "open_serial",
]
