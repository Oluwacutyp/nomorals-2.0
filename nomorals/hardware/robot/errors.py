"""Robot errors — every refusal names itself, never a fake reading.

Pattern from echelon/robot (AGPL-3.0, pattern only, no code taken):
the original's RCSerializer checks ACK/NACK and refuses a command
it cannot confirm. Devon does the same at every seam: a missing
device, a missing scan, or a bad drive command is a named refusal,
never a silent default or an invented map.
"""

from __future__ import annotations

#: Every refusal code the robot organ can raise. A code is stable
#: API: chat/CLI/tool replies carry it verbatim.
CODES = (
    "no_robot",      # no robot by that name exists (yet)
    "no_serial",     # no serial robot hardware attached / reachable
    "no_joystick",   # no joystick device attached
    "no_scan",       # no scan data to work with
    "no_depth",      # no depth source (cloudcam session / kinect) has frames
    "bad_drive",     # motor speeds out of range, or not numbers
    "bad_scan",      # scan angles/ranges malformed
    "bad_plan",      # drive plan malformed (not a list of steps)
    "bad_pose",      # pose values malformed
    "grid_empty",    # the map has no observations in it
    "pose_unknown",  # localization refused: too little map to match
    "drive_refused",  # the daemon refused a drive (stopped, or no robot)
    "bad_name",      # robot name malformed
    "already_serving",  # a TCP control server is already up for this robot
    "not_serving",   # no TCP control server running for this robot
)


class RobotError(Exception):
    """A robot refusal, carrying its stable code."""

    def __init__(self, kind: str, detail: str = "") -> None:
        if kind not in CODES:
            raise ValueError(f"unknown robot error code {kind!r}")
        self.kind = kind
        self.detail = detail or kind
        super().__init__(f"{kind}: {self.detail}")
