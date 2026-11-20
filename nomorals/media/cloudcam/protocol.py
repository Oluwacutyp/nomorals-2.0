"""CloudCam wire protocol — the streamed point-cloud packet format.

Pattern from storytold/cloudcamplugin (PATTERN ONLY, no code
copied): the Rust sender pushes frames over ZeroMQ and the UE
plugin polls them into a ring buffer. The wire is two packed
structs, little-endian, no padding:

* ``CloudCamPoint`` — 16 bytes: ``float x, y, z`` (millimetres,
  camera space, exactly as the Kinect depth unprojection emits)
  then ``uint8 r, g, b, a``.
* ``CloudCamDataLengthPacket`` — 8 bytes: ``int32 command``,
  ``int32 length`` (a pre-announcement; a sender may send it
  before the data, the poller does not need it).
* ``CloudCamPointCloudDataPacket`` — 8-byte header (same two
  ints) followed by ``length`` points.

``command == 2`` starts a new frame; any other command appends
to the frame in progress. That is the clone's rule, kept verbatim
so a real cloudcam sender drives Devon's poller unchanged.

Decoding is strict: a short packet, a length that does not match
the byte count, a negative length, or a packet over the cap is
refused by name — a corrupt stream never becomes a half-frame.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .errors import CloudCamError

__all__ = [
    "COMMAND_APPEND",
    "COMMAND_BEGIN",
    "MAX_POINTS_PER_FRAME",
    "MAX_POINTS_PER_PACKET",
    "CloudCamPoint",
    "decode_data_packet",
    "decode_length_packet",
    "encode_data_packet",
    "encode_length_packet",
]

#: Command 2 opens a new frame; anything else appends to the one
#: in progress (the clone's rule).
COMMAND_BEGIN = 2
COMMAND_APPEND = 1

#: A single packet never carries more than this many points — a
#: full Kinect NFOV frame is 368,640 points, so the cap sits above
#: any real frame the sender can emit in one packet.
MAX_POINTS_PER_PACKET = 500_000

#: A single frame never holds more than this many points.
MAX_POINTS_PER_FRAME = 1_000_000

_POINT_STRUCT = struct.Struct("<fffBBBB")
_HEADER_STRUCT = struct.Struct("<ii")


@dataclass(frozen=True)
class CloudCamPoint:
    """One coloured point, millimetres, camera space."""

    x: float
    y: float
    z: float
    r: int = 255
    g: int = 255
    b: int = 255
    a: int = 255

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise CloudCamError(
                    f"point {name} must be a finite number, got {value!r}",
                    kind="bad_point")
        for name in ("r", "g", "b", "a"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value <= 255:
                raise CloudCamError(
                    f"point {name} must be a byte 0-255, got {value!r}",
                    kind="bad_point")

    def as_tuple(self) -> tuple[float, float, float, int, int, int, int]:
        """(x, y, z, r, g, b, a) — the render/export shape."""
        return (float(self.x), float(self.y), float(self.z),
                self.r, self.g, self.b, self.a)


def encode_data_packet(points: list[CloudCamPoint], *,
                       command: int = COMMAND_BEGIN) -> bytes:
    """One data packet: 8-byte header + the points, packed."""
    if len(points) > MAX_POINTS_PER_PACKET:
        raise CloudCamError(
            f"packet of {len(points)} points exceeds the "
            f"{MAX_POINTS_PER_PACKET} cap",
            kind="packet_too_large")
    body = b"".join(_POINT_STRUCT.pack(
        float(p.x), float(p.y), float(p.z), p.r, p.g, p.b, p.a)
        for p in points)
    return _HEADER_STRUCT.pack(int(command), len(points)) + body


def decode_data_packet(data: bytes) -> tuple[int, list[CloudCamPoint]]:
    """Parse one data packet. Refuses short, mismatched, negative,
    or over-cap packets by name."""
    if len(data) < _HEADER_STRUCT.size:
        raise CloudCamError(
            f"data packet is {len(data)} bytes, shorter than the "
            f"{_HEADER_STRUCT.size}-byte header",
            kind="bad_packet")
    command, length = _HEADER_STRUCT.unpack_from(data, 0)
    if length < 0:
        raise CloudCamError(
            f"data packet declares a negative length {length}",
            kind="bad_packet")
    if length > MAX_POINTS_PER_PACKET:
        raise CloudCamError(
            f"data packet declares {length} points, over the "
            f"{MAX_POINTS_PER_PACKET} cap",
            kind="bad_packet")
    expected = _HEADER_STRUCT.size + length * _POINT_STRUCT.size
    if len(data) != expected:
        raise CloudCamError(
            f"data packet declares {length} points "
            f"({expected} bytes) but carries {len(data)} bytes",
            kind="bad_packet")
    points: list[CloudCamPoint] = []
    offset = _HEADER_STRUCT.size
    for _ in range(length):
        x, y, z, r, g, b, a = _POINT_STRUCT.unpack_from(data, offset)
        points.append(CloudCamPoint(x=x, y=y, z=z, r=r, g=g, b=b, a=a))
        offset += _POINT_STRUCT.size
    return command, points


def encode_length_packet(command: int, length: int) -> bytes:
    """A pre-announcement packet: command + the point count."""
    return _HEADER_STRUCT.pack(int(command), int(length))


def decode_length_packet(data: bytes) -> tuple[int, int]:
    """Parse a pre-announcement packet. Exactly 8 bytes."""
    if len(data) != _HEADER_STRUCT.size:
        raise CloudCamError(
            f"length packet is {len(data)} bytes, must be exactly "
            f"{_HEADER_STRUCT.size}",
            kind="bad_packet")
    command, length = _HEADER_STRUCT.unpack(data)
    if length < 0:
        raise CloudCamError(
            f"length packet declares a negative length {length}",
            kind="bad_packet")
    return command, length
