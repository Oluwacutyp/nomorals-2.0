"""The CloudCam ZeroMQ wire — Devon's publisher side.

The wire is storytold/cloudcam's ``zeromq/protocol.rs`` +
``pointcloud/point.rs`` (pattern only, no upstream code): a
``PUSH`` socket carries one frame as a run of little-endian
messages — an optional data-length preamble, then a BEGIN packet
and CONTINUE packets of at most 3000 points each. A point is 16
bytes: three little-endian f32 coordinates (millimetres, camera
space) and four u8 colour bytes (RGBA).

The clone's live sender (``main_zeromq.rs``) never sends the
length preamble — it sends BEGIN then CONTINUE until the frame
is drained — so :func:`encode_frame` does exactly that. The
length message is still encodable (:func:`encode_data_length`)
because the protocol defines it, and the decoder accepts it when
a sender does send it.

Improvements over the clone, all named here rather than hidden:

* the clone's sender drains its point vector in place and has no
  reassembly side at all — the decoder here (:class:`FrameDecoder`)
  reassembles BEGIN/CONTINUE runs into frames and counts malformed
  input instead of dying on it;
* a frame number is NOT on the wire (the clone has none — a
  consumer that added one would stop understanding the Unreal
  plugin). Frame sequencing lives in the session receipts and
  the store payloads, never in the bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

__all__ = [
    "DATA_LENGTH_COMMAND",
    "POINT_DATA_BEGIN_PAYLOAD_COMMAND",
    "POINT_DATA_CONTINUE_PAYLOAD_COMMAND",
    "MAX_SEND_POINTS_PER_PACKET",
    "Point",
    "WireError",
    "decode_frame",
    "encode_data_length",
    "encode_frame",
    "encode_point_data",
]

#: Denotes the command that sends the data length (optional
#: preamble — the clone's live sender does not send it).
DATA_LENGTH_COMMAND: int = 1
#: Denotes the command that sends the first chunk of a frame.
POINT_DATA_BEGIN_PAYLOAD_COMMAND: int = 2
#: Denotes the command that sends a following chunk of a frame.
POINT_DATA_CONTINUE_PAYLOAD_COMMAND: int = 3

#: The maximum number of points per ZeroMQ packet. The clone
#: defines this in Rust and in C++ — it must stay in step on
#: both sides, so it is a named constant here too.
MAX_SEND_POINTS_PER_PACKET: int = 3000

_POINT_STRUCT = struct.Struct("<fffBBBB")
_POINT_SIZE = _POINT_STRUCT.size  # 16


class WireError(ValueError):
    """A wire message that cannot be read — the reason names the
    command, the length, or the field that failed."""


@dataclass(frozen=True)
class Point:
    """One point-cloud point: millimetre coordinates + RGBA colour."""

    x: float
    y: float
    z: float
    r: int = 255
    g: int = 255
    b: int = 255
    a: int = 255

    @classmethod
    def size_bytes(cls) -> int:
        """Wire size of one point: 16 bytes (3xf32, 4xu8)."""
        return _POINT_SIZE

    def to_bytes(self) -> bytes:
        """The point as 16 little-endian bytes."""
        return _POINT_STRUCT.pack(
            float(self.x), float(self.y), float(self.z),
            int(self.r) & 0xFF, int(self.g) & 0xFF,
            int(self.b) & 0xFF, int(self.a) & 0xFF)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Point:
        """Read one point at ``offset``; short buffers are refused
        by name."""
        if len(data) - offset < _POINT_SIZE:
            raise WireError(
                f"point needs {_POINT_SIZE} bytes, got "
                f"{len(data) - offset} at offset {offset}")
        x, y, z, r, g, b, a = _POINT_STRUCT.unpack_from(data, offset)
        return cls(x=x, y=y, z=z, r=r, g=g, b=b, a=a)


def encode_data_length(count: int) -> bytes:
    """The fixed-size data-length preamble: command 1 + point
    count, both little-endian u32."""
    if count < 0:
        raise WireError(
            f"data length cannot be negative, got {count}")
    return struct.pack("<II", DATA_LENGTH_COMMAND, int(count))


def encode_point_data(points: list[Point],
                      is_beginning: bool) -> tuple[bytes, list[Point]]:
    """Encode the first chunk (at most 3000 points) of ``points``.

    Returns ``(message, rest)`` — the message bytes and the
    points still unsent. The input list is not mutated (the clone
    drains its vector in place; callers here keep their frame).
    The message is command (2 BEGIN / 3 CONTINUE) + chunk count
    + 16 bytes per point, all little-endian.
    """
    command = (POINT_DATA_BEGIN_PAYLOAD_COMMAND if is_beginning
               else POINT_DATA_CONTINUE_PAYLOAD_COMMAND)
    chunk = points[:MAX_SEND_POINTS_PER_PACKET]
    rest = points[MAX_SEND_POINTS_PER_PACKET:]
    buf = bytearray()
    buf += struct.pack("<II", command, len(chunk))
    for point in chunk:
        buf += point.to_bytes()
    return bytes(buf), rest


def encode_frame(points: list[Point]) -> list[bytes]:
    """A whole frame as wire messages: one BEGIN packet, then
    CONTINUE packets until the frame is drained. An empty frame
    is one BEGIN packet carrying zero points — the consumer still
    sees the frame boundary. This is exactly what the clone's
    live sender transmits."""
    messages: list[bytes] = []
    rest = list(points)
    first = True
    while True:
        message, rest = encode_point_data(rest, first)
        messages.append(message)
        first = False
        if not rest:
            break
    return messages


def _read_command(data: bytes) -> int:
    if len(data) < 8:
        raise WireError(
            f"wire message needs at least 8 bytes, got {len(data)}")
    command, _count = struct.unpack_from("<II", data, 0)
    return command


def decode_frame(messages: list[bytes]) -> list[Point]:
    """Reassemble one frame from its wire messages.

    Accepts an optional leading data-length message (its count is
    checked against the points that follow), then one BEGIN and
    any number of CONTINUEs. A message with an unknown command,
    a CONTINUE before a BEGIN, a second BEGIN, a truncated point,
    or a count that does not match its bytes is refused by name —
    the frame is not half-returned."""
    points: list[Point] = []
    expected: int | None = None
    begun = False
    for raw in messages:
        command = _read_command(raw)
        if command == DATA_LENGTH_COMMAND:
            if begun or points:
                raise WireError(
                    "data-length preamble arrived after frame data")
            _cmd, count = struct.unpack_from("<II", raw, 0)
            expected = count
            continue
        if command == POINT_DATA_BEGIN_PAYLOAD_COMMAND:
            if begun:
                raise WireError("second BEGIN packet in one frame")
            begun = True
        elif command == POINT_DATA_CONTINUE_PAYLOAD_COMMAND:
            if not begun:
                raise WireError("CONTINUE packet before any BEGIN")
        else:
            raise WireError(
                f"unknown wire command {command} (want 1, 2 or 3)")
        _cmd, count = struct.unpack_from("<II", raw, 0)
        body = len(raw) - 8
        if body != count * _POINT_SIZE:
            raise WireError(
                f"packet claims {count} points "
                f"({count * _POINT_SIZE} bytes) but carries {body}")
        for i in range(count):
            points.append(Point.from_bytes(raw, 8 + i * _POINT_SIZE))
    if not begun:
        raise WireError("no BEGIN packet in frame")
    if expected is not None and expected != len(points):
        raise WireError(
            f"data-length preamble promised {expected} points but "
            f"the frame carries {len(points)}")
    return points


class FrameDecoder:
    """A streaming reassembly of the wire: feed messages as they
    arrive, collect complete frames. Malformed input is counted
    in :attr:`errors` and the broken run is dropped — the decoder
    never raises on a bad packet, and never silently swallows one
    either."""

    def __init__(self) -> None:
        self._pending: list[bytes] = []
        self.errors: list[str] = []
        self.frames_decoded: int = 0
        self.points_decoded: int = 0

    def feed(self, raw: bytes) -> list[list[Point]]:
        """Feed one message; returns any frames completed by it
        (usually zero or one)."""
        try:
            command = _read_command(raw)
        except WireError as exc:
            self.errors.append(str(exc))
            return []
        if command == DATA_LENGTH_COMMAND:
            if self._pending:
                self.errors.append(
                    "data-length preamble arrived after frame data")
                self._pending = []
            self._pending.append(raw)
            return []
        if command == POINT_DATA_BEGIN_PAYLOAD_COMMAND:
            if self._pending:
                self.errors.append(
                    "BEGIN packet arrived with a frame still open — "
                    "the open frame is dropped")
            self._pending = [raw]
            return self._maybe_complete()
        if command == POINT_DATA_CONTINUE_PAYLOAD_COMMAND:
            if not self._pending:
                self.errors.append(
                    "CONTINUE packet with no open frame — dropped")
                return []
            self._pending.append(raw)
            return self._maybe_complete()
        self.errors.append(
            f"unknown wire command {command} (want 1, 2 or 3)")
        return []

    def _maybe_complete(self) -> list[list[Point]]:
        """Try to close the pending run — a frame completes when
        its last packet carries fewer than 3000 points, or when
        the preamble's count is met."""
        try:
            points = decode_frame(self._pending)
        except WireError as exc:
            self.errors.append(str(exc))
            self._pending = []
            return []
        # A full 3000-point packet may be followed by more; only
        # close when the last packet was short, or the preamble
        # count is satisfied.
        last = self._pending[-1]
        _cmd, count = struct.unpack_from("<II", last, 0)
        expected: int | None = None
        if struct.unpack_from("<I", self._pending[0], 0)[0] \
                == DATA_LENGTH_COMMAND:
            expected = struct.unpack_from(
                "<I", self._pending[0], 4)[0]
        if count < MAX_SEND_POINTS_PER_PACKET or (
                expected is not None and len(points) >= expected):
            self._pending = []
            self.frames_decoded += 1
            self.points_decoded += len(points)
            return [points]
        return []

    @property
    def pending(self) -> int:
        """Messages held for a frame that has not closed yet."""
        return len(self._pending)
