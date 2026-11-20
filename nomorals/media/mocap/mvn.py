"""Xsens MVN wire — Devon's own MXTP datagram organ.

Pattern only from storytold/xsens-packet-send: a 94-line Rust
program that replays recorded Xsens UDP packets to Unreal so MVN
Animate Pro is not needed. Devon does not copy it — she speaks the
wire herself, both ways, against her own mocap store.

The wire is the Xsens MVN real-time network streaming protocol
(Document MV0305P, Rev E, February 2012 — the public spec) as it
actually streams from MVN Animate 2022, verified byte-for-byte
against the clone's four recorded captures (``three_dancers_
animation.pcap`` etc.):

* One UDP datagram = a 24-byte big-endian header + items. Header:
  6-byte ID ``MXTP`` + 2-char message type, 4-byte sample
  counter, 1-byte datagram counter (7-bit index, MSB set = last
  datagram of the sample), 1-byte item count, 4-byte time code
  (ms), 1-byte avatar ID, 7 reserved bytes.
* Message types: ``01`` pose Euler (28 bytes/item: id + position
  + Euler), ``02`` pose quaternion (32 bytes/item: id + position
  + quaternion), ``03`` point positions (16 bytes/item),
  ``04`` MotionGrid tags (16 bytes/item), ``10`` scale info
  (52 bytes/item), ``11`` prop info (id + null-terminated name),
  ``12`` meta data (tagged text), ``13`` character info — the
  2022 name/offset table (count + per-entry name length, name,
  3-float default offset in metres, Z-up).
* Positions are centimetres, Z-up, right-handed; quaternions are
  (re, i, j, k), always normalised. The 2022 ``02`` datagram
  carries 63 items — the 23 body segments plus finger joints and
  points — in the ``13`` table's order, so an item's ID is its
  position in that table (1-based). The table's 63 names are
  exactly the bundled ``xsens`` remap dialect's 63 keys, so a
  decoded frame drives the rig through the dialect it already
  has — the dialect finally meets a real MVN frame.

Devon's frame is BVH-shaped: Y-up, metres, Euler (Z, X, Y)
degrees. The organ converts both ways with the standard Z-up to
Y-up rotation (``(x, y, z) -> (x, z, -y)``) applied to positions
and, as a similarity transform, to quaternions — documented in
:func:`mvn_to_devon_quat` and tested by round-trip.

The clone's one real behaviour — replay a ``.pcap`` of recorded
packets to a host — lives here as :class:`MvnReplayer`, but on a
real pcap reader: linktype-aware UDP extraction (NULL, Ethernet,
RAW), not the clone's fixed 32-byte strip, which only works for
loopback captures. Live UDP send is impossible in this sandbox
(it forbids UDP send outright) and there is no Xsens hardware on
this box, so the sender and replayer refuse by name when the
socket cannot send, and every proof here runs through fixture
pcaps and injectable sockets — never a faked success.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    hierarchy_joints,
    skeleton_from_dict,
    skeleton_to_dict,
)
from .errors import MocapError
from .state import MocapStateStore

__all__ = [
    "MVN_DEFAULT_PORT",
    "MVN_ID",
    "MVN_MESSAGE_TYPES",
    "MVN_NAMES",
    "MVN_OFFSETS_M",
    "MVN_PARENTS",
    "MVN_SEGMENT_IDS",
    "MvnAssembler",
    "MvnDatagram",
    "MvnEulerItem",
    "MvnHeader",
    "MvnNameTable",
    "MvnPointItem",
    "MvnReceiver",
    "MvnReplayer",
    "MvnSegmentItem",
    "MvnSender",
    "MvnSession",
    "PcapPacket",
    "decode_datagram",
    "decode_name_table",
    "devon_to_mvn_pos",
    "devon_to_mvn_quat",
    "encode_datagram",
    "euler_zxy_to_quat",
    "forward_kinematics",
    "get_session",
    "list_sessions",
    "mvn_skeleton",
    "mvn_to_devon_pos",
    "mvn_to_devon_quat",
    "name_for_id",
    "quat_to_euler_zxy",
    "read_pcap",
    "start_receiver",
    "start_replayer",
    "start_sender",
    "stop_session",
]

#: Default MVN network-streaming port (MV0305P §2.3: 9=X, M=6, E=3).
MVN_DEFAULT_PORT = 9763

#: Datagram ID prefix; the full 6-byte ID is ``MXTP`` + message type.
MVN_ID = b"MXTP"

#: Message types the codec understands (MV0305P §2.4.1 + the 2022
#: ``13`` character-info stream). ``05`` (Unity3D, Rev I) is known
#: but its body is not decoded — refused by name.
MVN_MESSAGE_TYPES = ("01", "02", "03", "04", "10", "11", "12", "13")

#: Fixed item sizes in bytes, by message type (``11``/``12``/``13``
#: are variable-length and have no entry here).
_ITEM_SIZE = {"01": 28, "02": 32, "03": 16, "04": 16, "10": 52}

#: 24-byte header, all big-endian: 6s id, I sample, B datagram
#: counter, B item count, I time code, B avatar, 7s reserved.
_HEADER = struct.Struct(">6sIBBIB7s")
HEADER_SIZE = 24


def _reserved_bytes(body_len: int) -> bytes:
    """The 7 trailing header bytes the way real MVN Studio
    emits them — identical in all four fixtures: ``0x17``,
    ``0x00``, ``0x28``, then the 4-byte big-endian body length
    (2016 for a full ``02`` datagram). The spec calls the field
    reserved for future use and gives it no meaning; 0x17 is
    plausibly the 23-segment count and 0x28 the 40-point count,
    but that is unconfirmed — the bytes are reproduced as
    observed, not as understood. The decoder ignores them."""
    return b"\x17\x00\x28" + struct.pack(">I", body_len)


#: A quaternion whose norm is further than this from 1.0 is refused
#: as corrupt, naming the item — the spec guarantees normalised
#: quaternions and the real captures measure 1.0000.
_QUAT_NORM_TOL = 1e-3

#: The 63 names of the 2022 MVN item table, in wire order (item ID
#: = position + 1). Decoded from the ``13`` datagrams of the
#: clone's ``three_dancers_animation.pcap``; the set is identical
#: to the bundled ``xsens`` remap dialect's 63 keys.
MVN_NAMES: tuple[str, ...] = (
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe",
    "LeftCarpus",
    "LeftFirstMC", "LeftFirstPP", "LeftFirstDP",
    "LeftSecondMC", "LeftSecondPP", "LeftSecondMP", "LeftSecondDP",
    "LeftThirdMC", "LeftThirdPP", "LeftThirdMP", "LeftThirdDP",
    "LeftFourthMC", "LeftFourthPP", "LeftFourthMP", "LeftFourthDP",
    "LeftFifthMC", "LeftFifthPP", "LeftFifthMP", "LeftFifthDP",
    "RightCarpus",
    "RightFirstMC", "RightFirstPP", "RightFirstDP",
    "RightSecondMC", "RightSecondPP", "RightSecondMP",
    "RightSecondDP",
    "RightThirdMC", "RightThirdPP", "RightThirdMP", "RightThirdDP",
    "RightFourthMC", "RightFourthPP", "RightFourthMP",
    "RightFourthDP",
    "RightFifthMC", "RightFifthPP", "RightFifthMP", "RightFifthDP",
)

#: Default offsets (metres, Z-up, absolute in the capture volume)
#: for the 63 names, in wire order — decoded from the same ``13``
#: datagrams as :data:`MVN_NAMES`. A live ``13`` table replaces
#: these for its avatar; they are the fallback for a stream that
#: never sent one (and the skeleton builder's source of truth when
#: no table has arrived yet).
MVN_OFFSETS_M: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.913), (-0.0108, 0.0, 1.0105),
    (-0.0108, 0.0, 1.1183), (-0.0109, 0.0, 1.2168),
    (-0.0109, 0.0, 1.3152), (-0.0109, 0.0, 1.453),
    (-0.0108, 0.0, 1.5446), (-0.0109, -0.0301, 1.3924),
    (-0.0109, -0.1712, 1.3924), (-0.0109, -0.4736, 1.3924),
    (-0.0109, -0.7208, 1.3924), (-0.0109, 0.0301, 1.3924),
    (-0.0109, 0.1712, 1.3924), (-0.0109, 0.4736, 1.3924),
    (-0.0109, 0.7208, 1.3924), (-0.0, -0.0803, 0.9131),
    (-0.0, -0.0803, 0.497), (-0.0, -0.0803, 0.0911),
    (0.1683, -0.0803, 0.0171), (-0.0, 0.0803, 0.9131),
    (-0.0, 0.0803, 0.497), (-0.0, 0.0803, 0.0911),
    (0.1683, 0.0803, 0.0171), (-0.0109, 0.7208, 1.3924),
    (0.0163, 0.7484, 1.3924), (0.0163, 0.7841, 1.3924),
    (0.0163, 0.8155, 1.3924), (0.0013, 0.7554, 1.3924),
    (0.0013, 0.8131, 1.3924), (0.0013, 0.8548, 1.3924),
    (0.0013, 0.8776, 1.3924), (-0.0109, 0.7573, 1.3924),
    (-0.0109, 0.8133, 1.3924), (-0.0109, 0.8589, 1.3924),
    (-0.0109, 0.8859, 1.3924), (-0.0232, 0.7554, 1.3924),
    (-0.0232, 0.8097, 1.3924), (-0.0232, 0.8531, 1.3924),
    (-0.0232, 0.8788, 1.3924), (-0.0329, 0.7508, 1.3924),
    (-0.0329, 0.8025, 1.3924), (-0.0329, 0.8387, 1.3924),
    (-0.0329, 0.8583, 1.3924), (-0.0109, -0.7208, 1.3924),
    (0.0163, -0.7484, 1.3924), (0.0163, -0.7841, 1.3924),
    (0.0163, -0.8155, 1.3924), (0.0013, -0.7554, 1.3924),
    (0.0013, -0.8131, 1.3924), (0.0013, -0.8548, 1.3924),
    (0.0013, -0.8776, 1.3924), (-0.0109, -0.7573, 1.3924),
    (-0.0109, -0.8133, 1.3924), (-0.0109, -0.8589, 1.3924),
    (-0.0109, -0.8859, 1.3924), (-0.0232, -0.7554, 1.3924),
    (-0.0232, -0.8097, 1.3924), (-0.0232, -0.8531, 1.3924),
    (-0.0232, -0.8788, 1.3924), (-0.0329, -0.7508, 1.3924),
    (-0.0329, -0.8025, 1.3924), (-0.0329, -0.8387, 1.3924),
    (-0.0329, -0.8583, 1.3924),
)

#: Segment IDs for the 23 body segments + props, MV0305P §3.1.
#: Note the gap: there is no ID 22 — Left Foot is 23, Left Toe
#: 24. The 2022 wire table has no gap (Left Foot is item 22), so
#: :func:`name_for_id` only uses this table for old 23-segment
#: streams (every item ID <= 28 and at most 27 items).
MVN_SEGMENT_IDS: dict[int, str] = {
    1: "Pelvis", 2: "L5", 3: "L3", 4: "T12", 5: "T8", 6: "Neck",
    7: "Head", 8: "RightShoulder", 9: "RightUpperArm",
    10: "RightForeArm", 11: "RightHand", 12: "LeftShoulder",
    13: "LeftUpperArm", 14: "LeftForeArm", 15: "LeftHand",
    16: "RightUpperLeg", 17: "RightLowerLeg", 18: "RightFoot",
    19: "RightToe", 20: "LeftUpperLeg", 21: "LeftLowerLeg",
    23: "LeftFoot", 24: "LeftToe", 25: "Prop1", 26: "Prop2",
    27: "Prop3", 28: "Prop4",
}

#: Parent for every one of the 63 names — the MVN hierarchy.
#: Pelvis is the root; finger metacarpals hang off the hand
#: (the carpus and the hand share one offset in the ``13``
#: table, so the hand is the parent either way); the thumb has
#: no middle phalanx (FirstMC/PP/DP only, per the table).
MVN_PARENTS: dict[str, str | None] = {
    "Pelvis": None,
    "L5": "Pelvis", "L3": "L5", "T12": "L3", "T8": "T12",
    "Neck": "T8", "Head": "Neck",
    "RightShoulder": "T8", "RightUpperArm": "RightShoulder",
    "RightForeArm": "RightUpperArm", "RightHand": "RightForeArm",
    "RightCarpus": "RightForeArm",
    "LeftShoulder": "T8", "LeftUpperArm": "LeftShoulder",
    "LeftForeArm": "LeftUpperArm", "LeftHand": "LeftForeArm",
    "LeftCarpus": "LeftForeArm",
    "RightUpperLeg": "Pelvis", "RightLowerLeg": "RightUpperLeg",
    "RightFoot": "RightLowerLeg", "RightToe": "RightFoot",
    "LeftUpperLeg": "Pelvis", "LeftLowerLeg": "LeftUpperLeg",
    "LeftFoot": "LeftLowerLeg", "LeftToe": "LeftFoot",
}
for _hand in ("Left", "Right"):
    for _finger in ("First", "Second", "Third", "Fourth", "Fifth"):
        _parts = ("MC", "PP", "MP", "DP") if _finger != "First" \
            else ("MC", "PP", "DP")
        _prev = f"{_hand}Hand"
        for _part in _parts:
            _name = f"{_hand}{_finger}{_part}"
            MVN_PARENTS[_name] = _prev
            _prev = _name
del _hand, _finger, _parts, _prev, _name, _part


def _mvn_error(message: str, *, kind: str) -> MocapError:
    return MocapError(message, kind=kind)


# -- datagram codec -------------------------------------------------

@dataclass(frozen=True)
class MvnHeader:
    """A decoded 24-byte MVN datagram header."""

    msg_type: str
    sample_counter: int
    datagram_counter: int
    last: bool
    item_count: int
    time_code_ms: int
    avatar_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "msg_type": self.msg_type,
            "sample_counter": self.sample_counter,
            "datagram_counter": self.datagram_counter,
            "last": self.last,
            "item_count": self.item_count,
            "time_code_ms": self.time_code_ms,
            "avatar_id": self.avatar_id,
        }


@dataclass(frozen=True)
class MvnSegmentItem:
    """One ``02`` pose item: segment ID, position (cm, Z-up),
    quaternion (re, i, j, k)."""

    segment_id: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "position": list(self.position),
            "quaternion": list(self.quaternion),
        }


@dataclass(frozen=True)
class MvnEulerItem:
    """One ``01`` pose item: segment ID, position (cm, Z-up),
    Euler rotation (x, y, z) in degrees."""

    segment_id: int
    position: tuple[float, float, float]
    euler: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "position": list(self.position),
            "euler": list(self.euler),
        }


@dataclass(frozen=True)
class MvnPointItem:
    """One ``03``/``04`` point item: point/tag ID, position
    (cm, Z-up)."""

    point_id: int
    position: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "position": list(self.position),
        }


@dataclass(frozen=True)
class MvnDatagram:
    """A decoded MVN datagram: header + typed items. ``11``/``12``/
    ``13`` datagrams carry their body unparsed in ``body`` (their
    items are variable-length); ``02``/``01``/``03``/``04``/``10``
    carry parsed ``items``."""

    header: MvnHeader
    items: tuple[Any, ...] = ()
    body: bytes = b""

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "body_len": len(self.body),
        }


def decode_header(data: bytes) -> MvnHeader:
    """Decode a 24-byte header. Short data and a bad ID are
    refused by name."""
    if len(data) < HEADER_SIZE:
        raise _mvn_error(
            f"MVN datagram is {len(data)} bytes, shorter than the "
            f"{HEADER_SIZE}-byte header", kind="short_datagram")
    raw_id, sample, counter, count, time_code, avatar, _res = \
        _HEADER.unpack(data[:HEADER_SIZE])
    if not raw_id.startswith(MVN_ID):
        raise _mvn_error(
            f"MVN datagram ID is {raw_id[:4]!r}, not b'MXTP'",
            kind="bad_magic")
    try:
        msg_type = raw_id[4:6].decode("ascii")
    except UnicodeDecodeError as exc:
        raise _mvn_error(
            f"MVN message type bytes {raw_id[4:6]!r} are not ASCII",
            kind="bad_msg_type") from exc
    return MvnHeader(
        msg_type=msg_type,
        sample_counter=sample,
        datagram_counter=counter & 0x7F,
        last=bool(counter & 0x80),
        item_count=count,
        time_code_ms=time_code,
        avatar_id=avatar,
    )


def encode_header(header: MvnHeader, *,
                  body_len: int = 0) -> bytes:
    """Encode a header. The message type must be two ASCII
    characters; the counters must fit their fields. ``body_len``
    fills the trailing reserved bytes the way real MVN Studio
    does (see :func:`_reserved_bytes`)."""
    if len(header.msg_type) != 2:
        raise _mvn_error(
            f"MVN message type {header.msg_type!r} must be 2 "
            f"characters", kind="bad_msg_type")
    try:
        raw_type = header.msg_type.encode("ascii")
    except UnicodeEncodeError as exc:
        raise _mvn_error(
            f"MVN message type {header.msg_type!r} is not ASCII",
            kind="bad_msg_type") from exc
    if not 0 <= header.sample_counter <= 0xFFFFFFFF:
        raise _mvn_error(
            f"sample counter {header.sample_counter} does not fit "
            f"in 32 bits", kind="bad_counter")
    if not 0 <= header.datagram_counter <= 0x7F:
        raise _mvn_error(
            f"datagram counter {header.datagram_counter} does not "
            f"fit in 7 bits", kind="bad_counter")
    if not 0 <= header.item_count <= 0xFF:
        raise _mvn_error(
            f"item count {header.item_count} does not fit in a "
            f"byte", kind="bad_item_count")
    counter = header.datagram_counter | (0x80 if header.last else 0)
    return _HEADER.pack(
        MVN_ID + raw_type, header.sample_counter, counter,
        header.item_count, header.time_code_ms, header.avatar_id,
        _reserved_bytes(body_len))


def _decode_fixed_items(msg_type: str, body: bytes,
                        count: int) -> tuple[Any, ...]:
    """Decode a fixed-size item body. A body whose length does not
    match the header's item count is refused with both numbers."""
    size = _ITEM_SIZE[msg_type]
    expected = count * size
    if len(body) != expected:
        raise _mvn_error(
            f"MVN type {msg_type} datagram claims {count} items "
            f"({expected} bytes) but carries {len(body)} bytes",
            kind="length_mismatch")
    items: list[Any] = []
    for index in range(count):
        chunk = body[index * size:(index + 1) * size]
        if msg_type == "01":
            seg, x, y, z, ex, ey, ez = struct.unpack(">I6f", chunk)
            items.append(MvnEulerItem(seg, (x, y, z), (ex, ey, ez)))
        elif msg_type == "02":
            seg, x, y, z, q1, q2, q3, q4 = struct.unpack(
                ">I7f", chunk)
            norm = math.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
            if abs(norm - 1.0) > _QUAT_NORM_TOL:
                raise _mvn_error(
                    f"MVN type 02 item {seg} quaternion "
                    f"({q1:.4f}, {q2:.4f}, {q3:.4f}, {q4:.4f}) has "
                    f"norm {norm:.4f}, not 1.0",
                    kind="bad_quaternion")
            items.append(MvnSegmentItem(
                seg, (x, y, z), (q1, q2, q3, q4)))
        elif msg_type in ("03", "04"):
            point, x, y, z = struct.unpack(">I3f", chunk)
            items.append(MvnPointItem(point, (x, y, z)))
        elif msg_type == "10":
            seg = struct.unpack(">I", chunk[:4])[0]
            values = struct.unpack(">12f", chunk[4:])
            items.append({"segment_id": seg,
                          "offset": list(values[:3]),
                          "matrix": list(values[3:])})
    return tuple(items)


def _decode_prop_items(body: bytes, count: int) -> tuple[Any, ...]:
    """Decode ``11`` prop items: 4-byte segment ID + a
    null-terminated name, repeated. A missing terminator or a
    short body is refused naming the item."""
    items: list[dict[str, Any]] = []
    off = 0
    for index in range(count):
        if off + 4 > len(body):
            raise _mvn_error(
                f"MVN type 11 item {index} has no segment ID "
                f"({len(body) - off} bytes left)",
                kind="short_datagram")
        seg = struct.unpack(">I", body[off:off + 4])[0]
        off += 4
        end = body.find(b"\x00", off)
        if end < 0:
            raise _mvn_error(
                f"MVN type 11 item {index} (segment {seg}) has no "
                f"null-terminated name", kind="short_datagram")
        try:
            name = body[off:end].decode("ascii")
        except UnicodeDecodeError as exc:
            raise _mvn_error(
                f"MVN type 11 item {index} (segment {seg}) name "
                f"is not ASCII", kind="bad_prop_name") from exc
        items.append({"segment_id": seg, "name": name})
        off = end + 1
    if off != len(body):
        raise _mvn_error(
            f"MVN type 11 body has {len(body) - off} trailing "
            f"bytes after {count} items", kind="length_mismatch")
    return tuple(items)


def decode_datagram(data: bytes) -> MvnDatagram:
    """Decode one MVN datagram, strictly. Unknown message types
    and the undecoded ``05`` Unity3D body are refused by name;
    ``11``/``12``/``13`` bodies come back raw for their own
    decoders."""
    header = decode_header(data)
    body = data[HEADER_SIZE:]
    if header.msg_type == "05":
        raise _mvn_error(
            "MVN type 05 (Unity3D) datagrams are not decoded — "
            "the Rev I body layout is not in this organ",
            kind="unsupported_type")
    if header.msg_type not in MVN_MESSAGE_TYPES:
        raise _mvn_error(
            f"MVN message type {header.msg_type!r} is unknown "
            f"(known: {', '.join(MVN_MESSAGE_TYPES)})",
            kind="unknown_type")
    if header.msg_type in _ITEM_SIZE:
        return MvnDatagram(
            header=header,
            items=_decode_fixed_items(header.msg_type, body,
                                      header.item_count))
    if header.msg_type == "11":
        return MvnDatagram(
            header=header,
            items=_decode_prop_items(body, header.item_count))
    if header.msg_type in ("12", "13"):
        return MvnDatagram(header=header, body=body)
    raise _mvn_error(
        f"MVN message type {header.msg_type!r} has no decoder",
        kind="unknown_type")


def _encode_fixed_items(msg_type: str,
                        items: tuple[Any, ...]) -> bytes:
    """Encode fixed-size items. A quaternion that is not
    normalised is refused naming the item."""
    out = bytearray()
    for item in items:
        if msg_type == "01":
            out += struct.pack(
                ">I6f", item.segment_id, *item.position,
                *item.euler)
        elif msg_type == "02":
            q = item.quaternion
            norm = math.sqrt(sum(v * v for v in q))
            if abs(norm - 1.0) > _QUAT_NORM_TOL:
                raise _mvn_error(
                    f"MVN type 02 item {item.segment_id} "
                    f"quaternion has norm {norm:.4f}, not 1.0",
                    kind="bad_quaternion")
            out += struct.pack(
                ">I7f", item.segment_id, *item.position, *q)
        elif msg_type in ("03", "04"):
            out += struct.pack(
                ">I3f", item.point_id, *item.position)
        elif msg_type == "10":
            out += struct.pack(
                ">I12f", item["segment_id"],
                *item["offset"], *item["matrix"])
    return bytes(out)


def encode_datagram(header: MvnHeader,
                    items: tuple[Any, ...] = (),
                    body: bytes = b"") -> bytes:
    """Encode one MVN datagram. Fixed-size types take ``items``
    (the header's item count must match); ``11``/``12``/``13``
    take a raw ``body``."""
    if header.msg_type in _ITEM_SIZE:
        if len(items) != header.item_count:
            raise _mvn_error(
                f"MVN type {header.msg_type} header claims "
                f"{header.item_count} items but {len(items)} "
                f"were given", kind="length_mismatch")
        if body:
            raise _mvn_error(
                f"MVN type {header.msg_type} takes items, not a "
                f"raw body", kind="bad_body")
        encoded = _encode_fixed_items(header.msg_type, items)
        return encode_header(header, body_len=len(encoded)) + encoded
    if header.msg_type == "11":
        if body:
            raise _mvn_error(
                "MVN type 11 takes items, not a raw body",
                kind="bad_body")
        out = bytearray()
        if len(items) != header.item_count:
            raise _mvn_error(
                f"MVN type 11 header claims "
                f"{header.item_count} items but {len(items)} "
                f"were given", kind="length_mismatch")
        for item in items:
            out += struct.pack(">I", item["segment_id"])
            out += item["name"].encode("ascii") + b"\x00"
        return encode_header(header, body_len=len(out)) + bytes(out)
    if header.msg_type in ("12", "13"):
        if items:
            raise _mvn_error(
                f"MVN type {header.msg_type} takes a raw body, "
                f"not items", kind="bad_body")
        return encode_header(header, body_len=len(body)) + body
    raise _mvn_error(
        f"MVN message type {header.msg_type!r} cannot be encoded",
        kind="unknown_type")


# -- character info ("13") ----------------------------------------

@dataclass(frozen=True)
class MvnNameTable:
    """A decoded ``13`` name table: item names in wire order plus
    their default offsets (metres, Z-up). ``consumed`` is how many
    body bytes the table took; ``extra`` is everything after it
    (further tables the organ does not decode); ``extra_names``
    is a best-effort list of printable names found in ``extra``."""

    names: tuple[str, ...]
    offsets_m: tuple[tuple[float, float, float], ...]
    consumed: int
    extra: bytes = b""
    extra_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "offsets_m": [list(o) for o in self.offsets_m],
            "consumed": self.consumed,
            "extra_bytes": len(self.extra),
            "extra_names": list(self.extra_names),
        }


def _scan_extra_names(extra: bytes) -> tuple[str, ...]:
    """Best-effort printable names in a ``13`` body's remainder:
    a 4-byte length (1..64) followed by that many printable ASCII
    bytes. Anything else is skipped, never forced into a name."""
    found: list[str] = []
    off = 0
    while off + 4 <= len(extra):
        size = struct.unpack(">I", extra[off:off + 4])[0]
        if 1 <= size <= 64 and off + 4 + size <= len(extra):
            raw = extra[off + 4:off + 4 + size]
            if all(32 <= b < 127 for b in raw):
                found.append(raw.decode("ascii"))
                off += 4 + size
                continue
        off += 1
    return tuple(found)


def decode_name_table(body: bytes) -> MvnNameTable:
    """Decode a ``13`` body: 4-byte entry count, then per entry a
    4-byte name length, the name, and 3 big-endian floats (the
    default offset, metres, Z-up). A malformed entry is refused
    naming the byte offset. Bytes after the table come back as
    ``extra`` with a best-effort name scan."""
    if len(body) < 4:
        raise _mvn_error(
            f"MVN type 13 body is {len(body)} bytes — no entry "
            f"count", kind="short_datagram")
    count = struct.unpack(">I", body[:4])[0]
    if count == 0 or count > 256:
        raise _mvn_error(
            f"MVN type 13 entry count {count} is not plausible "
            f"(1..256)", kind="bad_item_count")
    names: list[str] = []
    offsets: list[tuple[float, float, float]] = []
    off = 4
    for index in range(count):
        if off + 4 > len(body):
            raise _mvn_error(
                f"MVN type 13 entry {index} has no name length "
                f"at offset {off}", kind="short_datagram")
        size = struct.unpack(">I", body[off:off + 4])[0]
        off += 4
        if size == 0 or size > 64 or off + size + 12 > len(body):
            raise _mvn_error(
                f"MVN type 13 entry {index} name length {size} "
                f"at offset {off - 4} does not fit the body",
                kind="short_datagram")
        try:
            name = body[off:off + size].decode("ascii")
        except UnicodeDecodeError as exc:
            raise _mvn_error(
                f"MVN type 13 entry {index} name at offset "
                f"{off} is not ASCII", kind="bad_name") from exc
        off += size
        x, y, z = struct.unpack(">3f", body[off:off + 12])
        off += 12
        names.append(name)
        offsets.append((x, y, z))
    extra = body[off:]
    return MvnNameTable(
        names=tuple(names), offsets_m=tuple(offsets),
        consumed=off, extra=extra,
        extra_names=_scan_extra_names(extra))


def name_for_id(segment_id: int, *,
                table: MvnNameTable | None = None,
                legacy: bool = False) -> str:
    """The name for a wire item ID. A live ``13`` table is
    authoritative (ID = position + 1). Without one, the 2022
    wire order (63 segments) is the default — the format the
    fixtures and current MVN Studio use. ``legacy=True`` names
    IDs by the spec's 23-segment table instead (LeftFoot 23,
    LeftToe 24, no ID 22); use it only for old 23-segment
    captures. An ID no table covers is refused by name."""
    if table is not None:
        if 1 <= segment_id <= len(table.names):
            return table.names[segment_id - 1]
        raise _mvn_error(
            f"MVN item ID {segment_id} is outside the live name "
            f"table ({len(table.names)} entries)",
            kind="unknown_segment")
    if legacy:
        if segment_id in MVN_SEGMENT_IDS:
            return MVN_SEGMENT_IDS[segment_id]
        raise _mvn_error(
            f"MVN item ID {segment_id} is not in the spec's "
            f"23-segment table", kind="unknown_segment")
    if 1 <= segment_id <= len(MVN_NAMES):
        return MVN_NAMES[segment_id - 1]
    raise _mvn_error(
        f"MVN item ID {segment_id} is outside every known table",
        kind="unknown_segment")


# -- rotation math (stdlib only) ----------------------------------

#: Quaternion for RotX(-90°): the Z-up to Y-up axis change
#: ``(x, y, z) -> (x, z, -y)``. MVN is Z-up (X forward, Y left);
#: Devon/BVH is Y-up (X forward, Y up, Z right).
_Q_AXIS = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)
_Q_AXIS_INV = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)


def _quat_mul(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]
              ) -> tuple[float, float, float, float]:
    """Hamilton product: applies ``b`` first, then ``a``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_conj(q: tuple[float, float, float, float]
               ) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def _axis_angle(axis: tuple[float, float, float],
                angle_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(angle_deg) / 2.0
    s = math.sin(half)
    return (math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)


def euler_zxy_to_quat(z_deg: float, x_deg: float,
                      y_deg: float) -> tuple[float, float, float, float]:
    """BVH Euler (Z, X, Y) degrees -> quaternion (w, x, y, z).
    The rotation matrix is ``Rz(z) * Rx(x) * Ry(y)`` — the BVH
    channel order, applied Y first, then X, then Z."""
    qz = _axis_angle((0.0, 0.0, 1.0), z_deg)
    qx = _axis_angle((1.0, 0.0, 0.0), x_deg)
    qy = _axis_angle((0.0, 1.0, 0.0), y_deg)
    return _quat_mul(qz, _quat_mul(qx, qy))


def _quat_to_matrix(q: tuple[float, float, float, float]
                    ) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)),
    )


def quat_to_euler_zxy(q: tuple[float, float, float, float]
                      ) -> tuple[float, float, float]:
    """Quaternion (w, x, y, z) -> BVH Euler (Z, X, Y) degrees, the
    inverse of :func:`euler_zxy_to_quat` (gimbal-safe: at
    X = ±90° the Z angle is set to 0 and Y takes the sum)."""
    r = _quat_to_matrix(q)
    sx = max(-1.0, min(1.0, r[2][1]))
    x = math.asin(sx)
    if abs(math.cos(x)) > 1e-6:
        z = math.atan2(-r[0][1], r[1][1])
        y = math.atan2(-r[2][0], r[2][2])
    else:
        z = 0.0
        y = math.atan2(r[0][2], r[0][0])
    return (math.degrees(z), math.degrees(x), math.degrees(y))


def mvn_to_devon_quat(q: tuple[float, float, float, float]
                      ) -> tuple[float, float, float, float]:
    """An MVN quaternion (Z-up frame) into Devon's Y-up frame: a
    similarity transform with the axis-change rotation."""
    return _quat_mul(_Q_AXIS, _quat_mul(q, _Q_AXIS_INV))


def devon_to_mvn_quat(q: tuple[float, float, float, float]
                      ) -> tuple[float, float, float, float]:
    """Devon's Y-up quaternion back into MVN's Z-up frame."""
    return _quat_mul(_Q_AXIS_INV, _quat_mul(q, _Q_AXIS))


def mvn_to_devon_pos(pos_cm: tuple[float, float, float]
                     ) -> tuple[float, float, float]:
    """An MVN position (cm, Z-up) into Devon metres, Y-up."""
    x, y, z = pos_cm
    return (x / 100.0, z / 100.0, -y / 100.0)


def devon_to_mvn_pos(pos_m: tuple[float, float, float]
                     ) -> tuple[float, float, float]:
    """A Devon position (metres, Y-up) into MVN centimetres,
    Z-up."""
    x, y, z = pos_m
    return (x * 100.0, -z * 100.0, y * 100.0)


def _mat_mul(a: tuple[tuple[float, float, float], ...],
             b: tuple[tuple[float, float, float], ...]
             ) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3))
              for j in range(3))
        for i in range(3))


def _mat_vec(m: tuple[tuple[float, float, float], ...],
             v: tuple[float, float, float]
             ) -> tuple[float, float, float]:
    return tuple(sum(m[i][k] * v[k] for k in range(3))
                 for i in range(3))


def _euler_zxy_to_matrix(z_deg: float, x_deg: float,
                         y_deg: float
                         ) -> tuple[tuple[float, float, float], ...]:
    """Rotation matrix for BVH Euler (Z, X, Y) degrees."""
    z, x, y = (math.radians(a) for a in (z_deg, x_deg, y_deg))
    cz, sz, cx, sx, cy, sy = (math.cos(z), math.sin(z),
                              math.cos(x), math.sin(x),
                              math.cos(y), math.sin(y))
    rz = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    rx = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    ry = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    return _mat_mul(rz, _mat_mul(rx, ry))


def forward_kinematics(
        skeleton: Skeleton, frame: SkeletonFrame
        ) -> dict[str, dict[str, Any]]:
    """World positions (metres, Y-up) and quaternions per joint,
    from a BVH take's skeleton + one frame. The root's position
    comes from its 6 values; every other joint's position is its
    parent's position plus the parent's world rotation applied
    to the joint's offset."""
    ordered = hierarchy_joints(skeleton)
    parents: dict[str, str | None] = {}
    for joint in ordered:
        for child in joint.children:
            parents[child.name] = joint.name
    world: dict[str, dict[str, Any]] = {}
    for joint in ordered:
        values = frame.values.get(joint.name)
        if values is None:
            raise _mvn_error(
                f"frame has no values for joint {joint.name!r}",
                kind="bad_frame")
        if joint is skeleton.root:
            if len(values) != 6:
                raise _mvn_error(
                    f"root joint {joint.name!r} needs 6 values, "
                    f"got {len(values)}", kind="bad_frame")
            pos = (values[0], values[1], values[2])
            rot = euler_zxy_to_quat(values[3], values[4],
                                    values[5])
        else:
            if len(values) != 3:
                raise _mvn_error(
                    f"joint {joint.name!r} needs 3 values, got "
                    f"{len(values)}", kind="bad_frame")
            rot = euler_zxy_to_quat(values[0], values[1],
                                    values[2])
            parent = parents[joint.name]
            parent_world = world[parent]
            parent_rot = _quat_to_matrix(parent_world["quaternion"])
            pos = tuple(
                parent_world["position"][i]
                + _mat_vec(parent_rot, joint.offset)[i]
                for i in range(3))
            rot = _quat_mul(
                parent_world["quaternion"], rot)
        world[joint.name] = {"position": pos, "quaternion": rot}
    return world


# -- frame conversion ---------------------------------------------

def _legacy_names_offsets() -> tuple[list[str],
                                     list[tuple[float, float, float]]]:
    """The spec's 23-segment hierarchy as names + offsets: the
    wire table's first 21 entries, then LeftFoot/LeftToe at
    the spec's IDs 23/24 (wire IDs 22/23). Only for old
    23-segment captures."""
    return (list(MVN_NAMES[:21]) + ["LeftFoot", "LeftToe"],
            list(MVN_OFFSETS_M[:21])
            + [MVN_OFFSETS_M[21], MVN_OFFSETS_M[22]])


def mvn_skeleton(table: MvnNameTable | None = None, *,
                 legacy: bool = False,
                 frame_time: float = 1.0 / 120.0) -> Skeleton:
    """A BVH skeleton for the MVN hierarchy, from a live ``13``
    table or the bundled wire table. Offsets are parent-relative,
    converted from the table's absolute Z-up metres into Devon
    Y-up metres; the root's offset is (0, 0, 0) — its position
    rides in the frame's 6 root values. A name with no parent in
    :data:`MVN_PARENTS` is refused by name."""
    if table is None:
        if legacy:
            names, offsets = _legacy_names_offsets()
        else:
            names = MVN_NAMES
            offsets = MVN_OFFSETS_M
    else:
        names = table.names
        offsets = table.offsets_m
    absolute: dict[str, tuple[float, float, float]] = {}
    for name, (x, y, z) in zip(names, offsets, strict=True):
        absolute[name] = (x, z, -y)
    joints: dict[str, Joint] = {}
    for name in names:
        parent = MVN_PARENTS.get(name, "missing")
        if parent == "missing":
            raise _mvn_error(
                f"MVN name {name!r} has no parent in the MVN "
                f"hierarchy — cannot build its skeleton",
                kind="unknown_segment")
        if parent is None:
            offset = (0.0, 0.0, 0.0)
            channels = ("Xposition", "Yposition", "Zposition",
                        "Zrotation", "Xrotation", "Yrotation")
        else:
            mine = absolute[name]
            theirs = absolute[parent]
            offset = tuple(mine[i] - theirs[i] for i in range(3))
            channels = ("Zrotation", "Xrotation", "Yrotation")
        joints[name] = Joint(name=name, offset=offset,
                             channels=channels)
    for name, joint in joints.items():
        parent = MVN_PARENTS[name]
        if parent is not None:
            joints[parent].children.append(joint)
    root_name = next(name for name in names
                     if MVN_PARENTS[name] is None)
    return Skeleton(root=joints[root_name], frame_time=frame_time)


def mvn_pose_to_frame(
        items: tuple[Any, ...], *,
        table: MvnNameTable | None = None,
        legacy: bool = False) -> SkeletonFrame:
    """MVN pose items (``01``/``02``) into one BVH-shaped frame:
    Euler (Z, X, Y) degrees per segment, converted from the MVN
    Z-up frame; the Pelvis item also carries the root position.
    An item whose name no table covers is reported in the
    ``unmapped`` return — never silently dropped."""
    values: dict[str, tuple[float, ...]] = {}
    unmapped: list[str] = []
    for item in items:
        if isinstance(item, MvnSegmentItem):
            segment_id = item.segment_id
            quat = mvn_to_devon_quat(item.quaternion)
            euler = quat_to_euler_zxy(quat)
            pos = mvn_to_devon_pos(item.position)
        elif isinstance(item, MvnEulerItem):
            segment_id = item.segment_id
            quat = euler_zxy_to_quat(
                item.euler[2], item.euler[0], item.euler[1])
            quat = mvn_to_devon_quat(quat)
            euler = quat_to_euler_zxy(quat)
            pos = mvn_to_devon_pos(item.position)
        else:
            continue
        try:
            name = name_for_id(segment_id, table=table,
                               legacy=legacy)
        except MocapError:
            unmapped.append(f"id:{segment_id}")
            continue
        if name == "Pelvis":
            values[name] = (pos[0], pos[1], pos[2], *euler)
        else:
            values[name] = euler
    frame = SkeletonFrame(values=values)
    frame.unmapped = unmapped  # type: ignore[attr-defined]
    return frame


def frame_to_mvn_items(
        skeleton: Skeleton, frame: SkeletonFrame, *,
        name_map: dict[str, str] | None = None,
        table: MvnNameTable | None = None,
        legacy: bool = False
        ) -> tuple[tuple[MvnSegmentItem, ...], list[str]]:
    """A BVH take frame into ``02`` pose items: forward
    kinematics for world positions, Euler to quaternion, Devon
    Y-up to MVN Z-up. ``name_map`` maps take joint names to
    Xsens names (the tool layer builds it from the remap
    dialects); without one, only joints already named like the
    wire table pass through. ``legacy`` sends the spec's
    23-segment IDs instead of the 2022 wire IDs. Unmapped
    joints come back named, never silently dropped."""
    world = forward_kinematics(skeleton, frame)
    if table is not None:
        names = table.names
    elif legacy:
        names, _offsets = _legacy_names_offsets()
    else:
        names = MVN_NAMES
    index = {name: i + 1 for i, name in enumerate(names)}
    items: list[MvnSegmentItem] = []
    unmapped: list[str] = []
    for joint in hierarchy_joints(skeleton):
        xsens = None
        if joint.name in index:
            xsens = joint.name
        elif name_map is not None:
            xsens = name_map.get(joint.name)
        if xsens is None or xsens not in index:
            unmapped.append(joint.name)
            continue
        entry = world[joint.name]
        items.append(MvnSegmentItem(
            segment_id=index[xsens],
            position=devon_to_mvn_pos(entry["position"]),
            quaternion=devon_to_mvn_quat(entry["quaternion"])))
    return tuple(items), unmapped


# -- pcap reading -------------------------------------------------

#: Legacy pcap magic numbers (microsecond / nanosecond, either
#: byte order). The file's first 4 bytes decide the reader.
_PCAP_MAGICS = {
    0xA1B2C3D4: ("<", False),
    0xD4C3B2A1: (">", False),
    0xA1B23C4D: ("<", True),
    0x4D3CB2A1: (">", True),
}

#: Linktypes the reader strips: NULL (BSD loopback, 4 bytes),
#: Ethernet (14), RAW IP (no link header), Linux cooked (16).
_LINK_HEADER = {0: 4, 1: 14, 12: 0, 101: 0, 113: 16}


@dataclass(frozen=True)
class PcapPacket:
    """One UDP payload out of a pcap, with its capture time and
    endpoints."""

    timestamp_s: float
    src: str
    dst: str
    src_port: int
    dst_port: int
    payload: bytes


def _ipv4_udp(payload: bytes, *, where: str) -> PcapPacket | None:
    """Parse one link payload as IPv4 + UDP. Returns None for a
    non-IPv4/non-UDP packet (counted by the caller, never an
    error); malformed IP/UDP headers are refused by name."""
    if len(payload) < 20:
        raise _mvn_error(
            f"pcap {where}: IP packet is {len(payload)} bytes",
            kind="short_datagram")
    version = payload[0] >> 4
    if version != 4:
        return None
    ihl = (payload[0] & 0x0F) * 4
    if ihl < 20 or len(payload) < ihl:
        raise _mvn_error(
            f"pcap {where}: IPv4 header length {ihl} does not fit "
            f"a {len(payload)}-byte packet", kind="short_datagram")
    total = struct.unpack(">H", payload[2:4])[0]
    if total > len(payload):
        raise _mvn_error(
            f"pcap {where}: IPv4 total length {total} exceeds the "
            f"{len(payload)}-byte packet", kind="short_datagram")
    if payload[9] != 17:
        return None
    src = socket.inet_ntoa(payload[12:16])
    dst = socket.inet_ntoa(payload[16:20])
    udp = payload[ihl:total]
    if len(udp) < 8:
        raise _mvn_error(
            f"pcap {where}: UDP header is {len(udp)} bytes",
            kind="short_datagram")
    src_port, dst_port, length = struct.unpack(">HHH", udp[:6])
    if length < 8 or length > len(udp):
        raise _mvn_error(
            f"pcap {where}: UDP length {length} does not fit a "
            f"{len(udp)}-byte datagram", kind="short_datagram")
    return PcapPacket(
        timestamp_s=0.0, src=src, dst=dst, src_port=src_port,
        dst_port=dst_port, payload=udp[8:length])


def iter_pcap(data: bytes, *, where: str = "pcap"
              ) -> tuple[list[PcapPacket], dict[str, int]]:
    """Parse a whole legacy pcap file (bytes). Returns the UDP
    payloads plus counts: ``packets`` (records read),
    ``non_ip``/``non_udp`` (skipped by kind), ``truncated``
    (refused). A bad magic or a truncated record header is
    refused, naming the offset."""
    if len(data) < 24:
        raise _mvn_error(
            f"pcap {where} is {len(data)} bytes — no global "
            f"header", kind="short_datagram")
    magic = struct.unpack("<I", data[:4])[0]
    if magic not in _PCAP_MAGICS:
        magic_be = struct.unpack(">I", data[:4])[0]
        if magic_be in _PCAP_MAGICS:
            magic = magic_be
        else:
            raise _mvn_error(
                f"pcap {where} magic is 0x{magic:08x} — not a "
                f"legacy pcap file", kind="bad_magic")
    endian, _nano = _PCAP_MAGICS[magic]
    linktype = struct.unpack(endian + "I", data[20:24])[0]
    if linktype not in _LINK_HEADER:
        raise _mvn_error(
            f"pcap {where} linktype {linktype} is not supported "
            f"(NULL, Ethernet, RAW, Linux cooked)",
            kind="unsupported_linktype")
    strip = _LINK_HEADER[linktype]
    counts = {"packets": 0, "non_ip": 0, "non_udp": 0,
              "truncated": 0}
    packets: list[PcapPacket] = []
    off = 24
    record = struct.Struct(endian + "IIII")
    while off < len(data):
        if off + 16 > len(data):
            raise _mvn_error(
                f"pcap {where} record header at offset {off} is "
                f"truncated ({len(data) - off} bytes left)",
                kind="short_datagram")
        ts_s, ts_u, incl, _orig = record.unpack(data[off:off + 16])
        off += 16
        if off + incl > len(data):
            raise _mvn_error(
                f"pcap {where} record at offset {off} claims "
                f"{incl} bytes but {len(data) - off} remain",
                kind="short_datagram")
        raw = data[off:off + incl]
        off += incl
        counts["packets"] += 1
        link = raw[strip:]
        try:
            parsed = _ipv4_udp(link, where=where)
        except MocapError:
            counts["truncated"] += 1
            raise
        if parsed is None:
            if len(link) >= 1 and (link[0] >> 4) != 4:
                counts["non_ip"] += 1
            else:
                counts["non_udp"] += 1
            continue
        packets.append(PcapPacket(
            timestamp_s=ts_s + ts_u / 1_000_000.0,
            src=parsed.src, dst=parsed.dst,
            src_port=parsed.src_port, dst_port=parsed.dst_port,
            payload=parsed.payload))
    return packets, counts


def read_pcap(path: str | Path, *,
              where: str | None = None) -> tuple[list[PcapPacket],
                                                 dict[str, int]]:
    """Read a legacy pcap file from disk. A missing file is
    refused by name."""
    target = Path(path).expanduser()
    if not target.exists():
        raise _mvn_error(f"no pcap file at {target}",
                         kind="missing_pcap")
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise _mvn_error(
            f"pcap file at {target} is unreadable: {exc}",
            kind="bad_pcap") from exc
    return iter_pcap(data, where=where or str(target))


# -- datagram assembly --------------------------------------------

class MvnAssembler:
    """Reassemble multi-datagram samples: datagrams sharing a
    sample counter (and avatar + message type) are held until
    the last-flag datagram arrives, then yielded in counter
    order. A set that never completes is dropped when the
    buffer passes ``max_pending`` — counted, never silent."""

    def __init__(self, *, max_pending: int = 128) -> None:
        self._max_pending = max(1, int(max_pending))
        self._pending: dict[tuple[str, int, int], dict[int, bytes]] = {}
        self.dropped_incomplete = 0

    def feed(self, data: bytes) -> list[MvnDatagram]:
        """Feed one raw datagram; returns the complete messages
        (usually one) it finishes. A malformed datagram is
        refused by name — the caller's stream counts it."""
        datagram = decode_datagram(data)
        header = datagram.header
        key = (header.msg_type, header.avatar_id,
               header.sample_counter)
        if header.last and header.datagram_counter == 0:
            parts = self._pending.get(key)
            if not parts:
                return [datagram]
            # MVN Studio sends the tail of a split sample with
            # counter 0x80 (index 0 + last flag) — not 0x80 |
            # index as the spec's three-datagram example shows.
            # All four fixtures do this; the tail goes after
            # the highest pending index.
            parts[max(parts) + 1] = data
            return [self._merge(key)]
        parts = self._pending.setdefault(key, {})
        parts[header.datagram_counter] = data
        if not header.last:
            self._prune()
            return []
        expected = set(range(header.datagram_counter + 1))
        if set(parts) != expected:
            self.dropped_incomplete += 1
            del self._pending[key]
            raise _mvn_error(
                f"MVN sample {header.sample_counter} avatar "
                f"{header.avatar_id} is missing datagrams "
                f"{sorted(expected - set(parts))}",
                kind="incomplete_sample")
        return [self._merge(key)]

    def _merge(self, key: tuple[str, int, int]) -> MvnDatagram:
        """Join a completed set of parts into one datagram:
        the first part's header (item count summed), bodies in
        index order."""
        parts = self._pending.pop(key)
        ordered_parts = [decode_datagram(parts[i])
                         for i in sorted(parts)]
        first = ordered_parts[0]
        merged_header = MvnHeader(
            msg_type=first.header.msg_type,
            sample_counter=first.header.sample_counter,
            datagram_counter=0, last=True,
            item_count=sum(part.header.item_count
                           for part in ordered_parts),
            time_code_ms=first.header.time_code_ms,
            avatar_id=first.header.avatar_id)
        merged_body = b"".join(part.body
                               for part in ordered_parts)
        return decode_datagram(
            encode_header(merged_header, body_len=len(merged_body))
            + merged_body)

    def _prune(self) -> None:
        while len(self._pending) > self._max_pending:
            oldest = next(iter(self._pending))
            del self._pending[oldest]
            self.dropped_incomplete += 1


# -- sessions -----------------------------------------------------

@dataclass
class MvnSession:
    """One live MVN session: a sender, a receiver, or a
    replayer. Per-session, direct — no hub, no shared queue."""

    name: str
    kind: str
    target: str
    started_at: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind,
            "target": self.target, "started_at": self.started_at,
            "counts": dict(self.counts),
            "errors": list(self.errors[-5:]),
        }


_SESSIONS: dict[str, MvnSession] = {}


def list_sessions() -> list[dict[str, Any]]:
    """Every live MVN session, in start order."""
    return [session.to_dict() for session in _SESSIONS.values()]


def get_session(name: str) -> MvnSession:
    """One session by name — refused by name when unknown."""
    try:
        return _SESSIONS[name]
    except KeyError as exc:
        raise _mvn_error(f"no MVN session {name!r}",
                         kind="unknown_session") from exc


def stop_session(name: str) -> dict[str, Any]:
    """Stop a session by name. A receiver/replayer thread is
    asked to stop and the socket closed; the entry is removed
    once stopped."""
    session = get_session(name)
    stop = getattr(session, "_stop", None)
    if callable(stop):
        stop()
    return _SESSIONS.pop(name).to_dict()


def _new_session(name: str, kind: str, target: str) -> MvnSession:
    if not name:
        raise _mvn_error("MVN session name must not be empty",
                         kind="bad_session")
    if name in _SESSIONS:
        raise _mvn_error(
            f"MVN session {name!r} already exists",
            kind="duplicate_session")
    from datetime import UTC, datetime
    session = MvnSession(
        name=name, kind=kind, target=target,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"))
    _SESSIONS[name] = session
    return session


# -- receiver -----------------------------------------------------

class MvnReceiver:
    """MVN datagrams into the mocap store. ``feed`` takes raw
    datagram bytes (from a socket, a pcap, or a test); pose
    datagrams become ``skeleton``-channel take payloads the rig
    drives directly, and ``13`` datagrams refresh the name
    table for their avatar. Nothing is silently dropped: every
    refusal is counted by reason."""

    def __init__(self, store: MocapStateStore, *,
                 engine: str = "mvn",
                 channel: str = "skeleton",
                 name_map: dict[str, str] | None = None,
                 legacy: bool = False) -> None:
        self.store = store
        self.engine = engine
        self.channel = channel
        self.name_map = name_map
        self.legacy = legacy
        self._tables: dict[int, MvnNameTable] = {}
        self._assembler = MvnAssembler()
        self._last_time_ms: dict[int, int] = {}
        self._time_deltas: dict[int, list[int]] = {}
        self.counts: dict[str, int] = {
            "datagrams": 0, "frames": 0, "tables": 0,
            "dropped": 0,
        }
        self.dropped_by_reason: dict[str, int] = {}
        with suppress(MocapError):
            store.register_engine(
                engine, produces=(channel,), consumes=(),
                note="Xsens MVN receiver")

    def _drop(self, reason: str) -> None:
        self.counts["dropped"] += 1
        self.dropped_by_reason[reason] = \
            self.dropped_by_reason.get(reason, 0) + 1

    def table_for(self, avatar_id: int) -> MvnNameTable | None:
        """The live name table for an avatar, if one arrived."""
        return self._tables.get(avatar_id)

    def _frame_time(self, avatar_id: int,
                    time_code_ms: int) -> float:
        """The take's frame time, measured from that avatar's own
        time-code deltas (median of the last 8, in seconds).
        Avatars run on their own clocks — the tail of the
        three-dancer capture has one avatar's clock running
        ahead while another stalls, so a single shared tracker
        would read the divergence as the rate. The fixtures run
        near 240 Hz (time codes quantise to 4–5 ms); the spec's
        common rates are 60–120 Hz, so until two pose datagrams
        from one avatar have arrived the default is 1/120 s."""
        last = self._last_time_ms.get(avatar_id)
        deltas = self._time_deltas.setdefault(avatar_id, [])
        if last is not None:
            delta = time_code_ms - last
            if 0 < delta < 1000:
                deltas.append(delta)
                self._time_deltas[avatar_id] = deltas[-8:]
        self._last_time_ms[avatar_id] = time_code_ms
        if not deltas:
            return 1.0 / 120.0
        ordered = sorted(deltas)
        mid = len(ordered) // 2
        median = (ordered[mid] if len(ordered) % 2 == 1
                  else (ordered[mid - 1] + ordered[mid]) / 2.0)
        return median / 1000.0

    def feed(self, data: bytes) -> dict[str, Any]:
        """Feed one raw datagram. Returns what it did: frames
        written, tables refreshed, or the drop reason."""
        self.counts["datagrams"] += 1
        try:
            messages = self._assembler.feed(data)
        except MocapError as exc:
            self._drop(exc.kind)
            return {"dropped": exc.kind, "error": str(exc)}
        receipts: list[dict[str, Any]] = []
        for message in messages:
            receipts.append(self._handle(message))
        return {"messages": receipts,
                "counts": dict(self.counts)}

    def _handle(self, message: MvnDatagram) -> dict[str, Any]:
        header = message.header
        if header.msg_type == "13":
            try:
                table = decode_name_table(message.body)
            except MocapError as exc:
                self._drop(exc.kind)
                return {"dropped": exc.kind, "error": str(exc)}
            self._tables[header.avatar_id] = table
            self.counts["tables"] += 1
            return {"table": header.avatar_id,
                    "names": len(table.names),
                    "extra_bytes": len(table.extra)}
        if header.msg_type in ("01", "02"):
            table = self._tables.get(header.avatar_id)
            try:
                frame = mvn_pose_to_frame(message.items,
                                          table=table,
                                          legacy=self.legacy)
                skeleton = mvn_skeleton(
                    table, legacy=self.legacy,
                    frame_time=self._frame_time(
                        header.avatar_id, header.time_code_ms))
                payload = {
                    "skeleton": skeleton_to_dict(skeleton),
                    "frames": [{name: list(values)
                                for name, values in
                                frame.values.items()}],
                }
                self.store.write(
                    self.channel, self.engine, payload,
                    kind="snapshot")
            except MocapError as exc:
                self._drop(exc.kind)
                return {"dropped": exc.kind, "error": str(exc)}
            self.counts["frames"] += 1
            unmapped = getattr(frame, "unmapped", [])
            return {"frame": header.sample_counter,
                    "avatar": header.avatar_id,
                    "segments": len(message.items),
                    "unmapped": list(unmapped)}
        self._drop(f"unhandled_type_{header.msg_type}")
        return {"dropped": f"unhandled_type_{header.msg_type}",
                "msg_type": header.msg_type}


def _make_udp_socket(host: str, port: int) -> socket.socket:
    """A UDP socket bound to ``host:port``. A sandbox that
    forbids UDP refuses here by name, never with a bare
    traceback."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        raise _mvn_error(
            f"cannot bind UDP {host}:{port} for MVN: {exc}",
            kind="udp_unavailable") from exc
    return sock


def get_sender(name: str) -> MvnSender:
    """A session's sender — refused by name when the session
    is unknown or is not a sender."""
    session = get_session(name)
    sender = getattr(session, "_sender", None)
    if sender is None:
        raise _mvn_error(
            f"MVN session {name!r} is not a sender",
            kind="not_a_sender")
    return sender


def get_receiver(name: str) -> MvnReceiver:
    """A session's receiver — refused by name when the session
    is unknown or is not a receiver."""
    session = get_session(name)
    receiver = getattr(session, "_receiver", None)
    if receiver is None:
        raise _mvn_error(
            f"MVN session {name!r} is not a receiver",
            kind="not_a_receiver")
    return receiver


def get_replayer(name: str) -> MvnReplayer:
    """A session's replayer — refused by name when the session
    is unknown or is not a replayer."""
    session = get_session(name)
    replayer = getattr(session, "_replayer", None)
    if replayer is None:
        raise _mvn_error(
            f"MVN session {name!r} is not a replayer",
            kind="not_a_replayer")
    return replayer


def start_receiver(name: str, store: MocapStateStore, *,
                   host: str = "0.0.0.0",
                   port: int = MVN_DEFAULT_PORT,
                   engine: str = "mvn",
                   legacy: bool = False,
                   socket_factory: Callable[
                       [str, int], Any] | None = None
                   ) -> dict[str, Any]:
    """Listen for MVN datagrams on UDP ``host:port`` in a daemon
    thread, feeding a receiver. Returns the session. A socket
    that cannot bind refuses by name and no session is kept.
    ``legacy`` names segments by the spec's 23-segment table
    for old captures instead of the 2022 wire order."""
    factory = socket_factory or _make_udp_socket
    try:
        sock = factory(host, port)
    except (OSError, MocapError) as exc:
        if isinstance(exc, MocapError):
            raise
        raise _mvn_error(
            f"cannot bind UDP {host}:{port} for MVN: {exc}",
            kind="udp_unavailable") from exc
    session = _new_session(name, "receiver", f"udp://{host}:{port}")
    receiver = MvnReceiver(store, engine=engine, legacy=legacy)
    session.counts = receiver.counts
    stop_event = threading.Event()

    def _loop() -> None:
        with suppress(OSError):
            sock.settimeout(0.2)
        while not stop_event.is_set():
            try:
                data, _addr = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError as exc:
                session.errors.append(f"recv failed: {exc}")
                break
            try:
                receiver.feed(data)
            except Exception as exc:  # noqa: BLE001 - thread
                session.errors.append(f"feed failed: {exc}")
        with suppress(OSError):
            sock.close()

    thread = threading.Thread(target=_loop, daemon=True,
                              name=f"mvn-receiver-{name}")
    thread.start()

    def _stop() -> None:
        stop_event.set()
        thread.join(timeout=2.0)

    session._stop = _stop  # type: ignore[attr-defined]
    session._receiver = receiver  # type: ignore[attr-defined]
    return session.to_dict()


# -- sender -------------------------------------------------------

class MvnSender:
    """Store frames onto the MVN wire. ``send_frame`` reads the
    latest take off the ``skeleton`` channel, runs forward
    kinematics, and sends one ``02`` datagram per call; unmapped
    joints are counted by name, never silently dropped."""

    def __init__(self, store: MocapStateStore, *,
                 send: Callable[[bytes, tuple[str, int]], Any],
                 engine: str = "mvn",
                 channel: str = "skeleton",
                 avatar_id: int = 0,
                 name_map: dict[str, str] | None = None,
                 table: MvnNameTable | None = None) -> None:
        self.store = store
        self._send = send
        self.engine = engine
        self.channel = channel
        self.avatar_id = avatar_id
        self.name_map = name_map
        self.table = table
        self.sample_counter = 0
        self._started = time.monotonic()
        self.counts: dict[str, int] = {
            "sent": 0, "frames": 0, "dropped": 0,
        }
        self.unmapped: dict[str, int] = {}
        with suppress(MocapError):
            store.register_engine(
                engine, produces=(), consumes=(channel,),
                note="Xsens MVN sender")

    def send_frame(self, host: str,
                   port: int = MVN_DEFAULT_PORT) -> dict[str, Any]:
        """Send the latest store frame as one ``02`` datagram."""
        try:
            view = self.store.read(self.channel)
        except MocapError as exc:
            raise _mvn_error(
                f"channel {self.channel!r} holds no skeleton "
                f"take ({exc})", kind="no_take") from exc
        payload = view["payload"]
        if not isinstance(payload, dict) or "skeleton" not in payload:
            raise _mvn_error(
                f"channel {self.channel!r} holds no skeleton "
                f"take", kind="no_take")
        raw_frames = payload.get("frames", [])
        if not raw_frames:
            raise _mvn_error(
                f"channel {self.channel!r} holds no frames",
                kind="no_take")
        skeleton = skeleton_from_dict(payload["skeleton"])
        last = raw_frames[-1]
        frame = SkeletonFrame(values={
            name: tuple(float(v) for v in values)
            for name, values in last.items()})
        items, unmapped = frame_to_mvn_items(
            skeleton, frame, name_map=self.name_map,
            table=self.table)
        for joint in unmapped:
            self.unmapped[joint] = self.unmapped.get(joint, 0) + 1
        if not items:
            raise _mvn_error(
                "no take joint maps to an MVN segment — "
                f"unmapped: {', '.join(unmapped) or 'none'}",
                kind="no_mappable_joints")
        header = MvnHeader(
            msg_type="02", sample_counter=self.sample_counter,
            datagram_counter=0, last=True,
            item_count=len(items),
            time_code_ms=int(
                (time.monotonic() - self._started) * 1000),
            avatar_id=self.avatar_id)
        data = encode_datagram(header, items)
        try:
            self._send(data, (host, port))
        except OSError as exc:
            self.counts["dropped"] += 1
            raise _mvn_error(
                f"MVN send to {host}:{port} failed: {exc}",
                kind="udp_unavailable") from exc
        self.sample_counter += 1
        self.counts["sent"] += 1
        self.counts["frames"] += 1
        return {"sent": len(data), "segments": len(items),
                "sample": header.sample_counter,
                "unmapped": list(unmapped)}


def start_sender(name: str, store: MocapStateStore, *,
                 host: str = "127.0.0.1",
                 port: int = MVN_DEFAULT_PORT,
                 socket_factory: Callable[[], Any] | None = None,
                 **kwargs: Any) -> dict[str, Any]:
    """Start an MVN sender session: a UDP socket aimed at
    ``host:port``. Sending itself happens through
    :meth:`MvnSender.send_frame` (tool/CLI/chat call it per
    frame). A socket that cannot be made refuses by name."""
    if socket_factory is None:
        def _default() -> socket.socket:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect((host, port))
            except OSError as exc:
                sock.close()
                raise _mvn_error(
                    f"cannot open UDP to {host}:{port} for MVN: "
                    f"{exc}", kind="udp_unavailable") from exc
            return sock
        socket_factory = _default
    try:
        sock = socket_factory()
    except (OSError, MocapError) as exc:
        if isinstance(exc, MocapError):
            raise
        raise _mvn_error(
            f"cannot open UDP to {host}:{port} for MVN: {exc}",
            kind="udp_unavailable") from exc

    def _send(data: bytes, _addr: tuple[str, int]) -> None:
        sock.send(data)

    session = _new_session(name, "sender", f"udp://{host}:{port}")
    sender = MvnSender(store, send=_send, **kwargs)
    session.counts = sender.counts

    def _stop() -> None:
        with suppress(OSError):
            sock.close()

    session._stop = _stop  # type: ignore[attr-defined]
    session._sender = sender  # type: ignore[attr-defined]
    return session.to_dict()


# -- replayer (the clone's behaviour) -----------------------------

class MvnReplayer:
    """Replay a recorded ``.pcap`` of MVN packets to a host —
    the clone's whole behaviour, on a real pcap reader. One
    packet every 5 ms, a 500 ms gap between loops, looping
    until stopped. Sends and drops are counted; a send failure
    stops the replay with the reason named."""

    def __init__(self, pcap_path: str | Path, *,
                 send: Callable[[bytes, tuple[str, int]], Any],
                 host: str = "127.0.0.1",
                 port: int = MVN_DEFAULT_PORT,
                 pace_ms: float = 5.0,
                 loop_gap_ms: float = 500.0) -> None:
        self.pcap_path = str(pcap_path)
        self._send = send
        self.host = host
        self.port = port
        self.pace_s = pace_ms / 1000.0
        self.loop_gap_s = loop_gap_ms / 1000.0
        self.counts: dict[str, int] = {
            "sent": 0, "loops": 0, "dropped": 0,
        }
        self._stop_event = threading.Event()
        self._packets, self._pcap_counts = read_pcap(pcap_path)
        self._payloads = [packet.payload
                          for packet in self._packets]
        if not self._payloads:
            raise _mvn_error(
                f"pcap {pcap_path} holds no UDP payloads",
                kind="empty_pcap")

    @property
    def pcap_counts(self) -> dict[str, int]:
        """The pcap reader's counts (records, skipped kinds)."""
        return dict(self._pcap_counts)

    def play_once(self) -> dict[str, int]:
        """Send every payload once, paced. Returns the counts."""
        for payload in self._payloads:
            if self._stop_event.is_set():
                break
            try:
                self._send(payload, (self.host, self.port))
            except OSError as exc:
                self.counts["dropped"] += 1
                raise _mvn_error(
                    f"MVN replay send to {self.host}:{self.port} "
                    f"failed: {exc}", kind="udp_unavailable") from exc
            self.counts["sent"] += 1
            if self.pace_s > 0:
                time.sleep(self.pace_s)
        return dict(self.counts)

    def run_loop(self) -> dict[str, int]:
        """Loop :meth:`play_once` until :meth:`stop`, with the
        loop gap between passes."""
        while not self._stop_event.is_set():
            self.play_once()
            if self._stop_event.is_set():
                break
            self.counts["loops"] += 1
            if self.loop_gap_s > 0:
                self._stop_event.wait(self.loop_gap_s)
        return dict(self.counts)

    def stop(self) -> None:
        """Ask a running loop to stop after the current packet."""
        self._stop_event.set()


def start_replayer(name: str, pcap_path: str | Path, *,
                   host: str = "127.0.0.1",
                   port: int = MVN_DEFAULT_PORT,
                   socket_factory: Callable[[], Any] | None = None,
                   pace_ms: float = 5.0,
                   loop_gap_ms: float = 500.0) -> dict[str, Any]:
    """Start a replay session in a daemon thread: the pcap's
    packets to ``host:port``, looping. Returns the session. A
    socket that cannot be made, or an empty pcap, refuses by
    name and no session is kept."""
    if socket_factory is None:
        def _default() -> socket.socket:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect((host, port))
            except OSError as exc:
                sock.close()
                raise _mvn_error(
                    f"cannot open UDP to {host}:{port} for MVN "
                    f"replay: {exc}", kind="udp_unavailable") from exc
            return sock
        socket_factory = _default
    try:
        sock = socket_factory()
    except (OSError, MocapError) as exc:
        if isinstance(exc, MocapError):
            raise
        raise _mvn_error(
            f"cannot open UDP to {host}:{port} for MVN replay: "
            f"{exc}", kind="udp_unavailable") from exc

    def _send(data: bytes, _addr: tuple[str, int]) -> None:
        sock.send(data)

    replayer = MvnReplayer(
        pcap_path, send=_send, host=host, port=port,
        pace_ms=pace_ms, loop_gap_ms=loop_gap_ms)
    session = _new_session(
        name, "replayer", f"udp://{host}:{port} <- {pcap_path}")
    session.counts = replayer.counts

    thread = threading.Thread(target=replayer.run_loop,
                              daemon=True, name=f"mvn-replay-{name}")
    thread.start()

    def _stop() -> None:
        replayer.stop()
        thread.join(timeout=2.0)
        with suppress(OSError):
            sock.close()

    session._stop = _stop  # type: ignore[attr-defined]
    session._replayer = replayer  # type: ignore[attr-defined]
    return session.to_dict()
