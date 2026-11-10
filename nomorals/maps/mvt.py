"""Mapbox Vector Tile decoding — pure stdlib (Devon Atlas, M3).

Doc 21 S5 demoted every renderer to a replaceable boundary behind
Devon's own contracts; the vector renderer (M3) therefore starts
with a decoder Devon owns outright: hand-rolled protobuf wire
parsing for the MVT schema (vector-tile-spec 2.1), no protobuf
runtime, no network.  The schema, field by field:

* ``Tile`` — field 3 is the repeated ``Layer``.
* ``Layer`` — 1: name, 2: features, 3: keys, 4: values,
  5: extent (default 4096), 15: version (required by the spec).
* ``Feature`` — 1: id, 2: tags (packed key/value index pairs),
  3: GeomType (0 unknown, 1 point, 2 linestring, 3 polygon),
  4: geometry (packed command/parameter integers).
* ``Value`` — 1: string, 2: float, 3: double, 4: int64,
  5: uint64, 6: sint64 (zigzag), 7: bool.

Geometry is a command stream: each command integer carries a
command id in its low 3 bits (1 MoveTo, 2 LineTo, 7 ClosePath) and
a repeat count in the remaining bits; parameters are zigzag-encoded
deltas against a running cursor.  Decoding is typed end to end —
layers of named features whose geometry is a tuple of paths in the
layer's extent space — and it is *total or named*: a malformed
payload raises :class:`MvtError` saying exactly what is wrong
(truncated varint, unknown wire type, a field overrunning the
payload, a geometry command declaring more points than remain,
a tag index past the layer's key/value tables).  There is no
partial-silent decode: bytes that cannot be read as the schema
they claim to be are refused, never half-rendered.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

__all__ = [
    "COMMAND_CLOSE_PATH", "COMMAND_LINE_TO", "COMMAND_MOVE_TO",
    "DEFAULT_EXTENT", "GEOM_LINESTRING", "GEOM_POINT", "GEOM_POLYGON",
    "GEOM_UNKNOWN", "MvtError", "MvtFeature", "MvtLayer", "MvtPath",
    "MvtTile", "decode_mvt", "zigzag_decode",
]

#: Geometry commands (low 3 bits of a command integer).
COMMAND_MOVE_TO = 1
COMMAND_LINE_TO = 2
COMMAND_CLOSE_PATH = 7

#: GeomType values, as the wire carries them and as features name them.
GEOM_UNKNOWN = "unknown"
GEOM_POINT = "point"
GEOM_LINESTRING = "linestring"
GEOM_POLYGON = "polygon"

_GEOM_TYPES = {0: GEOM_UNKNOWN, 1: GEOM_POINT, 2: GEOM_LINESTRING,
               3: GEOM_POLYGON}

#: The extent a layer declares when it carries no extent field.
DEFAULT_EXTENT = 4096


class MvtError(ValueError):
    """An MVT defect, named in the message."""


def zigzag_decode(value: int) -> int:
    """One zigzag-encoded integer back to signed."""
    return (int(value) >> 1) ^ -(int(value) & 1)


# ── protobuf wire reading ─────────────────────────────────────────


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read one varint at ``pos``; return ``(value, next_pos)``."""
    result = 0
    shift = 0
    start = pos
    while True:
        if pos >= len(data):
            raise MvtError(
                "truncated varint at offset " + str(start))
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise MvtError(
                "varint at offset " + str(start) + " exceeds 64 bits")


def _read_fields(data: bytes) -> list[tuple[int, int, Any]]:
    """Parse a protobuf message body into ``(field, wire, value)``.

    Values are ``int`` for varints, ``bytes`` for length-delimited
    fields, and raw ``bytes`` for the fixed 32/64-bit wire types
    (only the caller knows their meaning; the schema this module
    serves uses none of them outside ``Value``'s float/double,
    which arrive length-delimited nowhere — they are wire 5/1
    fields on the ``Value`` message itself and are read here).
    """
    fields: list[tuple[int, int, Any]] = []
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        field_no, wire = key >> 3, key & 0x7
        if field_no == 0:
            raise MvtError("field number 0 at offset " + str(pos))
        if wire == 0:
            value, pos = _read_varint(data, pos)
            fields.append((field_no, wire, value))
        elif wire == 2:
            length, pos = _read_varint(data, pos)
            end = pos + length
            if end > len(data):
                raise MvtError(
                    "length-delimited field " + str(field_no)
                    + " overruns the payload: declares " + str(length)
                    + " bytes at offset " + str(pos) + ", only "
                    + str(len(data) - pos) + " remain")
            fields.append((field_no, wire, data[pos:end]))
            pos = end
        elif wire == 1:
            if pos + 8 > len(data):
                raise MvtError(
                    "64-bit field " + str(field_no) + " truncated at "
                    "offset " + str(pos))
            fields.append((field_no, wire, data[pos:pos + 8]))
            pos += 8
        elif wire == 5:
            if pos + 4 > len(data):
                raise MvtError(
                    "32-bit field " + str(field_no) + " truncated at "
                    "offset " + str(pos))
            fields.append((field_no, wire, data[pos:pos + 4]))
            pos += 4
        elif wire in (3, 4):
            raise MvtError(
                "unsupported wire type " + str(wire) + " (group) for "
                "field " + str(field_no) + " at offset " + str(pos))
        else:
            raise MvtError(
                "unknown wire type " + str(wire) + " for field "
                + str(field_no) + " at offset " + str(pos))
    return fields


def _packed_uints(blob: bytes, what: str) -> list[int]:
    """A packed repeated-uint32 body as a list of ints."""
    out: list[int] = []
    pos = 0
    while pos < len(blob):
        try:
            value, pos = _read_varint(blob, pos)
        except MvtError as exc:
            raise MvtError(what + ": " + str(exc)) from exc
        out.append(value)
    return out


def _as_uint_list(wire: int, value: Any, what: str) -> list[int]:
    """A repeated-uint32 field, packed (wire 2) or single (wire 0)."""
    if wire == 2:
        return _packed_uints(value, what)
    return [int(value)]


def _as_text(wire: int, value: Any, what: str) -> str:
    if wire != 2:
        raise MvtError(what + " is not a length-delimited string")
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MvtError(what + " is not valid UTF-8: " + str(exc)) from exc


# ── typed output ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MvtPath:
    """One path of a feature's geometry, in layer-extent units.

    ``points`` are absolute tile-space coordinates (the delta
    stream already applied).  ``closed`` records a ClosePath —
    for polygons the closing segment back to the first point is
    implied, never duplicated in ``points``.
    """

    points: tuple[tuple[int, int], ...]
    closed: bool = False


@dataclass(frozen=True)
class MvtFeature:
    """One decoded feature: id, resolved tags, typed paths."""

    id: int | None
    geom_type: str
    tags: dict[str, Any]
    paths: tuple[MvtPath, ...]


@dataclass(frozen=True)
class MvtLayer:
    """One decoded layer: name, extent, and its features."""

    name: str
    version: int
    extent: int
    features: tuple[MvtFeature, ...]


@dataclass(frozen=True)
class MvtTile:
    """A decoded tile: its layers, in wire order."""

    layers: tuple[MvtLayer, ...]

    def layer(self, name: str) -> MvtLayer | None:
        """The first layer called ``name``, or None."""
        for entry in self.layers:
            if entry.name == name:
                return entry
        return None

    def layer_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.layers)


# ── values, tags, geometry ────────────────────────────────────────


def _decode_value(body: bytes) -> Any:
    """One ``Value`` message → the Python value it carries."""
    result: Any = None
    for field_no, wire, value in _read_fields(body):
        if field_no == 1:
            result = _as_text(wire, value, "value string")
        elif field_no == 2:
            if wire != 5:
                raise MvtError("value float is not a 32-bit field")
            result = struct.unpack("<f", bytes(value))[0]
        elif field_no == 3:
            if wire != 1:
                raise MvtError("value double is not a 64-bit field")
            result = struct.unpack("<d", bytes(value))[0]
        elif field_no == 4:
            raw = int(value)
            result = raw - (1 << 64) if raw >= (1 << 63) else raw
        elif field_no == 5:
            result = int(value)
        elif field_no == 6:
            result = zigzag_decode(int(value))
        elif field_no == 7:
            result = bool(value)
        # unknown value fields are ignored (protobuf's own rule)
    return result


def _decode_tags(raw: list[int], keys: list[str],
                 values: list[Any]) -> dict[str, Any]:
    """Resolve a feature's tag index pairs against the layer tables."""
    if len(raw) % 2:
        raise MvtError(
            "feature tags are an odd count (" + str(len(raw))
            + "): key/value pairs required")
    tags: dict[str, Any] = {}
    for at in range(0, len(raw), 2):
        key_at, value_at = raw[at], raw[at + 1]
        if key_at >= len(keys):
            raise MvtError(
                "tag key index " + str(key_at) + " out of range: "
                "the layer declares " + str(len(keys)) + " keys")
        if value_at >= len(values):
            raise MvtError(
                "tag value index " + str(value_at) + " out of range: "
                "the layer declares " + str(len(values)) + " values")
        tags[keys[key_at]] = values[value_at]
    return tags


def _decode_geometry(raw: list[int],
                     geom_type: str) -> tuple[MvtPath, ...]:
    """The command stream → typed paths, cursor applied.

    MoveTo starts a new path; for point geometry every MoveTo
    coordinate is its own one-point path (a multipoint).  LineTo
    extends the open path — with none open it is a named defect,
    as is a ClosePath with nothing to close and any command whose
    declared count runs past the integers that remain.
    """
    paths: list[MvtPath] = []
    current: list[tuple[int, int]] | None = None
    closed = False
    cursor_x = 0
    cursor_y = 0
    pos = 0

    def flush() -> None:
        nonlocal current, closed
        if current is not None:
            paths.append(MvtPath(points=tuple(current), closed=closed))
        current = None
        closed = False

    while pos < len(raw):
        head = raw[pos]
        pos += 1
        command, count = head & 0x7, head >> 3
        if command == COMMAND_CLOSE_PATH:
            if current is None:
                raise MvtError(
                    "ClosePath with no open path at geometry index "
                    + str(pos - 1))
            closed = True
            flush()
            continue
        if command not in (COMMAND_MOVE_TO, COMMAND_LINE_TO):
            raise MvtError(
                "unknown geometry command " + str(command)
                + " at geometry index " + str(pos - 1))
        if pos + 2 * count > len(raw):
            raise MvtError(
                "geometry command overrun: command " + str(command)
                + " declares " + str(count) + " point(s) at geometry "
                "index " + str(pos - 1) + ", only "
                + str((len(raw) - pos) // 2) + " remain")
        points: list[tuple[int, int]] = []
        for _ in range(count):
            cursor_x += zigzag_decode(raw[pos])
            cursor_y += zigzag_decode(raw[pos + 1])
            pos += 2
            points.append((cursor_x, cursor_y))
        if command == COMMAND_MOVE_TO:
            flush()
            if geom_type == GEOM_POINT:
                for point in points:
                    paths.append(MvtPath(points=(point,)))
                current = None
            elif points:
                current = list(points)
            else:
                current = None
        else:  # LineTo
            if current is None:
                raise MvtError(
                    "LineTo before any MoveTo at geometry index "
                    + str(pos))
            current.extend(points)
    flush()
    return tuple(paths)


def _decode_feature(body: bytes, keys: list[str],
                    values: list[Any]) -> MvtFeature:
    feature_id: int | None = None
    geom_type = GEOM_UNKNOWN
    tag_raw: list[int] = []
    geom_raw: list[int] = []
    for field_no, wire, value in _read_fields(body):
        if field_no == 1:
            feature_id = int(value)
        elif field_no == 2:
            tag_raw.extend(_as_uint_list(wire, value, "feature tags"))
        elif field_no == 3:
            geom_type = _GEOM_TYPES.get(int(value), GEOM_UNKNOWN)
        elif field_no == 4:
            geom_raw.extend(
                _as_uint_list(wire, value, "feature geometry"))
        # field 5 (raster) and anything else: not geometry, ignored
    tags = _decode_tags(tag_raw, keys, values)
    paths = _decode_geometry(geom_raw, geom_type)
    return MvtFeature(id=feature_id, geom_type=geom_type, tags=tags,
                      paths=paths)


def _decode_layer(body: bytes) -> MvtLayer:
    version: int | None = None
    name = ""
    extent = DEFAULT_EXTENT
    keys: list[str] = []
    values: list[Any] = []
    feature_bodies: list[bytes] = []
    for field_no, wire, value in _read_fields(body):
        if field_no == 15:
            version = int(value)
        elif field_no == 1:
            name = _as_text(wire, value, "layer name")
        elif field_no == 5:
            extent = int(value)
        elif field_no == 3:
            keys.append(_as_text(wire, value, "layer key"))
        elif field_no == 4:
            values.append(_decode_value(bytes(value)))
        elif field_no == 2:
            if wire != 2:
                raise MvtError("layer feature is not a message")
            feature_bodies.append(bytes(value))
    if version is None:
        raise MvtError(
            "layer " + repr(name) + " carries no version field "
            "(the MVT spec requires one)")
    # Features decode after the whole layer is read: their tags
    # resolve against the key/value tables wherever those sit in
    # the wire order (encoders are not required to emit them first).
    features = tuple(
        _decode_feature(each, keys, values) for each in feature_bodies)
    return MvtLayer(name=name, version=version, extent=extent,
                    features=features)


def decode_mvt(data: bytes) -> MvtTile:
    """Decode one MVT payload into a typed :class:`MvtTile`.

    Raises :class:`MvtError` naming the defect on any malformed
    input — truncated, overrun, out-of-range, or uncommanded bytes
    are refused whole, never partially decoded.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise MvtError(
            "MVT payload must be bytes, got " + type(data).__name__)
    layers: list[MvtLayer] = []
    for field_no, wire, value in _read_fields(bytes(data)):
        if field_no == 3:
            if wire != 2:
                raise MvtError("tile layer is not a message")
            layers.append(_decode_layer(bytes(value)))
        # other top-level fields are not part of the 2.1 schema
    return MvtTile(layers=tuple(layers))
