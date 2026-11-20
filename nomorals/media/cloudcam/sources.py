"""Frame sources — where streamed point clouds come from.

Every source answers one question, non-blocking: ``poll()``
returns a list of frame events, each ``(is_begin, points)`` —
``is_begin`` opens a new frame (the wire's command 2), anything
else appends to the frame in progress. An empty list means
nothing new is ready. The session applies the events to the
ring; a source never blocks the worker thread.

* :class:`ZmqSource` — the clone's transport: a ZeroMQ PULL
  socket (bound by default on ``tcp://*:8888``, the clone's
  endpoint) reading CloudCam wire packets. pyzmq is optional —
  when it is not installed the source refuses by name instead
  of pretending to listen.
* :class:`StoreSource` — the mocap store's ``pointcloud``
  channel, as published by the Kinect bridge: depth-only points
  (``[x, y, z]`` in mm) get a depth-shaded colour, named as
  such. No device, no socket — the no-hardware path. New
  snapshots come from the store's delta log; when the log has
  rotated past what the source last saw, it re-syncs from the
  current snapshot and says so in the frame's source label.
* :class:`FileSource` — a JSON frames file
  (``{"frames": [[[x, y, z, r, g, b, a], …], …]}``) or an
  ASCII PLY file (``x y z`` plus ``red green blue alpha`` when
  present) — one frame per poll, then exhausted.
* :class:`CallableSource` — any zero-argument callable returning
  points (or ``None``) — the embedding and test path.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import CloudCamError
from .protocol import (
    COMMAND_BEGIN,
    CloudCamPoint,
    decode_data_packet,
)

__all__ = [
    "CallableSource",
    "FileSource",
    "FrameEvent",
    "FrameSource",
    "StoreSource",
    "ZmqSource",
    "depth_shade",
]

DEFAULT_ENDPOINT = "tcp://*:8888"

#: One frame event: (opens a new frame, points). A begin event
#: carries the first chunk of the new frame; an append carries
#: the next chunk of the frame in progress.
FrameEvent = tuple[bool, list[CloudCamPoint]]


def depth_shade(points: list[tuple[float, float, float]]
                ) -> list[CloudCamPoint]:
    """Depth-only points → coloured points. Colour is a ramp on
    depth (near = blue, far = red, mid green), alpha 255 — the
    colour is derived from the only signal a depth frame has,
    and it is named as derived wherever it is used."""
    if not points:
        return []
    depths = [p[2] for p in points]
    low, high = min(depths), max(depths)
    span = high - low
    coloured: list[CloudCamPoint] = []
    for x, y, z in points:
        t = 0.5 if span == 0 else (z - low) / span
        coloured.append(CloudCamPoint(
            x=float(x), y=float(y), z=float(z),
            r=int(round(t * 255)), g=64,
            b=int(round((1.0 - t) * 255)), a=255))
    return coloured


class FrameSource:
    """One pollable producer of frame events."""

    kind = "base"

    def poll(self) -> list[FrameEvent]:
        """Frame events since the last poll, or ``[]``."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the source (sockets, files)."""


class CallableSource(FrameSource):
    """A callable returning points (or ``None``) per poll."""

    kind = "callable"

    def __init__(self, fn: Callable[[], Any],
                 *, source: str = "callable") -> None:
        if not callable(fn):
            raise CloudCamError("callable source needs a callable",
                                kind="bad_source")
        self._fn = fn
        # The label IS the kind: an idle session reports "idle",
        # not the class name — status must name the source the
        # session was loaded with.
        self.kind = source

    def poll(self) -> list[FrameEvent]:
        raw = self._fn()
        if raw is None:
            return []
        return [(True, _coerce_points(raw))]


class ZmqSource(FrameSource):
    """ZeroMQ PULL, CloudCam wire packets. Binds by default (the
    clone's shape — the Rust sender PUSHes to it); ``bind=False``
    connects instead."""

    kind = "zmq"

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, *,
                 bind: bool = True) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise CloudCamError(
                "ZeroMQ streaming needs pyzmq — it is not "
                "installed on this machine (pip install pyzmq); "
                "use source=store or source=file instead",
                kind="missing_pyzmq") from exc
        self._zmq = zmq
        self.endpoint = endpoint
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PULL)
        try:
            if bind:
                self._socket.bind(endpoint)
            else:
                self._socket.connect(endpoint)
        except zmq.ZMQError as exc:
            self._socket.close()
            raise CloudCamError(
                f"ZeroMQ cannot {'bind' if bind else 'connect to'} "
                f"{endpoint}: {exc}",
                kind="zmq_bind") from exc

    def poll(self) -> list[FrameEvent]:
        """Drain every queued packet, in order. A bad packet
        refuses the whole poll by name — a corrupt stream never
        becomes a half-frame."""
        events: list[FrameEvent] = []
        while True:
            try:
                data = self._socket.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            except self._zmq.ZMQError as exc:
                raise CloudCamError(
                    f"ZeroMQ receive failed: {exc}",
                    kind="zmq_error") from exc
            command, points = decode_data_packet(data)
            events.append((command == COMMAND_BEGIN, points))
        return events

    def close(self) -> None:
        """Close the socket (the shared context stays)."""
        with contextlib.suppress(Exception):
            self._socket.close()


class StoreSource(FrameSource):
    """The mocap store's ``pointcloud`` channel. Each new
    snapshot becomes one frame; depth-only points are shaded
    by depth (see :func:`depth_shade`)."""

    kind = "store"

    def __init__(self, store: Any, *, channel: str = "pointcloud") -> None:
        if store is None:
            raise CloudCamError(
                "store source needs a mocap store — the cloudcam "
                "tool and CLI pass the shared mocap store",
                kind="bad_source")
        self._store = store
        self._channel = channel
        self._since_seq: int | None = None

    def poll(self) -> list[FrameEvent]:
        first = self._since_seq is None
        try:
            view = self._store.read(self._channel,
                                    since_seq=self._since_seq)
        except Exception:
            # Empty channel (no writes yet) and stale-channel
            # refusals both mean "nothing to stream right now" —
            # the session records nothing and polls again.
            return []
        seq = view.get("seq")
        if seq is not None:
            self._since_seq = seq
        changes = view.get("changes") or []
        if view.get("missed"):
            # The delta log rotated past what this source last
            # saw — re-sync from the current snapshot, one frame.
            payload = view.get("payload")
            points = _payload_points(payload)
            return [(True, points)] if points else []
        events: list[FrameEvent] = []
        for change in changes:
            points = _payload_points(change.get("payload"))
            if points:
                events.append((True, points))
        if first and not changes:
            # First poll, no delta log yet: the current snapshot
            # is the frame.
            points = _payload_points(view.get("payload"))
            if points:
                events.append((True, points))
        return events


class FileSource(FrameSource):
    """A JSON frames file or an ASCII PLY file. JSON yields one
    frame per poll until the list is spent; PLY yields one frame
    total, then is exhausted."""

    kind = "file"

    def __init__(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        if not target.exists():
            raise CloudCamError(f"no cloudcam file at {target}",
                                kind="missing_file")
        self.path = target
        self._frames: list[list[CloudCamPoint]] = []
        self._index = 0
        suffix = target.suffix.lower()
        try:
            if suffix == ".json":
                self._load_json(target)
            elif suffix == ".ply":
                self._load_ply(target)
            else:
                raise CloudCamError(
                    f"cloudcam file {target} must be .json or .ply, "
                    f"got {suffix or 'no suffix'}",
                    kind="bad_file")
        except CloudCamError:
            raise
        except (ValueError, OSError) as exc:
            raise CloudCamError(
                f"cloudcam file {target} is unreadable: {exc}",
                kind="bad_file") from exc

    def _load_json(self, target: Path) -> None:
        blob = json.loads(target.read_text(encoding="utf-8"))
        frames = blob.get("frames") if isinstance(blob, dict) else None
        if not isinstance(frames, list):
            raise CloudCamError(
                f"cloudcam JSON {target} needs a 'frames' list",
                kind="bad_file")
        for raw_frame in frames:
            self._frames.append(_coerce_points(raw_frame))

    def _load_ply(self, target: Path) -> None:
        text = target.read_text(encoding="utf-8", errors="strict")
        self._frames.append(_parse_ply(text, target))

    def poll(self) -> list[FrameEvent]:
        if self._index >= len(self._frames):
            return []
        frame = self._frames[self._index]
        self._index += 1
        return [(True, list(frame))]


def _payload_points(payload: Any) -> list[CloudCamPoint]:
    """A pointcloud-channel payload → shaded points. A payload
    without points yields ``[]`` (nothing to stream)."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("points")
    if not raw:
        return []
    return depth_shade([(float(p[0]), float(p[1]), float(p[2]))
                        for p in raw])


def _coerce_points(raw: Any) -> list[CloudCamPoint]:
    """A list of points in any of the accepted shapes —
    CloudCamPoint, ``[x, y, z]``, ``[x, y, z, a]``,
    ``[x, y, z, r, g, b, a]``, or ``{"x": …, "y": …, "z": …}``
    — into points. Anything else is refused by name."""
    if not isinstance(raw, (list, tuple)):
        raise CloudCamError(
            f"a frame must be a list of points, got "
            f"{type(raw).__name__}",
            kind="bad_frame")
    points: list[CloudCamPoint] = []
    for item in raw:
        if isinstance(item, CloudCamPoint):
            points.append(item)
        elif isinstance(item, dict):
            points.append(CloudCamPoint(
                x=float(item["x"]), y=float(item["y"]),
                z=float(item["z"]),
                r=int(item.get("r", 255)), g=int(item.get("g", 255)),
                b=int(item.get("b", 255)), a=int(item.get("a", 255))))
        elif isinstance(item, (list, tuple)) and len(item) in (3, 4, 7):
            if len(item) == 3:
                points.append(CloudCamPoint(
                    x=float(item[0]), y=float(item[1]),
                    z=float(item[2])))
            elif len(item) == 4:
                points.append(CloudCamPoint(
                    x=float(item[0]), y=float(item[1]),
                    z=float(item[2]), a=int(item[3])))
            else:
                points.append(CloudCamPoint(
                    x=float(item[0]), y=float(item[1]),
                    z=float(item[2]), r=int(item[3]),
                    g=int(item[4]), b=int(item[5]),
                    a=int(item[6])))
        else:
            raise CloudCamError(
                f"unreadable point {item!r} — use [x, y, z], "
                f"[x, y, z, r, g, b, a], or an object with x/y/z",
                kind="bad_point")
    return points


def _parse_ply(text: str, target: Path) -> list[CloudCamPoint]:
    """Minimal ASCII PLY 1.0 reader: vertex x y z, plus red/green/
    blue/alpha when the header declares them. Binary PLY is
    refused by name — it needs a real parser, not a guess."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise CloudCamError(f"PLY {target} does not start with 'ply'",
                            kind="bad_file")
    if len(lines) < 2 or lines[1].strip() != "format ascii 1.0":
        raise CloudCamError(
            f"PLY {target} must be 'format ascii 1.0' — binary PLY "
            f"is refused, not guessed",
            kind="bad_file")
    count = 0
    props: list[str] = []
    header_end = None
    in_vertex = False
    for i, line in enumerate(lines[1:], start=1):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
            continue
        if parts[0] == "property" and in_vertex:
            props.append(parts[-1])
            continue
        if parts[0] == "end_header":
            header_end = i
            break
    if header_end is None:
        raise CloudCamError(f"PLY {target} has no end_header",
                            kind="bad_file")
    if "x" not in props or "y" not in props or "z" not in props:
        raise CloudCamError(
            f"PLY {target} has no x/y/z vertex properties",
            kind="bad_file")
    idx = {name: props.index(name) for name in props}
    rows = lines[header_end + 1:header_end + 1 + count]
    if len(rows) < count:
        raise CloudCamError(
            f"PLY {target} declares {count} vertices but holds "
            f"{len(rows)} rows",
            kind="bad_file")
    points: list[CloudCamPoint] = []
    for row in rows:
        vals = row.split()
        points.append(CloudCamPoint(
            x=float(vals[idx["x"]]), y=float(vals[idx["y"]]),
            z=float(vals[idx["z"]]),
            r=int(float(vals[idx["red"]])) if "red" in idx else 255,
            g=int(float(vals[idx["green"]])) if "green" in idx else 255,
            b=int(float(vals[idx["blue"]])) if "blue" in idx else 255,
            a=int(float(vals[idx["alpha"]])) if "alpha" in idx else 255))
    return points
