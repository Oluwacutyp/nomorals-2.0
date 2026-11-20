"""Ring buffer of point-cloud frames — the clone's shape, fixed.

Pattern from storytold/cloudcamplugin (PATTERN ONLY): the ZeroMQ
poller writes frames into a circular buffer of N slots and the
render tick reads the latest. The clone's buffer has two holes
Devon's does not:

* No lock: the poller thread and the render tick share the read
  and write indices with no synchronisation, and ``GetPoints``
  hands out the live array. Devon's buffer holds one lock and
  ``latest()`` returns a copy — a reader can never see a frame
  half-written.
* Silent drops: when the writer laps the reader, the overwritten
  frame vanishes without a trace. Devon counts every frame
  overwritten before it was ever read (``dropped_frames``) and
  reports it in status — a fast sender over a slow reader is a
  named fact, never a mystery.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .errors import CloudCamError
from .protocol import MAX_POINTS_PER_FRAME, CloudCamPoint

__all__ = ["PointCloudFrame", "RingBuffer"]

DEFAULT_CAPACITY = 10


@dataclass
class PointCloudFrame:
    """One frame of points, with its provenance."""

    points: list[CloudCamPoint]
    seq: int
    source: str = ""
    received_at: float = field(default_factory=time.monotonic)

    @property
    def count(self) -> int:
        """Points in this frame."""
        return len(self.points)


class RingBuffer:
    """A circular buffer of point-cloud frames.

    ``new_frame`` opens a frame (the wire's command 2);
    ``append_to_frame`` adds points to the frame in progress (any
    other command). ``latest`` returns a copy of the newest frame,
    or ``None`` when nothing has arrived yet.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        capacity = int(capacity)
        if capacity < 1:
            raise CloudCamError(
                f"ring capacity must be at least 1, got {capacity}",
                kind="bad_capacity")
        self._capacity = capacity
        # RLock: append_to_frame opens a frame via new_frame while
        # already holding the lock.
        self._lock = threading.RLock()
        self._slots: list[PointCloudFrame | None] = [None] * capacity
        self._write_index = 0
        self._read_index = 0
        self._seq = 0
        self._last_read_seq = 0
        self._frames_received = 0
        self._dropped_frames = 0

    @property
    def capacity(self) -> int:
        """Frame slots in the ring."""
        return self._capacity

    @property
    def frames_received(self) -> int:
        """Frames opened since this buffer was created."""
        with self._lock:
            return self._frames_received

    @property
    def dropped_frames(self) -> int:
        """Frames overwritten before any reader ever saw them."""
        with self._lock:
            return self._dropped_frames

    @property
    def current_read_index(self) -> int:
        """Slot the latest frame sits in."""
        with self._lock:
            return self._read_index

    def new_frame(self, points: list[CloudCamPoint],
                  *, source: str = "") -> int:
        """Open a new frame, advancing the write index. A slot
        holding a frame no reader ever saw counts as dropped."""
        if len(points) > MAX_POINTS_PER_FRAME:
            raise CloudCamError(
                f"frame of {len(points)} points exceeds the "
                f"{MAX_POINTS_PER_FRAME} cap",
                kind="frame_too_large")
        with self._lock:
            self._seq += 1
            victim = self._slots[self._write_index]
            if victim is not None and victim.seq > self._last_read_seq:
                self._dropped_frames += 1
            self._read_index = self._write_index
            self._slots[self._write_index] = PointCloudFrame(
                points=list(points), seq=self._seq, source=source)
            self._write_index = (self._write_index + 1) % self._capacity
            self._frames_received += 1
            return self._seq

    def append_to_frame(self, points: list[CloudCamPoint],
                        *, source: str = "") -> int:
        """Append points to the frame in progress. With no frame
        open yet, this opens one (a sender that never sends
        command 2 still produces frames)."""
        with self._lock:
            current = self._slots[self._read_index]
            if current is None:
                return self.new_frame(points, source=source)
            if len(current.points) + len(points) > MAX_POINTS_PER_FRAME:
                raise CloudCamError(
                    f"appending {len(points)} points would push "
                    f"frame {current.seq} over the "
                    f"{MAX_POINTS_PER_FRAME} cap",
                    kind="frame_too_large")
            current.points.extend(points)
            return current.seq

    def latest(self) -> PointCloudFrame | None:
        """A copy of the newest frame, or ``None`` before the
        first frame arrives. Reading marks the frame seen, so it
        is never counted as dropped."""
        with self._lock:
            frame = self._slots[self._read_index]
            if frame is None:
                return None
            self._last_read_seq = max(self._last_read_seq, frame.seq)
            return PointCloudFrame(points=list(frame.points),
                                   seq=frame.seq, source=frame.source,
                                   received_at=frame.received_at)

    def has_points(self) -> bool:
        """True once at least one non-empty frame has arrived."""
        frame = self.latest()
        return frame is not None and frame.count > 0
