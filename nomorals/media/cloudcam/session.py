"""CloudCam sessions — one stream per session, direct, no hub.

A session binds ONE source to ONE ring buffer and ONE staging
grid. The worker thread polls the source and applies its frame
events to the ring; ``tick()`` stages the latest frame. There
is no shared queue and no centre: each session is independent,
and a session that falls behind drops frames in its own ring
and says so — one slow consumer never queues another's frames.

The poller never dies silently: a source error is recorded
(``last_error``, ``error_count``) and the thread backs off and
keeps polling. ``stop()`` joins the thread; a thread that will
not die in 5 seconds is refused by name, never abandoned.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .errors import CloudCamError
from .protocol import CloudCamPoint
from .ring import RingBuffer
from .sources import FrameSource
from .stage import StagedFrame, frame_bounds, placeholder_grid, stage_frame

__all__ = ["CloudCamSession"]

DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_CAPACITY = 10


class CloudCamSession:
    """One streamed point-cloud session."""

    def __init__(self, name: str, source: FrameSource, *,
                 width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT,
                 capacity: int = DEFAULT_CAPACITY) -> None:
        name = (name or "").strip()
        if not name:
            raise CloudCamError("session name must not be empty",
                                kind="bad_session")
        if width < 1 or height < 1:
            raise CloudCamError(
                f"stage grid must be at least 1x1, got "
                f"{width}x{height}",
                kind="bad_grid")
        self.name = name
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.ring = RingBuffer(capacity=capacity)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_frame_seq = 0
        self._staged_total = 0
        self._last_error: str | None = None
        self._error_count = 0
        self._started_at: float | None = None

    # -- lifecycle -------------------------------------------------

    @property
    def running(self) -> bool:
        """True while the poller thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> dict[str, Any]:
        """Start the poller thread. A running session refuses to
        start twice, by name."""
        if self.running:
            raise CloudCamError(
                f"cloudcam session {self.name!r} is already running",
                kind="already_running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name=f"cloudcam-{self.name}",
            daemon=True)
        self._thread.start()
        self._started_at = time.monotonic()
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Stop the poller and join it (5 s). A thread that will
        not die is refused by name — the session is left marked
        not-running, the thread stays named in the error."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise CloudCamError(
                    f"cloudcam session {self.name!r} poller did not "
                    f"stop in 5 s — it is still running",
                    kind="stop_timeout")
        self._thread = None
        return self.status()

    def _poll_loop(self) -> None:
        """Poll the source, apply events to the ring. Errors are
        recorded and backed off, never fatal to the loop."""
        while not self._stop.is_set():
            try:
                events = self.source.poll()
            except CloudCamError as exc:
                self._last_error = f"{exc.kind}: {exc.message}"
                self._error_count += 1
                self._stop.wait(0.5)
                continue
            except Exception as exc:  # noqa: BLE001 - poller never dies
                self._last_error = f"unexpected: {exc}"
                self._error_count += 1
                self._stop.wait(0.5)
                continue
            for is_begin, points in events:
                try:
                    if is_begin:
                        self.ring.new_frame(points,
                                            source=self.source.kind)
                    else:
                        self.ring.append_to_frame(
                            points, source=self.source.kind)
                except CloudCamError as exc:
                    self._last_error = f"{exc.kind}: {exc.message}"
                    self._error_count += 1
            self._stop.wait(0.005)

    # -- staging ---------------------------------------------------

    def tick(self) -> dict[str, Any]:
        """Stage the latest frame (or the placeholder grid when
        nothing has arrived yet). Returns the staging receipt
        plus the frame's bounds and the session's counters."""
        frame = self.ring.latest()
        if frame is None:
            staged = placeholder_grid(self.width, self.height)
            bounds = frame_bounds([])
            seq = 0
        else:
            staged = stage_frame(frame.points, self.width, self.height)
            bounds = frame_bounds(frame.points)
            seq = frame.seq
        self._last_frame_seq = seq
        self._staged_total += 1
        receipt = dict(staged.to_dict(), frame_seq=seq,
                       source=frame.source if frame is not None else "",
                       bounds=bounds)
        receipt["session"] = self.name
        receipt["running"] = self.running
        receipt["ring_dropped_frames"] = self.ring.dropped_frames
        receipt["ring_frames_received"] = self.ring.frames_received
        return receipt

    def latest_points(self) -> list[CloudCamPoint]:
        """The latest frame's points (a copy), or ``[]``."""
        frame = self.ring.latest()
        return list(frame.points) if frame is not None else []

    def staged(self) -> StagedFrame:
        """Stage the latest frame and keep the buffers (for a
        renderer that consumes them directly)."""
        frame = self.ring.latest()
        if frame is None:
            return placeholder_grid(self.width, self.height)
        return stage_frame(frame.points, self.width, self.height)

    # -- status ----------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Everything a status line needs, honest numbers only."""
        frame = self.ring.latest()
        return {
            "name": self.name,
            "running": self.running,
            "source": self.source.kind,
            "grid": f"{self.width}x{self.height}",
            "cells": self.width * self.height,
            "ring_capacity": self.ring.capacity,
            "ring_frames_received": self.ring.frames_received,
            "ring_dropped_frames": self.ring.dropped_frames,
            "last_frame_seq": self._last_frame_seq,
            "staged_total": self._staged_total,
            "latest_points": frame.count if frame is not None else 0,
            "last_error": self._last_error,
            "error_count": self._error_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Status, the registry's shape."""
        return self.status()
