"""The per-camera request queue — the clone's RequestQueue, in Python.

A queue of in-flight requests for one camera, at most
``max_length`` (4, the clone's number). The wall records a request
when a fetch starts and resolves it on load / fail / abort; the
queue itself never fetches — it is the bookkeeping the throttle
reads. ``purge_old()`` drops entries at or before the last loaded
request time (the clone's cache purge: a request superseded by a
newer loaded frame is dead weight). A frame whose request time is
at or before the last loaded one is stale and is never shown —
the queue enforces that, not the camera.
"""

from __future__ import annotations

import time
from typing import Any

__all__ = ["RequestQueue"]


class RequestQueue:
    """One camera's in-flight request queue (max 4, the clone's)."""

    def __init__(self, *, max_length: int = 4) -> None:
        self.max_length = max_length
        self.queue: list[dict[str, Any]] = []
        #: The request time of the last frame that actually loaded.
        self.last_loaded_request_time: float | None = None
        #: Entries purged as superseded, like the clone's cachePurge.
        self.purged = 0

    # -- recording --------------------------------------------------
    def request(self, url: str, *, now: float | None = None) -> float:
        """Record a request going out; returns its request time.

        When the queue is full, the oldest entry is dropped first
        (the clone shifts it out and nulls its image).
        """
        at = now if now is not None else time.time()
        while len(self.queue) >= self.max_length:
            self.queue.pop(0)
        self.queue.append({"url": url, "date": at, "state": "pending"})
        return at

    def resolve(self, request_time: float, state: str) -> None:
        """Mark the entry for ``request_time`` loaded/failed/aborted."""
        for entry in self.queue:
            if entry["date"] == request_time:
                entry["state"] = state
                if state == "loaded":
                    self.last_loaded_request_time = request_time
                return

    # -- reading ----------------------------------------------------
    def is_full(self) -> bool:
        """Queue capacity exhausted?"""
        return len(self.queue) >= self.max_length

    def percent_full(self) -> float:
        """Fraction of the queue used — the throttle reads this."""
        return len(self.queue) / self.max_length

    def newest(self) -> float | None:
        """Request time of the last queue item, or None when empty."""
        if not self.queue:
            return None
        return float(self.queue[-1]["date"])

    def oldest(self) -> float | None:
        """Request time of the first queue item, or None when empty."""
        if not self.queue:
            return None
        return float(self.queue[0]["date"])

    def is_stale(self, request_time: float) -> bool:
        """True when a frame from ``request_time`` must not display —
        a newer frame already loaded (the clone's onLoad guard)."""
        last = self.last_loaded_request_time
        return last is not None and request_time <= last

    # -- purging ----------------------------------------------------
    def purge_old(self) -> int:
        """Drop queue items at or before the last loaded request.

        Returns how many were dropped. The clone counts all but one
        of them as cache purges (the loaded one itself is not a
        purge); this port counts every dropped entry, and says so.
        """
        last = self.last_loaded_request_time
        if last is None:
            return 0
        found = 0
        for entry in self.queue:
            if float(entry["date"]) > last:
                break
            found += 1
        if not found:
            return 0
        self.purged += found
        del self.queue[:found]
        return found
