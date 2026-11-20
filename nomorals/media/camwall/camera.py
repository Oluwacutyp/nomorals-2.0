"""One camera — the viewer's Camera and CameraUrl, in Python.

``CameraUrl`` is the clone's URL manipulation, field for field:
a ``full-url`` is used as-is, otherwise the snapshot URL is built
from the current host plus the image path, and every request gets
a cache-busting timestamp (``?t=`` when the URL has no query,
``&t=`` when it does). ``setLocal()`` / ``setRemote()`` switch
hosts, defaulting to remote when both are given — exactly the
clone's rule. ``image-params`` adds the Linksys ``size``/``quality``
pair the clone appends.

``Camera`` is the clone's per-camera state: times (first/last
requested/loaded/failed/aborted), counts (request/load/fail/abort/
cachePurge), pause, throttle, and the current frame — here as
JPEG bytes with the fetch time, since Devon keeps frames as data,
not as browser Image objects.
"""

from __future__ import annotations

import time
from typing import Any

from .config import CameraConfig

__all__ = ["Camera", "CameraUrl"]


class CameraUrl:
    """A camera's snapshot URL, with the clone's manipulation."""

    def __init__(self, config: CameraConfig) -> None:
        self.full_url = config.full_url
        self.remote_host = config.remote_host
        self.local_host = config.local_host
        self.image_path = config.image_path
        self.params = config.image_params
        # Default to remote when both are specified — the clone's rule.
        self.current_host = (self.remote_host if self.remote_host
                             else self.local_host)

    def set_local(self) -> None:
        """Use the local host when one exists, else stay remote."""
        self.current_host = (self.local_host if self.local_host
                             else self.remote_host)

    def set_remote(self) -> None:
        """Use the remote host when one exists, else stay local."""
        self.current_host = (self.remote_host if self.remote_host
                             else self.local_host)

    def base_url(self) -> str:
        """The snapshot URL before the cache-busting timestamp."""
        if self.full_url:
            return self.full_url
        return f"http://{self.current_host}{self.image_path}"

    def get_image_url(self, *, now_ms: int | None = None) -> str:
        """A timestamped snapshot URL — the timestamp breaks caches.

        ``?t=<ms>`` when the URL has no query string, ``&t=<ms>``
        when it does — the clone's exact rule. ``image-params``
        appends the Linksys ``size``/``quality`` pair after it.
        """
        url = self.base_url()
        stamp = str(now_ms if now_ms is not None
                    else int(time.time() * 1000))
        url += ("?t=" if "?" not in url else "&t=") + stamp
        if self.params:
            url += "&size=2&quality=3"
        return url


class Camera:
    """One camera's live state — the clone's Camera, as data."""

    def __init__(self, config: CameraConfig, *, cam_id: int = 0) -> None:
        self.id = cam_id
        self.config = config
        self.name = config.name
        self.location = config.location
        self.vendor = config.vendor
        self.model = config.model
        self.url = CameraUrl(config)
        self.times: dict[str, float | None] = {
            "first_requested": None,
            "first_loaded": None,
            "last_requested": None,
            "last_loaded": None,
            "last_loaded_request_time": None,
            "last_failed": None,
            "last_aborted": None,
        }
        self.counts: dict[str, int] = {
            "request": 0,
            "load": 0,
            "fail": 0,
            "abort": 0,
            "cache_purge": 0,
        }
        self.is_paused = False
        #: Throttle to apply before the next request, in ms.
        #: The wall recomputes it every tick from queue fullness.
        self.throttle_ms = 0
        #: The current frame: JPEG bytes, or None while no frame
        #: has ever loaded. ``frame_at`` is its fetch time.
        self.cur_frame: bytes | None = None
        self.frame_at: float | None = None
        #: The refusal kind of the last failed fetch (None when the
        #: camera never failed) — the status says WHY it fails.
        self.last_error: str | None = None
        #: Snapshot transport: "snapshot" (polling) or "mjpeg".
        self.transport = ("mjpeg" if config.mjpeg_url() else "snapshot")

    # -- pause ------------------------------------------------------
    def set_pause(self, pause: bool = True) -> None:
        """Set the pause state (no argument means pause)."""
        self.is_paused = bool(pause)

    def toggle_pause(self) -> bool:
        """Flip the pause state; returns the new state."""
        self.is_paused = not self.is_paused
        return self.is_paused

    # -- frame bookkeeping (the queue calls these) ------------------
    def note_request(self, url: str, *, now: float | None = None) -> float:
        """Record a request going out; returns its request time."""
        at = now if now is not None else time.time()
        self.counts["request"] += 1
        self.times["last_requested"] = at
        if self.times["first_requested"] is None:
            self.times["first_requested"] = at
        return at

    def note_loaded(self, frame: bytes, request_time: float,
                    *, now: float | None = None) -> bool:
        """Record a loaded frame. Returns False when the frame is
        stale — older than the frame already showing — so the wall
        never displays an older image over a newer one (the clone's
        onLoad rule)."""
        last = self.times["last_loaded_request_time"]
        if last is not None and request_time <= last:
            return False
        at = now if now is not None else time.time()
        self.counts["load"] += 1
        self.cur_frame = frame
        self.frame_at = at
        self.times["last_loaded"] = at
        self.times["last_loaded_request_time"] = request_time
        if self.times["first_loaded"] is None:
            self.times["first_loaded"] = at
        return True

    def note_failed(self, kind: str | None = None, *,
                    now: float | None = None) -> None:
        """Record a failed request — and why it failed."""
        self.counts["fail"] += 1
        self.last_error = kind
        self.times["last_failed"] = now if now is not None else time.time()

    def note_aborted(self, *, now: float | None = None) -> None:
        """Record an aborted request."""
        self.counts["abort"] += 1
        self.times["last_aborted"] = (now if now is not None
                                      else time.time())

    # -- reading ----------------------------------------------------
    def fps(self, *, now: float | None = None) -> float:
        """Frames per second since the first request (the clone's
        titlebar figure: loads / seconds since first request)."""
        first = self.times["first_requested"]
        if first is None or self.counts["load"] == 0:
            return 0.0
        at = now if now is not None else time.time()
        elapsed = max(at - first, 0.001)
        return self.counts["load"] / elapsed

    def status(self) -> dict[str, Any]:
        """This camera, as data — what the wall, tool and page show."""
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "vendor": self.vendor,
            "model": self.model,
            "transport": self.transport,
            "paused": self.is_paused,
            "throttle_ms": self.throttle_ms,
            "url": self.url.base_url(),
            "has_frame": self.cur_frame is not None,
            "frame_bytes": (len(self.cur_frame)
                            if self.cur_frame is not None else 0),
            "frame_at": self.frame_at,
            "last_error": self.last_error,
            "times": dict(self.times),
            "counts": dict(self.counts),
            "fps": round(self.fps(), 3),
        }
