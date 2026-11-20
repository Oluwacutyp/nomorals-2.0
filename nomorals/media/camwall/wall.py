"""The cameras wall — the viewer's Controller, in Python.

A ``Wall`` owns its cameras (one per config entry), one request
queue per camera, and the main loop: every 800 ms (the clone's
interval) it purges stale queue entries, skips paused cameras,
and throttles each camera by its queue fullness — throttle is
``7s * queue.percent_full()``, the clone's ``7000 * percentFull``
in milliseconds — skipping a camera whose newest request plus
its throttle is still in the future. Fetches run on a per-camera
worker (one in-flight fetch per camera) so a slow camera's thread
waits while the rest keep refreshing: the loop never blocks on a
fetch, and a slow camera never stalls the wall.

Walls are per-name, like cloudcam's sessions: ``load_wall`` /
``get_wall`` / ``list_walls`` / ``drop_wall``. A wall is its own
state; there is no shared hub.
"""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from .camera import Camera
from .config import CameraConfig, camera_from_dict, load_config
from .errors import CamWallError
from .fetch import fetch_mjpeg_frame, fetch_snapshot
from .queue import RequestQueue

__all__ = [
    "LOOP_INTERVAL_S",
    "Wall",
    "drop_wall",
    "get_wall",
    "list_walls",
    "load_wall",
]

#: Main-loop interval, seconds — the clone's 800 ms setTimeout.
LOOP_INTERVAL_S = 0.8

#: Throttle scale, seconds — the clone's 7000 ms * percentFull.
THROTTLE_SCALE_S = 7.0

_log = get_logger(__name__)


class Wall:
    """One cameras wall: its cameras, their queues, the loop."""

    def __init__(self, name: str, cameras: list[CameraConfig], *,
                 rows: int = 2, server: str = "remote",
                 loop_interval_s: float = LOOP_INTERVAL_S) -> None:
        if not cameras:
            raise CamWallError("a wall needs at least one camera",
                               kind="no_cameras")
        self.name = name
        self.rows = max(1, min(int(rows), 6))
        self.server = "local" if server == "local" else "remote"
        self.loop_interval_s = float(loop_interval_s)
        self.cameras = [Camera(cfg, cam_id=i)
                        for i, cfg in enumerate(cameras)]
        if self.server == "local":
            for camera in self.cameras:
                camera.url.set_local()
        self.queues: dict[int, RequestQueue] = {
            camera.id: RequestQueue() for camera in self.cameras}
        #: Path the wall was loaded from, when known — `add` writes
        #: the config back there.
        self.config_path: str | None = None
        self._lock = threading.Lock()
        self._running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._in_flight: set[int] = set()
        self._futures: list[Any] = []

    # -- cameras ----------------------------------------------------
    def find(self, name_or_id: str | int) -> Camera:
        """A camera by name or numeric id, or a named refusal."""
        if isinstance(name_or_id, int) or (
                isinstance(name_or_id, str)
                and name_or_id.strip().isdigit()):
            wanted = int(name_or_id)
            for camera in self.cameras:
                if camera.id == wanted:
                    return camera
        else:
            wanted = str(name_or_id).strip().lower()
            for camera in self.cameras:
                if camera.name.lower() == wanted:
                    return camera
        raise CamWallError(
            f"wall {self.name!r} has no camera {name_or_id!r}",
            kind="unknown_camera")

    def add_camera(self, entry: dict[str, Any]) -> Camera:
        """Add a camera from a config entry (validated, then live)."""
        config = camera_from_dict(entry, index=len(self.cameras))
        camera = Camera(config, cam_id=len(self.cameras))
        if self.server == "local":
            camera.url.set_local()
        with self._lock:
            self.cameras.append(camera)
            self.queues[camera.id] = RequestQueue()
        if self.config_path:
            from .config import dump_config
            dump_config([cam.config for cam in self.cameras],
                        self.config_path)
        return camera

    # -- the loop ---------------------------------------------------
    def _fetch_one(self, camera: Camera, queue: RequestQueue,
                   request_time: float, url: str) -> None:
        """One fetch on a worker thread: load, fail or abort, then
        resolve the queue entry and clear the in-flight mark."""
        try:
            if camera.transport == "mjpeg":
                frame = fetch_mjpeg_frame(
                    url, timeout_s=camera.config.timeout_s,
                    username=camera.config.username,
                    password=camera.config.password)
            else:
                frame = fetch_snapshot(
                    url, timeout_s=camera.config.timeout_s,
                    username=camera.config.username,
                    password=camera.config.password)
        except CamWallError as exc:
            _log.debug("camwall %s camera %s: %s",
                       self.name, camera.name, exc)
            with self._lock:
                camera.note_failed(exc.kind)
                queue.resolve(request_time, "failed")
                self._in_flight.discard(camera.id)
            return
        except Exception as exc:  # noqa: BLE001 - a fetch never kills the wall
            _log.debug("camwall %s camera %s: %s",
                       self.name, camera.name, exc)
            with self._lock:
                camera.note_failed("fetch_failed")
                queue.resolve(request_time, "failed")
                self._in_flight.discard(camera.id)
            return
        with self._lock:
            if queue.is_stale(request_time):
                queue.resolve(request_time, "aborted")
                camera.note_aborted()
            else:
                camera.note_loaded(frame, request_time)
                queue.resolve(request_time, "loaded")
            self._in_flight.discard(camera.id)

    def tick(self, *, wait: bool = False,
             timeout: float = 30.0) -> list[int]:
        """One main-loop pass (the clone's mainLoop body).

        Purges stale entries, skips paused cameras and cameras
        with a fetch in flight, throttles by queue fullness, and
        dispatches the rest. Returns the ids it dispatched.
        With ``wait=True`` the dispatched fetches are joined first
        (tests and the CLI use this; the loop thread does not).
        The fetch workers exist from the first tick — a manual tick
        fetches concurrently exactly like the loop does, so a slow
        camera never stalls the rest even without ``start()``.
        """
        now = time.time()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, len(self.cameras)),
                thread_name_prefix=f"camwall-{self.name}")
        dispatched: list[tuple[int, float, str]] = []
        with self._lock:
            for camera in self.cameras:
                queue = self.queues[camera.id]
                queue.purge_old()
                if camera.is_paused:
                    continue
                if camera.id in self._in_flight:
                    continue
                camera.throttle_ms = round(
                    THROTTLE_SCALE_S * 1000 * queue.percent_full())
                newest = queue.newest()
                if newest is not None and (
                        newest + camera.throttle_ms / 1000.0 > now):
                    continue
                url = camera.url.get_image_url()
                request_time = queue.request(url, now=now)
                camera.note_request(url, now=now)
                dispatched.append((camera.id, request_time, url))
                self._in_flight.add(camera.id)
        futures = []
        for cam_id, request_time, url in dispatched:
            futures.append(self._executor.submit(
                self._fetch_one, self.cameras[cam_id],
                self.queues[cam_id], request_time, url))
        self._futures = futures
        if wait:
            for future in futures:
                # Fetch errors are recorded on the camera, not raised.
                with contextlib.suppress(Exception):
                    future.result(timeout=timeout)
        return [cam_id for cam_id, _, _ in dispatched]

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the loop never dies
                _log.debug("camwall %s tick failed: %s", self.name, exc)
            self._stop.wait(self.loop_interval_s)

    def start(self) -> dict[str, Any]:
        """Start the wall's loop (daemon thread + fetch workers)."""
        if self._running:
            return self.status()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, len(self.cameras)),
                thread_name_prefix=f"camwall-{self.name}")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"camwall-loop:{self.name}")
        self._thread.start()
        self._running = True
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Stop the loop and the fetch workers."""
        self._stop.set()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._running = False
        return self.status()

    # -- reading and control ----------------------------------------
    @property
    def running(self) -> bool:
        """True while the loop thread is up."""
        return self._running

    def status(self) -> dict[str, Any]:
        """The wall, as data."""
        with self._lock:
            cameras = [camera.status() for camera in self.cameras]
            totals = {
                "request": sum(c.counts["request"] for c in self.cameras),
                "load": sum(c.counts["load"] for c in self.cameras),
                "fail": sum(c.counts["fail"] for c in self.cameras),
                "abort": sum(c.counts["abort"] for c in self.cameras),
            }
        return {
            "wall": self.name,
            "running": self._running,
            "cameras": len(self.cameras),
            "rows": self.rows,
            "server": self.server,
            "loop_interval_s": self.loop_interval_s,
            "totals": totals,
            "camera_list": cameras,
        }

    def pause(self, name_or_id: str | int) -> dict[str, Any]:
        """Pause one camera (it keeps its last frame)."""
        camera = self.find(name_or_id)
        camera.set_pause(True)
        return camera.status()

    def resume(self, name_or_id: str | int) -> dict[str, Any]:
        """Resume one camera."""
        camera = self.find(name_or_id)
        camera.set_pause(False)
        return camera.status()

    def focus(self, name_or_id: str | int) -> dict[str, Any]:
        """Singleview: show one camera, pause all the rest."""
        wanted = self.find(name_or_id)
        for camera in self.cameras:
            camera.set_pause(camera.id != wanted.id)
        return self.status()

    def multiview(self) -> dict[str, Any]:
        """Back to the wall: unpause every camera."""
        for camera in self.cameras:
            camera.set_pause(False)
        return self.status()

    def set_server(self, server: str) -> dict[str, Any]:
        """Switch every camera between its local and remote host."""
        if server not in ("local", "remote"):
            raise CamWallError(
                f"camwall server is 'local' or 'remote', got {server!r}",
                kind="bad_config")
        self.server = server
        for camera in self.cameras:
            if server == "local":
                camera.url.set_local()
            else:
                camera.url.set_remote()
        return self.status()

    def set_rows(self, rows: int) -> dict[str, Any]:
        """Images per row in the wall view (1..6, the clone's range)."""
        if not isinstance(rows, int) or rows < 1 or rows > 6:
            raise CamWallError(
                f"camwall rows must be 1..6, got {rows!r}",
                kind="bad_run")
        self.rows = rows
        return self.status()

    def snapshot(self, name_or_id: str | int) -> bytes:
        """A camera's latest frame, or a named refusal."""
        camera = self.find(name_or_id)
        if camera.cur_frame is None:
            raise CamWallError(
                f"camera {camera.name!r} has staged no frame yet — "
                f"start the wall and tick first",
                kind="no_frame")
        return camera.cur_frame


# -- named walls ------------------------------------------------------

_WALLS: dict[str, Wall] = {}


def load_wall(name: str, source: Any, *,
              rows: int = 2, server: str = "remote") -> Wall:
    """Load (or replace) a named wall from a cameras config.

    ``source`` is anything :func:`config.load_config` takes: a
    dict, a JSON string, a file, or a path. A running wall of the
    same name is stopped first. When ``source`` is a path, the wall
    remembers it so `add` can write the config back.
    """
    configs = load_config(source)
    old = _WALLS.get(name)
    if old is not None and old.running:
        old.stop()
    wall = Wall(name, configs, rows=rows, server=server)
    if isinstance(source, (str, Path)) and not str(source).strip().startswith(("{", "[")):
        wall.config_path = str(Path(str(source)).expanduser())
    _WALLS[name] = wall
    return wall


def get_wall(name: str) -> Wall:
    """A loaded wall by name, or a named refusal."""
    wall = _WALLS.get(name)
    if wall is None:
        raise CamWallError(f"no camwall {name!r} is loaded",
                           kind="no_wall")
    return wall


def list_walls() -> list[dict[str, Any]]:
    """Every loaded wall, as status summaries."""
    return [wall.status() for wall in _WALLS.values()]


def drop_wall(name: str) -> dict[str, Any]:
    """Stop and forget a named wall."""
    wall = _WALLS.pop(name, None)
    if wall is None:
        raise CamWallError(f"no camwall {name!r} is loaded",
                           kind="no_wall")
    # Always stop: a wall ticked by hand owns fetch workers even
    # though its loop never started.
    wall.stop()
    return {"wall": name, "dropped": True}
