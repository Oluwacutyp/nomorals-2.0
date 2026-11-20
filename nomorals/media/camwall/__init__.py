"""Cameras wall — constantly refreshing IP cameras, in Python.

The port of echelon/ip-security-camera-viewer (MIT, Brandon
Thomas 2011 — see LICENSE): a wall of IP camera feeds that
refresh constantly, a slow camera never stalling the rest.
One camera = one snapshot URL (full-url, or a remote/local host
plus an image path) with a per-camera request queue; one wall =
the main loop that purges stale entries, throttles by queue
fullness and dispatches fetches on per-camera workers.

Public surface: :func:`load_wall`, :func:`get_wall`,
:func:`list_walls`, :func:`drop_wall`, :func:`export_web`,
:func:`serve_wall`, :func:`stop_wall`, :func:`list_served`,
plus the config and error types.
"""

from __future__ import annotations

from .camera import Camera, CameraUrl
from .config import (
    DEFAULT_CONFIG_PATH,
    CameraConfig,
    cameras_from_json,
    dump_config,
    load_config,
)
from .errors import CamWallError
from .fetch import (
    MAX_FRAME_BYTES,
    MJPEGReader,
    fetch_mjpeg_frame,
    fetch_snapshot,
    is_jpeg,
)
from .queue import RequestQueue
from .wall import Wall, drop_wall, get_wall, list_walls, load_wall
from .web import (
    WallServer,
    export_web,
    list_served,
    serve_wall,
    stop_wall,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MAX_FRAME_BYTES",
    "MJPEGReader",
    "Camera",
    "CameraConfig",
    "CameraUrl",
    "CamWallError",
    "RequestQueue",
    "Wall",
    "WallServer",
    "cameras_from_json",
    "drop_wall",
    "dump_config",
    "export_web",
    "fetch_mjpeg_frame",
    "fetch_snapshot",
    "get_wall",
    "is_jpeg",
    "list_served",
    "list_walls",
    "load_config",
    "load_wall",
    "serve_wall",
    "stop_wall",
]
