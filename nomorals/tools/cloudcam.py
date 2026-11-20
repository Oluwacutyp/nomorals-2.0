"""CloudCam: streamed point clouds (cloudcamplugin pattern port).

A thin surface over :mod:`nomorals.media.cloudcam` — the organ
owns the wire protocol, the ring buffer, the sources, the
staging and the sessions; the tool only routes arguments and
shapes the reply.

One session = one stream, direct, no hub: a worker thread polls
a source (ZeroMQ CloudCam wire, the mocap store's pointcloud
channel, a JSON/PLY file, or idle) into a ring buffer, and each
tick stages the latest frame into position + colour texture
buffers. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..media.cloudcam import (
    get_session,
    list_sessions,
    load_session,
    render_projection,
    write_ply,
)
from ..media.cloudcam.errors import CloudCamError

__all__ = ["register"]

_ACTIONS = ("start", "stop", "status", "list", "tick", "stage",
            "export", "render")


def _params(params: dict | None) -> dict:
    return dict(params or {})


def _session_store() -> Any:
    """The shared mocap store — the store source reads the
    pointcloud channel the Kinect bridge publishes there."""
    from .mocap import _STORE
    return _STORE


def _load(name: str, args: dict) -> Any:
    """Load (or replace) a session from tool params."""
    source = str(args.get("source") or "idle")
    store = _session_store() if source == "store" else None
    return load_session(
        name, source=source,
        endpoint=(str(args["endpoint"]) if args.get("endpoint") else None),
        path=(str(args["path"]) if args.get("path") else None),
        store=store,
        width=int(args.get("width") or 256),
        height=int(args.get("height") or 256),
        capacity=int(args.get("capacity") or 10))


def register(registry: Any) -> None:
    """Attach the cloudcam family to a registry."""
    @registry.register(
        "cloudcam",
        description=(
            "Streamed point clouds — a worker thread polls a frame "
            "source into a ring buffer and each tick stages the "
            "latest frame into position + colour texture buffers "
            "(cloudcamplugin pattern, pattern only). Sources: zmq "
            "(ZeroMQ PULL, CloudCam wire, default tcp://*:8888), "
            "store (the mocap store's pointcloud channel, Kinect "
            "depth, no device needed), file (a JSON frames file or "
            "an ASCII PLY file), idle (placeholder grid until a "
            "real source loads). Cloudcam actions: start=<open a "
            "stream, node=<session>, params=<{source, endpoint, "
            "path, width, height, capacity}>>; stop=<stop a "
            "session's poller, node=<session>>; status=<one "
            "session's state, node=<session>>; list=<loaded "
            "sessions>; tick=<stage the latest frame, node="
            "<session>>; stage=<stage the latest frame, node="
            "<session>>; export=<write the latest frame as PLY, "
            "node=<session>, output=<path.ply>>; render=<render the "
            "latest frame as a PNG projection, node=<session>, "
            "output=<path.png>, params=<{view: top|front, width, "
            "height}>>. action=<start|stop|status|list|tick|stage|"
            "export|render>; sources=<list of paths>; output=<path "
            "to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def cloudcam_tool(action: str = "status",
                      sources: list[str] | None = None,
                      output: str = "", kind: str = "",
                      node: str = "", value: Any = None,
                      command: str = "",
                      params: dict | None = None,
                      title: str = "",
                      ) -> dict[str, Any]:
        action = (action or "status").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown cloudcam action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        args = _params(params)
        sources = list(sources or [])
        name = (node or "").strip() or "default"
        if action == "list":
            return {"sessions": list_sessions()}
        if action == "start":
            session = _load(name, args)
            return dict(session.start(), ok=True)
        if action == "stop":
            return dict(get_session(name).stop(), ok=True)
        if action == "status":
            return dict(get_session(name).status())
        if action in ("tick", "stage"):
            return get_session(name).tick()
        if action in ("export", "render"):
            session = get_session(name)
            points = session.latest_points()
            if not points:
                raise CloudCamError(
                    f"cloudcam session {name!r} has staged no frame "
                    f"yet — start a source and tick first",
                    kind="no_frame")
            if not output:
                need = "path.ply" if action == "export" else "path.png"
                raise ValueError(f"{action} needs output=<{need}>")
            if action == "export":
                return dict(write_ply(points, output),
                            session=name, points=len(points))
            view = str(args.get("view") or "top")
            return dict(render_projection(
                points, output,
                width=int(args.get("width") or 640),
                height=int(args.get("height") or 360),
                view=view), session=name)
        raise ValueError(f"unhandled cloudcam action {action!r}")
