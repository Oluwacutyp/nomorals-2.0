"""Camwall: constantly refreshing IP cameras, as a brain tool.

A thin surface over :mod:`nomorals.media.camwall` — the organ
owns the cameras, their queues, the wall loop and the web
export; the tool only routes arguments and shapes the reply.

One wall = one named set of cameras, its own loop thread and
its own fetch workers. A slow camera never stalls the rest:
each camera fetches on its own worker while the loop keeps
ticking. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.camwall import (
    DEFAULT_CONFIG_PATH,
    CamWallError,
    drop_wall,
    export_web,
    get_wall,
    list_served,
    list_walls,
    load_wall,
    serve_wall,
    stop_wall,
)

__all__ = ["camwall_sync", "register"]

_ACTIONS = ("load", "list", "start", "stop", "drop", "status", "tick",
            "pause", "resume", "focus", "wall", "server", "rows",
            "add", "snapshot", "export", "serve", "unserve", "served")


def _params(params: dict | None) -> dict:
    return dict(params or {})


def _wall_name(node: str) -> str:
    return (node or "").strip() or "default"


def camwall(action: str = "status", **kwargs: Any) -> dict[str, Any]:
    """The camwall family, as a plain function (sync — the organ
    is thread-based, not async)."""
    action = (action or "status").lower()
    if action not in _ACTIONS:
        raise ValueError(f"unknown camwall action {action!r}; "
                         f"use one of {', '.join(_ACTIONS)}")
    args = _params(kwargs.get("params"))
    name = _wall_name(str(kwargs.get("node") or ""))
    if action == "list":
        return {"walls": list_walls()}
    if action == "served":
        return {"servers": list_served()}
    if action == "load":
        source = args.get("config")
        if source is None:
            sources = kwargs.get("sources")
            if isinstance(sources, list) and sources:
                source = sources[0]
        if source is None:
            source = DEFAULT_CONFIG_PATH
        wall = load_wall(
            name, source,
            rows=int(args.get("rows") or 2),
            server=str(args.get("server") or "remote"))
        return dict(wall.status(), ok=True)
    if action == "serve":
        output = str(kwargs.get("output") or "")
        if not output:
            raise ValueError("serve needs output=<export dir>")
        return dict(serve_wall(
            output, host=str(args.get("host") or "127.0.0.1"),
            port=int(args.get("port") or 0)), ok=True)
    if action == "unserve":
        output = str(kwargs.get("output") or "")
        if not output:
            raise ValueError("unserve needs output=<export dir>")
        return dict(stop_wall(output), ok=True)
    wall = get_wall(name)
    if action == "start":
        return dict(wall.start(), ok=True)
    if action == "stop":
        return dict(wall.stop(), ok=True)
    if action == "drop":
        return drop_wall(name)
    if action == "status":
        return wall.status()
    if action == "tick":
        # A tool call is synchronous: wait for the tick's fetches
        # unless the caller explicitly says not to.
        wait = args.get("wait")
        dispatched = wall.tick(
            wait=(True if wait is None else bool(wait)))
        return dict(wall.status(), ticked=dispatched)
    if action in ("pause", "resume", "focus"):
        camera = str(kwargs.get("value") or "")
        if not camera:
            raise ValueError(f"{action} needs a camera name or id")
        if action == "pause":
            return dict(wall.pause(camera), ok=True)
        if action == "resume":
            return dict(wall.resume(camera), ok=True)
        return dict(wall.focus(camera), ok=True)
    if action == "wall":
        return dict(wall.multiview(), ok=True)
    if action == "server":
        value = str(kwargs.get("value") or "")
        if value not in ("local", "remote"):
            raise ValueError("server needs value=local|remote")
        return dict(wall.set_server(value), ok=True)
    if action == "rows":
        return dict(wall.set_rows(int(kwargs.get("value") or 2)), ok=True)
    if action == "add":
        entry = args.get("camera")
        if not isinstance(entry, dict):
            raise ValueError("add needs params.camera={...} "
                             "(name + full-url, or remote-host + image)")
        added = wall.add_camera(entry)
        return dict(added.status(), ok=True)
    if action == "snapshot":
        camera = str(kwargs.get("value") or "")
        output = str(kwargs.get("output") or "")
        if not camera:
            raise ValueError("snapshot needs a camera name or id")
        if not output:
            raise ValueError("snapshot needs output=<path.jpg>")
        frame = wall.snapshot(camera)
        target = Path(output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(frame)
        return {"wall": name, "camera": camera, "output": str(target),
                "bytes": len(frame), "ok": True}
    if action == "export":
        output = str(kwargs.get("output") or "")
        if not output:
            raise ValueError("export needs output=<dir>")
        return dict(export_web(
            wall, output,
            title=(str(kwargs.get("title") or args.get("title"))
                   if (kwargs.get("title") or args.get("title"))
                   else None),
            overwrite=bool(args.get("overwrite"))), ok=True)
    raise ValueError(f"unhandled camwall action {action!r}")


def camwall_sync(action: str = "status", **kwargs: Any) -> dict[str, Any]:
    """Sync entry — the registry, the chat handler and the CLI call
    this directly; the organ is thread-based, so there is no event
    loop to bridge."""
    try:
        return camwall(action, **kwargs)
    except CamWallError as exc:
        return {"ok": False, "error": str(exc), "kind": exc.kind}


def register(registry: Any) -> None:
    """Attach the camwall family to a registry."""
    @registry.register(
        "camwall",
        description=(
            "IP cameras wall — constantly refreshing IP camera feeds "
            "(ip-security-camera-viewer port, MIT): a wall of cameras "
            "from a JSON config (full-url, or remote/local host + "
            "image path, optional MJPEG + basic auth), each with its "
            "own request queue and throttle, refreshed on a loop a "
            "slow camera never stalls; a web export is the wall as a "
            "page (multiview grid, click for singleview, right-click "
            "to pause). Camwall actions: load=<load cameras, node="
            "<wall>, params=<{config, rows, server}>>; list=<loaded "
            "walls>; start=<start a wall's loop, node=<wall>>; stop=<"
            "stop a wall's loop, node=<wall>>; drop=<forget a wall, "
            "node=<wall>>; status=<one wall's state, node=<wall>>; "
            "tick=<one loop pass, node=<wall>>; pause=<pause a camera, "
            "node=<wall>, value=<camera>>; resume=<resume a camera, "
            "node=<wall>, value=<camera>>; focus=<singleview one "
            "camera, node=<wall>, value=<camera>>; wall=<back to the "
            "wall, node=<wall>>; server=<local|remote hosts, node="
            "<wall>, value=<local|remote>>; rows=<images per row 1..6, "
            "node=<wall>, value=<n>>; add=<add a camera, node=<wall>, "
            "params=<{camera: {...}}>>; snapshot=<save a camera's "
            "latest frame, node=<wall>, value=<camera>, output=<"
            "path.jpg>>; export=<export the wall as a page, node="
            "<wall>, output=<dir>>; serve=<serve an exported wall, "
            "output=<dir>>; unserve=<stop serving, output=<dir>>; "
            "served=<servers running>. action=<load|list|start|stop|"
            "drop|status|tick|pause|resume|focus|wall|server|rows|add|"
            "snapshot|export|serve|unserve|served>; node=<wall name>; "
            "value=<camera name/id, or local|remote, or rows>; output="
            "<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def camwall_tool(action: str = "status",
                     sources: list[str] | None = None,
                     output: str = "", kind: str = "",
                     node: str = "", value: Any = None,
                     command: str = "",
                     params: dict | None = None,
                     title: str = "",
                     ) -> dict[str, Any]:
        return camwall(action, sources=list(sources or []),
                       output=output, kind=kind, node=node, value=value,
                       command=command, params=params, title=title)
