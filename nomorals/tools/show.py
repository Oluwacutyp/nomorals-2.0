"""Show: one live run of mocap + rig + cloudcam, through named
states (storyteller-bevy patterns, pattern only, no upstream
code).

A thin surface over :mod:`nomorals.media.show` — the organ owns
the state machine, the schedule of phases, the plugins, the
take player and the event journal; the tool only routes
arguments and shapes the reply.

One show is one run, direct, no hub: it holds its own rig
session, its own cloudcam session, its own take player and its
own journal. Starting a show with no rig attached auto-attaches
the default rig and an idle cloudcam — and says so, by name, in
the receipt. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..media.show import (
    ShowError,
    drop_show,
    get_show,
    list_shows,
    load_show,
)
from ..media.show.show import Show

__all__ = ["ensure_defaults", "register"]

_ACTIONS = ("create", "list", "status", "start", "pause", "resume",
            "stop", "tick", "run", "events", "export", "attach_rig",
            "attach_cloudcam", "load_take", "drop")


def _params(params: dict | None) -> dict:
    return dict(params or {})


def _session_store() -> Any:
    """The shared mocap store — the show coordinates the organs
    that already read and write it."""
    from .mocap import _STORE
    return _STORE


def ensure_defaults(show: Show) -> list[str]:
    """Attach the default rig and an idle cloudcam when the show
    has none, naming each one. A show with no rig has nothing to
    drive — the default is attached, never silently assumed."""
    from ..media.cloudcam import get_session as get_cc_session
    from ..media.rig import get_session as get_rig_session

    attached: list[str] = []
    try:
        show.resource("rig")
    except ShowError:
        show.attach_rig(get_rig_session("default"))
        attached.append("rig:default")
    try:
        show.resource("cloudcam")
    except ShowError:
        show.attach_cloudcam(get_cc_session("default"))
        attached.append("cloudcam:idle")
    return attached


def _create(name: str, args: dict) -> Show:
    """Load a show from tool params."""
    from ..media.cloudcam import load_session as load_cc_session
    from ..media.rig import get_session as get_rig_session

    rig = None
    if args.get("rig"):
        rig = get_rig_session(str(args["rig"]))
    cam = None
    if args.get("cloudcam"):
        from ..media.cloudcam import get_session as get_cc_session
        cam = get_cc_session(str(args["cloudcam"]))
    elif args.get("cloudcam_source"):
        source = str(args["cloudcam_source"]).strip().lower()
        cam = load_cc_session(
            f"{name}-cam", source=source,
            endpoint=(str(args["endpoint"])
                      if args.get("endpoint") else None),
            path=(str(args["path"]) if args.get("path") else None),
            store=_session_store() if source == "store" else None)
    show = load_show(
        name, _session_store(), rig=rig, cloudcam=cam,
        fps=float(args.get("fps") or 15.0),
        render_dir=(str(args["render_dir"])
                    if args.get("render_dir") else None),
        export_path=(str(args["export_path"])
                     if args.get("export_path") else None))
    if args.get("take"):
        show.load_take(str(args["take"]))
    return show


def register(registry: Any) -> None:
    """Attach the show family to a registry."""
    @registry.register(
        "show",
        description=(
            "A live show — one run of mocap + rig + cloudcam "
            "through named states (storyteller-bevy Bevy "
            "patterns, pattern only). States: idle → arming → "
            "live → paused → live → stopping → done (or failed); "
            "a transition is requested and applied at the next "
            "schedule boundary, never mid-tick. Every live tick "
            "runs capture → drive → render → publish in that "
            "order: the take player steps a BVH take into the "
            "store's skeleton channel, the rig drives the newest "
            "frame (holding the last pose when nothing new "
            "arrived), the posed frame renders, the cloudcam "
            "stages the latest point-cloud frame. Starting with "
            "no rig auto-attaches the default rig and an idle "
            "cloudcam, named in the receipt. Show actions: "
            "create=<load a show, node=<name>, params=<{rig, "
            "cloudcam, cloudcam_source, endpoint, path, take, "
            "fps, render_dir, export_path}>>; list=<loaded "
            "shows>; status=<one show's state, node=<name>>; "
            "start=<arm then go live, node=<name>>; pause=<hold "
            "the show, node=<name>>; resume=<resume a paused "
            "show, node=<name>>; stop=<export when configured, "
            "release the claim, stop the show-owned cloudcam, "
            "node=<name>>; tick=<one schedule step, node=<name>>; "
            "run=<run a bounded number of ticks, node=<name>, "
            "params=<{ticks, interval_s}>>; events=<the show's "
            "journal, node=<name>, params=<{limit}>>; export=<"
            "write the driven poses as BVH, node=<name>, output="
            "<path.bvh>>; attach_rig=<attach a rig session, node="
            "<show>, value=<rig session>>; attach_cloudcam=<"
            "attach a cloudcam session, node=<show>, value=<"
            "cloudcam session>>; load_take=<load a BVH/JSON take, "
            "node=<show>, sources=[path.bvh|path.json]>; drop=<"
            "stop and drop a show, node=<name>>. action=<create|"
            "list|status|start|pause|resume|stop|tick|run|events|"
            "export|attach_rig|attach_cloudcam|load_take|drop>; "
            "sources=<list of paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def show_tool(action: str = "status",
                  sources: list[str] | None = None,
                  output: str = "", kind: str = "",
                  node: str = "", value: Any = None,
                  command: str = "",
                  params: dict | None = None,
                  title: str = "",
                  ) -> dict[str, Any]:
        from ..media.cloudcam import get_session as get_cc_session
        from ..media.rig import get_session as get_rig_session

        action = (action or "status").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown show action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        args = _params(params)
        sources = list(sources or [])
        name = (node or "").strip() or "default"
        if action == "list":
            return {"shows": list_shows()}
        if action == "create":
            return _create(name, args).status()
        show = get_show(name)
        if action == "status":
            return show.status()
        if action == "start":
            attached = ensure_defaults(show)
            return dict(show.start(), auto_attached=attached)
        if action == "pause":
            return show.pause()
        if action == "resume":
            return show.resume()
        if action == "stop":
            return show.stop()
        if action == "tick":
            return show.step()
        if action == "run":
            attached = ensure_defaults(show)
            ticks = int(args.get("ticks") or 30)
            interval = float(args.get("interval_s") or 0.0)
            return dict(show.run(ticks, interval),
                        auto_attached=attached)
        if action == "events":
            limit = args.get("limit")
            return {
                "show": name,
                "events": show.events(
                    int(limit) if limit is not None else 50),
                "events_dropped": show.events_dropped,
            }
        if action == "export":
            if not output:
                raise ValueError("export needs output=<path.bvh>")
            return dict(show.export(output), show=name)
        if action == "attach_rig":
            session_name = str(value or "").strip()
            if not session_name:
                raise ValueError(
                    "attach_rig needs value=<rig session>")
            return show.attach_rig(get_rig_session(session_name))
        if action == "attach_cloudcam":
            session_name = str(value or "").strip()
            if not session_name:
                raise ValueError(
                    "attach_cloudcam needs value=<cloudcam session>")
            return show.attach_cloudcam(
                get_cc_session(session_name))
        if action == "load_take":
            if not sources:
                raise ValueError(
                    "load_take needs sources=[path.bvh|path.json]")
            return dict(show.load_take(sources[0]), show=name)
        if action == "drop":
            return drop_show(name)
        raise ValueError(f"unhandled show action {action!r}")
