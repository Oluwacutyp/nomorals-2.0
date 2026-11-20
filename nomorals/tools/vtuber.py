"""VTuber show: a scripted character performance on a stage
(AnimationTest pattern: UE5 VTuber show, pattern only).

A thin surface over :mod:`nomorals.media.vtuber` — the organ owns
the character, the act, the cameras, the act player and the show
plugin; the tool only routes arguments and shapes the reply.

One performance per show, direct, no hub: a character (named
expressions + gestures over the rig) performs an act (timed beats:
say / expression / gesture / camera / take / face_take) through a
show's own phases, with audience cues landing at the next tick
and a live MVN receiver attachable as the show's capture source
(the recorded take holds while live is attached). Writes output
files, so FS_WRITE.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..media.show import ShowError, drop_show, get_show, load_show
from ..media.show.show import Show
from ..media.vtuber.errors import VtuberError

__all__ = ["register"]

_ACTIONS = ("character", "act", "perform", "cue", "live", "stop",
            "status", "tick", "drop")


def _params(params: dict | None) -> dict:
    return dict(params or {})


def _session_store() -> Any:
    """The shared mocap store — the show coordinates the organs
    that already read and write it."""
    from .mocap import _STORE
    return _STORE


def _get_or_create_show(name: str) -> Show:
    """The named show, created on the shared store when missing
    (a performance starts from nothing — the show is the stage)."""
    try:
        return get_show(name)
    except ShowError:
        return load_show(name, _session_store())


def _performance(show: Show) -> Any:
    """The show's attached act player — refused by name."""
    player = getattr(show, "_vtuber_player", None)
    if player is None or not player._attached:
        raise VtuberError(
            f"show {show.name!r} performs nothing — perform an "
            f"act first (vtuber perform)",
            kind="not_performing")
    return player


def register(registry: Any) -> None:
    """Attach the vtuber family to a registry."""
    @registry.register(
        "vtuber",
        description=(
            "A VTuber show — a scripted character performance on a "
            "stage (AnimationTest UE5 VTuber show pattern, pattern "
            "only). A character (named expressions over the rig's "
            "52 ARKit blendshapes + named gestures over its 60 "
            "bones) performs an act (timed beats: say a line, make "
            "an expression, strike a gesture, cut camera, load a "
            "take, load a face take) through a show's own phases, "
            "with audience cues landing at the next tick and a live "
            "MVN receiver attachable as the show's capture source "
            "(the recorded take holds while live is attached). "
            "Vtuber actions: character=<load a character JSON, "
            "node=<name>, sources=[path]> | list loaded characters; "
            "act=<load a .vtact.json act, node=<name>, "
            "sources=[path]> | list loaded acts; perform=<attach "
            "an act to a show and start it, node=<show>, "
            "params=<{act, character}>>; cue=<an audience cue into "
            "a running show, node=<show>, params=<{kind: say | "
            "expression | gesture | camera, value, duration_s}>>; "
            "live=<attach a live MVN receiver, node=<show>, "
            "params=<{receiver}> | detach with params=<{detach: "
            "true}>>; stop=<detach the act and stop the show, "
            "node=<show>>; tick=<one schedule step, node=<show>>; "
            "status=<one show's performance state, node=<show>>; "
            "drop=<stop and drop a show, node=<show>>. action=<"
            "character|act|perform|cue|live|stop|status|tick|drop>; "
            "sources=<list of paths>; node=<name>."
        ),
        capability=Capability.FS_WRITE,
    )
    def vtuber_tool(action: str = "status",
                    sources: list[str] | None = None,
                    output: str = "", kind: str = "",
                    node: str = "", value: Any = None,
                    command: str = "",
                    params: dict | None = None,
                    title: str = "",
                    ) -> dict[str, Any]:
        from ..media.vtuber import (
            attach_live,
            attach_performance,
            detach_live,
            detach_performance,
            get_act,
            get_character,
            list_acts,
            list_characters,
            load_act,
            load_character,
        )
        from ..tools.show import ensure_defaults

        action = (action or "status").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown vtuber action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        args = _params(params)
        sources = list(sources or [])
        name = (node or "").strip() or "default"
        if action == "character":
            if sources:
                character = load_character(name, sources[0])
                return character.to_dict()
            return {"characters": [c["name"]
                                   for c in list_characters()]}
        if action == "act":
            if sources:
                act = load_act(name, sources[0])
                return {"name": act.name, "character": act.character,
                        "fps": act.fps, "beats": len(act.beats)}
            return {"acts": [a["name"] for a in list_acts()]}
        show = (_get_or_create_show(name) if action == "perform"
                else get_show(name))
        if action == "status":
            status = show.status()
            player = getattr(show, "_vtuber_player", None)
            live = getattr(show, "_live_mvn", None)
            if player is not None:
                # The performance state sits at the top level —
                # a status line names the act it is performing.
                status.update(player.state())
            status["live"] = (
                live["name"] if live is not None else None)
            return status
        if action == "perform":
            act_name = str(args.get("act") or "").strip()
            char_name = str(args.get("character") or "").strip()
            if not act_name or not char_name:
                raise ValueError(
                    "perform needs params={act, character}")
            act = get_act(act_name)
            character = get_character(char_name)
            attached = ensure_defaults(show)
            if getattr(show, "_vtuber_player", None) is not None:
                detach_performance(show)
            attach_performance(show, act, character)
            started = show.start()
            return dict(started, act=act.name,
                        character=character.name,
                        auto_attached=attached)
        if action == "cue":
            player = _performance(show)
            cue_kind = str(args.get("kind") or "").strip()
            cue_value = str(args.get("value") or "").strip()
            duration = args.get("duration_s")
            result = player.cue(
                cue_kind, cue_value,
                float(duration) if duration is not None else None)
            return dict(result, show=name,
                        state=player.state())
        if action == "live":
            if args.get("detach"):
                return detach_live(show)
            receiver_name = str(args.get("receiver") or "").strip()
            if not receiver_name:
                raise ValueError(
                    "live needs params={receiver}")
            from ..media.mocap.mvn import get_receiver
            receiver = get_receiver(receiver_name)
            return attach_live(show, receiver, receiver_name)
        if action == "stop":
            if getattr(show, "_vtuber_player", None) is not None:
                detach_performance(show)
            stopped = show.stop()
            return dict(stopped, ok=True)
        if action == "tick":
            return show.step()
        if action == "drop":
            return drop_show(name)
        raise ValueError(f"unhandled vtuber action {action!r}")
