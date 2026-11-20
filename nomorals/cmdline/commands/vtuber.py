"""``nm vtuber`` — a scripted character performance on a stage,
live from the terminal.

Thin CLI over the vtuber organ: the same characters, acts and
performances the ``vtuber`` brain tool drives, from the terminal.
Performing an act on a show with no rig attached auto-attaches
the default rig and an idle cloudcam — and says so, by name.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...media.show import drop_show, get_show, list_shows, load_show
from ...media.show.errors import ShowError
from ...media.vtuber.errors import VtuberError
from ...tools.show import ensure_defaults
from ...tools.vtuber import _session_store


def _out(args: argparse.Namespace, obj: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=2))
        return 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: {value}")
    else:
        print(obj)
    return 0


def _cmd_vtuber(args: argparse.Namespace, context: Any) -> int:
    """Route `nm vtuber` subcommands."""
    from ...media.vtuber import (
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

    action = args.vtuber_action
    name = (getattr(args, "session", "") or "").strip() or "default"
    try:
        if action == "character":
            path = getattr(args, "path", "") or ""
            if not path:
                return _out(args, {"characters": [
                    c["name"] for c in list_characters()]})
            load_name = (getattr(args, "name", "") or "").strip()
            return _out(args, load_character(load_name, path).to_dict())
        if action == "act":
            path = getattr(args, "path", "") or ""
            if not path:
                return _out(args, {"acts": [
                    a["name"] for a in list_acts()]})
            load_name = (getattr(args, "name", "") or "").strip()
            act = load_act(load_name, path)
            return _out(args, {"name": act.name,
                               "character": act.character,
                               "fps": act.fps,
                               "beats": len(act.beats)})
        if action == "list":
            return _out(args, {
                "characters": [c["name"] for c in list_characters()],
                "acts": [a["name"] for a in list_acts()],
                "shows": list_shows()})
        if action == "perform":
            act_name = (getattr(args, "act", "") or "").strip()
            char_name = (getattr(args, "character", "") or "").strip()
            if not act_name or not char_name:
                print("vtuber perform needs --act NAME --character "
                      "NAME", file=sys.stderr)
                return 2
            try:
                show = get_show(name)
            except ShowError:
                show = load_show(name, _session_store())
            attached = ensure_defaults(show)
            if getattr(show, "_vtuber_player", None) is not None:
                detach_performance(show)
            act = get_act(act_name)
            character = get_character(char_name)
            attach_performance(show, act, character)
            started = show.start()
            return _out(args, dict(started, act=act.name,
                                   character=character.name,
                                   auto_attached=attached))
        if action not in ("status", "tick", "cue", "live", "stop",
                          "drop"):
            print(f"unknown vtuber action: {action}",
                  file=sys.stderr)
            return 2
        show = get_show(name)
        if action == "status":
            status = show.status()
            player = getattr(show, "_vtuber_player", None)
            live = getattr(show, "_live_mvn", None)
            if player is not None:
                status.update(player.state())
            status["live"] = (
                live["name"] if live is not None else None)
            return _out(args, status)
        if action == "tick":
            return _out(args, show.step())
        if action == "cue":
            player = getattr(show, "_vtuber_player", None)
            if player is None or not player._attached:
                print(f"vtuber: show {name!r} performs nothing",
                      file=sys.stderr)
                return 1
            duration = getattr(args, "duration", 0.0) or None
            result = player.cue(
                getattr(args, "kind", "") or "",
                getattr(args, "value", "") or "",
                float(duration) if duration else None)
            return _out(args, dict(result, show=name,
                                   state=player.state()))
        if action == "live":
            if getattr(args, "detach", False):
                return _out(args, detach_live(show))
            receiver_name = (getattr(args, "receiver", "")
                             or "").strip()
            if not receiver_name:
                print("vtuber live needs --receiver NAME",
                      file=sys.stderr)
                return 2
            from ...media.mocap.mvn import get_receiver
            receiver = get_receiver(receiver_name)
            return _out(args, attach_live(show, receiver,
                                          receiver_name))
        if action == "stop":
            if getattr(show, "_vtuber_player", None) is not None:
                detach_performance(show)
            return _out(args, dict(show.stop(), ok=True))
        if action == "drop":
            return _out(args, drop_show(name))
    except (ShowError, VtuberError) as exc:
        print(f"vtuber: {exc}", file=sys.stderr)
        return 1
    print(f"unknown vtuber action: {action}", file=sys.stderr)
    return 2
