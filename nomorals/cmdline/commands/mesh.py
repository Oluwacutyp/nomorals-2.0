"""``nm mesh`` — device mesh: nodes, presence, task dispatch."""
from __future__ import annotations

import json
import sys
from typing import Any


def _transport(context: Any):
    from ...mesh import LocalTransport
    from ...storage.db import Database
    db = getattr(context, "db", None)
    if db is None:
        from pathlib import Path
        home = Path.home() / ".nomorals"
        home.mkdir(parents=True, exist_ok=True)
        db = Database(str(home / "nomorals.db"))
    return LocalTransport(db)


def _cmd_mesh(args: Any, context: Any) -> int:
    """Route ``nm mesh <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm mesh nodes [--json]\n"
              "       nm mesh register <name> [--platform P]\n"
              "       nm mesh heartbeat <node-id>\n"
              "       nm mesh dispatch <task-type> [--target NODE] [--json-args '{}']",
              file=sys.stderr)
        return 2
    verb = words[0]
    t = _transport(context)
    as_json = bool(getattr(args, "json", False))
    if verb == "nodes":
        nodes = t.active_nodes()
        if as_json:
            print(json.dumps([n.to_dict() for n in nodes], indent=2, default=str))
        else:
            if not nodes:
                print("no active nodes")
            for n in nodes:
                print(f"{n.node_id[:8]}  {n.name}  ({n.platform})  "
                      f"seen {n.age:.0f}s ago")
        return 0
    if verb == "register":
        if len(words) < 2:
            print("usage: nm mesh register <name> [--platform P]", file=sys.stderr)
            return 2
        platform = ""
        if "--platform" in words:
            i = words.index("--platform")
            if i + 1 < len(words):
                platform = words[i + 1]
        node = t.register(words[1], platform=platform)
        print(node.node_id)
        return 0
    if verb == "heartbeat":
        if len(words) < 2:
            print("usage: nm mesh heartbeat <node-id>", file=sys.stderr)
            return 2
        from ...mesh import NodeUnknown
        try:
            t.heartbeat(words[1])
        except NodeUnknown as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("ok")
        return 0
    if verb == "dispatch":
        if len(words) < 2:
            print("usage: nm mesh dispatch <task-type> [--target NODE]",
                  file=sys.stderr)
            return 2
        import socket
        target = None
        payload: dict[str, Any] = {}
        rest = words[2:]
        if "--target" in rest:
            i = rest.index("--target")
            if i + 1 < len(rest):
                target = rest[i + 1]
        if "--json-args" in rest:
            i = rest.index("--json-args")
            if i + 1 < len(rest):
                payload = json.loads(rest[i + 1])
        origin = getattr(context, "device_id", None) or socket.gethostname()
        job_id = t.dispatch(words[1], payload, origin_node=origin,
                            target_node=target)
        print(job_id)
        return 0
    print(f"unknown mesh verb: {verb}", file=sys.stderr)
    return 2
