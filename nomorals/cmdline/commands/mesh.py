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


def _flag(words: list[str], name: str) -> str | None:
    """Value of ``--name value`` in a word list, or None."""
    if name in words:
        i = words.index(name)
        if i + 1 < len(words):
            return words[i + 1]
    return None


def _parse_json_arg(raw: str, what: str) -> Any:
    """Parse a JSON CLI arg; raises ValueError with a clean message."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON for {what}: {exc}") from exc


def _cmd_mesh(args: Any, context: Any) -> int:
    """Route ``nm mesh <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm mesh nodes [--json]\n"
              "       nm mesh register <name> [--platform P]\n"
              "       nm mesh heartbeat <node-id>\n"
              "       nm mesh dispatch <task-type> [--target NODE] [--json-args '{}']\n"
              "       nm mesh poll <node-id> [--batch N] [--json]\n"
              "       nm mesh complete <job-id> [--result JSON]\n"
              "       nm mesh fail <job-id> [--error MSG] [--no-retry]\n"
              "       nm mesh pending [--node ID] [--json]\n"
              "       nm mesh prune [--stale-after SEC]",
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
        node = t.register(words[1], platform=_flag(words, "--platform") or "")
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
        rest = words[2:]
        target = _flag(rest, "--target")
        payload: dict[str, Any] = {}
        raw_args = _flag(rest, "--json-args")
        if raw_args is not None:
            try:
                payload = _parse_json_arg(raw_args, "--json-args")
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 2
            if not isinstance(payload, dict):
                print("--json-args must be a JSON object", file=sys.stderr)
                return 2
        origin = getattr(context, "device_id", None) or socket.gethostname()
        job_id = t.dispatch(words[1], payload, origin_node=origin,
                            target_node=target)
        print(job_id)
        return 0
    if verb == "poll":
        # Claim tasks for a node: its own topic first, then broadcast
        # (first poller wins — work-stealing).
        if len(words) < 2:
            print("usage: nm mesh poll <node-id> [--batch N]", file=sys.stderr)
            return 2
        try:
            batch = int(_flag(words, "--batch") or 5)
        except ValueError:
            print("--batch must be an integer", file=sys.stderr)
            return 2
        if batch < 1:
            print("--batch must be >= 1", file=sys.stderr)
            return 2
        try:
            tasks = t.poll(words[1], batch=batch)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if as_json:
            print(json.dumps([x.to_dict() for x in tasks], indent=2,
                             default=str))
        else:
            if not tasks:
                print("no tasks")
            for x in tasks:
                where = x.target_node or "broadcast"
                print(f"{x.job_id}  {x.task_type}  -> {where}  "
                      f"(from {x.origin_node}, attempt {x.attempts})")
        return 0
    if verb == "complete":
        if len(words) < 2:
            print("usage: nm mesh complete <job-id> [--result JSON]",
                  file=sys.stderr)
            return 2
        result = None
        raw = _flag(words, "--result")
        if raw is not None:
            try:
                result = _parse_json_arg(raw, "--result")
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 2
        t.complete(words[1], result=result)
        print("ok")
        return 0
    if verb == "fail":
        if len(words) < 2:
            print("usage: nm mesh fail <job-id> [--error MSG] [--no-retry]",
                  file=sys.stderr)
            return 2
        t.fail(words[1], error=_flag(words, "--error") or "",
               retry="--no-retry" not in words)
        print("ok")
        return 0
    if verb == "pending":
        tasks = getattr(t, "tasks", None)
        if tasks is None:
            print("pending: not supported by this transport", file=sys.stderr)
            return 1
        node = _flag(words, "--node")
        count = tasks.pending_count(node)
        if as_json:
            print(json.dumps({"node": node, "pending": count}))
        else:
            scope = f"node {node}" if node else "all mesh topics"
            print(f"{count} pending task(s) for {scope}")
        return 0
    if verb == "prune":
        nodes = getattr(t, "nodes", None)
        if nodes is None:
            print("prune: not supported by this transport", file=sys.stderr)
            return 1
        try:
            stale_after = float(_flag(words, "--stale-after") or 1200.0)
        except ValueError:
            print("--stale-after must be a number (seconds)", file=sys.stderr)
            return 2
        removed = nodes.prune(stale_after=stale_after)
        print(f"pruned {removed} node(s)")
        return 0
    print(f"unknown mesh verb: {verb}", file=sys.stderr)
    return 2
