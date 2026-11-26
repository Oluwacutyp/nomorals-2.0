"""``nm fleet`` — parallel coding-agent fleets from the terminal.

Create a fleet over a repo, dispatch coding briefs (each runs
isolated in its own git worktree), watch the Kanban board, review
a task's diff, approve to merge it into the repo. See
nomorals/agents/fleet.py.

``nm fleet create --repo DIR [--name NAME] [--base REV]``
``nm fleet dispatch <fleet> "<brief>" [--key KEY] [--wait SECONDS]``
``nm fleet board <fleet>``
``nm fleet list``
``nm fleet review <task>``
``nm fleet approve <task>``
``nm fleet cleanup <fleet>``

Dispatch returns at once unless ``--wait`` is given; a dispatched
task runs on a background thread inside this process, so ``--wait``
is the way to run one to completion from a single command. The
fleet home is ``<home>/fleet``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _store(context: Any):
    from ...agents.fleet import FleetStore

    settings = getattr(context, "settings", None)
    home = getattr(settings, "home", None) if settings is not None else None
    if not home:
        raise ValueError("fleet needs a home directory (no settings.home)")
    return FleetStore(Path(str(home)) / "fleet", context=context)


def _emit(args: argparse.Namespace, payload: dict[str, Any],
          text: str) -> int:
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)
    return 0


def _render_board(board: Any) -> str:
    from ...agents.fleet import render_board

    return render_board(board)


def _cmd_fleet(args: argparse.Namespace, context: Any) -> int:
    """Route `nm fleet` to create / dispatch / board / list /
    review / approve / cleanup."""
    from ...agents.fleet import FleetError

    action = args.fleet_action
    try:
        store = _store(context)
    except ValueError as exc:
        print(f"fleet: {exc}", file=sys.stderr)
        return 2
    try:
        if action == "create":
            record = store.create_fleet(
                args.name or "fleet", args.repo, base=args.base or None)
            return _emit(args, record,
                         f"fleet {record['name']} ({record['fleet_id']})\n"
                         f"repo {record['repo']} @ "
                         f"{record['base_revision'][:12]}")
        if action == "dispatch":
            record = store.dispatch(
                args.fleet, " ".join(args.brief),
                key=args.key or None, run_sync=bool(args.wait))
            if args.wait:
                record = store.wait(record["task_id"],
                                    timeout=float(args.wait))
            state = record.get("state", "")
            text = (f"task {record['task_id']} [{state}] "
                    f"{record.get('key', '')}")
            if record.get("error"):
                text += f"\n! {record['error']}"
            return _emit(args, record, text)
        if action == "board":
            board = store.board(args.fleet)
            payload = board.to_dict()
            payload["text"] = _render_board(board)
            return _emit(args, payload, payload["text"])
        if action == "list":
            fleets = store.list_fleets()
            text = "\n".join(
                f"{f['fleet_id'][:12]}  {f['name']}  {f['repo']}  "
                f"({f['tasks']} tasks)" for f in fleets) or "(no fleets)"
            return _emit(args, {"fleets": fleets}, text)
        if action == "review":
            payload = store.review(args.task)
            task = payload["task"]
            diff = payload.get("diff") or ""
            text = (f"task {task['task_id']} [{task['state']}] "
                    f"{task.get('key', '')}\n")
            if payload.get("stat"):
                text += payload["stat"] + "\n"
            text += diff if diff.strip() else (
                payload.get("note") or "(no changes)")
            return _emit(args, payload, text)
        if action == "approve":
            record = store.approve(args.task)
            files = (record.get("result") or {}).get("merged_files") or []
            text = (f"merged task {record['task_id']} "
                    f"({len(files)} files: {', '.join(files)})")
            return _emit(args, record, text)
        if action == "cleanup":
            payload = store.cleanup(args.fleet)
            return _emit(
                args, payload,
                f"fleet {payload['fleet_id']}: removed "
                f"{payload['removed']} worktrees, kept {payload['kept']}")
        print(f"unknown fleet action: {action}", file=sys.stderr)
        return 2
    except FleetError as exc:
        print(f"fleet: {exc}", file=sys.stderr)
        return 1
