"""``nm room`` — workspace rooms."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path



def _room_obj(context: Any) -> Any:
    """Build the RoomManager for the CLI context's workspace."""
    from ...workspace.rooms import RoomManager

    root = Path(context.settings.workspace_dir)
    return RoomManager(root, db=context.db)


def _cmd_room(args: argparse.Namespace, context: Any) -> int:
    """Route `nm room` to new / list / enter / status / archive / pause /
    resume / link / search / tick / stale."""
    as_json = getattr(args, "json", False)
    try:
        mgr = _room_obj(context)
    except Exception as exc:  # noqa: BLE001
        print(f"rooms unavailable: {exc}", file=sys.stderr)
        return 1
    action = args.room_action
    try:
        if action == "new":
            room = mgr.create(args.title, kind=args.kind,
                              linked_id=args.linked or "")
            payload = room.to_dict()
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                print(f"room {room.slug} ({room.kind}) — "
                      f"{mgr.rooms_dir / room.slug}")
            return 0
        if action == "list":
            rooms = mgr.list(status=args.status or "")
            if as_json:
                print(json.dumps([r.to_dict() for r in rooms], indent=2,
                                 default=str))
            else:
                for r in rooms:
                    print(f"{r.slug:28} {r.status:8} {r.kind:8} "
                          f"{r.current_step[:50] or '(no step)'}")
            return 0
        if action == "enter":
            with mgr.enter(args.slug) as ctx:
                md = (mgr.rooms_dir / args.slug / "ROOM.md").read_text(
                    encoding="utf-8")
            if as_json:
                print(json.dumps({"slug": args.slug, "room_md": md},
                                 indent=2))
            else:
                print(md)
                print(f"--- room {args.slug} entered; work in this session "
                      f"is scoped to {mgr.rooms_dir / args.slug} ---")
            return 0
        if action == "status":
            room = mgr.get(args.slug)
            if room is None:
                print(f"no room {args.slug!r}", file=sys.stderr)
                return 1
            payload = room.to_dict()
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                print(f"{room.slug} [{room.status}] {room.title}")
                print(f"kind: {room.kind}  linked: "
                      f"{room.linked_id or 'none'}")
                print(f"current step: {room.current_step or '(none)'}")
                print(f"blockers: {', '.join(room.blockers) or '(none)'}")
                for d in room.decisions[-5:]:
                    print(f"  - [{d.get('at', '?')}] {d.get('decision', '')}")
            return 0
        if action in ("archive", "pause", "resume"):
            room = {"archive": mgr.archive, "pause": mgr.pause,
                    "resume": mgr.resume}[action](args.slug)
            if as_json:
                print(json.dumps(room.to_dict(), indent=2, default=str))
            else:
                print(f"room {room.slug} → {room.status}")
            return 0
        if action == "link":
            result = mgr.link(args.slug_a, args.slug_b)
            if as_json:
                print(json.dumps(result, indent=2))
            else:
                print(f"linked {args.slug_a} <-> {args.slug_b}")
            return 0
        if action == "search":
            hits = mgr.search(args.query, deep=args.deep)
            if as_json:
                print(json.dumps(hits, indent=2, default=str))
            else:
                for h in hits:
                    print(f"{h['slug']:28} [{h['where']}] {h['title'][:60]}")
                if not hits:
                    print("(no matches)")
            return 0
        if action == "tick":
            from ...agents.goals import GoalSystem
            from ...agents.projects import ProjectManager
            result = mgr.tick(goal_system=GoalSystem(context),
                              project_manager=ProjectManager(context))
            if as_json:
                print(json.dumps(result, indent=2, default=str))
            else:
                for r in result["advanced"]:
                    print(f"{r['slug']}: advanced {r['steps']} step(s)")
                for r in result["skipped"]:
                    print(f"{r['slug']}: skipped ({r['reason']})")
                for r in result["reconciled"]:
                    print(f"{r['slug']}: reconciled dirty state")
                for r in result["errors"]:
                    print(f"{r['slug']}: ERROR {r['error']}")
            return 0
        if action == "stale":
            stale = mgr.stale_rooms(days=args.days)
            if as_json:
                print(json.dumps([r.to_dict() for r in stale], indent=2,
                                 default=str))
            else:
                for r in stale:
                    print(f"{r.slug:28} idle — archive?")
                if not stale:
                    print("(no stale rooms)")
            return 0
    except (KeyError, ValueError) as exc:
        print(f"room error: {exc}", file=sys.stderr)
        return 1
    print(f"unknown room action: {action}", file=sys.stderr)
    return 2
