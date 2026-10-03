"""``nm goal`` — goal surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from ..emit import _emit



def _cmd_goal(args: argparse.Namespace, context: Any) -> int:
    """Goals CLI: create/list/get/advance/tick + the project cascade."""
    from ...agents.goals import GoalSystem

    gs = GoalSystem(context)
    action = getattr(args, "action", "list") or "list"
    arg = getattr(args, "arg", "") or ""
    arg2 = getattr(args, "arg2", "") or ""
    goal_id = getattr(args, "id", "") or (arg if action != "create" else "")
    title = getattr(args, "title", "") or (arg if action == "create" else "")

    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    if action == "create":
        priority = _int(getattr(args, "priority", ""), 0)
        g = gs.create(title, getattr(args, "description", "") or "",
                      priority=priority)
        payload: dict[str, Any] = {"goal": g.to_dict(), "id": g.id}
        lines = [f"created {g.id} — {g.title}"]
        if getattr(args, "project", False):
            sp = gs.spawn_project(g.id)
            payload["project"] = sp
            lines.append(f"project spawned: {sp.get('project_id', '')} "
                         f"(steps={sp.get('steps', 0)})")
        _emit(args, payload, "\n".join(lines))
        return 0
    if action in ("spawn", "spawn_project"):
        sp = gs.spawn_project(goal_id)
        _emit(args, sp, f"project spawned: {sp.get('project_id', '')}")
        return 0
    if action == "list":
        goals = gs.list(status=getattr(args, "filter_status", "") or "",
                        limit=_int(getattr(args, "limit", ""), 20) or 20)
        payload = {"goals": [g.to_dict() for g in goals]}
        lines = [f"{g.id}  [{g.status}]  p{g.priority}  "
                 f"{int(g.progress * 100):3d}%  {g.title}" for g in goals]
        _emit(args, payload, "\n".join(lines) if lines else "no goals yet")
        return 0
    if action == "get":
        g = gs.get(goal_id)
        _emit(args, {"goal": g.to_dict() if g else None},
              json.dumps(g.to_dict(), indent=2, default=str) if g
              else f"goal not found: {goal_id}")
        return 0 if g else 1
    if action == "next":
        g = gs.next_goal()
        if g is None:
            _emit(args, {"goal": None}, "no goal ready")
            return 0
        _emit(args, {"goal": g.to_dict()}, f"next: {g.id} — {g.title}")
        return 0
    if action == "update":
        prio_arg = getattr(args, "priority", "")
        g = gs.update(
            goal_id,
            title=getattr(args, "title", "") or None,
            description=getattr(args, "description", "") or None,
            priority=_int(prio_arg) if prio_arg not in ("", None) else None,
        )
        _emit(args, {"goal": g.to_dict() if g else None},
              f"updated: {goal_id}" if g else f"goal not found: {goal_id}")
        return 0 if g else 1
    if action == "delete":
        ok = gs.delete(goal_id)
        _emit(args, {"deleted": ok, "goal_id": goal_id},
              f"deleted: {goal_id}" if ok else f"goal not found: {goal_id}")
        return 0 if ok else 1
    if action == "replan":
        g = gs.replan(goal_id, reason=getattr(args, "reason", "") or "")
        _emit(args, {"goal": g.to_dict() if g else None},
              f"replanned: {goal_id}" if g
              else f"goal not found: {goal_id}")
        return 0 if g else 1
    if action == "priority":
        g = gs.set_priority(goal_id, _int(arg2))
        _emit(args, {"goal": g.to_dict() if g else None},
              f"priority set: {goal_id} -> {arg2}" if g
              else f"goal not found: {goal_id}")
        return 0 if g else 1
    if action == "depends":
        dep_id = arg2 or getattr(args, "description", "") or ""
        if getattr(args, "remove", False):
            g = gs.remove_dependency(goal_id, dep_id)
            verb = "dependency removed"
        else:
            g = gs.add_dependency(goal_id, dep_id)
            verb = "dependency added"
        _emit(args, {"goal": g.to_dict() if g else None},
              f"{verb}: {goal_id} waits for {dep_id}" if g
              else "goal not found")
        return 0 if g else 1
    if action == "advance":
        g = gs.advance(goal_id)
        _emit(args, {"goal": g.to_dict()}, f"advanced: {g.id} "
              f"({int(g.progress * 100)}%)")
        return 0
    if action == "adapt":
        g = gs.adapt(goal_id, reason=getattr(args, "reason", "") or "")
        _emit(args, {"goal": g.to_dict()}, f"adapted: {g.id}")
        return 0
    if action in ("complete", "pause", "resume"):
        g = getattr(gs, action)(goal_id)
        _emit(args, {"goal": g.to_dict() if g else None},
              f"{action}d: {goal_id}" if g else "goal not found")
        return 0 if g else 1
    if action == "reflect":
        from ...agents.reflection import GoalReflector

        res = GoalReflector(context).reflect(goal_id, force=True)
        if res.get("ok"):
            kg_bits = res.get("kg", res.get("kg_nodes", "ok"))
            _emit(args, res, f"reflection for {goal_id} — kg: {kg_bits}, "
                            f"source: {res.get('source', 'heuristic')}")
        else:
            _emit(args, res, f"could not reflect {goal_id}: "
                  f"{res.get('error', 'unknown')}")
        return 0
    if action == "tick":
        advanced = gs.tick()
        payload = {"advanced": [g.to_dict() for g in advanced]}
        _emit(args, payload,
              f"ticked: {len(advanced)} goal(s) advanced" if advanced
              else "ticked: nothing to advance")
        return 0
    if action == "status":
        st = gs.status()
        _emit(args, st, json.dumps(st, indent=2, default=str))
        return 0
    print(f"goal: unknown action {action}", file=sys.stderr)
    return 2
