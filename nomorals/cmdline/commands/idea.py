"""``nm idea`` — idea capture, dismissal and promotion to goals."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import sys
from typing import Any
from ..emit import _emit


def _drive(coro: Any) -> Any:
    """Run a coroutine from sync command code.

    Works both from plain synchronous dispatch and from inside an already
    running event loop (in which case the coroutine is driven on a
    throwaway thread with its own loop).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _cmd_idea(args: argparse.Namespace, context: Any) -> int:
    """Ideas CLI: create/list/get/update/dismiss/promote/delete/search/stats."""
    from ...goals import GoalTracker, IdeaTracker

    db = context.db
    ideas = IdeaTracker(db)
    goals = GoalTracker(db)
    user = getattr(args, "user", "") or "owner"
    action = getattr(args, "action", "list") or "list"
    idea_id = getattr(args, "id", "") or getattr(args, "arg", "") or ""
    title = getattr(args, "title", "") or ""

    def _tags() -> list[str]:
        raw = getattr(args, "tags", "") or ""
        return [t.strip() for t in str(raw).split(",") if t.strip()]

    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    async def _run() -> int:
        if action == "create":
            idea = await ideas.create(
                user, title or getattr(args, "arg", "") or "untitled idea",
                description=getattr(args, "description", "") or "",
                tags=_tags(),
                source=getattr(args, "source", "") or "")
            _emit(args, {"idea": idea.to_dict()},
                  f"created {idea.idea_id} — {idea.title}")
            return 0
        if action == "list":
            rows = await ideas.list_ideas(
                user,
                status=getattr(args, "filter_status", "") or None,
                tags=_tags() or None,
                limit=_int(getattr(args, "limit", ""), 20) or 20)
            payload = {"ideas": [i.to_dict() for i in rows]}
            lines = [f"{i.idea_id}  [{i.status}]  {i.title}" for i in rows]
            _emit(args, payload, "\n".join(lines) if lines else "no ideas yet")
            return 0
        if action == "get":
            idea = await ideas.get(idea_id)
            _emit(args, {"idea": idea.to_dict() if idea else None},
                  json.dumps(idea.to_dict(), indent=2, default=str) if idea
                  else f"idea not found: {idea_id}")
            return 0 if idea else 1
        if action == "update":
            try:
                idea = await ideas.update(
                    idea_id,
                    title=getattr(args, "title", "") or None,
                    description=getattr(args, "description", "") or None,
                    tags=_tags() or None,
                    source=getattr(args, "source", "") or None)
            except KeyError:
                idea = None
            _emit(args, {"idea": idea.to_dict() if idea else None},
                  f"updated: {idea_id}" if idea else f"idea not found: {idea_id}")
            return 0 if idea else 1
        if action == "dismiss":
            ok = await ideas.dismiss(idea_id,
                                     reason=getattr(args, "reason", "") or "")
            _emit(args, {"dismissed": ok, "idea_id": idea_id},
                  f"dismissed: {idea_id}" if ok
                  else f"idea not found: {idea_id}")
            return 0 if ok else 1
        if action == "promote":
            goal_id = await ideas.promote_to_goal(idea_id, goals)
            _emit(args, {"idea_id": idea_id, "goal_id": goal_id},
                  f"promoted {idea_id} -> goal {goal_id}" if goal_id
                  else f"idea not found: {idea_id}")
            return 0 if goal_id else 1
        if action == "delete":
            ok = await ideas.delete(idea_id)
            _emit(args, {"deleted": ok, "idea_id": idea_id},
                  f"deleted: {idea_id}" if ok
                  else f"idea not found: {idea_id}")
            return 0 if ok else 1
        if action == "search":
            query = getattr(args, "arg", "") or ""
            rows = await ideas.search(user, query,
                                      limit=_int(getattr(args, "limit", ""), 20) or 20)
            payload = {"ideas": [i.to_dict() for i in rows]}
            lines = [f"{i.idea_id}  [{i.status}]  {i.title}" for i in rows]
            _emit(args, payload,
                  "\n".join(lines) if lines else f"no ideas match {query!r}")
            return 0
        if action == "stats":
            rows = await ideas.list_ideas(user)
            by_status: dict[str, int] = {}
            for i in rows:
                by_status[i.status] = by_status.get(i.status, 0) + 1
            payload = {"total": len(rows), "by_status": by_status}
            lines = [f"ideas: {len(rows)}"] + [
                f"  {s}: {n}" for s, n in sorted(by_status.items())]
            _emit(args, payload, "\n".join(lines))
            return 0
        print(f"idea: unknown action {action}", file=sys.stderr)
        return 2

    return _drive(_run())
