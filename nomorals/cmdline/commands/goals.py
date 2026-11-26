"""``nm goals`` — the goal-class board (WS2).

CLI parity for the chat ``/goals`` surface: the user-goal tracker's
goals grouped by domain class, its briefing / weekly review / focus
ranking, and create / re-file with a class.  Rendering lives in
:mod:`nomorals.goals.surface` — the CLI and chat surfaces render
the same text from the same functions.  (``nm goal`` stays the
agent GoalSystem; ``nm idea`` stays idea capture; /goal in chat
stays finance savings goals.)
"""

from __future__ import annotations

import argparse
from typing import Any

from ..emit import _emit


def _cmd_goals(args: argparse.Namespace, context: Any) -> int:
    """Goals CLI: list|briefing|review|focus|progress|add|class|classes|checkin|checkins|milestones."""
    from ...goals import GoalTracker, surface
    from ...goals.classes import GoalClassError

    tracker = GoalTracker(context.db)
    user = getattr(args, "user", "") or "owner"
    action = getattr(args, "action", "list") or "list"
    arg = getattr(args, "arg", "") or ""
    arg2 = getattr(args, "arg2", "") or ""

    if action in ("list", "board"):
        text = surface.render_class_board(
            tracker, user, only_class=arg or None)
        _emit(args, {"board": text}, text)
        return 0
    if action == "briefing":
        text = surface.render_briefing(tracker, user)
        _emit(args, {"briefing": text}, text)
        return 0
    if action == "review":
        text = surface.render_review(tracker, user)
        _emit(args, {"review": text}, text)
        return 0
    if action == "focus":
        text = surface.render_focus(tracker, user)
        _emit(args, {"focus": text}, text)
        return 0
    if action == "progress":
        if not arg:
            _emit(args, {"error": "usage"},
                  "usage: nm goals progress <goal id|#n|title part>")
            return 1
        text = surface.render_goal_progress(tracker, user, arg)
        _emit(args, {"progress": text}, text)
        return 0
    if action == "classes":
        if arg == "add":
            title = getattr(args, "title", "") or ""
            if not arg2 or not title:
                _emit(args, {"error": "usage"},
                      "usage: nm goals classes add <id> --title <title>")
                return 1
            try:
                descriptor = tracker.define_class(arg2, title)
            except GoalClassError as exc:
                _emit(args, {"error": str(exc)}, str(exc))
                return 1
            _emit(args, {"class": descriptor.to_dict()},
                  f"defined goal class {descriptor.title} (`{descriptor.id}`)")
            return 0
        if arg == "remove":
            if not arg2:
                _emit(args, {"error": "usage"},
                      "usage: nm goals classes remove <id>")
                return 1
            try:
                tracker.remove_class(arg2)
            except GoalClassError as exc:
                _emit(args, {"error": str(exc)}, str(exc))
                return 1
            _emit(args, {"removed": arg2}, f"removed goal class `{arg2}`.")
            return 0
        text = surface.render_classes(tracker)
        _emit(args, {"classes": text}, text)
        return 0
    if action == "checkin":
        if not arg:
            _emit(args, {"error": "usage"},
                  "usage: nm goals checkin <goal id|#n|title part> "
                  "[--progress <0-100>] [--note <text>]")
            return 1
        progress = None
        raw_progress = getattr(args, "progress", "") or ""
        if raw_progress != "":
            try:
                progress = float(str(raw_progress).rstrip("%"))
            except ValueError:
                _emit(args, {"error": "usage"},
                      f"progress must be a number 0-100, got {raw_progress!r}")
                return 1
            if not 0.0 <= progress <= 100.0:
                _emit(args, {"error": "usage"},
                      f"progress must be 0-100, got {progress}")
                return 1
        text = surface.check_in_goal(
            tracker, user, arg, progress=progress,
            note=getattr(args, "note", "") or "")
        _emit(args, {"checkin": text}, text)
        return 0
    if action == "checkins":
        text = surface.render_checkins_due(tracker, user)
        _emit(args, {"checkins": text}, text)
        return 0
    if action == "milestones":
        if not arg:
            _emit(args, {"error": "usage"},
                  "usage: nm goals milestones <goal id|#n|title part>")
            return 1
        text = surface.render_milestones(tracker, user, arg)
        _emit(args, {"milestones": text}, text)
        return 0
    if action == "add":
        title = getattr(args, "title", "") or arg
        if not title:
            _emit(args, {"error": "usage"},
                  "usage: nm goals add <title> [--class <class>]")
            return 1
        try:
            goal, class_id, suggested = surface.create_goal_filed(
                tracker, user, title,
                getattr(args, "goal_class", "") or None)
        except GoalClassError as exc:
            _emit(args, {"error": str(exc)}, str(exc))
            return 1
        note = " (suggested from the title)" if suggested else ""
        if class_id == "interests":
            watch = surface.interest_watch_note(tracker, goal)
            if watch:
                note += f"\n  {watch}"
        _emit(args, {"goal": goal.to_dict(), "id": goal.goal_id},
              f"created {goal.goal_id} — {goal.title} "
              f"[class: {class_id}]{note}")
        return 0
    if action == "class":
        if not arg or not arg2:
            _emit(args, {"error": "usage"},
                  "usage: nm goals class <goal> <class>")
            return 1
        try:
            text = surface.refile_goal(tracker, user, arg, arg2)
        except GoalClassError as exc:
            _emit(args, {"error": str(exc)}, str(exc))
            return 1
        _emit(args, {"refiled": text}, text)
        return 0
    _emit(args, {"error": f"unknown action {action!r}"},
          f"unknown goals action: {action}")
    return 1
