"""goal_class — the brain's tool over the user goal-class store.

The user-goal tracker (:mod:`nomorals.goals`) files every goal under a
domain class — careers, productivity, finance, interests, health,
learning, build, relationships, trading, general — plus any custom
class the owner defined as data (see ``GoalTracker.define_class``).
This tool is the brain's handle on that store: create a goal filed
under a class, list the board by class, read one goal's progress
(measured or self-reported, always labelled), check in on a goal,
complete one, list/define/remove classes, and see which classes are
due a check-in under their own cadence.

This is the USER goal store — not the finance savings goals behind
``/goal``, and not the agent ``goal`` tool's multi-step objectives
(``agents/goals.py`` keeps its own ``agent_goals`` table).  Rows live
under the ``owner`` user id, the same id the chat ``/goals`` surface
and the ``nm goals`` CLI use, so all three surfaces read one state.

A goal with no progress says so: ``progress`` returns the class
reading with its basis (``measured`` with the provider's source, or
``self-reported``), and ``checkins`` names classes whose cadence has
lapsed instead of inventing activity.  No database attached means
an honest ``ok: False`` error, never an empty board.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def _tracker(context: Any) -> Any:
    """The GoalTracker over the registry context's db, or None."""
    from ..goals import GoalTracker

    db = getattr(context, "db", None) if context is not None else None
    if db is None:
        return None
    return GoalTracker(db)


def _no_db() -> dict[str, Any]:
    return {"ok": False,
            "error": "no database attached — goals live in the database"}


def _split_list(value: Any) -> list[str]:
    """A tool argument as a string list: list passthrough, else comma-split."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _to_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None


def register(registry: Any) -> None:
    """Attach the ``goal_class`` tool."""
    context = registry.context

    @registry.register(
        "goal_class",
        description=(
            "The owner's goals, filed by class (careers, productivity, "
            "finance, interests, health, learning, build, relationships, "
            "trading, general, plus any custom class defined with "
            "define_class). action=create makes a goal filed under a "
            "class (title + goal_class; empty goal_class files from the "
            "title when it names a class, else general; the class's "
            "starter milestones are seeded as real milestone rows). "
            "action=list|board shows goals grouped by class with "
            "per-class rollups. action=classes lists every class. "
            "action=define_class defines a custom class as data "
            "(class_id slug + title + description + milestone_templates). "
            "action=remove_class removes an unused custom class. "
            "action=progress reports one goal's class reading + pace "
            "(ref = goal id, #n, or title part; progress is always "
            "labelled measured or self-reported). action=checkin records "
            "a check-in on a goal (progress optional, note optional). "
            "action=checkins lists classes whose check-in cadence has "
            "lapsed. action=milestones lists one goal's milestones. "
            "action=complete completes a goal. action=class re-files a "
            "goal under another class."
        ),
        capability="memory.write",
        parameters={
            "action": "str — create|list|board|classes|define_class|remove_class|progress|checkin|checkins|milestones|complete|class",
            "title": "str — goal title (create) or class title (define_class)",
            "goal_class": "str — class id to file under (create/class)",
            "ref": "str — goal id, #n, or title part (progress/checkin/milestones/complete/class)",
            "progress": "str — progress percent 0-100 (checkin)",
            "note": "str — check-in note (checkin)",
            "class_id": "str — class slug (define_class/remove_class)",
            "description": "str — class description (define_class)",
            "milestone_templates": "str — comma-separated starter milestones (define_class)",
            "limit": "int — max goals listed (list, default 20)",
        },
    )
    def goal_class(
        action: str = "board",
        *,
        title: str = "",
        goal_class: str = "",
        ref: str = "",
        progress: str = "",
        note: str = "",
        class_id: str = "",
        description: str = "",
        milestone_templates: str = "",
        limit: str = "20",
    ) -> dict[str, Any]:
        from ..goals import surface
        from ..goals.classes import GoalClassError

        tracker = _tracker(context)
        if tracker is None:
            return _no_db()
        user = surface.OWNER_USER_ID
        verb = (action or "board").strip().lower()
        try:
            n = int(limit or 20)
        except ValueError:
            n = 20

        if verb == "create":
            if not (title or "").strip():
                return {"ok": False, "error": "create needs a title"}
            try:
                goal, filed, suggested = surface.create_goal_filed(
                    tracker, user, title, goal_class or None)
            except GoalClassError as exc:
                return {"ok": False, "error": str(exc)}
            milestones = surface.run_sync(
                tracker.list_milestones(goal.goal_id))
            return {"ok": True, "action": "create",
                    "goal": goal.to_dict(), "goal_class": filed,
                    "suggested": suggested,
                    "milestones": [m.title for m in milestones]}

        if verb in ("list", "board"):
            board = surface.render_class_board(tracker, user)
            stats = surface.run_sync(tracker.stats(user))
            goals = surface.run_sync(
                tracker.list_goals(user, status="active", limit=max(1, n)))
            return {"ok": True, "action": "board", "board": board,
                    "stats": stats,
                    "goals": [g.to_dict() for g in goals]}

        if verb == "classes":
            return {"ok": True, "action": "classes",
                    "classes": [c.to_dict()
                                for c in tracker.list_all_classes()]}

        if verb == "define_class":
            if not (class_id or "").strip():
                return {"ok": False, "error": "define_class needs a class_id slug"}
            if not (title or "").strip():
                return {"ok": False, "error": "define_class needs a title"}
            try:
                descriptor = tracker.define_class(
                    class_id, title, description,
                    milestone_templates=_split_list(milestone_templates))
            except GoalClassError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "action": "define_class",
                    "class": descriptor.to_dict()}

        if verb == "remove_class":
            if not (class_id or "").strip():
                return {"ok": False, "error": "remove_class needs a class_id"}
            try:
                tracker.remove_class(class_id)
            except GoalClassError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "action": "remove_class", "class_id": class_id}

        if verb == "progress":
            if not (ref or "").strip():
                return {"ok": False, "error": "progress needs a goal ref"}
            goal, _candidates = surface.resolve_goal(tracker, user, ref)
            if goal is None:
                return {"ok": False,
                        "error": surface.render_goal_progress(tracker, user, ref)}
            reading = surface.run_sync(
                tracker.measured_progress(goal.goal_id))
            return {"ok": True, "action": "progress",
                    "goal": goal.to_dict(),
                    "reading": reading.to_dict(),
                    "text": surface.render_goal_progress(tracker, user, ref)}

        if verb == "checkin":
            if not (ref or "").strip():
                return {"ok": False, "error": "checkin needs a goal ref"}
            goal, _candidates = surface.resolve_goal(tracker, user, ref)
            if goal is None:
                return {"ok": False,
                        "error": surface.check_in_goal(tracker, user, ref)}
            return {"ok": True, "action": "checkin",
                    "text": surface.check_in_goal(
                        tracker, user, ref,
                        progress=_to_float(progress), note=note)}

        if verb == "checkins":
            due = surface.run_sync(tracker.class_checkins_due(user))
            return {"ok": True, "action": "checkins", "due": due,
                    "text": surface.render_checkins_due(tracker, user)}

        if verb == "milestones":
            if not (ref or "").strip():
                return {"ok": False, "error": "milestones needs a goal ref"}
            goal, _candidates = surface.resolve_goal(tracker, user, ref)
            if goal is None:
                return {"ok": False,
                        "error": surface.render_milestones(tracker, user, ref)}
            rows = surface.run_sync(tracker.list_milestones(goal.goal_id))
            return {"ok": True, "action": "milestones",
                    "goal": goal.to_dict(),
                    "milestones": [m.to_dict() for m in rows],
                    "text": surface.render_milestones(tracker, user, ref)}

        if verb == "complete":
            if not (ref or "").strip():
                return {"ok": False, "error": "complete needs a goal ref"}
            goal, _candidates = surface.resolve_goal(tracker, user, ref)
            if goal is None:
                return {"ok": False,
                        "error": f"no single goal matches {ref!r}"}
            surface.run_sync(tracker.complete(goal.goal_id))
            done = surface.run_sync(tracker.get(goal.goal_id))
            return {"ok": True, "action": "complete",
                    "goal": (done.to_dict() if done is not None
                             else goal.to_dict())}

        if verb == "class":
            if not (ref or "").strip() or not (goal_class or "").strip():
                return {"ok": False,
                        "error": "class needs a goal ref and a goal_class"}
            try:
                text = surface.refile_goal(tracker, user, ref, goal_class)
            except GoalClassError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "action": "class", "text": text}

        return {"ok": False, "error": f"unknown goal_class action {verb!r}"}
