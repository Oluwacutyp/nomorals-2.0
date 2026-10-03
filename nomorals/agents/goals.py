"""Long-term goal system — persistent, multi-step objectives that survive
sessions, broken into executable plans, tracked, and adaptively replanned.

A ``Goal`` is a durable objective (e.g. "ship the backup script and verify
it") that outlives any single conversation.  On creation it is decomposed
into an ordered plan of ``GoalStep``s (via the reasoning engine's
``decompose`` strategy when a model is available, else a heuristic).  The
system then works the goal step by step:

  * ``advance(goal_id)`` — execute the next ready step (via the tool
    registry / a devon-style run), record its result, and update progress.
  * ``adapt(goal_id)``   — when a step is blocked or failing, re-plan the
    remaining steps around what has already been learned (self-correcting).
  * ``tick()``           — advance every active goal a little; the hook the
    scheduler / autonomous mode calls for continuous progress.

Everything is stored durably (goals + goal_steps), so a goal that is half
done in one session is exactly where it left off in the next — minimal
supervision, maximum continuity.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["Goal", "GoalStep", "GoalSystem", "register"]

_GOAL_STATUSES = {"active", "paused", "done", "abandoned"}
_STEP_STATUSES = {"pending", "in_progress", "done", "blocked", "skipped"}


@dataclass
class GoalStep:
    id: str
    goal_id: str
    position: int
    description: str
    status: str = "pending"
    result: str = ""
    attempts: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "position": self.position,
                "description": self.description, "status": self.status,
                "result": self.result, "attempts": self.attempts}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "GoalStep":
        return cls(id=row["id"], goal_id=row["goal_id"],
                   position=int(row.get("position", 0)),
                   description=row.get("description", ""),
                   status=row.get("status", "pending"),
                   result=row.get("result", ""),
                   attempts=int(row.get("attempts", 0)),
                   created_at=float(row.get("created_at", 0)),
                   updated_at=float(row.get("updated_at", 0)))


@dataclass
class Goal:
    id: str
    title: str
    description: str = ""
    status: str = "active"
    progress: float = 0.0
    next_action: str = ""
    strategy: str = ""
    project_id: str = ""
    #: mission control (wave 62): higher = worked first by the loop
    priority: int = 0
    #: goal ids this goal cannot start until they are done
    depends_on: list[str] = field(default_factory=list)
    #: how many times the cognitive loop has self-healed this goal
    heals: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float = 0.0
    steps: list[GoalStep] = field(default_factory=list)

    @property
    def done_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "done")

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title,
            "description": self.description, "status": self.status,
            "progress": round(self.progress, 3),
            "next_action": self.next_action, "strategy": self.strategy,
            "project_id": self.project_id, "priority": self.priority,
            "depends_on": list(self.depends_on), "heals": self.heals,
            "done": self.done_steps, "total": self.total_steps,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Goal":
        raw = str(row.get("depends_on", "") or "").strip()
        if raw in ("", "[]"):
            deps: list[str] = []
        else:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                deps = [str(d) for d in parsed if str(d).strip()]
            else:
                deps = [d for d in raw.split(",") if d.strip()]
        return cls(id=row["id"], title=row["title"],
                   description=row.get("description", ""),
                   status=row.get("status", "active"),
                   progress=float(row.get("progress", 0)),
                   next_action=row.get("next_action", ""),
                   strategy=row.get("strategy", ""),
                   project_id=row.get("project_id", "") or "",
                   priority=int(row.get("priority", 0) or 0),
                   depends_on=deps,
                   heals=int(row.get("heals", 0) or 0),
                   created_at=float(row.get("created_at", 0)),
                   updated_at=float(row.get("updated_at", 0)),
                   finished_at=float(row.get("finished_at", 0) or 0.0))


class GoalSystem:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = context.db

    # ── create ──────────────────────────────────────────────────────────────
    def create(self, title: str, description: str = "", *,
               plan: list[str] | None = None, strategy: str = "",
               priority: int = 0) -> Goal:
        """Create a goal and decompose it into an executable plan.

        ``plan`` may be supplied directly; otherwise the reasoning engine's
        ``decompose`` strategy is used when a model is available, with a
        heuristic fallback so a goal can always be created (offline-safe).
        ``priority`` feeds mission control: the cognitive loop works
        higher-priority goals first.
        """
        now = time.time()
        goal_id = new_short_id("goal")
        title = (title or "").strip() or "untitled goal"
        if plan is None:
            # structured reading first (wave 76): the structuring
            # sub-agent's subgoals are a better plan floor than raw
            # decomposition when the description carries structure
            plan = self._structured_plan(title, description)
            if not plan:
                plan = self._decompose(title, description)
        if not plan:
            plan = [f"Clarify what '{title[:60]}' requires",
                    f"Produce the deliverable for {title[:60]}",
                    f"Verify the result against {title[:60]}"]
        self.db.execute(
            "INSERT INTO agent_goals (id, title, description, status, progress, "
            "next_action, strategy, created_at, updated_at, priority) "
            "VALUES (?,?,?,?,0.0,?,?,?,?,?)",
            (goal_id, title, (description or "")[:2000], "active",
             (plan[0] or "")[:300], (strategy or "")[:300], now, now,
             int(priority)))
        for i, step in enumerate(plan[:24]):
            self._add_step(goal_id, i, str(step)[:400])
        goal = self.get(goal_id) or Goal(id=goal_id, title=title)
        # Prompt 05: every goal gets a room (configurable, default ON);
        # room creation never blocks goal creation
        self._maybe_auto_room("goal", goal_id, title, plan)
        return goal

    # ── goal -> project cascade (wave 51) ─────────────────────────────────
    def spawn_project(self, goal_id: str, *, objective: str = "",
                      budget_wall: float = 0.0,
                      budget_tokens: int = 0) -> dict[str, Any]:
        """Turn a goal into an autonomous project that executes it.

        The goal's plan becomes the project's steps; the two are linked
        (goal.project_id <-> project.goal_id) so ``advance`` on the goal
        drives the project, and the project's completion drives the goal.
        A goal that already has a project returns that project (idempotent).
        """
        from .projects import ProjectManager

        goal = self.get(goal_id)
        if goal is None:
            raise ValueError(f"no goal {goal_id!r}")
        if goal.project_id:
            mgr = ProjectManager(self.context)
            proj = mgr._load(goal.project_id)
            if proj is not None:
                return {"project_id": proj.id, "status": proj.status,
                        "steps": len(proj.steps), "existing": True}
        mgr = ProjectManager(self.context)
        # Use the goal's pending steps as the project plan when they exist;
        # otherwise the project plans from the objective itself.
        pending = [s.description for s in goal.steps
                   if s.status in {"pending", "blocked", "in_progress"}]
        objective = (objective or goal.description
                     or f"achieve: {goal.title}")
        project = mgr.create(
            title=f"goal:{goal.title[:80]}", objective=objective,
            steps=pending or None, goal_id=goal.id,
            budget_wall=budget_wall, budget_tokens=budget_tokens)
        self.db.execute(
            "UPDATE agent_goals SET project_id=?, updated_at=? WHERE id=?",
            (project.id, time.time(), goal.id))
        # plan the project (no-op when steps were supplied)
        project = mgr.plan(project.id)
        return {"project_id": project.id, "status": project.status,
                "steps": len(project.steps), "existing": False}

    def _drive_project(self, goal_id: str, *,
                       executor: Any = None) -> Goal:
        """Advance the goal by driving its linked project one step, then
        sync the goal's progress/status from the project."""
        goal = self.get(goal_id)
        if goal is None or not goal.project_id:
            raise ValueError(f"no goal {goal_id!r} with a project")
        from .projects import ProjectManager

        mgr = ProjectManager(self.context)
        project = mgr._load(goal.project_id)
        if project is None:
            # stale link — clear it and fall back to normal stepping
            self.db.execute("UPDATE agent_goals SET project_id='' WHERE id=?",
                            (goal_id,))
            return self.get(goal_id) or goal
        mgr.advance(goal.project_id, executor=executor)
        return self._sync_goal_from_project(goal_id) or self.get(goal_id)

    def _sync_goal_from_project(self, goal_id: str) -> Goal | None:
        """Mirror the linked project's progress/status onto the goal."""
        goal = self.get(goal_id)
        if goal is None or not goal.project_id:
            return goal
        from .projects import ProjectManager

        project = ProjectManager(self.context)._load(goal.project_id)
        if project is None:
            return goal
        now = time.time()
        goal.progress = project.progress
        self.db.execute(
            "UPDATE agent_goals SET progress=?, next_action=?, updated_at=? "
            "WHERE id=?",
            (project.progress, (project.next_step.description[:300]
                                if project.next_step else ""),
             now, goal_id))
        if project.status == "done":
            # mark any remaining pending goal steps done (the project did them)
            for s in self._steps(goal_id):
                if s.status in {"pending", "in_progress", "blocked"}:
                    self._set_step(s.id, status="done",
                                   result="completed via project")
            self.db.execute(
                "UPDATE agent_goals SET status='done', progress=1.0, "
                "finished_at=?, updated_at=? WHERE id=?",
                (now, now, goal_id))
            self._record_completion_knowledge(goal_id)
            self._reflect(goal_id)
        elif project.status == "failed" and not any(
                s.status == "pending" for s in self._steps(goal_id)):
            self.db.execute(
                "UPDATE agent_goals SET status='paused', updated_at=? WHERE id=?",
                (now, goal_id))
        return self.get(goal_id)

    def _structured_plan(self, title: str, description: str) -> list[str]:
        """Wave 76: structure the goal text with the structuring
        sub-agent and use its subgoals as the plan floor.  Falls back to
        [] (the caller then uses _decompose) when there is no
        description to structure — a bare title is not a structurable
        objective, and we don't want to fabricate subgoals from one
        line of text."""
        if not (description or "").strip():
            return []
        try:
            from .structuring import structure_text
            brief = structure_text(self.context, description, for_="goal",
                                   polish=False)
            subgoals = [s for s in (brief.get("subgoals") or []) if s.strip()]
            # a goal whose only "subgoal" is the intent itself is not
            # really structured yet — let the decomposer have a go
            if len(subgoals) <= 1 and subgoals and \
                    subgoals[0].lower() == (brief.get("intent") or "").lower():
                return []
            return [str(s)[:400] for s in subgoals[:24]]
        except Exception:  # noqa: BLE001 - structuring is an enhancement
            return []

    def _decompose(self, title: str, description: str) -> list[str]:
        router = getattr(self.context, "router", None)
        if router is not None:
            try:
                from .reasoning import ReasoningEngine
                engine = ReasoningEngine(self.context, max_llm_calls=4,
                                         max_seconds=60.0)
                prompt = (f"Plan the concrete steps to achieve: {title}\n"
                          f"Context: {description[:400]}")
                # closed loop (wave 65): what the system already LEARNED
                # from past work steers the new plan.
                try:
                    from .skills import SkillLibrary

                    skills_block = SkillLibrary(self.db).context_block(
                        f"{title} {description}"[:240], limit=3)
                    if skills_block:
                        prompt = f"{skills_block}\n\n{prompt}"
                except Exception:  # noqa: BLE001 — skills never fatal
                    pass
                res = engine.reason(prompt, strategy="decompose", depth=1)
                if res.subgoals:
                    return [s[:400] for s in res.subgoals[:12]]
            except Exception:  # noqa: BLE001 — heuristic fallback below
                _log.debug("goal decompose fell back to heuristic")
        return []

    # ── read ────────────────────────────────────────────────────────────────
    def _steps(self, goal_id: str) -> list[GoalStep]:
        rows = self.db.query(
            "SELECT * FROM agent_goal_steps WHERE goal_id=? ORDER BY position",
            (goal_id,))
        return [GoalStep.from_row(r) for r in rows]

    def get(self, goal_id: str) -> Goal | None:
        row = self.db.query_one("SELECT * FROM agent_goals WHERE id=?", (goal_id,))
        if not row:
            return None
        goal = Goal.from_row(row)
        goal.steps = self._steps(goal_id)
        return goal

    def get_by_title(self, title: str) -> Goal | None:
        row = self.db.query_one("SELECT * FROM agent_goals WHERE title=?", (title,))
        if not row:
            return None
        goal = Goal.from_row(row)
        goal.steps = self._steps(goal.id)
        return goal

    def list(self, *, status: str = "", limit: int = 50) -> list[Goal]:
        if status:
            rows = self.db.query(
                "SELECT * FROM agent_goals WHERE status=? ORDER BY updated_at DESC "
                "LIMIT ?", (status, limit))
        else:
            rows = self.db.query(
                "SELECT * FROM agent_goals ORDER BY updated_at DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            g = Goal.from_row(r)
            g.steps = self._steps(g.id)
            out.append(g)
        return out

    # ── mission control (wave 62): priorities + dependencies ──────────────
    def set_priority(self, goal_id: str, priority: int) -> Goal | None:
        """Set a goal's priority; the loop works higher values first."""
        goal = self.get(goal_id)
        if goal is None:
            return None
        self.db.execute(
            "UPDATE agent_goals SET priority=?, updated_at=? WHERE id=?",
            (int(priority), time.time(), goal_id))
        return self.get(goal_id)

    def add_dependency(self, goal_id: str, dep_id: str) -> Goal | None:
        """Make ``goal_id`` wait for ``dep_id`` to finish.

        Refuses self-dependencies and cycles (a goal can never block
        itself, directly or transitively).
        """
        goal = self.get(goal_id)
        dep = self.get(dep_id)
        if goal is None or dep is None:
            return None
        if goal_id == dep_id:
            return goal
        deps = list(goal.depends_on)
        if dep_id in deps:
            return goal
        # cycle guard: walking the dependency graph from dep_id must never
        # reach goal_id
        seen: set[str] = set()
        frontier = [dep_id]
        while frontier:
            current = frontier.pop()
            if current == goal_id:
                return goal  # would create a cycle — leave unchanged
            if current in seen:
                continue
            seen.add(current)
            parent = self.get(current)
            if parent is not None:
                frontier.extend(parent.depends_on)
        deps.append(dep_id)
        self.db.execute(
            "UPDATE agent_goals SET depends_on=?, updated_at=? WHERE id=?",
            (",".join(deps), time.time(), goal_id))
        return self.get(goal_id)

    def remove_dependency(self, goal_id: str, dep_id: str) -> Goal | None:
        goal = self.get(goal_id)
        if goal is None or dep_id not in goal.depends_on:
            return goal
        deps = [d for d in goal.depends_on if d != dep_id]
        self.db.execute(
            "UPDATE agent_goals SET depends_on=?, updated_at=? WHERE id=?",
            (",".join(deps), time.time(), goal_id))
        return self.get(goal_id)

    def dependencies_met(self, goal: Goal) -> bool:
        """True when every goal this one waits on is done (or gone)."""
        for dep_id in goal.depends_on:
            dep = self.get(dep_id)
            if dep is None or dep.status != "done":
                return False
        return True

    def next_goal(self) -> Goal | None:
        """The mission-control pick: the best goal to work RIGHT NOW.

        Active goals with work left and all dependencies met, ordered by
        priority (higher first) then age (older first).
        """
        rows = self.db.query(
            "SELECT * FROM agent_goals WHERE status='active' "
            "ORDER BY priority DESC, created_at ASC")
        best: Goal | None = None
        for r in rows:
            g = Goal.from_row(r)
            g.steps = self._steps(g.id)
            if not any(s.status == "pending" for s in g.steps):
                continue
            if not self.dependencies_met(g):
                continue
            best = g
            break
        return best

    # ── progress ────────────────────────────────────────────────────────────
    # ── cross-goal intelligence (wave 63) ─────────────────────────────────
    def _record_completion_knowledge(self, goal_id: str) -> None:
        """When a goal finishes, write what it produced into the knowledge
        graph: a ``goal`` node (objective + step results) plus ``depends_on``
        edges from every goal that waits on it.  Downstream goals — and the
        KG-aware reasoning pre-flight — can then see what was actually done
        upstream, not just that it happened.  Best-effort, idempotent."""
        try:
            from .kg import KnowledgeGraph

            goal = self.get(goal_id)
            if goal is None or goal.status != "done":
                return
            kg = KnowledgeGraph(self.db)
            results = "; ".join(
                s.result for s in goal.steps if s.result and s.status == "done"
            )[:800]
            kg.upsert_node(goal.title, type="goal", properties={
                "id": goal.id,
                "objective": (goal.description or goal.title)[:500],
                "results": results,
                "steps_done": goal.done_steps,
                "finished_at": goal.finished_at,
            })
            rows = self.db.query("SELECT * FROM agent_goals WHERE status != 'done'")
            for r in rows:
                if goal_id in [d for d in str(r.get("depends_on", "") or "")
                               .split(",") if d.strip()]:
                    kg.link(r["title"], goal.title, "depends_on")
        except Exception as exc:  # noqa: BLE001 — memory, never fatal
            _log.debug("completion knowledge failed: %s", exc)

    def upstream_knowledge(self, goal_id: str) -> str:
        """What the dependencies of this goal ALREADY PRODUCED.

        Injected into project execution so a goal that builds on finished
        work starts from the real artifacts, not from a blank page.
        Empty string when there are no (finished) dependencies.
        """
        goal = self.get(goal_id)
        if goal is None or not goal.depends_on:
            return ""
        blocks: list[str] = []
        for dep_id in goal.depends_on:
            dep = self.get(dep_id)
            if dep is None or dep.status != "done":
                continue
            produced = "; ".join(
                s.result for s in dep.steps
                if s.result and s.status == "done"
            )[:500]
            line = f"{dep.title} (done){': ' + produced if produced else ''}"
            blocks.append(line)
        if not blocks:
            return ""
        return ("Completed upstream goals and what they produced:\n"
                + "\n".join(f"  - {b}" for b in blocks[:5]))

    def _add_step(self, goal_id: str, position: int, description: str) -> None:
        now = time.time()
        self.db.execute(
            "INSERT INTO agent_goal_steps (id, goal_id, position, description, "
            "status, created_at, updated_at) VALUES (?,?,?,?, 'pending',?,?)",
            (new_short_id("gstep"), goal_id, position, description, now, now))

    def _recompute_progress(self, goal: Goal) -> None:
        now = time.time()
        if goal.total_steps:
            progress = goal.done_steps / goal.total_steps
        else:
            progress = 0.0
        pending = [s for s in goal.steps if s.status == "pending"]
        blocked = [s for s in goal.steps if s.status == "blocked"]
        next_action = (pending[0].description if pending
                       else (blocked[0].description if blocked else ""))
        if progress >= 1.0 and not blocked:
            self.db.execute(
                "UPDATE agent_goals SET progress=1.0, status='done', "
                "next_action='', updated_at=?, finished_at=? WHERE id=?",
                (now, now, goal.id))
            goal.status = "done"
            goal.finished_at = now
            self._record_completion_knowledge(goal.id)
            self._reflect(goal.id)
        else:
            self.db.execute(
                "UPDATE agent_goals SET progress=?, next_action=?, updated_at=? "
                "WHERE id=?", (progress, next_action[:300], now, goal.id))
        goal.progress = progress
        goal.next_action = next_action

    def _set_step(self, step_id: str, *, status: str, result: str = "",
                  attempts: int | None = None) -> None:
        now = time.time()
        if attempts is None:
            self.db.execute(
                "UPDATE agent_goal_steps SET status=?, result=?, updated_at=? "
                "WHERE id=?", (status, result[:2000], now, step_id))
        else:
            self.db.execute(
                "UPDATE agent_goal_steps SET status=?, result=?, attempts=?, "
                "updated_at=? WHERE id=?",
                (status, result[:2000], attempts, now, step_id))

    def _room_manager(self) -> Any | None:
        """Lazily build the RoomManager (None when rooms unavailable).

        Cached per instance; a False sentinel avoids retrying a failed
        build.  Goals without rooms take the exact original code path.
        """
        if not hasattr(self, "_rooms_cache"):
            self._rooms_cache: Any = None
        if self._rooms_cache is False:
            return None
        if self._rooms_cache is None:
            try:
                from ..workspace.rooms import RoomManager
                from pathlib import Path
                root = Path(self.context.settings.workspace_dir)
                self._rooms_cache = RoomManager(root=root, db=self.db)
            except Exception:  # noqa: BLE001 — rooms are optional
                self._rooms_cache = False
                return None
        return self._rooms_cache

    def _linked_room(self, goal_id: str) -> Any | None:
        mgr = self._room_manager()
        if mgr is None:
            return None
        try:
            return mgr.get_by_linked("goal", goal_id)
        except Exception:  # noqa: BLE001
            return None

    def _rooms_auto_create(self) -> bool:
        try:
            return bool(getattr(self.context.settings,
                                "rooms_auto_create", True))
        except Exception:  # noqa: BLE001
            return True

    def _maybe_auto_room(self, kind: str, linked_id: str, title: str,
                         plan: list[str] | None = None) -> None:
        """Create+link a room for a new goal/project (configurable, ON)."""
        if not self._rooms_auto_create():
            return
        mgr = self._room_manager()
        if mgr is None:
            return
        try:
            if mgr.get_by_linked(kind, linked_id) is None:
                mgr.create(title, kind=kind, linked_id=linked_id, plan=plan)
        except Exception as exc:  # noqa: BLE001 — room failure never
            _log.warning("auto-room create failed for %s %s: %s",  # blocks creation
                         kind, linked_id, exc)

    def advance(self, goal_id: str, *, executor: Any = None) -> Goal:
        """Execute the next ready step of a goal and record its result.

        ``executor`` is an optional callable ``step_description -> str``
        that performs the step (e.g. a devon run or tool call).  When absent,
        the step is marked ``in_progress`` and its result set to a prompt for
        the caller — the goal still advances and tracks, which is what
        minimal-supervision continuity needs.
        """
        goal = self.get(goal_id)
        if goal is None:
            raise ValueError(f"no goal {goal_id!r}")
        if goal.status in {"done", "abandoned"}:
            return goal
        # Cascade: a goal with a linked project is driven through it.
        if goal.project_id:
            return self._drive_project(goal_id, executor=executor)
        ready = [s for s in goal.steps if s.status == "pending"]
        if not ready:
            self._recompute_progress(goal)
            return self.get(goal_id) or goal
        step = ready[0]
        self._set_step(step.id, status="in_progress")
        room = self._linked_room(goal_id)
        if room is None or executor is None:
            self._run_step_executor(step, executor)
        else:
            # room-linked step: run inside the RoomContext so outputs land
            # in files/, activity is logged, and state checkpoints after
            mgr = self._room_manager()
            assert mgr is not None
            with mgr.enter(room.slug) as ctx:
                ctx.set_step(step.description[:200])
                ctx.log(step.id, "start",
                        {"description": step.description[:500]})
                try:
                    self._run_step_executor(step, executor)
                except Exception:
                    ctx.log(step.id, "error", {"status": "executor raised"})
                    raise
                ctx.log(step.id, "end", {"status": "recorded"})
                ctx.checkpoint(f"goal step done: {step.description[:80]}")
        self._recompute_progress(self.get(goal_id) or goal)
        return self.get(goal_id) or goal

    def _run_step_executor(self, step: GoalStep, executor: Any) -> None:
        """The original step-execution body, unchanged.

        Extracted so the room-linked path wraps it without altering the
        roomless behavior byte-for-byte.
        """
        if executor is not None:
            try:
                result = str(executor(step.description))
                ok = "error" not in result.lower()[:40]
                self._set_step(step.id, status="done" if ok else "blocked",
                               result=result,
                               attempts=step.attempts + 1)
            except Exception as exc:  # noqa: BLE001 — a step failing is data
                self._set_step(step.id, status="blocked",
                               result=f"{type(exc).__name__}: {exc}",
                               attempts=step.attempts + 1)
        else:
            self._set_step(step.id, status="done",
                           result=f"step recorded: {step.description[:200]}")

    # ── adapt (self-correcting) ─────────────────────────────────────────────
    def adapt(self, goal_id: str, *, reason: str = "") -> Goal:
        """Re-plan the remaining (pending) steps around what has happened.

        A blocked/failed step triggers this: the done steps are kept, the
        blocked step is folded into the context, and the remainder is
        re-decomposed.  This is how a goal recovers instead of stalling.
        """
        goal = self.get(goal_id)
        if goal is None:
            raise ValueError(f"no goal {goal_id!r}")
        done = [s for s in goal.steps if s.status == "done"]
        blocked = [s for s in goal.steps if s.status == "blocked"]
        remaining = [s for s in goal.steps if s.status == "pending"]
        if not remaining:
            return goal
        context = (f"Already done: "
                   + "; ".join(s.description[:80] for s in done[-6:])
                   + (f"\nBlocked/failed: "
                      + "; ".join(s.description[:80] for s in blocked[-3:])
                      if blocked else "")
                   + (f"\nReason: {reason[:200]}" if reason else ""))
        new_plan = self._decompose(
            f"Continue: {goal.title} — finish the remaining work",
            goal.description + "\n" + context)
        if not new_plan:
            # no model: unblock the first blocked step and keep going
            for s in blocked:
                self._set_step(s.id, status="pending",
                               result="retried after adaptation")
            return self.get(goal_id) or goal
        now = time.time()
        self.db.execute(
            "DELETE FROM agent_goal_steps WHERE goal_id=? AND status='pending'",
            (goal_id,))
        base = len(done) + len(blocked)
        for i, step in enumerate(new_plan[:16]):
            self._add_step(goal_id, base + i, str(step)[:400])
        self.db.execute(
            "UPDATE agent_goals SET strategy=?, updated_at=? WHERE id=?",
            ((reason or "adapted after "
              + (blocked[0].description[:60] if blocked else "progress"))[:300],
             now, goal_id))
        return self.get(goal_id) or goal

    # ── lifecycle ───────────────────────────────────────────────────────────
    def pause(self, goal_id: str) -> Goal | None:
        self.db.execute("UPDATE agent_goals SET status='paused', updated_at=? "
                        "WHERE id=?", (time.time(), goal_id))
        return self.get(goal_id)

    def resume(self, goal_id: str) -> Goal | None:
        self.db.execute("UPDATE agent_goals SET status='active', updated_at=? "
                        "WHERE id=?", (time.time(), goal_id))
        return self.get(goal_id)

    def abandon(self, goal_id: str) -> Goal | None:
        self.db.execute("UPDATE agent_goals SET status='abandoned', "
                        "updated_at=? WHERE id=?", (time.time(), goal_id))
        return self.get(goal_id)

    def update(self, goal_id: str, *, title: str | None = None,
               description: str | None = None,
               priority: int | None = None) -> Goal | None:
        """Edit a goal's title / description / priority in place.

        Returns None when the goal does not exist.
        """
        goal = self.get(goal_id)
        if goal is None:
            return None
        updates: dict[str, Any] = {}
        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("goal title must not be empty")
            updates["title"] = title[:400]
        if description is not None:
            updates["description"] = description[:2000]
        if priority is not None:
            updates["priority"] = int(priority)
        if updates:
            updates["updated_at"] = time.time()
            set_clause = ", ".join(f"{k}=?" for k in updates)
            self.db.execute(
                f"UPDATE agent_goals SET {set_clause} WHERE id=?",
                (*updates.values(), goal_id))
        return self.get(goal_id)

    def delete(self, goal_id: str) -> bool:
        """Hard-delete a goal, its steps, and dangling dependency references.

        Returns False when the goal does not exist.
        """
        goal = self.get(goal_id)
        if goal is None:
            return False
        self.db.execute("DELETE FROM agent_goal_steps WHERE goal_id=?", (goal_id,))
        self.db.execute("DELETE FROM agent_goals WHERE id=?", (goal_id,))
        # scrub the deleted id out of other goals' dependency lists
        for other in self.list():
            if goal_id in other.depends_on:
                self.remove_dependency(other.id, goal_id)
        return True

    def replan(self, goal_id: str, *, reason: str = "") -> Goal | None:
        """Throw away every unfinished step and re-decompose from scratch.

        Unlike :meth:`adapt` (which surgically replaces the pending tail),
        ``replan`` wipes all non-done steps and rebuilds the full plan from
        the goal's current title/description.  Done steps and their results
        are kept and fed to the planner as context.
        """
        goal = self.get(goal_id)
        if goal is None:
            return None
        done = [s for s in goal.steps if s.status == "done"]
        context = ("Already done: "
                   + "; ".join(s.description[:80] for s in done[-8:])
                   + (f"\nReason for replan: {reason[:200]}" if reason else ""))
        new_plan = self._decompose(goal.title, goal.description + "\n" + context)
        if not new_plan:
            new_plan = [f"Clarify what '{goal.title[:60]}' requires",
                        f"Produce the deliverable for {goal.title[:60]}",
                        f"Verify the result against {goal.title[:60]}"]
        now = time.time()
        self.db.execute(
            "DELETE FROM agent_goal_steps WHERE goal_id=? AND status!='done'",
            (goal_id,))
        base = len(done)
        for i, step in enumerate(new_plan[:24]):
            self._add_step(goal_id, base + i, str(step)[:400])
        self.db.execute(
            "UPDATE agent_goals SET strategy=?, updated_at=? WHERE id=?",
            ((reason or "full replan")[:300], now, goal_id))
        goal = self.get(goal_id)
        if goal is not None:
            self._recompute_progress(goal)
            return self.get(goal_id)
        return None

    def complete(self, goal_id: str) -> Goal | None:
        now = time.time()
        self.db.execute("UPDATE agent_goal_steps SET status='done' WHERE goal_id=? "
                        "AND status IN ('pending','in_progress')", (goal_id,))
        self.db.execute("UPDATE agent_goals SET status='done', progress=1.0, "
                        "next_action='', updated_at=?, finished_at=? "
                        "WHERE id=?", (now, now, goal_id))
        self._record_completion_knowledge(goal_id)
        self._reflect(goal_id)
        return self.get(goal_id)

    # ── reflective checkpointing (wave 64) ─────────────────────────────────
    def _reflect(self, goal_id: str) -> None:
        """Distill what worked / didn't from the finished goal into the
        knowledge graph + skill library.  Best-effort and idempotent —
        reflection is a byproduct of completion, never a failure mode."""
        try:
            auto = getattr(self.context.settings, "autonomy", None)
            if auto is not None and not bool(
                    getattr(auto, "reflect_on_completion", True)):
                return
            from .reflection import GoalReflector

            GoalReflector(self.context).reflect(goal_id)
        except Exception as exc:  # noqa: BLE001
            _log.debug("goal reflection failed: %s", exc)

    # ── continuous ──────────────────────────────────────────────────────────
    def tick(self, *, executor: Any = None,
             max_goals: int = 5) -> list[Goal]:
        """Advance every active goal one step (bounded). The hook the
        scheduler / autonomous project mode calls for continuous progress.
        A goal with a blocked step is auto-adapted so it keeps moving.
        Goals still waiting on unfinished dependencies are left alone
        (mission control, wave 62)."""
        out: list[Goal] = []
        for goal in self.list(status="active", limit=max_goals):
            if not any(s.status == "pending" for s in goal.steps):
                if any(s.status == "blocked" for s in goal.steps):
                    goal = self.adapt(goal.id)
                continue
            if not self.dependencies_met(goal):
                continue
            out.append(self.advance(goal.id, executor=executor))
        return out

    def status(self) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT status, COUNT(*) AS n FROM agent_goals GROUP BY status")
        return {"by_status": {r["status"]: r["n"] for r in rows},
                "active": self.list(status="active", limit=10)}


# ── registry ────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "goal",
        description=(
            "Long-term goals + mission control: persistent multi-step "
            "objectives that survive sessions. action=create | advance | "
            "adapt | status | list | get | pause | resume | abandon | "
            "complete | tick | spawn_project | priority | depends | next | "
            "reflect (distill a finished goal's lessons into KG + skills)."
        ),
        capability="memory.write",
        parameters={
            "action": "str — create|advance|adapt|status|list|get|pause|resume|abandon|complete|tick|spawn_project|priority|depends|next|reflect",
            "title": "str — goal title (create)",
            "description": "str — goal detail (create)",
            "goal_id": "str — for most actions",
            "reason": "str — why to adapt",
            "status": "str — filter for list",
            "limit": "int",
            "priority": "int — new priority (priority action)",
            "depends_on": "str — goal id this goal waits for (depends)",
            "remove": "str — '1' to remove a dependency (depends)",
        },
    )
    def goal(
        action: str, *, title: str = "", description: str = "", goal_id: str = "",
        reason: str = "", status: str = "", limit: str = "20",
        priority: str = "", depends_on: str = "", remove: str = "",
    ) -> dict[str, Any]:
        system = GoalSystem(context)
        action = (action or "list").strip().lower()
        try:
            n = int(limit or 20)
        except ValueError:
            n = 20
        if action == "create":
            g = system.create(title, description)
            return {"ok": True, "goal": g.to_dict()}
        if action == "spawn_project":
            if not goal_id:
                return {"ok": False, "error": "goal_id required"}
            try:
                return {"ok": True, **system.spawn_project(goal_id,
                                                           objective=description)}
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
        if action == "get":
            g = system.get(goal_id)
            return {"ok": g is not None, "goal": g.to_dict() if g else None}
        if action == "list":
            return {"goals": [g.to_dict() for g in
                              system.list(status=status, limit=n)]}
        if action == "advance":
            g = system.advance(goal_id)
            return {"ok": True, "goal": g.to_dict()}
        if action == "adapt":
            g = system.adapt(goal_id, reason=reason)
            return {"ok": True, "goal": g.to_dict()}
        if action == "pause":
            g = system.pause(goal_id)
            return {"ok": g is not None}
        if action == "resume":
            g = system.resume(goal_id)
            return {"ok": g is not None}
        if action == "abandon":
            g = system.abandon(goal_id)
            return {"ok": g is not None}
        if action == "complete":
            g = system.complete(goal_id)
            return {"ok": g is not None, "goal": g.to_dict() if g else None}
        if action == "tick":
            gs = system.tick()
            return {"advanced": [g.to_dict() for g in gs]}
        if action == "priority":
            if not goal_id:
                return {"ok": False, "error": "goal_id required"}
            try:
                p = int(priority)
            except ValueError:
                return {"ok": False, "error": "priority must be an int"}
            g = system.set_priority(goal_id, p)
            return {"ok": g is not None,
                    "goal": g.to_dict() if g else None}
        if action == "depends":
            if not goal_id or not depends_on:
                return {"ok": False,
                        "error": "goal_id and depends_on required"}
            g = system.remove_dependency(goal_id, depends_on) \
                if remove in {"1", "true", "yes"} \
                else system.add_dependency(goal_id, depends_on)
            return {"ok": g is not None,
                    "goal": g.to_dict() if g else None}
        if action == "next":
            g = system.next_goal()
            return {"ok": True, "next": g.to_dict() if g else None}
        if action == "reflect":
            if not goal_id:
                return {"ok": False, "error": "goal_id required"}
            from .reflection import GoalReflector

            return GoalReflector(context).reflect(goal_id, force=True)
        return system.status()


# keep json import referenced (used by callers serializing steps)
_ = json
