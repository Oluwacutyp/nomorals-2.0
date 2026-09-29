"""Mission control (wave 62) — the portfolio view over the goal system.

Where the goal system tracks individual objectives, mission control looks
at the *whole portfolio*: what is ready to work (dependencies met, steps
left), what is blocked (on which goals), what is being executed (linked
projects), and what the next heartbeat will pick.

``MissionControl.plan()`` is the single source of truth the owner (and the
cognitive loop) uses to see the state of the entire autonomous stack in one
glance — priorities, progress, dependencies, project linkage, and readiness.

Modular and callable by the main AI and sub-agents via the ``mission`` tool.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.logging_setup import get_logger
from .goals import Goal, GoalSystem

_log = get_logger(__name__)

__all__ = ["MissionControl", "register"]

#: portfolio risk scoring (wave 64): priority sets the BASE value
#: (0.45 with no priority signal .. 1.0 at the portfolio's top priority);
#: dependency depth, heal history, and remaining size act as risk
#: MULTIPLIERS, so a blocked, failing, or sprawling goal loses expected
#: value even at high priority.  EV is clamped to 0..100.
_EV_BASE_MIN = 0.45
_EV_BASE_SPREAD = 0.55

#: cost-aware EV (wave 66): each PENDING step has an estimated model-call
#: cost — a build step runs the coding agent (draft -> run -> fix, up to 3
#: iterations) while a narration step is a single model call.  The
#: cost factor damps a goal's expected value in proportion to the calls
#: its remaining work will spend, so a 12-step build ranks below a
#: 2-step one when the daily budget is tight.
_CALLS_PER_BUILD_STEP = 3
_CALLS_PER_NARRATION_STEP = 1
_EV_COST_WEIGHT = 0.15
_COST_RISK_THRESHOLD = 12


def _project_state(context: Any, project_id: str) -> dict[str, Any]:
    """Lightweight status of a linked project (best-effort)."""
    if not project_id:
        return {}
    try:
        from .projects import ProjectManager

        mgr = ProjectManager(context)
        st = mgr.status(project_id)
        if not st.get("ok"):
            return {"project_id": project_id, "status": "missing"}
        return {"project_id": project_id, "status": st.get("status", ""),
                "progress": st.get("progress", 0.0),
                "steps_done": st.get("done", 0),
                "steps_total": st.get("total", 0)}
    except Exception as exc:  # noqa: BLE001 — view, never fatal
        _log.debug("mission: project status failed: %s", exc)
        return {"project_id": project_id, "status": "unknown"}


class MissionControl:
    """The portfolio: priorities, dependencies, readiness, next pick."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.settings = context.settings
        self.goals = GoalSystem(context)

    # ── the plan ───────────────────────────────────────────────────────────
    def plan(self, *, limit: int = 50) -> dict[str, Any]:
        """The full portfolio, partitioned by readiness.

        * ``next``      — what the cognitive loop will work first
        * ``ready``     — active, unblocked, work remaining (priority order)
        * ``blocked``   — active but waiting on unfinished dependencies
        * ``paused``    — paused goals (incl. ones whose project failed and
                          the loop may self-heal)
        * ``done``      — recently finished (progress 1.0)
        * ``projects``  — live projects with their linked goals
        """
        gs = self.goals
        ready: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        paused: list[dict[str, Any]] = []
        done: list[dict[str, Any]] = []
        ranked_goals: list[Goal] = []

        rows = self.context.db.query(
            "SELECT * FROM agent_goals ORDER BY priority DESC, created_at ASC "
            "LIMIT ?", (limit,))
        for r in rows:
            g = Goal.from_row(r)
            g.steps = gs._steps(g.id)
            if g.status in {"active", "paused"}:
                ranked_goals.append(g)
            if g.status == "done":
                done.append(self._row(g))
                continue
            if g.status == "paused":
                paused.append(self._row(g))
                continue
            if g.status != "active":
                continue
            has_work = any(s.status == "pending" for s in g.steps)
            if has_work and gs.dependencies_met(g):
                ready.append(self._row(g))
            elif has_work:
                entry = self._row(g)
                unmet = [d for d in g.depends_on
                         if (dep := gs.get(d)) is None or dep.status != "done"]
                entry["waiting_on"] = unmet
                blocked.append(entry)

        next_goal = gs.next_goal()
        projects: list[dict[str, Any]] = []
        try:
            from .projects import ProjectManager

            mgr = ProjectManager(self.context)
            for p in mgr.list_projects():
                if p.get("status") in {"done", "failed"}:
                    continue
                projects.append({
                    "project_id": p.get("id"), "title": p.get("title", ""),
                    "status": p.get("status"), "progress": p.get("progress"),
                    "goal_id": p.get("goal_id", ""),
                })
        except Exception as exc:  # noqa: BLE001
            _log.debug("mission: project list failed: %s", exc)

        cadence = {"base_hours": float(
            getattr(self.settings.autonomy, "interval_hours", 6.0) or 6.0),
                   "adaptive": bool(
                       getattr(self.settings.autonomy, "adaptive_cadence",
                               True)),
                   "effective_hours": None}
        try:
            from .cognition import CognitiveLoop

            cadence["effective_hours"] = round(
                CognitiveLoop(self.context).effective_interval(), 3)
        except Exception:  # noqa: BLE001
            pass
        # portfolio risk scoring (wave 64): the same goals ranked by
        # expected value — priority tempered by dependency depth, heal
        # history, and remaining size
        ranking: list[dict[str, Any]] = []
        if ranked_goals:
            max_priority = max((g.priority for g in ranked_goals), default=0)
            depth_cache: dict[str, int] = {}
            ranking = [self._score_goal(g, max_priority=max_priority,
                                        depth_cache=depth_cache)
                       for g in ranked_goals]
            ranking.sort(key=lambda r: (-r["expected_value"],
                                        -r["priority"], r["id"]))
        return {
            "generated_at": time.time(),
            "cadence": cadence,
            "next": next_goal.to_dict() if next_goal else None,
            "ready": ready,
            "blocked": blocked,
            "paused": paused,
            "done": done[-10:],
            "projects": projects,
            "ranking": ranking,
            "counts": {
                "ready": len(ready), "blocked": len(blocked),
                "paused": len(paused), "done": len(done),
                "live_projects": len(projects),
            },
        }

    def next(self) -> dict[str, Any]:
        """Just the pick: what to work on right now (or None)."""
        g = self.goals.next_goal()
        return {"ok": True, "next": g.to_dict() if g else None}

    # ── portfolio risk scoring (wave 64) ───────────────────────────────────
    def _unmet_depth(self, g: Goal, cache: dict[str, int],
                     seen: set[str]) -> int:
        """How deep this goal's chain of UNFINISHED dependencies goes
        (0 = ready to work).  Missing deps count one hop — they can never
        resolve.  Cycles are refused at add_dependency time; the seen-set
        is a backstop."""
        if g.id in cache:
            return cache[g.id]
        if g.id in seen:
            cache[g.id] = 0
            return 0
        seen.add(g.id)
        depth = 0
        for dep_id in g.depends_on:
            dep = self.goals.get(dep_id)
            if dep is None or dep.status != "done":
                child = (self._unmet_depth(dep, cache, set(seen))
                         if dep is not None else 0)
                depth = max(depth, 1 + child)
        cache[g.id] = depth
        return depth

    @staticmethod
    def _priority_norm(priority: int, max_priority: int) -> float:
        if max_priority > 0:
            return max(0.0, min(1.0, priority / max_priority))
        return 0.5  # no positive priorities in the portfolio: no signal

    def _step_cost(self, g: Goal) -> int:
        """Model calls one step of this goal's linked project costs
        (wave 66): build 3, narration 1, projectless goals 0 (a
        projectless goal's steps are bookkeeping — the loop doesn't
        spend model calls on them)."""
        if not g.project_id:
            return 0
        per = _CALLS_PER_NARRATION_STEP
        try:
            row = self.context.db.query_one(
                "SELECT task_kind FROM projects WHERE id=?", (g.project_id,))
            if row and (row.get("task_kind") or "") == "build":
                per = _CALLS_PER_BUILD_STEP
        except Exception:  # noqa: BLE001 — estimate, never fatal
            pass
        return per

    def _est_calls(self, g: Goal, pending: int) -> int:
        """Estimated model calls the goal's remaining work will spend
        (wave 66): pending steps x the per-step cost of its project."""
        if pending <= 0:
            return 0
        return pending * self._step_cost(g)

    def _budget_fit(self, est_calls: int, pending: int,
                    per_step: int) -> dict[str, Any]:
        """Budget-aware sizing (wave 67): how much of this goal's
        remaining work fits in TODAY's model budget.

        Unlimited budgets (the default) always fit.  With a cap set:
        * ``steps_today`` — steps the loop can actually afford to run
          today (each build step needs its full reservation, so it's a
          floor division, not a guess);
        * ``eta_days`` — days of full-budget work the remainder needs
          (1 when it finishes tomorrow, ceil(est/cap) beyond that);
        * ``fits_today`` — the whole remaining work fits in what's left.

        This is what turns "suspended at step 3 of 5" into visible,
        planned, multi-day work instead of a mystery pause.
        """
        if est_calls <= 0:
            return {"fits_today": True, "steps_today": pending,
                    "eta_days": 0}
        try:
            from .cognition import ModelBudget

            b = ModelBudget(self.context)
        except Exception:  # noqa: BLE001 — fit is a display, never fatal
            return {"fits_today": True, "steps_today": pending,
                    "eta_days": 0}
        if b.cap <= 0:
            return {"fits_today": True, "steps_today": pending,
                    "eta_days": 0}
        import math

        remaining = b.remaining_available()
        per = max(1, per_step)
        steps_today = max(0, min(pending, remaining // per))
        if est_calls <= remaining:
            eta = 0  # finishes in what's left of today
        elif est_calls <= b.cap:
            eta = 1  # a full day of budget covers it
        else:
            eta = math.ceil(est_calls / b.cap)
        return {"fits_today": est_calls <= remaining,
                "steps_today": steps_today, "eta_days": eta}

    def _score_goal(self, g: Goal, *, max_priority: int,
                    depth_cache: dict[str, int]) -> dict[str, Any]:
        """Risk- and cost-adjusted expected value (0-100) + the factors."""
        pending = sum(1 for s in g.steps
                      if s.status in {"pending", "in_progress", "blocked"})
        norm = self._priority_norm(g.priority, max_priority)
        depth = self._unmet_depth(g, depth_cache, set())
        base = _EV_BASE_MIN + _EV_BASE_SPREAD * norm
        depth_factor = 1.0 / (1.0 + depth)
        reliability = 1.0 / (1.0 + 0.5 * max(0, g.heals))
        size = 1.0 / (1.0 + 0.1 * pending)
        est_calls = self._est_calls(g, pending)
        cost_factor = 1.0 / (1.0 + _EV_COST_WEIGHT * est_calls)
        ev = 100.0 * base * depth_factor * reliability * size * cost_factor
        ev = max(0.0, min(100.0, ev))
        per_step = self._step_cost(g)
        fit = self._budget_fit(est_calls, pending, per_step)
        if depth >= 2 or g.heals >= 2:
            risk = "high"
        elif depth >= 1 or g.heals >= 1 or pending > 8 or \
                est_calls > _COST_RISK_THRESHOLD:
            risk = "medium"
        else:
            risk = "low"
        return {
            "id": g.id, "title": g.title, "status": g.status,
            "expected_value": round(ev, 1), "risk": risk,
            "priority": g.priority, "unmet_dependency_depth": depth,
            "heals": g.heals, "pending_steps": pending,
            "est_calls": est_calls,
            "progress": round(g.progress, 3),
            "fits_today": fit["fits_today"],
            "steps_today": fit["steps_today"],
            "eta_days": fit["eta_days"],
            "factors": {"base": round(base, 3),
                        "depth": round(depth_factor, 3),
                        "reliability": round(reliability, 3),
                        "size": round(size, 3),
                        "cost": round(cost_factor, 3)},
        }

    def ev_scores(self, goal_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Expected value per goal id (active/paused only; unknown ids are
        simply absent).  Used by ``mission plan`` and by the cognitive
        loop's goal ordering."""
        goals: list[Goal] = []
        for gid in goal_ids:
            g = self.goals.get(gid)
            if g is not None and g.status in {"active", "paused"}:
                goals.append(g)
        if not goals:
            return {}
        max_priority = max((g.priority for g in goals), default=0)
        depth_cache: dict[str, int] = {}
        return {g.id: self._score_goal(g, max_priority=max_priority,
                                       depth_cache=depth_cache)
                for g in goals}

    def ranking(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """The whole portfolio (active + paused) ranked by expected value."""
        rows = self.context.db.query(
            "SELECT * FROM agent_goals WHERE status IN ('active','paused') "
            "ORDER BY priority DESC, created_at ASC LIMIT ?", (limit,))
        goals: list[Goal] = []
        for r in rows:
            g = Goal.from_row(r)
            g.steps = self.goals._steps(g.id)
            goals.append(g)
        if not goals:
            return []
        max_priority = max((g.priority for g in goals), default=0)
        depth_cache: dict[str, int] = {}
        ranked = [self._score_goal(g, max_priority=max_priority,
                                   depth_cache=depth_cache) for g in goals]
        ranked.sort(key=lambda r: (-r["expected_value"], -r["priority"],
                                   r["id"]))
        return ranked

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _row(g: Goal) -> dict[str, Any]:
        d = g.to_dict()
        d["pending_steps"] = sum(
            1 for s in g.steps if s.status in {"pending", "in_progress",
                                               "blocked"})
        return d


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "mission",
        description=(
            "Mission control: the whole goal portfolio at a glance — "
            "what's ready, what's blocked on what, live projects, and the "
            "next pick. action=plan | next."
        ),
        capability="memory.read",
        parameters={
            "action": "str — plan|next",
            "limit": "int — max goals in the plan (default 50)",
        },
    )
    def mission(action: str = "plan", *, limit: str = "50") -> dict[str, Any]:
        mc = MissionControl(context)
        action = (action or "plan").strip().lower()
        try:
            n = int(limit or 50)
        except ValueError:
            n = 50
        if action == "next":
            return mc.next()
        return mc.plan(limit=n)
