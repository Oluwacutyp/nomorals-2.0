"""Thinking layer: plan -> act -> reflect -> remember.

One deliberate thought cycle for an autonomous task:

1. **Plan** — :func:`make_plan` turns a goal + steps into a
   :class:`Plan` (dataclass, pure: same inputs -> same shape, fresh id).
2. **Act** — :func:`record_act` turns one executed step into an
   :class:`ActRecord` (pure).
3. **Reflect** — :func:`reflect_on` splits acts into what worked / what
   failed and derives lessons + next steps (pure — no I/O, no model).
4. **Remember** — :meth:`ThinkingStore.remember` writes every act of a
   plan into the :class:`TrajectoryStore
   <nomorals.cognition.trajectories.TrajectoryStore>` so the broker and
   the failure KB learn from it, and returns the reflection.

:class:`ThinkingStore` is the in-memory coordinator: it holds plans and
acts, delegates the pure steps to the module functions, and owns the
single impure step (writing trajectories).  Nothing here calls a model
or takes an autonomous outward action — thinking is deliberation, and
deliberation never acts on its own.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_short_id

__all__ = [
    "Plan",
    "ActRecord",
    "ReflectResult",
    "make_plan",
    "record_act",
    "reflect_on",
    "ThinkingStore",
]


@dataclass
class Plan:
    """A deliberated plan: a goal decomposed into ordered steps."""

    id: str
    goal: str
    steps: list[str] = field(default_factory=list)
    task_kind: str = ""
    capability: str = ""
    model_id: str = ""
    predicted_confidence: float = 0.5
    status: str = "planned"  # planned | running | done | failed
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActRecord:
    """One executed step of a plan."""

    id: str
    plan_id: str
    step_index: int
    step: str
    action_class: str
    success: bool
    outcome: str = ""
    error: str = ""
    latency_s: float = 0.0
    created_at: float = 0.0


@dataclass
class ReflectResult:
    """What :func:`reflect_on` learned from a plan's acts."""

    plan_id: str
    worked: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    success_rate: float = 0.5
    total: int = 0
    next_steps: list[str] = field(default_factory=list)


def make_plan(
    goal: str,
    steps: list[str] | tuple[str, ...],
    *,
    task_kind: str = "",
    capability: str = "",
    model_id: str = "",
    predicted_confidence: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> Plan:
    """Build a :class:`Plan`.  Pure: no I/O, no store writes."""
    if not (goal or "").strip():
        raise ValueError("goal is required")
    step_list = [str(s) for s in (steps or [])]
    if not step_list:
        raise ValueError("a plan needs at least one step")
    conf = min(1.0, max(0.0, float(predicted_confidence)))
    return Plan(
        id=new_short_id("plan_"),
        goal=goal.strip(),
        steps=step_list,
        task_kind=task_kind or "",
        capability=capability or "",
        model_id=model_id or "",
        predicted_confidence=conf,
        status="planned",
        created_at=time.time(),
        metadata=dict(metadata or {}),
    )


def record_act(
    plan: Plan,
    step_index: int,
    action_class: str,
    success: bool,
    *,
    outcome: str = "",
    error: str = "",
    latency_s: float = 0.0,
) -> ActRecord:
    """Build an :class:`ActRecord` for one executed step.  Pure.

    ``step_index`` must address a step of ``plan``; the step text is
    copied onto the record so acts stay readable without the plan.
    """
    if not 0 <= int(step_index) < len(plan.steps):
        raise ValueError(
            f"step_index {step_index!r} out of range for {len(plan.steps)} steps")
    return ActRecord(
        id=new_short_id("act_"),
        plan_id=plan.id,
        step_index=int(step_index),
        step=plan.steps[int(step_index)],
        action_class=action_class or "",
        success=bool(success),
        outcome=outcome or "",
        error=error or "",
        latency_s=max(0.0, float(latency_s)),
        created_at=time.time(),
    )


def reflect_on(plan: Plan, acts: list[ActRecord]) -> ReflectResult:
    """Reflect on a plan's acts: what worked, what failed, what next.

    Pure: no I/O.  Acts belonging to another plan are ignored.  With
    no acts the result is honest — 0.5 prior, no lessons invented.
    """
    mine = [a for a in (acts or []) if a.plan_id == plan.id]
    worked = [a.step for a in mine if a.success]
    failed_acts = [a for a in mine if not a.success]
    failed = [a.step for a in failed_acts]
    lessons: list[str] = []
    for act in failed_acts:
        detail = f" ({act.error})" if act.error else ""
        lessons.append(
            f"Step {act.step_index + 1} {act.step!r} failed{detail} —"
            " retry with a different approach before repeating it verbatim.")
    if mine and not failed_acts:
        lessons.append("All executed steps succeeded — this approach is verified for reuse.")
    if not mine:
        lessons.append("No steps executed yet — nothing to learn from.")
    next_steps: list[str] = []
    for act in failed_acts:
        next_steps.append(f"Retry step {act.step_index + 1}: {act.step}")
    if not next_steps:
        next_steps.append("Proceed to the next goal — no retries outstanding.")
    rate = (len(worked) / len(mine)) if mine else 0.5
    return ReflectResult(
        plan_id=plan.id,
        worked=worked,
        failed=failed,
        lessons=lessons,
        success_rate=rate,
        total=len(mine),
        next_steps=next_steps,
    )


class ThinkingStore:
    """In-memory plan/act store + the single impure step (remember).

    The store never calls a model and never takes an outward action.
    ``remember`` is the only method that touches durable state, and it
    only *appends* trajectory rows — it never rewrites history.
    """

    def __init__(self, trajectory_store: Any | None = None) -> None:
        if trajectory_store is None:
            from .trajectories import TrajectoryStore

            trajectory_store = TrajectoryStore(":memory:")
        self.trajectory_store = trajectory_store
        self._lock = threading.RLock()
        self._plans: dict[str, Plan] = {}
        self._acts: dict[str, list[ActRecord]] = {}

    # ── plan ─────────────────────────────────────────────────────────
    def new_plan(self, goal: str, steps: list[str] | tuple[str, ...], **kwargs: Any) -> Plan:
        """Create and store a plan (see :func:`make_plan`)."""
        plan = make_plan(goal, steps, **kwargs)
        with self._lock:
            self._plans[plan.id] = plan
            self._acts.setdefault(plan.id, [])
        return plan

    def plan(self, plan_id: str) -> Plan | None:
        with self._lock:
            return self._plans.get(plan_id)

    # ── act ──────────────────────────────────────────────────────────
    def act(
        self,
        plan_id: str,
        step_index: int,
        action_class: str,
        success: bool,
        **kwargs: Any,
    ) -> ActRecord:
        """Execute (record) one step of a stored plan (see :func:`record_act`)."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise ValueError(f"unknown plan {plan_id!r}")
            rec = record_act(plan, step_index, action_class, success, **kwargs)
            self._acts.setdefault(plan_id, []).append(rec)
            plan.status = "running"
            return rec

    def acts(self, plan_id: str) -> list[ActRecord]:
        with self._lock:
            return list(self._acts.get(plan_id, []))

    # ── reflect ──────────────────────────────────────────────────────
    def reflect(self, plan_id: str) -> ReflectResult:
        """Reflect on a stored plan's acts (see :func:`reflect_on`)."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise ValueError(f"unknown plan {plan_id!r}")
            acts = list(self._acts.get(plan_id, []))
        return reflect_on(plan, acts)

    # ── remember ─────────────────────────────────────────────────────
    def remember(self, plan_id: str) -> dict[str, Any]:
        """Write a plan's acts to the trajectory store + reflect.

        Every act becomes one trajectory row (``task_kind`` /
        ``capability`` / ``model_id`` from the plan; ``success`` /
        ``error`` from the act), so the broker's success rates and the
        failure KB learn from this thought cycle.  Returns
        ``{"plan_id", "trajectories_written", "reflection"}`` and marks
        the plan ``done`` (``failed`` when every act failed and at
        least one act exists).
        """
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise ValueError(f"unknown plan {plan_id!r}")
            acts = list(self._acts.get(plan_id, []))
        reflection = reflect_on(plan, acts)
        written = 0
        for act in acts:
            self.trajectory_store.record(
                task_kind=plan.task_kind,
                capability=plan.capability,
                model_id=plan.model_id,
                success=act.success,
                error=act.error,
                latency_s=act.latency_s,
            )
            written += 1
        with self._lock:
            if acts and all(not a.success for a in acts):
                plan.status = "failed"
            else:
                plan.status = "done"
        return {
            "plan_id": plan_id,
            "trajectories_written": written,
            "reflection": reflection,
        }

    def summary(self, plan_id: str) -> str:
        """One-paragraph owner-readable summary of a thought cycle."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                return f"Unknown plan {plan_id!r}."
            acts = list(self._acts.get(plan_id, []))
        reflection = reflect_on(plan, acts)
        lines = [
            f"🧠 Plan {plan.goal!r} — {plan.status}"
            f" ({reflection.total} step(s),"
            f" {reflection.success_rate:.0%} succeeded)",
        ]
        if reflection.worked:
            lines.append("Worked: " + "; ".join(reflection.worked))
        if reflection.failed:
            lines.append("Failed: " + "; ".join(reflection.failed))
        for lesson in reflection.lessons:
            lines.append(f"• {lesson}")
        return "\n".join(lines)
