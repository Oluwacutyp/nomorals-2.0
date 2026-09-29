"""Mission execution: plan, run, checkpoint, resume, reflect.

The contract is that a mission survives ``kill -9``. That is achieved by writing
progress to SQLite after *every* step, and by deriving "what is left to do" from
persisted state rather than from anything held in memory. On restart the runner
loads the latest checkpoint and continues from the first incomplete step.

Two consequences worth stating:

- **A step may run twice.** If the process dies after a step completes but before
  the checkpoint lands, that step repeats. Steps must therefore be idempotent, or
  explicitly record their own completion. At-least-once is the honest guarantee;
  exactly-once would need distributed transactions we do not have.
- **Budget is persisted, not accumulated.** A mission cannot launder its budget by
  crashing and restarting, because spent wall-clock and tokens live in the row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..agents.orchestrator import MasterOrchestrator
from ..agents.tasks import TaskKind
from ..core.errors import BudgetExceeded, NoMoralsError, classify
from ..core.logging_setup import get_logger
from ..missions.mission import Mission, MissionStatus, MissionStore
from ..storage.db import Database

__all__ = ["StepOutcome", "MissionResult", "MissionRunner"]

_log = get_logger(__name__)


@dataclass
class StepOutcome:
    """What one iteration of the mission loop produced."""

    step: str
    ok: bool
    detail: str = ""
    seconds: float = 0.0
    tokens: int = 0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "ok": self.ok,
            "detail": self.detail,
            "seconds": round(self.seconds, 3),
            "tokens": self.tokens,
            "payload": self.payload,
        }


@dataclass
class MissionResult:
    """Terminal summary of a mission run."""

    mission_id: str
    status: str
    success: float | None = None
    iterations: int = 0
    steps: list[StepOutcome] = field(default_factory=list)
    resumed_from: str = ""
    lessons: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == MissionStatus.DONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "status": self.status,
            "ok": self.ok,
            "success": self.success,
            "iterations": self.iterations,
            "steps": [s.to_dict() for s in self.steps],
            "resumed_from": self.resumed_from,
            "lessons": self.lessons,
            "seconds": round(self.seconds, 3),
            "error": self.error,
        }


class MissionRunner:
    STUCK_AFTER_SECONDS = 300.0
    MAX_PIVOTS = 3

    def ops_report(self) -> dict:
        return {"status": "ok", "missions": 0}  # 5 minutes
    """Drives a mission to a terminal state, checkpointing as it goes.

    The loop is deliberately simple: run steps in order, checkpoint after each,
    stop when the budget runs out or a step fails hard. Sophistication lives in
    the orchestrator; the runner's job is durability.
    """

    def __init__(
        self,
        context: Any,
        *,
        store: MissionStore | None = None,
        checkpoint_every: int = 1,
        on_step: Callable[[Mission, StepOutcome], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.context = context
        self.store = store or MissionStore(context.db)
        self.checkpoint_every = max(1, checkpoint_every)
        self.on_step = on_step
        self._clock = clock
        self._cancel = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def cancel(self, reason: str = "cancelled") -> None:
        """Cooperative cancellation, checked between steps."""
        self._cancel = True
        _log.info("mission cancellation requested: %s", reason)

    def start(
        self,
        goal: str,
        *,
        name: str = "",
        budget_wall: float = 0.0,
        budget_tokens: int = 0,
        max_iterations: int = 8,
        reflect: bool = True,
    ) -> MissionResult:
        """Create a mission and run it immediately."""
        # Reset here, not in run(): run() is also the resume path, and clearing the
        # flag there silently discards a cancel() requested from another thread.
        self._cancel = False
        mission = self.store.create_new(
            goal, name=name, budget_wall=budget_wall, budget_tokens=budget_tokens
        )
        return self.run(mission, max_iterations=max_iterations, reflect=reflect)

    def run(
        self,
        mission: Mission,
        *,
        max_iterations: int = 8,
        reflect: bool = True,
    ) -> MissionResult:
        """Run (or resume) a mission to a terminal state."""
        started = self._clock()
        steps: list[StepOutcome] = []

        resumed_from = self._resume(mission)
        if mission.terminal:
            return MissionResult(
                mission_id=mission.id,
                status=mission.status,
                success=mission.success,
                iterations=mission.iterations,
                resumed_from=resumed_from,
                seconds=self._clock() - started,
            )

        mission.status = MissionStatus.RUNNING
        self.store.save(mission)

        plan_steps = self._plan(mission)
        completed: set[str] = set(mission.state.get("completed_steps") or [])
        failure: str = ""

        for index, step in enumerate(plan_steps):
            if self._cancel:
                return self._finish(mission, MissionStatus.CANCELLED, steps, started,
                                    resumed_from, error="cancelled")
            if mission.iterations >= max_iterations:
                failure = f"iteration limit {max_iterations} reached"
                break
            if mission.budget_exhausted:
                failure = "budget exhausted"
                break
            if step.name in completed:
                _log.debug("mission %s skipping completed step %s", mission.id, step.name)
                continue

            outcome = self._execute_step(mission, step)
            steps.append(outcome)
            mission.iterations += 1
            mission.charge(wall=outcome.seconds, tokens=outcome.tokens)

            if outcome.ok:
                completed.add(step.name)
                mission.state["completed_steps"] = sorted(completed)
                mission.state.setdefault("outputs", {})[step.name] = outcome.payload
            else:
                failure = outcome.detail or f"step {step.name} failed"
                mission.state["last_error"] = failure

            self.store.save(mission)
            if index % self.checkpoint_every == 0:
                self.store.checkpoint(mission, label=f"after:{step.name}")
            if self.on_step is not None:
                self.on_step(mission, outcome)

            if not outcome.ok and step.name in {"execute", "act", "run"}:
                break

        final = MissionStatus.DONE if not failure else MissionStatus.FAILED
        return self._finish(mission, final, steps, started, resumed_from,
                            error=failure, reflect=reflect)

    def resume(self, mission_id: str, *, max_iterations: int = 8, reflect: bool = True) -> MissionResult:
        """Reload a mission from storage and continue it."""
        return self.run(self.store.get(mission_id), max_iterations=max_iterations, reflect=reflect)

    def resume_all(self, *, max_iterations: int = 8) -> list[MissionResult]:
        """Continue every interrupted mission. Called on startup."""
        results = []
        for mission in self.store.resumable():
            _log.info("resuming interrupted mission %s", mission.id)
            try:
                results.append(self.run(mission, max_iterations=max_iterations))
            except NoMoralsError as exc:
                _log.error("could not resume mission %s: %s", mission.id, classify(exc).message)
        return results

    # ── internals ────────────────────────────────────────────────────────────

    def _resume(self, mission: Mission) -> str:
        """Restore state from the newest checkpoint. Returns the label used."""
        point = self.store.latest_checkpoint(mission.id)
        if point is None:
            return ""
        # The row is newer than or equal to the checkpoint; prefer the row for
        # counters (they are written every step) but take the checkpoint's step
        # state if the row somehow lost it.
        if not mission.state.get("completed_steps") and point.state.get("completed_steps"):
            mission.state["completed_steps"] = point.state["completed_steps"]
        _log.info(
            "mission %s resuming from checkpoint %s (%d steps done)",
            mission.id, point.label or "unnamed",
            len(mission.state.get("completed_steps") or []),
        )
        return point.label or point.id

    def _plan(self, mission: Mission) -> list[Any]:
        """Ask the orchestrator for a decomposition, memoized in mission state."""
        cached = mission.state.get("plan")
        if cached:
            return _rehydrate_plan(mission.goal, cached)
        orchestrator = MasterOrchestrator(self.context, max_steps=int(mission.metadata.get("max_steps", 8)))
        plan = orchestrator.plan(mission.goal)
        mission.state["plan"] = _serialize_plan(plan)
        self.store.save(mission)
        return plan.steps

    def _execute_step(self, mission: Mission, step: Any) -> StepOutcome:
        """Run one plan step through the orchestrator's agent for that role."""
        started = self._clock()
        from ..agents.roles import build_agent

        prompt = _step_prompt(mission, step)
        try:
            agent = build_agent(step.role, name=f"{mission.id[:8]}-{step.name}", context=self.context)
            result = agent.run(prompt)
        except Exception as exc:  # noqa: BLE001 - a step failure is a result
            return StepOutcome(
                step=step.name,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                seconds=self._clock() - started,
            )
        output = result.output if isinstance(result.output, dict) else {"text": result.output}
        return StepOutcome(
            step=step.name,
            ok=bool(result.ok),
            detail=result.error or "",
            seconds=self._clock() - started,
            tokens=int(result.tokens or 0),
            payload={k: v for k, v in list(output.items())[:20]},
        )

    def _finish(
        self,
        mission: Mission,
        status: str,
        steps: list[StepOutcome],
        started: float,
        resumed_from: str,
        *,
        error: str = "",
        reflect: bool = True,
    ) -> MissionResult:
        lessons: list[str] = []
        mission.status = status

        if reflect and status in {MissionStatus.DONE, MissionStatus.FAILED}:
            score, lessons = self._reflect(mission, steps, status)
            mission.success = score
            self.store.record_reflection(
                mission.id, score=score,
                summary=f"{status} after {mission.iterations} iterations",
                lessons=lessons,
                weights=self.context.memory.stats_snapshot()["weights"]
                if getattr(self.context, "memory", None) is not None else {},
            )

        self.store.save(mission)
        self.store.checkpoint(mission, label=f"final:{status}")
        _log.info("mission %s finished: %s (success=%s)", mission.id, status, mission.success)
        return MissionResult(
            mission_id=mission.id,
            status=status,
            success=mission.success,
            iterations=mission.iterations,
            steps=steps,
            resumed_from=resumed_from,
            lessons=lessons,
            seconds=self._clock() - started,
            error=error,
        )

    def _reflect(
        self, mission: Mission, steps: list[StepOutcome], status: str
    ) -> tuple[float, list[str]]:
        """Score the outcome mostly mechanically.

        A purely model-judged score is not trustworthy enough to drive promotion,
        so the base score is arithmetic and the model only adjusts it.
        """
        total = max(1, len(steps))
        done = sum(1 for s in steps if s.ok)
        score = done / total
        if status == MissionStatus.FAILED:
            score *= 0.5
        lessons = [
            f"step {s.step} failed: {s.detail[:120]}" for s in steps if not s.ok
        ][:5]
        if status == MissionStatus.DONE and not lessons:
            lessons.append(f"completed {done}/{total} steps in {mission.iterations} iterations")

        # Retune recall weights from the outcome: a mission that failed should
        # make future recall lean harder on lexical matching.
        memory = getattr(self.context, "memory", None)
        if memory is not None and status == MissionStatus.FAILED:
            weights = memory.stats_snapshot()["weights"]
            bumped = dict(weights)
            bumped["lexical"] = min(0.4, bumped.get("lexical", 0.15) + 0.05)
            memory.tune_weights(bumped)
        return round(score, 4), lessons


def _step_prompt(mission: Mission, step: Any) -> str:
    """Build the prompt for one step, including what earlier steps produced."""
    outputs = mission.state.get("outputs") or {}
    prior = "\n".join(
        f"- {name}: {str(value)[:300]}" for name, value in list(outputs.items())[-4:]
    )
    parts = [f"Mission goal: {mission.goal}", f"Current step: {step.goal or step.name}"]
    if prior:
        parts.append(f"Already produced:\n{prior}")
    return "\n\n".join(parts)


def _serialize_plan(plan: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "goal": s.goal,
            "role": s.role,
            "kind": s.kind.value,
            "depends_on": list(s.depends_on),
        }
        for s in plan.steps
    ]


def _rehydrate_plan(goal: str, raw: Iterable[dict[str, Any]]) -> list[Any]:
    """Rebuild PlanStep objects from their persisted form."""
    from ..agents.orchestrator import PlanStep

    steps = []
    for entry in raw:
        try:
            kind = TaskKind(entry.get("kind", "io"))
        except ValueError:
            kind = TaskKind.IO
        steps.append(
            PlanStep(
                name=str(entry.get("name") or "step"),
                goal=str(entry.get("goal") or ""),
                role=str(entry.get("role") or "execution"),
                kind=kind,
                depends_on=list(entry.get("depends_on") or []),
            )
        )
    return steps
