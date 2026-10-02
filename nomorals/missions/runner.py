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

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..agents.orchestrator import MasterOrchestrator
from ..core.errors import NoMoralsError, ValidationError, classify
from ..core.logging_setup import get_logger
from ..core.tasks import TaskKind
from ..missions.mission import Mission, MissionStatus, MissionStore
from .idempotency import IdempotencyStore, dedupe, step_idempotency_key
from .progress import (
    STALL_AFTER_FAILURES,
    MissionMilestones,
    StallCode,
    clear_stall,
    record_stall,
)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepOutcome:
        """Rebuild from :meth:`to_dict` — used when an idempotency hit
        replays a stored step outcome without re-executing the agent."""
        data = data or {}
        payload = data.get("payload")
        return cls(
            step=str(data.get("step") or ""),
            ok=bool(data.get("ok")),
            detail=str(data.get("detail") or ""),
            seconds=float(data.get("seconds") or 0.0),
            tokens=int(data.get("tokens") or 0),
            payload=dict(payload) if isinstance(payload, dict) else {},
        )


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
        milestones: bool = True,
        milestone_reporter: MissionMilestones | None = None,
        idempotency: IdempotencyStore | None = None,
    ) -> None:
        self.context = context
        self.store = store or MissionStore(context.db)
        self.checkpoint_every = max(1, checkpoint_every)
        self.on_step = on_step
        self._clock = clock
        self._cancel = False
        # Idempotency (Wave J): when set, each step execution is wrapped in
        # ``dedupe`` keyed by (mission, step name, goal, role). A step that
        # already completed is never re-executed on resume/retry — its stored
        # outcome is replayed instead — so a crash between a step's side
        # effects and its checkpoint cannot duplicate them. A step that ran
        # but reported ok=False is recorded as *failed* and may retry.
        # ``None`` (the default) keeps the historical always-execute path.
        self.idempotency = idempotency
        # OS control-plane hooks (Wave H2). Plain optional callables — the
        # runner never imports nomorals.os (L6); whoever wires them provides
        # the callables (see nomorals.os.mission_state.attach_runner and
        # nomorals.os.resources.advisor_callable).
        self._os_transition_hook: Callable[..., Any] | None = None
        self._resource_advisor: Callable[..., Any] | None = None
        # Milestone pushes (started / step / stalled / done) go through the
        # existing Notifier — never a parallel channel. ``milestones=False``
        # disables them; ``milestone_reporter`` injects a pre-built one
        # (tests use this for a fake clock + fake gateway).
        if milestone_reporter is not None:
            self.reporter: MissionMilestones | None = milestone_reporter
        elif milestones:
            from .progress import MissionWatchers

            self.reporter = MissionMilestones(
                context, store=self.store,
                watch_store=MissionWatchers(getattr(context, "db", None)))
        else:
            self.reporter = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def cancel(self, reason: str = "cancelled") -> None:
        """Cooperative cancellation, checked between steps."""
        self._cancel = True
        _log.info("mission cancellation requested: %s", reason)

    # ── milestones & stalls ──────────────────────────────────────────────

    def _heartbeat(self, mission: Mission) -> None:
        """Liveness marker for ``MissionStore.reconcile``.

        Written when a run starts and after every step. A mission that
        still says "running" with a stale heartbeat and a gone pid is a
        dead runner, not a live mission — reconcile() flips it to failed
        with an explicit reason instead of reporting "running" forever.
        """
        mission.state["heartbeat"] = {"pid": os.getpid(), "at": time.time()}

    def _report(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Fire a milestone event. Telemetry: never breaks a run."""
        if self.reporter is None:
            return
        try:
            getattr(self.reporter, method)(*args, **kwargs)
        except Exception:  # noqa: BLE001 - milestone pushes are best-effort
            _log.debug("mission milestone %s failed", method, exc_info=True)

    def _os_transition(self, mission_id: str, to_state: str, note: str = "") -> Any:
        """Fire the os state-machine hook.

        Defensive by design: a missing hook, a hook failure, or an illegal
        transition must never break a run — the os state machine is an
        observer, not a gate. Returns the hook's result (the freshly
        persisted mission) on success, else None — callers re-read from the
        store so the hook's writes are not clobbered by a stale in-memory
        copy.
        """
        hook = getattr(self, "_os_transition_hook", None)
        if hook is None:
            return None
        try:
            return hook(mission_id, to_state, note)
        except Exception:  # noqa: BLE001 - hooks never break a run
            _log.debug("os transition hook failed for %s -> %s",
                       mission_id, to_state, exc_info=True)
            return None

    def _advise_resources(self, mission: Mission) -> None:
        """Consult the resource advisor before a step.

        Advisory only: the advice is logged, never acted on here. A failing
        advisor is ignored — it must not be able to stall a mission.
        """
        advisor = getattr(self, "_resource_advisor", None)
        if advisor is None:
            return
        try:
            advice = advisor(mission)
        except Exception:  # noqa: BLE001 - advisory only
            _log.debug("resource advisor failed", exc_info=True)
            return
        if isinstance(advice, dict) and advice.get("throttled"):
            _log.info("mission %s resource-throttled: %s",
                      mission.id, advice.get("reasons"))

    def mark_stalled(
        self,
        mission_id: str,
        code: str,
        message: str,
        *,
        step: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a concrete stall reason and push it to chat.

        For the reasons only an operator knows: ``waiting_on_provider``
        ("provider X is rate-limiting us"), ``blocked_on_approval``
        ("needs your go-ahead on Y"), ``dependency_missing`` ("Z is not
        installed"). The runner records ``retry_budget_exhausted`` and
        ``budget_exhausted`` itself.
        """
        mission = self.store.get(mission_id)  # raises NotFound when unknown
        if mission.terminal:
            raise ValidationError(
                f"mission {mission_id} is {mission.status}: "
                "a terminal mission cannot stall")
        changed = record_stall(mission, code, message, step=step, extra=extra)
        self.store.save(mission)
        if changed:
            self._report("on_stalled", mission)
        return {"mission_id": mission_id, "changed": changed,
                "stall": mission.state.get("stall")}

    def clear_stalled(self, mission_id: str) -> bool:
        """Drop the stall record (progress resumed)."""
        mission = self.store.get(mission_id)  # raises NotFound when unknown
        cleared = clear_stall(mission)
        if cleared:
            self.store.save(mission)
        return cleared

    def _apply_step_result(
        self, mission: Mission, outcome: StepOutcome, step_name: str
    ) -> bool:
        """Track consecutive failures; declare a stall when the retry
        budget is spent. Returns True when a *new* stall was recorded
        (the caller pushes it). Mutates the in-memory mission; the caller
        saves.
        """
        if outcome.ok:
            mission.state["consec_failures"] = 0
            clear_stall(mission)  # progress resumed — the blocker is gone
            return False
        fails = int(mission.state.get("consec_failures") or 0) + 1
        mission.state["consec_failures"] = fails
        # The transition into stalled happens exactly once, at the budget
        # boundary — later failures keep the original stall record (and its
        # message) instead of manufacturing a "new" event per failure.
        if fails == STALL_AFTER_FAILURES:
            detail = (outcome.detail or "unknown error")[:160]
            return record_stall(
                mission,
                StallCode.RETRY_BUDGET_EXHAUSTED,
                f"{fails} consecutive step failures — last: {detail}",
                step=step_name,
            )
        return False

    def _record_budget_stall(self, mission: Mission) -> bool:
        """Stall with the exact budget numbers. Saves + reports; returns
        whether this is a new stall (duplicate pushes are suppressed)."""
        wall = f"{mission.spent_wall:.0f}s"
        if mission.budget_wall:
            wall += f"/{mission.budget_wall:.0f}s"
        tokens = f"{mission.spent_tokens}"
        if mission.budget_tokens:
            tokens += f"/{mission.budget_tokens}"
        changed = record_stall(
            mission,
            StallCode.BUDGET_EXHAUSTED,
            f"budget exhausted — wall {wall}, tokens {tokens}",
        )
        self.store.save(mission)
        if changed:
            self._report("on_stalled", mission)
        return changed

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
        # Refresh from the store: the hook persists its own copy, and the
        # stale in-memory mission must not clobber it on the next save.
        updated = self._os_transition(mission.id, "PLANNED", "mission created")
        return self.run(updated if updated is not None else mission,
                        max_iterations=max_iterations, reflect=reflect)

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
        self._heartbeat(mission)
        self.store.save(mission)
        self._report("on_started", mission)
        updated = self._os_transition(mission.id, "RUNNING", "run started")
        if updated is not None:
            mission = updated

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
                self._record_budget_stall(mission)
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

            new_stall = self._apply_step_result(mission, outcome, step.name)

            self._heartbeat(mission)
            self.store.save(mission)
            if index % self.checkpoint_every == 0:
                self.store.checkpoint(mission, label=f"after:{step.name}")
            if self.on_step is not None:
                self.on_step(mission, outcome)
            if new_stall:
                self._report("on_stalled", mission)
            elif outcome.ok:
                self._report("on_step", mission, outcome)

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
        """Ask the orchestrator for a decomposition, memoized in mission state.

        A degraded (template-fallback) plan is persisted to the router
        telemetry so ``nm mind`` shows the last plan_error — a mission that
        planned without a model must never look like a clean model plan.
        """
        cached = mission.state.get("plan")
        if cached:
            return _rehydrate_plan(mission.goal, cached)
        orchestrator = MasterOrchestrator(self.context, max_steps=int(mission.metadata.get("max_steps", 8)))
        plan = orchestrator.plan(mission.goal)
        plan_error = getattr(plan, "plan_error", "") or ""
        if plan_error:
            mission.state["plan_error"] = plan_error
            try:
                from ..storage import router_telemetry

                router_telemetry.record_plan_error(
                    getattr(self.context, "db", None), plan_error, route="mission")
            except Exception:  # noqa: BLE001 - telemetry never breaks a mission
                _log.debug("mission plan-error telemetry failed", exc_info=True)
        mission.state["plan"] = _serialize_plan(plan, plan_error=plan_error)
        self.store.save(mission)
        return plan.steps

    def _execute_step(self, mission: Mission, step: Any) -> StepOutcome:
        """Run one plan step through the orchestrator's agent for that role.

        With an idempotency store attached, the execution goes through
        :func:`dedupe`: a step whose key already completed returns its
        stored outcome (no duplicate side effects on the retry path), while
        a step that failed may retry.
        """
        started = self._clock()
        if self.idempotency is None:
            return self._run_step_agent(mission, step, started)
        key = step_idempotency_key(mission.id, step)

        def attempt() -> dict[str, Any]:
            return self._run_step_agent(mission, step, started).to_dict()

        result = dedupe(
            self.idempotency,
            key,
            attempt,
            owner=f"mission:{mission.id}",
            succeeded=lambda value: bool(
                value.get("ok")) if isinstance(value, dict) else True,
        )
        if not isinstance(result.value, dict):
            _log.error("idempotency record for step %s of mission %s is not "
                       "a dict; re-running without the stored outcome",
                       step.name, mission.id)
            return self._run_step_agent(mission, step, started)
        return StepOutcome.from_dict(result.value)

    def _run_step_agent(self, mission: Mission, step: Any,
                        started: float) -> StepOutcome:
        """The actual agent invocation for one step (always executes)."""
        self._advise_resources(mission)
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
        final_state = {
            MissionStatus.DONE: "COMPLETED",
            MissionStatus.FAILED: "FAILED",
            MissionStatus.CANCELLED: "CANCELLED",
        }.get(status)
        if final_state is not None:
            updated = self._os_transition(mission.id, final_state,
                                          f"finished: {status}")
            if updated is not None:
                mission = updated

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
        self._report("on_terminal", mission, status, error=error)
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
    """Build the prompt for one step, including what earlier steps produced.

    Compatibility wrapper: assembly now runs through
    :class:`nomorals.context.ContextEngine` (legacy-compatible mode), so
    existing callers see byte-identical output.  New callers can ask the
    engine for the full token-budgeted assembly via
    ``ContextEngine().build_step_prompt(mission, step, rich=True)``.
    """
    from ..context import ContextEngine

    return ContextEngine().build_step_prompt(mission, step)


def _serialize_plan(plan: Any, *, plan_error: str = "") -> list[dict[str, Any]]:
    entries = [
        {
            "name": s.name,
            "goal": s.goal,
            "role": s.role,
            "kind": s.kind.value,
            "depends_on": list(s.depends_on),
        }
        for s in plan.steps
    ]
    # The degradation marker rides along with the persisted plan so a
    # resumed mission stays honest about how it was planned.
    if plan_error:
        entries.append({"__plan_error__": plan_error})
    return entries


def _rehydrate_plan(goal: str, raw: Iterable[dict[str, Any]]) -> list[Any]:
    """Rebuild PlanStep objects from their persisted form."""
    from ..agents.orchestrator import PlanStep

    steps = []
    for entry in raw:
        if "__plan_error__" in entry:
            # Degradation marker (see _serialize_plan): not a step.
            continue
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
