"""Mission execution: plan, run, checkpoint, resume, reflect.

The contract is that a mission survives ``kill -9``. That is achieved by writing
progress to SQLite after *every* step, and by deriving "what is left to do" from
persisted state rather than from anything held in memory. On restart the runner
loads the latest checkpoint and continues from the first incomplete step.

Execution is dependency-aware: plan steps carry ``depends_on`` and run in
topological levels (Kahn's algorithm) — a cycle or dangling dependency fails
fast at plan time, a failed step blocks its dependents, and per-step
policies (``optional`` for degraded mode, ``on_failure: continue`` vs
``fail_fast``) decide what a failure means.  A fail-fast failure triggers
one replanning attempt (bounded, policy-gated) before the mission dies;
a terminal failure escalates to the owner with the evidence.

Two consequences worth stating:

- **A step may run twice.** If the process dies after a step completes but before
  the checkpoint lands, that step repeats. Steps must therefore be idempotent, or
  explicitly record their own completion. At-least-once is the honest guarantee;
  exactly-once would need distributed transactions we do not have.
- **Budget is persisted, not accumulated.** A mission cannot launder its budget by
  crashing and restarting, because spent wall-clock and tokens live in the row.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..agents.autonomy_ledger import record_ledger
from ..agents.orchestrator import MasterOrchestrator
from ..core.errors import NoMoralsError, ValidationError, classify
from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..core.tasks import TaskKind
from .mission import (
    Mission,
    MissionStatus,
    MissionStore,
    mission_liveness,
)
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

#: Health-watchdog default: a live mission with no fresh checkpoint *and* no
#: fresh heartbeat for this long counts as stuck (see ``MissionRunner.health``).
STUCK_AFTER_SECONDS = 1800.0

#: Per-step failure policies (read from ``step.payload``).
ON_FAILURE_FAIL_FAST = "fail_fast"
ON_FAILURE_CONTINUE = "continue"


def _step_policy(step: Any) -> dict[str, Any]:
    """Execution policy for one plan step, from ``step.payload``.

    * ``optional`` (bool): a failed optional step degrades instead of
      failing the mission — recorded, skipped, the mission continues and
      can still complete.
    * ``on_failure``: ``"fail_fast"`` (default — stop the mission, then
      try replanning) or ``"continue"`` (record the failure and keep
      going; the mission still ends FAILED, but later steps get a
      chance).
    """
    payload = getattr(step, "payload", None) or {}
    if not isinstance(payload, dict):
        payload = {}
    on_failure = str(payload.get("on_failure") or ON_FAILURE_FAIL_FAST).strip().lower()
    if on_failure not in {ON_FAILURE_FAIL_FAST, ON_FAILURE_CONTINUE}:
        on_failure = ON_FAILURE_FAIL_FAST
    return {
        "optional": bool(payload.get("optional")),
        "on_failure": on_failure,
    }


def _topo_levels(steps: list[Any]) -> list[list[Any]]:
    """Group plan steps into dependency levels (Kahn's algorithm).

    Every step in level N may run only after all steps in levels < N
    completed.  Steps inside one level are independent of each other and
    run in plan order (deterministic; the blackboard stays sequential).

    Raises :class:`ValidationError` on an unknown dependency or a
    dependency cycle — a plan that cannot be scheduled must fail at plan
    time, not mid-run.
    """
    by_name: dict[str, Any] = {}
    for step in steps:
        name = str(getattr(step, "name", "") or "")
        if not name:
            raise ValidationError("plan step without a name")
        if name in by_name:
            raise ValidationError(f"duplicate plan step name: {name!r}")
        by_name[name] = step
    for step in steps:
        for dep in (getattr(step, "depends_on", None) or []):
            if str(dep) not in by_name:
                raise ValidationError(
                    f"plan step {getattr(step, 'name', '?')!r} depends on "
                    f"unknown step {dep!r}")
    # Kahn's: repeatedly take steps whose deps are all emitted.
    remaining = {name: set(str(d) for d in (getattr(s, "depends_on", None) or []))
                 for name, s in by_name.items()}
    order = [s.name for s in steps]  # stable plan order within a level
    levels: list[list[Any]] = []
    while remaining:
        ready = [n for n in order if n in remaining and not remaining[n]]
        if not ready:
            cycle = sorted(remaining)
            raise ValidationError(
                "dependency cycle in plan steps: "
                + ", ".join(cycle))
        levels.append([by_name[n] for n in ready])
        done = set(ready)
        for n in ready:
            del remaining[n]
        for deps in remaining.values():
            deps -= done
    return levels


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
        artifact_store: Any | None = None,
        orchestrator_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.context = context
        self.store = store or MissionStore(context.db)
        self.checkpoint_every = max(1, checkpoint_every)
        self.on_step = on_step
        self._clock = clock
        self._cancel = False
        # Factory for the planner/executor orchestrator.  Production uses
        # MasterOrchestrator; tests inject a stub so replanning is
        # deterministic.  None = build the real one lazily per use.
        self._orchestrator_factory = orchestrator_factory
        # Idempotency (Wave J): when set, each step execution is wrapped in
        # ``dedupe`` keyed by (mission, step name, goal, role). A step that
        # already completed is never re-executed on resume/retry — its stored
        # outcome is replayed instead — so a crash between a step's side
        # effects and its checkpoint cannot duplicate them. A step that ran
        # but reported ok=False is recorded as *failed* and may retry.
        # ``None`` (the default) keeps the historical always-execute path.
        self.idempotency = idempotency
        # Artifact store for acceptance verification (see _verify_acceptance).
        # Injected in tests; otherwise built lazily from the context db so
        # merely constructing a runner never touches the filesystem.
        self.artifact_store = artifact_store
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

    def _emit_bus(self, topic: str, data: dict[str, Any]) -> None:
        """Publish a mission event for other systems (triggers, ledger,
        timeline).  Fail-open: a broken bus never breaks a run."""
        try:
            global_bus.publish(Event(topic=topic, data=data,
                                     source="nomorals.missions.runner"))
        except Exception:  # noqa: BLE001
            _log.debug("mission bus publish %s failed", topic, exc_info=True)

    def _ledger(self, kind: str, mission: Mission, summary: str, *,
                cost_seconds: float = 0.0, cost_tokens: int = 0,
                ok: bool = True, learned: str = "",
                metadata: dict[str, Any] | None = None) -> str:
        """Journal one entry to the unified autonomy ledger. Never raises."""
        try:
            return record_ledger(self.context, "mission", kind, mission.id,
                                 summary, cost_seconds=cost_seconds,
                                 cost_tokens=cost_tokens, ok=ok,
                                 learned=learned,
                                 metadata={"mission": mission.name, **(metadata or {})})
        except Exception:  # noqa: BLE001
            _log.debug("mission ledger write failed", exc_info=True)
            return ""

    def _make_orchestrator(self) -> Any:
        """Deprecated alias for :meth:`_new_orchestrator` (kept for any
        external callers)."""
        return self._new_orchestrator(8)

    def _new_orchestrator(self, max_steps: int) -> Any:
        """The planner/executor.  The injected factory wins (tests);
        otherwise the real MasterOrchestrator with the mission's
        max_steps — the historical default is 8."""
        if self._orchestrator_factory is not None:
            orch = self._orchestrator_factory()
            if hasattr(orch, "max_steps"):
                try:
                    orch.max_steps = max_steps
                except Exception:  # noqa: BLE001 - stub may be frozen
                    pass
            return orch
        from ..agents.orchestrator import MasterOrchestrator as _MO

        return _MO(self.context, max_steps=max_steps)

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
        acceptance: dict[str, Any] | Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MissionResult:
        """Create a mission and run it immediately.

        ``acceptance`` (optional) sets the mission's acceptance criteria at
        creation — the runner then verifies the finished mission through
        VERIFYING before it may complete (see :meth:`_verify_acceptance`).

        ``metadata`` may carry ``lock_key``: when set, starting fails fast
        if a live (non-terminal) mission already holds the same key — two
        missions racing the same goal is how duplicate side effects
        happen.
        """
        # Reset here, not in run(): run() is also the resume path, and clearing the
        # flag there silently discards a cancel() requested from another thread.
        self._cancel = False
        metadata = dict(metadata or {})
        lock_key = str(metadata.get("lock_key") or "").strip()
        if lock_key:
            live = self.store.live_with_lock(lock_key)
            if live:
                raise ValidationError(
                    f"a live mission already holds lock {lock_key!r}: "
                    f"{live[0].id} ({live[0].name}) — resume or cancel it first")
        mission = self.store.create_new(
            goal, name=name, budget_wall=budget_wall, budget_tokens=budget_tokens,
            acceptance=acceptance, metadata=metadata,
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
        # Dependency-aware execution: steps run in topological levels, so a
        # plan that fans out (research_a + research_b -> synthesize) actually
        # honors its graph instead of running flat in plan order.  A cycle
        # or dangling dependency fails fast here — never mid-run.
        levels = _topo_levels(plan_steps)
        completed: set[str] = set(mission.state.get("completed_steps") or [])
        failed: set[str] = set()
        failure: str = ""
        soft_failures: list[str] = []
        step_index = 0

        level_idx = 0
        while level_idx < len(levels):
            level = levels[level_idx]
            stop_levels = False
            for step in level:
                if self._cancel:
                    return self._finish(mission, MissionStatus.CANCELLED, steps, started,
                                        resumed_from, error="cancelled")
                if mission.iterations >= max_iterations:
                    failure = f"iteration limit {max_iterations} reached"
                    stop_levels = True
                    break
                if mission.budget_exhausted:
                    failure = "budget exhausted"
                    self._record_budget_stall(mission)
                    stop_levels = True
                    break
                if step.name in completed:
                    _log.debug("mission %s skipping completed step %s", mission.id, step.name)
                    continue
                if step.name in failed:
                    # a step that failed in THIS run already got its verdict
                    # (fail_fast / replan) — the fresh graph covers recovery;
                    # re-executing it would double-charge iterations.
                    _log.debug("mission %s skipping failed step %s",
                               mission.id, step.name)
                    continue
                # a failed dependency blocks its dependents: they never run
                dep_failed = [d for d in (step.depends_on or []) if d in failed]
                if dep_failed:
                    outcome = StepOutcome(
                        step=step.name, ok=False,
                        detail=(f"blocked: not executed — dependenc"
                                f"{'y' if len(dep_failed) == 1 else 'ies'} "
                                f"{', '.join(sorted(dep_failed))} failed"),
                    )
                    steps.append(outcome)
                    failed.add(step.name)
                    self._settle_step(mission, step, outcome, step_index,
                                      track_failure=False)
                    step_index += 1
                    continue

                outcome = self._execute_step(mission, step)
                steps.append(outcome)
                mission.iterations += 1
                mission.charge(wall=outcome.seconds, tokens=outcome.tokens)
                step_index += 1
                policy = _step_policy(step)

                if outcome.ok:
                    completed.add(step.name)
                    mission.state["completed_steps"] = sorted(completed)
                    mission.state.setdefault("outputs", {})[step.name] = outcome.payload
                    self._settle_step(mission, step, outcome, step_index)
                    continue

                # ── failure: degraded / continue / fail-fast ──────────
                failed.add(step.name)
                step_failure = outcome.detail or f"step {step.name} failed"
                mission.state["last_error"] = step_failure
                if policy["optional"]:
                    # degraded mode: the step failed, but the mission was
                    # designed to survive it — record and carry on.
                    outcome.payload["degraded"] = True
                    self._ledger(
                        "step", mission,
                        f"step {step.name} failed but is optional — degraded, continuing",
                        cost_seconds=outcome.seconds, cost_tokens=outcome.tokens,
                        ok=True, learned=step_failure[:200],
                        metadata={"step": step.name, "degraded": True})
                    self._settle_step(mission, step, outcome, step_index,
                                      track_failure=False)
                    continue
                self._settle_step(mission, step, outcome, step_index)
                if policy["on_failure"] == ON_FAILURE_CONTINUE:
                    soft_failures.append(f"{step.name}: {step_failure[:120]}")
                    continue
                # fail_fast: try to replan the remaining work before dying
                failure = step_failure
                replanned = self._maybe_replan(
                    mission, plan_steps, step.name, step_failure,
                    completed, failed)
                if replanned is not None:
                    _log.info("mission %s continuing on replanned graph",
                              mission.id)
                    failure = ""  # the replan takes over; the failure is history
                    levels = replanned
                    # restart the level scan on the fresh graph: the outer
                    # loop's `level_idx += 1` below lands this on level 0
                    # (completed steps skip fast).  Only `break`s out of the
                    # step loop — the failure stands only when no replan.
                    level_idx = -1
                    plan_steps = [s for lvl in replanned for s in lvl]
                    break
                stop_levels = True
                break
            if stop_levels:
                break
            level_idx += 1

        # observability: when the mission dies, the steps that never ran
        # get a verdict instead of vanishing — transitively blocked by a
        # failed dependency, or simply not reached.
        if failure:
            seen_steps = {s.step for s in steps}
            blocked: set[str] = set()
            progressed = True
            while progressed:
                progressed = False
                for step in plan_steps:
                    if (step.name in completed or step.name in failed
                            or step.name in blocked
                            or step.name in seen_steps):
                        continue
                    dep_failed = [d for d in (step.depends_on or [])
                                  if d in failed or d in blocked]
                    if dep_failed:
                        outcome = StepOutcome(
                            step=step.name, ok=False,
                            detail=(f"blocked: not executed — dependenc"
                                    f"{'y' if len(dep_failed) == 1 else 'ies'} "
                                    f"{', '.join(sorted(dep_failed))} failed"),
                        )
                        steps.append(outcome)
                        blocked.add(step.name)
                        self._settle_step(mission, step, outcome, step_index,
                                          track_failure=False)
                        step_index += 1
                        progressed = True
            for step in plan_steps:
                if (step.name in completed or step.name in failed
                        or step.name in blocked
                        or step.name in seen_steps):
                    continue
                outcome = StepOutcome(
                    step=step.name, ok=False,
                    detail=(f"not reached — mission stopped: "
                            f"{failure[:120]}"),
                )
                steps.append(outcome)
                self._settle_step(mission, step, outcome, step_index,
                                  track_failure=False)
                step_index += 1

        if not failure and soft_failures:
            failure = "failed steps (on_failure=continue): " + "; ".join(soft_failures)
        final = MissionStatus.DONE if not failure else MissionStatus.FAILED
        return self._finish(mission, final, steps, started, resumed_from,
                            error=failure, reflect=reflect)

    def _settle_step(self, mission: Mission, step: Any, outcome: StepOutcome,
                     index: int, *, track_failure: bool = True) -> None:
        """Post-step bookkeeping: stall tracking, heartbeat, checkpoint,
        callbacks, milestone reports, ledger, bus.  ``track_failure=False``
        for blocked/degraded steps — they are not real step failures, so
        they must not feed the consecutive-failure stall budget."""
        if track_failure:
            new_stall = self._apply_step_result(mission, outcome, step.name)
        else:
            new_stall = False
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
        self._ledger(
            "step", mission,
            f"step {step.name}: {'ok' if outcome.ok else 'FAILED'} — "
            f"{(outcome.detail or '')[:140]}",
            cost_seconds=outcome.seconds, cost_tokens=outcome.tokens,
            ok=outcome.ok,
            metadata={"step": step.name,
                      "role": str(getattr(step, "role", "") or "")})
        self._emit_bus("mission.step.finished", {
            "mission_id": mission.id, "mission_name": mission.name,
            "step": step.name, "ok": outcome.ok,
            "seconds": round(outcome.seconds, 2),
            "tokens": outcome.tokens,
        })

    def _maybe_replan(
        self,
        mission: Mission,
        plan_steps: list[Any],
        failed_step_name: str,
        failure_detail: str,
        completed: set[str],
        failed: set[str],
    ) -> list[list[Any]] | None:
        """Replan the remaining work after a fail-fast step failure.

        Returns fresh topological levels, or None when replanning is
        disabled, exhausted, or produced nothing usable.  The failed step
        stays failed — the new plan covers only the remaining work, with
        names uniquified and dependencies restricted to known steps.
        ``replan_policy`` (mission metadata): ``auto`` (default, once per
        mission) | ``always`` (up to 3) | ``off``.
        """
        policy = str((mission.metadata or {}).get("replan_policy")
                     or "auto").strip().lower()
        if policy not in {"auto", "always", "off"}:
            policy = "auto"
        replans = int(mission.state.get("replans") or 0)
        limit = {"auto": 1, "always": 3, "off": 0}[policy]
        if replans >= limit:
            _log.info("mission %s replan budget spent (%d/%d)",
                      mission.id, replans, limit)
            return None
        remaining = [s.name for s in plan_steps
                     if s.name not in completed and s.name not in failed]
        recovery_goal = (
            f"Recover the mission {mission.name!r} (goal: {mission.goal[:200]}). "
            f"Step {failed_step_name!r} just failed: {failure_detail[:300]} "
            f"Already completed: {sorted(completed) or 'none'}. "
            f"Still outstanding: {remaining or 'none'}. "
            f"Produce a revised plan for the REMAINING work only — "
            f"do not redo completed steps.")
        try:
            orchestrator = self._new_orchestrator(
                int(mission.metadata.get("max_steps", 8)))
            plan = orchestrator.plan(recovery_goal)
        except Exception as exc:  # noqa: BLE001 - a broken planner fails the mission, never the runner
            _log.warning("mission %s replan crashed: %s", mission.id, exc)
            return None
        plan_error = getattr(plan, "plan_error", "") or ""
        new_steps = list(getattr(plan, "steps", None) or [])
        if plan_error or not new_steps:
            _log.info("mission %s replan unusable (%s)",
                      mission.id, plan_error or "no steps")
            return None
        # sanitize: unique names, deps anchored on completed work or on
        # other new steps only — never on failed steps or on old steps
        # the replan is replacing.
        known = set(completed) | set(failed) | {s.name for s in plan_steps}
        dep_ok = set(completed)
        taken = set(known)
        clean: list[Any] = []
        for step in new_steps:
            base = str(getattr(step, "name", "") or "step").strip() or "step"
            name, i = base, 2
            while name in taken:
                name = f"{base}_r{i}"
                i += 1
            taken.add(name)
            try:
                step.name = name
            except Exception:  # noqa: BLE001 - frozen stub steps
                pass
            deps = [str(d) for d in (getattr(step, "depends_on", None) or [])
                    if str(d) in dep_ok]
            try:
                step.depends_on = deps
            except Exception:  # noqa: BLE001
                pass
            dep_ok.add(name)
            clean.append(step)
        if not clean:
            return None
        from types import SimpleNamespace

        combined = ([s for s in plan_steps
                     if s.name in completed or s.name in failed]
                    + clean)
        mission.state["plan"] = _serialize_plan(SimpleNamespace(steps=combined))
        mission.state["replans"] = replans + 1
        mission.state["last_replan"] = {
            "at": time.time(),
            "failed_step": failed_step_name,
            "new_steps": [s.name for s in clean],
        }
        self.store.save(mission)
        self._ledger("replanned", mission,
                     f"replanned after {failed_step_name} failed: "
                     f"{len(clean)} new step(s)",
                     ok=True, learned=failure_detail[:200],
                     metadata={"failed_step": failed_step_name,
                               "new_steps": [s.name for s in clean]})
        self._emit_bus("mission.replanned", {
            "mission_id": mission.id, "failed_step": failed_step_name,
            "new_steps": [s.name for s in clean],
        })
        _log.info("mission %s replanned (%d new steps)", mission.id, len(clean))
        return _topo_levels(combined)

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

    # ── watchdog ─────────────────────────────────────────────────────────────

    def health(self, *, limit: int = 50,
               stuck_after_seconds: float | None = None) -> dict[str, Any]:
        """Live-mission watchdog snapshot.

        A mission counts as *stuck* when it is live (running/paused) but has
        neither a fresh checkpoint nor a fresh heartbeat for longer than
        ``stuck_after_seconds`` — i.e. no sign of a worker at all. A mission
        on a long step keeps a fresh heartbeat and is *not* stuck.

        Returns ``{"active", "stuck", "stuck_after_seconds"}``; each active
        entry carries ``id``/``mission_id``, ``name``, ``status``,
        ``checkpoint_age_seconds`` (None when never checkpointed),
        ``heartbeat_age_seconds``, ``stuck`` and the ``stall`` record.
        Read-only: never mutates a mission.
        """
        threshold = (STUCK_AFTER_SECONDS if stuck_after_seconds is None
                     else max(1.0, float(stuck_after_seconds)))
        now = time.time()
        active: list[dict[str, Any]] = []
        stuck: list[str] = []
        for mission in self.store.resumable():
            if len(active) >= max(1, limit):
                break
            point = self.store.latest_checkpoint(mission.id)
            cp_age = (now - point.created_at) if point is not None else None
            hb_age = mission_liveness(mission)["heartbeat_age_s"]
            hb_fresh = hb_age <= threshold
            cp_fresh = cp_age is not None and cp_age <= threshold
            is_stuck = (
                mission.status in (MissionStatus.RUNNING, MissionStatus.PAUSED)
                and not hb_fresh and not cp_fresh
            )
            active.append({
                "id": mission.id,
                "mission_id": mission.id,
                "name": mission.name,
                "status": mission.status,
                "checkpoint_age_seconds": cp_age,
                "heartbeat_age_seconds": hb_age,
                "stuck": is_stuck,
                "stall": (mission.state or {}).get("stall"),
            })
            if is_stuck:
                stuck.append(mission.id)
        return {"active": active, "stuck": stuck,
                "stuck_after_seconds": threshold}

    def self_heal(self, mission_id: str, *, background: bool = True,
                  max_iterations: int = 8) -> dict[str, Any]:
        """One best-effort heal attempt for a stuck mission. Never raises.

        - unknown/terminal mission: not attempted (history is not rewritten);
        - worker provably dead (no live process *and* stale heartbeat, the
          same probe ``MissionStore.reconcile`` uses): the stale heartbeat is
          dropped and the mission is resumed so its remaining steps re-drive;
        - worker alive: not attempted — the watchdog cried wolf on a long step.

        The heal is recorded on the mission row (``state["self_heal"]``);
        callers gate repeat attempts themselves. Returns
        ``{"attempted": bool, "mission_id", ...}``.
        """
        try:
            mission = self.store.get(mission_id)  # raises NotFound when unknown
        except Exception as exc:  # noqa: BLE001 - healing never raises
            return {"attempted": False, "mission_id": mission_id,
                    "error": f"{type(exc).__name__}: {exc}"}
        if mission.terminal:
            return {"attempted": False, "mission_id": mission_id,
                    "reason": f"mission is {mission.status}"}
        try:
            if mission_liveness(mission)["alive"]:
                return {"attempted": False, "mission_id": mission_id,
                        "reason": "worker appears alive"}
            # The worker is dead: drop the stale heartbeat and re-drive the
            # remaining steps. Status stays live (running/paused), so the os
            # state machine needs no repair — run() re-fires RUNNING, which
            # is an idempotent no-op on the transition hook.
            mission.state.pop("heartbeat", None)
            mission.state["self_heal"] = {"at": time.time(),
                                         "reason": "worker dead; resumed remaining steps"}
            self.store.save(mission)
            _log.warning("self-healing mission %s (dead worker)", mission_id)
            if background:
                thread = threading.Thread(
                    target=self._self_heal_job,
                    args=(mission_id, max_iterations),
                    name=f"mission-selfheal-{mission_id[:8]}",
                    daemon=True,
                )
                thread.start()
                return {"attempted": True, "mission_id": mission_id,
                        "background": True}
            result = self.resume(mission_id, max_iterations=max_iterations)
            return {"attempted": True, "mission_id": mission_id,
                    "background": False, "status": result.status,
                    "ok": result.ok}
        except Exception as exc:  # noqa: BLE001 - healing never raises
            _log.warning("self-heal for mission %s failed: %s", mission_id, exc)
            return {"attempted": False, "mission_id": mission_id,
                    "error": f"{type(exc).__name__}: {exc}"}

    def _self_heal_job(self, mission_id: str, max_iterations: int) -> None:
        """Background half of :meth:`self_heal` — exceptions stay in the log."""
        try:
            self.resume(mission_id, max_iterations=max_iterations)
        except Exception:  # noqa: BLE001 - chat/watchdog must stay alive
            _log.exception("background self-heal failed for mission %s",
                           mission_id)

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
        max_steps = int(mission.metadata.get("max_steps", 8))
        orchestrator = self._new_orchestrator(max_steps)
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
        if status == MissionStatus.DONE:
            # Acceptance gate: a mission that finished its steps but carries
            # acceptance criteria must pass VERIFYING before it may COMPLETE.
            # A failed verification becomes a real FAILED — never a silent
            # success.
            mission, status, error = self._verify_acceptance(mission, error)
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
        self._ledger(
            "terminal", mission,
            f"mission {status}: {mission.name} "
            f"({mission.iterations} iterations)",
            cost_seconds=self._clock() - started,
            cost_tokens=mission.spent_tokens,
            ok=(status == MissionStatus.DONE),
            learned="; ".join(lessons)[:500],
            metadata={"status": status, "iterations": mission.iterations,
                      "success": mission.success})
        self._emit_bus("mission.terminal", {
            "mission_id": mission.id, "mission_name": mission.name,
            "status": status, "success": mission.success,
            "iterations": mission.iterations,
            "spent_wall": round(mission.spent_wall, 1),
            "spent_tokens": mission.spent_tokens,
        })
        if status == MissionStatus.FAILED:
            self._escalate(mission, error)
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

    def _escalate(self, mission: Mission, error: str) -> None:
        """Owner escalation on terminal failure: what failed, what was
        done, what it cost, and how to resume.  Best-effort — a broken
        notifier must not break mission teardown."""
        try:
            from ..agents.notifier import Notifier

            completed = mission.state.get("completed_steps") or []
            body = (
                f"Mission {mission.name!r} failed.\n"
                f"Goal: {mission.goal[:300]}\n"
                f"Steps done ({len(completed)}): "
                f"{', '.join(completed[:8]) or 'none'}\n"
                f"Cost: {mission.iterations} iterations, "
                f"{mission.spent_wall:.0f}s wall, {mission.spent_tokens} tokens\n"
                f"Error: {(error or '')[:500]}\n"
                f"Resume with: nm mission redrive {mission.id}"
            )
            Notifier(self.context).publish(
                "mission", f"\u274c mission failed: {mission.name}", body)
            self._ledger("escalated", mission,
                         f"escalated failure to owner: {(error or '')[:120]}",
                         ok=False)
            _log.warning("mission %s escalated to owner", mission.id)
        except Exception:  # noqa: BLE001
            _log.debug("mission escalation failed", exc_info=True)

    def _verify_acceptance(
        self, mission: Mission, error: str
    ) -> tuple[Mission, str, str]:
        """VERIFYING gate before a mission may COMPLETE.

        When the mission carries acceptance criteria
        (``state["acceptance"]``), transition RUNNING -> VERIFYING and run
        ``nomorals.os.mission_state.evaluate_acceptance`` against the
        mission's artifacts + persisted evidence. Returns
        ``(mission, status, error)``: DONE when verification passes, FAILED
        with a concrete reason when it does not — a failed verification is
        never a silent success. Missions without criteria keep the
        historical direct path (no VERIFYING detour).

        The os import is lazy and dynamic (``importlib``): missions (L5)
        must not import os (L6) statically — see
        ``nomorals.missions.wiring`` for the sanctioned pattern.
        """
        acceptance = mission.acceptance
        if not acceptance:
            return mission, MissionStatus.DONE, error
        updated = self._os_transition(mission.id, "VERIFYING",
                                      "acceptance verification")
        if updated is not None:
            mission = updated
        try:
            mission_state = importlib.import_module("nomorals.os.mission_state")
            acceptance_obj = mission_state.MissionAcceptance.from_dict(acceptance)
            passed, details = mission_state.evaluate_acceptance(
                mission, self._artifact_store(), acceptance_obj)
        except Exception as exc:  # noqa: BLE001 - a broken verifier fails the mission, never the runner
            passed = False
            details = {"passed": False,
                       "error": f"{type(exc).__name__}: {exc}",
                       "criteria": [],
                       "required_criteria_failed": [],
                       "artifact_types": {"required": [], "present": [],
                                          "missing": []}}
            _log.warning("mission %s acceptance evaluation failed: %s",
                         mission.id, exc)
        mission.state["verification"] = details
        self.store.save(mission)
        if passed:
            _log.info("mission %s acceptance verified", mission.id)
            return mission, MissionStatus.DONE, error
        parts: list[str] = []
        failed = details.get("required_criteria_failed") or []
        if failed:
            parts.append("failed criteria: " + ", ".join(str(n) for n in failed))
        missing = (details.get("artifact_types") or {}).get("missing") or []
        if missing:
            parts.append("missing artifact types: " + ", ".join(str(t) for t in missing))
        detail = "; ".join(parts) or "acceptance criteria not met"
        if details.get("error"):
            detail += f" ({details['error']})"
        verr = f"acceptance verification failed — {detail}"
        _log.warning("mission %s %s", mission.id, verr)
        return mission, MissionStatus.FAILED, verr

    def _artifact_store(self) -> Any:
        """ArtifactStore for acceptance evaluation.

        Injected via the constructor in tests; otherwise built lazily from
        the context db (same layout ``nm mission redrive`` uses) and
        memoized, so constructing a runner never touches the filesystem.
        """
        if self.artifact_store is not None:
            return self.artifact_store
        from pathlib import Path

        from ..storage.artifacts import ArtifactStore
        from ..storage.blob import BlobStore

        db = getattr(self.context, "db", None)
        if db is None:
            raise ValidationError(
                "acceptance verification needs a context with a db")
        db_path = getattr(db, "path", None)
        blob_dir = Path(db_path).parent / "blobs" if db_path else Path("data/blobs")
        self.artifact_store = ArtifactStore(db, BlobStore(db, blob_dir))
        return self.artifact_store

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
    entries = []
    for s in plan.steps:
        payload = getattr(s, "payload", None)
        entry = {
            "name": s.name,
            "goal": s.goal,
            "role": s.role,
            "kind": s.kind.value,
            "depends_on": list(s.depends_on),
        }
        # step execution policies ride along in the payload so a resumed
        # mission honors the same degraded/continue/fail-fast behavior
        if isinstance(payload, dict):
            policy = {k: payload[k] for k in ("optional", "on_failure")
                      if k in payload}
            if policy:
                entry["policy"] = policy
        entries.append(entry)
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
        payload = dict(entry.get("policy") or {})
        steps.append(
            PlanStep(
                name=str(entry.get("name") or "step"),
                goal=str(entry.get("goal") or ""),
                role=str(entry.get("role") or "execution"),
                kind=kind,
                depends_on=list(entry.get("depends_on") or []),
                payload=payload,
            )
        )
    return steps
