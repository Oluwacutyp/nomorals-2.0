"""Master Orchestrator.

The loop that makes this an *agent system* rather than a pile of utilities:

    plan → decompose → assign roles → execute in parallel → aggregate → reflect

Planning and reflection are themselves model calls, so the loop improves its own
plan over time: the reflector writes lessons into memory, and the next plan is
built with those lessons in context.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..llm.brain import brain_for
from ..core.errors import NoMoralsError, classify
from ..core.ids import new_id
from ..core.jsonutil import extract_json as _extract_json
from ..core.logging_setup import get_logger
from ..core.result import Ok, Err
from .base import Budget
from .blackboard import Blackboard
from .checkpoints import CheckpointStore
from .debate import (
    DEFAULT_RUBRIC,
    Critique,
    Debate,
    Issue,
    WorkArtifact,
)
from .role_specs import RoleRegistry, SwarmAgent
from .runtime import ExecutionReport, HybridExecutor
from .supervisor import Supervisor
from ..core.tasks import Task, TaskGraph, TaskKind, TaskState

__all__ = ["MasterOrchestrator", "Plan", "PlanStep", "OrchestrationResult",
           "Reevaluation"]

_log = get_logger(__name__)


@dataclass
class PlanStep:
    """One unit of a plan."""

    name: str
    goal: str
    role: str = "execution"
    kind: TaskKind = TaskKind.IO
    depends_on: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "goal": self.goal,
            "role": self.role,
            "kind": self.kind.value,
            "depends_on": self.depends_on,
        }


@dataclass
class Plan:
    """A decomposed goal."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    rationale: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "rationale": self.rationale,
            "model": self.model,
            "steps": [s.to_dict() for s in self.steps],
        }

    def as_task_graph(self, name: str = "mission") -> TaskGraph:
        """Convert to a TaskGraph. Handlers are attached by the orchestrator."""
        graph = TaskGraph(name=name)
        for step in self.steps:
            graph.add(
                Task(
                    name=step.name,
                    kind=step.kind,
                    role=step.role,
                    payload={"goal": step.goal, **step.payload},
                )
            )
        # Wire dependencies by name after every task exists.
        for step in self.steps:
            for dep in step.depends_on:
                target = graph.get(step.name)
                resolved = graph.get(dep)
                if target is not None and resolved is not None and resolved.id not in target.deps:
                    target.deps.append(resolved.id)
        return graph


@dataclass
class Reevaluation:
    """One mid-flight plan checkpoint decision.

    ``action`` is one of:

    * ``continue`` — the plan is still valid; nothing changed.
    * ``trim``     — remaining step(s) were superseded (an identical goal
                     already completed) and marked skipped.
    * ``revise``   — a settled task's result carried an explicit
                     ``revise_plan`` directive; the named remaining steps
                     were dropped.
    * ``abort``    — the plan's assumptions are invalid (failure cascade
                     or an explicit stop signal); remaining work cancelled.
    """

    action: str
    reason: str
    affected: list[str] = field(default_factory=list)
    at_task: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "affected": list(self.affected),
            "at_task": self.at_task,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class OrchestrationResult:
    goal: str
    plan: Plan
    report: ExecutionReport
    answer: str = ""
    lessons: list[str] = field(default_factory=list)
    score: float = 0.0
    seconds: float = 0.0
    #: every mid-flight checkpoint decision, in order (``continue`` included)
    reevaluations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report.failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "ok": self.ok,
            "answer": self.answer,
            "lessons": self.lessons,
            "score": self.score,
            "seconds": round(self.seconds, 3),
            "reevaluations": self.reevaluations,
            "plan": self.plan.to_dict(),
            "report": self.report.to_dict(),
        }


class MasterOrchestrator:
    """Plans, fans out, aggregates, and reflects.

    Mid-flight checkpoints: every ``checkpoint_every`` settled tasks (and
    always on a task failure) the remaining plan is re-evaluated against
    current state via :meth:`reevaluate`.  The plan can be revised, trimmed,
    or aborted with a recorded reason — long runs never blindly continue a
    plan the evidence has already invalidated.
    """

    def __init__(
        self,
        context: Any,
        *,
        executor: HybridExecutor | None = None,
        supervisor: Supervisor | None = None,
        blackboard: Blackboard | None = None,
        max_steps: int = 12,
        planner_prompt: str = "",
        roles: RoleRegistry | None = None,
        checkpoint_every: int = 1,
        debate_critic: Callable[[WorkArtifact, tuple[str, ...]], Critique] | None = None,
    ) -> None:
        self.context = context
        self.executor = executor or (context.executor if context is not None else None)
        self.supervisor = supervisor or Supervisor()
        if blackboard is not None:
            self.blackboard = blackboard
        elif context is not None and getattr(context, "blackboard", None) is not None:
            self.blackboard = context.blackboard
        else:
            self.blackboard = Blackboard()
        self.max_steps = max_steps
        self.planner_prompt = planner_prompt
        # Prompt 02: when a RoleRegistry is wired, plan roles resolve to
        # locked RoleSpecs (unknown roles get the safe execution spec).
        # When None, dispatch behaves exactly as before (backward compat).
        self.roles = roles
        self.history: list[OrchestrationResult] = []
        # Prompt 02: per-role telemetry — tasks, ok/failed, seconds, denials.
        self._role_stats: dict[str, dict[str, Any]] = {}
        # Wave F2: mid-flight plan checkpoints.  Every checkpoint_every
        # settled tasks (and always on failure) the remaining plan is
        # re-evaluated; decisions land here (bounded) and on each run's
        # result.reevaluations.
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.reevaluation_log: deque[Reevaluation] = deque(maxlen=64)
        # Injected critic for per-step debate (see _debate_step). When None,
        # a model critic is built on demand; with no model available the
        # debate is skipped honestly and the step result passes through.
        self.debate_critic = debate_critic
        # Run-state persistence (set per run() call, not at construction).
        self._run_checkpoint_store: CheckpointStore | None = None
        self._run_id: str = ""
        self._run_plan: Plan | None = None

    # ── planning ─────────────────────────────────────────────────────────────
    def plan(self, goal: str, *, context_hint: str = "") -> Plan:
        """Ask the model for a decomposition, then validate and repair it.

        The repair step is not optional: models routinely emit steps that depend
        on names they never defined, or duplicate names. Accepting that verbatim
        turns into a graph that cannot schedule.
        """
        prompt = self.planner_prompt or (
            "Decompose the goal into at most "
            f"{self.max_steps} concrete steps. Reply with JSON only, shaped as "
            '{"rationale": "...", "steps": [{"name": "short_id", "goal": "...", '
            '"role": "research|coding|vision|data_collection|execution|social", '
            '"kind": "io|cpu|async", "depends_on": ["short_id"], '
            '"debate": true}]}.\n'
            'Set "debate": true only on high-stakes steps whose output must '
            'survive a critic before it counts as done.\n'
            f"Goal: {goal}"
        )
        if context_hint:
            prompt = f"{prompt}\n\nContext: {context_hint}"
        # Repo orientation for self-referential goals ("link Spotify",
        # "where is the X code"): the decomposer must know Devon's real
        # layout and connector catalog instead of guessing.
        try:
            import re as _re
            from .orientation import repo_orientation_block
            if _re.search(r"\b(devon|nomorals|connect|link|integrat|"
                          r"connector|repo|codebase|spotify|github|gmail)\b",
                          goal, _re.I):
                prompt = f"{prompt}\n\n{repo_orientation_block()}"
        except Exception:  # noqa: BLE001 — orientation is a bonus
            pass
        # Prompt 01: lesson-memory injection at the planner choke point —
        # the decomposition sees known failure patterns before emitting steps.
        try:
            if self.context is not None:
                from .failure import enrich_with_lessons
                lessons_block = enrich_with_lessons(
                    self.context, goal, limit=3)
                if lessons_block:
                    prompt = f"{prompt}\n\n{lessons_block}"
        except Exception:  # noqa: BLE001 — lessons are a bonus, never fatal
            pass

        plan = Plan(goal=goal)
        router = getattr(self.context, "router", None) if self.context is not None else None
        plan_error = ""
        if router is None:
            # No model to ask: the template plan below is a degradation and
            # must carry the reason — a silent fallback would look like a
            # clean model-made plan.
            plan_error = "no LLM router configured — using template plan"
        else:
            from ..llm.base import Message, SamplingParams

            response = brain_for(self.context).chat(
                [Message.user(prompt)],
                SamplingParams(temperature=0.2, max_tokens=2048, json_mode=True),
            task_kind="plan")
            if response.ok:
                plan = self._parse_plan(goal, response.text) or plan
                plan.model = response.model
            if not plan.steps:
                if not response.ok:
                    reason = _plan_model_failure_reason(response)
                else:
                    reason = "model returned no usable plan"
                plan_error = f"{reason} — using template plan"
        if not plan.steps:
            plan = self._fallback_plan(goal)
        # plan_error is "" on the clean model path; non-empty whenever the
        # template fallback ran, so callers (mission runner, telemetry) can
        # tell a degraded plan from a model-made one.
        plan.plan_error = plan_error
        return self._repair(plan)

    def _parse_plan(self, goal: str, text: str) -> Plan | None:
        data = _extract_json(text)
        if not isinstance(data, dict):
            return None
        raw_steps = data.get("steps") or []
        if not isinstance(raw_steps, list):
            return None
        steps: list[PlanStep] = []
        for index, entry in enumerate(raw_steps[: self.max_steps]):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or f"step{index + 1}").strip().replace(" ", "_")[:48]
            kind_raw = str(entry.get("kind") or "io").lower()
            try:
                kind = TaskKind(kind_raw)
            except ValueError:
                kind = TaskKind.IO
            deps = entry.get("depends_on") or []
            # Step execution policies ride in the payload for the mission
            # runner: "optional" (a failed step degrades instead of failing
            # the mission) and "on_failure" ("fail_fast" | "continue").
            # "debate": true routes the step's output through coder→critic
            # rounds before it counts as done.
            payload: dict[str, Any] = {}
            if isinstance(entry.get("optional"), bool):
                payload["optional"] = entry["optional"]
            on_failure = str(entry.get("on_failure") or "").strip().lower()
            if on_failure in {"fail_fast", "continue"}:
                payload["on_failure"] = on_failure
            if entry.get("debate") is True:
                payload["debate"] = True
            steps.append(
                PlanStep(
                    name=name,
                    goal=str(entry.get("goal") or name),
                    role=str(entry.get("role") or "execution"),
                    kind=kind,
                    depends_on=[str(d) for d in deps] if isinstance(deps, list) else [],
                    payload=payload,
                )
            )
        if not steps:
            return None
        return Plan(goal=goal, steps=steps, rationale=str(data.get("rationale") or ""))

    def _fallback_plan(self, goal: str) -> Plan:
        """A sane plan when no model is available. Keeps the system usable offline."""
        return Plan(
            goal=goal,
            rationale="model returned no usable plan; using the default research→act→verify pipeline",
            steps=[
                PlanStep(name="research", goal=f"Gather what is needed for: {goal}", role="research"),
                PlanStep(
                    name="execute", goal=f"Produce the deliverable for: {goal}",
                    role="execution", depends_on=["research"],
                ),
                PlanStep(
                    name="verify", goal=f"Check the result against: {goal}",
                    role="critic", depends_on=["execute"],
                ),
            ],
        )

    def _repair(self, plan: Plan) -> Plan:
        """Make a model-produced plan schedulable."""
        seen: set[str] = set()
        repaired: list[PlanStep] = []
        for index, step in enumerate(plan.steps):
            name = step.name or f"step{index + 1}"
            while name in seen:
                name = f"{name}_{index}"
            seen.add(name)
            step.name = name
            repaired.append(step)
        # Drop dependencies on names that do not exist, and any self-dependency.
        for step in repaired:
            step.depends_on = [d for d in step.depends_on if d in seen and d != step.name]
        # Break cycles by removing back-edges in topological order.
        order: list[str] = []
        remaining = list(repaired)
        while remaining:
            ready = [s for s in remaining if all(d in order for d in s.depends_on)]
            if not ready:
                for step in remaining:
                    step.depends_on = [d for d in step.depends_on if d in order]
                ready = remaining[:1]
            for step in ready:
                order.append(step.name)
                remaining.remove(step)
        plan.steps = repaired[: self.max_steps]
        return plan

    # ── execution ────────────────────────────────────────────────────────────
    def run(
        self,
        goal: str,
        *,
        plan: Plan | None = None,
        handlers: dict[str, Callable[[Task], Any]] | None = None,
        default_handler: Callable[[Task], Any] | None = None,
        budget: Budget | None = None,
        fail_fast: bool = False,
        reflect: bool = True,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str = "",
    ) -> OrchestrationResult:
        """Execute a goal end to end.

        Mid-flight checkpoints: the executor calls back after every
        settled task; every ``checkpoint_every`` tasks (and always on a
        failure) the remaining plan is re-evaluated and can be revised,
        trimmed, or aborted with a recorded reason.  A checkpoint never
        raises and never blocks the run.

        With ``checkpoint_store``, the plan plus every task's state is
        persisted at each checkpoint under ``run_id`` — a dead run can be
        continued later with :meth:`resume` instead of restarting.
        """
        started = time.perf_counter()
        plan = plan or self.plan(goal)
        graph = plan.as_task_graph(name=goal[:48])
        self._begin_run_state(goal, graph, plan, checkpoint_store, run_id)
        self.supervisor.budget = budget

        handler = default_handler or self._default_handler
        role_handlers = handlers or {}
        dispatch = self._make_dispatch(goal, handler, role_handlers)
        for task in graph.tasks.values():
            task.fn = dispatch
            task.args = (task,)

        return self._execute_graph(
            graph, plan=plan, goal=goal, dispatch=dispatch,
            fail_fast=fail_fast, reflect=reflect, started=started)

    def resume(
        self,
        run_id: str,
        checkpoint_store: CheckpointStore,
        *,
        handlers: dict[str, Callable[[Task], Any]] | None = None,
        default_handler: Callable[[Task], Any] | None = None,
        budget: Budget | None = None,
        fail_fast: bool = False,
        reflect: bool = True,
        state_id: str = "",
    ) -> OrchestrationResult:
        """Continue an interrupted run from a checkpoint snapshot.

        Rebuilds the plan and task graph from the newest snapshot for
        ``run_id`` (or ``state_id`` explicitly), restores DONE / FAILED /
        SKIPPED tasks with their results, and executes only what is still
        PENDING — finished work is never re-run. Handlers are attached
        fresh, so new code or credentials apply to the resumed run.

        Raises ``ValueError`` when no usable snapshot exists.
        """
        started = time.perf_counter()
        states = checkpoint_store.list_states(run_id)
        if state_id:
            states = [s for s in states if s["state_id"] == state_id]
        if not states:
            raise ValueError(f"no checkpoint snapshots for run {run_id!r}")
        record = checkpoint_store.load_state(states[0]["state_id"])
        if record is None:
            raise ValueError(
                f"checkpoint {states[0]['state_id']!r} is missing or corrupt")
        payload = record.get("payload") or {}
        goal = str(payload.get("goal") or "")
        raw_steps = payload.get("steps") or []
        if not goal or not raw_steps:
            raise ValueError(
                f"checkpoint {states[0]['state_id']!r} has no goal/steps")

        plan = _plan_from_state(goal, raw_steps,
                                rationale=str(payload.get("rationale") or ""))
        graph = plan.as_task_graph(name=goal[:48])
        restored = self._restore_task_states(graph, payload.get("tasks") or {})
        self._begin_run_state(goal, graph, plan, checkpoint_store, run_id)
        self.supervisor.budget = budget

        handler = default_handler or self._default_handler
        role_handlers = handlers or {}
        dispatch = self._make_dispatch(goal, handler, role_handlers)
        for task in graph.tasks.values():
            task.fn = dispatch
            task.args = (task,)

        _log.info("resuming run %s from %s: %d task(s) restored, %d pending",
                  run_id, states[0]["state_id"], restored,
                  sum(1 for t in graph.tasks.values()
                      if t.state is TaskState.PENDING))
        return self._execute_graph(
            graph, plan=plan, goal=goal, dispatch=dispatch,
            fail_fast=fail_fast, reflect=reflect, started=started)

    # ── run internals (shared by run() and resume()) ─────────────────────
    def _begin_run_state(self, goal: str, graph: TaskGraph, plan: Plan,
                         store: CheckpointStore | None, run_id: str) -> None:
        """Per-run bookkeeping shared by :meth:`run` and :meth:`resume`."""
        self._settled = 0
        self._run_reevaluations = []
        self._run_goal = goal
        self._run_graph = graph
        self._run_plan = plan
        self._run_checkpoint_store = store
        self._run_id = (run_id or "").strip() or f"run-{new_id()[-8:]}"

    def _execute_graph(
        self,
        graph: TaskGraph,
        *,
        plan: Plan,
        goal: str,
        dispatch: Callable[[Task], Any],
        fail_fast: bool,
        reflect: bool,
        started: float,
    ) -> OrchestrationResult:
        """Execute an already-built, dispatch-wired graph to completion.

        Shared tail of :meth:`run` and :meth:`resume`: parallel execution
        with mid-flight checkpoints, supervised retry of failures,
        aggregation, reflection, and the finished result.
        """
        if self.executor is None:
            raise RuntimeError("orchestrator has no executor")
        report = self.executor.run(
            graph, fail_fast=fail_fast, on_task_done=self._checkpoint)

        # Give failed tasks one supervised retry before declaring the mission done.
        if report.failed and not fail_fast:
            self.supervisor.watch_graph(graph, dispatch)
            report = self._recount(graph, report)

        answer = self._aggregate(goal, graph)
        lessons: list[str] = []
        score = 0.0
        if reflect:
            score, lessons = self.reflect(goal, graph, report)

        result = OrchestrationResult(
            goal=goal,
            plan=plan,
            report=report,
            answer=answer,
            lessons=lessons,
            score=score,
            seconds=time.perf_counter() - started,
            reevaluations=[ev.to_dict() for ev in self._run_reevaluations],
        )
        self.history.append(result)
        if self.context is not None:
            self.context.emit(
                "mission.finished", goal=goal, ok=result.ok, score=score, seconds=result.seconds
            )
        return result

    def _make_dispatch(
        self,
        goal: str,
        handler: Callable[[Task], Any],
        role_handlers: dict[str, Callable[[Task], Any]],
    ) -> Callable[[Task], Any]:
        """Build the per-task dispatch closure: budget check, role routing,
        named errors, optional coder→critic debate, blackboard posting."""

        def dispatch(task: Task) -> Any:
            self.supervisor.check_budget()
            chosen = role_handlers.get(task.role) or role_handlers.get(task.name) or handler
            if self.context:
                self.context.emit("task.started", task=task.name, role=task.role, goal=goal)
            started = time.perf_counter()
            try:
                result = chosen(task)
            except Exception as exc:
                self._record_role(task.role, time.perf_counter() - started,
                                  ok=False, denials=0)
                # Name the step, its role, and the handler behind the
                # failure — the surfaced error must never be a bare
                # "something went wrong".  The original classification
                # (code/retryable) is preserved on the wrapper.
                raise _named_step_error(task, chosen, exc) from exc
            elapsed = time.perf_counter() - started
            denials = _result_denials(result)
            reported = result.get("error") if isinstance(result, dict) else ""
            if reported:
                # A handler that reports failure in its result failed the
                # step — returning it as a success would hide the failure
                # from the supervised retry, the mid-flight re-evaluation,
                # and the final ok flag.  Fail fast and name it.
                self._record_role(task.role, elapsed, ok=False, denials=denials)
                raise _named_step_error(task, chosen, str(reported))
            # Opt-in quality gate: high-stakes steps survive a critic
            # before they count as done.
            if task.payload.get("debate"):
                result = self._debate_step(task, chosen, result)
            self._record_role(task.role, time.perf_counter() - started,
                              ok=True, denials=denials)
            self.blackboard.post(
                f"task.{task.name}", result, author=task.role, topic=goal,
                metadata={"role": task.role},
            )
            return result

        return dispatch

    # ── per-step debate ──────────────────────────────────────────────────
    def _debate_step(self, task: Task, handler: Callable[[Task], Any],
                     handler_result: Any) -> Any:
        """Route a step's output through coder→critic rounds.

        The handler already did the work; the "coder" packages its output
        as a :class:`WorkArtifact` and the critic grades it against
        :data:`DEFAULT_RUBRIC`. ``approved`` passes the artifact through;
        ``rejected`` fails the step loudly (the supervisor retry and the
        mid-flight re-evaluation both see it); anything else keeps the
        artifact but stamps the verdict on the result so it is never
        silently accepted.
        """
        critic_fn = self.debate_critic
        if critic_fn is None:
            router = (getattr(self.context, "router", None)
                      if self.context is not None else None)
            if router is None:
                _log.warning("step %r asked for debate but no model is "
                             "available for the critic; passing the result "
                             "through unstamped", task.name)
                return handler_result
            critic_fn = self._model_critic

        summary = _stringify(handler_result)[:1500]
        first = WorkArtifact(content=handler_result,
                             summary=f"step {task.name!r}: {summary[:200]}")

        def coder_fn(brief: str, feedback: list[Issue] | None,
                     artifact: WorkArtifact | None) -> WorkArtifact:
            # Post-hoc review: the work is already done; the coder just
            # re-presents the artifact each round.
            return artifact if artifact is not None else first

        debate = Debate(
            coder_fn=coder_fn,
            critic_fn=critic_fn,
            blackboard=self.blackboard,
            max_rounds=2,
            context=self.context,
        )
        result = debate.run(
            f"Review the output of plan step {task.name!r} "
            f"(role {task.role!r}) for goal: {task.payload.get('goal', '')}",
            initial_artifact=first,
        )
        verdict = result.verdict
        if verdict == "approved":
            _log.info("debate approved step %r (%d round(s), score %.1f)",
                      task.name, result.rounds, result.score)
            return self._stamp_debate(handler_result, result)
        if verdict == "rejected":
            raise _named_step_error(
                task, handler,
                f"debate rejected the output: {result.reason}")
        _log.warning("debate %s on step %r: %s",
                     verdict, task.name, result.reason)
        return self._stamp_debate(handler_result, result)

    @staticmethod
    def _stamp_debate(handler_result: Any, result: Any) -> Any:
        """Stamp a debate verdict onto a step result without hiding it."""
        stamp = {"debate_verdict": result.verdict,
                 "debate_score": round(result.score, 1),
                 "debate_reason": result.reason}
        if isinstance(handler_result, dict):
            return {**handler_result, **stamp}
        return {"output": handler_result, **stamp}

    def _model_critic(self, artifact: WorkArtifact,
                      rubric: tuple[str, ...]) -> Critique:
        """Default critic: grades the artifact with the brain.

        Only used when a router exists (checked by the caller); the
        no-router path degrades to an explicit request_changes instead of
        crashing the debate contract.
        """
        router = getattr(self.context, "router", None) if self.context is not None else None
        if router is None:
            return Critique(verdict="request_changes", score=0.0,
                            notes="no model available for the critic")
        from ..llm.base import Message, SamplingParams

        rubric_lines = "\n".join(f"- {r}" for r in rubric)
        response = brain_for(self.context).chat(
            [
                Message.system(
                    "You are a ruthless critic reviewing an agent's work. "
                    "Grade it against this rubric:\n" + rubric_lines +
                    "\nReply with ONLY a JSON object: "
                    '{"verdict": "approve|request_changes|reject", '
                    '"score": 0-100, "notes": "...", '
                    '"issues": [{"id": "i1", "severity": "critical|major|minor", '
                    '"location": "...", "detail": "..."}]}'),
                Message.user(f"Work summary: {artifact.summary}\n\n"
                             f"Content:\n{_stringify(artifact.content)[:6000]}"),
            ],
            SamplingParams(temperature=0.1, max_tokens=800, json_mode=True),
        task_kind="chat")
        if not response.ok:
            return Critique(verdict="request_changes", score=0.0,
                            notes="critic model call failed: "
                                  f"{getattr(response, 'error', 'unknown')}")
        data = _extract_json(response.text or "")
        if not isinstance(data, dict):
            return Critique(verdict="request_changes", score=0.0,
                            notes="critic returned unparseable output")
        verdict = str(data.get("verdict") or "request_changes")
        if verdict not in {"approve", "request_changes", "reject"}:
            verdict = "request_changes"
        issues: list[Issue] = []
        for raw in (data.get("issues") or [])[:12]:
            if not isinstance(raw, dict):
                continue
            issues.append(Issue(
                id=str(raw.get("id") or f"i{len(issues) + 1}"),
                severity=str(raw.get("severity") or "minor"),
                location=str(raw.get("location") or ""),
                detail=str(raw.get("detail") or "")))
        try:
            score = float(data.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return Critique(verdict=verdict, issues=issues, score=score,
                        notes=str(data.get("notes") or ""))

    # ── run-state persistence ────────────────────────────────────────────
    def _persist_run_state(self) -> None:
        """Snapshot plan + task states to the checkpoint store.

        Called from :meth:`_checkpoint`; best-effort and never raises —
        persistence must not break a run.
        """
        store = getattr(self, "_run_checkpoint_store", None)
        graph = getattr(self, "_run_graph", None)
        plan = getattr(self, "_run_plan", None)
        if store is None or graph is None or plan is None:
            return
        try:
            tasks: dict[str, Any] = {}
            for task in graph.tasks.values():
                tasks[task.name] = {
                    "state": task.state.value,
                    "result": getattr(task, "result", None),
                    "error": task.error,
                    "attempts": task.attempts,
                }
            payload = {
                "goal": self._run_goal,
                "rationale": plan.rationale,
                "steps": [
                    {
                        "name": s.name, "goal": s.goal, "role": s.role,
                        "kind": s.kind.value, "depends_on": s.depends_on,
                        "payload": s.payload,
                    }
                    for s in plan.steps
                ],
                "tasks": tasks,
                "settled": self._settled,
            }
            # CheckpointStore.save_state coerces to JSON-safe itself.
            outcome = store.save_state(self._run_id, payload,
                                       label=f"checkpoint@{self._settled}")
            if not outcome.get("ok"):
                _log.debug("run-state snapshot failed: %s",
                           outcome.get("error"))
        except Exception as exc:  # noqa: BLE001 - persistence never breaks a run
            _log.debug("run-state snapshot crashed: %s", exc)

    @staticmethod
    def _restore_task_states(graph: TaskGraph,
                             states: dict[str, Any]) -> int:
        """Mark tasks DONE/FAILED/SKIPPED from a snapshot. Returns count."""
        restored = 0
        for task in graph.tasks.values():
            saved = states.get(task.name)
            if not isinstance(saved, dict):
                continue
            state = str(saved.get("state") or "")
            error = str(saved.get("error") or "")
            try:
                attempts = int(saved.get("attempts") or 0)
            except (TypeError, ValueError):
                attempts = 0
            task.attempts = attempts
            if state == TaskState.DONE.value:
                task.result = saved.get("result")
                task.mark_done(task.result)
                restored += 1
            elif state == TaskState.FAILED.value:
                task.mark_failed(error or "failed before checkpoint")
                restored += 1
            elif state in (TaskState.SKIPPED.value, TaskState.CANCELLED.value):
                task.state = TaskState.SKIPPED
                task.error = error or "skipped before checkpoint"
                task.finished_at = time.time()
                restored += 1
        return restored

    @staticmethod
    def _recount(graph: TaskGraph, report: ExecutionReport) -> ExecutionReport:
        counts = graph.counts()
        report.done = counts.get(TaskState.DONE.value, 0)
        report.failed = counts.get(TaskState.FAILED.value, 0)
        report.skipped = counts.get(TaskState.SKIPPED.value, 0)
        report.cancelled = counts.get(TaskState.CANCELLED.value, 0)
        report.results = graph.results()
        report.failures = graph.failures()
        return report

    # ── mid-flight plan re-evaluation (wave F2) ─────────────────────────────
    def _checkpoint(self, task: Task) -> None:
        """Executor ``on_task_done`` hook: reassess the remaining plan.

        Runs every ``checkpoint_every`` settled tasks, and always on a
        task failure.  The decision is recorded on the run's result and
        in the bounded :attr:`reevaluation_log`; non-``continue``
        decisions are logged, emitted on the event bus, and persisted to
        the router telemetry so ``nm mind`` can show them.  Never raises.
        """
        try:
            self._settled += 1
            graph = getattr(self, "_run_graph", None)
            if graph is None:
                return
            failed = task.state is TaskState.FAILED
            if not failed and self._settled % self.checkpoint_every:
                return
            evaluation = self.reevaluate(self._run_goal, graph, trigger=task)
        except Exception as exc:  # noqa: BLE001 — a checkpoint must not kill the run
            _log.warning("plan checkpoint failed: %s", exc)
            return
        self._run_reevaluations.append(evaluation)
        self.reevaluation_log.append(evaluation)
        # Persist run state at every checkpoint when a store is wired, so
        # a dead run can resume from here instead of restarting.
        self._persist_run_state()
        if evaluation.action == "continue":
            _log.debug("plan checkpoint @%s: %s",
                       evaluation.at_task, evaluation.reason)
            return
        _log.info("plan %s @%s: %s (affected: %s)",
                  evaluation.action, evaluation.at_task, evaluation.reason,
                  ",".join(evaluation.affected) or "-")
        self._emit_reevaluation(evaluation)

    def reevaluate(self, goal: str, graph: TaskGraph, *,
                   trigger: Task | None = None) -> Reevaluation:
        """Reassess the remaining plan against current state.

        Rules, in order of severity:

        1. **abort** — failure cascade: at least 2 failures and at least
           half of the settled steps failed.  "Failed" means the FAILED
           state *or* a DONE step whose result carries error evidence —
           a step that reports ``{"error": ...}`` in its result failed
           just as loudly as one that raised.  The plan's assumptions
           look invalid, so every non-terminal step is cancelled (the
           executor sees the cancelled graph and exits promptly).
        2. **revise** — a settled task's result carried an explicit
           ``revise_plan`` directive
           (``{"drop": [...step names...], "note": "..."}`` or
           ``{"abort": "reason"}``); the named remaining steps are
           dropped, or the run aborted.
        3. **abort** — the reasoning layer's ``course_correct`` returned a
           ``stop: ...`` pivot for the failed trigger task.
        4. **trim** — a remaining step's goal is identical (normalized)
           to an already-completed step's goal: it is superseded and
           marked skipped.

        Rules 2 and 4 can both apply in one pass; the returned action is
        the more severe one (``revise`` over ``trim``).  Otherwise the
        plan is still valid and ``continue`` is recorded with the
        evidence.  Applies real graph mutations — never advisory-only.
        """
        started = time.perf_counter()
        at = trigger.name if trigger is not None else ""
        settled = [t for t in graph.tasks.values()
                   if t.state in (TaskState.DONE, TaskState.FAILED)]
        # A DONE step whose result carries error evidence counts as failed:
        # through run() such results are converted to FAILED up front, but
        # reevaluate() is also called on hand-built graphs, and a result
        # that contradicts the plan's premise must trip the cascade rule.
        failed = [t for t in settled if not _step_succeeded(t)]
        remaining = [t for t in graph.tasks.values() if not t.is_terminal]

        def _finish(action: str, reason: str,
                    affected: list[str]) -> Reevaluation:
            return Reevaluation(
                action=action, reason=reason, affected=affected, at_task=at,
                seconds=time.perf_counter() - started)

        # 1) failure cascade → abort
        if len(failed) >= 2 and len(failed) / max(1, len(settled)) >= 0.5:
            names = [t.name for t in remaining]
            reason = (f"{len(failed)} of {len(settled)} settled steps failed "
                      f"— the plan's assumptions look invalid")
            cancelled = graph.cancel(reason)
            return _finish(
                "abort",
                f"{reason}; cancelled {cancelled} remaining step(s)", names)

        # 2) explicit handler directives → revise (or abort)
        revised: list[str] = []
        revised_note = ""
        for task in settled:
            directive = _revise_directive(getattr(task, "result", None))
            if not directive:
                continue
            if directive.get("abort"):
                why = str(directive["abort"])[:300] or "handler signalled abort"
                names = [t.name for t in remaining]
                graph.cancel(why)
                return _finish(
                    "abort",
                    f"step {task.name!r} signalled abort: {why}", names)
            drops = directive.get("drop") or []
            if isinstance(drops, str):
                drops = [drops]
            note = str(directive.get("note") or "").strip()[:300]
            for name in drops:
                candidate = graph.get(str(name))
                if candidate is not None and not candidate.is_terminal:
                    candidate.state = TaskState.SKIPPED
                    candidate.error = (
                        f"plan revised mid-flight by {task.name!r}"
                        + (f": {note}" if note else ""))
                    candidate.finished_at = time.time()
                    revised.append(candidate.name)
            if revised:
                revised_note = note

        # 3) reasoning-layer stop signal → abort
        if trigger is not None and trigger.state is TaskState.FAILED:
            pivot = self._course_correct_pivot(trigger)
            if pivot is not None:
                names = [t.name for t in remaining]
                graph.cancel(pivot)
                return _finish(
                    "abort",
                    f"reasoning course-correct stop after {trigger.name!r} "
                    f"failed: {pivot}", names)

        # 4) superseded steps → trim
        done_goals = {_norm_goal(t.payload.get("goal", ""))
                      for t in settled if _step_succeeded(t)}
        done_goals.discard("")
        trimmed: list[str] = []
        for task in remaining:
            if task.state is not TaskState.PENDING:
                continue
            if _norm_goal(task.payload.get("goal", "")) in done_goals:
                task.state = TaskState.SKIPPED
                task.error = ("superseded: an identical goal already "
                              "completed earlier in this run")
                task.finished_at = time.time()
                trimmed.append(task.name)

        if revised:
            return _finish(
                "revise",
                f"step(s) {', '.join(revised)} dropped mid-flight"
                + (f": {revised_note}" if revised_note else ""),
                revised)
        if trimmed:
            return _finish(
                "trim",
                f"{len(trimmed)} remaining step(s) superseded — an identical "
                f"goal already completed", trimmed)
        return _finish(
            "continue",
            f"plan still valid: {len(remaining)} step(s) remaining, "
            f"{len(failed)} failed of {len(settled)} settled", [])

    def _course_correct_pivot(self, task: Task) -> str | None:
        """Ask the reasoning layer whether the approach itself is wrong.

        Returns the ``stop: ...`` pivot text when ``course_correct``
        says the current approach should stop, else None.  Never raises;
        the deterministic signals work with no model at all.
        """
        try:
            from .reasoning import ReasoningAgent

            verdict = ReasoningAgent(self.context).course_correct(
                task.error or task.name, attempts=max(1, task.attempts))
        except Exception as exc:  # noqa: BLE001 — advisory, never fatal
            _log.debug("course-correct consult failed: %s", exc)
            return None
        if not isinstance(verdict, dict) or not verdict.get("should_pivot"):
            return None
        pivot = str(verdict.get("pivot") or "").strip()
        if pivot.lower().startswith("stop"):
            return pivot[:300]
        return None

    def _emit_reevaluation(self, evaluation: Reevaluation) -> None:
        """Make a non-``continue`` decision visible: event bus + telemetry.

        Never raises — observability must not break the run.
        """
        if self.context is None:
            return
        try:
            self.context.emit(
                "plan.reevaluated", action=evaluation.action,
                reason=evaluation.reason, affected=evaluation.affected,
                at_task=evaluation.at_task, goal=self._run_goal)
        except Exception:  # noqa: BLE001 - telemetry never breaks a run
            pass
        try:
            from ..storage import router_telemetry

            router_telemetry.record_reevaluation(
                getattr(self.context, "db", None), evaluation.action,
                evaluation.reason, goal=self._run_goal)
        except Exception:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("reevaluation telemetry write failed", exc_info=True)

    def _default_handler(self, task: Task) -> Any:
        """Delegate to the agent registered for this task's role.

        Prompt 02: when a RoleRegistry is wired, every role resolves to a
        locked RoleSpec (unknown roles get the safe ``execution`` spec —
        never full tool access).  The ``coding`` role keeps the real
        CodingRoleAgent; other roles get a generic SwarmAgent bound to
        their allowlist.  With no registry wired, this is exactly the
        old behavior.
        """
        if self.roles is not None and self.context is not None:
            tools = getattr(self.context, "tools", None)
            if task.role in ("coding", "coder"):
                agent_factory = getattr(tools, "agent_for", None)
                if callable(agent_factory):
                    agent = agent_factory("coding")
                    if agent is not None:
                        return agent.run(task.payload).output
            spec = self.roles.resolve(task.role)
            agent = SwarmAgent(self.context, spec, registry=tools)
            return agent.run(task.payload).output
        if self.context is not None:
            tools = getattr(self.context, "tools", None)
            agent_factory = getattr(tools, "agent_for", None)
            if callable(agent_factory):
                agent = agent_factory(task.role)
                if agent is not None:
                    return agent.run(task.payload).output
        return {"task": task.name, "goal": task.payload.get("goal", ""), "status": "no handler"}

    def _record_role(self, role: str, seconds: float, *,
                     ok: bool, denials: int) -> None:
        """Prompt 02: per-role telemetry for stats() and the event bus."""
        entry = self._role_stats.setdefault(
            role, {"tasks": 0, "ok": 0, "failed": 0,
                   "seconds": 0.0, "denials": 0})
        entry["tasks"] += 1
        entry["ok" if ok else "failed"] += 1
        entry["seconds"] = round(entry["seconds"] + seconds, 3)
        entry["denials"] += denials
        if self.context is not None:
            try:
                self.context.emit(
                    "swarm.task_done", role=role, ok=ok,
                    seconds=round(seconds, 3), denials=denials)
            except Exception:  # noqa: BLE001 - telemetry never breaks dispatch
                pass

    # ── aggregation & reflection ─────────────────────────────────────────────
    def _aggregate(self, goal: str, graph: TaskGraph) -> str:
        """Compose the final answer from completed step outputs."""
        router = getattr(self.context, "router", None) if self.context is not None else None
        results = {name: _stringify(value) for name, value in graph.results().items()}
        if router is None or not results:
            parts = [f"## {name}\n{value}" for name, value in results.items()]
            return "\n\n".join(parts) if parts else "(no steps completed)"

        from ..llm.base import Message, SamplingParams

        digest = "\n\n".join(f"[{name}]\n{value[:2000]}" for name, value in results.items())
        response = brain_for(self.context).chat(
            [
                Message.system("Combine the step outputs into one coherent answer. Be concise."),
                Message.user(f"Goal: {goal}\n\nStep outputs:\n{digest}"),
            ],
            SamplingParams(temperature=0.3, max_tokens=2048),
        task_kind="plan")
        return response.text if response.ok else "\n\n".join(f"## {k}\n{v}" for k, v in results.items())

    def reflect(
        self, goal: str, graph: TaskGraph, report: ExecutionReport
    ) -> tuple[float, list[str]]:
        """Score the run and extract lessons. Persists them to memory.

        The score is mostly mechanical (did the steps succeed?) with a model-based
        adjustment when one is available — a purely model-judged score is not
        trustworthy enough to drive promotion decisions.
        """
        total = len(graph) or 1
        base = report.done / total
        if report.failed:
            base *= 0.5
        lessons: list[str] = []

        router = getattr(self.context, "router", None) if self.context is not None else None
        if router is not None:
            from ..llm.base import Message, SamplingParams

            failures = "; ".join(f"{k}: {v[:120]}" for k, v in report.failures.items()) or "none"
            response = brain_for(self.context).chat(
                [
                    Message.user(
                        f"Goal: {goal}\nCompleted: {report.done}/{total}\nFailures: {failures}\n"
                        "Reply with one or two concrete lessons, each on its own line."
                    )
                ],
                SamplingParams(temperature=0.2, max_tokens=512),
            task_kind="plan")
            if response.ok and response.text.strip():
                lessons = [
                    line.strip("-• ").strip()
                    for line in response.text.strip().splitlines()
                    if line.strip()
                ][:4]

        if report.failures and not lessons:
            lessons = [f"step {name!r} failed: {error[:120]}" for name, error in list(report.failures.items())[:3]]

        score = round(min(1.0, max(0.0, base)), 3)
        memory = getattr(self.context, "memory", None) if self.context is not None else None
        if memory is not None and lessons:
            try:
                memory.remember(
                    f"Mission {goal[:80]!r} scored {score}. Lessons: " + " | ".join(lessons),
                    kind="lesson",
                    importance=max(0.4, score),
                    source="reflector",
                )
            except Exception as exc:  # noqa: BLE001 - reflection must not fail the mission
                _log.debug("could not persist reflection: %s", exc)
        return score, lessons

    # ── reporting ────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        runs = len(self.history)
        roles = {}
        for role, entry in self._role_stats.items():
            tasks = entry["tasks"] or 1
            roles[role] = {
                "tasks": entry["tasks"],
                "success_rate": round(entry["ok"] / tasks, 3),
                "avg_seconds": round(entry["seconds"] / tasks, 3),
                "denials": entry["denials"],
            }
        return {
            "runs": runs,
            "successes": sum(1 for r in self.history if r.ok),
            "avg_score": round(sum(r.score for r in self.history) / runs, 3) if runs else 0.0,
            "avg_seconds": round(sum(r.seconds for r in self.history) / runs, 3) if runs else 0.0,
            "supervisor": self.supervisor.snapshot(),
            "roles": roles,
            "tool_denials": sum(e["denials"] for e in self._role_stats.values()),
            "reevaluations": _reevaluation_summary(self.reevaluation_log),
        }


def _result_denials(result: Any) -> int:
    """Best-effort denial count from a handler's return value."""
    if result is None:
        return 0
    denials = getattr(result, "denials", None)
    if isinstance(denials, (int, float)):
        return int(denials)
    if isinstance(result, dict):
        value = result.get("denials", 0)
        return int(value) if isinstance(value, (int, float)) else 0
    return 0


def _norm_goal(text: Any) -> str:
    """Normalize a step goal for duplicate detection."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split())


def _step_succeeded(task: Task) -> bool:
    """A DONE task whose result carries no error evidence."""
    if task.state is not TaskState.DONE:
        return False
    result = getattr(task, "result", None)
    return not (isinstance(result, dict) and result.get("error"))


def _plan_model_failure_reason(response: Any) -> str:
    """Name what failed when the planner model call fails.

    Keeps the router's whole attempt chain — every provider that failed,
    in order — so the surfaced degradation names names instead of a bare
    "something went wrong".  Tolerates duck-typed routers (tests, stubs)
    that only set ``error``.
    """
    detail = getattr(response, "error", "") or "unknown error"
    reason = f"model call failed ({detail})"
    provider = getattr(response, "provider", "") or ""
    model = getattr(response, "model", "") or ""
    who = provider or model
    if who:
        reason += f" [provider: {who}]"
    chain = getattr(response, "fallback_note", "") or ""
    if chain and chain not in reason:
        reason += f" — {chain}"
    return reason


def _plan_from_state(goal: str, raw_steps: list[Any],
                     rationale: str = "") -> Plan:
    """Rebuild a :class:`Plan` from a checkpoint snapshot's step dicts.

    Tolerates the enriched step shape written by
    :meth:`MasterOrchestrator._persist_run_state` (name/goal/role/kind/
    depends_on/payload) and the leaner :meth:`PlanStep.to_dict` shape.
    """
    steps: list[PlanStep] = []
    for entry in raw_steps:
        if not isinstance(entry, dict):
            continue
        kind_raw = str(entry.get("kind") or "io").lower()
        try:
            kind = TaskKind(kind_raw)
        except ValueError:
            kind = TaskKind.IO
        deps = entry.get("depends_on") or []
        payload = entry.get("payload")
        steps.append(PlanStep(
            name=str(entry.get("name") or f"step{len(steps) + 1}"),
            goal=str(entry.get("goal") or ""),
            role=str(entry.get("role") or "execution"),
            kind=kind,
            depends_on=[str(d) for d in deps] if isinstance(deps, list) else [],
            payload=dict(payload) if isinstance(payload, dict) else {},
        ))
    return Plan(goal=goal, steps=steps, rationale=rationale)


def _named_step_error(task: Task, handler: Callable, cause: Any) -> NoMoralsError:
    """Wrap a step failure so the surfaced error names the step, its role,
    and the handler/tool behind it.

    The original error's ``code``/``retryable`` classification is preserved
    on the wrapper, so downstream retry and routing logic still sees the
    real failure kind — only the message gains its attribution.
    """
    name = getattr(handler, "__name__", None) or type(handler).__name__
    if isinstance(cause, BaseException):
        original = classify(cause)
        detail = f"{type(cause).__name__}: {original.message}"
        code, retryable = original.code, original.retryable
    else:
        detail, code, retryable = str(cause), "step.reported_error", False
    return NoMoralsError(
        f"step {task.name!r} (role {task.role!r}, handler {name}) failed: {detail}",
        code=code,
        retryable=retryable,
    )


def _revise_directive(result: Any) -> dict[str, Any] | None:
    """Extract a handler's explicit mid-flight plan directive.

    A task handler may return ``{"revise_plan": {"drop": [...],
    "note": "..."}}`` to drop remaining steps, or
    ``{"revise_plan": {"abort": "reason"}}`` to stop the run.  Anything
    else (including non-dict results) is not a directive.
    """
    if isinstance(result, dict):
        directive = result.get("revise_plan")
        if isinstance(directive, dict):
            return directive
    return None


def _reevaluation_summary(log: deque[Reevaluation]) -> dict[str, Any]:
    """Aggregate the bounded re-evaluation log for stats()."""
    entries = list(log)
    last = entries[-1].to_dict() if entries else None
    return {
        "total": len(entries),
        "aborts": sum(1 for e in entries if e.action == "abort"),
        "revises": sum(1 for e in entries if e.action == "revise"),
        "trims": sum(1 for e in entries if e.action == "trim"),
        "last": last,
    }


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)
