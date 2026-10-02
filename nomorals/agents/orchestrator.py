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
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.errors import classify
from ..core.jsonutil import extract_json as _extract_json
from ..core.logging_setup import get_logger
from ..core.result import Ok, Err
from .base import Budget
from .blackboard import Blackboard
from .role_specs import RoleRegistry, SwarmAgent
from .runtime import ExecutionReport, HybridExecutor
from .supervisor import Supervisor
from ..core.tasks import Task, TaskGraph, TaskKind, TaskState

__all__ = ["MasterOrchestrator", "Plan", "PlanStep", "OrchestrationResult"]

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
class OrchestrationResult:
    goal: str
    plan: Plan
    report: ExecutionReport
    answer: str = ""
    lessons: list[str] = field(default_factory=list)
    score: float = 0.0
    seconds: float = 0.0

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
            "plan": self.plan.to_dict(),
            "report": self.report.to_dict(),
        }


class MasterOrchestrator:
    """Plans, fans out, aggregates, and reflects."""

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
            '"kind": "io|cpu|async", "depends_on": ["short_id"]}]}.\n'
            f"Goal: {goal}"
        )
        if context_hint:
            prompt = f"{prompt}\n\nContext: {context_hint}"
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

            response = router.chat(
                [Message.user(prompt)],
                SamplingParams(temperature=0.2, max_tokens=2048, json_mode=True),
            )
            if response.ok:
                plan = self._parse_plan(goal, response.text) or plan
                plan.model = response.model
            if not plan.steps:
                reason = (
                    f"model call failed ({response.error or 'unknown error'})"
                    if not response.ok
                    else "model returned no usable plan"
                )
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
            steps.append(
                PlanStep(
                    name=name,
                    goal=str(entry.get("goal") or name),
                    role=str(entry.get("role") or "execution"),
                    kind=kind,
                    depends_on=[str(d) for d in deps] if isinstance(deps, list) else [],
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
    ) -> OrchestrationResult:
        """Execute a goal end to end."""
        started = time.perf_counter()
        plan = plan or self.plan(goal)
        graph = plan.as_task_graph(name=goal[:48])
        self.supervisor.budget = budget

        handler = default_handler or self._default_handler
        role_handlers = handlers or {}

        def dispatch(task: Task) -> Any:
            self.supervisor.check_budget()
            chosen = role_handlers.get(task.role) or role_handlers.get(task.name) or handler
            if self.context:
                self.context.emit("task.started", task=task.name, role=task.role, goal=goal)
            started = time.perf_counter()
            try:
                result = chosen(task)
                ok = not (isinstance(result, dict) and result.get("error"))
            except Exception:
                self._record_role(task.role, time.perf_counter() - started,
                                  ok=False, denials=0)
                raise
            elapsed = time.perf_counter() - started
            denials = _result_denials(result)
            self._record_role(task.role, elapsed, ok=ok, denials=denials)
            self.blackboard.post(
                f"task.{task.name}", result, author=task.role, topic=goal,
                metadata={"role": task.role},
            )
            return result

        for task in graph.tasks.values():
            task.fn = dispatch
            task.args = (task,)

        if self.executor is None:
            raise RuntimeError("orchestrator has no executor")
        report = self.executor.run(graph, fail_fast=fail_fast)

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
        )
        self.history.append(result)
        if self.context is not None:
            self.context.emit(
                "mission.finished", goal=goal, ok=result.ok, score=score, seconds=result.seconds
            )
        return result

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
        response = router.chat(
            [
                Message.system("Combine the step outputs into one coherent answer. Be concise."),
                Message.user(f"Goal: {goal}\n\nStep outputs:\n{digest}"),
            ],
            SamplingParams(temperature=0.3, max_tokens=2048),
        )
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
            response = router.chat(
                [
                    Message.user(
                        f"Goal: {goal}\nCompleted: {report.done}/{total}\nFailures: {failures}\n"
                        "Reply with one or two concrete lessons, each on its own line."
                    )
                ],
                SamplingParams(temperature=0.2, max_tokens=512),
            )
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


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)
