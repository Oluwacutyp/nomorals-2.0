"""Bridge: agentic loop → MasterOrchestrator for complex tasks.

When the loop's think step emits ``{"action": "plan", ...}``, this module
translates the goal into a MasterOrchestrator Plan, runs it with
tool-backed handlers, and feeds the result back as an observation the
loop can use to continue.

Two paths to a plan:
1. **Model-provided steps** (preferred): the think step includes explicit
   steps with tool mappings. The bridge validates and converts them.
2. **Orchestrator decomposition** (fallback): the bridge asks the
   MasterOrchestrator to decompose the goal, then makes one model call
   to map each step to the best tool.

Execution uses the MasterOrchestrator's machinery: TaskGraph with
dependencies, parallel execution via HybridExecutor, per-role handlers,
mid-flight checkpoints, and reflection that writes lessons to memory.
Those lessons are returned so the agentic loop can see them in future
think steps.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from ...core.tasks import Task, TaskKind
from ..blackboard import Blackboard
from ..orchestrator import MasterOrchestrator, OrchestrationResult, Plan, PlanStep
from ..runtime import HybridExecutor
from ...llm.base import Message, SamplingParams
from .tools import ToolAdapter

_log = logging.getLogger(__name__)

__all__ = [
    "PlannerBridge",
    "model_steps_to_plan",
    "observation_from_result",
]

_VALID_ROLES = frozenset({
    "research", "coding", "vision", "data_collection",
    "execution", "social",
})
_VALID_KINDS = {"io": TaskKind.IO, "cpu": TaskKind.CPU, "async": TaskKind.ASYNC}

_TOOL_MAP_SYSTEM = """You map plan steps to Devon tools.

RULES:
1. Output ONLY a JSON object: {"mappings": [{"step": "<step name>", "tool": "<tool name>", "args": {…}}]}
2. Use ONLY tools from the AVAILABLE TOOLS list. Every step MUST map to a real tool.
3. "args" must be a JSON object matching what the tool needs. Infer sensible arguments from the step goal.
4. If no tool fits a step well, pick the closest one and note it — do not invent tools.
"""


class _BridgeMemory:
    """Minimal memory stand-in that collects lessons for the loop."""

    def __init__(self) -> None:
        self.lessons: list[str] = []

    def remember(self, text: str, **kwargs: Any) -> None:
        # The reflector calls memory.remember(text, kind="lesson", ...).
        # We capture the text so the bridge can hand lessons to the loop.
        if text:
            self.lessons.append(str(text)[:500])


class PlannerBridge:
    """Runs complex goals through the MasterOrchestrator.

    The agentic loop handles simple tasks directly (1-3 tool calls).
    When it encounters something needing structured planning — parallel
    workstreams, role separation, dependencies — it emits the "plan"
    action and this bridge takes over for that portion of the work.
    """

    def __init__(
        self,
        llm: Any,
        tools: ToolAdapter,
        *,
        max_plan_steps: int = 12,
    ) -> None:
        if llm is None:
            raise ValueError("PlannerBridge needs an LLM")
        if tools is None:
            raise ValueError("PlannerBridge needs a ToolAdapter")
        self.llm = llm
        self.tools = tools
        self.max_plan_steps = max(1, int(max_plan_steps))
        self._orchestrator: MasterOrchestrator | None = None

    # ── public entry ─────────────────────────────────────────────────

    def run_planned(
        self,
        goal: str,
        *,
        model_steps: list[dict[str, Any]] | None = None,
        context_hint: str = "",
    ) -> tuple[bool, str, list[str]]:
        """Execute a goal via planned orchestration.

        Returns ``(ok, observation_text, lessons)``. Never raises for
        task-level issues — failures come back as ``ok=False`` with the
        reason in the observation text, so the agentic loop can decide
        how to recover.
        """
        goal = (goal or "").strip()
        if not goal:
            return False, "plan action needs a non-empty goal", []

        try:
            if model_steps:
                plan = model_steps_to_plan(goal, model_steps, self.tools)
                if plan is None:
                    return (
                        False,
                        "plan rejected: model-provided steps were invalid "
                        "(unknown tools, duplicate names, or bad dependencies). "
                        "Fall back to direct tool calls.",
                        [],
                    )
            else:
                plan = self._decompose_and_map(goal, context_hint)
                if plan is None:
                    return (
                        False,
                        "plan failed: could not decompose the goal into "
                        "executable steps. Fall back to direct tool calls.",
                        [],
                    )

            orchestrator = self._get_orchestrator()
            result = orchestrator.run(
                goal,
                plan=plan,
                default_handler=self._step_handler,
                reflect=True,
            )
            observation = observation_from_result(result)
            lessons = list(result.lessons)
            # Also surface persisted lessons from the bridge memory
            mem = getattr(orchestrator.context, "memory", None)
            if mem is not None and hasattr(mem, "lessons"):
                for lesson in mem.lessons:
                    if lesson not in lessons:
                        lessons.append(lesson)
            return result.ok, observation, lessons[:6]
        except Exception as exc:  # noqa: BLE001 — planned execution must not sink the loop
            _log.warning("planner bridge failed for goal %r: %s", goal[:80], exc)
            return False, f"planned execution failed: {exc}", []

    # ── plan construction ────────────────────────────────────────────

    def _get_orchestrator(self) -> MasterOrchestrator:
        if self._orchestrator is None:
            ctx = self._build_context()
            self._orchestrator = MasterOrchestrator(
                context=ctx,
                executor=HybridExecutor(threads=4),
                max_steps=self.max_plan_steps,
            )
        return self._orchestrator

    def _build_context(self) -> Any:
        """Minimal context the MasterOrchestrator needs.

        Provides: router (LLM for planning/reflection), memory (lesson
        capture), tools (registry access), blackboard, emit (no-op event
        sink). This is deliberately lightweight — the bridge owns the
        lifecycle, not the full agent context.
        """
        llm = self.llm

        class _Router:
            def chat(self, messages: list[Any], params: Any = None, **kw: Any) -> Any:
                return llm.chat(messages, params, **kw)

        ctx = SimpleNamespace(
            router=_Router(),
            memory=_BridgeMemory(),
            tools=SimpleNamespace(registry=self.tools.registry),
            blackboard=Blackboard(),
            executor=None,
        )

        def _emit(event: str, **kwargs: Any) -> None:
            _log.debug("bridge event %s: %s", event, kwargs)

        ctx.emit = _emit
        return ctx

    def _decompose_and_map(
        self, goal: str, context_hint: str
    ) -> Plan | None:
        """Fallback: decompose via orchestrator, then map steps to tools."""
        orchestrator = self._get_orchestrator()
        try:
            plan = orchestrator.plan(goal, context_hint=context_hint)
        except Exception as exc:  # noqa: BLE001
            _log.warning("orchestrator.plan failed: %s", exc)
            return None
        if not plan.steps:
            return None
        mappings = self._map_steps_to_tools(plan)
        if not mappings:
            return None
        # Attach tool mappings to the plan steps' payloads
        by_name = {m["step"]: m for m in mappings}
        for step in plan.steps:
            m = by_name.get(step.name)
            if m and self.tools.has(m["tool"]):
                step.payload["tool"] = m["tool"]
                step.payload["args"] = m["args"] if isinstance(m.get("args"), dict) else {}
            else:
                # Unmappable step — the handler will try relevance ranking
                _log.debug("no tool mapping for step %r", step.name)
        return plan

    def _map_steps_to_tools(
        self, plan: Plan
    ) -> list[dict[str, Any]] | None:
        """One model call: map every plan step to the best tool + args."""
        step_descs = "\n".join(
            f"- {s.name} (role={s.role}): {s.goal}" for s in plan.steps
        )
        tools_list = self.tools.describe_for(plan.goal)
        prompt = (
            f"PLAN STEPS:\n{step_descs}\n\n"
            f"AVAILABLE TOOLS:\n{tools_list}\n\n"
            "Map each step to a tool. Output ONLY the JSON object."
        )
        try:
            resp = self.llm.chat(
                [Message.system(_TOOL_MAP_SYSTEM), Message.user(prompt)],
                SamplingParams(temperature=0.1, max_tokens=2048),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("tool mapping model call failed: %s", exc)
            return None
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            return None
        try:
            from ...core.jsonutil import extract_json

            data = extract_json(text)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(data, dict):
            return None
        mappings = data.get("mappings")
        if not isinstance(mappings, list):
            return None
        out = []
        for m in mappings:
            if not isinstance(m, dict):
                continue
            step = str(m.get("step", "")).strip()
            tool = str(m.get("tool", "")).strip()
            args = m.get("args")
            if step and tool:
                out.append({
                    "step": step,
                    "tool": tool,
                    "args": args if isinstance(args, dict) else {},
                })
        return out or None

    # ── step execution ───────────────────────────────────────────────

    def _step_handler(self, task: Task) -> Any:
        """Execute one plan step as a tool call.

        The tool + args come from the step payload (model-provided or
        mapped). Returns {"result": ...} on success or {"error": ...} on
        failure — the orchestrator treats "error" as a step failure.
        """
        payload = task.payload or {}
        tool_name = str(payload.get("tool", "")).strip()
        args = payload.get("args")
        if not isinstance(args, dict):
            args = {}

        if not tool_name or not self.tools.has(tool_name):
            # Fallback: pick the best tool by relevance to the step goal
            goal = str(payload.get("goal", ""))
            tool_name = self._best_tool_for(goal)
            if not tool_name:
                return {"error": f"no tool available for step {task.name!r}: {goal[:100]}"}

        ok, observation = self.tools.call(tool_name, args)
        if not ok:
            return {"error": observation[:1000]}
        return {"result": observation[:4000], "tool": tool_name}


    def _best_tool_for(self, goal: str) -> str:
        """Pick the top-ranked tool for a goal string. Empty if none."""
        listing = self.tools.describe_for(goal, limit=1)
        # describe_for emits "- name(params): desc" lines
        for line in listing.splitlines():
            line = line.strip()
            if line.startswith("- "):
                # "- tool_name(params): description"
                rest = line[2:]
                name = rest.split("(")[0].strip()
                if name and self.tools.has(name):
                    return name
        return ""


def model_steps_to_plan(
    goal: str,
    model_steps: list[dict[str, Any]],
    tools: ToolAdapter,
) -> Plan | None:
    """Validate model-provided steps and convert to a Plan.

    Returns None if the steps are invalid: duplicate names, unknown
    tools, dependencies on undefined steps, or empty. The caller should
    treat None as "fall back to direct tool calls".
    """
    if not isinstance(model_steps, list) or not model_steps:
        return None

    seen: set[str] = set()
    steps: list[PlanStep] = []
    for i, raw in enumerate(model_steps):
        if not isinstance(raw, dict):
            _log.debug("plan step %d is not an object", i)
            return None
        name = str(raw.get("name", "")).strip()
        step_goal = str(raw.get("goal", "")).strip()
        tool_name = str(raw.get("tool", "")).strip()
        if not name or not step_goal:
            _log.debug("plan step %d missing name or goal", i)
            return None
        if name in seen:
            _log.debug("duplicate plan step name %r", name)
            return None
        seen.add(name)
        if tool_name and not tools.has(tool_name):
            _log.debug("plan step %r references unknown tool %r", name, tool_name)
            return None
        role = str(raw.get("role", "execution")).strip().lower()
        if role not in _VALID_ROLES:
            role = "execution"
        kind_key = str(raw.get("kind", "io")).strip().lower()
        kind = _VALID_KINDS.get(kind_key, TaskKind.IO)
        depends_on = raw.get("depends_on") or []
        if not isinstance(depends_on, list):
            depends_on = []
        depends_on = [str(d).strip() for d in depends_on if str(d).strip()]
        args = raw.get("args")
        payload: dict[str, Any] = {}
        if tool_name:
            payload["tool"] = tool_name
        if isinstance(args, dict):
            payload["args"] = args
        steps.append(
            PlanStep(
                name=name,
                goal=step_goal,
                role=role,
                kind=kind,
                depends_on=depends_on,
                payload=payload,
            )
        )

    # Validate dependencies reference defined steps
    for step in steps:
        for dep in step.depends_on:
            if dep not in seen:
                _log.debug(
                    "plan step %r depends on undefined step %r", step.name, dep
                )
                return None
            if dep == step.name:
                _log.debug("plan step %r depends on itself", step.name)
                return None

    return Plan(goal=goal, steps=steps, rationale="model-provided plan")


def observation_from_result(result: OrchestrationResult) -> str:
    """Render an OrchestrationResult as loop observation text."""
    lines = [
        f"planned execution {'succeeded' if result.ok else 'FAILED'}: "
        f"{result.report.done} done, {result.report.failed} failed, "
        f"{result.report.skipped} skipped "
        f"({result.seconds:.1f}s, score {result.score})",
    ]
    answer = (result.answer or "").strip()
    if answer:
        lines.append(f"result: {answer[:2000]}")
    if result.report.failures:
        failures = "; ".join(
            f"{k}: {v[:200]}" for k, v in list(result.report.failures.items())[:3]
        )
        lines.append(f"failures: {failures}")
    if result.reevaluations:
        # Surface non-trivial mid-flight decisions
        notable = [
            r for r in result.reevaluations
            if isinstance(r, dict) and r.get("action") != "continue"
        ]
        for r in notable[:3]:
            lines.append(
                f"mid-flight {r.get('action')}: {r.get('reason', '')[:200]}"
            )
    if result.lessons:
        lines.append("lessons: " + " | ".join(result.lessons[:3]))
    return "\n".join(lines)
