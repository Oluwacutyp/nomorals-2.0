"""The god-tier agentic loop: think → act → observe, repeated.

The model decides each step which tool to call (or that it's done), the
loop executes via the registry, feeds the result back, and repeats until
the model responds or the step budget runs out.

Designed for weaker models: the think prompt is highly structured, the
action space is tiny (tool / respond / ask), and output is parsed
defensively.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ...core.jsonutil import extract_json
from ...llm.base import Message, SamplingParams
from .context import LoopMemory, StepRecord
from .tools import ToolAdapter

_log = logging.getLogger(__name__)

__all__ = ["AgenticLoop", "LoopResult", "run_agentic"]

DEFAULT_STEP_BUDGET = 10
DEFAULT_MAX_THINK_CHARS = 4000


@dataclass
class LoopResult:
    """Outcome of one agentic loop run."""

    response: str  # final text for the owner
    steps_taken: int
    tools_called: list[str] = field(default_factory=list)
    asked_user: bool = False  # True if the loop paused for a clarifying question
    question: str = ""  # the clarifying question, if asked_user
    budget_exhausted: bool = False
    success: bool = True
    memory_snapshot: dict[str, Any] = field(default_factory=dict)
    # Serialized LoopMemory — persist this when asked_user is True (or
    # always) and pass it back as resume_from on the user's next message
    # to continue the plan instead of starting fresh.


THINK_SYSTEM = """You are Devon's agentic orchestrator. You solve the owner's request step by step by choosing tools.

RULES:
1. Output ONLY a JSON object, no other text. Format:
   {"thought": "<brief reasoning>", "action": "<tool|tools|plan|respond|ask>", "tool": "<name>", "args": {…}, "calls": [{"tool": "<name>", "args": {…}}], "goal": "<overall objective>", "steps": [{"name": "<id>", "goal": "<step goal>", "tool": "<tool>", "args": {…}, "role": "<role>", "depends_on": ["<id>"]}], "plan": "<your current plan>", "response": "<final answer or question>"}
2. action "tool": call ONE tool. Set "tool" to the exact tool name and "args" to its parameters object.
3. action "tools": call MULTIPLE INDEPENDENT tools at once. Set "calls" to a list of {"tool", "args"} objects. Use ONLY when the calls do not depend on each other's results — they run in parallel.
4. action "plan": the task is COMPLEX — needs 4+ steps, parallel workstreams, distinct roles (research + coding + execution), or dependencies between steps. Set "goal" to the overall objective and "steps" to the decomposed steps, each with a "name", "goal", "tool", "args", "role" (research|coding|vision|data_collection|execution|social), and "depends_on" (list of step names that must finish first). Prefer direct "tool"/"tools" calls for simple tasks (1-3 steps) — use "plan" only when the work genuinely needs structured orchestration.
5. action "respond": you are done. Put your final answer to the owner in "response". Use this when you have what you need.
6. action "ask": you genuinely cannot proceed without the owner clarifying something. Put the question in "response". Use sparingly.
7. "plan": keep your running plan updated here (one or two sentences). It persists across steps and messages.
8. If a tool failed, try a DIFFERENT tool or approach. Never retry the identical call.
9. Chain tools: use one tool's output as the next tool's input.
10. Keep "thought" to one or two sentences.
11. Do not invent tools. Only use tools from the AVAILABLE TOOLS list.
12. Prefer fewer steps. If you can answer now, use "respond".
"""

THINK_USER_TEMPLATE = """AVAILABLE TOOLS (ranked by relevance to this task):
{tools}

{context}

Decide your next action. Output ONLY the JSON object."""


CODE_SYSTEM = """You are Devon's agentic orchestrator. You solve the owner's request by writing Python code that calls tools as functions.

RULES:
1. Output ONLY a Python code block: ```python ... ```
2. Tools are functions: name(**kwargs) -> str (the observation text).
3. Use loops and conditionals freely — do in ONE block what would take many steps.
4. Assign your final answer to the variable `result`.
5. No imports, no file I/O, no network — only the tool functions and safe builtins.
6. If you need to ask the owner something, set result to "ASK: <your question>".
7. Keep it focused. Prefer fewer tool calls.
"""

CODE_USER_TEMPLATE = """AVAILABLE TOOL FUNCTIONS:
{tools}

{context}

Write the Python block to advance the task. Assign the outcome to `result`."""


class AgenticLoop:
    """ReAct loop over Devon's tool registry."""

    def __init__(
        self,
        llm: Any,
        tools: ToolAdapter,
        *,
        step_budget: int | None = None,
        sampling: SamplingParams | None = None,
        code_mode: bool | None = None,
        db: Any = None,
    ) -> None:
        from ...core.profiles import profile_value, get_profile_kind
        if step_budget is None:
            step_budget = int(profile_value("step_budget", DEFAULT_STEP_BUDGET))
        if step_budget < 1:
            raise ValueError("step_budget must be >= 1")
        self.llm = llm
        self.tools = tools
        # Optional skill database for the Hermes distillation hook. When
        # None the hook falls back to the app's default storage path.
        self.db = db
        self.step_budget = step_budget
        _max_tok = int(profile_value("max_tokens", 1024))
        self.sampling = sampling or SamplingParams(temperature=0.2, max_tokens=_max_tok)
        # code-first tool calls: default ON for termux (fewer round-trips =
        # less battery/latency), available everywhere.  Explicit arg wins.
        if code_mode is None:
            try:
                code_mode = get_profile_kind() == "termux"
            except Exception:  # noqa: BLE001
                code_mode = False
        self.code_mode = bool(code_mode)
        self._code_adapter: Any = None

    # ── public entry ─────────────────────────────────────────────────

    def run(self, message: str, memory: LoopMemory | None = None) -> LoopResult:
        """Run the loop to completion. Never raises for task-level issues.

        If ``memory`` is a snapshot from a previous run that paused with
        ``ask``, the new ``message`` is treated as the user's answer and
        the prior plan resumes instead of starting fresh.
        """
        if memory is None:
            memory = LoopMemory(user_message=message)
        elif memory.pending_ask():
            # Resuming: the user is answering our clarifying question.
            # Fold the Q&A into history so the model sees the full arc.
            question = memory.last_question()
            if question:
                memory.add_history("assistant", question)
            memory.add_history("user", message)
        if not memory.user_message:
            memory.user_message = message

        tools_called: list[str] = []

        for step_num in range(1, self.step_budget + 1):
            if self.code_mode:
                done, final = self._step_code(memory, step_num, tools_called)
                if done:
                    return final
                continue
            decision = self._think(memory, step_num)
            if decision is None:
                # Model output was unparseable — count as a failed step and
                # give it one more structured nudge before giving up.
                memory.record_step(
                    StepRecord(
                        step=step_num,
                        thought="model output was not valid JSON",
                        action="tool",
                        failed=True,
                        observation="your last output was not valid JSON. "
                        "Output ONLY the JSON object described in the rules.",
                    )
                )
                continue

            action = decision.get("action", "")
            thought = str(decision.get("thought", ""))[:500]
            # The model maintains its plan; it persists across steps and,
            # via the memory snapshot, across messages.
            plan = decision.get("plan")
            if isinstance(plan, str) and plan.strip():
                memory.set_plan(plan)

            if action == "respond":
                response = str(decision.get("response", "")).strip()
                memory.record_step(
                    StepRecord(step=step_num, thought=thought, action="respond")
                )
                result = LoopResult(
                    response=response or "(no response produced)",
                    steps_taken=step_num,
                    tools_called=tools_called,
                    memory_snapshot=memory.to_dict(),
                )
                # Hermes loop: distill successful multi-tool runs into
                # reusable skill drafts (inactive until promoted). The
                # loop's own LLM does the distillation — previously the
                # hook was called without one, so distill() always
                # returned None and nothing was ever installed.
                try:
                    from ..skill_distillation import maybe_distill

                    def _distill_llm(prompt: str) -> str:
                        resp = self.llm.chat(
                            [Message.user(prompt)], self.sampling)
                        return (getattr(resp, "text", None) or "")

                    maybe_distill(result, memory,
                                  llm_fn=_distill_llm, db=self.db)
                except Exception:  # noqa: BLE001 - distillation never breaks a run
                    pass
                return result

            if action == "ask":
                question = str(decision.get("response", "")).strip()
                memory.record_step(
                    StepRecord(
                        step=step_num,
                        thought=thought,
                        action="ask",
                        question=question,
                    )
                )
                return LoopResult(
                    response=question,
                    steps_taken=step_num,
                    tools_called=tools_called,
                    asked_user=True,
                    question=question,
                    memory_snapshot=memory.to_dict(),
                )

            if action == "tools":
                calls = decision.get("calls")
                if not isinstance(calls, list) or not calls:
                    memory.record_step(
                        StepRecord(
                            step=step_num,
                            thought=thought,
                            action="tool",
                            failed=True,
                            observation=(
                                'action "tools" needs a non-empty "calls" list '
                                'of {"tool", "args"} objects.'
                            ),
                        )
                    )
                    continue
                self._run_parallel_calls(
                    memory, step_num, thought, calls, tools_called
                )
                continue

            if action == "plan":
                self._run_planned_action(
                    memory, step_num, thought, decision, tools_called
                )
                continue

            if action == "tool":
                tool_name = str(decision.get("tool", "")).strip()
                args = decision.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                # Guard: don't blind-retry a tool that already failed
                if tool_name in memory.failed_tools() and self._same_args_as_failed(
                    memory, tool_name, args
                ):
                    memory.record_step(
                        StepRecord(
                            step=step_num,
                            thought=thought,
                            action="tool",
                            tool_name=tool_name,
                            tool_args=args,
                            failed=True,
                            observation=(
                                f"blocked: {tool_name} already failed with these "
                                "arguments. Choose a different tool or different arguments."
                            ),
                        )
                    )
                    continue
                ok, observation = self.tools.call(tool_name, args)
                tools_called.append(tool_name)
                memory.record_step(
                    StepRecord(
                        step=step_num,
                        thought=thought,
                        action="tool",
                        tool_name=tool_name,
                        tool_args=args,
                        observation=observation,
                        failed=not ok,
                    )
                )
                continue

            # Unknown action — treat as a failed step with guidance
            memory.record_step(
                StepRecord(
                    step=step_num,
                    thought=thought,
                    action="tool",
                    failed=True,
                    observation=(
                        f"unknown action {action!r}. "
                        "Valid actions: tool, tools, plan, respond, ask."
                    ),
                )
            )

        # Budget exhausted — summarize what happened honestly
        summary = self._budget_summary(memory)
        return LoopResult(
            response=summary,
            steps_taken=self.step_budget,
            tools_called=tools_called,
            budget_exhausted=True,
            success=False,
            memory_snapshot=memory.to_dict(),
        )

    # ── think ────────────────────────────────────────────────────────

    def _relevance_query(self, memory: LoopMemory) -> str:
        """Build the query used to rank tools for this think step."""
        parts = [memory.user_message, memory.plan]
        for s in memory.steps[-2:]:
            parts.append(s.thought)
            if s.tool_name:
                parts.append(s.tool_name.replace("_", " "))
        return " ".join(p for p in parts if p)

    # ── code-first mode ──────────────────────────────────────────────

    def _code_adapter_lazy(self):
        if self._code_adapter is None:
            from .tools import CodeAdapter
            self._code_adapter = CodeAdapter(self.tools)
        return self._code_adapter

    @staticmethod
    def _extract_code(text):
        """Pull the python block out of model output."""
        m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        stripped = text.strip()
        if stripped and not stripped.startswith("{"):
            return stripped
        return None

    def _step_code(self, memory, step_num, tools_called):
        """One code-mode step. Returns (done, final_result_or_None)."""
        from .code_exec import SafeCodeRunner
        adapter = self._code_adapter_lazy()
        prompt = CODE_USER_TEMPLATE.format(
            tools=adapter.describe_code(),
            context=memory.render(),
        )
        try:
            resp = self.llm.chat(
                [Message.system(CODE_SYSTEM), Message.user(prompt)],
                self.sampling,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("code-mode think step %d failed: %s", step_num, exc)
            return False, None
        text = (resp.text or "").strip()
        code = self._extract_code(text) if text else None
        if not code:
            memory.record_step(
                StepRecord(step=step_num, thought="no code block produced",
                           action="tool", failed=True,
                           observation="output ONLY a python code block."))
            return False, None
        runner = SafeCodeRunner(adapter.namespace())
        ok, obs = runner.run(code)
        ns = adapter.namespace()
        called = [n for n in ns if n in code]
        tools_called.extend(n for n in called if n not in tools_called)
        memory.record_step(
            StepRecord(step=step_num, thought="code-first block",
                       action="tool", tool_name="code_block",
                       tool_args={"code": code[:500]},
                       observation=obs[:2000], failed=not ok))
        if obs.startswith("ASK:"):
            question = obs[4:].strip()
            return True, LoopResult(
                response=question, steps_taken=step_num,
                tools_called=tools_called, asked_user=True,
                question=question, memory_snapshot=memory.to_dict())
        if ok and not obs.startswith("(code ran, no result set"):
            return True, LoopResult(
                response=obs, steps_taken=step_num,
                tools_called=tools_called,
                memory_snapshot=memory.to_dict())
        return False, None

    def _think(self, memory: LoopMemory, step_num: int) -> dict[str, Any] | None:
        """One model call. Returns the parsed decision dict, or None."""
        prompt = THINK_USER_TEMPLATE.format(
            tools=self.tools.describe_for(self._relevance_query(memory)),
            context=memory.render(),
        )
        try:
            resp = self.llm.chat(
                [
                    Message.system(THINK_SYSTEM),
                    Message.user(prompt),
                ],
                self.sampling,
            )
        except Exception as exc:  # noqa: BLE001 — model failure is a loop event
            _log.warning("agentic think step %d: model call failed: %s", step_num, exc)
            return None

        text = (resp.text or "").strip()
        if not text:
            return None
        try:
            decision = extract_json(text)
        except Exception:  # noqa: BLE001
            # Last resort: find a {...} block manually
            decision = self._manual_json_extract(text)
        if not isinstance(decision, dict):
            return None
        return decision

    @staticmethod
    def _manual_json_extract(text: str) -> Any:
        # Strip markdown fences, then grab the first balanced {...}
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        start = cleaned.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except Exception:  # noqa: BLE001
                        return None
        return None

    # ── parallel calls ─────────────────────────────────────────────

    def _run_parallel_calls(
        self,
        memory: LoopMemory,
        step_num: int,
        thought: str,
        calls: list[Any],
        tools_called: list[str],
    ) -> None:
        """Dispatch a batch of independent tool calls in one budget step.

        Each call is validated (known tool, not a blind retry) and skipped
        individually on failure — one bad call doesn't sink the batch.
        Results are recorded as one step record per call so the model sees
        every observation.
        """
        from .tools import _MAX_PARALLEL_CALLS

        valid: list[tuple[str, dict[str, Any]]] = []
        for entry in calls[:_MAX_PARALLEL_CALLS]:
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool", "")).strip()
            args = entry.get("args")
            if not isinstance(args, dict):
                args = {}
            if not tool_name:
                continue
            if tool_name in memory.failed_tools() and self._same_args_as_failed(
                memory, tool_name, args
            ):
                memory.record_step(
                    StepRecord(
                        step=step_num,
                        thought=thought,
                        action="tools",
                        tool_name=tool_name,
                        tool_args=args,
                        failed=True,
                        observation=(
                            f"blocked: {tool_name} already failed with these "
                            "arguments — skipped in this batch."
                        ),
                    )
                )
                continue
            valid.append((tool_name, args))

        if len(calls) > _MAX_PARALLEL_CALLS:
            memory.record_step(
                StepRecord(
                    step=step_num,
                    thought=thought,
                    action="tools",
                    failed=True,
                    observation=(
                        f"batch capped at {_MAX_PARALLEL_CALLS} calls; "
                        f"{len(calls) - _MAX_PARALLEL_CALLS} dropped — "
                        "split large batches across steps."
                    ),
                )
            )
        if not valid:
            return

        results = self.tools.call_many(valid)
        for (tool_name, args), (ok, observation) in zip(valid, results):
            tools_called.append(tool_name)
            memory.record_step(
                StepRecord(
                    step=step_num,
                    thought=thought,
                    action="tools",
                    tool_name=tool_name,
                    tool_args=args,
                    observation=observation,
                    failed=not ok,
                )
            )

    # ── planned execution (escalation to MasterOrchestrator) ──────────

    def _run_planned_action(
        self,
        memory: LoopMemory,
        step_num: int,
        thought: str,
        decision: dict[str, Any],
        tools_called: list[str],
    ) -> None:
        """Execute a "plan" action via the PlannerBridge.

        The model decided the task needs structured orchestration. The
        bridge translates the goal (+ optional model-provided steps) into
        a MasterOrchestrator plan, runs it, and the result comes back as
        an observation the loop can build on. Reflection lessons are
        added to memory so future think steps benefit.
        """
        from .planner_bridge import PlannerBridge

        goal = str(decision.get("goal", "")).strip() or memory.user_message
        raw_steps = decision.get("steps")
        model_steps = raw_steps if isinstance(raw_steps, list) else None

        bridge = PlannerBridge(self.llm, self.tools)
        ok, observation, lessons = bridge.run_planned(
            goal,
            model_steps=model_steps,
            context_hint=memory.plan,
        )
        tools_called.append("plan")
        for lesson in lessons:
            memory.add_lesson(lesson)
        memory.record_step(
            StepRecord(
                step=step_num,
                thought=thought,
                action="plan",
                tool_name="plan",
                tool_args={"goal": goal[:200]},
                observation=observation,
                failed=not ok,
            )
        )

    @staticmethod
    def _same_args_as_failed(
        memory: LoopMemory, tool_name: str, args: dict[str, Any]
    ) -> bool:
        for s in memory.steps:
            if s.failed and s.tool_name == tool_name and s.tool_args == args:
                return True
        return False

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _budget_summary(memory: LoopMemory) -> str:
        done = [s for s in memory.steps if not s.failed]
        failed = [s for s in memory.steps if s.failed]
        parts = [
            "I ran out of steps before finishing.",
            f"Completed {len(done)} step(s), {len(failed)} failed.",
        ]
        if done:
            last_ok = [s for s in done if s.action == "tool"]
            if last_ok:
                s = last_ok[-1]
                obs = str(s.observation)[:400]
                parts.append(f"Last useful result ({s.tool_name}): {obs}")
        parts.append("Tell me how to proceed and I'll continue.")
        return " ".join(parts)


def run_agentic(
    message: str,
    *,
    llm: Any,
    registry: Any,
    history: list[tuple[str, str]] | None = None,
    step_budget: int | None = None,
    actor: str = "owner-loop",
    capabilities: Any = None,
    resume_from: dict[str, Any] | None = None,
    db: Any = None,
) -> LoopResult:
    """One-call convenience wrapper.

    Plugs into the owner-chat path as an alternative to the rigid intent
    router — pass the live LLM provider and the tool registry.

    ``resume_from``: a ``LoopResult.memory_snapshot`` from a previous run
    that paused with ``ask``. The new ``message`` is treated as the user's
    answer and the prior plan continues instead of starting fresh.

    ``db``: skill database for the Hermes distillation hook. When None
    the hook falls back to the app's default storage path.
    """
    tools = ToolAdapter(registry, actor=actor, capabilities=capabilities)
    memory = LoopMemory.from_dict(resume_from) if resume_from else None
    if memory is None:
        memory = LoopMemory(user_message=message)
    for role, text in history or []:
        memory.add_history(role, text)
    loop = AgenticLoop(llm, tools, step_budget=step_budget, db=db)
    return loop.run(message, memory)
