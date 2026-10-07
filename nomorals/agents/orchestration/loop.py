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
   {"thought": "<brief reasoning>", "action": "<tool|tools|respond|ask>", "tool": "<name>", "args": {…}, "calls": [{"tool": "<name>", "args": {…}}], "plan": "<your current plan>", "response": "<final answer or question>"}
2. action "tool": call ONE tool. Set "tool" to the exact tool name and "args" to its parameters object.
3. action "tools": call MULTIPLE INDEPENDENT tools at once. Set "calls" to a list of {"tool", "args"} objects. Use ONLY when the calls do not depend on each other's results — they run in parallel.
4. action "respond": you are done. Put your final answer to the owner in "response". Use this when you have what you need.
5. action "ask": you genuinely cannot proceed without the owner clarifying something. Put the question in "response". Use sparingly.
6. "plan": keep your running plan updated here (one or two sentences). It persists across steps and messages.
7. If a tool failed, try a DIFFERENT tool or approach. Never retry the identical call.
8. Chain tools: use one tool's output as the next tool's input.
9. Keep "thought" to one or two sentences.
10. Do not invent tools. Only use tools from the AVAILABLE TOOLS list.
11. Prefer fewer steps. If you can answer now, use "respond".
"""

THINK_USER_TEMPLATE = """AVAILABLE TOOLS (ranked by relevance to this task):
{tools}

{context}

Decide your next action. Output ONLY the JSON object."""


class AgenticLoop:
    """ReAct loop over Devon's tool registry."""

    def __init__(
        self,
        llm: Any,
        tools: ToolAdapter,
        *,
        step_budget: int = DEFAULT_STEP_BUDGET,
        sampling: SamplingParams | None = None,
    ) -> None:
        if step_budget < 1:
            raise ValueError("step_budget must be >= 1")
        self.llm = llm
        self.tools = tools
        self.step_budget = step_budget
        self.sampling = sampling or SamplingParams(temperature=0.2, max_tokens=1024)

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
                return LoopResult(
                    response=response or "(no response produced)",
                    steps_taken=step_num,
                    tools_called=tools_called,
                    memory_snapshot=memory.to_dict(),
                )

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
                        "Valid actions: tool, tools, respond, ask."
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

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _same_args_as_failed(
        memory: LoopMemory, tool_name: str, args: dict[str, Any]
    ) -> bool:
        for s in memory.steps:
            if s.failed and s.tool_name == tool_name and s.tool_args == args:
                return True
        return False

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
    step_budget: int = DEFAULT_STEP_BUDGET,
    actor: str = "owner-loop",
    capabilities: Any = None,
    resume_from: dict[str, Any] | None = None,
) -> LoopResult:
    """One-call convenience wrapper.

    Plugs into the owner-chat path as an alternative to the rigid intent
    router — pass the live LLM provider and the tool registry.

    ``resume_from``: a ``LoopResult.memory_snapshot`` from a previous run
    that paused with ``ask``. The new ``message`` is treated as the user's
    answer and the prior plan continues instead of starting fresh.
    """
    tools = ToolAdapter(registry, actor=actor, capabilities=capabilities)
    memory = LoopMemory.from_dict(resume_from) if resume_from else None
    if memory is None:
        memory = LoopMemory(user_message=message)
    for role, text in history or []:
        memory.add_history(role, text)
    loop = AgenticLoop(llm, tools, step_budget=step_budget)
    return loop.run(message, memory)
