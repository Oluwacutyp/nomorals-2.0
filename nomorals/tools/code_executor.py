"""Plan mode and act mode for the coding agent.

Two execution modes:
- PLAN MODE: Agent analyzes the task and produces a plan, but doesn't execute.
  User reviews and approves before any changes are made.
- ACT MODE: Agent executes changes directly, with optional confirmation gates.

Usage:
    executor = CodeExecutor(agent, project_root="/path/to/repo")
    
    # Plan mode - see what would happen
    plan = await executor.plan("Add user authentication with JWT")
    print(plan.summary)
    for step in plan.steps:
        print(f"  {step.action}: {step.target}")
    
    # Execute plan
    if input("Execute plan? ").lower() == "y":
        result = await executor.execute_plan(plan)
    
    # Act mode - just do it
    result = await executor.act("Add user authentication with JWT")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..core.logging_setup import get_logger

# Deferred: EditLoop is instantiated in __init__ only, and EditResult /
# TestResult appear only in annotations. Importing edit_loop at module top
# pulled `nomorals.agents.coding` (via the old eager import) during tool
# registration at boot.
if TYPE_CHECKING:  # pragma: no cover
    from ..tools.edit_loop import EditLoop, EditResult, TestResult

__all__ = [
    "CodeExecutor",
    "ExecutionMode",
    "PlanStep",
    "ExecutionPlan",
    "ExecutionResult",
    "StepResult",
    "ReplanDecision",
]

_log = get_logger(__name__)


class ExecutionMode(str, Enum):
    """Execution mode for code operations."""
    
    PLAN = "plan"  # Analyze and plan, don't execute
    ACT = "act"    # Execute directly
    
    @classmethod
    def from_str(cls, s: str) -> ExecutionMode:
        if s.lower() in ("plan", "planning", "preview", "dry-run"):
            return cls.PLAN
        return cls.ACT


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    
    step_id: str
    action: str  # create_file, edit_file, delete_file, run_command, run_tests
    target: str  # File path or command
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    estimated_risk: str = "low"  # low, medium, high
    # What the step should produce - the executor and replanner judge
    # completion against this, not just "the command exited 0".
    expected_output: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "target": self.target,
            "description": self.description,
            "estimated_risk": self.estimated_risk,
            "expected_output": self.expected_output,
        }

    @property
    def signature(self) -> str:
        """Stable identity for loop detection."""
        return f"{self.action}:{self.target}"


@dataclass
class ExecutionPlan:
    """A plan for executing a code change."""
    
    plan_id: str
    goal: str
    summary: str
    steps: list[PlanStep] = field(default_factory=list)
    test_command: str = ""
    rollback_steps: list[PlanStep] = field(default_factory=list)
    estimated_time: str = ""
    risk_assessment: str = ""
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "test_command": self.test_command,
            "estimated_time": self.estimated_time,
            "risk_assessment": self.risk_assessment,
        }
    
    @property
    def step_count(self) -> int:
        return len(self.steps)
    
    @property
    def high_risk_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.estimated_risk == "high"]


@dataclass
class StepResult:
    """Result of executing a single plan step (LangGraph plan-and-execute)."""
    
    step_id: str
    action: str = ""
    target: str = ""
    description: str = ""
    output: str = ""
    success: bool = True
    error: str = ""
    duration: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "target": self.target,
            "description": self.description,
            "output": self.output[:2000],
            "success": self.success,
            "error": self.error,
            "duration": round(self.duration, 2),
        }


@dataclass
class ReplanDecision:
    """The replanner's verdict after a step: continue, replan, or finish."""
    
    action: str = "continue"  # continue | replan | finish
    revised_steps: list["PlanStep"] = field(default_factory=list)
    reasoning: str = ""
    final_answer: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "revised_steps": [s.to_dict() for s in self.revised_steps],
            "reasoning": self.reasoning,
            "final_answer": self.final_answer,
        }


@dataclass
class ExecutionResult:
    """Result of executing a plan or action."""
    
    success: bool
    goal: str
    steps_completed: int = 0
    steps_total: int = 0
    edit_results: list[EditResult] = field(default_factory=list)
    test_result: Optional[TestResult] = None
    error: str = ""
    duration: float = 0.0
    changes_made: list[str] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    replans: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal": self.goal,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "test_passed": self.test_result.passed if self.test_result else None,
            "error": self.error,
            "changes_made": self.changes_made,
            "steps": [s.to_dict() for s in self.step_results],
            "replans": self.replans,
        }


class CodeExecutor:
    """Plan/act executor for coding tasks."""
    
    def __init__(
        self,
        agent: Any,
        *,
        project_root: str = ".",
        mode: ExecutionMode = ExecutionMode.ACT,
        max_iterations: int = 3,
        auto_test: bool = True,
    ) -> None:
        self.agent = agent
        self.project_root = Path(project_root).resolve()
        self.mode = mode
        from ..tools.edit_loop import EditLoop  # deferred: see module docstring note
        self.edit_loop = EditLoop(
            agent,
            project_root=project_root,
            max_iterations=max_iterations,
        )
        self.auto_test = auto_test
        _log.info(f"CodeExecutor initialized in {mode.value} mode")
    
    async def plan(self, goal: str, *, context_files: list[str] | None = None) -> ExecutionPlan:
        """Create an execution plan for a goal.
        
        Args:
            goal: What to accomplish (natural language)
            context_files: Files to include as context
            
        Returns:
            ExecutionPlan with steps
        """
        from ..core.ids import new_id
        
        # Build project context (Phase D: ranked by relevance to the goal)
        project_context = await self._gather_project_context(
            context_files, task=goal)
        
        prompt = f"""You are a coding agent planning a code change.

PROJECT ROOT: {self.project_root}

PROJECT STRUCTURE:
{project_context.get('structure', 'Unknown')}

RELEVANT FILES:
{project_context.get('files', 'None provided')}

GOAL: {goal}

Create a step-by-step plan to accomplish this goal. For each step, specify:
1. action: create_file, edit_file, delete_file, run_command, or run_tests
2. target: file path or command
3. description: what this step does
4. estimated_risk: low, medium, or high

Also suggest a test command to verify the changes.

Return your plan as JSON:
```json
{{
    "summary": "Brief summary of the plan",
    "steps": [
        {{
            "action": "edit_file",
            "target": "src/auth.py",
            "description": "Add JWT validation middleware",
            "estimated_risk": "medium"
        }}
    ],
    "test_command": "pytest tests/test_auth.py -x",
    "estimated_time": "5 minutes",
    "risk_assessment": "Medium risk - touches authentication code"
}}
```
"""
        
        response = self.agent.chat(prompt)
        
        # Parse response
        import json
        import re
        
        json_match = re.search(r"```json\s*(.*?)\s*```", response.content, re.DOTALL)
        if json_match:
            plan_data = json.loads(json_match.group(1))
        else:
            # Fallback: create a simple plan
            plan_data = {
                "summary": response.content[:200],
                "steps": [{"action": "edit_file", "target": "", "description": goal, "estimated_risk": "medium"}],
                "test_command": "",
            }
        
        steps = []
        for i, step_data in enumerate(plan_data.get("steps", [])):
            steps.append(PlanStep(
                step_id=f"step_{i+1}",
                action=step_data.get("action", "edit_file"),
                target=step_data.get("target", ""),
                description=step_data.get("description", ""),
                estimated_risk=step_data.get("estimated_risk", "medium"),
            ))
        
        # Generate rollback steps
        rollback_steps = self._generate_rollback_steps(steps)
        
        return ExecutionPlan(
            plan_id=new_id("plan"),
            goal=goal,
            summary=plan_data.get("summary", ""),
            steps=steps,
            test_command=plan_data.get("test_command", ""),
            rollback_steps=rollback_steps,
            estimated_time=plan_data.get("estimated_time", ""),
            risk_assessment=plan_data.get("risk_assessment", ""),
        )
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        *,
        confirm_step: Any = None,
        max_replans: int = 2,
    ) -> ExecutionResult:
        """Execute an execution plan (plan-and-execute with replanning).

        The executor runs one step at a time and records a StepResult per
        step. When a step fails, the replanner is asked whether to continue,
        revise the remaining plan, or finish early - instead of dying at
        the first failure. Loop detection aborts when the same step fails
        three times in a row (the plan is broken, not the step).

        Args:
            plan: The plan to execute
            confirm_step: Optional callback to confirm each step
            max_replans: How many times the replanner may revise the plan

        Returns:
            ExecutionResult with per-step results and replan count
        """
        start = time.time()
        result = ExecutionResult(
            success=True,
            goal=plan.goal,
            steps_total=plan.step_count,
        )

        _log.info(f"Executing plan: {plan.plan_id} ({plan.step_count} steps)")

        remaining: list[PlanStep] = list(plan.steps)
        fail_streak: dict[str, int] = {}
        replans_used = 0

        while remaining:
            step = remaining.pop(0)
            idx = result.steps_completed
            _log.info(f"Step {idx+1}: {step.action} {step.target}")

            # Confirm step if callback provided
            if confirm_step:
                approved = await confirm_step(step)
                if not approved:
                    result.step_results.append(StepResult(
                        step_id=step.step_id, action=step.action,
                        target=step.target, description=step.description,
                        success=False, error="rejected by user"))
                    result.error = f"Step {step.step_id} rejected by user"
                    result.success = False
                    break

            step_result = await self._execute_step(step, plan)
            result.step_results.append(step_result)
            if getattr(step_result, "test_result", None) is not None:
                result.test_result = step_result.test_result
            if step_result.success:
                result.steps_completed += 1
                fail_streak.pop(step.signature, None)
                continue

            # --- failure path: loop detection, then the replanner ---
            streak = fail_streak.get(step.signature, 0) + 1
            fail_streak[step.signature] = streak
            if streak >= 3:
                result.success = False
                result.error = (
                    f"loop detected: {step.signature} failed {streak}x in a "
                    f"row ({step_result.error}). The plan is broken, not the "
                    "step - revise the goal or the plan.")
                _log.error(result.error)
                break

            if replans_used >= max_replans:
                result.success = False
                result.error = (f"Step {step.step_id} failed "
                                f"({step_result.error}); replan budget "
                                f"({max_replans}) exhausted")
                break

            decision = await self.replan(plan, result.step_results, remaining)
            replans_used += 1
            result.replans = replans_used
            _log.info("replan %d/%d: %s - %s", replans_used, max_replans,
                      decision.action, decision.reasoning[:120])
            if decision.action == "finish":
                result.success = False
                result.error = (decision.final_answer or
                                f"replanner gave up after {step.step_id}: "
                                f"{step_result.error} ({decision.reasoning})")
                break
            if decision.action == "replan" and decision.revised_steps:
                remaining = list(decision.revised_steps)
                _log.info("plan revised: %d remaining steps",
                          len(remaining))
            # "continue" falls through: retry the remaining plan as-is
            # (the failed step stays failed in the record; the next step
            # may still succeed - e.g. independent steps).

        # Run final tests if we have a test command and no test step ran
        if result.success and plan.test_command and not result.test_result:
            result.test_result = await self.edit_loop.run_tests(plan.test_command)
            if not result.test_result.passed:
                result.success = False
                result.error = "Final tests failed"

        # Automatic rollback on failure: undo the steps that ran, in reverse.
        if not result.success and plan.rollback_steps:
            _log.warning("plan failed (%s) - running %d rollback steps",
                         result.error, len(plan.rollback_steps))
            rollback_errors = []
            for rb in plan.rollback_steps:
                try:
                    if rb.action == "delete_file":
                        await self._delete_file(rb.target)
                    elif rb.action == "restore_file":
                        from ..tools.git import restore as _git_restore

                        _git_restore(rb.target, str(self.project_root))
                    elif rb.action == "run_command":
                        await self.edit_loop.run_tests(rb.target)
                    else:
                        _log.warning("no rollback handler for action %r", rb.action)
                except Exception as e:  # noqa: BLE001 - best-effort rollback
                    rollback_errors.append(f"{rb.step_id}: {e}")
            if rollback_errors:
                result.error += f" | rollback issues: {'; '.join(rollback_errors)}"

        # A failed FINAL step is a failed plan: the replanner's
        # "continue" only redeems mid-plan failures (later independent
        # steps may still succeed).
        if (result.success and result.step_results
                and not result.step_results[-1].success):
            last = result.step_results[-1]
            result.success = False
            result.error = (f"Step {last.step_id} failed ({last.error}); "
                            "no recovery")

        result.duration = time.time() - start
        result.steps_total = len(result.step_results)
        _log.info("Plan execution complete: %d/%d steps (%d replans)",
                  result.steps_completed, result.steps_total, result.replans)
        return result

    async def _execute_step(self, step: PlanStep,
                            plan: ExecutionPlan) -> StepResult:
        """Run one plan step; never raises - failures become StepResult."""
        started = time.time()
        sr = StepResult(step_id=step.step_id, action=step.action,
                        target=step.target, description=step.description)
        try:
            if step.action == "edit_file":
                edit_result = await self.edit_loop.edit_file(
                    step.target, step.description)
                if not edit_result.success:
                    sr.success = False
                    sr.error = f"Edit failed: {edit_result.error}"
                else:
                    sr.output = f"Edited: {step.target}"
            elif step.action == "create_file":
                await self._create_file(step.target, step.description)
                sr.output = f"Created: {step.target}"
            elif step.action == "delete_file":
                await self._delete_file(step.target)
                sr.output = f"Deleted: {step.target}"
            elif step.action == "restore_file":
                from ..tools.git import restore as _git_restore

                _git_restore(step.target, str(self.project_root))
                sr.output = f"Restored: {step.target}"
            elif step.action == "run_command":
                cmd_result = await self.edit_loop.run_tests(step.target)
                sr.output = (cmd_result.output or "")[:2000]
                if not cmd_result.passed:
                    sr.success = False
                    sr.error = f"Command failed: {step.target}"
            elif step.action == "run_tests":
                test_result = await self.edit_loop.run_tests(
                    step.target or plan.test_command)
                sr.test_result = test_result  # surfaced on ExecutionResult
                sr.output = (test_result.output or "")[:2000]
                if not test_result.passed:
                    sr.success = False
                    sr.error = "Tests failed"
            else:
                sr.success = False
                sr.error = f"unknown step action {step.action!r}"
        except Exception as e:  # noqa: BLE001 - step failures are results
            sr.success = False
            sr.error = str(e)
        sr.duration = time.time() - started
        return sr

    async def replan(self, plan: ExecutionPlan,
                     completed: list[StepResult],
                     remaining: list[PlanStep]) -> ReplanDecision:
        """Ask the agent whether to continue, revise, or finish.

        The replanner sees the goal, what each finished step produced,
        and the remaining steps, and returns a ReplanDecision. A broken
        or unparseable answer degrades to "continue" (never crash the
        run on the replanner itself).
        """
        import json as _json
        import re as _re

        done = "\n".join(
            f"- {s.step_id} [{s.action} {s.target}]: "
            f"{'OK' if s.success else 'FAILED: ' + s.error}"
            + (f" -> {s.output[:200]}" if s.success and s.output else "")
            for s in completed)
        todo = "\n".join(
            f"- {s.step_id} [{s.action} {s.target}]: {s.description}"
            for s in remaining)
        prompt = f"""You are replanning a coding task after a step failed.

GOAL: {plan.goal}

COMPLETED STEPS:
{done or "(none)"}

REMAINING STEPS:
{todo or "(none)"}

Decide: "continue" (keep the remaining plan as-is), "replan" (revise the
remaining steps to route around the failure), or "finish" (the goal is
unreachable - stop now). When replanning, emit the FULL revised remaining
step list (same shape as a plan: action, target, description,
estimated_risk, expected_output).

Reply as JSON only:
```json
{{"action": "continue|replan|finish",
  "reasoning": "why",
  "revised_steps": [{{"action": "edit_file", "target": "x.py",
                      "description": "...", "estimated_risk": "low",
                      "expected_output": "..."}}],
  "final_answer": "only when action=finish"}}
```"""
        try:
            response = self.agent.chat(prompt)
            content = getattr(response, "content", str(response))
            match = _re.search(r"```json\s*(.*?)\s*```", content, _re.DOTALL)
            data = _json.loads(match.group(1) if match else content)
        except Exception as e:  # noqa: BLE001 - replanner never sinks the run
            _log.warning("replan parse failed (%s); continuing", e)
            return ReplanDecision(action="continue",
                                  reasoning="replan output unparseable")
        action = str(data.get("action", "continue")).lower()
        if action not in ("continue", "replan", "finish"):
            action = "continue"
        revised: list[PlanStep] = []
        for i, sd in enumerate(data.get("revised_steps") or []):
            revised.append(PlanStep(
                step_id=f"replan_{i+1}",
                action=sd.get("action", "edit_file"),
                target=sd.get("target", ""),
                description=sd.get("description", ""),
                estimated_risk=sd.get("estimated_risk", "medium"),
                expected_output=sd.get("expected_output", ""),
            ))
        return ReplanDecision(action=action, revised_steps=revised,
                              reasoning=str(data.get("reasoning", "")),
                              final_answer=str(data.get("final_answer", "")))

    def synthesize(self, plan: ExecutionPlan,
                   result: ExecutionResult) -> str:
        """Combine step results into the final narrative (the Synthesizer).

        Pure function - no model call. The executor already recorded what
        happened; this renders it as the human-readable outcome.
        """
        from . import _style as _style

        lines = [_style.banner(f"Plan: {plan.goal}")]
        for sr in result.step_results:
            status = _style.OK if sr.success else _style.FAIL
            detail = sr.output or sr.error or sr.description
            lines.append(_style.status_line(
                status, f"{sr.step_id} [{sr.action} {sr.target}] {detail}"[:160]))
        verdict = ("SUCCEEDED" if result.success else "FAILED")
        lines.append("")
        lines.append(_style.status_line(
            _style.OK if result.success else _style.FAIL,
            f"{verdict}: {result.steps_completed}/{len(result.step_results)} "
            f"steps ok"
            + (f" ({result.replans} replans)" if result.replans else "")
            + (f" in {result.duration:.1f}s" if result.duration else "")))
        if result.error:
            lines.append(_style.status_line(_style.WARN, result.error[:300]))
        return "\n".join(lines)

    async def act(
        self,
        goal: str,
        *,
        context_files: list[str] | None = None,
        test_command: str = "",
    ) -> ExecutionResult:
        """Execute a goal directly in act mode.
        
        Args:
            goal: What to accomplish
            context_files: Files for context
            test_command: Tests to run after
            
        Returns:
            ExecutionResult
        """
        if self.mode == ExecutionMode.PLAN:
            _log.warning("Called act() in plan mode, switching to plan-then-execute")
            plan = await self.plan(goal, context_files=context_files)
            return await self.execute_plan(plan)
        
        # Create and execute plan inline
        plan = await self.plan(goal, context_files=context_files)
        
        if test_command:
            plan.test_command = test_command
        
        return await self.execute_plan(plan)
    
    class _RegistrySymbolSearch:
        """Symbol search over the registry's Phase-A code index.

        Plain ``__call__`` runs one query (warming the index first);
        ``search_many`` fires the index warm-up together with all initial
        queries in a single bounded ``call_many`` block (Phase D) instead
        of warm-then-serial-searches.  Every step is best-effort — ranking
        also works from targets, test files, and import neighbors when
        the index is unavailable.
        """

        def __init__(self, tools: Any, root: str | Path) -> None:
            self._tools = tools
            self._root = str(root)
            self._warmed = False

        def _warm(self) -> None:
            if self._warmed:
                return
            try:
                self._tools.call_many(
                    [("index_repo", {"path": self._root})], max_workers=1)
            except Exception as exc:  # noqa: BLE001 — warm-up is advisory
                _log.debug("code index warm-up failed: %s", exc)
            self._warmed = True

        @staticmethod
        def _paths(out: Any) -> list[str]:
            if not getattr(out, "ok", False):
                return []
            paths: list[str] = []
            for r in (out.value or {}).get("results", []):
                f = r.get("file")
                if f:
                    paths.append(f)
            return paths

        def __call__(self, query: str, limit: int = 10) -> list[str]:
            self._warm()
            try:
                out = self._tools.call("search_code", query=query,
                                       limit=limit)
            except Exception:  # noqa: BLE001 — search is advisory
                return []
            return self._paths(out)

        def search_many(self, queries: list[str],
                        limit: int = 10) -> dict[str, list[str]]:
            """Warm the index and run every query in one parallel block."""
            calls = [("index_repo", {"path": self._root})]
            calls += [("search_code", {"query": q, "limit": limit})
                      for q in queries]
            try:
                results = self._tools.call_many(
                    calls, max_workers=min(8, len(calls)))
            except Exception as exc:  # noqa: BLE001 — advisory
                _log.debug("parallel index warm + search failed: %s", exc)
                return {}
            self._warmed = True
            return {q: self._paths(res)
                    for q, res in zip(queries, results[1:])}

    def _symbol_search_fn(self) -> Any:
        """Build a symbol-search helper over the registry's ``search_code``
        tool (Phase A code index), or None when no registry is wired."""
        agent = self.agent
        context = getattr(agent, "context", None)
        tools = getattr(context, "tools", None)
        if tools is None or not hasattr(tools, "call_many"):
            return None
        return self._RegistrySymbolSearch(tools, self.project_root)

    async def _gather_project_context(
        self,
        context_files: list[str] | None = None,
        task: str = "",
        *,
        ranked: bool = True,
        budget_tokens: int = 12000,
    ) -> dict[str, str]:
        """Gather project context for planning.

        Phase D: ranked assembly under a token budget by default
        (``ranked=False`` keeps the old truncation for debugging).
        """
        if not ranked:
            from .repo_context import legacy_context

            return legacy_context(self.project_root, context_files)
        from .repo_context import rank_context

        ctx = rank_context(
            self.project_root,
            task or " ".join(context_files or []),
            targets=context_files,
            budget_tokens=budget_tokens,
            symbol_search=self._symbol_search_fn(),
        )
        files_block = "\n\n".join(
            f"--- {rel} ---\n{content}"
            for rel, content in ctx.files.items()
        )
        return {
            "structure": ctx.structure,
            "files": files_block,
            "metadata": ctx.metadata_json(),
        }

    async def _create_file(self, file_path: str, description: str) -> None:
        """Create a new file with LLM-generated content."""
        prompt = f"""Create a new file at `{file_path}`.

Description: {description}

Return the complete file contents in a code block:
```
<file contents>
```
"""
        response = self.agent.chat(prompt)
        
        # Extract code
        import re
        match = re.search(r"```\w*\n(.*?)```", response.content, re.DOTALL)
        content = match.group(1).strip() if match else response.content.strip()
        
        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        _log.info(f"Created file: {file_path}")
    
    async def _delete_file(self, file_path: str) -> None:
        """Delete a file."""
        full_path = self.project_root / file_path
        if full_path.exists():
            full_path.unlink()
            _log.info(f"Deleted file: {file_path}")
    
    @staticmethod
    def _generate_rollback_steps(steps: list[PlanStep]) -> list[PlanStep]:
        """Generate rollback steps for a plan.

        File restores go through the git tool (``restore_file`` action,
        handled in :meth:`execute_plan`) — never raw shell strings.
        """
        rollback = []

        for i, step in enumerate(reversed(steps)):
            if step.action == "create_file":
                rollback.append(PlanStep(
                    step_id=f"rollback_{i+1}",
                    action="delete_file",
                    target=step.target,
                    description=f"Delete created file: {step.target}",
                ))
            elif step.action == "edit_file":
                rollback.append(PlanStep(
                    step_id=f"rollback_{i+1}",
                    action="restore_file",
                    target=step.target,
                    description=f"Restore committed state: {step.target}",
                ))

        return rollback


# ── registry hook ──────────────────────────────────────────────────────────

_PLAN_ACTIONS = frozenset({
    "create_file", "edit_file", "delete_file", "restore_file",
    "run_command", "run_tests",
})


def _run_async(coro: Any) -> Any:
    """Drive a coroutine from sync tool code, even inside a running loop."""
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _parse_plan(plan_json: str) -> ExecutionPlan:
    """Parse and validate a JSON execution plan against the PlanStep schema."""
    import json as _json

    from ..core.errors import ValidationError
    from ..core.ids import new_id

    try:
        data = _json.loads(plan_json)
    except Exception as exc:
        raise ValidationError(f"plan_json is not valid JSON: {exc}", field="plan_json") from exc
    if not isinstance(data, dict):
        raise ValidationError("plan must be a JSON object", field="plan_json")
    steps_data = data.get("steps")
    if not isinstance(steps_data, list) or not steps_data:
        raise ValidationError("plan.steps must be a non-empty list", field="plan_json")

    steps = []
    for i, raw in enumerate(steps_data):
        if not isinstance(raw, dict):
            raise ValidationError(f"step {i} must be an object", field="plan_json")
        action = raw.get("action", "")
        target = raw.get("target", "")
        if action not in _PLAN_ACTIONS:
            raise ValidationError(
                f"step {i}: unknown action {action!r} "
                f"(allowed: {sorted(_PLAN_ACTIONS)})", field="plan_json")
        if not isinstance(target, str) or not target.strip():
            raise ValidationError(f"step {i}: target must be a non-empty string",
                                  field="plan_json")
        risk = raw.get("estimated_risk", "medium")
        if risk not in ("low", "medium", "high"):
            raise ValidationError(f"step {i}: bad estimated_risk {risk!r}",
                                  field="plan_json")
        steps.append(PlanStep(
            step_id=raw.get("step_id", f"step_{i + 1}"),
            action=action,
            target=target.strip(),
            description=raw.get("description", ""),
            estimated_risk=risk,
        ))

    plan = ExecutionPlan(
        plan_id=data.get("plan_id", new_id("plan")),
        goal=data.get("goal", ""),
        summary=data.get("summary", ""),
        steps=steps,
        test_command=data.get("test_command", ""),
    )
    plan.rollback_steps = CodeExecutor._generate_rollback_steps(steps)
    return plan


def register(registry: Any) -> None:
    """Attach the plan executor to a registry."""
    from ..core.errors import ToolError
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "execute_plan",
        description=("Execute a validated JSON coding plan (create/edit/delete "
                     "files, run commands/tests) with automatic rollback on "
                     "step failure."),
        capability=Capability.EXEC_CODE,
    )
    def execute_plan(plan_json: str) -> dict[str, Any]:
        from ..agents.coding import CodingAgent

        plan = _parse_plan(plan_json)
        try:
            agent = CodingAgent(context)
        except Exception as exc:
            raise ToolError(f"cannot build coding agent from context: {exc}") from exc
        settings = getattr(context, "settings", None) if context is not None else None
        root = str(settings.workspace_dir) if settings is not None else "."
        executor = CodeExecutor(agent, project_root=root)
        result = _run_async(executor.execute_plan(plan))
        return result.to_dict()
