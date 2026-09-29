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
from typing import Any, Optional

from ..core.logging_setup import get_logger
from ..tools.edit_loop import EditLoop, EditResult, TestResult

__all__ = [
    "CodeExecutor",
    "ExecutionMode",
    "PlanStep",
    "ExecutionPlan",
    "ExecutionResult",
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
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "target": self.target,
            "description": self.description,
            "estimated_risk": self.estimated_risk,
        }


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
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal": self.goal,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "test_passed": self.test_result.passed if self.test_result else None,
            "error": self.error,
            "changes_made": self.changes_made,
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
        
        # Build project context
        project_context = await self._gather_project_context(context_files)
        
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
    ) -> ExecutionResult:
        """Execute an execution plan.
        
        Args:
            plan: The plan to execute
            confirm_step: Optional callback to confirm each step
            
        Returns:
            ExecutionResult with status
        """
        start = time.time()
        result = ExecutionResult(
            success=True,
            goal=plan.goal,
            steps_total=plan.step_count,
        )
        
        _log.info(f"Executing plan: {plan.plan_id} ({plan.step_count} steps)")
        
        for i, step in enumerate(plan.steps):
            _log.info(f"Step {i+1}/{plan.step_count}: {step.action} {step.target}")
            
            # Confirm step if callback provided
            if confirm_step:
                approved = await confirm_step(step)
                if not approved:
                    result.error = f"Step {i+1} rejected by user"
                    result.success = False
                    break
            
            try:
                if step.action == "edit_file":
                    edit_result = await self.edit_loop.edit_file(
                        step.target,
                        step.description,
                    )
                    result.edit_results.append(edit_result)
                    
                    if not edit_result.success:
                        result.success = False
                        result.error = f"Edit failed: {edit_result.error}"
                        break
                    
                    result.changes_made.append(f"Edited: {step.target}")
                
                elif step.action == "create_file":
                    await self._create_file(step.target, step.description)
                    result.changes_made.append(f"Created: {step.target}")
                
                elif step.action == "delete_file":
                    await self._delete_file(step.target)
                    result.changes_made.append(f"Deleted: {step.target}")
                
                elif step.action == "run_command":
                    cmd_result = await self.edit_loop.run_tests(step.target)
                    if not cmd_result.passed:
                        _log.warning(f"Command failed: {step.target}")
                
                elif step.action == "run_tests":
                    result.test_result = await self.edit_loop.run_tests(
                        step.target or plan.test_command
                    )
                    if not result.test_result.passed:
                        result.success = False
                        result.error = "Tests failed"
                        break
                
                result.steps_completed = i + 1
                
            except Exception as e:
                _log.error(f"Step {i+1} failed: {e}")
                result.success = False
                result.error = str(e)
                break
        
        # Run final tests if we have a test command and no test step
        if result.success and plan.test_command and not result.test_result:
            result.test_result = await self.edit_loop.run_tests(plan.test_command)
            if not result.test_result.passed:
                result.success = False
                result.error = "Final tests failed"
        
        result.duration = time.time() - start
        _log.info(f"Plan execution complete: {result.steps_completed}/{result.steps_total} steps")
        
        return result
    
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
    
    async def _gather_project_context(
        self,
        context_files: list[str] | None = None,
    ) -> dict[str, str]:
        """Gather project context for planning."""
        context: dict[str, str] = {}
        
        # Project structure
        try:
            files = []
            for path in self.project_root.rglob("*"):
                if path.is_file() and not any(
                    skip in str(path)
                    for skip in [".git", "__pycache__", "node_modules", ".venv"]
                ):
                    rel = path.relative_to(self.project_root)
                    files.append(str(rel))
                    if len(files) > 100:  # Limit
                        break
            context["structure"] = "\n".join(sorted(files))
        except Exception:
            context["structure"] = "Unable to read project structure"
        
        # Context files
        if context_files:
            file_contents = []
            for f in context_files:
                full_path = self.project_root / f
                if full_path.exists():
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    file_contents.append(f"--- {f} ---\n{content[:2000]}")  # Limit
            context["files"] = "\n\n".join(file_contents)
        
        return context
    
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
    
    def _generate_rollback_steps(self, steps: list[PlanStep]) -> list[PlanStep]:
        """Generate rollback steps for a plan."""
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
                    action="run_command",
                    target=f"git checkout -- {step.target}",
                    description=f"Restore original: {step.target}",
                ))
        
        return rollback
