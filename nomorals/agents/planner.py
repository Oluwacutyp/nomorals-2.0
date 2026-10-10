"""Agent Planning Engine - Intelligent goal decomposition and execution.

This is the brain that ties all integrations together. It takes natural language
goals and breaks them down into executable multi-step plans using available
integrations (email, calendar, shopping, smart home, etc.).

Features:
- Natural language goal parsing
- LLM-powered plan generation
- Automatic integration selection
- Parallel task execution
- Progress tracking and status updates
- Failure recovery and retry logic
- Human-in-the-loop for sensitive operations

Usage:
    from nomorals.agents.planner import AgentPlanner
    
    planner = AgentPlanner(
        email_integration=email,
        calendar_integration=calendar,
        shopping_integration=shopping,
        llm_router=router,
    )
    
    # Execute a complex goal
    result = await planner.execute(
        goal="Plan a dinner party for 6 people this Saturday",
        account="bot@gmail.com",
        on_progress=lambda step, progress: print(f"{step}: {progress}%"),
    )
    
    # Check status
    status = planner.get_status(task_id)
    print(status.progress, status.current_step)
    
    # Cancel if needed
    await planner.cancel(task_id)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from ..core.error_intelligence import ErrorIntelligence, catch_and_analyze
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..search.adaptive import adaptive_result_limit
from ..integrations.calendar_integration import CalendarIntegration
from ..integrations.email_integration import EmailIntegration
from ..integrations.shopping_integration import ShoppingIntegration

__all__ = [
    "AgentPlanner",
    "Plan",
    "PlanStep",
    "PlanResult",
    "PlanStatus",
    "TaskExecution",
]

_log = get_logger(__name__)


class PlanStatus(str, Enum):
    """Status of a plan execution."""
    
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PlanStep:
    """One step in a plan."""
    
    step_id: str
    name: str
    description: str
    action: str  # e.g., "send_email", "add_calendar_event", "search_products"
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # Step IDs this depends on
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class Plan:
    """A decomposed goal with executable steps."""
    
    plan_id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0  # 0-100
    current_step: Optional[str] = None
    error: Optional[str] = None
    #: WHY the plan degraded ("" when the model planned it) — the template
    #: fallback is a degradation and must never look like a clean success.
    plan_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "current_step": self.current_step,
            "error": self.error,
            "plan_error": self.plan_error,
        }


@dataclass
class PlanResult:
    """Result of plan execution."""
    
    plan_id: str
    success: bool
    steps_completed: int
    steps_total: int
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    summary: str = ""
    #: WHY the plan degraded ("" when the model planned it) — the template
    #: fallback is a degradation and must never look like a clean success.
    plan_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "results": self.results,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "summary": self.summary,
            "plan_error": self.plan_error,
        }


@dataclass
class TaskExecution:
    """Tracks execution of a single task."""
    
    task_id: str
    plan: Plan
    on_progress: Optional[Callable[[str, float], None]] = None
    on_step_complete: Optional[Callable[[str, Any], None]] = None
    cancelled: bool = False


class AgentPlanner:
    """Intelligent goal decomposition and execution engine.
    
    Takes natural language goals and executes them using available integrations.
    """
    
    def __init__(
        self,
        email_integration: Optional[EmailIntegration] = None,
        calendar_integration: Optional[CalendarIntegration] = None,
        shopping_integration: Optional[ShoppingIntegration] = None,
        llm_router: Any = None,
    ) -> None:
        self.email = email_integration
        self.calendar = calendar_integration
        self.shopping = shopping_integration
        self.llm = llm_router
        self.error_intel = ErrorIntelligence()
        
        # Active executions
        self._executions: dict[str, TaskExecution] = {}
        
        # Action handlers
        self._action_handlers: dict[str, Callable] = {
            "send_email": self._action_send_email,
            "read_inbox": self._action_read_inbox,
            "search_emails": self._action_search_emails,
            "add_calendar_event": self._action_add_calendar_event,
            "list_calendar_events": self._action_list_calendar_events,
            "search_products": self._action_search_products,
            "compare_prices": self._action_compare_prices,
            "wait": self._action_wait,
        }
        
        _log.info("Agent Planner initialized with integrations")
    
    async def execute(
        self,
        goal: str,
        *,
        account: str,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_step_complete: Optional[Callable[[str, Any], None]] = None,
        step_timeout: float = 300.0,
        max_retries: int = 2,
    ) -> PlanResult:
        """Execute a goal by decomposing and running a plan.
        
        Args:
            goal: Natural language goal description
            account: Primary account to use
            on_progress: Callback for progress updates (step_name, progress_percent)
            on_step_complete: Callback when step completes (step_name, result)
            
        Returns:
            PlanResult with execution details
        """
        task_id = new_id("task")
        start_time = time.time()
        
        # Generate plan
        plan = await self._generate_plan(goal, account)
        
        # Create execution tracker
        execution = TaskExecution(
            task_id=task_id,
            plan=plan,
            on_progress=on_progress,
            on_step_complete=on_step_complete,
        )
        self._executions[task_id] = execution
        
        try:
            # Execute plan
            plan.status = PlanStatus.EXECUTING
            plan.started_at = time.time()
            
            # validate before touching anything (no mid-flight surprises)
            problems = self.validate_plan(plan)
            if problems:
                plan.status = PlanStatus.FAILED
                plan.completed_at = time.time()
                plan.error = "; ".join(problems[:5])
                _log.error(f"Plan invalid, not executed: {plan.error}")
                return PlanResult(
                    plan_id=plan.plan_id,
                    success=False,
                    steps_completed=0,
                    steps_total=len(plan.steps),
                    results={},
                    errors=[f"invalid plan: {p}" for p in problems],
                    duration_seconds=time.time() - start_time,
                    summary="plan failed validation and was not executed: "
                            + "; ".join(problems[:3]),
                    plan_error=plan.plan_error,
                )

            results = {}
            errors = []
            completed = 0
            total = len(plan.steps)

            async def _run_one(step: PlanStep) -> tuple[PlanStep, bool]:
                """Run one step with retry + timeout. Returns (step, ok)."""
                nonlocal completed
                if execution.cancelled:
                    step.status = "skipped"
                    step.error = "cancelled"
                    return step, False
                if not self._check_dependencies(step, results):
                    step.status = "skipped"
                    step.error = "dependency failed"
                    return step, False
                plan.current_step = step.name
                step.status = "running"
                step.started_at = time.time()
                last_exc: Exception | None = None
                for attempt in range(max(1, max_retries) + 1):
                    try:
                        if step_timeout and step_timeout > 0:
                            result = await asyncio.wait_for(
                                self._execute_step(step, account),
                                timeout=step_timeout)
                        else:
                            result = await self._execute_step(step, account)
                        step.result = result
                        step.status = "completed"
                        step.completed_at = time.time()
                        results[step.step_id] = result
                        completed += 1
                        if on_step_complete:
                            on_step_complete(step.name, result)
                        plan.progress = (completed / total) * 100 if total else 100.0
                        if on_progress:
                            on_progress(step.name, plan.progress)
                        _log.info(f"Step completed: {step.name}")
                        return step, True
                    except Exception as e:  # noqa: BLE001 - a step failing is data
                        last_exc = e
                        analysis = self.error_intel.analyze(
                            e, context={"step": step.name,
                                        "action": step.action})
                        retryable = bool(getattr(analysis, "retryable", False))
                        if retryable and attempt < max(1, max_retries):
                            delay = min(30.0, 2.0 ** attempt)
                            _log.warning(
                                f"Step {step.name} failed (retryable), "
                                f"retry {attempt + 1} in {delay:.0f}s: "
                                f"{getattr(analysis, 'explanation', e)}")
                            await asyncio.sleep(delay)
                            continue
                        step.status = "failed"
                        step.error = getattr(analysis, "explanation", str(e))
                        step.completed_at = time.time()
                        errors.append(f"{step.name}: {step.error}")
                        # non-retryable failures halt later batches (below)
                        step.non_retryable_fail = not retryable
                        _log.error(f"Step failed: {step.name} - {step.error}")
                        return step, False
                step.status = "failed"
                step.error = str(last_exc) if last_exc else "unknown"
                step.completed_at = time.time()
                errors.append(f"{step.name}: {step.error}")
                return step, False

            for batch in self.execution_batches(plan):
                if execution.cancelled:
                    break
                # independent steps in one batch run in parallel
                await asyncio.gather(*(_run_one(s) for s in batch))
                # a failed non-retryable step halts later batches that
                # depend on it; unrelated later work still runs (skip-on-
                # failed-deps is checked per step inside _run_one)
                if any(getattr(s, "non_retryable_fail", False)
                       for s in batch):
                    # non-retryable failure: stop launching new batches
                    for s in plan.steps:
                        if s.status == "pending":
                            s.status = "skipped"
                            s.error = ("halted after non-retryable "
                                       "step failure")
                    break
            
            # Finalize
            plan.status = PlanStatus.COMPLETED if not errors else PlanStatus.FAILED
            plan.completed_at = time.time()
            duration = plan.completed_at - start_time
            
            result = PlanResult(
                plan_id=plan.plan_id,
                success=not errors,
                steps_completed=completed,
                steps_total=len(plan.steps),
                results=results,
                errors=errors,
                duration_seconds=duration,
                summary=self._generate_summary(plan, results, errors),
                plan_error=plan.plan_error,
            )
            
            _log.info(f"Plan completed: {completed}/{len(plan.steps)} steps in {duration:.1f}s")
            return result
            
        finally:
            # Cleanup
            del self._executions[task_id]
    
    async def _generate_plan(self, goal: str, account: str) -> Plan:
        """Generate a plan from a goal using LLM, with an EXPLICIT template
        fallback: whenever the model path fails or is missing, the plan
        carries plan_error so a template plan never looks like a clean
        model-made success."""
        plan_id = new_id("plan")

        plan_error = ""
        if self.llm:
            plan_data, err = await self._llm_generate_plan(goal, account)
            if plan_data is None:
                plan_error = f"{err} — fell back to template plan"
                plan_data = self._template_generate_plan(goal, account)
        else:
            plan_error = "no LLM router configured — used template plan"
            plan_data = self._template_generate_plan(goal, account)

        # Convert to Plan object
        steps = []
        for step_data in plan_data.get("steps", []):
            steps.append(PlanStep(
                step_id=step_data.get("step_id", new_id("step")),
                name=step_data["name"],
                description=step_data["description"],
                action=step_data["action"],
                parameters=step_data.get("parameters", {}),
                depends_on=step_data.get("depends_on", []),
            ))

        return Plan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            plan_error=plan_error,
        )

    async def _llm_generate_plan(
        self, goal: str, account: str
    ) -> tuple[dict[str, Any] | None, str]:
        """Use LLM to generate a plan.

        Returns (plan_data, error). plan_data is None when the model path
        failed — the caller picks the template fallback and records
        plan_error explicitly. Never falls back silently.
        """
        # Build prompt
        available_actions = list(self._action_handlers.keys())
        
        prompt = f"""You are an AI agent that breaks down goals into actionable steps.

Goal: {goal}
Account: {account}

Available actions: {', '.join(available_actions)}

Generate a plan with steps. Each step should have:
- name: Short name
- description: What this step does
- action: One of the available actions
- parameters: Dict of parameters for the action
- depends_on: List of step IDs this depends on (optional)

Return JSON with format:
{{
  "steps": [
    {{
      "step_id": "step_1",
      "name": "Check calendar",
      "description": "Check if Saturday is free",
      "action": "list_calendar_events",
      "parameters": {{"days_ahead": 7}},
      "depends_on": []
    }}
  ]
}}

Generate the plan:"""
        
        try:
            response = await self.llm.generate(prompt, max_tokens=2000)
        except Exception as e:
            _log.error(f"LLM plan generation failed: {e}")
            return None, f"LLM plan generation failed: {e}"

        # Parse JSON from response
        json_match = response.content.find("{")
        if json_match < 0:
            return None, "LLM plan generation failed: no JSON in reply"
        json_str = response.content[json_match:]
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(json_str):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_str = json_str[:i+1]
                    break
        try:
            return json.loads(json_str), ""
        except Exception as e:
            return None, f"LLM plan generation failed: bad JSON ({e})"

    def _template_generate_plan(self, goal: str, account: str) -> dict[str, Any]:
        """Template-based plan generation (fallback)."""
        goal_lower = goal.lower()
        
        # Pattern matching for common goals
        if "dinner party" in goal_lower or "party" in goal_lower:
            return {
                "steps": [
                    {
                        "step_id": "step_1",
                        "name": "Check calendar",
                        "description": "Check if the date is available",
                        "action": "list_calendar_events",
                        "parameters": {"days_ahead": 7},
                        "depends_on": [],
                    },
                    {
                        "step_id": "step_2",
                        "name": "Send invitations",
                        "description": "Send email invitations to guests",
                        "action": "send_email",
                        "parameters": {
                            "subject": "You're invited to a dinner party!",
                            "body": "Join us for dinner this weekend!",
                        },
                        "depends_on": ["step_1"],
                    },
                ]
            }
        
        elif "shop" in goal_lower or "buy" in goal_lower or "purchase" in goal_lower:
            return {
                "steps": [
                    {
                        "step_id": "step_1",
                        "name": "Search products",
                        "description": "Find products matching the goal",
                        "action": "search_products",
                        "parameters": {"query": goal},
                        "depends_on": [],
                    },
                    {
                        "step_id": "step_2",
                        "name": "Compare prices",
                        "description": "Find the best price",
                        "action": "compare_prices",
                        "parameters": {"query": goal},
                        "depends_on": ["step_1"],
                    },
                ]
            }
        
        else:
            # Generic single-step plan
            return {
                "steps": [
                    {
                        "step_id": "step_1",
                        "name": "Execute goal",
                        "description": f"Work on: {goal}",
                        "action": "wait",
                        "parameters": {"seconds": 1},
                        "depends_on": [],
                    }
                ]
            }
    
    def validate_plan(self, plan: Plan) -> list[str]:
        """Validate a plan BEFORE execution (never mid-flight surprises).

        Checks: every ``depends_on`` id exists, no dependency cycles,
        every action has a handler, no duplicate step ids.  Returns a
        list of problem strings (empty = valid).  Never raises.
        """
        problems: list[str] = []
        try:
            ids = [s.step_id for s in plan.steps]
            seen: set[str] = set()
            for sid in ids:
                if sid in seen:
                    problems.append(f"duplicate step id: {sid}")
                seen.add(sid)
            idset = set(ids)
            for step in plan.steps:
                for dep in step.depends_on:
                    if dep not in idset:
                        problems.append(
                            f"step '{step.name}' depends on unknown "
                            f"step id '{dep}'")
                    if dep == step.step_id:
                        problems.append(
                            f"step '{step.name}' depends on itself")
                if step.action not in self._action_handlers:
                    problems.append(
                        f"step '{step.name}' uses unknown action "
                        f"'{step.action}'")
            # cycle detection (DFS on the dependency graph)
            graph = {s.step_id: [d for d in s.depends_on if d in idset]
                     for s in plan.steps}
            color: dict[str, int] = {}

            def _dfs(nid: str, stack: list[str]) -> bool:
                color[nid] = 1
                stack.append(nid)
                for dep in graph.get(nid, []):
                    if color.get(dep) == 1:
                        problems.append(
                            "dependency cycle: "
                            + " -> ".join(stack + [dep]))
                        return True
                    if color.get(dep) is None and _dfs(dep, stack):
                        return True
                stack.pop()
                color[nid] = 2
                return False

            for nid in graph:
                if color.get(nid) is None:
                    _dfs(nid, [])
        except Exception as exc:  # noqa: BLE001 - never raises
            problems.append(f"validation broke: {exc}")
        return problems

    def execution_batches(self, plan: Plan) -> list[list[PlanStep]]:
        """Topological batches: steps in one batch are independent and
        run in parallel; batches run in order.  Never raises (a cycle
        degrades to one step per batch in original order).
        """
        try:
            idset = {s.step_id for s in plan.steps}
            indeg = {s.step_id: 0 for s in plan.steps}
            followers: dict[str, list[str]] = {s.step_id: [] for s in plan.steps}
            for s in plan.steps:
                for dep in s.depends_on:
                    if dep in idset and dep != s.step_id:
                        indeg[s.step_id] += 1
                        followers[dep].append(s.step_id)
            by_id = {s.step_id: s for s in plan.steps}
            order_index = {s.step_id: i for i, s in enumerate(plan.steps)}
            remaining = set(idset)
            batches: list[list[PlanStep]] = []
            while remaining:
                ready = sorted((nid for nid in remaining if indeg[nid] == 0),
                               key=lambda nid: order_index[nid])
                if not ready:  # cycle: break it by original order
                    nid = min(remaining, key=lambda x: order_index[x])
                    ready = [nid]
                batches.append([by_id[nid] for nid in ready])
                for nid in ready:
                    remaining.discard(nid)
                    for f in followers[nid]:
                        indeg[f] -= 1
            return batches
        except Exception:  # noqa: BLE001 - never raises
            _log.debug("execution_batches failed", exc_info=True)
            return [[s] for s in plan.steps]

    def _check_dependencies(self, step: PlanStep, results: dict[str, Any]) -> bool:
        """Check if all dependencies are satisfied."""
        for dep_id in step.depends_on:
            if dep_id not in results:
                return False
        return True
    
    async def _execute_step(self, step: PlanStep, account: str) -> Any:
        """Execute a single step."""
        handler = self._action_handlers.get(step.action)
        
        if not handler:
            raise ValueError(f"Unknown action: {step.action}")
        
        # Inject account into parameters
        params = step.parameters.copy()
        if "account" not in params:
            params["account"] = account
        
        return await handler(**params)
    
    # ── Action Handlers ──────────────────────────────────────────────────────
    
    async def _action_send_email(self, **kwargs) -> dict[str, Any]:
        """Send an email."""
        if not self.email:
            raise RuntimeError("Email integration not available")
        
        to = kwargs.get("to", kwargs.get("recipient"))
        subject = kwargs.get("subject", "Message from bot")
        body = kwargs.get("body", "")
        account = kwargs["account"]
        
        message_id = await self.email.send(
            to=to,
            subject=subject,
            body=body,
            account=account,
        )
        
        return {"message_id": message_id, "sent": True}
    
    async def _action_read_inbox(self, **kwargs) -> dict[str, Any]:
        """Read inbox."""
        if not self.email:
            raise RuntimeError("Email integration not available")
        
        account = kwargs["account"]
        limit = kwargs.get("limit", 10)
        
        messages = await self.email.read_inbox(account=account, limit=limit)
        
        return {
            "count": len(messages),
            "messages": [m.to_dict() for m in messages],
        }
    
    async def _action_search_emails(self, **kwargs) -> dict[str, Any]:
        """Search emails."""
        if not self.email:
            raise RuntimeError("Email integration not available")
        
        query = kwargs.get("query", "")
        account = kwargs["account"]
        limit = kwargs.get("limit", 10)
        
        messages = await self.email.search(query=query, account=account, limit=limit)
        
        return {
            "count": len(messages),
            "messages": [m.to_dict() for m in messages],
        }
    
    async def _action_add_calendar_event(self, **kwargs) -> dict[str, Any]:
        """Add calendar event."""
        if not self.calendar:
            raise RuntimeError("Calendar integration not available")
        
        title = kwargs.get("title", "Event")
        start = kwargs["start"]
        end = kwargs["end"]
        account = kwargs["account"]
        
        event_id = await self.calendar.add_event(
            title=title,
            start=start,
            end=end,
            account=account,
            description=kwargs.get("description", ""),
            location=kwargs.get("location", ""),
        )
        
        return {"event_id": event_id, "created": True}
    
    async def _action_list_calendar_events(self, **kwargs) -> dict[str, Any]:
        """List calendar events."""
        if not self.calendar:
            raise RuntimeError("Calendar integration not available")
        
        account = kwargs["account"]
        days_ahead = kwargs.get("days_ahead", 7)
        
        events = await self.calendar.list_events(account=account, days_ahead=days_ahead)
        
        return {
            "count": len(events),
            "events": [e.to_dict() for e in events],
        }
    
    async def _action_search_products(self, **kwargs) -> dict[str, Any]:
        """Search products."""
        if not self.shopping:
            raise RuntimeError("Shopping integration not available")
        
        query = kwargs.get("query", "")
        # adaptive breadth when the caller didn't pick a count
        max_results = (kwargs.get("max_results")
                       or adaptive_result_limit(query, base=10,
                                                floor=5, ceiling=20))
        
        products = await self.shopping.search(query=query, max_results=max_results)
        
        return {
            "count": len(products),
            "products": [p.to_dict() for p in products],
        }
    
    async def _action_compare_prices(self, **kwargs) -> dict[str, Any]:
        """Compare prices."""
        if not self.shopping:
            raise RuntimeError("Shopping integration not available")
        
        query = kwargs.get("query", "")
        
        comparison = await self.shopping.compare_prices(query=query)
        
        return comparison.to_dict()
    
    async def _action_wait(self, **kwargs) -> dict[str, Any]:
        """Wait for specified time."""
        seconds = kwargs.get("seconds", 1)
        await asyncio.sleep(seconds)
        return {"waited": seconds}
    
    # ── Utility Methods ──────────────────────────────────────────────────────
    
    def _generate_summary(
        self,
        plan: Plan,
        results: dict[str, Any],
        errors: list[str],
    ) -> str:
        """Generate human-readable summary of plan execution."""
        if not errors:
            text = f"✅ Successfully completed all {len(plan.steps)} steps!"
        else:
            completed = len(plan.steps) - len(errors)
            text = (f"⚠️ Completed {completed}/{len(plan.steps)} steps. "
                    f"Errors: {'; '.join(errors[:3])}")
        # a template plan is a degradation — say so in the summary, not just
        # on the field, so nobody mistakes it for a model-made plan.
        if plan.plan_error:
            text = f"⚠️ template plan ({plan.plan_error}). {text}"
        return text
    
    def get_status(self, task_id: str) -> Optional[Plan]:
        """Get status of an active task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Plan object or None if not found
        """
        execution = self._executions.get(task_id)
        return execution.plan if execution else None
    
    async def cancel(self, task_id: str) -> bool:
        """Cancel an active task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled successfully
        """
        execution = self._executions.get(task_id)
        if not execution:
            return False
        
        execution.cancelled = True
        execution.plan.status = PlanStatus.CANCELLED
        _log.info(f"Cancelled task: {task_id}")
        
        return True
    
    def list_active_tasks(self) -> list[Plan]:
        """List all active tasks.
        
        Returns:
            List of Plan objects
        """
        return [exec.plan for exec in self._executions.values()]
    
    def register_action(self, name: str, handler: Callable) -> None:
        """Register a custom action handler.
        
        Args:
            name: Action name
            handler: Async function that handles the action
        """
        self._action_handlers[name] = handler
        _log.info(f"Registered custom action: {name}")


def register(registry: Any) -> None:
    """Expose the AgentPlanner as an agent tool for goal decomposition."""

    @registry.register(
        "plan_goal",
        description=(
            "Break a natural-language goal into an executable multi-step plan "
            "and run it. Uses available integrations (email, calendar, shopping). "
            "Returns the plan and execution results."
        ),
        capability="agent.plan",
        parameters={
            "goal": "str — natural language goal (e.g. 'Plan a dinner party for 6')",
            "account": "str — account identifier for integrations (optional)",
        },
    )
    def _plan_goal(goal: str, account: str = "") -> dict[str, Any]:
        import asyncio

        context = registry.context
        llm = getattr(context, "llm_router", None) or getattr(context, "llm", None)
        planner = AgentPlanner(llm_router=llm)
        try:
            result = asyncio.run(planner.execute(goal=goal, account=account or None))
            return {
                "ok": True,
                "status": str(result.status) if hasattr(result, "status") else "done",
                "result": result.to_dict() if hasattr(result, "to_dict") else str(result),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
