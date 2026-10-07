"""Plan mode for the coding agent — plan → approve → execute.

For non-trivial coding tasks (new files > 2, cross-module, or > 4 files
total), the agent produces a plan artifact FIRST and waits for the
owner's approval before touching anything.  The executor must then stay
within the approved scope — touching an unplanned file triggers a
re-plan instead of a silent scope creep.

Flow::

    result = agent.run(task, plan_mode="auto")   # or True / False
    if result.needs_approval:
        show(result.plan_text)                   # "approve?" in chat
        # on approval:
        result = agent.execute_plan(result.plan_id)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["CodePlan", "is_complex", "render_plan", "PlanStore"]


#: complexity thresholds
_MAX_AUTO_FILES = 4
_MAX_AUTO_NEW_FILES = 2
_MAX_AUTO_MODULES = 2


@dataclass
class CodePlan:
    id: str
    task: str
    files: list[dict[str, Any]] = field(default_factory=list)
    approach: str = ""
    risks: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    approved: bool = False
    approved_at: float = 0.0

    @property
    def paths(self) -> list[str]:
        return [str(f.get("path", "")) for f in self.files if f.get("path")]

    def in_scope(self, rel: str) -> bool:
        return str(rel) in self.paths


def is_complex(file_specs: list[dict[str, Any]]) -> bool:
    """Classify a file plan as complex (needs approval)."""
    if len(file_specs) > _MAX_AUTO_FILES:
        return True
    new_files = sum(1 for s in file_specs if s.get("new_file"))
    if new_files > _MAX_AUTO_NEW_FILES:
        return True
    modules = set()
    for s in file_specs:
        parts = str(s.get("path", "")).split("/")
        if len(parts) > 1:
            modules.add(parts[0])
        else:
            modules.add(".")
    return len(modules) > _MAX_AUTO_MODULES


def render_plan(plan: CodePlan) -> str:
    """Human-readable plan for the approval prompt."""
    lines = [f"📋 PLAN — {plan.task[:120]}", ""]
    lines.append("FILES:")
    for spec in plan.files:
        tag = "new" if spec.get("new_file") else "edit"
        why = str(spec.get("why", ""))[:100]
        lines.append(f"  [{tag}] {spec.get('path')} — {why}")
    if plan.approach:
        lines.append("")
        lines.append("APPROACH:")
        lines.append(f"  {plan.approach[:500]}")
    if plan.risks:
        lines.append("")
        lines.append("RISKS:")
        for r in plan.risks[:5]:
            lines.append(f"  ⚠️ {r[:120]}")
    lines.append("")
    lines.append("reply `approve` to execute, or describe changes.")
    return "\n".join(lines)


class PlanStore:
    """In-process store for pending/approved plans."""

    _plans: dict[str, CodePlan] = {}

    @classmethod
    def save(cls, plan: CodePlan) -> str:
        cls._plans[plan.id] = plan
        return plan.id

    @classmethod
    def get(cls, plan_id: str) -> CodePlan | None:
        return cls._plans.get(plan_id)

    @classmethod
    def approve(cls, plan_id: str) -> CodePlan | None:
        plan = cls._plans.get(plan_id)
        if plan is None:
            return None
        plan.approved = True
        plan.approved_at = time.time()
        _log.info("plan %s approved", plan_id)
        return plan

    @classmethod
    def new(cls, task: str, files: list[dict[str, Any]],
            approach: str = "", risks: list[str] | None = None) -> CodePlan:
        plan = CodePlan(
            id=uuid.uuid4().hex[:12],
            task=task,
            files=files,
            approach=approach,
            risks=risks or [],
        )
        cls.save(plan)
        return plan
