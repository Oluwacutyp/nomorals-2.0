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

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
    """Pending/approved plans, shared across processes.

    Plans live in memory for the session AND are persisted to a JSON
    file (``NM_PLAN_STORE``, default ``data/code-plans.json``) so a plan
    created by one ``nm code`` invocation can be approved and executed
    by a later one (``nm code approve <plan-id>``).  Only the most
    recent plans are kept — the file is pruned on every save.
    """

    _plans: dict[str, CodePlan] = {}
    _loaded: bool = False
    _KEEP = 50

    @classmethod
    def _path(cls) -> Path:
        return Path(os.environ.get("NM_PLAN_STORE", "data/code-plans.json"))

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        try:
            raw = cls._path().read_text(encoding="utf-8")
        except OSError:
            return
        try:
            rows = json.loads(raw)
        except ValueError:
            _log.warning("plan store file is not valid JSON — starting fresh")
            return
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict) or "id" not in row:
                continue
            try:
                plan = CodePlan(
                    id=str(row["id"]),
                    task=str(row.get("task", "")),
                    files=list(row.get("files") or []),
                    approach=str(row.get("approach", "")),
                    risks=list(row.get("risks") or []),
                    created_at=float(row.get("created_at", time.time())),
                    approved=bool(row.get("approved", False)),
                    approved_at=float(row.get("approved_at", 0.0)),
                )
            except (TypeError, ValueError):
                continue
            cls._plans[plan.id] = plan

    @classmethod
    def _persist(cls) -> None:
        try:
            path = cls._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [asdict(p) for p in
                    sorted(cls._plans.values(),
                           key=lambda p: p.created_at)[-cls._KEEP:]]
            path.write_text(json.dumps(rows, ensure_ascii=False),
                            encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 — persistence is best-effort
            _log.warning("could not persist plan store: %s", exc)

    @classmethod
    def save(cls, plan: CodePlan) -> str:
        cls._ensure_loaded()
        cls._plans[plan.id] = plan
        cls._persist()
        return plan.id

    @classmethod
    def get(cls, plan_id: str) -> CodePlan | None:
        cls._ensure_loaded()
        return cls._plans.get(plan_id)

    @classmethod
    def approve(cls, plan_id: str) -> CodePlan | None:
        cls._ensure_loaded()
        plan = cls._plans.get(plan_id)
        if plan is None:
            return None
        plan.approved = True
        plan.approved_at = time.time()
        _log.info("plan %s approved", plan_id)
        cls._persist()
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
