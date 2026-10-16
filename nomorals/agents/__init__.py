"""L5 multi-agent orchestration."""

from __future__ import annotations

from .base import Agent, AgentResult, Budget
from .blackboard import Blackboard
from .context import AgentContext
from .orchestrator import MasterOrchestrator, Plan, Reevaluation
from .runtime import ExecutionReport, HybridExecutor
from .supervisor import Supervisor
from ..core.tasks import Task, TaskGraph, TaskKind, TaskState

__all__ = [
    "Agent",
    "AgentContext",
    "AgentResult",
    "Blackboard",
    "Budget",
    "ExecutionReport",
    "HybridExecutor",
    "MasterOrchestrator",
    "Plan",
    "Reevaluation",
    "Supervisor",
    "Task",
    "TaskGraph",
    "TaskKind",
    "TaskState",
]

# Tier 0: agents runtime → VCPU farm adapter (new file, additive).
from .runtime_vcpu import RuntimeVCPUAdapter

__all__ += ["RuntimeVCPUAdapter"]

# GROUPS Tier 0 — runtime profile envelope + autoscale (new file, additive).
from .runtime_profile import (
    RuntimeProfileManager,
    autoscale,
    hybrid_kwargs,
    process_workers,
    shed_level,
    thread_workers,
    workers_for,
)

__all__ += [
    "RuntimeProfileManager",
    "autoscale",
    "hybrid_kwargs",
    "process_workers",
    "shed_level",
    "thread_workers",
    "workers_for",
]
