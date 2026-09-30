"""Agent-tool bridge.

Thin delegation layer: the canonical tool wrappers live with their agent
modules under ``nomorals/agents/`` (one home per concept). This module
re-exports them into the tool registry so ``ToolRegistry.register_builtins()``
wires the full agent surface — goal, project, skill, kg, improve,
model_route, failure_analyze, simulate, tool_create, reflection, autonomy, …
— without duplicating (or drifting from) their implementations.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

__all__ = ["register", "AGENT_TOOL_MODULES"]

_log = logging.getLogger(__name__)

# Modules under ``nomorals.agents`` that expose ``register(registry)``.
AGENT_TOOL_MODULES = (
    "benchmark",
    "cipher",
    "cognition",
    "decoder",
    "evolution",
    "failure",
    "goals",
    "improvement",
    "investigate",
    "kg",
    "mission",
    "monitor",
    "osint_graph",
    "projects",
    "reasoning",
    "reflection",
    "research_swarm",
    "router_select",
    "search.engine",
    "simulation",
    "skills",
    "structuring",
    "toolmaker",
)


def register(registry: Any) -> None:
    """Attach the agent-module tools to a registry.

    Each agent module owns its tool's real API; we import and call its
    ``register`` hook. A module that fails to import or register (optional
    dependency, unsupported context) is skipped with a debug note — the
    registry stays usable.
    """
    for name in AGENT_TOOL_MODULES:
        try:
            module = importlib.import_module(f"nomorals.agents.{name}")
            hook = getattr(module, "register", None)
            if hook is None:
                continue
            hook(registry)
        except Exception as exc:  # noqa: BLE001 — one bad module ≠ bad registry
            _log.debug("agent tool module %r failed to register: %s", name, exc)
