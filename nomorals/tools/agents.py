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
from types import SimpleNamespace
from typing import Any

from ..core.errors import CapabilityDenied, ToolNotFound
from ..core.policy import Capability, CapabilitySet
from ..core.result import Err

__all__ = ["register", "AGENT_TOOL_MODULES", "agent_for",
           "CODING_TOOLS", "CODING_CAPABILITIES", "CodingRoleAgent",
           "RoleScopedRegistry"]

_log = logging.getLogger(__name__)

# Modules under ``nomorals.agents`` that expose ``register(registry)``.
# Dotted names stay under nomorals.agents too ("search.engine" →
# ``nomorals.agents.search.engine``); a module living elsewhere in the
# package (like "memory.persona") is found via the nomorals.* fallback in
# register() below.
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
    "morning_briefing",
    "osint_graph",
    "projects",
    "reasoning",
    "reflection",
    "research_swarm",
    "router_select",
    "search.engine",
    "simulation",
    "skill_canary",
    "skill_evolution",
    "skill_synthesis",
    "skills",
    "structuring",
    "toolmaker",
    "watchers",
    "memory.persona",
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
            # most modules live under nomorals.agents (dotted names like
            # "search.engine" too); a few live elsewhere in the package
            # (e.g. "memory.persona" → nomorals.memory.persona)
            try:
                module = importlib.import_module(f"nomorals.agents.{name}")
            except ModuleNotFoundError:
                module = importlib.import_module(f"nomorals.{name}")
            hook = getattr(module, "register", None)
            if hook is None:
                continue
            hook(registry)
        except Exception as exc:  # noqa: BLE001 — one bad module ≠ bad registry
            _log.debug("agent tool module %r failed to register: %s", name, exc)


# ── Phase C: coding role handler with locked allowlist ─────────────────────
# The orchestrator offers "coding" as a plan role and delegates through
# ``context.tools.agent_for(role)``.  Before Phase C nothing answered, so
# coding-role tasks fell through to "no handler".  The handler below binds
# the role to a deny-by-default tool surface: exactly the tools a coding
# loop needs, enforced through the registry's capability system.

#: Exact tool surface for the ``coding`` role.  Nothing else — no browser,
#: no finance, no messaging, no network tools.
CODING_TOOLS: tuple[str, ...] = (
    "fs_read",
    "edit_file",
    "apply_patch",
    "shell_run",     # sandboxed
    "run_tests",     # Phase B pytest runner
    "lint",
    "git_status",
    "git_diff",
    "search_code",
    "index_repo",    # Phase D: warm the code index the coding loop searches
    "repo_map",      # structural repo map (tools/repo_index)
    "symbol_search", # ranked symbol search / who-imports / callers
)

#: Capability grant covering exactly the allowlisted tools' capabilities:
#: fs.read (fs_read, git_status, git_diff, repo_map, symbol_search),
#: fs.write (edit_file, apply_patch),
#: exec.shell (shell_run, run_tests, lint),
#: mem.read (search_code — reads the code index).
#: The name check in RoleScopedRegistry still denies every non-allowlisted
#: tool, whatever its capability.
CODING_CAPABILITIES: CapabilitySet = CapabilitySet.of(
    Capability.FS_READ,
    Capability.FS_WRITE,
    Capability.EXEC_SHELL,
    Capability.MEM_READ,
)


class RoleScopedRegistry:
    """Deny-by-default view over a ToolRegistry for one agent role.

    Name check first (anything not on the allowlist is refused, even if
    its capability would otherwise be granted), then the narrowed
    capability grant flows through the registry's own enforcement in
    :meth:`ToolRegistry.call`.  A refusal names the missing capability,
    e.g. ``role 'coding' may not call 'browser': capability
    'net.browser' is not granted to this role``.
    """

    def __init__(self, inner: Any, role: str,
                 allowlist: tuple[str, ...],
                 grant: CapabilitySet) -> None:
        self._inner = inner
        self._role = role
        self._allowlist = tuple(allowlist)
        self._grant = grant

    @property
    def context(self) -> Any:
        return getattr(self._inner, "context", None)

    def names(self) -> list[str]:
        return sorted(n for n in self._inner.names() if n in self._allowlist)

    def get(self, name: str) -> Any:
        spec = self._inner.get(name)
        return spec if spec is not None and name in self._allowlist else None

    def schemas(self) -> list[dict[str, Any]]:
        return [s for s in self._inner.schemas(capabilities=self._grant)
                if s.get("name") in self._allowlist]

    def prompt_listing(self) -> str:
        lines = []
        for schema in self.schemas():
            params = ", ".join(schema.get("parameters", ()))
            lines.append(f"- {schema['name']}({params}): "
                         f"{schema.get('description', '')}")
        return "\n".join(lines)

    def call(self, name: str, /, *args: Any, actor: str = "coding",
             capabilities: CapabilitySet | None = None,
             **kwargs: Any) -> Any:
        """Call an allowlisted tool; refuse anything else with a clear,
        capability-naming error."""
        spec = self._inner.get(name)
        if spec is None:
            return Err(ToolNotFound(f"unknown tool {name!r}"))
        if name not in self._allowlist:
            return Err(CapabilityDenied(
                f"role {self._role!r} may not call {name!r}: capability "
                f"{getattr(spec, 'capability', '?')!r} is not granted to "
                f"this role",
                capability=getattr(spec, "capability", ""),
                actor=actor,
            ))
        grant = (self._grant if capabilities is None
                 else self._grant.intersect(capabilities))
        return self._inner.call(name, *args, actor=actor,
                                capabilities=grant, **kwargs)

    def call_many(self, calls: list[tuple[str, dict[str, Any]]],
                  *, max_workers: int = 1, actor: str = "coding",
                  **common: Any) -> list[Any]:
        """Parallel dispatch honoring the role allowlist (Phase D).

        Pre-filters to allowlisted tools so a batch never leaks a denied
        call into the inner registry's parallel path; refusals come back
        as ``Err(CapabilityDenied)`` outcomes in position."""
        scoped: list[tuple[str, dict[str, Any]]] = []
        refused: dict[int, Any] = {}
        for i, (name, kwargs) in enumerate(calls):
            spec = self._inner.get(name)
            if spec is None:
                refused[i] = Err(ToolNotFound(f"unknown tool {name!r}"))
            elif name not in self._allowlist:
                refused[i] = Err(CapabilityDenied(
                    f"role {self._role!r} may not call {name!r}: capability "
                    f"{getattr(spec, 'capability', '?')!r} is not granted to "
                    f"this role",
                    capability=getattr(spec, "capability", ""),
                    actor=actor,
                ))
            else:
                scoped.append((name, kwargs))
        results = self._inner.call_many(
            scoped, max_workers=max_workers, actor=actor,
            capabilities=self._grant, **common)
        out: list[Any] = []
        it = iter(results)
        for i in range(len(calls)):
            out.append(refused[i] if i in refused else next(it))
        return out


class CodingRoleAgent:
    """The ``coding``-role agent: Phase B's CodingAgent bound to the
    locked tool surface.  ``run(payload)`` matches the orchestrator's
    ``agent.run(task.payload).output`` contract."""

    role = "coding"

    def __init__(self, context: Any, registry: Any) -> None:
        self.context = context
        self.tools = RoleScopedRegistry(registry, "coding",
                                        CODING_TOOLS, CODING_CAPABILITIES)

    def tool_names(self) -> list[str]:
        return self.tools.names()

    def run(self, payload: dict[str, Any] | None) -> Any:
        from ..agents.coding import CodingAgent

        payload = payload or {}
        task_text = (payload.get("task") or payload.get("goal") or "").strip()
        root = payload.get("root")
        agent = (CodingAgent(self.context, root=root) if root
                 else CodingAgent(self.context))
        result = agent.run(
            task_text or "(no task text)",
            max_iterations=int(payload.get("max_iterations", 5)),
            timeout=float(payload.get("timeout", 120)),
        )
        return SimpleNamespace(
            output={
                "ok": result.ok,
                "files": result.files,
                "summary": result.output[-2000:],
                "error": result.error,
                "review": getattr(result, "review", {}),
            },
            ok=result.ok,
        )


def agent_for(registry: Any, role: str) -> Any | None:
    """Resolve an orchestrator plan role to a bound agent.

    Returns None for unknown roles — the orchestrator then falls through
    to "no handler", exactly as before Phase C.
    """
    if role == "coding":
        return CodingRoleAgent(getattr(registry, "context", None), registry)
    return None
