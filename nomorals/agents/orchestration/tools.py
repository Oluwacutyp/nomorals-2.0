"""Tool adapter — exposes Devon's registry tools to the agentic loop.

Does NOT create new tools. Wraps the existing ToolRegistry so the loop's
think step can see what's available and the act step can call them with
capability gating and audit trail intact.
"""

from __future__ import annotations

import json
from typing import Any

from ...core.policy import CapabilitySet
from ...core.result import Outcome


class ToolAdapter:
    """Bridge between the ReAct loop and Devon's ToolRegistry.

    - ``describe()`` → compact tool listing for the think prompt.
    - ``call(name, args)`` → registry call with audit + capability checks.
    """

    def __init__(
        self,
        registry: Any,
        *,
        actor: str = "owner-loop",
        capabilities: CapabilitySet | None = None,
        max_tools_in_prompt: int = 40,
    ) -> None:
        self.registry = registry
        self.actor = actor
        self.capabilities = capabilities or CapabilitySet.all()
        self.max_tools_in_prompt = max_tools_in_prompt

    # ── for the think prompt ─────────────────────────────────────────

    def describe(self) -> str:
        """Compact tool listing, formatted for weaker models.

        One line per tool: name(params): description. Capped so the
        prompt stays small.
        """
        try:
            schemas = self.registry.schemas(capabilities=self.capabilities)
        except Exception:
            schemas = []
        lines = []
        for schema in schemas[: self.max_tools_in_prompt]:
            params = schema.get("parameters") or {}
            if isinstance(params, dict):
                param_names = ", ".join(params.keys())
            else:
                param_names = str(params)
            desc = (schema.get("description") or "")[:120]
            lines.append(f"- {schema['name']}({param_names}): {desc}")
        if len(schemas) > self.max_tools_in_prompt:
            lines.append(
                f"… and {len(schemas) - self.max_tools_in_prompt} more tools"
            )
        return "\n".join(lines) if lines else "(no tools available)"

    def has(self, name: str) -> bool:
        return self.registry.get(name) is not None

    # ── for the act step ─────────────────────────────────────────────

    def call(self, name: str, args: dict[str, Any] | None) -> tuple[bool, str]:
        """Call a tool through the registry.

        Returns (success, observation_text). Failures are returned as
        text, never raised — the loop decides how to recover.
        """
        args = args or {}
        if not isinstance(args, dict):
            return False, f"tool args must be an object, got {type(args).__name__}"
        if not self.has(name):
            return False, f"unknown tool {name!r}"

        try:
            outcome: Outcome = self.registry.call(
                name,
                actor=self.actor,
                capabilities=self.capabilities,
                **args,
            )
        except Exception as exc:  # noqa: BLE001 — registry should not raise, but be safe
            return False, f"tool {name} raised unexpectedly: {exc}"

        if outcome.ok:
            return True, self._format_result(outcome.value)
        err = outcome.error
        msg = getattr(err, "message", None) or str(err)
        return False, f"tool {name} failed: {msg}"

    @staticmethod
    def _format_result(value: Any) -> str:
        if value is None:
            return "(no output)"
        if isinstance(value, str):
            return value
        try:
            text = json.dumps(value, default=str)
        except Exception:  # noqa: BLE001
            text = str(value)
        return text
