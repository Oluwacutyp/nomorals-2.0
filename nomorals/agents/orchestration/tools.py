"""Tool adapter — exposes Devon's registry tools to the agentic loop.

Does NOT create new tools. Wraps the existing ToolRegistry so the loop's
think step can see what's available and the act step can call them with
capability gating and audit trail intact.

Tool selection is relevance-ranked: the think step sees the tools most
relevant to the current task first, so all 100+ registered tools stay
reachable instead of being cut off by an alphabetical cap.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ...core.policy import CapabilitySet
from ...core.result import Outcome

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_TOKENS = frozenset(
    "a an the to for of on in is it my me i you and or do does with at be "
    "please can could would should will just get set make have has had this "
    "that what when how".split()
)
_MAX_PARALLEL_CALLS = 8


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _relevance_score(schema: dict[str, Any], query_tokens: list[str]) -> float:
    """Score how relevant a tool is to the query tokens.

    Name matches weigh most, description matches less. Deterministic and
    cheap — no model call needed.
    """
    name = str(schema.get("name", ""))
    desc = str(schema.get("description", "") or "")
    name_tokens = set(_tokens(name.replace("_", " ")))
    desc_tokens = set(_tokens(desc))
    score = 0.0
    for tok in query_tokens:
        if tok in _STOP_TOKENS or len(tok) < 3:
            continue
        if tok in name_tokens:
            score += 3.0
        elif any(
            (tok in nt or nt in tok)
            for nt in name_tokens
            if len(nt) >= 3
        ):
            score += 1.5
        if tok in desc_tokens:
            score += 1.0
    return score


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

    def _schemas(self) -> list[dict[str, Any]]:
        try:
            schemas = self.registry.schemas(capabilities=self.capabilities)
        except Exception:
            schemas = []
        return [s for s in schemas if isinstance(s, dict) and s.get("name")]

    def describe(self) -> str:
        """Compact tool listing, formatted for weaker models.

        One line per tool: name(params): description. Capped so the
        prompt stays small.
        """
        return self.describe_for("", limit=self.max_tools_in_prompt)

    def describe_for(self, query: str, limit: int | None = None) -> str:
        """Tool listing ranked by relevance to ``query``.

        The most relevant tools come first, so the think step always sees
        the tools that matter for the current task — every registered tool
        stays reachable regardless of registry size.
        """
        limit = self.max_tools_in_prompt if limit is None else limit
        schemas = self._schemas()
        if query and query.strip():
            qtokens = _tokens(query)
            scored = [(_relevance_score(s, qtokens), s) for s in schemas]
            # Stable: relevance desc, then name asc for ties
            scored.sort(key=lambda p: (-p[0], str(p[1].get("name", ""))))
            ordered = [s for _, s in scored]
        else:
            ordered = schemas
        lines = []
        for schema in ordered[:limit]:
            params = schema.get("parameters") or {}
            if isinstance(params, dict):
                param_names = ", ".join(params.keys())
            else:
                param_names = str(params)
            desc = (schema.get("description") or "")[:120]
            lines.append(f"- {schema['name']}({param_names}): {desc}")
        if len(ordered) > limit:
            lines.append(
                f"… and {len(ordered) - limit} more tools "
                "(ranked lower for this task)"
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

    def call_many(
        self, calls: list[tuple[str, dict[str, Any]]]
    ) -> list[tuple[bool, str]]:
        """Call several tools, in parallel when the registry supports it.

        ``calls`` is a list of (tool_name, args). Returns a
        ``(success, observation)`` pair per call, in the same order.
        Failures are returned as text, never raised.
        """
        normed: list[tuple[str, dict[str, Any]]] = []
        for name, args in calls:
            name = str(name or "").strip()
            if not isinstance(args, dict):
                args = {}
            normed.append((name, args))
        if not normed:
            return []

        outcomes: list[Outcome] | None = None
        call_many_fn = getattr(self.registry, "call_many", None)
        if callable(call_many_fn):
            try:
                outcomes = call_many_fn(
                    normed,
                    max_workers=min(_MAX_PARALLEL_CALLS, len(normed)),
                    actor=self.actor,
                    capabilities=self.capabilities,
                )
            except Exception:  # noqa: BLE001 — fall back to serial
                outcomes = None
        if outcomes is None:
            # Registry without call_many (or it errored): serial fallback
            # through the audited single-call path.
            return [self.call(name, args) for name, args in normed]

        results: list[tuple[bool, str]] = []
        for (name, _args), outcome in zip(normed, outcomes):
            if outcome is None:
                results.append((False, f"tool {name} returned no outcome"))
            elif outcome.ok:
                results.append((True, self._format_result(outcome.value)))
            else:
                err = outcome.error
                msg = getattr(err, "message", None) or str(err)
                results.append((False, f"tool {name} failed: {msg}"))
        return results

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
