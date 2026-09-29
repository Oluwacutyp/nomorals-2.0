"""Tool registry: schema, dispatch, permission gating, and an audit trail.

Every tool declares the capability it needs. Every call is checked against the
caller's grant and logged — actor, argument digest, decision, outcome, duration.
That log is what makes "what did the agent do while I wasn't looking" answerable,
which is the minimum bar for leaving an autonomous system running unattended.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.errors import CapabilityDenied, NotFound, ToolError, ToolNotFound, classify
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.policy import CapabilitySet
from ..core.result import Err, Ok, Outcome

__all__ = ["ToolRegistry", "ToolSpec"]

_log = get_logger(__name__)


@dataclass
class ToolSpec:
    """A callable an agent may invoke."""

    name: str
    fn: Callable[..., Any]
    description: str = ""
    capability: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    confirm: bool = False
    kind: str = "io"
    metadata: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        """JSON-schema-ish description, for feeding a model a tool list."""
        return {
            "name": self.name,
            "description": self.description,
            "capability": self.capability,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registers, gates, dispatches, and audits tool calls."""

    def __init__(self, context: Any = None, *, enforce: bool = True) -> None:
        self.context = context
        self.enforce = enforce
        self._tools: dict[str, ToolSpec] = {}
        self.calls: list[dict[str, Any]] = []
        self._call_limit = 2000
        self.stats = {"calls": 0, "denied": 0, "errors": 0, "seconds": 0.0}
        self._builtin_registered = False

    # ── registration ─────────────────────────────────────────────────────────
    def register(
        self,
        name: str,
        fn: Callable[..., Any] | None = None,
        *,
        description: str = "",
        capability: str = "",
        parameters: dict[str, Any] | None = None,
        confirm: bool = False,
        kind: str = "io",
    ) -> Callable[..., Any]:
        """Register a tool, usable directly or as a decorator."""

        def do_register(func: Callable[..., Any]) -> Callable[..., Any]:
            doc = (inspect.getdoc(func) or "").strip()
            first_line = doc.splitlines()[0] if doc else ""
            spec = ToolSpec(
                name=name,
                fn=func,
                description=description or first_line or name,
                capability=capability,
                parameters=parameters or _infer_parameters(func),
                confirm=confirm,
                kind=kind,
            )
            self._tools[name] = spec
            return func

        if fn is not None:
            return do_register(fn)
        return do_register

    def register_builtins(self) -> "ToolRegistry":
        """Wire up the standard tool set. Imports are deferred per tool."""
        if self._builtin_registered:
            return self
        
        # Import all tool modules
        from . import (
            agents, archive, attacker, audio, book, browser, build_app, cards, cipher, compress, connectors, database,
            deals, decoder, decoder_agent, filesend, filesystem, finance, giftcard, hashcrack, imagedb,
            macros, media, media_hub, metadata, network, osint, osint_people, parsers,
            proxy, proxylab, run_code, sandbox_code, scriptgen, shell, ssh_socks,
            traindata, vision, web, workspace
        )
        
        # Register all tool modules
        for module in [
            agents, archive, attacker, audio, book, browser, build_app, cards, cipher, compress, connectors, database,
            deals, decoder, decoder_agent, filesend, filesystem, finance, giftcard, hashcrack, imagedb,
            macros, media, media_hub, metadata, network, osint, osint_people, parsers,
            proxy, proxylab, run_code, sandbox_code, scriptgen, shell, ssh_socks,
            traindata, vision, web, workspace
        ]:
            try:
                module.register(self)
            except Exception:
                pass  # Some modules may fail to register in certain contexts
        
        self._builtin_registered = True
        return self

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # ── introspection ────────────────────────────────────────────────────────
    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schemas(self, *, capabilities: CapabilitySet | None = None) -> list[dict[str, Any]]:
        """Tool list for a model prompt, filtered to what the caller may use."""
        out = []
        for spec in self._tools.values():
            if capabilities is not None and spec.capability and not capabilities.grants(spec.capability):
                continue
            out.append(spec.schema())
        return out

    def prompt_listing(self, *, capabilities: CapabilitySet | None = None) -> str:
        lines = []
        for schema in self.schemas(capabilities=capabilities):
            params = ", ".join(schema["parameters"])
            lines.append(f"- {schema['name']}({params}): {schema['description']}")
        return "\n".join(lines)

    # ── dispatch ─────────────────────────────────────────────────────────────
    def call(
        self,
        name: str,
        /,
        *args: Any,
        actor: str = "system",
        capabilities: CapabilitySet | None = None,
        confirmation: str | None = None,
        **kwargs: Any,
    ) -> Outcome[Any]:
        """Check the capability, run the tool, and audit the call."""
        # Reject extra positional arguments with a proper error outcome
        if args:
            return Err(ToolError(
                f"call() takes 1 positional argument but {1 + len(args)} were given; "
                f"pass tool parameters as keywords"
            ))
        
        spec = self._tools.get(name)
        if spec is None:
            return Err(ToolNotFound(f"unknown tool {name!r}; available: {self.names()[:20]}"))

        started = time.perf_counter()
        grant = capabilities if capabilities is not None else CapabilitySet.all()
        policy = getattr(self.context, "policy", None) if self.context else None

        if self.enforce and spec.capability and policy is not None:
            decision = policy.check(
                spec.capability, actor=actor, grant=grant,
                confirmation=confirmation, context={"tool": name},
            )
            if not decision.allowed:
                self._audit(name, actor, spec.capability, "deny", kwargs, 0.0, error=decision.reason)
                self.stats["denied"] += 1
                return Err(CapabilityDenied(decision.reason, capability=spec.capability, actor=actor))
        elif self.enforce and spec.capability and not grant.grants(spec.capability):
            self._audit(name, actor, spec.capability, "deny", kwargs, 0.0, error="not granted")
            self.stats["denied"] += 1
            return Err(
                CapabilityDenied(
                    f"actor {actor!r} lacks {spec.capability!r}",
                    capability=spec.capability, actor=actor,
                )
            )

        try:
            result = spec.fn(**kwargs) if kwargs else spec.fn()
        except Exception as exc:  # noqa: BLE001 - tool failures are results
            elapsed = time.perf_counter() - started
            error = classify(exc)
            self._audit(name, actor, spec.capability, "allow", kwargs, elapsed, error=error.message)
            self.stats["errors"] += 1
            self.stats["calls"] += 1
            self.stats["seconds"] += elapsed
            _log.debug("tool %s failed: %s", name, error.message)
            return Err(error)

        elapsed = time.perf_counter() - started
        self._audit(name, actor, spec.capability, "allow", kwargs, elapsed, ok=True)
        self.stats["calls"] += 1
        self.stats["seconds"] += elapsed
        return Ok(result)

    def call_many(self, calls: list[tuple[str, dict[str, Any]]], **common: Any) -> list[Outcome[Any]]:
        return [self.call(name, **{**common, **kwargs}) for name, kwargs in calls]

    # ── auditing ─────────────────────────────────────────────────────────────
    def _audit(
        self,
        name: str,
        actor: str,
        capability: str,
        decision: str,
        kwargs: dict[str, Any],
        elapsed: float,
        *,
        ok: bool = False,
        error: str = "",
    ) -> None:
        entry = {
            "id": new_id(),
            "tool": name,
            "actor": actor,
            "capability": capability,
            "decision": decision,
            "status": "ok" if ok else ("denied" if decision == "deny" else "error"),
            "args_digest": _digest(kwargs),
            "duration_ms": round(elapsed * 1000, 3),
            "error": error,
            "ts": time.time(),
        }
        self.calls.append(entry)
        if len(self.calls) > self._call_limit:
            del self.calls[: len(self.calls) - self._call_limit]
        db = getattr(self.context, "db", None) if self.context else None
        if db is not None:
            try:
                db.insert(
                    "tool_calls",
                    {
                        "id": entry["id"], "actor": actor, "tool": name,
                        "capability": capability, "decision": decision,
                        "args_digest": entry["args_digest"], "status": entry["status"],
                        "duration_ms": entry["duration_ms"], "error": error,
                        "created_at": entry["ts"],
                    },
                )
            except Exception as exc:  # noqa: BLE001 - auditing must never break a tool call
                _log.debug("could not persist tool call: %s", exc)

    def audit_trail(self, limit: int = 50, tool: str = "") -> list[dict[str, Any]]:
        items = self.calls if not tool else [c for c in self.calls if c["tool"] == tool]
        return items[-limit:]

    def stats_snapshot(self) -> dict[str, Any]:
        return {**self.stats, "tools": len(self._tools), "names": self.names()}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def _digest(kwargs: dict[str, Any]) -> str:
    """Stable, non-reversible digest of call arguments for the audit log."""
    try:
        payload = json.dumps(kwargs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(kwargs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _infer_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
    """Best-effort parameter schema from a signature."""
    out: dict[str, Any] = {}
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return out
    hints = getattr(fn, "__annotations__", {})
    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        entry: dict[str, Any] = {"type": _type_name(hints.get(name, ""))}
        if param.default is inspect.Parameter.empty:
            entry["required"] = True
        else:
            entry["default"] = param.default if _jsonable(param.default) else str(param.default)
        out[name] = entry
    return out


def _type_name(hint: Any) -> str:
    if not hint:
        return "any"
    if isinstance(hint, str):
        return hint.lower()
    return getattr(hint, "__name__", str(hint)).lower()


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None)))
