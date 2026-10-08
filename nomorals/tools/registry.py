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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    # Confirmation level: False = none, True = text confirmation token,
    # "biometric" = fingerprint approval (above True).
    confirm: bool | str = False
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
        # Phase D: guards shared audit/stats state for parallel call_many.
        self._lock = threading.Lock()

    # ── registration ─────────────────────────────────────────────────────────
    def register(
        self,
        name: str,
        fn: Callable[..., Any] | None = None,
        *,
        description: str = "",
        capability: str = "",
        parameters: dict[str, Any] | None = None,
        confirm: bool | str = False,
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

        # Modules that failed to import/register, with the error.  Surfaced
        # via failed_modules(), `nm tools`, and `nm doctor` — a broken tool
        # must never vanish invisibly.
        self._failed_modules: list[dict[str, str]] = []

        # Import and register each tool module SEPARATELY.  A single batched
        # `from . import (…)` made every builtin sink whenever ANY one tool's
        # optional dependency was missing; per-module import + register keeps
        # the rest of the registry alive (the honest degradation is "this
        # tool is unavailable", never "no tools at all").
        import importlib as _importlib

        for _name in (
            "archive", "attacker", "audio", "browser",
            "build_app", "captcha", "cipher", "code_executor", "compress",
            "connectors", "code_indexer",
            "database", "deals", "decoder", "decoder_agent", "deliver_report",
            "filesend",
            "filesystem", "finance", "giftcard", "git", "hashcrack", "imagedb",
            "lint",
            "edit_loop",
            "error_scan",
            "macros", "media", "media_edit", "media_pipeline", "metadata",
            "network", "osint", "osint_people", "parsers",
            "proxy", "proxylab", "pytest_runner", "sandbox_code", "scriptgen",
            "seer", "shell", "side_chats", "ssh_socks", "traindata", "vision", "web", "weather",
            "workspace",
            "trading",
            # the agent bridge registers last: agent modules own the real
            # implementations, and thin tools/ wrappers of the same name
            # must never shadow them (last registration wins).
            "agents",
        ):
            try:
                _module = _importlib.import_module(f".{_name}", __package__)
                _module.register(self)
            except Exception as exc:  # noqa: BLE001 — one broken tool is not all of them
                # A broken tool module must NEVER vanish invisibly.  Log it
                # with the module name and the exception so `nm tools` and
                # `nm doctor` can surface it.
                _log.warning("tool module %r failed to register: %s: %s",
                             _name, type(exc).__name__, exc)
                self._failed_modules.append({
                    "module": _name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
        
        # Custom tools authored by the tool creator (toolmaker.install drops
        # them into nomorals/tools/custom/). A fresh registry must pick them
        # up without any extra wiring — auto-registration is the contract.
        # The media system (music writer / player / video finder) lives in
        # its own package and registers through the same decorator surface.
        try:
            from ..books import tools as _books
            _books.register(self)
        except Exception as exc:  # noqa: BLE001
            _log.warning("tool module 'books' failed to register: %s: %s",
                         type(exc).__name__, exc)
            self._failed_modules.append({
                "module": "books",
                "error": f"{type(exc).__name__}: {exc}",
            })
        try:
            from ..media import music as _music, playback as _playback
            from ..media import video as _video
            from ..media import library as _library

            for _mod in (_music, _playback, _library, _video):
                try:
                    _mod.register(self)
                except Exception as exc:  # noqa: BLE001 — module-level opt-out
                    _log.warning("media tool module %r failed to register: %s: %s",
                                 getattr(_mod, "__name__", _mod),
                                 type(exc).__name__, exc)
                    self._failed_modules.append({
                        "module": getattr(_mod, "__name__", str(_mod)),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        except Exception as exc:  # noqa: BLE001
            _log.warning("media tool package failed to import: %s: %s",
                         type(exc).__name__, exc)
            self._failed_modules.append({
                "module": "media",
                "error": f"{type(exc).__name__}: {exc}",
            })

        try:
            self._load_custom_tools()
        except Exception as exc:  # noqa: BLE001 — a bad custom tool never sinks builtins
            _log.warning("custom tool loading failed: %s: %s",
                         type(exc).__name__, exc)
            self._failed_modules.append({
                "module": "custom",
                "error": f"{type(exc).__name__}: {exc}",
            })

        self._builtin_registered = True
        return self

    def _load_custom_tools(self) -> None:
        """Import every module in ``nomorals/tools/custom/`` and call its
        ``register(registry)`` hook."""
        import importlib
        import pkgutil

        from . import custom as custom_pkg

        for info in pkgutil.iter_modules(custom_pkg.__path__):
            if info.name.startswith("_"):
                continue
            try:
                module = importlib.import_module(f"{custom_pkg.__name__}.{info.name}")
                hook = getattr(module, "register", None)
                if hook is not None:
                    hook(self)
            except Exception as exc:  # noqa: BLE001 — skip a broken custom tool
                _log.warning("custom tool %r failed to register: %s: %s",
                             info.name, type(exc).__name__, exc)
                if not hasattr(self, "_failed_modules"):
                    self._failed_modules = []
                self._failed_modules.append({
                    "module": f"custom.{info.name}",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    def failed_modules(self) -> list[dict[str, str]]:
        """Modules that failed to register during ``register_builtins()``.

        Each entry is ``{"module": name, "error": "ExcType: message"}``.
        Empty when everything registered cleanly.  Surfaced by `nm tools`
        and `nm doctor` so a broken tool never vanishes invisibly.
        """
        return list(getattr(self, "_failed_modules", []))

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # ── introspection ────────────────────────────────────────────────────────
    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def agent_for(self, role: str) -> Any | None:
        """Resolve an orchestrator plan role to a bound agent (Phase C).

        Delegates to ``nomorals.tools.agents.agent_for``; unknown roles
        return None so the orchestrator falls through to "no handler".
        """
        from .agents import agent_for as _agent_for

        return _agent_for(self, role)

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
                self._bump_stats(denied=1)
                return Err(CapabilityDenied(decision.reason, capability=spec.capability, actor=actor))
        elif self.enforce and spec.capability and not grant.grants(spec.capability):
            self._audit(name, actor, spec.capability, "deny", kwargs, 0.0, error="not granted")
            self._bump_stats(denied=1)
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
            self._bump_stats(errors=1, calls=1, seconds=elapsed)
            _log.debug("tool %s failed: %s", name, error.message)
            self._record_failure_ledger(name, error.message, str(exc))
            return Err(error)

        elapsed = time.perf_counter() - started
        self._audit(name, actor, spec.capability, "allow", kwargs, elapsed, ok=True)
        self._bump_stats(calls=1, seconds=elapsed)
        return Ok(result)

    def _record_failure_ledger(self, tool: str, message: str,
                                detail: str) -> None:
        """Wave 76: real tool errors land in the failure ledger so the
        failure analyzer can learn from them (and routing can demote what
        keeps breaking). Best-effort, never raises, no recursion into the
        failure tooling itself."""
        db = getattr(self.context, "db", None) if self.context else None
        if db is None or getattr(self, "_recording", False):
            return
        self._recording = True
        try:
            db.execute(
                "INSERT INTO failures (id, source, summary, error, family, "
                "lesson, ts) VALUES (?,?,?,?,?,?,?)",
                (f"tool-{int(time.time_ns())}", "tool",
                 f"{tool} failed {message}"[:500], detail[:500],
                 self._failure_family(detail or message), "", time.time()))
        except Exception:  # noqa: BLE001 — the ledger never sinks the result
            pass
        finally:
            self._recording = False

    @staticmethod
    def _failure_family(text: str) -> str:
        try:
            from ..agents.failure import categorize

            return categorize(text)
        except Exception:  # noqa: BLE001
            return "tool"

    def _bump_stats(self, **deltas: float) -> None:
        """Thread-safe stats increments (Phase D parallel call_many)."""
        with self._lock:
            for key, delta in deltas.items():
                self.stats[key] = self.stats.get(key, 0.0) + delta

    def call_many(self, calls: list[tuple[str, dict[str, Any]]],
                  *, max_workers: int = 1, **common: Any) -> list[Outcome[Any]]:
        """Run several tool calls; ``max_workers > 1`` runs them in parallel
        (Phase D, bounded — default 8 in the explore phase) and preserves
        call order in the results.  ``max_workers=1`` is the old serial
        behavior."""
        if max_workers <= 1 or len(calls) <= 1:
            return [self.call(name, **{**common, **kwargs})
                    for name, kwargs in calls]

        def _one(call: tuple[str, dict[str, Any]]) -> Outcome[Any]:
            name, kwargs = call
            return self.call(name, **{**common, **kwargs})

        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="tool-call") as pool:
            return list(pool.map(_one, calls))

    def request_approval(
        self,
        name: str,
        *,
        actor: str = "",
        title: str = "",
        timeout_s: float = 60.0,
    ) -> str | None:
        """Mint a confirmation token for a sensitive tool via fingerprint.

        Uses the biometric path when the tool requires it — either declared
        (``spec.confirm == "biometric"``) or via the policy (the capability
        is in :attr:`Capability.BIOMETRIC` or a ``biometric`` rule matches).
        Returns None when no biometric is required (the text-confirm flow
        owns plain confirmations), when the tool is unknown, when no policy
        is attached, or on any failure. Never raises and never prompts on
        its own: callers invoke this explicitly from an interactive context
        only — ``call()`` itself never blocks on a fingerprint dialog.
        """
        try:
            spec = self._tools.get(name)
            if spec is None:
                return None
            policy = getattr(self.context, "policy", None) if self.context else None
            biometric_required = spec.confirm == "biometric"
            if not biometric_required and policy is not None and spec.capability:
                biometric_required = policy.requires_biometric(spec.capability)
            if not biometric_required or policy is None:
                return None
            from ..core.policy import approve_with_biometric

            return approve_with_biometric(
                policy,
                spec.capability,
                title=title or f"approve {name}",
                timeout_s=timeout_s,
            )
        except Exception:  # noqa: BLE001 - approval must never raise
            return None

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
        with self._lock:
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
