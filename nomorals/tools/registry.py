"""Tool registry: schema, dispatch, permission gating, and an audit trail.

Every tool declares the capability it needs. Every call is checked against the
caller's grant and logged — actor, argument digest, decision, outcome, duration.
That log is what makes "what did the agent do while I wasn't looking" answerable,
which is the minimum bar for leaving an autonomous system running unattended.
"""

from __future__ import annotations

from .typed_encoding import _decode_typed, _encode_typed

import contextvars
import fnmatch
import hashlib
import inspect
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from ..core.errors import (
    CapabilityDenied, NotFound, ToolError, ToolNotFound, ValidationError,
    classify,
)
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.policy import CapabilitySet
from ..core.result import Err, Ok, Outcome

__all__ = ["ToolRegistry", "ToolSpec", "ToolHealth", "sanitize_tool_description",
           "annotations_for_kind"]

_log = get_logger(__name__)


#: sentence-level patterns for instruction-like text smuggled into tool
#: descriptions. Tool metadata is shown to the model, so it is an
#: untrusted surface: descriptions must describe, never instruct.
_DESC_INSTRUCTION_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhidden\s+instruction\b", re.IGNORECASE),
    re.compile(r"\bignore\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"^\s*(system|developer|admin)\s*:", re.IGNORECASE),
    re.compile(r"\byou\s+(must|should|shall)\b", re.IGNORECASE),
    re.compile(
        r"\bdo\s+not\s+(mention|reveal|tell|disclose).{0,60}\binstruction\b",
        re.IGNORECASE,
    ),
)


def sanitize_tool_description(name: str, description: str) -> str:
    """Strip instruction-like sentences from a tool description.

    Keeps legitimate descriptive sentences; drops sentences that read as
    directives to the agent.  Never raises and never returns an empty
    description — worst case the original is kept with a warning, so
    registration can never break on sanitization.
    """
    try:
        text = (description or "").strip()
        if not text:
            return description
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [
            s for s in sentences
            if not any(p.search(s) for p in _DESC_INSTRUCTION_RES)
        ]
        if len(kept) == len(sentences):
            return description  # clean — return untouched
        if not any(k.strip() for k in kept):
            _log.warning(
                "tool %s description was entirely instruction-like; "
                "kept as-is (flagged)", name)
            return description
        _log.warning(
            "tool %s description contained instruction-like text; "
            "stripped %d of %d sentence(s)",
            name, len(sentences) - len(kept), len(sentences))
        return " ".join(kept).strip()
    except Exception:  # noqa: BLE001 - sanitization must never break registration
        _log.debug("tool %s description sanitization failed", name,
                   exc_info=True)
        return description


def _normalize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool's parameter declaration to the flat contract.

    The contract is ``{name: {type, required, default, description}}``.
    ``tools/social.py`` and ``tools/whatsapp.py`` historically passed a
    full JSON-Schema object (``{type: object, properties: {...},
    required: [...]}``) instead, which :meth:`ToolSpec.schema` then
    wrapped a second time — so listings rendered the schema's own keys
    (``type, properties, required``) as the tool's parameters. Detect
    the wrapped shape (top-level keys are all schema keywords, ``type``
    is ``object``, ``properties`` is a dict) and unwrap one level,
    marking the schema-level ``required`` names on their specs. Flat
    declarations pass through untouched.
    """
    if not isinstance(parameters, dict):
        return parameters
    if parameters.get("type") != "object":
        return parameters
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return parameters
    if not set(parameters) <= {
        "type", "properties", "required", "additionalProperties",
        "description", "$schema", "title",
    }:
        return parameters
    required = set(parameters.get("required") or [])
    flat: dict[str, Any] = {}
    for name, spec in properties.items():
        if isinstance(spec, dict):
            entry = dict(spec)
        elif isinstance(spec, str):
            entry = {"type": spec}
        else:
            entry = {"type": "any"}
        if name in required:
            entry["required"] = True
        flat[name] = entry
    return flat


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
    # Lifecycle: version of the tool implementation; deprecation marks the
    # tool as superseded (warned once per registry, still callable).
    version: str = ""
    deprecated: bool = False
    replaced_by: str = ""
    # ── MCP-style surface (mined: modelcontextprotocol.io tool annotations) ──
    # Human-readable display name; defaults to name.
    title: str = ""
    # Honest behavior flags hosts can gate on: read_only, destructive,
    # idempotent, open_world. Defaults are derived from ``kind`` when not
    # given explicitly (see annotations_for_kind).
    annotations: dict[str, bool] = field(default_factory=dict)
    # JSON schema describing the tool's result shape (optional but
    # recommended — models use it to parse structured output).
    output_schema: dict[str, Any] = field(default_factory=dict)
    # Short usage examples shown to the model.
    examples: list[str] = field(default_factory=list)
    # Long string results are truncated past this many chars, with an
    # explicit notice (MCP rule: never truncate silently).
    max_result_chars: int = 200_000

    def __post_init__(self) -> None:
        # Accept a full JSON-Schema object as `parameters` (social.py /
        # whatsapp.py historically passed one) and normalize it to the
        # flat contract, so every consumer sees real parameter names.
        self.parameters = _normalize_parameters(self.parameters)

    def resolved_annotations(self) -> dict[str, bool]:
        """Explicit annotations merged over kind-derived defaults."""
        merged = annotations_for_kind(self.kind)
        merged.update({k: bool(v) for k, v in self.annotations.items()})
        return merged

    def schema(self) -> dict[str, Any]:
        """JSON-schema-ish description, for feeding a model a tool list.

        Tightened per MCP best practice: every property carries its
        inferred type, ``required`` is exact, and
        ``additionalProperties`` is false so the model doesn't invent
        parameters.
        """
        params = dict(self.parameters or {})
        required = sorted(
            name for name, spec in params.items()
            if isinstance(spec, dict) and spec.get("required"))
        properties = {}
        for name, spec in params.items():
            entry = dict(spec) if isinstance(spec, dict) else {"type": "any"}
            entry.pop("required", None)
            # No stub description: a parameter with no recorded description
            # stays undescribed. The old setdefault wrote the parameter's
            # own name back as its description ("path" described as "path").
            properties[name] = entry
        out: dict[str, Any] = {
            "name": self.name,
            "title": self.title or self.name,
            "description": self.description,
            "capability": self.capability,
            "annotations": self.resolved_annotations(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
        if self.output_schema:
            out["output_schema"] = self.output_schema
        if self.examples:
            out["examples"] = list(self.examples)
        return out


def annotations_for_kind(kind: str) -> dict[str, bool]:
    """Sensible MCP-style behavior flags derived from a tool's kind.

    Explicit per-tool ``annotations`` always win over these defaults.
    """
    kind = (kind or "").lower()
    if kind in {"query", "read", "search", "list"}:
        return {"read_only": True, "destructive": False,
                "idempotent": True, "open_world": False}
    if kind in {"exec", "write", "delete", "mutate"}:
        return {"read_only": False, "destructive": kind == "delete",
                "idempotent": False, "open_world": False}
    if kind in {"net", "web", "external"}:
        return {"read_only": False, "destructive": False,
                "idempotent": True, "open_world": True}
    # "io" and anything unknown: conservative — not read-only, not
    # claimed idempotent, may touch the outside world.
    return {"read_only": False, "destructive": False,
            "idempotent": False, "open_world": True}


@dataclass
class ToolHealth:
    """Health state for one tool (opencapx ping pattern).

    ``status``: unknown → ok → degraded (1-2 probe failures) → down
    (3+ consecutive failures). Probes run with a timeout; a probe that
    raises or times out counts as a failure.
    """

    name: str
    status: str = "unknown"
    consecutive_failures: int = 0
    last_check: float = 0.0
    last_error: str = ""
    timeout_s: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "consecutive_failures": self.consecutive_failures,
            "last_check": self.last_check,
            "last_error": self.last_error,
            "timeout_s": self.timeout_s,
        }


def _format_params(schema: dict[str, Any]) -> str:
    """Render a tool's parameters as ``name`` / ``name=default`` pairs.

    ``schema["parameters"]`` is a JSON-Schema object (``type`` /
    ``properties`` / ``required`` / ``additionalProperties``) — joining its
    keys produced the nonsense ``(type, properties, required,
    additionalProperties)`` on every tool in every listing. This renders
    the actual parameter names in definition order: required params bare,
    optional params with their recorded default (repr, truncated to ~40
    chars), optional params without a recorded default bare.
    """
    params = schema.get("parameters") or {}
    properties = params.get("properties") or {}
    required = set(params.get("required") or [])
    parts = []
    for name, spec in properties.items():
        if name in required or not isinstance(spec, dict) or "default" not in spec:
            parts.append(name)
        else:
            default = repr(spec["default"])
            if len(default) > 40:
                default = default[:37] + "..."
            parts.append(f"{name}={default}")
    return ", ".join(parts)


@dataclass
class _IdempotencyRecord:
    """One durable idempotent effect: who ran what, with which args."""

    tool: str
    key: str
    actor: str
    args_digest: str
    tool_version: str
    outcome: Outcome[Any]
    created_at: float = 0.0


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
        # Per-tool stats and health (introspection + routing demotion).
        self._tool_stats: dict[str, dict[str, Any]] = {}
        self._health: dict[str, ToolHealth] = {}
        self._health_probes: dict[str, tuple[Callable[[], Any], float]] = {}
        self._deprecated_warned: set[str] = set()
        # Idempotency records: (tool name, idempotency_key) -> record.
        # Bound to (actor, argument digest, tool version) so a reused key
        # can never return an unrelated old result, and persisted to the
        # db when one is available so a restart keeps the protection.
        # Bounded in memory so a long-lived registry can't grow unbounded.
        self._idempotency_records: dict[tuple[str, str], _IdempotencyRecord] = {}
        self._idempotency_limit = 512
        # In-flight reservations: (tool, key) -> event set on completion.
        # A concurrent duplicate waits for the owner's result instead of
        # executing a second time.
        self._idempotency_inflight: dict[tuple[str, str], threading.Event] = {}
        self._idempotency_cond = threading.Condition(threading.Lock())
        self._idempotency_db_ready = False

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
        version: str = "",
        deprecated: bool = False,
        replaced_by: str = "",
        title: str = "",
        annotations: dict[str, bool] | None = None,
        output_schema: dict[str, Any] | None = None,
        examples: list[str] | None = None,
        max_result_chars: int = 200_000,
    ) -> Callable[..., Any]:
        """Register a tool, usable directly or as a decorator."""

        def do_register(func: Callable[..., Any]) -> Callable[..., Any]:
            doc = (inspect.getdoc(func) or "").strip()
            first_line = doc.splitlines()[0] if doc else ""
            spec = ToolSpec(
                name=name,
                fn=func,
                # tool metadata reaches the model: sanitize instruction-like
                # sentences (prompt-injection surface), never breaking
                # registration
                description=sanitize_tool_description(
                    name, description or first_line or name),
                capability=capability,
                parameters=parameters or _infer_parameters(func),
                confirm=confirm,
                kind=kind,
                version=version,
                deprecated=deprecated,
                replaced_by=replaced_by,
                title=title,
                annotations=annotations or {},
                output_schema=output_schema or {},
                examples=examples or [],
                max_result_chars=max_result_chars,
            )
            self._tools[name] = spec
            return func

        if fn is not None:
            return do_register(fn)
        return do_register

    def alias(self, new_name: str, target: str) -> bool:
        """Register ``new_name`` as an alias of the existing tool ``target``.

        The alias shares the implementation and capability; its metadata
        records ``alias_of``. Returns False when ``target`` is unknown.
        """
        spec = self._tools.get(target)
        if spec is None:
            return False
        self._tools[new_name] = replace(
            spec, name=new_name,
            metadata={**spec.metadata, "alias_of": target})
        return True

    def deprecate(self, name: str, *, replaced_by: str = "",
                  note: str = "") -> bool:
        """Mark a tool deprecated. It stays callable; calls log a warning
        once per registry (pointing at ``replaced_by`` when given)."""
        spec = self._tools.get(name)
        if spec is None:
            return False
        spec.deprecated = True
        spec.replaced_by = replaced_by
        if note:
            spec.metadata["deprecation_note"] = note
        return True

    def reload_builtin(self, name: str) -> Outcome[dict[str, Any]]:
        """Re-import ``nomorals.tools.<name>`` and re-run its register hook.

        Registration overwrites by name, so this is idempotent — the mining
        requirement that setup() be safe to run twice. Returns an Outcome;
        a broken module is reported, never raised.
        """
        import importlib

        try:
            module = importlib.import_module(f"{__package__}.{name}")
            module = importlib.reload(module)
        except Exception as exc:  # noqa: BLE001
            return Err(ToolError(f"cannot reload tool module {name!r}: {exc}"))
        hook = getattr(module, "register", None)
        if hook is None:
            return Err(ToolError(f"tool module {name!r} has no register() hook"))
        try:
            hook(self)
        except Exception as exc:  # noqa: BLE001
            return Err(ToolError(f"tool module {name!r} re-register failed: {exc}"))
        return Ok({"reloaded": name, "tools": len(self._tools)})

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
            "accounts",
            "build_app", "captcha", "cipher", "code_executor", "compress",
            "connectors", "code_indexer",
            "database", "deals", "decoder", "decoder_agent", "deliver_report",
            "commerce", "filesend",
            "filesystem", "finance", "giftcard", "git", "hashcrack", "imagedb",
            "lint",
            "edit_loop",
            "errorsys",  # system health / error budgets / incident history
            "error_scan",
            "macros", "media", "media_edit", "media_gen", "media_pipeline", "metadata",
            "music", "studio",
            "network", "osint", "osint_people", "parsers",
            "proxy", "proxylab", "pytest_runner", "sandbox_code", "scriptgen",
            "sceneintel", "directed", "seer", "shell", "side_chats", "ssh_socks", "traindata", "vision", "web", "weather",
            "security",
            "wisdom", "social", "whatsapp",
            "services",  # 28 service connectors bridged to the spine
            "research",
            "memory",  # spine-native granular memory tools (remember/recall/forget/...)
            "workspace",
            "trading",
            "characters",
            "games",
            "autonomy",
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
            from ..media import ace_step as _ace_step
            from ..media import vocals as _vocals
            from ..media import distribute as _distribute

            for _mod in (_music, _playback, _library, _video,
                         _ace_step, _vocals, _distribute):
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
            params = _format_params(schema)
            lines.append(f"- {schema['name']}({params}): {schema['description']}")
        return "\n".join(lines)

    def ranked_listing(
        self,
        query: str,
        *,
        capabilities: CapabilitySet | None = None,
        limit: int = 40,
    ) -> str:
        """Top-``limit`` tools by relevance to ``query``.

        Deterministic token-overlap scoring (name hits weigh most) —
        no model call needed. Always includes a small core of
        universally-useful tools so the model isn't blind when the
        query matches nothing well. Falls back to the full listing
        when the query is empty.
        """
        schemas = self.schemas(capabilities=capabilities)
        if not query or not query.strip():
            return self.prompt_listing(capabilities=capabilities)
        qtokens = [
            t.lower() for t in query.replace("_", " ").split()
            if len(t) >= 3
        ]
        if not qtokens:
            return self.prompt_listing(capabilities=capabilities)

        def _score(schema: dict[str, Any]) -> float:
            name = str(schema.get("name", ""))
            desc = str(schema.get("description", "") or "")
            name_toks = set(name.replace("_", " ").lower().split())
            desc_toks = set(desc.lower().split())
            score = 0.0
            for tok in qtokens:
                if tok in name_toks:
                    score += 3.0
                elif any(tok in nt or nt in tok for nt in name_toks
                         if len(nt) >= 3):
                    score += 1.5
                if tok in desc_toks:
                    score += 1.0
            return score

        # Core tools that should always be visible (memory, help, etc.)
        core = {"memory_recall", "help", "tools_list"}
        scored = [(_score(s), s) for s in schemas]
        scored.sort(key=lambda p: -p[0])
        picked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for s in schemas:
            if s["name"] in core:
                picked.append(s)
                seen.add(s["name"])
        for _, s in scored:
            if s["name"] in seen:
                continue
            if len(picked) >= limit:
                break
            # Only include tools with some relevance, unless we have
            # too few — then take the best scorers regardless.
            picked.append(s)
            seen.add(s["name"])
        # Trim to limit, keeping core first then by score.
        picked = picked[:limit]
        lines = []
        for schema in picked:
            params = _format_params(schema)
            lines.append(
                f"- {schema['name']}({params}): {schema['description']}")
        listing = "\n".join(lines)
        if len(schemas) > len(picked):
            listing += (
                f"\n({len(schemas) - len(picked)} more tools available — "
                f"ask with 'tools_list' to see them)")
        return listing

    def describe(self, name: str) -> dict[str, Any] | None:
        """Rich introspection for one tool: identity, source location,
        capability, lifecycle, health, and per-tool stats. Powers
        ``nm tools describe <name>``. None when unknown."""
        spec = self._tools.get(name)
        if spec is None:
            return None
        try:
            source = inspect.getsourcefile(spec.fn) or ""
        except (TypeError, OSError):
            source = ""
        try:
            _, lineno = inspect.getsourcelines(spec.fn)
        except (TypeError, OSError):
            lineno = 0
        health = self._health.get(name)
        return {
            "name": spec.name,
            "title": spec.title or spec.name,
            "description": spec.description,
            "capability": spec.capability,
            "confirm": spec.confirm,
            "kind": spec.kind,
            "annotations": spec.resolved_annotations(),
            "output_schema": spec.output_schema,
            "examples": spec.examples,
            "version": spec.version,
            "deprecated": spec.deprecated,
            "replaced_by": spec.replaced_by,
            "parameters": spec.parameters,
            "module": getattr(spec.fn, "__module__", ""),
            "source": source,
            "line": lineno,
            "alias_of": spec.metadata.get("alias_of", ""),
            "health": health.to_dict() if health else None,
            "stats": self.tool_stats(name),
        }

    def by_capability(self, capability: str) -> list[str]:
        """Tool names whose declared capability matches ``capability``
        (exact or fnmatch pattern)."""
        return sorted(
            name for name, spec in self._tools.items()
            if spec.capability and (
                spec.capability == capability
                or fnmatch.fnmatchcase(spec.capability, capability)
                or fnmatch.fnmatchcase(capability, spec.capability)))

    def by_kind(self, kind: str) -> list[str]:
        return sorted(name for name, spec in self._tools.items()
                      if spec.kind == kind)

    def capabilities_used(self) -> list[str]:
        """Every capability declared by at least one tool, sorted."""
        return sorted({spec.capability for spec in self._tools.values()
                       if spec.capability})

    def tool_stats(self, name: str) -> dict[str, Any]:
        """Per-tool counters: calls, denied, errors, seconds, last call info.
        Unknown tools yield a zeroed record."""
        with self._lock:
            stats = self._tool_stats.get(name)
            return dict(stats) if stats else {
                "calls": 0, "denied": 0, "errors": 0, "seconds": 0.0,
                "last_ts": 0.0, "last_status": "", "last_error": "",
                "last_duration_ms": 0.0,
            }

    # ── health ───────────────────────────────────────────────────────────────
    def register_health(self, name: str, probe: Callable[[], Any], *,
                        timeout_s: float = 5.0) -> bool:
        """Attach a health probe to a tool.

        The probe is a zero-arg callable: truthy (or ``(True, msg)``) means
        healthy; falsy, a raised exception, or a timeout means failure.
        Returns False when the tool is unknown.
        """
        if name not in self._tools:
            return False
        with self._lock:
            self._health_probes[name] = (probe, timeout_s)
            existing = self._health.get(name)
            if existing is None:
                self._health[name] = ToolHealth(name=name, timeout_s=timeout_s)
            else:
                existing.timeout_s = timeout_s
        return True

    def check_health(self, name: str) -> ToolHealth | None:
        """Run one tool's probe now; updates and returns its health record."""
        with self._lock:
            entry = self._health.get(name)
            probe_pack = self._health_probes.get(name)
        if entry is None or probe_pack is None:
            return None
        probe, timeout_s = probe_pack
        ok, error = self._run_probe(probe, timeout_s)
        with self._lock:
            entry.last_check = time.time()
            entry.timeout_s = timeout_s
            if ok:
                entry.status = "ok"
                entry.consecutive_failures = 0
                entry.last_error = ""
            else:
                entry.consecutive_failures += 1
                entry.last_error = error
                # opencapx rule: 3 consecutive failures → down
                entry.status = ("down" if entry.consecutive_failures >= 3
                                else "degraded")
            return entry

    def check_all_health(self, *, max_workers: int = 8) -> dict[str, ToolHealth]:
        """Run every registered probe (bounded parallelism); returns the
        health records keyed by tool name."""
        with self._lock:
            names = list(self._health_probes)
        if not names:
            return {}
        with ThreadPoolExecutor(max_workers=max(1, max_workers),
                                thread_name_prefix="tool-health") as pool:
            results = list(pool.map(self.check_health, names))
        return {name: entry for name, entry in zip(names, results)
                if entry is not None}

    def health_report(self) -> dict[str, Any]:
        """Aggregate health: counts per status plus the per-tool records."""
        with self._lock:
            records = {name: entry.to_dict()
                       for name, entry in self._health.items()}
        counts: dict[str, int] = {}
        for record in records.values():
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        return {
            "tools": len(self._tools),
            "probed": len(records),
            "counts": counts,
            "records": records,
            "failed_modules": self.failed_modules(),
        }

    @staticmethod
    def _run_probe(probe: Callable[[], Any],
                   timeout_s: float) -> tuple[bool, str]:
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["result"] = probe()
            except Exception as exc:  # noqa: BLE001
                box["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=target, daemon=True,
                                  name="tool-health-probe")
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            return False, f"probe timed out after {timeout_s:g}s"
        if "error" in box:
            return False, str(box["error"])
        result = box.get("result", False)
        if isinstance(result, tuple):
            ok = bool(result[0])
            msg = str(result[1]) if len(result) > 1 else ""
            return ok, "" if ok else (msg or "probe reported unhealthy")
        return bool(result), "" if result else "probe returned falsy"

    # ── dispatch ─────────────────────────────────────────────────────────────
    def _call_inner(
        self,
        name: str,
        /,
        *args: Any,
        actor: str = "system",
        capabilities: CapabilitySet | None = None,
        confirmation: str | None = None,
        **kwargs: Any,
    ) -> Outcome[Any]:
        """Check the capability, run the tool, and audit the call.

        Idempotency lives in :meth:`call` (the wrapper); this core
        always executes.
        """
        # Reject extra positional arguments with a proper error outcome
        if args:
            return Err(ToolError(
                f"call() takes 1 positional argument but {1 + len(args)} were given; "
                f"pass tool parameters as keywords"
            ))

        spec = self._tools.get(name)
        if spec is None:
            return Err(ToolNotFound(_unknown_tool_message(name, self._tools)))
        if spec.deprecated and name not in self._deprecated_warned:
            with self._lock:
                self._deprecated_warned.add(name)
            _log.warning(
                "tool %r is deprecated%s", name,
                f"; use {spec.replaced_by!r} instead" if spec.replaced_by
                else "")

        started = time.perf_counter()
        grant = capabilities if capabilities is not None else CapabilitySet.all()
        policy = getattr(self.context, "policy", None) if self.context else None

        if self.enforce and spec.capability and policy is not None:
            # Argument-bound enforcement: the policy sees the exact
            # arguments, not just the capability string, so a
            # confirmation for one target cannot be redirected to
            # another, and writes to the oversight machinery itself
            # (policy / social gate / audit / vault) require a token
            # bound to those exact arguments. See core/oversight.py.
            from ..core.oversight import args_digest, is_oversight_target

            digest = args_digest(kwargs)
            decision = policy.check(
                spec.capability, actor=actor, grant=grant,
                confirmation=confirmation,
                context={
                    "tool": name,
                    "args_digest": digest,
                    "oversight": is_oversight_target(
                        name, spec.capability, kwargs),
                },
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
        return Ok(_maybe_truncate(result, spec.max_result_chars))

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

    # -- idempotency: durable effect contract --
    @staticmethod
    def _args_digest(kwargs: dict[str, Any]) -> str:
        try:
            from ..core.oversight import args_digest

            return args_digest(kwargs)
        except Exception:  # noqa: BLE001 - digest must never break dispatch
            try:
                canon = json.dumps(kwargs or {}, sort_keys=True, default=str)
            except Exception:  # noqa: BLE001
                canon = repr(kwargs or {})
            return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def _idempotency_table(self) -> bool:
        """Create the durable record table once. False when no db."""
        if self._idempotency_db_ready:
            return True
        db = getattr(self.context, "db", None) if self.context else None
        if db is None:
            return False
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS idempotency_records ("
                "tool TEXT NOT NULL, key TEXT NOT NULL, actor TEXT NOT NULL,"
                " args_digest TEXT NOT NULL, tool_version TEXT NOT NULL,"
                " outcome_json TEXT NOT NULL, ts REAL NOT NULL,"
                " PRIMARY KEY (tool, key))")
            self._idempotency_db_ready = True
            return True
        except Exception:  # noqa: BLE001 - durability is best-effort
            return False

    def _idempotency_load(self, name: str, key: str) -> _IdempotencyRecord | None:
        """Memory first, then the durable table (restart recovery)."""
        rec = self._idempotency_records.get((name, key))
        if rec is not None:
            return rec
        if not self._idempotency_table():
            return None
        db = getattr(self.context, "db", None)
        try:
            row = db.query_one(
                "SELECT * FROM idempotency_records WHERE tool = ? AND key = ?",
                (name, key))
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        try:
            outcome: Outcome[Any] = Ok(
                _decode_typed(json.loads(row["outcome_json"])))
        except Exception:  # noqa: BLE001 - a corrupt row is a miss
            return None
        rec = _IdempotencyRecord(
            tool=name, key=key, actor=str(row["actor"]),
            args_digest=str(row["args_digest"]),
            tool_version=str(row["tool_version"]),
            outcome=outcome, created_at=float(row["ts"] or 0.0))
        with self._lock:
            self._idempotency_records[(name, key)] = rec
        return rec

    def _idempotency_store(self, rec: _IdempotencyRecord) -> None:
        """Persist a successful idempotent result (memory + db)."""
        with self._lock:
            self._idempotency_records[(rec.tool, rec.key)] = rec
            while len(self._idempotency_records) > self._idempotency_limit:
                self._idempotency_records.pop(
                    next(iter(self._idempotency_records)))
        if self._idempotency_table():
            db = getattr(self.context, "db", None)
            try:
                db.execute(
                    "INSERT OR REPLACE INTO idempotency_records "
                    "(tool, key, actor, args_digest, tool_version,"
                    " outcome_json, ts) VALUES (?,?,?,?,?,?,?)",
                    (rec.tool, rec.key, rec.actor, rec.args_digest,
                     rec.tool_version,
                     json.dumps(_encode_typed(rec.outcome.value)),
                     rec.created_at))
            except Exception:  # noqa: BLE001 - durability is best-effort
                pass

    def _idempotency_replay(
        self, name: str, key: str, actor: str, digest: str,
        tool_version: str = "",
    ) -> Outcome[Any] | None:
        """Replay / reject for a stored record. None = no record (miss)."""
        rec = self._idempotency_load(name, key)
        if rec is None:
            return None
        if rec.actor != actor:
            return Err(ValidationError(
                f"idempotency key {key!r} for tool {name!r} belongs to "
                f"another actor; result retrieval needs that actor's "
                f"authorization"))
        if rec.args_digest != digest:
            return Err(ValidationError(
                f"idempotency key {key!r} for tool {name!r} was already used "
                f"with different arguments (conflict) — refusing to return "
                f"an unrelated result or re-execute"))
        if rec.tool_version != tool_version:
            return Err(ValidationError(
                f"idempotency key {key!r} for tool {name!r} was recorded "
                f"under tool version {rec.tool_version!r} but the tool is "
                f"now version {tool_version!r} (conflict) — refusing to "
                f"return a stale result or re-execute"))
        return rec.outcome

    def _idempotency_acquire(self, name: str, key: str) -> bool:
        """Reserve (tool, key) for this caller. True = we own it.

        A concurrent duplicate waits for the owner's completion instead
        of executing twice. The wait is bounded: a pathological stall
        degrades to executing rather than deadlocking.
        """
        slot = (name, key)
        with self._idempotency_cond:
            while slot in self._idempotency_inflight:
                event = self._idempotency_inflight[slot]
                waited = self._idempotency_cond.wait_for(
                    lambda: slot not in self._idempotency_inflight,
                    timeout=120.0)
                if not waited:
                    _log.warning(
                        "idempotency: waited out for %s %r; executing anyway",
                        name, key)
                    return True
                _ = event
            self._idempotency_inflight[slot] = threading.Event()
            return True

    def _idempotency_release(self, name: str, key: str) -> None:
        slot = (name, key)
        with self._idempotency_cond:
            event = self._idempotency_inflight.pop(slot, None)
            if event is not None:
                event.set()
            self._idempotency_cond.notify_all()

    def _emit_tool_call(self, *, name: str, actor: str, digest: str,
                        ok: bool, seconds: float, error: str = "",
                        phase: str) -> None:
        """Publish a ``tool.call`` bus event (timeline/replay join key).

        Best-effort: a broken bus must never break tool execution.
        Correlation ids ride along so replay can join this call to its
        request/run/mission.
        """
        try:
            from ..core.correlation import current_correlation
            from ..core.events import Event, global_bus

            corr = current_correlation()
            global_bus.publish(Event(
                topic="tool.call",
                data={
                    "tool": name, "actor": actor, "args_digest": digest,
                    "ok": ok, "seconds": round(seconds, 3), "error": error,
                    "phase": phase,
                    "request_id": corr.get("request_id", ""),
                    "run_id": corr.get("run_id", ""),
                    "mission_id": corr.get("mission_id", ""),
                    "correlation_id": corr.get("run_id", "")
                    or corr.get("request_id", ""),
                },
                source="nomorals.tools.registry"))
        except Exception:  # noqa: BLE001 - events never break tools
            _log.debug("tool.call event failed", exc_info=True)

    def call(
        self,
        name: str,
        /,
        *args: Any,
        actor: str = "system",
        capabilities: CapabilitySet | None = None,
        confirmation: str | None = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> Outcome[Any]:
        """Check the capability, run the tool, and audit the call.

        ``idempotency_key`` makes a call safely repeatable: a previous
        successful result for the same (tool, key, actor, argument
        digest, tool version) is returned instead of re-executing —
        the MCP two-step rule for destructive work (preview, then commit
        with a key so a retry can't double-apply). Reuse with different
        arguments is a conflict error, never a stale replay; reuse by a
        different actor is denied. Concurrent duplicates execute once:
        the second caller waits for the owner's result. Successful
        results persist to the db when one is available, so a restart
        keeps the protection.
        """
        # Reject extra positional arguments with a proper error outcome
        if args:
            return Err(ToolError(
                f"call() takes 1 positional argument but {1 + len(args)} were given; "
                f"pass tool parameters as keywords"
            ))
        spec = self._tools.get(name)
        if spec is None:
            return Err(ToolNotFound(_unknown_tool_message(name, self._tools)))
        if not idempotency_key:
            started = time.perf_counter()
            self._emit_tool_call(name=name, actor=actor, digest="",
                                 ok=True, seconds=0.0, phase="start")
            outcome = self._call_inner(
                name, actor=actor, capabilities=capabilities,
                confirmation=confirmation, **kwargs)
            elapsed = time.perf_counter() - started
            self._emit_tool_call(
                name=name, actor=actor, digest=self._args_digest(kwargs),
                ok=bool(getattr(outcome, "ok", False)), seconds=elapsed,
                error="" if getattr(outcome, "ok", False)
                else str(getattr(getattr(outcome, "error", None),
                                 "message", "") or ""),
                phase="finish")
            return outcome

        digest = self._args_digest(kwargs)
        hit = self._idempotency_replay(name, idempotency_key, actor, digest,
                                       tool_version=spec.version)
        if hit is not None:
            if hit.ok:
                self._audit(name, actor, spec.capability, "allow", kwargs,
                            0.0, ok=True)
            return hit
        # Reserve: a concurrent duplicate waits for our result.
        self._idempotency_acquire(name, idempotency_key)
        try:
            # A waiter that found the slot freed re-checks: the owner
            # may have stored a result (replay) or failed (execute).
            hit = self._idempotency_replay(name, idempotency_key, actor,
                                           digest,
                                           tool_version=spec.version)
            if hit is not None:
                if hit.ok:
                    self._audit(name, actor, spec.capability, "allow", kwargs,
                                0.0, ok=True)
                return hit
            started = time.perf_counter()
            self._emit_tool_call(name=name, actor=actor, digest=digest,
                                 ok=True, seconds=0.0, phase="start")
            outcome = self._call_inner(
                name, actor=actor, capabilities=capabilities,
                confirmation=confirmation, **kwargs)
            elapsed = time.perf_counter() - started
            self._emit_tool_call(
                name=name, actor=actor, digest=digest,
                ok=bool(getattr(outcome, "ok", False)), seconds=elapsed,
                error="" if getattr(outcome, "ok", False)
                else str(getattr(getattr(outcome, "error", None),
                                 "message", "") or ""),
                phase="finish")
        finally:
            self._idempotency_release(name, idempotency_key)
        if outcome.ok:
            self._idempotency_store(_IdempotencyRecord(
                tool=name, key=idempotency_key, actor=actor,
                args_digest=digest, tool_version=spec.version,
                outcome=outcome, created_at=time.time()))
        return outcome

    def call_many(self, calls: list[tuple[str, dict[str, Any]]],
                  *, max_workers: int = 1, **common: Any) -> list[Outcome[Any]]:
        """Run several tool calls; ``max_workers > 1`` runs them in parallel
        (Phase D, bounded — default 8 in the explore phase) and preserves
        call order in the results.  ``max_workers=1`` is the old serial
        behavior."""
        if max_workers <= 1 or len(calls) <= 1:
            return [self.call(name, **{**common, **kwargs})
                    for name, kwargs in calls]

        # Pool threads do not inherit the caller's contextvars, so a
        # bare ``self.call`` on a worker thread runs with an EMPTY
        # context: correlation ids (request/run/mission) vanish from
        # the tool.call events those calls emit, and replay cannot
        # join them to their turn. Capture one context copy per call
        # HERE, on the submitting thread (each copy is entered by
        # exactly one worker), mirroring core/jobs.py's submit.
        captured = [(call, contextvars.copy_context()) for call in calls]

        def _one(item: tuple[tuple[str, dict[str, Any]],
                             contextvars.Context]) -> Outcome[Any]:
            (name, kwargs), ctx = item
            return ctx.run(self.call, name, **{**common, **kwargs})

        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="tool-call") as pool:
            return list(pool.map(_one, captured))

    def request_approval(
        self,
        name: str,
        *,
        actor: str = "",
        title: str = "",
        timeout_s: float = 60.0,
        args_digest: str | None = None,
    ) -> str | None:
        """Mint a confirmation token for a sensitive tool via fingerprint.

        ``args_digest`` binds the minted token to the exact arguments
        approved (see :mod:`nomorals.core.oversight`); oversight
        actions require a bound token.

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
            from ..native.biometric import biometric_available, request_biometric

            return approve_with_biometric(
                policy,
                spec.capability,
                title=title or f"approve {name}",
                timeout_s=timeout_s,
                biometric=(biometric_available, request_biometric),
                args_digest=args_digest,
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
            # Per-tool stats: what lets routing demote what keeps breaking.
            per = self._tool_stats.setdefault(name, {
                "calls": 0, "denied": 0, "errors": 0, "seconds": 0.0,
                "last_ts": 0.0, "last_status": "", "last_error": "",
                "last_duration_ms": 0.0,
            })
            status = entry["status"]
            per["calls"] += 1
            if status == "denied":
                per["denied"] += 1
            elif status == "error":
                per["errors"] += 1
            per["seconds"] = round(per["seconds"] + elapsed, 3)
            per["last_ts"] = entry["ts"]
            per["last_status"] = status
            per["last_error"] = error
            per["last_duration_ms"] = entry["duration_ms"]
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


def _unknown_tool_message(name: str, tools: dict[str, ToolSpec]) -> str:
    """Teachable 'unknown tool' error: suggest close names and the
    discovery tools instead of dumping the whole catalog."""
    import difflib

    candidates = difflib.get_close_matches(name, tools.keys(), n=5, cutoff=0.5)
    msg = f"unknown tool {name!r}."
    if candidates:
        msg += f" Did you mean: {', '.join(candidates)}?"
    msg += (" Use 'tools_list' to browse the catalog or describe() for one"
            " tool's schema.")
    return msg


def _maybe_truncate(result: Any, limit: int) -> Any:
    """Truncate over-long string results WITH an explicit notice.

    MCP rule: never truncate silently — the model must know output was
    cut so it can ask for the rest (pagination, offsets) instead of
    reasoning from a fragment as if it were complete.
    """
    if not isinstance(result, str) or limit <= 0 or len(result) <= limit:
        return result
    notice = (f"\n\n[…truncated: showing {limit:,} of {len(result):,} chars — "
              f"narrow the request or page through the source for the rest]")
    return result[:limit] + notice


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


#: Python type names -> JSON Schema type names. ``_type_name`` maps only
#: these exact simple names; anything else (``list[str]``, ``str | None``,
#: ``Optional[str]``, …) passes through unchanged.
_TYPE_ALIASES = {
    "str": "string",
    "int": "integer",
    "bool": "boolean",
    "float": "number",
    "list": "array",
    "dict": "object",
    "bytes": "string",
}


def _type_name(hint: Any) -> str:
    if not hint:
        return "any"
    if isinstance(hint, str):
        name = hint.lower()
    else:
        name = getattr(hint, "__name__", str(hint)).lower()
    return _TYPE_ALIASES.get(name, name)


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None)))
