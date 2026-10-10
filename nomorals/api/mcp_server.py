"""Model Context Protocol (MCP) server — Devon as infrastructure.

Exposes Devon's memory (query/write) and tool registry to external agents
(Claude, Cursor, Cline, ...) over the Model Context Protocol
(spec: https://modelcontextprotocol.io, protocol version 2025-11-25).
Transport: JSON-RPC 2.0 over stdio (newline-delimited; ``Content-Length``
framing also accepted) or HTTP ``POST /mcp`` (buffered).

Run over stdio (how MCP clients spawn the server)::

    python -m nomorals.api.mcp_server --stdio

Or mount on the existing API server::

    from nomorals.api.mcp_server import MCPServer, register_mcp
    mcp = MCPServer(context)
    register_mcp(api_server, mcp)   # adds POST /mcp

COVERAGE — honest accounting of the spec surface:

  Implemented (client → server direction):
    initialize (+ notifications/initialized, ping),
    tools/list, tools/call (memory_query, memory_write, tool_call,
    research_run; every tool carries spec ToolAnnotations, a title, and an
    outputSchema — results return both text content and structuredContent),
    resources/list, resources/templates/list, resources/read
    (devon://memory/* incl. devon://memory/facts/{id}),
    prompts/list, prompts/get,
    completion/complete (prompt-argument and resource-URI autocompletion),
    logging/setLevel (+ notifications/message),
    notifications/progress (honors _meta.progressToken on tools/call).

  NOT implemented → JSON-RPC -32601 (method_not_found), never fake success:
    sampling/* (server→client LLM requests), elicitation/*,
    roots/*, notifications/cancelled — this server never asks the client
    for anything; every action is gated by the caller's capability grant.

Auth: the SAME Principal/capability model as ``nomorals.api.server`` —
nothing weaker. Read-only methods (memory_query, resources) run under
the request principal (DEFAULT_PRINCIPAL = bounded local grant when no
token is configured). Write/tool methods require the principal to hold
the relevant capability: memory_write needs mem.write; tool_call is
gated per-tool by the registry against the principal's grant; the
registry denies what the grant lacks and audits the rest.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

MCP_PROTOCOL_VERSION = "2025-11-25"

#: Protocol versions we can speak, newest first. initialize negotiates the
#: highest version both sides support (never above our own).
_MCP_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

SERVER_NAME = "devon"
SERVER_VERSION = "2.0"

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
#: Resource not found (spec §server/resources error codes).
RESOURCE_NOT_FOUND = -32002

#: MCP log levels (syslog severity names, per the spec).
_LOG_LEVELS = ("debug", "info", "notice", "warning", "error", "critical",
               "alert", "emergency")

EmitFn = Callable[[dict[str, Any]], None]


class MCPError(Exception):
    def __init__(self, message: str, code: int = INTERNAL_ERROR) -> None:
        super().__init__(message)
        self.code = code


# ── tool definitions ──────────────────────────────────────────────────────
# Annotations follow the spec's ToolAnnotations (readOnlyHint,
# destructiveHint, idempotentHint, openWorldHint): hints, not enforcement —
# the capability grant in _tools_call is what actually gates.

_MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_query",
        "title": "Query memory",
        "description": (
            "Query Devon's two-tier memory (facts + events). Facts rank "
            "first for direct questions; events for temporal context. "
            "Read-only."
        ),
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "natural-language memory query"},
                "limit": {"type": "integer", "default": 8, "minimum": 1,
                          "maximum": 50},
                "hybrid": {"type": "boolean", "default": False,
                           "description": "fuse a BM25 keyword lane via RRF"},
            },
            "required": ["query"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "facts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query", "facts"],
        },
    },
    {
        "name": "memory_write",
        "title": "Write memory fact",
        "description": (
            "Write a durable first-person fact to Devon's memory "
            "(ADD-only: never deletes, only supersedes). Requires the "
            "owner token (mem.write)."
        ),
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": "atomic first-person fact, e.g. "
                                        "'my girlfriend is Ada'"},
                "confidence": {"type": "number", "default": 0.7,
                               "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["text"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "text": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["id", "text", "confidence"],
        },
    },
    {
        "name": "tool_call",
        "title": "Call a Devon tool",
        "description": (
            "Call one of Devon's registry tools by name. Capability-gated: "
            "the call runs with the caller's principal grant and the "
            "registry denies what the grant lacks."
        ),
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string",
                         "description": "registered tool name"},
                "arguments": {"type": "object", "default": {},
                              "description": "tool parameters"},
            },
            "required": ["tool"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "ok": {"type": "boolean"},
            },
            "required": ["tool", "ok"],
        },
    },
    {
        "name": "research_run",
        "title": "Run web research",
        "description": (
            "Run an ad-hoc web research job on a topic. Returns "
            "deduplicated findings (title, url, snippet). Budget-capped."
        ),
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"},
                            "description": "search queries (default: [topic])"},
                "max_results": {"type": "integer", "default": 6,
                                "minimum": 1, "maximum": 20},
            },
            "required": ["topic"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                            "detail": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["topic", "findings"],
        },
    },
]

_RESOURCES: list[dict[str, Any]] = [
    {"uri": "devon://memory/facts", "name": "memory-facts",
     "description": "Active first-person facts in Devon's two-tier memory",
     "mimeType": "application/json"},
    {"uri": "devon://memory/entities", "name": "memory-entities",
     "description": "Typed memory entities (people, projects, commitments)",
     "mimeType": "application/json"},
    {"uri": "devon://memory/timeline", "name": "memory-timeline",
     "description": "How beliefs evolved (supersede chains, newest first)",
     "mimeType": "application/json"},
]

_RESOURCE_TEMPLATES: list[dict[str, Any]] = [
    {"uriTemplate": "devon://memory/facts/{id}", "name": "memory-fact",
     "description": "A single active memory fact, addressed by its id",
     "mimeType": "application/json"},
]

_PROMPTS: list[dict[str, Any]] = [
    {"name": "devon-brief",
     "description": "Ask Devon for a concise briefing on a topic, "
                    "grounded in her memory",
     "arguments": [{"name": "topic", "description": "what to brief on",
                    "required": True}]},
]


class MCPServer:
    """MCP server endpoint. Transport-agnostic core; see module docstring."""

    def __init__(
        self,
        context: Any = None,
        registry: Any = None,
        *,
        principal: Any = None,
        principal_resolver: Callable[[], Any] | None = None,
        memory: Any = None,
    ) -> None:
        self.context = context
        self.registry = registry if registry is not None else \
            getattr(context, "tools", None)
        self._principal = principal
        self._principal_resolver = principal_resolver
        self._memory = memory
        self._lock = threading.RLock()
        # logging/setLevel state; None = client never asked for log
        # notifications, so _notify() stays silent.
        self._log_level: str | None = None
        # The emit fn for the in-flight request (dispatch installs it per
        # call); used by _notify() and progress reporting.
        self._emit_tls = threading.local()
        if self._principal is None and self._principal_resolver is None:
            from .server import DEFAULT_PRINCIPAL
            self._principal = DEFAULT_PRINCIPAL

    # ── principal / memory ─────────────────────────────────────────────
    @property
    def principal(self) -> Any:
        if self._principal_resolver is not None:
            try:
                return self._principal_resolver()
            except Exception:  # noqa: BLE001 - resolver must never break us
                pass
        return self._principal

    @property
    def two_tier(self) -> Any:
        if self._memory is None:
            from ..memory.tiers import TwoTierMemory
            self._memory = TwoTierMemory()
        return self._memory

    # ── JSON-RPC plumbing ──────────────────────────────────────────────
    def _current_emit(self) -> EmitFn | None:
        return getattr(self._emit_tls, "emit", None)

    def _notify(self, level: str, message: str,
                *, logger: str = "devon") -> None:
        """Push a ``notifications/message`` to the client when it asked for
        log notifications (``logging/setLevel``) and ``level`` passes the
        threshold. Silent otherwise — stdio stays pure JSON-RPC lines."""
        if self._log_level is None:
            return
        if _LOG_LEVELS.index(level) < _LOG_LEVELS.index(self._log_level):
            return
        emit = self._current_emit()
        if emit is None:
            return
        try:
            emit({"jsonrpc": "2.0", "method": "notifications/message",
                  "params": {"level": level, "logger": logger,
                             "data": message}})
        except Exception:  # noqa: BLE001 - notifications never break us
            pass

    def _report_progress(self, token: str | int | None, progress: float,
                         *, total: float | None = None,
                         message: str = "") -> None:
        """Push a ``notifications/progress`` for a client-supplied
        ``_meta.progressToken``. No token → no-op (spec: the receiver MAY
        choose not to send any)."""
        if token is None:
            return
        emit = self._current_emit()
        if emit is None:
            return
        params: dict[str, Any] = {"progressToken": token,
                                  "progress": progress}
        if total is not None:
            params["total"] = total
        if message:
            params["message"] = message
        try:
            emit({"jsonrpc": "2.0", "method": "notifications/progress",
                  "params": params})
        except Exception:  # noqa: BLE001 - notifications never break us
            pass
    @staticmethod
    def _ok(msg_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": code, "message": message}}

    def _require_cap(self, capability: str) -> None:
        from ..core.policy import Capability
        from .server import _require_capability
        _ = Capability  # namespace import check
        _require_capability(self.principal, capability)

    def dispatch(self, msg: Any, emit: EmitFn) -> dict[str, Any] | None:
        """Handle one JSON-RPC message. Never raises.

        ``emit`` carries server→client notifications for the duration of
        this call (progress, log messages); it is installed on a
        thread-local so tool handlers can reach it.
        """
        previous = getattr(self._emit_tls, "emit", None)
        self._emit_tls.emit = emit
        try:
            return self._dispatch_inner(msg, emit)
        finally:
            if previous is None:
                try:
                    del self._emit_tls.emit
                except AttributeError:
                    pass
            else:
                self._emit_tls.emit = previous

    def _dispatch_inner(self, msg: Any, emit: EmitFn) -> dict[str, Any] | None:
        _ = emit  # notifications flow through the thread-local emit
        try:
            if not isinstance(msg, dict):
                return self._err(None, INVALID_REQUEST,
                                 "message must be a JSON object")
            if msg.get("jsonrpc") != "2.0" or "method" not in msg:
                return self._err(msg.get("id"), INVALID_REQUEST,
                                 "not a JSON-RPC 2.0 request")
            method = msg["method"]
            msg_id = msg.get("id")  # absent → notification
            params = msg.get("params") or {}
            if not isinstance(params, dict):
                if msg_id is None:
                    return None
                return self._err(msg_id, INVALID_PARAMS,
                                 "params must be an object")
            try:
                result = self._handle(method, params, emit)
            except MCPError as exc:
                if msg_id is None:
                    return None
                return self._err(msg_id, exc.code, str(exc))
            except Exception as exc:  # noqa: BLE001 - never leak a traceback
                _log.exception("mcp: error in %s", method)
                if msg_id is None:
                    return None
                return self._err(msg_id, INTERNAL_ERROR,
                                 f"{type(exc).__name__}: {exc}")
            if msg_id is None:
                return None
            return self._ok(msg_id, result)
        except Exception as exc:  # noqa: BLE001 - dispatch never raises
            _log.exception("mcp: dispatch crashed")
            return self._err(None, INTERNAL_ERROR, f"{type(exc).__name__}")

    def handle_batch(self, msgs: Any, emit: EmitFn) -> Any:
        if isinstance(msgs, list):
            if not msgs:
                return self._err(None, INVALID_REQUEST, "empty batch")
            out = [self.dispatch(m, emit) for m in msgs]
            return [r for r in out if r is not None]
        return self.dispatch(msgs, emit)

    # ── method handlers ────────────────────────────────────────────────
    def _handle(self, method: str, params: dict[str, Any],
                emit: EmitFn) -> Any:
        if method == "initialize":
            return self._initialize(params)
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": _MCP_TOOLS}
        if method == "tools/call":
            return self._tools_call(params, emit)
        if method == "resources/list":
            return {"resources": _RESOURCES}
        if method == "resources/templates/list":
            return {"resourceTemplates": _RESOURCE_TEMPLATES}
        if method == "resources/read":
            return self._resources_read(params)
        if method == "prompts/list":
            return {"prompts": _PROMPTS}
        if method == "prompts/get":
            return self._prompts_get(params)
        if method == "completion/complete":
            return self._completion_complete(params)
        if method == "logging/setLevel":
            return self._logging_set_level(params)
        # Honest non-coverage: sampling/*, elicitation/*, roots/* and
        # every other unimplemented method.
        raise MCPError(f"method not implemented: {method}",
                       METHOD_NOT_FOUND)

    @staticmethod
    def _negotiate_version(requested: Any) -> str:
        """Highest protocol version both sides support (never above ours)."""
        if isinstance(requested, str) and requested in _MCP_VERSIONS:
            return requested
        return MCP_PROTOCOL_VERSION

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        client_version = (params or {}).get("protocolVersion", "")
        version = self._negotiate_version(client_version)
        _log.info("mcp: initialize from client protocolVersion=%r "
                  "→ negotiated %r", client_version, version)
        self._notify("info",
                     f"client initialized (protocol {version})")
        return {
            "protocolVersion": version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
                "logging": {},
                "completions": {},
            },
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _logging_set_level(self, params: dict[str, Any]) -> dict[str, Any]:
        level = str(params.get("level") or "").lower()
        if level not in _LOG_LEVELS:
            raise MCPError(
                f"unknown log level {params.get('level')!r}; "
                f"expected one of {', '.join(_LOG_LEVELS)}", INVALID_PARAMS)
        self._log_level = level
        _log.info("mcp: client set log level %s", level)
        return {}

    # ── tools ──────────────────────────────────────────────────────────
    def _tools_call(self, params: dict[str, Any],
                    emit: EmitFn) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise MCPError("tools/call requires 'name'", INVALID_PARAMS)
        if not isinstance(arguments, dict):
            raise MCPError("'arguments' must be an object", INVALID_PARAMS)
        meta = params.get("_meta") or {}
        progress_token = meta.get("progressToken") if isinstance(
            meta, dict) else None
        if progress_token is not None and not isinstance(
                progress_token, (str, int)):
            raise MCPError("'progressToken' must be a string or integer",
                           INVALID_PARAMS)
        handler = {
            "memory_query": self._tool_memory_query,
            "memory_write": self._tool_memory_write,
            "tool_call": self._tool_tool_call,
            "research_run": self._tool_research_run,
        }.get(name)
        if handler is None:
            self._notify("error", f"unknown tool {name!r}")
            return self._tool_error(f"unknown tool {name!r}")
        # Progress: the spec lets the receiver MAY send notifications; the
        # progress value MUST increase and MUST stop after completion.
        self._report_progress(progress_token, 1, message=f"running {name}")
        try:
            return handler(arguments)
        except MCPError as exc:
            self._notify("error", f"tool {name} failed: {exc}")
            return self._tool_error(str(exc))
        except Exception as exc:  # noqa: BLE001
            _log.exception("mcp: tool %s failed", name)
            self._notify("error", f"tool {name} failed: {type(exc).__name__}")
            return self._tool_error(f"{type(exc).__name__}: {exc}")
        finally:
            self._report_progress(progress_token, 2,
                                  message=f"finished {name}")

    @staticmethod
    def _tool_ok(text: str,
                 structured: dict[str, Any] | None = None) -> dict[str, Any]:
        # Both shapes, like the official SDK: structuredContent for modern
        # clients, plain text content for backward compatibility.
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }
        if structured is not None:
            result["structuredContent"] = structured
        return result

    @staticmethod
    def _tool_error(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}], "isError": True}

    def _tool_memory_query(self, args: dict[str, Any]) -> dict[str, Any]:
        query = (args.get("query") or "").strip()
        if not query:
            raise MCPError("'query' must not be empty", INVALID_PARAMS)
        limit = max(1, min(50, int(args.get("limit", 8))))
        hybrid = bool(args.get("hybrid", False))
        recall = self.two_tier.recall(query, limit=limit, hybrid=hybrid)
        facts = list(recall.texts or [])[:limit]
        structured = {"query": query, "facts": facts}
        return self._tool_ok(json.dumps(structured, ensure_ascii=False),
                             structured)

    def _tool_memory_write(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require_cap("mem.write")
        text = (args.get("text") or "").strip()
        if not text:
            raise MCPError("'text' must not be empty", INVALID_PARAMS)
        confidence = max(0.0, min(1.0, float(args.get("confidence", 0.7))))
        fact = self.two_tier.facts.add_fact(text, confidence=confidence)
        structured = {"id": fact.id, "text": fact.text,
                      "confidence": fact.confidence}
        self._notify("info", f"memory fact written: {fact.id}")
        return self._tool_ok(json.dumps(structured, ensure_ascii=False),
                             structured)

    def _tool_tool_call(self, args: dict[str, Any]) -> dict[str, Any]:
        tool = args.get("tool")
        tool_args = args.get("arguments") or {}
        if self.registry is None:
            return self._tool_error("no tool registry configured")
        from ..core.result import Ok
        outcome = self.registry.call(
            tool, actor=f"mcp:{self.principal.name}",
            capabilities=self.principal.grant, **tool_args)
        if isinstance(outcome, Ok):
            value = outcome.unwrap()
            structured = {"tool": tool, "ok": True}
            text = (value if isinstance(value, str)
                    else json.dumps(value, ensure_ascii=False, default=str))
            return self._tool_ok(text, structured)
        err = outcome.error if hasattr(outcome, "error") else outcome
        return self._tool_error(str(getattr(err, "message", err)))

    def _tool_research_run(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require_cap("model.call")
        topic = (args.get("topic") or "").strip()
        if not topic:
            raise MCPError("'topic' must not be empty", INVALID_PARAMS)
        queries = args.get("queries") or [topic]
        max_results = max(1, min(20, int(args.get("max_results", 6))))
        if self.registry is None:
            return self._tool_error("no tool registry configured "
                                    "(research needs web_search)")
        from ..research.pipeline import ResearchContext, ResearchJob, run_job
        job = ResearchJob(topic=topic, queries=list(queries),
                          max_results=max_results)
        rctx = ResearchContext(db=getattr(self.context, "db", None),
                               registry=self.registry)
        findings = run_job(job, rctx)
        structured = {
            "topic": topic,
            "findings": [
                {"title": f.title, "url": f.url, "snippet": f.snippet}
                for f in findings
            ],
        }
        return self._tool_ok(json.dumps(structured, ensure_ascii=False),
                             structured)

        structured = {
            "topic": topic,
            "findings": [
                {"title": f.title, "url": f.url, "snippet": f.snippet,
                 "detail": f.detail}
                for f in findings
            ],
        }
        return self._tool_ok(json.dumps(structured, ensure_ascii=False),
                             structured)

    # ── completion ─────────────────────────────────────────────────────
    def _completion_complete(self, params: dict[str, Any]) -> dict[str, Any]:
        """Argument autocompletion (spec: server/utilities/completion).

        ``ref`` is ``{"type": "ref/prompt", "name": ...}`` or
        ``{"type": "ref/resource", "uri": ...}``; ``argument`` is
        ``{"name": ..., "value": ...}``. The referenced prompt/resource
        must exist. At most 100 values are returned (spec cap).
        """
        ref = params.get("ref") or {}
        argument = params.get("argument") or {}
        arg_name = argument.get("name")
        value = str(argument.get("value") or "")
        ref_type = ref.get("type")
        if ref_type == "ref/prompt":
            name = ref.get("name")
            if not any(p["name"] == name for p in _PROMPTS):
                raise MCPError(f"unknown prompt {name!r}", INVALID_PARAMS)
            values = self._complete_prompt_argument(
                str(name), str(arg_name or ""), value)
        elif ref_type == "ref/resource":
            uri = str(ref.get("uri") or "")
            values = self._complete_resource_argument(
                uri, str(arg_name or ""), value)
        else:
            raise MCPError(f"unknown completion ref type {ref_type!r}",
                           INVALID_PARAMS)
        values = [v for v in values if v.startswith(value)][:100]
        return {"completion": {"values": values, "total": len(values),
                               "hasMore": False}}

    def _complete_prompt_argument(self, prompt_name: str, arg_name: str,
                                  value: str) -> list[str]:
        _ = value  # prefix filtering happens in _completion_complete
        if prompt_name == "devon-brief" and arg_name == "topic":
            # Real suggestions: recent active memory facts make good
            # briefing topics.
            try:
                facts = self._list_active_facts(limit=50)
                seen: set[str] = set()
                out: list[str] = []
                for f in facts:
                    topic = str(f.get("text") or "").strip()
                    if topic and topic not in seen:
                        seen.add(topic)
                        out.append(topic[:120])
                return out
            except Exception:  # noqa: BLE001 - completion never breaks
                return []
        return []

    def _complete_resource_argument(self, uri: str, arg_name: str,
                                    value: str) -> list[str]:
        _ = value
        if uri == "devon://memory/facts/{id}" and arg_name == "id":
            try:
                return [str(f.get("id"))
                        for f in self._list_active_facts(limit=100)
                        if f.get("id")]
            except Exception:  # noqa: BLE001 - completion never breaks
                return []
        if not uri or uri == "devon://memory/facts":
            return [r["uri"] for r in _RESOURCES] + [
                t["uriTemplate"] for t in _RESOURCE_TEMPLATES]
        return []

    # ── resources ──────────────────────────────────────────────────────
    def _resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if uri == "devon://memory/facts":
            payload = self._list_active_facts()
        elif uri == "devon://memory/entities":
            payload = self._read_entities()
        elif uri == "devon://memory/timeline":
            payload = self._read_timeline()
        elif isinstance(uri, str) and uri.startswith(
                "devon://memory/facts/"):
            fact_id = uri[len("devon://memory/facts/"):]
            fact = self._read_fact(fact_id)
            if fact is None:
                raise MCPError(f"unknown resource {uri!r}",
                               RESOURCE_NOT_FOUND)
            payload = fact
        else:
            raise MCPError(f"unknown resource {uri!r}", INVALID_PARAMS)
        return {"contents": [{
            "uri": uri, "mimeType": "application/json",
            "text": json.dumps(payload, ensure_ascii=False)}]}

    def _read_fact(self, fact_id: str) -> dict[str, Any] | None:
        """One active fact by id (backs the resource template)."""
        try:
            rows = self.two_tier.db.query(
                "SELECT id, text, confidence, valid_from, valid_to "
                "FROM tier_facts WHERE id = ? AND active = 1", (fact_id,))
        except Exception:  # noqa: BLE001
            _log.debug("mcp: fact read failed", exc_info=True)
            return None
        if not rows:
            return None
        r = rows[0]
        return {"id": str(r["id"]), "text": str(r["text"]),
                "confidence": float(r["confidence"]),
                "valid_from": float(r["valid_from"]),
                "valid_to": (float(r["valid_to"])
                             if r["valid_to"] is not None else None)}

    def _list_active_facts(self, limit: int = 100) -> list[dict[str, Any]]:
        try:
            rows = self.two_tier.db.query(
                "SELECT id, text, confidence, valid_from, valid_to "
                "FROM tier_facts WHERE active = 1 "
                "ORDER BY created_at DESC LIMIT ?", (limit,))
            return [{"id": str(r["id"]), "text": str(r["text"]),
                     "confidence": float(r["confidence"]),
                     "valid_from": float(r["valid_from"]),
                     "valid_to": (float(r["valid_to"])
                                  if r["valid_to"] is not None else None)}
                    for r in rows]
        except Exception:  # noqa: BLE001
            _log.debug("mcp: fact listing failed", exc_info=True)
            return []

    def _read_timeline(self, limit: int = 100) -> list[dict[str, Any]]:
        try:
            rows = self.two_tier.db.query(
                "SELECT id FROM tier_facts ORDER BY created_at DESC LIMIT ?",
                (limit,))
            payload: list[dict[str, Any]] = []
            seen: set[str] = set()
            for r in rows:
                for f in self.two_tier.facts.history(str(r["id"])):
                    if f.id not in seen:
                        seen.add(f.id)
                        payload.append({
                            "id": f.id, "text": f.text,
                            "valid_from": f.valid_from,
                            "valid_to": f.valid_to,
                            "supersedes": f.supersedes})
            payload.sort(key=lambda f: f["valid_from"], reverse=True)
            return payload[:limit]
        except Exception:  # noqa: BLE001
            _log.debug("mcp: timeline read failed", exc_info=True)
            return []

    def _read_entities(self) -> list[dict[str, Any]]:
        try:
            from ..memory.types import ENTITY_TYPES
            from pathlib import Path
            import os as _os
            base = Path(_os.path.expanduser("~/.nomorals/memory/entities"))
            out: list[dict[str, Any]] = []
            if base.is_dir():
                for etype in ENTITY_TYPES:
                    d = base / etype
                    if not d.is_dir():
                        continue
                    for fp in sorted(d.glob("*.json"))[:200]:
                        try:
                            out.append(json.loads(fp.read_text()))
                        except Exception:  # noqa: BLE001 - skip bad files
                            continue
            return out
        except Exception:  # noqa: BLE001
            _log.debug("mcp: entity read failed", exc_info=True)
            return []

    def _prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if name != "devon-brief":
            raise MCPError(f"unknown prompt {name!r}", INVALID_PARAMS)
        args = params.get("arguments") or {}
        topic = (args.get("topic") or "today").strip()
        return {
            "description": "Ask Devon for a concise briefing",
            "messages": [{
                "role": "user",
                "content": {"type": "text",
                            "text": f"Give me a concise briefing on {topic}, "
                                    "grounded in what you remember about me."},
            }],
        }

    # ── stdio transport ────────────────────────────────────────────────
    @staticmethod
    def _read_stdio_message(stream: Any) -> dict[str, Any] | None:
        first = stream.readline()
        if not first:
            return None
        if first.strip().lower().startswith(b"content-length:"):
            length = int(first.split(b":", 1)[1].strip())
            while True:
                line = stream.readline()
                if not line or line.strip() in (b"", b"\r"):
                    break
            raw = stream.read(length)
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
        line = first.strip()
        if not line:
            return None
        return json.loads(line.decode("utf-8"))

    def run_stdio(self, stdin: Any = None, stdout: Any = None) -> int:
        """Serve MCP over stdio. Logs go to stderr; stdout is pure
        JSON-RPC. Returns 0 on clean EOF."""
        stdin = stdin if stdin is not None else sys.stdin.buffer
        stdout = stdout if stdout is not None else sys.stdout.buffer

        def emit(notification: dict[str, Any]) -> None:
            # Real server→client notifications (notifications/progress,
            # notifications/message) ride the same stdout channel as
            # responses — plain newline-delimited JSON-RPC frames.
            stdout.write((json.dumps(notification, ensure_ascii=False) + "\n"
                          ).encode("utf-8"))
            stdout.flush()

        _log.info("mcp: serving over stdio")
        while True:
            try:
                msg = self._read_stdio_message(stdin)
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
                stdout.write((json.dumps(
                    self._err(None, PARSE_ERROR, f"parse error: {exc}")
                ) + "\n").encode("utf-8"))
                stdout.flush()
                continue
            except OSError as exc:
                _log.info("mcp: stdio read failed: %s", exc)
                break
            if msg is None:
                break
            response = self.handle_batch(msg, emit)
            if response is None:
                continue
            if isinstance(response, list) and not response:
                continue
            stdout.write((json.dumps(response, ensure_ascii=False) + "\n"
                          ).encode("utf-8"))
            stdout.flush()
        _log.info("mcp: stdio EOF, shutting down")
        return 0

    # ── HTTP transport ─────────────────────────────────────────────────
    def handle_http(self, body: Any) -> Any:
        """Buffered HTTP: one JSON-RPC message/batch → response payload.

        Server→client notifications (progress, log messages) are only
        delivered over stdio; buffered HTTP returns just the final
        response (same documented trade-off as ACP's handle_http)."""

        def emit(notification: dict[str, Any]) -> None:
            _ = notification

        return self.handle_batch(body, emit)


# ── mounting on the existing API server ────────────────────────────────

def register_mcp(api_server: Any, mcp: MCPServer) -> None:
    """Add ``POST /mcp`` (buffered JSON-RPC) to an APIServer, following the
    register_acp pattern."""
    api_server._mcp_server = mcp  # noqa: SLF001 - available for introspection

    @api_server.route("POST", "/mcp",
                      description="Model Context Protocol: JSON-RPC 2.0 "
                                  "endpoint (initialize, tools/list, "
                                  "tools/call, resources/*). Buffered.")
    def mcp_endpoint(body: dict[str, Any],
                     query: dict[str, str]) -> dict[str, Any]:
        _ = query
        return mcp.handle_http(body)


# ── stdio entry point ─────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Devon MCP server (Model Context Protocol) over stdio")
    parser.add_argument("--stdio", action="store_true",
                        help="serve JSON-RPC 2.0 over stdin/stdout")
    args = parser.parse_args(argv)
    if not args.stdio:
        parser.error("--stdio is required (MCP clients spawn this over stdio)")
    # Lazy: stdio mode must not require a full app boot.
    mcp = MCPServer()
    return mcp.run_stdio()


if __name__ == "__main__":
    sys.exit(main())
