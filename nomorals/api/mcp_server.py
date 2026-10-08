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
    research_run), resources/list, resources/read (devon://memory/*),
    prompts/list, prompts/get.

  NOT implemented → JSON-RPC -32601 (method_not_found), never fake success:
    sampling/* (server→client LLM requests), elicitation/*,
    roots/*, completion/*, logging/*, notifications/cancelled,
    notifications/progress — this server never asks the client for
    anything; every action is gated by the caller's capability grant.

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
SERVER_NAME = "devon"
SERVER_VERSION = "2.0"

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

EmitFn = Callable[[dict[str, Any]], None]


class MCPError(Exception):
    def __init__(self, message: str, code: int = INTERNAL_ERROR) -> None:
        super().__init__(message)
        self.code = code


# ── tool definitions ──────────────────────────────────────────────────────

_MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_query",
        "description": (
            "Query Devon's two-tier memory (facts + events). Facts rank "
            "first for direct questions; events for temporal context. "
            "Read-only."
        ),
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
    },
    {
        "name": "memory_write",
        "description": (
            "Write a durable first-person fact to Devon's memory "
            "(ADD-only: never deletes, only supersedes). Requires the "
            "owner token (mem.write)."
        ),
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
    },
    {
        "name": "tool_call",
        "description": (
            "Call one of Devon's registry tools by name. Capability-gated: "
            "the call runs with the caller's principal grant and the "
            "registry denies what the grant lacks."
        ),
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
    },
    {
        "name": "research_run",
        "description": (
            "Run an ad-hoc web research job on a topic. Returns "
            "deduplicated findings (title, url, snippet). Budget-capped."
        ),
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
        """Handle one JSON-RPC message. Never raises."""
        _ = emit  # MCP responses carry no server→client notifications
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
                result = self._handle(method, params)
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
    def _handle(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return self._initialize(params)
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": _MCP_TOOLS}
        if method == "tools/call":
            return self._tools_call(params)
        if method == "resources/list":
            return {"resources": _RESOURCES}
        if method == "resources/read":
            return self._resources_read(params)
        if method == "prompts/list":
            return {"prompts": _PROMPTS}
        if method == "prompts/get":
            return self._prompts_get(params)
        # Honest non-coverage: sampling/*, elicitation/*, roots/*,
        # completion/*, logging/* and every other unimplemented method.
        raise MCPError(f"method not implemented: {method}",
                       METHOD_NOT_FOUND)

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        client_version = (params or {}).get("protocolVersion", "")
        _log.info("mcp: initialize from client protocolVersion=%r",
                  client_version)
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    # ── tools ──────────────────────────────────────────────────────────
    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise MCPError("tools/call requires 'name'", INVALID_PARAMS)
        if not isinstance(arguments, dict):
            raise MCPError("'arguments' must be an object", INVALID_PARAMS)
        handler = {
            "memory_query": self._tool_memory_query,
            "memory_write": self._tool_memory_write,
            "tool_call": self._tool_tool_call,
            "research_run": self._tool_research_run,
        }.get(name)
        if handler is None:
            return self._tool_error(f"unknown tool {name!r}")
        try:
            return handler(arguments)
        except MCPError as exc:
            return self._tool_error(str(exc))
        except Exception as exc:  # noqa: BLE001
            _log.exception("mcp: tool %s failed", name)
            return self._tool_error(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _tool_ok(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}], "isError": False}

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
        return self._tool_ok(json.dumps({
            "query": query,
            "facts": list(recall.texts or [])[:limit],
        }, ensure_ascii=False))

    def _tool_memory_write(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require_cap("mem.write")
        text = (args.get("text") or "").strip()
        if not text:
            raise MCPError("'text' must not be empty", INVALID_PARAMS)
        confidence = max(0.0, min(1.0, float(args.get("confidence", 0.7))))
        fact = self.two_tier.facts.add_fact(text, confidence=confidence)
        return self._tool_ok(json.dumps({
            "id": fact.id, "text": fact.text, "confidence": fact.confidence,
        }, ensure_ascii=False))

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
            return self._tool_ok(
                value if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, default=str))
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
        return self._tool_ok(json.dumps([
            {"title": f.title, "url": f.url, "snippet": f.snippet,
             "detail": f.detail}
            for f in findings
        ], ensure_ascii=False))

    # ── resources ──────────────────────────────────────────────────────
    def _resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if uri == "devon://memory/facts":
            payload = self._list_active_facts()
        elif uri == "devon://memory/entities":
            payload = self._read_entities()
        elif uri == "devon://memory/timeline":
            payload = self._read_timeline()
        else:
            raise MCPError(f"unknown resource {uri!r}", INVALID_PARAMS)
        return {"contents": [{
            "uri": uri, "mimeType": "application/json",
            "text": json.dumps(payload, ensure_ascii=False)}]}

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
            _ = notification  # no server→client notifications in MCP core

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
        """Buffered HTTP: one JSON-RPC message/batch → response payload."""

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
