"""Agent Client Protocol (ACP) — agent-side implementation.

Makes Devon interoperable with ACP clients (Zed, JetBrains IDEs, Neovim,
...) as a backend coding agent. Spec: https://agentclientprotocol.com
(protocol version 1). Transport: JSON-RPC 2.0 over stdio
(newline-delimited; ``Content-Length`` framing also accepted) or HTTP
``POST /acp`` (buffered, or SSE streaming with
``Accept: text/event-stream``).

Run over stdio (how editors spawn the agent)::

    python -m nomorals.api.acp --stdio

Or mount on the existing API server::

    from nomorals.api.acp import ACPServer, register_acp
    acp = ACPServer(context)
    register_acp(api_server, acp)   # adds POST /acp (+ SSE)

COVERAGE — honest accounting of the spec surface:

  Implemented (agent → client direction):
    initialize, authenticate (accepted; this agent declares no
    authMethods, so it is a no-op), session/new, session/load,
    session/prompt, session/cancel (notification; cooperative — checked
    at tool-call boundaries), session/set_mode, session/set_model,
    devon/tools (Devon extension, NOT part of the ACP spec: lists the
    tool registry filtered to the session principal's capability grant).

  session/new mcpServers — declared servers are wired into each
    turn's registry view (per-turn scope: the view wraps the shared
    registry, the session's MCP connections live on the session and
    close when it is pruned or expires). initialize reports
    mcpCapabilities http True (Streamable HTTP is spoken), sse False
    (the legacy SSE transport is not implemented).

  session/update notifications: agent_message_chunk (completed message in
    chunks — the loop does not token-stream), tool_call / tool_call_update
    (live, around each real tool invocation, with spec ToolKind, rawInput,
    rawOutput, and file locations for follow-along), plan (turn goal,
    in_progress → completed), available_commands_update (on session/new,
    from the capability-filtered registry).

  NOT implemented → JSON-RPC -32601 (method_not_found), never fake success:
    session/request_permission — agent→client direction; this agent never
    requests permissions because every tool call is gated by the session
    principal's capability grant instead (deny-by-default).
    fs/*, terminal/* — client-side methods; not our role as the agent.

  Streaming notes: the orchestration loop does not token-stream, so
  ``agent_message_chunk`` updates carry the completed message in chunks,
  while ``tool_call`` / ``tool_call_update`` notifications are emitted
  live around each real tool invocation. ``session/prompt`` resolves with
  a real ``stopReason``: ``end_turn``, ``cancelled``, or ``refusal``
  (loop asked the user / failed closed).

Auth: the SAME Principal/capability model as ``nomorals.api.server`` —
nothing weaker. Every tool call inside a turn runs as
``actor="acp:<session-id>"`` with the session principal's grant; the
registry denies what the grant lacks and audits the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.ids import new_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Resolver identities whose failure has already been warned about —
#: the tail-66 visibility sketch: a broken resolver warns ONCE per
#: identity, so the failure is findable without spamming every call.
_RESOLVER_FAILURE_WARNED: set[int] = set()


def _warn_resolver_failure(resolver: Any, exc: BaseException) -> None:
    key = id(resolver)
    if key in _RESOLVER_FAILURE_WARNED:
        return
    _RESOLVER_FAILURE_WARNED.add(key)
    _log.warning(
        "acp: principal resolver failed (%s: %s); falling back to the "
        "configured principal", type(exc).__name__, exc)

PROTOCOL_VERSION = 1

# ── JSON-RPC error codes ──────────────────────────────────────────────────
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

_STOP_REASONS = ("end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled")

#: Methods this agent answers. Everything else → -32601.
_IMPLEMENTED = frozenset({
    "initialize",
    "authenticate",
    "session/new",
    "session/load",
    "session/prompt",
    "session/cancel",
    "session/set_mode",
    "session/set_model",
    "devon/tools",
})

#: Registry tool name → ACP ToolKind. First token-prefix match wins;
#: "other" is the schema default. (Spec: the kind "helps clients choose
#: appropriate icons and UI treatment".)
_TOOL_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("read", ("read", "get", "list", "show", "cat", "load")),
    ("edit", ("write", "edit", "update", "set", "create", "apply", "patch",
               "append")),
    ("delete", ("delete", "remove", "rm", "purge", "drop", "forget")),
    ("move", ("move", "rename", "mv")),
    ("search", ("search", "query", "grep", "find", "lookup", "recall")),
    ("execute", ("run", "exec", "shell", "bash", "command", "eval",
                 "spawn")),
    ("fetch", ("fetch", "download", "http", "web", "browser", "curl",
               "wget", "scrape")),
    ("think", ("think", "plan", "reason", "reflect", "decide")),
)

#: Tool kwarg names that carry an absolute file path → ToolCallLocation
#: ("enables follow-along features in clients").
_PATH_ARG_KEYS = ("path", "file", "filepath", "filename", "target_path")


def _tool_kind(name: str) -> str:
    """Map a registry tool name to a spec ToolKind by token-prefix."""
    tokens = [t for t in "".join(
        c if c.isalnum() else " " for c in name.lower()).split() if t]
    for kind, prefixes in _TOOL_KIND_RULES:
        for token in tokens:
            if token == "tool" or token == "devon":
                continue
            if any(token.startswith(p) for p in prefixes):
                return kind
    return "other"


def _tool_locations(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Absolute-path kwargs → ToolCallLocation list."""
    locations: list[dict[str, Any]] = []
    for key in _PATH_ARG_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and os.path.isabs(value):
            locations.append({"path": value})
    return locations


def _safe_json(value: Any, *, max_string: int = 4000) -> Any:
    """Make ``value`` JSON-safe for rawInput/rawOutput (default=str,
    long strings truncated, total payload capped). Never raises."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - defensive, should not happen
        return str(value)[:max_string]
    if len(text) > 2 * max_string:
        # Too big for a notification frame: keep a readable prefix.
        return text[:2 * max_string] + "…"
    try:
        decoded = json.loads(text)
    except Exception:  # noqa: BLE001 - defensive
        return str(value)[:max_string]
    if isinstance(decoded, str) and len(decoded) > max_string:
        return decoded[:max_string] + "…"
    return decoded


class ACPError(Exception):
    """An error that maps to a JSON-RPC error response."""

    def __init__(self, message: str, code: int = INTERNAL_ERROR) -> None:
        super().__init__(message)
        self.code = code


class _CancelTurn(Exception):
    """Raised at a tool-call boundary when session/cancel arrived."""


# ── sessions ──────────────────────────────────────────────────────────────

@dataclass
class ACPSession:
    """One ACP conversation session."""

    id: str
    cwd: str
    model: str = ""
    mode: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    history: list[tuple[str, str]] = field(default_factory=list)
    memory_snapshot: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event,
                                          repr=False, compare=False)
    #: The session's live MCP connections (an MCPClientManager, built
    #: lazily on the first turn that needs it; closed when the session
    #: is pruned or expires). Not part of to_dict().
    mcp_manager: Any = field(default=None, repr=False, compare=False)

    def touch(self) -> None:
        self.last_active = time.time()
        self.cancel_event.clear()

    def close_mcp(self) -> None:
        """Drop the session's MCP connections. Never raises."""
        manager, self.mcp_manager = self.mcp_manager, None
        if manager is None:
            return
        try:
            manager.close()
        except Exception:  # noqa: BLE001 - close must never raise
            _log.debug("acp: mcp close failed", exc_info=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.id,
            "cwd": self.cwd,
            "model": self.model,
            "mode": self.mode,
            "createdAt": self.created_at,
            "lastActive": self.last_active,
            "historyTurns": len(self.history) // 2,
        }


# ── session MCP wiring ────────────────────────────────────────────────
#
# The session-scoping decision, settled from the code rather than by
# asking anyone: _default_turn already wraps registry.call per turn
# with traced_call under _turn_lock, so the scoping IS a per-turn
# registry view wrapping the shared one. The view exposes the session's
# declared MCP servers' tools as real ToolSpecs — named
# mcp_<server>_<tool> so they can never shadow a native tool — and the
# session principal's capability grant still gates every call
# (NET_OUT). The view never raises: a dead server contributes nothing
# and the shared tools keep working. Its tool results are translated
# exactly like the spine mcp_call tool's, so the loop sees identical
# observations from both paths.

_MCP_NAME_RE = re.compile(r"[^a-z0-9]+")


def _sanitize_mcp_name(raw: str, *, fallback: str) -> str:
    """Lowercase snake_case for a tool name; never empty."""
    text = _MCP_NAME_RE.sub("_", (raw or "").lower()).strip("_")
    return text or fallback


def _mcp_server_config(entry: dict[str, Any], *, index: int) -> Any | None:
    """One ACP-declared mcpServers entry -> MCPServerConfig, or None.

    Liberal about shapes: ACP clients send stdio {command, args, env}
    or http {url, headers} entries, optionally named; Devon-shaped
    entries (transport, timeout_s, env as a list of names) work too.
    A malformed entry is skipped with a log line — it never breaks the
    session, and a session whose declarations are ALL unusable simply
    behaves as if it declared none.
    """
    from ..tools.mcp_client import MCPClientError, MCPServerConfig
    if not isinstance(entry, dict):
        _log.warning("acp: skipping non-object mcpServers entry %r",
                     type(entry).__name__)
        return None
    transport = str(entry.get("transport") or "").strip().lower()
    url = str(entry.get("url") or "")
    if not transport:
        transport = "http" if url else "stdio"
    if transport not in ("stdio", "http"):
        _log.warning("acp: skipping mcpServers entry %r: unknown "
                     "transport %r", entry.get("name") or index, transport)
        return None
    name = str(entry.get("name") or "").strip()
    if not name:
        if transport == "stdio":
            name = os.path.splitext(
                os.path.basename(str(entry.get("command") or "")))[0]
        else:
            try:
                from urllib.parse import urlparse
                name = urlparse(url).hostname or ""
            except Exception:  # noqa: BLE001 - any malformed URL
                name = ""
    name = _sanitize_mcp_name(name, fallback=f"server{index + 1}")
    env_values: dict[str, str] = {}
    env_names: list[str] = []
    env = entry.get("env")
    if isinstance(env, dict):
        env_values = {str(k): str(v) for k, v in env.items()}
    elif isinstance(env, list):
        env_names = [str(e) for e in env]
    headers = entry.get("headers")
    headers = ({str(k): str(v) for k, v in headers.items()}
               if isinstance(headers, dict) else {})
    try:
        timeout_s = float(entry.get("timeout_s") or 0.0)
    except (TypeError, ValueError):
        timeout_s = 0.0
    args = entry.get("args")
    args = [str(a) for a in args] if isinstance(args, (list, tuple)) else []
    cfg = MCPServerConfig(
        name=name,
        command=str(entry.get("command") or ""),
        args=args,
        env=env_names,
        transport=transport,
        url=url,
        enabled=True,
        timeout_s=timeout_s,
        env_values=env_values,
        headers=headers,
    )
    try:
        cfg.validate()
    except MCPClientError as exc:
        _log.warning("acp: skipping mcpServers entry %r: %s", name, exc)
        return None
    return cfg


def _mcp_tool_result(manager: Any, server: str, tool: str,
                     arguments: dict[str, Any]) -> dict[str, Any]:
    """Call one MCP tool; the mcp_call-shaped dict, never a raise."""
    from ..tools.mcp_client import MCPClientError
    try:
        result = manager.call_tool(server, tool, arguments)
    except MCPClientError as exc:
        return {"ok": False, "error": str(exc)}
    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    if result.get("isError"):
        return {"ok": False,
                "error": "\n".join(parts) or "tool reported an error",
                "structured": result.get("structuredContent")}
    return {"ok": True, "text": "\n".join(parts),
            "structured": result.get("structuredContent")}


class _SessionRegistryView:
    """Per-turn registry view: the shared registry + this session's
    MCP tools. Satisfies the ToolAdapter interface (schemas / get /
    call / call_many) so the agentic loop needs no special-casing.

    The MCP tool list is fetched lazily on first use and cached for
    the turn; servers connect on demand through the session's manager
    (connections cache across turns on the session). MCP calls are
    synchronous and serial (call_many runs them one by one); the
    idempotency_key the adapter may pass is accepted and ignored — an
    MCP call executes live each time, so there is nothing to replay.
    """

    def __init__(self, shared: Any, session: ACPSession,
                 manager: Any, grant: Any) -> None:
        self._shared = shared
        self._session = session
        self._manager = manager
        self._grant = grant
        self._index: dict[str, tuple[str, str, dict[str, Any]]] | None = None

    # -- MCP tool index (lazy, per turn) --
    def _mcp_index(self) -> dict[str, tuple[str, str, dict[str, Any]]]:
        if self._index is not None:
            return self._index
        index: dict[str, tuple[str, str, dict[str, Any]]] = {}
        for cfg in self._manager.servers():
            try:
                tools = self._manager.list_tools(cfg.name)
            except Exception as exc:  # noqa: BLE001 - dead server fails open
                _log.warning("acp: mcp server %r tools/list failed: %s",
                             cfg.name, exc)
                continue
            for tool in tools:
                raw = str(tool.get("name") or "")
                if not raw:
                    continue
                reg_name = ("mcp_"
                            + _sanitize_mcp_name(cfg.name, fallback="server")
                            + "_"
                            + _sanitize_mcp_name(raw, fallback="tool"))
                if self._shared is not None \
                        and self._shared.get(reg_name) is not None:
                    _log.warning("acp: mcp tool %r would shadow a native "
                                 "tool; skipping", reg_name)
                    continue
                if reg_name in index:
                    continue
                index[reg_name] = (cfg.name, raw, tool)
        self._index = index
        return index

    def _mcp_spec(self, reg_name: str, server: str,
                  tool: dict[str, Any]) -> Any:
        from ..core.policy import Capability
        from ..tools.registry import ToolSpec
        manager = self._manager
        raw_name = str(tool.get("name") or "")
        description = str(tool.get("description") or "").strip()
        params = tool.get("inputSchema")
        if not isinstance(params, dict):
            params = {}

        def _fn(**kwargs: Any) -> dict[str, Any]:
            return _mcp_tool_result(manager, server, raw_name, kwargs)

        return ToolSpec(
            name=reg_name,
            fn=_fn,
            description=(f"[mcp:{server}] {description}" if description
                         else f"MCP tool {server}/{raw_name}"),
            capability=Capability.NET_OUT,
            parameters=params,
            kind="net",
        )

    def _net_out_allowed(self, capabilities: Any) -> bool:
        from ..core.policy import Capability
        grant = capabilities if capabilities is not None else self._grant
        if grant is None:
            return True
        return grant.grants(Capability.NET_OUT)

    def _audit_mcp(self, name: str, actor: str, kwargs: dict[str, Any],
                   elapsed: float, *, decision: str, ok: bool,
                   error: str = "") -> None:
        from ..core.policy import Capability
        if self._shared is None:
            return
        try:
            self._shared._audit(name, actor, Capability.NET_OUT, decision,
                                kwargs, elapsed, ok=ok, error=error)
        except Exception:  # noqa: BLE001 - auditing never breaks a call
            _log.debug("acp: mcp audit failed", exc_info=True)

    # -- ToolAdapter interface --
    def schemas(self, *, capabilities: Any = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self._shared is not None:
            try:
                out = list(self._shared.schemas(capabilities=capabilities))
            except Exception:  # noqa: BLE001 - shared listing never breaks
                _log.debug("acp: shared schemas failed", exc_info=True)
                out = []
        if not self._net_out_allowed(capabilities):
            return out
        try:
            index = self._mcp_index()
        except Exception:  # noqa: BLE001 - MCP listing never breaks shared
            _log.debug("acp: mcp index failed", exc_info=True)
            return out
        for reg_name, (server, _raw, tool) in index.items():
            try:
                out.append(self._mcp_spec(reg_name, server, tool).schema())
            except Exception:  # noqa: BLE001 - one bad tool never breaks
                _log.debug("acp: mcp spec %r failed", reg_name,
                           exc_info=True)
        return out

    def get(self, name: str) -> Any | None:
        if self._shared is not None:
            spec = self._shared.get(name)
            if spec is not None:
                return spec
        if not self._net_out_allowed(None):
            return None
        try:
            entry = self._mcp_index().get(name)
        except Exception:  # noqa: BLE001 - MCP listing never breaks shared
            _log.debug("acp: mcp index failed", exc_info=True)
            return None
        if entry is None:
            return None
        server, _raw, tool = entry
        return self._mcp_spec(name, server, tool)

    def call(self, name: str, /, *args: Any, actor: str = "system",
             capabilities: Any = None, confirmation: str | None = None,
             idempotency_key: str | None = None,
             **kwargs: Any) -> Any:
        from ..core.errors import CapabilityDenied, ToolError
        from ..core.policy import Capability
        from ..core.result import Err, Ok
        if args:
            return Err(ToolError(
                f"call() takes 1 positional argument but {1 + len(args)} "
                "were given"))
        entry = None
        try:
            entry = self._mcp_index().get(name)
        except Exception:  # noqa: BLE001 - MCP listing never breaks shared
            _log.debug("acp: mcp index failed", exc_info=True)
            entry = None
        if entry is None:
            if self._shared is None:
                return Err(ToolError(f"unknown tool {name!r}"))
            return self._shared.call(
                name, actor=actor, capabilities=capabilities,
                confirmation=confirmation,
                idempotency_key=idempotency_key, **kwargs)
        server, raw_tool, _tool = entry
        if not self._net_out_allowed(capabilities):
            self._audit_mcp(name, actor, kwargs, 0.0, decision="deny",
                            ok=False,
                            error=f"actor {actor!r} lacks 'net.out'")
            return Err(CapabilityDenied(
                f"actor {actor!r} lacks 'net.out'",
                capability=Capability.NET_OUT, actor=actor))
        started = time.perf_counter()
        result = _mcp_tool_result(self._manager, server, raw_tool, kwargs)
        elapsed = time.perf_counter() - started
        if result.get("ok"):
            self._audit_mcp(name, actor, kwargs, elapsed, decision="allow",
                            ok=True)
            return Ok(result)
        err = str(result.get("error") or "tool reported an error")
        self._audit_mcp(name, actor, kwargs, elapsed, decision="allow",
                        ok=False, error=err)
        return Err(ToolError(err))

    def call_many(self, calls: list[tuple[str, dict[str, Any]]],
                  *, max_workers: int = 1, **common: Any) -> list[Any]:
        out: list[Any] = []
        for name, args in calls:
            args = dict(args) if isinstance(args, dict) else {}
            key = args.pop("idempotency_key", None)
            out.append(self.call(
                str(name),
                actor=common.get("actor", "system"),
                capabilities=common.get("capabilities"),
                idempotency_key=key, **args))
        return out



# ── server ────────────────────────────────────────────────────────────────

EmitFn = Callable[[dict[str, Any]], None]
TurnFn = Callable[[ACPSession, str, EmitFn], str]


class ACPServer:
    """ACP agent endpoint. Transport-agnostic core; see module docstring."""

    def __init__(
        self,
        context: Any = None,
        registry: Any = None,
        *,
        turn_fn: TurnFn | None = None,
        principal: Any = None,
        principal_resolver: Callable[[], Any] | None = None,
        session_ttl_s: float = 3600.0,
    ) -> None:
        self.context = context
        self.registry = registry if registry is not None else \
            getattr(context, "tools", None)
        self._turn_fn = turn_fn or self._default_turn
        self._principal = principal
        self._principal_resolver = principal_resolver
        # Explicit per-call principal (dispatch's ``principal=``) for
        # transports whose worker thread cannot see the caller's own
        # thread-local — the SSE path runs the batch on a worker.
        self._principal_tls = threading.local()
        self.session_ttl_s = session_ttl_s
        self._sessions: dict[str, ACPSession] = {}
        self._lock = threading.RLock()
        # Serializes the default turn's registry.call tracing: two
        # concurrent turns must not interleave their monkeypatches.
        self._turn_lock = threading.RLock()
        # Client capabilities from initialize (fs.readTextFile/writeTextFile,
        # terminal, auth, elicitation) — recorded so future agent→client
        # calls (fs/*, terminal/*) know what the client supports.
        self.client_capabilities: dict[str, Any] = {}
        if self._principal is None and self._principal_resolver is None:
            from .server import DEFAULT_PRINCIPAL
            self._principal = DEFAULT_PRINCIPAL

    # ── principal ─────────────────────────────────────────────────────
    @property
    def principal(self) -> Any:
        override = getattr(self._principal_tls, "principal", None)
        if override is not None:
            return override
        if self._principal_resolver is not None:
            try:
                resolved = self._principal_resolver()
            except Exception as exc:  # noqa: BLE001 - resolver must never break us
                _warn_resolver_failure(self._principal_resolver, exc)
                resolved = None
            if resolved is not None:
                return resolved
        return self._principal

    # ── JSON-RPC plumbing ─────────────────────────────────────────────
    @staticmethod
    def _ok(msg_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
        }
        if msg_id is not None:
            out["id"] = msg_id
        return out

    @staticmethod
    def _notification(method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "method": method, "params": params}

    def _session_update(self, session_id: str,
                        update: dict[str, Any]) -> dict[str, Any]:
        return self._notification("session/update", {
            "sessionId": session_id,
            "update": update,
        })

    # ── sessions ──────────────────────────────────────────────────────
    def _get_session(self, session_id: str) -> ACPSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ACPError(f"unknown session {session_id!r} "
                           f"(expired or never created)", INVALID_PARAMS)
        if (time.time() - session.last_active) > self.session_ttl_s:
            with self._lock:
                self._sessions.pop(session_id, None)
            session.close_mcp()
            raise ACPError(f"session {session_id!r} expired", INVALID_PARAMS)
        return session

    def prune(self) -> int:
        """Drop sessions idle longer than the TTL. Returns the count."""
        now = time.time()
        with self._lock:
            dead = [sid for sid, s in self._sessions.items()
                    if now - s.last_active > self.session_ttl_s]
            sessions = [self._sessions.pop(sid) for sid in dead]
        for session in sessions:
            session.close_mcp()
        return len(dead)

    def _session_mcp_manager(self, session: ACPSession) -> Any | None:
        """The session's MCP manager, built lazily from its declared
        servers. None when the session declared no usable servers.
        Built once per session (connections cache across turns) under
        the server lock; closed when the session is pruned or expires."""
        from ..tools.mcp_client import MCPClientManager
        with self._lock:
            manager = session.mcp_manager
            if manager is not None:
                return manager
            configs: dict[str, Any] = {}
            for index, entry in enumerate(session.mcp_servers or []):
                cfg = _mcp_server_config(entry, index=index)
                if cfg is None:
                    continue
                if cfg.name in configs:
                    _log.warning("acp: duplicate mcpServers name %r; "
                                 "keeping the first", cfg.name)
                    continue
                configs[cfg.name] = cfg
            if not configs:
                return None
            manager = MCPClientManager(servers=configs)
            session.mcp_manager = manager
            return manager

    def _session_view(self, session: ACPSession, grant: Any) -> Any:
        """Per-turn registry view for a session: the shared registry
        plus the session's declared MCP servers' tools. A session with
        no usable servers gets the shared registry itself (byte-
        identical behavior to before this lane)."""
        if self.registry is None or not session.mcp_servers:
            return self.registry
        manager = self._session_mcp_manager(session)
        if manager is None:
            return self.registry
        return _SessionRegistryView(self.registry, session, manager, grant)

    def _new_session(self, params: dict[str, Any]) -> ACPSession:
        cwd = params.get("cwd") or ""
        if not cwd:
            raise ACPError("session/new requires an absolute 'cwd'",
                           INVALID_PARAMS)
        if not os.path.isabs(cwd):
            raise ACPError(f"cwd must be absolute, got {cwd!r}", INVALID_PARAMS)
        session = ACPSession(
            id=f"sess_{new_id()[:12]}",
            cwd=cwd,
            model=str(params.get("model") or ""),
            mode=str(params.get("mode") or ""),
            mcp_servers=list(params.get("mcpServers") or []),
        )
        if session.mcp_servers:
            _log.info("acp: session %s declared %d mcpServers (wired "
                      "into the per-turn registry view)", session.id,
                      len(session.mcp_servers))
        with self._lock:
            self.prune()
            self._sessions[session.id] = session
        return session

    # ── prompt text extraction ────────────────────────────────────────
    @staticmethod
    def _prompt_text(prompt: Any) -> str:
        """Content blocks → plain text. Only text blocks are supported
        (promptCapabilities advertises text only); anything else is
        skipped with a log line, never silently misread."""
        if isinstance(prompt, str):
            return prompt
        parts: list[str] = []
        for block in prompt or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text") or ""))
            else:
                _log.info("acp: dropping unsupported prompt block type %r",
                          btype)
        return "\n".join(p for p in parts if p).strip()

    # ── the default turn: real orchestration loop ─────────────────────
    def _default_turn(self, session: ACPSession, prompt_text: str,
                      emit: EmitFn) -> str:
        """Drive one turn through the agentic loop, emitting live
        tool_call / tool_call_update notifications. Cooperative cancel
        is honored at tool-call boundaries."""
        from ..agents.orchestration import run_agentic
        llm = getattr(self.context, "llm", None) or \
            getattr(self.context, "router", None)
        if llm is None or self.registry is None:
            raise ACPError(
                "ACP turns need an LLM and a tool registry on the context; "
                "construct ACPServer with turn_fn= to drive turns yourself",
                INTERNAL_ERROR,
            )
        registry = self.registry
        grant = self.principal.grant
        actor = f"acp:{session.id}"
        if session.mcp_servers:
            # Per-turn scope: the session's declared MCP servers are
            # visible through this view only; the shared registry is
            # untouched and a session that declared nothing gets it
            # directly (behavior identical to before this lane).
            registry = self._session_view(session, grant)
        orig_call = registry.call

        def traced_call(name: str, /, *args: Any, **kwargs: Any) -> Any:
            if session.cancel_event.is_set():
                raise _CancelTurn()
            tool_call_id = f"tc_{new_id()[:8]}"
            kind = _tool_kind(name)
            raw_input = _safe_json(kwargs if kwargs else list(args))
            locations = _tool_locations(
                kwargs if isinstance(kwargs, dict) else {})
            tool_call: dict[str, Any] = {
                "sessionUpdate": "tool_call",
                "toolCallId": tool_call_id,
                "title": name,
                "name": name,
                "kind": kind,
                "status": "in_progress",
                "rawInput": raw_input,
            }
            if locations:
                tool_call["locations"] = locations
            emit(self._session_update(session.id, tool_call))
            outcome = orig_call(name, *args, **kwargs)
            status = "completed"
            content: list[dict[str, Any]] = []
            raw_output: Any = None
            try:
                ok = bool(getattr(outcome, "ok", False)) if hasattr(
                    outcome, "ok") else True
                if not ok:
                    status = "failed"
                val = getattr(outcome, "value", outcome)
                if hasattr(outcome, "error") and not ok:
                    val = getattr(outcome, "error")
                raw_output = _safe_json(val)
                content = [{"type": "content",
                            "content": {"type": "text",
                                        "text": str(val)[:4000]}}]
            except Exception:  # noqa: BLE001 - tracing must never break a turn
                pass
            update: dict[str, Any] = {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": status,
                "content": content,
            }
            if raw_output is not None:
                update["rawOutput"] = raw_output
            emit(self._session_update(session.id, update))
            if session.cancel_event.is_set():
                raise _CancelTurn()
            return outcome

        # The patch, the run, and the restore all happen under the turn
        # lock: concurrent turns must never interleave monkeypatches.
        with self._turn_lock:
            registry.call = traced_call  # type: ignore[method-assign]
            try:
                result = run_agentic(
                    prompt_text,
                    llm=llm,
                    registry=registry,
                    history=list(session.history),
                    actor=actor,
                    capabilities=grant,
                    resume_from=session.memory_snapshot,
                )
            except _CancelTurn:
                return "__CANCELLED__"
            finally:
                registry.call = orig_call  # type: ignore[method-assign]
        # Persist resumable state for session/load.
        session.history.append(("user", prompt_text))
        session.history.append(("assistant", result.response))
        session.memory_snapshot = result.memory_snapshot
        if result.asked_user:
            return "__REFUSAL__" + result.question
        if not result.success:
            return "__REFUSAL__" + (result.response or
                                    "the turn failed closed")
        return result.response

    def _emit_message_chunks(self, session_id: str, text: str,
                             emit: EmitFn, width: int = 800) -> None:
        for i in range(0, max(len(text), 1), width):
            emit(self._session_update(session_id, {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text",
                            "text": text[i:i + width] or ""},
            }))

    def _emit_plan(self, session: ACPSession, goal: str, emit: EmitFn,
                   *, status: str) -> None:
        """Report the turn's execution plan (spec: the client replaces its
        whole plan with each update). One entry — the real goal of this
        turn — in_progress at turn start, completed when the turn ends."""
        try:
            emit(self._session_update(session.id, {
                "sessionUpdate": "plan",
                "entries": [{
                    "content": (goal[:200] or "Handle the prompt"),
                    "priority": "medium",
                    "status": status,
                }],
            }))
        except Exception:  # noqa: BLE001 - plan is visibility, not control
            _log.debug("acp: plan emit failed", exc_info=True)

    def _emit_available_commands(self, session: ACPSession,
                                 emit: EmitFn) -> None:
        """Advertise slash commands from the capability-filtered registry
        (spec: available_commands_update — "commands are ready or have
        changed")."""
        if self.registry is None:
            return
        try:
            schemas = self.registry.schemas(
                capabilities=self.principal.grant)
            commands = [
                {"name": s.get("name"),
                 "description": str(s.get("description") or "")}
                for s in schemas
                if isinstance(s, dict) and s.get("name")
            ]
        except Exception:  # noqa: BLE001 - commands are additive
            _log.debug("acp: available-commands emit failed", exc_info=True)
            return
        try:
            emit(self._session_update(session.id, {
                "sessionUpdate": "available_commands_update",
                "availableCommands": commands,
            }))
        except Exception:  # noqa: BLE001 - commands are additive
            _log.debug("acp: available-commands emit failed", exc_info=True)

    # ── method handlers ───────────────────────────────────────────────
    def _handle(self, method: str, params: dict[str, Any],
                emit: EmitFn) -> Any:
        if method == "initialize":
            requested = params.get("protocolVersion", PROTOCOL_VERSION)
            try:
                requested_version = (
                    PROTOCOL_VERSION if requested is None
                    else int(requested))
            except (TypeError, ValueError):
                raise ACPError(
                    f"protocolVersion must be an integer, got "
                    f"{requested!r}", INVALID_PARAMS) from None
            if requested_version < 1:
                requested_version = PROTOCOL_VERSION
            version = min(requested_version, PROTOCOL_VERSION)
            # Record what the client can do (fs.readTextFile/writeTextFile,
            # terminal, auth, elicitation) for future agent→client calls.
            client_caps = params.get("clientCapabilities")
            if isinstance(client_caps, dict):
                self.client_capabilities = client_caps
            return {
                "protocolVersion": version,
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {
                        "text": True,
                        "image": False,
                        "audio": False,
                        "embeddedContext": False,
                    },
                    # Declared session servers are wired into the
                    # turn's registry view. Streamable HTTP is spoken;
                    # the legacy SSE transport is not implemented.
                    "mcpCapabilities": {"http": True, "sse": False},
                },
                "authMethods": [],
                "agentInfo": {"name": "devon", "version": "2.0"},
            }
        if method == "authenticate":
            # We declare no authMethods, so clients should not call this;
            # accept it anyway rather than failing a lenient client.
            return {}
        if method == "session/new":
            session = self._new_session(params)
            _log.info("acp: new session %s cwd=%s", session.id, session.cwd)
            self._emit_available_commands(session, emit)
            return {"sessionId": session.id}
        if method == "session/load":
            session = self._get_session(str(params.get("sessionId") or ""))
            if params.get("cwd"):
                session.cwd = str(params["cwd"])
            session.touch()
            return {"sessionId": session.id}
        if method == "session/set_mode":
            session = self._get_session(str(params.get("sessionId") or ""))
            session.mode = str(params.get("mode") or "")
            session.touch()
            return {}
        if method == "session/set_model":
            session = self._get_session(str(params.get("sessionId") or ""))
            session.model = str(params.get("model") or "")
            session.touch()
            # Recorded on the session and reported back; the model/provider
            # actually used for a turn remains server-configured.
            return {}
        if method == "session/cancel":
            session = self._get_session(str(params.get("sessionId") or ""))
            session.cancel_event.set()
            return {}
        if method == "session/prompt":
            session = self._get_session(str(params.get("sessionId") or ""))
            text = self._prompt_text(params.get("prompt"))
            if not text:
                raise ACPError("session/prompt needs at least one text "
                               "content block", INVALID_PARAMS)
            session.touch()
            self._emit_plan(session, text, emit, status="in_progress")
            try:
                final = self._turn_fn(session, text, emit)
            except _CancelTurn:
                final = "__CANCELLED__"
            if final == "__CANCELLED__":
                stop = "cancelled"
                self._emit_message_chunks(session.id,
                                          "Turn cancelled.", emit)
            elif final.startswith("__REFUSAL__"):
                stop = "refusal"
                self._emit_message_chunks(session.id,
                                          final[len("__REFUSAL__"):], emit)
            else:
                stop = "end_turn"
                self._emit_message_chunks(session.id, final, emit)
            # PlanEntryStatus has no "failed" — a stopped turn still
            # completes its (single) plan entry.
            self._emit_plan(session, text, emit, status="completed")
            session.touch()
            return {"stopReason": stop}
        if method == "devon/tools":
            # Devon extension (not ACP spec): capability-filtered registry.
            if self.registry is None:
                return {"tools": []}
            schemas = self.registry.schemas(
                capabilities=self.principal.grant)
            return {"tools": [
                {
                    "name": s.get("name"),
                    "description": s.get("description"),
                    "inputSchema": {
                        "type": "object",
                        "properties": s.get("parameters") or {},
                    },
                }
                for s in schemas
            ]}
        raise ACPError(f"method not found: {method}", METHOD_NOT_FOUND)

    # ── transport-agnostic dispatch ───────────────────────────────────
    def dispatch(self, msg: Any, emit: EmitFn, *,
                 principal: Any = None) -> dict[str, Any] | None:
        """Handle one JSON-RPC message. Returns the response dict, or
        ``None`` for notifications (never respond to those). Never raises.

        ``principal`` pins this call to an explicit principal (the SSE
        transport's worker thread cannot see the HTTP request's own
        thread-local); without it the principal resolves per the server
        configuration.
        """
        previous_principal = getattr(self._principal_tls, "principal", None)
        if principal is not None:
            self._principal_tls.principal = principal
        try:
            return self._dispatch_inner(msg, emit)
        finally:
            if principal is not None:
                if previous_principal is None:
                    try:
                        del self._principal_tls.principal
                    except AttributeError as exc:
                        _log.debug("acp.dispatch failed", exc_info=True)
                else:
                    self._principal_tls.principal = previous_principal

    def _dispatch_inner(self, msg: Any, emit: EmitFn) -> dict[str, Any] | None:
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
            except ACPError as exc:
                if msg_id is None:
                    _log.debug("acp: notification %s failed: %s", method, exc)
                    return None
                return self._err(msg_id, exc.code, str(exc))
            except Exception as exc:  # noqa: BLE001 - never leak a traceback
                _log.exception("acp: error in %s", method)
                if msg_id is None:
                    return None
                return self._err(msg_id, INTERNAL_ERROR,
                                 f"{type(exc).__name__}: {exc}")
            if msg_id is None:
                return None
            return self._ok(msg_id, result)
        except Exception as exc:  # noqa: BLE001 - dispatch never raises
            _log.exception("acp: dispatch crashed")
            return self._err(None, INTERNAL_ERROR, f"{type(exc).__name__}")

    def handle_batch(self, msgs: Any, emit: EmitFn, *,
                     principal: Any = None) -> Any:
        """A single message or a batch list → response or list of them."""
        if isinstance(msgs, list):
            if not msgs:
                return self._err(None, INVALID_REQUEST, "empty batch")
            out = [self.dispatch(m, emit, principal=principal) for m in msgs]
            return [r for r in out if r is not None]
        return self.dispatch(msgs, emit, principal=principal)

    # ── stdio transport ───────────────────────────────────────────────
    @staticmethod
    def _read_stdio_message(stream: Any) -> dict[str, Any] | None:
        """Read one message: newline-delimited JSON, or Content-Length
        framed (LSP-style). Returns None on clean EOF."""
        first = stream.readline()
        if not first:
            return None
        if first.strip().lower().startswith(b"content-length:"):
            length = int(first.split(b":", 1)[1].strip())
            # consume remaining header lines
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
        """Serve ACP over stdio. Logs go to stderr; stdout is pure
        JSON-RPC. Returns 0 on clean EOF."""
        stdin = stdin if stdin is not None else sys.stdin.buffer
        stdout = stdout if stdout is not None else sys.stdout.buffer

        def emit(notification: dict[str, Any]) -> None:
            raw = (json.dumps(notification, ensure_ascii=False) + "\n"
                   ).encode("utf-8")
            stdout.write(raw)
            stdout.flush()

        _log.info("acp: serving over stdio")
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
                _log.info("acp: stdio read failed: %s", exc)
                break
            if msg is None:  # clean EOF
                break
            response = self.handle_batch(msg, emit)
            if response is None:
                continue
            if isinstance(response, list) and not response:
                continue
            stdout.write((json.dumps(response, ensure_ascii=False) + "\n"
                          ).encode("utf-8"))
            stdout.flush()
        _log.info("acp: stdio EOF, shutting down")
        return 0

    # ── HTTP transport ────────────────────────────────────────────────
    def handle_http(self, body: Any, *, principal: Any = None) -> Any:
        """Buffered HTTP: one JSON-RPC message/batch → response payload.
        session/update notifications are only delivered over stdio or SSE;
        buffered HTTP returns just the final response (documented)."""
        dropped: list[dict[str, Any]] = []

        def emit(notification: dict[str, Any]) -> None:
            dropped.append(notification)

        return self.handle_batch(body, emit, principal=principal)

    def iter_sse(self, body: Any, *, principal: Any = None) -> Any:
        """SSE streaming for POST /acp with ``Accept: text/event-stream``.

        Yields ``data: <json>`` frames: first every session/update
        notification as it happens, then the final JSON-RPC response.
        """
        q: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        result_box: dict[str, Any] = {}

        def emit(notification: dict[str, Any]) -> None:
            q.put(notification)

        def worker() -> None:
            try:
                result_box["response"] = self.handle_batch(
                    body, emit, principal=principal)
            except Exception as exc:  # noqa: BLE001
                result_box["response"] = self._err(None, INTERNAL_ERROR,
                                                   f"{type(exc).__name__}")
            finally:
                q.put(None)

        thread = threading.Thread(target=worker, name="acp-sse",
                                  daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        response = result_box.get("response")
        if response is not None and not (isinstance(response, list)
                                         and not response):
            yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n"


# ── mounting on the existing API server ───────────────────────────────

def register_acp(api_server: Any, acp: ACPServer) -> None:
    """Add ``POST /acp`` (buffered JSON-RPC) to an APIServer, following the
    route pattern. SSE streaming is wired by a small special-case in the
    HTTP handler (see ``serve_acp_sse``), mirroring the /stream precedent."""
    api_server._acp_server = acp  # noqa: SLF001 - read by the SSE special-case
    if acp._principal_resolver is None:  # noqa: SLF001 - mount wiring
        # The mounted server must act as the principal each HTTP request
        # resolved to — never as one fixed server-wide principal (that
        # both escalated narrow named principals to the bounded local
        # grant and downgraded the owner to it). Outside an in-flight
        # request (direct handler calls, tests) the server's own
        # configured principal applies.
        own = acp._principal  # noqa: SLF001

        def _resolve_inflight() -> Any:
            inflight = getattr(api_server, "_inflight_principal", None)
            resolved = inflight() if inflight is not None else None
            return resolved if resolved is not None else own

        acp._principal_resolver = _resolve_inflight  # noqa: SLF001
    # JSON-RPC batches are valid bodies for this route.
    batch_routes = getattr(api_server, "_batch_routes", None)
    if batch_routes is not None:
        batch_routes.add(("POST", "/acp"))

    @api_server.route("POST", "/acp",
                      description="Agent Client Protocol: JSON-RPC 2.0 "
                                  "endpoint (initialize, session/new, "
                                  "session/prompt, ...). Buffered; send "
                                  "Accept: text/event-stream for SSE.")
    def acp_endpoint(body: dict[str, Any],
                     query: dict[str, str]) -> dict[str, Any]:
        _ = query
        return acp.handle_http(body)


def serve_acp_sse(handler: Any, api_server: Any, acp: ACPServer,
                  raw_body: bytes, *, principal: Any = None) -> None:
    """Serve one SSE stream for POST /acp. Called from the HTTP handler's
    special-case (mirrors the /stream SSE precedent in server.py)."""
    try:
        body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        handler.send_response(400)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(b'{"error": "body is not valid JSON"}')
        return
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Referrer-Policy", "no-referrer")
    handler.send_header("X-Accel-Buffering", "no")
    request_id = getattr(handler, "_request_id", "")
    if request_id:
        handler.send_header("X-Request-ID", request_id)
    handler.end_headers()
    try:
        for frame in acp.iter_sse(body, principal=principal):
            handler.wfile.write(frame.encode("utf-8"))
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError) as exc:
        _log.debug("acp.serve_acp_sse failed", exc_info=True)


def _wants_sse(handler: Any) -> bool:
    accept = handler.headers.get("Accept", "")
    return "text/event-stream" in accept.lower()


# ── stdio entry point ─────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Devon ACP agent (Agent Client Protocol) over stdio")
    parser.add_argument("--stdio", action="store_true",
                        help="serve JSON-RPC 2.0 over stdin/stdout")
    parser.add_argument("--session-ttl", type=float, default=3600.0,
                        help="idle seconds before a session expires")
    args = parser.parse_args(argv)
    if not args.stdio:
        parser.error("--stdio is required (editors spawn this over stdio)")
    # Lazy context: stdio mode must not require a full app boot.
    acp = ACPServer(session_ttl_s=args.session_ttl)
    return acp.run_stdio()


if __name__ == "__main__":
    sys.exit(main())
