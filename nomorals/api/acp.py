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
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.ids import new_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

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

    def touch(self) -> None:
        self.last_active = time.time()
        self.cancel_event.clear()

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
        self.session_ttl_s = session_ttl_s
        self._sessions: dict[str, ACPSession] = {}
        self._lock = threading.RLock()
        # Serializes the default turn's registry.call tracing: two
        # concurrent turns must not interleave their monkeypatches.
        self._turn_lock = threading.RLock()
        if self._principal is None and self._principal_resolver is None:
            from .server import DEFAULT_PRINCIPAL
            self._principal = DEFAULT_PRINCIPAL

    # ── principal ─────────────────────────────────────────────────────
    @property
    def principal(self) -> Any:
        if self._principal_resolver is not None:
            try:
                return self._principal_resolver()
            except Exception:  # noqa: BLE001 - resolver must never break us
                pass
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
            raise ACPError(f"session {session_id!r} expired", INVALID_PARAMS)
        return session

    def prune(self) -> int:
        """Drop sessions idle longer than the TTL. Returns the count."""
        now = time.time()
        with self._lock:
            dead = [sid for sid, s in self._sessions.items()
                    if now - s.last_active > self.session_ttl_s]
            for sid in dead:
                del self._sessions[sid]
        return len(dead)

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
            _log.info("acp: session %s declared %d mcpServers (accepted, "
                      "not wired to MCP)", session.id, len(session.mcp_servers))
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
        orig_call = registry.call

        def traced_call(name: str, /, *args: Any, **kwargs: Any) -> Any:
            if session.cancel_event.is_set():
                raise _CancelTurn()
            tool_call_id = f"tc_{new_id()[:8]}"
            emit(self._session_update(session.id, {
                "sessionUpdate": "tool_call",
                "toolCallId": tool_call_id,
                "title": name,
                "kind": "other",
                "status": "in_progress",
            }))
            outcome = orig_call(name, *args, **kwargs)
            status = "completed"
            content: list[dict[str, Any]] = []
            try:
                ok = bool(getattr(outcome, "ok", False)) if hasattr(
                    outcome, "ok") else True
                if not ok:
                    status = "failed"
                val = getattr(outcome, "value", outcome)
                if hasattr(outcome, "error") and not ok:
                    val = getattr(outcome, "error")
                content = [{"type": "content",
                            "content": {"type": "text",
                                        "text": str(val)[:4000]}}]
            except Exception:  # noqa: BLE001 - tracing must never break a turn
                pass
            emit(self._session_update(session.id, {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": status,
                "content": content,
            }))
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

    # ── method handlers ───────────────────────────────────────────────
    def _handle(self, method: str, params: dict[str, Any],
                emit: EmitFn) -> Any:
        if method == "initialize":
            requested = params.get("protocolVersion", PROTOCOL_VERSION)
            version = min(int(requested or PROTOCOL_VERSION), PROTOCOL_VERSION)
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
    def dispatch(self, msg: Any, emit: EmitFn) -> dict[str, Any] | None:
        """Handle one JSON-RPC message. Returns the response dict, or
        ``None`` for notifications (never respond to those). Never raises."""
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

    def handle_batch(self, msgs: Any, emit: EmitFn) -> Any:
        """A single message or a batch list → response or list of them."""
        if isinstance(msgs, list):
            if not msgs:
                return self._err(None, INVALID_REQUEST, "empty batch")
            out = [self.dispatch(m, emit) for m in msgs]
            return [r for r in out if r is not None]
        return self.dispatch(msgs, emit)

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
    def handle_http(self, body: Any) -> Any:
        """Buffered HTTP: one JSON-RPC message/batch → response payload.
        session/update notifications are only delivered over stdio or SSE;
        buffered HTTP returns just the final response (documented)."""
        dropped: list[dict[str, Any]] = []

        def emit(notification: dict[str, Any]) -> None:
            dropped.append(notification)

        return self.handle_batch(body, emit)

    def iter_sse(self, body: Any) -> Any:
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
                result_box["response"] = self.handle_batch(body, emit)
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
                  raw_body: bytes) -> None:
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
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()
    try:
        for frame in acp.iter_sse(body):
            handler.wfile.write(frame.encode("utf-8"))
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass


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
