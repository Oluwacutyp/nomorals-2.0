"""ACP (Agent Client Protocol) agent-side tests — all offline."""
import io
import json
import time

import pytest

from nomorals.api.acp import (
    ACPServer,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    PROTOCOL_VERSION,
)
from nomorals.api.server import DEFAULT_PRINCIPAL
from nomorals.tools.registry import ToolRegistry


def _rpc(mid, method, params=None):
    msg = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _server(**kw):
    emitted = []
    kw.setdefault("turn_fn", lambda session, text, emit: "turn-result")
    server = ACPServer(**kw)
    return server, emitted


def _new_session(server, mid=1, cwd="/tmp"):
    resp = server.dispatch(
        _rpc(mid, "session/new", {"cwd": cwd}), lambda n: None)
    assert "result" in resp, resp
    return resp["result"]["sessionId"]


# ── initialize ──────────────────────────────────────────────────────────

def test_initialize_handshake():
    server, _ = _server()
    resp = server.dispatch(
        _rpc(0, "initialize",
             {"protocolVersion": 1,
              "clientCapabilities": {},
              "clientInfo": {"name": "test", "version": "0"}}),
        lambda n: None)
    result = resp["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["agentCapabilities"]["loadSession"] is True
    assert result["agentCapabilities"]["promptCapabilities"]["text"] is True
    assert result["authMethods"] == []


def test_initialize_negotiates_down():
    server, _ = _server()
    resp = server.dispatch(_rpc(0, "initialize", {"protocolVersion": 99}),
                           lambda n: None)
    assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_authenticate_noop():
    server, _ = _server()
    resp = server.dispatch(_rpc(0, "authenticate", {}), lambda n: None)
    assert resp["result"] == {}


# ── sessions ────────────────────────────────────────────────────────────

def test_session_new_returns_id():
    server, _ = _server()
    sid = _new_session(server)
    assert sid.startswith("sess_")


def test_session_new_requires_absolute_cwd():
    server, _ = _server()
    resp = server.dispatch(_rpc(1, "session/new", {"cwd": "relative/path"}),
                           lambda n: None)
    assert resp["error"]["code"] == INVALID_PARAMS


def test_session_new_requires_cwd():
    server, _ = _server()
    resp = server.dispatch(_rpc(1, "session/new", {}), lambda n: None)
    assert resp["error"]["code"] == INVALID_PARAMS


def test_session_load_roundtrip():
    server, _ = _server()
    sid = _new_session(server)
    resp = server.dispatch(_rpc(2, "session/load", {"sessionId": sid}),
                           lambda n: None)
    assert resp["result"]["sessionId"] == sid


def test_session_load_unknown():
    server, _ = _server()
    resp = server.dispatch(_rpc(2, "session/load", {"sessionId": "sess_nope"}),
                           lambda n: None)
    assert resp["error"]["code"] == INVALID_PARAMS


def test_session_expiry():
    server, _ = _server(session_ttl_s=0.01)
    sid = _new_session(server)
    time.sleep(0.02)
    resp = server.dispatch(_rpc(2, "session/load", {"sessionId": sid}),
                           lambda n: None)
    assert resp["error"]["code"] == INVALID_PARAMS


def test_set_mode_and_model():
    server, _ = _server()
    sid = _new_session(server)
    server.dispatch(_rpc(2, "session/set_mode",
                         {"sessionId": sid, "mode": "ask"}), lambda n: None)
    server.dispatch(_rpc(3, "session/set_model",
                         {"sessionId": sid, "model": "test-model"}),
                    lambda n: None)
    assert server._sessions[sid].mode == "ask"
    assert server._sessions[sid].model == "test-model"


# ── prompt turn ─────────────────────────────────────────────────────────

def test_prompt_emits_updates_and_stop_reason():
    emitted = []

    def turn_fn(session, text, emit):
        assert text == "hello agent"
        emit({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": session.id,
                         "update": {"sessionUpdate": "tool_call",
                                    "toolCallId": "tc_1",
                                    "title": "read",
                                    "status": "in_progress"}}})
        return "all done"

    server = ACPServer(turn_fn=turn_fn)
    sid = _new_session(server)
    resp = server.dispatch(
        _rpc(5, "session/prompt",
             {"sessionId": sid,
              "prompt": [{"type": "text", "text": "hello agent"}]}),
        emitted.append)
    assert resp["result"] == {"stopReason": "end_turn"}
    kinds = [n["params"]["update"]["sessionUpdate"] for n in emitted]
    assert "tool_call" in kinds
    assert "agent_message_chunk" in kinds
    chunk_text = "".join(
        n["params"]["update"]["content"]["text"] for n in emitted
        if n["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
    assert chunk_text == "all done"


def test_prompt_rejects_empty():
    server, _ = _server()
    sid = _new_session(server)
    resp = server.dispatch(
        _rpc(5, "session/prompt",
             {"sessionId": sid, "prompt": [{"type": "text", "text": "  "}]}),
        lambda n: None)
    assert resp["error"]["code"] == INVALID_PARAMS


def test_prompt_unknown_session():
    server, _ = _server()
    resp = server.dispatch(
        _rpc(5, "session/prompt",
             {"sessionId": "sess_nope",
              "prompt": [{"type": "text", "text": "hi"}]}),
        lambda n: None)
    assert resp["error"]["code"] == INVALID_PARAMS


def test_cancel_notification_sets_event_no_response():
    server, _ = _server()
    sid = _new_session(server)
    session = server._sessions[sid]
    assert not session.cancel_event.is_set()
    # notification: no id → no response
    resp = server.dispatch(
        {"jsonrpc": "2.0", "method": "session/cancel",
         "params": {"sessionId": sid}},
        lambda n: None)
    assert resp is None
    assert session.cancel_event.is_set()


def test_cancelled_turn_stop_reason():
    def turn_fn(session, text, emit):
        return "__CANCELLED__"

    server = ACPServer(turn_fn=turn_fn)
    sid = _new_session(server)
    resp = server.dispatch(
        _rpc(5, "session/prompt",
             {"sessionId": sid, "prompt": [{"type": "text", "text": "x"}]}),
        lambda n: None)
    assert resp["result"] == {"stopReason": "cancelled"}


# ── errors ──────────────────────────────────────────────────────────────

def test_unknown_method():
    server, _ = _server()
    resp = server.dispatch(_rpc(9, "frobnicate", {}), lambda n: None)
    assert resp["error"]["code"] == METHOD_NOT_FOUND
    assert "frobnicate" in resp["error"]["message"]


def test_not_implemented_spec_methods_are_32601():
    # session/request_permission is agent→client; fs/* are client-side.
    # None of them are served here — honest method_not_found, not fakes.
    server, _ = _server()
    for method in ("session/request_permission", "fs/read_text_file",
                   "terminal/create"):
        resp = server.dispatch(_rpc(9, method, {}), lambda n: None)
        assert resp["error"]["code"] == METHOD_NOT_FOUND, method


def test_malformed_message():
    server, _ = _server()
    resp = server.dispatch({"nope": True}, lambda n: None)
    assert "error" in resp


def test_batch():
    server, _ = _server()
    out = server.handle_batch([
        _rpc(1, "initialize", {"protocolVersion": 1}),
        _rpc(2, "session/new", {"cwd": "/tmp"}),
        {"jsonrpc": "2.0", "method": "session/cancel",
         "params": {"sessionId": "sess_x"}},  # notification → dropped
    ], lambda n: None)
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["id"] == 1 and out[1]["id"] == 2


# ── devon/tools: capability-filtered ────────────────────────────────────

def test_devon_tools_matches_registry_filtered_by_grant():
    registry = ToolRegistry()

    @registry.register("safe_read", description="reads things",
                       capability="")
    def _safe():
        return "ok"

    @registry.register("privileged_write", description="writes things",
                       capability="__never_granted__")
    def _priv():
        return "ok"

    server = ACPServer(registry=registry, principal=DEFAULT_PRINCIPAL)
    resp = server.dispatch(_rpc(1, "devon/tools", {}), lambda n: None)
    names = [t["name"] for t in resp["result"]["tools"]]
    assert "safe_read" in names
    assert "privileged_write" not in names
    # JSON-schema shape for clients
    tool = next(t for t in resp["result"]["tools"] if t["name"] == "safe_read")
    assert tool["inputSchema"]["type"] == "object"


def test_devon_tools_empty_without_registry():
    server = ACPServer(registry=None)
    resp = server.dispatch(_rpc(1, "devon/tools", {}), lambda n: None)
    assert resp["result"] == {"tools": []}


# ── stdio framing ───────────────────────────────────────────────────────

def _run_stdio(lines: bytes) -> list[str]:
    server, _ = _server()
    stdin = io.BytesIO(lines)
    stdout = io.BytesIO()
    server.run_stdio(stdin=stdin, stdout=stdout)
    return [ln for ln in stdout.getvalue().decode().splitlines() if ln.strip()]


def test_stdio_newline_delimited():
    out = _run_stdio(
        (json.dumps(_rpc(1, "initialize", {"protocolVersion": 1})) + "\n"
         ).encode())
    resp = json.loads(out[0])
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_stdio_content_length_framed():
    body = json.dumps(_rpc(7, "initialize", {"protocolVersion": 1})).encode()
    framed = b"Content-Length: %d\r\n\r\n" % len(body) + body
    out = _run_stdio(framed)
    resp = json.loads(out[0])
    assert resp["id"] == 7
    assert "result" in resp


def test_stdio_notification_gets_no_response():
    server, _ = _server()
    sid = _new_session(server)
    out = _run_stdio(
        (json.dumps({"jsonrpc": "2.0", "method": "session/cancel",
                     "params": {"sessionId": sid}}) + "\n").encode())
    # separate server instance in _run_stdio; just assert no crash and
    # the framing round-trips (unknown session → logged, no response)
    assert out == []


def test_stdio_parse_error_responds():
    out = _run_stdio(b"this is not json\n")
    resp = json.loads(out[0])
    assert resp["error"]["code"] == -32700


# ── SSE ─────────────────────────────────────────────────────────────────

def test_iter_sse_streams_updates_then_response():
    def turn_fn(session, text, emit):
        emit({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": session.id,
                         "update": {"sessionUpdate": "agent_message_chunk",
                                    "content": {"type": "text",
                                                "text": "hi"}}}})
        return "hi"

    server = ACPServer(turn_fn=turn_fn)
    sid = _new_session(server)
    frames = list(server.iter_sse(
        _rpc(3, "session/prompt",
             {"sessionId": sid, "prompt": [{"type": "text", "text": "go"}]})))
    assert all(f.startswith("data: ") for f in frames)
    payloads = [json.loads(f[len("data: "):]) for f in frames]
    assert payloads[0]["method"] == "session/update"
    assert payloads[-1]["id"] == 3
    assert payloads[-1]["result"] == {"stopReason": "end_turn"}


# ── HTTP buffered ───────────────────────────────────────────────────────

def test_handle_http_buffered():
    server, _ = _server()
    resp = server.handle_http(_rpc(1, "initialize", {"protocolVersion": 1}))
    assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION
