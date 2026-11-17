"""ACP session mcpServers wiring — all offline (fake stdio MCP server).

A session that declares mcpServers must be able to call those servers'
tools inside its turns, through a per-turn registry view wrapping the
shared registry; a session that declares nothing behaves exactly as
before. The declared servers are the session's own enablement — no
config master switch — while the session principal's capability grant
still gates every call (NET_OUT).
"""
import os
import sys
import textwrap
import time
from types import SimpleNamespace

from nomorals.api.acp import ACPServer
from nomorals.api.server import Principal
from nomorals.core.policy import CapabilitySet
from nomorals.tools.registry import ToolRegistry

#: Fake MCP server over stdio (same shape as tests/test_mcp_client.py).
_FAKE_SERVER = textwrap.dedent('''
    import json, os, sys
    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if mid is None:
            continue
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp", "version": "0.1"}}})
            continue
        if method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
            continue
        if method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "echo", "description": "Echo back",
                 "inputSchema": {"type": "object",
                                 "properties": {"text": {"type": "string"}},
                                 "required": ["text"]}},
                {"name": "failer", "description": "Always errors",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "ping", "description": "Fake ping",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "env_echo", "description": "Echo an env var",
                 "inputSchema": {"type": "object",
                                 "properties": {"name": {"type": "string"}}}},
            ]}})
            continue
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                text = args.get("text", "")
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": f"hello {text}"}],
                    "structuredContent": {"text": text},
                    "isError": False}})
            elif name == "failer":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "boom"}],
                    "isError": True}})
            elif name == "ping":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "fake-pong"}],
                    "isError": False}})
            elif name == "env_echo":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": os.environ.get(
                                     args.get("name", ""), "")}],
                    "isError": False}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "unknown tool"}],
                    "isError": True}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "error": {
            "code": -32601, "message": f"method not found: {method}"}})
''')


def _write_fake_server(tmp_path) -> str:
    path = os.path.join(str(tmp_path), "fake_mcp_server.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_FAKE_SERVER)
    return path


def _rpc(mid, method, params=None):
    msg = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _registry_with_ping() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        "ping",
        description="Native ping tool.",
        capability="",
        parameters={},
    )
    def _ping() -> dict:
        return {"ok": True, "text": "pong"}

    return registry


def _server(registry, *, grant=None, **kw):
    principal = Principal(
        name="test",
        grant=grant if grant is not None else CapabilitySet.all())
    return ACPServer(
        context=SimpleNamespace(llm=object()),
        registry=registry,
        principal=principal,
        **kw)


def _new_session(server, mid=1, *, mcp_servers=None):
    params = {"cwd": "/tmp"}
    if mcp_servers is not None:
        params["mcpServers"] = mcp_servers
    resp = server.dispatch(_rpc(mid, "session/new", params), lambda n: None)
    assert "result" in resp, resp
    return resp["result"]["sessionId"]


def _fake_decl(script: str) -> dict:
    return {"name": "fake", "command": sys.executable,
            "args": [script]}


# ── initialize ──────────────────────────────────────────────────────────

def test_initialize_reports_mcp_http_capability():
    server = _server(_registry_with_ping())
    resp = server.dispatch(
        _rpc(0, "initialize", {"protocolVersion": 1}),
        lambda n: None)
    caps = resp["result"]["agentCapabilities"]["mcpCapabilities"]
    # The client speaks Streamable HTTP; the legacy SSE transport is
    # NOT implemented (see tools/mcp_client.py) — sse stays False.
    assert caps == {"http": True, "sse": False}


# ── the per-turn view ───────────────────────────────────────────────────

def test_view_lists_session_mcp_tools(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[_fake_decl(script)])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    names = [s["name"] for s in view.schemas(
        capabilities=CapabilitySet.all())]
    assert "ping" in names  # native tool still listed
    assert "mcp_fake_echo" in names
    assert "mcp_fake_failer" in names
    assert "mcp_fake_ping" in names  # prefixed: never shadows native ping


def test_view_calls_session_mcp_tool(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[_fake_decl(script)])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    outcome = view.call("mcp_fake_echo", actor="acp:test",
                        capabilities=CapabilitySet.all(), text="world")
    assert outcome.ok, outcome
    assert outcome.value["ok"] is True
    assert outcome.value["text"] == "hello world"
    assert outcome.value["structured"] == {"text": "world"}


def test_view_mcp_tool_error_is_a_result_not_a_raise(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[_fake_decl(script)])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    outcome = view.call("mcp_fake_failer", actor="acp:test",
                        capabilities=CapabilitySet.all())
    assert not outcome.ok
    assert "boom" in str(outcome.error)


def test_view_call_many_routes_mcp_and_native(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[_fake_decl(script)])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    outcomes = view.call_many(
        [("mcp_fake_echo", {"text": "x"}), ("ping", {})],
        actor="acp:test", capabilities=CapabilitySet.all())
    assert len(outcomes) == 2
    assert outcomes[0].ok and outcomes[0].value["text"] == "hello x"
    assert outcomes[1].ok


# ── the turn path ───────────────────────────────────────────────────────

def test_turn_uses_per_turn_view_for_declared_servers(tmp_path, monkeypatch):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[_fake_decl(script)])
    seen: dict = {}
    emitted: list = []

    def fake_run(message, *, llm, registry, history=None, actor="owner-loop",
                 capabilities=None, resume_from=None, **kw):
        from nomorals.agents.orchestration.tools import ToolAdapter
        seen["registry"] = registry
        adapter = ToolAdapter(registry, actor=actor,
                              capabilities=capabilities)
        ok, obs = adapter.call("mcp_fake_echo", {"text": "world"})
        assert ok, obs
        return SimpleNamespace(response=f"echo said: {obs}",
                               memory_snapshot=None, asked_user=False,
                               success=True)

    import nomorals.agents.orchestration as orch
    monkeypatch.setattr(orch, "run_agentic", fake_run)
    resp = server.dispatch(
        _rpc(2, "session/prompt",
             {"sessionId": sid,
              "prompt": [{"type": "text", "text": "say hello"}]}),
        emitted.append)
    assert resp["result"]["stopReason"] == "end_turn"
    # The turn ran through the per-turn view, not the shared registry.
    assert seen["registry"] is not registry
    assert "hello world" in "".join(
        u["params"]["update"].get("content", {}).get("text", "")
        for u in emitted
        if u["params"]["update"].get("sessionUpdate")
        == "agent_message_chunk")
    # The MCP tool call was visible to the client as a tool_call update.
    kinds = [u["params"]["update"].get("sessionUpdate") for u in emitted]
    assert "tool_call" in kinds


def test_turn_without_servers_uses_shared_registry(tmp_path, monkeypatch):
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server)
    seen: dict = {}

    def fake_run(message, *, llm, registry, history=None, actor="owner-loop",
                 capabilities=None, resume_from=None, **kw):
        seen["registry"] = registry
        return SimpleNamespace(response="done", memory_snapshot=None,
                               asked_user=False, success=True)

    import nomorals.agents.orchestration as orch
    monkeypatch.setattr(orch, "run_agentic", fake_run)
    resp = server.dispatch(
        _rpc(2, "session/prompt",
             {"sessionId": sid,
              "prompt": [{"type": "text", "text": "hi"}]}),
        lambda n: None)
    assert resp["result"]["stopReason"] == "end_turn"
    assert seen["registry"] is registry


# ── capability gating ───────────────────────────────────────────────────

def test_mcp_tools_hidden_without_net_out(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    narrow = CapabilitySet.of("fs.read")
    server = _server(registry, grant=narrow)
    sid = _new_session(server, mcp_servers=[_fake_decl(script)])
    session = server._sessions[sid]
    view = server._session_view(session, narrow)
    names = [s["name"] for s in view.schemas(capabilities=narrow)]
    assert "ping" in names
    assert not [n for n in names if n.startswith("mcp_")]
    outcome = view.call("mcp_fake_echo", actor="acp:test",
                        capabilities=narrow, text="world")
    assert not outcome.ok
    assert "net.out" in str(outcome.error)


# ── failure modes ───────────────────────────────────────────────────────

def test_dead_server_fails_open(tmp_path):
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[
        {"name": "dead", "command": "/nonexistent/binary-xyz"}])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    # Shared tools still listed; the dead server contributes nothing.
    names = [s["name"] for s in view.schemas(
        capabilities=CapabilitySet.all())]
    assert "ping" in names
    assert not [n for n in names if n.startswith("mcp_")]
    # Calling into the dead server is an error result, never a raise,
    # and the native tool still works through the same view.
    outcome = view.call("mcp_dead_echo", actor="acp:test",
                        capabilities=CapabilitySet.all(), text="x")
    assert not outcome.ok
    ok = view.call("ping", actor="acp:test",
                   capabilities=CapabilitySet.all())
    assert ok.ok


def test_bad_declaration_skipped(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[
        "not-a-dict",
        {"name": "no-command", "transport": "stdio"},
        {"name": "bad-transport", "transport": "carrier-pigeon",
         "command": "x"},
        _fake_decl(script),
    ])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    names = [s["name"] for s in view.schemas(
        capabilities=CapabilitySet.all())]
    assert "mcp_fake_echo" in names
    assert not [n for n in names if "no-command" in n or "bad" in n]
    outcome = view.call("mcp_fake_echo", actor="acp:test",
                        capabilities=CapabilitySet.all(), text="ok")
    assert outcome.ok


def test_duplicate_server_name_deduped(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[
        _fake_decl(script),
        _fake_decl(script),
    ])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    names = [s["name"] for s in view.schemas(
        capabilities=CapabilitySet.all())]
    assert names.count("mcp_fake_echo") == 1


# ── env values + lifecycle ──────────────────────────────────────────────

def test_declared_env_values_reach_server(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry)
    sid = _new_session(server, mcp_servers=[{
        "name": "fake", "command": sys.executable, "args": [script],
        "env": {"ACP_FAKE_TOKEN": "s3cr3t"},
    }])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    outcome = view.call("mcp_fake_env_echo", actor="acp:test",
                        capabilities=CapabilitySet.all(), name="ACP_FAKE_TOKEN")
    assert outcome.ok, outcome
    assert outcome.value["text"] == "s3cr3t"


def test_http_headers_reach_transport():
    from nomorals.tools.mcp_client import MCPServerConfig, _StreamableHttpTransport
    cfg = MCPServerConfig(
        name="h", transport="http", url="https://example.invalid/mcp",
        headers={"Authorization": "Bearer abc"})
    transport = _StreamableHttpTransport(cfg, max_payload_bytes=1000)
    headers = transport._headers()
    assert headers["Authorization"] == "Bearer abc"
    assert headers["MCP-Protocol-Version"] == "2025-11-25"


def test_prune_closes_session_mcp(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry, session_ttl_s=0.05)
    sid = _new_session(server, mcp_servers=[_fake_decl(script)])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    view.schemas(capabilities=CapabilitySet.all())  # connects
    assert session.mcp_manager is not None
    client = session.mcp_manager._clients["fake"]
    assert client.connected
    time.sleep(0.1)
    assert server.prune() == 1
    assert not client.connected
    assert sid not in server._sessions


def test_session_expiry_closes_mcp(tmp_path):
    script = _write_fake_server(tmp_path)
    registry = _registry_with_ping()
    server = _server(registry, session_ttl_s=0.05)
    sid = _new_session(server, mcp_servers=[_fake_decl(script)])
    session = server._sessions[sid]
    view = server._session_view(session, CapabilitySet.all())
    view.schemas(capabilities=CapabilitySet.all())  # connects
    client = session.mcp_manager._clients["fake"]
    assert client.connected
    time.sleep(0.1)
    resp = server.dispatch(_rpc(2, "session/load", {"sessionId": sid}),
                           lambda n: None)
    assert resp["error"]["code"] == -32602
    assert not client.connected
