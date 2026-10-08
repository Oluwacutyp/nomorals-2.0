"""MCP server tests (build-map #41). All offline — no real MCP client."""
import io
import json
import tempfile

import pytest

from nomorals.api.mcp_server import (
    MCPServer, register_mcp, MCP_PROTOCOL_VERSION, METHOD_NOT_FOUND,
)
from nomorals.api.server import APIServer, DEFAULT_PRINCIPAL


def _req(mid, method, params=None):
    msg = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


@pytest.fixture()
def server(tmp_path):
    return MCPServer(memory=None, principal=DEFAULT_PRINCIPAL)


@pytest.fixture()
def mem_server(tmp_path):
    # MCPServer with an isolated temp two-tier DB
    from nomorals.memory.tiers import TwoTierMemory
    mem = TwoTierMemory(db=str(tmp_path / "mcp.db"))
    return MCPServer(memory=mem, principal=DEFAULT_PRINCIPAL), mem


def _call(srv, mid, method, params=None):
    return srv.dispatch(_req(mid, method, params), lambda n: None)


def test_initialize_handshake(server):
    r = _call(server, 1, "initialize",
              {"protocolVersion": "2025-11-25", "clientInfo": {"name": "t"}})
    assert r["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert r["result"]["serverInfo"]["name"] == "devon"
    assert "tools" in r["result"]["capabilities"]
    assert "resources" in r["result"]["capabilities"]


def test_ping_and_initialized_notification(server):
    assert _call(server, 2, "ping")["result"] == {}
    # notification (no id) → None, never a response
    assert server.dispatch(
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        lambda n: None) is None


def test_tools_list_has_four_tools(server):
    r = _call(server, 3, "tools/list")
    names = {t["name"] for t in r["result"]["tools"]}
    assert names == {"memory_query", "memory_write", "tool_call",
                     "research_run"}
    for t in r["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"


def test_memory_query_roundtrip(mem_server):
    srv, mem = mem_server
    mem.facts.add_fact("my girlfriend is Ada", confidence=0.9)
    r = _call(srv, 4, "tools/call",
              {"name": "memory_query",
               "arguments": {"query": "what's my girlfriend's name?"}})
    assert r["result"]["isError"] is False
    payload = json.loads(r["result"]["content"][0]["text"])
    assert any("Ada" in f for f in payload["facts"])


def test_memory_write_and_readback(mem_server):
    srv, mem = mem_server
    r = _call(srv, 5, "tools/call",
              {"name": "memory_write",
               "arguments": {"text": "I prefer morning briefings"}})
    assert r["result"]["isError"] is False
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["text"] == "I prefer morning briefings"
    # and it shows up in the facts resource
    r2 = _call(srv, 6, "resources/read",
               {"uri": "devon://memory/facts"})
    facts = json.loads(r2["result"]["contents"][0]["text"])
    assert any("morning briefings" in f["text"] for f in facts)


def test_memory_write_denied_without_capability(mem_server):
    from nomorals.core.policy import CapabilitySet
    from nomorals.api.server import Principal
    srv, _ = mem_server
    srv._principal = Principal(name="nobody", grant=CapabilitySet.none())
    r = _call(srv, 7, "tools/call",
              {"name": "memory_write", "arguments": {"text": "x"}})
    # capability denied → tool-level error, not a protocol crash
    assert r["result"]["isError"] is True
    assert "mem.write" in r["result"]["content"][0]["text"]


def test_tool_call_unknown_tool_is_error(server):
    from nomorals.tools.registry import ToolRegistry
    server.registry = ToolRegistry()
    r = _call(server, 8, "tools/call",
              {"name": "tool_call",
               "arguments": {"tool": "nope_not_real", "arguments": {}}})
    assert r["result"]["isError"] is True


def test_tool_call_routes_through_registry(server):
    from nomorals.tools.registry import ToolRegistry
    from nomorals.core.result import Ok
    reg = ToolRegistry()
    reg.register("echo_caps", lambda text="": text.upper(),
                 description="echo", capability="")
    server.registry = reg
    r = _call(server, 9, "tools/call",
              {"name": "tool_call",
               "arguments": {"tool": "echo_caps",
                             "arguments": {"text": "hello"}}})
    assert r["result"]["isError"] is False
    assert "HELLO" in r["result"]["content"][0]["text"]


def test_resources_list(mem_server):
    srv, _ = mem_server
    r = _call(srv, 10, "resources/list")
    uris = {res["uri"] for res in r["result"]["resources"]}
    assert uris == {"devon://memory/facts", "devon://memory/entities",
                    "devon://memory/timeline"}


def test_resources_read_timeline(mem_server):
    srv, mem = mem_server
    f1 = mem.facts.add_fact("my favorite color is blue")
    mem.facts.supersede_fact(f1.id, "my favorite color is green")
    r = _call(srv, 11, "resources/read",
              {"uri": "devon://memory/timeline"})
    items = json.loads(r["result"]["contents"][0]["text"])
    texts = [i["text"] for i in items]
    assert "my favorite color is blue" in texts
    assert "my favorite color is green" in texts


def test_resources_read_unknown_uri(mem_server):
    srv, _ = mem_server
    r = _call(srv, 12, "resources/read", {"uri": "devon://nope"})
    assert r["error"]["code"] == -32602


def test_prompts_list_and_get(server):
    r = _call(server, 13, "prompts/list")
    assert any(p["name"] == "devon-brief"
               for p in r["result"]["prompts"])
    r2 = _call(server, 14, "prompts/get",
               {"name": "devon-brief",
                "arguments": {"topic": "the NITDA challenge"}})
    msgs = r2["result"]["messages"]
    assert msgs and "NITDA challenge" in msgs[0]["content"]["text"]


def test_unknown_method_is_32601(server):
    r = _call(server, 15, "sampling/createMessage", {})
    assert r["error"]["code"] == METHOD_NOT_FOUND
    r2 = _call(server, 16, "elicitation/create", {})
    assert r2["error"]["code"] == METHOD_NOT_FOUND


def test_batch_and_parse_errors(server):
    out = server.handle_batch(
        [_req(1, "ping"), _req(2, "nope/method")], lambda n: None)
    assert len(out) == 2
    assert out[0]["result"] == {}
    assert out[1]["error"]["code"] == METHOD_NOT_FOUND


def test_stdio_framing(mem_server):
    srv, mem = mem_server
    mem.facts.add_fact("my girlfriend is Ada", confidence=0.9)
    stdin = io.BytesIO(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        b'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n')
    stdout = io.BytesIO()
    rc = srv.run_stdio(stdin=stdin, stdout=stdout)
    assert rc == 0
    lines = [json.loads(l) for l in stdout.getvalue().split(b"\n") if l.strip()]
    assert len(lines) == 2
    assert lines[0]["result"]["serverInfo"]["name"] == "devon"
    assert len(lines[1]["result"]["tools"]) == 4


def test_stdio_content_length_framing(server):
    body = b'{"jsonrpc":"2.0","id":9,"method":"ping","params":{}}'
    raw = (b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    stdin = io.BytesIO(raw)
    stdout = io.BytesIO()
    server.run_stdio(stdin=stdin, stdout=stdout)
    lines = [json.loads(l) for l in stdout.getvalue().split(b"\n") if l.strip()]
    assert lines[0]["id"] == 9 and lines[0]["result"] == {}


def test_register_mcp_mounts_on_api_server(mem_server):
    srv, _ = mem_server
    api = APIServer(context=object())
    register_mcp(api, srv)
    assert ("POST", "/mcp") in api._routes
    handler = api._routes[("POST", "/mcp")]
    resp = handler({"jsonrpc": "2.0", "id": 1, "method": "ping",
                    "params": {}}, {})
    assert resp["result"] == {}


def test_dispatch_never_raises(server):
    assert server.dispatch(None, lambda n: None)["error"]["code"] == -32600
    assert server.dispatch("garbage", lambda n: None)["error"]["code"] == -32600
    r = _call(server, 99, "tools/call", {"name": "", "arguments": {}})
    assert "error" in r or "result" in r
