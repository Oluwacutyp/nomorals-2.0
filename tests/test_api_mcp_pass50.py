"""Pass #50 — API + MCP/ACP surface: principal parity, route gates,
protocol honesty, lifecycle.

Headline regression: the protocol servers mounted by ``serve()``
(``register_mcp`` / ``register_acp`` with bare servers) used to run every
HTTP request as the bounded DEFAULT_PRINCIPAL — a narrow named principal
could ESCALATE through /mcp (memory_write, MEM_WRITE tools) and the
owner was DOWNGRADED on /mcp + /acp. The mounts now resolve the
in-flight request principal; outside a request the server's own
configured principal applies (stdio/direct behavior unchanged).

Also pinned here: capability gates on the core routes (memory/chat/
backup/agents/events/models), grant-filtered GET /tools, HTTP
/tools/call idempotency keys + reserved argument names, JSON-RPC
batches over HTTP for the protocol routes only, MCP resources/read
-32002 for unknown URIs, ACP initialize INVALID_PARAMS for a
non-integer protocolVersion, research_run findings carrying their
advertised 'detail' field, GET /stream following the API server's CORS
posture, and serve() failing a busy port with a clear error.

Hermetic: ephemeral ports only, in-process handlers, no network.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from nomorals.api.acp import ACPServer, PROTOCOL_VERSION, register_acp
from nomorals.api.mcp_server import MCPServer, register_mcp
from nomorals.api.server import (
    APIServer,
    OWNER_PRINCIPAL,
    Principal,
    _make_handler,
    serve,
)
from nomorals.core.errors import NoMoralsError
from nomorals.core.policy import Capability, CapabilitySet
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry

OWNER_TOKEN = "owner-token-pass50"
READER_TOKEN = "reader-token-pass50"
EMPTY_TOKEN = "empty-token-pass50"

READER = Principal("reader", CapabilitySet.of(Capability.FS_READ))
EMPTY = Principal("empty", CapabilitySet.none())


# ── fixtures ──────────────────────────────────────────────────────────


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("note_read", lambda: "read-ok",
                 capability=Capability.MEM_READ, description="read a note")
    reg.register("note_write", lambda: "wrote-it",
                 capability=Capability.MEM_WRITE, description="write a note")
    reg.register("shellish", lambda: "did-it",
                 capability=Capability.EXEC_SHELL, description="shell")
    return reg


class _FakeMemory:
    def __init__(self) -> None:
        self.remembered: list[str] = []

    def remember(self, content, kind="episode", source="api",
                 importance=0.5):
        self.remembered.append(content)
        return "rec_1"

    def recall(self, query, limit=10, kind=None):
        return SimpleNamespace(
            to_dict=lambda: {"records": [], "total": 0, "took_ms": 0.0})

    def stats_snapshot(self):
        return {"records": len(self.remembered)}


def _context(tmp_path) -> SimpleNamespace:
    db = Database(str(tmp_path / "test.db"))
    settings = SimpleNamespace(
        api=SimpleNamespace(token="", max_body_mb=0),
        backup_dir=str(tmp_path), workspace_dir=str(tmp_path))
    return SimpleNamespace(
        db=db, settings=settings, tools=_registry(), bus=None,
        memory=_FakeMemory(), started_at=0)


def _api(ctx) -> APIServer:
    """An APIServer mounted exactly the way serve() mounts it."""
    api = APIServer(
        ctx, token=OWNER_TOKEN,
        principals={READER_TOKEN: READER, EMPTY_TOKEN: EMPTY},
        max_body_bytes=64 * 1024)
    register_acp(api, ACPServer(ctx))
    register_mcp(api, MCPServer(ctx))
    return api


class _Live:
    def __init__(self, api: APIServer) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                         _make_handler(api))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def req(self, method: str, path: str, token: str | None = None,
            body: bytes | None = None,
            headers: dict[str, str] | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port,
                                          timeout=10)
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = f"Bearer {token}"
        if headers:
            h.update(headers)
        conn.request(method, path, body=body, headers=h)
        resp = conn.getresponse()
        raw = resp.read()
        status, hdrs = resp.status, dict(resp.getheaders())
        conn.close()
        return status, raw, hdrs


@pytest.fixture()
def mounted(tmp_path):
    ctx = _context(tmp_path)
    api = _api(ctx)
    live = _Live(api)
    try:
        yield ctx, api, live
    finally:
        live.close()


def _rpc(method: str, params: dict | None = None, msg_id: int = 1) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method,
                       "params": params or {}}).encode()


def _tool_call(tool: str, arguments: dict | None = None,
               msg_id: int = 1) -> bytes:
    return _rpc("tools/call", {"name": "tool_call",
                               "arguments": {"tool": tool,
                                             "arguments": arguments or {}}},
                msg_id)


# ── F1: the mounted protocol servers act as the REQUEST principal ────


def test_mcp_http_narrow_principal_cannot_escalate(mounted) -> None:
    _, _, live = mounted
    # memory_write requires mem.write; the reader holds only fs.read.
    # Before the fix this SUCCEEDED (server ran as DEFAULT_PRINCIPAL).
    status, raw, _ = live.req(
        "POST", "/mcp", READER_TOKEN,
        _rpc("tools/call", {"name": "memory_write",
                            "arguments": {"text": "reader was here"}}))
    assert status == 200
    result = json.loads(raw)["result"]
    assert result["isError"] is True
    assert "mem.write" in result["content"][0]["text"]
    # Same story through the tool_call meta-tool (registry-gated).
    status, raw, _ = live.req("POST", "/mcp", READER_TOKEN,
                              _tool_call("note_write", msg_id=2))
    result = json.loads(raw)["result"]
    assert result["isError"] is True
    assert "mem.write" in result["content"][0]["text"]
    # Control: the plain HTTP surface denied the same call all along.
    status, _, _ = live.req("POST", "/tools/call", READER_TOKEN,
                            json.dumps({"name": "note_write"}).encode())
    assert status == 403


def test_mcp_http_owner_is_not_downgraded(mounted) -> None:
    _, _, live = mounted
    # The owner holds exec.shell via /tools/call…
    status, raw, _ = live.req("POST", "/tools/call", OWNER_TOKEN,
                              json.dumps({"name": "shellish"}).encode())
    assert status == 200 and json.loads(raw)["result"] == "did-it"
    # …and now via /mcp too (before the fix: "lacks 'exec.shell'").
    status, raw, _ = live.req("POST", "/mcp", OWNER_TOKEN,
                              _tool_call("shellish"))
    result = json.loads(raw)["result"]
    assert result["isError"] is False
    assert result["content"][0]["text"] == "did-it"


def test_mcp_http_idempotency_replay_survives_the_resolver(mounted) -> None:
    # Pass #44's rule, still holding with per-request principals: the
    # same JSON-RPC request retried replays; a new request id executes.
    ctx, _, live = mounted
    calls: list[int] = []
    ctx.tools.register(
        "counter", lambda: calls.append(1) or len(calls),
        capability=Capability.EXEC_SHELL, description="counting tool")
    body = _tool_call("counter", msg_id=42)
    live.req("POST", "/mcp", OWNER_TOKEN, body)
    status, raw, _ = live.req("POST", "/mcp", OWNER_TOKEN, body)
    assert json.loads(raw)["result"]["content"][0]["text"] == "1"
    assert len(calls) == 1  # replayed, not re-executed
    live.req("POST", "/mcp", OWNER_TOKEN, _tool_call("counter", msg_id=43))
    assert len(calls) == 2


def test_acp_http_tools_filtered_per_principal(mounted) -> None:
    _, _, live = mounted
    _, raw, _ = live.req("POST", "/acp", OWNER_TOKEN,
                         _rpc("devon/tools"))
    owner_tools = {t["name"] for t in json.loads(raw)["result"]["tools"]}
    assert {"note_read", "note_write", "shellish"} <= owner_tools
    _, raw, _ = live.req("POST", "/acp", READER_TOKEN,
                         _rpc("devon/tools"))
    reader_tools = {t["name"] for t in json.loads(raw)["result"]["tools"]}
    # fs.read alone unlocks none of the three test tools — and crucially
    # the reader no longer sees the DEFAULT grant's tools either.
    assert reader_tools == set()


def test_acp_sse_tools_filtered_per_principal(mounted) -> None:
    # The SSE path runs the batch on a worker thread: the request
    # principal must be handed over explicitly, not read from a
    # thread-local the worker cannot see.
    _, _, live = mounted
    for token, expect_shellish in ((OWNER_TOKEN, True),
                                   (READER_TOKEN, False)):
        conn = http.client.HTTPConnection("127.0.0.1", live.port,
                                          timeout=10)
        conn.request("POST", "/acp", body=_rpc("devon/tools"),
                     headers={"Authorization": f"Bearer {token}",
                              "Content-Type": "application/json",
                              "Accept": "text/event-stream"})
        resp = conn.getresponse()
        assert resp.status == 200
        payload = None
        for _ in range(40):
            line = resp.fp.readline().decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                frame = json.loads(line[len("data: "):])
                if isinstance(frame, dict) and "result" in frame:
                    payload = frame
                    break
        resp.close()
        conn.close()
        assert payload is not None, f"no response frame for {token}"
        names = {t["name"] for t in payload["result"]["tools"]}
        assert ("shellish" in names) is expect_shellish


def test_mounted_server_falls_back_to_own_principal_off_request(
        tmp_path) -> None:
    # Direct handler calls (no HTTP request in flight) use the protocol
    # server's own configured principal — the stdio/direct contract.
    ctx = _context(tmp_path)
    api = APIServer(ctx, token=OWNER_TOKEN, max_body_bytes=64 * 1024)
    custom = Principal("custom", CapabilitySet.of(Capability.MEM_READ))
    register_mcp(api, MCPServer(ctx, principal=custom))
    handler = api._routes[("POST", "/mcp")]
    out = handler(json.loads(_tool_call("note_read")), {})
    assert out["result"]["isError"] is False
    out = handler(json.loads(_tool_call("shellish")), {})
    assert out["result"]["isError"] is True


def test_register_mcp_respects_an_explicit_resolver(tmp_path) -> None:
    ctx = _context(tmp_path)
    api = APIServer(ctx, token=OWNER_TOKEN, max_body_bytes=64 * 1024)
    resolver = lambda: OWNER_PRINCIPAL  # noqa: E731 - identity is the point
    mcp = MCPServer(ctx, principal_resolver=resolver)
    register_mcp(api, mcp)
    assert mcp._principal_resolver is resolver


def test_unmounted_protocol_servers_unchanged(tmp_path) -> None:
    # stdio-style construction with an explicit principal behaves
    # exactly as before (no API server involved at all).
    ctx = _context(tmp_path)
    mcp = MCPServer(ctx, principal=READER)
    out = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "tool_call",
                    "arguments": {"tool": "note_write", "arguments": {}}}},
        lambda n: None)
    assert out["result"]["isError"] is True
    acp = ACPServer(ctx, principal=READER)
    out = acp.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "devon/tools", "params": {}},
        lambda n: None)
    assert out["result"]["tools"] == []


# ── F2: capability gates on the core routes ───────────────────────────


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/memory/remember", {"content": "x"}),
    ("POST", "/memory/recall", {"query": "x"}),
    ("GET", "/memory/stats", {}),
    ("POST", "/chat", {"prompt": "hi"}),
    ("POST", "/backup", {}),
    ("POST", "/agents/run", {"goal": "g"}),
    ("GET", "/events", {}),
    ("GET", "/models", {}),
])
def test_core_routes_deny_a_grantless_principal(mounted, method, path,
                                                body) -> None:
    _, api, _ = mounted
    status, payload = api.dispatch(method, path, body, {}, principal=EMPTY)
    assert status == 403, (path, payload)
    assert payload["kind"] == "CapabilityDenied"


def test_core_routes_owner_passes_the_gates(mounted) -> None:
    ctx, api, _ = mounted
    owner = api.resolve_principal(f"Bearer {OWNER_TOKEN}")
    status, _ = api.dispatch("POST", "/memory/remember",
                             {"content": "kept"}, {}, principal=owner)
    assert status == 200 and ctx.memory.remembered == ["kept"]
    status, _ = api.dispatch("POST", "/backup", {}, {}, principal=owner)
    assert status == 200
    # /models: the gate passes for the owner; the handler then needs a
    # migrated models table, which this bare fixture DB lacks (400
    # StorageError) — what matters here is that it is NOT a 403.
    status, payload = api.dispatch("GET", "/models", {}, {},
                                   principal=owner)
    assert not (status == 403 and payload.get("kind") == "CapabilityDenied")
    status, _ = api.dispatch("GET", "/events", {}, {}, principal=owner)
    assert status == 200
    # /agents/run: the gate passes; the run itself may fail against the
    # fake context, but it must NOT be a capability denial.
    status, payload = api.dispatch("POST", "/agents/run", {"goal": "g"},
                                   {}, principal=owner)
    assert not (status == 403 and payload.get("kind") == "CapabilityDenied")


def test_default_principal_keeps_local_surface_but_not_the_destructive(
        mounted) -> None:
    # Tokenless local mode: memory + chat stay available (their
    # capabilities are in DEFAULT_GRANT); backup and agent spawn are
    # beyond the bounded local grant by design and now say so.
    _, api, _ = mounted
    status, _ = api.dispatch("POST", "/memory/remember",
                             {"content": "local note"}, {})
    assert status == 200
    status, _ = api.dispatch("POST", "/memory/recall", {"query": "x"}, {})
    assert status == 200
    status, _ = api.dispatch("POST", "/backup", {}, {})
    assert status == 403
    status, _ = api.dispatch("POST", "/agents/run", {"goal": "g"}, {})
    assert status == 403


def test_tools_listing_filtered_by_grant(mounted) -> None:
    _, api, _ = mounted
    owner = api.resolve_principal(f"Bearer {OWNER_TOKEN}")
    _, payload = api.dispatch("GET", "/tools", {}, {}, principal=owner)
    owner_names = {t["name"] for t in payload["tools"]}
    assert "shellish" in owner_names
    _, payload = api.dispatch("GET", "/tools", {}, {}, principal=EMPTY)
    empty_names = {t["name"] for t in payload["tools"]}
    assert "shellish" not in empty_names
    assert "note_write" not in empty_names
    # The bounded local principal sees exactly the tools it may call.
    _, payload = api.dispatch("GET", "/tools", {}, {})
    local_names = {t["name"] for t in payload["tools"]}
    assert "note_read" in local_names and "note_write" in local_names
    assert "shellish" not in local_names


# ── F3: /tools/call parity — idempotency + reserved names ─────────────


def test_tools_call_idempotency_key_replays(mounted) -> None:
    ctx, api, _ = mounted
    calls: list[int] = []
    ctx.tools.register(
        "counter", lambda: calls.append(1) or len(calls),
        capability=Capability.FS_READ, description="counting tool")
    owner = api.resolve_principal(f"Bearer {OWNER_TOKEN}")
    body = {"name": "counter", "idempotency_key": "req-123"}
    s1, p1 = api.dispatch("POST", "/tools/call", body, {}, principal=owner)
    s2, p2 = api.dispatch("POST", "/tools/call", body, {}, principal=owner)
    assert (s1, s2) == (200, 200)
    assert p1["result"] == p2["result"] == 1
    assert len(calls) == 1  # the retry replayed instead of re-executing
    # No key → each call executes, as before.
    api.dispatch("POST", "/tools/call", {"name": "counter"}, {},
                 principal=owner)
    assert len(calls) == 2


@pytest.mark.parametrize("reserved", ["actor", "capabilities",
                                      "confirmation", "idempotency_key"])
def test_tools_call_reserved_argument_names_rejected(mounted,
                                                     reserved) -> None:
    _, api, _ = mounted
    owner = api.resolve_principal(f"Bearer {OWNER_TOKEN}")
    status, payload = api.dispatch(
        "POST", "/tools/call",
        {"name": "note_read", "arguments": {reserved: "spoof"}},
        {}, principal=owner)
    # Before the fix: HTTP 500 {"error": "TypeError"} from the keyword
    # collision inside registry.call.
    assert status == 200
    assert payload["ok"] is False
    assert "reserved" in payload["error"] and reserved in payload["error"]


def test_mcp_tool_call_reserved_argument_names_rejected(tmp_path) -> None:
    ctx = _context(tmp_path)
    mcp = MCPServer(ctx, principal=OWNER_PRINCIPAL)
    for reserved in ("actor", "capabilities", "confirmation"):
        out = mcp.dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "tool_call",
                        "arguments": {"tool": "note_read",
                                      "arguments": {reserved: "spoof"}}}},
            lambda n: None)
        assert out["result"]["isError"] is True
        assert "reserved" in out["result"]["content"][0]["text"]
    out = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "tool_call",
                    "arguments": {"tool": "note_read",
                                  "arguments": ["not", "an", "object"]}}},
        lambda n: None)
    assert out["result"]["isError"] is True
    assert "must be an object" in out["result"]["content"][0]["text"]


# ── F4/F5/F8: protocol honesty ────────────────────────────────────────


def test_batches_work_over_http_on_protocol_routes(mounted) -> None:
    _, _, live = mounted
    batch = json.dumps([
        {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
    ]).encode()
    status, raw, _ = live.req("POST", "/mcp", OWNER_TOKEN, batch)
    assert status == 200
    responses = json.loads(raw)
    assert [r["id"] for r in responses] == [1, 2]
    batch = json.dumps([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": 1}},
        {"jsonrpc": "2.0", "id": 2, "method": "devon/tools", "params": {}},
    ]).encode()
    status, raw, _ = live.req("POST", "/acp", OWNER_TOKEN, batch)
    assert status == 200
    responses = json.loads(raw)
    assert [r["id"] for r in responses] == [1, 2]
    assert "shellish" in {t["name"] for t in responses[1]["result"]["tools"]}


def test_batch_still_rejected_on_plain_routes(mounted) -> None:
    _, _, live = mounted
    status, raw, _ = live.req("POST", "/tools/call", OWNER_TOKEN,
                              json.dumps([{"name": "note_read"}]).encode())
    assert status == 400
    assert "body must be a JSON object" in raw.decode()


def test_acp_initialize_rejects_a_non_integer_version(tmp_path) -> None:
    acp = ACPServer(_context(tmp_path))
    out = acp.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "abc"}}, lambda n: None)
    # Before the fix: -32603 "ValueError: invalid literal for int()…".
    assert out["error"]["code"] == -32602
    out = acp.dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "initialize",
         "params": {"protocolVersion": 1}}, lambda n: None)
    assert out["result"]["protocolVersion"] == 1
    out = acp.dispatch(
        {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {}},
        lambda n: None)
    assert out["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_research_run_findings_carry_detail(tmp_path, monkeypatch) -> None:
    import nomorals.research.pipeline as pipeline

    finding = SimpleNamespace(title="T", url="https://x", snippet="S",
                              detail="the long fetched body")
    seen: dict[str, Any] = {}

    def _run_job(job, rctx):
        seen["job"] = job
        return [finding]

    monkeypatch.setattr(pipeline, "run_job", _run_job)
    ctx = _context(tmp_path)
    mcp = MCPServer(ctx, principal=OWNER_PRINCIPAL)
    out = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "research_run",
                    "arguments": {"topic": "anything"}}},
        lambda n: None)
    result = out["result"]
    assert result["isError"] is False
    # The handler must hand run_job a fully-formed ResearchJob — it used
    # to omit the required `id`, so research_run failed on EVERY call.
    assert seen["job"].id and seen["job"].topic == "anything"
    found = result["structuredContent"]["findings"]
    assert found[0]["detail"] == "the long fetched body"


# ── F6/F7: lifecycle + CORS posture ───────────────────────────────────


def test_serve_on_a_busy_port_fails_with_a_clear_error(tmp_path) -> None:
    ctx = _context(tmp_path)
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(NoMoralsError, match="cannot bind"):
            serve(ctx, host="127.0.0.1", port=port, protocols=False)
    finally:
        blocker.close()


def test_stream_cors_follows_the_api_servers_posture(tmp_path) -> None:
    ctx = _context(tmp_path)
    for cors_origins, expect_header in ((None, False), (("*",), True)):
        api = APIServer(ctx, token=OWNER_TOKEN,
                        cors_origins=cors_origins,
                        max_body_bytes=64 * 1024)
        live = _Live(api)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", live.port,
                                              timeout=10)
            conn.request("GET", "/stream",
                         headers={"Authorization": f"Bearer {OWNER_TOKEN}"})
            resp = conn.getresponse()
            assert resp.status == 200
            # Before the fix this was "*" even with CORS off.
            assert (resp.getheader("Access-Control-Allow-Origin")
                    is not None) is expect_header
            resp.close()
            conn.close()
        finally:
            live.close()
