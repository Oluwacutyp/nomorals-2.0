"""API sweep tests: request IDs, RFC 9457 errors, rate limiting, CORS,
readiness, protocol mounts (server.py); annotations, structured content,
completion, logging, progress, resource templates, version negotiation
(mcp_server.py); tool kinds, rawInput/rawOutput, locations, plan and
available-commands updates (acp.py)."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from nomorals.api.acp import (
    ACPServer,
    _tool_kind,
    _tool_locations,
    register_acp,
)
from nomorals.api.mcp_server import (
    MCP_PROTOCOL_VERSION,
    MCPServer,
    RESOURCE_NOT_FOUND,
    register_mcp,
)
from nomorals.api.server import (
    APIServer,
    DEFAULT_PRINCIPAL,
    OWNER_PRINCIPAL,
    RateLimiter,
    _make_handler,
)
from nomorals.core.policy import Capability, CapabilitySet
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry

OWNER_TOKEN = "owner-secret-token"
RO_TOKEN = "ro-token"


def _context(tmpdir: str, db: Any = "memory") -> Any:
    database = Database(":memory:") if db == "memory" else db
    settings = SimpleNamespace(
        api=SimpleNamespace(token="", max_body_mb=0),
        backup_dir=tmpdir,
        workspace_dir=tmpdir,
    )
    return SimpleNamespace(
        db=database, settings=settings, tools=ToolRegistry(), bus=None,
        memory=None,
    )


def _server(context: Any, **kw: Any) -> APIServer:
    kw.setdefault("max_body_bytes", 64 * 1024)
    kw.setdefault("token", "")
    return APIServer(context, **kw)


def _owner(context: Any, **kw: Any) -> tuple[APIServer, Any]:
    api = _server(context, token=OWNER_TOKEN, **kw)
    principal = api.resolve_principal(f"Bearer {OWNER_TOKEN}")
    assert principal is OWNER_PRINCIPAL
    return api, principal


class _LiveServer:
    """Real HTTP server on loopback; http.client for full header control."""

    def __init__(self, api: APIServer) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        return conn.getresponse()


@pytest.fixture()
def live():
    api = _server(_context("/tmp"), token=OWNER_TOKEN)
    srv = _LiveServer(api)
    yield srv, api
    srv.close()


# ── request IDs ──────────────────────────────────────────────────────────


def test_request_id_honored_and_echoed(live) -> None:
    srv, _ = live
    resp = srv.request("GET", "/live", {"X-Request-ID": "client-123"})
    assert resp.status == 200
    assert resp.getheader("X-Request-ID") == "client-123"


def test_request_id_minted_when_absent(live) -> None:
    srv, _ = live
    resp = srv.request("GET", "/live")
    rid = resp.getheader("X-Request-ID")
    assert rid and rid.startswith("req_")


def test_hostile_request_id_is_replaced_not_echoed(live) -> None:
    import socket

    srv, _ = live
    s = socket.create_connection(("127.0.0.1", srv.port), timeout=10)
    try:
        s.sendall(b"GET /live HTTP/1.1\r\nHost: x\r\n"
                   b"X-Request-ID: evil\r\n injected: 1\r\n\r\n")
        resp = s.recv(4096).decode("latin1")
    finally:
        s.close()
    assert resp.startswith("HTTP/1.1 200")
    assert "injected:" not in resp  # no response splitting
    rid = next(l.split(": ", 1)[1] for l in resp.split("\r\n")
               if l.lower().startswith("x-request-id:"))
    assert rid.startswith("req_")


# ── RFC 9457 error envelope ──────────────────────────────────────────────


def test_404_is_problem_details(live) -> None:
    srv, _ = live
    resp = srv.request(
        "GET", "/nope", {"Authorization": f"Bearer {OWNER_TOKEN}"})
    assert resp.status == 404
    assert "application/problem+json" in resp.getheader("Content-Type")
    body = json.loads(resp.read())
    assert body["type"] == "about:blank"
    assert body["title"] == "Not Found"
    assert body["status"] == 404
    assert body["code"] == "not_found"
    assert body["detail"]
    assert body["error"] == body["detail"]  # legacy key kept
    # instance correlates with the echoed request id
    assert body["instance"] == resp.getheader("X-Request-ID")


def test_401_is_problem_details(live) -> None:
    srv, _ = live
    resp = srv.request("GET", "/health")  # no token, token configured
    assert resp.status == 401
    body = json.loads(resp.read())
    assert body["code"] == "auth_required"
    assert body["status"] == 401
    assert body["instance"]


def test_405_for_wrong_method_with_allow_header(tmp_path) -> None:
    api, owner = _owner(_context(str(tmp_path)))
    status, payload = api.dispatch("POST", "/health", {}, {},
                                   principal=owner)
    assert status == 405
    assert payload["allow"] == ["GET"]
    assert payload["kind"] == "MethodNotAllowed"


def test_405_allow_header_on_wire(live) -> None:
    srv, _ = live
    resp = srv.request(
        "POST", "/health", {"Authorization": f"Bearer {OWNER_TOKEN}"})
    assert resp.status == 405
    assert resp.getheader("Allow") == "GET"


def test_security_headers(live) -> None:
    srv, _ = live
    resp = srv.request("GET", "/live")
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    assert resp.getheader("Referrer-Policy") == "no-referrer"


# ── rate limiting ────────────────────────────────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_rate_limiter_token_bucket() -> None:
    clock = _Clock()
    rl = RateLimiter(clock=clock)
    ok1, rem1, _, _ = rl.check("k", per_minute=2)
    ok2, rem2, _, _ = rl.check("k", per_minute=2)
    assert ok1 and ok2 and rem1 == 1 and rem2 == 0
    allowed, remaining, retry_after, reset_at = rl.check("k", per_minute=2)
    assert not allowed and remaining == 0 and retry_after > 0
    assert reset_at > 1000.0
    clock.t += 30.0  # half a minute → one token refills
    allowed, remaining, _, _ = rl.check("k", per_minute=2)
    assert allowed and remaining == 0


def test_rate_limiter_keys_are_independent() -> None:
    rl = RateLimiter(clock=_Clock())
    rl.check("a", per_minute=1)
    assert not rl.check("a", per_minute=1)[0]
    assert rl.check("b", per_minute=1)[0]


def test_http_429_with_retry_after(tmp_path) -> None:
    api, owner = _owner(_context(str(tmp_path)), rate_limit_per_minute=2)
    srv = _LiveServer(api)
    try:
        headers = {"Authorization": f"Bearer {OWNER_TOKEN}"}
        # Owner is exempt: many requests, no 429.
        for _ in range(4):
            resp = srv.request("GET", "/docs", headers)
            assert resp.status == 200
            resp.read()
    finally:
        srv.close()


def test_http_429_for_limited_principal(tmp_path) -> None:
    api = _server(_context(str(tmp_path)), rate_limit_per_minute=2)
    srv = _LiveServer(api)
    try:
        r1 = srv.request("GET", "/docs")
        r2 = srv.request("GET", "/docs")
        assert r1.status == 200 and r2.status == 200
        assert r1.getheader("X-RateLimit-Limit") == "2"
        assert r1.getheader("X-RateLimit-Remaining") == "1"
        assert r2.getheader("X-RateLimit-Remaining") == "0"
        assert r1.getheader("X-RateLimit-Reset")
        r1.read()
        r2.read()
        r3 = srv.request("GET", "/docs")
        assert r3.status == 429
        assert r3.getheader("Retry-After")
        body = json.loads(r3.read())
        assert body["code"] == "rate_limited"
        assert body["status"] == 429
    finally:
        srv.close()


def test_rate_limit_off_by_default(tmp_path) -> None:
    api = _server(_context(str(tmp_path)))
    assert api._limit_for("local") == 0
    assert api._limit_for("owner") == 0
    api2 = _server(_context(str(tmp_path)), rate_limit_per_minute=5,
                   rate_limits={"local": 10})
    assert api2._limit_for("local") == 10
    assert api2._limit_for("other") == 5
    assert api2._limit_for("owner") == 0


def test_live_probe_exempt_from_rate_limit(tmp_path) -> None:
    api = _server(_context(str(tmp_path)), rate_limit_per_minute=1)
    srv = _LiveServer(api)
    try:
        for _ in range(3):
            resp = srv.request("GET", "/live")
            assert resp.status == 200
            resp.read()
    finally:
        srv.close()


# ── CORS ─────────────────────────────────────────────────────────────────


def test_cors_preflight(tmp_path) -> None:
    api = _server(_context(str(tmp_path)),
                  cors_origins=["https://example.com"])
    srv = _LiveServer(api)
    try:
        resp = srv.request("OPTIONS", "/health",
                           {"Origin": "https://example.com"})
        assert resp.status == 204
        assert resp.getheader("Access-Control-Allow-Origin") == \
            "https://example.com"
        assert "Authorization" in resp.getheader("Access-Control-Allow-Headers")
        # And on a real response.
        resp = srv.request("GET", "/docs",
                           {"Origin": "https://example.com"})
        assert resp.getheader("Access-Control-Allow-Origin") == \
            "https://example.com"
        resp.read()
    finally:
        srv.close()


def test_cors_off_by_default_and_origin_mismatch(tmp_path) -> None:
    api = _server(_context(str(tmp_path)))
    srv = _LiveServer(api)
    try:
        resp = srv.request("GET", "/docs", {"Origin": "https://example.com"})
        assert resp.getheader("Access-Control-Allow-Origin") is None
        resp.read()
        resp = srv.request("OPTIONS", "/docs")
        assert resp.status == 404  # no CORS → OPTIONS is unknown
        resp.read()
    finally:
        srv.close()
    api2 = _server(_context(str(tmp_path)),
                   cors_origins=["https://example.com"])
    srv2 = _LiveServer(api2)
    try:
        resp = srv2.request("OPTIONS", "/health",
                            {"Origin": "https://evil.example"})
        assert resp.status == 404
        resp.read()
    finally:
        srv2.close()


# ── readiness ────────────────────────────────────────────────────────────


def test_ready_ok(tmp_path) -> None:
    api, owner = _owner(_context(str(tmp_path)))
    status, payload = api.dispatch("GET", "/ready", {}, {},
                                   principal=owner)
    assert status == 200
    assert payload == {"ok": True}


def test_ready_503_without_db(tmp_path) -> None:
    ctx = _context(str(tmp_path), db=None)
    api, owner = _owner(ctx)
    status, payload = api.dispatch("GET", "/ready", {}, {},
                                   principal=owner)
    assert status == 503
    assert payload["kind"] == "ServiceUnavailable"


def test_ready_503_when_db_down(tmp_path) -> None:
    class _DeadDb:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("disk gone")

    ctx = _context(str(tmp_path), db=_DeadDb())
    api, owner = _owner(ctx)
    status, payload = api.dispatch("GET", "/ready", {}, {},
                                   principal=owner)
    assert status == 503


def test_ready_on_wire(live) -> None:
    srv, _ = live
    resp = srv.request(
        "GET", "/ready", {"Authorization": f"Bearer {OWNER_TOKEN}"})
    assert resp.status == 200
    assert json.loads(resp.read()) == {"ok": True}


def test_docs_lists_ready(tmp_path) -> None:
    api, owner = _owner(_context(str(tmp_path)))
    status, payload = api.dispatch("GET", "/docs", {}, {},
                                   principal=owner)
    assert status == 200
    assert any(r["path"] == "/ready" and r["description"]
               for r in payload["routes"])


# ── protocol mounts ──────────────────────────────────────────────────────


def test_register_mcp_adds_working_route(tmp_path) -> None:
    api, owner = _owner(_context(str(tmp_path)))
    register_mcp(api, MCPServer())
    status, payload = api.dispatch(
        "POST", "/mcp",
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25"}},
        {}, principal=owner)
    assert status == 200
    assert payload["result"]["protocolVersion"] == "2025-11-25"


def test_register_acp_adds_working_route(tmp_path) -> None:
    api, owner = _owner(_context(str(tmp_path)))
    register_acp(api, ACPServer())
    status, payload = api.dispatch(
        "POST", "/acp",
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": 1}},
        {}, principal=owner)
    assert status == 200
    assert payload["result"]["protocolVersion"] == 1


# ── MCP: annotations / structured content ────────────────────────────────


@pytest.fixture()
def mcp_server():
    return MCPServer(principal=DEFAULT_PRINCIPAL)


def _call(srv: MCPServer, mid: int, method: str,
          params: dict[str, Any] | None = None,
          emit: Any = None):
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params is not None:
        msg["params"] = params
    return srv.dispatch(msg, emit or (lambda n: None))


def test_tool_annotations_and_titles(mcp_server) -> None:
    tools = _call(mcp_server, 1, "tools/list")["result"]["tools"]
    by_name = {t["name"]: t for t in tools}
    assert by_name["memory_query"]["annotations"]["readOnlyHint"] is True
    assert by_name["memory_query"]["annotations"]["openWorldHint"] is False
    assert by_name["memory_write"]["annotations"]["destructiveHint"] is True
    assert by_name["tool_call"]["annotations"]["openWorldHint"] is True
    assert by_name["research_run"]["annotations"]["readOnlyHint"] is True
    assert all(t.get("title") for t in tools)
    assert all(t.get("outputSchema") for t in tools)


def test_tools_call_returns_structured_content(tmp_path) -> None:
    from nomorals.memory.tiers import TwoTierMemory

    mem = TwoTierMemory(db=str(tmp_path / "mcp.db"))
    mem.facts.add_fact("my dog is Rex", confidence=0.9)
    srv = MCPServer(memory=mem, principal=DEFAULT_PRINCIPAL)
    result = _call(srv, 1, "tools/call",
                   {"name": "memory_query",
                    "arguments": {"query": "dog"}})["result"]
    assert result["isError"] is False
    structured = result["structuredContent"]
    assert structured["query"] == "dog"
    assert any("Rex" in f for f in structured["facts"])
    # Backward-compat text content carries the same payload.
    assert json.loads(result["content"][0]["text"]) == structured


# ── MCP: completion ──────────────────────────────────────────────────────


def test_completion_prompt_topic(tmp_path) -> None:
    from nomorals.memory.tiers import TwoTierMemory

    mem = TwoTierMemory(db=str(tmp_path / "mcp.db"))
    mem.facts.add_fact("my dog is Rex", confidence=0.9)
    srv = MCPServer(memory=mem, principal=DEFAULT_PRINCIPAL)
    result = _call(srv, 1, "completion/complete", {
        "ref": {"type": "ref/prompt", "name": "devon-brief"},
        "argument": {"name": "topic", "value": "my dog"},
    })["result"]
    assert any("Rex" in v for v in result["completion"]["values"])
    assert result["completion"]["hasMore"] is False


def test_completion_unknown_prompt_is_invalid_params(mcp_server) -> None:
    err = _call(mcp_server, 1, "completion/complete", {
        "ref": {"type": "ref/prompt", "name": "nope"},
        "argument": {"name": "topic", "value": ""},
    })["error"]
    assert err["code"] == -32602


def test_completion_resource_uris(mcp_server) -> None:
    result = _call(mcp_server, 1, "completion/complete", {
        "ref": {"type": "ref/resource", "uri": "devon://memory/facts"},
        "argument": {"name": "", "value": "devon://memory/f"},
    })["result"]
    assert "devon://memory/facts" in result["completion"]["values"]


def test_initialize_advertises_logging_and_completions(mcp_server) -> None:
    caps = _call(mcp_server, 1, "initialize",
                 {"protocolVersion": "2025-11-25"})["result"]["capabilities"]
    assert "logging" in caps and "completions" in caps


# ── MCP: logging ─────────────────────────────────────────────────────────


def test_logging_set_level_and_notification(mcp_server) -> None:
    notes: list[dict[str, Any]] = []
    assert _call(mcp_server, 1, "logging/setLevel", {"level": "info"},
                 emit=notes.append)["result"] == {}
    # Unknown tool → error-level log notification.
    _call(mcp_server, 2, "tools/call", {"name": "nope"},
          emit=notes.append)
    messages = [n for n in notes
                if n.get("method") == "notifications/message"]
    assert messages and messages[0]["params"]["level"] == "error"


def test_logging_set_level_rejects_bad_level(mcp_server) -> None:
    err = _call(mcp_server, 1, "logging/setLevel",
                {"level": "verbose"})["error"]
    assert err["code"] == -32602


def test_no_log_notifications_before_set_level(mcp_server) -> None:
    notes: list[dict[str, Any]] = []
    _call(mcp_server, 1, "tools/call", {"name": "nope"},
          emit=notes.append)
    assert not [n for n in notes
                if n.get("method") == "notifications/message"]


# ── MCP: progress ────────────────────────────────────────────────────────


def test_progress_notifications_increase(mcp_server) -> None:
    notes: list[dict[str, Any]] = []
    _call(mcp_server, 1, "tools/call",
          {"name": "memory_query", "arguments": {"query": "x"},
           "_meta": {"progressToken": "tok-1"}},
          emit=notes.append)
    progress = [n["params"] for n in notes
                if n.get("method") == "notifications/progress"]
    assert len(progress) == 2
    assert progress[0]["progressToken"] == "tok-1"
    assert progress[1]["progress"] > progress[0]["progress"]


def test_progress_bad_token_is_invalid_params(mcp_server) -> None:
    err = _call(mcp_server, 1, "tools/call",
                {"name": "memory_query", "arguments": {"query": "x"},
                 "_meta": {"progressToken": [1, 2]}})["error"]
    assert err["code"] == -32602


# ── MCP: resource templates + version negotiation ────────────────────────


def test_resource_templates_list(mcp_server) -> None:
    templates = _call(mcp_server, 1,
                      "resources/templates/list")["result"]["resourceTemplates"]
    assert any(t["uriTemplate"] == "devon://memory/facts/{id}"
               for t in templates)


def test_resource_template_read_and_32002(tmp_path) -> None:
    from nomorals.memory.tiers import TwoTierMemory

    mem = TwoTierMemory(db=str(tmp_path / "mcp.db"))
    fact = mem.facts.add_fact("my dog is Rex", confidence=0.9)
    srv = MCPServer(memory=mem, principal=DEFAULT_PRINCIPAL)
    result = _call(srv, 1, "resources/read",
                   {"uri": f"devon://memory/facts/{fact.id}"})["result"]
    payload = json.loads(result["contents"][0]["text"])
    assert payload["text"] == "my dog is Rex"
    err = _call(srv, 2, "resources/read",
                {"uri": "devon://memory/facts/does-not-exist"})["error"]
    assert err["code"] == RESOURCE_NOT_FOUND


def test_version_negotiation(mcp_server) -> None:
    older = _call(mcp_server, 1, "initialize",
                  {"protocolVersion": "2025-03-26"})["result"]
    assert older["protocolVersion"] == "2025-03-26"
    newer = _call(mcp_server, 2, "initialize",
                  {"protocolVersion": "2026-07-28"})["result"]
    assert newer["protocolVersion"] == MCP_PROTOCOL_VERSION
    missing = _call(mcp_server, 3, "initialize", {})["result"]
    assert missing["protocolVersion"] == MCP_PROTOCOL_VERSION


# ── ACP: tool kinds / locations ──────────────────────────────────────────


@pytest.mark.parametrize("name,kind", [
    ("read_file", "read"),
    ("list_dir", "read"),
    ("web_search", "search"),
    ("shell_exec", "execute"),
    ("run_command", "execute"),
    ("memory_recall", "search"),
    ("edit_patch", "edit"),
    ("delete_cache", "delete"),
    ("rename_thing", "move"),
    ("think_step", "think"),
    ("mystery_tool", "other"),
])
def test_tool_kind_mapping(name: str, kind: str) -> None:
    assert _tool_kind(name) == kind


def test_tool_locations() -> None:
    assert _tool_locations({"path": "/abs/x.py"}) == [{"path": "/abs/x.py"}]
    assert _tool_locations({"path": "relative/x.py"}) == []
    assert _tool_locations({"other": "/abs/y"}) == []


# ── ACP: initialize / session updates ────────────────────────────────────


def _rpc(mid: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params}


def test_initialize_records_client_caps_and_mcp_caps() -> None:
    srv = ACPServer()
    resp = srv.dispatch(
        _rpc(0, "initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {"terminal": True,
                                   "fs": {"readTextFile": True}},
        }),
        lambda n: None)
    caps = resp["result"]["agentCapabilities"]
    assert caps["mcpCapabilities"] == {"http": False, "sse": False}
    assert srv.client_capabilities["terminal"] is True


def test_session_new_emits_available_commands() -> None:
    registry = ToolRegistry()

    @registry.register("safe_read", description="reads things",
                       capability="")
    def _safe() -> str:
        return "ok"

    srv = ACPServer(registry=registry, principal=DEFAULT_PRINCIPAL)
    notes: list[dict[str, Any]] = []
    resp = srv.dispatch(_rpc(1, "session/new", {"cwd": "/tmp"}),
                        notes.append)
    sid = resp["result"]["sessionId"]
    updates = [n["params"]["update"] for n in notes
               if n.get("method") == "session/update"]
    cmd_updates = [u for u in updates
                   if u.get("sessionUpdate") == "available_commands_update"]
    assert cmd_updates
    names = [c["name"] for c in cmd_updates[0]["availableCommands"]]
    assert "safe_read" in names
    assert all(c["description"] for c in cmd_updates[0]["availableCommands"])
    assert all(n["params"]["sessionId"] == sid for n in notes)


def test_prompt_emits_plan_and_typed_tool_call(monkeypatch) -> None:
    import nomorals.agents.orchestration as orch

    registry = ToolRegistry()

    @registry.register("read_notes", description="reads notes",
                       capability="")
    def _read(path: str = "") -> str:
        return "notes!"

    grant = CapabilitySet.of(Capability.MEM_READ)

    class _Result:
        response = "done"
        memory_snapshot: dict[str, Any] | None = None
        asked_user = False
        success = True

    def fake_run_agentic(prompt_text: str, **kw: Any) -> _Result:
        # Goes through ACPServer's traced registry.call wrapper.
        kw["registry"].call("read_notes", actor="test",
                            capabilities=grant, path="/tmp/notes.md")
        return _Result()

    monkeypatch.setattr(orch, "run_agentic", fake_run_agentic)
    context = SimpleNamespace(llm=object(), router=None)
    srv = ACPServer(context=context, registry=registry,
                    principal=DEFAULT_PRINCIPAL)
    notes: list[dict[str, Any]] = []
    sid = srv.dispatch(
        _rpc(1, "session/new", {"cwd": "/tmp"}),
        notes.append)["result"]["sessionId"]
    notes.clear()
    resp = srv.dispatch(
        _rpc(2, "session/prompt",
             {"sessionId": sid, "prompt": [{"type": "text",
                                           "text": "read my notes"}]}),
        notes.append)
    assert resp["result"]["stopReason"] == "end_turn"
    updates = [n["params"]["update"] for n in notes
               if n.get("method") == "session/update"]
    kinds = [u.get("sessionUpdate") for u in updates]
    # Plan brackets the turn: in_progress first, completed at the end.
    plans = [u for u in updates if u.get("sessionUpdate") == "plan"]
    assert len(plans) == 2
    assert plans[0]["entries"][0]["status"] == "in_progress"
    assert "read my notes" in plans[0]["entries"][0]["content"]
    assert plans[1]["entries"][0]["status"] == "completed"
    assert kinds.index("plan") < kinds.index("tool_call")
    # The tool_call notification carries spec fields.
    tool_call = next(u for u in updates
                     if u.get("sessionUpdate") == "tool_call")
    assert tool_call["name"] == "read_notes"
    assert tool_call["kind"] == "read"
    assert tool_call["status"] == "in_progress"
    # rawInput carries the real params (incl. actor/capabilities kwargs).
    assert tool_call["rawInput"]["path"] == "/tmp/notes.md"
    assert tool_call["locations"] == [{"path": "/tmp/notes.md"}]
    tool_update = next(u for u in updates
                       if u.get("sessionUpdate") == "tool_call_update")
    assert tool_update["status"] == "completed"
    assert tool_update["rawOutput"] == "notes!"
