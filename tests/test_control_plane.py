"""F3 control plane: discovery, routing, governed MCP gateway,
OpenAPI→MCP, skill serving, provenance receipts.

Discovery and routing run against a SMALL fake registry (three tools),
never the full 787 — the full-registry count pin lives in
test_registry_count below and in the pin files.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import Any

from nomorals.core.policy import Capability, CapabilitySet
from nomorals.hub.control import (
    ControlError,
    ControlPlane,
    openapi_to_mcp,
)
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


def _small_registry() -> ToolRegistry:
    registry = ToolRegistry(SimpleNamespace())
    registry._tools.clear()

    @registry.register(
        "memory_recall",
        description="Recall a memory from the journal by topic",
        parameters={"query": "str — what to recall"},
    )
    def _recall(query: str = "") -> dict[str, Any]:
        return {"ok": True, "text": "recalled " + query}

    @registry.register(
        "echo_tool",
        description="Echo text back for testing",
        parameters={"text": "str — text to echo"},
    )
    def _echo(text: str = "") -> dict[str, Any]:
        return {"ok": True, "echo": text}

    @registry.register(
        "file_write",
        description="Write a file to disk",
        capability=Capability.FS_WRITE,
        parameters={"path": "str — file path"},
    )
    def _write(path: str = "") -> dict[str, Any]:
        return {"ok": True, "path": path}

    return registry


def _plane(**kwargs: Any) -> ControlPlane:
    db = kwargs.pop("db", None) or Database(":memory:")
    registry = kwargs.pop("registry", None) or _small_registry()
    return ControlPlane(registry=registry, settings=SimpleNamespace(),
                        db=db, **kwargs)


_PETSTORE_DOC = {
    "openapi": "3.0.3",
    "info": {"title": "Petstore", "version": "1.0.0"},
    "servers": [{"url": "https://pets.example.invalid/v1"}],
    "paths": {
        "/pets": {
            "get": {
                "operationId": "listPets",
                "summary": "List all pets",
                "parameters": [
                    {"name": "limit", "in": "query",
                     "schema": {"type": "integer"}},
                ],
            },
            "post": {
                "operationId": "createPet",
                "summary": "Create a pet",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "tag": {"type": "string"},
                        },
                        "required": ["name"],
                    }}},
                },
            },
        },
        "/pets/{petId}": {
            "get": {
                "summary": "Info for a pet",
                "parameters": [
                    {"name": "petId", "in": "path", "required": True,
                     "schema": {"type": "string"}},
                ],
            },
        },
    },
}


class FakeMCPConfig:
    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self.transport = "http"
        self.url = "https://fake.example.invalid"
        self.env: list[str] = []
        self.notes = "a fake server for the gateway tests"
        self.timeout_s = 0.0


class FakeMCPManager:
    """Stands in for MCPClientManager: no processes, no network."""

    def __init__(self, servers: list[FakeMCPConfig] | None = None) -> None:
        self._servers = {cfg.name: cfg for cfg in (servers or [])}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def server(self, name: str) -> FakeMCPConfig:
        if name not in self._servers:
            raise ValueError(f"unknown MCP server {name!r}")
        return self._servers[name]

    def servers(self) -> list[FakeMCPConfig]:
        return list(self._servers.values())

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "servers": {
                name: {"enabled": cfg.enabled, "transport": cfg.transport,
                       "connected": True, "tools": 1, "error": "",
                       "notes": cfg.notes}
                for name, cfg in self._servers.items()
            },
        }

    def list_tools(self, name: str) -> list[dict[str, Any]]:
        self.server(name)
        return [{"name": "ping", "description": "ping the fake server"}]

    def call_tool(self, name: str, tool: str,
                  args: dict[str, Any] | None = None) -> dict[str, Any]:
        self.server(name)
        self.calls.append((name, tool, dict(args or {})))
        return {"content": [{"type": "text", "text": "pong"}],
                "isError": False}


class FakeDelegation:
    def __init__(self) -> None:
        self.spawned: list[str] = []

    def list_runs(self, *, limit: int = 20) -> list[Any]:
        return []

    def plan_and_spawn(self, goal: str, **kwargs: Any) -> dict[str, Any]:
        self.spawned.append(goal)
        return {"ok": True, "goal": goal, "runs": []}


class ControlDiscoveryTests(unittest.TestCase):
    def test_discover_lists_live_tools_not_a_hardcoded_list(self) -> None:
        plane = _plane()
        found = plane.discover()
        tools = found["sections"]["tools"]
        self.assertTrue(tools["available"])
        self.assertEqual(tools["count"], 3)
        names = {t["name"] for t in tools["tools"]}
        self.assertEqual(names, {"memory_recall", "echo_tool", "file_write"})
        # A capability-gated caller never sees the tool it cannot call.
        gated = plane.discover(grant=CapabilitySet.none())
        gated_names = {t["name"]
                       for t in gated["sections"]["tools"]["tools"]}
        self.assertNotIn("file_write", gated_names)
        self.assertIn("memory_recall", gated_names)

    def test_discover_sections_name_what_is_missing(self) -> None:
        plane = _plane()
        found = plane.discover()
        sections = found["sections"]
        # 12 shipped MCP defaults, all disabled by default — none
        # invented, none connected.
        mcp = sections["mcp_servers"]
        self.assertTrue(mcp["available"])
        self.assertEqual(mcp["count"], 12)
        self.assertTrue(all(not s["enabled"] for s in mcp["servers"]))
        # A2A is empty by design: nothing connects by default.
        self.assertEqual(sections["a2a_agents"]["count"], 0)
        self.assertEqual(sections["skills"]["count"], 0)
        # The delegation roster is the real 7-role roster.
        roles = {r["role"] for r in sections["delegation"]["roles"]}
        self.assertEqual(
            roles, {"planner", "implementer", "reviewer", "tester",
                    "refactorer", "dep_hunter", "api_designer"})
        self.assertEqual(sections["mesh_nodes"]["count"], 0)
        self.assertEqual(sections["approvals_pending"]["count"], 0)

    def test_describe_tool_and_role(self) -> None:
        plane = _plane()
        described = plane.describe("tool", "memory_recall")
        self.assertEqual(described["kind"], "tool")
        self.assertEqual(described["description"],
                         "Recall a memory from the journal by topic")
        self.assertEqual(described["credentials"], [])
        self.assertEqual(described["approvals_pending"], 0)
        role = plane.describe("role", "planner")
        self.assertEqual(role["kind"], "role")
        self.assertTrue(role["description"])
        with self.assertRaises(ControlError):
            plane.describe("tool", "no_such_tool")
        with self.assertRaises(ControlError):
            plane.describe("bogus_kind", "x")


class ControlRoutingTests(unittest.TestCase):
    def test_route_ranks_by_name_first(self) -> None:
        plane = _plane()
        result = plane.route("memory recall")
        self.assertTrue(result["ok"])
        self.assertFalse(result["executed"])
        top = result["candidates"][0]
        self.assertEqual(top["name"], "memory_recall")
        self.assertEqual(top["kind"], "tool")
        self.assertTrue(top["granted"])

    def test_route_denied_top_candidate_names_the_capability(self) -> None:
        plane = _plane()
        result = plane.route("write a file", grant=CapabilitySet.none())
        self.assertTrue(result["ok"])  # ranking itself succeeds
        top = result["candidates"][0]
        self.assertEqual(top["name"], "file_write")
        self.assertFalse(top["granted"])
        refused = plane.route("write a file", grant=CapabilitySet.none(),
                              execute=True)
        self.assertFalse(refused["ok"])
        self.assertIn("fs.write", refused["error"])

    def test_route_executes_through_the_registry_and_receipts_it(self) -> None:
        plane = _plane()
        result = plane.route("echo hello", execute=True,
                             args={"text": "hi"})
        self.assertTrue(result["ok"])
        self.assertTrue(result["executed"])
        self.assertEqual(result["result"], {"ok": True, "echo": "hi"})
        receipts = plane.list_receipts(limit=10)
        actions = {r["action"] for r in receipts}
        self.assertIn("route.decide", actions)
        self.assertIn("route.execute", actions)
        # Arguments are stored as a digest, never raw.
        for receipt in receipts:
            self.assertNotIn("hi", receipt["args_digest"])
            self.assertEqual(len(receipt["args_digest"]), 64)

    def test_route_to_delegation_uses_the_manager(self) -> None:
        fake = FakeDelegation()
        plane = _plane(delegation_manager=fake)
        # "planner" is a delegation role; a goal naming it outranks
        # the tools (name hit weighs most).
        result = plane.route("planner plan this", execute=True,
                             grant=CapabilitySet.of(Capability.AGENT_SPAWN))
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidate"]["kind"], "delegation")
        self.assertEqual(fake.spawned, ["planner plan this"])

    def test_route_empty_goal_is_refused(self) -> None:
        plane = _plane()
        with self.assertRaises(ControlError):
            plane.route("   ")


class ControlGatewayTests(unittest.TestCase):
    def test_gateway_refusals_name_the_reason(self) -> None:
        manager = FakeMCPManager([FakeMCPConfig("fake"),
                                  FakeMCPConfig("off", enabled=False)])
        plane = _plane(mcp_manager=manager)
        with self.assertRaises(ControlError) as ctx:
            plane.gateway_call("nope", "ping", {})
        self.assertIn("unknown", str(ctx.exception))
        with self.assertRaises(ControlError) as ctx:
            plane.gateway_call("off", "ping", {})
        self.assertIn("disabled", str(ctx.exception))
        with self.assertRaises(ControlError) as ctx:
            plane.gateway_call("fake", "ping", {},
                               grant=CapabilitySet.none())
        self.assertIn("net.out", str(ctx.exception))
        with self.assertRaises(ControlError) as ctx:
            plane.gateway_call("fake", "not_listed", {})
        self.assertIn("does not list", str(ctx.exception))

    def test_gateway_call_goes_through_and_receipts(self) -> None:
        manager = FakeMCPManager([FakeMCPConfig("fake")])
        plane = _plane(mcp_manager=manager)
        result = plane.gateway_call("fake", "ping", {"x": 1})
        self.assertEqual(result["content"][0]["text"], "pong")
        self.assertEqual(manager.calls, [("fake", "ping", {"x": 1})])
        receipts = plane.list_receipts(limit=5)
        self.assertEqual(receipts[0]["action"], "mcp.call")
        self.assertEqual(receipts[0]["target"], "mcp:fake:ping")

    def test_gateway_servers_lists_live_state(self) -> None:
        manager = FakeMCPManager([FakeMCPConfig("fake")])
        plane = _plane(mcp_manager=manager)
        # gateway_servers reads the EFFECTIVE config (12 defaults) for
        # the list, so the fake manager's status only annotates names
        # it knows; the fake server itself is not in the effective
        # config and must not appear.
        servers = plane.gateway_servers()
        self.assertEqual(len(servers), 12)
        self.assertTrue(all(not s["enabled"] for s in servers))


class ControlOpenAPITests(unittest.TestCase):
    def test_openapi_to_mcp_names_and_schemas(self) -> None:
        defs, operations = openapi_to_mcp(_PETSTORE_DOC)
        by_name = {d["name"]: d for d in defs}
        self.assertEqual(set(by_name), {"listPets", "createPet",
                                        "get_pets_petId"})
        list_pets = by_name["listPets"]
        self.assertEqual(list_pets["description"], "List all pets")
        self.assertEqual(list_pets["inputSchema"]["properties"]["limit"],
                         {"type": "integer"})
        create = by_name["createPet"]
        self.assertEqual(create["inputSchema"]["required"], ["name"])
        self.assertEqual(create["inputSchema"]["properties"]["name"],
                         {"type": "string"})
        self.assertEqual(operations["listPets"]["method"], "GET")
        self.assertEqual(operations["listPets"]["server_url"],
                         "https://pets.example.invalid/v1")

    def test_openapi_malformed_docs_fail_closed(self) -> None:
        with self.assertRaises(ControlError):
            openapi_to_mcp({"paths": {}})
        with self.assertRaises(ControlError):
            openapi_to_mcp({"swagger": "2.0", "paths": {"/x": {}}})
        with self.assertRaises(ControlError):
            openapi_to_mcp({"openapi": "3.0.0", "paths": {}})

    def test_openapi_import_counts_tools(self) -> None:
        plane = _plane()
        imported = plane.openapi_import(_PETSTORE_DOC)
        self.assertTrue(imported["ok"])
        self.assertEqual(imported["count"], 3)

    def test_openapi_call_executes_over_http(self) -> None:
        seen: dict[str, Any] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                seen["path"] = self.path
                seen["auth"] = self.headers.get("Authorization")
                body = json.dumps({"pets": [{"name": "Rex"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            doc = dict(_PETSTORE_DOC)
            doc["servers"] = [{"url": f"http://127.0.0.1:{server.server_port}"}]
            plane = _plane()
            result = plane.call_openapi_tool(doc, "listPets", {"limit": 2})
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], 200)
            self.assertEqual(result["result"], {"pets": [{"name": "Rex"}]})
            self.assertEqual(seen["path"], "/pets?limit=2")
        finally:
            server.shutdown()
            server.server_close()

    def test_openapi_call_with_security_needs_named_auth(self) -> None:
        doc = {
            "openapi": "3.0.3",
            "info": {"title": "t", "version": "1"},
            "servers": [{"url": "https://x.example.invalid"}],
            "paths": {"/s": {"get": {
                "operationId": "secret",
                "security": [{"ApiKey": []}],
            }}},
            "components": {"securitySchemes": {
                "ApiKey": {"type": "apiKey", "in": "header", "name": "X-Key"}}},
        }
        plane = _plane()
        with self.assertRaises(ControlError) as ctx:
            plane.call_openapi_tool(doc, "secret", {})
        self.assertIn("auth_env", str(ctx.exception))
        with self.assertRaises(ControlError) as ctx:
            plane.call_openapi_tool(doc, "secret", {}, auth_env="NOPE_KEY_XYZ")
        self.assertIn("not set", str(ctx.exception))


class ControlReceiptTests(unittest.TestCase):
    def test_chain_verifies_and_tamper_fails_by_id(self) -> None:
        plane = _plane()
        plane.route("echo hello", execute=True, args={"text": "a"})
        plane.route("echo hello", execute=True, args={"text": "b"})
        chain = plane.verify_chain()
        self.assertTrue(chain["ok"])
        self.assertGreaterEqual(chain["checked"], 4)
        receipts = plane.list_receipts(limit=10)
        victim = receipts[0]["id"]
        plane.db.execute(
            "UPDATE control_receipts SET result = 'forged' WHERE id = ?",
            (victim,))
        chain = plane.verify_chain()
        self.assertFalse(chain["ok"])
        self.assertIn(victim, chain["broken"])

    def test_receipts_list_newest_first(self) -> None:
        plane = _plane()
        plane.route("echo one", execute=True, args={"text": "1"})
        plane.route("echo two", execute=True, args={"text": "2"})
        receipts = plane.list_receipts(limit=2)
        self.assertEqual(len(receipts), 2)
        self.assertGreaterEqual(receipts[0]["created_at"],
                                receipts[1]["created_at"])


class ControlSkillTests(unittest.TestCase):
    def test_serve_skill_from_the_live_registry(self) -> None:
        from nomorals.skills.registry import SkillRegistry

        db = Database(":memory:")
        registry = SkillRegistry(db)
        registry.install({
            "name": "greeter",
            "version": "1.0.0",
            "description": "Greets the owner",
            "tools": ["memory_recall"],
            "tags": ["hello"],
        })
        plane = ControlPlane(registry=_small_registry(),
                             settings=SimpleNamespace(), db=db,
                             skill_registry=registry)
        served = plane.serve_skill("greeter")
        self.assertTrue(served["ok"])
        self.assertEqual(served["name"], "greeter")
        self.assertEqual(served["tools"], ["memory_recall"])
        self.assertEqual(served["description"], "Greets the owner")
        with self.assertRaises(ControlError):
            plane.serve_skill("no_such_skill")

    def test_describe_skill(self) -> None:
        from nomorals.skills.registry import SkillRegistry

        db = Database(":memory:")
        registry = SkillRegistry(db)
        registry.install({
            "name": "greeter",
            "version": "1.0.0",
            "description": "Greets the owner",
            "tools": ["memory_recall"],
        })
        plane = ControlPlane(registry=_small_registry(),
                             settings=SimpleNamespace(), db=db,
                             skill_registry=registry)
        described = plane.describe("skill", "greeter")
        self.assertEqual(described["kind"], "skill")
        self.assertEqual(described["version"], "1.0.0")


class ControlChatTests(unittest.TestCase):
    def test_control_chat_summary_and_route(self) -> None:
        from nomorals.tools.control import control_chat

        context = SimpleNamespace(db=Database(":memory:"),
                                  settings=SimpleNamespace(),
                                  tools=_small_registry())
        summary = control_chat("", context)
        self.assertIn("Control plane", summary)
        self.assertIn("tools — 3", summary)
        routed = control_chat("route echo hello", context)
        self.assertIn("echo_tool", routed)
        receipts = control_chat("receipts", context)
        self.assertIn("Provenance chain: OK", receipts)

    def test_control_chat_unknown_verb(self) -> None:
        from nomorals.tools.control import control_chat

        context = SimpleNamespace(db=Database(":memory:"),
                                  settings=SimpleNamespace(),
                                  tools=_small_registry())
        self.assertIn("Unknown /control verb",
                      control_chat("frobnicate", context))


class ControlRoutesTests(unittest.TestCase):
    def _fake_server(self, context: Any) -> tuple[Any, dict[str, Any]]:
        from nomorals.api.control_routes import register_control_routes
        from nomorals.core.policy import CapabilitySet

        handlers: dict[str, Any] = {}

        class FakeServer:
            def route(self, method: str, path: str,
                      description: str = "") -> Any:
                def decorator(fn: Any) -> Any:
                    handlers[(method, path)] = fn
                    return fn
                return decorator

            def _current_principal(self) -> Any:
                return SimpleNamespace(name="owner",
                                       grant=CapabilitySet.all())

        register_control_routes(FakeServer(), context)
        return handlers

    def test_routes_agents_and_describe(self) -> None:
        context = SimpleNamespace(db=Database(":memory:"),
                                  settings=SimpleNamespace(),
                                  tools=_small_registry())
        handlers = self._fake_server(context)
        agents = handlers[("GET", "/control/agents")]({}, {})
        self.assertTrue(agents["ok"])
        self.assertEqual(agents["sections"]["tools"]["count"], 3)
        described = handlers[("GET", "/control/describe")](
            {}, {"kind": "tool", "name": "echo_tool"})
        self.assertTrue(described["ok"])
        self.assertEqual(described["name"], "echo_tool")
        missing = handlers[("GET", "/control/describe")]({}, {})
        self.assertFalse(missing["ok"])

    def test_routes_route_and_receipts(self) -> None:
        context = SimpleNamespace(db=Database(":memory:"),
                                  settings=SimpleNamespace(),
                                  tools=_small_registry())
        handlers = self._fake_server(context)
        routed = handlers[("POST", "/control/route")](
            {"goal": "echo hello", "execute": False}, {})
        self.assertTrue(routed["ok"])
        self.assertEqual(routed["candidates"][0]["name"], "echo_tool")
        receipts = handlers[("GET", "/control/receipts")]({}, {})
        self.assertTrue(receipts["ok"])
        self.assertTrue(receipts["chain"]["ok"])


class RegistryCountTests(unittest.TestCase):
    def test_registry_count_is_808_after_f3(self) -> None:
        registry = ToolRegistry(SimpleNamespace()).register_builtins()
        tools = registry.schemas()
        self.assertEqual(len(tools), 808,
                         "808 after this lane (F3) added control")


if __name__ == "__main__":
    unittest.main()
