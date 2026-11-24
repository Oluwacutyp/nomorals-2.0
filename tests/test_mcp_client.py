"""MCP client tests — Devon calling external MCP servers.

A fake in-process MCP server (a small Python script run as a
subprocess over stdio) is the thing the client talks to: list tools,
call a tool, timeout, malformed JSON, server crash. A second fake
speaks Streamable HTTP from an in-process HTTP server (JSON body and
SSE stream). Registry/config/command-surface pins ride along.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.core.config import MCPServerEntry, Settings, load_settings  # noqa: E402
from nomorals.core.errors import ConfigError  # noqa: E402
from nomorals.core.policy import CapabilitySet  # noqa: E402
from nomorals.social.chat.control import (  # noqa: E402
    _HELP_GROUPS,
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
)
from nomorals.tools import mcp_client as mc  # noqa: E402
from nomorals.tools.registry import ToolRegistry  # noqa: E402

#: A fake MCP server over stdio. Modes via argv[1]:
#:   good — initialize/tools/list/call/resources/prompts/ping
#:   malformed — a garbage line on stdout after initialize
#:   crash — exit(1) at once; crash-after-init — exit after initialize
#:   slow — sleep 5s before answering tools/list (timeout test)
_FAKE_SERVER = textwrap.dedent('''
    import json, sys, time
    mode = sys.argv[1] if len(sys.argv) > 1 else "good"
    if mode == "crash":
        sys.exit(1)
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
            send({"jsonrpc": "2.0", "id": None,
                  "error": {"code": -32700, "message": "parse error"}})
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if mid is None:
            continue
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}, "resources": {},
                                 "prompts": {}},
                "serverInfo": {"name": "fake-mcp", "version": "0.1"}}})
            if mode == "malformed":
                sys.stdout.write("this is not json\\n")
                sys.stdout.flush()
            if mode == "crash-after-init":
                sys.exit(1)
            continue
        if method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
            continue
        if method == "tools/list":
            if mode == "slow":
                time.sleep(5)
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "echo", "description": "Echo back",
                 "inputSchema": {"type": "object",
                                 "properties": {"text": {"type": "string"}}}},
                {"name": "failer", "description": "Always errors",
                 "inputSchema": {"type": "object", "properties": {}}},
            ]}})
            continue
        if method == "tools/call":
            params = msg.get("params") or {}
            if params.get("name") == "echo":
                text = (params.get("arguments") or {}).get("text", "")
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": f"hello {text}"}],
                    "structuredContent": {"text": text},
                    "isError": False}})
            elif params.get("name") == "failer":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "boom"}],
                    "isError": True}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": "unknown tool"}],
                    "isError": True}})
            continue
        if method == "resources/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"resources": [
                {"uri": "fake://one", "name": "one",
                 "mimeType": "text/plain"}]}})
            continue
        if method == "resources/read":
            send({"jsonrpc": "2.0", "id": mid, "result": {"contents": [
                {"uri": "fake://one", "mimeType": "text/plain",
                 "text": "resource-body"}]}})
            continue
        if method == "prompts/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"prompts": [
                {"name": "greet", "description": "Say hi",
                 "arguments": [{"name": "who", "required": True}]}]}})
            continue
        if method == "prompts/get":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "description": "Say hi",
                "messages": [{"role": "user", "content": {
                    "type": "text", "text": "hi"}}]}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "error": {
            "code": -32601, "message": f"method not found: {method}"}})
''')


def _write_fake_server(tmp: str) -> str:
    path = os.path.join(tmp, "fake_mcp_server.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_FAKE_SERVER)
    return path


def _settings_with_fake(script: str, mode: str = "good", *,
                        name: str = "fake", enabled: bool = True,
                        timeout_s: float = 30.0,
                        max_servers: int = 8) -> Settings:
    settings = Settings()
    settings.mcp.enabled = True
    settings.mcp.max_servers = max_servers
    settings.mcp.servers[name] = MCPServerEntry(
        command=sys.executable, args=[script, mode],
        enabled=enabled, timeout_s=timeout_s)
    return settings


class MCPClientDefaultsTests(unittest.TestCase):
    def test_twelve_known_servers_all_disabled_by_default(self):
        self.assertEqual(
            set(mc.DEFAULT_MCP_SERVERS),
            {"context7", "github", "playwright", "serena", "graphify",
             "blender", "since-cutoff", "bestax", "upload-post",
             "humanpages", "sato-hub", "ceki"})
        for cfg in mc.DEFAULT_MCP_SERVERS.values():
            self.assertFalse(cfg.enabled, cfg.name)

    def test_http_servers_and_stdio_servers(self):
        for name, url in (("graphify", "https://api.graphify.com/mcp"),
                          ("sato-hub", "https://satohub.ai/api/mcp"),
                          ("ceki", "https://api.ceki.me/mcp/agent")):
            self.assertEqual(mc.DEFAULT_MCP_SERVERS[name].transport,
                             "http", name)
            self.assertEqual(mc.DEFAULT_MCP_SERVERS[name].url, url)
        for name in ("context7", "github", "playwright", "serena",
                     "blender", "since-cutoff", "bestax", "upload-post",
                     "humanpages"):
            self.assertEqual(
                mc.DEFAULT_MCP_SERVERS[name].transport, "stdio", name)

    def test_github_names_its_token_env(self):
        self.assertIn("GITHUB_PERSONAL_ACCESS_TOKEN",
                      mc.DEFAULT_MCP_SERVERS["github"].env)

    def test_master_switch_off_by_default(self):
        self.assertFalse(Settings().mcp.enabled)

    def test_config_override_enables_one_server(self):
        settings = load_settings(overrides={
            "mcp.enabled": True,
            "mcp.servers.context7.enabled": True,
        })
        self.assertTrue(settings.mcp.enabled)
        registry = mc.effective_mcp_servers(settings)
        self.assertTrue(registry["context7"].enabled)
        self.assertFalse(registry["github"].enabled)
        self.assertEqual(registry["context7"].command, "npx")

    def test_config_unknown_server_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            load_settings(overrides={
                "mcp.servers.context7.bogus": True,
            })

    def test_config_new_server_adds_to_registry(self):
        settings = load_settings(overrides={
            "mcp.servers.custom.command": "npx",
            "mcp.servers.custom.args": ["-y", "some-mcp"],
        })
        registry = mc.effective_mcp_servers(settings)
        self.assertIn("custom", registry)
        self.assertEqual(registry["custom"].args, ["-y", "some-mcp"])
        self.assertFalse(registry["custom"].enabled)


class MCPClientStdioTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.script = _write_fake_server(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _client(self, mode: str = "good", **kw: Any) -> mc.MCPClient:
        settings = _settings_with_fake(self.script, mode, **kw)
        cfg = mc.effective_mcp_servers(settings)["fake"]
        return mc.MCPClient(cfg, timeout_s=kw.get("timeout_s", 30.0))

    def test_initialize_negotiates_version(self):
        client = self._client()
        try:
            client.connect()
            self.assertTrue(client.connected)
            self.assertEqual(client.protocol_version, "2025-11-25")
            self.assertEqual(client.server_info.get("name"), "fake-mcp")
        finally:
            client.close()

    def test_list_tools_and_call_tool(self):
        client = self._client()
        try:
            tools = client.list_tools()
            self.assertEqual([t["name"] for t in tools],
                             ["echo", "failer"])
            result = client.call_tool("echo", {"text": "world"})
            self.assertFalse(result.get("isError"))
            self.assertEqual(result["structuredContent"], {"text": "world"})
            self.assertEqual(client.call_tool_text("echo", {"text": "w"}),
                             "hello w")
        finally:
            client.close()

    def test_call_tool_iserror_is_a_result_not_a_raise(self):
        client = self._client()
        try:
            result = client.call_tool("failer", {})
            self.assertTrue(result.get("isError"))
            self.assertEqual(client.call_tool_text("failer", {}), "boom")
        finally:
            client.close()

    def test_ping_resources_prompts(self):
        client = self._client()
        try:
            self.assertEqual(client.ping(), {})
            resources = client.list_resources()
            self.assertEqual(resources[0]["uri"], "fake://one")
            read = client.read_resource("fake://one")
            self.assertIn("resource-body",
                          json.dumps(read))
            prompts = client.list_prompts()
            self.assertEqual(prompts[0]["name"], "greet")
            got = client.get_prompt("greet", {"who": "Ada"})
            self.assertEqual(got["messages"][0]["role"], "user")
        finally:
            client.close()

    def test_unknown_method_surfaces_server_error(self):
        client = self._client()
        try:
            client.connect()
            with self.assertRaises(mc.MCPClientError) as ctx:
                client._request("nope/nope", {})
            self.assertIn("method not found", str(ctx.exception))
        finally:
            client.close()

    def test_call_timeout(self):
        client = self._client("slow", timeout_s=0.5)
        try:
            with self.assertRaises(mc.MCPTimeoutError):
                client.list_tools()
        finally:
            client.close()

    def test_malformed_json_from_server(self):
        client = self._client("malformed")
        try:
            with self.assertRaises(mc.MCPClientError) as ctx:
                client.list_tools()
            self.assertIn("malformed JSON", str(ctx.exception))
        finally:
            client.close()

    def test_server_crash_at_spawn(self):
        client = self._client("crash")
        with self.assertRaises(mc.MCPClientError) as ctx:
            client.connect()
        self.assertIn("exited", str(ctx.exception))
        client.close()

    def test_server_crash_after_init(self):
        client = self._client("crash-after-init")
        with self.assertRaises(mc.MCPClientError) as ctx:
            client.list_tools()
        self.assertIn("exited", str(ctx.exception))
        client.close()

    def test_call_tool_validates_arguments(self):
        client = self._client()
        try:
            with self.assertRaises(mc.MCPClientError):
                client.call_tool("", {})
            with self.assertRaises(mc.MCPClientError):
                client.call_tool("echo", ["not", "a", "dict"])  # type: ignore[arg-type]
        finally:
            client.close()


class MCPClientManagerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.script = _write_fake_server(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_refuses_when_master_off(self):
        settings = Settings()  # master off by default
        manager = mc.MCPClientManager.from_settings(settings)
        with self.assertRaises(mc.MCPClientError) as ctx:
            manager.connect("context7")
        self.assertIn("mcp client is off", str(ctx.exception))

    def test_refuses_disabled_server(self):
        settings = _settings_with_fake(self.script, enabled=False)
        manager = mc.MCPClientManager.from_settings(settings)
        with self.assertRaises(mc.MCPClientError) as ctx:
            manager.connect("fake")
        self.assertIn("disabled", str(ctx.exception))

    def test_refuses_unknown_server(self):
        settings = _settings_with_fake(self.script)
        manager = mc.MCPClientManager.from_settings(settings)
        with self.assertRaises(mc.MCPClientError) as ctx:
            manager.connect("nope")
        self.assertIn("unknown mcp server", str(ctx.exception))

    def test_connection_budget(self):
        settings = _settings_with_fake(self.script, max_servers=1)
        settings.mcp.servers["fake2"] = MCPServerEntry(
            command=sys.executable,
            args=[self.script, "good"], enabled=True)
        manager = mc.MCPClientManager.from_settings(settings)
        try:
            manager.connect("fake")
            with self.assertRaises(mc.MCPClientError) as ctx:
                manager.connect("fake2")
            self.assertIn("budget", str(ctx.exception))
        finally:
            manager.close()

    def test_status_makes_no_connections(self):
        settings = _settings_with_fake(self.script)
        manager = mc.MCPClientManager.from_settings(settings)
        status = manager.status()
        self.assertTrue(status["enabled"])
        self.assertFalse(status["servers"]["fake"]["connected"])
        self.assertIsNone(status["servers"]["fake"]["tools"])

    def test_manager_call_and_close(self):
        settings = _settings_with_fake(self.script)
        manager = mc.MCPClientManager.from_settings(settings)
        try:
            text = manager.call_tool_text("fake", "echo", {"text": "Ada"})
            self.assertEqual(text, "hello Ada")
            status = manager.status()
            self.assertTrue(status["servers"]["fake"]["connected"])
            self.assertEqual(status["servers"]["fake"]["tools"], 2)
        finally:
            manager.close()
            self.assertEqual(manager.status()["connected"], 0)


def _http_handler(mode: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self):  # noqa: N802 - stdlib name
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            msg = json.loads(body.decode("utf-8"))
            method = msg.get("method")
            mid = msg.get("id")
            if method == "initialize":
                result = {"protocolVersion": "2025-11-25",
                          "capabilities": {},
                          "serverInfo": {"name": "fake-http",
                                         "version": "0.1"}}
            elif method == "tools/list":
                result = {"tools": [{"name": "echo", "description": "Echo",
                                     "inputSchema": {"type": "object"}}]}
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text": "http-hi"}],
                          "isError": False}
            else:
                result = {}
            payload = {"jsonrpc": "2.0", "id": mid, "result": result}
            raw = json.dumps(payload).encode("utf-8")
            if mode == "sse":
                sse = (b"event: message\ndata: " + raw + b"\n\n")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(sse)))
                self.end_headers()
                self.wfile.write(sse)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

    return Handler


class MCPClientHttpTests(unittest.TestCase):
    def _serve(self, mode: str):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _http_handler(mode))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def _client_for(self, server: ThreadingHTTPServer) -> mc.MCPClient:
        url = f"http://127.0.0.1:{server.server_address[1]}/mcp"
        cfg = mc.MCPServerConfig(name="fake-http", transport="http",
                                 url=url, enabled=True)
        return mc.MCPClient(cfg)

    def test_streamable_http_json_body(self):
        server = self._serve("json")
        client = self._client_for(server)
        try:
            tools = client.list_tools()
            self.assertEqual([t["name"] for t in tools], ["echo"])
            self.assertEqual(client.call_tool_text("echo", {}), "http-hi")
        finally:
            client.close()
            server.shutdown()
            server.server_close()

    def test_streamable_http_sse_body(self):
        server = self._serve("sse")
        client = self._client_for(server)
        try:
            tools = client.list_tools()
            self.assertEqual([t["name"] for t in tools], ["echo"])
        finally:
            client.close()
            server.shutdown()
            server.server_close()

    def test_http_unreachable_is_a_client_error(self):
        cfg = mc.MCPServerConfig(
            name="gone", transport="http",
            url="http://127.0.0.1:1/mcp", enabled=True)
        client = mc.MCPClient(cfg, timeout_s=2.0)
        try:
            with self.assertRaises(mc.MCPClientError):
                client.list_tools()
        finally:
            client.close()


class MCPRegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.script = _write_fake_server(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _registry(self, settings: Settings) -> ToolRegistry:
        context = SimpleNamespace(settings=settings)
        registry = ToolRegistry(context=context)
        registry.register_builtins()
        return registry

    def test_tools_registered_with_net_out(self):
        registry = self._registry(Settings())
        for name in ("mcp_call", "mcp_list_tools", "mcp_status"):
            spec = registry.get(name)
            self.assertIsNotNone(spec, name)
            self.assertEqual(str(spec.capability), "net.out")

    def test_mcp_call_through_registry(self):
        settings = _settings_with_fake(self.script)
        registry = self._registry(settings)
        outcome = registry.call(
            "mcp_call", actor="test", capabilities=CapabilitySet.all(),
            server="fake", tool="echo", arguments={"text": "Ada"})
        self.assertTrue(outcome.ok, outcome)
        value = outcome.unwrap()
        self.assertTrue(value["ok"])
        self.assertEqual(value["text"], "hello Ada")

    def test_mcp_call_reports_tool_error_honestly(self):
        settings = _settings_with_fake(self.script)
        registry = self._registry(settings)
        outcome = registry.call(
            "mcp_call", actor="test", capabilities=CapabilitySet.all(),
            server="fake", tool="failer", arguments={})
        value = outcome.unwrap()
        self.assertFalse(value["ok"])
        self.assertEqual(value["error"], "boom")

    def test_mcp_call_refuses_disabled_server(self):
        settings = _settings_with_fake(self.script)
        registry = self._registry(settings)
        outcome = registry.call(
            "mcp_call", actor="test", capabilities=CapabilitySet.all(),
            server="context7", tool="echo", arguments={})
        value = outcome.unwrap()
        self.assertFalse(value["ok"])
        self.assertIn("disabled", value["error"])

    def test_mcp_status_tool_makes_no_connections(self):
        settings = _settings_with_fake(self.script)
        registry = self._registry(settings)
        outcome = registry.call(
            "mcp_status", actor="test", capabilities=CapabilitySet.all())
        value = outcome.unwrap()
        self.assertTrue(value["ok"])
        self.assertFalse(value["status"]["servers"]["fake"]["connected"])


class MCPCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.script = _write_fake_server(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_command_registered_with_help(self):
        self.assertIn("mcp", CONTROL_COMMANDS)
        self.assertEqual(CONTROL_COMMANDS["mcp"], (0, None))
        details = COMMAND_DETAILS["mcp"]
        for key in ("what", "usage", "example", "related"):
            self.assertTrue(str(details.get(key) or "").strip(), key)
        self.assertIn("mcp", LIST_ONELINERS)
        self.assertTrue(any("mcp" in g for _, g in LIST_GROUPS),
                        "mcp missing from LIST_GROUPS")
        self.assertTrue(any("mcp" in g for _, g in _HELP_GROUPS),
                        "mcp missing from _HELP_GROUPS")

    def test_control_status_off_by_default(self):
        text = mc.control_mcp("", Settings())
        self.assertIn("OFF", text)
        self.assertIn("context7", text)

    def test_control_list(self):
        text = mc.control_mcp("list", Settings())
        for name in ("context7", "github", "playwright", "serena",
                     "graphify", "blender", "since-cutoff", "bestax",
                     "upload-post", "humanpages", "sato-hub", "ceki"):
            self.assertIn(name, text)

    def test_control_tools_and_call(self):
        settings = _settings_with_fake(self.script)
        tools = mc.control_mcp("tools fake", settings)
        self.assertIn("echo", tools)
        self.assertIn("failer", tools)
        called = mc.control_mcp("call fake echo {\"text\": \"Ada\"}",
                                settings)
        self.assertEqual(called, "hello Ada")

    def test_control_call_bad_json(self):
        settings = _settings_with_fake(self.script)
        text = mc.control_mcp("call fake echo {bad", settings)
        self.assertIn("Bad JSON", text)

    def test_control_unknown_verb_shows_usage(self):
        text = mc.control_mcp("frobnicate", Settings())
        self.assertIn("/mcp", text)


if __name__ == "__main__":
    unittest.main()
