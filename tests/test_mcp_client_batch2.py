"""MCP client batch 2 tests — the six servers wired by the MCP Servers
lane (since-cutoff, bestax, upload-post, humanpages, sato-hub, ceki).

The defaults are the thing under test: exact spawn commands / endpoint
URLs (verified against each project's own docs 2026-11-20), all disabled
by default, gated by config, and consumable through the existing
mcp_call / mcp_list_tools / mcp_status tools and /mcp — no new brain
tool (registry stays at 787). The round-trip runs against a REAL local
stdio MCP fixture (no network, no paid services); the HTTP servers are
asserted on shape only and never dialled.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.core.config import MCPServerEntry, Settings, load_settings  # noqa: E402
from nomorals.core.policy import CapabilitySet  # noqa: E402
from nomorals.tools import mcp_client as mc  # noqa: E402
from nomorals.tools.registry import ToolRegistry  # noqa: E402

#: Minimal real stdio MCP server (same shape as the fixture in
#: tests/test_mcp_client.py, good mode only): initialize / tools/list /
#: tools/call (echo, failer) / ping.
_FAKE_SERVER = textwrap.dedent('''
    import json, sys
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
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp-batch2",
                               "version": "0.1"}}})
            continue
        if method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
            continue
        if method == "tools/list":
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
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "boom"}],
                    "isError": True}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "error": {
            "code": -32601, "message": f"method not found: {method}"}})
''')


def _write_fake_server(tmp: str) -> str:
    path = os.path.join(tmp, "fake_mcp_server.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_FAKE_SERVER)
    return path


#: The six new defaults: name -> (transport, command, args, env, url).
#: Commands / URLs verified against each project's own docs 2026-11-20.
NEW_SERVERS: dict[str, tuple[str, str, list[str], list[str], str]] = {
    "since-cutoff": ("stdio", "uvx", ["since-cutoff@latest", "mcp"], [],
                     ""),
    "bestax": ("stdio", "npx", ["-y", "bestax-mcp"], [], ""),
    "upload-post": ("stdio", "npx", ["-y", "@upload-post/mcp"],
                    ["UPLOAD_POST_API_KEY"], ""),
    "humanpages": ("stdio", "npx", ["-y", "humanpages"], [], ""),
    "sato-hub": ("http", "", [], [], "https://satohub.ai/api/mcp"),
    "ceki": ("http", "", [], [], "https://api.ceki.me/mcp/agent"),
}


class Batch2DefaultsTests(unittest.TestCase):
    def test_each_new_server_entry_is_exact(self):
        for name, (transport, command, args, env, url) in NEW_SERVERS.items():
            cfg = mc.DEFAULT_MCP_SERVERS[name]
            self.assertEqual(cfg.transport, transport, name)
            self.assertEqual(cfg.command, command, name)
            self.assertEqual(list(cfg.args), args, name)
            self.assertEqual(list(cfg.env), env, name)
            self.assertEqual(cfg.url, url, name)
            self.assertFalse(cfg.enabled, name)
            self.assertTrue(cfg.notes.strip(), name)

    def test_upload_post_names_its_token_env(self):
        self.assertIn("UPLOAD_POST_API_KEY",
                      mc.DEFAULT_MCP_SERVERS["upload-post"].env)

    def test_http_servers_carry_no_command(self):
        for name in ("sato-hub", "ceki"):
            self.assertEqual(mc.DEFAULT_MCP_SERVERS[name].command, "", name)

    def test_registry_grew_six_to_twelve(self):
        self.assertEqual(len(mc.DEFAULT_MCP_SERVERS), 12)


class Batch2GatingTests(unittest.TestCase):
    def test_each_new_server_enables_alone_via_config(self):
        for name in NEW_SERVERS:
            settings = load_settings(overrides={
                "mcp.enabled": True,
                f"mcp.servers.{name}.enabled": True,
            })
            registry = mc.effective_mcp_servers(settings)
            self.assertTrue(registry[name].enabled, name)
            for other in NEW_SERVERS:
                if other != name:
                    self.assertFalse(registry[other].enabled,
                                     f"{name} enabled, {other} leaked on")
            self.assertFalse(registry["context7"].enabled, name)

    def test_new_server_override_keeps_shipped_command(self):
        settings = load_settings(overrides={
            "mcp.enabled": True,
            "mcp.servers.since-cutoff.enabled": True,
        })
        registry = mc.effective_mcp_servers(settings)
        self.assertEqual(registry["since-cutoff"].command, "uvx")
        self.assertEqual(list(registry["since-cutoff"].args),
                         ["since-cutoff@latest", "mcp"])


class Batch2RoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.script = _write_fake_server(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _settings(self, name: str) -> Settings:
        settings = Settings()
        settings.mcp.enabled = True
        settings.mcp.servers[name] = MCPServerEntry(
            command=sys.executable, args=[self.script, "good"],
            enabled=True, timeout_s=30.0)
        return settings

    def test_stdio_round_trip_under_new_names(self):
        # The uvx family (since-cutoff) and the npx family (bestax):
        # the same client path serves both once the default is enabled.
        for name in ("since-cutoff", "bestax"):
            settings = self._settings(name)
            manager = mc.MCPClientManager.from_settings(settings)
            try:
                tools = manager.list_tools(name)
                self.assertEqual([t.get("name") for t in tools],
                                 ["echo", "failer"])
                result = manager.call_tool(name, "echo", {"text": "Ada"})
                text = "".join(
                    str(i.get("text") or "")
                    for i in result.get("content") or [])
                self.assertEqual(text, "hello Ada")
            finally:
                manager.close()

    def test_mcp_call_tool_through_registry_under_new_name(self):
        settings = self._settings("since-cutoff")
        context = SimpleNamespace(settings=settings)
        registry = ToolRegistry(context=context)
        registry.register_builtins()
        outcome = registry.call(
            "mcp_call", actor="test", capabilities=CapabilitySet.all(),
            server="since-cutoff", tool="echo", arguments={"text": "Ada"})
        value = outcome.unwrap()
        self.assertTrue(value["ok"], value)
        self.assertEqual(value["text"], "hello Ada")

    def test_control_mcp_call_under_new_name(self):
        settings = self._settings("since-cutoff")
        called = mc.control_mcp(
            "call since-cutoff echo {\"text\": \"Ada\"}", settings)
        self.assertEqual(called, "hello Ada")

    def test_refusals_name_the_new_server(self):
        # Master off.
        with self.assertRaises(mc.MCPClientError) as ctx:
            mc.MCPClientManager.from_settings(Settings()).connect(
                "since-cutoff")
        self.assertIn("off", str(ctx.exception))
        # Master on, server disabled.
        settings = Settings()
        settings.mcp.enabled = True
        with self.assertRaises(mc.MCPClientError) as ctx:
            mc.MCPClientManager.from_settings(settings).connect("ceki")
        self.assertIn("disabled", str(ctx.exception))
        # Unknown server.
        with self.assertRaises(mc.MCPClientError) as ctx:
            mc.MCPClientManager.from_settings(settings).connect("nope")
        self.assertIn("unknown", str(ctx.exception))


class Batch2HttpShapeTests(unittest.TestCase):
    def test_http_defaults_never_dialled(self):
        # Status makes no connections: 0 connected, both HTTP servers
        # listed disabled, with their URLs in the registry.
        manager = mc.MCPClientManager.from_settings(Settings())
        status = manager.status()
        self.assertEqual(status["connected"], 0)
        self.assertFalse(status["servers"]["sato-hub"]["enabled"])
        self.assertFalse(status["servers"]["ceki"]["enabled"])
        self.assertEqual(status["servers"]["sato-hub"]["transport"], "http")
        self.assertEqual(status["servers"]["ceki"]["transport"], "http")

    def test_control_list_and_status_name_new_servers(self):
        text = mc.control_mcp("list", Settings())
        for name in NEW_SERVERS:
            self.assertIn(name, text)
        status = mc.control_mcp("", Settings())
        self.assertIn("OFF", status)
        for name in ("since-cutoff", "sato-hub", "ceki"):
            self.assertIn(name, status)


class Batch2OverrideTransportTests(unittest.TestCase):
    """Regression: an ``{enabled: true}`` config override must never
    flip a shipped HTTP server to stdio. Before this fix, merge() took
    the override's defaulted "stdio" transport unconditionally, so
    enabling graphify (and the new sato-hub / ceki) from config raised
    ``mcp server '<name>': stdio needs a command`` — proven on the
    pristine base for graphify."""

    def test_http_defaults_enable_via_config_keep_transport(self):
        for name, url in (("graphify", "https://api.graphify.com/mcp"),
                          ("sato-hub", "https://satohub.ai/api/mcp"),
                          ("ceki", "https://api.ceki.me/mcp/agent")):
            settings = load_settings(overrides={
                "mcp.enabled": True,
                f"mcp.servers.{name}.enabled": True,
            })
            registry = mc.effective_mcp_servers(settings)
            self.assertEqual(registry[name].transport, "http", name)
            self.assertEqual(registry[name].url, url, name)
            self.assertTrue(registry[name].enabled, name)

    def test_merge_inherits_transport_when_override_unset(self):
        default = mc.DEFAULT_MCP_SERVERS["sato-hub"]
        merged = default.merge(
            mc.MCPServerConfig.from_dict("sato-hub", {"enabled": True}))
        self.assertEqual(merged.transport, "http")
        self.assertEqual(merged.url, "https://satohub.ai/api/mcp")
        self.assertTrue(merged.enabled)

    def test_merge_explicit_transport_flip_wins(self):
        stdio = mc.DEFAULT_MCP_SERVERS["context7"]
        flipped = stdio.merge(mc.MCPServerConfig.from_dict(
            "context7", {"transport": "http",
                         "url": "https://example.invalid/mcp"}))
        self.assertEqual(flipped.transport, "http")
        http = mc.DEFAULT_MCP_SERVERS["ceki"]
        back = http.merge(mc.MCPServerConfig.from_dict(
            "ceki", {"transport": "stdio", "command": "ceki-local"}))
        self.assertEqual(back.transport, "stdio")
        self.assertEqual(back.command, "ceki-local")

    def test_config_entry_without_transport_keeps_http(self):
        from nomorals.core.config import MCPServerEntry
        settings = Settings()
        settings.mcp.enabled = True
        settings.mcp.servers["sato-hub"] = MCPServerEntry(enabled=True)
        self.assertEqual(settings.mcp.servers["sato-hub"].transport, "")
        registry = mc.effective_mcp_servers(settings)
        self.assertEqual(registry["sato-hub"].transport, "http")
        self.assertEqual(registry["sato-hub"].url,
                         "https://satohub.ai/api/mcp")

    def test_new_server_without_transport_defaults_stdio(self):
        settings = load_settings(overrides={
            "mcp.enabled": True,
            "mcp.servers.brand-new.enabled": True,
            "mcp.servers.brand-new.command": "brand-new-mcp",
        })
        registry = mc.effective_mcp_servers(settings)
        self.assertEqual(registry["brand-new"].transport, "stdio")
        self.assertEqual(registry["brand-new"].command, "brand-new-mcp")


class Batch2ToolSurfaceTests(unittest.TestCase):
    def test_no_new_brain_tool_registry_stays_805(self):
        # 794, not 789: the trendclip port added 5 tools
        # 789, not 788: the D5 (walabel) ship's whatsapp_account_label tool landed
        # after this pin's base (parent re-bump, 2026-11-25).
        context = SimpleNamespace(settings=Settings())
        registry = ToolRegistry(context=context)
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 805)  # trendclip adds 5 tools
        for name in ("mcp_call", "mcp_list_tools", "mcp_status"):
            self.assertIsNotNone(registry.get(name), name)

    def test_mcp_call_description_names_new_servers(self):
        context = SimpleNamespace(settings=Settings())
        registry = ToolRegistry(context=context)
        registry.register_builtins()
        description = str(registry.get("mcp_call").description)
        for name in ("since-cutoff", "bestax", "Upload-Post",
                     "Human Pages", "Sato Hub", "Ceki"):
            self.assertIn(name, description)


if __name__ == "__main__":
    unittest.main()
