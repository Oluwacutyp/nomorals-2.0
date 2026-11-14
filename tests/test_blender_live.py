"""Blender live-session organ + Blender Skills port (2026-11-14).

Pins: the live session talks to a running Blender through the MCP
for Blender server (a fake stdio MCP server stands in for it — list
tools, run code, screenshot, scene summary, addon status), the
iterate loop is bounded and stops early when the goal is met, the
no-Blender path says so plainly, and the 94 vendored skills are
present with the router registered in library/INDEX.md.
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.core.config import MCPServerEntry, Settings  # noqa: E402
from nomorals.media import blender_live as bl  # noqa: E402
from nomorals.skills.filebook import parse_skill_text, seed_library_dir  # noqa: E402
from nomorals.tools import mcp_client as mc  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "nomorals" / "skills" / "library"

#: A 1x1 transparent PNG, base64 — the fake `look` screenshot.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAw"
    "MCAO+ip1sAAAAASUVORK5CYII="
)

#: A fake MCP for Blender server over stdio. Modes via argv[1]:
#:   good — the four live tools answer
#:   crash — exit(1) at once (no Blender, no server)
#:   noimage — `look` returns text only, no image item
_FAKE_BLENDER_SERVER = textwrap.dedent('''\
    import json, sys
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
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if mid is None:
            continue
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp-for-blender", "version": "1.0"}}})
            continue
        if method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
            continue
        if method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "execute_blender_code", "description": "Run Python",
                 "inputSchema": {"type": "object",
                                 "properties": {"code": {"type": "string"}}}},
                {"name": "look", "description": "Screenshot",
                 "inputSchema": {"type": "object",
                                 "properties": {"mode": {"type": "string"}}}},
                {"name": "get_scene_info", "description": "Scene summary",
                 "inputSchema": {"type": "object",
                                 "properties": {"limit": {"type": "integer"}}}},
                {"name": "get_addon_status", "description": "Addon status",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "generate_3d", "description": "Generate a model",
                 "inputSchema": {"type": "object", "properties": {}}},
            ]}})
            continue
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "execute_blender_code":
                code = args.get("code", "")
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": f"ran: {code[:60]}"}],
                    "isError": False}})
            elif name == "look":
                content = [{"type": "text",
                            "text": "Angles: front, back, left, right, top."}]
                if mode != "noimage":
                    content.append({"type": "image",
                                    "data": "__TINY_PNG_B64__",
                                    "mimeType": "image/png"})
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": content, "isError": False}})
            elif name == "get_scene_info":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": "Scene: 3 objects\\nCube at (0, 0, 0), size 2m\\nCamera at (7, -7, 5)\\nLight Sun, power 3"}],
                    "isError": False}})
            elif name == "get_addon_status":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": "Blender 4.2.3, addon 1.0, libraries on"}],
                    "isError": False}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "unknown tool"}],
                    "isError": True}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "error": {
            "code": -32601, "message": f"method not found: {method}"}})
'''.replace("__TINY_PNG_B64__", _TINY_PNG_B64))


def _write_fake_server(tmp: str) -> str:
    path = os.path.join(tmp, "fake_blender_mcp.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_FAKE_BLENDER_SERVER)
    return path


def _settings_with_fake(script: str, mode: str = "good", *,
                        name: str = "blender") -> Settings:
    settings = Settings()
    settings.mcp.enabled = True
    settings.mcp.servers[name] = MCPServerEntry(
        command=sys.executable, args=[script, mode],
        enabled=True, timeout_s=30.0)
    return settings


def _session(mode: str = "good") -> tuple[bl.BlenderLiveSession, str]:
    tmp = tempfile.mkdtemp(prefix="blender-live-test-")
    script = _write_fake_server(tmp)
    manager = mc.MCPClientManager.from_settings(
        _settings_with_fake(script, mode))
    return bl.BlenderLiveSession(manager), tmp


class BlenderRegistryTests(unittest.TestCase):
    def test_blender_is_a_known_server(self):
        self.assertIn("blender", mc.DEFAULT_MCP_SERVERS)
        cfg = mc.DEFAULT_MCP_SERVERS["blender"]
        self.assertEqual(cfg.transport, "stdio")
        self.assertEqual(cfg.command, "uvx")
        self.assertEqual(cfg.args, ["mcp-for-blender"])
        self.assertIn("add-on", cfg.notes)

    def test_blender_tool_names(self):
        for tool in ("execute_blender_code", "look", "get_scene_info",
                     "get_addon_status"):
            self.assertIn(tool, bl.BLENDER_MCP_TOOLS)
        self.assertEqual(len(bl.BLENDER_MCP_TOOLS), 10)


class BlenderLiveSessionTests(unittest.TestCase):
    def test_connect_lists_the_live_tools(self):
        session, _ = _session()
        try:
            result = session.connect()
            self.assertTrue(result["ok"], result)
            self.assertIn("execute_blender_code", result["tools"])
            self.assertIn("look", result["tools"])
            self.assertIn("get_scene_info", result["tools"])
            self.assertIn("get_addon_status", result["tools"])
            self.assertEqual(result["server"].get("name"), "mcp-for-blender")
        finally:
            session.close()

    def test_status_scene_run_and_screenshot(self):
        session, _ = _session()
        try:
            status = session.status()
            self.assertTrue(status["ok"], status)
            self.assertIn("4.2.3", status["text"])
            summary = session.scene_summary()
            self.assertTrue(summary["ok"], summary)
            self.assertIn("Cube", summary["text"])
            run = session.run_code("bpy.ops.mesh.primitive_cube_add()")
            self.assertTrue(run["ok"], run)
            self.assertIn("primitive_cube_add", run["text"])
            shot = session.screenshot()
            self.assertTrue(shot["ok"], shot)
            self.assertIn("Angles", shot["text"])
            raw = base64.b64decode(shot["image_b64"])
            self.assertTrue(raw.startswith(b"\x89PNG"))
            self.assertEqual(shot["mime"], "image/png")
        finally:
            session.close()

    def test_screenshot_without_image_is_still_ok(self):
        session, _ = _session("noimage")
        try:
            shot = session.screenshot()
            self.assertTrue(shot["ok"], shot)
            self.assertNotIn("image_b64", shot)
        finally:
            session.close()

    def test_run_code_rejects_empty_code(self):
        session, _ = _session()
        try:
            result = session.run_code("   ")
            self.assertFalse(result["ok"])
        finally:
            session.close()

    def test_no_blender_found_says_so_plainly(self):
        session, _ = _session("crash")
        try:
            result = session.connect()
            self.assertFalse(result["ok"])
            self.assertIn("No live Blender found", result["error"])
            self.assertIn("install-addon", result["error"])
        finally:
            session.close()

    def test_calls_before_connect_fail_plainly(self):
        session, _ = _session("crash")
        try:
            with self.assertRaises(mc.MCPClientError) as ctx:
                session.scene_summary()
            self.assertIn("No live Blender found", str(ctx.exception))
        finally:
            session.close()

    def test_iterate_stops_early_when_the_goal_is_met(self):
        session, _ = _session()
        try:
            calls: list[str] = []

            def code_fn(goal: str, feedback: str, target: Any) -> str:
                calls.append(goal)
                return "bpy.ops.mesh.primitive_cube_add()"

            result = session.iterate("a cube", code_fn, max_iterations=5)
            self.assertTrue(result["met"])
            self.assertEqual(len(result["passes"]), 1)
            self.assertGreaterEqual(result["score"], 0.8)
            self.assertEqual(calls, ["a cube"])
            self.assertIn("Cube", result["scene_summary"])
        finally:
            session.close()

    def test_iterate_is_bounded_and_reports_when_not_met(self):
        session, _ = _session()
        try:
            def code_fn(goal: str, feedback: str, target: Any) -> str:
                return "bpy.ops.mesh.primitive_cube_add()"

            result = session.iterate("a dragon", code_fn, max_iterations=2)
            self.assertFalse(result["met"])
            self.assertEqual(len(result["passes"]), 2)
            self.assertEqual(result["score"], 0.0)
            self.assertIn("screenshot", result)
            self.assertTrue(result["screenshot"]["ok"])
        finally:
            session.close()

    def test_iterate_uses_a_custom_scorer(self):
        session, _ = _session()
        try:
            def code_fn(goal: str, feedback: str, target: Any) -> str:
                return "bpy.ops.mesh.primitive_cube_add()"

            def score_fn(summary: str, shot: dict[str, Any],
                         target: Any) -> tuple[float, str]:
                return 0.9, "custom scorer says good"

            result = session.iterate("anything", code_fn,
                                     max_iterations=3, score_fn=score_fn)
            self.assertTrue(result["met"])
            self.assertEqual(len(result["passes"]), 1)
        finally:
            session.close()


class BlenderSkillsPortTests(unittest.TestCase):
    def test_router_parses_with_valid_frontmatter(self):
        path = LIB / "blender-skills.md"
        self.assertTrue(path.is_file())
        doc = parse_skill_text(path.read_text(encoding="utf-8"))
        self.assertEqual(doc.id, "blender-skills")
        self.assertEqual(doc.layer, "media")
        self.assertEqual(doc.version, 1)

    def test_router_is_registered_in_index(self):
        ids = set()
        for line in (LIB / "INDEX.md").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("- `"):
                ids.add(line[3:].split("`", 1)[0])
        self.assertIn("blender-skills", ids)

    def test_94_skills_vendored(self):
        skills_dir = LIB / "blender-skills" / "skills"
        found = sorted(p.parent.name for p in skills_dir.rglob("SKILL.md"))
        self.assertEqual(len(found), 94, found[:5])
        for name in ("animation", "blender-modeler", "rigging",
                     "materials", "export-pipeline", "godot-export"):
            self.assertIn(name, found)

    def test_vendored_skills_are_plain_markdown_with_frontmatter(self):
        skills_dir = LIB / "blender-skills" / "skills"
        for skill_file in sorted(skills_dir.rglob("SKILL.md")):
            text = skill_file.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"), skill_file)
            self.assertIn("\nname: ", text, skill_file)

    def test_license_and_version_pin(self):
        self.assertTrue((LIB / "blender-skills" / "LICENSE").is_file())
        license_text = (LIB / "blender-skills" / "LICENSE").read_text(
            encoding="utf-8")
        self.assertIn("MIT License", license_text)
        router = (LIB / "blender-skills.md").read_text(encoding="utf-8")
        self.assertIn("v1.3.0", router)
        self.assertIn("94 skills", router)

    def test_router_names_the_addon_bridge_and_the_organ(self):
        router = (LIB / "blender-skills.md").read_text(encoding="utf-8")
        self.assertIn("install-addon", router)
        self.assertIn("blender_live.py", router)
        self.assertIn("NEVER starts Blender", router)

    def test_seed_library_dir_is_the_checked_in_library(self):
        self.assertEqual(seed_library_dir(), LIB)


if __name__ == "__main__":
    unittest.main()
