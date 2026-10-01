"""God-tier builder tests: the four new stacks (django, nextjs,
bot-telegram, go-cli) — file layouts, generated-code sanity, manifest
run commands, and error paths.  Existing-stack coverage lives elsewhere;
this file only tests what's new.

No network, no package installs: builds run into a temp workspace and
generated Python is validated with py_compile, JSON with json.loads.
"""

from __future__ import annotations

import json
import py_compile
import shutil
import tempfile
import unittest
from pathlib import Path

from nomorals.builders import STACKS, AppBuilder
from nomorals.core.errors import ToolError

NEW_STACKS = ("django", "nextjs", "bot-telegram", "go-cli")

EXPECTED_FILES = {
    "django": {
        "manage.py", "requirements.txt", "README.md",
        "config/__init__.py", "config/settings.py", "config/urls.py",
        "config/wsgi.py",
        "items/__init__.py", "items/models.py", "items/views.py",
        "items/urls.py", "items/admin.py",
        "items/migrations/__init__.py",
        "items/templates/items/item_list.html",
        "items/templates/items/item_form.html",
    },
    "nextjs": {
        "package.json", "tsconfig.json", "next.config.mjs", ".gitignore",
        "app/layout.tsx", "app/page.tsx", "app/globals.css",
        "app/api/health/route.ts",
        "app/api/items/_store.ts", "app/api/items/route.ts",
        "app/api/items/[id]/route.ts",
        "README.md",
    },
    "bot-telegram": {
        "bot.py", "requirements.txt", "Dockerfile", ".env.example",
        "README.md",
    },
    "go-cli": {
        "go.mod", "main.go", "README.md",
    },
}

# (relative path, required marker substrings) — sanity, not a compiler
MARKERS = {
    "django": [
        ("manage.py", ["execute_from_command_line", "config.settings"]),
        ("config/settings.py", ['"items"', "sqlite3"]),
        ("items/models.py", ["class Item(models.Model)", "CharField"]),
        ("items/views.py", ["ItemListView", "ItemCreateView", "health"]),
        ("items/urls.py", ["item-list", "health/"]),
        ("items/templates/items/item_list.html",
         ["{% for item in items %}", "{% url 'item-create' %}"]),
        ("requirements.txt", ["django>="]),
    ],
    "nextjs": [
        ("app/page.tsx", ['"use client"', "/api/items", "useState"]),
        ("app/layout.tsx", ["RootLayout", "<html"]),
        ("app/api/health/route.ts", ["status", "ok"]),
        ("app/api/items/route.ts", ["export async function GET",
                                    "export async function POST"]),
        ("app/api/items/[id]/route.ts", ["PATCH", "DELETE"]),
        ("app/api/items/_store.ts", ["takeId", "Item"]),
        ("package.json", ['"next"', '"react"']),
    ],
    "bot-telegram": [
        ("bot.py", ["BOT_TOKEN", "CommandHandler", "run_polling",
                    "/start", "/help", "echo"]),
        ("requirements.txt", ["python-telegram-bot>="]),
        ("Dockerfile", ["FROM python", "bot.py"]),
    ],
    "go-cli": [
        ("main.go", ["package main", "func main()", "data.json",
                     '"add"', '"list"', '"clear"']),
        ("go.mod", ["module "]),
    ],
}

EXPECTED_PORTS = {"django": 8000, "nextjs": 3000}


class _Settings:
    def __init__(self, ws: str) -> None:
        self.workspace_dir = ws


class _Context:
    def __init__(self, ws: str) -> None:
        self.settings = _Settings(ws)


class NewStacksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="builders-godtier-")
        self.builder = AppBuilder(_Context(self.tmp))
        self._built: dict[str, dict] = {}

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── helpers ──────────────────────────────────────────────────────
    def _build(self, stack: str) -> dict:
        if stack not in self._built:
            self._built[stack] = self.builder.build({
                "name": f"godtier-{stack}",
                "stack": stack,
                "title": f"Godtier {stack}",
                "description": "god-tier scaffold test app",
                "features": ["track items", "ship fast"],
            })
        return self._built[stack]

    def _app_dir(self, stack: str) -> Path:
        return Path(self._build(stack)["dir"])

    # ── stack registry ───────────────────────────────────────────────
    def test_all_ten_stacks_listed(self) -> None:
        listed = self.builder.stacks()
        names = [s["stack"] for s in listed]
        self.assertEqual(len(names), 10)
        self.assertEqual(set(names), set(STACKS))
        for s in listed:
            self.assertIn("description", s)
        for stack in NEW_STACKS:
            self.assertIn(stack, names)

    # ── file layouts ─────────────────────────────────────────────────
    def test_expected_files_present(self) -> None:
        for stack in NEW_STACKS:
            with self.subTest(stack=stack):
                files = set(self._build(stack)["files"]) - {"manifest.json"}
                missing = EXPECTED_FILES[stack] - files
                self.assertFalse(missing, f"{stack} missing: {missing}")

    def test_manifest_run_commands_present(self) -> None:
        for stack in NEW_STACKS:
            with self.subTest(stack=stack):
                run = self._build(stack)["run"]
                self.assertIsInstance(run, str)
                self.assertTrue(run.strip(), f"{stack} has empty run command")
                # info() (manifest on disk) agrees
                info = self.builder.info(f"godtier-{stack}")
                self.assertEqual(info["run"], run)
                self.assertEqual(info["stack"], stack)

    def test_default_ports(self) -> None:
        for stack, port in EXPECTED_PORTS.items():
            with self.subTest(stack=stack):
                self._build(stack)
                info = self.builder.info(f"godtier-{stack}")
                self.assertEqual(info["port"], port)

    # ── generated code sanity ────────────────────────────────────────
    def test_marker_strings_present(self) -> None:
        for stack in NEW_STACKS:
            app_dir = self._app_dir(stack)
            for rel, markers in MARKERS[stack]:
                with self.subTest(stack=stack, file=rel):
                    text = (app_dir / rel).read_text(encoding="utf-8")
                    self.assertTrue(text.strip(), f"{rel} is empty")
                    for marker in markers:
                        self.assertIn(marker, text,
                                      f"{rel} missing marker {marker!r}")

    def test_generated_python_compiles(self) -> None:
        for stack in ("django", "bot-telegram"):
            app_dir = self._app_dir(stack)
            for rel in EXPECTED_FILES[stack]:
                if not rel.endswith(".py"):
                    continue
                with self.subTest(stack=stack, file=rel):
                    py_compile.compile(str(app_dir / rel), doraise=True)

    def test_generated_json_parses(self) -> None:
        app_dir = self._app_dir("nextjs")
        for rel in ("package.json", "tsconfig.json"):
            with self.subTest(file=rel):
                data = json.loads(
                    (app_dir / rel).read_text(encoding="utf-8"))
                self.assertIsInstance(data, dict)
        self.assertIn("next", json.loads(
            (app_dir / "package.json").read_text())["dependencies"])

    def test_js_ts_go_files_nonempty(self) -> None:
        app_dir = self._app_dir("nextjs")
        for rel in EXPECTED_FILES["nextjs"]:
            if Path(rel).suffix in (".ts", ".tsx", ".mjs"):
                with self.subTest(file=rel):
                    self.assertTrue(
                        (app_dir / rel).read_text(
                            encoding="utf-8").strip())
        go_dir = self._app_dir("go-cli")
        for rel in ("go.mod", "main.go"):
            with self.subTest(file=rel):
                self.assertTrue(
                    (go_dir / rel).read_text(encoding="utf-8").strip())

    def test_builder_validation_ok_for_new_stacks(self) -> None:
        for stack in NEW_STACKS:
            with self.subTest(stack=stack):
                v = self._build(stack)["validation"]
                self.assertTrue(v["ok"], f"{stack} validation failed: "
                                         f"{v['failed']}")
                self.assertFalse(v["failed"])

    # ── error paths ──────────────────────────────────────────────────
    def test_unknown_stack_raises_tool_error(self) -> None:
        with self.assertRaises(ToolError):
            self.builder.build({"name": "nope", "stack": "cobol"})

    def test_duplicate_build_without_overwrite_raises(self) -> None:
        self._build("django")
        with self.assertRaises(ToolError):
            self.builder.build({"name": "godtier-django", "stack": "django"})

    def test_serve_rejects_non_server_stacks(self) -> None:
        for stack in ("bot-telegram", "go-cli"):
            self._build(stack)
            with self.subTest(stack=stack), \
                    self.assertRaises(ToolError):
                self.builder.serve(f"godtier-{stack}")


if __name__ == "__main__":
    unittest.main()
