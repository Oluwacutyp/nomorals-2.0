"""Tests for the queryable code graph (tools/code_graph.py).

Builds the graph over a small fixture repo with known call edges
and pins the queries; proves incremental rebuild (only the changed
file re-parses); proves the rung works with the stdlib ast fallback
when tree-sitter is absent; wires the tools registry + `nm graph`.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.tools.code_graph import (
    CodeGraph,
    default_store_path,
    ensure_parser,
)

FILES = {
    "pkg/__init__.py": '"""Package init."""\n',
    "pkg/a.py": (
        '"""Alpha module."""\n\n\n'
        "def bar():\n"
        '    """Bar the helper."""\n'
        "    return 1\n\n\n"
        "def foo():\n"
        '    """Foo calls bar."""\n'
        "    bar()\n"
        "    return helper()\n\n\n"
        "class Baz:\n"
        '    """Baz the class."""\n\n'
        "    def run(self):\n"
        '        """Run calls bar."""\n'
        "        bar()\n"
    ),
    "pkg/b.py": (
        "from pkg.a import bar\n\n\n"
        "def helper():\n"
        '    """Helper calls bar."""\n'
        "    bar()\n\n\n"
        "def use_foo():\n"
        "    from pkg.a import foo\n"
        "    foo()\n"
    ),
    "pkg/c.py": (
        "class Animal:\n"
        '    """Base animal."""\n\n'
        "    def speak(self):\n"
        '        """Speak."""\n'
        "        return '...'\n\n\n"
        "class Dog(Animal):\n"
        '    """Dog inherits Animal."""\n\n'
        "    def speak(self):\n"
        "        return 'woof'\n"
    ),
    "top.py": (
        "def top_fn():\n"
        '    """Top-level function."""\n'
        "    return 0\n"
    ),
}


def make_repo(files: dict[str, str]) -> str:
    root = tempfile.mkdtemp(prefix="code_graph_test_")
    for rel, content in files.items():
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


class FakeRegistry:
    def __init__(self, workspace: str):
        self.context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=workspace))
        self.tools: dict = {}

    def register(self, name: str, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


class CodeGraphTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo(FILES)
        self.tmp = tempfile.mkdtemp(prefix="code_graph_home_")
        # Keep the store out of the real Devon home.
        self._env = mock.patch.dict(
            os.environ, {"NM_HOME": self.tmp})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(shutil.rmtree, self.root, True)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db_path = str(Path(self.tmp) / "graph.sqlite")
        self.graph = CodeGraph(self.root, db_path=self.db_path)
        self.addCleanup(self.graph.close)
        self.stats = self.graph.build(provision=False)

    def test_build_counts(self):
        self.assertEqual(self.stats.files_total, 5)
        self.assertEqual(self.stats.parsed, 5)
        self.assertEqual(self.stats.skipped, 0)
        self.assertEqual(self.stats.parser, "ast")
        self.assertGreaterEqual(self.stats.nodes, 15)
        self.assertGreaterEqual(self.stats.edges, 10)

    def test_who_calls_bar(self):
        callers = {c["caller"] for c in self.graph.who_calls("bar")}
        self.assertEqual(callers, {
            "pkg.a.foo", "pkg.a.Baz.run", "pkg.b.helper"})

    def test_calls_foo_is_exact(self):
        # foo() calls bar(); the unresolvable helper() call is dropped,
        # not guessed — calls("foo") is exactly [bar].
        callees = {c["callee"] for c in self.graph.calls("foo")}
        self.assertEqual(callees, {"pkg.a.bar"})

    def test_calls_use_foo_resolves_import(self):
        callees = {c["callee"] for c in self.graph.calls("use_foo")}
        self.assertEqual(callees, {"pkg.a.foo"})

    def test_calls_self_method(self):
        callees = {c["callee"] for c in self.graph.calls("run")}
        self.assertEqual(callees, {"pkg.a.bar"})

    def test_where_defined(self):
        defs = self.graph.where_defined("Dog")
        self.assertEqual(len(defs), 1)
        self.assertEqual(defs[0]["file"], "pkg/c.py")
        self.assertEqual(defs[0]["kind"], "class")
        bar = self.graph.where_defined("bar")
        self.assertEqual(bar[0]["qualified"], "pkg.a.bar")
        self.assertEqual(bar[0]["file"], "pkg/a.py")

    def test_owner(self):
        info = self.graph.owner("pkg/a.py")
        self.assertEqual(info["owner"], "pkg")
        self.assertEqual(info["module"], "pkg.a")
        quals = {s["qualified"] for s in info["symbols"]}
        self.assertIn("pkg.a.foo", quals)
        self.assertIn("pkg.a.Baz.run", quals)
        top = self.graph.owner("top.py")
        self.assertEqual(top["owner"], Path(self.root).name)
        self.assertEqual(top["module"], "top")

    def test_search_ranks_exact_first(self):
        hits = self.graph.search("foo")
        self.assertGreaterEqual(len(hits), 2)
        self.assertEqual(hits[0]["qualified"], "pkg.a.foo")
        self.assertIn("pkg.b.use_foo",
                      {h["qualified"] for h in hits})

    def test_search_docstring(self):
        hits = self.graph.search("helper")
        quals = {h["qualified"] for h in hits}
        self.assertIn("pkg.b.helper", quals)
        # bar's docstring is "Bar the helper." — docstring hits count.
        self.assertIn("pkg.a.bar", quals)

    def test_inherits_and_imports_edges(self):
        db = self.graph._open()
        inherits = db.query(
            "SELECT src, dst FROM graph_edges WHERE kind = 'inherits'")
        self.assertIn(("def:pkg.c.Dog", "def:pkg.c.Animal"),
                      [(r["src"], r["dst"]) for r in inherits])
        imports = db.query(
            "SELECT src, dst FROM graph_edges WHERE kind = 'imports'")
        self.assertIn(("module:pkg.b", "module:pkg.a"),
                      [(r["src"], r["dst"]) for r in imports])

    def test_incremental_rebuild_skips_unchanged(self):
        again = self.graph.build(provision=False)
        self.assertEqual(again.parsed, 0)
        self.assertEqual(again.skipped, 5)
        self.assertEqual(again.nodes, self.stats.nodes)
        self.assertEqual(again.edges, self.stats.edges)

    def test_incremental_rebuild_parses_only_changed(self):
        (Path(self.root) / "pkg" / "b.py").write_text(
            "from pkg.a import bar\n\n\n"
            "def helper():\n"
            "    bar()\n\n\n"
            "def brand_new():\n"
            '    """Brand new."""\n'
            "    bar()\n",
            encoding="utf-8")
        stats = self.graph.build(provision=False)
        self.assertEqual(stats.parsed, 1)
        self.assertEqual(stats.skipped, 4)
        self.assertEqual(
            self.graph.where_defined("brand_new")[0]["file"], "pkg/b.py")
        # Old edges survive the incremental rebuild.
        self.assertEqual(
            {c["caller"] for c in self.graph.who_calls("bar")},
            {"pkg.a.foo", "pkg.a.Baz.run", "pkg.b.helper",
             "pkg.b.brand_new"})

    def test_deleted_file_drops_out(self):
        (Path(self.root) / "top.py").unlink()
        stats = self.graph.build(provision=False)
        self.assertEqual(stats.removed, 1)
        self.assertEqual(self.graph.where_defined("top_fn"), [])

    def test_unparseable_file_never_fails_build(self):
        (Path(self.root) / "broken.py").write_text(
            "def oops(:\n", encoding="utf-8")
        stats = self.graph.build(provision=False)
        self.assertEqual(stats.parsed, 1)
        self.assertEqual(self.graph.where_defined("oops"), [])
        # Second build skips it (fingerprint recorded) — no re-parse.
        again = self.graph.build(provision=False)
        self.assertEqual(again.parsed, 0)

    def test_status(self):
        st = self.graph.status()
        self.assertEqual(st["files"], 5)
        self.assertEqual(st["nodes"], self.stats.nodes)
        self.assertEqual(st["edges"], self.stats.edges)
        self.assertEqual(st["parser"], "ast")
        self.assertEqual(st["store"], self.db_path)

    def test_store_path_is_per_root(self):
        other = default_store_path("/some/other/root")
        self.assertNotEqual(other, default_store_path(self.root))
        self.assertTrue(str(other).endswith(".sqlite"))


class ParserFallbackTests(unittest.TestCase):
    def test_ensure_parser_falls_back_without_provision(self):
        parser, detail = ensure_parser(provision=False)
        if importlib.util.find_spec("tree_sitter") is None:
            self.assertEqual(parser, "ast")
            self.assertIn("ast", detail)
        else:
            self.assertEqual(parser, "tree-sitter")

    def test_graph_builds_with_ast_when_tree_sitter_absent(self):
        if importlib.util.find_spec("tree_sitter") is not None:
            self.skipTest("tree-sitter is installed here")
        root = make_repo({"a.py": "def f():\n    g()\n\n\ndef g():\n    pass\n"})
        tmp = tempfile.mkdtemp(prefix="code_graph_fb_")
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(shutil.rmtree, tmp, True)
        graph = CodeGraph(root, db_path=str(Path(tmp) / "g.sqlite"))
        self.addCleanup(graph.close)
        stats = graph.build(provision=False)
        self.assertEqual(stats.parser, "ast")
        self.assertEqual({c["caller"] for c in graph.who_calls("g")},
                         {"a.f"})

    @unittest.skipUnless(
        importlib.util.find_spec("tree_sitter") is not None
        and importlib.util.find_spec("tree_sitter_python") is not None,
        "tree-sitter not installed")
    def test_tree_sitter_parse_matches_ast_shape(self):
        from nomorals.tools.code_graph import _parse_ast, _parse_tree_sitter
        src = (Path(__file__).parent.parent / "nomorals" / "tools"
               / "code_graph.py").read_bytes()
        ast_parse = _parse_ast("nomorals/tools/code_graph.py", src)
        ts_parse = _parse_tree_sitter("nomorals/tools/code_graph.py",
                                      src)
        self.assertIsNotNone(ast_parse)
        self.assertIsNotNone(ts_parse)
        assert ast_parse is not None and ts_parse is not None
        ast_syms = {s.qualified for s in ast_parse.symbols}
        ts_syms = {s.qualified for s in ts_parse.symbols}
        # Same top-level symbols from both parsers (subset: the
        # tree-sitter walk covers functions/classes; nested defs may
        # differ in edge cases, so top-level equality is the pin).
        ast_top = {q for q in ast_syms if q.count(".") <= 4}
        ts_top = {q for q in ts_syms if q.count(".") <= 4}
        self.assertEqual(ast_top, ts_top)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo(FILES)
        self.tmp = tempfile.mkdtemp(prefix="code_graph_home_")
        self._env = mock.patch.dict(
            os.environ, {"NM_HOME": self.tmp})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(shutil.rmtree, self.root, True)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        from nomorals.tools import code_graph
        reg = FakeRegistry(self.root)
        code_graph.register(reg)
        self.tools = reg.tools

    def test_tools_registered(self):
        self.assertIn("graph_build", self.tools)
        self.assertIn("graph_query", self.tools)
        self.assertIn("graph_status", self.tools)

    def test_graph_build_tool(self):
        out = self.tools["graph_build"](path=".")
        self.assertEqual(out["files_total"], 5)
        self.assertEqual(out["parser"], "ast")

    def test_graph_query_tool(self):
        self.tools["graph_build"](path=".")
        out = self.tools["graph_query"](
            path=".", op="who_calls", name="bar")
        self.assertEqual(
            {c["caller"] for c in out["callers"]},
            {"pkg.a.foo", "pkg.a.Baz.run", "pkg.b.helper"})
        from nomorals.core.errors import ToolError
        with self.assertRaises(ToolError):
            self.tools["graph_query"](path=".", op="nope", name="bar")

    def test_graph_status_tool(self):
        self.tools["graph_build"](path=".")
        out = self.tools["graph_status"](path=".")
        self.assertEqual(out["files"], 5)
        self.assertGreater(out["nodes"], 0)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo(FILES)
        self.tmp = tempfile.mkdtemp(prefix="code_graph_home_")
        self._env = mock.patch.dict(
            os.environ, {"NM_HOME": self.tmp})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(shutil.rmtree, self.root, True)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        from nomorals.cmdline.commands.graph import _cmd_graph
        self._cmd_graph = _cmd_graph

    def _args(self, action: str, **kw: object) -> argparse.Namespace:
        base = {"graph_action": action, "root": self.root,
                "json": False, "limit": 20, "force": False,
                "name": "", "query": ""}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_cli_build_and_status(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = self._cmd_graph(self._args("build"), None)
        self.assertEqual(rc, 0)
        self.assertIn("graph built", out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            rc = self._cmd_graph(self._args("status"), None)
        self.assertEqual(rc, 0)
        self.assertIn("nodes:", out.getvalue())

    def test_cli_who_calls(self):
        with redirect_stdout(io.StringIO()):
            self._cmd_graph(self._args("build"), None)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = self._cmd_graph(
                self._args("who-calls", name="bar"), None)
        self.assertEqual(rc, 0)
        self.assertIn("pkg.a.foo", out.getvalue())
        self.assertIn("pkg.b.helper", out.getvalue())

    def test_cli_search_json(self):
        with redirect_stdout(io.StringIO()):
            self._cmd_graph(self._args("build"), None)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = self._cmd_graph(
                self._args("search", query="foo", json=True), None)
        self.assertEqual(rc, 0)
        import json
        payload = json.loads(out.getvalue())
        self.assertIn("pkg.a.foo",
                      {r["qualified"] for r in payload["results"]})


if __name__ == "__main__":
    unittest.main()
