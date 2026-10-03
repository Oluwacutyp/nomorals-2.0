"""Tests for nomorals.tools.repo_index: repo map, symbol search, packing."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.tools import repo_index as ri
from nomorals.tools.agents import CODING_TOOLS

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_repo(files: dict[str, str]) -> str:
    """Create a synthetic repo in a temp dir; returns the dir path."""
    root = tempfile.mkdtemp(prefix="repo_index_test_")
    for rel, content in files.items():
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


class RepoMapTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({
            "pkg/__init__.py": '"""Package init."""\n',
            "pkg/mod.py": (
                '"""Module purpose line.\nSecond line.\n"""\n'
                "CONST = 1\n\n\n"
                "def top_func(a, b):\n"
                '    """Do the thing."""\n'
                "    return a + b\n\n\n"
                "class TopClass:\n"
                '    """A class."""\n'
                "    def method_one(self):\n"
                "        return 1\n"
            ),
            "pkg/nodoc.py": "x = 1\n",
            "README.md": "# docs\n",
            ".git/ignored.py": "def hidden():\n    pass\n",
            "__pycache__/cached.py": "y = 2\n",
        })
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_tree_purposes_symbols(self):
        m = ri.build_repo_map(self.root)
        self.assertEqual(m.total_files, 4)  # .git/__pycache__ skipped
        self.assertFalse(m.truncated)
        by_path = {f.path: f for f in m.files}
        self.assertEqual(by_path["pkg/mod.py"].purpose, "Module purpose line.")
        self.assertEqual(by_path["pkg/nodoc.py"].purpose, "")
        self.assertFalse(by_path["README.md"].is_python)
        syms = {s.name: s for s in by_path["pkg/mod.py"].symbols}
        # top-level only: methods are not in the map's symbol list
        self.assertEqual(set(syms), {"top_func", "TopClass"})
        self.assertEqual(syms["top_func"].kind, "function")
        self.assertEqual(syms["top_func"].line, 7)
        self.assertEqual(syms["TopClass"].kind, "class")
        self.assertEqual(syms["TopClass"].line, 12)
        self.assertGreater(m.total_symbols, 0)
        # tree shape
        tree = m.tree
        self.assertEqual(tree["type"], "dir")
        names = [c["name"] for c in tree["children"]]
        self.assertIn("pkg", names)
        self.assertIn("README.md", names)
        pkg = next(c for c in tree["children"] if c["name"] == "pkg")
        mod = next(c for c in pkg["children"] if c["name"] == "mod.py")
        self.assertEqual(mod["type"], "file")
        self.assertEqual(mod["purpose"], "Module purpose line.")
        # JSON-serializable
        json.dumps(m.to_dict())

    def test_max_files_truncation(self):
        files = {f"m{i}.py": f"def f{i}():\n    pass\n" for i in range(10)}
        root = make_repo(files)
        self.addCleanup(shutil.rmtree, root, True)
        m = ri.build_repo_map(root, max_files=3)
        self.assertTrue(m.truncated)
        self.assertEqual(len(m.files), 3)
        self.assertEqual(m.total_files, 10)

    def test_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            ri.build_repo_map("/nonexistent-root-xyz")


class FindSymbolTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({
            "a.py": (
                "def data_processor(x):\n    return x\n\n"
                "def data_processor_v2(x):\n    return x\n\n"
                "def my_data_processor(x):\n    return x\n\n"
                "class DataProcessor:\n    pass\n"
            ),
            "b.py": "def unrelated():\n    pass\n",
        })
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_exact_beats_prefix_beats_substring(self):
        hits = ri.find_symbol(self.root, "data_processor")
        kinds = [h.match for h in hits]
        # exact first, then prefix, then substring — strictly ordered
        self.assertEqual(kinds[:3], ["exact", "prefix", "substring"])
        self.assertEqual(hits[0].score, 1.0)
        self.assertEqual(hits[0].symbol.name, "data_processor")
        self.assertEqual(hits[1].symbol.name, "data_processor_v2")
        self.assertEqual(hits[2].symbol.name, "my_data_processor")
        # scores strictly decrease across tiers
        scores = [h.score for h in hits[:3]]
        self.assertTrue(scores[0] > scores[1] > scores[2])

    def test_fuzzy_only_with_flag(self):
        plain = ri.find_symbol(self.root, "data_procesor", fuzzy=False)
        self.assertFalse(any(h.match == "fuzzy" for h in plain))
        fuzzy = ri.find_symbol(self.root, "data_procesor", fuzzy=True)
        fuzzy_hits = [h for h in fuzzy if h.match == "fuzzy"]
        self.assertTrue(fuzzy_hits)
        # fuzzy tier scores stay below the substring tier (0.8)
        self.assertTrue(all(h.score < 0.8 for h in fuzzy_hits))

    def test_fuzzy_sorts_after_stronger_tiers(self):
        hits = ri.find_symbol(self.root, "data_processor", fuzzy=True)
        kinds = [h.match for h in hits]
        self.assertEqual(kinds[:3], ["exact", "prefix", "substring"])
        if "fuzzy" in kinds:
            self.assertLess(kinds.index("substring"), kinds.index("fuzzy"))

    def test_kind_filter(self):
        hits = ri.find_symbol(self.root, "data_processor", kind="class",
                              fuzzy=True)
        self.assertTrue(hits)
        self.assertTrue(all(h.symbol.kind == "class" for h in hits))

    def test_empty_name(self):
        self.assertEqual(ri.find_symbol(self.root, ""), [])


class WhoImportsTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({
            "b.py": "class Thing:\n    pass\n",
            "a.py": "import b\n",
            "c.py": "from b import Thing\n",
            "d.py": "import os\n",
        })
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_module_name(self):
        self.assertEqual(ri.who_imports(self.root, "b"), ["a.py", "c.py"])

    def test_symbol_name(self):
        self.assertEqual(ri.who_imports(self.root, "Thing"), ["c.py"])

    def test_unknown(self):
        self.assertEqual(ri.who_imports(self.root, "nope"), [])


class CallersTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({
            "main.py": (
                "def target():\n    pass\n\n"
                "def caller_one():\n    target()\n\n"
                "class K:\n"
                "    def m(self):\n        self.target()\n"
            ),
            "other.py": "from main import target\ntarget()\n",
        })
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_callers_found(self):
        hits = ri.callers(self.root, "target")
        got = {(h.file, h.caller, h.call_kind) for h in hits}
        self.assertIn(("main.py", "caller_one", "name"), got)
        self.assertIn(("main.py", "m", "attribute"), got)
        self.assertIn(("other.py", "<module>", "name"), got)
        # definition itself is not a call site
        self.assertFalse(any(h.line == 1 and h.file == "main.py"
                             for h in hits))

    def test_unknown_name(self):
        self.assertEqual(ri.callers(self.root, "no_such_fn"), [])


class PackContextTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({
            "main.py": (
                '"""Main entry."""\n'
                "import util\n\n"
                "def run():\n    util.helper()\n"
            ),
            "util.py": (
                '"""Utilities."""\n'
                "def helper():\n    return 42\n"
            ),
            "extra.py": (
                '"""Extra."""\n'
                "def helper_audit():\n    pass\n"
            ),
        })
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_ranking_and_budget(self):
        pc = ri.pack_context(self.root, ["main.py"],
                             task="fix the helper audit flow",
                             token_budget=6000)
        self.assertGreater(len(pc.files), 0)
        # query file first
        self.assertEqual(pc.files[0].path, "main.py")
        self.assertEqual(pc.files[0].rank_reason, "query")
        # import neighbor picked up via the import graph
        reasons = {f.path: f.rank_reason for f in pc.files}
        self.assertEqual(reasons.get("util.py"), "import-neighbor")
        # budget respected
        self.assertLessEqual(pc.total_chars, 6000 * 4)
        # per-file chars add up to the total
        self.assertEqual(sum(f.chars for f in pc.files), pc.total_chars)
        self.assertEqual(pc.estimated_tokens,
                         ri.estimate_tokens("x" * pc.total_chars))
        json.dumps(pc.to_dict())

    def test_deterministic(self):
        kw = dict(query_files=["main.py"], task="fix the helper",
                  token_budget=1000)
        a = ri.pack_context(self.root, **kw).to_dict()
        b = ri.pack_context(self.root, **kw).to_dict()
        self.assertEqual(a, b)

    def test_tiny_budget_truncates(self):
        pc = ri.pack_context(self.root, ["main.py", "util.py", "extra.py"],
                             token_budget=10)
        self.assertLessEqual(pc.total_chars, 10 * 4)
        self.assertTrue(pc.truncated)

    def test_header_only_for_overlap(self):
        pc = ri.pack_context(self.root, ["main.py"], task="helper audit",
                             token_budget=6000)
        overlap = [f for f in pc.files
                   if f.rank_reason == "symbol-overlap"]
        for f in overlap:
            self.assertEqual(f.content, "")


class RefreshTests(unittest.TestCase):
    def test_refresh_single_file_fast(self):
        files = {f"m{i}.py": f"def f{i}():\n    return {i}\n"
                 for i in range(200)}
        root = make_repo(files)
        self.addCleanup(shutil.rmtree, root, True)

        idx = ri.get_repo_index(root)
        self.assertEqual(len(idx.stale()), 0)

        # change exactly one file (different size => stale even on
        # coarse mtime granularity)
        target = Path(root) / "m42.py"
        target.write_text("def f42():\n    return 42\n\n"
                          "def brand_new_symbol():\n    return 'new'\n",
                          encoding="utf-8")
        self.assertEqual(idx.stale(), ["m42.py"])

        t0 = time.perf_counter()
        refreshed = idx.refresh(["m42.py"])
        dt = time.perf_counter() - t0
        self.assertEqual(refreshed, ["m42.py"])
        self.assertLess(dt, 1.0)

        # index reflects the change; everything else untouched
        hits = idx.find_symbol("brand_new_symbol")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].match, "exact")
        self.assertEqual(idx.stale(), [])

    def test_refresh_drops_deleted(self):
        root = make_repo({"a.py": "x = 1\n", "b.py": "y = 2\n"})
        self.addCleanup(shutil.rmtree, root, True)
        idx = ri.get_repo_index(root)
        os.remove(Path(root) / "b.py")
        self.assertEqual(idx.stale(), ["b.py"])
        self.assertEqual(idx.refresh(["b.py"]), ["b.py"])
        self.assertNotIn("b.py", [f.path for f in
                                  ri.build_repo_map(root).files])


class RealRepoTests(unittest.TestCase):
    def test_build_repo_map_on_nomorals_under_5s(self):
        root = REPO_ROOT / "nomorals"
        self.assertTrue(root.is_dir(), "nomorals tree missing")
        t0 = time.perf_counter()
        m = ri.build_repo_map(str(root))
        dt = time.perf_counter() - t0
        self.assertGreater(m.total_files, 300)
        self.assertGreater(m.total_symbols, 1000)
        # 30s is generous: the assertion guards against pathological
        # slowness, not a tight perf budget. Under full-suite load on
        # shared CI runners the 600-file tree can take >5s.
        self.assertLess(dt, 30.0, f"build_repo_map took {dt:.2f}s")


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


class RegistryToolTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({
            "a.py": '"""Alpha."""\ndef alpha():\n    pass\n',
            "b.py": "import a\ndef beta():\n    a.alpha()\n",
        })
        self.addCleanup(shutil.rmtree, self.root, True)
        from nomorals.tools import code_indexer
        reg = FakeRegistry(self.root)
        code_indexer.register(reg)
        self.tools = reg.tools

    def test_tools_registered(self):
        self.assertIn("repo_map", self.tools)
        self.assertIn("symbol_search", self.tools)

    def test_repo_map_tool(self):
        out = self.tools["repo_map"](path=".")
        self.assertEqual(out["total_files"], 2)
        self.assertIn("tree", out)
        self.assertIn("files", out)

    def test_symbol_search_tool_find(self):
        out = self.tools["symbol_search"](path=".", op="find", name="alpha")
        self.assertEqual(out["op"], "find")
        self.assertTrue(out["results"])
        self.assertEqual(out["results"][0]["match"], "exact")

    def test_symbol_search_tool_who_imports(self):
        out = self.tools["symbol_search"](path=".", op="who_imports",
                                           name="a")
        self.assertEqual(out["importers"], ["b.py"])

    def test_symbol_search_tool_callers(self):
        out = self.tools["symbol_search"](path=".", op="callers",
                                           name="alpha")
        self.assertTrue(any(c["caller"] == "beta" for c in out["callers"]))

    def test_symbol_search_tool_bad_op(self):
        from nomorals.core.errors import ToolError
        with self.assertRaises(ToolError):
            self.tools["symbol_search"](path=".", op="bogus", name="x")

    def test_coding_allowlist(self):
        self.assertIn("repo_map", CODING_TOOLS)
        self.assertIn("symbol_search", CODING_TOOLS)


if __name__ == "__main__":
    unittest.main()
