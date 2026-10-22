"""Pass #62: import-graph test selection + optional full-suite final gate.

The defect: ``pytest_runner`` selected verifying tests by filename
stem — a test was picked because its name resembled the changed file,
not because it exercises the changed code.  Renamed modules, indirect
coverage and behavior-named tests were invisible; same-stem tests that
import nothing relevant burned the verify budget.

These tests pin the new contract on fixture repos with known import
shapes:

- graph selection picks the tests whose imports cover a changed module
  (direct imports, transitive chains, from-imports, relative imports);
- stem picks are NEVER dropped (the graph only extends selection);
- every selection records WHY each test was picked;
- the graph cache invalidates when a file's imports change;
- a graph build failure degrades to stem selection, noted in results;
- pass #58's zero-selection honesty is intact on foreground AND
  background paths with the graph active;
- the optional full-suite final gate (default OFF) runs once after a
  green selection run and reports a DISTINCT verdict — a gate failure
  on unrelated pre-existing breakage is attributable and never
  rewrites the selection verdict, in the runner or in CodingAgent.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import bg_tests
from nomorals.agents.coding import CodingAgent
from nomorals.tools import import_graph, pytest_runner


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class GraphFixture:
    """Known import shapes:

    pkg/core.py      <- imported by pkg/api.py, pkg/sub/thing.py (via
                        a relative import) and tests/test_zebra.py
    pkg/api.py       <- imported by tests/test_alpha.py (transitive
                        coverage of core)
    pkg/renamed.py   <- imported by tests/test_gadget.py via
                        ``from pkg import renamed`` (behavior-named
                        test file: the stem heuristic cannot find it)
    pkg/plain.py     <- imported by tests/test_core.py — a SAME-STEM
                        test for core.py that covers plain.py instead
    tests/test_nothing.py imports nothing relevant.
    """

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass62_graph_"))
        _write(self.root, "pkg/__init__.py", "")
        _write(self.root, "pkg/core.py", "CORE = 1\n")
        _write(self.root, "pkg/api.py",
               "import pkg.core\n\nAPI = pkg.core.CORE + 1\n")
        _write(self.root, "pkg/renamed.py", "VALUE = 7\n")
        _write(self.root, "pkg/plain.py", "PLAIN = 3\n")
        _write(self.root, "pkg/sub/__init__.py", "")
        _write(self.root, "pkg/sub/thing.py",
               "from ..core import CORE\n\nVALUE = CORE\n")
        _write(self.root, "pkg/lazy.py",
               "def load():\n    import pkg.core\n"
               "    return pkg.core.CORE\n")
        _write(self.root, "tests/__init__.py", "")
        _write(self.root, "tests/test_zebra.py",
               "import pkg.core\n\n\ndef test_core_value():\n"
               "    assert pkg.core.CORE == 1\n")
        _write(self.root, "tests/test_alpha.py",
               "import pkg.api\n\n\ndef test_api_value():\n"
               "    assert pkg.api.API == 2\n")
        _write(self.root, "tests/test_gadget.py",
               "from pkg import renamed\n\n\ndef test_gadget_value():\n"
               "    assert renamed.VALUE == 7\n")
        _write(self.root, "tests/test_core.py",
               "import pkg.plain\n\n\ndef test_plain_value():\n"
               "    assert pkg.plain.PLAIN == 3\n")
        _write(self.root, "tests/test_rel.py",
               "from pkg.sub import thing\n\n\ndef test_thing_value():\n"
               "    assert thing.VALUE == 1\n")
        _write(self.root, "tests/test_lazy.py",
               "import pkg.lazy\n\n\ndef test_lazy_value():\n"
               "    assert pkg.lazy.load() == 1\n")
        _write(self.root, "tests/test_nothing.py",
               "def test_ok():\n    assert True\n")

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class ImportGraphSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fx = GraphFixture()
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(import_graph.clear_cache)

    def _select(self, *changed):
        return pytest_runner.select_changed_tests_detailed(
            str(self.fx.root), list(changed))

    def test_direct_import_selected_by_graph_not_stem(self):
        _selected, reasons, meta = self._select("pkg/core.py")
        self.assertEqual(meta["graph_status"], "ok")
        # "core" is not in "test_zebra": only the graph can pick it.
        self.assertIn("graph-direct", reasons["tests/test_zebra.py"])

    def test_transitive_chain_selected(self):
        _selected, reasons, _meta = self._select("pkg/core.py")
        # test_alpha imports pkg.api, which imports pkg.core.
        self.assertIn("graph-transitive", reasons["tests/test_alpha.py"])
        # test_rel reaches core through pkg.sub.thing's relative import.
        self.assertIn("graph-transitive", reasons["tests/test_rel.py"])
        # test_lazy reaches core through a FUNCTION-LEVEL lazy import
        # in pkg/lazy.py — statement-level traversal must still see it.
        self.assertIn("graph-transitive", reasons["tests/test_lazy.py"])

    def test_from_import_of_renamed_module_selected(self):
        _selected, reasons, _meta = self._select("pkg/renamed.py")
        self.assertIn("graph-direct", reasons["tests/test_gadget.py"])

    def test_stem_pick_never_dropped(self):
        selected, reasons, _meta = self._select("pkg/core.py")
        # test_core.py imports pkg.plain, NOT pkg.core — the graph does
        # not cover it, but the stem heuristic picks it and the union
        # must keep it.
        self.assertIn("tests/test_core.py", selected)
        self.assertEqual(reasons["tests/test_core.py"], "stem")

    def test_irrelevant_test_not_selected(self):
        selected, _reasons, _meta = self._select("pkg/core.py")
        self.assertNotIn("tests/test_nothing.py", selected)
        self.assertNotIn("tests/test_gadget.py", selected)

    def test_runner_result_records_reasons_and_executes_graph_picks(self):
        res = pytest_runner.run_tests(
            repo=str(self.fx.root), changed_files=["pkg/core.py"],
            timeout=120)
        self.assertTrue(res["ok"], res)
        # exactly the 5 covering tests ran: zebra, alpha, rel, lazy
        # (graph) + test_core (stem).  test_nothing/test_gadget did
        # not run.
        self.assertEqual(res["passed"], 5)
        self.assertEqual(res["ran"], 5)
        reasons = res["selection_reasons"]
        self.assertIn("graph-direct", reasons["tests/test_zebra.py"])
        self.assertIn("graph-transitive", reasons["tests/test_alpha.py"])
        self.assertEqual(res["selection"]["graph_status"], "ok")
        self.assertEqual(
            res["selection"]["changed_files"], ["pkg/core.py"])

    def test_cache_invalidates_when_imports_change(self):
        _s, reasons, _m = self._select("pkg/core.py")
        self.assertIn("tests/test_alpha.py", reasons)
        # pkg.api stops importing pkg.core: the transitive edge dies.
        _write(self.fx.root, "pkg/api.py", "API = 2\n")
        _s2, reasons2, _m2 = self._select("pkg/core.py")
        self.assertNotIn("tests/test_alpha.py", reasons2)
        self.assertIn("tests/test_zebra.py", reasons2)  # direct stands

    def test_unparsable_file_contributes_no_edges_but_never_fails(self):
        # a broken file whose import edge is LOST to the syntax error:
        # recorded in unparsed, and the build does not fail.
        _write(self.fx.root, "pkg/broken.py",
               "import pkg.core\ndef broken(:\n")
        graph = import_graph.build_import_graph(self.fx.root)
        self.assertIn("pkg/broken.py", graph.unparsed)
        self.assertNotIn("pkg/broken.py", graph.file_edges)

    def test_module_names(self):
        root = self.fx.root
        self.assertEqual(
            import_graph.module_name_for(root, root / "pkg/__init__.py"),
            "pkg")
        self.assertEqual(
            import_graph.module_name_for(
                root, root / "pkg/sub/thing.py"),
            "pkg.sub.thing")
        self.assertEqual(
            import_graph.module_name_for(
                root, root / "tests/test_zebra.py"),
            "tests.test_zebra")


@unittest.skipUnless(shutil.which("git"), "git not available")
class GitDerivedSelectionTests(unittest.TestCase):
    """No explicit changed_files: the git-derived changed set gets the
    same graph union."""

    def setUp(self):
        self.fx = GraphFixture()
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(import_graph.clear_cache)
        _git("init", "-q", cwd=self.fx.root)
        _git("config", "user.email", "t@example.com", cwd=self.fx.root)
        _git("config", "user.name", "T", cwd=self.fx.root)
        _git("add", ".", cwd=self.fx.root)
        _git("commit", "-qm", "init", cwd=self.fx.root)

    def test_git_selection_unions_graph(self):
        _write(self.fx.root, "pkg/core.py", "CORE = 1  # touched\n")
        selected = pytest_runner.select_changed_tests(str(self.fx.root))
        self.assertIn("tests/test_zebra.py", selected)   # graph-direct
        self.assertIn("tests/test_alpha.py", selected)   # graph-transitive
        self.assertIn("tests/test_core.py", selected)    # stem
        self.assertNotIn("tests/test_nothing.py", selected)


class GraphDegradationTests(unittest.TestCase):
    def setUp(self):
        self.fx = GraphFixture()
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(import_graph.clear_cache)

    def test_graph_failure_degrades_to_stem_with_note(self):
        with mock.patch.object(import_graph, "build_import_graph",
                               side_effect=RuntimeError("boom")):
            selected, reasons, meta = \
                pytest_runner.select_changed_tests_detailed(
                    str(self.fx.root), ["pkg/core.py"])
            self.assertEqual(meta["graph_status"], "degraded")
            self.assertIn("boom", meta["graph_error"])
            # stem selection still works, graph picks are absent.
            self.assertEqual(selected, ["tests/test_core.py"])
            self.assertEqual(reasons["tests/test_core.py"], "stem")
            res = pytest_runner.run_tests(
                repo=str(self.fx.root), changed_files=["pkg/core.py"],
                timeout=120)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["selection"]["graph_status"], "degraded")
            self.assertIn("degraded", res["note"])


class ZeroSelectionHonestyTests(unittest.TestCase):
    """Pass #58's contract with the graph active: zero selection is
    still an honest nothing-ran, on foreground and background paths."""

    def setUp(self):
        self.fx = GraphFixture()
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(import_graph.clear_cache)

    def test_foreground_nothing_ran_still_honest(self):
        _write(self.fx.root, "lonely.py", "L = 1\n")
        res = pytest_runner.run_tests(
            repo=str(self.fx.root), changed_files=["lonely.py"],
            timeout=60)
        self.assertTrue(res["ok"])  # skip semantics unchanged
        self.assertEqual(res["ran"], 0)
        self.assertTrue(res["nothing_ran"])
        self.assertEqual(res["selection_reasons"], {})

    def test_background_honest_skip_carries_selection(self):
        _write(self.fx.root, "lonely.py", "L = 1\n")
        res = bg_tests.background_run_tests(
            str(self.fx.root), changed_only=True,
            changed_files=["lonely.py"])
        self.assertIsInstance(res, dict)
        self.assertTrue(res["nothing_ran"])
        self.assertEqual(res["selection_reasons"], {})

    def test_background_graph_run_reports_reasons(self):
        run = bg_tests.background_run_tests(
            str(self.fx.root), changed_only=True,
            changed_files=["pkg/core.py"], cap_seconds=120)
        self.assertNotIsInstance(run, dict)
        result = run.wait()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["passed"], 5)
        self.assertIn("graph-direct",
                      result["selection_reasons"]["tests/test_zebra.py"])


class GateFixture:
    """Selection-green, full-suite-red: tests/test_broken.py fails and
    covers nothing — it is pre-existing breakage the gate must
    attribute, not a verdict on the change."""

    def __init__(self, broken: bool = True):
        self.root = Path(tempfile.mkdtemp(prefix="pass62_gate_"))
        _write(self.root, "pkg/__init__.py", "")
        _write(self.root, "pkg/core.py", "CORE = 1\n")
        _write(self.root, "tests/__init__.py", "")
        _write(self.root, "tests/test_zebra.py",
               "import pkg.core\n\n\ndef test_core_value():\n"
               "    assert pkg.core.CORE == 1\n")
        if broken:
            _write(self.root, "tests/test_broken.py",
                   "def test_bad():\n    assert False, 'pre-existing'\n")

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class FinalGateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.fx = GateFixture(broken=True)
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(import_graph.clear_cache)

    def test_gate_off_by_default(self):
        res = pytest_runner.run_tests(
            repo=str(self.fx.root), changed_files=["pkg/core.py"],
            timeout=120)
        self.assertTrue(res["ok"], res)
        self.assertNotIn("final_gate", res)

    def test_gate_fail_is_distinct_and_attributed(self):
        res = pytest_runner.run_tests(
            repo=str(self.fx.root), changed_files=["pkg/core.py"],
            timeout=120, full_suite_gate=True)
        # the SELECTION verdict stands: the change's own tests passed.
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["passed"], 1)
        gate = res["final_gate"]
        self.assertTrue(gate["ran"])
        self.assertEqual(gate["verdict"], "fail")
        self.assertEqual(res["final_gate_verdict"], "fail")
        self.assertTrue(
            any("test_broken" in t for t in gate["failed"]), gate)
        self.assertIn("final gate", res["note"])

    def test_gate_pass(self):
        fx = GateFixture(broken=False)
        self.addCleanup(fx.cleanup)
        res = pytest_runner.run_tests(
            repo=str(fx.root), changed_files=["pkg/core.py"],
            timeout=120, full_suite_gate=True)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["final_gate"]["verdict"], "pass")
        self.assertEqual(res["final_gate"]["failed"], [])

    def test_gate_skipped_when_selection_red(self):
        _write(self.fx.root, "tests/test_zebra.py",
               "import pkg.core\n\n\ndef test_core_value():\n"
               "    assert pkg.core.CORE == 99\n")
        res = pytest_runner.run_tests(
            repo=str(self.fx.root), changed_files=["pkg/core.py"],
            timeout=120, full_suite_gate=True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["final_gate"]["verdict"], "skipped")
        self.assertFalse(res["final_gate"]["ran"])

    def test_gate_skipped_when_nothing_ran(self):
        _write(self.fx.root, "lonely.py", "L = 1\n")
        res = pytest_runner.run_tests(
            repo=str(self.fx.root), changed_files=["lonely.py"],
            timeout=60, full_suite_gate=True)
        self.assertTrue(res["nothing_ran"])
        self.assertEqual(res["final_gate"]["verdict"], "skipped")


# ── loop-level ────────────────────────────────────────────────────────

class ScriptedRouter:
    """Deterministic stand-in for the LLM: canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, params=None):
        self.calls += 1
        text = self._responses.pop(0) if self._responses else ""
        return SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "empty")


class StubDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE coding_log (id TEXT, task TEXT, filename TEXT,"
            " attempt INT, exit_code INT, timed_out INT, stdout TEXT,"
            " stderr TEXT, code TEXT, created_at REAL)")

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


class StubContext:
    def __init__(self, router):
        self.router = router
        self.db = StubDB()
        self.settings = SimpleNamespace(reasoning_mode="off")


def _json_block(payload):
    return "```json\n" + json.dumps(payload) + "\n```"


_PLAN = _json_block({"files": [
    {"path": "widget.py", "why": "test", "new_file": False}]})
_EDIT = _json_block({"edits": [
    {"old_text": "VALUE = 1", "new_text": "VALUE = 2"}]})


@unittest.skipUnless(shutil.which("git"), "git not available")
class FinalGateLoopTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(import_graph.clear_cache)

    def _repo(self, extra_broken: bool = False) -> Path:
        root = Path(tempfile.mkdtemp(prefix="pass62_loop_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        _git("init", "-q", cwd=root)
        _git("config", "user.email", "t@example.com", cwd=root)
        _git("config", "user.name", "T", cwd=root)
        _write(root, "widget.py", "VALUE = 1\n")
        _write(root, "tests/__init__.py", "")
        # behavior-named on purpose: stem "widget" is not in
        # "test_behavior" — only the import graph connects them.
        _write(root, "tests/test_behavior.py",
               "import widget\n\n\ndef test_value():\n"
               "    assert widget.VALUE == 2\n")
        if extra_broken:
            _write(root, "tests/test_unrelated.py",
                   "def test_bad():\n    assert False, 'pre-existing'\n")
        _git("add", ".", cwd=root)
        _git("commit", "-qm", "init", cwd=root)
        return root

    def _agent(self, root, responses):
        # pass #67 fixture note: the run-start brief now fires for
        # EVERY task (it used to be length-gated off for these short
        # ones), so each script gains a leading "{}" — a model reply
        # with no objective, which sends the brief to its heuristic
        # floor and leaves the task text (and this script) unchanged.
        return CodingAgent(
            StubContext(ScriptedRouter(["{}"] + list(responses))),
            root=str(root))

    def test_loop_selection_follows_changed_files_via_graph(self):
        root = self._repo()
        agent = self._agent(root, [_PLAN, _EDIT])
        result = agent.run("Fix widget value", max_iterations=3,
                           timeout=30)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertEqual(result.verification, "tests")
        # no gate was requested: the field stays empty.
        self.assertEqual(result.final_gate, {})

    def test_loop_gate_pass_recorded(self):
        root = self._repo()
        agent = self._agent(root, [_PLAN, _EDIT])
        result = agent.run("Fix widget value", max_iterations=3,
                           timeout=30, full_suite_gate=True)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertEqual(result.final_gate["verdict"], "pass")
        self.assertEqual(
            result.to_dict()["final_gate"]["verdict"], "pass")

    def test_loop_gate_fail_never_rewrites_run_verdict(self):
        root = self._repo(extra_broken=True)
        agent = self._agent(root, [_PLAN, _EDIT])
        result = agent.run("Fix widget value", max_iterations=3,
                           timeout=30, full_suite_gate=True)
        # selection verdict: the change is verified, the run is green.
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertEqual(result.verification, "tests")
        # the gate's distinct verdict names the unrelated breakage.
        gate = result.final_gate
        self.assertEqual(gate["verdict"], "fail")
        self.assertTrue(
            any("test_unrelated" in t for t in gate["failed"]), gate)
        self.assertIn("final gate", result.output)


if __name__ == "__main__":
    unittest.main()
