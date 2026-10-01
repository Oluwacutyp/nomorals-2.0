"""Tests for the specialist subagent roster (nomorals/agents/subagents.py).

All offline: model-dependent paths use the mock LLM provider with scripted
replies; rule-based paths (reviewer AST checks, dep hunter stdlib/repo
checks) need no model at all.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.agents import subagents as S
from nomorals.agents.subagents import (
    AgentRun,
    ApiDesigner,
    ApiDesignInput,
    DepHunter,
    DepInput,
    FlawCategory,
    FusedResult,
    Implementer,
    ImplementInput,
    Planner,
    PlanInput,
    Refactorer,
    RefactorInput,
    Reviewer,
    ReviewInput,
    ROSTER,
    Tester,
    TestInput,
    run_parallel,
)
from nomorals.llm.providers.mock import MockProvider


def _mock(scripted: dict[str, str]) -> MockProvider:
    return MockProvider(scripted=scripted)


def _reply(obj: object) -> str:
    return json.dumps(obj)


# ── fixtures ─────────────────────────────────────────────────────────────

_NEW_FILE_DIFF = """--- /dev/null
+++ b/greet.py
@@ -0,0 +1,4 @@
+def greet(name):
+    print(undeff_name_here)
+    return "hi " + name
+
"""

_CLEAN_DIFF = """--- a/util.py
+++ b/util.py
@@ -1,3 +1,3 @@
 def double(x):
-    return x * 2
+    return x * 3
"""


class _Boom(S.Subagent):
    roster_type = "reviewer"

    def run(self, inp):  # noqa: ANN001, ANN202
        raise RuntimeError("boom")


class _Slow(S.Subagent):
    roster_type = "reviewer"

    def run(self, inp):  # noqa: ANN001, ANN202
        time.sleep(1.0)
        return {"done": True}


# ── planner ──────────────────────────────────────────────────────────────


class PlannerTests(unittest.TestCase):
    def test_happy_path(self):
        router = _mock({"file-level implementation plan": _reply({"items": [
            {"path": "a.py", "why": "core", "new_file": False,
             "acceptance": "unit tests"},
            {"path": "b.py", "why": "helper", "new_file": True,
             "acceptance": "imports"},
        ]})})
        result = Planner(router=router, project_root=".").run(
            PlanInput(task="do things",
                      repo_map=[{"path": "a.py", "purpose": "core"}]))
        self.assertFalse(result.fallback)
        self.assertEqual([i.path for i in result.items], ["a.py", "b.py"])
        self.assertFalse(result.items[0].new_file)
        self.assertTrue(result.items[1].new_file)

    def test_fallback_without_model(self):
        result = Planner(router=None, project_root=".").run(
            PlanInput(task="do things",
                      repo_map=[{"path": "a.py", "purpose": "core"}]))
        self.assertTrue(result.fallback)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].path, "a.py")
        self.assertIn("no model", result.note)

    def test_fallback_without_model_or_map(self):
        result = Planner(router=None, project_root=".").run(
            PlanInput(task="do things"))
        self.assertTrue(result.fallback)
        self.assertEqual(result.items[0].path, "notes/plan.md")
        self.assertTrue(result.items[0].new_file)

    def test_path_escape_dropped(self):
        router = _mock({"file-level implementation plan": _reply({"items": [
            {"path": "../../etc/passwd", "why": "evil", "new_file": True,
             "acceptance": "no"},
            {"path": "ok.py", "why": "fine", "new_file": False,
             "acceptance": "tests"},
        ]})})
        result = Planner(router=router, project_root=".").run(
            PlanInput(task="t"))
        self.assertFalse(result.fallback)
        self.assertEqual([i.path for i in result.items], ["ok.py"])
        self.assertIn("dropped 1", result.note)

    def test_all_bad_paths_falls_back(self):
        router = _mock({"file-level implementation plan": _reply({"items": [
            {"path": "/abs/x.py", "why": "evil", "new_file": True,
             "acceptance": "no"},
        ]})})
        result = Planner(router=router, project_root=".").run(
            PlanInput(task="t", repo_map=[{"path": "a.py", "purpose": "x"}]))
        self.assertTrue(result.fallback)

    def test_truncates_to_bound(self):
        items = [{"path": f"m{i}.py", "why": "w", "new_file": True,
                  "acceptance": "a"} for i in range(15)]
        router = _mock({"file-level implementation plan": _reply(
            {"items": items})})
        result = Planner(router=router, project_root=".").run(
            PlanInput(task="t", max_files=12))
        self.assertEqual(len(result.items), 12)
        self.assertIn("truncated", result.note)

    def test_garbage_model_output_falls_back(self):
        router = _mock({"file-level implementation plan": "not json at all"})
        result = Planner(router=router, project_root=".").run(
            PlanInput(task="t"))
        self.assertTrue(result.fallback)


# ── implementer ──────────────────────────────────────────────────────────


class ImplementerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel: str, content: str) -> None:
        (self.root / rel).write_text(content, encoding="utf-8")

    def test_surgical_edit_applies(self):
        self._write("calc.py", "def add(a, b):\n    return a + b\n")
        router = _mock({"surgical edits": _reply({"edits": [
            {"old_text": "    return a + b",
             "new_text": "    return a + b + 0"}]})})
        result = Implementer(router=router,
                             project_root=str(self.root)).run(
            ImplementInput(path="calc.py", instruction="no-op tweak"))
        self.assertTrue(result.ok, result.note)
        self.assertTrue(result.edits[0].applied)
        self.assertIn("return a + b + 0",
                      (self.root / "calc.py").read_text())
        self.assertIn("@@", result.edits[0].diff)

    def test_failed_edit_attributed_and_file_untouched(self):
        self._write("calc.py", "x = 1\n")
        router = _mock({"surgical edits": _reply({"edits": [
            {"old_text": "nope, not here", "new_text": "whatever"}]})})
        result = Implementer(router=router,
                             project_root=str(self.root)).run(
            ImplementInput(path="calc.py", instruction="bad edit"))
        self.assertFalse(result.ok)
        self.assertFalse(result.edits[0].applied)
        self.assertTrue(result.edits[0].error)
        self.assertEqual((self.root / "calc.py").read_text(), "x = 1\n")

    def test_new_file_created(self):
        router = _mock({"complete contents": _reply(
            {"content": "VALUE = 42\n"})})
        result = Implementer(router=router,
                             project_root=str(self.root)).run(
            ImplementInput(path="newmod.py", instruction="create",
                           new_file=True))
        self.assertTrue(result.ok, result.note)
        self.assertEqual((self.root / "newmod.py").read_text(), "VALUE = 42\n")

    def test_no_model(self):
        result = Implementer(router=None,
                             project_root=str(self.root)).run(
            ImplementInput(path="x.py", instruction="y", new_file=True))
        self.assertFalse(result.ok)
        self.assertIn("no model", result.note)

    def test_path_escape_rejected(self):
        result = Implementer(router=None,
                             project_root=str(self.root)).run(
            ImplementInput(path="../evil.py", instruction="y"))
        self.assertFalse(result.ok)


# ── reviewer ─────────────────────────────────────────────────────────────


class ReviewerTests(unittest.TestCase):
    def test_undefined_name_detected_rule_based(self):
        result = Reviewer(router=None, project_root=".").run(
            ReviewInput(diff=_NEW_FILE_DIFF))
        self.assertEqual(result.model_review, "skipped")
        hits = [f for f in result.flaws
                if f.category == FlawCategory.UNDEFINED_NAME.value]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].file, "greet.py")
        self.assertEqual(hits[0].line, 2)
        self.assertIn("undeff_name_here", hits[0].message)
        self.assertEqual(hits[0].severity, "high")

    def test_no_false_positive_on_defined_names(self):
        result = Reviewer(router=None, project_root=".").run(
            ReviewInput(diff=_CLEAN_DIFF))
        undef = [f for f in result.flaws
                 if f.category == FlawCategory.UNDEFINED_NAME.value]
        self.assertEqual(undef, [])

    def test_unused_import_is_dead_code(self):
        diff = ("--- /dev/null\n+++ b/m.py\n@@ -0,0 +1,4 @@\n"
                "+import os\n+\n+def f():\n+    return 1\n")
        result = Reviewer(router=None, project_root=".").run(
            ReviewInput(diff=diff))
        dead = [f for f in result.flaws
                if f.category == FlawCategory.DEAD_CODE.value]
        self.assertEqual(len(dead), 1)
        self.assertIn("os", dead[0].message)

    def test_syntax_error_maps_to_contract_violation(self):
        diff = ("--- /dev/null\n+++ b/broken.py\n@@ -0,0 +1,2 @@\n"
                "+def f(:\n+    pass\n")
        result = Reviewer(router=None, project_root=".").run(
            ReviewInput(diff=diff))
        self.assertEqual(len(result.flaws), 1)
        flaw = result.flaws[0]
        self.assertEqual(flaw.category,
                         FlawCategory.CONTRACT_VIOLATION.value)
        self.assertEqual(flaw.severity, "critical")

    def test_model_flaws_merged_and_invalid_dropped(self):
        router = _mock({"unified diff": _reply({"flaws": [
            {"category": "security-risk", "file": "util.py", "line": 2,
             "severity": "high", "message": "uses eval on input"},
            {"category": "bogus-category", "file": "util.py", "line": 1,
             "severity": "low", "message": "dropped"},
            {"category": "dead-code", "file": "util.py", "line": 9,
             "severity": "weird", "message": "coerced severity"},
        ]})})
        result = Reviewer(router=router, project_root=".").run(
            ReviewInput(diff=_CLEAN_DIFF))
        self.assertEqual(result.model_review, "ok")
        by_cat = {f.category for f in result.flaws}
        self.assertIn("security-risk", by_cat)
        self.assertNotIn("bogus-category", by_cat)
        coerced = [f for f in result.flaws if f.message == "coerced severity"]
        self.assertEqual(coerced[0].severity, "low")
        self.assertIn("dropped 1", result.note)

    def test_empty_diff(self):
        result = Reviewer(router=None, project_root=".").run(
            ReviewInput(diff=""))
        self.assertEqual(result.flaws, [])
        self.assertEqual(result.model_review, "skipped")


# ── tester ───────────────────────────────────────────────────────────────


_TEST_CODE = """import unittest

import calc as _target


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(_target.add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(_target.add(-1, 1), 0)
"""


class TesterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "calc.py").write_text(
            "def add(a, b):\n    return a + b\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_generated_tests_run_and_pass(self):
        router = _mock({"focused unittest suite": _reply({"code": _TEST_CODE})})
        result = Tester(router=router, project_root=str(self.root)).run(
            TestInput(changed_files=["calc.py"]))
        self.assertTrue(result.ok, result.note)
        self.assertEqual(len(result.tests_written), 1)
        self.assertEqual(result.discarded, [])
        self.assertGreaterEqual(result.passed, 2)
        self.assertEqual(result.failed, [])

    def test_syntax_error_discarded_and_reported(self):
        router = _mock({"focused unittest suite": _reply(
            {"code": "def broken(:"})})
        result = Tester(router=router, project_root=str(self.root)).run(
            TestInput(changed_files=["calc.py"]))
        self.assertFalse(result.ok)
        self.assertEqual(result.tests_written, [])
        self.assertEqual(len(result.discarded), 1)
        self.assertIn("syntax error", result.discarded[0]["reason"])

    def test_uncollectable_discarded_and_reported(self):
        router = _mock({"focused unittest suite": _reply(
            {"code": "import unittest\nNOTHING_HERE = 1\n"})})
        result = Tester(router=router, project_root=str(self.root)).run(
            TestInput(changed_files=["calc.py"]))
        self.assertFalse(result.ok)
        self.assertEqual(result.tests_written, [])
        self.assertEqual(len(result.discarded), 1)
        self.assertIn("no test cases", result.discarded[0]["reason"])

    def test_no_model(self):
        result = Tester(router=None, project_root=str(self.root)).run(
            TestInput(changed_files=["calc.py"]))
        self.assertFalse(result.ok)
        self.assertIn("no model", result.note)


# ── refactorer ───────────────────────────────────────────────────────────


class RefactorerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "mod.py").write_text(
            "def compute(x):\n    y = x * 2\n    return y\n",
            encoding="utf-8")
        tests_dir = self.root / "tests"
        tests_dir.mkdir()
        # TestCase-based: the repo's pytest_runner falls back to unittest
        # when pytest is absent, and unittest only collects TestCases.
        (tests_dir / "test_mod.py").write_text(
            "import unittest\n\nimport mod\n\n\n"
            "class TestCompute(unittest.TestCase):\n"
            "    def test_compute(self):\n"
            "        self.assertEqual(mod.compute(3), 6)\n",
            encoding="utf-8")
        self.original = (self.root / "mod.py").read_text()

    def tearDown(self):
        self.tmp.cleanup()

    def test_verified_refactor_applies(self):
        router = _mock({"surgical refactor edits": _reply({"edits": [
            {"old_text": "    y = x * 2\n    return y",
             "new_text": "    result = x * 2\n    return result"}]})})
        result = Refactorer(router=router,
                            project_root=str(self.root)).run(
            RefactorInput(path="mod.py", goal="clearer local name"))
        self.assertTrue(result.applied, result.note)
        self.assertFalse(result.reverted)
        self.assertTrue(result.tests_before["ok"])
        self.assertTrue(result.tests_after["ok"])
        self.assertIn("result = x * 2",
                      (self.root / "mod.py").read_text())

    def test_regression_reverts_honestly(self):
        router = _mock({"surgical refactor edits": _reply({"edits": [
            {"old_text": "    return y",
             "new_text": "    return zzz"}]})})
        result = Refactorer(router=router,
                            project_root=str(self.root)).run(
            RefactorInput(path="mod.py", goal="break it deliberately"))
        self.assertFalse(result.applied)
        self.assertTrue(result.reverted)
        self.assertIn("reverted", result.note)
        self.assertEqual((self.root / "mod.py").read_text(), self.original)

    def test_no_model(self):
        result = Refactorer(router=None,
                            project_root=str(self.root)).run(
            RefactorInput(path="mod.py", goal="x"))
        self.assertFalse(result.applied)
        self.assertIn("no model", result.note)


# ── dependency hunter ────────────────────────────────────────────────────


class DepHunterTests(unittest.TestCase):
    def test_stdlib(self):
        result = DepHunter().run(DepInput(name="json"))
        self.assertEqual(result.verdict, "stdlib")
        self.assertIn("standard library", result.detail)

    def test_stdlib_pathlib(self):
        result = DepHunter().run(DepInput(name="pathlib"))
        self.assertEqual(result.verdict, "stdlib")

    def test_repo_existing(self):
        result = DepHunter().run(DepInput(name="web"))
        self.assertEqual(result.verdict, "repo-existing")
        self.assertIn("nomorals.tools.web", result.detail)
        self.assertIn("reuse", result.recommendation)

    def test_unknown_package(self):
        result = DepHunter().run(DepInput(name="no-such-pkg-xyz-999"))
        # PyPI 404 or no network both land here — never blocks.
        self.assertEqual(result.verdict, "unknown")
        self.assertIn("do not add blindly", result.recommendation)

    def test_empty_name(self):
        result = DepHunter().run(DepInput(name="  "))
        self.assertEqual(result.verdict, "unknown")

    def test_verdict_vocabulary(self):
        self.assertEqual(
            set(DepHunter().run(DepInput(name="os")).to_dict()),
            {"name", "verdict", "detail", "recommendation"})


# ── API designer ─────────────────────────────────────────────────────────


_STUB = '''"""Widget API."""

from dataclasses import dataclass

__all__ = ["Widget", "make_widget", "WidgetResult"]


@dataclass
class WidgetResult:
    ok: bool
    name: str = ""

    def to_dict(self):
        return {"ok": self.ok, "name": self.name}


class Widget:
    def __init__(self, name: str):
        self.name = name


def make_widget(name: str) -> WidgetResult:
    return WidgetResult(ok=True, name=name)
'''


class ApiDesignerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "sample_a.py").write_text(
            '"""Sample A."""\n\n__all__ = ["thing"]\n\n\n'
            'def thing():\n    """Do the thing."""\n    return 1\n',
            encoding="utf-8")
        (self.root / "sample_b.py").write_text(
            '"""Sample B."""\n\n__all__ = []\n', encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_stub_imports_and_notes(self):
        router = _mock({"public Python API": _reply({"stub": _STUB})})
        result = ApiDesigner(router=router,
                             project_root=str(self.root)).run(
            ApiDesignInput(feature="a widget factory",
                           sample_modules=["sample_a.py", "sample_b.py"]))
        self.assertTrue(result.imports_ok, result.import_error)
        self.assertTrue(result.stub_code)
        self.assertTrue(result.consistency_notes)
        self.assertTrue(any(n.startswith("ok:") for n in result.consistency_notes))

    def test_broken_stub_reported(self):
        router = _mock({"public Python API": _reply(
            {"stub": "import nonexistent_module_xyz_123\n"})})
        result = ApiDesigner(router=router,
                             project_root=str(self.root)).run(
            ApiDesignInput(feature="x",
                           sample_modules=["sample_a.py"]))
        self.assertFalse(result.imports_ok)
        self.assertTrue(result.import_error)

    def test_no_model(self):
        result = ApiDesigner(router=None,
                             project_root=str(self.root)).run(
            ApiDesignInput(feature="x"))
        self.assertFalse(result.imports_ok)
        self.assertIn("no model", result.note)


# ── fused runner ─────────────────────────────────────────────────────────


class FusedRunnerTests(unittest.TestCase):
    def test_reviewer_merge_dedupes(self):
        router = _mock({"unified diff": _reply({"flaws": []})})
        agents = [Reviewer(router=router, project_root="."),
                  Reviewer(router=router, project_root=".")]
        fused = run_parallel(agents,
                             [ReviewInput(diff=_NEW_FILE_DIFF),
                              ReviewInput(diff=_NEW_FILE_DIFF)],
                             timeout=30.0)
        self.assertIsInstance(fused, FusedResult)
        self.assertTrue(fused.ok)
        self.assertEqual(fused.succeeded, 2)
        self.assertEqual(fused.failed, 0)
        # both reviewers flag the same undefined name -> one merged flaw
        self.assertEqual(fused.merged["count"], 1)
        self.assertEqual(fused.merge, "reviewer")

    def test_failure_is_recorded_not_raised(self):
        router = _mock({"unified diff": _reply({"flaws": []})})
        agents = [Reviewer(router=router, project_root="."),
                  _Boom(project_root=".")]
        fused = run_parallel(agents,
                             [ReviewInput(diff=_CLEAN_DIFF),
                              ReviewInput(diff=_CLEAN_DIFF)],
                             timeout=30.0)
        self.assertFalse(fused.ok)
        self.assertEqual(fused.succeeded, 1)
        self.assertEqual(fused.failed, 1)
        boom_run = [r for r in fused.runs if r.name.startswith("reviewer-1")][0]
        self.assertFalse(boom_run.ok)
        self.assertIn("RuntimeError: boom", boom_run.error)
        # the good reviewer's result still merged
        self.assertIn("flaws", fused.merged)

    def test_timeout_is_recorded(self):
        fused = run_parallel([_Slow(project_root=".")], [{"x": 1}],
                             timeout=0.2)
        self.assertFalse(fused.ok)
        self.assertEqual(fused.failed, 1)
        self.assertIn("timed out", fused.runs[0].error)

    def test_planner_merge_unions_by_path(self):
        reply = _reply({"items": [
            {"path": "a.py", "why": "w", "new_file": False,
             "acceptance": "t"},
            {"path": "b.py", "why": "w", "new_file": True,
             "acceptance": "t"},
        ]})
        agents = [Planner(router=_mock({"file-level implementation plan": reply}),
                          project_root="."),
                  Planner(router=_mock({"file-level implementation plan": reply}),
                          project_root=".")]
        fused = run_parallel(agents, [PlanInput(task="t"), PlanInput(task="t")],
                             timeout=30.0, merge="planner")
        self.assertTrue(fused.ok)
        self.assertEqual(fused.merged["count"], 2)
        self.assertEqual([i["path"] for i in fused.merged["items"]],
                         ["a.py", "b.py"])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            run_parallel([Reviewer(router=None)], [], timeout=5.0)

    def test_empty_batch(self):
        fused = run_parallel([], [], timeout=5.0)
        self.assertFalse(fused.ok)
        self.assertEqual(fused.runs, [])


# ── roster ───────────────────────────────────────────────────────────────


class RosterTests(unittest.TestCase):
    def test_roster_keys(self):
        self.assertEqual(set(ROSTER),
                         {"planner", "implementer", "reviewer", "tester",
                          "refactorer", "dep_hunter", "api_designer"})

    def test_each_has_run_and_roster_type(self):
        for name, cls in ROSTER.items():
            agent = cls(router=None)
            self.assertTrue(callable(agent.run))
            self.assertEqual(agent.roster_type, name)
            self.assertIsNotNone(agent.role_spec)

    def test_result_dataclasses_serialize(self):
        flaw = S.Flaw(category="dead-code", file="x.py", line=1,
                      severity="low", message="m")
        self.assertEqual(flaw.to_dict()["category"], "dead-code")
        run = AgentRun(name="r-0", ok=True, seconds=0.1, result=None)
        self.assertEqual(run.to_dict()["name"], "r-0")


if __name__ == "__main__":
    unittest.main()
