"""Pass #68: per-step isolated verification — breaks are attributed
to the step that caused them.

The defect, from the coding-loop survey: verification ran ONCE per
round over the run's ACCUMULATED changed files (pass #62 selection
over the union of every step's files).  The verdict was a single
boolean for the whole union; on failure the loop stamped the whole
formatted union output onto every file touched that round, and
``CodingResult.steps`` carried no record of which tests cover a
step, whether they ever ran, or which ones failed.  A break
introduced by step 2 of 5 surfaced (if at all) with no per-step
account — the runner computed per-test failure ids and the exact
selection, and the loop discarded both.

The fix (agents/coding.py only): after each round's drafting, every
step that drafted gets an ISOLATED check — the same #62 selection
(stem ∪ import graph, via ``pytest_runner.run_tests`` /
``bg_tests.background_run_tests``) scoped to THAT step's changed
files — recorded additively on the step record as ``verify`` =
{ran, ok, tests, failed, checks, executed_total, note}:

- a step whose files select no tests records ran=False / ok=None
  (unverified at this step — pass #58 honesty: not a pass, not a
  failure);
- a failed check names the failing tests on the step's record
  (``failed`` keeps the trail across repairs) and feeds the failing
  tests into the step's error slot, so the EXISTING repair
  machinery (per-file minimum + shared pool, #60) repairs THAT
  step and the step re-checks after repair;
- the round's union verify remains the sole authority for
  run-green: a break visible only in the union (a file no step
  claims) still fails the run even when every step check is green.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.coding import CodingAgent


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class RuleRouter:
    """Content-routed fake LLM (pass #60 harness): the handler
    answers by what the call is asking, so call ORDER never
    matters — only content does."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, str]] = []

    def chat(self, messages, params=None):
        texts = [getattr(m, "content", "") or "" for m in messages]
        system = texts[0] if texts else ""
        user = texts[-1] if texts else ""
        self.calls.append((system, user))
        text = self.handler(system, user)
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


class FixtureRepo:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass68_"))
        _git("init", "-q", cwd=self.root)
        _git("config", "user.email", "t@example.com", cwd=self.root)
        _git("config", "user.name", "T", cwd=self.root)

    def write(self, rel, content):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def commit_all(self, msg="init"):
        _git("add", ".", cwd=self.root)
        _git("commit", "-qm", msg, cwd=self.root)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _steps_json(steps):
    return "```json\n" + json.dumps({"steps": steps}) + "\n```"


def _code(code):
    return "```python\n" + code + "\n```"


def _edits(edits):
    return "```json\n" + json.dumps({"edits": edits}) + "\n```"


def _by_path(result):
    return {s["path"]: s for s in result.steps}


# ── (a) attribution + repair ─────────────────────────────────────

_CALC_STEPS = [
    {"path": "calc.py", "why": "the math", "new_file": True,
     "instructions": "implement add(a, b) returning the sum",
     "acceptance": "test_calc passes"},
    {"path": "report.py", "why": "reporting", "new_file": True,
     "instructions": "write banner() returning the report banner",
     "depends_on": ["s1"], "acceptance": "test_report passes"},
]


def _calc_report_repo(fx):
    fx.write("tests/__init__.py", "")
    fx.write("tests/test_calc.py",
             "import calc\n\n\ndef test_add():\n"
             "    assert calc.add(2, 3) == 5\n")
    fx.write("tests/test_report.py",
             "import report\n\n\ndef test_banner():\n"
             "    assert report.banner() == 'REPORT'\n")
    fx.commit_all()
    return fx


@unittest.skipUnless(shutil.which("git"), "git not available")
class AttributionRepairTests(unittest.TestCase):
    """(a) step 2 breaks a test step 1's check passed: the break is
    attributed to step 2 by name, repair lands on step 2, the step
    re-checks, and the run's final state reflects the truth."""

    def test_step_two_break_attributed_repaired_and_rechecked(self):
        fx = _calc_report_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        report_drafts = {"n": 0}

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(_CALC_STEPS)
            if "Write complete, runnable Python" in system:
                if "for the file 'calc.py'" in system:
                    return _code("def add(a, b):\n    return a + b\n")
                if "for the file 'report.py'" in system:
                    report_drafts["n"] += 1
                    if report_drafts["n"] == 1:
                        # step 2's first draft breaks its own test
                        return _code("def banner():\n"
                                     "    return 'REPRT'\n")
                    return _code("def banner():\n"
                                 "    return 'REPORT'\n")
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Build calc and report", max_iterations=5,
                           timeout=30, plan_mode="never")
        self.assertTrue(result.ok, result.error)
        self.assertGreaterEqual(report_drafts["n"], 2,
                                "step 2 was re-drafted (repair)")
        steps = _by_path(result)
        calc, report = steps["calc.py"], steps["report.py"]
        self.assertEqual(calc["status"], "done")
        self.assertEqual(report["status"], "done")
        # step 1's isolated check ran and passed; it never broke a test
        self.assertTrue(calc["verify"]["ran"])
        self.assertTrue(calc["verify"]["ok"])
        self.assertEqual(calc["verify"]["failed"], [])
        self.assertIn("tests/test_calc.py", calc["verify"]["tests"])
        self.assertNotIn("tests/test_report.py", calc["verify"]["tests"])
        # step 2: the failing test is NAMED on its record (the trail
        # survives the repair), it re-checked after repair, and the
        # latest check is green
        self.assertTrue(report["verify"]["ran"])
        self.assertTrue(report["verify"]["ok"])
        self.assertGreaterEqual(report["verify"]["checks"], 2)
        self.assertTrue(
            any(t.startswith("tests/test_report.py::test_banner")
                for t in report["verify"]["failed"]),
            report["verify"])
        self.assertIn("tests/test_report.py", report["verify"]["tests"])
        # (e) the records ride to_dict / JSON consumers unchanged
        self.assertEqual(result.to_dict()["steps"], result.steps)
        json.dumps(result.to_dict())

    def test_background_mode_records_step_verify(self):
        """(4) with background tests enabled (#66), the isolated
        check runs through the background machinery and still lands
        on the step record."""
        fx = _calc_report_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(_CALC_STEPS)
            if "Write complete, runnable Python" in system:
                if "for the file 'calc.py'" in system:
                    return _code("def add(a, b):\n    return a + b\n")
                if "for the file 'report.py'" in system:
                    return _code("def banner():\n"
                                 "    return 'REPORT'\n")
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Build calc and report", max_iterations=5,
                           timeout=30, plan_mode="never",
                           background_tests=True)
        self.assertTrue(result.ok, result.error)
        steps = _by_path(result)
        self.assertTrue(steps["calc.py"]["verify"]["ran"])
        self.assertTrue(steps["calc.py"]["verify"]["ok"])
        self.assertIn("tests/test_calc.py",
                      steps["calc.py"]["verify"]["tests"])


# ── (b) no covering tests: honest unverified-at-step ─────────────

@unittest.skipUnless(shutil.which("git"), "git not available")
class UnverifiedStepTests(unittest.TestCase):
    """(b) a step whose files select NO tests records
    unverified-at-step (ran=False, ok=None — not a pass, not a
    failure) and the run can still go green at the union verify."""

    def test_step_without_covering_tests_records_unverified(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("tests/__init__.py", "")
        fx.write("tests/test_calc.py",
                 "import calc\n\n\ndef test_add():\n"
                 "    assert calc.add(2, 3) == 5\n")
        fx.commit_all()
        steps = [
            {"path": "calc.py", "why": "math", "new_file": True,
             "instructions": "implement add", "acceptance": "tests pass"},
            {"path": "notes.py", "why": "scratch", "new_file": True,
             "instructions": "write note()", "acceptance": "imports"},
        ]

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(steps)
            if "Write complete, runnable Python" in system:
                if "for the file 'calc.py'" in system:
                    return _code("def add(a, b):\n    return a + b\n")
                if "for the file 'notes.py'" in system:
                    return _code("def note():\n    return 'noted'\n")
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Build calc and notes", max_iterations=5,
                           timeout=30, plan_mode="never")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.verification, "tests")
        steps = _by_path(result)
        self.assertEqual(steps["notes.py"]["status"], "done")
        verify = steps["notes.py"]["verify"]
        self.assertFalse(verify["ran"])
        self.assertIsNone(verify["ok"])
        self.assertEqual(verify["tests"], [])
        self.assertEqual(verify["failed"], [])
        self.assertIn("no tests cover", verify["note"])
        # the covered step, by contrast, was really checked
        self.assertTrue(steps["calc.py"]["verify"]["ok"])
        self.assertIn("tests/test_calc.py",
                      steps["calc.py"]["verify"]["tests"])


# ── (c) union-only break: the round verify stays the authority ───

@unittest.skipUnless(shutil.which("git"), "git not available")
class UnionOnlyBreakTests(unittest.TestCase):
    """(c) a multi-file patch changes a file NO step claims; the
    break is visible only in the union selection.  Every step
    check is green, yet the run must NOT go green — per-step green
    never substitutes for the union verify."""

    def test_union_only_break_still_fails_the_run(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("calc.py", "VALUE = 1\n")
        fx.write("helper.py", "BONUS = 10\n")
        fx.write("tests/__init__.py", "")
        fx.write("tests/test_calc.py",
                 "import calc\n\n\ndef test_value():\n"
                 "    assert calc.VALUE == 2\n")
        fx.write("tests/test_helper.py",
                 "import helper\n\n\ndef test_bonus():\n"
                 "    assert helper.BONUS == 10\n")
        fx.commit_all()
        steps = [{"path": "calc.py", "why": "bump", "new_file": False,
                  "instructions": "make VALUE 2",
                  "acceptance": "test_calc passes"}]
        diff = ("```diff\n"
                "--- a/calc.py\n+++ b/calc.py\n"
                "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
                "--- a/helper.py\n+++ b/helper.py\n"
                "@@ -1 +1 @@\n-BONUS = 10\n+BONUS = 99\n"
                "```\n")

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(steps)
            if "surgical code editor" in system:
                return diff
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Bump the value", max_iterations=2,
                           timeout=30, plan_mode="never")
        self.assertFalse(result.ok)
        step = result.steps[0]
        # the step's own covering tests passed in isolation…
        self.assertTrue(step["verify"]["ran"])
        self.assertTrue(step["verify"]["ok"], step["verify"])
        self.assertIn("tests/test_calc.py", step["verify"]["tests"])
        self.assertNotIn("tests/test_helper.py", step["verify"]["tests"])
        # …but the union verify saw the helper break and the run
        # failed on it (the journal carries the union verdict text)
        rows = agent.context.db.query(
            "SELECT stdout FROM coding_log WHERE filename='calc.py'")
        self.assertTrue(any("test_helper" in (r["stdout"] or "")
                            for r in rows), rows)


# ── (d) failure beyond repair: dependents skip, cause named ──────

@unittest.skipUnless(shutil.which("git"), "git not available")
class DependencyAttributionTests(unittest.TestCase):
    """(d) step 1's isolated check fails beyond repair: step 1
    settles failed NAMING the failing tests (record and reason),
    and the dependent step 2 — which never drafted — skips with
    the dependency as its named cause."""

    def test_failed_step_names_tests_dependent_skips(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("a.py", "VALUE = 0\n")
        fx.write("b.py", "B = 1\n")
        fx.write("tests/__init__.py", "")
        fx.write("tests/test_a.py",
                 "import a\n\n\ndef test_value():\n"
                 "    assert a.VALUE == 1\n")
        fx.commit_all()
        steps = [
            {"path": "a.py", "why": "fix value", "new_file": False,
             "instructions": "make VALUE 1", "acceptance": "test_a passes"},
            {"path": "b.py", "why": "companion", "new_file": False,
             "instructions": "leave b alone", "depends_on": ["s1"],
             "acceptance": "b unchanged"},
        ]

        def handler(system, user):
            if "build planner" in system:
                if "RE-PLAN" in user:
                    return "no usable plan in this reply"
                return _steps_json(steps)
            if "surgical code editor" in system:
                if "File: a.py" in user:
                    if "VALUE = -0" in user:
                        return _edits([{"old_text": "VALUE = -0",
                                        "new_text": "VALUE = 0"}])
                    return _edits([{"old_text": "VALUE = 0",
                                    "new_text": "VALUE = -0"}])
                return _edits([])  # b.py: no change, ever
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Fix the value module", max_iterations=4,
                           timeout=30, plan_mode="never")
        self.assertFalse(result.ok)
        steps = _by_path(result)
        a, b = steps["a.py"], steps["b.py"]
        self.assertEqual(a["status"], "failed")
        self.assertTrue(a["verify"]["ran"])
        self.assertFalse(a["verify"]["ok"])
        self.assertTrue(
            any(t.startswith("tests/test_a.py::test_value")
                for t in a["verify"]["failed"]), a["verify"])
        # the failing tests ride the step's reason trail too
        self.assertIn("test_value", a["reason"])
        # the dependent never attempted and names its cause
        self.assertEqual(b["status"], "skipped")
        self.assertEqual(b["attempts"], 0)
        self.assertIn("dependency s1", b["reason"])
        self.assertFalse(b["verify"]["ran"])
        self.assertIsNone(b["verify"]["ok"])


# ── (e) additive shape: plan pause + result defaults ─────────────

@unittest.skipUnless(shutil.which("git"), "git not available")
class AdditiveShapeTests(unittest.TestCase):
    """(e) the verify record is present (blank, honest) on step
    dicts from every settle path — including the plan-approval
    pause, where no check has run yet."""

    def test_paused_steps_carry_blank_verify(self):
        fx = _calc_report_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(_CALC_STEPS)
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        paused = agent.run("Build calc and report", max_iterations=5,
                           timeout=30, plan_mode=True)
        self.assertTrue(paused.needs_approval)
        for step in paused.steps:
            self.assertIn("verify", step)
            self.assertFalse(step["verify"]["ran"])
            self.assertIsNone(step["verify"]["ok"])
            self.assertEqual(step["verify"]["tests"], [])


if __name__ == "__main__":
    unittest.main()
