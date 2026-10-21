"""Pass #60: the plan is a step plan (W4), and the repair budget no
longer collapses with plan size (W2).

The defects, from the 2026-10-20 coding-agent survey:

- W4: ``_plan_files`` emitted ``{path, why, new_file}`` only — no
  per-step instructions, dependencies, or acceptance criteria;
  ``CodingResult`` carried no per-step status; the plan was never
  revised mid-run, so a weak step could only be named post-mortem.
- W2: ``per_file = max(1, max_iterations // len(plan))`` — at the chat
  caller's budget (5), a 5-file plan got ONE draft attempt per file and
  ZERO repair rounds.  Multi-file (real) work got the least iteration.

These tests pin the new contract:

- steps carry instructions / depends_on / acceptance, and the
  instructions + acceptance ride the step's draft calls;
- ``CodingResult.steps`` reports every step's outcome
  (done/failed/skipped + reason + attempts), including steps
  superseded by a re-plan;
- on a step's repeated failure the run re-plans the remaining steps
  exactly once (recorded in ``result.replans``) instead of grinding
  the dead plan;
- a step whose dependency failed is skipped, naming the dependency;
- the budget is a per-file minimum (draft + >=1 repair) plus a shared
  repair pool: a 5-file plan at the chat budget drafts every file
  3 times (2 minimum + 1 pool draw), where the old division gave 1;
- a 1-file plan still cannot exceed the caller's rounds — the pool
  funds repairs for failing steps, it does not inflate the budget.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.coding import CodingAgent


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class RuleRouter:
    """Content-routed fake LLM: the handler answers by what the call
    is asking (planner / whole-file draft / surgical edit), so call
    ORDER never matters — only content does."""

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

    def planner_calls(self):
        return [c for c in self.calls if "build planner" in c[0]]

    def drafts(self, filename):
        return [c for c in self.calls
                if "Write complete, runnable Python" in c[0]
                and f"for the file '{filename}'" in c[0]]

    def edits(self, filename):
        return [c for c in self.calls
                if "surgical code editor" in c[0]
                and f"File: {filename}" in c[1]]


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
        self.root = Path(tempfile.mkdtemp(prefix="pass60_"))
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


def _calc_repo(fx):
    fx.write("tests/__init__.py", "")
    fx.write("tests/test_calc.py",
             "import calc\n\n\ndef test_add():\n"
             "    assert calc.add(2, 3) == 5\n")
    fx.commit_all()
    return fx


_CALC_STEPS = [
    {"path": "calc.py", "why": "the math", "new_file": True,
     "instructions": "implement add(a, b) returning the sum",
     "acceptance": "test_calc passes"},
    {"path": "report.py", "why": "reporting", "new_file": True,
     "instructions": "write summary() returning the calc result",
     "depends_on": ["s1"], "acceptance": "report imports cleanly"},
]


def _calc_handler(system, user):
    if "build planner" in system:
        return _steps_json(_CALC_STEPS)
    if "Write complete, runnable Python" in system:
        if "for the file 'calc.py'" in system:
            return _code("def add(a, b):\n    return a + b\n")
        if "for the file 'report.py'" in system:
            return _code("import calc\n\n\ndef summary():\n"
                         "    return calc.add(2, 3)\n")
    return ""


@unittest.skipUnless(shutil.which("git"), "git not available")
class StepPlanReportTests(unittest.TestCase):
    """W4: steps carry criteria into the run and outcomes out of it."""

    def test_result_defaults_are_additive(self):
        from nomorals.agents.coding import CodingResult
        bare = CodingResult(ok=True, iterations=0)
        self.assertEqual(bare.steps, [])
        self.assertEqual(bare.replans, 0)
        self.assertEqual(bare.to_dict()["steps"], [])
        self.assertEqual(bare.to_dict()["replans"], 0)

    def test_green_run_reports_steps_done_with_criteria(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        router = RuleRouter(_calc_handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Build a tiny calculator", max_iterations=5,
                           timeout=30, plan_mode="never")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.replans, 0)
        self.assertEqual(len(result.steps), 2)
        by_path = {s["path"]: s for s in result.steps}
        calc, report = by_path["calc.py"], by_path["report.py"]
        self.assertEqual(calc["status"], "done")
        self.assertEqual(report["status"], "done")
        self.assertEqual(calc["attempts"], 1)
        self.assertEqual(calc["id"], "s1")
        self.assertEqual(report["depends_on"], ["s1"])
        self.assertEqual(calc["acceptance"], "test_calc passes")
        self.assertTrue(calc["instructions"].startswith("implement add"))
        # the step's instructions + acceptance rode its draft call
        draft_users = [u for _, u in router.drafts("calc.py")]
        self.assertTrue(any(
            "implement add(a, b) returning the sum" in u
            and "accepted when: test_calc passes" in u
            for u in draft_users))
        # to_dict carries the same report (chat/mission consumers)
        self.assertEqual(result.to_dict()["steps"], result.steps)

    def test_pause_reports_planned_steps_then_execute_carries_them(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        router = RuleRouter(_calc_handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        paused = agent.run("Build a tiny calculator", max_iterations=5,
                           timeout=30, plan_mode=True)
        self.assertTrue(paused.needs_approval)
        self.assertEqual([s["status"] for s in paused.steps],
                         ["planned", "planned"])
        # the approval text shows the steps, not just the file list
        self.assertIn("implement add(a, b) returning the sum",
                      paused.plan_text)
        self.assertIn("accepted when: test_calc passes", paused.plan_text)
        result = agent.approve_and_execute(paused.plan_id)
        self.assertTrue(result.ok, result.error)
        by_path = {s["path"]: s for s in result.steps}
        self.assertEqual(by_path["calc.py"]["status"], "done")
        self.assertEqual(by_path["calc.py"]["acceptance"],
                         "test_calc passes")
        self.assertEqual(by_path["report.py"]["depends_on"], ["s1"])
        # exactly one planner call: execution replays the approved plan
        self.assertEqual(len(router.planner_calls()), 1)


@unittest.skipUnless(shutil.which("git"), "git not available")
class BudgetFloorTests(unittest.TestCase):
    """W2: minimum per-file budget + shared repair pool."""

    def _five_file_repo(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("tests/__init__.py", "")
        fx.write("tests/test_a.py",
                 "import a\n\n\ndef test_value():\n"
                 "    assert a.VALUE == 1\n")
        fx.commit_all()
        return fx

    def test_five_file_plan_keeps_repair_rounds_at_chat_budget(self):
        fx = self._five_file_repo()
        steps = [{"path": f"{ch}.py", "why": "module", "new_file": True,
                  "instructions": f"write module {ch}",
                  "acceptance": "suite passes"}
                 for ch in "abcde"]

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(steps)  # initial plan AND the re-plan
            if "Write complete, runnable Python" in system:
                return _code("VALUE = 0\n")  # always wrong: never green
            if "surgical code editor" in system:
                # after the re-plan the files exist, so the loop repairs
                # them through the edit protocol — still wrong, still
                # a real (spending) attempt
                return _edits([{"old_text": "VALUE = 0",
                                "new_text": "VALUE = -0"}])
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Build the five modules", max_iterations=5,
                           timeout=30, plan_mode="never")
        self.assertFalse(result.ok)
        # the old division (5 // 5) gave each file ONE attempt, total.
        # Now: 2 guaranteed (draft + repair) + 1 shared-pool draw each
        # (the pool of 5 is spent across the 5 failing files in round
        # 3; round 4 finds the pool empty and stops).
        for ch in "abcde":
            attempts = (len(router.drafts(f"{ch}.py"))
                        + len(router.edits(f"{ch}.py")))
            self.assertEqual(attempts, 3, f"{ch}.py attempt count")
        self.assertEqual(len(router.planner_calls()), 2)
        self.assertEqual(result.replans, 1)
        self.assertIn("files that did not converge", result.error)
        by_path = {s["path"]: s for s in result.steps}
        for ch in "abcde":
            self.assertEqual(by_path[f"{ch}.py"]["status"], "failed")
            self.assertEqual(by_path[f"{ch}.py"]["attempts"], 3)

    def test_single_file_plan_cannot_exceed_the_callers_rounds(self):
        fx = self._five_file_repo()
        steps = [{"path": "a.py", "why": "module", "new_file": True,
                  "instructions": "write module a",
                  "acceptance": "test_a passes"}]

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(steps)
            if "Write complete, runnable Python" in system:
                return _code("VALUE = 0\n")
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Build module a", max_iterations=3,
                           timeout=30, plan_mode="never")
        self.assertFalse(result.ok)
        # the pool funds repairs, never extra rounds: 3 rounds, 3 drafts
        self.assertEqual(len(router.drafts("a.py")), 3)
        self.assertEqual(result.replans, 0)
        self.assertEqual(result.steps[0]["status"], "failed")
        self.assertEqual(result.steps[0]["attempts"], 3)


@unittest.skipUnless(shutil.which("git"), "git not available")
class ReplanRecoveryTests(unittest.TestCase):
    """W4: a repeatedly failing step triggers ONE recorded re-plan."""

    def test_replan_supersedes_dead_step_and_run_recovers(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("extra.py", "# placeholder\n")
        _calc_repo(fx)
        v1 = [
            {"path": "calc.py", "why": "math", "new_file": True,
             "instructions": "implement add by repeated subtraction",
             "acceptance": "test_calc passes"},
            {"path": "extra.py", "why": "tidy", "new_file": False,
             "instructions": "tidy the placeholder module",
             "depends_on": ["s1"], "acceptance": "extra still imports"},
        ]
        v2 = [
            {"path": "calc.py", "why": "math, fixed approach",
             "new_file": True,
             "instructions": "implement add as a + b",
             "acceptance": "test_calc passes"},
        ]

        def handler(system, user):
            if "build planner" in system:
                return _steps_json(v2 if "RE-PLAN" in user else v1)
            if "Write complete, runnable Python" in system:
                if "as a + b" in user:
                    return _code("def add(a, b):\n    return a + b\n")
                return _code("def add(a, b):\n    return a - b\n")
            if "surgical code editor" in system:
                if "File: calc.py" in user and "as a + b" in user:
                    # the re-planned step arrives as a surgical edit
                    # (calc.py exists by now): apply the new approach
                    return _edits([{"old_text": "return a - b",
                                    "new_text": "return a + b"}])
                return _edits([])  # extra.py: model judges no change
            return ""

        router = RuleRouter(handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("Build the calculator core", max_iterations=5,
                           timeout=30, plan_mode="never")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.replans, 1)
        self.assertEqual(len(router.planner_calls()), 2)
        by_path = {s["path"]: s for s in result.steps}
        self.assertEqual(by_path["calc.py"]["status"], "done")
        self.assertEqual(by_path["extra.py"]["status"], "skipped")
        self.assertEqual(by_path["extra.py"]["reason"],
                         "superseded by re-plan")
        self.assertEqual(by_path["extra.py"]["attempts"], 0)


@unittest.skipUnless(shutil.which("git"), "git not available")
class DependencySkipTests(unittest.TestCase):
    """W4: a step whose dependency never lands is skipped, naming it."""

    def test_dependent_step_skipped_when_dependency_fails(self):
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
        by_path = {s["path"]: s for s in result.steps}
        self.assertEqual(by_path["a.py"]["status"], "failed")
        self.assertEqual(by_path["a.py"]["attempts"], 4)
        self.assertEqual(by_path["b.py"]["status"], "skipped")
        self.assertEqual(by_path["b.py"]["attempts"], 0)
        self.assertIn("dependency s1", by_path["b.py"]["reason"])
        self.assertEqual(result.replans, 0)
        # the failed run rolled the tree back
        self.assertEqual((fx.root / "a.py").read_text(), "VALUE = 0\n")


if __name__ == "__main__":
    unittest.main()
