"""Pass #65 (W7): token metering in the coding loop + optional spend cap.

The defect, from the 2026-10-20 coding-agent survey:

- W7: ``CodingResult`` carried ``seconds`` but NO token fields, and
  neither ``agents/coding.py`` nor ``tools/edit_loop.py`` ever read
  ``.usage`` — every model response the loop paid for had its usage
  discarded.  Spend was bounded only by rounds x files; nothing could
  answer "what did that run cost?", and no caller could cap it.

These tests pin the new contract:

- metering is ALWAYS ON: every model call through the loop's chat
  funnel (plan / draft / edit-loop chat) lands on
  ``CodingResult.usage`` with prompt/completion/total tokens, a call
  count, and per-phase attribution; ``to_dict`` carries the same dict;
- diff-review spend surfaces too: ``reasoning.review_text`` reports
  the usage of its internal call through the pass-#63 ``status``
  channel (additively — ``status["usage"]``), and the loop meters it
  under the "review" phase;
- a response WITHOUT usage meters as 0 tokens for that call and
  increments ``usage_unreported`` — never a crash, never a fabricated
  number;
- metering alone never blocks: with no cap set, a run whose spend
  would exceed any plausible cap still completes;
- the OPTIONAL cap (``run(max_tokens=...)``, default None = off):
  when the running total crosses it, the loop stops BEFORE issuing
  the next model call and settles honestly — not-green with a
  cap-named error and unattempted steps "skipped" with the cap as
  their reason — UNLESS a verify-green already stands and no edit
  has landed since, in which case the verdict stands and the cap is
  noted on the result.  The cap never rewrites an earned verdict;
- the mission bridge forwards a step payload's ``max_tokens`` to the
  loop (absent = None = no cap), the settlement dict carries the
  usage evidence, and the cap survives plan serialization like
  ``accept`` does (pass #61).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nomorals.agents.coding as coding_module
import nomorals.agents.roles as roles
from nomorals.agents.coding import CodingAgent, CodingResult
from nomorals.agents.orchestrator import Plan, PlanStep
from nomorals.agents.reasoning import review_text
from nomorals.agents.roles import MissionCodingAgent
from nomorals.missions.runner import _serialize_plan
from nomorals.storage.db import Database


# ── fakes ────────────────────────────────────────────────────────────


def _usage(prompt, completion):
    return SimpleNamespace(prompt_tokens=prompt,
                           completion_tokens=completion,
                           total_tokens=prompt + completion)


class UsageRouter:
    """Content-routed fake LLM (the pass-#60 RuleRouter shape) whose
    responses CARRY usage: each chat() call pops the next
    (prompt, completion) pair from the script.  With
    ``with_usage=False`` the responses omit ``.usage`` entirely —
    the provider/test-double shape the meter must survive honestly."""

    def __init__(self, handler, usage_script=(), with_usage=True):
        self.handler = handler
        self._usage_script = list(usage_script)
        self.with_usage = with_usage
        self.calls: list[tuple[str, str]] = []

    def chat(self, messages, params=None):
        texts = [getattr(m, "content", "") or "" for m in messages]
        system = texts[0] if texts else ""
        user = texts[-1] if texts else ""
        self.calls.append((system, user))
        text = self.handler(system, user)
        resp = SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "empty")
        if self.with_usage:
            prompt, completion = (self._usage_script.pop(0)
                                  if self._usage_script else (1, 1))
            resp.usage = _usage(prompt, completion)
        return resp


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


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class FixtureRepo:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass65_"))
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


_CALC_STEPS = [
    {"path": "calc.py", "why": "the math", "new_file": True,
     "instructions": "implement add(a, b) returning the sum",
     "acceptance": "test_calc passes"},
    {"path": "report.py", "why": "reporting", "new_file": True,
     "instructions": "write summary() returning the calc result",
     "depends_on": ["s1"], "acceptance": "report imports cleanly"},
]


def _calc_handler(system, user):
    # pass #67: the run-start brief now fires for every task (it used
    # to be length-gated off for "Build a tiny calculator"), so the
    # fake model answers the brief prompt like a real one would —
    # one JSON brief, objective unchanged — and each usage script
    # below carries the brief's call as its first entry.
    if "structure raw owner requests" in system:
        return json.dumps({
            "objective": "Build a tiny calculator",
            "constraints": [],
            "success_criteria": ["test_calc passes"],
            "steps": ["implement calc.py", "write report.py"],
            "persona": "", "tool_hints": []})
    if "build planner" in system:
        return _steps_json(_CALC_STEPS)
    if "Write complete, runnable Python" in system:
        if "for the file 'calc.py'" in system:
            return _code("def add(a, b):\n    return a + b\n")
        if "for the file 'report.py'" in system:
            return _code("import calc\n\n\ndef summary():\n"
                         "    return calc.add(2, 3)\n")
    return ""


def _calc_repo(fx):
    fx.write("tests/__init__.py", "")
    fx.write("tests/test_calc.py",
             "import calc\n\n\ndef test_add():\n"
             "    assert calc.add(2, 3) == 5\n")
    fx.commit_all()
    return fx


class MeteringReviewer:
    """review_text double on the post-#65 contract: fills the #63
    status channel AND reports its call's usage through
    ``status["usage"]``.  ``with_usage=False`` plays the pre-#65
    double (status without usage) — the loop must count that call
    as unreported, never invent tokens for it."""

    def __init__(self, per_call=(10, 5), with_usage=True):
        self.calls = []
        self.per_call = per_call
        self.with_usage = with_usage

    def __call__(self, context, text, *, focus="", max_calls=None,
                 status=None):
        self.calls.append({"focus": focus, "max_calls": max_calls})
        if status is not None:
            status.update(available=True, error="")
            if self.with_usage:
                prompt, completion = self.per_call
                status["usage"] = {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                    "calls": 1, "usage_unreported": 0}
        return []


# ── (a)+(b) metering totals ──────────────────────────────────────────


@unittest.skipUnless(shutil.which("git"), "git not available")
class MeteringTotalsTests(unittest.TestCase):

    def test_a_usage_totals_sum_every_funneled_call(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        # router calls, in order: brief, planner, draft calc, draft
        # report (the brief leads since pass #67 — see _calc_handler)
        router = UsageRouter(_calc_handler, usage_script=[
            (10, 5), (100, 50), (200, 100), (300, 150)])
        reviewer = MeteringReviewer(per_call=(10, 5))
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        with mock.patch("nomorals.agents.reasoning.review_text",
                        reviewer):
            result = agent.run("Build a tiny calculator",
                               max_iterations=5, timeout=30,
                               plan_mode="never")
        self.assertTrue(result.ok, result.error)
        usage = result.usage
        # brief: 15 over 1 call (pass #67 — folded via _meter_external)
        # funnel: 150 + 300 + 450 = 900 over 3 calls
        # reviews: draft-review + per-diff review per file (2x2), and
        # the Phase C gate at DEEP tier (new top-level defs are a
        # public-api-change risk in the #63 table): per-file x2 +
        # combined x1 = 3 -> 7 review calls x 15 tokens = 105
        self.assertEqual(len(reviewer.calls), 7)
        self.assertEqual(usage["calls"], 11)
        self.assertEqual(usage["prompt_tokens"], 610 + 70)
        self.assertEqual(usage["completion_tokens"], 305 + 35)
        self.assertEqual(usage["total_tokens"], 1020)
        self.assertEqual(usage["usage_unreported"], 0)
        by_phase = usage["by_phase"]
        self.assertEqual(by_phase["brief"]["total_tokens"], 15)
        self.assertEqual(by_phase["brief"]["calls"], 1)
        self.assertEqual(by_phase["plan"]["total_tokens"], 150)
        self.assertEqual(by_phase["plan"]["calls"], 1)
        self.assertEqual(by_phase["edit"]["total_tokens"], 750)
        self.assertEqual(by_phase["edit"]["calls"], 2)
        self.assertEqual(by_phase["review"]["total_tokens"], 105)
        self.assertEqual(by_phase["review"]["calls"], 7)
        # to_dict carries the same metering (chat/mission consumers)
        self.assertEqual(result.to_dict()["usage"], usage)
        # no cap was set: nothing about a cap on the result
        self.assertNotIn("cap", usage)
        self.assertFalse(usage.get("capped", False))

    def test_b_responses_without_usage_count_unreported(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        router = UsageRouter(_calc_handler, with_usage=False)
        reviewer = MeteringReviewer(with_usage=False)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        with mock.patch("nomorals.agents.reasoning.review_text",
                        reviewer):
            result = agent.run("Build a tiny calculator",
                               max_iterations=5, timeout=30,
                               plan_mode="never")
        # metering never breaks the run, and never invents a number
        self.assertTrue(result.ok, result.error)
        usage = result.usage
        self.assertEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["prompt_tokens"], 0)
        self.assertEqual(usage["completion_tokens"], 0)
        # 11 calls: the pass-#67 brief + 3 funnel + 7 reviews, every
        # one of them unreported rather than estimated
        self.assertEqual(usage["calls"], 11)
        self.assertEqual(usage["usage_unreported"], 11)
        self.assertEqual(usage["by_phase"]["brief"]["usage_unreported"],
                         1)
        self.assertEqual(usage["by_phase"]["review"]["usage_unreported"],
                         7)

    def test_result_usage_defaults_are_additive(self):
        bare = CodingResult(ok=True, iterations=0)
        self.assertEqual(bare.usage, {})
        self.assertEqual(bare.to_dict()["usage"], {})


# ── (c)+(d)+(e) the spend cap ────────────────────────────────────────


@unittest.skipUnless(shutil.which("git"), "git not available")
class SpendCapTests(unittest.TestCase):

    def test_c_no_cap_metering_never_blocks(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        router = UsageRouter(_calc_handler, usage_script=[
            (10 ** 6, 10 ** 6), (10 ** 6, 10 ** 6), (10 ** 6, 10 ** 6)])
        reviewer = MeteringReviewer(per_call=(10 ** 6, 10 ** 6))
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        with mock.patch("nomorals.agents.reasoning.review_text",
                        reviewer):
            result = agent.run("Build a tiny calculator",
                               max_iterations=5, timeout=30,
                               plan_mode="never")
        self.assertTrue(result.ok, result.error)
        self.assertGreater(result.usage["total_tokens"], 10 ** 7)
        self.assertFalse(result.usage.get("capped", False))

    def test_d_cap_stops_before_next_model_call(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        # the brief spends 15 (pass #67), the planner 150, the first
        # draft 300 -> total 465; the cap (200) is crossed by the
        # draft, so the NEXT model call (the attempt-1 draft review)
        # must never be issued.
        router = UsageRouter(_calc_handler, usage_script=[
            (10, 5), (100, 50), (200, 100), (300, 150)])
        reviewer = MeteringReviewer(per_call=(10, 5))
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        with mock.patch("nomorals.agents.reasoning.review_text",
                        reviewer):
            result = agent.run("Build a tiny calculator",
                               max_iterations=5, timeout=30,
                               plan_mode="never", max_tokens=200)
        self.assertFalse(result.ok)
        # stopped BEFORE the next call: brief + planner + one draft,
        # and the reviewer was never reached at all
        self.assertEqual(len(router.calls), 3)
        self.assertEqual(len(reviewer.calls), 0)
        # the error names the cap and the spend; usage still reported
        self.assertIn("spend cap", result.error)
        self.assertIn("465", result.error)
        self.assertIn("200", result.error)
        self.assertEqual(result.usage["total_tokens"], 465)
        self.assertEqual(result.usage["cap"], 200)
        self.assertTrue(result.usage["capped"])
        # per-step status preserved: nothing landed, every step is
        # skipped with the cap as its reason
        by_path = {s["path"]: s for s in result.steps}
        self.assertEqual(by_path["calc.py"]["status"], "skipped")
        self.assertEqual(by_path["report.py"]["status"], "skipped")
        for step in result.steps:
            self.assertIn("spend cap", step["reason"])

    def test_e_cap_crossed_after_green_verify_verdict_stands(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        # the brief spends 15 first (pass #67); funnel: 150 + 300 +
        # 300 = 750; four pre-gate reviews x 15 bring the running
        # total to 825 when the Phase C gate review is due.  Cap 815
        # is crossed only there — after the verify round already went
        # green and no edit has landed since.
        router = UsageRouter(_calc_handler, usage_script=[
            (10, 5), (100, 50), (200, 100), (200, 100)])
        reviewer = MeteringReviewer(per_call=(10, 5))
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        with mock.patch("nomorals.agents.reasoning.review_text",
                        reviewer):
            result = agent.run("Build a tiny calculator",
                               max_iterations=5, timeout=30,
                               plan_mode="never", max_tokens=815)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.verification, "tests")
        self.assertEqual(len(router.calls), 4)
        self.assertEqual(len(reviewer.calls), 4)  # gate never issued
        by_path = {s["path"]: s for s in result.steps}
        self.assertEqual(by_path["calc.py"]["status"], "done")
        self.assertEqual(by_path["report.py"]["status"], "done")
        # the cap is noted, and the gate honestly reports it never
        # completed — the green verdict itself is untouched
        self.assertTrue(result.usage["capped"])
        self.assertEqual(result.usage["total_tokens"], 825)
        self.assertFalse(result.review["available"])
        self.assertIn("spend cap",
                      " ".join(result.review.get("notes", [])))


# ── review_text surfaces its own usage (real reasoning path) ────────


class ReviewTextUsageTests(unittest.TestCase):

    def test_review_text_reports_usage_through_status(self):
        def handler(system, user):
            return json.dumps({"passed": True, "flaws": []})
        router = UsageRouter(handler, usage_script=[(40, 8)])
        status: dict = {}
        flaws = review_text(StubContext(router), "some artifact",
                            status=status)
        self.assertEqual(flaws, [])
        self.assertTrue(status["available"])
        usage = status["usage"]
        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["prompt_tokens"], 40)
        self.assertEqual(usage["completion_tokens"], 8)
        self.assertEqual(usage["total_tokens"], 48)
        self.assertEqual(usage["usage_unreported"], 0)

    def test_review_text_without_usage_counts_unreported(self):
        def handler(system, user):
            return json.dumps({"passed": True, "flaws": []})
        router = UsageRouter(handler, with_usage=False)
        status: dict = {}
        review_text(StubContext(router), "some artifact", status=status)
        usage = status["usage"]
        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["usage_unreported"], 1)


# ── (f) the mission bridge ───────────────────────────────────────────


class _FakeLoop:
    """Stand-in for the real CodingAgent at the bridge boundary
    (the pass-#61 pattern): records run() kwargs, returns a scripted
    CodingResult."""

    scripted: CodingResult | None = None
    instances: list["_FakeLoop"] = []

    def __init__(self, context, root=None):
        self.context = context
        self.root = root
        self.run_calls: list[tuple[str, dict]] = []
        _FakeLoop.instances.append(self)

    def run(self, task, **kwargs):
        self.run_calls.append((task, kwargs))
        assert _FakeLoop.scripted is not None
        return _FakeLoop.scripted


def _bridge_ctx():
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db, tools=None, memory=None, router=None,
                           settings=SimpleNamespace(), extras={})


class BridgeCapTests(unittest.TestCase):

    def setUp(self):
        _FakeLoop.scripted = CodingResult(
            ok=True, iterations=1, files=["a.py"], verification="tests",
            usage={"prompt_tokens": 30, "completion_tokens": 12,
                   "total_tokens": 42, "calls": 2,
                   "usage_unreported": 0, "by_phase": {}})
        _FakeLoop.instances = []
        patcher = mock.patch.object(coding_module, "CodingAgent",
                                    _FakeLoop)
        patcher.start()
        self.addCleanup(patcher.stop)

    @property
    def loop(self):
        self.assertEqual(len(_FakeLoop.instances), 1)
        return _FakeLoop.instances[0]

    def test_payload_cap_reaches_run_and_usage_settles(self):
        agent = MissionCodingAgent(context=_bridge_ctx())
        agent.bind_step(PlanStep(
            name="build", goal="build it", role="coding",
            payload={"max_tokens": 1234}))
        out = agent.work("build it")
        _, kwargs = self.loop.run_calls[0]
        self.assertEqual(kwargs["max_tokens"], 1234)
        # the settlement carries the run's usage evidence additively
        self.assertEqual(out["usage"]["total_tokens"], 42)
        self.assertEqual(out["usage"]["calls"], 2)

    def test_absent_cap_forwards_none(self):
        agent = MissionCodingAgent(context=_bridge_ctx())
        agent.work("build it")
        _, kwargs = self.loop.run_calls[0]
        self.assertIn("max_tokens", kwargs)
        self.assertIsNone(kwargs["max_tokens"])

    def test_cap_survives_plan_serialization(self):
        plan = Plan(goal="g", steps=[PlanStep(
            name="build", goal="build it", role="coding",
            payload={"max_tokens": 777, "accept": "make check"})])
        entry = _serialize_plan(plan)[0]
        self.assertEqual(entry["policy"]["max_tokens"], 777)
        self.assertEqual(entry["policy"]["accept"], "make check")


if __name__ == "__main__":
    unittest.main()
