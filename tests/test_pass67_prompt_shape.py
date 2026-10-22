"""Pass #67: the prompt-shape gating class.

The defect (named at pass #63, fixed there only inside the coding
loop's reviewer): a capability, review, or pre-flight surface is
suppressed — silently never runs — because of the SHAPE of the
user's prompt (its length, a keyword/regex "complexity" guess, a
constant standing in for judgment) instead of running on structural
facts about the WORK, or running always with depth tiered by
structure.  The companion sin: the suppressed surface's absence is
indistinguishable from it having run.

These tests pin the post-#67 contract per fixed site:

- ``BriefAgent.refine`` attempts the model brief for EVERY request
  when a brain is available (a short, single-clause ask most of
  all — it was the one silently skipped); with no brain the brief
  is honestly labelled ``by="heuristic"`` — the old ``by="skip"``
  state is gone, and a model brief's usage rides on the brief so
  the coding loop can meter it (pass #65's channel).
- The coding loop folds that brief usage under the ``"brief"``
  phase — a short coding task's run now shows the brief call it
  actually made.
- ``DevonAgent._reason_review_plan`` reviews every non-empty plan
  when the owner's reasoning mode permits — a one-step plan from a
  three-word prompt gets the same pre-flight as a nine-step plan.
- The partner brain's reasoning digest runs on every owner turn in
  auto mode at a focused budget tier; ``always`` keeps the full
  budget; ``off`` still blocks it entirely (the owner's knob).
- Projects' failure consult (``_course_pivot``) and the researcher's
  notification critique (``_reasoning_critique``) run in auto mode:
  their triggers are structural (a step failed; an item scored
  worth notifying), so the mode flag may only veto, never judge.
- ``reasoning_permitted`` is the honest form of the old
  ``reasoning_enabled(context, complex_ok=True)`` constant: a pure
  owner-mode check.  ``reasoning_enabled`` itself is untouched —
  its judgment-parameter contract is pinned by wave 85.
- ``plan_mode.is_complex`` is pinned as STRUCTURE, not shape: it
  classifies the produced plan (file/module counts) and never
  reads the prompt; the plan-mode knob itself is the owner's.

Hermetic throughout: fake routers/brains, stub contexts, tmp dirs.
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

from tests.test_search_and_model import _FakeRouter

import nomorals.agents.reasoning as reasoning_module
from nomorals.agents.brief import BriefAgent
from nomorals.agents.context import build_context
from nomorals.agents.devon import DevonAgent
from nomorals.agents.plan_mode import is_complex
from nomorals.agents.projects import ProjectManager
from nomorals.agents.reasoning import ReasoningAgent
from nomorals.agents.researcher import ResearchAgent
from nomorals.agents.structuring import PromptArchitect
from nomorals.core.config import load_settings


# ── shared fixtures ──────────────────────────────────────────────────


def _settings(tmp: str):
    return load_settings(overrides={"home": tmp,
                                    "partner.platforms": "local",
                                    "chat.local_enabled": "true"})


def _make_context(tmp: str):
    context = build_context(_settings(tmp), with_executor=False,
                            with_tools=False, with_router=False,
                            with_memory=False)
    context.router = _FakeRouter()
    return context


def _lock_power_off(context) -> None:
    """Pin power mode INACTIVE so auto-mode behavior is observable
    (the default-on posture would otherwise force reasoning on)."""
    from nomorals.agents.power import power_mode_for
    power_mode_for(context).lock(actor="test")


class _StubBrain:
    """brain_for double: chat_json returns canned data + a response
    whose usage the brief must surface."""

    def __init__(self, data, usage=(11, 7)):
        self.data = data
        self.calls = 0
        self._usage = usage

    def chat_json(self, messages, **_kw):
        self.calls += 1
        prompt, completion = self._usage
        resp = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=prompt + completion))
        return self.data, resp


_BRIEF_JSON = {
    "objective": "Fix the parser bug in the tokenizer",
    "constraints": ["no public API change"],
    "success_criteria": ["parser tests pass"],
    "steps": ["patch the tokenizer", "run the parser tests"],
    "persona": "",
    "tool_hints": [],
}


# ── the mode predicate ───────────────────────────────────────────────


class ReasoningPermittedTests(unittest.TestCase):
    """reasoning_permitted: the owner-mode check with NO judgment
    parameter — False only when the owner set reasoning off."""

    def _ctx(self, mode):
        return SimpleNamespace(
            settings=SimpleNamespace(reasoning_mode=mode))

    def test_off_is_the_only_veto(self):
        from nomorals.agents.reasoning import reasoning_permitted
        self.assertFalse(reasoning_permitted(self._ctx("off")))

    def test_auto_always_and_unset_all_permit(self):
        from nomorals.agents.reasoning import reasoning_permitted
        self.assertTrue(reasoning_permitted(self._ctx("auto")))
        self.assertTrue(reasoning_permitted(self._ctx("always")))
        self.assertTrue(reasoning_permitted(self._ctx("")))
        self.assertTrue(reasoning_permitted(SimpleNamespace()))

    def test_reasoning_enabled_contract_unchanged(self):
        # wave 85 pins this; #67 must not move it: in auto with power
        # inactive, the judgment parameter still decides.
        tmp = tempfile.TemporaryDirectory(prefix="nm-p67-")
        self.addCleanup(tmp.cleanup)
        context = _make_context(tmp.name)
        self.addCleanup(context.close)
        _lock_power_off(context)
        context.settings.reasoning_mode = "auto"
        self.assertFalse(
            reasoning_module.reasoning_enabled(context, complex_ok=False))
        self.assertTrue(
            reasoning_module.reasoning_enabled(context, complex_ok=True))
        context.settings.reasoning_mode = "off"
        self.assertFalse(
            reasoning_module.reasoning_enabled(context, complex_ok=True))


# ── (1) the brief gate ───────────────────────────────────────────────


class BriefGateTests(unittest.TestCase):

    def test_short_request_gets_the_model_brief(self):
        # "fix the parser bug": 18 chars, one clause — should_brief()
        # said no, so refine() returned by="skip" and the model was
        # never consulted.  Post-#67 the structuring RUNS.
        brain = _StubBrain(dict(_BRIEF_JSON))
        agent = BriefAgent(SimpleNamespace(router=object()))
        with mock.patch("nomorals.llm.brain.brain_for",
                        return_value=brain):
            brief = agent.refine("fix the parser bug", kind="code")
        self.assertEqual(brain.calls, 1)
        self.assertEqual(brief.by, "model")
        self.assertEqual(brief.objective,
                         "Fix the parser bug in the tokenizer")
        # the run honestly records its cost for the #65 meter
        self.assertIsNotNone(brief.usage)
        self.assertEqual(brief.usage["calls"], 1)
        self.assertEqual(brief.usage["total_tokens"], 18)
        self.assertEqual(brief.usage["usage_unreported"], 0)
        self.assertEqual(brief.to_dict()["usage"], brief.usage)

    def test_model_brief_without_usage_counts_unreported(self):
        brain = _StubBrain(dict(_BRIEF_JSON))
        brain._usage = (0, 0)
        agent = BriefAgent(SimpleNamespace(router=object()))
        with mock.patch("nomorals.llm.brain.brain_for",
                        return_value=brain):
            brief = agent.refine("fix the parser bug", kind="code")
        self.assertEqual(brief.by, "model")
        self.assertEqual(brief.usage["calls"], 1)
        self.assertEqual(brief.usage["total_tokens"], 0)
        self.assertEqual(brief.usage["usage_unreported"], 1)

    def test_no_brain_reports_heuristic_never_skip(self):
        # no router and no env brain: the surface cannot run — the
        # brief must SAY so (by="heuristic"), not pass as structured
        # and not vanish into the old "skip" state.
        agent = BriefAgent(SimpleNamespace(router=None))
        with mock.patch.dict("os.environ", {}, clear=False):
            brief = agent.refine("fix the parser bug", kind="code")
        self.assertEqual(brief.by, "heuristic")
        self.assertEqual(brief.objective, "fix the parser bug")
        self.assertIsNone(brief.usage)

    def test_long_request_still_gets_the_model_brief(self):
        brain = _StubBrain(dict(_BRIEF_JSON))
        agent = BriefAgent(SimpleNamespace(router=object()))
        long_req = ("Please fix the parser bug in the tokenizer, and "
                    "make sure the regression is covered by a test, "
                    "and keep the public API exactly as it is today.")
        with mock.patch("nomorals.llm.brain.brain_for",
                        return_value=brain):
            brief = agent.refine(long_req, kind="code")
        self.assertEqual(brief.by, "model")
        self.assertEqual(brain.calls, 1)


# ── (3) the orchestrator plan pre-flight ─────────────────────────────


class DevonPlanReviewGateTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-p67-")
        self.context = _make_context(self.tmp.name)
        _lock_power_off(self.context)
        self.agent = DevonAgent.__new__(DevonAgent)
        self.agent.context = self.context
        self.agent._tools = {"fs_read": object()}
        self.agent.max_steps = 8

    def tearDown(self):
        self.context.close()
        self.tmp.cleanup()

    def _one_step_plan(self):
        return [{"tool": "fs_read", "args": {"path": "x.py"},
                 "why": "look before deleting"}]

    def test_short_prompt_one_step_plan_is_reviewed_in_auto(self):
        # pre-#67: looks_complex("clean up") is False and len(steps)
        # is 1, so complex_ok was False and auto mode returned the
        # plan UNREVIEWED.  The pre-flight now runs on the work —
        # every non-empty plan — not on the prompt's shape.
        self.context.settings.reasoning_mode = "auto"
        with mock.patch.object(reasoning_module, "review_text",
                               return_value=[]) as review:
            out = self.agent._reason_review_plan(
                "clean up", self._one_step_plan())
        self.assertEqual(review.call_count, 1)
        self.assertEqual(out, self._one_step_plan())

    def test_off_mode_still_blocks_the_review(self):
        self.context.settings.reasoning_mode = "off"
        with mock.patch.object(reasoning_module, "review_text",
                               return_value=[]) as review:
            self.agent._reason_review_plan(
                "clean up", self._one_step_plan())
        self.assertEqual(review.call_count, 0)

    def test_empty_plan_is_a_no_work_guard_not_a_gate(self):
        self.context.settings.reasoning_mode = "auto"
        with mock.patch.object(reasoning_module, "review_text",
                               return_value=[]) as review:
            out = self.agent._reason_review_plan("clean up", [])
        self.assertEqual(out, [])
        self.assertEqual(review.call_count, 0)


# ── (4) the partner brain's answer review ────────────────────────────


class _FakeEngine:
    instances: list = []

    def __init__(self, context, **kw):
        self.kw = kw
        _FakeEngine.instances.append(self)

    def reason(self, text, strategy="auto"):
        return SimpleNamespace(
            answer="the ledger balances", stopped="complete",
            trace=[SimpleNamespace(kind="note",
                                   text="checked the ledger twice")])


class PartnerDigestGateTests(unittest.TestCase):

    def setUp(self):
        _FakeEngine.instances = []
        from nomorals.agents.partner.brain import PartnerBrain
        self.brain = PartnerBrain.__new__(PartnerBrain)
        self.brain.context = SimpleNamespace()

    def _digest(self, mode, text):
        self.brain.settings = SimpleNamespace(
            partner=SimpleNamespace(reasoning=mode))
        with mock.patch.object(reasoning_module, "ReasoningEngine",
                               _FakeEngine):
            return self.brain._reasoning_digest(text)

    def test_short_message_is_reviewed_in_auto(self):
        # "sounds good, do it": short, no "?", no complexity keywords
        # — looks_complex() said no and the digest never ran.
        out = self._digest("auto", "sounds good, do it")
        self.assertIn("checked the ledger twice", out)
        self.assertEqual(len(_FakeEngine.instances), 1)

    def test_auto_uses_the_focused_budget_tier(self):
        self._digest("auto", "sounds good, do it")
        kw = _FakeEngine.instances[0].kw
        self.assertEqual(kw.get("max_llm_calls"), 3)
        self.assertEqual(kw.get("max_seconds"), 30.0)

    def test_always_keeps_the_full_budget(self):
        out = self._digest("always", "sounds good, do it")
        self.assertIn("checked the ledger twice", out)
        kw = _FakeEngine.instances[0].kw
        self.assertEqual(kw.get("max_llm_calls"), 6)
        self.assertEqual(kw.get("max_seconds"), 45.0)

    def test_off_mode_still_blocks_the_digest(self):
        out = self._digest("off", "why is the sky blue, really?")
        self.assertEqual(out, "")
        self.assertEqual(_FakeEngine.instances, [])


# ── (5) projects + researcher pre-flights in auto ────────────────────


class ProjectsPivotGateTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-p67-")
        self.context = _make_context(self.tmp.name)
        _lock_power_off(self.context)
        self.manager = ProjectManager(self.context)

    def tearDown(self):
        self.context.close()
        self.tmp.cleanup()

    def test_failure_consult_runs_in_auto(self):
        # the trigger is structural — a step FAILED — so auto mode
        # may not suppress the consult (pre-#67 the bare
        # reasoning_enabled() call made it dead code in auto).
        self.context.settings.reasoning_mode = "auto"
        pivot = {"should_pivot": True, "pivot": "use the sandbox",
                 "rationale": "direct runs keep failing"}
        with mock.patch.object(ReasoningAgent, "course_correct",
                               return_value=pivot) as consult:
            out = self.manager._course_pivot("the fragile step",
                                             "boom", 1)
        self.assertEqual(consult.call_count, 1)
        self.assertEqual(out, pivot)

    def test_off_mode_still_blocks_the_consult(self):
        self.context.settings.reasoning_mode = "off"
        with mock.patch.object(ReasoningAgent, "course_correct",
                               return_value={}) as consult:
            out = self.manager._course_pivot("the fragile step",
                                             "boom", 1)
        self.assertEqual(out, {})
        self.assertEqual(consult.call_count, 0)


class ResearcherCritiqueGateTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-p67-")
        self.context = _make_context(self.tmp.name)
        _lock_power_off(self.context)
        self.agent = ResearchAgent(self.context)

    def tearDown(self):
        self.context.close()
        self.tmp.cleanup()

    def test_critique_runs_in_auto(self):
        # the call site already tiers by the work's measured score
        # (only about-to-notify items reach here); auto mode may not
        # veto the pass itself.
        self.context.settings.reasoning_mode = "auto"
        with mock.patch.object(
                ReasoningAgent, "advise",
                return_value="too vague — no concrete action") as adv:
            out = self.agent._reasoning_critique(
                "quantum sensors", "fund a quantum sensing lab")
        self.assertEqual(adv.call_count, 1)
        self.assertIn("too vague", out)

    def test_off_mode_still_blocks_the_critique(self):
        self.context.settings.reasoning_mode = "off"
        with mock.patch.object(ReasoningAgent, "advise",
                               return_value="too vague") as adv:
            out = self.agent._reasoning_critique(
                "quantum sensors", "fund a quantum sensing lab")
        self.assertEqual(out, "")
        self.assertEqual(adv.call_count, 0)


# ── (6) the structuring constant ─────────────────────────────────────


class StructuringPolishGateTests(unittest.TestCase):

    _BASE = {"intent": "raw ask", "subgoals": ["do the thing"],
             "inputs": [], "constraints": [], "acceptance": [],
             "tools": [], "risks": []}

    def _ctx(self, mode):
        return SimpleNamespace(
            router=object(),
            settings=SimpleNamespace(reasoning_mode=mode))

    def test_polish_runs_in_auto(self):
        brain = _StubBrain({"intent": "sharper intent",
                            "subgoals": ["step one"]})
        architect = PromptArchitect(self._ctx("auto"))
        with mock.patch("nomorals.llm.brain.brain_for",
                        return_value=brain):
            out = architect._model_polish("do the thing",
                                          dict(self._BASE))
        self.assertIsNotNone(out)
        self.assertEqual(out["intent"], "sharper intent")

    def test_polish_never_consults_the_judgment_parameter(self):
        # the pre-#67 form: reasoning_enabled(context, complex_ok=True)
        # — a constant wearing a judgment's clothes.  The migrated
        # site consults only the owner-mode predicate, so a poisoned
        # reasoning_enabled must not matter.
        brain = _StubBrain({"intent": "sharper intent",
                            "subgoals": ["step one"]})
        architect = PromptArchitect(self._ctx("auto"))
        with mock.patch.object(
                reasoning_module, "reasoning_enabled",
                side_effect=AssertionError("judgment gate consulted")), \
             mock.patch("nomorals.llm.brain.brain_for",
                        return_value=brain):
            out = architect._model_polish("do the thing",
                                          dict(self._BASE))
        self.assertIsNotNone(out)

    def test_polish_blocked_when_owner_mode_is_off(self):
        brain = _StubBrain({"intent": "sharper intent",
                            "subgoals": ["step one"]})
        architect = PromptArchitect(self._ctx("off"))
        with mock.patch("nomorals.llm.brain.brain_for",
                        return_value=brain):
            out = architect._model_polish("do the thing",
                                          dict(self._BASE))
        self.assertIsNone(out)
        self.assertEqual(brain.calls, 0)


# ── (2) plan_mode.is_complex is structure, not shape ────────────────


class PlanModeStructureTests(unittest.TestCase):
    """Pinned as STRUCTURE: the classifier reads the produced
    plan's file/module counts.  It never sees the prompt, so a
    two-word prompt over a six-file plan is complex and a page-long
    prompt over a one-file plan is not."""

    def test_file_and_module_counts_decide(self):
        def spec(path, new_file=False):
            return {"path": path, "new_file": new_file}
        # >4 files: complex however plain each one is
        self.assertTrue(is_complex([spec(f"m/f{i}.py") for i in range(5)]))
        # one file: not complex however long the prompt behind it was
        self.assertFalse(is_complex([spec("hello.py")]))
        # >2 modules: complex on spread alone
        self.assertTrue(is_complex([spec("a/x.py"), spec("b/y.py"),
                                    spec("c/z.py")]))
        # several files in ONE module, none new: still not complex
        self.assertFalse(is_complex([spec("pkg/a.py"), spec("pkg/b.py"),
                                     spec("pkg/c.py")]))


if __name__ == "__main__":
    unittest.main()


# ── (1, coding site) the brief runs + meters inside the loop ─────────


class _StubDB:
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


class _StubContext:
    def __init__(self, router):
        self.router = router
        self.db = _StubDB()
        self.settings = SimpleNamespace(reasoning_mode="off")


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class _FixtureRepo:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass67_"))
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


class _BlobRouter:
    """Content-routed fake whose responses are real LLMResponse
    objects (the shape both the loop funnel and Brain.chat_json
    accept), with per-call usage popped from a script.  Every call's
    full message blob is recorded so call ORDER is assertable."""

    def __init__(self, handler, usage_script=()):
        self.handler = handler
        self._usage_script = list(usage_script)
        self.blobs: list[str] = []

    def chat(self, messages, params=None, **_kw):
        from nomorals.llm.base import LLMResponse, Usage
        blob = "\n".join(getattr(m, "content", "") or "" for m in messages)
        self.blobs.append(blob)
        text = self.handler(blob)
        prompt, completion = (self._usage_script.pop(0)
                              if self._usage_script else (1, 1))
        return LLMResponse(
            text=text, model="fake",
            usage=Usage(prompt_tokens=prompt,
                        completion_tokens=completion,
                        total_tokens=prompt + completion))


class _ReviewerDouble:
    """review_text double on the #63/#65 contract (status channel,
    no usage — the loop must count those calls unreported)."""

    def __init__(self):
        self.calls = []

    def __call__(self, context, text, *, focus="", max_calls=None,
                 status=None):
        self.calls.append({"focus": focus})
        if status is not None:
            status.update(available=True, error="")
        return []


_HELLO_STEPS = [
    {"path": "hello.py", "why": "the greeting module is missing",
     "new_file": True,
     "instructions": "create greet() returning 'hello'",
     "acceptance": "test_hello passes"},
]

_BRIEF_MODEL_JSON = {
    "objective": "Fix the greeting so greet() returns 'hello'",
    "constraints": [],
    "success_criteria": ["test_hello passes"],
    "steps": ["create hello.py with greet() returning 'hello'"],
    "persona": "",
    "tool_hints": [],
}


def _hello_handler(blob):
    if "structure raw owner requests" in blob:
        return json.dumps(_BRIEF_MODEL_JSON)
    if "build planner" in blob:
        return "```json\n" + json.dumps({"steps": _HELLO_STEPS}) + "\n```"
    if "Write complete, runnable Python" in blob:
        return "```python\ndef greet():\n    return 'hello'\n```"
    return ""


@unittest.skipUnless(shutil.which("git"), "git not available")
class CodingBriefMeteringTests(unittest.TestCase):

    def _repo(self):
        fx = _FixtureRepo()
        fx.write("tests/__init__.py", "")
        fx.write("tests/test_hello.py",
                 "import hello\n\n\ndef test_greet():\n"
                 "    assert hello.greet() == 'hello'\n")
        fx.commit_all()
        return fx

    def test_short_task_brief_runs_first_and_is_metered(self):
        from nomorals.agents.coding import CodingAgent
        fx = self._repo()
        self.addCleanup(fx.cleanup)
        # router calls in order: brief, planner, draft
        router = _BlobRouter(_hello_handler,
                             usage_script=[(7, 3), (100, 50), (200, 100)])
        agent = CodingAgent(_StubContext(router), root=str(fx.root))
        with mock.patch("nomorals.agents.reasoning.review_text",
                        _ReviewerDouble()):
            result = agent.run("Fix the greeting", max_iterations=5,
                               timeout=30, plan_mode="never")
        self.assertTrue(result.ok, result.error)
        # the brief ran FIRST — before the planner — even though the
        # task is 16 chars and one clause (previously: never ran)
        self.assertIn("structure raw owner requests", router.blobs[0])
        # …and its spend is on the result under its own phase (#65)
        by_phase = result.usage["by_phase"]
        self.assertIn("brief", by_phase)
        self.assertEqual(by_phase["brief"]["calls"], 1)
        self.assertEqual(by_phase["brief"]["total_tokens"], 10)
        self.assertEqual(by_phase["brief"]["usage_unreported"], 0)
        # the totals include the brief call: 10 + 150 + 300
        self.assertEqual(result.usage["total_tokens"], 460)
