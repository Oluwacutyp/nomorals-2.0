"""Wave 65 — the system knows whether to build.

Hermetic end-to-end tests:

  1. task-type classifier — pure text is classified (build | investigate |
     research | chat) by a deterministic keyword layer first; a model
     tiebreak is spent ONLY when the keywords are silent.  A build carries
     the artifact (the concrete thing that must exist when done).
  2. real build execution   — a build-typed project routes its steps
     through the real coding agent (draft -> sandbox run -> fix): a real
     file lands in the workspace, a real exit code decides the step, and
     multi-step builds EXTEND the previous step's code.  Non-build
     projects keep the narration executor and never write files.
  3. closed-loop skills     — proven skills are recalled into plan AND
     execute prompts (projects + goals), and each step outcome is fed
     back via record_use so a skill's success rate tracks reality.
  4. devon routing          — "build X" escalates to the mission path
     deterministically (offline-safe), investigation stays in the
     diagnostic tools, and the LLM planner sees the pre-classification.

No network egress.  LLM paths run through a scripted FakeRouter; the
sandbox, filesystem, and database are the real ones.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nomorals.agents.context import build_context       # noqa: E402
from nomorals.agents.projects import ProjectManager     # noqa: E402
from nomorals.agents.skills import SkillLibrary         # noqa: E402
from nomorals.agents.task_type import (                 # noqa: E402
    TASK_KINDS, TaskType, artifact_filename, classify_task)
from nomorals.core.config import load_settings          # noqa: E402
from nomorals.llm.base import LLMResponse               # noqa: E402
from nomorals.tools.filesystem import safe_path         # noqa: E402


class FakeRouter:
    """Scripted router: first marker found in the last user prompt wins."""

    def __init__(self, script: dict[str, str] | None = None,
                 default: str = "") -> None:
        self.script = dict(script or {})
        self.default = default
        self.calls = 0
        self.prompts: list[str] = []

    def chat(self, messages, params=None, **kw):
        prompt = messages[-1].content if messages else ""
        self.calls += 1
        self.prompts.append(prompt)
        for marker, reply in self.script.items():
            if marker in prompt:
                return LLMResponse(text=reply, model="fake")
        return LLMResponse(text=self.default, model="fake")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-wave65-")
        settings = load_settings(
            overrides={"home": self.tmp.name, "partner.platforms": "local",
                       "chat.local_enabled": "true"})
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)
        self.context.__enter__()

    def tearDown(self) -> None:
        try:
            self.context.__exit__(None, None, None)
        finally:
            self.tmp.cleanup()


# ── 1. the task-type classifier ───────────────────────────────────────────────

class TaskTypeKeywordTest(_Base):
    """The deterministic layer: offline, zero model calls."""

    def test_build_verb_plus_artifact(self) -> None:
        for text in ("build a todo cli", "write a python script that scrapes",
                     "create a csv parser", "develop a rest api for the goals",
                     "make me a password generator", "build a calculator"):
            tt = classify_task(self.context, text)
            self.assertEqual(tt.kind, "build", text)
            self.assertEqual(tt.source, "keywords", text)
            self.assertTrue(tt.artifact, text)
            self.assertGreaterEqual(tt.confidence, 0.8, text)

    def test_strong_build_phrase_without_noun(self) -> None:
        tt = classify_task(self.context, "make a thing that beeps")
        self.assertEqual(tt.kind, "build")
        self.assertEqual(tt.source, "keywords")

    def test_artifact_anchors_on_last_noun(self) -> None:
        self.assertEqual(
            classify_task(self.context, "build a todo cli").artifact,
            "todo cli")
        tt = classify_task(self.context, "write a python script that scrapes")
        self.assertEqual(tt.artifact, "python script")
        self.assertEqual(
            classify_task(self.context, "build a website for my band").artifact,
            "website")

    def test_artifact_strips_leading_verb_and_article(self) -> None:
        tt = classify_task(self.context, "make a script")
        self.assertEqual(tt.artifact, "script")
        tt = classify_task(self.context, "build a backup pipeline")
        self.assertEqual(tt.artifact, "backup pipeline")

    def test_investigate_markers(self) -> None:
        for text in ("why did the deploy crash",
                     "the message flow is broken, investigate",
                     "check if the gateway is sending messages",
                     "there is a traceback in the log"):
            tt = classify_task(self.context, text)
            self.assertEqual(tt.kind, "investigate", text)
            self.assertEqual(tt.source, "keywords", text)

    def test_research_markers(self) -> None:
        for text in ("what is the latest on GGUF quantization",
                     "research docker network modes",
                     "look up the best way to shard a kubernetes cluster"):
            tt = classify_task(self.context, text)
            self.assertEqual(tt.kind, "research", text)
            self.assertEqual(tt.source, "keywords", text)

    def test_chat_default_when_silent(self) -> None:
        tt = classify_task(self.context, "hey how are you")
        self.assertEqual(tt.kind, "chat")
        self.assertEqual(tt.source, "default")
        tt = classify_task(self.context, "")
        self.assertEqual(tt.kind, "chat")

    def test_build_beats_investigate_when_artifact_present(self) -> None:
        # a broken-thing BUILDER is still a build
        tt = classify_task(self.context, "build a crash monitor script")
        self.assertEqual(tt.kind, "build")
        self.assertEqual(tt.artifact, "crash monitor script")

    def test_no_model_calls_in_keyword_layer(self) -> None:
        router = FakeRouter(script={}, default="X")
        self.context.router = router
        classify_task(self.context, "build a todo cli")
        self.assertEqual(router.calls, 0)

    def test_all_kinds_are_valid(self) -> None:
        self.assertEqual(TASK_KINDS, ("build", "investigate", "research",
                                      "chat"))


class TaskTypeModelTiebreakTest(_Base):
    """The model tiebreak: only when the keywords are silent."""

    def test_ambiguous_text_uses_model(self) -> None:
        self.context.router = FakeRouter(
            default='{"kind": "build", "confidence": 0.85, '
                    '"artifact": "price watcher", "verify": "run it"}')
        tt = classify_task(self.context,
                           "put together something that watches prices")
        self.assertEqual(tt.kind, "build")
        self.assertEqual(tt.source, "model")
        self.assertEqual(tt.artifact, "price watcher")
        self.assertEqual(tt.verify, "run it")
        self.assertAlmostEqual(tt.confidence, 0.85)

    def test_model_garbage_falls_back_to_default(self) -> None:
        self.context.router = FakeRouter(default="I think it's a bit of a build")
        tt = classify_task(self.context, "put together something")
        self.assertEqual(tt.kind, "chat")
        self.assertEqual(tt.source, "default")

    def test_model_rejects_unknown_kind(self) -> None:
        self.context.router = FakeRouter(
            default='{"kind": "vibrate", "confidence": 0.9}')
        tt = classify_task(self.context, "put together something")
        self.assertEqual(tt.kind, "chat")
        self.assertEqual(tt.source, "default")

    def test_non_build_model_answer_has_no_artifact(self) -> None:
        self.context.router = FakeRouter(
            default='{"kind": "research", "confidence": 0.7, '
                    '"artifact": "should be empty", "verify": "x"}')
        tt = classify_task(self.context, "tell me about the state of RAG")
        self.assertEqual(tt.kind, "research")
        self.assertEqual(tt.artifact, "")
        self.assertEqual(tt.verify, "")

    def test_keywords_hit_skips_model(self) -> None:
        router = FakeRouter(default="X")
        self.context.router = router
        classify_task(self.context, "build a todo cli")
        self.assertEqual(router.calls, 0)

    def test_use_model_false_never_calls(self) -> None:
        router = FakeRouter(default="X")
        self.context.router = router
        tt = classify_task(self.context, "put together something",
                           use_model=False)
        self.assertEqual(router.calls, 0)
        self.assertEqual(tt.kind, "chat")

    def test_no_router_stays_default(self) -> None:
        tt = classify_task(None, "put together something")
        self.assertEqual(tt.kind, "chat")
        self.assertEqual(tt.source, "default")


class ArtifactFilenameTest(unittest.TestCase):
    def test_slug(self) -> None:
        self.assertEqual(artifact_filename("todo cli"), "todo-cli.py")
        self.assertEqual(artifact_filename("Python Script"), "python-script.py")
        self.assertEqual(artifact_filename("rest api"), "rest-api.py")

    def test_default(self) -> None:
        self.assertEqual(artifact_filename(""), "main.py")

    def test_existing_filename_passthrough(self) -> None:
        self.assertEqual(artifact_filename("app.py"), "app.py")


# ── 2. real build execution ───────────────────────────────────────────────────

class _BuildRouter(FakeRouter):
    """Plan call -> JSON array of steps; coding draft -> a code block."""

    def __init__(self, code: str, steps: list[str] | None = None,
                 default_reply: str = "step done") -> None:
        super().__init__(script={}, default=default_reply)
        self.code = code
        self.steps = steps or ["do the work"]
        self.plan_json = "[" + ", ".join(
            f'"{s}"' for s in self.steps) + "]"
        self.draft_count = 0

    def chat(self, messages, params=None, **kw):
        prompt = messages[-1].content if messages else ""
        self.calls += 1
        self.prompts.append(prompt)
        if prompt.startswith("Objective:"):
            return LLMResponse(text=self.plan_json, model="fake")
        if "Rewrite it as a different" in prompt:
            return LLMResponse(text="a genuinely different approach",
                               model="fake")
        # coding draft (also the reason-review calls, which look for a code
        # block and keep the draft when none is in the reply)
        self.draft_count += 1
        return LLMResponse(text=self.code, model="fake")


class _ScriptingRouter:
    """A router driven by a stateful script function (for multi-draft
    scenarios); records every prompt for assertions."""

    def __init__(self, script_fn) -> None:
        self.script_fn = script_fn
        self.prompts: list[str] = []
        self.calls = 0

    def chat(self, messages, params=None, **kw):
        prompt = messages[-1].content if messages else ""
        self.calls += 1
        self.prompts.append(prompt)
        return self.script_fn(prompt)


def _multi_step_script(first: str, second: str):
    """Reply script: plan -> two steps; draft 1 = first, drafts 2+ = second."""
    state = {"n": 0}

    def script(prompt: str) -> LLMResponse:
        if prompt.startswith("Objective:"):
            return LLMResponse(text='["step one", "step two"]', model="fake")
        n = min(state["n"], 1)
        state["n"] += 1
        return LLMResponse(text=(first if n == 0 else second), model="fake")

    return script


def _fixing_script(broken: str, fixed: str):
    """Reply script: plan -> one step; draft 1 broken, draft 2 fixed."""
    state = {"n": 0}

    def script(prompt: str) -> LLMResponse:
        if prompt.startswith("Objective:"):
            return LLMResponse(text='["make it work"]', model="fake")
        if "Rewrite it as a different" in prompt:
            return LLMResponse(text="try again", model="fake")
        n = min(state["n"], 1)
        state["n"] += 1
        return LLMResponse(text=(broken if n == 0 else fixed), model="fake")

    return script


class BuildProjectCreateTest(_Base):
    def test_create_classifies_build(self) -> None:
        mgr = ProjectManager(self.context)
        p = mgr.create("Todo CLI", objective="build a todo cli")
        self.assertEqual(p.task_kind, "build")
        self.assertEqual(p.artifact, "todo cli")

    def test_create_classifies_research(self) -> None:
        mgr = ProjectManager(self.context)
        p = mgr.create("Net research",
                       objective="research docker network modes")
        self.assertEqual(p.task_kind, "research")
        self.assertEqual(p.artifact, "")

    def test_create_classifies_investigate(self) -> None:
        mgr = ProjectManager(self.context)
        p = mgr.create("Crash", objective="why did the deploy crash")
        self.assertEqual(p.task_kind, "investigate")

    def test_persistence_roundtrip(self) -> None:
        mgr = ProjectManager(self.context)
        p = mgr.create("Calc", objective="build a calculator script")
        loaded = mgr._load(p.id)
        self.assertEqual(loaded.task_kind, "build")
        self.assertEqual(loaded.artifact, "calculator script")
        self.assertEqual(loaded.to_dict()["task_kind"], "build")
        self.assertEqual(loaded.to_dict()["artifact"], "calculator script")

    def test_migration_added_columns(self) -> None:
        cols = [r["name"] for r in self.context.db.query(
            "PRAGMA table_info(projects)")]
        self.assertIn("task_kind", cols)
        self.assertIn("artifact", cols)

    def test_report_and_list_show_kind(self) -> None:
        mgr = ProjectManager(self.context)
        p = mgr.create("Calc", objective="build a calculator script",
                       steps=["step a"])
        rep = mgr.report(p.id)
        self.assertEqual(rep["task_kind"], "build")
        self.assertEqual(rep["artifact"], "calculator script")
        self.assertIn("[build", rep["report"])
        row = mgr.list_projects()[0]
        self.assertEqual(row["task_kind"], "build")
        self.assertEqual(row["artifact"], "calculator script")


class BuildProjectExecutionTest(_Base):
    """The core of wave 65: build steps run for real in the sandbox."""

    def test_build_step_produces_a_real_file(self) -> None:
        router = _BuildRouter(
            code='```python\nprint("hello from the real build")\n```',
            steps=["draft the core", "wire it up", "verify the output"])
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("Todo CLI", objective="build a todo cli")
        p = mgr.plan(p.id)
        self.assertEqual(len(p.steps), 3)
        # the planning prompt carried the build guidance
        self.assertIn("This is a BUILD task", router.prompts[0])
        self.assertIn("todo cli", router.prompts[0])

        result = mgr.run(p.id, max_steps=10)
        self.assertEqual(result["status"], "done")
        p2 = mgr._load(p.id)
        # every step reports the REAL execution, not a narration
        for s in p2.steps:
            self.assertEqual(s.status, "done")
            self.assertIn("built todo-cli.py for real", s.result)
            self.assertIn("exit 0", s.result)
        # and the file actually exists in the workspace
        fpath = safe_path(self.context, "todo-cli.py")
        self.assertTrue(fpath.exists())
        self.assertIn("hello from the real build", fpath.read_text())
        # and the real stdout is in the step result
        self.assertTrue(any("hello from the real build" in s.result
                            for s in p2.steps))

    def test_coding_agent_fixes_its_own_failure(self) -> None:
        """First draft crashes, second works — inside one step."""
        self.context.router = _ScriptingRouter(_fixing_script(
            '```python\nraise ValueError("boom")\n```',
            '```python\nprint("fixed it")\n```'))
        mgr = ProjectManager(self.context)
        p = mgr.create("Fixer", objective="build a fixer script",
                       steps=["make it work"])
        result = mgr.run(p.id, max_steps=5)
        self.assertEqual(result["status"], "done")
        p2 = mgr._load(p.id)
        self.assertIn("iteration 2", p2.steps[0].result)
        self.assertIn("fixed it", p2.steps[0].result)
        # the sandbox journal records the real failure then the fix
        rows = self.context.db.query(
            "SELECT exit_code FROM coding_log ORDER BY created_at")
        self.assertEqual([r["exit_code"] for r in rows], [1, 0])

    def test_hard_failure_fails_the_project_with_a_lesson(self) -> None:
        router = FakeRouter(
            script={"Objective:": '["do it"]'},
            default='```python\nraise SystemExit("boom")\n```')
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("Broken", objective="build a broken thing",
                       steps=["do it"])
        result = mgr.run(p.id, max_steps=5, max_attempts=1)
        self.assertEqual(result["status"], "failed")
        p2 = mgr._load(p.id)
        self.assertIn("failed after", p2.steps[0].result)
        self.assertIn("main.py", p2.steps[0].result)
        # the failure entered the learning pipeline (lesson + prevention skill)
        rows = self.context.db.query(
            "SELECT source, root_cause FROM lessons "
            "ORDER BY created_at DESC LIMIT 1")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["source"], "project")
        skills = self.context.db.query(
            "SELECT name FROM skills WHERE kind='prevention'")
        self.assertTrue(skills)

    def test_multi_step_build_extends_previous_code(self) -> None:
        first = '```python\nVALUE = 41\nprint(VALUE + 1)\n```'
        second = '```python\nVALUE = 41\nprint(VALUE + 1)\nprint("step 2")\n```'
        router = _ScriptingRouter(_multi_step_script(first, second))
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("Extender", objective="build a demo pipeline",
                       steps=["step one", "step two"])
        self.assertEqual(p.artifact, "demo pipeline")
        result = mgr.run(p.id, max_steps=10)
        self.assertEqual(result["status"], "done")
        # the second step's FIRST draft saw the first step's code
        seeded = [prm for prm in router.prompts
                  if "Existing code from the previous step" in prm]
        self.assertTrue(seeded)
        self.assertIn("VALUE = 41", seeded[0])
        # final file has both steps
        fpath = safe_path(self.context, "demo-pipeline.py")
        self.assertIn("step 2", fpath.read_text())

    def test_non_build_project_never_writes_files(self) -> None:
        self.context.router = FakeRouter(
            script={"Objective:": '["survey", "summarize"]'},
            default="surveyed the field and summarized findings")
        mgr = ProjectManager(self.context)
        p = mgr.create("Research", objective="research docker network modes",
                       steps=["survey", "summarize"])
        result = mgr.run(p.id, max_steps=10)
        self.assertEqual(result["status"], "done")
        p2 = mgr._load(p.id)
        self.assertIn("surveyed the field", p2.steps[0].result)
        ws = Path(self.context.settings.workspace_dir)
        self.assertFalse(list(ws.glob("*.py")),
                         "a research project must not build files")

    def test_build_executor_wins_over_orchestrator(self) -> None:
        class _Orch:
            def run(self, description: str) -> str:
                return "narration from the orchestrator"

        self.context.router = _BuildRouter(
            code='```python\nprint("orch override")\n```',
            steps=["step"])
        self.context.orchestrator = _Orch()
        mgr = ProjectManager(self.context)
        p = mgr.create("Override", objective="build a demo script",
                       steps=["step"])
        result = mgr.run(p.id, max_steps=5)
        self.assertEqual(result["status"], "done")
        p2 = mgr._load(p.id)
        # real build, not the orchestrator's narration
        self.assertIn("built demo-script.py for real", p2.steps[0].result)
        self.assertTrue(safe_path(self.context, "demo-script.py").exists())

    def test_build_with_explicit_steps_skips_planning(self) -> None:
        self.context.router = _BuildRouter(code='```python\nprint("x")\n```')
        mgr = ProjectManager(self.context)
        p = mgr.create("Fast", objective="build a fast script",
                       steps=["one", "two"])
        self.assertEqual(p.status, "running")
        self.assertEqual(p.task_kind, "build")
        result = mgr.run(p.id, max_steps=5)
        self.assertEqual(result["status"], "done")


# ── 3. closed-loop skills ─────────────────────────────────────────────────────

class ClosedLoopSkillsTest(_Base):
    def _save_skill(self, name: str = "verify-by-running",
                    body: str = "always run the artifact and check stdout") -> str:
        return SkillLibrary(self.context.db).save(
            name, kind="strategy", body=body,
            tags=["build", "verify"]).id

    def test_project_plan_prompt_contains_skills(self) -> None:
        sid = self._save_skill()
        self.context.router = FakeRouter(
            script={"Objective:": '["a", "b"]'}, default="x")
        mgr = ProjectManager(self.context)
        p = mgr.create("Builder", objective="build a parser script")
        mgr.plan(p.id)
        self.assertIn("verify-by-running", self.context.router.prompts[0])

    def test_goal_decompose_prompt_contains_skills(self) -> None:
        self._save_skill(name="parser-lesson",
                         body="parse with the csv module, not split")
        self.context.router = FakeRouter(default='["a", "b"]')
        from nomorals.agents.goals import GoalSystem
        gs = GoalSystem(self.context)
        gs.create("Build a csv parser", description="make a parser")
        self.assertTrue(any("parser-lesson" in prm
                            for prm in self.context.router.prompts))

    def test_narration_step_records_skill_success(self) -> None:
        sid = self._save_skill(name="citation-habit",
                               body="cite sources when doing research")
        self.context.router = FakeRouter(
            script={"Objective:": '["survey the research field"]'},
            default="surveyed with citations")
        mgr = ProjectManager(self.context)
        p = mgr.create("Res", objective="research the research field",
                       steps=["survey the research field"])
        mgr.run(p.id, max_steps=5)
        s = SkillLibrary(self.context.db).get(sid)
        self.assertGreaterEqual(s.uses, 1)
        self.assertGreaterEqual(s.success_count, 1)

    def test_failed_narration_step_records_skill_failure(self) -> None:
        sid = self._save_skill(name="flaky-habit",
                               body="a habit for flaky research tasks")

        class _Broken:
            def chat(self, messages, params=None, **kw):
                prompt = messages[-1].content if messages else ""
                if prompt.startswith("Objective:"):
                    return LLMResponse(text='["flaky research task"]',
                                       model="fake")
                return LLMResponse(text="", model="fake",
                                   error="model down")

        self.context.router = _Broken()
        mgr = ProjectManager(self.context)
        p = mgr.create("Flaky", objective="research the flaky research task",
                       steps=["flaky research task"])
        result = mgr.run(p.id, max_steps=5, max_attempts=1)
        self.assertEqual(result["status"], "failed")
        s = SkillLibrary(self.context.db).get(sid)
        self.assertGreaterEqual(s.failure_count, 1)

    def test_build_step_records_skill_success(self) -> None:
        sid = self._save_skill()
        self.context.router = _BuildRouter(code='```python\nprint("ok")\n```',
                                           steps=["one"])
        mgr = ProjectManager(self.context)
        p = mgr.create("B", objective="build a parser script",
                       steps=["one"])
        mgr.run(p.id, max_steps=5)
        s = SkillLibrary(self.context.db).get(sid)
        self.assertGreaterEqual(s.uses, 1)
        self.assertGreaterEqual(s.success_count, 1)

    def test_empty_library_is_harmless(self) -> None:
        self.context.router = _BuildRouter(code='```python\nprint("ok")\n```',
                                           steps=["one"])
        mgr = ProjectManager(self.context)
        p = mgr.create("B", objective="build a parser script",
                       steps=["one"])
        result = mgr.run(p.id, max_steps=5)
        self.assertEqual(result["status"], "done")


# ── 4. devon routing ──────────────────────────────────────────────────────────

class DevonRoutingTest(_Base):
    def _agent(self):
        from nomorals.agents.devon import DevonAgent
        return DevonAgent(self.context)

    def test_build_escalates_to_mission_offline(self) -> None:
        plan = self._agent()._heuristic_plan(
            "build a crawler script for job listings")
        self.assertEqual(plan[0]["tool"], "mission")
        self.assertEqual(plan[0]["args"]["goal"],
                         "build a crawler script for job listings")
        self.assertEqual(plan[0]["args"]["name"], "devon-mission")

    def test_build_a_test_script_goes_to_mission_not_tests(self) -> None:
        plan = self._agent()._heuristic_plan(
            "build a test script for the parser")
        self.assertEqual(plan[0]["tool"], "mission")
        self.assertNotIn("run_tests", [s["tool"] for s in plan])

    def test_investigation_stays_in_diagnostics(self) -> None:
        plan = self._agent()._heuristic_plan("why did the deploy crash")
        tools = [s["tool"] for s in plan]
        self.assertNotIn("mission", tools)
        self.assertIn("logs_tail", tools)

    def test_big_dataset_task_escalates_to_mission(self) -> None:
        # the real 161-word phone task: "create a big dataset of at least
        # 40000 ..." — a long CREATE task goes to the mission path, not to
        # research tools.
        plan = self._agent()._heuristic_plan(
            "create a big dataset of at least 40000 dataset for your "
            "training around all different types of topic research and "
            "scrape the internet for free datasets if you need to")
        self.assertEqual(plan[0]["tool"], "mission")
        self.assertIn("create a big dataset", plan[0]["args"]["goal"])

    def test_llm_plan_prompt_carries_classification(self) -> None:
        self.context.router = FakeRouter(
            default='{"steps": [{"tool": "git_status", "args": {}}]}')
        agent = self._agent()
        agent._llm_plan("build a todo cli", "")
        self.assertIn("Task type (pre-classified): build",
                      self.context.router.prompts[0])

    def test_chat_task_gets_no_type_hint(self) -> None:
        self.context.router = FakeRouter(
            default='{"steps": [{"tool": "mood", "args": {}}]}')
        agent = self._agent()
        agent._llm_plan("how are you feeling", "")
        self.assertNotIn("Task type (pre-classified)",
                         self.context.router.prompts[0])


# ── 5. CLI surface ────────────────────────────────────────────────────────────

class _CliHome:
    """CLI tests get an isolated NM_HOME (never the real one)."""

    def setUp(self) -> None:
        super().setUp()
        self._cli_tmp = tempfile.TemporaryDirectory(prefix="nm-wave65-cli-")
        self._old_home = os.environ.get("NM_HOME")
        os.environ["NM_HOME"] = self._cli_tmp.name

    def tearDown(self) -> None:
        try:
            if self._old_home is None:
                os.environ.pop("NM_HOME", None)
            else:
                os.environ["NM_HOME"] = self._old_home
        finally:
            self._cli_tmp.cleanup()
            super().tearDown()

    def _run_cli(self, *argv: str) -> tuple[int, str]:
        from nomorals.cli import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()


class ProjectCliTest(_CliHome, _Base):
    def test_create_prints_kind_and_artifact(self) -> None:
        code, out = self._run_cli("project", "create", "Todo CLI",
                                  "build a todo cli")
        self.assertEqual(code, 0)
        self.assertIn("kind: build", out)
        self.assertIn("artifact: todo cli", out)

    def test_list_shows_kind_marker(self) -> None:
        self._run_cli("project", "create", "Todo CLI", "build a todo cli")
        code, out = self._run_cli("project", "list")
        self.assertEqual(code, 0)
        self.assertIn("[build]", out)

    def test_report_shows_kind(self) -> None:
        code, out = self._run_cli("project", "create", "Todo CLI",
                                  "build a todo cli")
        pid = out.strip().splitlines()[0].split()[1]
        code, out = self._run_cli("project", "report", pid)
        self.assertEqual(code, 0)
        self.assertIn("[build", out)


if __name__ == "__main__":
    unittest.main()
