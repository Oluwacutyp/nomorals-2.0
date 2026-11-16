"""Failing-first tests for /plan + learn-as-distillation (gapfix).

/plan surveys the real code graph before planning, saves a reviewable
Implementation Plan artifact, shows it, and runs it (via the existing
DelegationManager.plan_and_spawn) — no approval gate, by owner order
2026-11-16. /skills distill distills a finished mission or a session
description into a draft skill via the existing distill_task.

Written BEFORE the wiring: every test here must FAIL on base 94db727b
(missing modules) and pass after.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# ── fakes ────────────────────────────────────────────────────────────

class FakeGraph:
    """A code graph that records what it was asked about."""

    def __init__(self, rows=None):
        self.rows = rows or [
            {"name": "distill_task", "file": "nomorals/skills/distill_files.py", "line": 44},
            {"name": "FileSkillLibrary", "file": "nomorals/skills/filebook.py", "line": 210},
        ]
        self.calls: list[tuple[str, str]] = []

    def search(self, query, limit=10):
        self.calls.append(("search", query))
        return [dict(r) for r in self.rows[:limit]]

    def where_defined(self, name, limit=20):
        self.calls.append(("where_defined", name))
        return [dict(r) for r in self.rows if r["name"] == name][:limit]

    def who_calls(self, name, limit=20):
        self.calls.append(("who_calls", name))
        return [{"name": "caller_fn", "file": "nomorals/missions/runner.py", "line": 2606}]


class FailingGraph:
    """A code graph that cannot build."""

    def search(self, query, limit=10):
        raise RuntimeError("graph store is corrupt")

    def where_defined(self, name, limit=20):
        raise RuntimeError("graph store is corrupt")

    def who_calls(self, name, limit=20):
        raise RuntimeError("graph store is corrupt")


class FakeManager:
    """A DelegationManager stand-in recording plan_and_spawn calls."""

    def __init__(self):
        self.calls: list[dict] = []
        self.runs_by_id: dict[str, object] = {}

    def plan_and_spawn(self, goal, *, mission_id="", chat_key="", context_hint=""):
        self.calls.append({"goal": goal, "chat_key": chat_key, "context_hint": context_hint})
        runs = [
            SimpleNamespace(id="run-impl-1", role="implementer", contract_id="c-1", status="running"),
            SimpleNamespace(id="run-rev-1", role="reviewer", contract_id="c-2", status="running"),
        ]
        for run in runs:
            self.runs_by_id[run.id] = run
        return {
            "plan": {
                "goal": goal,
                "rationale": "Add the verb over the existing organ.",
                "model": "template",
                "steps": [
                    {"name": "survey", "goal": "Survey the skillbook code", "role": "research",
                     "kind": "io", "depends_on": []},
                    {"name": "implement", "goal": "Add the distill subverb", "role": "coding",
                     "kind": "io", "depends_on": ["survey"],
                     "payload": {"writes": ["nomorals/agents/partner/runtime_expression.py"]}},
                    {"name": "verify", "goal": "Verify the draft lands", "role": "critic",
                     "kind": "io", "depends_on": ["implement"], "payload": {"debate": True}},
                ],
            },
            "plan_error": "no LLM router configured — using template plan",
            "runs": runs,
            "unassigned": [
                {"step": "train", "role": "training", "reason": "no roster worker for plan role 'training'"},
            ],
        }

    def get_run(self, run_id):
        return self.runs_by_id.get(run_id)


def _mission(**kw):
    """A finished-mission shape for distill tests."""
    base = {
        "goal": "Add the distill subverb to the skillbook",
        "mission_id": "m-1",
        "outcome": "done",
        "summary": "done after 4 iterations",
        "steps": [
            {"name": "survey", "role": "research", "tools": ["graph_query"]},
            {"name": "implement", "role": "coding", "tools": ["edit"]},
            {"name": "verify", "role": "critic", "tools": ["pytest"]},
        ],
        "lessons": ["the skillbook handler lives in runtime_expression, not runtime_games"],
        "tags": ["skills"],
    }
    base.update(kw)
    return base


# ── survey ───────────────────────────────────────────────────────────

class SurveyTests(unittest.TestCase):
    def test_survey_consults_the_graph(self):
        from nomorals.agents.plan.survey import survey_task

        graph = FakeGraph()
        survey = survey_task("add a distill verb to the skillbook", graph=graph)
        self.assertEqual(survey["unavailable"], "")
        kinds = {kind for kind, _ in graph.calls}
        self.assertIn("search", kinds)
        self.assertIn("where_defined", kinds)
        files = {f["path"] for f in survey["files"]}
        self.assertIn("nomorals/skills/distill_files.py", files)
        self.assertTrue(survey["keywords"])

    def test_survey_unavailable_when_graph_fails(self):
        from nomorals.agents.plan.survey import survey_task

        survey = survey_task("add a distill verb", graph=FailingGraph())
        self.assertTrue(survey["unavailable"])
        self.assertEqual(survey["files"], [])


# ── plan service ─────────────────────────────────────────────────────

class PlanServiceTests(unittest.TestCase):
    def test_plan_fails_without_survey(self):
        from nomorals.agents.plan.errors import PlanError
        from nomorals.agents.plan.service import plan_and_run
        from nomorals.agents.plan.store import PlanStore

        manager = FakeManager()
        store = PlanStore(path=None, plans_dir=None)
        with self.assertRaises(PlanError):
            plan_and_run(None, "add a distill verb", store=store, manager=manager, graph=FailingGraph())
        self.assertEqual(manager.calls, [])
        self.assertEqual(store.list_plans(), [])

    def test_plan_saves_shows_and_runs(self):
        from nomorals.agents.plan.service import plan_and_run
        from nomorals.agents.plan.store import PlanStore

        manager = FakeManager()
        store = PlanStore(path=None, plans_dir=None)
        result = plan_and_run(
            None, "add a distill verb to the skillbook",
            store=store, manager=manager, graph=FakeGraph())
        self.assertTrue(result["ok"])
        self.assertTrue(result["plan_id"].startswith("pl-"))
        # The survey rode into the planner as the context hint.
        self.assertIn("distill_files.py", manager.calls[0]["context_hint"])
        record = store.get_plan(result["plan_id"])
        self.assertIsNotNone(record)
        self.assertEqual(record["task"], "add a distill verb to the skillbook")
        self.assertEqual(len(record["runs"]), 2)
        self.assertEqual(len(record["unassigned"]), 1)
        self.assertIn("template plan", record["plan_error"])
        # No approval state exists anywhere in the record.
        self.assertNotIn("awaiting_approval", json.dumps(record))
        self.assertNotIn("approved", json.dumps(record))
        display = result["display"]
        self.assertIn("implement", display)
        self.assertIn("verify", display)  # the critic gate is shown
        self.assertIn("training", display)  # unassigned is named, not hidden

    def test_plan_state_derives_from_runs(self):
        from nomorals.agents.plan.service import plan_and_run
        from nomorals.agents.plan.store import PlanStore, derive_plan_state

        manager = FakeManager()
        store = PlanStore(path=None, plans_dir=None)
        result = plan_and_run(
            None, "add a distill verb to the skillbook",
            store=store, manager=manager, graph=FakeGraph())
        record = store.get_plan(result["plan_id"])
        self.assertEqual(derive_plan_state(record, manager.get_run), "running")
        manager.runs_by_id["run-impl-1"].status = "completed"
        manager.runs_by_id["run-rev-1"].status = "completed"
        self.assertEqual(derive_plan_state(record, manager.get_run), "done")
        manager.runs_by_id["run-rev-1"].status = "failed"
        self.assertEqual(derive_plan_state(record, manager.get_run), "failed")


# ── plan store ───────────────────────────────────────────────────────

class PlanStoreTests(unittest.TestCase):
    def test_save_list_get_resolve(self):
        from nomorals.agents.plan.store import PlanStore

        store = PlanStore(path=None, plans_dir=None)
        record = store.save_plan(
            task="add a distill verb", survey={"keywords": ["distill"], "files": [], "symbols": [], "unavailable": ""},
            plan={"goal": "add a distill verb", "rationale": "", "model": "", "steps": []},
            runs=[], unassigned=[], plan_error="", chat_key="dm:owner")
        self.assertTrue(record["id"].startswith("pl-"))
        self.assertEqual(len(store.list_plans()), 1)
        self.assertEqual(store.get_plan(record["id"])["task"], "add a distill verb")
        self.assertEqual(store.resolve(record["id"][:6])["id"], record["id"])

    def test_corrupt_store_degrades_to_empty_with_reason(self):
        from nomorals.agents.plan.store import PlanStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.json"
            path.write_text("{not json", encoding="utf-8")
            store = PlanStore(path=path, plans_dir=None)
            self.assertEqual(store.list_plans(), [])
            self.assertTrue(store.degraded_reason)

    def test_bad_row_skipped(self):
        from nomorals.agents.plan.store import PlanStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.json"
            path.write_text(json.dumps({
                "plans": [
                    {"id": "pl-good", "task": "good", "state": "running"},
                    {"task": "no id at all"},
                    {"id": "pl-bad-state", "task": "bad", "state": "nonsense"},
                ]}), encoding="utf-8")
            store = PlanStore(path=path, plans_dir=None)
            ids = [r["id"] for r in store.list_plans()]
            # The id-less row is skipped; a bad state resets to the
            # initial state (mirroring WorkflowStore) instead of
            # dropping the row.
            self.assertEqual(ids, ["pl-good", "pl-bad-state"])
            bad = store.get_plan("pl-bad-state")
            self.assertEqual(bad["state"], "running")


# ── render ───────────────────────────────────────────────────────────

class RenderTests(unittest.TestCase):
    def test_render_has_approach_steps_gates_risks(self):
        from nomorals.agents.plan.render import render_plan_md

        record = {
            "id": "pl-abc", "task": "add a distill verb",
            "survey": {"keywords": ["distill"], "files": [{"path": "nomorals/skills/distill_files.py", "hits": 2}], "symbols": [], "unavailable": ""},
            "plan": {"goal": "add a distill verb", "rationale": "Reuse the organ.",
                     "model": "template",
                     "steps": [
                         {"name": "implement", "goal": "Add the subverb", "role": "coding", "kind": "io", "depends_on": []},
                         {"name": "verify", "goal": "Verify the draft", "role": "critic", "kind": "io", "depends_on": ["implement"]},
                     ]},
            "runs": [], "unassigned": [{"step": "train", "role": "training", "reason": "no roster worker"}],
            "plan_error": "no LLM router configured — using template plan",
            "state": "running",
        }
        text = render_plan_md(record)
        self.assertIn("Approach", text)
        self.assertIn("Reuse the organ", text)
        self.assertIn("implement", text)
        self.assertIn("Verification", text)
        self.assertIn("verify", text)
        self.assertIn("Risks", text)
        self.assertIn("template plan", text)
        self.assertIn("training", text)


# ── distillation ─────────────────────────────────────────────────────

class DistillTests(unittest.TestCase):
    def _library(self, tmp):
        from nomorals.skills.filebook import FileSkillLibrary

        # Explicit dirs — no NM_HOME env leak into other test files.
        seed = Path(tmp) / "seed"
        seed.mkdir(parents=True, exist_ok=True)
        return FileSkillLibrary(seed_dir=seed, user_dir=Path(tmp) / "skills")

    def test_distill_text_form_writes_draft_in_existing_format(self):
        from nomorals.skills.distill_files import distill_record, task_from_text

        with tempfile.TemporaryDirectory() as tmp:
            library = self._library(tmp)
            task = task_from_text(
                "the skillbook handler lives in runtime_expression, "
                "not runtime_games — check the dispatch table first")
            result = distill_record(library, task)
            self.assertTrue(result["ok"])
            draft = library.get_draft(result["draft_id"])
            self.assertIsNotNone(draft)
            self.assertTrue(draft.is_draft)
            path = Path(result["path"])
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("id: " + result["draft_id"], text)
            self.assertIn("## Lessons", text)
            # Drafts are routable but labelled unproven, and the user
            # INDEX.md is untouched (only promote rewrites it).
            self.assertFalse((Path(tmp) / "skills" / "INDEX.md").exists())

    def test_distill_mission_with_nothing_to_teach_writes_nothing(self):
        from nomorals.skills.distill_files import distill_record, task_from_mission

        with tempfile.TemporaryDirectory() as tmp:
            library = self._library(tmp)
            task = task_from_mission(SimpleNamespace(
                goal="tiny fix", id="m-9", status="done", state={},
                iterations=1, tags=[], metadata={}))
            result = distill_record(library, task)
            self.assertFalse(result["ok"])
            self.assertEqual(library.drafts(), [])

    def test_distill_mission_form(self):
        from nomorals.skills.distill_files import distill_record, task_from_mission

        with tempfile.TemporaryDirectory() as tmp:
            library = self._library(tmp)
            mission = SimpleNamespace(
                goal="Add the distill subverb to the skillbook", id="m-1",
                status="done",
                state={"plan": {"steps": [
                    {"name": "survey", "role": "research", "tools": ["graph_query"]},
                    {"name": "implement", "role": "coding", "tools": ["edit"]},
                    {"name": "verify", "role": "critic", "tools": ["pytest"]},
                ]}},
                iterations=4, tags=["skills"],
                metadata={"lessons": ["check the dispatch table first"]})
            task = task_from_mission(mission)
            result = distill_record(library, task)
            self.assertTrue(result["ok"])
            self.assertEqual(result["mission_id"], "m-1")
            self.assertEqual(result["revises"], "")
            draft = library.get_draft(result["draft_id"])
            self.assertIsNotNone(draft)
            self.assertIn("survey", draft.body)


# ── names ────────────────────────────────────────────────────────────

class NameTests(unittest.TestCase):
    def test_plan_free_and_learn_goal_untouched(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            PUBLIC_COMMANDS,
        )

        self.assertIn("plan", CONTROL_COMMANDS)
        self.assertIn("learn", CONTROL_COMMANDS)
        self.assertIn("goal", CONTROL_COMMANDS)
        # /plan is owner-only like /workflow — never public.
        self.assertNotIn("plan", PUBLIC_COMMANDS)
        # The taken names still mean what they meant.
        self.assertIn("tutor", COMMAND_DETAILS["learn"]["what"])
        self.assertIn("money goals", COMMAND_DETAILS["goal"]["what"])
        # The distill verb lives under /skills.
        self.assertIn("distill", COMMAND_DETAILS["skills"]["usage"])


if __name__ == "__main__":
    unittest.main()
