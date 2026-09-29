"""Wave 66 — multiversal-grade execution: the system plans around cost.

Hermetic end-to-end tests:

  1. predictive budget reservation — a build step reserves its model calls
     BEFORE the draft loop starts; when the daily cap can't cover it the
     project suspends (honest report, step stays pending, resumes after
     rollover) instead of dying mid-build with the budget half-eaten.
  2. cost-aware expected value  — mission ranking damps a goal's EV by
     the estimated model calls its remaining work will spend (build
     projects cost 3 calls/step, narration 1), and expensive goals earn
     the medium risk tier.
  3. real acceptance criteria   — a build step is done only when its REAL
     sandbox command exits 0; the classifier's verify hint is compiled
     into that command (prose falls back to running the artifact).
  4. build-session skill distill — a coding session that had to fix
     itself becomes a reusable code skill named after the error signature,
     so the next build recalls the proven fix pattern.

No network egress.  LLM paths run through scripted routers; the sandbox,
filesystem, and database are the real ones.
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

from nomorals.agents.context import build_context        # noqa: E402
from nomorals.agents.cognition import ModelBudget        # noqa: E402
from nomorals.agents.mission import MissionControl       # noqa: E402
from nomorals.agents.projects import (                   # noqa: E402
    BudgetSuspended, ProjectManager)
from nomorals.agents.skills import SkillLibrary          # noqa: E402
from nomorals.agents.task_type import acceptance_command  # noqa: E402
from nomorals.core.config import load_settings           # noqa: E402
from nomorals.llm.base import LLMResponse                # noqa: E402


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-wave66-")
        settings = load_settings(
            overrides={"home": self.tmp.name, "partner.platforms": "local",
                       "chat.local_enabled": "true", **self._extra()})
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)
        self.context.__enter__()

    def _extra(self) -> dict:
        return {}

    def tearDown(self) -> None:
        try:
            self.context.__exit__(None, None, None)
        finally:
            self.tmp.cleanup()


class _CodeRouter:
    """Plan -> JSON steps; drafts pop from a script (code blocks)."""

    def __init__(self, drafts: list[str],
                 steps: list[str] | None = None) -> None:
        self.drafts = list(drafts)
        self.steps = steps or ["do it"]
        self.prompts: list[str] = []
        self.draft_count = 0

    def chat(self, messages, params=None, **kw):
        prompt = messages[-1].content if messages else ""
        self.prompts.append(prompt)
        if prompt.startswith("Objective:"):
            return LLMResponse(text="[" + ", ".join(
                f'"{s}"' for s in self.steps) + "]", model="fake")
        if "Rewrite it as a different" in prompt:
            return LLMResponse(text="a different approach", model="fake")
        i = min(self.draft_count, len(self.drafts) - 1)
        self.draft_count += 1
        return LLMResponse(text=self.drafts[i], model="fake")


# ── 1. predictive budget reservation ────────────────────────────────────────

class ReservationLedgerTest(_Base):
    def test_reserve_commit_release(self) -> None:
        b = ModelBudget(self.context)
        self.assertEqual(b.cap, 0)  # unlimited by default
        r = b.reserve(10, task="build step")
        self.assertTrue(r["ok"])
        self.assertEqual(b.held(), 10)
        # unlimited: even a huge reservation is fine, remaining stays -1
        r2 = b.reserve(5000, task="big build")
        self.assertTrue(r2["ok"])
        self.assertEqual(b.remaining_available(), -1)
        b.release(r["reservation_id"])
        b.commit(r2["reservation_id"])
        # idempotent: resolving a closed reservation again is a no-op
        self.assertFalse(b.release(r["reservation_id"]))

    def test_persistence_across_objects(self) -> None:
        b = ModelBudget(self.context)
        r = b.reserve(7, task="build")
        b2 = ModelBudget(self.context)
        self.assertEqual(b2.held(), 7)
        b2.commit(r["reservation_id"])
        self.assertEqual(ModelBudget(self.context).held(), 0)
        rows = b2.reservations()
        self.assertEqual(rows[0]["status"], "committed")

    def test_report_includes_reserved(self) -> None:
        b = ModelBudget(self.context)
        r = b.reserve(4, task="build")
        rep = b.report()
        self.assertEqual(rep["reserved"], 4)
        b.release(r["reservation_id"])


class _CappedContext(_Base):
    def _extra(self) -> dict:
        return {"autonomy.daily_model_calls": "5"}


class BuildSuspensionTest(_CappedContext):
    def test_unaffordable_build_suspends_not_fails(self) -> None:
        ModelBudget(self.context).record(3, 0)  # only 2 calls left
        self.context.router = _CodeRouter(
            ['```python\nprint("x")\n```'], steps=["do it"])
        mgr = ProjectManager(self.context)
        p = mgr.create("B", objective="build a demo script",
                       steps=["do it"])
        result = mgr.run(p.id, max_steps=5, max_attempts=3)
        p2 = mgr._load(p.id)
        self.assertEqual(p2.status, "paused")
        self.assertIn("budget", p2.report.lower())
        self.assertIn("reserve", p2.report.lower())
        # the step is PENDING (resumable), not failed/burned
        self.assertEqual(p2.steps[0].status, "pending")
        self.assertEqual(result["status"], "paused")

    def test_suspension_raises_budget_suspended(self) -> None:
        ModelBudget(self.context).record(5, 0)  # 0 left
        self.context.router = _CodeRouter(
            ['```python\nprint("x")\n```'], steps=["do it"])
        mgr = ProjectManager(self.context)
        p = mgr.create("B", objective="build a demo script",
                       steps=["do it"])
        p.status = "running"
        mgr._upsert_row(p)
        mgr._current_project_id = p.id
        # the executor itself raises; advance() converts that into a pause
        with self.assertRaises(BudgetSuspended):
            mgr._default_executor("do it")

    def test_resume_after_rollover_completes(self) -> None:
        ModelBudget(self.context).record(3, 0)
        self.context.router = _CodeRouter(
            ['```python\nprint("x")\n```'], steps=["do it"])
        mgr = ProjectManager(self.context)
        p = mgr.create("B", objective="build a demo script",
                       steps=["do it"])
        mgr.run(p.id, max_steps=5)
        self.assertEqual(mgr._load(p.id).status, "paused")
        # day rolls over: fresh ledger
        self.context.db.execute(
            "DELETE FROM kv_store WHERE key='autonomy.budget'")
        p2 = mgr._load(p.id)
        p2.status = "running"
        mgr._upsert_row(p2)
        result = mgr.run(p.id, max_steps=5)
        self.assertEqual(result["status"], "done")

    def test_affordable_build_still_runs(self) -> None:
        ModelBudget(self.context).record(1, 0)  # 4 left, need 3
        self.context.router = _CodeRouter(
            ['```python\nprint("x")\n```'], steps=["do it"])
        mgr = ProjectManager(self.context)
        p = mgr.create("B", objective="build a demo script",
                       steps=["do it"])
        result = mgr.run(p.id, max_steps=5)
        self.assertEqual(result["status"], "done")
        # the reservation closed after the build
        self.assertEqual(ModelBudget(self.context).held(), 0)


# ── 2. cost-aware expected value ────────────────────────────────────────────

class CostAwareEVTest(_Base):
    def _goal_with_project(self, title: str, objective: str,
                           n_steps: int = 3) -> str:
        from nomorals.agents.goals import GoalSystem
        gs = GoalSystem(self.context)
        g = gs.create(title, description="x",
                      plan=[f"s{i}" for i in range(n_steps)])
        proj = ProjectManager(self.context).create(
            f"proj-{title}", objective=objective,
            steps=[f"s{i}" for i in range(n_steps)], goal_id=g.id)
        self.context.db.execute(
            "UPDATE agent_goals SET project_id=? WHERE id=?", (proj.id, g.id))
        return g.id

    def test_build_costs_three_per_step(self) -> None:
        g_build = self._goal_with_project("Build", "build a demo pipeline")
        g_narr = self._goal_with_project("Research",
                                         "research network topologies")
        scores = MissionControl(self.context).ev_scores(
            [g_build, g_narr])
        sb, sn = scores[g_build], scores[g_narr]
        self.assertEqual(sb["est_calls"], 9)
        self.assertEqual(sn["est_calls"], 3)
        self.assertIn("cost", sb["factors"])
        self.assertLess(sb["factors"]["cost"], sn["factors"]["cost"])
        self.assertLess(sb["expected_value"], sn["expected_value"])

    def test_expensive_goal_earns_medium_risk(self) -> None:
        g_big = self._goal_with_project("Big", "build a large pipeline",
                                        n_steps=5)
        s = MissionControl(self.context).ev_scores([g_big])[g_big]
        self.assertEqual(s["est_calls"], 15)
        self.assertEqual(s["risk"], "medium")

    def test_projectless_goal_costs_zero(self) -> None:
        from nomorals.agents.goals import GoalSystem
        g = GoalSystem(self.context).create(
            "Free", description="x", plan=["a", "b"])
        s = MissionControl(self.context).ev_scores([g.id])[g.id]
        self.assertEqual(s["est_calls"], 0)
        self.assertEqual(s["factors"]["cost"], 1.0)

    def test_ranking_exposes_est_calls(self) -> None:
        g = self._goal_with_project("Build", "build a demo pipeline")
        ranked = MissionControl(self.context).ranking()
        self.assertTrue(any(r["id"] == g for r in ranked))
        row = next(r for r in ranked if r["id"] == g)
        self.assertEqual(row["est_calls"], 9)


# ── 3. real acceptance criteria ─────────────────────────────────────────────

class AcceptanceCommandTest(unittest.TestCase):
    def test_empty_verify_uses_default(self) -> None:
        self.assertEqual(acceptance_command("", "todo-cli.py"),
                         'python3 "todo-cli.py"')

    def test_command_verify_passes_through(self) -> None:
        self.assertEqual(
            acceptance_command("python3 todo-cli.py --check", "todo-cli.py"),
            "python3 todo-cli.py --check")
        self.assertEqual(
            acceptance_command("python3 -m pytest todo_cli.py -q",
                               "todo_cli.py"),
            "python3 -m pytest todo_cli.py -q")

    def test_pipe_is_a_command(self) -> None:
        self.assertEqual(acceptance_command("python3 app.py | grep ok",
                                            "app.py"),
                         "python3 app.py | grep ok")

    def test_relative_path_binary(self) -> None:
        self.assertEqual(acceptance_command("./run.sh", "app.py"),
                         "./run.sh")

    def test_prose_falls_back_to_default(self) -> None:
        self.assertEqual(acceptance_command("run it and see", "app.py"),
                         'python3 "app.py"')
        self.assertEqual(acceptance_command("verify it works somehow",
                                            "app.py"),
                         'python3 "app.py"')

    def test_model_tiebreak_carries_command(self) -> None:
        class Ctx:
            router = None

        class FakeRouter:
            def chat(self, messages, params=None, **kw):
                return LLMResponse(text=(
                    '{"kind": "build", "confidence": 0.8, '
                    '"artifact": "todo cli", '
                    '"verify": "python3 todo-cli.py --check"}'),
                    model="fake")

        class C:
            pass

        c = C()
        c.router = FakeRouter()
        from nomorals.agents.task_type import classify_task
        tt = classify_task(c, "put together a thing that tracks todos")
        self.assertEqual(tt.verify, "python3 todo-cli.py --check")
        self.assertEqual(
            acceptance_command(tt.verify, "todo-cli.py"),
            "python3 todo-cli.py --check")

    def test_project_stores_verify_cmd(self) -> None:
        import tempfile as _t
        tmp = _t.TemporaryDirectory(prefix="nm-w66-ac-")
        s = load_settings(
            overrides={"home": tmp.name, "partner.platforms": "local",
                       "chat.local_enabled": "true"})
        ctx = build_context(s, with_executor=False, with_tools=False)
        ctx.__enter__()
        try:
            mgr = ProjectManager(ctx)
            p = mgr.create("T", objective="build a todo cli")
            self.assertEqual(p.verify_cmd, 'python3 "todo-cli.py"')
            loaded = mgr._load(p.id)
            self.assertEqual(loaded.verify_cmd, 'python3 "todo-cli.py"')
        finally:
            ctx.__exit__(None, None, None)
            tmp.cleanup()


class RealAcceptanceExecutionTest(_Base):
    def test_step_must_pass_the_acceptance_command(self) -> None:
        """Code that exits 1 without --check: the step only counts as done
        because the acceptance command RAN WITH THE ARGUMENT."""
        router = _CodeRouter(
            ['```python\nimport sys\n'
             'if "--check" not in sys.argv:\n'
             '    sys.exit(1)\n'
             'print("check passed")\n```'],
            steps=["wire the cli"])
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("T", objective="build a todo cli",
                       steps=["wire the cli"])
        # enforce the strict acceptance
        p.verify_cmd = acceptance_command("python3 todo-cli.py --check",
                                          "todo-cli.py")
        mgr._upsert_row(p)
        result = mgr.run(p.id, max_steps=5)
        p2 = mgr._load(p.id)
        self.assertEqual(result["status"], "done")
        self.assertIn("check passed", p2.steps[0].result)

    def test_strict_acceptance_rejects_lazy_code(self) -> None:
        """The acceptance checks the REAL output: lazy code that prints
        nothing useful fails the grep and the step fails — the loop
        can't fake it."""
        router = _CodeRouter(
            ['```python\nprint("did nothing")\n```'],
            steps=["wire the cli"])
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("T", objective="build a todo cli",
                       steps=["wire the cli"])
        p.verify_cmd = acceptance_command(
            'python3 todo-cli.py --check | grep "READY"', "todo-cli.py")
        mgr._upsert_row(p)
        result = mgr.run(p.id, max_steps=5, max_attempts=1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("failed", mgr._load(p.id).steps[0].result)

    def test_planner_sees_the_acceptance_command(self) -> None:
        self.context.router = _CodeRouter(
            ['```python\nprint("x")\n```'], steps=["a", "b"])
        mgr = ProjectManager(self.context)
        p = mgr.create("T", objective="build a todo cli")
        p.verify_cmd = "python3 todo-cli.py --check"
        mgr._upsert_row(p)
        mgr.plan(p.id)
        self.assertIn("python3 todo-cli.py --check",
                      self.context.router.prompts[0])


# ── 4. build-session skill distillation ─────────────────────────────────────

class SessionDistillationTest(_Base):
    def test_self_corrected_session_distills_a_code_skill(self) -> None:
        router = _CodeRouter(
            ['```python\nimport module_that_does_not_exist_xyz\n```',
             '```python\nprint("ok")\n```'],
            steps=["fix it"])
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("D", objective="build a distill demo script",
                       steps=["fix it"])
        result = mgr.run(p.id, max_steps=5)
        self.assertEqual(result["status"], "done")
        lib = SkillLibrary(self.context.db)
        skills = [s for s in lib.list()
                  if s.kind == "code" and s.source == "coding_session"]
        self.assertEqual(len(skills), 1)
        self.assertIn("modulenotfounderror", skills[0].name)
        self.assertIn("does_not_exist", skills[0].body)
        self.assertIn("attempt 1", skills[0].body)

    def test_one_shot_success_does_not_distill(self) -> None:
        lib = SkillLibrary(self.context.db)
        before = len(lib.list())
        router = _CodeRouter(['```python\nprint("first try")\n```'],
                             steps=["one shot"])
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("O", objective="build a one-shot script",
                       steps=["one shot"])
        mgr.run(p.id, max_steps=5)
        skills = [s for s in lib.list() if s.source == "coding_session"]
        self.assertEqual(len(skills), 0)
        self.assertEqual(len(lib.list()), before)

    def test_distilled_skill_is_recallable(self) -> None:
        """The closed loop: distill now, recall later on a similar task."""
        router = _CodeRouter(
            ['```python\nraise ValueError("bad index")\n```',
             '```python\nprint("fixed")\n```'],
            steps=["fix it"])
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("D", objective="build a parser script",
                       steps=["fix it"])
        mgr.run(p.id, max_steps=5)
        lib = SkillLibrary(self.context.db)
        hits = lib.recall("build a parser script with index handling")
        self.assertTrue(hits)


# ── 5. CLI surface ──────────────────────────────────────────────────────────

class _CliHome:
    def setUp(self) -> None:
        super().setUp()
        self._cli_tmp = tempfile.TemporaryDirectory(prefix="nm-wave66-cli-")
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


class Wave66CliTest(_CliHome, _Base):
    def _cli_context(self):
        """A context bound to the CLI home (NM_HOME env), not the _Base
        one — the CLI reads its own database."""
        from nomorals.core.config import load_settings as _ls
        s = _ls()
        ctx = build_context(s, with_executor=False, with_tools=False)
        ctx.__enter__()
        return ctx

    def test_budget_report_shows_reserved(self) -> None:
        ctx = self._cli_context()
        try:
            b = ModelBudget(ctx)
            r = b.reserve(4, task="build step")
            try:
                code, out = self._run_cli("autonomy", "budget")
                self.assertEqual(code, 0)
                self.assertIn("reserved 4", out)
            finally:
                b.release(r["reservation_id"])
        finally:
            ctx.__exit__(None, None, None)

    def test_mission_plan_shows_est_calls(self) -> None:
        code, out = self._run_cli("goal", "create", "Build goal")
        self.assertEqual(code, 0)
        gid = out.strip().splitlines()[0].split()[1]
        code, out = self._run_cli(
            "project", "create", "demo pipeline",
            "build a demo pipeline")
        self.assertEqual(code, 0)
        pid = out.strip().splitlines()[0].split()[1]
        ctx = self._cli_context()
        try:
            ctx.db.execute("UPDATE agent_goals SET project_id=? WHERE id=?",
                           (pid, gid))
            ctx.db.execute("UPDATE projects SET goal_id=? WHERE id=?",
                           (gid, pid))
        finally:
            ctx.__exit__(None, None, None)
        code, out = self._run_cli("mission", "plan")
        self.assertEqual(code, 0)
        self.assertIn("~9 calls", out)


if __name__ == "__main__":
    unittest.main()
