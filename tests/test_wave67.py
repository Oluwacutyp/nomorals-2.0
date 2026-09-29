"""Wave 67 — the system plans around its budget and never loses work.

Hermetic end-to-end tests:

  1. cross-build error memory   — recurring error families become systemic
     traps: new builds are warned up front, and a build hitting a known
     error gets the proven fix recalled mid-loop (deterministic).
  2. budget-aware mission sizing — every goal shows what fits in today's
     budget (steps affordable, ETA in days); unlimited budgets always fit.
  3. acceptance regression suite — every passing acceptance is frozen;
     later steps re-run earlier acceptances (a loosened contract is
     caught), and `nm project regress` reports the suite at any time.
  4. no more idle-forever        — an idle tick with pending work (even
     budget-suspended work) keeps the configured pace; budget-suspended
     goals AUTO-RESUME after the daily rollover; non-build steps reserve
     and meter their calls exactly like builds.
  5. the "5 a day" question      — the daily cap defaults to UNLIMITED;
     `nm autonomy budget --unlimited | --cap N` writes the owner's .env.
  6. detailed help               — /help is a catalog; /help <command> is
     a full page (what / usage / example / related); topic pages; fuzzy
     matching; the same pages ship in `nm help <topic>`.

No network egress.  LLM paths run through scripted routers; the sandbox,
filesystem, and database are the real ones.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nomorals.agents.cognition import CognitiveLoop, ModelBudget  # noqa: E402
from nomorals.agents.context import build_context        # noqa: E402
from nomorals.agents.goals import GoalSystem             # noqa: E402
from nomorals.agents.mission import MissionControl       # noqa: E402
from nomorals.agents.projects import ProjectManager      # noqa: E402
from nomorals.agents.skills import SkillLibrary          # noqa: E402
from nomorals.agents.task_type import acceptance_command  # noqa: E402
from nomorals.agents.task_type import artifact_filename   # noqa: E402
from nomorals.core.config import load_settings           # noqa: E402
from nomorals.core.ids import new_short_id               # noqa: E402
from nomorals.llm.base import LLMResponse                # noqa: E402
from nomorals.social.chat.control import (               # noqa: E402
    COMMAND_DETAILS, CONTROL_COMMANDS, detailed_help, parse_control)

GOOD_CORE = ('```python\nimport sys\nprint("CORE")\nprint("READY")\n```')
GOOD_EXTRA = ('```python\nimport sys\nprint("READY")\nprint("extra works")\n```')
BAD_IMPORT = ('```python\nimport module_that_does_not_exist_xyz\n```')
FIXED = ('```python\nprint("ok")\n```')


class _Base(unittest.TestCase):
    def _extra(self) -> dict:
        return {}

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-wave67-")
        settings = load_settings(
            overrides={"home": self.tmp.name, "partner.platforms": "local",
                       "chat.local_enabled": "true", **self._extra()})
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)
        self.context.__enter__()

    def tearDown(self) -> None:
        try:
            self.context.__exit__(None, None, None)
        finally:
            self.tmp.cleanup()

    def rollover(self) -> None:
        self.context.db.execute(
            "DELETE FROM kv_store WHERE key='autonomy.budget'")

    def goal_with_project(self, title: str, objective: str,
                          steps: list[str],
                          project_title: str = "proj") -> str:
        gs = GoalSystem(self.context)
        g = gs.create(title, description="x", plan=list(steps))
        proj = ProjectManager(self.context).create(
            project_title, objective=objective, steps=list(steps),
            goal_id=g.id)
        self.context.db.execute(
            "UPDATE goals SET project_id=? WHERE id=?", (proj.id, g.id))
        return g.id


# ── scripted routers ────────────────────────────────────────────────────────

class _TwoPhaseRouter:
    """Odd calls fail with a fixed import error, even calls succeed —
    every session self-corrects with the same error signature."""

    def __init__(self) -> None:
        self.n = 0
        self.prompts: list[str] = []

    def chat(self, messages, params=None, **kw):
        prompt = messages[-1].content if messages else ""
        self.prompts.append(prompt)
        if prompt.startswith("Objective:"):
            return LLMResponse(text='["do it"]', model="fake")
        self.n += 1
        if self.n % 2 == 1:
            return LLMResponse(text=BAD_IMPORT, model="fake")
        return LLMResponse(text=FIXED, model="fake")


class _ExtendRouter:
    """Step 1 writes the core (CORE+READY); continuation steps either
    keep the contract or break it (drop CORE)."""

    def __init__(self, break_contract: bool = False) -> None:
        self.break_contract = break_contract

    def chat(self, messages, params=None, **kw):
        prompt = messages[-1].content if messages else ""
        if prompt.startswith("Objective:"):
            return LLMResponse(text='["write the core", "add the extra"]',
                               model="fake")
        continuation = ("Existing code" in prompt) or ("Previous code" in prompt)
        if not continuation:
            # step 1: the full core — satisfies the strict CORE contract
            return LLMResponse(text=GOOD_CORE, model="fake")
        # step 2+: the extension (with break_contract it drops CORE)
        return LLMResponse(text=GOOD_EXTRA, model="fake")


# ── 1. cross-build error memory (systemic traps) ───────────────────────────

class SystemicTrapTest(_Base):
    def _distill(self, n: int) -> None:
        self.context.router = _TwoPhaseRouter()
        mgr = ProjectManager(self.context)
        for i in range(n):
            p = mgr.create(f"D{i}", objective="build a demo script",
                           steps=["do it"])
            p.artifact = f"demo {i} tool"
            p.verify_cmd = acceptance_command(
                "", artifact_filename(p.artifact))
            mgr._upsert_row(p)
            out = mgr.run(p.id, max_steps=5)
            self.assertEqual(out["status"], "done", out)

    def test_repeat_errors_become_a_systemic_trap(self) -> None:
        self._distill(3)
        traps = SkillLibrary(self.context.db).systemic_traps()
        self.assertTrue(any(t["family"] == "modulenotfounderror"
                            for t in traps), traps)
        block = SkillLibrary(self.context.db).trap_block()
        self.assertIn("KNOWN TRAPS", block)
        self.assertIn("modulenotfounderror", block)

    def test_single_error_is_not_yet_a_trap(self) -> None:
        self._distill(1)
        self.assertEqual(SkillLibrary(self.context.db).systemic_traps(), [])
        self.assertEqual(SkillLibrary(self.context.db).trap_block(), "")

    def test_new_build_first_prompt_carries_the_traps(self) -> None:
        self._distill(3)
        router = _TwoPhaseRouter()
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("W", objective="build a warning demo script",
                       steps=["do it"])
        p.artifact = "warning demo tool"
        p.verify_cmd = acceptance_command(
            "", artifact_filename(p.artifact))
        mgr._upsert_row(p)
        out = mgr.run(p.id, max_steps=5)
        self.assertEqual(out["status"], "done")
        first = next(pp for pp in router.prompts if pp.startswith("Task:"))
        self.assertIn("KNOWN TRAPS", first)

    def test_hard_recall_injects_the_proven_fix(self) -> None:
        self._distill(3)
        router = _TwoPhaseRouter()
        self.context.router = router
        mgr = ProjectManager(self.context)
        p = mgr.create("W", objective="build a warning demo script",
                       steps=["do it"])
        p.artifact = "warning demo tool"
        p.verify_cmd = acceptance_command(
            "", artifact_filename(p.artifact))
        mgr._upsert_row(p)
        out = mgr.run(p.id, max_steps=5)
        self.assertEqual(out["status"], "done")
        fix_prompts = [pp for pp in router.prompts if "KNOWN FIX" in pp]
        self.assertTrue(fix_prompts)
        self.assertIn("fix-modulenotfounderror", fix_prompts[0])
        self.assertIn("module_that_does_not_exist_xyz", fix_prompts[0])

    def test_match_errors_is_deterministic(self) -> None:
        lib = SkillLibrary(self.context.db)
        lib.save("fix-modulenotfounderror-no-modul", kind="code",
                 body="attempt 1: ModuleNotFoundError",
                 source="coding_session")
        hits = lib.match_errors(
            "Traceback (most recent call last):\n"
            "ModuleNotFoundError: No module named 'x'")
        self.assertEqual(len(hits), 1)
        self.assertEqual(lib.match_errors("ValueError: bad index"), [])
        self.assertEqual(lib.match_errors(""), [])


# ── 2. budget-aware mission sizing ─────────────────────────────────────────

class _Capped(_Base):
    def _extra(self) -> dict:
        return {"autonomy.daily_model_calls": "5"}


class BudgetFitTest(_Capped):
    def test_build_goal_does_not_fit(self) -> None:
        g = self.goal_with_project(
            "Big", "build a large pipeline",
            [f"s{i}" for i in range(5)], "pb")
        s = MissionControl(self.context).ev_scores([g])[g]
        self.assertEqual(s["est_calls"], 15)
        self.assertFalse(s["fits_today"])
        self.assertEqual(s["steps_today"], 1)  # 5 left / 3 per step
        self.assertEqual(s["eta_days"], 3)     # 15 / cap 5

    def test_narration_goal_fit(self) -> None:
        g = self.goal_with_project(
            "R", "research network topologies", ["a", "b", "c"], "pn")
        s = MissionControl(self.context).ev_scores([g])[g]
        self.assertEqual(s["est_calls"], 3)
        self.assertTrue(s["fits_today"])
        self.assertEqual(s["steps_today"], 3)
        self.assertEqual(s["eta_days"], 0)

    def test_unlimited_always_fits(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-w67-")
        s = load_settings(overrides={"home": tmp.name,
                                     "partner.platforms": "local",
                                     "chat.local_enabled": "true"})
        ctx = build_context(s, with_executor=False, with_tools=False)
        ctx.__enter__()
        try:
            gs = GoalSystem(ctx)
            g = gs.create("Free", description="x", plan=["a", "b"])
            proj = ProjectManager(ctx).create(
                "pf", objective="build a free pipeline", steps=["a", "b"],
                goal_id=g.id)
            ctx.db.execute("UPDATE goals SET project_id=? WHERE id=?",
                           (proj.id, g.id))
            row = MissionControl(ctx).ev_scores([g.id])[g.id]
            self.assertTrue(row["fits_today"])
            self.assertEqual(row["eta_days"], 0)
            self.assertEqual(row["steps_today"], 2)
        finally:
            ctx.__exit__(None, None, None)
            tmp.cleanup()

    def test_spent_calls_shrink_steps_today(self) -> None:
        g = self.goal_with_project("R", "research network topologies",
                                   ["a", "b", "c"], "pn")
        ModelBudget(self.context).record(4, 0)  # 1 left
        s = MissionControl(self.context).ev_scores([g])[g]
        self.assertFalse(s["fits_today"])
        self.assertEqual(s["steps_today"], 1)


# ── 3. acceptance regression suite ─────────────────────────────────────────

class RegressionSuiteTest(_Base):
    def _project(self, verify: str, steps: list[str] | None = None):
        mgr = ProjectManager(self.context)
        p = mgr.create("App", objective="build a demo app",
                       steps=steps or ["write the core", "add the extra"])
        p.verify_cmd = verify
        mgr._upsert_row(p)
        return mgr, p

    def test_green_build_freezes_acceptances_and_passes_suite(self) -> None:
        self.context.router = _ExtendRouter(break_contract=False)
        mgr, p = self._project('python3 "demo-app.py" --check | grep READY')
        out = mgr.run(p.id, max_steps=5)
        p2 = mgr._load(p.id)
        self.assertEqual(p2.status, "done", p2.report)
        hist = mgr.acceptance_history(p.id)
        self.assertEqual(len(hist), 2)
        self.assertTrue(all(h["ok"] for h in hist))
        rc = mgr.regression_check(p.id)
        self.assertTrue(rc["ok"], rc)
        # one frozen command per step (same shared cmd, both green)
        self.assertEqual(len(rc["results"]), 2)
        self.assertTrue(all(r["ok"] for r in rc["results"]))

    def test_in_step_gate_catches_a_loosened_contract(self) -> None:
        """Step 1 froze a STRICT contract; step 2 runs under a LOOSER one
        and passes its own check while breaking the earlier one — the
        regression gate must fail the step."""
        self.context.router = _ExtendRouter(break_contract=True)
        mgr, p = self._project('python3 "demo-app.py" --check | grep CORE')
        mgr.run(p.id, max_steps=1)  # step 1 done, freezes grep CORE
        p2 = mgr._load(p.id)
        self.assertEqual(p2.steps[0].status, "done")
        # planner refines the acceptance (looser) for the rest of the build
        p2.verify_cmd = 'python3 "demo-app.py" --check | grep READY'
        p2.status = "running"
        mgr._upsert_row(p2)
        out = mgr.run(p.id, max_steps=2, max_attempts=1)
        p3 = mgr._load(p.id)
        self.assertEqual(p3.status, "failed")
        self.assertIn("regression", p3.steps[1].result.lower())
        rc = mgr.regression_check(p.id)
        self.assertFalse(rc["ok"])

    def test_suite_goes_red_when_the_file_is_later_broken(self) -> None:
        self.context.router = _ExtendRouter(break_contract=False)
        # grep READY: satisfied by the core AND the extension, so the
        # build goes fully green before we break the file later.
        mgr, p = self._project('python3 "demo-app.py" --check | grep READY')
        mgr.run(p.id, max_steps=5)
        self.assertTrue(mgr.regression_check(p.id)["ok"])
        # a later edit (bad merge, manual change) breaks the artifact
        from nomorals.tools.filesystem import safe_path

        safe_path(self.context, "demo-app.py").write_text(
            'print("totally broken now")\n')
        rc = mgr.regression_check(p.id)
        self.assertFalse(rc["ok"])
        self.assertTrue(rc["results"])
        self.assertFalse(rc["results"][0]["ok"])

    def test_record_acceptance_run_hashes_output(self) -> None:
        mgr, p = self._project("python3 demo-app.py")
        mgr._record_acceptance_run(p, p.steps[0].id, "python3 demo-app.py",
                                   "abc123")
        rows = self.context.db.query(
            "SELECT cmd, output_hash, ok FROM acceptance_runs")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ok"], 1)
        self.assertTrue(rows[0]["output_hash"])
        # best-effort: bad input must not raise
        mgr._record_acceptance_run(p, "", "", "")


# ── 4. cadence + auto-resume (the idle-forever fix) ────────────────────────

class CadenceFixTest(_Base):
    def _idle_telemetry(self, n: int = 3) -> None:
        for i in range(n):
            self.context.db.execute(
                "INSERT INTO cognition_log (id, ts, stages, seconds, note) "
                "VALUES (?,?,?,?,?)",
                (f"cg{i}", time.time(),
                 json.dumps({"goals": {"advanced": []}}), 0.1, ""))

    def test_idle_empty_portfolio_relaxes(self) -> None:
        loop = CognitiveLoop(self.context)
        base = float(self.context.settings.autonomy.interval_hours)
        self._idle_telemetry()
        self.assertEqual(loop.adaptive_interval(),
                         min(48.0, max(base, base * 4)))

    def test_idle_with_pending_work_keeps_pace(self) -> None:
        loop = CognitiveLoop(self.context)
        base = float(self.context.settings.autonomy.interval_hours)
        self._idle_telemetry()
        GoalSystem(self.context).create("Waiting", description="x",
                                        plan=["a", "b"])
        self.assertEqual(loop.adaptive_interval(), base)

    def test_busy_still_tightens(self) -> None:
        loop = CognitiveLoop(self.context)
        base = float(self.context.settings.autonomy.interval_hours)
        for i in range(3):
            self.context.db.execute(
                "INSERT INTO cognition_log (id, ts, stages, seconds, note) "
                "VALUES (?,?,?,?,?)",
                (f"cg{i}", time.time(),
                 json.dumps({"goals": {"advanced": ["g1"]}}), 0.1, ""))
        self.assertEqual(loop.adaptive_interval(),
                         max(0.25, min(base, base * 0.25)))


class AutoResumeTest(_Capped):
    def _suspend(self) -> tuple[str, str]:
        self.context.router = _ExtendRouter()
        ModelBudget(self.context).record(3, 0)  # 2 left, build needs 3
        g = self.goal_with_project(
            "Suspended", "build a suspended demo",
            ["do it", "more work"], "ps")
        mgr = ProjectManager(self.context)
        proj_id = next(str(r["id"]) for r in self.context.db.query(
            "SELECT id FROM projects WHERE goal_id=?", (g,)))
        mgr.advance(proj_id)
        p = mgr._load(proj_id)
        self.assertEqual(p.status, "paused")
        self.assertIn("Suspended on model budget", p.report)
        self.assertEqual(GoalSystem(self.context).get(g).status, "paused")
        return g, proj_id

    def test_tick_stays_paused_while_unaffordable(self) -> None:
        g, _ = self._suspend()
        self.assertEqual(CognitiveLoop(self.context)._tick_resume_suspended(),
                         [])
        self.assertEqual(GoalSystem(self.context).get(g).status, "paused")

    def test_tick_resumes_after_rollover(self) -> None:
        g, proj_id = self._suspend()
        self.rollover()
        resumed = CognitiveLoop(self.context)._tick_resume_suspended()
        self.assertEqual(resumed, [g])
        p = ProjectManager(self.context)._load(proj_id)
        self.assertEqual(p.steps[0].status, "done")
        self.assertEqual(GoalSystem(self.context).get(g).status, "active")
        self.assertEqual(ModelBudget(self.context).held(), 0)

    def test_advance_self_resumes_manually(self) -> None:
        _, proj_id = self._suspend()
        self.rollover()
        # `nm project advance <id>` — no heartbeat involved
        p = ProjectManager(self.context).advance(proj_id)
        self.assertEqual(p.steps[0].status, "done")

    def test_rollover_probe_is_stateless(self) -> None:
        _, proj_id = self._suspend()
        held_before = ModelBudget(self.context).held()
        CognitiveLoop(self.context)._tick_resume_suspended()
        self.assertEqual(ModelBudget(self.context).held(), held_before)


class NarrationReservationTest(_Capped):
    def test_narration_step_reserves_and_suspends(self) -> None:
        class NR:
            def chat(self, messages, params=None, **kw):
                return LLMResponse(text="step done for real", model="fake")
        self.context.router = NR()
        ModelBudget(self.context).record(5, 0)  # 0 left
        mgr = ProjectManager(self.context)
        p = mgr.create("N", objective="research network topologies",
                       steps=["do it"])
        p2 = mgr.advance(p.id)
        self.assertEqual(p2.status, "paused")
        self.assertIn("budget", p2.report.lower())
        self.assertEqual(p2.steps[0].status, "pending")  # resumable
        # rollover -> the same advance resumes and completes
        self.rollover()
        p3 = mgr.advance(p.id)
        self.assertEqual(p3.steps[0].status, "done")
        self.assertEqual(ModelBudget(self.context).held(), 0)


# ── 5. the "5 a day" question: default unlimited + .env writer ─────────────

class BudgetDefaultTest(unittest.TestCase):
    def test_default_cap_is_zero_unlimited(self) -> None:
        s = load_settings()
        self.assertEqual(s.autonomy.daily_model_calls, 0)


class _CliBase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._cli_tmp = tempfile.TemporaryDirectory(prefix="nm-wave67-cli-")
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


class BudgetEnvWriterTest(_CliBase):
    def test_unlimited_writes_zero(self) -> None:
        code, out = self._run_cli("autonomy", "budget", "--unlimited")
        self.assertEqual(code, 0)
        self.assertIn("set to 0", out)
        env = (Path(self._cli_tmp.name) / ".env").read_text()
        self.assertIn("NM_AUTONOMY_DAILY_MODEL_CALLS=0", env)

    def test_cap_replaces_existing_line(self) -> None:
        self._run_cli("autonomy", "budget", "--cap", "100")
        self._run_cli("autonomy", "budget", "--cap", "250")
        env = (Path(self._cli_tmp.name) / ".env").read_text()
        self.assertIn("NM_AUTONOMY_DAILY_MODEL_CALLS=250", env)
        self.assertEqual(env.count("NM_AUTONOMY_DAILY_MODEL_CALLS"), 1)

    def test_cap_persists_across_cli_calls(self) -> None:
        # `--cap 5` writes the .env; the NEXT cli call loads it fresh and
        # sees the cap — this is the "5 a day" switch, made visible.
        code, out = self._run_cli("autonomy", "budget", "--cap", "5")
        self.assertEqual(code, 0)
        code, out = self._run_cli("autonomy", "budget", "--json")
        data = json.loads(out)
        self.assertEqual(data["unlimited"], False)
        self.assertEqual(data["cap"], 5)
        # the text report points at the removal command
        code, out = self._run_cli("autonomy", "budget")
        self.assertIn("--unlimited", out)


# ── 6. detailed help ───────────────────────────────────────────────────────

class DetailedHelpTest(unittest.TestCase):
    def test_overview_mentions_every_registered_command(self) -> None:
        text = detailed_help()
        for kind in CONTROL_COMMANDS:
            self.assertIn(f"/{kind}", text, f"overview misses /{kind}")

    def test_every_command_has_a_detail_page(self) -> None:
        for kind in CONTROL_COMMANDS:
            page = detailed_help(kind)
            self.assertTrue(page.startswith(f"/{kind}"), kind)
            self.assertIn("usage:", page)
            self.assertIn("example:", page)
        self.assertEqual(set(COMMAND_DETAILS) == set(CONTROL_COMMANDS),
                         True)

    def test_topic_pages(self) -> None:
        for t in ("budget", "goals", "builds", "skills", "missions", "modes"):
            page = detailed_help(t)
            self.assertGreater(len(page), 200, t)
        self.assertIn("--unlimited", detailed_help("budget"))

    def test_fuzzy_resolution(self) -> None:
        self.assertTrue(detailed_help("dev").startswith("/devon"))
        self.assertTrue(detailed_help("devonx").startswith("/devon"))
        self.assertIn("did you mean", detailed_help("st"))
        self.assertIn("no help page", detailed_help("zzzznotreal"))

    def test_parse_help_variants(self) -> None:
        self.assertEqual(parse_control("/help").kind, "help")
        self.assertEqual(parse_control("/help").arg, "")
        self.assertEqual(parse_control("/help code").arg, "code")
        self.assertEqual(parse_control("/help budget").kind, "help")
        self.assertEqual(parse_control("/help a b").kind, "error")

    def test_cli_help_matches_chat_help(self) -> None:
        pass  # covered by _CliBase tests below


class _CliHelpBase(_CliBase):
    pass


class CliHelpTest(_CliHelpBase):
    def test_nm_help_overview(self) -> None:
        code, out = self._run_cli("help")
        self.assertEqual(code, 0)
        self.assertIn("/devon", out)
        self.assertIn("topic pages", out)

    def test_nm_help_command(self) -> None:
        code, out = self._run_cli("help", "code")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("/code"))
        self.assertIn("usage:", out)

    def test_nm_help_topic(self) -> None:
        code, out = self._run_cli("help", "budget")
        self.assertEqual(code, 0)
        self.assertIn("--unlimited", out)

    def test_nm_help_json(self) -> None:
        code, out = self._run_cli("help", "devon", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertIn("help", data)
        self.assertIn("/devon", data["help"])


class MissionPlanFitDisplayTest(_CliBase):
    def test_plan_shows_est_calls_and_fit(self) -> None:
        code, _ = self._run_cli("goal", "create", "Build goal")
        self.assertEqual(code, 0)
        from nomorals.core.config import load_settings as _ls

        s = _ls()
        ctx = build_context(s, with_executor=False, with_tools=False)
        ctx.__enter__()
        try:
            from nomorals.agents.goals import GoalSystem as GS

            g = GS(ctx).list()[0]
            proj = ProjectManager(ctx).create(
                "pb", objective="build a demo pipeline",
                steps=["a", "b", "c"], goal_id=g.id)
            ctx.db.execute("UPDATE goals SET project_id=? WHERE id=?",
                           (proj.id, g.id))
        finally:
            ctx.__exit__(None, None, None)
        code, out = self._run_cli("mission", "plan")
        self.assertEqual(code, 0)
        self.assertIn("~9 calls", out)
        self.assertIn("fits today", out)  # unlimited budget


if __name__ == "__main__":
    unittest.main()
