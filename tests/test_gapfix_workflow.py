"""Failing-first tests for the Devon /workflow build.

Antigravity Dynamic Workflows, Devon version: saved reviewable
workflow scripts with schema-typed handoffs, per-agent worktrees,
multi-turn handles, scheduler runner. NO review-before-run gate
(owner order 2026-11-16: "Not necessary I review") — a drafted
script runs as soon as it is written; the saved script is the
review surface.

Tests are written against the organ API defined in
~/workspace/workflow-verify/WORKFLOW_SURVEY.md before the wiring
exists; they must FAIL first (missing modules) and pass after.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.agents.workflow.errors import WorkflowError
from nomorals.agents.workflow.schema import validate_result
from nomorals.agents.workflow.script import (
    ScriptRuntime,
    WorkflowHandle,
    validate_script,
)
from nomorals.agents.workflow.store import (
    WorkflowStore,
    default_workflows_dir,
    default_workflows_path,
    new_workflow_id,
)
from nomorals.agents.workflow.worktree import (
    create_worktree,
    remove_worktree,
)


class FakeDelegationRun:
    """Minimal stand-in for DelegationRun: status + result_dict()."""

    def __init__(self, run_id, *, status="completed", outcome=None,
                 error=""):
        self.id = run_id
        self.status = status
        self._outcome = outcome if outcome is not None else {"ok": True}
        self.error = error

    def result_dict(self):
        return dict(self._outcome)


class FakeManager:
    """Scripted stand-in for DelegationManager.spawn/wait/get_run.

    outcomes: {(role, goal-substring): outcome-dict or Exception}.
    Every spawn is recorded on .spawns as (role, goal, input).
    """

    ROSTER = ("planner", "implementer", "reviewer", "tester",
              "refactorer", "dep_hunter", "api_designer")

    def __init__(self, outcomes=None):
        self.outcomes = dict(outcomes or {})
        self.spawns = []
        self._runs = {}
        self._n = 0

    def spawn(self, *, role, goal, input=None, mission_id="",
              scope=(), step_budget=0, chat_key="", name=""):
        if role not in self.ROSTER:
            raise ValueError(
                f"unknown roster role {role!r}; roster: "
                f"{sorted(self.ROSTER)}")
        if role == "implementer" and not (input or {}).get("path"):
            raise ValueError("spawn needs path for implementer")
        self._n += 1
        run_id = f"dr-fake{self._n:03d}"
        self.spawns.append((role, goal, dict(input or {})))
        outcome = {"ok": True, "result": f"did: {goal}"}
        for (r, needle), value in self.outcomes.items():
            if r == role and needle in goal:
                outcome = value
                break
        if isinstance(outcome, BaseException):
            raise outcome
        status = "completed" if outcome.get("ok") else "failed"
        self._runs[run_id] = FakeDelegationRun(
            run_id, status=status, outcome=outcome,
            error="" if outcome.get("ok") else str(outcome.get("error")))
        return self._runs[run_id]

    def wait(self, run_id, *, timeout=30.0):
        run = self._runs.get(run_id)
        if run is None:
            raise ValueError(f"no delegation run {run_id!r}")
        return run

    def get_run(self, run_id):
        return self._runs.get(run_id)


def _tmp_home(testcase):
    import tempfile
    tmp = tempfile.mkdtemp(dir=str(
        Path.home() / "workspace" / "workflow-verify"))
    testcase.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
    return Path(tmp)


class StoreTests(unittest.TestCase):
    def test_save_and_get_round_trips_script_and_state(self):
        store = WorkflowStore()
        wf = store.save_workflow(
            name="review-changes", task="review the changed files",
            script="phase(\"Review\")\nlog(\"hi\")\n")
        self.assertEqual(wf["state"], "queued")
        self.assertEqual(wf["script"], "phase(\"Review\")\nlog(\"hi\")\n")
        self.assertTrue(wf["id"].startswith("wf-"))
        got = store.get_workflow(wf["id"])
        self.assertEqual(got["name"], "review-changes")
        self.assertEqual(got["task"], "review the changed files")

    def test_duplicate_name_is_refused_by_name(self):
        store = WorkflowStore()
        store.save_workflow(name="dup", task="one", script="log(1)\n")
        with self.assertRaises(WorkflowError) as caught:
            store.save_workflow(name="dup", task="two", script="log(2)\n")
        self.assertIn("dup", str(caught.exception))

    def test_resolve_by_id_prefix_and_name(self):
        store = WorkflowStore()
        wf = store.save_workflow(name="my-flow", task="t", script="log(1)\n")
        self.assertEqual(store.resolve(wf["id"])["name"], "my-flow")
        self.assertEqual(
            store.resolve(wf["id"][:6])["name"], "my-flow")
        self.assertEqual(store.resolve("my-flow")["name"], "my-flow")
        self.assertIsNone(store.resolve("no-such-workflow"))

    def test_saved_workflow_is_queued_without_any_approval(self):
        # Owner correction 2026-11-16 ("Not necessary I review"):
        # no review-before-run gate — a saved script is queued and
        # the runner picks it up. There is no approve/deny step.
        store = WorkflowStore()
        wf = store.save_workflow(name="a", task="t", script="log(1)\n")
        self.assertEqual(store.get_workflow(wf["id"])["state"],
                         "queued")
        self.assertFalse(hasattr(store, "approve"))
        self.assertFalse(hasattr(store, "request_approval"))

    def test_cancel_keeps_the_script_for_reading(self):
        store = WorkflowStore()
        wf = store.save_workflow(name="d", task="t", script="log(1)\n")
        store.cancel(wf["id"], by="owner")
        got = store.get_workflow(wf["id"])
        self.assertEqual(got["state"], "cancelled")
        self.assertEqual(got["script"], "log(1)\n")

    def test_cancel_and_resume_states(self):
        store = WorkflowStore()
        wf = store.save_workflow(name="c", task="t", script="log(1)\n")
        self.assertEqual(store.get_workflow(wf["id"])["state"],
                         "queued")
        store.mark_running(wf["id"])
        store.cancel(wf["id"], by="owner")
        self.assertEqual(store.get_workflow(wf["id"])["state"],
                         "cancelled")
        with self.assertRaises(WorkflowError):
            store.resume(wf["id"], by="owner")
        store.requeue(wf["id"], by="owner")
        self.assertEqual(store.get_workflow(wf["id"])["state"],
                         "queued")
        wf2 = store.save_workflow(name="r", task="t", script="log(1)\n")
        store.mark_running(wf2["id"])
        store.mark_interrupted(wf2["id"])
        store.resume(wf2["id"], by="owner")
        self.assertEqual(store.get_workflow(wf2["id"])["state"],
                         "queued")

    def test_phase_journal_survives_disk_reload(self):
        home = _tmp_home(self)
        store = WorkflowStore(path=default_workflows_path(home),
                              scripts_dir=default_workflows_dir(home))
        wf = store.save_workflow(name="j", task="t",
                                 script="phase(\"A\")\nlog(\"x\")\n")
        store.start_phase(wf["id"], "A")
        store.append_log(wf["id"], "hello phase")
        store.record_agent_result(wf["id"], "reviewer", "ok")
        store.finish_phase(wf["id"], "A")
        store2 = WorkflowStore(path=default_workflows_path(home),
                               scripts_dir=default_workflows_dir(home))
        got = store2.get_workflow(wf["id"])
        self.assertEqual(len(got["phases"]), 1)
        self.assertEqual(got["phases"][0]["name"], "A")
        self.assertIn("hello phase", got["phases"][0]["log"])
        self.assertEqual(got["phases"][0]["agents"][0]["role"],
                         "reviewer")

    def test_script_file_is_written_beside_the_index(self):
        home = _tmp_home(self)
        store = WorkflowStore(path=default_workflows_path(home),
                              scripts_dir=default_workflows_dir(home))
        wf = store.save_workflow(name="f", task="t",
                                 script="log(\"saved\")\n")
        script_path = Path(wf["script_path"])
        self.assertTrue(script_path.exists())
        self.assertEqual(script_path.read_text(encoding="utf-8"),
                         "log(\"saved\")\n")

    def test_corrupt_index_degrades_to_empty_with_reason(self):
        home = _tmp_home(self)
        path = default_workflows_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        store = WorkflowStore(path=path,
                              scripts_dir=default_workflows_dir(home))
        self.assertEqual(store.list_workflows(), [])
        self.assertIn("corrupt", store.degraded_reason)
        quarantined = list(path.parent.glob("workflows.json.corrupt-*"))
        self.assertEqual(len(quarantined), 1)

    def test_bad_rows_are_skipped(self):
        home = _tmp_home(self)
        path = default_workflows_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"workflows": [
            {"id": "wf-good", "name": "good", "task": "t",
             "script": "log(1)\n", "state": "draft"},
            {"id": 42, "name": "bad-row"},
            {"name": "no-id"},
        ]}), encoding="utf-8")
        store = WorkflowStore(path=path,
                              scripts_dir=default_workflows_dir(home))
        names = [w["name"] for w in store.list_workflows()]
        self.assertEqual(names, ["good"])

    def test_new_workflow_id_shape(self):
        wid = new_workflow_id()
        self.assertTrue(wid.startswith("wf-"))
        self.assertEqual(len(wid), 15)


class SchemaTests(unittest.TestCase):
    def test_string_and_array_and_object_pass_through(self):
        self.assertEqual(
            validate_result("hi", {"type": "string"}), "hi")
        self.assertEqual(
            validate_result(["a.py"], {"type": "array",
                                       "items": {"type": "string"}}),
            ["a.py"])
        self.assertEqual(
            validate_result({"n": 1}, {"type": "object",
                                       "properties": {
                                           "n": {"type": "integer"}},
                                       "required": ["n"]}),
            {"n": 1})

    def test_wrong_type_names_the_field(self):
        with self.assertRaises(WorkflowError) as caught:
            validate_result(3, {"type": "string"}, where="result")
        self.assertIn("result", str(caught.exception))

    def test_missing_required_key_is_named(self):
        with self.assertRaises(WorkflowError) as caught:
            validate_result({}, {"type": "object", "required": ["files"]})
        self.assertIn("files", str(caught.exception))

    def test_bool_is_not_an_integer(self):
        with self.assertRaises(WorkflowError):
            validate_result(True, {"type": "integer"})

    def test_enum_violation_is_named(self):
        with self.assertRaises(WorkflowError) as caught:
            validate_result("maybe", {"type": "string",
                                      "enum": ["yes", "no"]})
        self.assertIn("maybe", str(caught.exception))

    def test_unknown_schema_type_is_an_error(self):
        with self.assertRaises(WorkflowError):
            validate_result("x", {"type": "wizard"})


class ScriptValidatorTests(unittest.TestCase):
    CLEAN = (
        "phase(\"Discover\")\n"
        "files = agent(\"List the changed files.\",\n"
        "             schema={\"type\": \"array\","
        " \"items\": {\"type\": \"string\"}})\n"
        "log(\"found\", len(files))\n"
        "reviews = parallel(files, lambda f: agent(\"Review \" + f))\n"
        "done = pipeline(files, lambda f: log(f))\n"
    )

    def test_clean_script_passes(self):
        self.assertEqual(validate_script(self.CLEAN), [])

    def test_import_is_rejected_naming_the_line(self):
        errs = validate_script("import os\nlog(1)\n")
        self.assertTrue(errs)
        self.assertIn("1", errs[0])
        self.assertIn("import", errs[0].lower())

    def test_from_import_is_rejected(self):
        errs = validate_script("from pathlib import Path\nlog(1)\n")
        self.assertTrue(errs)

    def test_open_exec_eval_are_rejected(self):
        for bad in ("open('x').read()\n", "eval('1')\n",
                    "exec('log(1)')\n", "__import__('os')\n"):
            with self.subTest(bad=bad):
                self.assertTrue(validate_script(bad), bad)

    def test_dunder_attribute_is_rejected(self):
        errs = validate_script("x = agent.__class__\n")
        self.assertTrue(errs)

    def test_syntax_error_is_reported_not_raised(self):
        errs = validate_script("phase(\n")
        self.assertTrue(errs)
        self.assertIn("line", errs[0].lower())

    def test_while_loop_is_rejected(self):
        # Scripts run on a background thread with no kill switch:
        # unbounded loops hang the run. Finite work uses for /
        # parallel / pipeline over a list.
        errs = validate_script("while True:\n    log(1)\n")
        self.assertTrue(errs)

    def test_empty_script_is_rejected(self):
        self.assertTrue(validate_script(""))

    def test_unknown_name_fails_at_runtime_not_statically(self):
        # The static validator rejects dangerous constructs only;
        # an unknown name is a runtime NameError, recorded as a
        # failed run by the runner (pinned in ScriptRuntimeTests).
        self.assertEqual(validate_script("frobnicate(1)\n"), [])


class ScriptRuntimeTests(unittest.TestCase):
    def _runtime(self, manager=None, outcomes=None):
        store = WorkflowStore()
        wf = store.save_workflow(name="rt", task="t", script="log(1)\n")
        manager = manager if manager is not None else FakeManager(outcomes)
        runtime = ScriptRuntime(run=wf, store=store, manager=manager)
        return runtime, store, wf, manager

    def test_phase_and_log_journal_durably(self):
        runtime, store, wf, _ = self._runtime()
        runtime.phase("Discover")
        runtime.log("found", 3, "files")
        got = store.get_workflow(wf["id"])
        self.assertEqual(got["phases"][0]["name"], "Discover")
        self.assertIn("found 3 files", got["phases"][0]["log"])

    def test_agent_one_shot_returns_validated_result(self):
        runtime, _, _, manager = self._runtime(
            outcomes={("reviewer", "Review"): {
                "ok": True, "result": "riskiest: none"}})
        out = runtime.agent(
            "Review the diff.", schema={"type": "string"},
            role="reviewer")
        self.assertEqual(out, "riskiest: none")
        self.assertEqual(manager.spawns[0][0], "reviewer")

    def test_agent_schema_failure_returns_error_dict(self):
        from nomorals.agents.workflow import failures as wfailures
        wfailures.clear_failures()
        runtime, _, _, _ = self._runtime(
            outcomes={("planner", "List"): {
                "ok": True, "result": 42}})
        out = runtime.agent("List files.", schema={"type": "string"},
                            role="planner")
        self.assertFalse(out["ok"])
        self.assertIn("error", out)
        self.assertGreaterEqual(wfailures.failure_count(), 1)

    def test_agent_unknown_role_names_the_roster(self):
        runtime, _, _, _ = self._runtime()
        out = runtime.agent("Do it.", role="wizard")
        self.assertFalse(out["ok"])
        self.assertIn("planner", out["error"])

    def test_agent_missing_required_input_names_the_field(self):
        runtime, _, _, _ = self._runtime()
        out = runtime.agent("Fix it.", role="implementer")
        self.assertFalse(out["ok"])
        self.assertIn("path", out["error"])

    def test_parallel_keeps_order_and_isolates_failure(self):
        runtime, _, _, _ = self._runtime(
            outcomes={("reviewer", "bad"): {
                "ok": False, "error": "boom"}})
        results = runtime.parallel(
            ["good", "bad", "good2"],
            lambda item: runtime.agent(
                f"Review {item}", role="reviewer"))
        self.assertEqual(len(results), 3)
        # Success returns the unwrapped value; a failure returns
        # the error envelope — the script decides what to do.
        self.assertEqual(results[0], "did: Review good")
        self.assertFalse(results[1]["ok"])
        self.assertIn("boom", results[1]["error"])
        self.assertEqual(results[2], "did: Review good2")
        # The two successes, in order, are the plain values.
        self.assertEqual(
            [r for r in results if isinstance(r, str)],
            ["did: Review good", "did: Review good2"])

    def test_pipeline_is_sequential_and_ordered(self):
        runtime, _, _, _ = self._runtime()
        order = []
        results = runtime.pipeline(
            ["a", "b", "c"],
            lambda item: order.append(item) or f"done-{item}")
        self.assertEqual(results, ["done-a", "done-b", "done-c"])
        self.assertEqual(order, ["a", "b", "c"])

    def test_multi_turn_handle_carries_prior_turns(self):
        runtime, _, _, manager = self._runtime()
        handle = runtime.agent(role="reviewer")
        self.assertIsInstance(handle, WorkflowHandle)
        handle("List the bugs.")
        handle("Verify the fixes.")
        self.assertEqual(len(manager.spawns), 2)
        second_goal = manager.spawns[1][1]
        self.assertIn("List the bugs.", second_goal)
        self.assertIn("Verify the fixes.", second_goal)

    def test_trajectory_reads_the_conversation_log(self):
        class FakeLog:
            def history(self, conversation_id, *, limit):
                # Real ConversationLog.history is newest-first.
                return [{"role": "assistant",
                         "content": "earlier reply"},
                        {"role": "user", "content": "earlier task"}]

        runtime, _, _, _ = self._runtime()
        runtime._conversation_log = FakeLog()
        runtime._conversation_id = "conv-1"
        steps = runtime.trajectory(limit=2)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["content"], "earlier task")

    def test_run_executes_a_clean_script_end_to_end(self):
        runtime, store, wf, _ = self._runtime()
        summary = runtime.run(
            "phase(\"Done\")\n"
            "log(\"all good\")\n")
        self.assertEqual(summary["state"], "done")
        got = store.get_workflow(wf["id"])
        self.assertIn("all good", got["phases"][0]["log"])

    def test_run_rejects_an_invalid_script_without_executing(self):
        # run() never raises: an invalid script is a failed run
        # with the reason recorded — and nothing in it executes.
        runtime, store, wf, _ = self._runtime()
        summary = runtime.run("import os\nlog(\"ran\")\n")
        self.assertEqual(summary["state"], "failed")
        self.assertIn("invalid", summary["error"])
        got = store.get_workflow(wf["id"])
        self.assertEqual(got["state"], "failed")
        self.assertIn("import", got["error"])
        self.assertEqual(got["phases"], [])

    def test_run_records_a_script_error_as_failed(self):
        runtime, store, wf, _ = self._runtime()
        summary = runtime.run("raise ValueError('kaboom')\n")
        self.assertEqual(summary["state"], "failed")
        self.assertIn("kaboom", store.get_workflow(wf["id"])["error"])

    def test_run_records_unknown_name_as_failed(self):
        runtime, store, wf, _ = self._runtime()
        summary = runtime.run("frobnicate(1)\n")
        self.assertEqual(summary["state"], "failed")
        self.assertIn("frobnicate", store.get_workflow(wf["id"])["error"])


if __name__ == "__main__":
    unittest.main()


def _make_git_repo(path: Path) -> Path:
    """A real tiny git repo at path (one commit on main)."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)],
                   check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email",
                    "wf@test.local"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name",
                    "Workflow Test"], check=True)
    (path / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."],
                   check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m",
                    "first"], check=True)
    rev = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    return rev


class WorktreeTests(unittest.TestCase):
    def test_create_and_remove_against_a_real_repo(self):
        home = _tmp_home(self)
        repo = home / "repo"
        rev = _make_git_repo(repo)
        dest = home / "wt" / "agent-1"
        mechanism = create_worktree(repo, dest, rev)
        self.assertEqual(mechanism, "worktree")
        self.assertTrue((dest / "a.txt").exists())
        self.assertEqual((dest / "a.txt").read_text(
            encoding="utf-8"), "hello\n")
        head = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(head, rev)
        self.assertTrue(remove_worktree(repo, dest))
        self.assertFalse(dest.exists())

    def test_bogus_revision_is_an_honest_error(self):
        home = _tmp_home(self)
        repo = home / "repo"
        _make_git_repo(repo)
        with self.assertRaises(WorkflowError) as caught:
            create_worktree(repo, home / "wt" / "x", "deadbeef")
        self.assertIn("deadbeef", str(caught.exception))

    def test_worktree_failure_falls_back_to_copy(self):
        home = _tmp_home(self)
        repo = home / "repo"
        rev = _make_git_repo(repo)
        calls = []

        def failing_first_git(args, **kw):
            calls.append(list(args))
            if "worktree" in args:
                return subprocess.CompletedProcess(args, 128, "", "nope")
            return subprocess.run(args, **kw)

        dest = home / "wt" / "agent-2"
        mechanism = create_worktree(repo, dest, rev, git=failing_first_git)
        self.assertEqual(mechanism, "copy")
        self.assertTrue((dest / "a.txt").exists())


class _FakeNotifier:
    def __init__(self):
        self.sent = []

    def publish(self, kind, title="", body="", **kw):
        self.sent.append((kind, title, body))
        return {"ok": True}


class _FakeContext:
    def __init__(self, db=None, home=None):
        self.db = db
        self.home = home


class RunnerTests(unittest.TestCase):
    def _queued_run(self, store, name="flow"):
        wf = store.save_workflow(name=name, task="do the thing",
                                 script="phase(\"Work\")\nlog(\"done\")\n",
                                 chat_key="owner-dm")
        return wf

    def test_tick_starts_queued_run_and_delivers_summary(self):
        from nomorals.agents.workflow.runner import (
            run_workflow_tick_once,
        )
        store = WorkflowStore()
        wf = self._queued_run(store)
        notifier = _FakeNotifier()
        manager = FakeManager()
        out = run_workflow_tick_once(
            _FakeContext(), store=store, manager=manager,
            notifier=notifier)
        self.assertTrue(out["ok"])
        self.assertEqual(out["started"], 1)
        # The run executes on its thread; give it a moment, then tick.
        deadline = time.time() + 10
        while time.time() < deadline:
            got = store.get_workflow(wf["id"])
            if got["state"] in ("done", "failed"):
                break
            run_workflow_tick_once(
                _FakeContext(), store=store, manager=manager,
                notifier=notifier)
            time.sleep(0.05)
        # Delivery is a tick action, not a thread action: the
        # thread marks the run done, the NEXT tick delivers it.
        run_workflow_tick_once(
            _FakeContext(), store=store, manager=manager,
            notifier=notifier)
        got = store.get_workflow(wf["id"])
        self.assertEqual(got["state"], "done")
        self.assertTrue(
            any("done" in body for _, _, body in notifier.sent))

    def test_tick_never_raises_when_the_store_raises(self):
        from nomorals.agents.workflow.runner import (
            run_workflow_tick_once,
        )

        class BrokenStore:
            def list_workflows(self):
                raise RuntimeError("store is down")

        out = run_workflow_tick_once(_FakeContext(),
                                     store=BrokenStore())
        self.assertFalse(out["ok"])
        self.assertIn("store is down", out["error"])

    def test_restart_orphaned_run_is_marked_interrupted(self):
        from nomorals.agents.workflow.runner import (
            run_workflow_tick_once,
        )
        store = WorkflowStore()
        wf = self._queued_run(store)
        store.mark_running(wf["id"])
        store.start_phase(wf["id"], "Work")
        store.append_log(wf["id"], "halfway")
        # No live thread exists for this run (process restarted):
        # the tick must mark it interrupted, journal intact.
        out = run_workflow_tick_once(_FakeContext(), store=store,
                                     manager=FakeManager())
        self.assertTrue(out["ok"])
        got = store.get_workflow(wf["id"])
        self.assertEqual(got["state"], "interrupted")
        self.assertIn("halfway", got["phases"][0]["log"])

    def test_tick_with_nothing_to_do_is_quiet(self):
        from nomorals.agents.workflow.runner import (
            run_workflow_tick_once,
        )
        out = run_workflow_tick_once(_FakeContext(),
                                     store=WorkflowStore(),
                                     manager=FakeManager())
        self.assertTrue(out["ok"])
        self.assertEqual(out["started"], 0)

    def test_ensure_workflow_job_is_idempotent(self):
        from nomorals.agents.workflow.runner import (
            WORKFLOW_JOB,
            WORKFLOW_TOOL,
            ensure_workflow_job,
        )

        class FakeScheduler:
            def __init__(self):
                self.jobs = []

            def list_jobs(self):
                return list(self.jobs)

            def add(self, name, spec, kind, payload, routine=False):
                job = {"id": f"job-{len(self.jobs)}", "name": name,
                       "spec": spec, "kind": kind, "payload": payload,
                       "routine": routine}
                self.jobs.append(job)
                return job

            def remove(self, job_id):
                self.jobs = [j for j in self.jobs if j["id"] != job_id]

        sched = FakeScheduler()
        first = ensure_workflow_job(sched)
        second = ensure_workflow_job(sched)
        self.assertEqual(first, second)
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(first["name"], WORKFLOW_JOB)
        self.assertEqual(first["payload"]["tool"], WORKFLOW_TOOL)
        # A stale spec is replaced, not duplicated.
        stale = ensure_workflow_job(sched, spec="every 1m")
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(stale["spec"], "every 1m")


class LifecycleTests(unittest.TestCase):
    def test_rerun_queues_again_without_approval(self):
        from nomorals.agents.workflow.runner import rerun_workflow
        store = WorkflowStore()
        wf = store.save_workflow(name="again", task="t",
                                 script="log(1)\n")
        store.mark_running(wf["id"])
        store.mark_done(wf["id"], summary="finished")
        rerun_workflow(store, wf["id"], by="owner")
        self.assertEqual(store.get_workflow(wf["id"])["state"],
                         "queued")

    def test_cancel_from_queued_is_at_once(self):
        from nomorals.agents.workflow.runner import cancel_workflow
        store = WorkflowStore()
        wf = store.save_workflow(name="stop", task="t",
                                 script="log(1)\n")
        cancel_workflow(store, wf["id"], by="owner")
        self.assertEqual(store.get_workflow(wf["id"])["state"],
                         "cancelled")

    def test_lifecycle_verbs_need_an_actor(self):
        from nomorals.agents.workflow.runner import (
            cancel_workflow,
            rerun_workflow,
            resume_workflow,
        )
        store = WorkflowStore()
        wf = store.save_workflow(name="g", task="t", script="log(1)\n")
        for fn in (rerun_workflow, cancel_workflow, resume_workflow):
            with self.assertRaises(WorkflowError):
                fn(store, wf["id"], by="")
        cancel_workflow(store, wf["id"], by="owner")
        self.assertEqual(store.get_workflow(wf["id"])["state"],
                         "cancelled")


class LayeringGuardTests(unittest.TestCase):
    def test_organ_never_imports_the_runtime_or_tools_at_top_level(self):
        base = Path(__file__).resolve().parent.parent / "nomorals" \
            / "agents" / "workflow"
        banned = ("agents.partner", "tools.autonomy", "nomorals.os",
                  "agents.partner.runtime")
        for path in sorted(base.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            head = text[:2000]
            for bad in banned:
                self.assertNotIn(f"from ..{bad}", head, path.name)
                self.assertNotIn(f"from nomorals.{bad}", head, path.name)
                self.assertNotIn(f"import {bad}", head, path.name)

    def test_tick_tool_imports_the_organ_lazily_only(self):
        text = (Path(__file__).resolve().parent.parent / "nomorals"
                / "tools" / "autonomy.py").read_text(encoding="utf-8")
        self.assertIn("__workflow_tick__", text)
        head = text.split("def ")[0]
        self.assertNotIn("workflow", head)


class SkillFileTests(unittest.TestCase):
    def test_skill_doc_exists_and_states_the_sync_form(self):
        skill = (Path(__file__).resolve().parent.parent / "nomorals"
                 / "skills" / "library" / "dynamic-workflows.md")
        self.assertTrue(skill.exists())
        text = skill.read_text(encoding="utf-8")
        for token in ("phase(", "parallel(", "pipeline(", "agent(",
                      "planner", "reviewer", "queued"):
            self.assertIn(token, text)

    def test_skill_index_lists_it(self):
        index = (Path(__file__).resolve().parent.parent / "nomorals"
                 / "skills" / "library" / "INDEX.md")
        self.assertIn("dynamic-workflows", index.read_text(
            encoding="utf-8"))


class ControlMixinTests(unittest.TestCase):
    """The /workflow chat control on the mixin (fake brain)."""

    _SCRIPT = 'phase("Work")\nlog("hi")\n'

    def _fake_runtime(self, script_text=_SCRIPT):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_workflow import (
            RuntimeWorkflowMixin,
        )

        home = _tmp_home(self)

        class FakeRuntime(RuntimeWorkflowMixin):
            pass

        runtime = FakeRuntime()
        runtime.settings = SimpleNamespace(home_path=home)
        runtime.context = SimpleNamespace()
        runtime._workflow_llm = lambda prompt: script_text
        return runtime

    def test_draft_task_saves_shows_and_queues(self):
        runtime = self._fake_runtime()
        reply = runtime._control_workflow("review the retry path")
        self.assertIn("saved as review-the-retry-path", reply)
        self.assertIn("queued", reply)
        self.assertIn("```python", reply)
        self.assertIn('phase("Work")', reply)
        store = runtime._workflow_store()
        rows = store.list_workflows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "queued")
        script_file = Path(rows[0]["script_path"])
        self.assertTrue(script_file.exists())
        self.assertIn('phase("Work")',
                      script_file.read_text(encoding="utf-8"))

    def test_list_and_show_and_status(self):
        runtime = self._fake_runtime()
        runtime._control_workflow("review the retry path")
        self.assertIn("review-the-retry-path",
                      runtime._control_workflow("list"))
        shown = runtime._control_workflow("show review-the-retry-path")
        self.assertIn('phase("Work")', shown)
        self.assertIn("[queued]", shown)
        status = runtime._control_workflow(
            "status review-the-retry-path")
        self.assertIn("[queued]", status)
        self.assertIn("review the retry path", status)

    def test_run_requeues_a_done_workflow(self):
        runtime = self._fake_runtime()
        store = runtime._workflow_store()
        row = store.save_workflow(
            name="done-one", task="t", script=self._SCRIPT)
        store.mark_running(row["id"])
        store.mark_done(row["id"], summary="all good")
        reply = runtime._control_workflow("run done-one")
        self.assertIn("queued again", reply)
        self.assertEqual(
            store.get_workflow(row["id"])["state"], "queued")

    def test_cancel_cancels_a_queued_workflow(self):
        runtime = self._fake_runtime()
        runtime._control_workflow("review the retry path")
        reply = runtime._control_workflow(
            "cancel review-the-retry-path")
        self.assertIn("cancelled", reply)
        store = runtime._workflow_store()
        row = store.list_workflows()[0]
        self.assertEqual(row["state"], "cancelled")

    def test_status_reports_a_failed_workflow(self):
        runtime = self._fake_runtime()
        store = runtime._workflow_store()
        row = store.save_workflow(
            name="bad-one", task="t", script=self._SCRIPT)
        store.mark_running(row["id"])
        store.mark_failed(row["id"], error="ValueError: kaboom")
        status = runtime._control_workflow("status bad-one")
        self.assertIn("[failed]", status)
        self.assertIn("kaboom", status)

    def test_invalid_draft_saves_nothing(self):
        runtime = self._fake_runtime(script_text="import os\n")
        reply = runtime._control_workflow("do something shady")
        self.assertIn("couldn't draft", reply)
        self.assertIn("import", reply)
        self.assertEqual(
            runtime._workflow_store().list_workflows(), [])

    def test_unknown_ref_is_honest(self):
        runtime = self._fake_runtime()
        self.assertIn("no workflow 'nope'",
                      runtime._control_workflow("run nope"))

    def test_verb_word_task_drafts_instead_of_resolving(self):
        runtime = self._fake_runtime()
        reply = runtime._control_workflow("show me the files")
        self.assertIn("saved as show-me-the-files", reply)
        self.assertEqual(
            len(runtime._workflow_store().list_workflows()), 1)

    def test_brain_failure_is_an_honest_reply(self):
        runtime = self._fake_runtime()
        def _boom(prompt):
            raise RuntimeError("brain down")
        runtime._workflow_llm = _boom
        reply = runtime._control_workflow("review the retry path")
        self.assertIn("couldn't draft", reply)
        self.assertIn("brain down", reply)
        self.assertEqual(
            runtime._workflow_store().list_workflows(), [])
