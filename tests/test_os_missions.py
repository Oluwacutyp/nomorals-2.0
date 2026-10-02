"""L6 — os mission state machine: transitions, acceptance, repair loop, hooks.

Unit tier: fully offline. Real MissionStore / ArtifactStore on an
in-memory database; the MissionRunner is driven with a stub context and a
stubbed plan so no LLM or agent is ever constructed.
"""

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.errors import NotFound
from nomorals.core.events import Event, global_bus
from nomorals.core.tasks import AcceptanceCriterion
from nomorals.missions import MissionRunner, MissionStatus, MissionStore
from nomorals.os.mission_state import (
    CANCELLED,
    COMPLETED,
    CREATED,
    FAILED,
    PAUSED,
    PLANNED,
    RUNNING,
    VERIFYING,
    InvalidTransition,
    MissionAcceptance,
    attach_runner,
    current_state,
    evaluate_acceptance,
    transition,
    verify_repair_loop,
)
from nomorals.os.verifiers import (
    CodeTestsVerifier,
    DocsRenderVerifier,
    Verdict,
    VerifierRegistry,
    _import_docs_check,
    default_registry,
)
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database


def make_db():
    db = Database(":memory:")
    db.migrate()
    return db


def make_artifact_store(test):
    tmp = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = make_db()
    test.addCleanup(db.close)
    return ArtifactStore(db, BlobStore(db, Path(tmp) / "blobs"))


def make_runner(db, **kwargs):
    ctx = SimpleNamespace(db=db, memory=None)
    return MissionRunner(ctx, milestones=False, **kwargs)


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)

    def test_happy_path_planned_running_completed(self):
        m = self.store.create_new("ship it")
        self.assertEqual(current_state(self.store.get(m.id)), CREATED)
        transition(self.store, m.id, PLANNED, note="planned")
        transition(self.store, m.id, RUNNING)
        out = transition(self.store, m.id, COMPLETED)
        self.assertEqual(out.status, MissionStatus.DONE)
        self.assertEqual(current_state(out), COMPLETED)

    def test_status_mapping_is_synced(self):
        m = self.store.create_new("m")
        transition(self.store, m.id, PLANNED)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.PENDING)
        transition(self.store, m.id, RUNNING)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.RUNNING)
        transition(self.store, m.id, PAUSED)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.PAUSED)
        transition(self.store, m.id, RUNNING)
        transition(self.store, m.id, FAILED)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.FAILED)

    def test_illegal_moves_raise(self):
        m = self.store.create_new("m")
        with self.assertRaises(InvalidTransition):
            transition(self.store, m.id, RUNNING)  # CREATED -> RUNNING
        with self.assertRaises(InvalidTransition):
            transition(self.store, m.id, COMPLETED)  # CREATED -> COMPLETED
        with self.assertRaises(InvalidTransition):
            transition(self.store, m.id, "BOGUS")
        transition(self.store, m.id, PLANNED)
        with self.assertRaises(InvalidTransition):
            transition(self.store, m.id, COMPLETED)  # PLANNED -> COMPLETED

    def test_terminal_states_have_no_outgoing(self):
        for terminal in (COMPLETED, FAILED, CANCELLED):
            m = self.store.create_new("m")
            transition(self.store, m.id, CANCELLED if terminal == CANCELLED
                       else PLANNED)
            if terminal != CANCELLED:
                transition(self.store, m.id, RUNNING)
                transition(self.store, m.id, terminal)
            with self.assertRaises(InvalidTransition):
                transition(self.store, m.id, RUNNING)
            with self.assertRaises(InvalidTransition):
                transition(self.store, m.id, PLANNED)

    def test_verifying_path(self):
        m = self.store.create_new("m")
        transition(self.store, m.id, PLANNED)
        transition(self.store, m.id, RUNNING)
        transition(self.store, m.id, VERIFYING)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.RUNNING)
        transition(self.store, m.id, COMPLETED)
        self.assertEqual(current_state(self.store.get(m.id)), COMPLETED)

    def test_same_state_is_idempotent_noop(self):
        m = self.store.create_new("m")
        transition(self.store, m.id, PLANNED, note="first")
        out = transition(self.store, m.id, PLANNED, note="second")
        log = out.state["transition_log"]
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["note"], "first")

    def test_transition_log_appends(self):
        m = self.store.create_new("m")
        transition(self.store, m.id, PLANNED, note="n1")
        transition(self.store, m.id, RUNNING, note="n2")
        log = self.store.get(m.id).state["transition_log"]
        self.assertEqual([(e["from"], e["to"], e["note"]) for e in log],
                         [(CREATED, PLANNED, "n1"), (PLANNED, RUNNING, "n2")])
        self.assertTrue(all(e["ts"] > 0 for e in log))

    def test_unknown_mission_raises_not_found(self):
        with self.assertRaises(NotFound):
            transition(self.store, "nope", PLANNED)

    def test_mission_transition_event_emitted(self):
        seen: list[Event] = []
        sub = global_bus.subscribe("mission.transition", seen.append, sync=True)
        try:
            m = self.store.create_new("m")
            transition(self.store, m.id, PLANNED, note="hello")
        finally:
            global_bus.unsubscribe(sub)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].data,
                         {"mission_id": m.id, "from_state": CREATED,
                          "to_state": PLANNED, "note": "hello"})
        # idempotent no-ops do not emit
        sub2 = global_bus.subscribe("mission.transition", seen.append, sync=True)
        try:
            transition(self.store, m.id, PLANNED)
        finally:
            global_bus.unsubscribe(sub2)
        self.assertEqual(len(seen), 1)


class AttachRunnerTests(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)

    def test_attach_is_duck_typed(self):
        fake_runner = SimpleNamespace()
        hook = attach_runner(fake_runner, self.store)
        self.assertIs(fake_runner._os_transition_hook, hook)
        m = self.store.create_new("m")
        hook(m.id, PLANNED, "via hook")
        self.assertEqual(current_state(self.store.get(m.id)), PLANNED)

    def test_hook_propagates_invalid_transition(self):
        fake_runner = SimpleNamespace()
        hook = attach_runner(fake_runner, self.store)
        m = self.store.create_new("m")
        with self.assertRaises(InvalidTransition):
            hook(m.id, RUNNING)  # CREATED -> RUNNING is illegal

    def test_runner_finish_fires_hook_without_breaking(self):
        m = self.store.create_new("m")
        runner = make_runner(self.db, store=self.store)
        attach_runner(runner, self.store)
        # drive the sequence the runner itself produces: PLANNED (start),
        # RUNNING (run), then _finish maps DONE -> COMPLETED.
        runner._os_transition(m.id, PLANNED, "created")
        runner._os_transition(m.id, RUNNING, "run started")
        result = runner._finish(self.store.get(m.id), MissionStatus.DONE, [],
                                time.monotonic(), "", reflect=False)
        self.assertTrue(result.ok)
        done = self.store.get(m.id)
        self.assertEqual(done.status, MissionStatus.DONE)
        self.assertEqual(current_state(done), COMPLETED)
        self.assertEqual(
            [(e["from"], e["to"]) for e in done.state["transition_log"]],
            [(CREATED, PLANNED), (PLANNED, RUNNING), (RUNNING, COMPLETED)])

    def test_runner_finish_failed_maps(self):
        m = self.store.create_new("m")
        runner = make_runner(self.db, store=self.store)
        attach_runner(runner, self.store)
        runner._os_transition(m.id, PLANNED)
        runner._os_transition(m.id, RUNNING)
        result = runner._finish(self.store.get(m.id), MissionStatus.FAILED,
                                [], time.monotonic(), "", reflect=False,
                                error="boom")
        self.assertFalse(result.ok)
        self.assertEqual(current_state(self.store.get(m.id)), FAILED)

    def test_run_without_hook_still_works(self):
        # no attach_runner: the hooks are no-ops, the run is unaffected
        m = self.store.create_new("m")
        runner = make_runner(self.db, store=self.store)
        runner._plan = lambda mission: []  # stub: no agent ever built
        result = runner.run(self.store.get(m.id), reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.DONE)
        self.assertNotIn("os_state", self.store.get(m.id).state)

    def test_full_run_with_hooks(self):
        m = self.store.create_new("m")
        runner = make_runner(self.db, store=self.store)
        attach_runner(runner, self.store)
        runner._plan = lambda mission: []
        runner._os_transition(m.id, PLANNED, "created")  # what start() does
        result = runner.run(self.store.get(m.id), reflect=False)
        self.assertTrue(result.ok)
        done = self.store.get(m.id)
        self.assertEqual(current_state(done), COMPLETED)
        self.assertEqual(
            [(e["from"], e["to"]) for e in done.state["transition_log"]],
            [(CREATED, PLANNED), (PLANNED, RUNNING), (RUNNING, COMPLETED)])

    def test_illegal_runner_sequence_does_not_break_run(self):
        # attach, but skip PLANNED: RUNNING from CREATED is illegal, the
        # runner swallows it and the mission still completes.
        m = self.store.create_new("m")
        runner = make_runner(self.db, store=self.store)
        attach_runner(runner, self.store)
        runner._plan = lambda mission: []
        result = runner.run(self.store.get(m.id), reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.DONE)

    def test_resource_advisor_is_advisory(self):
        m = self.store.create_new("m")
        runner = make_runner(self.db, store=self.store)
        calls = []
        runner._resource_advisor = lambda mission: calls.append(mission) or {
            "ok": True, "throttled": True, "reasons": ["hot"]}
        mission = self.store.get(m.id)
        runner._advise_resources(mission)  # the _execute_step call site
        self.assertEqual(calls, [mission])

    def test_raising_advisor_is_swallowed(self):
        m = self.store.create_new("m")
        runner = make_runner(self.db, store=self.store)

        def bad(mission):
            raise RuntimeError("advisor exploded")

        runner._resource_advisor = bad
        runner._advise_resources(self.store.get(m.id))  # must not raise


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.arts = make_artifact_store(self)
        self.db = make_db()
        self.addCleanup(self.db.close)
        self.store = MissionStore(self.db)

    def _mission_with(self, **state):
        m = self.store.create_new("m")
        m.state.update(state)
        return self.store.save(m)

    def test_criteria_pass(self):
        m = self._mission_with(metrics={"quality": 0.9})
        acc = MissionAcceptance(criteria=[
            AcceptanceCriterion(name="quality",
                                spec={"metric": "quality", "gte": 0.8})])
        passed, details = evaluate_acceptance(m, self.arts, acc)
        self.assertTrue(passed)
        self.assertEqual(details["criteria"][0]["name"], "quality")
        self.assertTrue(details["criteria"][0]["passed"])

    def test_criteria_fail(self):
        m = self._mission_with(metrics={"quality": 0.5})
        acc = MissionAcceptance(criteria=[
            AcceptanceCriterion(name="quality",
                                spec={"metric": "quality", "gte": 0.8})])
        passed, details = evaluate_acceptance(m, self.arts, acc)
        self.assertFalse(passed)
        self.assertEqual(details["required_criteria_failed"], ["quality"])

    def test_required_artifact_types(self):
        m = self.store.create_new("m")
        self.arts.put_text("report body", type="report", mission_id=m.id)
        acc = MissionAcceptance(required_artifact_types=["report"])
        passed, details = evaluate_acceptance(m, self.arts, acc)
        self.assertTrue(passed)
        self.assertEqual(details["artifact_types"]["missing"], [])

        acc2 = MissionAcceptance(required_artifact_types=["report", "video"])
        passed2, details2 = evaluate_acceptance(m, self.arts, acc2)
        self.assertFalse(passed2)
        self.assertEqual(details2["artifact_types"]["missing"], ["video"])
        self.assertEqual(details2["artifact_types"]["present"], ["report"])

    def test_empty_acceptance_passes(self):
        m = self.store.create_new("m")
        passed, _ = evaluate_acceptance(m, self.arts, MissionAcceptance())
        self.assertTrue(passed)


class StubVerifier:
    name = "stub"

    def __init__(self, failures_before_pass=1):
        self.calls = 0
        self.failures_before_pass = failures_before_pass

    def verify(self, target):
        self.calls += 1
        if self.calls <= self.failures_before_pass:
            return Verdict(passed=False, details=f"attempt {self.calls} broken")
        return Verdict(passed=True, details="fixed")


class VerifyRepairLoopTests(unittest.TestCase):
    def test_repairs_then_passes(self):
        verifier = StubVerifier(failures_before_pass=1)

        def planner():
            return {"v": 1}

        def executor(plan):
            return {"plan_v": plan["v"]}

        def repairer(observation, verdict):
            return {"v": observation["plan_v"] + 1}

        report = verify_repair_loop(
            verifier=verifier, planner=planner, executor=executor,
            repairer=repairer, max_repair_rounds=3)
        self.assertTrue(report["passed"])
        self.assertEqual(report["rounds"], 2)
        self.assertEqual(report["repairs"], 1)
        self.assertEqual(report["plan"], {"v": 2})
        self.assertEqual(len(report["verdicts"]), 2)
        self.assertFalse(report["verdicts"][0]["passed"])
        self.assertTrue(report["verdicts"][1]["passed"])

    def test_gives_up_after_max_rounds(self):
        verifier = StubVerifier(failures_before_pass=99)
        report = verify_repair_loop(
            verifier=verifier,
            planner=lambda: {"v": 0},
            executor=lambda plan: {"ok": False},
            repairer=lambda obs, v: {"v": 1},
            max_repair_rounds=2)
        self.assertFalse(report["passed"])
        self.assertEqual(report["rounds"], 3)  # 1 initial + 2 repairs
        self.assertEqual(report["repairs"], 2)

    def test_passes_first_try_no_repair(self):
        verifier = StubVerifier(failures_before_pass=0)
        repairs = []
        report = verify_repair_loop(
            verifier=verifier,
            planner=lambda: {"v": 1},
            executor=lambda plan: {"done": True},
            repairer=lambda obs, v: repairs.append(1) or {},
            max_repair_rounds=3)
        self.assertTrue(report["passed"])
        self.assertEqual(report["rounds"], 1)
        self.assertEqual(repairs, [])


REPO_ROOT = Path(__file__).resolve().parent.parent


class VerifierRegistryTests(unittest.TestCase):
    def test_register_get_list(self):
        reg = VerifierRegistry()
        v = CodeTestsVerifier()
        reg.register(v)
        self.assertIs(reg.get("code_tests"), v)
        self.assertEqual(reg.list(), ["code_tests"])
        self.assertIn("code_tests", reg)

    def test_verify_unknown_raises(self):
        with self.assertRaises(KeyError):
            VerifierRegistry().verify("nope", {})

    def test_default_registry_ships_both(self):
        reg = default_registry()
        self.assertEqual(reg.list(), ["code_tests", "docs_render"])


class CodeTestsVerifierTests(unittest.TestCase):
    def test_passing_suite(self):
        v = CodeTestsVerifier(timeout=120.0)
        verdict = v.verify({
            "test_ids": ["tests.test_os_artifacts.ArtifactGraphTests"
                         ".test_rebuild_unknown_id_raises"],
            "cwd": str(REPO_ROOT),
        })
        self.assertTrue(verdict.passed, verdict.details)
        self.assertIn("Ran 1 test", verdict.details)
        self.assertIn("OK", verdict.details)

    def test_failing_suite(self):
        v = CodeTestsVerifier(timeout=120.0)
        verdict = v.verify({
            "test_ids": ["tests.test_os_artifacts.NoSuchClass.test_nope"],
            "cwd": str(REPO_ROOT),
        })
        self.assertFalse(verdict.passed)

    def test_no_test_ids_fails_closed(self):
        verdict = CodeTestsVerifier().verify({})
        self.assertFalse(verdict.passed)
        self.assertIn("no test_ids", verdict.details)


class DocsRenderVerifierTests(unittest.TestCase):
    def _make_root(self, readme: str, n_test_defs: int = 10,
                   n_lines: int = 100, bad_md: bool = False) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "README.md").write_text(readme, encoding="utf-8")
        pkg = root / "nomorals"
        pkg.mkdir()
        (pkg / "mod.py").write_text(
            "\n".join(f"x{i} = {i}" for i in range(n_lines)),
            encoding="utf-8")
        tests = root / "tests"
        tests.mkdir()
        # NOTE: no tests/__init__.py here on purpose — the synthetic root
        # exercises the file-location fallback. (With a package import, the
        # already-imported real `tests` package would shadow it via
        # sys.modules and measure the wrong tree.) The synthetic module
        # mirrors the real one's measurement surface.
        (tests / "test_docs_consistency.py").write_text(
            "import re\n"
            "from pathlib import Path\n"
            "ROOT = Path(__file__).resolve().parent.parent\n"
            "PKG = ROOT / 'nomorals'\n"
            "TESTS_DIR = ROOT / 'tests'\n"
            "_TEST_DEF = re.compile(r'^\\s*def\\s+test_', re.MULTILINE)\n"
            "def _measure_modules():\n"
            "    return [p for p in PKG.rglob('*.py') if p.is_file()]\n"
            "def _measure_lines(modules):\n"
            "    return sum(len(p.read_text(encoding='utf-8',"
            " errors='ignore').splitlines()) for p in modules)\n"
            "def _measure_tests():\n"
            "    total = 0\n"
            "    for p in sorted(TESTS_DIR.glob('test_*.py')):\n"
            "        total += len(_TEST_DEF.findall("
            "p.read_text(encoding='utf-8', errors='ignore')))\n"
            "    return total\n"
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            + "\n".join(f"    def test_{i}(self): pass"
                        for i in range(n_test_defs))
            + "\n",
            encoding="utf-8")
        (root / "notes.md").write_text("# notes\n", encoding="utf-8")
        if bad_md:
            (root / "broken.md").write_bytes(b"\xff\xfe\x00not utf-8")
        return root

    def test_clean_tree_passes(self):
        root = self._make_root(
            "# P\n~10 tests\n100 lines of Python\n1+ modules\n")
        verdict = DocsRenderVerifier().verify({"root": str(root)})
        self.assertTrue(verdict.passed, verdict.details)

    def test_stale_claim_fails(self):
        root = self._make_root(
            "# P\n~10 tests\n100000 lines of Python\n1+ modules\n")
        verdict = DocsRenderVerifier().verify({"root": str(root)})
        self.assertFalse(verdict.passed)
        self.assertIn("line count", verdict.details)

    def test_unreadable_markdown_fails(self):
        root = self._make_root(
            "# P\n~10 tests\n100 lines of Python\n1+ modules\n", bad_md=True)
        verdict = DocsRenderVerifier().verify({"root": str(root)})
        self.assertFalse(verdict.passed)
        self.assertIn("broken.md", verdict.details)

    def test_missing_readme_fails(self):
        root = self._make_root("# P\n~10 tests\n100 lines of Python\n1+ modules\n")
        (root / "README.md").unlink()
        verdict = DocsRenderVerifier().verify({"root": str(root)})
        self.assertFalse(verdict.passed)
        self.assertIn("README.md missing", verdict.details)

    def test_package_import_branch_reuses_real_module(self):
        # When tests/ IS a package, the canonical test module is reused.
        mod = _import_docs_check(REPO_ROOT)
        self.assertIsNotNone(mod)
        modules = mod._measure_modules()
        self.assertGreater(len(modules), 100)
        self.assertGreater(mod._measure_lines(modules), 10000)
        self.assertGreater(mod._measure_tests(), 100)

    def test_real_repo_verdict(self):
        # The real README is honest (H1) — this must pass, but tolerate
        # drift the same 15% the guard test uses.
        verdict = DocsRenderVerifier().verify({"root": str(REPO_ROOT)})
        self.assertTrue(verdict.passed, verdict.details)


if __name__ == "__main__":
    unittest.main()
