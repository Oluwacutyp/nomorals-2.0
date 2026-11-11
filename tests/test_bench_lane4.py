"""Multi-cursor bench, Lane 4 — contracts, spawning, wiring.

Pins: (a) contract-first decomposition — a contract publishes
generation 1 on promotion, bound tasks claim against it, a
challenge suspends them (claims refused naming the contract)
and reopens the contract task, the revision publishes
generation 2 and rebases every binding, and generation 1 stays
readable; (b) swarm spawning — the monitor enforces a disjoint,
contained partition before registering anything (overlap and
escape refusals name the collision), roles canonicalise through
the registry, children depend on the parent and claim only
after it promotes; (c) orchestrator bench mode — a plan becomes
registered bench tasks, a step without writes/acceptance is
skipped with its reason, and nothing executes; (d) the verifier
pool — a submitted task is assigned to a verifier that is not
its worker, and a pool whose only verifier IS the worker
assigns nothing; (e) the /runs surface — bench runs and tasks
project as tickets with verdicts and the open escalation
(arbiter: the owner) visible, internal ids never rendered;
(f) the composed runtime cycle verifies and promotes end to
end, and a bench run registers as a mission.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.bench import BenchMonitor, TaskSpec
from nomorals.agents.bench.model import BenchError
from nomorals.agents.bench.pool import VerifierPool
from nomorals.agents.bench.wiring import build_bench
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.agents.run_ticket import ticket_for
from nomorals.agents.ticket_board import board_rows, render_board, render_ticket
from nomorals.missions.mission import MissionStore
from nomorals.storage.db import Database

T0 = 1_700_000_000.0

_IDENTITY = ("-c", "user.name=Bench Tests",
             "-c", "user.email=bench-tests@nomorals.invalid",
             "-c", "commit.gpgsign=false")

IFACE_V1 = 'def greet(name):\n    ...\n'
IFACE_V2 = 'def greet(name):\n    ...\n\n\ndef farewell(name):\n    ...\n'


def _git(cwd: Path, *args: str):
    return subprocess.run(
        ["git", *_IDENTITY, *args], cwd=str(cwd),
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120)


def _spec(write_set, *, role="coder", instructions="do the thing",
          acceptance=("tests pass",), deps=()):
    return TaskSpec(instructions=instructions,
                    acceptance=tuple(acceptance), role=role,
                    write_set=tuple(write_set), deps=tuple(deps))


class _Lane4Case(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.repo = self.home / "repo"
        self.repo.mkdir()
        self.assertEqual(_git(self.repo, "init", "-q").returncode, 0)
        files = {
            "iface.py": 'def placeholder():\n    ...\n',
            "impl.py": 'def run():\n    return "v0"\n',
            "calc.py": (
                "def add(a, b):\n    return a + b\n\n\n"
                "def mul(a, b):\n    return a - b\n"),
            "nomorals/storage/migrations.py": (
                "# the frozen file, fixture shape\nVALUE = 1\n"),
            "tests/test_impl.py": (
                "import impl\n\n\n"
                "def test_run():\n    assert impl.run() == 'v0'\n"),
            "tests/test_add.py": (
                "import calc\n\n\n"
                "def test_add():\n    assert calc.add(2, 3) == 5\n"),
            "tests/test_mul.py": (
                "import calc\n\n\n"
                "def test_mul():\n    assert calc.mul(2, 3) == 6\n"),
            "tests/test_layering.py": (
                "from pathlib import Path\n\n\n"
                "def test_no_forbidden_module():\n"
                "    root = Path(__file__).parent.parent\n"
                "    assert not (root / 'forbidden.py').exists()\n"),
        }
        for rel, content in files.items():
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.assertEqual(_git(self.repo, "add", "-A").returncode, 0)
        self.assertEqual(
            _git(self.repo, "commit", "-q", "-m", "base").returncode, 0)
        self.base = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        self.db = Database(str(self.home / "bench.db"))
        self.db.migrate()
        self.runtime = build_bench(self.db, self.repo,
                                   self.home / "benchroot")
        self.monitor = self.runtime.monitor
        self.contracts = self.runtime.contracts

    # -- helpers ----------------------------------------------------

    def open_run(self, goal="lane 4 goal"):
        return self.runtime.open_run(goal, self.base, now=T0)

    def register(self, run_id, key, write_set, **kw):
        return self.monitor.register_task(
            run_id, key, _spec(write_set, **kw), now=T0)

    def claim(self, run_id, key, cursor, *, now=T0):
        got = self.monitor.claim(run_id, key, cursor, now=now)
        self.assertTrue(got["claimed"], got)
        return got

    def edit(self, run_id, key, cursor, relpath, content):
        path = self.runtime.views.own_staging_path(
            run_id, key, cursor) / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def submit(self, run_id, key, cursor, generation, *, now):
        got = self.monitor.submit(run_id, key, cursor, generation,
                                  now=now)
        self.assertTrue(got["submitted"], got)
        return got

    def task(self, run_id, key):
        return self.monitor.store.get_task(run_id, key)

    def events(self, run_id, kind, key=None):
        task_id = self.task(run_id, key)["id"] if key else ""
        return self.monitor.store.events(run_id, task_id, kind)

    def drive_contract(self, run_id, content, *, now):
        """Claim the contract task, write the artifact, submit,
        verify through the pool, promote through the queue."""
        got = self.claim(run_id, "api", "cursor-arch", now=now)
        self.edit(run_id, "api", "cursor-arch", "iface.py", content)
        self.submit(run_id, "api", "cursor-arch", got["generation"],
                    now=now + 1)
        outcome = self.runtime.pool.assign_next(run_id, now=now + 2)
        self.assertTrue(outcome["assigned"], outcome)
        self.assertEqual(outcome["verdict"], "supported", outcome)
        promoted = self.runtime.integrator.promote_next(
            run_id, now=now + 3)
        self.assertTrue(promoted["promoted"], promoted)


class ContractTests(_Lane4Case):

    def _setup_contract_and_impls(self):
        run = self.open_run("contract goal")
        run_id = run["id"]
        self.contracts.define(run_id, "api", "iface.py", now=T0)
        self.register(run_id, "impl", ("impl.py", "tests/test_impl.py"),
                      deps=("api",),
                      acceptance=("test:tests/test_impl.py",))
        self.register(run_id, "impl2", ("impl2.py",), deps=("api",),
                      acceptance=("test:tests/test_impl.py",))
        self.contracts.bind(run_id, "impl", "api", now=T0)
        self.contracts.bind(run_id, "impl2", "api", now=T0)
        return run_id

    def test_contract_lifecycle_challenge_and_rebase(self):
        run_id = self._setup_contract_and_impls()
        # Bound at generation 0 while the contract is a draft,
        # and unclaimable: the contract is not promoted yet.
        binding = self.contracts.binding_for(
            self.task(run_id, "impl")["id"])
        self.assertEqual(binding["generation"], 0)
        refused = self.monitor.claim(run_id, "impl", "cursor-impl",
                                     now=T0)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "deps_unpromoted")
        with self.assertRaises(BenchError):
            self.contracts.challenge(run_id, "api",
                                     challenger="cursor-impl",
                                     now=T0)
        # Generation 1 publishes on promotion.
        self.drive_contract(run_id, IFACE_V1, now=T0 + 10)
        published = self.contracts.sync_promotions(run_id, now=T0 + 20)
        self.assertEqual(
            [(p["contract"], p["generation"]) for p in published],
            [("api", 1)])
        contract = self.contracts.get(run_id, "api")
        self.assertEqual(contract["state"], "current")
        self.assertEqual(contract["generation"], 1)
        # Bindings rebased onto generation 1, journaled.
        for key in ("impl", "impl2"):
            binding = self.contracts.binding_for(
                self.task(run_id, key)["id"])
            self.assertEqual(binding["generation"], 1)
            rebased = self.events(run_id, "task_rebased", key)
            self.assertEqual(len(rebased), 1)
            self.assertEqual(rebased[0]["detail"]["to_generation"], 1)
        # The frozen generation reads back byte for byte.
        current = self.contracts.current(run_id, "api")
        self.assertEqual(current["content"].decode(), IFACE_V1)
        self.assertEqual(
            self.contracts.at_generation(run_id, "api", 1)["content"]
            .decode(), IFACE_V1)
        # Bound tasks claim against generation 1.
        self.claim(run_id, "impl", "cursor-impl", now=T0 + 30)
        ready = [t["key"] for t in self.monitor.ready_tasks(run_id)]
        self.assertIn("impl2", ready)
        # The challenge: bound tasks suspend, the contract
        # reopens, and a claim names the contract.
        challenged = self.contracts.challenge(
            run_id, "api", challenger="cursor-impl",
            reason="the interface has no farewell()", now=T0 + 40)
        self.assertEqual(challenged["state"], "challenged")
        self.assertEqual(
            sorted(challenged["suspended"]), ["impl", "impl2"])
        self.assertEqual(self.task(run_id, "api")["state"], "requeued")
        self.assertTrue(self.events(run_id, "task_suspended", "impl2"))
        refused = self.monitor.claim(run_id, "impl2", "cursor-2",
                                     now=T0 + 41)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "contract_challenged")
        self.assertEqual(refused["contract"], "api")
        ready = [t["key"] for t in self.monitor.ready_tasks(run_id)]
        self.assertNotIn("impl2", ready)
        # The revision publishes generation 2 and rebases.
        self.drive_contract(run_id, IFACE_V2, now=T0 + 50)
        published = self.contracts.sync_promotions(run_id, now=T0 + 60)
        self.assertEqual(
            [(p["contract"], p["generation"]) for p in published],
            [("api", 2)])
        self.assertTrue(self.events(run_id, "contract_revised", "api"))
        binding = self.contracts.binding_for(
            self.task(run_id, "impl2")["id"])
        self.assertEqual(binding["generation"], 2)
        rebased = self.events(run_id, "task_rebased", "impl2")
        self.assertEqual(rebased[-1]["detail"]["to_generation"], 2)
        self.assertTrue(rebased[-1]["detail"]["suspension_lifted"])
        # The suspension is lifted; generation 1 stays readable.
        self.claim(run_id, "impl2", "cursor-2", now=T0 + 70)
        self.assertEqual(
            self.contracts.at_generation(run_id, "api", 1)["content"]
            .decode(), IFACE_V1)
        self.assertEqual(
            self.contracts.current(run_id, "api")["content"].decode(),
            IFACE_V2)
        # A sync over the same journal publishes nothing twice.
        self.assertEqual(
            self.contracts.sync_promotions(run_id, now=T0 + 80), [])

    def test_bind_and_define_refusals(self):
        run = self.open_run()
        run_id = run["id"]
        self.contracts.define(run_id, "api", "iface.py", now=T0)
        with self.assertRaises(BenchError):
            self.contracts.define(run_id, "api", "iface.py", now=T0)
        with self.assertRaises(BenchError):
            self.contracts.bind(run_id, "ghost", "api", now=T0)
        self.register(run_id, "impl", ("impl.py",))
        with self.assertRaises(BenchError):
            self.contracts.bind(run_id, "impl", "nope", now=T0)
        with self.assertRaises(BenchError):
            self.contracts.current(run_id, "api")


class SpawnTests(_Lane4Case):

    def test_spawn_partition_enforced(self):
        run = self.open_run("spawn goal")
        run_id = run["id"]
        self.register(run_id, "parent", ("pkg/**",))
        got = self.claim(run_id, "parent", "spawner")
        gen = got["generation"]
        # A stale spawner spawns nothing.
        refused = self.monitor.spawn(
            run_id, "parent", "spawner", gen + 99,
            [("a", _spec(("pkg/a.py",)))], now=T0)
        self.assertFalse(refused["spawned"])
        self.assertEqual(refused["reason"], "stale_generation")
        # An escaping child is refused, named, unregistered.
        refused = self.monitor.spawn(
            run_id, "parent", "spawner", gen,
            [("e", _spec(("outside.py",)))], now=T0)
        self.assertFalse(refused["spawned"])
        self.assertEqual(refused["reason"], "write_set_escape")
        self.assertEqual(refused["entries"], ["outside.py"])
        self.assertIsNone(self.task(run_id, "e"))
        # Overlapping children are refused naming the collision.
        refused = self.monitor.spawn(
            run_id, "parent", "spawner", gen,
            [("c", _spec(("pkg/x.py",))),
             ("d", _spec(("pkg/x.py",)))], now=T0)
        self.assertFalse(refused["spawned"])
        self.assertEqual(refused["reason"], "partition_overlap")
        self.assertEqual(refused["children"], ["c", "d"])
        self.assertEqual(refused["files"], ["pkg/x.py"])
        self.assertIsNone(self.task(run_id, "c"))
        self.assertIsNone(self.task(run_id, "d"))
        # A role the registry does not know is refused.
        refused = self.monitor.spawn(
            run_id, "parent", "spawner", gen,
            [("f", _spec(("pkg/f.py",), role="wizard"))], now=T0)
        self.assertFalse(refused["spawned"])
        self.assertEqual(refused["reason"], "unknown_role")
        # The clean partition registers: roles canonicalise,
        # children depend on the parent, spawning is journaled.
        spawned = self.monitor.spawn(
            run_id, "parent", "spawner", gen,
            [("a", _spec(("pkg/a.py",), role="coding")),
             ("b", _spec(("pkg/b.py",), role="coder"))], now=T0)
        self.assertTrue(spawned["spawned"], spawned)
        self.assertEqual(self.task(run_id, "a")["role"], "coder")
        self.assertEqual(self.task(run_id, "a")["deps"], ("parent",))
        self.assertTrue(self.events(run_id, "task_spawned", "a"))
        # Children are unclaimable until the parent promotes.
        refused = self.monitor.claim(run_id, "a", "cursor-a", now=T0)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "deps_unpromoted")
        # The parent's (empty) candidate promotes; the child
        # claims at the next generation of its own lifecycle.
        self.submit(run_id, "parent", "spawner", gen, now=T0 + 1)
        outcome = self.runtime.pool.assign_next(run_id, now=T0 + 2)
        self.assertEqual(outcome["verdict"], "supported", outcome)
        promoted = self.runtime.integrator.promote_next(
            run_id, now=T0 + 3)
        self.assertTrue(promoted["promoted"], promoted)
        self.claim(run_id, "a", "cursor-a", now=T0 + 4)


class OrchestratorBenchModeTests(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(tmp.cleanup)
        self.db = Database(os.path.join(tmp.name, "bench.db"))
        self.db.migrate()
        self.monitor = BenchMonitor(self.db, lease_ttl_s=10.0)
        context = SimpleNamespace(executor=None, blackboard=None,
                                  router=None)
        self.orchestrator = MasterOrchestrator(context)

    def test_bench_plan_registers_and_executes_nothing(self):
        plan = Plan(goal="bench goal", steps=[
            PlanStep(name="one", goal="do one", role="coding",
                     payload={"writes": ["x.py"],
                              "acceptance": ["test:tests/test_x.py"]}),
            PlanStep(name="two", goal="do two", role="coding",
                     depends_on=["one"],
                     payload={"writes": ["y.py"],
                              "acceptance": ["cmd:true"]}),
            PlanStep(name="three", goal="do three", role="coding",
                     payload={"acceptance": ["cmd:true"]}),
            PlanStep(name="four", goal="do four", role="coding",
                     depends_on=["three"],
                     payload={"writes": ["z.py"],
                              "acceptance": ["cmd:true"]}),
        ])
        result = self.orchestrator.bench_plan(
            "bench goal", self.monitor, plan=plan,
            base_revision="rev-0", now=T0)
        self.assertEqual(sorted(result["tasks"]), ["one", "two"])
        skipped = {row["step"]: row["reason"]
                   for row in result["skipped"]}
        self.assertEqual(skipped["three"], "no_write_set")
        self.assertEqual(skipped["four"],
                         "dependency_not_registered")
        run_id = result["run_id"]
        one = self.monitor.store.get_task(run_id, "one")
        two = self.monitor.store.get_task(run_id, "two")
        self.assertEqual(one["state"], "pending")
        self.assertEqual(two["deps"], ("one",))
        self.assertEqual(two["write_set"], ("y.py",))
        # The run registered as a mission carrying the run id.
        self.assertTrue(result["mission_id"])
        mission = MissionStore(self.db).get(result["mission_id"])
        self.assertEqual(mission.state["bench_run_id"], run_id)
        self.assertIn("bench", mission.tags)
        # Planner-only: nothing executed, nothing settled.
        self.assertEqual(self.orchestrator.history, [])
        events = self.monitor.store.events(run_id, "",
                                           "plan_registered")
        self.assertEqual(len(events), 1)


class PoolTests(_Lane4Case):

    def _submitted_fix(self, run_id):
        self.register(run_id, "fix", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim(run_id, "fix", "cursor-a")
        self.edit(run_id, "fix", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit(run_id, "fix", "cursor-a", got["generation"],
                    now=T0 + 1)

    def test_pool_assigns_a_verifier_who_is_not_the_worker(self):
        run = self.open_run("pool goal")
        run_id = run["id"]
        self._submitted_fix(run_id)
        pool = VerifierPool(self.monitor, self.home / "pool",
                            verifier_ids=("cursor-a",
                                          "bench-verifier-9"),
                            test_timeout_s=180.0)
        outcome = pool.assign_next(run_id, now=T0 + 2)
        self.assertTrue(outcome["assigned"], outcome)
        self.assertEqual(outcome["verifier"], "bench-verifier-9")
        self.assertEqual(outcome["verdict"], "supported", outcome)
        rows = self.monitor.store.verdicts(
            self.task(run_id, "fix")["id"])
        self.assertEqual(rows[-1]["verifier"], "bench-verifier-9")

    def test_pool_with_only_the_worker_assigns_nothing(self):
        run = self.open_run("pool goal 2")
        run_id = run["id"]
        self._submitted_fix(run_id)
        pool = VerifierPool(self.monitor, self.home / "pool2",
                            verifier_ids=("cursor-a",),
                            test_timeout_s=180.0)
        outcome = pool.assign_next(run_id, now=T0 + 2)
        self.assertFalse(outcome["assigned"])
        self.assertEqual(outcome["reason"],
                         "no_independent_verifier")
        self.assertEqual(
            self.monitor.store.verdicts(
                self.task(run_id, "fix")["id"]), [])
        self.assertEqual(self.task(run_id, "fix")["state"],
                         "submitted")


class SurfaceTests(_Lane4Case):

    def test_runs_surface_shows_bench_state_and_escalation(self):
        run = self.open_run("escalation goal")
        run_id = run["id"]
        self.register(run_id, "bad", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim(run_id, "bad", "cursor-bad")
        # A candidate that does NOT fix the bug: contradicted,
        # and with an attempt budget of 1 it escalates at once.
        self.edit(run_id, "bad", "cursor-bad", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a - b  # still wrong\n")
        self.submit(run_id, "bad", "cursor-bad", got["generation"],
                    now=T0 + 1)
        pool = VerifierPool(self.monitor, self.home / "poolx",
                            verifier_ids=("bench-verifier-9",),
                            attempt_budget=1, test_timeout_s=180.0)
        outcome = pool.assign_next(run_id, now=T0 + 2)
        self.assertEqual(outcome["verdict"], "contradicted", outcome)
        self.assertTrue(outcome.get("escalated"), outcome)
        task_id = self.task(run_id, "bad")["id"]
        # The run ticket carries tasks, verdicts, escalation.
        ticket = ticket_for(f"bench:{run_id}", self.db, now=T0 + 3)
        self.assertTrue(ticket["found"])
        self.assertEqual(ticket["kind"], "bench_run")
        self.assertEqual(
            ticket["evidence"]["verdicts_by_task"]["bad"],
            "contradicted")
        escalations = ticket["detail"]["escalations"]
        self.assertEqual(len(escalations), 1)
        self.assertEqual(escalations[0]["task"], "bad")
        self.assertEqual(escalations[0]["arbiter"], "owner")
        # The task ticket carries the verdict and the escalation.
        task_ticket = ticket_for(f"bench-task:{task_id}", self.db,
                                 now=T0 + 3)
        self.assertTrue(task_ticket["found"])
        self.assertEqual(
            task_ticket["evidence"]["latest_verdict"],
            "contradicted")
        self.assertEqual(
            len(task_ticket["detail"]["escalations"]), 1)
        # An unpromoted candidate ref is invisible on the ticket.
        self.assertNotIn("candidate_ref",
                         task_ticket["fencing"])
        # The board composes bench rows; the rendered ticket
        # never prints an internal id.
        board = board_rows(self.db)
        kinds = [row["kind"] for row in board["rows"]]
        self.assertIn("bench_run", kinds)
        self.assertIn("bench_task", kinds)
        self.assertIn("escalation goal", render_board(board))
        run_row = next(row for row in board["rows"]
                       if row["kind"] == "bench_run")
        text = render_ticket(run_row)
        self.assertIn("arbiter", text)
        self.assertNotIn(run_id, text)
        self.assertNotIn(task_id, text)
        # Unknown bench refs answer typed not-found.
        missing = ticket_for("bench:br-nope", self.db, now=T0 + 3)
        self.assertFalse(missing["found"])

    def test_runtime_cycle_verifies_and_promotes(self):
        run = self.open_run("cycle goal")
        run_id = run["id"]
        self.register(run_id, "fix", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim(run_id, "fix", "cursor-a")
        self.edit(run_id, "fix", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit(run_id, "fix", "cursor-a", got["generation"],
                    now=T0 + 1)
        result = self.runtime.cycle(now=T0 + 2)
        self.assertEqual(self.task(run_id, "fix")["state"],
                         "promoted")
        self.assertTrue(result["promoted"], result)
        # The mission registration exists for the run.
        from nomorals.agents.bench.wiring import mission_for_run
        self.assertEqual(mission_for_run(self.db, run_id),
                         run["mission_id"])


if __name__ == "__main__":
    unittest.main()
