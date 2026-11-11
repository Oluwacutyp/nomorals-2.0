"""Multi-cursor bench, Lane 5 — the acceptance run (dossier §5).

Seven proofs, each driving the REAL bench organs end to end
(monitor, staging, verifier/pool, integrator, contracts) with
scripted cursor runs — no mocks of the bench itself:

1. a format/parser candidate whose own tests pass only because
   its fixture encoder shares the decoder's misreading is
   contradicted under the real-artifact rule and never lands;
2. two cursors with overlapping write-sets — the second claim
   is refused naming the holder and the colliding files, and no
   partial lease survives;
3. a peer cursor cannot see another's unpromoted candidate —
   neither through the read view nor through staging access;
4. a claimant that dies mid-work (pid death, lease lapse) has
   its task requeued and its staging quarantined, its late
   submit is fenced ``stale_generation``, and a successor
   completes the task;
5. a candidate that conflicts at promotion (an indirect file in
   neither write-set) lands ``conflicted`` with journaled
   evidence, the tree untouched, and the queue promotes later
   tasks;
6. an integrator crash between the git commit and the store
   transaction retries to exactly one promotion;
7. the tightly-coupled job: two cursors build opposite sides of
   one interface under a promoted contract; mid-run the
   contract is challenged and revised to a new generation;
   dependents suspend, rebase, and both sides land consistent
   with the NEW generation.

Proof 8 — verdict stability: the Lane 4 pool flow returns
``supported`` repeatedly inside a combined process (the mock
suites run in-process first), pinning the stale-bytecode fix
in the verifier's two-revision scratch (see
``test_bench_verifier.StaleBytecodeTests`` for the
deterministic mechanism pin).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

from nomorals.agents.bench import BenchIntegrator, TaskSpec
from nomorals.agents.bench.model import (
    TASK_CONFLICTED,
    TASK_PROMOTED,
    TASK_REQUEUED,
    TASK_VERIFYING,
    BenchError,
)
from nomorals.agents.bench.pool import VerifierPool
from nomorals.agents.bench.wiring import build_bench
from nomorals.storage.db import Database
from nomorals.tools.pytest_runner import run_tests

T0 = 1_700_000_000.0

_IDENTITY = ("-c", "user.name=Bench Tests",
             "-c", "user.email=bench-tests@nomorals.invalid",
             "-c", "commit.gpgsign=false")

#: The stored real artifact for Proof 1: four (x, y) points in
#: row-major order. A correct decoder returns them as written;
#: the planted decoder transposes every pair.
REAL_TILE = bytes((1, 2, 3, 4, 5, 6, 7, 8))
REAL_POINTS = [(1, 2), (3, 4), (5, 6), (7, 8)]

FIXED_CALC = ("def add(a, b):\n    return a + b\n\n\n"
              "def mul(a, b):\n    return a * b\n")

IFACE_V1 = 'def greet(name):\n    return "hi " + name\n'
IFACE_V2 = ('def greet(name):\n    return "hi " + name\n\n\n'
            'def farewell(name):\n    return "bye " + name\n')


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


FIXTURE_FILES = {
    "iface.py": 'def placeholder():\n    ...\n',
    "impl.py": 'def run():\n    return "v0"\n',
    "calc.py": ("def add(a, b):\n    return a + b\n\n\n"
                "def mul(a, b):\n    return a - b\n"),
    "notes.txt": "line one\nline two\nline three\n",
    "data/real_tile.bin": REAL_TILE,
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


def _build_bench_home(home: Path):
    """A fresh fixture repo + composed bench in ``home``.
    Returns (runtime, base_revision)."""
    repo = home / "repo"
    repo.mkdir(parents=True)
    assert _git(repo, "init", "-q").returncode == 0
    for rel, content in FIXTURE_FILES.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    assert _git(repo, "add", "-A").returncode == 0
    assert _git(repo, "commit", "-q", "-m", "base").returncode == 0
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    db = Database(str(home / "bench.db"))
    db.migrate()
    runtime = build_bench(db, repo, home / "benchroot")
    return runtime, base


class _Lane5Case(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.runtime, self.base = _build_bench_home(self.home)
        self.repo = self.home / "repo"
        self.monitor = self.runtime.monitor
        self.contracts = self.runtime.contracts
        self.integrator = self.runtime.integrator

    # -- helpers ----------------------------------------------------

    def open_run(self, goal="lane 5 goal"):
        return self.runtime.open_run(goal, self.base, now=T0)

    def register(self, run_id, key, write_set, **kw):
        return self.monitor.register_task(
            run_id, key, _spec(write_set, **kw), now=T0)

    def claim(self, run_id, key, cursor, *, now=T0, pid=0):
        got = self.monitor.claim(run_id, key, cursor, pid=pid,
                                 now=now)
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

    def commit_count(self):
        return int(_git(self.repo, "rev-list", "--count",
                        "HEAD").stdout.strip())

    def head(self):
        return _git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def promoted_text(self, relpath, revision=""):
        return self.runtime.staging.read_at_revision(
            revision or self.head(), relpath).decode()

    def drive_fix(self, run_id, key, cursor, *, now):
        """Claim, fix mul, submit, verify supported, promote.
        Returns the promotion outcome."""
        got = self.claim(run_id, key, cursor, now=now)
        self.edit(run_id, key, cursor, "calc.py", FIXED_CALC)
        self.submit(run_id, key, cursor, got["generation"],
                    now=now + 1)
        outcome = self.runtime.pool.assign_next(run_id, now=now + 2)
        self.assertEqual(outcome["verdict"], "supported", outcome)
        promoted = self.integrator.promote_next(run_id, now=now + 3)
        self.assertTrue(promoted["promoted"], promoted)
        return promoted

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
        promoted = self.integrator.promote_next(run_id, now=now + 3)
        self.assertTrue(promoted["promoted"], promoted)

class FixtureRotationTests(_Lane5Case):
    """Proof 1 — the planted fixture-rotation defect.

    The builder's decoder transposes every (x, y) pair, and its
    fixture encoder applies the SAME transposition — so the
    candidate's own round-trip test passes while the decoder
    misreads the stored real artifact. The verifier must return
    ``contradicted`` under the real-artifact rule (the
    acceptance names no stored artifact at all), and the
    candidate must never reach the promoted tree."""

    def test_fixture_shared_misreading_is_contradicted(self):
        run = self.open_run("fixture rotation goal")
        run_id = run["id"]
        self.register(run_id, "fmt", ("fmt.py", "tests/test_fmt.py"),
                      role="format",
                      acceptance=("test:tests/test_fmt.py",))
        got = self.claim(run_id, "fmt", "cursor-a")
        decoder = ("def decode(data):\n"
                   "    pts = []\n"
                   "    for i in range(0, len(data) - 1, 2):\n"
                   "        pts.append((data[i + 1], data[i]))\n"
                   "    return pts\n")
        self.edit(run_id, "fmt", "cursor-a", "fmt.py", decoder)
        fixture_test = (
            "import fmt\n\n\n"
            "def _encode(points):\n"
            "    out = bytearray()\n"
            "    for x, y in points:\n"
            "        out += bytes((y, x))\n"
            "    return bytes(out)\n\n\n"
            "def test_round_trip():\n"
            "    points = [(1, 2), (3, 4), (5, 6), (7, 8)]\n"
            "    assert fmt.decode(_encode(points)) == points\n")
        self.edit(run_id, "fmt", "cursor-a", "tests/test_fmt.py",
                  fixture_test)
        self.submit(run_id, "fmt", "cursor-a", got["generation"],
                    now=T0 + 1)
        outcome = self.runtime.pool.assign_next(run_id, now=T0 + 2)
        self.assertEqual(outcome["verdict"], "contradicted", outcome)
        bundle = json.loads(
            Path(outcome["evidence_ref"]).read_text(
                encoding="utf-8"))
        checks = {c["name"]: c for c in bundle["checks"]}
        # The trap is real: the candidate's own tests pass.
        self.assertEqual(checks["acceptance_tests"]["status"], "pass")
        self.assertEqual(checks["real_artifact"]["status"], "fail")
        self.assertIn("real_artifact", bundle["failed_checks"])
        # The candidate's decoder really does misread the
        # stored real artifact — read it off the candidate ref
        # itself and decode the artifact's bytes.
        task = self.task(run_id, "fmt")
        code = self.runtime.staging.read_at_revision(
            str(task["candidate_ref"]), "fmt.py")
        namespace: dict[str, object] = {}
        exec(code, namespace)  # noqa: S102 - fixture decoder
        decoded = namespace["decode"](REAL_TILE)
        self.assertEqual(decoded,
                         [(y, x) for x, y in REAL_POINTS])
        self.assertNotEqual(decoded, REAL_POINTS)
        # And it never lands: the task is returned, the queue
        # has nothing to promote, the promoted tree has no fmt.
        self.assertEqual(task["state"], "returned")
        self.assertIsNone(
            self.integrator.promote_next(run_id, now=T0 + 3))
        with self.assertRaises(BenchError):
            self.runtime.staging.read_at_revision(
                self.head(), "fmt.py")


class CollisionTests(_Lane5Case):
    """Proof 2 — overlapping write-sets refuse the second claim
    naming the holder and the colliding files, and no partial
    lease survives the refusal."""

    def test_second_claim_refused_and_holds_nothing(self):
        run = self.open_run("collision goal")
        run_id = run["id"]
        self.register(run_id, "a", ("calc.py", "tests/test_add.py"))
        self.register(run_id, "b", ("calc.py", "tests/test_mul.py"))
        self.claim(run_id, "a", "cursor-a")
        refused = self.monitor.claim(run_id, "b", "cursor-b", now=T0)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "write_set_held")
        self.assertEqual(refused["holder"], "cursor-a")
        self.assertEqual(refused["holder_task"], "a")
        self.assertIn("calc.py", refused["files"])
        # The refusal is journaled, and the refused task holds
        # nothing: still pending, no holder, no leases.
        events = self.events(run_id, "claim_refused", "b")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reason"], "write_set_held")
        task_b = self.task(run_id, "b")
        self.assertEqual(task_b["state"], "pending")
        self.assertEqual(task_b["holder"], "")
        self.assertEqual(task_b["leases"], [])
        # A third cursor can still claim a file the refused
        # cursor would have half-leased — nothing was granted.
        self.register(run_id, "c", ("tests/test_mul.py",))
        self.claim(run_id, "c", "cursor-c")
        self.assertEqual(self.task(run_id, "c")["holder"],
                         "cursor-c")


class InvisibilityTests(_Lane5Case):
    """Proof 3 — a peer cursor cannot see another's unpromoted
    candidate: not in the read view, not through staging access,
    not as a dependency artifact."""

    def test_unpromoted_candidate_is_absent_for_peers(self):
        run = self.open_run("invisibility goal")
        run_id = run["id"]
        self.register(run_id, "fix", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim(run_id, "fix", "cursor-a")
        self.edit(run_id, "fix", "cursor-a", "calc.py", FIXED_CALC)
        self.submit(run_id, "fix", "cursor-a", got["generation"],
                    now=T0 + 1)
        task = self.task(run_id, "fix")
        self.assertEqual(task["state"], "submitted")
        candidate_ref = str(task["candidate_ref"])
        self.assertTrue(candidate_ref)
        # The peer's view carries the state and the spec — but
        # the candidate ref is structurally absent.
        peer = [t for t in self.monitor.store.peer_view(
            run_id, for_cursor="cursor-b") if t["key"] == "fix"][0]
        self.assertEqual(peer["candidate_ref"], "")
        self.assertFalse(peer["candidate_visible"])
        # The holder's own view still resolves it.
        own = [t for t in self.monitor.store.peer_view(
            run_id, for_cursor="cursor-a") if t["key"] == "fix"][0]
        self.assertTrue(own["candidate_visible"])
        self.assertEqual(own["candidate_ref"], candidate_ref)
        # No staging route reaches a peer's area, and an
        # unpromoted task offers no dependency artifact.
        with self.assertRaises(BenchError):
            self.runtime.views.own_staging_path(
                run_id, "fix", "cursor-b")
        with self.assertRaises(BenchError):
            self.runtime.views.dependency_artifact(
                run_id, "fix", "calc.py")

class KillAndFenceTests(_Lane5Case):
    """Proof 4 — a claimant that dies mid-work loses its task
    and its staging is quarantined (preserved, never reused);
    its late submit is fenced, and a successor completes the
    task."""

    def _dead_pid(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        return proc.pid

    def test_pid_death_requeues_quarantines_and_fences(self):
        run = self.open_run("kill and fence goal")
        run_id = run["id"]
        self.register(run_id, "fix", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim(run_id, "fix", "cursor-a", now=T0,
                         pid=self._dead_pid())
        partial = ("def add(a, b):\n"
                   "    return a + b  # partial\n\n\n"
                   "def mul(a, b):\n    return a - b\n")
        self.edit(run_id, "fix", "cursor-a", "calc.py", partial)
        swept = self.monitor.sweep(now=T0 + 5)
        self.assertIn("fix", swept["expired"])
        task = self.task(run_id, "fix")
        self.assertEqual(task["state"], "requeued")
        self.assertEqual(task["holder"], "")
        self.assertEqual(task["claim_generation"], 0)
        expired = self.events(run_id, "claim_expired", "fix")
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0]["reason"], "pid_dead")
        self.assertTrue(self.events(run_id, "task_requeued", "fix"))
        # The staging is quarantined with the partial work
        # inside it — kept for inspection, never a start point.
        quarantined = self.events(run_id, "staging_quarantined",
                                  "fix")
        self.assertEqual(len(quarantined), 1)
        area = Path(quarantined[0]["detail"]["path"])
        self.assertTrue(area.is_dir())
        self.assertIn("# partial",
                      (area / "calc.py").read_text(encoding="utf-8"))
        # The dead cursor's late submit is fenced by generation.
        late = self.monitor.submit(run_id, "fix", "cursor-a",
                                   got["generation"], now=T0 + 6)
        self.assertFalse(late["submitted"])
        self.assertEqual(late["reason"], "stale_generation")
        # A successor claims fresh, completes, and promotes.
        self.drive_fix(run_id, "fix", "cursor-b", now=T0 + 10)
        self.assertEqual(
            self.task(run_id, "fix")["state"], TASK_PROMOTED)
        self.assertIn("return a * b", self.promoted_text("calc.py"))

    def test_lease_lapse_requeues_and_fences(self):
        run = self.open_run("lease lapse goal")
        run_id = run["id"]
        self.register(run_id, "fix", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim(run_id, "fix", "cursor-a", now=T0)
        # Inside the lease the watchdog leaves the claim alone.
        self.assertEqual(
            self.monitor.sweep(now=T0 + 5)["expired"], [])
        # Past the lease TTL the claim is dead at read.
        swept = self.monitor.sweep(now=T0 + 121)
        self.assertIn("fix", swept["expired"])
        expired = self.events(run_id, "claim_expired", "fix")
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0]["reason"], "lease_expired")
        self.assertEqual(self.task(run_id, "fix")["state"],
                         "requeued")
        late = self.monitor.submit(run_id, "fix", "cursor-a",
                                   got["generation"], now=T0 + 122)
        self.assertFalse(late["submitted"])
        self.assertEqual(late["reason"], "stale_generation")
        self.drive_fix(run_id, "fix", "cursor-b", now=T0 + 130)
        self.assertEqual(
            self.task(run_id, "fix")["state"], TASK_PROMOTED)


class ConflictTests(_Lane5Case):
    """Proof 5 — a candidate that conflicts at promotion lands
    ``conflicted`` with journaled evidence, the tree untouched,
    and the queue continues promoting later tasks.

    The conflicting file (``notes.txt``) is in NEITHER task's
    write-set: task ``x`` touches it indirectly in its staging
    area, while task ``y`` owns it and promotes first. The
    verifier's write-set rule refuses ``x``'s candidate at the
    gate (asserted below), so the proof presents ``x`` to the
    integrator the way a post-gate candidate arrives — a
    ``supported`` verdict for its captured candidate ref
    recorded through the store's own verdict seam, the
    verifier's write path — and asserts the integrator's
    honesty from there. The candidate ref itself is real:
    captured through the ordinary submit path from a real
    staging area."""

    def test_indirect_conflict_is_honest_and_queue_continues(self):
        run = self.open_run("conflict goal")
        run_id = run["id"]
        # Task x: its declared work is x_impl.py, but its
        # staging also touches notes.txt (an indirect touch no
        # lease covers — neither write-set names it).
        self.register(run_id, "x", ("x_impl.py",),
                      acceptance=("cmd:true",))
        got_x = self.claim(run_id, "x", "cursor-x")
        self.edit(run_id, "x", "cursor-x", "x_impl.py", "x = 1\n")
        self.edit(run_id, "x", "cursor-x", "notes.txt",
                  "line one (x)\nline two\nline three\n")
        self.submit(run_id, "x", "cursor-x", got_x["generation"],
                    now=T0 + 1)
        task_x = self.task(run_id, "x")
        captured = self.events(run_id, "candidate_captured", "x")
        self.assertIn("notes.txt", captured[0]["detail"]["files"])
        # Task y owns notes.txt and promotes first, changing the
        # very line x's candidate also changes.
        self.register(run_id, "y", ("notes.txt",),
                      acceptance=("cmd:true",))
        got_y = self.claim(run_id, "y", "cursor-y", now=T0 + 2)
        self.edit(run_id, "y", "cursor-y", "notes.txt",
                  "line one (y)\nline two\nline three\n")
        self.submit(run_id, "y", "cursor-y", got_y["generation"],
                    now=T0 + 3)
        # The pool verifies the oldest submitted first: x's
        # candidate, whose indirect touch the gate refuses.
        outcome_x = self.runtime.pool.assign_next(run_id, now=T0 + 4)
        self.assertEqual(outcome_x["key"], "x")
        self.assertEqual(outcome_x["verdict"], "contradicted",
                         outcome_x)
        bundle_x = json.loads(
            Path(outcome_x["evidence_ref"]).read_text(
                encoding="utf-8"))
        self.assertIn("write_set", bundle_x["failed_checks"])
        self.assertEqual(self.task(run_id, "x")["state"], "returned")
        outcome_y = self.runtime.pool.assign_next(run_id, now=T0 + 5)
        self.assertEqual(outcome_y["key"], "y")
        self.assertEqual(outcome_y["verdict"], "supported",
                         outcome_y)
        promoted_y = self.integrator.promote_next(run_id, now=T0 + 6)
        self.assertTrue(promoted_y["promoted"], promoted_y)
        rev1 = self.head()
        self.assertEqual(self.promoted_text("notes.txt"),
                         "line one (y)\nline two\nline three\n")
        # Present x to the integrator as a verified candidate.
        self.monitor.store.record_verdict(
            run_id, task_x["id"],
            candidate_ref=str(task_x["candidate_ref"]),
            verifier="bench-verifier-9", verdict="supported",
            detail={"staged": "post-gate candidate"}, now=T0 + 7)
        moved = self.monitor.store.transition(
            task_x["id"], ["returned"], TASK_VERIFYING, now=T0 + 7)
        self.assertTrue(moved)
        outcome = self.integrator.promote_next(run_id, now=T0 + 8)
        self.assertTrue(outcome.get("conflicted"), outcome)
        self.assertEqual(self.task(run_id, "x")["state"],
                         TASK_CONFLICTED)
        conflicted = self.events(run_id, "task_conflicted", "x")
        self.assertEqual(len(conflicted), 1)
        detail = conflicted[0]["detail"]
        self.assertIn("notes.txt", detail["files"])
        # The evidence names the conflict's shape: the file that
        # blocked the promotion is in neither write-set.
        self.assertEqual(detail["in_write_set"], {"notes.txt": False})
        defects = self.events(run_id, "monitor_defect", "x")
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["reason"],
                         "promotion_conflict_after_admission")
        # The tree is untouched by the conflict: HEAD unmoved,
        # y's version of the file still the promoted content,
        # x's version nowhere in it.
        self.assertEqual(self.head(), rev1)
        self.assertEqual(self.promoted_text("notes.txt"),
                         "line one (y)\nline two\nline three\n")
        # And the queue continues: a later clean task promotes.
        self.register(run_id, "z", ("z_impl.py",),
                      acceptance=("cmd:true",))
        got_z = self.claim(run_id, "z", "cursor-z", now=T0 + 10)
        self.edit(run_id, "z", "cursor-z", "z_impl.py", "z = 1\n")
        self.submit(run_id, "z", "cursor-z", got_z["generation"],
                    now=T0 + 11)
        outcome_z = self.runtime.pool.assign_next(run_id, now=T0 + 12)
        self.assertEqual(outcome_z["verdict"], "supported", outcome_z)
        promoted_z = self.integrator.promote_next(run_id, now=T0 + 13)
        self.assertTrue(promoted_z["promoted"], promoted_z)
        self.assertNotEqual(self.head(), rev1)
        self.assertEqual(self.promoted_text("z_impl.py"), "z = 1\n")


class ExactlyOnceTests(_Lane5Case):
    """Proof 6 — the integrator crashes between the git commit
    and the store transaction; the retry adopts the landed
    commit exactly once and HEAD advances by one."""

    def test_crash_mid_promotion_retries_to_exactly_once(self):
        run = self.open_run("exactly once goal")
        run_id = run["id"]
        self.register(run_id, "fix", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim(run_id, "fix", "cursor-a")
        self.edit(run_id, "fix", "cursor-a", "calc.py", FIXED_CALC)
        self.submit(run_id, "fix", "cursor-a", got["generation"],
                    now=T0 + 1)
        outcome = self.runtime.pool.assign_next(run_id, now=T0 + 2)
        self.assertEqual(outcome["verdict"], "supported", outcome)
        base_count = self.commit_count()

        class CrashOnce(BenchIntegrator):
            armed = True

            def _promotion_checkpoint(self, phase, task_, revision=""):
                if self.armed and phase == "after_git_commit":
                    self.armed = False
                    raise RuntimeError("simulated crash")

        crashing = CrashOnce(self.monitor, stale_after_s=0.2)
        with self.assertRaises(RuntimeError):
            crashing.promote_next(run_id, now=T0 + 3)
        # The git commit landed; the journal did not.
        self.assertEqual(self.commit_count(), base_count + 1)
        self.assertEqual(self.task(run_id, "fix")["state"],
                         TASK_VERIFYING)
        journal = self.monitor.store.get_run(run_id)
        self.assertEqual(journal["promoted_revision"], self.base)
        self.assertNotEqual(self.head(), self.base)
        # The retry adopts the landed commit and journals it.
        retry = crashing.promote_next(run_id, now=T0 + 4)
        self.assertTrue(retry["promoted"])
        self.assertTrue(retry["adopted"])
        self.assertEqual(self.commit_count(), base_count + 1)
        journal = self.monitor.store.get_run(run_id)
        self.assertEqual(journal["promoted_revision"],
                         retry["revision"])
        self.assertEqual(journal["promoted_revision"], self.head())
        self.assertEqual(self.task(run_id, "fix")["state"],
                         TASK_PROMOTED)
        # A third call has nothing to do and changes nothing.
        self.assertIsNone(crashing.promote_next(run_id, now=T0 + 5))
        self.assertEqual(self.commit_count(), base_count + 1)

class CoupledJobTests(_Lane5Case):
    """Proof 7 — the tightly-coupled job (the owner's case).

    Two cursors build opposite sides of one interface under a
    promoted contract. Mid-run the contract is challenged and
    revised to a new generation; the bound dependents suspend,
    rebase onto the new generation, and both sides land
    consistent with it — the promoted tree satisfies the
    revised contract's acceptance."""

    def test_two_sides_land_consistent_with_revised_contract(self):
        run = self.open_run("coupled goal")
        run_id = run["id"]
        self.contracts.define(run_id, "api", "iface.py", now=T0)
        self.register(run_id, "server",
                      ("server.py", "tests/test_server.py"),
                      deps=("api",),
                      acceptance=("test:tests/test_server.py",))
        self.register(run_id, "client",
                      ("client.py", "tests/test_client.py"),
                      deps=("api",),
                      acceptance=("test:tests/test_client.py",))
        self.register(run_id, "impl3", ("impl3.py",), deps=("api",),
                      acceptance=("cmd:true",))
        for key in ("server", "client", "impl3"):
            self.contracts.bind(run_id, key, "api", now=T0)
        # Generation 1 publishes; both sides claim and start
        # building against it.
        self.drive_contract(run_id, IFACE_V1, now=T0 + 10)
        published = self.contracts.sync_promotions(run_id, now=T0 + 20)
        self.assertEqual(
            [(p["contract"], p["generation"]) for p in published],
            [("api", 1)])
        server_v1 = ("import iface\n\n\n"
                     "def serve(name):\n"
                     '    return iface.greet(name)\n')
        server_test_v1 = ("import server\n\n\n"
                          "def test_serve():\n"
                          '    assert server.serve("ana") == "hi ana"\n')
        client_v1 = ("import iface\n\n\n"
                     "def call(name):\n"
                     '    return iface.greet(name)\n')
        client_test_v1 = ("import client\n\n\n"
                          "def test_call():\n"
                          '    assert client.call("ana") == "hi ana"\n')
        got_server = self.claim(run_id, "server", "cursor-server",
                                now=T0 + 30)
        self.edit(run_id, "server", "cursor-server", "server.py",
                  server_v1)
        self.edit(run_id, "server", "cursor-server",
                  "tests/test_server.py", server_test_v1)
        got_client = self.claim(run_id, "client", "cursor-client",
                                now=T0 + 30)
        self.edit(run_id, "client", "cursor-client", "client.py",
                  client_v1)
        self.edit(run_id, "client", "cursor-client",
                  "tests/test_client.py", client_test_v1)
        # Mid-run the contract is challenged: the interface has
        # no farewell(). Bound tasks suspend; the contract task
        # reopens for its author's revision.
        challenged = self.contracts.challenge(
            run_id, "api", challenger="cursor-client",
            reason="the interface has no farewell()", now=T0 + 40)
        self.assertEqual(challenged["state"], "challenged")
        self.assertEqual(sorted(challenged["suspended"]),
                         ["client", "impl3", "server"])
        self.assertEqual(self.task(run_id, "api")["state"],
                         TASK_REQUEUED)
        for key in ("server", "client", "impl3"):
            self.assertTrue(
                self.events(run_id, "task_suspended", key))
        refused = self.monitor.claim(run_id, "impl3", "cursor-3",
                                     now=T0 + 41)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "contract_challenged")
        self.assertEqual(refused["contract"], "api")
        self.assertEqual(refused["generation"], 1)
        # The revision publishes generation 2 and rebases every
        # binding onto it, lifting the suspension.
        self.drive_contract(run_id, IFACE_V2, now=T0 + 50)
        published = self.contracts.sync_promotions(run_id, now=T0 + 60)
        self.assertEqual(
            [(p["contract"], p["generation"]) for p in published],
            [("api", 2)])
        self.assertTrue(self.events(run_id, "contract_revised", "api"))
        for key in ("server", "client", "impl3"):
            binding = self.contracts.binding_for(
                self.task(run_id, key)["id"])
            self.assertEqual(binding["generation"], 2)
            rebased = self.events(run_id, "task_rebased", key)
            self.assertEqual(rebased[-1]["detail"]["to_generation"], 2)
            self.assertTrue(
                rebased[-1]["detail"]["suspension_lifted"])
        # Generation 1 stays readable; generation 2 is in force.
        self.assertEqual(
            self.contracts.at_generation(
                run_id, "api", 1)["content"].decode(), IFACE_V1)
        self.assertEqual(
            self.contracts.current(run_id, "api")["content"].decode(),
            IFACE_V2)
        # The guard lifts: the third bound task claims now.
        self.claim(run_id, "impl3", "cursor-3", now=T0 + 70)
        # Both sides rebase: yield the generation-1 staging and
        # re-claim at the generation-2 base, then build the new
        # side of the interface against it.
        gen2_rev = self.contracts.get(run_id, "api")[
            "current_revision"]
        for key, cursor, gen in (("server", "cursor-server",
                                  got_server["generation"]),
                                 ("client", "cursor-client",
                                  got_client["generation"])):
            released = self.monitor.release(
                run_id, key, cursor, gen, reason="contract rebase",
                now=T0 + 71)
            self.assertTrue(released["released"], released)
            self.assertEqual(self.task(run_id, key)["state"],
                             "pending")
        server_v2 = ("import iface\n\n\n"
                     "def serve(name):\n"
                     '    return iface.greet(name)\n\n\n'
                     "def part(name):\n"
                     '    return iface.farewell(name)\n')
        server_test_v2 = ("import server\n\n\n"
                          "def test_serve():\n"
                          '    assert server.serve("ana") == "hi ana"\n'
                          "\n\n"
                          "def test_part():\n"
                          '    assert server.part("ana") == "bye ana"\n')
        client_v2 = ("import iface\n\n\n"
                     "def call(name):\n"
                     '    return iface.greet(name)\n\n\n'
                     "def leave(name):\n"
                     '    return iface.farewell(name)\n')
        client_test_v2 = ("import client\n\n\n"
                          "def test_call():\n"
                          '    assert client.call("ana") == "hi ana"\n'
                          "\n\n"
                          "def test_leave():\n"
                          '    assert client.leave("ana") == "bye ana"\n')
        got_server2 = self.claim(run_id, "server", "cursor-server",
                                 now=T0 + 72)
        self.assertEqual(got_server2["base_revision"], gen2_rev)
        self.edit(run_id, "server", "cursor-server", "server.py",
                  server_v2)
        self.edit(run_id, "server", "cursor-server",
                  "tests/test_server.py", server_test_v2)
        self.submit(run_id, "server", "cursor-server",
                    got_server2["generation"], now=T0 + 73)
        got_client2 = self.claim(run_id, "client", "cursor-client",
                                 now=T0 + 72)
        self.assertEqual(got_client2["base_revision"], gen2_rev)
        self.edit(run_id, "client", "cursor-client", "client.py",
                  client_v2)
        self.edit(run_id, "client", "cursor-client",
                  "tests/test_client.py", client_test_v2)
        self.submit(run_id, "client", "cursor-client",
                    got_client2["generation"], now=T0 + 73)
        outcome_s = self.runtime.pool.assign_next(run_id, now=T0 + 74)
        self.assertEqual(outcome_s["verdict"], "supported", outcome_s)
        outcome_c = self.runtime.pool.assign_next(run_id, now=T0 + 75)
        self.assertEqual(outcome_c["verdict"], "supported", outcome_c)
        promoted = self.integrator.promote_all(run_id, now=T0 + 76)
        self.assertEqual(len(promoted), 2, promoted)
        self.assertTrue(all(p["promoted"] for p in promoted))
        self.assertEqual(self.task(run_id, "server")["state"],
                         TASK_PROMOTED)
        self.assertEqual(self.task(run_id, "client")["state"],
                         TASK_PROMOTED)
        # The promoted tree satisfies the revised contract: the
        # new operation exists on both sides, and both sides'
        # tests run green against the promoted tree itself.
        self.assertIn("def farewell", self.promoted_text("iface.py"))
        self.assertIn("farewell", self.promoted_text("server.py"))
        self.assertIn("farewell", self.promoted_text("client.py"))
        tree_run = run_tests(
            paths=["tests/test_server.py", "tests/test_client.py"],
            repo=str(self.repo), timeout=180.0)
        self.assertTrue(tree_run and tree_run.get("ok"), tree_run)


class StabilityTests(_Lane5Case):
    """Proof 8 — verdict stability.

    The Lane 4 pool test twice returned ``contradicted`` inside
    combined-suite processes while passing solo every time. The
    cause was stale bytecode in the verifier's two-revision
    scratch (a same-size one-character fix whose base checkout
    landed in the same second as the compile executed the
    candidate's code on the base side — see
    ``test_bench_verifier.StaleBytecodeTests``). This test pins
    the composed behaviour: after the mock suites have run in
    this same process, the pool flow returns ``supported``
    repeatedly."""

    def _one_pool_flow(self, index):
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name)
        runtime, base = _build_bench_home(home)
        monitor = runtime.monitor
        run = runtime.open_run("stability goal", base, now=T0)
        run_id = run["id"]
        monitor.register_task(
            run_id, "fix",
            _spec(("calc.py", "tests/test_mul.py"), role="bugfix",
                  acceptance=("test:tests/test_mul.py",)), now=T0)
        got = monitor.claim(run_id, "fix", "cursor-a", now=T0)
        self.assertTrue(got["claimed"], got)
        path = runtime.views.own_staging_path(
            run_id, "fix", "cursor-a") / "calc.py"
        path.write_text(FIXED_CALC, encoding="utf-8")
        submitted = monitor.submit(run_id, "fix", "cursor-a",
                                   got["generation"], now=T0 + 1)
        self.assertTrue(submitted["submitted"], submitted)
        pool = VerifierPool(monitor, home / "pool",
                            verifier_ids=("cursor-a",
                                          "bench-verifier-9"),
                            test_timeout_s=180.0)
        outcome = pool.assign_next(run_id, now=T0 + 2)
        self.assertTrue(outcome["assigned"], outcome)
        self.assertEqual(outcome["verifier"], "bench-verifier-9")
        self.assertEqual(outcome["verdict"], "supported", outcome)

    def test_pool_flow_supported_repeatedly_after_mock_suites(self):
        inner_bt = self.home / "inner-bt"
        inner_bt.mkdir(parents=True, exist_ok=True)
        rc = pytest.main([
            "-q", "tests/test_mock_screens.py",
            "tests/test_mock_screens_phase2b.py",
            "tests/test_mock_formats_wiring.py",
            "--basetemp=" + str(inner_bt),
            "-p", "no:cacheprovider",
        ])
        self.assertEqual(rc, 0)
        for index in range(3):
            self._one_pool_flow(index)


if __name__ == "__main__":
    unittest.main()
