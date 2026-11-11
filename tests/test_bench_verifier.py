"""Multi-cursor bench, Lane 3 — the verifier cursor + the gate.

Pins: (a) the verifier role is capability-fenced — its write
tools are denied on production paths by the enforcing
registry's own rule, while test execution stays allowed (the
tester-style carve-out); (b) the verify request type carries
exactly spec + base + candidate — no field exists for a
builder's self-report; (c) NO verdict -> NO promotion: the
queue refuses a merely submitted candidate and says why via
``blocked()``; (d) a supported verdict promotes, end to end;
(e) a contradicted verdict returns the task WITH its evidence
bundle, the builder resubmits as a NEW candidate, and the old
verdict is powerless over the new ref; (f) builder-as-own-
verifier is structurally refused by the delegation store —
the verdict cannot be written and no bench verdict row
exists; (g) the check table: write-set escape contradicts
naming the file; a bugfix whose proving test passes on base
too is contradicted; a migrations.py change is contradicted
by hash; format evidence resting on the candidate's own
fixtures is contradicted by rule while evidence against a
stored real artifact passes; a layering failure contradicts;
(h) pre-existing adjoining failures make a verdict partial —
which does not promote; a candidate nothing executable ran
against is uncertain — which does not promote either;
(i) a task sitting submitted past its budget returns to the
pool (the submitted-state timeout); (j) exhausting the
attempt budget escalates — a skeptic challenge on the
contract and a journal event naming the owner as arbiter.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from nomorals.agents.bench import (
    BenchIntegrator,
    BenchMonitor,
    BenchVerifier,
    StagingManager,
    TaskSpec,
    VerifyRequest,
)
from nomorals.agents.bench.staging import BenchViews
from nomorals.agents.delegation import IndependenceError
from nomorals.agents.role_specs import BUILTIN_ROLES, check_spec_call
from nomorals.storage.db import Database
from nomorals.tools.pytest_runner import run_tests

T0 = 1_700_000_000.0

_IDENTITY = ("-c", "user.name=Bench Tests",
             "-c", "user.email=bench-tests@nomorals.invalid",
             "-c", "commit.gpgsign=false")


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


class _VerifierCase(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.repo = self.home / "repo"
        self.repo.mkdir()
        self.assertEqual(_git(self.repo, "init", "-q").returncode, 0)
        files = {
            "calc.py": (
                "def add(a, b):\n    return a + b\n\n\n"
                "def mul(a, b):\n    return a - b\n"),
            "shared.py": "def helper():\n    return 1\n",
            "broken.py": "def broken():\n    return False\n",
            "decoder.py": "def decode(data):\n    return list(data)\n",
            "nomorals/storage/migrations.py": (
                "# the frozen file, fixture shape\nVALUE = 1\n"),
            "tests/test_add.py": (
                "import calc\n\n\n"
                "def test_add():\n    assert calc.add(2, 3) == 5\n"),
            "tests/test_mul.py": (
                "import calc\n\n\n"
                "def test_mul():\n    assert calc.mul(2, 3) == 6\n"),
            "tests/test_shared.py": (
                "import shared\n\n\n"
                "def test_helper():\n"
                "    assert shared.helper() == 1\n"),
            "tests/test_broken.py": (
                "import broken\nimport shared\n\n\n"
                "def test_broken():\n"
                "    assert shared.helper() == 1\n"
                "    assert broken.broken() is True\n"),
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
        artifact = self.repo / "fixtures" / "sample.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"\x01\x02\x03real-bytes")
        self.assertEqual(_git(self.repo, "add", "-A").returncode, 0)
        self.assertEqual(
            _git(self.repo, "commit", "-q", "-m", "base").returncode, 0)
        self.base = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        self.db = Database(str(self.home / "bench.db"))
        self.staging = StagingManager(self.repo,
                                      self.home / "benchroot")
        self.monitor = BenchMonitor(
            self.db, lease_ttl_s=10000.0, stall_after_s=5.0,
            staging=self.staging)
        self.views = BenchViews(self.monitor.store, self.staging)
        self.integrator = BenchIntegrator(self.monitor,
                                          stale_after_s=0.2)
        self.verifier = BenchVerifier(
            self.monitor, self.home / "vscratch",
            test_timeout_s=180.0)
        self.run = self.monitor.open_run("goal", self.base, now=T0)
        self.run_id = self.run["id"]

    # -- helpers ----------------------------------------------------

    def register(self, key, write_set, **kw):
        return self.monitor.register_task(
            self.run_id, key, _spec(write_set, **kw), now=T0)

    def claim(self, key, cursor, *, now=T0):
        got = self.monitor.claim(self.run_id, key, cursor, now=now)
        self.assertTrue(got["claimed"], got)
        return got

    def edit(self, key, cursor, relpath, content):
        path = self.views.own_staging_path(
            self.run_id, key, cursor) / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def submit(self, key, cursor, generation, *, now):
        got = self.monitor.submit(self.run_id, key, cursor,
                                  generation, now=now)
        self.assertTrue(got["submitted"], got)
        return got

    def task(self, key):
        return self.monitor.store.get_task(self.run_id, key)

    def events(self, kind, key=None):
        task_id = self.task(key)["id"] if key else ""
        return self.monitor.store.events(self.run_id, task_id, kind)

    def verdicts(self, key):
        return self.monitor.store.verdicts(self.task(key)["id"])

    def failed_checks(self, key):
        rows = self.verdicts(key)
        self.assertTrue(rows)
        return rows[-1]["detail"]["failed_checks"]


class RoleAndRequestTests(_VerifierCase):

    def test_verifier_role_is_capability_fenced(self):
        spec = BUILTIN_ROLES["verifier"]
        # Production writes are denied by the registry's own
        # rule — the role cannot fix what it judges.
        denied = check_spec_call(
            spec, "fs_write",
            {"path": "nomorals/agents/bench/verifier.py"})
        self.assertIsNotNone(denied)
        self.assertEqual(denied.reason, "path_guard")
        # The tester-style carve-out: test paths writable, test
        # execution allowed.
        self.assertIsNone(check_spec_call(
            spec, "fs_write", {"path": "tests/test_x.py"}))
        self.assertIsNone(check_spec_call(spec, "run_tests", {}))
        self.assertIsNone(check_spec_call(spec, "shell_run", {}))
        # And tools outside the allowlist are denied outright.
        self.assertIsNotNone(check_spec_call(spec, "git_push", {}))

    def test_verify_request_has_no_self_report_field(self):
        names = [f.name for f in dataclasses.fields(VerifyRequest)]
        self.assertEqual(
            names,
            ["run_id", "task_id", "task_key", "instructions",
             "acceptance", "role", "write_set", "base_revision",
             "candidate_ref"])
        for banned in ("summary", "report", "claimed", "holder",
                       "notes", "self_report"):
            self.assertNotIn(banned, names)


class GateTests(_VerifierCase):

    def test_no_verdict_no_promotion(self):
        self.register("a", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        # Submitted is not promotable — the queue is empty and
        # says why it is holding the task.
        self.assertEqual(self.integrator.pending(self.run_id), [])
        self.assertIsNone(self.integrator.promote_next(self.run_id))
        blocked = self.integrator.blocked(self.run_id)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["key"], "a")
        self.assertEqual(blocked[0]["reason"], "awaiting_verdict")
        self.assertEqual(self.task("a")["state"], "submitted")

    def test_supported_verdict_promotes_end_to_end(self):
        self.register("a", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertTrue(outcome["verified"], outcome)
        self.assertEqual(outcome["verdict"], "supported")
        # The verdict lives in BOTH ledgers: the bench row the
        # gate consumes and the delegation claim it validated.
        rows = self.verdicts("a")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verdict"], "supported")
        self.assertTrue(rows[0]["detail"]["claim_id"])
        self.assertEqual(self.task("a")["state"], "verifying")
        # The gate now admits exactly this task, and promotion
        # lands the candidate's content.
        outcome = self.integrator.promote_next(self.run_id,
                                               now=T0 + 20)
        self.assertTrue(outcome["promoted"], outcome)
        self.assertEqual(self.task("a")["state"], "promoted")
        self.assertIn(b"a * b", self.views.read_promoted(
            self.run_id, "calc.py"))


class ReturnLoopTests(_VerifierCase):

    def test_contradicted_returns_with_evidence_old_verdict_powerless(
            self):
        self.register("a", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim("a", "cursor-a")
        gen = got["generation"]
        # v1: a comment-only "fix" — mul still subtracts.
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    # fixed!\n    return a - b\n")
        self.submit("a", "cursor-a", gen, now=T0 + 1)
        first_ref = self.task("a")["candidate_ref"]
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "contradicted")
        self.assertTrue(outcome["returned"])
        self.assertEqual(self.task("a")["state"], "returned")
        # The evidence bundle is a real artifact carrying the
        # failing checks — not the builder's account of them.
        bundle = json.loads(
            Path(outcome["evidence_ref"]).read_text(encoding="utf-8"))
        self.assertEqual(bundle["verdict"], "contradicted")
        self.assertIn("acceptance_tests", bundle["failed_checks"])
        returned = self.events("task_returned", "a")
        self.assertEqual(len(returned), 1)
        self.assertEqual(returned[0]["reason"], "contradicted")
        # Returned does not promote.
        self.assertIsNone(self.integrator.promote_next(self.run_id))
        # The cursor reworks in its own staging and resubmits:
        # a NEW candidate ref, over which the old verdict has
        # no power at all.
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit("a", "cursor-a", gen, now=T0 + 30)
        task = self.task("a")
        self.assertNotEqual(task["candidate_ref"], first_ref)
        self.assertIsNone(self.monitor.store.latest_verdict(
            task["id"], task["candidate_ref"]))
        self.assertIsNone(self.integrator.promote_next(self.run_id))
        # Verified afresh, the new candidate promotes.
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 40)
        self.assertEqual(outcome["verdict"], "supported")
        outcome = self.integrator.promote_next(self.run_id,
                                               now=T0 + 50)
        self.assertTrue(outcome["promoted"], outcome)


class IndependenceTests(_VerifierCase):

    def test_builder_cannot_verify_own_candidate(self):
        self.register("a", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        own = BenchVerifier(self.monitor, self.home / "vown",
                            verifier_id="cursor-a",
                            test_timeout_s=180.0)
        with self.assertRaises(IndependenceError):
            own.verify_task(self.run_id, "a", now=T0 + 10)
        # The refusal is structural: no bench verdict row was
        # written and the task is back at submitted, untouched
        # by any half-state.
        self.assertEqual(self.verdicts("a"), [])
        self.assertEqual(self.task("a")["state"], "submitted")


class CheckTableTests(_VerifierCase):

    def test_write_set_escape_contradicts_naming_the_file(self):
        self.register("a", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.edit("a", "cursor-a", "rogue.py", "rogue = True\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "contradicted")
        self.assertIn("write_set", self.failed_checks("a"))
        bundle = json.loads(
            Path(outcome["evidence_ref"]).read_text(encoding="utf-8"))
        detail = [c for c in bundle["checks"]
                  if c["name"] == "write_set"][0]["detail"]
        self.assertIn("rogue.py", detail)

    def test_bugfix_whose_test_passes_on_base_is_contradicted(self):
        # The candidate really fixes mul — but its named
        # acceptance only exercises test_add, which passes on
        # the base too. The proving test proves nothing.
        self.register("a", ("calc.py",), role="bugfix",
                      acceptance=("test:tests/test_add.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "contradicted")
        self.assertIn("proving_test", self.failed_checks("a"))
        self.assertNotIn("acceptance_tests", self.failed_checks("a"))

    def test_migrations_change_is_contradicted_by_hash(self):
        self.register(
            "a", ("nomorals/storage/migrations.py", "calc.py"),
            role="schema",
            acceptance=("test:tests/test_add.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "nomorals/storage/migrations.py",
                  "# the frozen file, fixture shape\nVALUE = 2\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "contradicted")
        self.assertIn("migrations_frozen", self.failed_checks("a"))

    def test_format_fixture_only_evidence_is_contradicted(self):
        self.register("a", ("decoder.py", "tests/test_add.py"),
                      role="format-parser",
                      acceptance=("test:tests/test_add.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "decoder.py",
                  "def decode(data):\n    return list(data)  # v2\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "contradicted")
        self.assertIn("real_artifact", self.failed_checks("a"))

    def test_format_candidate_authored_artifact_is_contradicted(self):
        self.register(
            "a", ("decoder.py", "fixtures/own.bin",
                  "tests/test_add.py"),
            role="format-parser",
            acceptance=("test:tests/test_add.py",
                        "artifact:fixtures/own.bin"))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "decoder.py",
                  "def decode(data):\n    return list(data)  # v2\n")
        path = self.views.own_staging_path(
            self.run_id, "a", "cursor-a") / "fixtures" / "own.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x09\x09fixture-i-made")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "contradicted")
        self.assertIn("real_artifact", self.failed_checks("a"))

    def test_format_real_artifact_evidence_passes(self):
        self.register(
            "a", ("decoder.py", "tests/test_add.py"),
            role="format-parser",
            acceptance=("test:tests/test_add.py",
                        "artifact:fixtures/sample.bin"))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "decoder.py",
                  "def decode(data):\n    return list(data)  # v2\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "supported")
        self.assertEqual(self.task("a")["state"], "verifying")

    def test_layering_failure_is_contradicted(self):
        self.register("a", ("forbidden.py", "tests/test_shared.py"),
                      role="feature",
                      acceptance=("test:tests/test_shared.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "forbidden.py", "x = 1\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "contradicted")
        self.assertIn("layering", self.failed_checks("a"))

    def test_preexisting_adjoining_failure_is_partial(self):
        # test_broken fails on base and candidate alike (the
        # candidate never touches broken.py). Acceptance is
        # green; the honest verdict is partial — and partial
        # does not promote.
        self.register("a", ("shared.py",), role="feature",
                      acceptance=("test:tests/test_shared.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "shared.py",
                  "def helper():\n    return 1\n\n\n"
                  "def extra():\n    return 2\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "partial")
        self.assertEqual(self.failed_checks("a"), [])
        self.assertEqual(self.task("a")["state"], "returned")
        self.assertIsNone(self.integrator.promote_next(self.run_id))

    def test_unexecuted_candidate_is_uncertain(self):
        self.register("a", ("lonely.py",), role="feature",
                      acceptance=("it works nicely",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "lonely.py",
                  "def lonely():\n    return 1\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = self.verifier.verify_task(self.run_id, "a",
                                            now=T0 + 10)
        self.assertEqual(outcome["verdict"], "uncertain")
        self.assertEqual(self.task("a")["state"], "returned")
        self.assertIsNone(self.integrator.promote_next(self.run_id))


class TimeoutAndEscalationTests(_VerifierCase):

    def test_submitted_past_budget_returns_to_pool(self):
        impatient = BenchVerifier(
            self.monitor, self.home / "vtimeout",
            verify_timeout_s=50.0, test_timeout_s=180.0)
        self.register("a", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        self.assertEqual(impatient.expire_stale(now=T0 + 30), [])
        out = impatient.expire_stale(now=T0 + 100)
        self.assertEqual(len(out), 1)
        task = self.task("a")
        self.assertEqual(task["state"], "requeued")
        self.assertEqual(task["candidate_ref"], "")
        self.assertEqual(
            len(self.events("verification_timeout", "a")), 1)
        # Idempotent, and the freed write-set claims again.
        self.assertEqual(impatient.expire_stale(now=T0 + 200), [])
        again = self.monitor.claim(self.run_id, "a", "cursor-b",
                                    now=T0 + 300)
        self.assertTrue(again["claimed"], again)

    def test_attempt_budget_exhaustion_escalates_to_owner(self):
        strict = BenchVerifier(
            self.monitor, self.home / "vstrict", attempt_budget=1,
            test_timeout_s=180.0)
        self.register("a", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    # fixed!\n    return a - b\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcome = strict.verify_task(self.run_id, "a", now=T0 + 10)
        self.assertEqual(outcome["verdict"], "contradicted")
        self.assertTrue(outcome["escalated"])
        events = self.events("verification_escalated", "a")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["detail"]["arbiter"], "owner")
        self.assertTrue(events[0]["detail"]["challenge_id"])


class StaleBytecodeTests(_VerifierCase):
    """Lane 5, Proof 8 — the verdict-stability root cause.

    A same-size one-character fix (``a - b`` -> ``a * b``)
    compiles the candidate's module into the verifier scratch's
    ``__pycache__``; CPython validates a ``.pyc`` by source
    mtime in whole seconds plus size, so a base checkout landing
    in the same second as the compile resurrected the
    candidate's bytecode on the base side — the proving test
    passed on "base" and a sound fix was contradicted. The
    verifier purges bytecode caches on every revision switch;
    this test plants the collision deterministically (the base
    source is laid over the scratch with the recorded mtime
    second) and asserts the base-side run still executes the
    base source. Without the purge it passes and this fails."""

    def test_base_side_run_ignores_stale_candidate_bytecode(self):
        self.register("a", ("calc.py", "tests/test_mul.py"),
                      role="bugfix",
                      acceptance=("test:tests/test_mul.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "calc.py",
                  "def add(a, b):\n    return a + b\n\n\n"
                  "def mul(a, b):\n    return a * b\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        request = VerifyRequest.from_task(self.task("a"))
        scratch = self.verifier._materialise(request)
        self.addCleanup(shutil.rmtree, scratch, True)
        acc = run_tests(paths=["tests/test_mul.py"],
                        repo=str(scratch), timeout=180.0)
        self.assertTrue(acc and acc.get("ok"), acc)
        pycs = list((scratch / "__pycache__").glob("calc.*.pyc"))
        self.assertEqual(len(pycs), 1, pycs)
        recorded = struct.unpack(
            "<I", pycs[0].read_bytes()[8:12])[0]

        def _on_base():
            # Force the collision the wall clock only sometimes
            # delivers: the base source, same size, same second.
            calc = scratch / "calc.py"
            os.utime(calc, (recorded, recorded))
            return run_tests(paths=["tests/test_mul.py"],
                             repo=str(scratch), timeout=180.0)

        base_run = self.verifier._run_on_base(request, scratch,
                                              _on_base)
        self.assertIsNotNone(base_run)
        self.assertFalse(base_run.get("ok"), base_run)
        self.assertIn("tests/test_mul.py::test_mul",
                      [str(f.get("test_id"))
                       for f in base_run.get("failed") or []])


if __name__ == "__main__":
    unittest.main()
