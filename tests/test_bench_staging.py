"""Multi-cursor bench, Lane 2 — staging + integration queue.

Pins: (a) claiming creates an isolated staging area pinned to
the promoted base — a git worktree on this box, proven against
a real repo — and the plain-copy fallback carries the same
semantics where worktrees don't hold; (b) a peer cursor cannot
read another cursor's staging through ANY bench view (the
holder check refuses by construction, promoted reads miss the
unpromoted files, peer_view keeps hiding the candidate);
(c) submit captures the staging diff as an immutable candidate
commit (tracked edits + new files), recorded on the task;
(d) two cursors' candidates promote SERIALLY in submit order,
the second rebased onto the first's promoted revision;
(e) a planted conflict becomes a named ``task_conflicted``
journal event (+ a ``monitor_defect``), releases the leases,
and the queue CONTINUES — and the conflicted task returns for
rebase, re-claims at the new base, and lands; (f) a crash
mid-promotion (after the git commit, before the journal)
retries into reconciliation and lands EXACTLY ONCE — and so
does a promotion whose idempotency key was left stale-running
by a dead owner; (g) promotion unblocks dependents, whose
claims pin the NEW promoted revision and whose staging starts
from it; dependency artifacts read at their promotion
revision; (h) the watchdog's stall detection reads artifact
change in the staging tree as progress (same ``stalled``
event, artifact evidence attached); (i) a dead claim's staging
is quarantined — preserved for inspection, never handed to
the next cursor; (j) an empty submit promotes without moving
the promoted revision.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.agents.bench import (
    BenchIntegrator,
    BenchMonitor,
    StagingManager,
    TaskSpec,
    promotion_key,
)
from nomorals.agents.bench.model import (
    TASK_SUBMITTED,
    TASK_VERIFYING,
    BenchError,
)
from nomorals.agents.bench.staging import BenchViews
from nomorals.missions.idempotency import IdempotencyStore
from nomorals.storage.db import Database

T0 = 1_700_000_000.0

_IDENTITY = ("-c", "user.name=Bench Tests",
             "-c", "user.email=bench-tests@nomorals.invalid",
             "-c", "commit.gpgsign=false")


def _git(cwd: Path, *args: str, stdin: str | None = None):
    return subprocess.run(
        ["git", *_IDENTITY, *args], cwd=str(cwd), input=stdin,
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120)


def _spec(write_set, *, deps=(), instructions="do the thing",
          acceptance=("tests pass",)):
    return TaskSpec(instructions=instructions,
                    acceptance=tuple(acceptance),
                    write_set=tuple(write_set), deps=tuple(deps))


class _BenchCase(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.repo = self.home / "repo"
        self.repo.mkdir()
        self.assertEqual(_git(self.repo, "init", "-q").returncode, 0)
        (self.repo / "shared.txt").write_text(
            "line one\nline two\nline three\nline four\nline five\n",
            encoding="utf-8")
        (self.repo / "lib.py").write_text("VALUE = 1\n",
                                          encoding="utf-8")
        (self.repo / "generated.txt").write_text("generated base\n",
                                                 encoding="utf-8")
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

    def staging_path(self, key, cursor):
        return self.views.own_staging_path(self.run_id, key, cursor)

    def edit(self, key, cursor, relpath, content):
        path = self.staging_path(key, cursor) / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def submit(self, key, cursor, generation, *, now):
        got = self.monitor.submit(self.run_id, key, cursor,
                                  generation, now=now)
        self.assertTrue(got["submitted"], got)
        # Lane 3: promotion is verdict-gated — the queue only
        # consumes verifying tasks whose latest verdict for the
        # current candidate ref is supported. These Lane-2
        # suites pin staging/integration mechanics, so the
        # helper drives the gate exactly as a supported
        # verifier verdict leaves it (the verifier organ itself
        # is pinned in tests/test_bench_verifier.py).
        task = self.task(key)
        self.assertTrue(self.monitor.store.transition(
            task["id"], [TASK_SUBMITTED], TASK_VERIFYING,
            now=now + 0.1))
        self.monitor.store.record_verdict(
            self.run_id, task["id"],
            candidate_ref=task["candidate_ref"],
            verifier="bench-verifier", verdict="supported",
            evidence_ref="lane2-helper", now=now + 0.2)
        return got

    def task(self, key):
        return self.monitor.store.get_task(self.run_id, key)

    def events(self, kind, key=None):
        task_id = self.task(key)["id"] if key else ""
        return self.monitor.store.events(self.run_id, task_id, kind)

    def commit_count(self):
        out = _git(self.repo, "rev-list", "--count", "HEAD")
        return int(out.stdout.strip())

    def promoted_content(self, relpath):
        return self.views.read_promoted(self.run_id, relpath)


class StagingCreationTests(_BenchCase):

    def test_claim_creates_a_worktree_pinned_to_base(self):
        self.register("a", ("a_impl.py",))
        got = self.claim("a", "cursor-a")
        area = self.staging.area(self.run_id, got["task_id"])
        self.assertIsNotNone(area)
        # The mechanism proven on THIS box: a real git worktree.
        self.assertEqual(area.mechanism, "worktree")
        self.assertEqual(area.base_revision, self.base)
        listing = _git(self.repo, "worktree", "list",
                       "--porcelain").stdout
        self.assertTrue(
            str(area.path) in listing
            or os.path.realpath(area.path) in listing,
            listing)
        # Pinned to base: the staging tree reads base content.
        self.assertEqual(
            (area.path / "lib.py").read_text(encoding="utf-8"),
            "VALUE = 1\n")
        created = self.events("staging_created", "a")
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["detail"]["mechanism"],
                         "worktree")

    def test_copy_fallback_carries_the_same_semantics(self):
        self.staging._create_worktree = (  # noqa: SLF001 - forced wall
            lambda dest, rev: None)
        self.register("a", ("a_impl.py",))
        got = self.claim("a", "cursor-a")
        area = self.staging.area(self.run_id, got["task_id"])
        self.assertEqual(area.mechanism, "copy")
        self.edit("a", "cursor-a", "a_impl.py", "impl via copy\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        outcomes = self.integrator.promote_all(self.run_id)
        self.assertTrue(outcomes[0]["promoted"])
        self.assertEqual(self.promoted_content("a_impl.py"),
                         b"impl via copy\n")


class InvisibilityTests(_BenchCase):

    def test_peer_cannot_read_staging_through_any_view(self):
        self.register("a", ("a_impl.py",))
        self.register("b", ("b_impl.py",))
        got_a = self.claim("a", "cursor-a")
        self.claim("b", "cursor-b")
        self.edit("a", "cursor-a", "secret.txt", "sentinel-secret\n")
        self.edit("a", "cursor-a", "a_impl.py", "impl A\n")
        # The holder-checked view refuses the peer by construction.
        with self.assertRaises(BenchError):
            self.views.own_staging_path(self.run_id, "a", "cursor-b")
        # Promoted reads see only the promoted revision: neither
        # of A's files exists there yet.
        with self.assertRaises(BenchError):
            self.views.read_promoted(self.run_id, "secret.txt")
        with self.assertRaises(BenchError):
            self.views.read_promoted(self.run_id, "a_impl.py")
        self.submit("a", "cursor-a", got_a["generation"], now=T0 + 1)
        # After submit the candidate stays invisible to the peer.
        peer = self.monitor.store.peer_view(self.run_id,
                                            for_cursor="cursor-b")
        mine = [row for row in peer if row["key"] == "a"][0]
        self.assertFalse(mine["candidate_visible"])
        self.assertEqual(mine["candidate_ref"], "")
        with self.assertRaises(BenchError):
            self.views.read_promoted(self.run_id, "a_impl.py")
        # Promotion is the moment the work becomes visible.
        self.integrator.promote_all(self.run_id)
        self.assertEqual(self.promoted_content("a_impl.py"),
                         b"impl A\n")


class CaptureTests(_BenchCase):

    def test_submit_captures_the_staging_diff_as_a_commit(self):
        self.register("a", ("lib.py", "newmod.py"))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "lib.py", "VALUE = 2\n")
        self.edit("a", "cursor-a", "newmod.py", "new module\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        task = self.task("a")
        ref = task["candidate_ref"]
        self.assertTrue(ref)
        self.assertEqual(
            _git(self.repo, "cat-file", "-t", ref).stdout.strip(),
            "commit")
        names = _git(self.repo, "diff", "--name-status",
                     self.base, ref).stdout
        self.assertIn("M\tlib.py", names)
        self.assertIn("A\tnewmod.py", names)
        # The candidate is pinned under its ref for the verifier.
        pinned = _git(self.repo, "rev-parse",
                      "refs/bench/candidates/" + task["id"])
        self.assertEqual(pinned.stdout.strip(), ref)
        captured = self.events("candidate_captured", "a")
        self.assertEqual(len(captured), 1)
        self.assertEqual(sorted(captured[0]["detail"]["files"]),
                         ["lib.py", "newmod.py"])
        self.assertEqual(captured[0]["detail"]["candidate_ref"], ref)

    def test_empty_submit_promotes_without_moving_the_revision(self):
        self.register("a", ("a_impl.py",))
        got = self.claim("a", "cursor-a")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        self.assertEqual(self.task("a")["candidate_ref"], "")
        outcome = self.integrator.promote_next(self.run_id)
        self.assertTrue(outcome["promoted"])
        self.assertEqual(outcome["revision"], self.base)
        run = self.monitor.store.get_run(self.run_id)
        self.assertEqual(run["promoted_revision"], self.base)
        self.assertEqual(self.task("a")["state"], "promoted")


class SerialPromotionTests(_BenchCase):

    def test_two_cursors_promote_serially_second_rebased(self):
        self.register("a", ("a_impl.py",))
        self.register("b", ("b_impl.py",))
        got_a = self.claim("a", "cursor-a")
        got_b = self.claim("b", "cursor-b")
        # Disjoint declared write-sets; both ALSO touch shared.txt
        # at disjoint hunks — the second candidate was captured
        # against the old base and must rebase onto the first's
        # promotion.
        shared_a = ("line one A\nline two\nline three\n"
                    "line four\nline five\n")
        shared_b = ("line one\nline two\nline three\n"
                    "line four\nline five B\n")
        self.edit("a", "cursor-a", "shared.txt", shared_a)
        self.edit("b", "cursor-b", "shared.txt", shared_b)
        self.submit("a", "cursor-a", got_a["generation"], now=T0 + 1)
        self.submit("b", "cursor-b", got_b["generation"], now=T0 + 2)
        pending = self.integrator.pending(self.run_id)
        self.assertEqual([t["key"] for t in pending], ["a", "b"])
        outcomes = self.integrator.promote_all(self.run_id)
        self.assertEqual([o["task"] for o in outcomes], ["a", "b"])
        self.assertTrue(all(o["promoted"] for o in outcomes))
        rev_a, rev_b = outcomes[0]["revision"], outcomes[1]["revision"]
        # Serial: B's promotion commit sits directly on A's.
        parent = _git(self.repo, "rev-list", "--parents", "-n", "1",
                      rev_b).stdout.split()
        self.assertEqual(parent[1], rev_a)
        run = self.monitor.store.get_run(self.run_id)
        self.assertEqual(run["promoted_revision"], rev_b)
        # Both changes live in the promoted tree.
        final = self.promoted_content("shared.txt").decode()
        self.assertIn("line one A", final)
        self.assertIn("line five B", final)
        # Leases released by both promotions.
        for key in ("a", "b"):
            task = self.task(key)
            self.assertEqual(task["state"], "promoted")
            self.assertEqual(task["leases"], [])
        lease = self.monitor.leases.current("bench-file:a_impl.py")
        self.assertIsNone(lease)
        promoted_events = self.monitor.store.events(
            self.run_id, kind="task_promoted")
        self.assertEqual(
            [e["detail"]["revision"] for e in promoted_events],
            [rev_a, rev_b])


class ConflictTests(_BenchCase):

    def _plant_conflict(self):
        self.register("a", ("a_impl.py",))
        self.register("c", ("c_impl.py",))
        self.register("d", ("d_impl.py",))
        got_a = self.claim("a", "cursor-a")
        got_c = self.claim("c", "cursor-c")
        got_d = self.claim("d", "cursor-d")
        # a_impl/c_impl/d_impl are the declared, disjoint sets;
        # generated.txt is in NEITHER — an indirect overlap the
        # admission check cannot see. A and C rewrite it
        # incompatibly.
        self.edit("a", "cursor-a", "a_impl.py", "impl A\n")
        self.edit("a", "cursor-a", "generated.txt", "from A\n")
        self.edit("c", "cursor-c", "c_impl.py", "impl C\n")
        self.edit("c", "cursor-c", "generated.txt", "from C\n")
        self.edit("d", "cursor-d", "d_impl.py", "impl D\n")
        self.submit("a", "cursor-a", got_a["generation"], now=T0 + 1)
        self.submit("c", "cursor-c", got_c["generation"], now=T0 + 2)
        self.submit("d", "cursor-d", got_d["generation"], now=T0 + 3)
        return got_c

    def test_conflict_is_a_named_event_and_the_queue_continues(self):
        self._plant_conflict()
        outcomes = self.integrator.promote_all(self.run_id)
        by_task = {o["task"]: o for o in outcomes}
        self.assertTrue(by_task["a"]["promoted"])
        self.assertTrue(by_task["c"]["conflicted"])
        self.assertEqual(by_task["c"]["files"], ["generated.txt"])
        # The queue continued past the conflict.
        self.assertTrue(by_task["d"]["promoted"])
        task_c = self.task("c")
        self.assertEqual(task_c["state"], "conflicted")
        self.assertEqual(task_c["leases"], [])
        # No silent merge: the promoted file is exactly A's.
        self.assertEqual(self.promoted_content("generated.txt"),
                         b"from A\n")
        conflicted = self.events("task_conflicted", "c")
        self.assertEqual(len(conflicted), 1)
        detail = conflicted[0]["detail"]
        self.assertEqual(detail["files"], ["generated.txt"])
        self.assertEqual(detail["in_write_set"],
                         {"generated.txt": False})
        self.assertTrue(detail["output"])
        defects = self.events("monitor_defect", "c")
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["reason"],
                         "promotion_conflict_after_admission")

    def test_conflicted_task_returns_for_rebase_and_lands(self):
        self._plant_conflict()
        self.integrator.promote_all(self.run_id)
        back = self.integrator.return_for_rebase(self.run_id, "c")
        self.assertTrue(back["returned"])
        self.assertEqual(self.task("c")["state"], "requeued")
        run = self.monitor.store.get_run(self.run_id)
        again = self.claim("c", "cursor-c", now=T0 + 10)
        # The re-claim pins the CURRENT promoted base, and the
        # fresh staging starts from it (A's content is there).
        self.assertEqual(again["base_revision"],
                         run["promoted_revision"])
        staging = self.staging_path("c", "cursor-c")
        self.assertEqual(
            (staging / "generated.txt").read_text(encoding="utf-8"),
            "from A\n")
        self.edit("c", "cursor-c", "generated.txt",
                  "from A, merged with C\n")
        self.submit("c", "cursor-c", again["generation"],
                    now=T0 + 11)
        outcome = self.integrator.promote_next(self.run_id)
        self.assertTrue(outcome["promoted"])
        self.assertEqual(self.promoted_content("generated.txt"),
                         b"from A, merged with C\n")
        self.assertEqual(self.task("c")["state"], "promoted")


class ExactlyOnceTests(_BenchCase):

    def _one_task_with_work(self):
        self.register("a", ("a_impl.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "a_impl.py", "impl A\n")
        self.submit("a", "cursor-a", got["generation"], now=T0 + 1)
        return self.task("a")

    def test_crash_mid_promotion_retries_to_exactly_once(self):
        task = self._one_task_with_work()
        base_count = self.commit_count()

        class CrashOnce(BenchIntegrator):
            armed = True

            def _promotion_checkpoint(self, phase, task_, revision=""):
                if self.armed and phase == "after_git_commit":
                    self.armed = False
                    raise RuntimeError("simulated crash")

        crashing = CrashOnce(self.monitor, stale_after_s=0.2)
        with self.assertRaises(RuntimeError):
            crashing.promote_next(self.run_id)
        # The git commit landed; the journal did not — the exact
        # divergence reconciliation exists for.
        self.assertEqual(self.commit_count(), base_count + 1)
        # Lane 3 re-pin: the task entered promotion from
        # `verifying` (the verdict gate), so the crash leaves it
        # verifying — not submitted as in the Lane-2 shape.
        self.assertEqual(self.task("a")["state"], "verifying")
        run = self.monitor.store.get_run(self.run_id)
        self.assertEqual(run["promoted_revision"], self.base)
        self.assertNotEqual(self.integrator.staging.head(),
                            self.base)
        # The retry adopts the landed commit and journals it.
        outcome = crashing.promote_next(self.run_id)
        self.assertTrue(outcome["promoted"])
        self.assertTrue(outcome["adopted"])
        self.assertEqual(self.commit_count(), base_count + 1)
        run = self.monitor.store.get_run(self.run_id)
        self.assertEqual(run["promoted_revision"],
                         outcome["revision"])
        self.assertEqual(self.task("a")["state"], "promoted")
        # A third call has nothing to do and changes nothing.
        self.assertIsNone(crashing.promote_next(self.run_id))
        self.assertEqual(self.commit_count(), base_count + 1)
        self.assertEqual(task["id"],
                         self.task("a")["id"])

    def test_stale_running_key_is_stolen_and_lands_once(self):
        task = self._one_task_with_work()
        base_count = self.commit_count()
        IdempotencyStore(self.db)  # creates the table
        key = promotion_key(self.run_id, task)
        old = time.time() - 100.0
        self.db.execute(
            "INSERT INTO idempotency_keys (key, status,"
            " result_json, error, owner, created_at, updated_at)"
            " VALUES (?, 'running', '{}', '', 'dead-owner', ?, ?)",
            (key, old, old))
        outcome = self.integrator.promote_next(self.run_id)
        self.assertTrue(outcome["promoted"])
        self.assertTrue(outcome["executed"])
        self.assertEqual(self.commit_count(), base_count + 1)
        # A replay of the same promotion is the recorded outcome,
        # not a second landing.
        replay = self.integrator.pending(self.run_id)
        self.assertEqual(replay, [])


class DependencyTests(_BenchCase):

    def test_promotion_unblocks_dependents_at_the_new_base(self):
        self.register("lib", ("lib.py",))
        self.register("user", ("user.py",), deps=("lib",))
        refused = self.monitor.claim(self.run_id, "user",
                                     "cursor-u", now=T0)
        self.assertFalse(refused["claimed"])
        self.assertEqual(refused["reason"], "deps_unpromoted")
        got = self.claim("lib", "cursor-l")
        self.edit("lib", "cursor-l", "lib.py", "VALUE = 2\n")
        self.submit("lib", "cursor-l", got["generation"], now=T0 + 1)
        self.integrator.promote_all(self.run_id)
        # The dep's artifact reads at its promotion revision.
        self.assertEqual(
            self.views.dependency_artifact(self.run_id, "lib",
                                           "lib.py"),
            b"VALUE = 2\n")
        run = self.monitor.store.get_run(self.run_id)
        claim = self.claim("user", "cursor-u", now=T0 + 2)
        self.assertEqual(claim["base_revision"],
                         run["promoted_revision"])
        self.assertNotEqual(claim["base_revision"], self.base)
        # The dependent's staging starts from the new base.
        staging = self.staging_path("user", "cursor-u")
        self.assertEqual(
            (staging / "lib.py").read_text(encoding="utf-8"),
            "VALUE = 2\n")


class WatchdogArtifactTests(_BenchCase):

    def test_artifact_change_counts_as_progress(self):
        self.register("a", ("a_impl.py",))
        self.claim("a", "cursor-a")
        # Silence + no artifact change: the stall lands, and it
        # carries the artifact evidence it was judged on.
        sweep = self.monitor.sweep(now=T0 + 10)
        self.assertEqual(sweep["stalled"], ["a"])
        stalled = self.events("stalled", "a")
        self.assertEqual(len(stalled), 1)
        artifact = stalled[0]["detail"]["artifact"]
        self.assertEqual(len(artifact["digest"]), 64)
        self.assertGreaterEqual(artifact["files"], 3)
        # The cursor edits its staging: the artifact signal is
        # progress even though no heartbeat ever said so.
        self.edit("a", "cursor-a", "a_impl.py", "work in progress\n")
        sweep = self.monitor.sweep(now=T0 + 12)
        self.assertEqual(sweep["stalled"], [])
        task = self.task("a")
        self.assertEqual(task["last_progress_at"], T0 + 12)
        self.assertFalse(task["stalled"])
        # Silence again past the horizon: a second stall lands.
        sweep = self.monitor.sweep(now=T0 + 20)
        self.assertEqual(sweep["stalled"], ["a"])
        self.assertEqual(len(self.events("stalled", "a")), 2)

    def test_dead_claims_staging_is_quarantined_not_reused(self):
        self.register("a", ("a_impl.py",))
        got = self.claim("a", "cursor-a")
        self.edit("a", "cursor-a", "ghost.txt", "dead work\n")
        area = self.staging.area(self.run_id, got["task_id"])
        sweep = self.monitor.sweep(now=T0 + 20000)
        self.assertEqual(sweep["expired"], ["a"])
        quarantined = self.staging.area(self.run_id, got["task_id"])
        self.assertEqual(quarantined.state, "quarantined")
        self.assertNotEqual(quarantined.path, area.path)
        # Preserved for inspection...
        self.assertEqual(
            (quarantined.path / "ghost.txt").read_text(
                encoding="utf-8"),
            "dead work\n")
        self.assertEqual(len(self.events("staging_quarantined",
                                         "a")), 1)
        # ...but never handed to the next cursor as a start.
        self.claim("a", "cursor-b", now=T0 + 20001)
        staging = self.staging_path("a", "cursor-b")
        self.assertFalse((staging / "ghost.txt").exists())


if __name__ == "__main__":
    unittest.main()
