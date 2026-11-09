"""Batch C — recoverability (whole-system audit F06 + F07 + mission drill).

Tier: unit/integration, fully offline. Real ``BackupManager`` over real
SQLite files, and a mission-recovery drill that runs a real
``MissionRunner`` in a child process, SIGKILLs it mid-flight, and resumes
through the real startup path (``resume_all``) in a fresh process-side
runner.

F07 — snapshot identities: same-second / same-label / sanitized-label
creates must never alias one path; the manifest must agree with disk.
F06 — restore: verification happens *before* publication, into staging;
a damaged snapshot is refused, the previous target survives, and the
safety copy of an active-WAL target captures WAL content.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.core.errors import StorageError
from nomorals.storage.backup import BackupManager
from nomorals.storage.db import Database

FROZEN_EPOCH = 1798761600.0  # 2027-01-01T00:00:00Z


class _FrozenTime:
    """Drop-in for the ``time`` module reference inside storage.backup,
    with the clock nailed to one instant: every create lands in the same
    second with the same created_at."""

    def __init__(self, now: float) -> None:
        self._now = now

    def time(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def gmtime(self, t=None):  # noqa: ANN001, ANN202 - mirrors time.gmtime
        return time.gmtime(self._now if t is None else t)

    def strftime(self, fmt, t=None):  # noqa: ANN001, ANN202 - mirrors time.strftime
        return time.strftime(fmt, t if t is not None else time.gmtime(self._now))


class _BackupFixture:
    def __init__(self, *, include_blobs: bool = False) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        self.db = Database(root / "live.db")
        self.db.migrate()
        self.db.insert(
            "conversations",
            {"id": "c1", "title": "t", "created_at": 1.0, "updated_at": 1.0},
        )
        self.blobs = root / "blobs"
        self.blobs.mkdir()
        (self.blobs / "a.bin").write_bytes(b"hello-blob-a" * 100)
        (self.blobs / "sub").mkdir()
        (self.blobs / "sub" / "b.bin").write_bytes(b"hello-blob-b" * 200)
        self.manager = BackupManager(
            self.db,
            root / "backups",
            keep=3,
            include_blobs=include_blobs,
            blob_dir=self.blobs if include_blobs else None,
        )

    def cleanup(self) -> None:
        self.db.close()
        self.tmp.cleanup()


def _repair_manifest_sha(manager: BackupManager, name: str, path: Path,
                         field: str = "sha256") -> None:
    """Point a manifest entry's checksum at the file's current bytes, so a
    corruption must be caught by a deeper check than the checksum."""
    manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        if entry["name"] == name:
            entry[field] = hashlib.sha256(path.read_bytes()).hexdigest()
    manager.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ── F07: snapshot identities ─────────────────────────────────────────────

class SnapshotIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _BackupFixture()

    def tearDown(self) -> None:
        self.fx.cleanup()

    def test_same_second_same_label_creates_are_distinct(self) -> None:
        mgr = self.fx.manager
        with mock.patch("nomorals.storage.backup.time", _FrozenTime(FROZEN_EPOCH)):
            infos = [mgr.create(label="nightly") for _ in range(3)]
        self.assertEqual(len({i.path for i in infos}), 3)
        self.assertEqual(len({i.name for i in infos}), 3)
        self.assertEqual(len({i.snapshot_id for i in infos}), 3)
        entries = json.loads(mgr.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(entries), 3)
        self.assertEqual(len({e["path"] for e in entries}), 3)
        listed = {b.path: b for b in mgr.list()}
        for entry in entries:
            on_disk = Path(entry["path"])
            self.assertTrue(on_disk.is_file())
            self.assertEqual(
                entry["sha256"], hashlib.sha256(on_disk.read_bytes()).hexdigest(),
                "manifest checksum must describe the file actually on disk",
            )
            self.assertEqual(entry["snapshot_id"], listed[entry["path"]].snapshot_id)
        for info in infos:
            self.assertEqual(mgr.verify(info), [])

    def test_sanitized_label_collisions_are_distinct(self) -> None:
        mgr = self.fx.manager
        with mock.patch("nomorals.storage.backup.time", _FrozenTime(FROZEN_EPOCH)):
            infos = [mgr.create(label=label) for label in ("a b", "a/b", "a:b")]
        # All three labels sanitize to "a-b"; the snapshots must not.
        self.assertTrue(all("-a-b-" in i.name for i in infos))
        self.assertEqual(len({i.path for i in infos}), 3)
        for info in infos:
            self.assertEqual(mgr.verify(info), [])

    def test_rotation_with_identical_timestamps(self) -> None:
        mgr = self.fx.manager  # keep=3
        with mock.patch("nomorals.storage.backup.time", _FrozenTime(FROZEN_EPOCH)):
            infos = [mgr.create(label="rot") for _ in range(5)]
            removed = mgr.rotate()
        self.assertEqual(sorted(removed), sorted(i.name for i in infos[:2]))
        remaining = [b.name for b in mgr.list()]
        self.assertEqual(remaining, [i.name for i in infos[2:]])
        for info in infos[2:]:
            self.assertTrue(Path(info.path).is_file())
            self.assertEqual(mgr.verify(info), [])
        for info in infos[:2]:
            self.assertFalse(Path(info.path).exists())

    def test_simultaneous_creates_across_managers_share_no_path(self) -> None:
        directory = Path(self.fx.tmp.name) / "shared-backups"
        errors: list[BaseException] = []

        def worker() -> None:
            db = Database(":memory:")
            db.migrate()
            try:
                mgr = BackupManager(db, directory, keep=50)
                mgr.create(label="race")
                mgr.create(label="race")
            except BaseException as exc:  # noqa: BLE001 - collected + asserted
                errors.append(exc)
            finally:
                db.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(errors, [])
        probe = BackupManager(self.fx.db, directory, keep=50)
        entries = json.loads(probe.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(entries), 4, "no manifest entry may be lost to a race")
        self.assertEqual(len({e["path"] for e in entries}), 4)
        for backup in probe.list():
            self.assertEqual(probe.verify(backup), [])

    def test_verify_flags_aliased_manifest(self) -> None:
        mgr = self.fx.manager
        info = mgr.create(label="victim")
        entries = json.loads(mgr.manifest_path.read_text(encoding="utf-8"))
        forged = dict(entries[0])
        forged["sha256"] = "0" * 64  # a second entry claiming the same path
        forged["snapshot_id"] = "forged-identity"
        entries.append(forged)
        mgr.manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        problems = mgr.verify(info)
        self.assertTrue(
            any("manifest alias" in p for p in problems),
            f"expected an alias problem, got: {problems}",
        )


# ── F06: verify-into-staging restore ─────────────────────────────────────

class StagedRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _BackupFixture()

    def tearDown(self) -> None:
        self.fx.cleanup()

    def _target_with_marker(self) -> Path:
        target = self.fx.root / "target.db"
        conn = sqlite3.connect(str(target))
        conn.execute("CREATE TABLE marker (v TEXT)")
        conn.execute("INSERT INTO marker VALUES ('previous-good')")
        conn.commit()
        conn.close()
        return target

    def test_restore_refuses_checksum_mismatch_and_preserves_target(self) -> None:
        mgr = self.fx.manager
        info = mgr.create(label="good")
        target = self._target_with_marker()
        before = target.read_bytes()
        data = bytearray(Path(info.path).read_bytes())
        data[len(data) // 2] ^= 0xFF
        Path(info.path).write_bytes(bytes(data))
        with self.assertRaises(StorageError):
            mgr.restore(info, target=target, overwrite=True)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(list(self.fx.root.glob(".restore-*")), [])

    def test_restore_refuses_gzip_of_non_database(self) -> None:
        # The audit's probe, harder: the manifest checksum is repaired, so
        # only the staged integrity gate can catch the junk payload.
        mgr = self.fx.manager
        info = mgr.create(label="good")
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as fh:
            fh.write(b"definitely not a sqlite database" * 64)
        Path(info.path).write_bytes(buf.getvalue())
        _repair_manifest_sha(mgr, info.name, Path(info.path))
        target = self._target_with_marker()
        before = target.read_bytes()
        fresh = {b.name: b for b in mgr.list()}[info.name]
        with self.assertRaises(StorageError):
            mgr.restore(fresh, target=target, overwrite=True)
        self.assertEqual(target.read_bytes(), before)

    def test_restore_refuses_truncated_gzip(self) -> None:
        mgr = self.fx.manager
        info = mgr.create(label="good")
        raw = Path(info.path).read_bytes()
        Path(info.path).write_bytes(raw[: len(raw) // 2])
        _repair_manifest_sha(mgr, info.name, Path(info.path))
        target = self._target_with_marker()
        before = target.read_bytes()
        fresh = {b.name: b for b in mgr.list()}[info.name]
        with self.assertRaises(StorageError):
            mgr.restore(fresh, target=target, overwrite=True)
        self.assertEqual(target.read_bytes(), before)

    def test_failed_publication_leaves_target_untouched(self) -> None:
        mgr = self.fx.manager
        info = mgr.create(label="good")
        target = self._target_with_marker()
        before = target.read_bytes()
        with mock.patch.object(os, "replace",
                               side_effect=OSError("simulated publish failure")), \
                self.assertRaises(OSError):
            mgr.restore(info, target=target, overwrite=True)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(list(self.fx.root.glob(".restore-*")), [])

    def test_restore_over_active_wal_target(self) -> None:
        mgr = self.fx.manager
        info = mgr.create(label="rescuer")
        # The previous target: an ACTIVE WAL database whose committed row
        # exists only in the -wal sidecar while the connection is open.
        target = self.fx.root / "wal-target.db"
        conn = sqlite3.connect(str(target))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('wal-only-row')")
        conn.commit()
        self.assertTrue(Path(f"{target}-wal").is_file())
        try:
            restored = mgr.restore(info, target=target, overwrite=True)
        finally:
            conn.close()
        self.assertEqual(restored, target)
        # The restored file is the snapshot's database…
        check = sqlite3.connect(str(target))
        try:
            self.assertEqual(
                check.execute("SELECT COUNT(*) FROM conversations").fetchone()[0], 1)
            tables = {r[0] for r in check.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("t", tables, "old generation must not resurrect")
            self.assertEqual(
                check.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            check.close()
        # …no stale sidecars survive beside it…
        self.assertFalse(Path(f"{target}-wal").exists())
        self.assertFalse(Path(f"{target}-shm").exists())
        # …and the safety copy captured the WAL-only row (a plain file copy
        # of the main database file would not even contain table t).
        spares = list(self.fx.root.glob("wal-target.db.pre-restore-*"))
        self.assertEqual(len(spares), 1)
        spare = sqlite3.connect(str(spares[0]))
        try:
            self.assertEqual(
                spare.execute("SELECT v FROM t").fetchall(), [("wal-only-row",)])
        finally:
            spare.close()

    def test_no_overwrite_refusal_preserved(self) -> None:
        mgr = self.fx.manager
        info = mgr.create(label="good")
        target = self._target_with_marker()
        with self.assertRaises(StorageError):
            mgr.restore(info, target=target)
        conn = sqlite3.connect(str(target))
        try:
            self.assertEqual(
                conn.execute("SELECT v FROM marker").fetchall(),
                [("previous-good",)])
        finally:
            conn.close()

    def test_cli_restore_of_corrupt_snapshot_fails(self) -> None:
        from argparse import Namespace
        from types import SimpleNamespace

        from nomorals.cmdline.commands.backup import _cmd_backup

        mgr = self.fx.manager
        info = mgr.create(label="cli")
        Path(info.path).write_bytes(b"corrupted beyond recognition")
        settings = SimpleNamespace(
            backup_dir=mgr.directory,
            backup=SimpleNamespace(keep=3, compress=True, git_repo="",
                                   include_blobs=False),
            blob_dir=self.fx.blobs,
        )
        context = SimpleNamespace(db=self.fx.db, settings=settings)
        args = Namespace(verb=None, create=False, verify=False,
                         restore=info.name, push=False, json=False)
        with self.assertRaises(StorageError):
            _cmd_backup(args, context)
        default_target = Path(info.path).with_suffix("").with_suffix(".restored.db")
        if str(default_target).endswith(".gz"):
            default_target = default_target.with_suffix("")
        self.assertFalse(default_target.exists(),
                         "a refused restore must not write its target")


class StagedRestoreBlobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _BackupFixture(include_blobs=True)

    def tearDown(self) -> None:
        self.fx.cleanup()

    def test_blob_mismatch_blocks_restore(self) -> None:
        import tarfile

        mgr = self.fx.manager
        info = mgr.create(label="blobbed")
        archive = mgr.directory / info.blob_name
        # Rebuild the archive with one blob's bytes changed, then repair
        # the archive checksum — the per-blob manifest check must catch it.
        tmpdir = Path(tempfile.mkdtemp(dir=self.fx.root))
        mgr._extract_blob_archive(archive, tmpdir)  # noqa: SLF001 - test seam
        (tmpdir / "a.bin").write_bytes(b"tampered")
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(tmpdir.rglob("*")):
                if path.is_file():
                    tar.add(str(path), arcname=path.relative_to(tmpdir).as_posix())
        _repair_manifest_sha(mgr, info.name, archive, field="blob_sha256")
        target = self.fx.root / "blob-target.db"
        target.write_bytes(b"previous")
        fresh = {b.name: b for b in mgr.list()}[info.name]
        with self.assertRaises(StorageError):
            mgr.restore(fresh, target=target, overwrite=True)
        self.assertEqual(target.read_bytes(), b"previous")

    def test_restore_publishes_blobs(self) -> None:
        mgr = self.fx.manager
        info = mgr.create(label="blobbed")
        (self.fx.blobs / "a.bin").unlink()
        (self.fx.blobs / "sub" / "b.bin").unlink()
        target = self.fx.root / "fresh.db"
        restored = mgr.restore(info, target=target)
        self.assertTrue(restored.is_file())
        self.assertEqual((self.fx.blobs / "a.bin").read_bytes(), b"hello-blob-a" * 100)
        self.assertEqual(
            (self.fx.blobs / "sub" / "b.bin").read_bytes(), b"hello-blob-b" * 200)

    def test_restore_refuses_blobbed_backup_without_blob_dir(self) -> None:
        mgr = self.fx.manager
        info = mgr.create(label="blobbed")
        bare = BackupManager(self.fx.db, mgr.directory, keep=3)
        target = self.fx.root / "bare-target.db"
        with self.assertRaises(StorageError) as caught:
            bare.restore(info, target=target)
        self.assertIn("blob_dir", str(caught.exception))
        self.assertFalse(target.exists())


# ── Mission-recovery timed proof ─────────────────────────────────────────
#
# A real MissionRunner in a child process runs a six-step mission whose
# step effects are recorded in a durable side database. Step s4 blocks
# forever; the parent SIGKILLs the child mid-flight, confirms the durable
# interrupted state through a fresh connection (status running, s1–s3 in
# completed_steps, checkpoints on disk, the dead child's run claim), then
# runs the real startup recovery — wired_runner(...).resume_all() — and
# requires: mission done, every step's effect recorded exactly once
# (completed steps reconciled, never repeated), inside the time budget
# derived from the idempotency stale-claim contract (see below). This is
# the audit's "larger, separately timed mission-recovery run": it takes
# ~5 minutes by design and does not belong in a 100s sample budget.

_CHILD_SOURCE = '''
import os
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

workdir = Path(sys.argv[1])

import nomorals.agents.roles as roles_mod
from nomorals.agents.orchestrator import Plan, PlanStep
from nomorals.core.tasks import TaskKind
from nomorals.missions import wired_runner
from nomorals.storage.db import Database

STEPS = ["s1", "s2", "s3", "s4", "s5", "s6"]
BLOCK_STEP = "s4"


def record(step):
    conn = sqlite3.connect(str(workdir / "effects.db"))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS effects (step TEXT, at REAL)")
        conn.execute("INSERT INTO effects VALUES (?, ?)", (step, time.time()))
        conn.commit()
    finally:
        conn.close()


class Agent:
    def __init__(self, step):
        self.step = step

    def run(self, prompt):
        if self.step == BLOCK_STEP:
            (workdir / "blocked.txt").write_text(str(os.getpid()))
            while True:
                time.sleep(0.05)
        record(self.step)
        return SimpleNamespace(ok=True, output={"done": self.step},
                               error="", tokens=0)


def fake_build_agent(role, name="", context=None):
    return Agent(name.rsplit("-", 1)[-1])


roles_mod.build_agent = fake_build_agent


class Orchestrator:
    def plan(self, goal):
        return Plan(goal=goal, steps=[
            PlanStep(name=s, goal=f"do {s}", role="execution",
                     kind=TaskKind.IO, payload={})
            for s in STEPS
        ])


db = Database(workdir / "mission.db")
db.migrate()
ctx = SimpleNamespace(db=db, memory=None)
runner = wired_runner(ctx, milestones=False, orchestrator_factory=Orchestrator)
result = runner.start("batch C recovery drill", name="batchc-drill",
                      max_iterations=0, reflect=False)
print(f"child finished on its own: {result.status}", flush=True)
'''

_STEPS = ["s1", "s2", "s3", "s4", "s5", "s6"]
# The recovery budget is set by a production contract, not by optimism:
# the killed step's idempotency claim (missions/idempotency.py dedupe)
# is stolen by age, not by pid — ``stale_after=300s`` — because a claim
# that still looks fresh may belong to a live owner inside one long
# step. A resumed mission therefore waits out that window before the
# in-flight step may execute again; measured recovery was 300.1s. The
# budget is the stale window plus execution margin. (The mission-level
# run claim, by contrast, IS pid-aware and is stolen immediately.)
_RECOVERY_BUDGET_S = 420.0


def _effects(workdir: Path) -> list[str]:
    path = workdir / "effects.db"
    if not path.is_file():
        return []
    conn = sqlite3.connect(str(path))
    try:
        return [row[0] for row in conn.execute("SELECT step FROM effects")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


class MissionRecoveryDrillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name) / "drill"
        self.workdir.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_killed_mission_resumes_and_completes(self) -> None:
        import nomorals.agents.roles as roles_mod
        from nomorals.agents.orchestrator import Plan, PlanStep
        from nomorals.core.tasks import TaskKind
        from nomorals.missions import MissionStore, wired_runner

        workdir = self.workdir
        script = workdir / "mission_child.py"
        script.write_text(_CHILD_SOURCE, encoding="utf-8")
        tree_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tree_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [sys.executable, str(script), str(workdir)],
            env=env, cwd=str(tree_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        def parent_agent(role: str, name: str = "", context=None):  # noqa: ANN202
            step = name.rsplit("-", 1)[-1]

            class _Agent:
                def run(self, prompt):  # noqa: ANN202
                    conn = sqlite3.connect(str(workdir / "effects.db"))
                    try:
                        conn.execute(
                            "CREATE TABLE IF NOT EXISTS effects (step TEXT, at REAL)")
                        conn.execute("INSERT INTO effects VALUES (?, ?)",
                                     (step, time.time()))
                        conn.commit()
                    finally:
                        conn.close()
                    return SimpleNamespace(ok=True, output={"done": step},
                                           error="", tokens=0)

            return _Agent()

        original_build_agent = roles_mod.build_agent
        try:
            # ── phase 1: the child runs and blocks inside s4 ────────────
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    out = proc.stdout.read() if proc.stdout else ""
                    self.fail(f"mission child exited early (rc={proc.returncode}):\n{out}")
                if (workdir / "blocked.txt").is_file() and \
                        Counter(_effects(workdir)) == Counter({"s1": 1, "s2": 1, "s3": 1}):
                    break
                time.sleep(0.1)
            else:
                self.fail("mission child never reached the blocking step")

            # ── the interrupted state is durable, read fresh ────────────
            db = Database(workdir / "mission.db")
            store = MissionStore(db)
            missions = store.list()
            self.assertEqual(len(missions), 1)
            mission = missions[0]
            mission_id = mission.id
            self.assertEqual(mission.status, "running")
            self.assertEqual(
                sorted(mission.state.get("completed_steps") or []),
                ["s1", "s2", "s3"])
            self.assertGreaterEqual(len(store.checkpoint_history(mission_id)), 1)
            claim = (mission.state.get("run_claim") or {})
            self.assertEqual(claim.get("pid"), proc.pid,
                             "the run claim must name the (about to die) child")
            db.close()

            # ── kill it mid-flight ───────────────────────────────────────
            proc.kill()
            proc.wait(timeout=30)
            self.assertEqual(proc.returncode, -9)

            # ── phase 2: the real startup recovery path ─────────────────
            class _Orchestrator:
                def plan(self, goal):  # noqa: ANN202
                    return Plan(goal=goal, steps=[
                        PlanStep(name=s, goal=f"do {s}", role="execution",
                                 kind=TaskKind.IO, payload={})
                        for s in _STEPS
                    ])

            roles_mod.build_agent = parent_agent
            db2 = Database(workdir / "mission.db")
            try:
                runner2 = wired_runner(
                    SimpleNamespace(db=db2, memory=None),
                    milestones=False, orchestrator_factory=_Orchestrator)
                started = time.monotonic()
                results = runner2.resume_all()
                elapsed = time.monotonic() - started
            finally:
                roles_mod.build_agent = original_build_agent

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok, f"resume result: {results[0]}")
            final = MissionStore(db2).get(mission_id)
            self.assertEqual(final.status, "done")
            self.assertTrue(final.success)
            self.assertEqual(
                Counter(_effects(workdir)), Counter({s: 1 for s in _STEPS}),
                "every step effect exactly once: completed steps reconciled, "
                "never repeated; the killed step and its successors ran once")
            self.assertLess(
                elapsed, _RECOVERY_BUDGET_S,
                f"recovery took {elapsed:.1f}s (budget {_RECOVERY_BUDGET_S:.0f}s)")
            print(f"mission recovery drill: resumed + completed in {elapsed:.2f}s")
            db2.close()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
