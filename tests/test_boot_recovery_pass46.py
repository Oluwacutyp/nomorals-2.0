"""Pass 46 — boot-time recovery of interrupted self-updates.

Pass 42 built and hardened ``UpdateManager.recover_interrupted()``:
replay the update journal, roll back (state snapshot AND code tree) a
run that died at/after a dangerous step, mark a run that died earlier
aborted, and record a ``crash_recovery`` terminal event so it never
re-fires.  But it had ZERO production callers — in real life an update
interrupted by a crash or power loss was never recovered at boot, and
Devon would start on a possibly half-migrated database paired with
new code.

These tests pin the wiring: ``PartnerRuntime.start()`` (the boot that
``scripts/run_chat_bot.py`` runs) replays the journal once, beside the
TrialFlow boot recoveries, in its own guard so a recovery failure can
never block boot; an acting receipt is reported to the owner through
the durable Notifier (the same mechanism the TrialFlow recovery beside
it uses) and logged at warning level or above; a clean boot is a
provably unchanged no-op (one journal read, no restore, no notice).

Everything runs against temp dirs and a local bare remote — never the
real remote or a live install.  The runtime is exercised through the
real boot seam (``PartnerRuntime._recover_interrupted_update``) on a
bare instance whose settings point at the temp world, the same seam
``start()`` calls.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.agents.partner.runtime as runtime_module
from nomorals.os.snapshots import SnapshotManager
from nomorals.os.update import JOURNAL_FILENAME, UpdateManager

_GIT_C = ["-c", "user.email=t@t", "-c", "user.name=t"]
_RUNTIME_LOG = "nomorals.agents.partner.runtime"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *_GIT_C, "-C", str(cwd), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"git {args} failed: {proc.stderr}")
    return proc


def _sha(cwd: Path) -> str:
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE state (k TEXT PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO state VALUES ('schema', 'v1')")
    conn.commit()
    conn.close()


def _db_schema(path: Path) -> str:
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT v FROM state WHERE k = 'schema'").fetchone()
    conn.close()
    return str(row[0]) if row else ""


def _set_db_schema(path: Path, value: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("UPDATE state SET v = ? WHERE k = 'schema'", (value,))
    conn.commit()
    conn.close()


class _World:
    """A bare remote, a seed clone, and an 'installed' clone at v1."""

    def __init__(self, tmp: str) -> None:
        self.tmp = Path(tmp)
        self.home = self.tmp / "home"
        self.db = self.tmp / "db.sqlite"
        _make_db(self.db)
        self.remote = self.tmp / "remote.git"
        subprocess.run(["git", *_GIT_C, "init", "--bare", str(self.remote)],
                       capture_output=True, check=True)
        self.seed = self.tmp / "seed"
        subprocess.run(["git", *_GIT_C, "clone", str(self.remote),
                        str(self.seed)], capture_output=True, check=True)
        (self.seed / "marker.txt").write_text("v1\n")
        _git(self.seed, "add", "-A")
        _git(self.seed, "commit", "-m", "v1")
        _git(self.seed, "push", "origin", "HEAD:main")
        self.install = self.tmp / "install"
        subprocess.run(["git", *_GIT_C, "clone", "-b", "main",
                        str(self.remote), str(self.install)],
                       capture_output=True, check=True)
        self.v1 = _sha(self.install)

    def push_v2(self) -> str:
        (self.seed / "marker.txt").write_text("v2-broken\n")
        _git(self.seed, "add", "-A")
        _git(self.seed, "commit", "-m", "v2")
        _git(self.seed, "push", "origin", "HEAD:main")
        return _sha(self.seed)

    def manager(self, checks) -> UpdateManager:
        snaps = SnapshotManager(self.home, self.db)
        return UpdateManager(self.install, snaps, health_checks=checks)

    def journal(self) -> list[dict]:
        path = self.home / JOURNAL_FILENAME
        if not path.exists():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _FakeNotifier:
    """Records publishes instead of delivering them."""

    instances: list["_FakeNotifier"] = []

    def __init__(self, context, gateway=None) -> None:
        self.published: list[dict] = []
        _FakeNotifier.instances.append(self)

    def publish(self, kind, title, body="", **kw):
        entry = {"kind": kind, "title": title, "body": body, **kw}
        self.published.append(entry)
        return {"ok": True, "delivered": True}


def _boot_runtime(world: _World):
    """A bare PartnerRuntime whose settings point at the temp world.

    Only the attributes the boot seam reads are set — the seam itself
    (construction of SnapshotManager/UpdateManager from settings +
    repo root, journal replay, owner report) is the real code.
    """
    from nomorals.agents.partner.runtime import PartnerRuntime

    rt = PartnerRuntime.__new__(PartnerRuntime)
    settings = SimpleNamespace(home_path=world.home, db_path=world.db,
                               blob_dir=None)
    rt.context = SimpleNamespace(settings=settings)
    rt.gateway = object()
    rt.dry_run = False
    return rt


def _published() -> list[dict]:
    return [p for inst in _FakeNotifier.instances for p in inst.published]


class BootWiringCharacterizationTests(unittest.TestCase):
    def test_start_calls_update_recovery(self) -> None:
        src = Path(runtime_module.__file__).read_text(encoding="utf-8")
        self.assertIn("recover_interrupted", src,
                      "runtime never replays the update journal")
        start = src[src.index("    def start(self)"):]
        start = start[:start.index("    def _arena_notify")]
        self.assertIn("_recover_interrupted_update()", start,
                      "start() does not invoke the update recovery seam")


class BootRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeNotifier.instances = []
        patcher = mock.patch("nomorals.agents.notifier.Notifier",
                             _FakeNotifier)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _crash_mid_migration(self, w: _World) -> None:
        """A real update that pulls v2 then 'dies' inside _migrate."""
        w.push_v2()
        mgr = w.manager([lambda: (True, "ok")])
        with mock.patch.object(UpdateManager, "_migrate",
                               side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                mgr.run(pull=True)
        self.assertNotEqual(_sha(w.install), w.v1)
        # the half-applied schema the dead process left behind
        _set_db_schema(w.db, "v2-half")

    def test_boot_rolls_back_mid_migration_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            self._crash_mid_migration(w)
            rt = _boot_runtime(w)
            receipt = rt._recover_interrupted_update(repo_root=w.install)
            self.assertIsNotNone(receipt)
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "rolled_back")
            self.assertTrue(receipt["code_restored"])
            self.assertEqual(_sha(w.install), w.v1,
                             "boot recovery left the pulled code in place")
            self.assertEqual((w.install / "marker.txt").read_text(), "v1\n")
            self.assertEqual(_db_schema(w.db), "v1",
                             "boot recovery did not restore the database")
            published = _published()
            self.assertEqual(len(published), 1,
                             "acting recovery must notify the owner once")
            self.assertEqual(published[0]["kind"], "update")
            self.assertTrue(published[0].get("force"))
            self.assertIn("rolled back",
                          published[0]["title"] + published[0]["body"])

    def test_second_boot_after_recovery_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            self._crash_mid_migration(w)
            rt = _boot_runtime(w)
            first = rt._recover_interrupted_update(repo_root=w.install)
            self.assertTrue(first["recovered"])
            second = rt._recover_interrupted_update(repo_root=w.install)
            self.assertFalse(second["recovered"],
                             "crash_recovery terminal event must close "
                             "the run — a second boot re-rolled back")
            self.assertEqual(len(_published()), 1,
                             "second boot must not re-notify")
            self.assertEqual(_sha(w.install), w.v1)

    def test_boot_with_clean_journal_is_silent_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            v2 = w.push_v2()
            report = w.manager([lambda: (True, "ok")]).run(pull=True)
            self.assertTrue(report.ok)
            rt = _boot_runtime(w)
            receipt = rt._recover_interrupted_update(repo_root=w.install)
            self.assertFalse(receipt["recovered"])
            self.assertEqual(_sha(w.install), v2,
                             "clean boot touched the tree")
            self.assertEqual(_published(), [],
                             "clean boot must not notify the owner")

    def test_boot_with_no_journal_is_silent_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            rt = _boot_runtime(w)
            receipt = rt._recover_interrupted_update(repo_root=w.install)
            self.assertFalse(receipt["recovered"])
            self.assertEqual(_published(), [])

    def test_boot_marks_pre_dangerous_death_aborted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            v2 = w.push_v2()
            # the pull landed, then the process died BEFORE migrations
            _git(w.install, "fetch", "origin")
            _git(w.install, "reset", "--hard", "origin/main")
            self.assertEqual(_sha(w.install), v2)
            snaps = SnapshotManager(w.home, w.db)
            snap = snaps.create(label="pre-update")
            entries = [
                {"run_id": "run-early", "event": "step", "step": "snapshot",
                 "snapshot": snap.id},
                {"run_id": "run-early", "event": "step", "step": "git_pull",
                 "from_sha": w.v1, "to_sha": v2},
            ]
            w.home.mkdir(parents=True, exist_ok=True)
            (w.home / JOURNAL_FILENAME).write_text(
                "".join(json.dumps(e) + "\n" for e in entries),
                encoding="utf-8")
            rt = _boot_runtime(w)
            receipt = rt._recover_interrupted_update(repo_root=w.install)
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "aborted")
            self.assertEqual(_sha(w.install), v2,
                             "aborted recovery must not restore anything")
            self.assertEqual(_db_schema(w.db), "v1")
            self.assertEqual(len(_published()), 1,
                             "an aborted close-out is still reported")
            # and the run is closed: a later boot is a no-op
            again = rt._recover_interrupted_update(repo_root=w.install)
            self.assertFalse(again["recovered"])
            self.assertEqual(len(_published()), 1)

    def test_recovery_failure_never_breaks_boot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            rt = _boot_runtime(w)
            with mock.patch.object(
                    UpdateManager, "recover_interrupted",
                    side_effect=RuntimeError("journal exploded")):
                with self.assertLogs(_RUNTIME_LOG, level="WARNING"):
                    receipt = rt._recover_interrupted_update(
                        repo_root=w.install)
            self.assertIsNone(receipt,
                              "a failed recovery must surface as None, "
                              "never as an exception out of boot")

    def test_recovery_construction_failure_never_breaks_boot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            rt = _boot_runtime(w)
            with mock.patch(
                    "nomorals.os.snapshots.SnapshotManager",
                    side_effect=RuntimeError("no home")):
                with self.assertLogs(_RUNTIME_LOG, level="WARNING"):
                    receipt = rt._recover_interrupted_update(
                        repo_root=w.install)
            self.assertIsNone(receipt)

    def test_failed_rollback_is_loud(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            # dangerous journal whose snapshot no longer exists: the
            # restore raises, the receipt is rollback_failed, and the
            # owner notice must be critical + error-logged.
            entries = [
                {"run_id": "run-bad", "event": "step", "step": "snapshot",
                 "snapshot": "snap-does-not-exist"},
                {"run_id": "run-bad", "event": "step", "step": "git_pull",
                 "from_sha": w.v1},
                {"run_id": "run-bad", "event": "step", "step": "migrate"},
            ]
            w.home.mkdir(parents=True, exist_ok=True)
            (w.home / JOURNAL_FILENAME).write_text(
                "".join(json.dumps(e) + "\n" for e in entries),
                encoding="utf-8")
            rt = _boot_runtime(w)
            with self.assertLogs(_RUNTIME_LOG, level="ERROR"):
                receipt = rt._recover_interrupted_update(repo_root=w.install)
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "rollback_failed")
            published = _published()
            self.assertEqual(len(published), 1)
            self.assertTrue(published[0].get("critical"),
                            "rollback_failed must bypass the mute gates")
            self.assertIn("FAILED", published[0]["title"])


if __name__ == "__main__":
    unittest.main()
