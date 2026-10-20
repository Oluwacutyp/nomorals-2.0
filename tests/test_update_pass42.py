"""Pass 42 — self-update transaction integrity.

The UpdateManager contract (module docstring): "The system is never
left half-updated" and a failed update "restores the pre-update
snapshot".  These tests pin what that must mean for the CODE TREE, not
just the state database:

* a health/migration failure rolls the tree back to the pre-update
  commit (previously only the DB was restored — the new, broken code
  stayed in place while the report claimed a rollback);
* a pull that lands on a quarantined commit undoes the pull instead of
  leaving the tree sitting on a known-bad version;
* uncommitted owner changes are never silently destroyed: colliding
  changes refuse the update up front, disjoint changes survive an
  update AND its rollback untouched;
* a crash DURING migrations is recovered as dangerous (the migrate
  step used to be journaled only after it finished, so the most
  dangerous crash point was classified "died before dangerous steps");
* the import health check exercises the code in the tree (fresh
  interpreter), not the already-imported package of the process
  running the update;
* two updates cannot run concurrently over one home.

Everything runs against temp dirs and a local bare remote — never the
real remote or a live install.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nomorals.os.snapshots import SnapshotManager
from nomorals.os.update import UpdateManager, default_health_checks

_GIT_C = ["-c", "user.email=t@t", "-c", "user.name=t"]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *_GIT_C, "-C", str(cwd), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"git {args} failed: {proc.stderr}")
    return proc


def _sha(cwd: Path) -> str:
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


class _World:
    """A bare remote, a seed clone, and an 'installed' clone at v1."""

    def __init__(self, tmp: str) -> None:
        self.tmp = Path(tmp)
        self.remote = self.tmp / "remote.git"
        subprocess.run(["git", *_GIT_C, "init", "--bare", str(self.remote)],
                       capture_output=True, check=True)
        self.seed = self.tmp / "seed"
        subprocess.run(["git", *_GIT_C, "clone", str(self.remote),
                        str(self.seed)], capture_output=True, check=True)
        (self.seed / "marker.txt").write_text("v1\n")
        (self.seed / "other.txt").write_text("other-v1\n")
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
        snaps = SnapshotManager(self.tmp / "home", self.tmp / "db.sqlite")
        return UpdateManager(self.install, snaps, health_checks=checks)


class UpdateTransactionTests(unittest.TestCase):
    def test_health_failure_restores_code_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            v2 = w.push_v2()
            self.assertNotEqual(v2, w.v1)
            mgr = w.manager([lambda: (False, "v2 is broken")])
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertTrue(report.rolled_back)
            self.assertEqual(_sha(w.install), w.v1,
                             "rollback left the tree on the failed commit")
            self.assertEqual((w.install / "marker.txt").read_text(), "v1\n")

    def test_migration_failure_restores_code_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            w.push_v2()
            mgr = w.manager([lambda: (True, "ok")])
            with mock.patch.object(UpdateManager, "_migrate",
                                   side_effect=RuntimeError("boom")):
                report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertTrue(report.rolled_back)
            self.assertEqual(_sha(w.install), w.v1)
            self.assertEqual((w.install / "marker.txt").read_text(), "v1\n")

    def test_quarantined_pull_undoes_the_pull(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            v2 = w.push_v2()
            mgr = w.manager([lambda: (True, "ok")])
            mgr._quarantine_add(v2, "failed a previous update")
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertEqual(_sha(w.install), w.v1,
                             "tree left sitting on the quarantined commit")
            self.assertEqual((w.install / "marker.txt").read_text(), "v1\n")

    def test_disjoint_local_edit_survives_update_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            w.push_v2()
            (w.install / "other.txt").write_text("OWNER LOCAL EDIT\n")
            mgr = w.manager([lambda: (False, "v2 is broken")])
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertTrue(report.rolled_back)
            self.assertEqual(_sha(w.install), w.v1)
            self.assertEqual((w.install / "other.txt").read_text(),
                             "OWNER LOCAL EDIT\n")

    def test_colliding_local_edit_refuses_before_applying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            w.push_v2()
            (w.install / "marker.txt").write_text("OWNER LOCAL EDIT\n")
            mgr = w.manager([lambda: (True, "ok")])
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertIn("uncommitted", report.error)
            self.assertEqual(_sha(w.install), w.v1, "update was applied")
            self.assertEqual((w.install / "marker.txt").read_text(),
                             "OWNER LOCAL EDIT\n")

    def test_crash_during_migrate_rolls_back_on_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            w.push_v2()
            mgr = w.manager([lambda: (True, "ok")])
            with mock.patch.object(UpdateManager, "_migrate",
                                   side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    mgr.run(pull=True)
            # the pull landed; the process "died" mid-migration
            self.assertNotEqual(_sha(w.install), w.v1)
            receipt = w.manager([lambda: (True, "ok")]).recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "rolled_back")
            self.assertEqual(_sha(w.install), w.v1,
                             "crash recovery left the pulled code in place")

    def test_concurrent_update_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            entered = threading.Event()
            release = threading.Event()

            def blocking_check():
                entered.set()
                release.wait(timeout=30)
                return True, "ok"

            first = w.manager([blocking_check])
            outcome: dict = {}
            thread = threading.Thread(
                target=lambda: outcome.setdefault("r", first.run(pull=False)))
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=30))
                second = w.manager([lambda: (True, "ok")])
                report2 = second.run(pull=False)
                self.assertFalse(report2.ok)
                self.assertIn("already running", report2.error)
            finally:
                release.set()
                thread.join(timeout=30)
            self.assertTrue(outcome["r"].ok)

    # ── pins: the happy paths must keep working exactly as before ──
    def test_clean_update_applies_and_moves_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            v2 = w.push_v2()
            mgr = w.manager([lambda: (True, "ok")])
            report = mgr.run(pull=True)
            self.assertTrue(report.ok, report.error)
            self.assertEqual(_sha(w.install), v2)
            self.assertEqual((w.install / "marker.txt").read_text(),
                             "v2-broken\n")

    def test_disjoint_local_edit_survives_successful_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            v2 = w.push_v2()
            (w.install / "other.txt").write_text("OWNER LOCAL EDIT\n")
            mgr = w.manager([lambda: (True, "ok")])
            report = mgr.run(pull=True)
            self.assertTrue(report.ok, report.error)
            self.assertEqual(_sha(w.install), v2)
            self.assertEqual((w.install / "other.txt").read_text(),
                             "OWNER LOCAL EDIT\n")


class ImportHealthCheckTests(unittest.TestCase):
    def test_import_check_exercises_the_tree_not_this_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            pkg = repo / "nomorals"
            pkg.mkdir()
            (pkg / "__init__.py").write_text(
                "raise RuntimeError('v2 does not import')\n")
            (pkg / "version.py").write_text("__version__ = '9.9'\n")
            checks = default_health_checks(repo_dir=repo)
            ok, detail = checks[0]()
            self.assertFalse(
                ok, "import check passed against a tree that cannot import")
            self.assertIn("v2 does not import", detail)


if __name__ == "__main__":
    unittest.main()
