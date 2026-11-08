"""Regression pins for the 2026-11-08 /update audit fixes.

Four defects, each proven against the pre-fix code first:

1. The health gate's layering check imported ``test_layering``
   in-process, so a FAILED update (which never restarts the
   process) left the failed tree's module cached in
   ``sys.modules`` — a retry then judged the new tree by the OLD
   tree's grandfathered baseline and scanner. The fd6c871 class:
   the gate itself can roll back the fix.
2. The pre-pull collision guard only looked at TRACKED files. An
   untracked file sitting at a path the incoming commit CREATES
   was invisible to it, so the pull died mid-checkout ("untracked
   working tree file would be overwritten") instead of the guard
   refusing up front, the way it does for a modified tracked file.
3. A deferred database restore whose snapshot had been DELETED
   (the owner hand-deleted snapshots during the 2026-11-08 disk
   saga) was treated as a transient failure: the marker stayed,
   and every boot published a "restore FAILED" notice, forever,
   with no path to clear it short of a successful /update.
4. A successful /update restarted the process on the dying
   process's say-so: "restarting now" from the process about to
   die, then silence whether the new process came up on the new
   code or never came up at all. The restart journey needs the
   NEW process to confirm, correlated by the version landed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

_GIT_C = ["-c", "user.email=t@example.com", "-c", "user.name=t",
          "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main"]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *_GIT_C, "-C", str(cwd), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"git {args} failed: {proc.stderr}")
    return proc


def _sha(cwd: Path) -> str:
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


def _manager(repo: Path, home: Path):
    from nomorals.os.snapshots import SnapshotManager
    from nomorals.os.update import UpdateManager

    snaps = SnapshotManager(home, home / "db.sqlite")
    return UpdateManager(repo, snaps, health_checks=[])


# ── Fix 1: the layering gate judges the tree on disk ─────────────────

_DISK_LAYERING = '''
KNOWN_VIOLATIONS = frozenset({("nomorals.v", "nomorals.core")})


def scan_layering():
    return ([], ["nomorals/v.py:1 (L0) imports nomorals.core (L1)"],
            set(KNOWN_VIOLATIONS), {"nomorals.v"})
'''


class TestLayeringGateUsesTreeBaseline(unittest.TestCase):
    """The retry-after-failure shape: process cache holds the FAILED
    tree's test_layering (baseline WITHOUT the pair, scanner that
    reports the pair as new); the tree on disk — the retry's pull —
    grandfathered the pair. The gate must judge the tree on disk."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.mkdtemp(dir=None)
        self.addCleanup(self._cleanup)
        self._saved_module = sys.modules.get("test_layering")
        self._saved_path = list(sys.path)
        root = Path(self._tmp)
        self.repo = root / "repo"
        (self.repo / "tests").mkdir(parents=True)
        (self.repo / "nomorals").mkdir()
        (self.repo / "tests" / "test_layering.py").write_text(
            _DISK_LAYERING, encoding="utf-8")

    def _cleanup(self) -> None:
        import shutil

        if self._saved_module is None:
            sys.modules.pop("test_layering", None)
        else:
            sys.modules["test_layering"] = self._saved_module
        sys.path[:] = self._saved_path
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _layering_check(self):
        from nomorals.os.update import default_health_checks

        checks = default_health_checks(repo_dir=str(self.repo))
        return next(c for c in checks if c.__name__ == "_layering")

    def _plant_stale_module(self, baseline, new_lines) -> None:
        stale = types.ModuleType("test_layering")
        # Same file path as the tree on disk — the poison is the
        # module OBJECT the earlier (failed) tree loaded, which is
        # exactly what an in-process import returns on a retry.
        stale.__file__ = str(self.repo / "tests" / "test_layering.py")
        stale.KNOWN_VIOLATIONS = baseline
        stale.scan_layering = lambda: (list(new_lines), [],
                                       set(baseline), {"nomorals.v"})
        sys.modules["test_layering"] = stale

    def test_stale_cached_baseline_must_not_fail_the_retry(self):
        self._plant_stale_module(
            frozenset(),
            ["nomorals/v.py:1 (L0) imports nomorals.core (L1)"])
        ok, detail = self._layering_check()()
        self.assertTrue(
            ok, "gate judged the retry tree by the FAILED tree's "
                f"cached baseline instead of the tree on disk: {detail}")
        self.assertIn("0 new", detail)

    def test_no_cache_loads_the_tree_module(self):
        sys.modules.pop("test_layering", None)
        ok, detail = self._layering_check()()
        self.assertTrue(ok, detail)

    def test_matching_baseline_keeps_the_injected_module_seam(self):
        # The layering pins inject a module whose baseline equals
        # the tree's; the gate must honor THAT module's scanner
        # (the seam tests/test_update_health_layering.py relies on).
        self._plant_stale_module(
            frozenset({("nomorals.v", "nomorals.core")}),
            ["nomorals/w.py:9 (L0) imports nomorals.core (L1)"])
        ok, detail = self._layering_check()()
        self.assertFalse(ok, "injected scanner's new violation "
                             f"was ignored: {detail}")
        self.assertIn("1 new violation", detail)


# ── Fix 2: preflight refuses untracked collisions up front ───────────


class TestPreflightUntrackedCollision(unittest.TestCase):

    def _world(self, tmp: Path):
        remote = tmp / "remote.git"
        subprocess.run(["git", *_GIT_C, "init", "--bare", str(remote)],
                       capture_output=True, check=True)
        seed = tmp / "seed"
        subprocess.run(["git", *_GIT_C, "clone", str(remote), str(seed)],
                       capture_output=True, check=True)
        (seed / "code.py").write_text("code-v1\n")
        _git(seed, "add", "-A")
        _git(seed, "commit", "-m", "v1")
        _git(seed, "push", "origin", "HEAD:main")
        install = tmp / "install"
        subprocess.run(["git", *_GIT_C, "clone", "-b", "main",
                        str(remote), str(install)],
                       capture_output=True, check=True)
        (seed / "newfile.txt").write_text("new-v2\n")
        _git(seed, "add", "-A")
        _git(seed, "commit", "-m", "v2")
        _git(seed, "push", "origin", "HEAD:main")
        return install

    def test_untracked_file_at_an_incoming_path_refuses_up_front(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            install = self._world(Path(td))
            (install / "newfile.txt").write_text("owner's own file\n")
            mgr = _manager(install, Path(td) / "home")
            proceed, detail = mgr._preflight_pull()
            self.assertFalse(
                proceed, "preflight waved through an untracked file "
                         "the pull would die on mid-checkout")
            self.assertIn("newfile.txt", detail)
            self.assertIn("untracked", detail)
            # Nothing was pulled and the owner's file is untouched.
            self.assertEqual((install / "newfile.txt").read_text(),
                             "owner's own file\n")

    def test_disjoint_untracked_file_still_proceeds(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            install = self._world(Path(td))
            (install / "notes.txt").write_text("owner notes\n")
            mgr = _manager(install, Path(td) / "home")
            proceed, detail = mgr._preflight_pull()
            self.assertTrue(proceed, detail)


# ── Fix 3: a deferred restore whose snapshot is gone is discarded ────


class TestPendingRestoreMissingSnapshot(unittest.TestCase):

    def _manager_with_marker(self, td: str):
        home = Path(td) / "home"
        home.mkdir(parents=True)
        repo = Path(td) / "not-a-repo"
        repo.mkdir()
        mgr = _manager(repo, home)
        mgr.pending_restore_path.write_text(json.dumps({
            "snapshot": "20990101-000000-deadbeef",
            "run_id": "run-x",
            "target_sha": "t" * 40,
            "from_sha": "f" * 40,
            "ts": 1.0,
        }), encoding="utf-8")
        return mgr

    def test_missing_snapshot_discards_the_marker(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mgr = self._manager_with_marker(td)
            res = mgr.apply_pending_restore()
            self.assertTrue(
                res.get("discarded_snapshot_missing"),
                f"missing snapshot treated as a retryable failure: {res}")
            self.assertNotIn("error", res)
            self.assertFalse(
                mgr.pending_restore_path.exists(),
                "marker kept — every boot would re-report the failure")

    def test_boot_receipt_names_the_unfulfillable_restore(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mgr = self._manager_with_marker(td)
            receipt = mgr.recover_interrupted()
            self.assertTrue(receipt.get("recovered"), receipt)
            self.assertEqual(receipt.get("action"),
                             "pending_restore_unfulfillable", receipt)


# ── Fix 4: the new process confirms the restart ──────────────────────


class TestRestartIntent(unittest.TestCase):

    def _repo(self, tmp: Path) -> tuple[Path, str, str]:
        repo = tmp / "repo"
        repo.mkdir()
        _git(repo, "init")
        (repo / "a.txt").write_text("v1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "v1")
        s1 = _sha(repo)
        (repo / "a.txt").write_text("v2\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "v2")
        return repo, s1, _sha(repo)

    def _mgr(self, repo: Path, tmp: Path):
        home = tmp / "home"
        home.mkdir(exist_ok=True)
        snaps = SimpleNamespace(home=home, db_path=home / "db.sqlite")
        from nomorals.os.update import UpdateManager

        return UpdateManager(repo, snaps, health_checks=[])

    def _report(self, from_sha: str, to_sha: str):
        from nomorals.os.update import UpdateReport

        return UpdateReport(ok=True, from_sha=from_sha, to_sha=to_sha)

    def test_confirmed_roundtrip_is_one_shot(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo, s1, s2 = self._repo(tmp)
            mgr = self._mgr(repo, tmp)
            self.assertTrue(mgr.note_restart_intent(self._report(s1, s2)))
            receipt = mgr.consume_restart_intent()
            self.assertIsNotNone(receipt)
            self.assertEqual(receipt["action"], "restart_confirmed")
            self.assertEqual(receipt["running_sha"], s2)
            self.assertIsNone(mgr.consume_restart_intent(),
                              "a later boot must not re-report the "
                              "same restart")

    def test_unconfirmed_when_head_is_not_the_landed_version(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo, s1, s2 = self._repo(tmp)
            mgr = self._mgr(repo, tmp)
            self.assertTrue(mgr.note_restart_intent(self._report(s1, s2)))
            _git(repo, "checkout", "-q", s1)
            receipt = mgr.consume_restart_intent()
            self.assertEqual(receipt["action"], "restart_unconfirmed")
            self.assertEqual(receipt["to_sha"], s2)
            self.assertEqual(receipt["running_sha"], s1)
            self.assertIsNone(mgr.consume_restart_intent())

    def test_noop_update_writes_no_intent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo, s1, _s2 = self._repo(tmp)
            mgr = self._mgr(repo, tmp)
            self.assertFalse(mgr.note_restart_intent(self._report(s1, s1)))
            self.assertIsNone(mgr.consume_restart_intent())

    def test_no_marker_no_receipt(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo, _s1, _s2 = self._repo(tmp)
            self.assertIsNone(self._mgr(repo, tmp).consume_restart_intent())


if __name__ == "__main__":
    unittest.main()
