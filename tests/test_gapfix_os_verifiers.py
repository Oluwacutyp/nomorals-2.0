"""Gapfix R2-38 — the OS pre-update gates actually gate an update.

``gate_registry()`` (lint + git_clean + disk_space, as the
``pre_update_gates`` all-of composite) existed with zero production
callers: an update could proceed with a dirty tree, failing lint, or
too little disk. These tests pin the wiring at its real seam —
``UpdateManager`` runs the gates as a pre-pull step (step 0.65, after
the disk preflight, before quiesce/snapshot/pull) and refuses the
update, naming the failing gate, before anything is written.

Everything runs against temp dirs and local git repos — never the real
remote or a live install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.os.update import UpdateManager
from nomorals.os.verifiers import CompositeVerifier, Verdict

_GIT_C = ["-c", "user.email=t@t", "-c", "user.name=t"]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *_GIT_C, "-C", str(cwd), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"git {args} failed: {proc.stderr}")
    return proc


def _sha(cwd: Path) -> str:
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


def _step(report, name: str):
    return next((s for s in report.steps if s["step"] == name), None)


def _step_names(report) -> list[str]:
    return [s["step"] for s in report.steps]


class _FakeSnapshots:
    """SnapshotManager stand-in in the updater-hardening idiom."""

    def __init__(self, home: str) -> None:
        self.home = home
        self.db_path = f"{home}/devon.db"
        self.blob_dir = None
        self.created: list[str] = []
        self.restored: list[str] = []
        Path(home).mkdir(parents=True, exist_ok=True)

    def create(self, label: str = "") -> SimpleNamespace:
        self.created.append(label)
        return SimpleNamespace(id=f"snap-{len(self.created)}")

    def restore(self, snap_id: str, force: bool = False) -> None:
        self.restored.append(snap_id)

    def prune_pre_update(self, protect=None) -> list[str]:
        return []


def _make_repo(base: Path, *, broken_lint: bool = False) -> Path:
    """A clean temp git repo with a valid (or broken) nomorals package."""
    repo = base / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    (repo / "file.txt").write_text("v1\n")
    pkg = repo / "nomorals"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__version__ = '1'\n")
    if broken_lint:
        (pkg / "broken.py").write_text("def broken(:\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "v1")
    return repo


class _RemoteWorld:
    """A bare remote plus an installed clone at v1.

    The dirty-tree gate tests need this shape, not a standalone repo:
    step 0.5 (``_preflight_pull``) refuses outright when local changes
    exist and no upstream can be fetched — so a repo with no remote
    never reaches the gate at all. With a real remote the fetch
    succeeds, the disjoint edit proceeds past preflight, and the gate
    is what refuses it — the EC2 shape.
    """

    def __init__(self, base: Path) -> None:
        self.base = base
        self.remote = base / "remote.git"
        subprocess.run(["git", *_GIT_C, "init", "--bare", str(self.remote)],
                       capture_output=True, check=True)
        seed = base / "seed"
        subprocess.run(["git", *_GIT_C, "clone", str(self.remote),
                        str(seed)], capture_output=True, check=True)
        (seed / "file.txt").write_text("v1\n")
        pkg = seed / "nomorals"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("__version__ = '1'\n")
        _git(seed, "add", "-A")
        _git(seed, "commit", "-m", "v1")
        _git(seed, "push", "origin", "HEAD:main")
        self.install = base / "install"
        subprocess.run(["git", *_GIT_C, "clone", "-b", "main",
                        str(self.remote), str(self.install)],
                       capture_output=True, check=True)


def _production_manager(repo: Path, home: Path) -> UpdateManager:
    """The production construction: no injected checks, no injected
    verifier — the defaults (including the gates) apply."""
    return UpdateManager(repo, _FakeSnapshots(str(home)))


class _FailingGate:
    """Duck-typed gate (the Verifier protocol) that always refuses."""

    name = "test_gate"

    def __init__(self, detail: str = "test_gate: refused on purpose") -> None:
        self.detail = detail

    def verify(self, target) -> Verdict:
        return Verdict(passed=False, details=self.detail)


class _PassingGate:
    """Duck-typed gate that always passes."""

    name = "test_gate"

    def verify(self, target) -> Verdict:
        return Verdict(passed=True, details="test_gate: fine")


class TestDirtyTreeRefused(unittest.TestCase):
    def test_dirty_tree_refused_before_anything_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            world = _RemoteWorld(base)
            repo = world.install
            # The owner's disjoint local edit: tracked file modified.
            # No v2 was pushed, so the incoming change set is empty
            # and preflight lets the disjoint edit through — the gate
            # is what refuses it.
            (repo / "file.txt").write_text("OWNER LOCAL EDIT\n")
            home = base / "home"
            snaps = _FakeSnapshots(str(home))
            mgr = UpdateManager(repo, snaps)
            before_sha = _sha(repo)
            # Tracked status only: the lint gate itself compiles the
            # tree, which leaves untracked __pycache__ debris behind
            # (benign on the real repo, where it is gitignored) — the
            # assertion is that no TRACKED file moved.
            before_status = _git(
                repo, "status", "--porcelain",
                "--untracked-files=no").stdout
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            gate = _step(report, "pre_update_gates")
            self.assertIsNotNone(gate, "no pre-update-gates step recorded")
            self.assertFalse(gate["ok"])
            self.assertIn("git_clean", gate["detail"])
            self.assertIn("git_clean", report.error)
            self.assertIn("working tree dirty", report.error)
            # Nothing was written: no snapshot, no pull, tree untouched.
            self.assertEqual(snaps.created, [])
            self.assertIsNone(_step(report, "snapshot"))
            self.assertIsNone(_step(report, "git_pull"))
            self.assertEqual(_sha(repo), before_sha)
            self.assertEqual(_git(repo, "status", "--porcelain",
                                  "--untracked-files=no").stdout,
                             before_status)
            self.assertEqual((repo / "file.txt").read_text(),
                             "OWNER LOCAL EDIT\n")

    def test_broken_lint_refused_naming_lint(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo(base, broken_lint=True)
            mgr = _production_manager(repo, base / "home")
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            gate = _step(report, "pre_update_gates")
            self.assertIsNotNone(gate)
            self.assertFalse(gate["ok"])
            self.assertIn("lint", gate["detail"])
            self.assertIn("lint", report.error)
            self.assertIsNone(_step(report, "git_pull"))

    def test_too_little_disk_refused_naming_disk_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo(base)
            home = base / "home"
            mgr = _production_manager(repo, home)
            _Usage = namedtuple("usage", "total used free")
            real_usage = shutil.disk_usage

            def _usage(path):
                text = str(path)
                if text == str(repo) or text.startswith(str(repo) + os.sep):
                    # The volume holding the repo is nearly full;
                    # the snapshots home (elsewhere) is fine, so the
                    # measured step-0.6 preflight still passes and
                    # only the gate's flat floor refuses.
                    return _Usage(10_000_000_000, 9_900_000_000,
                                  100_000_000)
                return real_usage(path)

            with mock.patch("shutil.disk_usage", side_effect=_usage):
                report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            gate = _step(report, "pre_update_gates")
            self.assertIsNotNone(gate)
            self.assertFalse(gate["ok"])
            self.assertIn("disk_space", gate["detail"])
            self.assertIn("disk_space", report.error)
            self.assertIsNone(_step(report, "git_pull"))


class TestAllGreenProceeds(unittest.TestCase):
    def test_clean_tree_gate_passes_and_run_continues_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo(base)
            snaps = _FakeSnapshots(str(base / "home"))
            mgr = UpdateManager(repo, snaps)
            report = mgr.run(pull=True)
            gate = _step(report, "pre_update_gates")
            self.assertIsNotNone(gate, "gate step missing on a clean run")
            self.assertTrue(gate["ok"], gate["detail"])
            names = _step_names(report)
            self.assertLess(names.index("pre_update_gates"),
                            names.index("snapshot"),
                            "gate must run before the snapshot")
            # The run continued exactly as before: snapshot taken,
            # pull attempted (and failing only for lack of a remote).
            self.assertEqual(len(snaps.created), 1)
            pull = _step(report, "git_pull")
            self.assertIsNotNone(pull)
            self.assertFalse(pull["ok"])
            self.assertNotIn("pre_update_gates", report.error)

    def test_production_default_is_the_real_gate_composite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo(base)
            mgr = _production_manager(repo, base / "home")
            gates = mgr._pre_update_gates
            self.assertIsNotNone(gates, "production default has no gates")
            self.assertEqual(getattr(gates, "name", None),
                             "pre_update_gates")


class TestNothingHalfApplied(unittest.TestCase):
    def test_gate_refusal_leaves_status_exactly_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            world = _RemoteWorld(base)
            repo = world.install
            (repo / "file.txt").write_text("OWNER LOCAL EDIT\n")
            # Tracked status only (see the note in the dirty-tree
            # test: the lint gate leaves __pycache__ debris).
            pre_status = _git(repo, "status", "--porcelain",
                              "--untracked-files=no").stdout
            pre_sha = _sha(repo)
            mgr = _production_manager(repo, base / "home")
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertIn("git_clean", report.error)
            self.assertEqual(_sha(repo), pre_sha)
            self.assertEqual(_git(repo, "status", "--porcelain",
                                  "--untracked-files=no").stdout,
                             pre_status)
            self.assertEqual((repo / "file.txt").read_text(),
                             "OWNER LOCAL EDIT\n")

    def test_post_pull_verify_failure_undoes_the_pull(self):
        # Pins the existing half of the contract: a verifier failure
        # AFTER the pull is undone by the real tree restore.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo(base)
            (repo / "file.txt").write_text("v2\n")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-m", "v2")
            v2 = _sha(repo)
            _git(repo, "reset", "--hard", "HEAD~1")
            v1 = _sha(repo)
            self.assertEqual((repo / "file.txt").read_text(), "v1\n")
            calls: list[tuple] = []

            def failing_verifier(repo_dir, new_sha, previous_sha,
                                 git_bin="git"):
                calls.append((str(repo_dir), new_sha, previous_sha))
                return {"ok": False,
                        "detail": "simulated install-verification failure"}

            mgr = UpdateManager(
                repo, _FakeSnapshots(str(base / "home")),
                health_checks=[lambda: (True, "ok")],
                verifier=failing_verifier)
            def _pull():
                _git(repo, "reset", "--hard", v2)
                return True, "pulled"

            mgr._git_pull = _pull
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], v2)
            self.assertEqual(calls[0][2], v1)
            self.assertIn("failed install verification", report.error)
            restore = _step(report, "git_restore")
            self.assertIsNotNone(restore)
            self.assertTrue(restore["ok"], restore["detail"])
            self.assertEqual(_sha(repo), v1)
            self.assertEqual((repo / "file.txt").read_text(), "v1\n")


class TestInjectedSeamsHonoured(unittest.TestCase):
    def test_injected_verifier_still_wins_and_run_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo(base)
            (repo / "file.txt").write_text("v2\n")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-m", "v2")
            v2 = _sha(repo)
            _git(repo, "reset", "--hard", "HEAD~1")
            v1 = _sha(repo)
            calls: list[tuple] = []

            def injected_verifier(repo_dir, new_sha, previous_sha,
                                  git_bin="git"):
                calls.append((str(repo_dir), new_sha, previous_sha))
                return {"ok": True, "detail": "injected verifier: ok"}

            mgr = UpdateManager(
                repo, _FakeSnapshots(str(base / "home")),
                health_checks=[lambda: (True, "ok")],
                verifier=injected_verifier)
            def _pull():
                _git(repo, "reset", "--hard", v2)
                return True, "pulled"

            mgr._git_pull = _pull
            mgr._migrate = lambda: ([], 3)
            report = mgr.run(pull=True)
            self.assertTrue(report.ok, report.error)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], v2)
            self.assertEqual(calls[0][2], v1)
            # Injected constructions get no default gates.
            self.assertIsNone(_step(report, "pre_update_gates"))
            self.assertIsNone(mgr._pre_update_gates)

    def test_explicit_gates_win_over_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo(base)
            # A failing gate refuses even on an injected construction.
            mgr = UpdateManager(
                repo, _FakeSnapshots(str(base / "home")),
                health_checks=[lambda: (True, "ok")],
                pre_update_gates=CompositeVerifier(
                    "pre_update_gates", [_FailingGate()], mode="all"))
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertIn("test_gate", report.error)
            self.assertIsNone(_step(report, "git_pull"))
            # A passing gate lets the same construction proceed.
            mgr2 = UpdateManager(
                repo, _FakeSnapshots(str(base / "home")),
                health_checks=[lambda: (True, "ok")],
                pre_update_gates=CompositeVerifier(
                    "pre_update_gates", [_PassingGate()], mode="all"))
            mgr2._git_pull = lambda: (False, "fatal: no remote")
            report2 = mgr2.run(pull=True)
            gate = _step(report2, "pre_update_gates")
            self.assertIsNotNone(gate)
            self.assertTrue(gate["ok"], gate["detail"])
            self.assertIsNotNone(_step(report2, "git_pull"))


if __name__ == "__main__":
    unittest.main()
