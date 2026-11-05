"""Verify-then-install (Echelon Wave 8).

Tier: unit. Two halves:

* ``verify_install`` against synthetic git repos — a genuinely landed
  commit verifies; a tampered worktree file, a wrong HEAD, and an
  oversized new blob each fail their named check; a first install
  skips the previous-relative checks by name.
* the ``UpdateManager`` wiring, in the scripted-seams idiom of
  tests/test_updater_hardening.py — a failed verification undoes the
  pull and aborts; a passing one is recorded and the update proceeds;
  a no-op pull never invokes the verifier.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.os.verify_install import verify_install

_GIT = shutil.which("git")
_GIT_C = ["-c", "user.email=t@t", "-c", "user.name=t"]


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", *_GIT_C, "-C", str(cwd), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"git {args} failed: {proc.stderr}")
    return proc.stdout.strip()


def _two_commit_repo(root: Path) -> tuple[str, str]:
    _git(root, "init", "-q")
    (root / "app.py").write_text("v = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "one")
    sha_a = _git(root, "rev-parse", "HEAD")
    (root / "app.py").write_text("v = 2\n", encoding="utf-8")
    (root / "new.py").write_text("x = 'landed'\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "two")
    sha_b = _git(root, "rev-parse", "HEAD")
    return sha_a, sha_b


def _check(result: dict, name: str) -> dict:
    return next(c for c in result["checks"] if c["name"] == name)


@unittest.skipUnless(_GIT, "git not installed")
class TestVerifyInstall(unittest.TestCase):
    def test_genuinely_landed_commit_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b = _two_commit_repo(root)
            result = verify_install(root, sha_b, sha_a)
            self.assertTrue(result["ok"], result["checks"])
            for name in ("head_matches", "descendant", "objects_present",
                         "worktree_matches", "no_oversized_new_blobs"):
                self.assertTrue(_check(result, name)["ok"], name)

    def test_tampered_worktree_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b = _two_commit_repo(root)
            (root / "app.py").write_text("v = 'tampered'\n",
                                         encoding="utf-8")
            result = verify_install(root, sha_b, sha_a)
            self.assertFalse(result["ok"])
            self.assertFalse(_check(result, "worktree_matches")["ok"])

    def test_wrong_head_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b = _two_commit_repo(root)
            _git(root, "checkout", "-q", sha_a)
            result = verify_install(root, sha_b, sha_a)
            self.assertFalse(result["ok"])
            self.assertFalse(_check(result, "head_matches")["ok"])

    def test_oversized_new_blob_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha_a, sha_b = _two_commit_repo(root)
            result = verify_install(root, sha_b, sha_a,
                                    max_new_blob_bytes=4)
            self.assertFalse(result["ok"])
            self.assertFalse(
                _check(result, "no_oversized_new_blobs")["ok"])

    def test_first_install_skips_previous_relative_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sha_a, sha_b = _two_commit_repo(root)
            result = verify_install(root, sha_b, "")
            self.assertTrue(result["ok"], result["checks"])
            self.assertTrue(_check(result, "descendant")["skipped"])
            self.assertTrue(
                _check(result, "no_oversized_new_blobs")["skipped"])


# ── UpdateManager wiring (scripted seams, test_updater_hardening idiom)

SHA_A = "a" * 40
SHA_B = "b" * 40


class _FakeSnapshots:
    def __init__(self, home: str):
        self.home = home
        self.db_path = f"{home}/devon.db"

    def create(self, label: str = "") -> SimpleNamespace:
        return SimpleNamespace(id="snap-1")

    def restore(self, snap_id: str, force: bool = False) -> None:
        pass


def _manager(tmp: str, *, move: bool, verifier):
    from nomorals.os.update import UpdateManager

    mgr = UpdateManager(tmp, _FakeSnapshots(tmp), health_checks=[],
                        verifier=verifier)
    state = {"head": SHA_A}
    mgr._preflight_pull = lambda: (True, "")
    mgr._git_sha = lambda: state["head"]

    def _pull():
        if move:
            state["head"] = SHA_B
        return True, "ok"

    mgr._git_pull = _pull
    mgr._migrate = lambda: ([], 3)
    mgr._restored: list[str] = []

    def _restore(sha: str):
        mgr._restored.append(sha)
        state["head"] = sha
        return True, "restored"

    mgr._restore_tree = _restore
    return mgr


def _step(report, name: str):
    return next((s for s in report.steps if s["step"] == name), None)


class TestUpdateManagerVerifyWiring(unittest.TestCase):
    def test_failed_verification_undoes_the_pull_and_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[tuple] = []

            def verifier(repo, expected, previous, *, git_bin="git"):
                calls.append((expected, previous))
                return {"ok": False, "detail": "worktree hash mismatch"}

            mgr = _manager(tmp, move=True, verifier=verifier)
            report = mgr.run()
            self.assertFalse(report.ok)
            self.assertEqual(calls, [(SHA_B, SHA_A)])
            self.assertIn("failed install verification", report.error)
            self.assertEqual(mgr._restored, [SHA_A])
            step = _step(report, "verify_install")
            self.assertIsNotNone(step)
            self.assertFalse(step["ok"])

    def test_passing_verification_is_recorded_and_update_proceeds(
            self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            def verifier(repo, expected, previous, *, git_bin="git"):
                return {"ok": True, "detail": "install verified"}

            mgr = _manager(tmp, move=True, verifier=verifier)
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            step = _step(report, "verify_install")
            self.assertIsNotNone(step)
            self.assertTrue(step["ok"])
            self.assertEqual(mgr._restored, [])

    def test_noop_pull_never_invokes_the_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[tuple] = []

            def verifier(repo, expected, previous, *, git_bin="git"):
                calls.append((expected, previous))
                return {"ok": False, "detail": "must not be called"}

            mgr = _manager(tmp, move=False, verifier=verifier)
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertEqual(calls, [])


class TestVerifierDefaultResolution(unittest.TestCase):
    """The production construction verifies; a scripted construction
    (injected health gates, stubbed git seams) is not verified against
    a real repo it does not have — the health-gate idiom."""

    def test_production_construction_gets_the_real_verifier(self) -> None:
        from nomorals.os.update import UpdateManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = UpdateManager(tmp, _FakeSnapshots(tmp))
            self.assertIs(mgr._verifier, verify_install)

    def test_scripted_construction_gets_none_unless_explicit(self) -> None:
        from nomorals.os.update import UpdateManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = UpdateManager(tmp, _FakeSnapshots(tmp), health_checks=[])
            self.assertIsNone(mgr._verifier)
            sentinel = object()
            mgr = UpdateManager(tmp, _FakeSnapshots(tmp), health_checks=[],
                                verifier=sentinel)
            self.assertIs(mgr._verifier, sentinel)


if __name__ == "__main__":
    unittest.main()
