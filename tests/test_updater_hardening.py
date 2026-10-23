"""Updater hardening — the 2026-10-23 EC2 /update incident, pinned.

What happened on the owner's EC2 host: /update reported "already up
to date" at 91b6ac0, then ran migrate + health anyway; the health
gate included the advisory error_scan, whose 333 standing findings
(331x E103 + 2x E102 on the healthy tree) failed it; the rollback
then failed copying the live database the running Devon process held
open (PermissionError, WinError 32); and the current, WORKING version
91b6ac0 was quarantined.

The five fixes, each pinned at its real seam:

1. a no-op update (HEAD == remote HEAD) short-circuits — no migrate,
   no health gate, no rollback, no quarantine, no restart;
2. quarantine names only a version the update itself pulled; a stale
   entry naming the already-running version is voided at run start;
3. a rollback whose database restore is blocked by the live DB lock
   defers via a pending-restore marker (applied at next boot) and
   reports a NOTE — never "ROLLBACK FAILED";
4. error_scan is advisory-only: out of the health gate, counts in the
   report, never able to fail/rollback/quarantine;
5. the layering/error scans enumerate the TRACKED tree (git index),
   not a filesystem rglob, and health failure lines carry the full
   offending path + the interpreter version.
"""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_X = "c" * 40

_GIT_C = ["-c", "user.email=t@t", "-c", "user.name=t"]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *_GIT_C, "-C", str(cwd), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"git {args} failed: {proc.stderr}")
    return proc


# ── scripted seams (the pass-77 idiom) ─────────────────────────────────

class _FakeSnapshots:
    def __init__(self, home: str, restore_exc: Exception | None = None):
        self.home = home
        self.db_path = f"{home}/devon.db"
        self.restored: list[str] = []
        self._restore_exc = restore_exc

    def create(self, label: str = "") -> SimpleNamespace:
        return SimpleNamespace(id="snap-1")

    def restore(self, snap_id: str, force: bool = False) -> None:
        if self._restore_exc is not None:
            raise self._restore_exc
        self.restored.append(snap_id)


def _manager(tmp: str, *, health=(), move: bool = True, start: str = SHA_A,
             snaps: _FakeSnapshots | None = None, advisory=None):
    """An UpdateManager whose git/migrate seams are scripted.

    ``move=True`` simulates a pull landing SHA_B; False simulates an
    up-to-date pull (HEAD stays where it started).
    """
    from nomorals.os.update import UpdateManager

    snaps = snaps or _FakeSnapshots(tmp)
    kwargs: dict = {"health_checks": list(health)}
    if advisory is not None:
        kwargs["advisory_checks"] = list(advisory)
    mgr = UpdateManager(tmp, snaps, **kwargs)
    state = {"head": start}
    mgr._preflight_pull = lambda: (True, "")
    mgr._git_sha = lambda: state["head"]

    def _pull():
        if move:
            state["head"] = SHA_B
        return True, "ok"

    mgr._git_pull = _pull
    mgr._migrate = lambda: ([], 3)
    return mgr, snaps


def _step(report, name: str):
    return next((s for s in report.steps if s["step"] == name), None)


# ── Fix 1: no-op short-circuit ─────────────────────────────────────────

class TestNoOpShortCircuit(unittest.TestCase):
    def test_up_to_date_stops_before_migrate_health_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[str] = []

            def spy_health():
                calls.append("health")
                return True, "fine"

            mgr, snaps = _manager(tmp, health=[spy_health], move=False)
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertEqual(calls, [], "health gate ran on a no-op update")
            self.assertIsNotNone(_step(report, "up_to_date"))
            self.assertIsNone(_step(report, "migrate"))
            self.assertFalse(any(s["step"].startswith("health:")
                                 for s in report.steps))
            self.assertFalse(any(s["step"].startswith("advisory:")
                                 for s in report.steps))
            self.assertEqual(snaps.restored, [])
            self.assertFalse(report.rolled_back)
            self.assertEqual(report.quarantined, "")
            self.assertEqual(mgr.quarantine_list(), [])
            self.assertIn("already up to date", report.render())

    def test_ec2_incident_replay(self):
        # The full incident conditions at once: HEAD already at the
        # remote version, a health gate that WOULD fail, and a stale
        # quarantine entry naming the running version. The fixed
        # updater reports up-to-date, voids the stale entry, and the
        # failing gate is never even consulted.
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[str] = []

            def failing_spy():
                calls.append("health")
                return False, "333 error_scan errors"

            mgr, snaps = _manager(tmp, health=[failing_spy], move=False)
            mgr._quarantine_add(SHA_A, "health check failed (incident)")
            # The incident's own journal shape: the run STARTED at the
            # version it quarantined (from_sha == to_sha == SHA_A).
            mgr._journal({"run_id": "incident", "event": "step",
                          "step": "git_pull", "from_sha": SHA_A,
                          "to_sha": SHA_A, "ts": time.time()})
            mgr._journal({"run_id": "incident", "event": "rolled_back",
                          "quarantined": SHA_A, "ts": time.time()})
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertEqual(calls, [])
            self.assertIsNotNone(_step(report, "quarantine"))
            self.assertIn("voided", _step(report, "quarantine")["detail"])
            self.assertFalse(mgr._is_quarantined(SHA_A))
            self.assertEqual(report.quarantined, "")
            self.assertEqual(snaps.restored, [])

    def test_real_pull_still_runs_the_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp, health=[lambda: (True, "fine")],
                              move=True)
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertIsNotNone(_step(report, "migrate"))
            self.assertTrue(any(s["step"].startswith("health:")
                                for s in report.steps))
            self.assertIsNone(_step(report, "up_to_date"))


# ── Fix 2: quarantine only the newly-pulled version ────────────────────

class TestQuarantineOnlyNew(unittest.TestCase):
    def test_stale_entry_for_running_version_voided(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp, health=[lambda: (True, "fine")],
                              move=True)
            mgr._quarantine_add(SHA_A, "wrongly quarantined")
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            step = _step(report, "quarantine")
            self.assertIsNotNone(step)
            self.assertIn("voided", step["detail"])
            self.assertIn(SHA_A[:12], step["detail"])
            self.assertFalse(mgr._is_quarantined(SHA_A))
            self.assertEqual(report.to_sha, SHA_B)

    def test_legit_quarantine_is_never_voided(self):
        # A version an update genuinely PULLED and rolled back keeps
        # its quarantine: the journal shows from_sha != to_sha.
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp, health=[lambda: (True, "fine")],
                              move=False, start=SHA_B)
            mgr._quarantine_add(SHA_B, "failed a previous update")
            mgr._journal({"run_id": "old", "event": "step",
                          "step": "git_pull", "from_sha": SHA_X,
                          "to_sha": SHA_B, "ts": time.time()})
            mgr._journal({"run_id": "old", "event": "rolled_back",
                          "quarantined": SHA_B, "ts": time.time()})
            report = mgr.run()
            # The pre-existing behaviour stands: landing (or sitting)
            # on a legitimately quarantined version refuses, naming
            # the quarantine, and the entry is NOT voided.
            self.assertFalse(report.ok)
            self.assertIn("quarantined", report.error)
            self.assertIsNone(_step(report, "quarantine"))
            self.assertTrue(mgr._is_quarantined(SHA_B))

    def test_health_failure_quarantines_pulled_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp, health=[lambda: (False, "broken")],
                              move=True)
            mgr._restore_tree = lambda sha: (True, "tree restored")
            report = mgr.run()
            self.assertFalse(report.ok)
            self.assertEqual(report.quarantined, SHA_B)
            self.assertTrue(mgr._is_quarantined(SHA_B))
            self.assertFalse(mgr._is_quarantined(SHA_A))

    def test_rollback_guard_refuses_to_quarantine_pre_update_sha(self):
        # The guard at its own seam: even if a caller hands _rollback
        # the pre-update sha, it is never quarantined.
        with tempfile.TemporaryDirectory() as tmp:
            from nomorals.os.update import UpdateReport

            mgr, _ = _manager(tmp)
            mgr._tx_from_sha = SHA_A
            mgr._restore_tree = lambda sha: (True, "tree restored")
            report = UpdateReport(pre_update_snapshot="snap-1")
            out = mgr._rollback(report, time.time(), "boom",
                                quarantine_sha=SHA_A)
            self.assertEqual(out.quarantined, "")
            self.assertEqual(mgr.quarantine_list(), [])
            out2 = mgr._rollback(UpdateReport(pre_update_snapshot="s"),
                                 time.time(), "boom", quarantine_sha=SHA_B)
            self.assertEqual(out2.quarantined, SHA_B)
            self.assertTrue(mgr._is_quarantined(SHA_B))


# ── Fix 3: locked-DB rollback defers, never "fails" ────────────────────

class TestLockedDbRollbackDefers(unittest.TestCase):
    def _locked_manager(self, tmp: str):
        exc = PermissionError("the database file is locked")
        exc.winerror = 32  # WinError 32: file in use by another process
        snaps = _FakeSnapshots(tmp, restore_exc=exc)
        mgr, snaps = _manager(
            tmp, health=[lambda: (False, "v2 is broken")], move=True,
            snaps=snaps)
        mgr._restore_tree = lambda sha: (True, "tree restored")
        return mgr, snaps

    def test_locked_db_rollback_defers_with_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = self._locked_manager(tmp)
            report = mgr.run()
            self.assertFalse(report.ok)  # the UPDATE failed…
            # …but the rollback did not: code restored, DB deferred.
            self.assertTrue(report.rolled_back)
            self.assertIn("PermissionError", report.db_restore_deferred)
            bad = [s for s in report.steps
                   if s["step"] == "rollback" and not s["ok"]]
            self.assertEqual(bad, [], "rollback rendered as FAILED")
            note = _step(report, "rollback_db")
            self.assertIsNotNone(note)
            self.assertTrue(note["ok"])
            self.assertIn("deferred", note["detail"])
            self.assertIn("database restore deferred", report.render())
            # The pulled (broken) version is still quarantined; the
            # deferral changes nothing about quarantine truth.
            self.assertEqual(report.quarantined, SHA_B)
            # The pending-restore marker names the snapshot.
            import json
            marker = json.loads(
                Path(mgr.pending_restore_path).read_text(encoding="utf-8"))
            self.assertEqual(marker["snapshot"], "snap-1")

    def test_pending_restore_applies_at_boot(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, snaps = _manager(tmp)
            mgr._write_pending_restore("snap-1", "PermissionError: locked")
            receipt = mgr.recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "pending_restore_applied")
            self.assertEqual(receipt["restored_snapshot"], "snap-1")
            self.assertEqual(snaps.restored, ["snap-1"])
            self.assertFalse(Path(mgr.pending_restore_path).exists())

    def test_pending_restore_failure_keeps_marker_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            snaps = _FakeSnapshots(
                tmp, restore_exc=RuntimeError("still broken"))
            mgr, _ = _manager(tmp, snaps=snaps)
            mgr._write_pending_restore("snap-1", "PermissionError: locked")
            receipt = mgr.recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "pending_restore_failed")
            self.assertIn("still broken", receipt["error"])
            self.assertTrue(Path(mgr.pending_restore_path).exists())

    def test_non_lock_restore_failure_still_fails_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            snaps = _FakeSnapshots(
                tmp, restore_exc=RuntimeError("disk gone"))
            mgr, _ = _manager(
                tmp, health=[lambda: (False, "broken")], move=True,
                snaps=snaps)
            mgr._restore_tree = lambda sha: (True, "tree restored")
            report = mgr.run()
            self.assertFalse(report.ok)
            self.assertFalse(report.rolled_back)
            self.assertEqual(report.db_restore_deferred, "")
            bad = _step(report, "rollback")
            self.assertIsNotNone(bad)
            self.assertFalse(bad["ok"])
            self.assertFalse(Path(mgr.pending_restore_path).exists())

    def test_chat_line_names_deferral_not_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            from nomorals.os.update import UpdateReport
            from nomorals.social.chat.update_control import (
                ChatUpdateController)

            report = UpdateReport(
                ok=False, rolled_back=True, pre_update_snapshot="snap-1",
                db_restore_deferred="PermissionError: file locked")
            report.from_sha, report.to_sha = SHA_A, SHA_B
            report.record("health:_import", False, "import exploded")
            ctl = ChatUpdateController(
                mission_store=_EmptyStore(),
                update_runner=lambda pull: report)
            out = ctl.execute("update")
            self.assertFalse(out["ok"])
            self.assertIn("deferred", out["message"])
            self.assertNotIn("ROLLBACK FAILED", out["message"])


class _EmptyStore:
    def list(self, active_only: bool = True, limit: int = 200):
        return []

    def resumable(self):
        return []


# ── Fix 4: error_scan is advisory only ─────────────────────────────────

class TestErrorScanAdvisory(unittest.TestCase):
    def test_gate_composition_excludes_error_scan(self):
        from nomorals.os.update import (
            default_advisory_checks, default_health_checks)

        with tempfile.TemporaryDirectory() as tmp:
            gate = [c.__name__ for c in default_health_checks(
                repo_dir=tmp, db_path=Path(tmp) / "db.sqlite")]
            self.assertEqual(gate, ["_import", "_layering", "_db"])
            advisory = [c.__name__ for c in
                        default_advisory_checks(repo_dir=tmp)]
            self.assertEqual(advisory, ["_error_scan"])

    def _repo_with_e103(self, tmp: str) -> Path:
        repo = Path(tmp) / "repo"
        pkg = repo / "nomorals"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "swallow.py").write_text(
            "def f():\n"
            "    try:\n"
            "        risky()\n"
            "    except Exception:\n"
            "        pass\n")
        _git(repo, "init", "-q")
        _git(repo, "add", "-A")
        return repo

    def test_advisory_reports_counts_and_cannot_gate(self):
        from nomorals.os.update import default_advisory_checks

        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repo_with_e103(tmp)
            checks = default_advisory_checks(repo_dir=repo)
            ok, detail = checks[0]()
            self.assertTrue(ok)
            self.assertIn("1 error(s)", detail)
            self.assertIn("advisory", detail)

            # And through a full update run: findings present, the
            # update still succeeds, nothing rolls back/quarantines.
            mgr, snaps = _manager(tmp, health=[lambda: (True, "fine")],
                                  advisory=checks)
            report = mgr.run(pull=False)
            self.assertTrue(report.ok, report.error)
            step = _step(report, "advisory:_error_scan")
            self.assertIsNotNone(step)
            self.assertTrue(step["ok"])
            self.assertIn("1 error(s)", step["detail"])
            self.assertEqual(snaps.restored, [])
            self.assertEqual(report.quarantined, "")

    def test_advisory_injection_cannot_fail_an_update(self):
        # Even a hostile advisory callable (returns not-ok / raises)
        # is recorded as information with ok=True.
        with tempfile.TemporaryDirectory() as tmp:
            def hostile():
                return False, "advisory says no"

            def raiser():
                raise RuntimeError("advisory exploded")

            mgr, _ = _manager(tmp, health=[lambda: (True, "fine")],
                              advisory=[hostile, raiser])
            report = mgr.run(pull=False)
            self.assertTrue(report.ok, report.error)
            advisory_steps = [s for s in report.steps
                              if s["step"].startswith("advisory:")]
            self.assertEqual(len(advisory_steps), 2)
            self.assertTrue(all(s["ok"] for s in advisory_steps))


# ── Fix 5: tracked-tree scoping + diagnosable failure lines ────────────

class TestTrackedTreeScoping(unittest.TestCase):
    def test_error_scan_tracked_only_ignores_strays(self):
        from nomorals.tools.error_scan import scan

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            pkg = repo / "nomorals"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("")
            (pkg / "tracked_bad.py").write_text(
                "def f():\n"
                "    try:\n"
                "        risky()\n"
                "    except Exception:\n"
                "        pass\n")
            _git(repo, "init", "-q")
            _git(repo, "add", "-A")
            # Untracked stray: worse findings, must be invisible to a
            # tracked-tree scan.
            (pkg / "stray.py").write_text(
                "def g():\n"
                "    try:\n"
                "        risky()\n"
                "    except:\n"
                "        pass\n")
            tracked = scan([str(pkg)], tracked_only=True)
            tracked_files = {Path(f.file).name for f in tracked.findings}
            self.assertEqual(tracked_files, {"tracked_bad.py"})
            self.assertEqual(tracked.files_scanned, 2)
            full = scan([str(pkg)])
            full_files = {Path(f.file).name for f in full.findings}
            self.assertEqual(full_files, {"tracked_bad.py", "stray.py"})

    def _layering_tree(self, tmp: str) -> Path:
        """A minimal real tree: the suite's own test_layering.py over
        a tiny tracked package."""
        repo = Path(tmp) / "repo"
        (repo / "tests").mkdir(parents=True)
        (repo / "nomorals" / "core").mkdir(parents=True)
        real = Path(__file__).resolve().parent / "test_layering.py"
        (repo / "tests" / "test_layering.py").write_text(
            real.read_text(encoding="utf-8"), encoding="utf-8")
        (repo / "nomorals" / "__init__.py").write_text("")
        (repo / "nomorals" / "core" / "__init__.py").write_text("")
        (repo / "nomorals" / "core" / "ok.py").write_text("VALUE = 1\n")
        _git(repo, "init", "-q")
        _git(repo, "add", "-A")
        return repo

    def _load_layering_copy(self, repo: Path):
        spec = importlib.util.spec_from_file_location(
            "test_layering_copied", repo / "tests" / "test_layering.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_layering_ignores_untracked_stray_until_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._layering_tree(tmp)
            stray = repo / "nomorals" / "core" / "build.py"
            stray.write_text("def broken(:\n")  # syntax error, untracked
            mod = self._load_layering_copy(repo)
            names = {p.name for p in mod.iter_modules()}
            self.assertNotIn("build.py", names)
            case = mod.TestLayering("test_no_upward_imports")
            case.test_no_upward_imports()  # must not raise
            # Once the file is IN the index, the gate sees it — the
            # scoping is by tracked set, not by ignoring errors.
            _git(repo, "add", "nomorals/core/build.py")
            names = {p.name for p in mod.iter_modules()}
            self.assertIn("build.py", names)
            with self.assertRaises(SyntaxError):
                case.test_no_upward_imports()

    def test_layering_failure_line_has_full_path_and_interpreter(self):
        from nomorals.os.update import default_health_checks

        with tempfile.TemporaryDirectory() as tmp:
            repo = self._layering_tree(tmp)
            bad = repo / "nomorals" / "core" / "build.py"
            bad.write_text('x = "unterminated\n')
            _git(repo, "add", "-A")  # tracked: the gate must catch it
            checks = default_health_checks(repo_dir=repo)
            layering = next(c for c in checks
                            if c.__name__ == "_layering")
            # The health closure imports test_layering by name; make
            # sure it resolves to THIS tree's copy, not a cached one.
            saved = sys.modules.pop("test_layering", None)
            try:
                ok, detail = layering()
            finally:
                sys.modules.pop("test_layering", None)
                if saved is not None:
                    sys.modules["test_layering"] = saved
            self.assertFalse(ok)
            self.assertIn(str(bad.resolve()), detail)
            self.assertIn(f"python {platform.python_version()}", detail)


if __name__ == "__main__":
    unittest.main()
