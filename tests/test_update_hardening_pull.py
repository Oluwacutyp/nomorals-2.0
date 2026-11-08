"""Update hardening — the 2026-11-07 EC2 locked-bridge incident, pinned.

What happened on the owner's EC2 host: /update pulled 8d9f1d7→2c51381
and died at git's checkout — ``unable to unlink old
'bridge/bin/whatsapp-bridge.exe': Invalid argument`` — because the
WhatsApp bridge process Devon herself spawned was still running and
Windows locks a running executable. Git had already written much of
the tree, so the install was left HALF-APPLIED (HEAD unmoved, files
mixed, new files untracked), while the report claimed the update
"stopped before applying changes". The owner's follow-up manual
pull then refused on the debris.

The four fixes, each pinned at its real seam:

1. Quiesce before pull: the update stops the repo-file children
   (BridgeSupervisor.stop_all_processes — terminate, wait, kill;
   launches held against supervision beats) BEFORE pulling, through
   UpdateManager's quiesce seam; a child that will not stop — or a
   survivor bridge this process does not own — aborts the update
   before the pull, naming what is still running.
2. Never half-apply: a failed pull restores the working tree to
   the exact status captured before it (status-delta restore,
   verified), the owner's disjoint edits preserved.
3. Verify after pull: the working tree must match the landed
   commit for the whole incoming change set before migrate/health
   run on it; a mismatch undoes the pull.
4. Truthful reports: the quiesce step is named; a failed restore
   is a ROLLBACK FAILED in chat, and "no rollback performed" is
   left for the paths where literally nothing was applied.

Plus crash recovery: a run that dies INSIDE the pull (the pull's
begin entry journaled, its completion never written) is recovered
by restoring the tree from the journaled pre-pull status — not by
the old "died before anything changed" verdict.

Everything runs against temp dirs and a local bare remote — never
the real remote or a live install.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SHA_A = "a" * 40
SHA_B = "b" * 40

_GIT_C = ["-c", "user.email=t@t", "-c", "user.name=t"]

EXE = "bridge/bin/whatsapp-bridge.exe"
CODE = "code.py"
OTHER = "other.txt"
DOOMED = "doomed.txt"
NEW = "newfile.txt"


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


# ── scripted seams (the updater-hardening idiom) ──────────────────────

class _FakeSnapshots:
    def __init__(self, home: str):
        self.home = home
        self.db_path = f"{home}/devon.db"
        self.restored: list[str] = []

    def create(self, label: str = "") -> SimpleNamespace:
        return SimpleNamespace(id="snap-1")

    def restore(self, snap_id: str, force: bool = False) -> None:
        self.restored.append(snap_id)


def _scripted(tmp: str, *, quiesce=None):
    """An UpdateManager whose git/migrate seams are scripted; the
    pull lands SHA_B. Returns (manager, pull_calls)."""
    from nomorals.os.update import UpdateManager

    mgr = UpdateManager(tmp, _FakeSnapshots(tmp),
                        health_checks=[lambda: (True, "fine")],
                        quiesce=quiesce)
    state = {"head": SHA_A}
    pull_calls: list[str] = []
    mgr._preflight_pull = lambda: (True, "")
    mgr._git_sha = lambda: state["head"]

    def _pull():
        pull_calls.append("pull")
        state["head"] = SHA_B
        return True, "ok"

    mgr._git_pull = _pull
    mgr._migrate = lambda: ([], 3)
    return mgr, pull_calls


# ── Fix 1 (manager seam): quiesce gates the pull ──────────────────────

class TestQuiesceGate(unittest.TestCase):
    def test_quiesce_runs_before_the_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            events: list[str] = []

            def quiesce():
                events.append("quiesce")
                return True, "stopped the WhatsApp bridge process(es)"

            mgr, pull_calls = _scripted(tmp, quiesce=quiesce)
            orig_pull = mgr._git_pull

            def pull():
                events.append("pull")
                return orig_pull()

            mgr._git_pull = pull
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertEqual(events, ["quiesce", "pull"])
            step = _step(report, "quiesce")
            self.assertIsNotNone(step)
            self.assertTrue(step["ok"])
            self.assertIn("stopped the WhatsApp bridge", step["detail"])

    def test_quiesce_refusal_aborts_before_any_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            def quiesce():
                return False, ("bridge processes still running: "
                               "whatsapp:2348012345678 (go engine) is "
                               "still running after terminate and kill")

            mgr, pull_calls = _scripted(tmp, quiesce=quiesce)
            report = mgr.run()
            self.assertFalse(report.ok)
            self.assertEqual(pull_calls, [], "pulled into a locked tree")
            step = _step(report, "quiesce")
            self.assertIsNotNone(step)
            self.assertFalse(step["ok"])
            self.assertIn("still running", report.error)
            self.assertIn("stopped before pulling", report.error)
            self.assertIsNone(_step(report, "git_pull"))
            self.assertIsNone(_step(report, "migrate"))

    def test_quiesce_exception_aborts_before_any_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            def quiesce():
                raise RuntimeError("supervisor unreachable")

            mgr, pull_calls = _scripted(tmp, quiesce=quiesce)
            report = mgr.run()
            self.assertFalse(report.ok)
            self.assertEqual(pull_calls, [])
            self.assertIn("quiesce", report.error)
            self.assertIsNone(_step(report, "git_pull"))

    def test_no_quiesce_hook_is_recorded_not_faked(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _scripted(tmp, quiesce=None)
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            step = _step(report, "quiesce")
            self.assertIsNotNone(step)
            self.assertTrue(step["ok"])
            self.assertIn("no quiesce hook attached", step["detail"])


# ── the real-git world ────────────────────────────────────────────────

class _World:
    """A bare remote, a seed clone, and an 'installed' clone at v1.

    v2 rewrites the bridge binary + code.py, adds newfile.txt and
    deletes doomed.txt — the incoming change set the incident's
    pull carried.
    """

    def __init__(self, tmp: str) -> None:
        self.tmp = Path(tmp)
        self.remote = self.tmp / "remote.git"
        subprocess.run(["git", *_GIT_C, "init", "--bare", str(self.remote)],
                       capture_output=True, check=True)
        self.seed = self.tmp / "seed"
        subprocess.run(["git", *_GIT_C, "clone", str(self.remote),
                        str(self.seed)], capture_output=True, check=True)
        (self.seed / "bridge" / "bin").mkdir(parents=True)
        (self.seed / EXE).write_text("bridge-v1\n")
        (self.seed / CODE).write_text("code-v1\n")
        (self.seed / OTHER).write_text("other-v1\n")
        (self.seed / DOOMED).write_text("doomed-v1\n")
        _git(self.seed, "add", "-A")
        _git(self.seed, "commit", "-m", "v1")
        _git(self.seed, "push", "origin", "HEAD:main")
        self.install = self.tmp / "install"
        subprocess.run(["git", *_GIT_C, "clone", "-b", "main",
                        str(self.remote), str(self.install)],
                       capture_output=True, check=True)
        self.v1 = _sha(self.install)

    def push_v2(self) -> str:
        (self.seed / EXE).write_text("bridge-v2\n")
        (self.seed / CODE).write_text("code-v2\n")
        (self.seed / NEW).write_text("new-v2\n")
        (self.seed / DOOMED).unlink()
        _git(self.seed, "add", "-A")
        _git(self.seed, "commit", "-m", "v2")
        _git(self.seed, "push", "origin", "HEAD:main")
        return _sha(self.seed)

    def manager(self, checks, **kwargs):
        from nomorals.os.snapshots import SnapshotManager
        from nomorals.os.update import UpdateManager

        snaps = SnapshotManager(self.tmp / "home", self.tmp / "db.sqlite")
        return UpdateManager(self.install, snaps, health_checks=checks,
                             **kwargs)

    def status(self) -> str:
        return _git(self.install, "status", "--porcelain",
                    "--untracked-files=all").stdout


# ── Fix 2: a failed pull never leaves a half-applied tree ─────────────

class TestFailedPullRestoresTree(unittest.TestCase):
    def test_half_applied_pull_is_fully_undone(self):
        # The incident itself: git wrote most of the tree, then died
        # on the locked bridge binary. HEAD never moved.
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            w.push_v2()
            (w.install / OTHER).write_text("OWNER LOCAL EDIT\n")
            pre_status = w.status()
            mgr = w.manager([lambda: (True, "ok")])

            def half_pull():
                (w.install / EXE).write_text("bridge-v2\n")
                (w.install / CODE).write_text("code-v2\n")
                (w.install / NEW).write_text("new-v2\n")
                (w.install / DOOMED).unlink()
                return False, ("error: unable to unlink old "
                               "'bridge/bin/whatsapp-bridge.exe': "
                               "Invalid argument")

            mgr._git_pull = half_pull
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            self.assertFalse(_step(report, "git_pull")["ok"])
            restore = _step(report, "git_restore")
            self.assertIsNotNone(restore,
                                 "failed pull recorded no tree restore")
            self.assertTrue(restore["ok"], restore["detail"])
            self.assertIn("Nothing was applied", report.error)
            # The tree is EXACTLY as before the attempt.
            self.assertEqual(w.status(), pre_status)
            self.assertEqual(_sha(w.install), w.v1)
            self.assertEqual((w.install / EXE).read_text(), "bridge-v1\n")
            self.assertEqual((w.install / CODE).read_text(), "code-v1\n")
            self.assertFalse((w.install / NEW).exists())
            self.assertEqual((w.install / DOOMED).read_text(), "doomed-v1\n")
            # The owner's disjoint edit survived the failure too.
            self.assertEqual((w.install / OTHER).read_text(),
                             "OWNER LOCAL EDIT\n")

    def test_clean_pull_failure_leaves_status_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            w.push_v2()
            pre_status = w.status()
            mgr = w.manager([lambda: (True, "ok")])
            mgr._git_pull = lambda: (False, "fatal: no remote")
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            restore = _step(report, "git_restore")
            self.assertIsNotNone(restore)
            self.assertTrue(restore["ok"], restore["detail"])
            self.assertEqual(w.status(), pre_status)
            self.assertEqual(_sha(w.install), w.v1)


# ── Fix 3: post-pull verification ─────────────────────────────────────

class TestPostPullVerify(unittest.TestCase):
    def test_success_path_records_quiesce_then_verify_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            v2 = w.push_v2()
            mgr = w.manager(
                [lambda: (True, "ok")],
                quiesce=lambda: (True, "stopped the bridge"))
            report = mgr.run(pull=True)
            self.assertTrue(report.ok, report.error)
            names = [s["step"] for s in report.steps]
            self.assertLess(names.index("quiesce"),
                            names.index("git_pull"))
            verify = _step(report, "verify_tree")
            self.assertIsNotNone(verify)
            self.assertTrue(verify["ok"], verify["detail"])
            self.assertEqual(_sha(w.install), v2)
            self.assertEqual((w.install / CODE).read_text(), "code-v2\n")

    def test_tree_mismatch_after_pull_undoes_the_pull(self):
        # The pull reports success but the landed tree is not the
        # pulled commit (a changed file corrupted on its way in):
        # the update must refuse and put the tree back, not migrate
        # and health-check a lie.
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            w.push_v2()
            mgr = w.manager([lambda: (True, "ok")])
            orig_pull = mgr._git_pull

            def corrupting_pull():
                ok, detail = orig_pull()
                (w.install / CODE).write_text("corrupted-after-pull\n")
                return ok, detail

            mgr._git_pull = corrupting_pull
            report = mgr.run(pull=True)
            self.assertFalse(report.ok)
            verify = _step(report, "verify_tree")
            self.assertIsNotNone(verify)
            self.assertFalse(verify["ok"])
            self.assertIn(CODE, verify["detail"])
            restore = _step(report, "git_restore")
            self.assertIsNotNone(restore)
            self.assertTrue(restore["ok"], restore["detail"])
            self.assertIn("working-tree check", report.error)
            self.assertEqual(_sha(w.install), w.v1)
            self.assertEqual((w.install / CODE).read_text(), "code-v1\n")
            self.assertIsNone(_step(report, "migrate"))


# ── Crash recovery: died inside the pull ──────────────────────────────

class TestCrashMidPullRecovery(unittest.TestCase):
    def test_crash_mid_pull_restores_tree_at_boot(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = _World(tmp)
            w.push_v2()
            (w.install / OTHER).write_text("OWNER LOCAL EDIT\n")
            pre_status = w.status()
            mgr = w.manager([lambda: (True, "ok")])

            def dying_pull():
                (w.install / EXE).write_text("bridge-v2\n")
                (w.install / NEW).write_text("new-v2\n")
                raise KeyboardInterrupt

            mgr._git_pull = dying_pull
            with self.assertRaises(KeyboardInterrupt):
                mgr.run(pull=True)
            # The pull's writes are sitting in the tree, HEAD unmoved.
            self.assertEqual((w.install / EXE).read_text(), "bridge-v2\n")
            receipt = w.manager([lambda: (True, "ok")]) \
                .recover_interrupted()
            self.assertTrue(receipt["recovered"])
            self.assertEqual(receipt["action"], "tree_restored",
                             receipt)
            self.assertEqual(w.status(), pre_status)
            self.assertEqual(_sha(w.install), w.v1)
            self.assertEqual((w.install / EXE).read_text(), "bridge-v1\n")
            self.assertFalse((w.install / NEW).exists())
            self.assertEqual((w.install / OTHER).read_text(),
                             "OWNER LOCAL EDIT\n")


# ── Fix 1 (lifecycle): the supervisor's quiesce stop ──────────────────

ACCT = "whatsapp:2348012345678"
PORT = 18931


class _FakeProc:
    """A bridge process: alive until terminate lands; stubborn
    ones ignore terminate; unkillable ones ignore kill too."""

    def __init__(self, *, stubborn: bool = False,
                 unkillable: bool = False):
        self.terminated = False
        self.killed = False
        self._poll: int | None = None
        self.stdout = iter(())
        self._stubborn = stubborn
        self._unkillable = unkillable

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        if not self._stubborn:
            self._poll = 0

    def kill(self):
        self.killed = True
        if not self._unkillable:
            self._poll = 0


def _supervisor(tmp: str, spawn, **kwargs):
    from nomorals.social.pairing import AccountRecord, BridgeSupervisor, PairingRegistry

    registry = PairingRegistry(Path(tmp) / "paired.json")
    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=Path(tmp) / "sessions", spawn=spawn,
        base_port=PORT,
        binary_resolver=lambda: "/staged/whatsapp-bridge", **kwargs)
    rec = AccountRecord(
        id=ACCT, platform="whatsapp", identity="2348012345678",
        status="linked", port=PORT,
        creds_dir=str(Path(tmp) / "sessions" / "wa-2348012345678"))
    registry.upsert(rec)
    return sup, registry


class TestSupervisorQuiesce(unittest.TestCase):
    def test_stop_all_stops_owned_process_and_keeps_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _FakeProc()
            sup, registry = _supervisor(tmp, lambda argv, env: proc)
            sup.start(ACCT)
            self.assertTrue(sup.is_running(ACCT))
            ok, detail = sup.stop_all_processes()
            self.assertTrue(ok, detail)
            self.assertTrue(proc.terminated)
            self.assertFalse(sup.is_running(ACCT))
            self.assertIn("stopped the WhatsApp bridge", detail)
            # NOT an owner stop: the record keeps its state, so the
            # ordinary reconcile — not an owner /pair start — is
            # what brings the bridge back after the update.
            self.assertEqual(registry.get(ACCT).status, "linked")

    def test_launch_hold_blocks_relaunch_until_released(self):
        from nomorals.social.pairing import PairingUnavailable

        with tempfile.TemporaryDirectory() as tmp:
            spawns: list[_FakeProc] = []

            def spawn(argv, env):
                proc = _FakeProc()
                spawns.append(proc)
                return proc

            sup, _ = _supervisor(tmp, spawn)
            sup.start(ACCT)
            ok, _ = sup.stop_all_processes()
            self.assertTrue(ok)
            with self.assertRaises(PairingUnavailable):
                sup.start(ACCT)
            serving, reason = sup.ensure_running(ACCT)
            self.assertFalse(serving)
            self.assertIn("held", reason)
            self.assertEqual(len(spawns), 1, "a beat relaunched "
                             "into the held window")
            sup.release_launch_hold()
            sup.start(ACCT)
            self.assertEqual(len(spawns), 2)
            self.assertTrue(sup.is_running(ACCT))

    def test_stubborn_process_escalates_to_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _FakeProc(stubborn=True)
            sup, _ = _supervisor(tmp, lambda argv, env: proc)
            sup.start(ACCT)
            ok, detail = sup.stop_all_processes(timeout=0.2)
            self.assertTrue(ok, detail)
            self.assertTrue(proc.terminated)
            self.assertTrue(proc.killed)

    def test_unkillable_process_is_a_named_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _FakeProc(stubborn=True, unkillable=True)
            sup, _ = _supervisor(tmp, lambda argv, env: proc)
            sup.start(ACCT)
            ok, detail = sup.stop_all_processes(timeout=0.2)
            self.assertFalse(ok)
            self.assertIn(ACCT, detail)
            self.assertIn("still running", detail)

    def test_survivor_bridge_is_a_named_blocker(self):
        # A bridge Devon did not spawn (it survived a restart and
        # was adopted by port) has no process handle: the quiesce
        # must refuse, naming it, rather than report all-clear.
        with tempfile.TemporaryDirectory() as tmp:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            try:

                sup, registry = _supervisor(
                    tmp, lambda argv, env: _FakeProc())
                rec = registry.get(ACCT)
                rec.port = port
                registry.upsert(rec)
                self.assertIsNotNone(rec)
                ok, detail = sup.stop_all_processes()
                self.assertFalse(ok)
                self.assertIn(ACCT, detail)
                self.assertIn("not a process this Devon started",
                              detail)
            finally:
                sock.close()

    def test_node_engine_process_is_stopped_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _FakeProc()
            sup, _ = _supervisor(
                tmp, lambda argv, env: proc, engine="node",
                node_resolver=lambda: "/usr/bin/node")
            sup.start(ACCT)
            self.assertTrue(sup.is_running(ACCT))
            ok, detail = sup.stop_all_processes()
            self.assertTrue(ok, detail)
            self.assertTrue(proc.terminated)
            self.assertIn("node", detail)


# ── Fix 4: the chat lines tell the truth ──────────────────────────────

class _EmptyStore:
    def list(self, active_only: bool = True, limit: int = 200):
        return []

    def resumable(self):
        return []


def _ctl_for(report):
    from nomorals.social.chat.update_control import ChatUpdateController

    return ChatUpdateController(
        mission_store=_EmptyStore(),
        update_runner=lambda pull: report)


class TestChatLines(unittest.TestCase):
    def _report(self, steps):
        from nomorals.os.update import UpdateReport

        report = UpdateReport(ok=False, from_sha=SHA_A)
        report.pre_update_snapshot = "snap-1"
        for name, ok, detail in steps:
            report.record(name, ok, detail)
        return report

    def test_failed_pull_with_restore_says_put_back(self):
        report = self._report([
            ("quiesce", True, "stopped the bridge"),
            ("git_pull", False, "unable to unlink old bridge exe"),
            ("git_restore", True, "working tree restored"),
        ])
        report.error = "git pull failed: … Nothing was applied: …"
        out = _ctl_for(report).execute("update")
        self.assertFalse(out["ok"])
        self.assertIn("put back to the pre-update commit",
                      out["message"])
        self.assertNotIn("no rollback performed", out["message"])

    def test_failed_pull_with_failed_restore_says_rollback_failed(self):
        report = self._report([
            ("git_pull", False, "unable to unlink old bridge exe"),
            ("git_restore", False, "could not restore: bridge exe"),
        ])
        report.error = "git pull failed: … could NOT be put back …"
        out = _ctl_for(report).execute("update")
        self.assertFalse(out["ok"])
        self.assertIn("ROLLBACK FAILED", out["message"])
        self.assertNotIn("no rollback performed", out["message"])

    def test_quiesce_abort_is_the_one_honest_no_rollback_case(self):
        # Nothing was applied — the update stopped before the pull —
        # so here (and only here) the old line remains the truth.
        report = self._report([
            ("quiesce", False, "bridge processes still running: …"),
        ])
        report.error = ("update stopped before pulling — bridge "
                        "processes still running")
        out = _ctl_for(report).execute("update")
        self.assertFalse(out["ok"])
        self.assertIn("stopped before pulling", out["message"])
        self.assertIn("no rollback performed", out["message"])


# ── the runtime seam itself ───────────────────────────────────────────

class TestRuntimeQuiesceSeam(unittest.TestCase):
    def _runtime(self, sup):
        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt._pair_supervisor = sup
        return rt

    def test_quiesce_delegates_to_the_supervisor(self):
        sup = SimpleNamespace(
            stop_all_processes=lambda: (False, "named blocker"),
            release_launch_hold=lambda: None)
        rt = self._runtime(sup)
        self.assertEqual(rt._quiesce_bridges_for_update(),
                         (False, "named blocker"))

    def test_quiesce_failure_to_run_is_a_refusal_not_a_pass(self):
        def boom():
            raise RuntimeError("supervisor gone")

        sup = SimpleNamespace(stop_all_processes=boom,
                              release_launch_hold=lambda: None)
        rt = self._runtime(sup)
        ok, detail = rt._quiesce_bridges_for_update()
        self.assertFalse(ok)
        self.assertIn("could not run", detail)

    def test_release_delegates(self):
        released: list[bool] = []
        sup = SimpleNamespace(
            stop_all_processes=lambda: (True, ""),
            release_launch_hold=lambda: released.append(True))
        rt = self._runtime(sup)
        rt._release_bridge_launch_hold()
        self.assertEqual(released, [True])


if __name__ == "__main__":
    unittest.main()
