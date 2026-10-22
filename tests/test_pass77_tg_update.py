"""Pass #77 — Telegram /update truth.

The chat layer over UpdateManager (step 9) already registered the four
kinds owner-only and dispatched them. What it did NOT do:

* an up-to-date ``/update`` still bounced the process (restart was
  requested whenever the report was ok, moved tree or not);
* the report carried no from→to revisions, so neither the owner nor
  the controller could tell "moved" from "already current";
* a second concurrent ``/update`` (UpdateManager's lock refusal) was
  rendered as a FAILED update instead of an honest "already running";
* failure reports never named the rollback state in chat.

These tests pin the fixes at the real seams: UpdateManager driven
with injected collaborators, the ChatUpdateController with crafted
reports, and PartnerRuntime.handle_control with UpdateManager stubbed
where the runtime constructs it.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

SHA_A = "a" * 40
SHA_B = "b" * 40


# ── UpdateManager report bookkeeping ───────────────────────────────────

class _FakeSnapshots:
    def __init__(self, home: str) -> None:
        self.home = home
        self.db_path = f"{home}/devon.db"
        self.restored: list[str] = []

    def create(self, label: str = "") -> SimpleNamespace:
        return SimpleNamespace(id="snap-1")

    def restore(self, snap_id: str, force: bool = False) -> None:
        self.restored.append(snap_id)


def _manager(tmp: str, *, health=(), pull_result=(True, "ok"), move=True):
    """An UpdateManager whose git/migrate seams are scripted.

    ``move=True`` simulates a pull that lands SHA_B; False simulates
    an up-to-date pull (HEAD stays at SHA_A).
    """
    from nomorals.os.update import UpdateManager

    snaps = _FakeSnapshots(tmp)
    mgr = UpdateManager(tmp, snaps, health_checks=list(health))
    state = {"head": SHA_A}
    mgr._preflight_pull = lambda: (True, "")
    mgr._git_sha = lambda: state["head"]

    def _pull():
        ok, detail = pull_result
        if ok and move:
            state["head"] = SHA_B
        return ok, detail

    mgr._git_pull = _pull
    mgr._migrate = lambda: ([], 3)
    return mgr, snaps


class TestReportRevisions(unittest.TestCase):
    def test_success_records_from_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp, health=[lambda: (True, "fine")])
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertEqual(report.from_sha, SHA_A)
            self.assertEqual(report.to_sha, SHA_B)
            data = report.to_dict()
            self.assertEqual(data["from_sha"], SHA_A)
            self.assertEqual(data["to_sha"], SHA_B)
            self.assertIn(f"{SHA_A[:12]} → {SHA_B[:12]}", report.render())

    def test_up_to_date_records_equal_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp, move=False)
            report = mgr.run()
            self.assertTrue(report.ok, report.error)
            self.assertEqual(report.from_sha, SHA_A)
            self.assertEqual(report.to_sha, SHA_A)
            self.assertIn("already up to date", report.render())

    def test_pull_failure_records_from_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp, pull_result=(False, "fatal: no remote"))
            report = mgr.run()
            self.assertFalse(report.ok)
            self.assertEqual(report.from_sha, SHA_A)
            self.assertEqual(report.to_sha, "")
            self.assertIn("git pull failed", report.error)
            self.assertFalse(report.rolled_back)

    def test_health_failure_rolls_back_and_records_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, snaps = _manager(
                tmp, health=[lambda: (False, "import exploded")])
            mgr._restore_tree = lambda sha: (True, "tree restored")
            report = mgr.run()
            self.assertFalse(report.ok)
            self.assertTrue(report.rolled_back)
            self.assertEqual(snaps.restored, ["snap-1"])
            self.assertEqual(report.from_sha, SHA_A)
            self.assertEqual(report.to_sha, SHA_B)
            self.assertEqual(report.quarantined, SHA_B)
            self.assertIn("rolled back", report.render())

    def test_lock_refusal_is_structural(self):
        # The real lock machinery: hold the per-home lock, then run() —
        # the refusal report must carry the "lock" step the chat layer
        # keys on (never a bare error string).
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = _manager(tmp)
            fd = mgr._acquire_update_lock()
            self.assertIsNotNone(fd)
            try:
                report = mgr.run()
            finally:
                mgr._release_update_lock(fd)
            self.assertFalse(report.ok)
            self.assertTrue(report.steps)
            self.assertEqual(report.steps[0]["step"], "lock")
            self.assertFalse(report.steps[0]["ok"])
            self.assertIn("already running", report.error)


# ── ChatUpdateController truth ─────────────────────────────────────────

class _FakeMission:
    def __init__(self, mid: str, name: str, terminal: bool = False) -> None:
        self.id = mid
        self.name = name
        self.goal = name
        self.terminal = terminal


class _FakeStore:
    def __init__(self, missions=()) -> None:
        self._missions = list(missions)
        self.checkpointed: list[str] = []

    def list(self, active_only: bool = True, limit: int = 200):
        return [m for m in self._missions if not m.terminal]

    def resumable(self):
        return [m for m in self._missions if not m.terminal]

    def checkpoint(self, mission, label: str = ""):
        self.checkpointed.append(mission.id)
        return SimpleNamespace(id=f"cp-{mission.id}")


def _report(**kw):
    from nomorals.os.update import UpdateReport

    rep = UpdateReport(**{k: v for k, v in kw.items()
                          if k in ("ok", "rolled_back", "pre_update_snapshot",
                                   "error", "quarantined")})
    for name, ok, detail in kw.get("steps", []):
        rep.record(name, ok, detail)
    if "from_sha" in kw:
        rep.from_sha = kw["from_sha"]
    if "to_sha" in kw:
        rep.to_sha = kw["to_sha"]
    return rep


def _moved_report():
    return _report(
        ok=True, from_sha=SHA_A, to_sha=SHA_B,
        steps=[("git_pull", True, "Fast-forward"),
               ("migrate", True, "schema v3"),
               ("health:_import", True, "nomorals 9.9 imports")])


def _uptodate_report():
    return _report(
        ok=True, from_sha=SHA_A, to_sha=SHA_A,
        steps=[("git_pull", True, "Already up to date."),
               ("health:_import", True, "nomorals 9.9 imports")])


class TestControllerTruth(unittest.TestCase):
    def _ctl(self, report, missions=()):
        from nomorals.social.chat.update_control import ChatUpdateController

        return ChatUpdateController(
            mission_store=_FakeStore(missions),
            update_runner=lambda pull: report)

    def test_update_up_to_date_does_not_restart(self):
        out = self._ctl(_uptodate_report()).execute("update")
        self.assertTrue(out["ok"])
        self.assertFalse(out["restart_requested"])
        self.assertIn("already up to date", out["message"])

    def test_deploy_up_to_date_does_not_restart(self):
        out = self._ctl(_uptodate_report()).execute("deploy")
        self.assertTrue(out["ok"])
        self.assertFalse(out["restart_requested"])

    def test_pull_up_to_date_is_plain(self):
        out = self._ctl(_uptodate_report()).execute("pull")
        self.assertTrue(out["ok"])
        self.assertFalse(out["restart_requested"])
        self.assertIn("already up to date", out["message"])

    def test_update_moved_restarts_and_reports_revisions(self):
        out = self._ctl(_moved_report()).execute("update")
        self.assertTrue(out["ok"])
        self.assertTrue(out["restart_requested"])
        self.assertIn(f"{SHA_A[:12]} → {SHA_B[:12]}", out["message"])
        self.assertIn("nomorals 9.9 imports", out["message"])

    def test_pull_moved_reports_revisions_without_restart(self):
        out = self._ctl(_moved_report()).execute("pull")
        self.assertTrue(out["ok"])
        self.assertFalse(out["restart_requested"])
        self.assertIn(f"{SHA_A[:12]} → {SHA_B[:12]}", out["message"])
        self.assertIn("next restart", out["message"])

    def test_lock_refusal_says_already_running_not_failed(self):
        report = _report(ok=False)
        report.error = ("another update is already running over this "
                        "install — refusing to run two updates at once")
        report.record("lock", False, report.error)
        out = self._ctl(report).execute("update")
        self.assertFalse(out["ok"])
        self.assertFalse(out["restart_requested"])
        self.assertIn("already running", out["message"])
        self.assertNotIn("FAILED", out["message"])

    def test_pull_failure_names_no_rollback(self):
        report = _report(
            ok=False, from_sha=SHA_A,
            steps=[("snapshot", True, "pre-update snapshot snap-1"),
                   ("git_pull", False, "fatal: no remote")])
        report.error = "git pull failed: fatal: no remote"
        out = self._ctl(report).execute("update")
        self.assertFalse(out["ok"])
        self.assertFalse(out["restart_requested"])
        self.assertIn("git pull failed", out["message"])
        self.assertIn("no rollback", out["message"])

    def test_health_failure_names_rollback(self):
        report = _report(
            ok=False, rolled_back=True, pre_update_snapshot="snap-1",
            from_sha=SHA_A, to_sha=SHA_B,
            steps=[("git_pull", True, "Fast-forward"),
                   ("health:_import", False, "import exploded"),
                   ("rollback", True, "restored snap-1")])
        report.error = "health check failed: _import: import exploded"
        out = self._ctl(report).execute("update")
        self.assertFalse(out["ok"])
        self.assertFalse(out["restart_requested"])
        self.assertIn("rolled back", out["message"])
        self.assertIn("snap-1", out["message"])

    def test_failed_rollback_is_named_as_failed(self):
        report = _report(
            ok=False, rolled_back=False, from_sha=SHA_A, to_sha=SHA_B,
            steps=[("health:_import", False, "import exploded"),
                   ("rollback", False, "disk gone")])
        report.error = "health check failed: _import: import exploded"
        out = self._ctl(report).execute("update")
        self.assertFalse(out["ok"])
        self.assertIn("ROLLBACK FAILED", out["message"])

    def test_gate_still_refuses_mid_mission(self):
        ctl = self._ctl(_moved_report(),
                        missions=[_FakeMission("m1", "build thing")])
        out = ctl.execute("update")
        self.assertFalse(out["ok"])
        self.assertIn("build thing", out["message"])
        self.assertFalse(out["restart_requested"])


# ── Catalog + gate facts ───────────────────────────────────────────────

class TestCatalogGate(unittest.TestCase):
    KINDS = ("update", "pull", "deploy", "restart")

    def test_not_in_public_catalog(self):
        from nomorals.social.chat.control import (
            PUBLIC_COMMANDS, public_menu_text)

        for kind in self.KINDS:
            self.assertNotIn(kind, PUBLIC_COMMANDS)
            self.assertNotIn(f"/{kind}", public_menu_text())

    def test_listed_on_personal_botfather_menu(self):
        from nomorals.social.chat.personal_commands import (
            personal_botfather_commands)

        names = {name for name, _desc in personal_botfather_commands()}
        for kind in self.KINDS:
            self.assertIn(kind, names)

    def test_caller_may_run_matrix(self):
        from nomorals.social.chat.control import caller_may_run

        for kind in self.KINDS:
            self.assertTrue(caller_may_run(kind, True))
            self.assertFalse(caller_may_run(kind, False))


# ── Runtime dispatch (the live surface) ────────────────────────────────

class TestRuntimeDispatch(unittest.TestCase):
    def _make_runtime(self, platform: str = "telegram-personal"):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner.runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self):
                super().__init__(media_dir="/tmp/nm-test-media")
                self.name = platform

            def run(self, handler):
                pass

            def send(self, chat, text, *, reply_to="", buttons=None, **kw):
                return SendResult(ok=True, platform=self.name, message_id="m1")

        tmp = tempfile.TemporaryDirectory(prefix="nm-test-pass77-")
        settings = load_settings(overrides={"home": tmp.name,
                                            "partner.platforms": platform})
        context = build_context(settings, with_executor=False, with_tools=False)
        adapter = FakeAdapter()
        gateway = ChatGateway({platform: adapter}, db=context.db)
        runtime = PartnerRuntime(context, gateway=gateway)
        return runtime, context, tmp

    def _msg(self, owner: bool, platform: str, text: str = "/update"):
        from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

        return ChatMessage(
            chat=ChatRef(platform=platform, chat_id="c1",
                         kind=ChatKind.DM, peer="someone"),
            incoming=True, text=text, sender="someone",
            meta={"is_owner": True} if owner else {})

    def _cleanup(self, runtime, context, tmp):
        try:
            runtime.stop()
        except Exception:
            pass
        try:
            context.close()
        except Exception:
            pass
        tmp.cleanup()

    def test_owner_personal_update_runs_reports_then_bounces(self):
        runtime, context, tmp = self._make_runtime("telegram-personal")
        order: list = []
        release = threading.Event()
        real_sleep = time.sleep

        class FakeUpdater:
            def __init__(self, repo, snapshots):
                pass

            def run(self, pull=True):
                return _moved_report()

        def fake_sleep(seconds):
            # Park ONLY the restart thread inside its delay until the
            # test has the reply in hand — the delay is the mechanism
            # that guarantees report-before-bounce, so hold it open
            # and prove nothing past it has happened yet. Every other
            # thread keeps (a capped version of) the real sleep.
            if threading.current_thread().name == "partner-restart":
                order.append(("sleep", seconds))
                release.wait(timeout=10)
            else:
                real_sleep(min(seconds, 0.05))

        def fake_execv(file, argv):
            order.append(("execv", argv[1] if len(argv) > 1 else ""))

        try:
            with mock.patch("nomorals.os.update.UpdateManager", FakeUpdater), \
                    mock.patch("os.execv", fake_execv), \
                    mock.patch("time.sleep", fake_sleep), \
                    mock.patch.object(runtime, "stop",
                                      lambda: order.append(("stop",))):
                reply = runtime.handle_control(
                    "/update", "telegram-personal:c1",
                    message=self._msg(True, "telegram-personal"))
                order.append(("reply", reply))
                # The report is the reply: revisions + health, in full.
                self.assertIn(f"{SHA_A[:12]} → {SHA_B[:12]}", reply)
                self.assertIn("nomorals 9.9 imports", reply)
                # The bounce is still parked in its 2.0s delay: the
                # reply exists, stop/execv have NOT happened yet.
                kinds = [entry[0] for entry in order]
                self.assertIn(("sleep", 2.0), order)
                self.assertNotIn("stop", kinds)
                self.assertNotIn("execv", kinds)
                # Release the delay and let the bounce complete WHILE
                # the fakes are still patched — an execv that escapes
                # the patch window would re-exec this test process.
                release.set()
                restart = next(
                    (t for t in threading.enumerate()
                     if t.name == "partner-restart"), None)
                self.assertIsNotNone(restart, "no restart thread spawned")
                restart.join(timeout=5)
                self.assertFalse(restart.is_alive(),
                                 "restart thread never reached execv")
                kinds = [entry[0] for entry in order]
                self.assertLess(kinds.index("reply"), kinds.index("execv"))
                self.assertLess(kinds.index("sleep"), kinds.index("stop"))
                self.assertLess(kinds.index("stop"), kinds.index("execv"))
        finally:
            release.set()
            self._cleanup(runtime, context, tmp)

    def test_owner_userbot_pull_up_to_date_no_bounce(self):
        runtime, context, tmp = self._make_runtime("telegram")

        class FakeUpdater:
            def __init__(self, repo, snapshots):
                pass

            def run(self, pull=True):
                return _uptodate_report()

        try:
            with mock.patch("nomorals.os.update.UpdateManager", FakeUpdater), \
                    mock.patch("os.execv",
                               side_effect=AssertionError("must not bounce")):
                reply = runtime.handle_control(
                    "/pull", "telegram:c1",
                    message=self._msg(True, "telegram", text="/pull"))
            self.assertIn("already up to date", reply)
        finally:
            self._cleanup(runtime, context, tmp)

    def test_owner_on_public_bot_refused_at_surface(self):
        runtime, context, tmp = self._make_runtime("telegram-bot")
        try:
            with mock.patch(
                    "nomorals.os.update.UpdateManager",
                    side_effect=AssertionError("must not be constructed")):
                reply = runtime.handle_control(
                    "/update", "telegram-bot:c1",
                    message=self._msg(True, "telegram-bot"))
            self.assertIn("public chat", reply)
        finally:
            self._cleanup(runtime, context, tmp)

    def test_non_owner_on_personal_refused(self):
        runtime, context, tmp = self._make_runtime("telegram-personal")
        try:
            with mock.patch(
                    "nomorals.os.update.UpdateManager",
                    side_effect=AssertionError("must not be constructed")):
                reply = runtime.handle_control(
                    "/update", "telegram-personal:c1",
                    message=self._msg(False, "telegram-personal"))
            self.assertIn("just for the owner", reply)
        finally:
            self._cleanup(runtime, context, tmp)


if __name__ == "__main__":
    unittest.main()
