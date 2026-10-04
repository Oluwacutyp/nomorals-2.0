"""Wave F3 — missions: full chat control (pause/resume/cancel/retry/watch/templates).

Covers the ``_control_mission`` verbs added in F3:

- pause / resume / cancel / retry (mocked MissionRunner + threads)
- watch / unwatch persistence via MissionWatchers (kv_store)
- new <template> creation via MissionStore.create_new
- help/usage text mentions every new verb
- milestone fan-out to subscribed chats through MissionMilestones
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.missions import (
    MissionMilestones,
    MissionRunner,
    MissionStatus,
    MissionStore,
    MissionWatchers,
)
from nomorals.storage.db import Database


# ── fakes ────────────────────────────────────────────────────────────────────

class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok


class FakeGateway:
    """Records every send; owner channels report running."""

    def __init__(self, live_platforms=("telegram",)):
        self.live = set(live_platforms)
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        return {p: {"running_in_session": True} for p in self.live}

    def send(self, platform, chat_ref, text):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)


class _SyncThread:
    """threading.Thread stand-in that runs the target synchronously."""

    def __init__(self, target=None, **kw):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


def make_partner(**kw):
    base = dict(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=0,   # 0 == 0 -> quiet hours disabled (deterministic tests)
        quiet_end=0,
        timezone="UTC",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def make_ctx(test=None, partner=None, gateway=None):
    tmp = tempfile.mkdtemp(prefix="mission-control-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(workspace_dir=tmp,
                               partner=partner or make_partner())
    ctx = SimpleNamespace(db=db, settings=settings,
                          extras={"gateway": gateway} if gateway else {})
    return ctx, tmp


def make_mission(store, name="m1", goal="test goal",
                 plan_names=("a", "b", "c")):
    mission = store.create_new(goal, name=name)
    mission.state["plan"] = [
        {"name": n, "goal": f"do {n}", "role": "execution",
         "kind": "io", "depends_on": []}
        for n in plan_names
    ]
    store.save(mission)
    return mission


class _RuntimeStub:
    """Drives _control_mission without building a full PartnerRuntime."""

    def __init__(self, ctx):
        self.context = ctx
        self._control_mission = PartnerRuntime._control_mission.__get__(self)
        # _find_mission / _mission_template_spec are staticmethods: class
        # attribute access already yields the plain function — no __get__.
        self._find_mission = PartnerRuntime._find_mission
        self._mission_template_spec = PartnerRuntime._mission_template_spec
        # _resume_job/_retry_job one-shot threads call this in a finally
        # block to drop their per-thread DB connection; bind the real
        # method so the stub exercises the same release path.
        self._release_db_thread = PartnerRuntime._release_db_thread.__get__(self)


# ── verb tests ───────────────────────────────────────────────────────────────

class ControlVerbTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = make_ctx(self)
        self.store = MissionStore(self.ctx.db)
        self.rt = _RuntimeStub(self.ctx)

    def call(self, tail, chat_key="telegram:9"):
        return self.rt._control_mission(tail, chat_key=chat_key)

    # pause ──

    def test_pause_flips_status(self):
        m = make_mission(self.store)
        out = self.call(f"pause {m.id}")
        self.assertIn("paused", out)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.PAUSED)

    def test_pause_unknown_mission(self):
        out = self.call("pause nope")
        self.assertIn("no mission matching", out)

    def test_pause_terminal_is_idempotent_noop(self):
        m = make_mission(self.store)
        self.store.set_status(m.id, MissionStatus.DONE)
        out = self.call(f"pause {m.id}")
        self.assertIn("already done", out)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.DONE)

    # resume ──

    def test_resume_runs_via_runner_in_background(self):
        m = make_mission(self.store)
        self.store.set_status(m.id, MissionStatus.PAUSED)
        with patch("nomorals.missions.wired_runner") as mr, \
                patch("nomorals.agents.partner.runtime.threading.Thread",
                      _SyncThread):
            out = self.call(f"resume {m.id}")
        self.assertIn("background", out)
        mr.return_value.resume.assert_called_once_with(m.id)

    def test_resume_terminal_refused(self):
        m = make_mission(self.store)
        self.store.set_status(m.id, MissionStatus.FAILED)
        with patch("nomorals.missions.wired_runner") as mr:
            out = self.call(f"resume {m.id}")
        self.assertIn("can't resume", out)
        mr.assert_not_called()

    # cancel ──

    def test_cancel_sets_terminal_and_reason(self):
        m = make_mission(self.store)
        with patch("nomorals.missions.wired_runner") as mr:
            out = self.call(f"cancel {m.id} provider keeps timing out")
        self.assertIn("cancelled", out)
        self.assertEqual(self.store.get(m.id).status, MissionStatus.CANCELLED)
        self.assertEqual(
            self.store.get(m.id).state.get("status_note"),
            "provider keeps timing out")
        mr.return_value.cancel.assert_called_once_with(
            "provider keeps timing out")

    def test_cancel_default_reason(self):
        m = make_mission(self.store)
        with patch("nomorals.missions.wired_runner"):
            self.call(f"cancel {m.id}")
        self.assertEqual(self.store.get(m.id).status, MissionStatus.CANCELLED)

    def test_cancel_unknown_mission(self):
        out = self.call("cancel nope")
        self.assertIn("no mission matching", out)

    # retry ──

    def test_retry_creates_fresh_mission_and_keeps_terminal_old(self):
        m = make_mission(self.store, name="build widget")
        self.store.set_status(m.id, MissionStatus.FAILED)
        with patch("nomorals.missions.wired_runner") as mr, \
                patch("nomorals.agents.partner.runtime.threading.Thread",
                      _SyncThread):
            out = self.call(f"retry {m.id}")
        self.assertIn("retry", out)
        old = self.store.get(m.id)
        # terminal history is never rewritten: the failed record stays failed
        self.assertEqual(old.status, MissionStatus.FAILED)
        # the runner was told to run a *new* mission, not the old one
        run_mission = mr.return_value.run.call_args[0][0]
        self.assertNotEqual(run_mission.id, m.id)
        self.assertEqual(run_mission.goal, m.goal)
        self.assertEqual(run_mission.metadata.get("retry_of"), m.id)
        self.assertEqual(run_mission.state.get("plan"), m.state.get("plan"))
        self.assertEqual(run_mission.status, MissionStatus.PENDING)

    def test_retry_refuses_running_mission(self):
        m = make_mission(self.store)
        self.store.set_status(m.id, MissionStatus.RUNNING)
        with patch("nomorals.missions.wired_runner") as mr:
            out = self.call(f"retry {m.id}")
        self.assertIn("still running", out)
        mr.assert_not_called()

    # usage ──

    def test_usage_mentions_all_verbs(self):
        out = self.call("frobnicate")
        for verb in ("pause", "resume", "cancel", "retry", "watch",
                     "unwatch", "new", "stall", "clear", "status", "list"):
            self.assertIn(verb, out, verb)

    def test_empty_tail_defaults_to_status(self):
        m = make_mission(self.store)
        out = self.call("")
        self.assertIn("m1", out)


# ── watch / unwatch ──────────────────────────────────────────────────────────

class WatchTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = make_ctx(self)
        self.store = MissionStore(self.ctx.db)
        self.rt = _RuntimeStub(self.ctx)

    def call(self, tail, chat_key="telegram:9"):
        return self.rt._control_mission(tail, chat_key=chat_key)

    def test_watch_subscribes_chat(self):
        m = make_mission(self.store)
        out = self.call(f"watch {m.id}")
        self.assertIn("watching", out)
        self.assertEqual(MissionWatchers(self.ctx.db).watchers(m.id),
                         ["telegram:9"])

    def test_watch_is_idempotent(self):
        m = make_mission(self.store)
        self.call(f"watch {m.id}")
        out = self.call(f"watch {m.id}")
        self.assertIn("already watching", out)
        self.assertEqual(MissionWatchers(self.ctx.db).watchers(m.id),
                         ["telegram:9"])

    def test_watch_persists_across_store_instances(self):
        m = make_mission(self.store)
        self.call(f"watch {m.id}", chat_key="telegram:42")
        # a brand-new MissionWatchers reads the same kv_store rows
        self.assertEqual(MissionWatchers(self.ctx.db).watchers(m.id),
                         ["telegram:42"])

    def test_unwatch_removes_chat(self):
        m = make_mission(self.store)
        self.call(f"watch {m.id}")
        out = self.call(f"unwatch {m.id}")
        self.assertIn("stopped watching", out)
        self.assertEqual(MissionWatchers(self.ctx.db).watchers(m.id), [])

    def test_unwatch_when_not_watched(self):
        m = make_mission(self.store)
        out = self.call(f"unwatch {m.id}")
        self.assertIn("isn't watched", out)

    def test_watch_needs_chat_context(self):
        m = make_mission(self.store)
        out = self.call(f"watch {m.id}", chat_key="")
        self.assertIn("chat context", out)
        out = self.call(f"unwatch {m.id}", chat_key="")
        self.assertIn("chat context", out)


# ── templates ────────────────────────────────────────────────────────────────

class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = make_ctx(self)
        self.store = MissionStore(self.ctx.db)
        self.rt = _RuntimeStub(self.ctx)

    def call(self, tail, chat_key="telegram:9"):
        return self.rt._control_mission(tail, chat_key=chat_key)

    def test_new_research_template(self):
        out = self.call("new research rust borrow checker")
        self.assertIn("mission created", out)
        self.assertIn("/mission resume", out)
        missions = self.store.list()
        self.assertEqual(len(missions), 1)
        m = missions[0]
        self.assertEqual(m.metadata.get("template"), "research")
        self.assertIn("rust borrow checker", m.goal)
        plan = m.state.get("plan") or []
        self.assertEqual([s["name"] for s in plan],
                         ["gather sources", "synthesize findings",
                          "write report"])

    def test_new_build_template(self):
        self.call("new build a telegram poll bot")
        m = self.store.list()[0]
        self.assertEqual(m.metadata.get("template"), "build")
        self.assertEqual(m.status, MissionStatus.PENDING)

    def test_new_fix_template(self):
        self.call("new fix the flaky checkout test")
        m = self.store.list()[0]
        self.assertEqual(m.metadata.get("template"), "fix")
        plan = m.state.get("plan") or []
        self.assertEqual([s["name"] for s in plan],
                         ["reproduce", "root-cause", "patch", "verify"])

    def test_new_unknown_template_shows_usage(self):
        out = self.call("new frobnicate the thing")
        self.assertIn("templates", out)
        self.assertEqual(self.store.list(), [])

    def test_new_without_args_shows_usage(self):
        out = self.call("new research")
        self.assertIn("usage", out)
        self.assertEqual(self.store.list(), [])

    def test_template_spec_unit(self):
        spec = self.rt._mission_template_spec("research", "qwen3")
        self.assertEqual(spec["metadata"]["template"], "research")
        self.assertIn("qwen3", spec["goal"])
        self.assertIsNone(self.rt._mission_template_spec("nope", "x"))
        self.assertIsNone(self.rt._mission_template_spec("research", ""))


# ── milestone fan-out to watched chats ───────────────────────────────────────

class FanoutTests(unittest.TestCase):
    def setUp(self):
        self.gateway = FakeGateway()
        self.ctx, _ = make_ctx(self, gateway=self.gateway)
        self.store = MissionStore(self.ctx.db)

    def test_terminal_push_fans_out_to_watched_chat(self):
        m = make_mission(self.store)
        MissionWatchers(self.ctx.db).subscribe(m.id, "telegram:777")
        reporter = MissionMilestones(self.ctx, store=self.store)
        reporter.on_terminal(m, MissionStatus.DONE)
        chats = [c for _, c, _ in self.gateway.sent]
        self.assertIn("777", chats)       # the watched chat got it
        self.assertIn("111", chats)       # owner channel still got it
        texts = [t for _, c, t in self.gateway.sent if c == "777"]
        self.assertTrue(any("mission done" in t for t in texts))

    def test_owner_chat_not_double_sent(self):
        m = make_mission(self.store)
        MissionWatchers(self.ctx.db).subscribe(m.id, "telegram:111")
        reporter = MissionMilestones(self.ctx, store=self.store)
        reporter.on_started(m)
        hits = [c for _, c, _ in self.gateway.sent if c == "111"]
        self.assertEqual(len(hits), 1)

    def test_no_watchers_no_extra_sends(self):
        m = make_mission(self.store)
        reporter = MissionMilestones(self.ctx, store=self.store)
        reporter.on_started(m)
        chats = {c for _, c, _ in self.gateway.sent}
        self.assertEqual(chats, {"111"})

    def test_fanout_never_breaks_push(self):
        m = make_mission(self.store)
        MissionWatchers(self.ctx.db).subscribe(m.id, "bogus-no-colon")
        reporter = MissionMilestones(self.ctx, store=self.store)
        res = reporter.on_terminal(m, MissionStatus.DONE)
        self.assertEqual(res.get("kind"), "mission")  # publish still went out
        self.assertTrue(res.get("delivered"))


# ── runner wiring ────────────────────────────────────────────────────────────

class RunnerWiringTests(unittest.TestCase):
    def test_runner_builds_reporter_with_watch_store(self):
        ctx, _ = make_ctx(self)
        runner = MissionRunner(ctx, store=MissionStore(ctx.db))
        self.assertIsInstance(runner.reporter.watch_store, MissionWatchers)


if __name__ == "__main__":
    unittest.main()
