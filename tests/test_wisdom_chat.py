"""Chat integration tests: WisdomKeeper practice sessions in chat.

Covers the pacer (start/pause/resume/stop with a fake clock and a mock
Notifier — message sequence + timing logic, no real sleeping), the
manager's word routing + journaling cycle, the /wisdom chat dispatch,
and the `nm wisdom practice --chat` handoff.
"""
from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from nomorals.agents.partner.runtime_wisdom import RuntimeWisdomMixin
from nomorals.cmdline.commands.wisdom import _cmd_wisdom
from nomorals.social.chat.control import help_text, list_catalog, parse_control
from nomorals.wisdom import (PracticeError, PracticeGuide, SAFETY_TEXT,
                             WisdomChatManager)
from nomorals.wisdom import chat_session as chat_session_mod


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-chat-test-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings), tmp


class FakeClock:
    """Injectable clock: records sleeps, never actually waits."""

    def __init__(self, now=1700000000.0):
        self._now = now
        self.ticks: list[float] = []

    def sleep(self, seconds):
        self.ticks.append(float(seconds))

    def now(self):
        return self._now


class BlockingClock(FakeClock):
    """sleep() parks until released — deterministic mid-phase control."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.release = threading.Event()
        self.entered = threading.Event()

    def sleep(self, seconds):
        self.ticks.append(float(seconds))
        self.entered.set()
        self.release.wait(15)


class MockNotifier:
    """Stands in for Notifier: records publish calls, never sends."""

    def __init__(self):
        self.calls: list[dict] = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({
            "kind": kind, "title": title, "body": body,
            "critical": kw.get("critical"), "force": kw.get("force"),
            "channels": kw.get("channels"),
        })
        return {"id": "n1", "kind": kind, "title": title,
                "delivered": True, "delivery_state": "sent"}

    @property
    def bodies(self):
        return [c["body"] for c in self.calls]


class ChatPracticeSessionTests(unittest.TestCase):
    def _manager(self, ctx, clock=None, notifier=None):
        return WisdomChatManager(ctx, clock=clock or FakeClock(),
                                 notifier=notifier or MockNotifier())

    def test_start_sends_safety_first_then_phases_in_order(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        clock, notifier = FakeClock(), MockNotifier()
        mgr = self._manager(ctx, clock, notifier)
        try:
            ack = mgr.start_session("telegram:1", "box-breathing",
                                    platform="telegram")
            self.assertIn("Box Breathing", ack)
            session = mgr.active_session("telegram:1")
            self.assertTrue(session.join(timeout=10))
            self.assertEqual(
                mgr.status("telegram:1")["journal_await"], "box-breathing")
        finally:
            mgr.shutdown()

        calls = notifier.calls
        # SAFETY_TEXT first, verbatim, critical (quiet-hours bypass)
        self.assertEqual(calls[0]["body"], SAFETY_TEXT)
        self.assertTrue(calls[0]["critical"])
        self.assertTrue(calls[0]["force"])
        self.assertEqual(calls[0]["channels"], ["telegram"])
        # then every phase message in order
        self.assertEqual([c["body"] for c in calls[1:-1]],
                         guide.phases_for_chat("box-breathing"))
        # timing logic: paced sleeps sum to the phase durations
        timed = guide.timed_phases_for_chat("box-breathing")
        self.assertAlmostEqual(sum(clock.ticks),
                               sum(s for _, s in timed))
        # journal prompt last
        self.assertIn("How was that?", calls[-1]["body"])
        # practice log recorded a completed chat session
        hist = guide.history(limit=5)
        sess = [e for e in hist if e.get("type") == "session"][0]
        self.assertTrue(sess["completed"])
        self.assertEqual(sess["via"], "chat")

    def test_unknown_session_id_fails_fast(self):
        ctx, _ = _ctx()
        notifier = MockNotifier()
        mgr = self._manager(ctx, notifier=notifier)
        with self.assertRaises(PracticeError) as cm:
            mgr.start_session("telegram:1", "nope-not-real", platform="t")
        self.assertIn("available", str(cm.exception))
        self.assertEqual(notifier.calls, [])
        self.assertIsNone(mgr.active_session("telegram:1"))

    def test_double_start_rejected(self):
        ctx, _ = _ctx()
        clock = BlockingClock()
        notifier = MockNotifier()
        mgr = self._manager(ctx, clock, notifier)
        try:
            mgr.start_session("telegram:1", "box-breathing", platform="t")
            session = mgr.active_session("telegram:1")
            self.assertTrue(clock.entered.wait(5))
            reply = mgr.start_session("telegram:1", "box-breathing",
                                      platform="t")
            self.assertIn("already running", reply)
        finally:
            clock.release.set()
            mgr.shutdown()

    def test_pause_resume(self):
        ctx, _ = _ctx()
        clock, notifier = BlockingClock(), MockNotifier()
        mgr = self._manager(ctx, clock, notifier)
        try:
            mgr.start_session("telegram:1", "box-breathing", platform="t")
            session = mgr.active_session("telegram:1")
            self.assertTrue(clock.entered.wait(5))  # parked in phase-1 sleep
            n_before = len(notifier.calls)  # safety + phase 1

            reply = mgr.handle_incoming("telegram:1", "pause")
            self.assertIn("paused", reply)
            self.assertTrue(session.is_paused)

            clock.release.set()  # let the phase sleep finish
            self.assertTrue(session._hold.wait(5))  # thread parks on pause
            time.sleep(0.3)
            # no phase 2 went out while paused
            self.assertEqual(len(notifier.calls), n_before)

            reply = mgr.handle_incoming("telegram:1", "resume")
            self.assertIn("resum", reply)
            self.assertTrue(session.join(timeout=10))
        finally:
            clock.release.set()
            mgr.shutdown()

        bodies = notifier.bodies
        guide = PracticeGuide(ctx)
        self.assertEqual(bodies[1:-1], guide.phases_for_chat("box-breathing"))
        self.assertIn("How was that?", bodies[-1])

    def test_stop_ends_gracefully_and_still_journals(self):
        ctx, _ = _ctx()
        clock, notifier = BlockingClock(), MockNotifier()
        mgr = self._manager(ctx, clock, notifier)
        guide = mgr._practice_guide
        try:
            mgr.start_session("telegram:1", "box-breathing", platform="t")
            session = mgr.active_session("telegram:1")
            self.assertTrue(clock.entered.wait(5))

            stopper = threading.Thread(
                target=lambda: mgr.handle_incoming("telegram:1", "stop"))
            stopper.start()
            time.sleep(0.2)  # let stop() land before the release
            clock.release.set()
            stopper.join(timeout=10)
            self.assertTrue(session.join(timeout=10))
            self.assertEqual(session.state, "stopped")
        finally:
            clock.release.set()
            mgr.shutdown()

        # journal prompt still went out; reply is awaited
        self.assertIn("How was that?", notifier.bodies[-1])
        self.assertIn("ended early", notifier.bodies[-1])
        self.assertEqual(mgr.status("telegram:1")["journal_await"],
                         "box-breathing")
        hist = guide.history(limit=5)
        sess = [e for e in hist if e.get("type") == "session"][0]
        self.assertFalse(sess["completed"])

    def test_stop_with_nothing_running(self):
        ctx, _ = _ctx()
        mgr = self._manager(ctx)
        self.assertIn("no practice session",
                      mgr.stop_session("telegram:1"))
        # plain "stop" with no session/journal falls through to normal flow
        self.assertIsNone(mgr.handle_incoming("telegram:1", "stop"))

    def test_journal_flow_prompt_reply_stored(self):
        ctx, _ = _ctx()
        notifier = MockNotifier()
        mgr = self._manager(ctx, notifier=notifier)
        guide = mgr._practice_guide
        try:
            mgr.start_session("telegram:1", "four-seven-eight", platform="t")
            session = mgr.active_session("telegram:1")
            self.assertTrue(session.join(timeout=10))
            self.assertEqual(
                mgr.status("telegram:1")["journal_await"], "four-seven-eight")

            reply = mgr.handle_incoming("telegram:1",
                                        "I felt calm and saw warm colors")
            self.assertIn("saved", reply)
            # stored via PracticeGuide.journal
            hist = guide.history(limit=5)
            journals = [e for e in hist if e.get("type") == "journal"]
            self.assertEqual(journals[0]["notes"],
                             "I felt calm and saw warm colors")
            self.assertEqual(journals[0]["session_id"], "four-seven-eight")
            # await cleared — a further message falls through to normal flow
            self.assertIsNone(mgr.status("telegram:1")["journal_await"])
            self.assertIsNone(mgr.handle_incoming("telegram:1", "hello again"))
        finally:
            mgr.shutdown()

    def test_words_ignored_without_session(self):
        ctx, _ = _ctx()
        mgr = self._manager(ctx)
        self.assertIsNone(mgr.handle_incoming("telegram:9", "pause"))
        self.assertIsNone(mgr.handle_incoming("telegram:9", "resume"))
        # slash commands are never swallowed, even with a session live
        clock = BlockingClock()
        mgr2 = self._manager(ctx, clock, MockNotifier())
        try:
            mgr2.start_session("telegram:1", "box-breathing", platform="t")
            self.assertTrue(clock.entered.wait(5))
            self.assertIsNone(
                mgr2.handle_incoming("telegram:1", "/wisdom practice stop"))
        finally:
            clock.release.set()
            mgr2.shutdown()

    def test_state_file_persists_journal_await(self):
        import json as _json
        ctx, tmp = _ctx()
        mgr = self._manager(ctx)
        try:
            mgr.start_session("telegram:1", "four-seven-eight", platform="t")
            mgr.active_session("telegram:1").join(timeout=10)
        finally:
            mgr.shutdown()
        path = mgr._state_path()
        self.assertTrue(path.is_file())
        raw = _json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["journal_await"].get("telegram:1"),
                         "four-seven-eight")
        # a fresh manager picks the journal-await back up
        mgr2 = self._manager(ctx)
        self.assertEqual(mgr2.status("telegram:1")["journal_await"],
                         "four-seven-eight")


class FakeRuntime(RuntimeWisdomMixin):
    """Test double carrying the real mixin methods."""

    def __init__(self, ctx, manager):
        self.context = ctx
        self._manager = manager
        self.sent: list[str] = []

    def _ref_from_key(self, key):
        plat, _, cid = key.partition(":")
        return SimpleNamespace(platform=plat or "local", chat_id=cid,
                               key=key)

    def _send_long_checked(self, platform, chat, text):
        self.sent.append(text)
        return ""

    def _wisdom_chat_manager(self):
        return self._manager


class ChatDispatchTests(unittest.TestCase):
    def test_parse_control_routes_wisdom_and_alias(self):
        cmd = parse_control("/wisdom practice list")
        self.assertEqual(cmd.kind, "wisdom")
        self.assertEqual(cmd.tail, "practice list")
        cmd = parse_control("/wis ask kundalini")
        self.assertEqual(cmd.kind, "wis")
        self.assertEqual(cmd.tail, "ask kundalini")

    def test_wisdom_in_help_and_catalog(self):
        self.assertIn("/wisdom", help_text())
        catalog = list_catalog()
        self.assertIn("/wisdom", catalog)

    def _runtime(self, ctx, manager=None):
        return FakeRuntime(
            ctx, manager or WisdomChatManager(
                ctx, clock=FakeClock(), notifier=MockNotifier()))

    def test_control_wisdom_practice_list(self):
        ctx, _ = _ctx()
        rt = self._runtime(ctx)
        reply = rt._control_wisdom("practice list", chat_key="telegram:1")
        self.assertIn("box-breathing", reply)
        self.assertIn("/wisdom practice", reply)

    def test_control_wisdom_unknown_session_lists_available(self):
        ctx, _ = _ctx()
        rt = self._runtime(ctx)
        reply = rt._control_wisdom("practice nope", chat_key="telegram:1")
        self.assertIn("box-breathing", reply)  # available ids listed

    def test_control_wisdom_status(self):
        ctx, _ = _ctx()
        rt = self._runtime(ctx)
        reply = rt._control_wisdom("status", chat_key="telegram:1")
        self.assertIn("corpus", reply)

    def test_control_wisdom_ask_empty_corpus(self):
        ctx, _ = _ctx()
        rt = self._runtime(ctx)
        rt._control_wisdom("ask what is breath", chat_key="telegram:1")
        self.assertTrue(rt.sent)  # long reply went through the checker
        self.assertIn("No passages", rt.sent[0])

    def test_control_wisdom_usage_and_unknown_verb(self):
        ctx, _ = _ctx()
        rt = self._runtime(ctx)
        self.assertIn("usage",
                      rt._control_wisdom("", chat_key="telegram:1"))
        self.assertIn("unknown /wisdom verb",
                      rt._control_wisdom("frobnicate", chat_key="telegram:1"))

    def test_control_wisdom_practice_start_stop(self):
        ctx, _ = _ctx()
        notifier = MockNotifier()
        mgr = WisdomChatManager(ctx, clock=FakeClock(), notifier=notifier)
        rt = self._runtime(ctx, mgr)
        try:
            reply = rt._control_wisdom("practice box-breathing",
                                       chat_key="telegram:1")
            self.assertIn("starting", reply)
            session = mgr.active_session("telegram:1")
            self.assertTrue(session.join(timeout=10))
            # safety went out critical via the notifier
            self.assertEqual(notifier.calls[0]["body"], SAFETY_TEXT)
            self.assertTrue(notifier.calls[0]["critical"])

            reply = rt._control_wisdom("practice stop", chat_key="telegram:1")
            self.assertIn("no practice session", reply)  # already finished
        finally:
            mgr.shutdown()

    def test_control_wis_alias_routes_same_handler(self):
        # /wis parses to kind "wis"; the runtime dispatches both to
        # _control_wisdom — the same handler serves the alias.
        ctx, _ = _ctx()
        rt = self._runtime(ctx)
        reply = rt._control_wisdom("status", chat_key="telegram:1")
        self.assertIn("corpus", reply)


class CliChatHandoffTests(unittest.TestCase):
    def _args(self, **kw):
        d = dict(task=[], chat=False, json=False, limit=0, rounds=0,
                 tradition="", start=-3000, end=2100)
        d.update(kw)
        return SimpleNamespace(**d)

    def test_chat_without_gateway_fails_fast(self):
        ctx, _ = _ctx()  # no gateway attribute at all
        args = self._args(task=["practice", "box-breathing"], chat=True)
        err = io.StringIO()
        with redirect_stderr(err):
            rc = _cmd_wisdom(args, ctx)
        self.assertEqual(rc, 1)
        self.assertIn("no live chat gateway", err.getvalue())
        self.assertIn("/wisdom practice box-breathing", err.getvalue())

    def test_chat_with_gateway_delivers_session(self):
        class FakeGateway:
            def __init__(self):
                self.sent = []

            def status(self):
                return {"telegram": {"running_in_session": True}}

            def send(self, platform, chat, text):
                self.sent.append((platform, chat.chat_id, text))
                return SimpleNamespace(ok=True)

        ctx, tmp = _ctx()
        gw = FakeGateway()
        partner = SimpleNamespace(owner_chats="telegram:123")
        ctx.settings = SimpleNamespace(workspace_dir=tmp, partner=partner)
        ctx.db = None
        ctx.gateway = gw

        orig = chat_session_mod.WisdomChatManager

        class FastManager(orig):
            def __init__(self, context):
                super().__init__(context, clock=FakeClock())

        chat_session_mod.WisdomChatManager = FastManager
        try:
            args = self._args(task=["practice", "box-breathing"], chat=True)
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = _cmd_wisdom(args, ctx)
        finally:
            chat_session_mod.WisdomChatManager = orig

        self.assertEqual(rc, 0)
        self.assertIn("starting", out.getvalue())
        self.assertTrue(gw.sent, "nothing was delivered to chat")
        # SAFETY_TEXT went out through the Notifier path (title-prefixed)
        self.assertTrue(any("Safety notes for breathing practice" in t
                            for _, _, t in gw.sent))
        # journal-await persisted for the owner chat
        mgr = WisdomChatManager(ctx)
        self.assertEqual(mgr.status("telegram:123")["journal_await"],
                         "box-breathing")

    def test_chat_unknown_session_fails_fast(self):
        class FakeGateway:
            def status(self):
                return {"telegram": {"running_in_session": True}}

            def send(self, platform, chat, text):
                return SimpleNamespace(ok=True)

        ctx, tmp = _ctx()
        ctx.settings = SimpleNamespace(
            workspace_dir=tmp,
            partner=SimpleNamespace(owner_chats="telegram:123"))
        ctx.db = None
        ctx.gateway = FakeGateway()
        args = self._args(task=["practice", "nope"], chat=True)
        err = io.StringIO()
        with redirect_stderr(err):
            rc = _cmd_wisdom(args, ctx)
        self.assertEqual(rc, 1)
        self.assertIn("available", err.getvalue())


if __name__ == "__main__":
    unittest.main()
