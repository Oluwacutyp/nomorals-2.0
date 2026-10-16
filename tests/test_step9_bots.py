"""Step 9 Bots tests — one invariant test per new module.

Each test pins the module's headline invariant, not incidental detail.
"""

from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path


class TestTelegramMenu(unittest.TestCase):
    def test_validate_menu_covers_registry(self):
        from nomorals.social.chat.telegram_menu import validate_menu
        ok, problems = validate_menu()
        self.assertTrue(ok, f"menu drift: {problems}")

    def test_public_menu_excludes_owner_only(self):
        from nomorals.social.chat.telegram_menu import build_telegram_menu
        public = build_telegram_menu(False)
        kinds = {k for v in public.values() for k in v}
        for owner_only in ("quit", "power", "update", "pull", "deploy", "restart"):
            self.assertNotIn(owner_only, kinds)


class TestActionFooters(unittest.TestCase):
    def test_tool_footer_retry_expand_next(self):
        from nomorals.social.chat.action_footers import tool_result_footer
        kb = tool_result_footer("/game trivia")
        flat = [b for row in kb for b in row]
        by_label = {label: data for label, data in flat}
        self.assertEqual(by_label["🔁 Retry"], "game trivia")
        self.assertEqual(by_label["🔍 Expand"], "help game")
        self.assertEqual(by_label["⏭️ Next"], "list")
        self.assertNotIn("💾 Save", by_label)
        self.assertNotIn("📤 Send", by_label)

    def test_footer_for_plain_text_is_none(self):
        from nomorals.social.chat.action_footers import footer_for_text
        self.assertIsNone(footer_for_text("just chatting", "/game trivia"))
        self.assertIsNone(footer_for_text("✅ done", ""))


class TestSingleConsumer(unittest.TestCase):
    def test_second_acquire_refuses(self):
        from nomorals.social.chat.single_consumer import SingleConsumerLock
        with tempfile.TemporaryDirectory() as tmp:
            lock1 = SingleConsumerLock(Path(tmp) / "bot.lock")
            lock2 = SingleConsumerLock(Path(tmp) / "bot.lock")
            self.assertTrue(lock1.acquire())
            try:
                self.assertFalse(lock2.acquire())
                self.assertTrue(lock1.is_held())
            finally:
                lock1.release()
            self.assertTrue(lock2.acquire())
            lock2.release()

    def test_stale_lock_reclaimed(self):
        from nomorals.social.chat.single_consumer import SingleConsumerLock
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bot.lock"
            # Dead PID: no such process (use a huge pid that cannot exist
            # as a live pid in this container... use os.getpid()+10**6
            # guarded: pid_alive check treats it as dead).
            path.write_text(json.dumps({"pid": 10 ** 9, "started_at": 0}))
            lock = SingleConsumerLock(path)
            self.assertTrue(lock.acquire())
            lock.release()


class TestProgressThrottle(unittest.TestCase):
    def test_edit_only_on_change_and_interval(self):
        from nomorals.social.chat.progress_throttle import ProgressThrottler
        clock = [0.0]
        th = ProgressThrottler(min_interval_s=3.0, clock=lambda: clock[0])
        self.assertTrue(th.should_edit("step 1"))
        self.assertFalse(th.should_edit("step 1"))  # unchanged
        clock[0] = 1.0
        self.assertFalse(th.should_edit("step 2"))  # too soon
        clock[0] = 3.1
        self.assertTrue(th.should_edit("step 2"))

    def test_long_output_becomes_document(self):
        from nomorals.social.chat.progress_throttle import split_or_document
        kind, payload = split_or_document("short text")
        self.assertEqual(kind, "chunks")
        big = "\n".join(f"line {i} " + "x" * 100 for i in range(600))
        kind, payload = split_or_document(big)
        self.assertEqual(kind, "document")
        self.assertEqual(payload, big.encode("utf-8"))


class _FakeMission:
    def __init__(self, mid, name, terminal=False):
        self.id = mid
        self.name = name
        self._terminal = terminal

    @property
    def terminal(self):
        return self._terminal


class _FakeStore:
    def __init__(self, missions=()):
        self._missions = list(missions)
        self.checkpointed = []

    def list(self, active_only=False, limit=200):
        if active_only:
            return [m for m in self._missions if not m.terminal]
        return list(self._missions)

    def resumable(self):
        return [m for m in self._missions if not m.terminal]

    def checkpoint(self, mission, label=""):
        self.checkpointed.append(mission.id)
        class CP:
            id = f"cp-{mission.id}"
        return CP()


class _FakeReport:
    ok = True

    def render(self):
        return "update — ✓ OK (0.1s)"


class TestUpdateControl(unittest.TestCase):
    def test_gate_refuses_mid_mission(self):
        from nomorals.social.chat.update_control import ChatUpdateController
        store = _FakeStore([_FakeMission("m1", "build thing")])
        ctl = ChatUpdateController(mission_store=store, update_runner=lambda p: _FakeReport())
        result = ctl.execute("update")
        self.assertFalse(result["ok"])
        self.assertFalse(result["restart_requested"])
        self.assertIn("still running", result["message"])

    def test_restart_checkpoints_and_requests_restart(self):
        from nomorals.social.chat.update_control import ChatUpdateController
        store = _FakeStore([_FakeMission("m1", "done thing", terminal=True)])
        ctl = ChatUpdateController(mission_store=store, update_runner=lambda p: _FakeReport())
        result = ctl.execute("restart")
        self.assertTrue(result["ok"])
        self.assertTrue(result["restart_requested"])

    def test_pull_does_not_restart(self):
        from nomorals.social.chat.update_control import ChatUpdateController
        store = _FakeStore([])
        ctl = ChatUpdateController(mission_store=store, update_runner=lambda p: _FakeReport())
        result = ctl.execute("pull")
        self.assertTrue(result["ok"])
        self.assertFalse(result["restart_requested"])

    def test_no_store_fails_closed(self):
        from nomorals.social.chat.update_control import ChatUpdateController
        ctl = ChatUpdateController(mission_store=None)
        self.assertFalse(ctl.execute("update")["ok"])


class TestWhatsAppDigest(unittest.TestCase):
    def test_digest_counts_mentions_decisions_resolved(self):
        from nomorals.social.chat.whatsapp_digest import build_digest
        msgs = [
            {"sender": "Ada", "text": "@death should we ship Friday?"},
            {"sender": "Bo", "text": "resolved — agreed, shipping Friday ✅"},
            {"sender": "Cy", "text": "lol ok"},
        ]
        d = build_digest("team", msgs, policy="always", me="death")
        self.assertTrue(d["included"])
        self.assertEqual(d["mentions"], 1)
        self.assertEqual(d["decisions"], 1)  # resolved message is not a decision
        self.assertEqual(d["resolved"], 1)

    def test_never_policy_excludes(self):
        from nomorals.social.chat.whatsapp_digest import build_digest
        d = build_digest("spam", [{"sender": "X", "text": "@death help?"}],
                         policy="never", me="death")
        self.assertFalse(d["included"])
        self.assertEqual(d["text"], "")


class TestWhatsAppCommunities(unittest.TestCase):
    def test_empty_and_overcap_refuse(self):
        from nomorals.social.chat.whatsapp_communities import (
            CommunityPost, plan_crosspost)
        self.assertFalse(plan_crosspost(CommunityPost("c", "", ["g1"])).ok)
        self.assertFalse(plan_crosspost(
            CommunityPost("c", "hi", [f"g{i}" for i in range(25)])).ok)

    def test_rate_limit_refuses_second_post(self):
        from nomorals.social.chat.whatsapp_communities import (
            CommunityPost, CommunityRateLimiter, plan_crosspost)
        limiter = CommunityRateLimiter(posts_per_minute=1, clock=lambda: 1000.0)
        ok1 = plan_crosspost(CommunityPost("c", "hello", ["g1"]), limiter=limiter)
        self.assertTrue(ok1.ok)
        self.assertIn("hello", ok1.preview)
        ok2 = plan_crosspost(CommunityPost("c", "again", ["g1"]), limiter=limiter)
        self.assertFalse(ok2.ok)


class TestWhatsAppRender(unittest.TestCase):
    def test_table_becomes_real_png(self):
        from nomorals.social.chat.whatsapp_render import render_for_whatsapp
        text = "| name | coins |\n|---|---|\n| ada | 50 |\n| bo | 30 |"
        with tempfile.TemporaryDirectory() as tmp:
            kind, path = render_for_whatsapp(text, tmp)
            self.assertEqual(kind, "image")
            data = Path(path).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG"))
            self.assertGreater(len(data), 100)

    def test_plain_text_stays_text(self):
        from nomorals.social.chat.whatsapp_render import render_for_whatsapp
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(render_for_whatsapp("hello there", tmp), ("text", "hello there"))

    def test_diff_becomes_png(self):
        from nomorals.social.chat.whatsapp_render import render_for_whatsapp
        diff = "--- a/x\n+++ b/x\n-old line\n+new line\n context\n"
        with tempfile.TemporaryDirectory() as tmp:
            kind, path = render_for_whatsapp(diff, tmp)
            self.assertEqual(kind, "image")
            self.assertTrue(Path(path).read_bytes().startswith(b"\x89PNG"))


class TestWhatsAppTransportCost(unittest.TestCase):
    def test_cloud_service_priced_from_rate_card(self):
        from nomorals.social.chat.whatsapp_transport_cost import estimate
        from nomorals.social.whatsapp_cost import RATE_SERVICE_KOBO
        r = estimate(transport="cloud-api", category="service", conversations=2)
        self.assertEqual(r["total_kobo"], 2 * RATE_SERVICE_KOBO)

    def test_baileys_zero_marginal_cold_outreach_prefers_cloud(self):
        from nomorals.social.chat.whatsapp_transport_cost import compare, estimate
        self.assertEqual(estimate(transport="baileys")["total_kobo"], 0)
        self.assertEqual(compare(cold_outreach=True)["recommendation"], "cloud-api")


class TestBaileysGuard(unittest.TestCase):
    def test_warmup_cap_and_burst_guard(self):
        from nomorals.social.chat.baileys_guard import BaileysPacer
        clock = [0.0]
        pacer = BaileysPacer(session_age_days=0, clock=lambda: clock[0])
        self.assertEqual(pacer.daily_cap(), 20)
        for _ in range(5):
            ok, _ = pacer.may_send()
            self.assertTrue(ok)
            pacer.record_sent()
        ok, reason = pacer.may_send()
        self.assertFalse(ok)
        self.assertIn("burst", reason)

    def test_first_warning_latches(self):
        from nomorals.social.chat.baileys_guard import WarningDetector
        det = WarningDetector()
        self.assertTrue(det.may_send())
        self.assertTrue(det.note("account temporarily banned for bulk behaviour"))
        self.assertFalse(det.may_send())
        det.note("all clear now")
        self.assertFalse(det.may_send())  # latched: never talks past a warning

    def test_encrypted_backup_roundtrip(self):
        from nomorals.social.chat.baileys_guard import (
            backup_session_encrypted, restore_session_encrypted)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "session"
            src.mkdir()
            (src / "auth.json").write_text('{"key": "secret"}')
            backup = backup_session_encrypted(src, Path(tmp) / "bak.json", "pw-123")
            self.assertNotIn(b"secret", backup.read_bytes())
            dest = restore_session_encrypted(backup, Path(tmp) / "restored", "pw-123")
            self.assertEqual((dest / "auth.json").read_text(), '{"key": "secret"}')
            with self.assertRaises(ValueError):
                restore_session_encrypted(backup, Path(tmp) / "bad", "wrong-pw")


class TestPersonalPresentation(unittest.TestCase):
    def test_non_owner_denied(self):
        from nomorals.core.policy import Policy
        from nomorals.social.chat.personal_presentation import check_personal_presentation
        d = check_personal_presentation(Policy(), actor="stranger", is_owner=False)
        self.assertFalse(d.allowed)

    def test_owner_needs_confirmation_then_allowed(self):
        from nomorals.core.policy import Capability, CapabilitySet, Policy
        from nomorals.social.chat.personal_presentation import check_personal_presentation
        policy = Policy(default_grant=CapabilitySet.all())
        d = check_personal_presentation(policy, actor="owner", is_owner=True)
        self.assertFalse(d.allowed)
        self.assertTrue(d.needs_confirmation)
        token = policy.issue_confirmation(Capability.SOCIAL_DM)
        d2 = check_personal_presentation(policy, actor="owner", is_owner=True,
                                         confirmation=token)
        self.assertTrue(d2.allowed)

    def test_group_send_needs_bulk(self):
        from nomorals.social.chat.personal_presentation import capability_for_send
        from nomorals.core.policy import Capability
        self.assertEqual(capability_for_send(recipients=1), Capability.SOCIAL_DM)
        self.assertEqual(capability_for_send(recipients=1, is_group=True),
                         Capability.SOCIAL_BULK)
        self.assertEqual(capability_for_send(recipients=5), Capability.SOCIAL_BULK)


class TestChatUpdateDispatch(unittest.TestCase):
    """Runtime dispatch for /update etc. — proves the branch runs (this
    is where a missing ``Path`` import hid: no test exercised it)."""

    def _make_runtime(self):
        import tempfile
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self):
                super().__init__(media_dir="/tmp/nm-test-media")
                self.name = "telegram-bot"

            def run(self, handler):
                pass

            def send(self, chat, text, *, reply_to="", buttons=None, **kw):
                return SendResult(ok=True, platform=self.name, message_id="m1")

        tmp = tempfile.TemporaryDirectory(prefix="nm-test-update-dispatch-")
        settings = load_settings(overrides={"home": tmp.name,
                                            "partner.platforms": "telegram-bot"})
        context = build_context(settings, with_executor=False, with_tools=False)
        adapter = FakeAdapter()
        gateway = ChatGateway({"telegram-bot": adapter}, db=context.db)
        runtime = PartnerRuntime(context, gateway=gateway)
        return runtime, context, tmp

    def _msg(self, owner: bool):
        from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
        return ChatMessage(
            chat=ChatRef(platform="telegram-bot", chat_id="c1",
                         kind=ChatKind.DM, peer="someone"),
            incoming=True, text="/update", sender="someone",
            meta={"is_owner": True} if owner else {})

    def test_non_owner_denied_before_update_machinery(self):
        from unittest import mock
        runtime, context, tmp = self._make_runtime()
        try:
            with mock.patch(
                    "nomorals.social.chat.update_control.ChatUpdateController",
                    side_effect=AssertionError("must not be constructed")):
                reply = runtime.handle_control("/update", "telegram-bot:c1",
                                               message=self._msg(owner=False))
            self.assertEqual(reply, "that one's just for the owner.")
        finally:
            try:
                runtime.stop()
            except Exception:
                pass
            context.close()
            tmp.cleanup()

    def test_owner_pull_dispatches_to_controller(self):
        from unittest import mock
        runtime, context, tmp = self._make_runtime()
        seen = {}

        class FakeCtl:
            def __init__(self, **kw):
                seen["kw"] = kw

            def execute(self, kind):
                seen["kind"] = kind
                return {"ok": True, "message": "fake update done",
                        "restart_requested": False, "checkpoints": [],
                        "update_report": None}

        try:
            with mock.patch(
                    "nomorals.social.chat.update_control.ChatUpdateController",
                    FakeCtl):
                reply = runtime.handle_control("/pull", "telegram-bot:c1",
                                               message=self._msg(owner=True))
            self.assertEqual(reply, "fake update done")
            self.assertEqual(seen["kind"], "pull")
        finally:
            try:
                runtime.stop()
            except Exception:
                pass
            context.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
