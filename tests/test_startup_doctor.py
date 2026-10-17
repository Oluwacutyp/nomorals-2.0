"""Infra spec P5 (§8.4 + §8.6): startup progress + /doctor health.

Startup: ping first (before heavy work), per-stage progress,
"still working" heartbeat, honest failure (real cause, DM + console),
never silent. /doctor: registry + config + stores + adapters +
providers, each pass/fail with the real cause.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.startup_progress import (
    StartupProgress,
    ping_owner_dm,
    verify_stores,
)


class TestStartupProgress(unittest.TestCase):
    def test_stages_recorded_in_order(self):
        seen: list[str] = []
        progress = StartupProgress(sink=seen.append)
        for stage in ("starting", "config loaded", "adapters up",
                      "stores verified", "ready"):
            progress.stage(stage)
        self.assertEqual(progress.stages,
                         ["starting", "config loaded", "adapters up",
                          "stores verified", "ready"])
        self.assertTrue(all("Devon:" in line for line in seen))

    def test_heartbeat_fires_while_stage_runs_long(self):
        seen: list[str] = []
        progress = StartupProgress(sink=seen.append, heartbeat_s=1.0)
        progress.start()
        try:
            progress.stage("config loaded")
            time.sleep(2.2)
        finally:
            progress.stop()
        self.assertTrue(any("still working" in line and "config loaded" in line
                            for line in seen),
                        f"no heartbeat in {seen}")

    def test_fail_names_real_cause_and_stops(self):
        seen: list[str] = []
        progress = StartupProgress(sink=seen.append)
        progress.stage("adapters up")
        message = progress.fail(None, RuntimeError("boom: adapter exploded"))
        self.assertIn("adapters up", message)
        self.assertIn("adapter exploded", message)
        self.assertTrue(any("failed to start" in line for line in seen))

    def test_fail_redacts_secrets(self):
        progress = StartupProgress(sink=lambda line: None)
        message = progress.fail(None, RuntimeError("bad api_key=sk-1234567890"))
        self.assertNotIn("sk-1234567890", message)


class TestPingOwnerDm(unittest.TestCase):
    def test_no_token_never_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(ping_owner_dm(None, "hello"))

    def test_failed_request_returns_false(self):
        settings = SimpleNamespace(
            partner=SimpleNamespace(owner_chats="telegram:1"))
        with mock.patch.dict(
                os.environ, {"NM_CHAT_TELEGRAM_BOT_TOKEN": "x"}, clear=False), \
                mock.patch("nomorals.agents.startup_progress.urllib.request.urlopen",
                           side_effect=OSError("no net")):
            self.assertFalse(ping_owner_dm(settings, "hello"))

    def test_accepted_message_returns_true(self):
        settings = SimpleNamespace(
            partner=SimpleNamespace(owner_chats="telegram:1"))
        body = mock.Mock()
        body.read.return_value = b'{"ok": true}'
        body.__enter__ = lambda self: self
        body.__exit__ = lambda *args: False
        with mock.patch.dict(
                os.environ, {"NM_CHAT_TELEGRAM_BOT_TOKEN": "x"}, clear=False), \
                mock.patch("nomorals.agents.startup_progress.urllib.request.urlopen",
                           return_value=body):
            self.assertTrue(ping_owner_dm(settings, "hello"))


class TestVerifyStores(unittest.TestCase):
    def test_healthy_stores_no_problems(self):
        tmp = tempfile.TemporaryDirectory(prefix="nm-test-startup-")
        env = mock.patch.dict(os.environ, {
            "NOMORALS_CHAT_MODES_PATH": os.path.join(tmp.name, "modes.json"),
            "NOMORALS_STYLE_OVERRIDES_PATH": os.path.join(tmp.name, "style.json"),
        })
        env.start()
        try:
            self.assertEqual(verify_stores(SimpleNamespace(db=None)), [])
        finally:
            env.stop()
            tmp.cleanup()

    def test_corrupt_stores_degrade_not_fail(self):
        tmp = tempfile.TemporaryDirectory(prefix="nm-test-startup-")
        modes = os.path.join(tmp.name, "modes.json")
        with open(modes, "w", encoding="utf-8") as fh:
            fh.write("{corrupt")
        env = mock.patch.dict(os.environ, {
            "NOMORALS_CHAT_MODES_PATH": modes,
            "NOMORALS_STYLE_OVERRIDES_PATH": os.path.join(tmp.name, "style.json"),
        })
        env.start()
        try:
            self.assertEqual(verify_stores(SimpleNamespace(db=None)), [])
        finally:
            env.stop()
            tmp.cleanup()


class TestDoctorCommand(unittest.TestCase):
    def _runtime(self, tmp_home: str):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self):
                super().__init__(media_dir="/tmp/nm-test-media")
            def run(self, handler):
                pass
            def send(self, chat, text, *, reply_to="", buttons=None, **kwargs):
                return SendResult(ok=True, platform="telegram", message_id="m1")

        settings = load_settings(overrides={
            "home": tmp_home,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        context = build_context(settings, with_executor=False, with_tools=False)
        gateway = ChatGateway({"telegram": FakeAdapter()}, db=context.db)
        return PartnerRuntime(context, gateway=gateway)

    def test_doctor_reports_all_five_checks(self):
        tmp = tempfile.TemporaryDirectory(prefix="nm-test-doctor-")
        try:
            rt = self._runtime(tmp.name)
            with mock.patch.dict(os.environ, {}, clear=True):
                out = rt.handle_control("/doctor", "telegram:5478650254")
        finally:
            tmp.cleanup()
        self.assertIn("doctor —", out)
        for section in ("registry:", "config:", "stores:",
                        "adapters:", "providers:"):
            self.assertIn(section, out, out)

    def test_doctor_config_flags_ignored_keys(self):
        tmp = tempfile.TemporaryDirectory(prefix="nm-test-doctor-")
        try:
            rt = self._runtime(tmp.name)
            with mock.patch.dict(os.environ, {"NM_FOO_BAR": "1"}, clear=True):
                out = rt.handle_control("/doctor", "telegram:5478650254")
        finally:
            tmp.cleanup()
        self.assertIn("FAIL config:", out)
        self.assertIn("NM_FOO_BAR", out)

    def test_doctor_is_owner_only(self):
        from nomorals.social.chat.control import caller_may_run
        self.assertTrue(caller_may_run("doctor", True))
        self.assertFalse(caller_may_run("doctor", False))


if __name__ == "__main__":
    unittest.main()
