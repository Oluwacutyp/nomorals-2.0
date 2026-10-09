"""Delivery-path hardening: the /notify 0-delivery root cause + fallbacks.

Covers the delivery side of the notification system (the scheduler
foundation is another worker's lane):
- bare numeric ``owner_chats`` entries (``"123456789"``, exactly what
  docs/TERMUX_ENV_TEMPLATE.txt documents) resolve to a platform instead
  of silently producing zero targets — the audit's 7/10 "failed" root
  cause;
- platform aliasing (``telegram`` vs ``telegram-bot``);
- fallback chain: a failed channel doesn't stop the next one, and every
  attempt is recorded;
- per-channel delivery tracking (``notification_deliveries``):
  sent/failed/skipped + message_id / error per notification;
- never-raises: broken gateway.status(), raising send(), no gateway;
- /notify shows per-channel reasons, not a bare "failed";
- Telegram Bot API 429s honor retry_after with bounded retries; 4xx
  fails fast.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents.notifier import (
    ATTEMPT_FAILED,
    ATTEMPT_SENT,
    ATTEMPT_SKIPPED,
    DeliveryOutcome,
    Notifier,
)
from nomorals.social.chat.gateway import resolve_owner_targets
from nomorals.storage.db import Database


class _SendResult:
    def __init__(self, ok=True, message_id="", error=""):
        self.ok = ok
        self.message_id = message_id
        self.error = error


class FakeGateway:
    """Configurable fake: per-platform running flags, send outcomes."""

    def __init__(self, running=(), fail=(), raise_on_status=False,
                 raise_on_send=()):
        self.running = set(running)
        self.fail = set(fail)
        self.raise_on_status = raise_on_status
        self.raise_on_send = set(raise_on_send)
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        if self.raise_on_status:
            raise RuntimeError("status boom")
        return {p: {"running_in_session": True} for p in self.running}

    def send(self, platform, chat_ref, text):
        if platform in self.raise_on_send:
            raise RuntimeError("send boom")
        self.sent.append((platform, chat_ref.chat_id, text))
        if platform in self.fail:
            return _SendResult(ok=False, error="simulated send failure")
        return _SendResult(ok=True, message_id="mid-1")

    def platforms_for_chat_id(self, chat_id):
        return []


def make_partner(**kw):
    base = dict(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=22,
        quiet_end=8,
        timezone="UTC",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def make_ctx(test=None, partner=None, gateway=None):
    tmp = tempfile.mkdtemp(prefix="delivery-test-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner or make_partner())
    ctx = SimpleNamespace(db=db, settings=settings, extras={})
    return ctx, tmp


class BareIdResolutionTests(unittest.TestCase):
    """The audit root cause: NM_PARTNER_OWNER_CHATS=<bare numeric id>."""

    def test_bare_numeric_id_reaches_telegram_bot(self):
        gw = FakeGateway(running=("telegram-bot",))
        ctx, _ = make_ctx(self,
                          partner=make_partner(owner_chats="123456789"),
                          gateway=gw)
        res = Notifier(ctx, gateway=gw).publish(
            "alert", "job done", "it worked", force=True)
        self.assertTrue(res["delivered"], res)
        self.assertEqual(res["delivery_state"], "sent")
        self.assertEqual([(p, c) for p, c, _ in gw.sent],
                         [("telegram-bot", "123456789")])

    def test_bare_id_zero_targets_before_fix_would_fail(self):
        # resolution itself: a bare id must produce a target, never []
        targets = resolve_owner_targets(
            "123456789",
            status={"telegram-bot": {"running_in_session": True}})
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].platform, "telegram-bot")

    def test_platform_alias_telegram_to_telegram_bot(self):
        gw = FakeGateway(running=("telegram-bot",))
        ctx, _ = make_ctx(
            self, partner=make_partner(owner_chats="telegram:123456789"),
            gateway=gw)
        res = Notifier(ctx, gateway=gw).publish(
            "alert", "hi", "body", force=True)
        self.assertTrue(res["delivered"], res)
        self.assertEqual(gw.sent[0][0], "telegram-bot")
        attempts = Notifier(ctx, gateway=gw).delivery_attempts(res["id"])
        self.assertTrue(any("alias" in (a.get("note") or "")
                            for a in attempts),
                        attempts)

    def test_explicit_platform_still_wins(self):
        gw = FakeGateway(running=("telegram", "telegram-bot"))
        ctx, _ = make_ctx(
            self, partner=make_partner(owner_chats="telegram:111"),
            gateway=gw)
        Notifier(ctx, gateway=gw).publish("alert", "t", "b", force=True)
        self.assertEqual(gw.sent[0][0], "telegram")

    def test_phone_shaped_bare_id_goes_to_whatsapp(self):
        targets = resolve_owner_targets(
            "+2348012345678",
            status={"whatsapp": {"running_in_session": True},
                    "telegram-bot": {"running_in_session": True}})
        self.assertEqual([t.platform for t in targets], ["whatsapp"])

    def test_numeric_id_never_offered_to_sms_whatsapp(self):
        # a Telegram user id must not become an SMS/WhatsApp recipient
        targets = resolve_owner_targets(
            "123456789",
            status={"whatsapp": {"running_in_session": True},
                    "sms": {"running_in_session": True}})
        plats = {t.platform for t in targets}
        self.assertFalse(plats & {"whatsapp", "sms"}, targets)

    def test_console_last_resort_when_nothing_resolves(self):
        gw = FakeGateway(running=("local",))
        ctx, _ = make_ctx(self, partner=make_partner(owner_chats=""),
                          gateway=gw)
        res = Notifier(ctx, gateway=gw).publish(
            "alert", "t", "b", force=True)
        self.assertTrue(res["delivered"], res)
        self.assertEqual([(p, c) for p, c, _ in gw.sent],
                         [("local", "console")])

    def test_registry_lookup_wins_over_id_shape(self):
        gw = FakeGateway(running=("telegram-bot", "discord"))
        gw.platforms_for_chat_id = lambda cid: ["discord"]  # type: ignore
        ctx, _ = make_ctx(self,
                          partner=make_partner(owner_chats="123456789"),
                          gateway=gw)
        Notifier(ctx, gateway=gw).publish("alert", "t", "b", force=True)
        self.assertEqual(gw.sent[0][0], "discord")


class FallbackChainTests(unittest.TestCase):
    def test_failed_channel_falls_through_to_next(self):
        gw = FakeGateway(running=("telegram-bot", "discord"),
                         fail=("telegram-bot",))
        ctx, _ = make_ctx(
            self,
            partner=make_partner(
                owner_chats="telegram-bot:111,discord:222"),
            gateway=gw)
        n = Notifier(ctx, gateway=gw)
        res = n.publish("alert", "t", "b", force=True)
        self.assertTrue(res["delivered"], res)
        self.assertEqual(res["delivery_state"], "sent")
        # both attempted, in config order
        self.assertEqual([p for p, _, _ in gw.sent],
                         ["telegram-bot", "discord"])
        attempts = n.delivery_attempts(res["id"])
        by_plat = {a["platform"]: a for a in attempts}
        self.assertEqual(by_plat["telegram-bot"]["state"], ATTEMPT_FAILED)
        self.assertIn("simulated send failure",
                      by_plat["telegram-bot"]["error"])
        self.assertEqual(by_plat["discord"]["state"], ATTEMPT_SENT)
        self.assertEqual(by_plat["discord"]["message_id"], "mid-1")

    def test_all_channels_down_records_each_reason(self):
        gw = FakeGateway(running=("telegram-bot", "discord"),
                         fail=("telegram-bot", "discord"))
        ctx, _ = make_ctx(
            self,
            partner=make_partner(
                owner_chats="telegram-bot:111,discord:222"),
            gateway=gw)
        n = Notifier(ctx, gateway=gw)
        res = n.publish("alert", "t", "b", force=True)
        self.assertFalse(res["delivered"])
        self.assertEqual(res["delivery_state"], "failed")
        attempts = n.delivery_attempts(res["id"])
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(a["state"] == ATTEMPT_FAILED for a in attempts))
        self.assertTrue(all(a["error"] for a in attempts))

    def test_not_running_platform_recorded_as_skipped(self):
        gw = FakeGateway(running=("telegram-bot",))  # discord configured, down
        ctx, _ = make_ctx(
            self,
            partner=make_partner(owner_chats="discord:222"),
            gateway=gw)
        n = Notifier(ctx, gateway=gw)
        res = n.publish("alert", "t", "b", force=True)
        self.assertFalse(res["delivered"])
        attempts = n.delivery_attempts(res["id"])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["state"], ATTEMPT_SKIPPED)
        self.assertIn("not running", attempts[0]["error"])

    def test_channels_filter_is_family_aware(self):
        gw = FakeGateway(running=("telegram-bot", "discord"))
        ctx, _ = make_ctx(
            self,
            partner=make_partner(
                owner_chats="telegram-bot:111,discord:222"),
            gateway=gw)
        Notifier(ctx, gateway=gw).publish("alert", "t", "b", force=True,
                                           channels=["telegram"])
        self.assertEqual([p for p, _, _ in gw.sent], ["telegram-bot"])


class NeverRaisesTests(unittest.TestCase):
    def test_status_raises_still_stores(self):
        gw = FakeGateway(running=("telegram-bot",), raise_on_status=True)
        ctx, _ = make_ctx(self, gateway=gw)
        res = Notifier(ctx, gateway=gw).publish(
            "alert", "t", "b", force=True)
        self.assertEqual(res["delivery_state"], "failed")
        self.assertFalse(res["delivered"])

    def test_send_raises_does_not_propagate(self):
        gw = FakeGateway(running=("telegram-bot",),
                         raise_on_send=("telegram-bot",))
        ctx, _ = make_ctx(self, gateway=gw)
        res = Notifier(ctx, gateway=gw).publish(
            "alert", "t", "b", force=True)
        self.assertFalse(res["delivered"])
        n = Notifier(ctx, gateway=gw)
        attempts = n.delivery_attempts(res["id"])
        self.assertEqual(attempts[0]["state"], ATTEMPT_FAILED)
        self.assertIn("send boom", attempts[0]["error"])

    def test_no_gateway_never_raises(self):
        ctx, _ = make_ctx(self)  # no gateway at all
        res = Notifier(ctx).publish("alert", "t", "b", force=True)
        self.assertIn(res["delivery_state"], ("pending", "sent"))

    def test_publish_returns_attempts(self):
        gw = FakeGateway(running=("telegram-bot",))
        ctx, _ = make_ctx(self, gateway=gw)
        n = Notifier(ctx, gateway=gw)
        res = n.publish("alert", "t", "b", force=True)
        self.assertIn("attempts", res)
        self.assertTrue(res["attempts"])


class DeliveryOutcomeTests(unittest.TestCase):
    def test_truthiness(self):
        self.assertFalse(DeliveryOutcome())
        self.assertTrue(DeliveryOutcome(
            [{"state": ATTEMPT_SENT, "platform": "telegram-bot"}]))
        self.assertFalse(DeliveryOutcome(
            [{"state": ATTEMPT_FAILED, "platform": "x"}]))

    def test_first_channel(self):
        o = DeliveryOutcome([
            {"state": ATTEMPT_FAILED, "platform": "discord"},
            {"state": ATTEMPT_SENT, "platform": "telegram-bot"},
        ])
        self.assertEqual(o.first_channel, "telegram-bot")
        self.assertEqual(o.delivered, 1)

    def test_describe_failures(self):
        o = DeliveryOutcome([
            {"state": ATTEMPT_SKIPPED, "platform": "discord",
             "chat_id": "222", "error": "not running"},
        ])
        self.assertIn("not running", o.describe_failures())


class NotifyHonestyTests(unittest.TestCase):
    def _run_notify(self, ctx, gw, arg="10"):
        from nomorals.agents.partner.runtime_schedule import (
            RuntimeScheduleMixin)
        stub = SimpleNamespace(context=ctx, gateway=gw)
        return RuntimeScheduleMixin._control_notify(stub, arg)

    def test_failed_row_shows_channel_reasons(self):
        gw = FakeGateway(running=("telegram-bot", "discord"),
                         fail=("telegram-bot", "discord"))
        ctx, _ = make_ctx(
            self,
            partner=make_partner(
                owner_chats="telegram-bot:111,discord:222"),
            gateway=gw)
        Notifier(ctx, gateway=gw).publish("alert", "job done", "b",
                                           force=True)
        out = self._run_notify(ctx, gw)
        self.assertIn("✗ failed", out)
        self.assertIn("telegram-bot:111", out)
        self.assertIn("discord:222", out)
        self.assertIn("simulated send failure", out)

    def test_sent_row_shows_channel_and_message_id(self):
        gw = FakeGateway(running=("telegram-bot",))
        ctx, _ = make_ctx(self, gateway=gw)
        Notifier(ctx, gateway=gw).publish("alert", "job done", "b",
                                           force=True)
        out = self._run_notify(ctx, gw)
        self.assertIn("✓ sent", out)
        self.assertIn("telegram-bot:111", out)
        self.assertIn("mid-1", out)

    def test_skipped_reason_visible(self):
        gw = FakeGateway(running=())  # nothing running, no console either
        ctx, _ = make_ctx(
            self, partner=make_partner(owner_chats="discord:222"),
            gateway=gw)
        Notifier(ctx, gateway=gw).publish("alert", "job done", "b",
                                           force=True)
        out = self._run_notify(ctx, gw)
        self.assertIn("✗ failed", out)
        self.assertIn("not running", out)


class RedeliverTrackingTests(unittest.TestCase):
    def test_redeliver_records_attempts_against_row(self):
        ctx, _ = make_ctx(self)  # no gateway -> pending
        n = Notifier(ctx)
        res = n.publish("alert", "t", "b", force=True)
        self.assertEqual(res["delivery_state"], "pending")
        gw = FakeGateway(running=("telegram-bot",))
        n.gateway = gw
        self.assertEqual(n.redeliver(), 1)
        attempts = n.delivery_attempts(res["id"])
        self.assertTrue(any(a["state"] == ATTEMPT_SENT for a in attempts),
                        attempts)


class Telegram429Tests(unittest.TestCase):
    def _adapter(self):
        from nomorals.social.chat.telegram import TelegramBotAdapter
        ad = TelegramBotAdapter.__new__(TelegramBotAdapter)
        ad.token = "tok"
        ad.poll_timeout = 1
        return ad

    def _resp(self, payload):
        class R:
            def json(self_inner):
                return payload
        return R()

    def test_429_honors_retry_after_then_succeeds(self):
        ad = self._adapter()
        calls = []

        class Sess:
            def post_json(self_inner, url, params, timeout=None):
                calls.append(params)
                if len(calls) == 1:
                    return self._resp({"ok": False, "error_code": 429,
                                       "description": "Too Many Requests",
                                       "parameters": {"retry_after": 1}})
                return self._resp({"ok": True,
                                   "result": {"message_id": 42}})

        ad._session = Sess()
        with patch("time.sleep") as slp:
            result = ad._api("sendMessage", chat_id=1, text="hi")
        self.assertEqual(result, {"message_id": 42})
        self.assertEqual(len(calls), 2)
        slp.assert_called_once()
        # retry_after=1 -> waits 1+1 buffer
        self.assertEqual(slp.call_args[0][0], 2)

    def test_429_gives_up_after_bounded_retries(self):
        from nomorals.core.errors import ValidationError
        ad = self._adapter()

        class Sess:
            def post_json(self_inner, url, params, timeout=None):
                return self._resp({"ok": False, "error_code": 429,
                                   "description": "Too Many Requests",
                                   "parameters": {"retry_after": 1}})

        ad._session = Sess()
        with patch("time.sleep"):
            with self.assertRaises(ValidationError):
                ad._api("sendMessage", chat_id=1, text="hi")

    def test_403_fails_fast_without_retry(self):
        from nomorals.core.errors import ValidationError
        ad = self._adapter()
        calls = []

        class Sess:
            def post_json(self_inner, url, params, timeout=None):
                calls.append(1)
                return self._resp({"ok": False, "error_code": 403,
                                   "description": "Forbidden: bot was blocked"})

        ad._session = Sess()
        with patch("time.sleep") as slp:
            with self.assertRaises(ValidationError):
                ad._api("sendMessage", chat_id=1, text="hi")
        self.assertEqual(len(calls), 1)
        slp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
