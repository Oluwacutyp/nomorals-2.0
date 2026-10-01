"""Proactive delivery: she speaks first, owner-only.

Covers the proactive push layer built on the Notifier:
- gateway resolution from ``context.extras["gateway"]`` (the runtime's
  location — ``context.gateway`` does not exist on AgentContext)
- owner-only delivery: pushes go to ``partner.owner_chats`` keys and
  never to anyone else (stranger regression test)
- proactive master switch + per-kind toggles (briefing / watchers)
- quiet-hours policy helper (briefing exempt as a scheduled send,
  watchers own their per-watcher quiet hours)
- delivery states recorded per send: sent / failed / pending /
  held-quiet-hours / disabled / muted / deduped
- ``nm briefing status`` helper: switches + recent sends
- watcher ``_send`` audit log stays honest (no fake "sent")
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import notifier as notmod
from nomorals.agents.notifier import Notifier
from nomorals.storage.db import Database


class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok


class FakeGateway:
    """Records every send; owner channels report running."""

    def __init__(self, live_platforms=("telegram", "whatsapp")):
        self.live = set(live_platforms)
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        return {p: {"running_in_session": True} for p in self.live}

    def send(self, platform, chat_ref, text):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)


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


def make_ctx(partner=None, gateway=None, **kw):
    tmp = tempfile.mkdtemp(prefix="proactive-test-")
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner or make_partner(), **kw)
    ctx = SimpleNamespace(db=db, settings=settings,
                          extras={"gateway": gateway} if gateway else {})
    return ctx, tmp


class GatewayResolutionTests(unittest.TestCase):
    def test_resolves_from_extras(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        n = Notifier(ctx)
        self.assertIs(gw, n.gateway)

    def test_explicit_gateway_wins(self):
        gw1, gw2 = FakeGateway(), FakeGateway()
        ctx, _ = make_ctx(gateway=gw1)
        n = Notifier(ctx, gateway=gw2)
        self.assertIs(gw2, n.gateway)

    def test_none_without_gateway(self):
        ctx, _ = make_ctx()
        n = Notifier(ctx)
        self.assertIsNone(n.gateway)


class OwnerOnlyDeliveryTests(unittest.TestCase):
    def test_sends_only_to_owner_chats(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(
            partner=make_partner(owner_chats="telegram:111,whatsapp:222"),
            gateway=gw)
        n = Notifier(ctx)
        res = n.publish("briefing", "morning", "hello owner")
        self.assertTrue(res["delivered"])
        self.assertEqual(res["delivery_state"], "sent")
        targets = {(p, c) for p, c, _ in gw.sent}
        self.assertEqual(targets, {("telegram", "111"), ("whatsapp", "222")})

    def test_stranger_never_messaged(self):
        # regression: even if a stranger's chat key somehow appears in a
        # message, delivery only iterates owner_chats
        gw = FakeGateway()
        ctx, _ = make_ctx(
            partner=make_partner(owner_chats="telegram:111"),
            gateway=gw)
        n = Notifier(ctx)
        n.publish("watcher", "price alert", "stranger chat telegram:999 "
                  "should never receive this")
        for _p, chat_id, text in gw.sent:
            self.assertEqual(chat_id, "111")
            self.assertNotIn("999", chat_id)

    def test_dead_platform_not_counted(self):
        gw = FakeGateway(live_platforms=("telegram",))
        ctx, _ = make_ctx(
            partner=make_partner(owner_chats="telegram:111,whatsapp:222"),
            gateway=gw)
        n = Notifier(ctx)
        res = n.publish("briefing", "t", "b")
        self.assertTrue(res["delivered"])  # telegram reached
        self.assertEqual(
            {(p, c) for p, c, _ in gw.sent}, {("telegram", "111")})

    def test_no_live_channel_is_failed_not_sent(self):
        gw = FakeGateway(live_platforms=())
        ctx, _ = make_ctx(gateway=gw)
        n = Notifier(ctx)
        res = n.publish("briefing", "t", "b")
        self.assertFalse(res["delivered"])
        self.assertEqual(res["delivery_state"], "failed")

    def test_no_gateway_is_pending(self):
        ctx, _ = make_ctx()  # no gateway at all
        n = Notifier(ctx)
        res = n.publish("briefing", "t", "b")
        self.assertFalse(res["delivered"])
        self.assertEqual(res["delivery_state"], "pending")


class ProactiveGateTests(unittest.TestCase):
    def test_master_off_disables_briefing(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_enabled=False),
                          gateway=gw)
        n = Notifier(ctx)
        res = n.publish("briefing", "morning", "hi")
        self.assertFalse(res["delivered"])
        self.assertEqual(res["delivery_state"], "disabled")
        self.assertEqual(gw.sent, [])

    def test_master_off_disables_watchers(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_enabled=False),
                          gateway=gw)
        n = Notifier(ctx)
        res = n.publish("watcher", "hit", "hi")
        self.assertEqual(res["delivery_state"], "disabled")
        self.assertEqual(gw.sent, [])

    def test_kind_toggle_briefing_only(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_briefing=False),
                          gateway=gw)
        n = Notifier(ctx)
        res = n.publish("briefing", "morning", "hi")
        self.assertEqual(res["delivery_state"], "disabled")
        # watchers unaffected by the briefing toggle
        res2 = n.publish("watcher", "hit", "hi")
        self.assertEqual(res2["delivery_state"], "sent")

    def test_kind_toggle_watchers_only(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_watchers=False),
                          gateway=gw)
        n = Notifier(ctx)
        res = n.publish("watcher", "hit", "hi")
        self.assertEqual(res["delivery_state"], "disabled")
        res2 = n.publish("briefing", "morning", "hi")
        self.assertEqual(res2["delivery_state"], "sent")

    def test_non_proactive_kind_unaffected(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_enabled=False),
                          gateway=gw)
        n = Notifier(ctx)
        # "news" is not a proactive kind — keeps historical behavior
        res = n.publish("news", "digest", "hi")
        self.assertEqual(res["delivery_state"], "sent")

    def test_critical_bypasses_proactive_gate(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_enabled=False),
                          gateway=gw)
        n = Notifier(ctx)
        res = n.publish("watcher", "urgent", "hi", critical=True)
        self.assertEqual(res["delivery_state"], "sent")

    def test_force_still_respects_proactive_off(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_enabled=False),
                          gateway=gw)
        n = Notifier(ctx)
        res = n.publish("briefing", "late one", "hi", force=True)
        self.assertEqual(res["delivery_state"], "disabled")
        self.assertEqual(gw.sent, [])

    def test_dedupe_still_works(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        n = Notifier(ctx)
        n.publish("briefing", "same title", "one")
        res = n.publish("briefing", "same title", "two")
        self.assertTrue(res.get("deduped"))
        self.assertEqual(res["delivery_state"], "deduped")

    def test_disabled_rows_not_resurrected_by_redeliver(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_enabled=False),
                          gateway=gw)
        n = Notifier(ctx)
        n.publish("briefing", "morning", "hi")
        self.assertEqual(n.redeliver(), 0)
        self.assertEqual(gw.sent, [])

    def test_pending_redelivered_when_gateway_arrives(self):
        ctx, _ = make_ctx()  # stored as pending: no gateway
        n = Notifier(ctx)
        n.publish("briefing", "morning", "hi")
        self.assertEqual(len(n.pending()), 1)
        gw = FakeGateway()
        n.gateway = gw
        self.assertEqual(n.redeliver(), 1)
        self.assertEqual(len(gw.sent), 1)


class QuietHoursPolicyTests(unittest.TestCase):
    def test_in_quiet_hours_now(self):
        ctx, _ = make_ctx(
            partner=make_partner(quiet_start=22, quiet_end=8,
                                 timezone="UTC"))
        real_dt = __import__("datetime").datetime
        # 23:00 UTC is inside 22:00-08:00
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.side_effect = lambda tz=None: real_dt(
                2026, 10, 1, 23, 0, tzinfo=tz)
            mock_dt.fromtimestamp = real_dt.fromtimestamp
            self.assertTrue(notmod._in_quiet_hours_now(ctx))
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.side_effect = lambda tz=None: real_dt(
                2026, 10, 1, 12, 0, tzinfo=tz)
            mock_dt.fromtimestamp = real_dt.fromtimestamp
            self.assertFalse(notmod._in_quiet_hours_now(ctx))

    def test_briefing_exempt_from_quiet_hours(self):
        # the scheduled briefing is an explicit send (like an alarm) —
        # it goes out even inside quiet hours
        ctx, _ = make_ctx(partner=make_partner())
        self.assertIsNone(notmod.proactive_gate(ctx, "briefing"))

    def test_watcher_exempt_from_notifier_quiet_hours(self):
        # watchers carry their own per-watcher quiet-hours logic
        ctx, _ = make_ctx(partner=make_partner())
        self.assertIsNone(notmod.proactive_gate(ctx, "watcher"))


class BriefingStatusTests(unittest.TestCase):
    def test_status_reports_switches_and_states(self):
        from nomorals.agents import morning_briefing as mb
        gw = FakeGateway()
        ctx, _ = make_ctx(
            partner=make_partner(proactive_briefing=False),
            gateway=gw)
        n = Notifier(ctx)
        n.publish("briefing", "morning", "hi")   # disabled
        n.publish("watcher", "hit", "hi")        # sent
        payload = mb.proactive_status(ctx)
        self.assertFalse(payload["settings"]["proactive_briefing"])
        self.assertTrue(payload["settings"]["proactive_enabled"])
        states = {r["delivery_state"] for r in payload["recent"]}
        self.assertIn("disabled", states)
        self.assertIn("sent", states)

    def test_run_briefing_returns_delivery_state(self):
        from nomorals.agents import morning_briefing as mb
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        res = mb.run_briefing(ctx)
        self.assertIn("delivery_state", res)
        self.assertEqual(res["delivery_state"], "sent")
        self.assertTrue(gw.sent)


class WatcherSendHonestyTests(unittest.TestCase):
    def test_send_audit_not_fake_sent_when_undelivered(self):
        from nomorals.agents.watchers import WatcherStore, AlertEngine, Watcher
        ctx, _ = make_ctx()  # no gateway -> pending
        store = WatcherStore(ctx.db)
        w = Watcher(id="w1", name="btc", kind="price", severity="important",
                    cooldown_s=60)
        store.save(w)
        alerter = AlertEngine(store, ctx)
        out = alerter._send(w, "title", "body", time.time())
        self.assertEqual(out["action"], "sent")
        self.assertFalse(out["delivered"])
        log = store.alert_log("w1", limit=1)
        self.assertEqual(log[0]["status"], "pending")

    def test_send_audit_sent_when_delivered(self):
        from nomorals.agents.watchers import WatcherStore, AlertEngine, Watcher
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        store = WatcherStore(ctx.db)
        w = Watcher(id="w1", name="btc", kind="price", severity="important",
                    cooldown_s=60)
        store.save(w)
        alerter = AlertEngine(store, ctx)
        alerter._send(w, "title", "body", time.time())
        log = store.alert_log("w1", limit=1)
        self.assertEqual(log[0]["status"], "sent")

    def test_watcher_disabled_by_toggle(self):
        from nomorals.agents.watchers import WatcherStore, AlertEngine, Watcher
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(proactive_watchers=False),
                          gateway=gw)
        store = WatcherStore(ctx.db)
        w = Watcher(id="w1", name="btc", kind="price", severity="important",
                    cooldown_s=60)
        store.save(w)
        alerter = AlertEngine(store, ctx)
        out = alerter._send(w, "title", "body", time.time())
        self.assertEqual(gw.sent, [])
        log = store.alert_log("w1", limit=1)
        self.assertEqual(log[0]["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
