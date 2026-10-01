"""Prompt 04 — morning briefing: the overnight digest.

Covers: section composition from pluggable providers, empty-section
skipping, the ~600-word cap with "+N more", stored briefings
(``nm briefing today``), follow-up item resolution, catch-up (exactly one
late briefing after downtime), total-failure fallback, scheduler job
registration, and engagement demotion/pinning.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import morning_briefing as mb
from nomorals.storage.db import Database


def make_ctx(**settings_kw):
    tmp = tempfile.mkdtemp(prefix="briefing-test-")
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(workspace_dir=tmp, **settings_kw)
    return SimpleNamespace(db=db, settings=settings), tmp


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()

    def test_empty_world_still_composes(self):
        # no calendar, no topics, no symbols, no watchers — clean, no errors
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        self.assertIsInstance(b, mb.Briefing)
        # sections that need data are skipped silently
        names = {s.name for s in b.sections}
        self.assertNotIn("news", names)      # no news rows
        self.assertNotIn("markets", names)   # no symbols configured
        self.assertNotIn("calendar", names)  # nothing configured

    def test_held_info_alerts_become_section_and_are_digested(self):
        from nomorals.agents.watchers import WatcherStore
        store = WatcherStore(self.ctx.db)
        store.record_alert("w1", severity="info", channel="digest",
                           status="held", title="BTC dipped 3%",
                           body="price check")
        store.record_alert("w2", severity="info", channel="digest",
                           status="held", title="new HN post",
                           body="keyword hit")
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        sec = next(s for s in b.sections if s.name == "alerts")
        self.assertEqual(len(sec.items), 2)
        # consumed by the briefing — no double delivery
        self.assertEqual(store.held_alerts(), [])

    def test_news_filtered_by_topics(self):
        from nomorals.core.ids import new_id
        for i, (title, src) in enumerate([
                ("AI breakthrough in Lagos", "BBC"),
                ("football results", "BBC"),
                ("BTC hits new high", "Verge")]):
            self.ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at) VALUES (?,?,?,?,?,?,?)",
                (new_id(), src, title, f"https://x/{i}", "sum", 0, time.time()))
        prefs = mb._prefs(self.ctx)
        prefs["topics"] = ["AI", "BTC"]
        mb._save_prefs(self.ctx, prefs)
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        sec = next(s for s in b.sections if s.name == "news")
        titles = " ".join(i["title"] for i in sec.items)
        self.assertIn("AI breakthrough", titles)
        self.assertIn("BTC", titles)
        self.assertNotIn("football", titles)

    def test_anti_monopoly_max_3_per_source(self):
        from nomorals.core.ids import new_id
        for i in range(6):
            self.ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at) VALUES (?,?,?,?,?,?,?)",
                (new_id(), "BBC", f"story {i}", f"https://x/{i}", "sum",
                 0, time.time()))
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        sec = next(s for s in b.sections if s.name == "news")
        self.assertLessEqual(len(sec.items), 3)


class LengthCapTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()

    def test_200_news_items_capped_with_note(self):
        from nomorals.core.ids import new_id
        for i in range(200):
            long_title = f"headline number {i} " + "word " * 80
            self.ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at) VALUES (?,?,?,?,?,?,?)",
                (new_id(), f"src-{i % 5}", long_title, f"https://x/{i}",
                 "summary " * 40, 0, time.time()))
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        self.assertLessEqual(b.word_count(), mb.MAX_WORDS)
        self.assertIn("+", b.truncated_note)
        self.assertIn("more", b.truncated_note)


class RunDeliverTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()

    def test_run_stores_and_delivers(self):
        delivered = []

        class FakeNotifier:
            def __init__(self, context, gateway=None):
                pass

            def publish(self, kind, title, body, **kw):
                delivered.append((kind, title, body))
                return {"delivered": True}

        with patch("nomorals.agents.notifier.Notifier", FakeNotifier):
            res = mb.run_briefing(self.ctx)
        self.assertTrue(res["ok"])
        self.assertTrue(delivered)
        self.assertEqual(delivered[0][0], "briefing")
        # stored for `nm briefing today`
        stored = mb.latest_briefing(self.ctx)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["date"], res["date"])

    def test_quiet_world_gets_quiet_note(self):
        class FakeNotifier:
            def __init__(self, context, gateway=None):
                pass

            def publish(self, kind, title, body, **kw):
                return {"delivered": True}

        # the ambient weather/USA providers are live-network: simulate a
        # fully quiet world (all providers empty) to keep this hermetic.
        from nomorals.agents import weather as _wx
        with patch.object(_wx.WeatherProvider, "collect",
                          return_value=None), \
             patch.object(_wx.USASituationsProvider, "collect",
                          return_value=None), \
             patch("nomorals.agents.notifier.Notifier", FakeNotifier):
            res = mb.run_briefing(self.ctx)
        self.assertIn("quiet night", res["text"])

    def test_total_failure_sends_fallback(self):
        class FakeNotifier:
            def __init__(self, context, gateway=None):
                pass

            def publish(self, kind, title, body, **kw):
                FakeNotifier.last = body
                return {"delivered": True}

        with patch.object(mb.BriefingComposer, "compose",
                          side_effect=RuntimeError("boom")):
            with patch("nomorals.agents.notifier.Notifier", FakeNotifier):
                res = mb.run_briefing(self.ctx)
        self.assertFalse(res["ok"])
        self.assertTrue(res["fallback"])
        self.assertIn("Briefing failed", FakeNotifier.last)
        self.assertNotIn("Traceback", FakeNotifier.last)


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()
        from nomorals.core.ids import new_id
        for i in range(4):
            self.ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at) VALUES (?,?,?,?,?,?,?)",
                (new_id(), "BBC", f"story {i}", f"https://x/{i}", "sum",
                 0, time.time()))
        # keep this fixture news-only: the ambient weather/USA providers
        # are live-network and would shift item numbering.
        from nomorals.agents import weather as _wx
        with patch.object(_wx.WeatherProvider, "collect",
                          return_value=None), \
             patch.object(_wx.USASituationsProvider, "collect",
                          return_value=None):
            b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        mb.store_briefing(self.ctx, b)

    def test_item_3_resolves(self):
        item = mb.followup_item(self.ctx, 3)
        self.assertIsNotNone(item)
        self.assertEqual(item["n"], 3)
        self.assertIn("title", item["item"])

    def test_out_of_range_returns_none(self):
        self.assertIsNone(mb.followup_item(self.ctx, 999))
        self.assertIsNone(mb.followup_item(self.ctx, 0))

    def test_followup_records_engagement(self):
        store = mb._engagement_store(self.ctx)
        before = store.get("news")["followups"]
        mb.followup_item(self.ctx, 1)
        self.assertEqual(store.get("news")["followups"], before + 1)


class EngagementTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()
        from nomorals.core.ids import new_id
        for i in range(3):
            self.ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at) VALUES (?,?,?,?,?,?,?)",
                (new_id(), "BBC", f"story {i}", f"https://x/{i}", "sum",
                 0, time.time()))

    def test_sustained_low_engagement_demotes(self):
        store = mb._engagement_store(self.ctx)
        for _ in range(7):
            store.record_view("news")  # seen 7x, never followed up
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        sec = next(s for s in b.sections if s.name == "news")
        self.assertGreater(sec.priority, 50)  # demoted from 50

    def test_pin_protects_from_demotion(self):
        prefs = mb._prefs(self.ctx)
        prefs["pinned_sections"] = ["news"]
        mb._save_prefs(self.ctx, prefs)
        store = mb._engagement_store(self.ctx)
        for _ in range(7):
            store.record_view("news")
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        sec = next(s for s in b.sections if s.name == "news")
        self.assertEqual(sec.priority, 50)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()

    def test_ensure_briefing_job_idempotent(self):
        r1 = mb.ensure_briefing_job(self.ctx)
        r2 = mb.ensure_briefing_job(self.ctx)
        self.assertTrue(r1.get("scheduled") or r1.get("already_scheduled"))
        self.assertTrue(r2.get("already_scheduled"))
        from nomorals.agents.scheduler import Scheduler
        jobs = [j for j in Scheduler(self.ctx).list_jobs()
                if j.get("name") == mb.BRIEFING_JOB_NAME]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["spec"], "07:00")  # scheduler stores HH:MM

    def test_catchup_delivers_once_after_downtime(self):
        # simulate: briefing time (07:00) already passed, no briefing today
        real_today = mb._today_str
        real_time = mb.briefing_time
        try:
            mb._today_str = lambda ctx: "2026-10-01"
            mb.briefing_time = lambda ctx: "00:01"  # long past
            r = mb.check_catchup(self.ctx)
            self.assertTrue(r.get("catchup"))
            self.assertTrue(r.get("late"))
            # second boot: already delivered → no duplicate
            r2 = mb.check_catchup(self.ctx)
            self.assertFalse(r2.get("catchup"))
        finally:
            mb._today_str = real_today
            mb.briefing_time = real_time

    def test_catchup_skipped_before_briefing_time(self):
        real_today = mb._today_str
        real_time = mb.briefing_time
        try:
            mb._today_str = lambda ctx: "2026-10-01"
            mb.briefing_time = lambda ctx: "23:59"  # not yet
            r = mb.check_catchup(self.ctx)
            self.assertFalse(r.get("catchup"))
            self.assertIsNone(mb.latest_briefing(self.ctx, "2026-10-01"))
        finally:
            mb._today_str = real_today
            mb.briefing_time = real_time


class RepoSectionTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()

    def test_repo_watcher_alerts_become_section(self):
        from nomorals.agents.watchers import WatcherStore, Watcher
        from nomorals.core.ids import new_id
        store = WatcherStore(self.ctx.db)
        w = store.create(Watcher(id=new_id(), name="my repo", kind="repo",
                                 target={"repo": "x/y"}))
        store.record_alert(w.id, severity="info", channel="digest",
                           status="held", title="new issue #42",
                           body="issue opened")
        # a non-repo held alert must NOT leak into the repo section
        store.record_alert("other", severity="info", channel="digest",
                           status="held", title="price moved",
                           body="btc")
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        sec = next(s for s in b.sections if s.name == "repos")
        self.assertEqual(len(sec.items), 1)
        self.assertIn("issue #42", sec.items[0]["title"])


class RoomsSectionTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()

    def test_dirty_room_appears_in_briefing(self):
        from nomorals.workspace.rooms import RoomManager
        mgr = RoomManager(self.tmp, db=self.ctx.db)
        room = mgr.create("Crashy Project")
        # simulate a crashed session: dirty state persisted to the DB
        room.state["dirty"] = True
        room.blockers.append("simulated crash")
        mgr._persist(room)
        b = mb.BriefingComposer().compose(self.ctx, "2026-10-01")
        sec = next((s for s in b.sections if s.name == "rooms"), None)
        self.assertIsNotNone(sec)
        self.assertIn("Crashy Project", sec.render_text())


class MarketsProtocolTests(unittest.TestCase):
    """The briefing reuses the MarketDataProvider protocol defined in
    nomorals/integrations/sentinel_bridge.py (it was put there explicitly
    for Prompt 04) — no parallel protocol."""

    def test_shared_protocol_imported(self):
        from nomorals.integrations.sentinel_bridge import (
            MarketDataProvider as Shared)
        self.assertIs(mb.MarketDataProvider, Shared)

    def test_coingecko_provider_implements_protocol(self):
        ctx, _ = make_ctx()
        prov = mb.CoinGeckoMarketProvider(ctx)
        # structural: quote(symbol, market=...) + overnight_movers(...)
        import inspect
        self.assertIn("market", inspect.signature(prov.quote).parameters)
        self.assertIn("market",
                      inspect.signature(prov.overnight_movers).parameters)

    def test_overnight_movers_sorts_by_abs_change(self):
        ctx, _ = make_ctx()
        prov = mb.CoinGeckoMarketProvider(ctx)
        prov.quote = lambda s, market="crypto": {  # type: ignore[method-assign]
            "symbol": s, "price": 1.0,
            "change_pct_24h": {"A": 5.0, "B": -12.0, "C": 1.0}[s]}
        movers = prov.overnight_movers(["A", "B", "C"])
        self.assertEqual([m["symbol"] for m in movers], ["B", "A", "C"])


class ToolRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.tmp = make_ctx()

    def test_briefing_tool_config_action(self):
        from nomorals.core.policy import CapabilitySet
        from nomorals.tools.registry import ToolRegistry
        reg = ToolRegistry(self.ctx)
        mb.register(reg)
        out = reg.call("briefing", capabilities=CapabilitySet.all(),
                       action="config")
        self.assertTrue(out.ok)
        self.assertEqual(out.value["time"], "07:00")
        self.assertEqual(out.value["max_words"], mb.MAX_WORDS)


if __name__ == "__main__":
    unittest.main()
