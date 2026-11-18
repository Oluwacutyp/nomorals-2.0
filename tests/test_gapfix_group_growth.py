"""Churn / growth / topic-cluster organ for group health
(register §8 L1397-1398 — R2-27 remainder).

The /ghealth organ (nomorals/social/chat/group_health.py) reports
window churn counts, rhythm and silent members, but it has no churn
RATE, no growth number or trend, and no idea what the group actually
talks about. This file pins the new organ (nomorals/community/
growth.py) and its wiring: the organ's pure math, the health build
gaining ``growth`` + ``topics``, the /ghealth command feeding the
last 7 days of recorded messages into the topic clusters, the
``group_health`` brain tool, and the "how is <group> doing?"
natural phrasing.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.conversations import ConversationLog
from nomorals.storage.db import Database

WA = "whatsapp:120363025000000@g.us"
NOW = 1780000000.0  # fixed anchor; tests pass explicit `now`
DAY = 86400.0


# ── organ fixtures ─────────────────────────────────────────────────

def _member(uid, *, joined_ago, left_ago=None, bot=False,
            deleted=False, seen_ago=None, messages=0):
    return {
        "user_id": uid,
        "joined_ts": NOW - joined_ago * DAY,
        "left_ts": (NOW - left_ago * DAY) if left_ago is not None else 0.0,
        "is_bot": bot,
        "deleted": deleted,
        "last_seen": NOW - (seen_ago if seen_ago is not None else joined_ago) * DAY,
        "message_count": messages,
    }


def _ledger():
    """11 current members; window math is exact (see module docstring).

    Current 7d window: 2 joins (I, J), 1 leave (K) → start 10.
    Prior 7d window: 2 joins (L, M), 2 leaves (L, N) → start 10.
    """
    members = [_member(f"m{i}", joined_ago=60) for i in range(1, 9)]
    members.append(_member("I", joined_ago=3))
    members.append(_member("J", joined_ago=3))
    members.append(_member("K", joined_ago=60, left_ago=2))
    members.append(_member("L", joined_ago=10, left_ago=9))
    members.append(_member("M", joined_ago=10))
    members.append(_member("N", joined_ago=60, left_ago=9))
    return members


def _msgs():
    return [
        {"sender": "Ada", "text": "arsenal won the match, what a football night",
         "ts": NOW - 1 * DAY},
        {"sender": "Bob", "text": "that match was unreal, arsenal's defence held",
         "ts": NOW - 1 * DAY},
        {"sender": "Ada", "text": "football again on saturday? same time for the match",
         "ts": NOW - 2 * DAY},
        {"sender": "Cid", "text": "arsenal score predictions for the weekend match?",
         "ts": NOW - 2 * DAY},
        {"sender": "Bob", "text": "i missed the match, did arsenal win the football game",
         "ts": NOW - 3 * DAY},
        {"sender": "Dee", "text": "match highlights are up, great football from arsenal",
         "ts": NOW - 3 * DAY},
        {"sender": "Ada", "text": "gold price is climbing, market looks strong",
         "ts": NOW - 1 * DAY},
        {"sender": "Bob", "text": "sold some gold before the market dipped, good trade",
         "ts": NOW - 2 * DAY},
        {"sender": "Cid", "text": "gold market update: price holding above the level",
         "ts": NOW - 2 * DAY},
        {"sender": "Dee", "text": "is the gold trade still on? market seems flat",
         "ts": NOW - 4 * DAY},
        {"sender": "Ada", "text": "hello everyone", "ts": NOW - 1 * DAY},
        {"sender": "Bob", "text": "good morning all", "ts": NOW - 1 * DAY},
    ]


class GrowthOrganTests(unittest.TestCase):
    def test_churn_and_growth_rates(self):
        from nomorals.community.growth import churn_and_growth

        out = churn_and_growth(_ledger(), now=NOW)
        cur = out["current"]
        self.assertEqual(cur["joins"], 2)
        self.assertEqual(cur["leaves"], 1)
        self.assertEqual(cur["members_at_start"], 10)
        self.assertEqual(cur["net"], 1)
        self.assertAlmostEqual(cur["churn_rate"], 0.1)
        self.assertAlmostEqual(cur["growth_rate"], 0.1)
        prior = out["prior"]
        self.assertEqual(prior["joins"], 2)
        self.assertEqual(prior["leaves"], 2)
        self.assertEqual(prior["members_at_start"], 10)
        self.assertEqual(prior["net"], 0)
        self.assertAlmostEqual(prior["churn_rate"], 0.2)
        self.assertAlmostEqual(prior["growth_rate"], 0.0)
        self.assertAlmostEqual(out["churn_trend"], -0.1)
        self.assertAlmostEqual(out["growth_trend"], 0.1)
        self.assertEqual(out["direction"], "improving")

    def test_churn_member_who_joins_and_leaves_in_window(self):
        from nomorals.community.growth import churn_and_growth

        members = [_member("A", joined_ago=60),
                   _member("B", joined_ago=2, left_ago=1)]
        out = churn_and_growth(members, now=NOW)
        self.assertEqual(out["current"]["joins"], 1)
        self.assertEqual(out["current"]["leaves"], 1)
        # B was never a member at the window start.
        self.assertEqual(out["current"]["members_at_start"], 1)
        self.assertAlmostEqual(out["current"]["churn_rate"], 1.0)
        self.assertAlmostEqual(out["current"]["growth_rate"], 0.0)

    def test_empty_ledger_is_honest(self):
        from nomorals.community.growth import churn_and_growth

        out = churn_and_growth([], now=NOW)
        self.assertEqual(out["current"]["joins"], 0)
        self.assertEqual(out["current"]["leaves"], 0)
        self.assertIsNone(out["current"]["churn_rate"])
        self.assertIsNone(out["current"]["growth_rate"])
        self.assertIsNone(out["churn_trend"])
        self.assertEqual(out["direction"], "no data")

    def test_no_baseline_is_not_zero(self):
        from nomorals.community.growth import churn_and_growth

        # Everybody joined inside the window: there is no start
        # baseline, so a rate would be invented — None instead.
        members = [_member("A", joined_ago=2),
                   _member("B", joined_ago=1)]
        out = churn_and_growth(members, now=NOW)
        self.assertEqual(out["current"]["members_at_start"], 0)
        self.assertIsNone(out["current"]["churn_rate"])
        self.assertIsNone(out["current"]["growth_rate"])
        self.assertEqual(out["current"]["joins"], 2)

    def test_bots_and_deleted_are_excluded(self):
        from nomorals.community.growth import churn_and_growth

        members = [_member("A", joined_ago=60),
                   _member("bot", joined_ago=1, bot=True),
                   _member("gone", joined_ago=60, left_ago=1,
                           deleted=True)]
        out = churn_and_growth(members, now=NOW)
        self.assertEqual(out["current"]["joins"], 0)
        self.assertEqual(out["current"]["leaves"], 0)
        self.assertEqual(out["current"]["members_at_start"], 1)

    def test_worsening_direction(self):
        from nomorals.community.growth import churn_and_growth

        members = [_member(f"m{i}", joined_ago=60) for i in range(10)]
        members.append(_member("K1", joined_ago=60, left_ago=1))
        members.append(_member("K2", joined_ago=60, left_ago=2))
        out = churn_and_growth(members, now=NOW)
        self.assertEqual(out["direction"], "worsening")


class TopicClusterTests(unittest.TestCase):
    def test_two_topics_cluster_apart(self):
        from nomorals.community.growth import topic_clusters

        out = topic_clusters(_msgs())
        self.assertEqual(out["messages_scanned"], 12)
        clusters = out["clusters"]
        self.assertEqual(len(clusters), 2)
        labels = [" ".join(c["label"]) for c in clusters]
        self.assertTrue(any("arsenal" in lab and "match" in lab
                            for lab in labels), labels)
        self.assertTrue(any("gold" in lab and "market" in lab
                            for lab in labels), labels)
        counts = sorted((c["messages"] for c in clusters), reverse=True)
        self.assertEqual(counts, [6, 4])
        self.assertEqual(out["unclustered"], 2)
        # Shares are over the messages that clustered.
        self.assertAlmostEqual(clusters[0]["share"], 0.6)

    def test_empty_messages_is_honest(self):
        from nomorals.community.growth import topic_clusters

        out = topic_clusters([])
        self.assertEqual(out["clusters"], [])
        self.assertEqual(out["messages_scanned"], 0)
        self.assertEqual(out["unclustered"], 0)

    def test_single_message_never_clusters(self):
        from nomorals.community.growth import topic_clusters

        out = topic_clusters(
            [{"sender": "Ada", "text": "arsenal won the match",
              "ts": NOW}])
        self.assertEqual(out["clusters"], [])
        self.assertEqual(out["unclustered"], 1)

    def test_deterministic(self):
        from nomorals.community.growth import topic_clusters

        first = topic_clusters(_msgs())
        second = topic_clusters(list(reversed(_msgs())))
        self.assertEqual(first, second)

    def test_cluster_names_top_senders_and_example(self):
        from nomorals.community.growth import topic_clusters

        out = topic_clusters(_msgs())
        top = out["clusters"][0]
        self.assertTrue(top["senders"])
        self.assertEqual(top["senders"][0][0], "Ada")
        self.assertIn("arsenal", top["example"].lower())


# ── health-build wiring ────────────────────────────────────────────

def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _seed_group(surface, key=WA, *, members=(), now=NOW):
    for uid, name in members:
        surface.note_join(key, uid, username=name.lower(),
                          name=name, ts=now)
    return surface


def _member_row(surface, key, uid):
    return surface.store.group(key)["members"][uid]


class HealthBuildTests(unittest.TestCase):
    def test_build_gains_growth(self):
        from nomorals.social.chat import group_health as gh

        with tempfile.TemporaryDirectory() as tmp:
            surface = _surface(Path(tmp))
            # Base room joined long ago; K leaves inside the window,
            # M joined in the prior window — 10 members at the
            # window start, like the organ fixture.
            for uid, name in [(f"m{i}", f"M{i}") for i in range(1, 9)]:
                surface.note_join(WA, uid, username=name.lower(),
                                  name=name, ts=NOW - 60 * DAY)
            surface.note_join(WA, "K", username="k", name="K",
                              ts=NOW - 60 * DAY)
            _member_row(surface, WA, "K")["left_ts"] = NOW - 2 * DAY
            surface.note_join(WA, "M", username="m", name="M",
                              ts=NOW - 10 * DAY)
            # Two joins inside the window.
            surface.note_join(WA, "I", username="i", name="I",
                              ts=NOW - 3 * DAY)
            surface.note_join(WA, "J", username="j", name="J",
                              ts=NOW - 3 * DAY)
            health = gh.build_group_health(surface, WA, now=NOW)
            growth = health["growth"]
            self.assertEqual(growth["current"]["joins"], 2)
            self.assertEqual(growth["current"]["leaves"], 1)
            self.assertEqual(growth["current"]["net"], 1)
            self.assertAlmostEqual(growth["current"]["churn_rate"], 0.1)
            self.assertIn("Growth", gh.render_group_health(health))

    def test_build_topics_from_messages(self):
        from nomorals.social.chat import group_health as gh

        with tempfile.TemporaryDirectory() as tmp:
            surface = _surface(Path(tmp))
            _seed_group(surface, members=[("a", "Ada")])
            health = gh.build_group_health(surface, WA, now=NOW,
                                           messages=_msgs())
            topics = health["topics"]
            self.assertEqual(topics["messages_scanned"], 12)
            self.assertEqual(len(topics["clusters"]), 2)
            text = gh.render_group_health(health)
            self.assertIn("Talking about", text)
            self.assertIn("arsenal", text)

    def test_build_without_messages_is_honest(self):
        from nomorals.social.chat import group_health as gh

        with tempfile.TemporaryDirectory() as tmp:
            surface = _surface(Path(tmp))
            _seed_group(surface, members=[("a", "Ada")])
            health = gh.build_group_health(surface, WA, now=NOW)
            self.assertIsNone(health["topics"])
            self.assertIn("no message record", gh.render_group_health(health))

    def test_overview_totals_gain_net(self):
        from nomorals.social.chat import group_health as gh

        with tempfile.TemporaryDirectory() as tmp:
            surface = _surface(Path(tmp))
            _seed_group(surface, members=[("a", "Ada"), ("b", "Bob")])
            overview = gh.build_overview(surface, now=NOW)
            self.assertIn("net_7d", overview["totals"])
            self.assertIn("net", gh.render_overview(overview).lower())


# ── /ghealth dispatch ──────────────────────────────────────────────

def _log(tmp_db=None):
    db = tmp_db if tmp_db is not None else Database(":memory:")
    db.migrate()
    return ConversationLog(db)


_ROW_SEQ = [0]


def _seed_messages(log, key, rows, *, now=NOW):
    log.ensure_conversation(key, title=key, agent="partner",
                            channel=key.split(":", 1)[0], now=now)
    for sender, text, off in rows:
        _ROW_SEQ[0] += 1
        log.append_message(row_id=f"m{_ROW_SEQ[0]}",
                           conversation_id=key,
                           role="user", content=text, name=sender,
                           model="", created_at=now + off)
    return log


def _group_msg(key=WA, *, sender_id="admin1", platform="whatsapp"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind=ChatKind.GROUP,
                             chat_id=key.split(":", 1)[1]),
        sender_id=sender_id, sender="Admin",
        sender_username="boss", text="", meta={}, reply_to="",
        message_id="900")


def _rig(tmp_path: Path, adapter=None, *, db=None):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = _surface(tmp_path)
    surface.set_designated(WA, True, by="owner")
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    mixin._operator_or_false = lambda message: True  # type: ignore[method-assign]
    mixin._group_admin_ok = lambda message: True  # type: ignore[method-assign]
    mixin._console_is_admin = lambda k, sender: True  # type: ignore[method-assign]
    if db is not None:
        mixin.context = SimpleNamespace(db=db)
    return mixin, surface


class GhealthDispatchTests(unittest.TestCase):
    def test_in_group_report_carries_topics_from_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = Database(":memory:")
            log = _log(db)
            # Messages live in the real 7-day window — the dispatch
            # reads the log at real time, not the fixed NOW anchor.
            live_now = time.time()
            _seed_messages(log, WA, [
                ("Ada", "arsenal won the match, great football", -1 * DAY),
                ("Bob", "that match was unreal, arsenal held on", -1 * DAY),
                ("Ada", "gold price climbing, market strong", -1 * DAY),
                ("Bob", "sold some gold before the market dipped", -2 * DAY),
            ], now=live_now)
            mixin, surface = _rig(tmp_path, db=db)
            _seed_group(surface, members=[("a", "Ada"), ("b", "Bob")])
            out = mixin._control_ghealth("", _group_msg())
            text = getattr(out, "text", out)
            self.assertIn("Talking about", text)
            self.assertIn("arsenal", text)
            self.assertIn("Growth", text)

    def test_in_group_without_log_is_honest(self):
        with tempfile.TemporaryDirectory() as tmp:
            mixin, surface = _rig(Path(tmp))
            _seed_group(surface, members=[("a", "Ada")])
            out = mixin._control_ghealth("", _group_msg())
            text = getattr(out, "text", out)
            self.assertIn("no message record", text)
            self.assertIn("Growth", text)


# ── brain tool + NL phrasing ───────────────────────────────────────

class ToolTests(unittest.TestCase):
    def _registry(self, tmp):
        from nomorals.tools.registry import ToolRegistry

        settings = SimpleNamespace(home=tmp, home_path=tmp,
                                   workspace_dir=str(tmp))
        ctx = SimpleNamespace(settings=settings, db=Database(":memory:"))
        ctx.db.migrate()
        reg = ToolRegistry(ctx)
        reg.register_builtins()
        return reg, ctx

    def test_tool_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg, _ctx = self._registry(tmp)
            self.assertIn("group_health", reg._tools)

    def test_tool_report_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg, ctx = self._registry(tmp)
            # Seed the default surface under tmp/data/social.
            from nomorals.social.chat.group_surface import default_surface

            surface = default_surface(tmp)
            surface.note_join(WA, "a", username="ada", name="Ada",
                              ts=NOW)
            fn = reg._tools["group_health"].fn
            out = fn(action="report", group=WA)
            self.assertTrue(out.get("ok"))
            self.assertIn("Growth", out.get("text", ""))

    def test_tool_unknown_group_is_honest(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg, _ctx = self._registry(tmp)
            fn = reg._tools["group_health"].fn
            out = fn(action="report", group="no-such-group")
            self.assertTrue(out.get("ok"))
            self.assertIn("no record", out.get("text", "").lower())

    def test_tool_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg, _ctx = self._registry(tmp)
            fn = reg._tools["group_health"].fn
            out = fn(action="overview")
            self.assertTrue(out.get("ok"))
            self.assertIn("WhatsApp group", out.get("text", ""))


class ParserTests(unittest.TestCase):
    KNOWN = ("Lagos Life", WA)

    def test_how_is_group_doing_phrasings(self):
        from nomorals.tools.group_health import (
            parse_group_health_question,
        )

        for text in (
            "how is Lagos Life doing?",
            "how's the group doing",
            "is Lagos Life growing?",
            "what is Lagos Life talking about?",
            "what's the group talking about",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(
                    parse_group_health_question(text, self.KNOWN), text)
        self.assertEqual(
            parse_group_health_question("how is Lagos Life doing?",
                                        self.KNOWN),
            ("report", "Lagos Life"))
        self.assertEqual(
            parse_group_health_question(
                "what is Lagos Life talking about?", self.KNOWN),
            ("topics", "Lagos Life"))
        self.assertEqual(
            parse_group_health_question("is Lagos Life growing?",
                                        self.KNOWN),
            ("growth", "Lagos Life"))

    def test_unrelated_questions_are_not_stolen(self):
        from nomorals.tools.group_health import (
            parse_group_health_question,
        )

        for text in (
            "how is the weather doing?",
            "how are you doing?",
            "what is the group chat called?",
            "is the market growing?",
        ):
            with self.subTest(text=text):
                self.assertIsNone(
                    parse_group_health_question(text, self.KNOWN), text)


if __name__ == "__main__":
    unittest.main()
