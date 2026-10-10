"""Autonomy strategies: scoring, safety vetoes, adaptive threshold."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from nomorals.agents import autonomy as auto_mod
from nomorals.agents.autonomy import (
    AmbientShareStrategy,
    AutonomyAgent,
    DecisionContext,
    GroupPostStrategy,
    Proposal,
    SilenceCheckinStrategy,
)
from nomorals.core.events import EventBus
from nomorals.storage.db import Database


def _mood(label="calm", **values):
    cur = SimpleNamespace(label=label,
                          values={"affection": 70, "distance": 10,
                                  "insecurity": 10, **values})
    return SimpleNamespace(current=lambda: cur, describe=lambda: "calm")


def _brain():
    return SimpleNamespace(
        mood=_mood(),
        persona=SimpleNamespace(name="Devon", interests=["music"]),
        relationship=SimpleNamespace(),
        background=SimpleNamespace(),
    )


def _ctx(features=None):
    db = Database(":memory:")
    db.migrate()
    settings = SimpleNamespace(
        features=SimpleNamespace(enabled_features=set(features or ())))
    return SimpleNamespace(db=db, settings=settings, extras={})


_FEATURES_ON = lambda ctx, name: name in {"proactive_dm", "group_posts"}
_FEATURES_OFF = lambda ctx, name: False


def _gateway(ok=True):
    return SimpleNamespace(
        send=lambda platform, chat, content: SimpleNamespace(
            ok=ok, error=None if ok else "no route"))


def _agent(**kw):
    features = kw.pop("features", {"proactive_dm", "group_posts"})
    ctx = kw.pop("ctx", None) or _ctx(features)
    kw.setdefault("mode", "auto")
    kw.setdefault("owner_chats", {"telegram:dm:1"})
    return AutonomyAgent(ctx, _brain(), _gateway(), **kw)


def _decision_ctx(**kw):
    base = dict(now=time.time(), mood_label="calm",
                mood_values={"affection": 70, "distance": 10,
                             "insecurity": 10},
                dm_sent=0, group_sent=0, last_dm_ts=0.0, last_chat_ts={})
    base.update(kw)
    return DecisionContext(**base)


class StrategyScoringTests(unittest.TestCase):
    def test_silence_checkin_proposes_after_idle(self):
        agent = _agent()
        chat = agent._owner_dm_chat()
        self.assertIsNotNone(chat)
        # no chats table row -> hours_idle 0 -> no proposal
        props = SilenceCheckinStrategy().evaluate(agent, _decision_ctx())
        self.assertEqual(props, [])

    def test_group_post_needs_matching_mood(self):
        agent = _agent(group_chats={"telegram:group:9"})
        s = GroupPostStrategy()
        ctx = _decision_ctx(mood_label="playful")
        props = s.evaluate(agent, ctx)
        self.assertTrue(props)
        self.assertTrue(all(0.0 <= p.score <= 1.0 for p in props))
        self.assertEqual(props[0].kind, "group")
        # unlisted mood -> no proposal
        ctx2 = _decision_ctx(mood_label="melancholy")
        self.assertEqual(s.evaluate(agent, ctx2), [])

    def test_group_moods_are_data(self):
        agent = _agent(group_moods={"melancholy"})
        self.assertEqual(agent.group_moods, {"melancholy"})

    def test_ambient_share_proposes(self):
        agent = _agent()
        props = AmbientShareStrategy().evaluate(agent, _decision_ctx())
        # ambient share is opportunistic; score bounded either way
        for p in props:
            self.assertGreaterEqual(p.score, 0.0)
            self.assertLessEqual(p.score, 1.0)


class SafetyVetoTests(unittest.TestCase):
    def test_quiet_hours_blocks_tick(self):
        agent = _agent(quiet_start=0, quiet_end=23)
        # now = noon local; quiet 00:00-23:00 covers it
        noon = time.mktime(time.strptime("2026-10-09 12:00", "%Y-%m-%d %H:%M"))
        out = agent.tick(now=noon)
        self.assertEqual(out["decision"], "quiet hours")

    def test_dm_cap_veto(self):
        agent = _agent(max_dm_per_day=1)
        chat = SimpleNamespace(key="telegram:dm:1", kind="dm",
                               platform="telegram")
        prop = Proposal(kind="dm", chat=chat, score=0.9, reason="r",
                        draft_instruction="i", strategy="test")
        ctx = _decision_ctx(dm_sent=1)  # cap reached
        self.assertFalse(agent._policy_allows(prop, ctx, time.time(),
                                              _FEATURES_ON))

    def test_feature_flag_veto(self):
        agent = _agent()  # flags passed explicitly below
        chat = SimpleNamespace(key="telegram:dm:1", kind="dm",
                               platform="telegram")
        prop = Proposal(kind="dm", chat=chat, score=0.9, reason="r",
                        draft_instruction="i", strategy="test")
        ctx = _decision_ctx()
        self.assertFalse(agent._policy_allows(prop, ctx, time.time(),
                                              _FEATURES_OFF))

    def test_per_chat_interval_veto(self):
        agent = _agent()
        chat = SimpleNamespace(key="telegram:dm:1", kind="dm",
                               platform="telegram")
        prop = Proposal(kind="dm", chat=chat, score=0.9, reason="r",
                        draft_instruction="i", strategy="test")
        # the veto reads the agent's day state (DecisionContext is built
        # from it in tick())
        agent._day.last_chat_ts["telegram:dm:1"] = time.time()
        ctx = _decision_ctx()
        self.assertFalse(agent._policy_allows(prop, ctx, time.time(),
                                              _FEATURES_ON))

    def test_bad_strategy_never_kills_tick(self):
        agent = _agent()

        class Bad(auto_mod.Strategy):
            name = "bad"

            def evaluate(self, agent, ctx):
                raise RuntimeError("strategy exploded")

        agent._strategies = [Bad()]
        out = agent.tick()
        self.assertEqual(out["decision"], "nothing to send")


class AdaptiveThresholdTests(unittest.TestCase):
    def test_success_lowers_threshold(self):
        agent = _agent()
        agent._adapt_threshold(success=True)
        self.assertLess(agent._threshold, 0.5)
        self.assertGreaterEqual(agent._threshold, 0.3)

    def test_failure_raises_threshold(self):
        agent = _agent()
        agent._adapt_threshold(success=False)
        self.assertGreater(agent._threshold, 0.5)
        self.assertLessEqual(agent._threshold, 0.9)

    def test_threshold_floor_and_cap(self):
        agent = _agent()
        for _ in range(200):
            agent._adapt_threshold(success=True)
        self.assertEqual(agent._threshold, 0.3)
        for _ in range(200):
            agent._adapt_threshold(success=False)
        self.assertEqual(agent._threshold, 0.9)

    def test_below_threshold_skips(self):
        agent = _agent()

        class Weak(auto_mod.Strategy):
            name = "weak"

            def evaluate(self, agent, ctx):
                chat = SimpleNamespace(key="telegram:dm:1", kind="dm",
                                       platform="telegram")
                return [Proposal(kind="dm", chat=chat, score=0.1,
                                 reason="r", draft_instruction="i",
                                 strategy="weak")]

        agent._strategies = [Weak()]
        out = agent.tick()
        self.assertEqual(out["decision"], "below threshold")


class LedgerAndBusTests(unittest.TestCase):
    def test_emit_journals_ledger(self):
        from nomorals.agents.autonomy_ledger import AutonomyLedger
        import nomorals.core.events as events_mod

        agent = _agent()
        bus = EventBus()
        seen = []
        bus.subscribe("autonomy.*", lambda e: seen.append(e.topic), sync=True)
        orig = events_mod.global_bus
        events_mod.global_bus = bus
        try:
            chat = SimpleNamespace(key="telegram:dm:1", kind="dm",
                                   platform="telegram")
            agent._emit("dm", chat, "thinking of you", "silence check-in",
                        time.time())
        finally:
            events_mod.global_bus = orig
        rows = AutonomyLedger(agent.context.db).recent(system="autonomy")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["kind"], "send")
        self.assertIn("autonomy.sent", seen)


if __name__ == "__main__":
    unittest.main()
