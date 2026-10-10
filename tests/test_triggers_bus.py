"""Trigger engine bus source: cross-system event-driven triggers."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from nomorals.core.events import Event, EventBus
from nomorals.storage.db import Database
from nomorals.triggers.engine import TriggerEngine
from nomorals.triggers.models import (
    OUTCOME_FIRED,
    SOURCE_BUS,
    Trigger,
    TriggerError,
    validate_definition,
)


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx():
    db = _db()
    return SimpleNamespace(db=db, tools=SimpleNamespace(), memory=None,
                           settings=SimpleNamespace(), extras={})


def _make_trigger(engine, condition, name="bus-trigger",
                  action="message", action_params=None):
    trigger = Trigger(
        id=f"trg-{name}", name=name, enabled=True, source=SOURCE_BUS,
        condition=condition, action=action,
        action_params=action_params or {"chat": "local:console",
                                       "text": "fired"})
    engine.store.save(trigger)
    return trigger


def _fired_rows(engine):
    return [r for r in engine.store.history(limit=100)
            if r["outcome"] == OUTCOME_FIRED]


class BusValidationTests(unittest.TestCase):
    def test_bus_source_constant(self):
        self.assertEqual(SOURCE_BUS, "bus")

    def test_requires_topic(self):
        with self.assertRaises(TriggerError):
            validate_definition("bus", {"source": "bus"}, "notify", {})

    def test_topic_and_match(self):
        condition, params, cooldown = validate_definition(
            "bus", {"topic": "scheduler.job.finished",
                    "match": {"ok": True}}, "notify", {})
        self.assertEqual(condition, {"topic": "scheduler.job.finished",
                                    "match": {"ok": True}})
        self.assertEqual(params, {})
        self.assertEqual(cooldown, 0.0)

    def test_wildcard_topic(self):
        condition, _, _ = validate_definition(
            "bus", {"source": "bus", "topic": "mission.*"}, "notify", {})
        self.assertEqual(condition["topic"], "mission.*")

    def test_match_must_be_object(self):
        with self.assertRaises(TriggerError):
            validate_definition("bus", {"source": "bus", "topic": "a.*",
                                        "match": "nope"}, "notify", {})


class BusFiringTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.bus = EventBus()
        self.engine = TriggerEngine(self.ctx.db, self.ctx,
                                    send_message=lambda c, t: True)
        self.engine.attach_bus(self.bus)
        _make_trigger(self.engine, {"topic": "scheduler.job.finished",
                                    "match": {"ok": True}})

    def tearDown(self):
        self.engine.detach_bus()

    def test_matching_event_fires(self):
        self.bus.publish(Event(topic="scheduler.job.finished",
                               data={"ok": True, "name": "pulse"},
                               source="test"))
        time.sleep(0.2)
        self.assertEqual(len(_fired_rows(self.engine)), 1)

    def test_non_matching_data_ignored(self):
        self.bus.publish(Event(topic="scheduler.job.finished",
                               data={"ok": False}, source="test"))
        time.sleep(0.2)
        self.assertEqual(len(_fired_rows(self.engine)), 0)

    def test_wildcard_topic(self):
        _make_trigger(self.engine, {"topic": "mission.*"}, name="wild")
        self.bus.publish(Event(topic="mission.terminal",
                               data={"status": "failed"}, source="test"))
        time.sleep(0.2)
        self.assertEqual(len(_fired_rows(self.engine)), 1)

    def test_detach_stops_firing(self):
        self.engine.detach_bus()
        self.bus.publish(Event(topic="scheduler.job.finished",
                               data={"ok": True}, source="test"))
        time.sleep(0.2)
        self.assertEqual(len(_fired_rows(self.engine)), 0)

    def test_ledger_records_firing(self):
        from nomorals.agents.autonomy_ledger import AutonomyLedger

        self.bus.publish(Event(topic="scheduler.job.finished",
                               data={"ok": True}, source="test"))
        time.sleep(0.2)
        rows = AutonomyLedger(self.ctx.db).recent(system="trigger",
                                                  kind="fired")
        self.assertTrue(rows)


class BusChainTests(unittest.TestCase):
    def test_trigger_fired_cascades(self):
        """trigger.fired on the bus wakes a second trigger — the chain
        primitive for scheduler → trigger → mission flows.  The second
        trigger matches only the first's name, so its own firing does
        not loop back.  Uses the shared global bus, the production path
        (``_fire`` publishes trigger.fired there)."""
        from nomorals.core.events import global_bus

        ctx = _ctx()
        engine = TriggerEngine(ctx.db, ctx, send_message=lambda c, t: True)
        engine.attach_bus()  # global bus, like production
        try:
            _make_trigger(engine, {"topic": "scheduler.job.finished",
                                   "match": {"name": "pulse"}}, name="first")
            _make_trigger(engine, {"topic": "trigger.fired",
                                   "match": {"name": "first"}},
                          name="second")
            global_bus.publish(Event(topic="scheduler.job.finished",
                                     data={"name": "pulse", "ok": True},
                                     source="test"))
            time.sleep(0.3)
            # first trigger fires, its trigger.fired wakes the second
            self.assertEqual(len(_fired_rows(engine)), 2)
        finally:
            engine.detach_bus()

    def test_chain_depth_guard(self):
        """A trigger listening to its own trigger.fired must not loop
        forever — the depth guard caps the cascade at 8."""
        from nomorals.core.events import global_bus

        ctx = _ctx()
        engine = TriggerEngine(ctx.db, ctx, send_message=lambda c, t: True)
        engine.attach_bus()  # global bus, like production
        try:
            _make_trigger(engine, {"topic": "trigger.fired"})
            global_bus.publish(Event(topic="trigger.fired",
                                     data={"trigger_id": 999}, source="test"))
            time.sleep(0.5)
            self.assertEqual(len(_fired_rows(engine)), 8)
        finally:
            engine.detach_bus()


if __name__ == "__main__":
    unittest.main()
