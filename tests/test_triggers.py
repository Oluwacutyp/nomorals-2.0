"""Trigger engine: validation, sources, actions, engine, webhook, hook."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.storage.db import Database
from nomorals.triggers import (
    TriggerEngine,
    TriggerError,
    attach,
    message_hook,
    validate_definition,
)
from nomorals.triggers.actions import build_command_argv, default_run_command
from nomorals.triggers.engine import ENGINE_KEY
from nomorals.triggers.models import (
    OUTCOME_ERROR,
    OUTCOME_FIRED,
    OUTCOME_NO_MATCH,
    OUTCOME_SKIPPED,
    normalize_schedule_condition,
)
from nomorals.triggers.sources import schedule_plan
from nomorals.triggers.store import TriggerStore


def _db():
    return Database(":memory:")


def _engine(**kw):
    kw.setdefault("notify_fn", lambda trig, title, body, eng: {"ok": True})
    return TriggerEngine(_db(), **kw)


class ValidationTests(unittest.TestCase):
    def test_unknown_source(self):
        with self.assertRaises(TriggerError):
            validate_definition("bogus", {}, "notify", {})

    def test_unknown_action(self):
        with self.assertRaises(TriggerError):
            validate_definition("webhook", {}, "bogus", {})

    def test_bad_cron(self):
        with self.assertRaises(TriggerError):
            validate_definition("schedule", {"cron": "not a cron"},
                                "notify", {})

    def test_schedule_needs_exactly_one_key(self):
        with self.assertRaises(TriggerError):
            validate_definition("schedule", {}, "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition(
                "schedule", {"cron": "0 9 * * *", "interval": "1h"},
                "notify", {})

    def test_interval_normalization(self):
        cases = {
            "30m": "*/30 * * * *",
            "1m": "*/1 * * * *",
            "60m": "0 * * * *",
            "2h": "0 */2 * * *",
            "24h": "0 0 * * *",
            "1d": "0 0 * * *",
            "7d": "0 0 * * SUN",
            "120s": "*/2 * * * *",
        }
        for text, cron in cases.items():
            cond, _, _ = validate_definition(
                "schedule", {"interval": text}, "notify", {})
            self.assertEqual(cond, {"cron": cron}, text)

    def test_interval_rejected(self):
        for text in ["45s", "90m", "25h", "3d", "2w", "xyz", "0m", "-5m"]:
            with self.assertRaises(TriggerError, msg=text):
                validate_definition(
                    "schedule", {"interval": text}, "notify", {})

    def test_daily_weekly(self):
        cond, _, _ = validate_definition(
            "schedule", {"daily": "09:30"}, "notify", {})
        self.assertEqual(cond, {"cron": "30 9 * * *"})
        cond, _, _ = validate_definition(
            "schedule", {"weekly": "MON 09:30"}, "notify", {})
        self.assertEqual(cond, {"cron": "30 9 * * MON"})
        with self.assertRaises(TriggerError):
            validate_definition("schedule", {"daily": "9:30"},
                                "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition("schedule", {"daily": "25:00"},
                                "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition("schedule", {"weekly": "FUNDAY 09:30"},
                                "notify", {})

    def test_once(self):
        future = time.time() + 3600
        cond, _, _ = validate_definition(
            "schedule", {"once": future}, "notify", {})
        self.assertEqual(cond, {"once": future})
        with self.assertRaises(TriggerError):
            validate_definition("schedule", {"once": time.time() - 10},
                                "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition("schedule", {"once": "someday"},
                                "notify", {})

    def test_file(self):
        cond, _, _ = validate_definition(
            "file", {"path": "/tmp/x.log"}, "notify", {})
        self.assertEqual(cond, {"path": "/tmp/x.log", "on": "change"})
        with self.assertRaises(TriggerError):
            validate_definition("file", {}, "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition("file", {"path": "/tmp/x", "on": "bogus"},
                                "notify", {})

    def test_price(self):
        cond, _, _ = validate_definition(
            "price", {"symbol": "btc", "op": "lt", "value": 60000},
            "notify", {})
        self.assertEqual(cond["symbol"], "BTC")
        self.assertEqual(cond["op"], "lt")
        with self.assertRaises(TriggerError):
            validate_definition("price", {}, "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition(
                "price", {"symbol": "BTC", "market": "bogus", "op": "lt",
                          "value": 1}, "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition(
                "price", {"symbol": "BTC", "op": "sideways", "value": 1},
                "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition(
                "price", {"symbol": "BTC", "op": "lt"}, "notify", {})

    def test_message(self):
        cond, _, _ = validate_definition(
            "message", {"pattern": "hello (.*)", "chat": "telegram:1"},
            "notify", {})
        self.assertEqual(cond["pattern"], "hello (.*)")
        with self.assertRaises(TriggerError):
            validate_definition("message", {}, "notify", {})
        with self.assertRaises(TriggerError):
            validate_definition("message", {"pattern": "(["}, "notify", {})

    def test_webhook_secret_optional(self):
        cond, _, _ = validate_definition("webhook", {}, "notify", {})
        self.assertEqual(cond, {})
        cond, _, _ = validate_definition(
            "webhook", {"secret": "s3cr3t"}, "notify", {})
        self.assertEqual(cond, {"secret": "s3cr3t"})

    def test_action_params(self):
        # message action
        with self.assertRaises(TriggerError):
            validate_definition("webhook", {}, "message", {"text": "hi"})
        with self.assertRaises(TriggerError):
            validate_definition(
                "webhook", {}, "message", {"chat": "telegram:1"})
        # command action
        with self.assertRaises(TriggerError):
            validate_definition("webhook", {}, "command", {})
        with self.assertRaises(TriggerError):
            validate_definition(
                "webhook", {}, "command",
                {"argv": ["a"], "command": "b"})
        with self.assertRaises(TriggerError):
            validate_definition(
                "webhook", {}, "command", {"argv": "notalist"})
        # mission action
        with self.assertRaises(TriggerError):
            validate_definition("webhook", {}, "mission", {})
        with self.assertRaises(TriggerError):
            validate_definition(
                "webhook", {}, "mission",
                {"goal": "x", "max_iterations": 0})
        # notify action needs nothing
        validate_definition("webhook", {}, "notify", {})

    def test_cooldown(self):
        _, _, cd = validate_definition("webhook", {}, "notify", {},
                                       cooldown_s=60)
        self.assertEqual(cd, 60.0)
        with self.assertRaises(TriggerError):
            validate_definition("webhook", {}, "notify", {},
                                cooldown_s=-1)

    def test_schedule_plan(self):
        self.assertEqual(schedule_plan({"cron": "0 9 * * *"}),
                         ("cron", {"cron_expr": "0 9 * * *"}))
        kind, plan = schedule_plan(
            normalize_schedule_condition({"interval": "1h"}))
        self.assertEqual((kind, plan), ("cron", {"cron_expr": "0 */1 * * *"}))
        kind, plan = schedule_plan({"once": 1234.0})
        self.assertEqual((kind, plan), ("once", {"run_at": 1234.0}))


class CrudTests(unittest.TestCase):
    def setUp(self):
        self.eng = _engine()

    def test_add_get_list_remove(self):
        t = self.eng.add("w1", "webhook", {}, "notify", {"title": "hi"})
        self.assertTrue(t.id.startswith("trigger"))
        self.assertEqual(self.eng.get(t.id).name, "w1")
        self.assertEqual(len(self.eng.list()), 1)
        self.assertTrue(self.eng.remove(t.id))
        self.assertIsNone(self.eng.get(t.id))
        self.assertFalse(self.eng.remove(t.id))
        self.assertEqual(self.eng.list(), [])

    def test_add_needs_name(self):
        with self.assertRaises(TriggerError):
            self.eng.add("", "webhook", {}, "notify", {})

    def test_add_invalid_leaves_no_row(self):
        with self.assertRaises(TriggerError):
            self.eng.add("bad", "schedule", {"cron": "nope"}, "notify", {})
        self.assertEqual(self.eng.list(), [])

    def test_enable_disable(self):
        t = self.eng.add("w", "webhook", {}, "notify", {})
        self.eng.set_enabled(t.id, False)
        self.assertFalse(self.eng.get(t.id).enabled)
        self.eng.set_enabled(t.id, True)
        self.assertTrue(self.eng.get(t.id).enabled)
        with self.assertRaises(TriggerError):
            self.eng.set_enabled("trigger_nope", True)

    def test_persistence_across_restart(self):
        tmp = tempfile.mkdtemp(prefix="trig-store-")
        path = os.path.join(tmp, "t.db")
        e1 = TriggerEngine(Database(path), notify_fn=lambda *a: {})
        t = e1.add("persist-me", "webhook", {"secret": "s"},
                   "notify", {"title": "t"})
        e1.store.record(t.id, OUTCOME_FIRED, {"x": 1}, fired=True)
        # brand-new engine on the same file: definitions + history survive
        e2 = TriggerEngine(Database(path), notify_fn=lambda *a: {})
        got = e2.get(t.id)
        self.assertIsNotNone(got)
        self.assertEqual(got.condition, {"secret": "s"})
        self.assertEqual(got.fire_count, 1)
        hist = e2.history(t.id)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["outcome"], OUTCOME_FIRED)


class ScheduleWiringTests(unittest.TestCase):
    def setUp(self):
        self.eng = _engine()

    def _cron_rows(self):
        return self.eng.db.query(
            "SELECT c.task_id, c.cron_expr, t.action "
            "FROM cron_jobs c JOIN scheduled_tasks t "
            "ON c.task_id = t.task_id")

    def test_cron_wired_into_scheduler(self):
        t = self.eng.add("morn", "schedule", {"cron": "0 9 * * *"},
                         "notify", {"title": "gm"})
        rows = self._cron_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], f"trigger:{t.id}")
        self.assertEqual(rows[0]["cron_expr"], "0 9 * * *")
        self.assertEqual(rows[0]["action"], "__trigger_fire__")
        # handler registered
        self.assertIn("__trigger_fire__",
                      self.eng._ensure_scheduler()._action_handlers)

    def test_scheduler_fire_fires_trigger(self):
        fired = []
        eng = TriggerEngine(
            _db(), notify_fn=lambda trig, ti, bo, en: fired.append(ti) or {})
        t = eng.add("morn", "schedule", {"cron": "0 9 * * *"},
                    "notify", {"title": "gm"})
        asyncio.run(eng._on_scheduler_fire(t.id))
        self.assertEqual(fired, ["gm"])
        self.assertEqual(eng.get(t.id).fire_count, 1)

    def test_disabled_schedule_never_fires(self):
        eng = _engine()
        t = eng.add("morn", "schedule", {"cron": "0 9 * * *"},
                    "notify", {"title": "gm"})
        eng.set_enabled(t.id, False)
        # scheduler rows cleared on disable
        self.assertEqual(
            eng.db.query("SELECT * FROM cron_jobs WHERE task_id = ?",
                         (f"trigger:{t.id}",)), [])
        # and even a stale queued fire is refused at fire time
        result = eng._fire_by_id(t.id, {"source": "schedule"})
        self.assertFalse(result["fired"])
        self.assertEqual(result["outcome"], OUTCOME_SKIPPED)
        hist = eng.history(t.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_SKIPPED)

    def test_reenable_rewires(self):
        eng = _engine()
        t = eng.add("morn", "schedule", {"interval": "1h"},
                    "notify", {"title": "gm"})
        eng.set_enabled(t.id, False)
        eng.set_enabled(t.id, True)
        rows = eng.db.query(
            "SELECT cron_expr FROM cron_jobs WHERE task_id = ?",
            (f"trigger:{t.id}",))
        self.assertEqual(rows[0]["cron_expr"], "0 */1 * * *")

    def test_remove_unwires(self):
        eng = _engine()
        t = eng.add("morn", "schedule", {"cron": "0 9 * * *"},
                    "notify", {})
        eng.remove(t.id)
        self.assertEqual(
            eng.db.query("SELECT * FROM scheduled_tasks WHERE task_id = ?",
                         (f"trigger:{t.id}",)), [])

    def test_once_wired(self):
        t = self.eng.add("once", "schedule",
                         {"once": time.time() + 7200}, "notify", {})
        rows = self.eng.db.query(
            "SELECT task_id, action FROM scheduled_tasks WHERE task_id = ?",
            (f"trigger:{t.id}",))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "__trigger_fire__")


class FileSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="trig-file-")
        self.path = os.path.join(self.tmp, "watched.txt")
        self.eng = _engine()

    def _write(self, text):
        with open(self.path, "w") as fh:
            fh.write(text)

    def test_change_baseline_then_fire(self):
        self._write("v1")
        t = self.eng.add("fw", "file", {"path": self.path}, "notify", {})
        # first tick: baseline established, no fire
        summary = self.eng.tick()
        self.assertEqual(summary["fired"], 0)
        self.assertEqual(self.eng.get(t.id).fire_count, 0)
        # unchanged: no fire, outcome recorded
        summary = self.eng.tick()
        self.assertEqual(summary["fired"], 0)
        # change: fires
        self._write("v2")
        summary = self.eng.tick()
        self.assertEqual(summary["fired"], 1)
        self.assertEqual(self.eng.get(t.id).fire_count, 1)
        outcomes = [h["outcome"] for h in self.eng.history(t.id)]
        self.assertIn(OUTCOME_NO_MATCH, outcomes)
        self.assertIn(OUTCOME_FIRED, outcomes)

    def test_create_mode(self):
        t = self.eng.add("fc", "file",
                         {"path": self.path, "on": "create"}, "notify", {})
        self.eng.tick()  # missing: no fire
        self.assertEqual(self.eng.get(t.id).fire_count, 0)
        self._write("hello")
        self.eng.tick()  # created: fires
        self.assertEqual(self.eng.get(t.id).fire_count, 1)

    def test_delete_mode(self):
        self._write("v1")
        t = self.eng.add("fd", "file",
                         {"path": self.path, "on": "delete"}, "notify", {})
        self.eng.tick()  # baseline
        self.assertEqual(self.eng.get(t.id).fire_count, 0)
        os.unlink(self.path)
        self.eng.tick()  # deleted: fires
        self.assertEqual(self.eng.get(t.id).fire_count, 1)

    def test_missing_path_never_fires(self):
        t = self.eng.add("fm", "file",
                         {"path": os.path.join(self.tmp, "nope.txt")},
                         "notify", {})
        for _ in range(3):
            self.eng.tick()
        self.assertEqual(self.eng.get(t.id).fire_count, 0)


class PriceSourceTests(unittest.TestCase):
    def setUp(self):
        self.eng = _engine()

    def _with_quote(self, price):
        return mock.patch(
            "nomorals.triggers.sources.market_data.quote",
            return_value={"symbol": "BTC", "price": price,
                          "currency": "USD", "source": "test"})

    def test_threshold_fires_when_already_below(self):
        # watchers semantics: lt fires on the first check if already true
        t = self.eng.add("pd", "price",
                         {"symbol": "BTC", "op": "lt", "value": 60000},
                         "notify", {})
        with self._with_quote(59000):
            summary = self.eng.tick()
        self.assertEqual(summary["fired"], 1)
        self.assertEqual(self.eng.get(t.id).fire_count, 1)

    def test_threshold_no_fire_above(self):
        t = self.eng.add("pd", "price",
                         {"symbol": "BTC", "op": "lt", "value": 60000},
                         "notify", {})
        with self._with_quote(61000):
            summary = self.eng.tick()
        self.assertEqual(summary["fired"], 0)
        hist = self.eng.history(t.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_NO_MATCH)

    def test_changed_by_pct_baselines_then_fires(self):
        t = self.eng.add("pc", "price",
                         {"symbol": "BTC", "op": "changed_by_pct",
                          "value": 5}, "notify", {})
        with self._with_quote(60000):
            self.eng.tick()
        self.assertEqual(self.eng.get(t.id).fire_count, 0)
        with self._with_quote(63000):  # +5%
            self.eng.tick()
        self.assertEqual(self.eng.get(t.id).fire_count, 1)

    def test_quote_failure_recorded_not_raised(self):
        t = self.eng.add("pq", "price",
                         {"symbol": "BTC", "op": "lt", "value": 1},
                         "notify", {})
        other = self.eng.add("ok", "webhook", {}, "notify", {})
        with mock.patch(
                "nomorals.triggers.sources.market_data.quote",
                return_value=None):
            summary = self.eng.tick()  # must not raise
        self.assertEqual(summary["errors"], 1)
        hist = self.eng.history(t.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_ERROR)
        self.assertIn("no quote", hist[0]["error"])
        # other triggers unaffected
        self.assertEqual(self.eng.get(other.id).fire_count, 0)


class MessageSourceTests(unittest.TestCase):
    def setUp(self):
        self.fired = []
        self.eng = TriggerEngine(
            _db(),
            notify_fn=lambda trig, ti, bo, en: self.fired.append(ti) or {})

    def test_regex_match_fires(self):
        t = self.eng.add("mm", "message", {"pattern": r"deploy (failed|ok)"},
                         "notify", {"title": "deploy event"})
        fired = self.eng.on_message("deploy failed on prod", "telegram:1")
        self.assertEqual(fired, [t.id])
        self.assertEqual(self.fired, ["deploy event"])
        hist = self.eng.history(t.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_FIRED)
        self.assertIn("failed", hist[0]["detail"]["evidence"]["matched"])

    def test_no_match_recorded(self):
        t = self.eng.add("mm", "message", {"pattern": r"^alert:"},
                         "notify", {})
        fired = self.eng.on_message("just chatting", "telegram:1")
        self.assertEqual(fired, [])
        hist = self.eng.history(t.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_NO_MATCH)

    def test_chat_filter(self):
        t = self.eng.add("mm", "message",
                         {"pattern": "hi", "chat": "telegram:1"}, "notify",
                         {})
        self.assertEqual(self.eng.on_message("hi", "telegram:2"), [])
        self.assertEqual(len(self.eng.on_message("hi", "telegram:1")), 1)
        self.assertEqual(self.eng.get(t.id).fire_count, 1)

    def test_sender_filter(self):
        t = self.eng.add("mm", "message",
                         {"pattern": "hi", "sender": "alice"}, "notify",
                         {})
        self.assertEqual(
            self.eng.on_message("hi", "telegram:1", sender="bob"), [])
        self.assertEqual(
            len(self.eng.on_message("hi", "telegram:1", sender="alice")), 1)


class WebhookTests(unittest.TestCase):
    def setUp(self):
        self.eng = _engine()

    def test_fire_ok(self):
        t = self.eng.add("wh", "webhook", {"secret": "s3cr3t"},
                         "notify", {"title": "hook"})
        result = self.eng.fire_webhook(
            t.id, payload={"a": 1}, secret="s3cr3t")
        self.assertTrue(result["fired"])
        hist = self.eng.history(t.id)
        self.assertEqual(hist[0]["detail"]["evidence"]["payload"], {"a": 1})

    def test_unknown_id(self):
        with self.assertRaises(TriggerError):
            self.eng.fire_webhook("trigger_nope")

    def test_wrong_source(self):
        t = self.eng.add("w", "message", {"pattern": "x"}, "notify", {})
        with self.assertRaises(TriggerError):
            self.eng.fire_webhook(t.id)

    def test_bad_secret(self):
        t = self.eng.add("wh", "webhook", {"secret": "s3cr3t"},
                         "notify", {})
        with self.assertRaises(TriggerError):
            self.eng.fire_webhook(t.id, secret="wrong")
        self.assertEqual(self.eng.get(t.id).fire_count, 0)

    def test_no_secret_configured_any_secret_ok(self):
        t = self.eng.add("wh", "webhook", {}, "notify", {})
        result = self.eng.fire_webhook(t.id)
        self.assertTrue(result["fired"])

    def test_disabled_webhook_fails_fast(self):
        t = self.eng.add("wh", "webhook", {}, "notify", {})
        self.eng.set_enabled(t.id, False)
        with self.assertRaises(TriggerError):
            self.eng.fire_webhook(t.id)


class ActionTests(unittest.TestCase):
    def test_notify_uses_injected_fn(self):
        seen = []
        eng = TriggerEngine(
            _db(),
            notify_fn=lambda trig, title, body, en: seen.append(
                (trig.id, title, body)) or {"sent": True})
        t = eng.add("n", "webhook", {}, "notify",
                    {"title": "T", "body": "B"})
        result = eng.manual_fire(t.id)
        self.assertTrue(result["fired"])
        self.assertEqual(seen[0][1:], ("T", "B"))

    def test_notify_defaults_title_to_trigger_name(self):
        seen = []
        eng = TriggerEngine(
            _db(),
            notify_fn=lambda trig, title, body, en: seen.append(title) or {})
        t = eng.add("my-trigger", "webhook", {}, "notify", {})
        eng.manual_fire(t.id)
        self.assertEqual(seen, ["my-trigger"])

    def test_message_action(self):
        sent = []
        eng = TriggerEngine(
            _db(), send_message=lambda chat, text: sent.append((chat, text)))
        t = eng.add("m", "webhook", {}, "message",
                    {"chat": "telegram:9", "text": "hello there"})
        result = eng.manual_fire(t.id)
        self.assertTrue(result["fired"])
        self.assertEqual(sent, [("telegram:9", "hello there")])

    def test_message_action_no_sender_fails_fast(self):
        eng = _engine()  # no send_message bound
        t = eng.add("m", "webhook", {}, "message",
                    {"chat": "telegram:9", "text": "hi"})
        result = eng.manual_fire(t.id)
        self.assertFalse(result["fired"])
        self.assertEqual(result["outcome"], OUTCOME_ERROR)
        self.assertIn("no sender", result["error"])
        hist = eng.history(t.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_ERROR)

    def test_command_action_injected(self):
        calls = []
        eng = TriggerEngine(
            _db(),
            run_command=lambda argv, to: calls.append((argv, to)) or
            {"exit_code": 0})
        t = eng.add("c", "webhook", {}, "command",
                    {"argv": ["status", "--json"]})
        result = eng.manual_fire(t.id)
        self.assertTrue(result["fired"])
        argv, timeout = calls[0]
        self.assertEqual(argv[:3], [sys.executable, "-m", "nomorals"])
        self.assertEqual(argv[3:], ["status", "--json"])

    def test_command_string_split(self):
        argv = build_command_argv({"command": "finance quote BTC"})
        self.assertEqual(argv[3:], ["finance", "quote", "BTC"])

    def test_command_nonzero_exit_is_trigger_error(self):
        with self.assertRaises(TriggerError) as ctx:
            default_run_command(
                [sys.executable, "-c", "import sys; sys.exit(3)"], 30)
        self.assertIn("exited 3", str(ctx.exception))

    def test_command_timeout(self):
        with self.assertRaises(TriggerError) as ctx:
            default_run_command(
                [sys.executable, "-c", "import time; time.sleep(5)"], 0.2)
        self.assertIn("timed out", str(ctx.exception))

    def test_mission_action_injected(self):
        started = []
        eng = TriggerEngine(
            _db(),
            start_mission=lambda goal, n, en: started.append((goal, n)) or
            {"mission_id": "m1"})
        t = eng.add("mi", "webhook", {}, "mission",
                    {"goal": "do the thing", "max_iterations": 3})
        result = eng.manual_fire(t.id)
        self.assertTrue(result["fired"])
        self.assertEqual(started, [("do the thing", 3)])
        self.assertEqual(result["result"]["mission_id"], "m1")

    def test_mission_action_no_context_fails(self):
        from nomorals.triggers.actions import default_start_mission
        eng = SimpleNamespace(context=None)
        with self.assertRaises(TriggerError):
            default_start_mission("goal", 8, eng)


class ResilienceTests(unittest.TestCase):
    def test_one_bad_action_does_not_kill_others(self):
        def flaky(trigger, title, body, eng):
            if trigger.name == "bad":
                raise RuntimeError("boom")
            return {"sent": True}

        eng = TriggerEngine(_db(), notify_fn=flaky)
        t1 = eng.add("bad", "webhook", {}, "notify", {})
        t2 = eng.add("good", "webhook", {}, "notify", {})
        r1 = eng.manual_fire(t1.id)
        self.assertEqual(r1["outcome"], OUTCOME_ERROR)
        self.assertIn("RuntimeError", r1["error"])
        r2 = eng.manual_fire(t2.id)
        self.assertTrue(r2["fired"])
        hist = eng.history(t1.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_ERROR)
        self.assertEqual(hist[0]["trigger_id"], t1.id)

    def test_error_history_carries_trigger_id(self):
        eng = TriggerEngine(_db(), notify_fn=lambda *a: 1 / 0)
        t = eng.add("x", "webhook", {}, "notify", {})
        eng.manual_fire(t.id)
        hist = eng.history(t.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_ERROR)
        self.assertIn("ZeroDivisionError", hist[0]["error"])

    def test_cooldown_skips_and_records(self):
        eng = _engine()
        t = eng.add("cd", "webhook", {}, "notify", {}, cooldown_s=3600)
        r1 = eng.manual_fire(t.id)
        self.assertTrue(r1["fired"])
        r2 = eng.manual_fire(t.id)
        self.assertFalse(r2["fired"])
        self.assertEqual(r2["outcome"], OUTCOME_SKIPPED)
        self.assertEqual(r2["reason"], "cooldown")
        self.assertEqual(eng.get(t.id).fire_count, 1)
        hist = eng.history(t.id)
        self.assertEqual(hist[0]["outcome"], OUTCOME_SKIPPED)

    def test_disabled_manual_fire_fails_fast(self):
        eng = _engine()
        t = eng.add("d", "webhook", {}, "notify", {})
        eng.set_enabled(t.id, False)
        with self.assertRaises(TriggerError):
            eng.manual_fire(t.id)


class MessageHookTests(unittest.TestCase):
    def test_hook_no_engine_is_noop(self):
        ctx = SimpleNamespace(extras={})
        self.assertEqual(message_hook(ctx, "hi", "telegram:1"), [])

    def test_hook_fires_attached_engine(self):
        fired = []
        eng = TriggerEngine(
            _db(),
            notify_fn=lambda trig, ti, bo, en: fired.append(ti) or {})
        ctx = SimpleNamespace(extras={})
        attach(eng, ctx)
        t = eng.add("mh", "message", {"pattern": "ping"}, "notify",
                    {"title": "pong"})
        self.assertEqual(message_hook(ctx, "ping!", "telegram:1"), [t.id])
        self.assertEqual(fired, ["pong"])

    def test_hook_exotic_context(self):
        eng = _engine()
        ctx = SimpleNamespace()  # no extras dict
        attach(eng, ctx)
        self.assertEqual(ctx.trigger_engine, eng)
        self.assertEqual(message_hook(ctx, "hi", "telegram:1"), [])

    def test_engine_key_constant(self):
        self.assertEqual(ENGINE_KEY, "trigger_engine")


class WebhookRouteTests(unittest.TestCase):
    def _server(self):
        routes = {}

        class FakeServer:
            def route(self, method, path):
                def deco(fn):
                    routes[(method, path)] = fn
                    return fn
                return deco
        return FakeServer(), routes

    def test_route_registered_and_fires(self):
        from nomorals.triggers.webhook import (
            WEBHOOK_PATH, register_trigger_routes)
        server, routes = self._server()
        ctx = SimpleNamespace(db=_db(), extras={})
        register_trigger_routes(server, ctx)
        self.assertIn(("POST", WEBHOOK_PATH), routes)
        handler = routes[("POST", WEBHOOK_PATH)]
        # unknown id -> ok False
        out = handler({"trigger_id": "trigger_nope"}, {})
        self.assertEqual(out, {"ok": False, "error": mock.ANY})
        self.assertIn("unknown trigger", out["error"])
        # real trigger fires
        from nomorals.triggers import TriggerEngine as TE
        eng = TE(ctx.db, notify_fn=lambda *a: {"sent": True})
        t = eng.add("wh", "webhook", {"secret": "s"}, "notify", {})
        out = handler({"trigger_id": t.id, "secret": "s",
                       "payload": {"k": "v"}}, {})
        self.assertTrue(out["ok"])
        self.assertTrue(out["fired"])
        # bad secret rejected
        out = handler({"trigger_id": t.id, "secret": "wrong"}, {})
        self.assertFalse(out["ok"])
        self.assertIn("secret", out["error"])


class LifecycleTests(unittest.TestCase):
    def test_start_needs_file_db(self):
        eng = _engine()  # :memory:
        with self.assertRaises(TriggerError):
            eng.start()

    def test_start_stop_poll_thread(self):
        tmp = tempfile.mkdtemp(prefix="trig-life-")
        path = os.path.join(tmp, "t.db")
        fired = []
        eng = TriggerEngine(
            Database(path), poll_interval=0.2,
            notify_fn=lambda trig, ti, bo, en: fired.append(ti) or {})
        watched = os.path.join(tmp, "w.txt")
        with open(watched, "w") as fh:
            fh.write("v1")
        eng.add("lf", "file", {"path": watched}, "notify",
                {"title": "changed!"})
        eng.start()
        try:
            time.sleep(0.5)  # baseline tick
            self.assertEqual(fired, [])
            with open(watched, "w") as fh:
                fh.write("v2")
            deadline = time.time() + 5
            while not fired and time.time() < deadline:
                time.sleep(0.1)
            self.assertEqual(fired, ["changed!"])
        finally:
            eng.stop()
        # second stop is a no-op
        eng.stop()


class StoreTests(unittest.TestCase):
    def test_history_limit_and_purge(self):
        store = TriggerStore(_db())
        from nomorals.triggers.models import Trigger
        trg = Trigger(id="trigger_1", name="t", source="webhook",
                      action="notify")
        store.save(trg)
        for i in range(5):
            store.record("trigger_1", OUTCOME_NO_MATCH, {"i": i})
        rows = store.history("trigger_1", limit=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["detail"]["i"], 4)  # newest first
        # purge everything older than now
        removed = store.purge_old(ttl_s=-1)
        self.assertEqual(removed, 5)
        self.assertEqual(store.history("trigger_1"), [])

    def test_count(self):
        eng = _engine()
        self.assertEqual(eng.store.count(), 0)
        eng.add("a", "webhook", {}, "notify", {})
        eng.add("b", "webhook", {}, "notify", {})
        self.assertEqual(eng.store.count(), 2)


if __name__ == "__main__":
    unittest.main()
