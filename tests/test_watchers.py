"""Acceptance tests for Prompt 03 — Watchers: background monitors + smart alerts.

Covers the spec's contract: structured conditions (parsed once, evaluated
deterministically); NL watcher creation with the ambiguity policy; the six
watcher kinds; smart alerting (cooldowns, digest batching, quiet hours, flap
suppression); scheduler integration (single sweeper job); restart catch-up;
watcher history + the alert audit log; the read-only security guarantee; and
MonitorAgent API stability after the Prompt 03 extraction.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import watchers as W
from nomorals.agents.watchers import (
    AlertEngine,
    Condition,
    Watcher,
    WatcherAgent,
    WatcherContext,
    WatcherStore,
    ensure_sweeper_job,
    evaluate_condition,
    parse_condition,
)
from nomorals.storage.db import Database


# ── fakes ────────────────────────────────────────────────────────────────────

class _FakeNotifier:
    """Stands in for Notifier: records publishes instead of sending."""

    def __init__(self):
        self.sent: list[dict] = []

    def publish(self, kind, title, body="", **kw):
        self.sent.append({"kind": kind, "title": title, "body": body,
                          "critical": kw.get("critical", False)})
        return {"id": f"n{len(self.sent)}", "kind": kind, "title": title,
                "delivered": True}


class _Outcome:
    def __init__(self, ok, value=None, error=None):
        self.ok = ok
        self.value = value
        self.error = error


class _FakeRegistry:
    """Minimal tool registry: name → callable. Returns Outcome-shaped objects."""

    def __init__(self, tools):
        self._tools = tools

    def call(self, name, **kwargs):
        fn = self._tools.get(name)
        if fn is None:
            return _Outcome(False, error=f"unknown tool {name!r}")
        try:
            return _Outcome(True, value=fn(**kwargs))
        except Exception as exc:  # noqa: BLE001
            return _Outcome(False, error=str(exc))


def _test_spec():
    from nomorals.agents.role_specs import RoleSpec

    return RoleSpec(
        name="watcher-test",
        description="test",
        system_prompt="test",
        tool_allowlist=("web_search", "deals", "fake_tool", "fake_quote"),
        read_only=True,
    )


class _Ctx:
    """Harness: temp DB (migrated), controllable clock, fake tools/notifier."""

    def __init__(self, tools=None):
        self.tmp = tempfile.mkdtemp(prefix="watchers-test-")
        self.db_path = os.path.join(self.tmp, "test.db")
        self.db = Database(self.db_path)
        self.db.migrate()
        self.now = [1_700_000_000.0]
        self.notifier = _FakeNotifier()
        self.tools = tools or {}
        self.registry = _FakeRegistry(self.tools)
        # settings.workspace_dir points at the temp dir so sandboxed file
        # access (MonitorAgent, FileKind) stays inside the test sandbox
        self.settings = SimpleNamespace(workspace_dir=self.tmp, timezone="UTC")
        self.context = SimpleNamespace(db=self.db, settings=self.settings,
                                       gateway=None, tools=self.registry)

    def agent(self):
        a = WatcherAgent(self.context, notifier=self.notifier,
                         registry=self.registry,
                         now=lambda: self.now[0])
        # short digest window so digest tests don't need 6h jumps
        a._engine = AlertEngine(a.store, self.context,
                                notifier=self.notifier,
                                digest_window_s=600.0)
        return a

    def advance(self, seconds):
        self.now[0] += seconds

    def close(self):
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)


def _allow_test_tools(test):
    """Patch the read-only gate to allow the harness's fake tools."""
    return patch.object(W, "_read_only_spec", lambda: _test_spec())


# ── structured conditions ────────────────────────────────────────────────────

class ConditionTests(unittest.TestCase):
    def test_threshold_ops(self):
        for op, val, hit in [("lt", 60, True), ("lt", 40, False),
                             ("lte", 50, True), ("gt", 40, True),
                             ("gte", 50, True), ("eq", 50, True),
                             ("ne", 50, False)]:
            c = parse_condition({"op": op, "field": "price", "value": val})
            trig, _ = evaluate_condition(c, old=None,
                                         new={"price": 50})
            self.assertEqual(trig, hit, f"{op} {val}")

    def test_threshold_evaluates_on_first_check(self):
        # already-below-threshold on the first check still alerts
        c = parse_condition({"op": "lt", "field": "price", "value": 60000})
        trig, note = evaluate_condition(c, old=None,
                                        new={"price": 59000})
        self.assertTrue(trig)
        self.assertIn("59000", note)

    def test_changed_needs_baseline(self):
        c = parse_condition({"op": "changed", "field": "value"})
        trig, note = evaluate_condition(c, old=None, new="x")
        self.assertFalse(trig)
        self.assertIn("baseline", note)
        trig, _ = evaluate_condition(c, old="x", new="y")
        self.assertTrue(trig)
        trig, _ = evaluate_condition(c, old="x", new="x")
        self.assertFalse(trig)

    def test_changed_by_pct(self):
        c = parse_condition({"op": "changed_by_pct", "field": "price",
                             "value": 5})
        trig, note = evaluate_condition(c, old={"price": 100},
                                        new={"price": 106})
        self.assertTrue(trig)
        self.assertIn("6.0%", note)
        trig, _ = evaluate_condition(c, old={"price": 100},
                                     new={"price": 102})
        self.assertFalse(trig)

    def test_contains_and_matches(self):
        c = parse_condition({"op": "contains", "field": "title",
                             "value": "release"})
        trig, _ = evaluate_condition(c, old=None,
                                     new={"title": "v2 release notes"})
        self.assertTrue(trig)
        c = parse_condition({"op": "matches", "field": "title",
                             "value": r"v\d+"})
        trig, _ = evaluate_condition(c, old=None,
                                     new={"title": "v2 release notes"})
        self.assertTrue(trig)

    def test_bad_condition_fails_fast(self):
        with self.assertRaises(ValueError):
            parse_condition({"op": "explode", "field": "x"})


# ── natural-language creation ────────────────────────────────────────────────

class ParseTests(unittest.TestCase):
    def setUp(self):
        self.h = _Ctx()

    def tearDown(self):
        self.h.close()

    def test_parse_btc(self):
        agent = self.h.agent()
        parsed = agent.parse("tell me if BTC drops below $60000")
        self.assertTrue(parsed["ok"])
        d = parsed["draft"]
        self.assertEqual(d["kind"], "price")
        self.assertEqual(d["target"]["source"], "market")
        self.assertEqual(d["target"]["symbol"], "BTC")
        # stored structured, not free text
        self.assertEqual(d["condition"],
                         {"op": "lt", "field": "price", "value": 60000.0})
        self.assertIn("BTC", parsed["echo"])
        self.assertIn("60000", parsed["echo"])

    def test_add_btc_creates_structured_watcher(self):
        agent = self.h.agent()
        res = agent.add("tell me if BTC drops below $60000")
        self.assertTrue(res["ok"])
        w = res["watcher"]
        self.assertEqual(w["condition"]["op"], "lt")
        self.assertEqual(w["condition"]["value"], 60000.0)
        row = self.h.db.query_one("SELECT condition FROM watchers WHERE id=?",
                                  (w["id"],))
        stored = json.loads(row["condition"])
        self.assertEqual(stored["op"], "lt")  # structured in the DB

    def test_parse_jumia_price(self):
        agent = self.h.agent()
        parsed = agent.parse(
            "watch the Jumia price of Sony WH-1000XM5 and tell me if it "
            "drops below ₦150000")
        self.assertTrue(parsed["ok"])
        d = parsed["draft"]
        self.assertEqual(d["kind"], "price")
        self.assertEqual(d["target"]["source"], "deals")
        self.assertIn("sony", d["target"]["query"].lower())
        self.assertEqual(d["condition"]["value"], 150000.0)

    def test_parse_repo(self):
        agent = self.h.agent()
        parsed = agent.parse(
            "watch github.com/Oluwacutyp/nomorals-2.0 for new releases")
        self.assertTrue(parsed["ok"])
        d = parsed["draft"]
        self.assertEqual(d["kind"], "repo")
        self.assertEqual(d["target"]["owner"], "Oluwacutyp")
        self.assertEqual(d["target"]["repo"], "nomorals-2.0")
        self.assertEqual(d["target"]["feed"], "releases")

    def test_parse_keyword(self):
        agent = self.h.agent()
        parsed = agent.parse("tell me about news about electric cars")
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["draft"]["kind"], "keyword")
        self.assertIn("electric cars",
                      parsed["draft"]["target"]["keywords"])

    def test_parse_url(self):
        agent = self.h.agent()
        parsed = agent.parse("watch https://example.com/prices for changes")
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["draft"]["kind"], "url")
        self.assertEqual(parsed["draft"]["target"]["url"],
                         "https://example.com/prices")

    def test_ambiguous_price_asks_one_question(self):
        agent = self.h.agent()
        res = agent.add("watch the price of headphones")
        self.assertFalse(res.get("ok"))
        self.assertTrue(res.get("needs_clarification"))
        self.assertIn("threshold", res["question"])
        # nothing was created on a guess
        self.assertEqual(agent.list(), [])

    def test_gibberish_asks(self):
        agent = self.h.agent()
        res = agent.add("blorple the wumpus")
        self.assertFalse(res.get("ok"))
        self.assertTrue(res.get("needs_clarification"))


# ── smart alerting ───────────────────────────────────────────────────────────

class _FakeHttpResp:
    def __init__(self, status, body):
        self.status = status
        self.body = body
        self.text = ""


class _FakeHttpClient:
    """Stand-in for nomorals.core.http.HttpClient (price API)."""
    prices = [59000.0]

    def __init__(self, *a, **k):
        pass

    def get(self, url, headers=None, timeout=None):
        price = _FakeHttpClient.prices[0]
        body = json.dumps(
            {"bitcoin": {"usd": price}}).encode("utf-8")
        return _FakeHttpResp(200, body)


class AlertingTests(unittest.TestCase):
    def setUp(self):
        self.h = _Ctx()

    def tearDown(self):
        self.h.close()

    def _market_watcher(self, severity="important", **kw):
        agent = self.h.agent()
        res = agent.add("tell me if BTC drops below $60000",
                        severity=severity, **kw)
        self.assertTrue(res["ok"])
        return agent, res["watcher"]["id"]

    def test_threshold_cross_sends_one_alert_then_cooldown(self):
        agent, wid = self._market_watcher()
        with patch("nomorals.core.http.HttpClient", _FakeHttpClient):
            out = agent.sweep()
        self.assertEqual(out["checked"], 1)
        self.assertEqual(len(self.h.notifier.sent), 1)
        self.assertIn("BTC", self.h.notifier.sent[0]["title"])
        # price keeps falling but the cooldown holds: no second alert.
        # (advance past the 1h interval but stay inside the 6h cooldown)
        _FakeHttpClient.prices = [58000.0, 57000.0]
        with patch("nomorals.core.http.HttpClient", _FakeHttpClient):
            self.h.advance(3700)
            out = agent.sweep()
            self.assertEqual(out["checked"], 1)
            self.h.advance(3700)
            out = agent.sweep()
            self.assertEqual(out["checked"], 1)
        self.assertEqual(len(self.h.notifier.sent), 1)
        # the audit log records the suppression ("why didn't you tell me")
        log = agent.alert_log(wid)
        statuses = [a["status"] for a in log]
        self.assertIn("sent", statuses)
        self.assertIn("suppressed", statuses)

    def test_info_hits_batch_into_one_digest(self):
        agent = self.h.agent()
        ids = []
        for i in range(3):
            w = Watcher(id=f"winfo{i}", name=f"info watcher {i}",
                        kind="condition",
                        target={"tool": "fake_tool", "args": {}},
                        condition=Condition(op="eq", field="v", value=1),
                        interval_s=60, severity="info", created_at=self.h.now[0])
            agent.store.create(w)
            ids.append(w.id)
        self.h.tools["fake_tool"] = lambda **k: {"v": 1}
        # steady state: a digest just went out, so the window isn't due yet
        agent.store.set_state("last_digest_ts", str(self.h.now[0]))
        with _allow_test_tools(self):
            agent.sweep()
        # three info hits → zero immediate messages, three held
        self.assertEqual(len(self.h.notifier.sent), 0)
        held = agent.store.held_alerts()
        self.assertEqual(len(held), 3)
        # no more hits; past the digest window → ONE digest for all three
        self.h.tools["fake_tool"] = lambda **k: {"v": 0}
        self.h.advance(601)
        with _allow_test_tools(self):
            out = agent.sweep()
        self.assertEqual(out["digest"]["action"], "digested")
        self.assertEqual(out["digest"]["count"], 3)
        self.assertEqual(len(self.h.notifier.sent), 1)
        body = self.h.notifier.sent[0]["body"]
        for i in range(3):
            self.assertIn(f"info watcher {i}", body)
        # held rows are now marked digested in the audit log
        self.assertEqual(agent.store.held_alerts(), [])

    def test_flapping_watcher_auto_pauses_with_single_notice(self):
        agent = self.h.agent()
        w = Watcher(id="wflap", name="flappy", kind="condition",
                    target={"tool": "fake_tool", "args": {}},
                    condition=Condition(op="eq", field="v", value=1),
                    interval_s=60, severity="important",
                    created_at=self.h.now[0])
        agent.store.create(w)
        values = [{"v": 1}, {"v": 0}, {"v": 1}, {"v": 0}]
        for i, val in enumerate(values):
            self.h.tools["fake_tool"] = (lambda v: (lambda **k: v))(val)
            with _allow_test_tools(self):
                self.h.advance(61)
                agent.sweep()
        w = agent.store.get("wflap")
        self.assertEqual(w.state, "paused")
        notices = [s for s in self.h.notifier.sent
                   if "paused" in s["title"]]
        self.assertEqual(len(notices), 1)
        self.assertIn("flapping", notices[0]["body"])
        # stays paused: no more checks, no more notices
        sent_before = len(self.h.notifier.sent)
        with _allow_test_tools(self):
            self.h.advance(61)
            out = agent.sweep()
        self.assertEqual(out["checked"], 0)
        self.assertEqual(len(self.h.notifier.sent), sent_before)
        # resume re-arms it
        agent.resume("wflap")
        self.assertEqual(agent.store.get("wflap").state, "active")

    def test_quiet_hours_hold_non_urgent_but_not_urgent(self):
        agent = self.h.agent()
        # 2026-10-01 03:00 UTC — inside 22:00–07:00
        t3am = datetime.datetime(2026, 10, 1, 3, 0,
                                 tzinfo=datetime.timezone.utc).timestamp()
        self.h.now[0] = t3am
        for wid, sev in (("wquiet", "important"), ("wurgent", "urgent")):
            w = Watcher(id=wid, name=f"{sev} watcher", kind="condition",
                        target={"tool": "fake_tool", "args": {}},
                        condition=Condition(op="eq", field="v", value=1),
                        interval_s=60, severity=sev,
                        quiet_hours={"start": "22:00", "end": "07:00",
                                     "tz": "UTC"},
                        created_at=t3am)
            agent.store.create(w)
        self.h.tools["fake_tool"] = lambda **k: {"v": 1}
        agent.store.set_state("last_digest_ts", str(t3am))
        with _allow_test_tools(self):
            agent.sweep()
        titles = [s["title"] for s in self.h.notifier.sent]
        # urgent went out immediately; important was held, not sent
        self.assertTrue(any("urgent watcher" in t for t in titles))
        self.assertFalse(any("important watcher" in t for t in titles))
        held = agent.store.held_alerts()
        self.assertEqual(len(held), 1)
        self.assertIn("quiet hours", held[0]["title"])
        # the digest does NOT fire at 3am either — it's the morning digest
        self.assertEqual(
            len([s for s in self.h.notifier.sent if "digest" in s["title"].lower()]),
            0)
        # 08:00, quiet hours over, no new hits → the morning digest arrives
        t8am = datetime.datetime(2026, 10, 1, 8, 0,
                                 tzinfo=datetime.timezone.utc).timestamp()
        self.h.now[0] = t8am
        self.h.tools["fake_tool"] = lambda **k: {"v": 0}
        with _allow_test_tools(self):
            out = agent.sweep()
        self.assertEqual(out["digest"]["action"], "digested")
        self.assertEqual(out["digest"]["count"], 1)
        digests = [s for s in self.h.notifier.sent
                   if "digest" in s["title"].lower()]
        self.assertEqual(len(digests), 1)
        self.assertIn("important watcher", digests[0]["body"])

    def test_error_streak_notifies_once_then_stays_quiet(self):
        agent = self.h.agent()
        w = Watcher(id="werr", name="erroring", kind="condition",
                    target={"tool": "fake_tool", "args": {}},
                    condition=Condition(op="eq", field="v", value=1),
                    interval_s=60, severity="important",
                    created_at=self.h.now[0])
        agent.store.create(w)

        def _boom(**k):
            raise RuntimeError("connection reset")

        self.h.tools["fake_tool"] = _boom
        with _allow_test_tools(self):
            for _ in range(5):
                self.h.advance(61)
                agent.sweep()
        notices = [s for s in self.h.notifier.sent if "erroring" in s["title"]]
        self.assertEqual(len(notices), 1)  # one notice, then quiet
        # recovery resets the streak
        self.h.tools["fake_tool"] = lambda **k: {"v": 1}
        with _allow_test_tools(self):
            self.h.advance(61)
            agent.sweep()
        self.assertEqual(agent.store.get("werr").error_streak, 0)

    def test_history_and_alert_log(self):
        agent = self.h.agent()
        w = Watcher(id="whist", name="history", kind="condition",
                    target={"tool": "fake_tool", "args": {}},
                    condition=Condition(op="changed", field="value"),
                    interval_s=60, severity="important",
                    created_at=self.h.now[0])
        agent.store.create(w)
        self.h.tools["fake_tool"] = lambda **k: {"v": "a"}
        with _allow_test_tools(self):
            agent.sweep()  # baseline
            self.h.advance(61)
            self.h.tools["fake_tool"] = lambda **k: {"v": "b"}
            agent.sweep()  # change → alert
        hist = agent.history("whist")
        self.assertIsNotNone(hist)
        self.assertEqual(len(hist["checks"]), 2)
        self.assertFalse(hist["checks"][1]["changed"])  # newest first
        self.assertTrue(hist["checks"][0]["changed"])
        log = agent.alert_log("whist")
        self.assertTrue(any(a["status"] == "sent" for a in log))
        row = log[0]
        for key in ("watcher_id", "severity", "channel", "created_at"):
            self.assertIn(key, row)


# ── restart / catch-up ───────────────────────────────────────────────────────

class RestartTests(unittest.TestCase):
    def test_watchers_resume_and_catch_up_once(self):
        h = _Ctx()
        agent = h.agent()
        w = Watcher(id="wrestart", name="restart me", kind="condition",
                    target={"tool": "fake_tool", "args": {}},
                    condition=Condition(op="changed", field="value"),
                    interval_s=60, severity="info",
                    created_at=h.now[0])
        agent.store.create(w)
        h.tools["fake_tool"] = lambda **k: {"v": "a"}
        with _allow_test_tools(self):
            agent.sweep()
        self.assertEqual(len(agent.store.check_history("wrestart")), 1)

        # "restart": brand-new agent over the same DB file, 1h of downtime
        h2now = [h.now[0] + 3600]
        ctx2 = SimpleNamespace(db=Database(h.db_path), settings=None,
                               gateway=None, tools=h.registry)
        agent2 = WatcherAgent(ctx2, notifier=h.notifier,
                             registry=h.registry,
                             now=lambda: h2now[0])
        with _allow_test_tools(self):
            out = agent2.sweep()
        # the missed hour collapses into ONE catch-up check, not 60
        self.assertEqual(out["checked"], 1)
        self.assertEqual(len(agent2.store.check_history("wrestart")), 2)
        # and the watcher is still active afterwards
        self.assertEqual(agent2.store.get("wrestart").state, "active")
        with _allow_test_tools(self):
            out = agent2.sweep()
        self.assertEqual(out["checked"], 0)
        h.close()


# ── read-only security ───────────────────────────────────────────────────────

class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.h = _Ctx()

    def tearDown(self):
        self.h.close()

    def test_condition_watcher_cannot_invoke_write_tool(self):
        # uses the REAL read-only spec — no test patch
        agent = self.h.agent()
        w = Watcher(id="wevil", name="evil", kind="condition",
                    target={"tool": "shell_run", "args": {"cmd": "rm -rf /"}},
                    condition=Condition(op="changed", field="value"),
                    interval_s=60, created_at=self.h.now[0])
        agent.store.create(w)
        called = []
        self.h.tools["shell_run"] = lambda **k: called.append(k) or {}
        out = agent.sweep()  # real spec: no _allow_test_tools patch
        self.assertEqual(out["checked"], 1)
        self.assertEqual(called, [])  # the write tool never ran
        hist = agent.store.check_history("wevil")
        self.assertTrue(hist)
        w = agent.store.get("wevil")
        self.assertGreater(w.error_streak, 0)

    def test_read_only_spec_allows_legit_tools(self):
        agent = self.h.agent()
        w = Watcher(id="wok", name="ok", kind="condition",
                    target={"tool": "web_search", "args": {"query": "x"}},
                    condition=Condition(op="changed", field="value"),
                    interval_s=60, created_at=self.h.now[0])
        agent.store.create(w)
        seen = []
        self.h.tools["web_search"] = lambda **k: seen.append(k) or {"r": []}
        out = agent.sweep()
        self.assertEqual(len(seen), 1)
        self.assertEqual(out["errors"], 0)

    def test_expiry_parsed_and_auto_removed(self):
        agent = self.h.agent()
        res = agent.add("tell me if BTC drops below $60000 for 3 days")
        self.assertTrue(res["ok"])
        w = agent.store.get(res["watcher"]["id"])
        self.assertAlmostEqual(w.expires_at - self.h.now[0], 3 * 86400,
                               delta=1.0)
        self.assertIn("auto-removing", res["echo"])
        # sweep past expiry → watcher auto-removed, reported
        self.h.now[0] += 4 * 86400
        out = agent.sweep()
        self.assertIn(w.id, out["expired_removed"])
        self.assertIsNone(agent.store.get(w.id))

    def test_robots_disallow_skips_without_error(self):
        from unittest.mock import patch
        from nomorals.agents import watchers as W

        agent = self.h.agent()
        w = Watcher(id="wrobots", name="robots", kind="url",
                    target={"url": "https://example.com/page"},
                    condition=Condition(op="changed", field="value"),
                    interval_s=60, created_at=self.h.now[0])
        agent.store.create(w)
        with patch.object(W._robots_cache, "allowed",
                          return_value=False):
            out = agent.sweep()
        self.assertEqual(out["errors"], 0)
        hist = agent.history("wrobots")["checks"]
        self.assertTrue(any("robots.txt" in c["summary"] for c in hist))

    def test_channel_mention_parsed(self):
        agent = self.h.agent()
        res = agent.add("tell me if BTC drops below $60000, alert me only "
                        "on telegram")
        self.assertTrue(res["ok"])
        self.assertEqual(res["watcher"]["channels"], ["telegram"])
        self.assertIn("on telegram", res["echo"])
        res2 = agent.add("watch https://example.com for changes via "
                         "whatsapp and email")
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["watcher"]["channels"],
                         ["whatsapp", "email"])

    def test_channel_subset_restricts_delivery(self):
        from nomorals.agents.notifier import Notifier

        delivered = []

        class _FakeGateway:
            def status(self):
                return {"telegram": {"running_in_session": True},
                        "whatsapp": {"running_in_session": True}}

            def send(self, plat, ref, text):
                delivered.append(plat)
                return SimpleNamespace(ok=True)

        self.h.settings.partner = SimpleNamespace(
            owner_chats="telegram:1,whatsapp:2")
        n = Notifier(self.h.context, gateway=_FakeGateway())
        n.publish("watcher", "chan test", "b", force=True,
                  channels=["telegram"])
        self.assertEqual(delivered, ["telegram"])
        # and the watcher's channel preference flows through _send
        agent = self.h.agent()
        w = Watcher(id="wchan", name="chan watcher", kind="condition",
                    target={"tool": "fake_tool", "args": {}},
                    condition=Condition(op="eq", field="v", value=1),
                    interval_s=60, severity="important",
                    channels=["telegram"], created_at=self.h.now[0])
        agent.store.create(w)
        self.h.tools["fake_tool"] = lambda **k: {"v": 1}
        agent._engine._notifier = n  # noqa: SLF001 — test seam
        with _allow_test_tools(self):
            agent.sweep()
        self.assertEqual(delivered, ["telegram", "telegram"])
        log = agent.alert_log("wchan")
        self.assertTrue(any(a["channel"] == "telegram" and
                            a["status"] == "sent" for a in log))


# ── scheduler + autonomy-loop integration ──────────────────────────────────

class SchedulerTests(unittest.TestCase):
    def test_sweeper_job_is_single_and_idempotent(self):
        from nomorals.agents.scheduler import Scheduler
        from nomorals.agents.watchers import SWEEPER_JOB_NAME

        h = _Ctx()
        first = ensure_sweeper_job(h.context)
        second = ensure_sweeper_job(h.context)
        self.assertTrue(first.get("scheduled"))
        self.assertTrue(second.get("already_scheduled"))
        self.assertEqual(first["job_id"], second["job_id"])
        sched = Scheduler(h.context)
        jobs = [j for j in sched.list_jobs()
                if j.get("name") == SWEEPER_JOB_NAME]
        # one sweeper job for ALL watchers — never one job per watcher
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["spec"], "every 60s")
        # the job invokes the watch tool's sweep action
        row = h.db.query_one(
            "SELECT payload FROM schedule_jobs WHERE name=?",
            (SWEEPER_JOB_NAME,))
        payload = json.loads(row["payload"])
        self.assertEqual(payload["tool"], "watch")
        self.assertEqual(payload["args"], {"action": "sweep"})
        h.close()

    def test_cognitive_loop_runs_watchers_stage(self):
        from nomorals.agents.cognition import CognitiveLoop

        h = _Ctx()
        h.settings.autonomy = SimpleNamespace(
            tick_goals=False, tick_improvement=False, tick_train=False,
            tick_watchers=True)
        w = Watcher(id="waut", name="autonomy check", kind="condition",
                    target={"tool": "fake_tool", "args": {}},
                    condition=Condition(op="changed", field="value"),
                    interval_s=60, severity="info",
                    created_at=h.now[0])
        WatcherStore(h.db).create(w)
        h.tools["fake_tool"] = lambda **k: {"v": "a"}
        loop = CognitiveLoop(h.context)
        with _allow_test_tools(self):
            summary = loop.tick()
        stage = summary["stages"].get("watchers", {})
        # the autonomy tick swept the watcher (baseline check, no model used)
        self.assertEqual(stage.get("checked"), 1)
        self.assertNotIn("error", stage)
        h.close()


# ── MonitorAgent stability after the extraction ──────────────────────────────

class MonitorStabilityTests(unittest.TestCase):
    def test_extracted_primitives_match_old_behavior(self):
        from nomorals.agents.monitor import (
            MonitorAgent, fetch_file_bytes, hash_bytes,
            unified_content_diff)

        self.assertEqual(hash_bytes(b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9"
                         "cb410ff61f20015ad")
        diff = unified_content_diff("a\nb\n", "a\nc\n")
        self.assertIn("-b", diff)
        self.assertIn("+c", diff)
        # MonitorAgent._diff still delegates
        self.assertEqual(MonitorAgent._diff("x", "y"),
                         unified_content_diff("x", "y"))

    def test_monitor_agent_file_tick_still_works(self):
        from nomorals.agents.monitor import MonitorAgent

        h = _Ctx()
        # file monitors are sandboxed to settings.workspace_dir
        path = os.path.join(h.tmp, "watched.txt")
        with open(path, "w") as fh:
            fh.write("hello")
        try:
            mon = MonitorAgent(h.context)
            mon.add("watched.txt", kind="file", interval=30)
            first = mon.tick()
            self.assertEqual(first["checked"], 1)
            self.assertEqual(first["changed"], [])
            with open(path, "w") as fh:
                fh.write("hello world")
            mon.db.execute("UPDATE monitors SET last_ts=0")
            second = mon.tick()
            self.assertEqual(len(second["changed"]), 1)
            self.assertIn("diff", second["changed"][0])
        finally:
            os.unlink(path)
            h.close()


# ── real CLI coverage ────────────────────────────────────────────────────────

class CliTests(unittest.TestCase):
    def _run(self, home, *argv):
        env = dict(os.environ)
        env["HOME"] = home
        env["TMPDIR"] = home
        proc = subprocess.run(
            [sys.executable, "-m", "nomorals.cli", "watch", *argv],
            cwd="/home/hatch/workspace/devon", env=env,
            capture_output=True, text=True, timeout=120)
        return proc

    def test_cli_add_and_list(self):
        home = tempfile.mkdtemp(prefix="nm-watch-cli-")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        proc = self._run(home, "add", "tell me if BTC drops below $60000")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("BTC", proc.stdout)
        self.assertIn("60000", proc.stdout)
        proc = self._run(home, "list")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("BTC", proc.stdout)

    def test_cli_add_ambiguous_asks(self):
        home = tempfile.mkdtemp(prefix="nm-watch-cli-")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        proc = self._run(home, "add", "watch the price of headphones")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("threshold", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
