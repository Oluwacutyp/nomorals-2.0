"""Trigger sweep tests: url source, conditions, modes, templates, digest,
HMAC webhooks, display, saved-search snooze/digest, store extras."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.storage.db import Database
from nomorals.triggers import (
    TriggerEngine,
    TriggerError,
    TriggerSpec,
    display,
    list_templates,
    render_template as render_blueprint,
    to_cloudevent,
    validate_definition,
    validate_trigger_spec,
    verify_webhook_signature,
)
from nomorals.triggers.actions import render_template
from nomorals.triggers.models import (
    OUTCOME_ERROR,
    OUTCOME_FIRED,
    OUTCOME_NO_MATCH,
    OUTCOME_SKIPPED,
)
from nomorals.triggers.saved_search import (
    Match,
    SavedSearchStore,
    check_all,
    deliver_digests,
    format_digest,
    format_search,
    register_matcher,
)
from nomorals.triggers.sources import evaluate_url, match_message
from nomorals.triggers.webhook import WEBHOOK_PATH, register_trigger_routes


def _db():
    return Database(":memory:")


def _engine(**kw):
    kw.setdefault("notify_fn", lambda trig, title, body, eng: {"ok": True})
    return TriggerEngine(_db(), **kw)


# ── models: new validation ───────────────────────────────────────────

class UrlValidationTests(unittest.TestCase):
    def test_missing_url(self):
        with self.assertRaises(TriggerError):
            validate_definition("url", {}, "notify", {})

    def test_bad_scheme(self):
        with self.assertRaises(TriggerError):
            validate_definition("url", {"url": "ftp://x/y"}, "notify", {})

    def test_ok(self):
        cond, _, _ = validate_definition(
            "url", {"url": "https://example.com", "text_only": True,
                    "timeout_s": 5}, "notify", {})
        self.assertEqual(cond["url"], "https://example.com")
        self.assertTrue(cond["text_only"])

    def test_bad_regex(self):
        with self.assertRaises(TriggerError):
            validate_definition("url", {"url": "https://example.com",
                                        "regex": "(["}, "notify", {})

    def test_bad_jsonpath(self):
        with self.assertRaises(TriggerError):
            validate_definition("url", {"url": "https://example.com",
                                        "jsonpath": "data.price"},
                                "notify", {})

    def test_bad_timeout(self):
        with self.assertRaises(TriggerError):
            validate_definition("url", {"url": "https://example.com",
                                        "timeout_s": 9999}, "notify", {})


class ConditionValidationTests(unittest.TestCase):
    def test_unknown_type(self):
        with self.assertRaises(TriggerError):
            validate_trigger_spec("webhook", {}, "notify", {},
                                  conditions=[{"type": "bogus"}])

    def test_time_window_ok(self):
        spec = validate_trigger_spec(
            "webhook", {}, "notify", {},
            conditions=[{"type": "time_window", "after": "22:00",
                         "before": "06:00"}])
        self.assertEqual(spec.conditions[0]["after"], "22:00")

    def test_time_window_bad(self):
        with self.assertRaises(TriggerError):
            validate_trigger_spec(
                "webhook", {}, "notify", {},
                conditions=[{"type": "time_window", "after": "nope",
                             "before": "06:00"}])

    def test_rate_ok_and_bad(self):
        spec = validate_trigger_spec(
            "webhook", {}, "notify", {},
            conditions=[{"type": "rate", "max": 3, "window_s": 3600}])
        self.assertEqual(spec.conditions[0]["max"], 3)
        with self.assertRaises(TriggerError):
            validate_trigger_spec(
                "webhook", {}, "notify", {},
                conditions=[{"type": "rate", "max": 0, "window_s": 3600}])

    def test_evidence_ok_and_bad(self):
        spec = validate_trigger_spec(
            "webhook", {}, "notify", {},
            conditions=[{"type": "evidence",
                         "match": {"symbol": "BTC"}}])
        self.assertEqual(spec.conditions[0]["match"], {"symbol": "BTC"})
        with self.assertRaises(TriggerError):
            validate_trigger_spec(
                "webhook", {}, "notify", {},
                conditions=[{"type": "evidence", "match": {}}])

    def test_mode_validation(self):
        spec = validate_trigger_spec("webhook", {}, "notify", {},
                                     mode="single")
        self.assertEqual(spec.mode, "single")
        with self.assertRaises(TriggerError):
            validate_trigger_spec("webhook", {}, "notify", {},
                                  mode="turbo")

    def test_poll_s_validation(self):
        spec = validate_trigger_spec("webhook", {}, "notify", {},
                                     poll_s=120)
        self.assertEqual(spec.poll_s, 120.0)
        with self.assertRaises(TriggerError):
            validate_trigger_spec("webhook", {}, "notify", {},
                                  poll_s=-1)

    def test_backward_compat_3tuple(self):
        out = validate_definition("webhook", {}, "notify", {})
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[0], dict)


class WebhookSchemeValidationTests(unittest.TestCase):
    def test_plain_default(self):
        # legacy shape stays byte-identical: no scheme key when unset
        cond, _, _ = validate_definition("webhook", {}, "notify", {})
        self.assertEqual(cond, {})
        cond, _, _ = validate_definition(
            "webhook", {"secret": "s"}, "notify", {})
        self.assertEqual(cond, {"secret": "s"})

    def test_hmac_needs_secret(self):
        with self.assertRaises(TriggerError):
            validate_definition("webhook", {"scheme": "github"},
                                "notify", {})

    def test_github_ok(self):
        cond, _, _ = validate_definition(
            "webhook", {"scheme": "github", "secret": "s3cr3t",
                        "tolerance_s": 120}, "notify", {})
        self.assertEqual(cond["scheme"], "github")
        self.assertEqual(cond["tolerance_s"], 120.0)

    def test_bad_scheme(self):
        with self.assertRaises(TriggerError):
            validate_definition("webhook", {"scheme": "rot13", "secret": "x"},
                                "notify", {})

    def test_bad_tolerance(self):
        with self.assertRaises(TriggerError):
            validate_definition(
                "webhook", {"scheme": "stripe", "secret": "x",
                            "tolerance_s": -5}, "notify", {})


class MessageConditionTests(unittest.TestCase):
    def test_exclude_and_case_insensitive(self):
        cond, _, _ = validate_definition(
            "message", {"pattern": "alert", "exclude": "test",
                        "case_insensitive": True}, "notify", {})
        self.assertEqual(cond["exclude"], "test")
        self.assertTrue(cond["case_insensitive"])

    def test_bad_exclude(self):
        with self.assertRaises(TriggerError):
            validate_definition("message", {"pattern": "x",
                                            "exclude": "(["}, "notify", {})


class DigestParamTests(unittest.TestCase):
    def test_digest_ok(self):
        _, params, _ = validate_definition(
            "webhook", {}, "notify",
            {"digest": True, "digest_every_s": 120, "digest_max": 5})
        self.assertEqual(params["digest_every_s"], 120.0)

    def test_digest_bad(self):
        with self.assertRaises(TriggerError):
            validate_definition("webhook", {}, "notify",
                                {"digest": True, "digest_every_s": 5})


class ScheduleOptionTests(unittest.TestCase):
    def test_misfire_passthrough(self):
        cond, _, _ = validate_definition(
            "schedule", {"daily": "09:30", "misfire": "skip",
                         "overlap": "queue"}, "notify", {})
        self.assertEqual(cond["misfire"], "skip")
        self.assertEqual(cond["overlap"], "queue")
        # legacy shape stays byte-identical when no options are set
        cond, _, _ = validate_definition(
            "schedule", {"daily": "09:30"}, "notify", {})
        self.assertEqual(cond, {"cron": "30 9 * * *"})

    def test_bad_misfire(self):
        with self.assertRaises(TriggerError):
            validate_definition("schedule", {"daily": "09:30",
                                             "misfire": "yolo"},
                                "notify", {})


# ── templates ────────────────────────────────────────────────────────

class TemplateTests(unittest.TestCase):
    def test_list(self):
        names = {t["name"] for t in list_templates()}
        self.assertIn("price_alert", names)

    def test_price_alert_render(self):
        spec = render_blueprint("price_alert",
                               {"symbol": "BTC", "direction": "below",
                                "value": "60000"})
        self.assertEqual(spec["source"], "price")
        self.assertEqual(spec["condition"]["op"], "lt")
        self.assertIn("{{price}}", spec["action_params"]["body"])

    def test_missing_input_raises(self):
        with self.assertRaises(TriggerError):
            render_blueprint("price_alert", {"symbol": "BTC"})

    def test_unknown_template(self):
        with self.assertRaises(TriggerError):
            render_blueprint("nope", {})

    def test_engine_add_from_template(self):
        eng = _engine()
        t = eng.add_from_template("message_keyword",
                                  {"pattern": "urgent", "chat": "c1"})
        self.assertEqual(t.source, "message")
        self.assertEqual(t.condition["chat"], "c1")
        self.assertEqual(t.cooldown_s, 300.0)


# ── actions: evidence templates + digest ─────────────────────────────

class EvidenceTemplateTests(unittest.TestCase):
    def test_dotted_paths(self):
        ev = {"price": 61234.5, "match": {"title": "2bed Yaba"}}
        self.assertEqual(
            render_template("now {{price}} — {{match.title}}", ev),
            "now 61234.5 — 2bed Yaba")

    def test_missing_renders_empty(self):
        self.assertEqual(render_template("a{{nope}}b", {}), "ab")

    def test_never_raises(self):
        self.assertIsInstance(render_template("{{x}}", {"x": object()}), str)

    def test_notify_body_uses_evidence(self):
        eng = _engine()
        t = eng.add("t1", "webhook", {}, "notify",
                    {"title": "alert", "body": "price {{payload.p}}"})
        out = eng.fire_webhook(t.id, payload={"p": "61k"})
        self.assertTrue(out["fired"])
        # notify_fn fake ignores content; check history detail kept evidence
        rows = eng.history(t.id, limit=1)
        self.assertEqual(rows[0]["outcome"], OUTCOME_FIRED)


class DigestActionTests(unittest.TestCase):
    def test_digest_buffers_and_flushes(self):
        sent = []
        eng = _engine(send_message=lambda chat, text: sent.append((chat, text)))
        t = eng.add("d1", "webhook", {}, "message",
                    {"chat": "c1", "text": "hit {{payload.n}}",
                     "digest": True, "digest_every_s": 60,
                     "digest_max": 100})
        eng.fire_webhook(t.id, payload={"n": "1"})
        eng.fire_webhook(t.id, payload={"n": "2"})
        self.assertEqual(sent, [])  # buffered, not sent
        self.assertEqual(len(eng.store.digest_pending(t.id)), 2)
        summary = eng.flush_digests(force=True)
        self.assertEqual(summary["sent"], 1)
        self.assertEqual(len(sent), 1)
        chat, text = sent[0]
        self.assertEqual(chat, "c1")
        self.assertIn("hit 1", text)
        self.assertIn("hit 2", text)
        self.assertEqual(eng.store.digest_pending(t.id), [])

    def test_digest_flush_respects_cadence(self):
        eng = _engine(send_message=lambda c, t: None)
        t = eng.add("d2", "webhook", {}, "notify",
                    {"digest": True, "digest_every_s": 3600})
        eng.fire_webhook(t.id)
        summary = eng.flush_digests()  # not due yet
        self.assertEqual(summary["sent"], 0)
        self.assertEqual(len(eng.store.digest_pending(t.id)), 1)


# ── engine: conditions + modes ───────────────────────────────────────

class ConditionGateTests(unittest.TestCase):
    def test_time_window_blocks(self):
        eng = _engine()
        # window that can never contain "now": use after==before-1min trick —
        # instead compute a window guaranteed outside now
        now_hm = time.strftime("%H:%M")
        t = eng.add("c1", "webhook", {}, "notify", {},
                    conditions=[{"type": "time_window", "after": now_hm,
                                 "before": now_hm}])
        # after == before → empty window → always blocked
        out = eng.fire_webhook(t.id)
        self.assertFalse(out["fired"])
        self.assertEqual(out["reason"], "condition")

    def test_rate_gate(self):
        eng = _engine()
        t = eng.add("c2", "webhook", {}, "notify", {},
                    conditions=[{"type": "rate", "max": 2,
                                 "window_s": 3600}])
        self.assertTrue(eng.fire_webhook(t.id)["fired"])
        self.assertTrue(eng.fire_webhook(t.id)["fired"])
        out = eng.fire_webhook(t.id)
        self.assertFalse(out["fired"])
        self.assertEqual(out["reason"], "condition")
        rows = eng.history(t.id, limit=1)
        self.assertEqual(rows[0]["outcome"], OUTCOME_SKIPPED)

    def test_evidence_gate(self):
        eng = _engine()
        t = eng.add("c3", "webhook", {}, "notify", {},
                    conditions=[{"type": "evidence",
                                 "match": {"source": "webhook"}}])
        out = eng.fire_webhook(t.id)
        self.assertTrue(out["fired"])  # evidence always has source=webhook


class ModeTests(unittest.TestCase):
    def test_single_skips_while_running(self):
        import tempfile
        entered = threading.Event()
        release = threading.Event()

        def slow_notify(trig, title, body, eng):
            entered.set()
            release.wait(timeout=5)
            return {"ok": True}

        # file-backed DB: :memory: is per-connection, invisible to threads
        db = Database(os.path.join(tempfile.mkdtemp(), "t.db"))
        eng = TriggerEngine(
            db, notify_fn=slow_notify)
        t = eng.add("m1", "webhook", {}, "notify", {}, mode="single")
        results = []
        th = threading.Thread(
            target=lambda: results.append(eng.fire_webhook(t.id)))
        th.start()
        self.assertTrue(entered.wait(timeout=5))
        out = eng.fire_webhook(t.id)  # second fire while first runs
        self.assertFalse(out["fired"])
        self.assertEqual(out["reason"], "already_running")
        release.set()
        th.join(timeout=5)
        self.assertTrue(results[0]["fired"])

    def test_bad_mode_rejected(self):
        eng = _engine()
        with self.assertRaises(TriggerError):
            eng.add("m2", "webhook", {}, "notify", {}, mode="turbo")


# ── engine: poll_s + url tick ────────────────────────────────────────

class _MutableHandler(BaseHTTPRequestHandler):
    content = b"<html><body><p>hello</p></body></html>"

    def do_GET(self):
        body = _MutableHandler.content
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class UrlSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _MutableHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)

    def setUp(self):
        _MutableHandler.content = b"<html><body><p>hello</p></body></html>"

    def _url(self):
        return f"http://127.0.0.1:{self.port}/"

    def test_baseline_then_change(self):
        from types import SimpleNamespace
        trig = SimpleNamespace(condition={"url": self._url(),
                                          "text_only": True})
        state: dict = {}
        fired, ev = evaluate_url(trig, state)
        self.assertFalse(fired)
        self.assertEqual(ev["event"], "baseline")
        _MutableHandler.content = b"<html><body><p>world</p></body></html>"
        fired, ev = evaluate_url(trig, state)
        self.assertTrue(fired)
        self.assertEqual(ev["event"], "changed")
        self.assertIn("world", ev["snippet"])

    def test_regex_gate(self):
        from types import SimpleNamespace
        trig = SimpleNamespace(condition={"url": self._url(),
                                          "regex": "nope-never"})
        state: dict = {}
        evaluate_url(trig, state)  # baseline
        _MutableHandler.content = b"<html><body>changed</body></html>"
        fired, ev = evaluate_url(trig, state)
        self.assertFalse(fired)
        self.assertEqual(ev["event"], "changed_no_regex")

    def test_jsonpath(self):
        from types import SimpleNamespace

        class _JsonHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"data": {"price": 42}}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), _JsonHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            trig = SimpleNamespace(
                condition={"url": f"http://127.0.0.1:{port}/",
                           "jsonpath": "$.data.price"})
            state: dict = {}
            fired, ev = evaluate_url(trig, state)
            self.assertFalse(fired)
            self.assertIn("42", ev["snippet"])
        finally:
            srv.shutdown()
            th.join(timeout=5)

    def test_engine_tick_polls_url(self):
        now = [1000.0]
        eng = _engine(clock=lambda: now[0])
        t = eng.add("u1", "url", {"url": self._url()}, "notify", {})
        summary = eng.tick()
        self.assertEqual(summary["evaluated"], 1)
        rows = eng.history(t.id, limit=1)
        self.assertEqual(rows[0]["outcome"], OUTCOME_NO_MATCH)
        _MutableHandler.content = b"<html><body><p>CHANGED</p></body></html>"
        now[0] += 31.0  # past the default 30s poll interval
        summary = eng.tick()
        self.assertEqual(summary["fired"], 1)

    def test_poll_s_gating(self):
        eng = _engine(poll_interval=30)
        t = eng.add("u2", "url", {"url": self._url()}, "notify", {},
                    poll_s=3600)
        eng.tick()
        summary = eng.tick()  # second tick: not due
        self.assertEqual(summary["not_due"], 1)
        self.assertEqual(summary["evaluated"], 0)


# ── webhook HMAC + idempotency ───────────────────────────────────────

class WebhookHmacTests(unittest.TestCase):
    def test_github_ok_and_bad(self):
        secret, body = "s3cr3t", b'{"a":1}'
        sig = "sha256=" + hmac.new(secret.encode(), body,
                                   hashlib.sha256).hexdigest()
        verify_webhook_signature(secret, "github", signature=sig, body=body)
        with self.assertRaises(TriggerError):
            verify_webhook_signature(secret, "github",
                                     signature="sha256=dead", body=body)
        with self.assertRaises(TriggerError):
            verify_webhook_signature(secret, "github", signature="",
                                     body=body)

    def test_stripe_ok_stale_bad(self):
        secret, body = "whsec_x", b'{"a":1}'
        ts = str(int(time.time()))
        signed = f"{ts}.".encode() + body
        v1 = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        verify_webhook_signature(secret, "stripe",
                                 signature=f"t={ts},v1={v1}", body=body)
        old_ts = str(int(time.time()) - 9999)
        signed_old = f"{old_ts}.".encode() + body
        v1_old = hmac.new(secret.encode(), signed_old,
                          hashlib.sha256).hexdigest()
        with self.assertRaises(TriggerError):
            verify_webhook_signature(secret, "stripe",
                                     signature=f"t={old_ts},v1={v1_old}",
                                     body=body, tolerance_s=300)
        with self.assertRaises(TriggerError):
            verify_webhook_signature(secret, "stripe",
                                     signature=f"t={ts},v1=bad", body=body)

    def test_engine_fire_webhook_github(self):
        eng = _engine()
        secret = "s3cr3t"
        t = eng.add("w1", "webhook",
                    {"scheme": "github", "secret": secret}, "notify", {})
        payload = {"n": 1}
        raw = json.dumps(payload, sort_keys=True,
                         separators=(",", ":")).encode()
        sig = "sha256=" + hmac.new(secret.encode(), raw,
                                   hashlib.sha256).hexdigest()
        out = eng.fire_webhook(t.id, payload=payload, signature=sig,
                               raw_body=raw)
        self.assertTrue(out["fired"])
        with self.assertRaises(TriggerError):
            eng.fire_webhook(t.id, payload=payload, signature="sha256=x",
                             raw_body=raw)

    def test_webhook_idempotency(self):
        eng = _engine()
        t = eng.add("w2", "webhook", {}, "notify", {})
        out = eng.fire_webhook(t.id, event_id="evt-1")
        self.assertTrue(out["fired"])
        with self.assertRaises(TriggerError) as ctx:
            eng.fire_webhook(t.id, event_id="evt-1")
        self.assertIn("duplicate", str(ctx.exception))

    def test_webhook_route(self):
        routes = {}

        class FakeServer:
            def route(self, method, path):
                def deco(fn):
                    routes[(method, path)] = fn
                    return fn
                return deco

        srv = FakeServer()
        register_trigger_routes(srv, type("C", (), {"db": _db()})())
        self.assertIn(("POST", WEBHOOK_PATH), routes)


# ── engine: status / next_run / stats ────────────────────────────────

class StatusTests(unittest.TestCase):
    def test_next_run_cron(self):
        eng = _engine()
        t = eng.add("s1", "schedule", {"cron": "0 9 * * *"}, "notify", {})
        nxt = eng.next_run(t.id)
        self.assertIsNotNone(nxt)
        self.assertGreater(nxt, time.time())

    def test_next_run_once(self):
        eng = _engine()
        future = time.time() + 3600
        t = eng.add("s2", "schedule", {"once": future}, "notify", {})
        self.assertAlmostEqual(eng.next_run(t.id), future, delta=1)

    def test_next_run_non_schedule(self):
        eng = _engine()
        t = eng.add("s3", "webhook", {}, "notify", {})
        self.assertIsNone(eng.next_run(t.id))

    def test_status(self):
        eng = _engine()
        t = eng.add("s4", "webhook", {}, "notify", {})
        eng.fire_webhook(t.id)
        st = eng.status(t.id)
        self.assertEqual(st["id"], t.id)
        self.assertEqual(st["stats"][OUTCOME_FIRED], 1)
        self.assertFalse(st["running"])
        with self.assertRaises(TriggerError):
            eng.status("nope")

    def test_store_stats(self):
        eng = _engine()
        t = eng.add("s5", "webhook", {}, "notify", {})
        eng.fire_webhook(t.id)
        stats = eng.store.stats(t.id)
        self.assertEqual(stats[OUTCOME_FIRED], 1)


# ── message source extensions ────────────────────────────────────────

class MessageMatchTests(unittest.TestCase):
    def test_exclude(self):
        from types import SimpleNamespace
        trig = SimpleNamespace(condition={"pattern": "alert",
                                          "exclude": "test"})
        hit, ev = match_message(trig, "alert: fire", "c", "")
        self.assertTrue(hit)
        hit, ev = match_message(trig, "alert: test fire", "c", "")
        self.assertFalse(hit)
        self.assertEqual(ev["reason"], "excluded")

    def test_case_insensitive(self):
        from types import SimpleNamespace
        trig = SimpleNamespace(condition={"pattern": "ALERT",
                                          "case_insensitive": True})
        hit, _ = match_message(trig, "alert now", "c", "")
        self.assertTrue(hit)
        trig2 = SimpleNamespace(condition={"pattern": "ALERT"})
        hit, _ = match_message(trig2, "alert now", "c", "")
        self.assertFalse(hit)


# ── CloudEvents ──────────────────────────────────────────────────────

class CloudEventTests(unittest.TestCase):
    def test_envelope(self):
        from types import SimpleNamespace
        ev = SimpleNamespace(topic="mission.terminal", source="missions",
                             event_id="e1", data={"ok": True},
                             subject="m-9")
        ce = to_cloudevent(ev)
        self.assertEqual(ce["specversion"], "1.0")
        self.assertEqual(ce["id"], "e1")
        self.assertEqual(ce["type"], "mission.terminal")
        self.assertEqual(ce["data"], {"ok": True})
        self.assertIn("time", ce)


# ── display ──────────────────────────────────────────────────────────

class DisplayTests(unittest.TestCase):
    def _trigger(self):
        eng = _engine()
        return eng, eng.add("pretty", "price",
                            {"symbol": "BTC", "op": "lt", "value": 60000},
                            "notify", {"title": "dip"})

    def test_outcome_glyph(self):
        self.assertEqual(display.outcome_glyph("fired"), "✅")
        self.assertEqual(display.outcome_glyph("no_match"), "⚪")
        self.assertEqual(display.outcome_glyph("skipped"), "⏸️")
        self.assertEqual(display.outcome_glyph("error"), "❌")

    def test_trigger_line(self):
        eng, t = self._trigger()
        line = display.format_trigger_line(t)
        self.assertIn("pretty", line)
        self.assertIn("price", line)
        self.assertIn("notify", line)

    def test_history_row(self):
        row = {"at": time.time(), "outcome": "fired", "trigger_id": "abc",
               "detail": {"evidence": {"price": 61000}}, "error": None}
        text = display.format_history_row(row)
        self.assertIn("✅", text)
        self.assertIn("price=61000", text)

    def test_detail_box(self):
        eng, t = self._trigger()
        st = eng.status(t.id)
        box = display.format_trigger_detail(st)
        self.assertIn("pretty", box)
        self.assertIn("╭", box)

    def test_status_table(self):
        eng, t = self._trigger()
        table = display.format_status_table([t])
        self.assertIn("1/1 enabled", table)
        self.assertEqual(display.format_status_table([]),
                         "no triggers — add one with `nm trigger add`")

    def test_digest_preview(self):
        text = display.format_digest_preview(
            "w", [{"title": "a", "text": "one"}])
        self.assertIn("📦", text)


# ── saved_search: snooze + digest ────────────────────────────────────

class SavedSearchSweepTests(unittest.TestCase):
    def _store(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "ss.db")
        return SavedSearchStore(path)

    def test_snooze(self):
        store = self._store()
        s = store.create("property", "2bed Yaba", {})
        self.assertIsNotNone(s)
        assert s is not None
        self.assertTrue(s.due(now=time.time()))
        self.assertTrue(store.snooze(s.id, 5))
        s2 = store.get(s.id)
        assert s2 is not None
        self.assertTrue(s2.snoozed)
        self.assertFalse(s2.due(now=time.time()))
        self.assertTrue(store.unsnooze(s.id))
        self.assertFalse(store.get(s.id).snoozed)  # type: ignore[union-attr]

    def test_digest_at_validation(self):
        store = self._store()
        s = store.create("property", "2bed Yaba", {})
        assert s is not None
        self.assertTrue(store.set_digest_at(s.id, "08:30"))
        self.assertEqual(store.get(s.id).digest_at, "08:30")  # type: ignore[union-attr]
        self.assertFalse(store.set_digest_at(s.id, "25:00"))
        self.assertTrue(store.set_digest_at(s.id, ""))
        self.assertEqual(store.get(s.id).digest_at, "")  # type: ignore[union-attr]

    def test_digest_flow(self):
        store = self._store()
        s = store.create("property", "2bed Yaba", {})
        assert s is not None
        store.set_digest_at(s.id, "00:00")  # always due
        pushed: list[str] = []
        register_matcher("property", lambda d, q, f: [
            {"title": "flat A", "price_kobo": 150000000, "area": "Yaba",
             "url": "http://x/1"},
            {"title": "flat B", "price_kobo": 120000000, "area": "Yaba",
             "url": "http://x/2"},
        ])
        try:
            found = check_all(store, sender=pushed.append,
                              now=time.time())
            self.assertEqual(len(found), 2)
            self.assertEqual(pushed, [])  # queued, not pushed
            self.assertEqual(store.pending_digest_count(s.id), 2)
            sent = deliver_digests(store, sender=pushed.append,
                                   now=time.time())
            self.assertEqual(sent, 1)
            self.assertEqual(len(pushed), 1)
            self.assertIn("flat A", pushed[0])
            self.assertIn("flat B", pushed[0])
            self.assertEqual(store.pending_digest_count(s.id), 0)
        finally:
            from nomorals.triggers.saved_search import _matchers
            _matchers.pop("property", None)

    def test_format_digest(self):
        from nomorals.triggers.saved_search import SavedSearch
        search = SavedSearch(id="w1", domain="property",
                             query="2bed Yaba")
        m = Match(search_id="w1", title="flat A", price_kobo=150000000,
                  area="Yaba", url="http://x/1")
        text = format_digest(search, [m])
        self.assertIn("📦", text)
        self.assertIn("flat A", text)

    def test_format_search_shows_snooze_digest(self):
        from nomorals.triggers.saved_search import SavedSearch
        s = SavedSearch(id="w1", domain="property", query="q",
                        snooze_until=time.time() + 3600,
                        digest_at="08:00")
        text = format_search(s)
        self.assertIn("snoozed", text)
        self.assertIn("08:00", text)

    def test_control_watch_snooze_digest(self):
        from nomorals.triggers.saved_search import control_watch
        store = self._store()
        ctx = type("C", (), {"saved_search_store": store})()
        out = control_watch("2bed Yaba under 1.5m", ctx)
        self.assertIn("watching", out)
        sid = store.list()[0].id
        out = control_watch(f"snooze {sid} 3", ctx)
        self.assertIn("snoozed", out)
        out = control_watch(f"digest {sid} 08:00", ctx)
        self.assertIn("08:00", out)
        out = control_watch(f"digest {sid} off", ctx)
        self.assertIn("instant", out)


if __name__ == "__main__":
    unittest.main()
