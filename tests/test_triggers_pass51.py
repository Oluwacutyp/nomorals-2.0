"""Pass #51 — triggers + webhooks + automation-firing surface.

Pins the fixes:

* POST /triggers/webhook answers non-2xx on authentication failure
  (bad secret / bad HMAC / stale timestamp) — a 200 told external
  senders the delivery succeeded when it was rejected.  Duplicate
  deliveries are acknowledged (200, ``duplicate: true``) without
  re-firing; unknown/disabled targets are 400s.
* A delivery to a disabled trigger no longer burns its event_id:
  after re-enable, the sender's legitimate retry fires.  Concurrent
  duplicate deliveries claim the event id atomically — exactly one
  fires.
* POST /triggers (API) has engine parity: conditions/mode/poll_s are
  honored (not silently dropped) and schedule triggers are actually
  wired into the scheduler (previously persisted-but-dead).
* GET /triggers and the create response never disclose webhook
  secrets; the stored trigger still fires with the real secret.
* A secret-less "plain" webhook trigger says so at creation (API
  ``warnings`` + ``webhook_auth: "none"``), in the CLI, and in list
  rendering — the exposure is explicit, not silent.
* ``time_window`` conditions accept an IANA ``timezone`` and evaluate
  the window in that zone instead of silently using server-local time.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from nomorals.api.server import APIServer, Principal
from nomorals.core.errors import CapabilityDenied, NoMoralsError
from nomorals.core.policy import Capability, CapabilitySet
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry
from nomorals.triggers.engine import TriggerEngine
from nomorals.triggers.models import (
    TriggerError,
    WebhookAuthError,
    WebhookReplayError,
    webhook_protection,
)

OWNER_TOKEN = "owner-" + "tok-p51"
RO_TOKEN = "ro-" + "tok-p51"


def _db() -> Database:
    return Database(":memory:")


def _engine(db: Database | None = None) -> TriggerEngine:
    return TriggerEngine(db or _db(),
                         notify_fn=lambda *a: {"sent": True})


def _context(db: Database) -> Any:
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(
            api=SimpleNamespace(token="", max_body_mb=0),
            backup_dir="/tmp", workspace_dir="/tmp"),
        tools=ToolRegistry(), bus=None, memory=None)


def _server(db: Database) -> tuple[APIServer, Any, Any]:
    api = APIServer(
        _context(db), token=OWNER_TOKEN,
        principals={RO_TOKEN: Principal(
            "ro", CapabilitySet.of(Capability.DB_READ))})
    owner = api.resolve_principal(f"Bearer {OWNER_TOKEN}")
    ro = api.resolve_principal(f"Bearer {RO_TOKEN}")
    return api, owner, ro


def _create_webhook(api: APIServer, owner: Any,
                    condition: dict[str, Any]) -> dict[str, Any]:
    status, payload = api.dispatch("POST", "/triggers", {
        "name": "hook", "source": "webhook", "condition": condition,
        "action": "notify", "action_params": {"title": "t"}}, {},
        principal=owner)
    assert status == 200, payload
    return payload


# ── webhook HTTP status contract ─────────────────────────────────────────

class WebhookStatusContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p51-")
        self.db = Database(f"{self.tmp}/t.db")
        self.api, self.owner, self.ro = _server(self.db)
        payload = _create_webhook(self.api, self.owner, {"secret": "s3cr3t"})
        self.tid = payload["trigger"]["id"]

    def _post(self, body: dict[str, Any]) -> tuple[int, Any]:
        return self.api.dispatch("POST", "/triggers/webhook", body, {},
                                 principal=self.owner)

    def test_wrong_secret_is_not_200(self) -> None:
        status, payload = self._post(
            {"trigger_id": self.tid, "secret": "WRONG", "payload": {}})
        self.assertNotEqual(status, 200)
        self.assertEqual(status, 403)
        self.assertEqual(payload["kind"], "CapabilityDenied")
        self.assertIn("secret", payload["error"])

    def test_correct_secret_fires_200(self) -> None:
        status, payload = self._post(
            {"trigger_id": self.tid, "secret": "s3cr3t", "payload": {}})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["fired"])

    def test_unknown_trigger_is_400(self) -> None:
        status, payload = self._post({"trigger_id": "trg_nope"})
        self.assertEqual(status, 400)
        self.assertIn("unknown trigger", payload["error"])

    def test_not_a_webhook_trigger_is_400(self) -> None:
        status, created = self.api.dispatch("POST", "/triggers", {
            "name": "m", "source": "message", "condition": {"pattern": "x"},
            "action": "notify", "action_params": {}}, {}, principal=self.owner)
        self.assertEqual(status, 200)
        status, payload = self._post(
            {"trigger_id": created["trigger"]["id"]})
        self.assertEqual(status, 400)
        self.assertIn("not a webhook trigger", payload["error"])

    def test_disabled_trigger_is_400(self) -> None:
        eng = _engine(self.db)
        eng.set_enabled(self.tid, False)
        status, payload = self._post(
            {"trigger_id": self.tid, "secret": "s3cr3t"})
        self.assertEqual(status, 400)
        self.assertIn("disabled", payload["error"])

    def test_duplicate_delivery_acknowledged_not_refired(self) -> None:
        body = {"trigger_id": self.tid, "secret": "s3cr3t",
                "event_id": "evt-42", "payload": {}}
        status, payload = self._post(body)
        self.assertEqual(status, 200)
        self.assertTrue(payload["fired"])
        status, payload = self._post(body)
        self.assertEqual(status, 200)  # ack: sender must stop retrying
        self.assertFalse(payload["fired"])
        self.assertTrue(payload["duplicate"])
        eng = _engine(self.db)
        self.assertEqual(eng.get(self.tid).fire_count, 1)

    def test_github_bad_signature_is_403(self) -> None:
        payload = _create_webhook(
            self.api, self.owner, {"secret": "whsec", "scheme": "github"})
        tid = payload["trigger"]["id"]
        status, payload = self._post(
            {"trigger_id": tid, "payload": {"a": 1},
             "signature": "sha256=deadbeef"})
        self.assertEqual(status, 403)
        # and a correct HMAC over the canonical body fires
        body = json.dumps({"a": 1}, sort_keys=True,
                          separators=(",", ":")).encode()
        sig = "sha256=" + hmac_mod.new(b"whsec", body,
                                       hashlib.sha256).hexdigest()
        status, payload = self._post(
            {"trigger_id": tid, "payload": {"a": 1}, "signature": sig})
        self.assertEqual(status, 200)
        self.assertTrue(payload["fired"])

    def test_stripe_stale_timestamp_is_403(self) -> None:
        payload = _create_webhook(
            self.api, self.owner, {"secret": "whsec", "scheme": "stripe"})
        tid = payload["trigger"]["id"]
        stale = 1577836800  # 2020-01-01: outside any 300s tolerance, forever
        body = json.dumps({}, sort_keys=True, separators=(",", ":")).encode()
        sig_hex = hmac_mod.new(b"whsec", f"{stale}.".encode() + body,
                               hashlib.sha256).hexdigest()
        status, payload = self._post(
            {"trigger_id": tid, "payload": {},
             "signature": f"t={stale},v1={sig_hex}"})
        self.assertEqual(status, 403)
        self.assertIn("tolerance", payload["error"])


# ── engine-level typed errors + dedup integrity ──────────────────────────

class WebhookEngineIntegrityTests(unittest.TestCase):
    def test_bad_secret_raises_auth_error(self) -> None:
        eng = _engine()
        t = eng.add("wh", "webhook", {"secret": "s"}, "notify", {})
        with self.assertRaises(WebhookAuthError):
            eng.fire_webhook(t.id, secret="wrong")
        # still a TriggerError for legacy catchers
        with self.assertRaises(TriggerError):
            eng.fire_webhook(t.id, secret="wrong")

    def test_duplicate_raises_replay_error(self) -> None:
        eng = _engine()
        t = eng.add("wh", "webhook", {"secret": "s"}, "notify", {})
        eng.fire_webhook(t.id, secret="s", event_id="e1")
        with self.assertRaises(WebhookReplayError):
            eng.fire_webhook(t.id, secret="s", event_id="e1")
        self.assertEqual(eng.get(t.id).fire_count, 1)

    def test_disabled_delivery_does_not_burn_event_id(self) -> None:
        eng = _engine()
        t = eng.add("wh", "webhook", {"secret": "s"}, "notify", {})
        eng.set_enabled(t.id, False)
        with self.assertRaises(TriggerError):
            eng.fire_webhook(t.id, secret="s", event_id="evt-x")
        eng.set_enabled(t.id, True)
        result = eng.fire_webhook(t.id, secret="s", event_id="evt-x")
        self.assertTrue(result["fired"])

    def test_concurrent_duplicate_deliveries_fire_once(self) -> None:
        # file-backed DB: :memory: is per-connection, invisible to threads
        tmp = tempfile.mkdtemp(prefix="p51-race-")
        eng = _engine(Database(f"{tmp}/t.db"))
        t = eng.add("wh", "webhook", {"secret": "s"}, "notify", {})
        results: list[str] = []
        barrier = threading.Barrier(2)

        def deliver() -> None:
            barrier.wait(timeout=10)
            try:
                out = eng.fire_webhook(t.id, secret="s", event_id="evt-race")
                results.append("fired" if out["fired"] else "not-fired")
            except WebhookReplayError:
                results.append("replay")
            except Exception as exc:  # noqa: BLE001
                results.append(f"other:{type(exc).__name__}")

        threads = [threading.Thread(target=deliver) for _ in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)
        self.assertEqual(sorted(results), ["fired", "replay"])
        self.assertEqual(eng.get(t.id).fire_count, 1)


# ── API create parity + secret redaction ─────────────────────────────────

class ApiCreateParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p51-")
        self.db = Database(f"{self.tmp}/t.db")
        self.api, self.owner, self.ro = _server(self.db)

    def test_api_schedule_trigger_is_wired(self) -> None:
        status, payload = self.api.dispatch("POST", "/triggers", {
            "name": "api-cron", "source": "schedule",
            "condition": {"cron": "0 9 * * *"},
            "action": "notify", "action_params": {}}, {},
            principal=self.owner)
        self.assertEqual(status, 200, payload)
        tid = payload["trigger"]["id"]
        rows = self.db.query(
            "SELECT task_id FROM cron_jobs WHERE task_id = ?",
            (f"trigger:{tid}",))
        self.assertEqual(len(rows), 1)

    def test_api_create_keeps_conditions_mode_poll(self) -> None:
        status, payload = self.api.dispatch("POST", "/triggers", {
            "name": "gated", "source": "webhook",
            "condition": {"secret": "s"}, "action": "notify",
            "action_params": {},
            "conditions": [{"type": "rate", "max": 1, "window_s": 60}],
            "mode": "single", "poll_s": 5}, {}, principal=self.owner)
        self.assertEqual(status, 200, payload)
        t = payload["trigger"]
        self.assertEqual(t["conditions"],
                         [{"type": "rate", "max": 1, "window_s": 60.0}])
        self.assertEqual(t["mode"], "single")
        self.assertEqual(t["poll_s"], 5.0)

    def test_api_create_rejects_bad_conditions(self) -> None:
        status, payload = self.api.dispatch("POST", "/triggers", {
            "name": "bad", "source": "webhook", "condition": {},
            "action": "notify", "action_params": {},
            "conditions": [{"type": "rate", "max": 0, "window_s": 60}]},
            {}, principal=self.owner)
        self.assertEqual(status, 400)

    def test_listing_never_discloses_webhook_secret(self) -> None:
        _create_webhook(self.api, self.owner, {"secret": "topsecret"})
        for principal in (self.owner, self.ro):
            status, payload = self.api.dispatch(
                "GET", "/triggers", {}, {}, principal=principal)
            self.assertEqual(status, 200)
            self.assertNotIn("topsecret", json.dumps(payload))
            trig = payload["triggers"][0]
            self.assertEqual(trig["condition"]["secret"], "********")
            self.assertEqual(trig["webhook_auth"], "shared-secret")
        # the stored trigger still fires with the real secret
        status, payload = self.api.dispatch("POST", "/triggers/webhook", {
            "trigger_id": payload_id(self.api, self.owner),
            "secret": "topsecret", "payload": {}}, {}, principal=self.owner)
        self.assertEqual(status, 200)
        self.assertTrue(payload["fired"])

    def test_create_response_redacts_secret(self) -> None:
        payload = _create_webhook(self.api, self.owner,
                                  {"secret": "topsecret"})
        self.assertNotIn("topsecret", json.dumps(payload))
        self.assertEqual(
            payload["trigger"]["condition"]["secret"], "********")


def payload_id(api: APIServer, owner: Any) -> str:
    status, payload = api.dispatch("GET", "/triggers", {}, {},
                                   principal=owner)
    assert status == 200
    return payload["triggers"][0]["id"]


# ── open (secret-less) webhook explicitness ──────────────────────────────

class OpenWebhookExplicitnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="p51-")
        self.db = Database(f"{self.tmp}/t.db")
        self.api, self.owner, self.ro = _server(self.db)

    def test_protection_classification(self) -> None:
        self.assertEqual(webhook_protection({}), "none")
        self.assertEqual(webhook_protection({"secret": "x"}),
                         "shared-secret")
        self.assertEqual(
            webhook_protection({"secret": "x", "scheme": "plain"}),
            "shared-secret")
        self.assertEqual(
            webhook_protection({"secret": "x", "scheme": "github"}),
            "hmac:github")
        self.assertEqual(
            webhook_protection({"secret": "x", "scheme": "stripe"}),
            "hmac:stripe")

    def test_api_create_warns_for_open_webhook(self) -> None:
        payload = _create_webhook(self.api, self.owner, {})
        self.assertEqual(payload["trigger"]["webhook_auth"], "none")
        warnings = payload.get("warnings") or []
        self.assertTrue(
            any("no secret" in w.lower() for w in warnings), payload)

    def test_api_create_no_warning_when_protected(self) -> None:
        payload = _create_webhook(self.api, self.owner, {"secret": "s"})
        self.assertEqual(payload["trigger"]["webhook_auth"],
                         "shared-secret")
        self.assertFalse(payload.get("warnings"))

    def test_listing_marks_open_webhook(self) -> None:
        _create_webhook(self.api, self.owner, {})
        status, payload = self.api.dispatch(
            "GET", "/triggers", {}, {}, principal=self.ro)
        self.assertEqual(status, 200)
        self.assertEqual(payload["triggers"][0]["webhook_auth"], "none")

    def test_display_line_flags_open_webhook(self) -> None:
        from nomorals.triggers import display

        eng = _engine()
        open_t = eng.add("open", "webhook", {}, "notify", {})
        shut_t = eng.add("shut", "webhook", {"secret": "s"}, "notify", {})
        self.assertIn("open webhook", display.format_trigger_line(open_t))
        self.assertNotIn("open webhook",
                         display.format_trigger_line(shut_t))

    def test_cli_add_warns_for_open_webhook(self) -> None:
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from nomorals.cmdline.commands.trigger import _cmd_trigger

        ctx = _context(_db())
        args = SimpleNamespace(
            task=["add"], name="cli-open", source="webhook",
            action="notify", condition="", params="", cooldown=0.0,
            secret="", title="hi", body="", chat="", text="",
            command_text="", argv=None, goal="", max_iterations=None,
            cron="", interval="", daily="", weekly="", once="",
            path="", on="", symbol="", market="", op="", value=None,
            pattern="", sender="", json=False)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = _cmd_trigger(args, ctx)
        self.assertEqual(rc, 0)
        self.assertIn("no secret", err.getvalue().lower())


# ── time_window timezone ─────────────────────────────────────────────────

class TimeWindowTimezoneTests(unittest.TestCase):
    def test_window_evaluates_in_named_zone(self) -> None:
        eng = _engine()
        t = eng.add("win", "webhook", {}, "notify", {},
                    conditions=[{"type": "time_window", "after": "02:00",
                                 "before": "04:00",
                                 "timezone": "America/New_York"}])
        # 2026-01-15 03:00 UTC == 22:00 EST (previous day) in New York:
        # outside 02:00-04:00 there, inside it in UTC.
        now = datetime(2026, 1, 15, 3, 0,
                       tzinfo=timezone.utc).timestamp()
        ok, reason = eng.check_conditions(t, {}, now=now)
        self.assertFalse(ok)
        self.assertIn("time_window", reason)
        t2 = eng.add("win2", "webhook", {}, "notify", {},
                     conditions=[{"type": "time_window", "after": "02:00",
                                  "before": "04:00", "timezone": "UTC"}])
        ok, _ = eng.check_conditions(t2, {}, now=now)
        self.assertTrue(ok)

    def test_bad_timezone_rejected_at_creation(self) -> None:
        eng = _engine()
        with self.assertRaises(TriggerError):
            eng.add("win", "webhook", {}, "notify", {},
                    conditions=[{"type": "time_window", "after": "02:00",
                                 "before": "04:00",
                                 "timezone": "Mars/Olympus"}])

    def test_window_without_timezone_still_valid(self) -> None:
        eng = _engine()
        t = eng.add("win", "webhook", {}, "notify", {},
                    conditions=[{"type": "time_window", "after": "00:00",
                                 "before": "23:59"}])
        self.assertEqual(t.conditions[0]["type"], "time_window")
        self.assertNotIn("timezone", t.conditions[0])


if __name__ == "__main__":
    unittest.main()
