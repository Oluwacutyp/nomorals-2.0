"""Header-signature delivery — direct GitHub/Stripe webhooks (live HTTP).

N4 plumbed the raw request bytes to the route, but a real provider
delivery still could not land: GitHub sends its signature in the
``X-Hub-Signature-256`` header and Stripe in ``Stripe-Signature``,
neither carries a ``trigger_id`` in the body, and the API server never
passed request headers to any handler. Only a relay that wrapped the
event into Devon's envelope (signature in the query string) worked.

These pins prove the direct-delivery contract end to end over live HTTP:

* a GitHub push event with ``X-Hub-Signature-256`` over the raw body
  and ``?trigger=<id>`` verifies and fires — the whole provider body
  is the payload the action sees;
* a Stripe event with ``Stripe-Signature`` (``t=<ts>,v1=<sig>``) and
  ``?trigger=<id>`` verifies and fires;
* replay is still acknowledged (200, ``duplicate: true``) without
  re-firing — via ``X-GitHub-Delivery`` for GitHub, the body's
  ``id`` (``evt_...``) for Stripe;
* wrong header signature / tampered body / stale Stripe timestamp
  still 403;
* a delivery with no trigger identification anywhere is 400;
* the envelope/relay path is unchanged (query-string signature still
  fires), and a relay that forwards the provider's headers still fires
  via its envelope signature — the route tries each signature
  candidate in order and accepts the first that verifies.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import http.client
import json
import socket
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

from nomorals.api.server import APIServer, serve
from nomorals.core.policy import Capability, CapabilitySet
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry
from nomorals.triggers.engine import TriggerEngine

OWNER_TOKEN = "hdr-owner-token"
RO_TOKEN = "hdr-ro-token"


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _context(db: Database, tmp: str) -> Any:
    settings = SimpleNamespace(
        api=SimpleNamespace(token="", max_body_mb=0, cors_origins=None),
        backup_dir=tmp,
        workspace_dir=tmp,
    )
    return SimpleNamespace(
        db=db, settings=settings, tools=ToolRegistry(), bus=None,
        memory=None)


def _wait_up(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/live")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"server did not come up on port {port}")


def _post_raw(port: int, path: str, raw: bytes,
              headers: dict[str, str] | None = None) -> tuple[int, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        all_headers = {"Content-Type": "application/json"}
        all_headers.update(headers or {})
        conn.request("POST", path, body=raw, headers=all_headers)
        resp = conn.getresponse()
        data = resp.read()
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            parsed = {"_unparsed": data.decode("utf-8", "replace")}
        return resp.status, parsed
    finally:
        conn.close()


def _github_sig(secret: str, raw: bytes) -> str:
    return "sha256=" + hmac_mod.new(
        secret.encode(), raw, hashlib.sha256).hexdigest()


def _stripe_sig(secret: str, ts: str, raw: bytes) -> str:
    hexed = hmac_mod.new(
        secret.encode(), f"{ts}.".encode() + raw,
        hashlib.sha256).hexdigest()
    return f"t={ts},v1={hexed}"


def _github_push_body() -> bytes:
    # A realistic push event, serialized non-canonically (insertion
    # order, spaces after separators) so a signature over the canonical
    # re-encoding would NOT match — only the raw bytes verify.
    push = {
        "ref": "refs/heads/main",
        "repository": {"full_name": "Oluwacutyp/nomorals-2.0"},
        "pusher": {"name": "Oluwacutyp"},
        "commits": [{"id": "abc123", "message": "test"}],
    }
    return json.dumps(push).encode()


def _stripe_event_body() -> bytes:
    event = {
        "id": "evt_3P1234567890abcdef",
        "object": "event",
        "type": "payment_intent.succeeded",
        "data": {"object": {"id": "pi_123", "amount": 1000}},
    }
    return json.dumps(event).encode()


class DirectDeliveryTests(unittest.TestCase):
    """Live-HTTP pins: provider headers + query-string trigger id."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hdr-")
        self.db = Database(f"{self.tmp}/t.db")
        eng = TriggerEngine(self.db, notify_fn=lambda *a: {"sent": True})
        gh = eng.add("gh", "webhook",
                     {"secret": "whsec", "scheme": "github"}, "notify", {})
        st = eng.add("st", "webhook",
                     {"secret": "whsec", "scheme": "stripe"}, "notify", {})
        self.gh_tid, self.st_tid = gh.id, st.id
        self.port = _free_port()
        serve(_context(self.db, self.tmp), host="127.0.0.1",
              port=self.port, background=True, protocols=False, token="")
        _wait_up(self.port)

    def _history_payload(self, tid: str) -> dict[str, Any]:
        eng = TriggerEngine(self.db, notify_fn=lambda *a: {"sent": True})
        rows = eng.store.history(tid, limit=1)
        self.assertTrue(rows, "no history row recorded")
        # A fired trigger records {"evidence": {...}, "result": ...};
        # the webhook evidence carries the delivered payload.
        return rows[0]["detail"].get("evidence", {}).get("payload", {})

    def test_github_direct_delivery_fires(self) -> None:
        raw = _github_push_body()
        sig = _github_sig("whsec", raw)
        status, payload = _post_raw(
            self.port, "/triggers/webhook?trigger=" + self.gh_tid, raw,
            headers={"X-Hub-Signature-256": sig,
                     "X-GitHub-Delivery": "deliv-1",
                     "X-GitHub-Event": "push"})
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])
        # The action sees the provider event itself, not an empty payload.
        delivered = self._history_payload(self.gh_tid)
        self.assertEqual(delivered.get("ref"), "refs/heads/main")
        self.assertEqual(
            delivered.get("repository", {}).get("full_name"),
            "Oluwacutyp/nomorals-2.0")

    def test_github_direct_delivery_replay_via_delivery_header(self) -> None:
        raw = _github_push_body()
        sig = _github_sig("whsec", raw)
        path = "/triggers/webhook?trigger=" + self.gh_tid
        headers = {"X-Hub-Signature-256": sig,
                   "X-GitHub-Delivery": "deliv-replay-1"}
        status, payload = _post_raw(self.port, path, raw, headers=headers)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])
        status, payload = _post_raw(self.port, path, raw, headers=headers)
        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["fired"])
        self.assertTrue(payload["duplicate"])
        eng = TriggerEngine(self.db, notify_fn=lambda *a: {"sent": True})
        self.assertEqual(eng.get(self.gh_tid).fire_count, 1)

    def test_github_direct_wrong_header_signature_is_403(self) -> None:
        raw = _github_push_body()
        sig = _github_sig("WRONG", raw)
        status, payload = _post_raw(
            self.port, "/triggers/webhook?trigger=" + self.gh_tid, raw,
            headers={"X-Hub-Signature-256": sig})
        self.assertEqual(status, 403, payload)
        self.assertEqual(payload["kind"], "CapabilityDenied")

    def test_github_direct_tampered_body_is_403(self) -> None:
        raw = _github_push_body()
        sig = _github_sig("whsec", raw)
        tampered = json.dumps({
            "ref": "refs/heads/main",
            "repository": {"full_name": "Oluwacutyp/nomorals-2.0"},
            "pusher": {"name": "Oluwacutyp"},
            "commits": [{"id": "TAMPERED", "message": "test"}],
        }).encode()
        status, payload = _post_raw(
            self.port, "/triggers/webhook?trigger=" + self.gh_tid, tampered,
            headers={"X-Hub-Signature-256": sig})
        self.assertEqual(status, 403, payload)

    def test_stripe_direct_delivery_fires(self) -> None:
        raw = _stripe_event_body()
        ts = str(int(time.time()))
        sig = _stripe_sig("whsec", ts, raw)
        status, payload = _post_raw(
            self.port, "/triggers/webhook?trigger=" + self.st_tid, raw,
            headers={"Stripe-Signature": sig})
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])
        delivered = self._history_payload(self.st_tid)
        self.assertEqual(delivered.get("id"), "evt_3P1234567890abcdef")
        self.assertEqual(delivered.get("type"),
                         "payment_intent.succeeded")

    def test_stripe_direct_delivery_replay_via_event_id(self) -> None:
        raw = _stripe_event_body()
        ts = str(int(time.time()))
        sig = _stripe_sig("whsec", ts, raw)
        path = "/triggers/webhook?trigger=" + self.st_tid
        headers = {"Stripe-Signature": sig}
        status, payload = _post_raw(self.port, path, raw, headers=headers)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])
        status, payload = _post_raw(self.port, path, raw, headers=headers)
        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["fired"])
        self.assertTrue(payload["duplicate"])
        eng = TriggerEngine(self.db, notify_fn=lambda *a: {"sent": True})
        self.assertEqual(eng.get(self.st_tid).fire_count, 1)

    def test_stripe_direct_stale_timestamp_is_403(self) -> None:
        raw = _stripe_event_body()
        ts = str(int(time.time()) - 3600)
        sig = _stripe_sig("whsec", ts, raw)
        status, payload = _post_raw(
            self.port, "/triggers/webhook?trigger=" + self.st_tid, raw,
            headers={"Stripe-Signature": sig})
        self.assertEqual(status, 403, payload)
        self.assertIn("tolerance", payload["error"])

    def test_missing_trigger_identification_is_400(self) -> None:
        raw = _github_push_body()
        sig = _github_sig("whsec", raw)
        status, payload = _post_raw(
            self.port, "/triggers/webhook", raw,
            headers={"X-Hub-Signature-256": sig})
        self.assertEqual(status, 400, payload)
        self.assertIn("trigger", payload["error"])

    def test_envelope_path_unchanged(self) -> None:
        # The N4 relay/envelope contract still works: signature in the
        # query string, trigger_id + payload in the body.
        raw = (
            b'{"payload": {"b": 1, "a": 2}, "trigger_id": "' +
            self.gh_tid.encode() + b'", "event_id": "e-env-1"}'
        )
        sig = _github_sig("whsec", raw)
        status, payload = _post_raw(
            self.port,
            "/triggers/webhook?signature=" + quote(sig, safe=""),
            raw)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])

    def test_relay_envelope_with_forwarded_provider_header_still_fires(
            self) -> None:
        # A relay that forwards the provider's original signature header
        # along with its own envelope signature must not break: the
        # header candidate is over different bytes (the provider event,
        # not the envelope) and fails; the envelope signature verifies.
        raw = (
            b'{"payload": {"b": 1, "a": 2}, "trigger_id": "' +
            self.gh_tid.encode() + b'", "event_id": "e-relay-1"}'
        )
        sig = _github_sig("whsec", raw)
        stale_provider_sig = _github_sig(
            "whsec", b'{"ref": "refs/heads/main"}')
        status, payload = _post_raw(
            self.port,
            "/triggers/webhook?signature=" + quote(sig, safe=""),
            raw,
            headers={"X-Hub-Signature-256": stale_provider_sig})
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])


class DispatchHeadersTests(unittest.TestCase):
    """Direct dispatch: the headers plumbing reaches the route."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hdr-dispatch-")
        self.db = Database(f"{self.tmp}/t.db")
        self.api = APIServer(
            _context(self.db, self.tmp), token=OWNER_TOKEN,
            principals={RO_TOKEN: ("ro", CapabilitySet.of(Capability.DB_READ))})
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")
        eng = TriggerEngine(self.db, notify_fn=lambda *a: {"sent": True})
        gh = eng.add("gh", "webhook",
                     {"secret": "whsec", "scheme": "github"}, "notify", {})
        self.gh_tid = gh.id

    def test_dispatch_passes_headers_to_route(self) -> None:
        raw = _github_push_body()
        sig = _github_sig("whsec", raw)
        body = json.loads(raw.decode("utf-8"))
        status, payload = self.api.dispatch(
            "POST", "/triggers/webhook", body,
            {"trigger": self.gh_tid}, principal=self.owner, raw_body=raw,
            headers={"X-Hub-Signature-256": sig})
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])

    def test_dispatch_headers_ignored_for_plain_handlers(self) -> None:
        # A handler that does not declare ``headers`` keeps the plain
        # (body, query) call shape — passing headers to dispatch must
        # not break it.
        status, payload = self.api.dispatch(
            "GET", "/live", {}, {}, principal=self.owner,
            headers={"X-Hub-Signature-256": "sha256=deadbeef"})
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
