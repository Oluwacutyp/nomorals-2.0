"""N4 — webhook HMAC runs over the raw request bytes (live HTTP).

Pass #51 pinned the webhook contract through direct ``dispatch`` calls,
where no raw body exists, so the engine falls back to the canonical
re-encoding of ``payload``. Real senders sign the exact bytes they send:
a GitHub delivery whose JSON re-encodes differently (key order,
whitespace) fails verification even though the sender signed it
correctly. These pins prove the contract end to end over live HTTP:

* a signature over the raw request bytes verifies (github + stripe);
* a signature over the canonical re-encoding is REJECTED over HTTP
  (the old way stops working once raw bytes reach the route — the
  contract change this lane makes, pinned explicitly);
* wrong secret / tampered body / stale timestamp still 403;
* replay is still acknowledged (200, ``duplicate: true``) without
  re-firing;
* the plain shared-secret scheme is untouched by the byte change.

The signature must travel outside the signed bytes (query string
``?signature=...`` here — a signature inside the body cannot sign the
body that contains it). Direct ``dispatch`` without ``raw_body`` keeps
the canonical fallback (pass #51's contract, unchanged).
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

OWNER_TOKEN = "n4-owner-token"
RO_TOKEN = "n4-ro-token"


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


def _post_raw(port: int, path: str, raw: bytes) -> tuple[int, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", path, body=raw, headers={
            "Content-Type": "application/json"})
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


class RawBodyWebhookTests(unittest.TestCase):
    """Live-HTTP pins: HMAC over the raw request bytes."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="n4-")
        self.db = Database(f"{self.tmp}/t.db")
        eng = TriggerEngine(self.db, notify_fn=lambda *a: {"sent": True})
        gh = eng.add("gh", "webhook",
                     {"secret": "whsec", "scheme": "github"}, "notify", {})
        st = eng.add("st", "webhook",
                     {"secret": "whsec", "scheme": "stripe"}, "notify", {})
        pl = eng.add("pl", "webhook", {"secret": "s3cr3t"}, "notify", {})
        self.gh_tid, self.st_tid, self.pl_tid = gh.id, st.id, pl.id
        self.port = _free_port()
        serve(_context(self.db, self.tmp), host="127.0.0.1",
              port=self.port, background=True, protocols=False, token="")
        _wait_up(self.port)

    def _envelope_raw(self, tid: str, payload: dict[str, Any],
                      event_id: str) -> bytes:
        # Deliberately non-canonical: keys out of sorted order, spaces
        # after separators — the canonical re-encoding of this body is
        # different bytes, so a signature over one does not match the other.
        return (
            b'{"payload": ' + json.dumps(payload).encode() +
            b', "trigger_id": "' + tid.encode() +
            b'", "event_id": "' + event_id.encode() + b'"}'
        )

    def test_github_signature_over_raw_body_fires(self) -> None:
        raw = self._envelope_raw(self.gh_tid, {"b": 1, "a": 2}, "e-raw-1")
        sig = _github_sig("whsec", raw)
        status, payload = _post_raw(
            self.port,
            "/triggers/webhook?signature=" + quote(sig, safe=""),
            raw)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])

    def test_github_canonical_signature_in_body_rejected_over_http(self) -> None:
        # Contract change, pinned explicitly: over live HTTP the HMAC is
        # over the raw request bytes. A signature computed over the
        # canonical re-encoding of ``payload`` (the old way) no longer
        # verifies — the sender must sign the bytes it sends.
        canonical = json.dumps({"a": 2, "b": 1}, sort_keys=True,
                               separators=(",", ":")).encode()
        sig = _github_sig("whsec", canonical)
        raw = json.dumps({
            "trigger_id": self.gh_tid, "payload": {"b": 1, "a": 2},
            "signature": sig, "event_id": "e-old-1"}).encode()
        status, payload = _post_raw(self.port, "/triggers/webhook", raw)
        self.assertEqual(status, 403, payload)

    def test_github_wrong_signature_is_403(self) -> None:
        raw = self._envelope_raw(self.gh_tid, {"b": 1, "a": 2}, "e-raw-2")
        sig = _github_sig("WRONG", raw)
        status, payload = _post_raw(
            self.port,
            "/triggers/webhook?signature=" + quote(sig, safe=""),
            raw)
        self.assertEqual(status, 403, payload)
        self.assertEqual(payload["kind"], "CapabilityDenied")

    def test_github_tampered_body_is_403(self) -> None:
        raw = self._envelope_raw(self.gh_tid, {"b": 1, "a": 2}, "e-raw-3")
        sig = _github_sig("whsec", raw)
        tampered = self._envelope_raw(
            self.gh_tid, {"b": 1, "a": 999}, "e-raw-3")
        status, payload = _post_raw(
            self.port,
            "/triggers/webhook?signature=" + quote(sig, safe=""),
            tampered)
        self.assertEqual(status, 403, payload)

    def test_stripe_signature_over_raw_body_fires(self) -> None:
        raw = self._envelope_raw(self.st_tid, {"type": "evt"}, "e-st-1")
        ts = str(int(time.time()))
        sig = _stripe_sig("whsec", ts, raw)
        status, payload = _post_raw(
            self.port,
            "/triggers/webhook?signature=" + quote(sig, safe=""),
            raw)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])

    def test_stripe_stale_timestamp_is_403(self) -> None:
        raw = self._envelope_raw(self.st_tid, {"type": "evt"}, "e-st-2")
        ts = str(int(time.time()) - 3600)
        sig = _stripe_sig("whsec", ts, raw)
        status, payload = _post_raw(
            self.port,
            "/triggers/webhook?signature=" + quote(sig, safe=""),
            raw)
        self.assertEqual(status, 403, payload)
        self.assertIn("tolerance", payload["error"])

    def test_replay_acknowledged_not_refired_over_http(self) -> None:
        raw = self._envelope_raw(self.gh_tid, {"b": 1}, "e-replay-1")
        sig = _github_sig("whsec", raw)
        path = "/triggers/webhook?signature=" + quote(sig, safe="")
        status, payload = _post_raw(self.port, path, raw)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])
        status, payload = _post_raw(self.port, path, raw)
        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["fired"])
        self.assertTrue(payload["duplicate"])
        eng = TriggerEngine(self.db, notify_fn=lambda *a: {"sent": True})
        self.assertEqual(eng.get(self.gh_tid).fire_count, 1)

    def test_plain_secret_still_fires_over_http(self) -> None:
        raw = json.dumps({
            "trigger_id": self.pl_tid, "secret": "s3cr3t",
            "payload": {}}).encode()
        status, payload = _post_raw(self.port, "/triggers/webhook", raw)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])


class DispatchRawBodyTests(unittest.TestCase):
    """Direct dispatch: the raw_body plumbing reaches the route."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="n4-dispatch-")
        self.db = Database(f"{self.tmp}/t.db")
        self.api = APIServer(
            _context(self.db, self.tmp), token=OWNER_TOKEN,
            principals={RO_TOKEN: ("ro", CapabilitySet.of(Capability.DB_READ))})
        self.owner = self.api.resolve_principal(f"Bearer {OWNER_TOKEN}")
        eng = TriggerEngine(self.db, notify_fn=lambda *a: {"sent": True})
        gh = eng.add("gh", "webhook",
                     {"secret": "whsec", "scheme": "github"}, "notify", {})
        self.gh_tid = gh.id

    def test_dispatch_passes_raw_body_to_route(self) -> None:
        raw = (
            b'{"payload": {"b": 1, "a": 2}, "trigger_id": "' +
            self.gh_tid.encode() + b'", "event_id": "e-d-1"}'
        )
        sig = _github_sig("whsec", raw)
        body = {"trigger_id": self.gh_tid, "payload": {"b": 1, "a": 2},
                "event_id": "e-d-1"}
        status, payload = self.api.dispatch(
            "POST", "/triggers/webhook", body, {"signature": sig},
            principal=self.owner, raw_body=raw)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["fired"])


if __name__ == "__main__":
    unittest.main()
