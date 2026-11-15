"""Webhook source: HTTP endpoint for trigger firing.

Hangs on the existing API server (``nomorals.api.server.APIServer``,
L7): the server imports this module (downward), never the reverse.

Two delivery shapes land on ``POST /triggers/webhook``:

* **Envelope / relay delivery** (the pass #51 + N4 contract) — JSON
  body ``{"trigger_id": "<id>", "secret": "<optional>", "payload":
  {...}, "signature": "<hmac, for hmac schemes>", "timestamp": "<ts>",
  "event_id": "<idempotency key>"}``. The signature travels outside
  the signed bytes, in the query string (``?signature=...``).
* **Direct provider delivery** — the provider's event as the body
  (``trigger_id`` is NOT in it), the trigger identified in the query
  string (``?trigger=<id>``), the signature in the provider's header
  (GitHub ``X-Hub-Signature-256``, Stripe ``Stripe-Signature``). The
  whole body is the payload the action sees; the idempotency key comes
  from ``X-GitHub-Delivery`` (GitHub) or the body's ``id`` (Stripe
  ``evt_...``).

Auth schemes (set on the trigger's condition): ``plain`` compares the
shared ``secret``; ``github`` verifies ``signature`` as
``sha256=<hmac_sha256(body)>``; ``stripe`` verifies ``signature`` as
``t=<ts>,v1=<hmac_sha256(ts + "." + body)>`` with a timestamp tolerance
(default 300s, ``tolerance_s``).  For the HMAC schemes the HMAC is
computed over the **raw request body bytes** exactly as received over
HTTP — the bytes the sender signed.

Signature candidates are tried in order — the provider header first,
then the envelope/query signature — and the first that verifies wins.
A relay that forwards the provider's original header along with its
own envelope signature therefore still fires (the header candidate is
over different bytes and fails; the envelope candidate verifies).

Response contract (pass #51 — external senders read the HTTP status,
not our JSON, so a rejection answered 200 was a silent failure for
them):

* fired / evaluated — ``200`` with ``{"ok": true, ...}``.
* authentication failed (bad secret, bad/missing signature, stale
  timestamp) — **403** problem body (the dispatcher's
  ``CapabilityDenied`` mapping).  Senders mark the delivery failed
  and retry/alert instead of believing it landed.
* replayed ``event_id`` — ``200`` with ``{"ok": true, "fired":
  false, "duplicate": true}``: the delivery was authentic and
  already processed, so it is acknowledged (senders stop retrying)
  but never re-fires.
* unknown id, non-webhook trigger, disabled trigger, missing trigger
  identification — **400** problem body with the real cause.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

WEBHOOK_PATH = "/triggers/webhook"


def _stripe_event_id(data: dict[str, Any]) -> str:
    """The Stripe event id (``evt_...``) from a direct-delivery body —
    the idempotency key when no explicit ``event_id`` travels with the
    delivery. Anything that is not a Stripe event id is not one."""
    evt = data.get("id")
    return evt if isinstance(evt, str) and evt.startswith("evt_") else ""


def register_trigger_routes(server: Any, context: Any) -> Any:
    """Register the trigger webhook route on an API server instance."""

    @server.route("POST", WEBHOOK_PATH)
    def trigger_webhook(body: dict[str, Any],
                        query: dict[str, str],
                        raw_body: bytes = b"",
                        headers: dict[str, str] | None = None
                        ) -> dict[str, Any]:
        from ..core.errors import CapabilityDenied, NoMoralsError
        from .engine import TriggerEngine
        from .models import (
            TriggerError,
            WebhookAuthError,
            WebhookReplayError,
        )

        data = body or {}
        query = query or {}
        headers = headers or {}
        # The trigger is identified in the body (envelope/relay) or in
        # the query string (direct provider delivery — the provider's
        # event carries no trigger_id). The store has no provider →
        # trigger mapping, so the query string is the identification.
        trigger_id = (
            str(data.get("trigger_id") or "")
            or str(query.get("trigger") or "")
            or str(query.get("trigger_id") or "")
        )
        if not trigger_id:
            raise NoMoralsError(
                "missing trigger_id — a direct provider delivery "
                "identifies its trigger in the query string "
                "(?trigger=<id>); an envelope delivery carries "
                "trigger_id in the body")
        # Signature candidates, in order: the provider's header first,
        # then the envelope/query signature. The first that verifies
        # wins (see the module docstring).
        candidates: list[str] = []
        header_sig = (
            str(headers.get("x-hub-signature-256") or "")
            or str(headers.get("stripe-signature") or "")
        )
        if header_sig:
            candidates.append(header_sig)
        envelope_sig = (
            str(data.get("signature") or "")
            or str(query.get("signature") or "")
        )
        if envelope_sig and envelope_sig not in candidates:
            candidates.append(envelope_sig)
        if not candidates:
            candidates = [""]
        # Direct provider delivery: the whole body IS the event payload.
        # Envelope delivery: the inner payload.
        payload = data.get("payload") if "payload" in data else data
        event_id = (
            str(data.get("event_id") or "")
            or str(query.get("event_id") or "")
            or str(headers.get("x-github-delivery") or "")
            or _stripe_event_id(data)
        )
        engine = TriggerEngine(context.db, context)
        last_auth_error: WebhookAuthError | None = None
        for signature in candidates:
            try:
                result = engine.fire_webhook(
                    trigger_id,
                    payload=payload,
                    secret=data.get("secret") or query.get("secret"),
                    signature=signature,
                    timestamp=data.get("timestamp")
                    or query.get("timestamp"),
                    event_id=event_id,
                    raw_body=raw_body,
                )
            except WebhookAuthError as exc:
                # Auth happens before any state change (the dedup claim
                # is after auth passes), so trying the next candidate is
                # side-effect-free. A plain-scheme trigger ignores the
                # signature and fails on the secret the same way twice.
                last_auth_error = exc
                continue
            except WebhookReplayError as exc:
                # Authentic but already processed: acknowledge (200) so
                # the sender stops retrying; the action does not re-fire.
                _log.info("trigger webhook duplicate delivery: %s", exc)
                return {"ok": True, "fired": False, "duplicate": True,
                        "error": str(exc)}
            except TriggerError as exc:
                _log.info("trigger webhook rejected: %s", exc)
                raise NoMoralsError(str(exc)) from exc
            return {"ok": True, **result}
        # Every candidate failed auth: the sender must see this delivery
        # as FAILED.
        _log.info("trigger webhook auth rejected: %s", last_auth_error)
        raise CapabilityDenied(str(last_auth_error)) from last_auth_error

    _log.info("registered trigger webhook route POST %s", WEBHOOK_PATH)
    return trigger_webhook
