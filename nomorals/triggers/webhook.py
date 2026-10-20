"""Webhook source: HTTP endpoint for trigger firing.

Hangs on the existing API server (``nomorals.api.server.APIServer``,
L7): the server imports this module (downward), never the reverse.

``POST /triggers/webhook`` with JSON body::

    {"trigger_id": "<id>", "secret": "<optional>", "payload": {...},
     "signature": "<hmac, for hmac schemes>", "timestamp": "<ts>",
     "event_id": "<idempotency key>"}

Auth schemes (set on the trigger's condition): ``plain`` compares the
shared ``secret``; ``github`` verifies ``signature`` as
``sha256=<hmac_sha256(body)>``; ``stripe`` verifies ``signature`` as
``t=<ts>,v1=<hmac_sha256(ts + "." + body)>`` with a timestamp tolerance
(default 300s, ``tolerance_s``).  The server route only sees the parsed
body, so HMAC is computed over the canonical re-encoding of
``payload`` — for raw-byte verification, call
``TriggerEngine.fire_webhook(..., raw_body=<bytes>)`` directly.

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
* unknown id, non-webhook trigger, disabled trigger — **400**
  problem body with the real cause.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

WEBHOOK_PATH = "/triggers/webhook"


def register_trigger_routes(server: Any, context: Any) -> Any:
    """Register the trigger webhook route on an API server instance."""

    @server.route("POST", WEBHOOK_PATH)
    def trigger_webhook(body: dict[str, Any],
                        query: dict[str, str]) -> dict[str, Any]:
        from ..core.errors import CapabilityDenied, NoMoralsError
        from .engine import TriggerEngine
        from .models import (
            TriggerError,
            WebhookAuthError,
            WebhookReplayError,
        )

        data = body or {}
        query = query or {}
        engine = TriggerEngine(context.db, context)
        try:
            result = engine.fire_webhook(
                str(data.get("trigger_id") or ""),
                payload=data.get("payload"),
                secret=data.get("secret") or query.get("secret"),
                signature=str(data.get("signature") or "")
                or str(query.get("signature") or ""),
                timestamp=data.get("timestamp") or query.get("timestamp"),
                event_id=str(data.get("event_id") or "")
                or str(query.get("event_id") or ""),
            )
        except WebhookAuthError as exc:
            # Non-2xx: the sender must see this delivery as FAILED.
            _log.info("trigger webhook auth rejected: %s", exc)
            raise CapabilityDenied(str(exc)) from exc
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

    _log.info("registered trigger webhook route POST %s", WEBHOOK_PATH)
    return trigger_webhook
