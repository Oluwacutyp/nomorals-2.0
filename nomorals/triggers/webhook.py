"""Webhook source: HTTP endpoint for trigger firing.

Hangs on the existing API server (``nomorals.api.server.APIServer``,
L7): the server imports this module (downward), never the reverse.

``POST /triggers/webhook`` with JSON body::

    {"trigger_id": "<id>", "secret": "<optional>", "payload": {...}}

The trigger's own secret (if configured) is the auth; a wrong secret,
unknown id, non-webhook trigger, or disabled trigger fails fast with a
clear ``{"ok": false, "error": ...}`` body — never a silent drop.
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
        from .engine import TriggerEngine
        from .models import TriggerError

        data = body or {}
        engine = TriggerEngine(context.db, context)
        try:
            result = engine.fire_webhook(
                str(data.get("trigger_id") or ""),
                payload=data.get("payload"),
                secret=data.get("secret"),
            )
        except TriggerError as exc:
            _log.info("trigger webhook rejected: %s", exc)
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **result}

    _log.info("registered trigger webhook route POST %s", WEBHOOK_PATH)
    return trigger_webhook
