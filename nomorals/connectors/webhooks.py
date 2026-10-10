"""Inbound webhook primitives shared by every connector.

Three providers, three HMAC schemes, one verifier — Stripe's 2026 best
practice distilled:

* verify against the **raw body**, never re-serialized JSON;
* constant-time signature comparison (timing attacks are real);
* reject stale timestamps (replay mitigation);
* at-least-once delivery is the norm → dedupe by event id in the same
  breath as the business work (:class:`WebhookDeduper`).

Connectors that verify webhooks (Paystack already did; Stripe and GitHub
now do) delegate to :func:`verify_signature` instead of hand-rolling
their own HMAC — one scheme per provider, implemented once, tested once.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .base import ConnectorError, ConnectorValidationError

__all__ = [
    "WebhookDeduper",
    "WebhookEvent",
    "parse_event",
    "supported_providers",
    "verify_signature",
]

#: Max age (seconds) for a timestamped webhook signature (Stripe scheme).
#: Stripe's SDK defaults to 300s; captured webhooks older than this are
#: rejected as replays.
DEFAULT_REPLAY_WINDOW = 300.0

_PROVIDERS = ("stripe", "paystack", "github")


def supported_providers() -> tuple[str, ...]:
    """Providers with a known signature scheme."""
    return _PROVIDERS


def _hmac_hex(secret: str, payload: bytes, digest: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), payload, getattr(hashlib, digest)
    ).hexdigest()


def _verify_stripe(
    payload: bytes, signature: str, secret: str,
    *,
    replay_window: float,
    now: float,
) -> None:
    """Stripe scheme: ``t=<ts>,v1=<hmac>`` over ``"<ts>.<raw-body>"``."""
    parts: dict[str, str] = {}
    for chunk in (signature or "").split(","):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            parts[k.strip()] = v.strip()
    ts = parts.get("t", "")
    v1 = parts.get("v1", "")
    if not ts or not v1:
        raise ConnectorValidationError(
            "stripe webhook signature malformed: expected 't=...,v1=...'"
        )
    try:
        ts_f = float(ts)
    except ValueError as exc:
        raise ConnectorValidationError(
            "stripe webhook signature has a non-numeric timestamp"
        ) from exc
    if abs(now - ts_f) > replay_window:
        raise ConnectorValidationError(
            f"stripe webhook timestamp {ts} is outside the "
            f"{replay_window:.0f}s replay window — possible replay attack"
        )
    expected = _hmac_hex(secret, f"{ts}.".encode() + payload, "sha256")
    if not hmac.compare_digest(expected, v1):
        raise ConnectorValidationError(
            "stripe webhook signature mismatch — wrong secret or "
            "tampered payload"
        )


def _verify_paystack(payload: bytes, signature: str, secret: str) -> None:
    """Paystack scheme: ``x-paystack-signature`` = HMAC-SHA512 hex."""
    expected = _hmac_hex(secret, payload, "sha512")
    if not hmac.compare_digest(expected, (signature or "").strip().lower()):
        raise ConnectorValidationError(
            "paystack webhook signature mismatch — wrong secret or "
            "tampered payload"
        )


def _verify_github(payload: bytes, signature: str, secret: str) -> None:
    """GitHub scheme: ``x-hub-signature-256`` = ``sha256=<hmac>``."""
    sig = (signature or "").strip()
    if not sig.startswith("sha256="):
        raise ConnectorValidationError(
            "github webhook signature malformed: expected 'sha256=...'"
        )
    expected = "sha256=" + _hmac_hex(secret, payload, "sha256")
    if not hmac.compare_digest(expected, sig):
        raise ConnectorValidationError(
            "github webhook signature mismatch — wrong secret or "
            "tampered payload"
        )


def verify_signature(
    provider: str,
    payload: bytes,
    signature: str,
    secret: str,
    *,
    replay_window: float = DEFAULT_REPLAY_WINDOW,
    now: float | None = None,
) -> None:
    """Verify an inbound webhook signature; raises on any problem.

    ``provider`` is one of ``"stripe"``, ``"paystack"``, ``"github"``.
    ``payload`` must be the **raw request body bytes**. ``signature`` is
    the provider's signature header value. ``secret`` is the webhook
    signing secret (never logged — only presence is ever reported).

    Raises :class:`ConnectorValidationError` (a ``ConnectorError``) when
    the signature is missing, malformed, stale, or mismatched. Returns
    ``None`` on success.
    """
    if not secret:
        raise ConnectorError(
            f"cannot verify {provider} webhook: no signing secret configured"
        )
    if provider == "stripe":
        _verify_stripe(payload, signature, secret,
                       replay_window=replay_window,
                       now=time.time() if now is None else now)
    elif provider == "paystack":
        _verify_paystack(payload, signature, secret)
    elif provider == "github":
        _verify_github(payload, signature, secret)
    else:
        raise ConnectorError(
            f"unknown webhook provider {provider!r}; "
            f"supported: {', '.join(_PROVIDERS)}"
        )


@dataclass
class WebhookEvent:
    """One normalized inbound webhook event (provider-agnostic)."""

    id: str
    provider: str
    type: str
    created_at: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "type": self.type,
            "created_at": self.created_at,
            "data": dict(self.data),
        }


def parse_event(provider: str, payload: bytes | str | dict[str, Any]) -> WebhookEvent:
    """Parse a webhook body into a normalized :class:`WebhookEvent`.

    Verifies nothing — call :func:`verify_signature` first. Raises
    :class:`ConnectorValidationError` on unparsable bodies.
    """
    if isinstance(payload, dict):
        body = payload
    else:
        raw = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ConnectorValidationError(
                f"{provider} webhook body is not valid JSON"
            ) from exc
    if not isinstance(body, dict):
        raise ConnectorValidationError(
            f"{provider} webhook body is not a JSON object"
        )
    if provider == "stripe":
        return WebhookEvent(
            id=str(body.get("id", "")),
            provider=provider,
            type=str(body.get("type", "")),
            created_at=float(body.get("created", 0) or 0),
            data=body.get("data", {}).get("object", {})
            if isinstance(body.get("data"), dict) else {},
            raw=body,
        )
    if provider == "paystack":
        return WebhookEvent(
            id=str(body.get("id", "")) or str(body.get("reference", "")),
            provider=provider,
            type=str(body.get("event", "")),
            created_at=time.time(),
            data=body.get("data", {}) if isinstance(body.get("data"), dict) else {},
            raw=body,
        )
    if provider == "github":
        return WebhookEvent(
            id=str(body.get("delivery", "")),
            provider=provider,
            type=str(body.get("event", "")),
            created_at=time.time(),
            data={k: v for k, v in body.items()
                  if k not in ("delivery", "event")},
            raw=body,
        )
    raise ConnectorError(f"unknown webhook provider {provider!r}")


class WebhookDeduper:
    """At-least-once delivery guard: remember seen event ids with a TTL.

    Stripe retries webhooks with exponential backoff for up to 72h; the
    same ``event.id`` arrives multiple times. ``check_and_mark`` returns
    True when this id was already seen (skip the business work), False on
    first sight (do the work, in the same breath). Thread-safe, in-memory,
    TTL-purged — for durable dedupe, persist ``event.id`` with a UNIQUE
    constraint in the app DB alongside the business write.
    """

    def __init__(self, ttl_seconds: float = 7 * 86400.0) -> None:
        self.ttl = max(60.0, float(ttl_seconds))
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_mark(self, event_id: str, *, now: float | None = None) -> bool:
        """True if ``event_id`` was already seen (duplicate); else mark it."""
        now = time.time() if now is None else now
        with self._lock:
            self._purge_locked(now)
            if event_id in self._seen:
                return True
            self._seen[event_id] = now
            return False

    def seen(self, event_id: str) -> bool:
        with self._lock:
            self._purge_locked(time.time())
            return event_id in self._seen

    def purge(self, now: float | None = None) -> int:
        """Drop expired entries. Returns the number removed."""
        with self._lock:
            return self._purge_locked(time.time() if now is None else now)

    def _purge_locked(self, now: float) -> int:
        expired = [k for k, ts in self._seen.items() if now - ts > self.ttl]
        for k in expired:
            del self._seen[k]
        return len(expired)

    def __len__(self) -> int:
        with self._lock:
            self._purge_locked(time.time())
            return len(self._seen)
