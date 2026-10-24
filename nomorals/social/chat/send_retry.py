"""Retry with classification for chat sends (G1 + G6).

Adapters report ordinary failures as :class:`SendResult` (they never
raise), so classification works on the result's ``error`` string as well
as on raised exceptions. Classes:

* **transient** — ``429`` / ``5xx`` / timeout / connection reset:
  exponential backoff + jitter, retry (default 5 attempts).
* **terminal** — ``400`` / chat-not-found / blocked / invalid token:
  fail loudly, **never retry** (exactly 1 attempt).
* **unknown** — anything else: retry twice (3 attempts), then treat as
  terminal.

The loop is built on :mod:`nomorals.core.retry` primitives
(:class:`BackoffPolicy` for the schedule, :class:`RetryBudget` per
platform so one dead platform cannot consume the queue) rather than
reimplementing backoff. It does not use :func:`retry_call` directly
because the send function returns a result instead of raising — the
classification step is what turns a result into a retry decision.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from ...core.errors import (
    AuthError,
    CapabilityDenied,
    NoMoralsError,
    NotFound,
    ParseError,
    ProviderError,
    RateLimited,
    TimeoutError_,
    ValidationError,
    classify,
)
from ...core.retry import BackoffPolicy, RetryBudget
from .base import SendResult
from .failures import record_failure

__all__ = [
    "classify_send_error",
    "classify_error",
    "send_with_retry",
    "get_retry_budget",
    "reset_retry_budgets",
    "DEFAULT_TRANSIENT_ATTEMPTS",
    "DEFAULT_UNKNOWN_ATTEMPTS",
]

DEFAULT_TRANSIENT_ATTEMPTS = 5
DEFAULT_UNKNOWN_ATTEMPTS = 3  # 1 attempt + 2 retries

#: Substring patterns (lowercased error text). Terminal is checked FIRST:
#: a permanent error must never be retried even if it also mentions a
#: transient-sounding word.
_TERMINAL_PATTERNS: tuple[str, ...] = (
    "chat not found", "chat_not_found", "peer_id_invalid",
    "user not found", "user_not_found",
    "bot was blocked", "blocked by user", "blocked", "forbidden",
    "chat_write_forbidden", "not a member", "kicked", "deactivated",
    "unauthorized", "unauthorised", "invalid token", "invalid api",
    "auth failed", "authentication", "invalid", "bad request",
    "400", "401", "403", "404", "revoked",
    "no adapter", "not configured", "no destination",
    "file not found", "media has no path",
    "message too long", "too long", "not found", "deleted", "disabled",
    "twilio not configured",
)

_TRANSIENT_PATTERNS: tuple[str, ...] = (
    "429", "rate limit", "rate_limited", "too many requests", "flood",
    "500", "502", "503", "504", "server error", "bad gateway",
    "service unavailable", "gateway timeout", "timeout", "timed out",
    "connection reset", "connection refused", "connection aborted",
    "connection error", "econnreset", "econnrefused",
    "temporarily", "try again", "network", "unreachable",
    "not connected", "temporarily unavailable", "overloaded",
    "over capacity", "slow down",
)

#: Framework error classes whose meaning is already decided.
_TERMINAL_CLASSES: tuple[type[NoMoralsError], ...] = (
    AuthError, NotFound, ValidationError, CapabilityDenied, ParseError,
)
_TRANSIENT_CLASSES: tuple[type[NoMoralsError], ...] = (
    RateLimited, TimeoutError_, ProviderError,
)


def classify_send_error(error: str | BaseException | None) -> str:
    """``"transient"`` | ``"terminal"`` | ``"unknown"`` for a send failure.

    Accepts a :class:`SendResult` error string, a raised exception, or
    None (unknown). Never raises.
    """
    try:
        if error is None:
            return "unknown"
        if isinstance(error, BaseException):
            classified = classify(error)
            text = str(error) or type(error).__name__
            lowered = text.lower()
            if any(p in lowered for p in _TERMINAL_PATTERNS):
                return "terminal"
            if isinstance(classified, _TERMINAL_CLASSES):
                return "terminal"
            if isinstance(classified, _TRANSIENT_CLASSES):
                return "transient"
            if classified.retryable:
                return "transient"
            if any(p in lowered for p in _TRANSIENT_PATTERNS):
                return "transient"
            return "unknown"
        lowered = str(error).lower()
        if not lowered.strip():
            return "unknown"
        if any(p in lowered for p in _TERMINAL_PATTERNS):
            return "terminal"
        if any(p in lowered for p in _TRANSIENT_PATTERNS):
            return "transient"
        return "unknown"
    except Exception as exc:  # noqa: BLE001 - classification never breaks a send
        record_failure('send_retry.classify_send_error', exc)
        return "unknown"


#: Alias — the name the spec uses.
classify_error = classify_send_error


#: Per-platform retry budgets. Ratio 0.5 (a retry costs 2 tokens, a call
#: deposits 1) bounds retry storms while letting a genuinely flaky
#: platform recover; the 20-token floor lets a fresh platform retry
#: through a short outage before the budget bites.
_BUDGETS: dict[str, RetryBudget] = {}
_BUDGETS_LOCK = threading.Lock()


def get_retry_budget(platform: str) -> RetryBudget:
    """The retry budget for one platform (created on first use)."""
    key = str(platform or "default").strip().lower() or "default"
    with _BUDGETS_LOCK:
        budget = _BUDGETS.get(key)
        if budget is None:
            budget = RetryBudget(retry_ratio=0.5, min_tokens=20.0)
            _BUDGETS[key] = budget
        return budget


def reset_retry_budgets() -> None:
    """Drop all per-platform budgets (tests)."""
    with _BUDGETS_LOCK:
        _BUDGETS.clear()


def send_with_retry(
    send_fn: Callable[[], SendResult],
    *,
    platform: str = "",
    policy: BackoffPolicy | None = None,
    budget: RetryBudget | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_transient_attempts: int = DEFAULT_TRANSIENT_ATTEMPTS,
    max_unknown_attempts: int = DEFAULT_UNKNOWN_ATTEMPTS,
) -> tuple[SendResult, int]:
    """Call ``send_fn`` with classified retries. Never raises: an
    exception from ``send_fn`` is converted to a failed result and
    classified like any other error.

    Returns ``(result, attempts)``. Terminal errors cost exactly 1
    attempt; transient errors retry up to ``max_transient_attempts``;
    unknown errors retry up to ``max_unknown_attempts``. When the
    per-platform budget is spent, the last result is returned immediately
    instead of retrying — the failure is loud, not looped.
    """
    effective_policy = policy or BackoffPolicy(
        base=0.5, factor=2.0, cap=30.0, max_attempts=DEFAULT_TRANSIENT_ATTEMPTS,
    )
    effective_budget = budget if budget is not None else get_retry_budget(platform)
    effective_budget.before_call()
    attempts = 0
    last = SendResult(ok=False, platform=platform, error="send failed")
    while True:
        attempts += 1
        try:
            result = send_fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            result = SendResult(
                ok=False, platform=platform,
                error=f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(result, SendResult):
            result = SendResult(ok=False, platform=platform,
                                error=f"send returned {type(result).__name__}")
        if result.ok:
            return result, attempts
        last = result
        kind = classify_send_error(result.error)
        if kind == "terminal":
            limit = 1
        elif kind == "transient":
            limit = max(1, int(max_transient_attempts))
        else:
            limit = max(1, int(max_unknown_attempts))
        if attempts >= limit:
            return result, attempts
        if not effective_budget.allow_retry():
            return result, attempts
        delay = effective_policy.delay(attempts)
        if delay > 0:
            sleep(delay)
