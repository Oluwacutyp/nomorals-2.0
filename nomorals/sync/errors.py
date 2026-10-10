"""Sync errors.

The hierarchy separates *transient* failures (worth retrying) from
*permanent* ones (retrying is pointless), so retry policy can be correct
without string-matching messages:

- :class:`SyncConnectionError` — transport failure after retries. Transient.
- :class:`SyncAuthError` — the hub rejected our credentials (HTTP 401).
  Permanent until the token changes.
- :class:`SyncHubError` — the hub answered with an error status (other
  4xx/5xx). Carries ``status_code`` and ``detail``.

Everything subclasses :class:`SyncError`, so ``except SyncError`` keeps
working everywhere it did before.
"""

from __future__ import annotations

__all__ = [
    "SyncError",
    "SyncConnectionError",
    "SyncAuthError",
    "SyncHubError",
]


class SyncError(Exception):
    """Base for all sync errors."""


class SyncConnectionError(SyncError):
    """The peer could not be reached (transient — safe to retry)."""


class SyncAuthError(SyncError):
    """The peer rejected our credentials (HTTP 401 — fix the token)."""

    def __init__(self, message: str = "", *, status_code: int = 401,
                 detail: str = "") -> None:
        super().__init__(message or f"hub auth rejected ({status_code})"
                         + (f": {detail}" if detail else ""))
        self.status_code = status_code
        self.detail = detail


class SyncHubError(SyncError):
    """The hub answered with an error status (hub-side failure).

    ``retryable`` marks statuses worth retrying (5xx) versus permanent
    ones (4xx) — the engine's backoff consults it.
    """

    def __init__(self, message: str = "", *, status_code: int = 0,
                 detail: str = "", retryable: bool = False) -> None:
        super().__init__(message or f"hub error {status_code}"
                         + (f": {detail}" if detail else ""))
        self.status_code = status_code
        self.detail = detail
        self.retryable = retryable
