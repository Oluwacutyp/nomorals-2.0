"""Stream errors."""

from __future__ import annotations

__all__ = ["StreamError", "SubscriberLimitExceeded", "StreamClosed"]


class StreamError(Exception):
    """Base for all stream errors."""


class SubscriberLimitExceeded(StreamError):
    """The hub already has ``max_subscribers`` live subscriptions."""


class StreamClosed(StreamError):
    """The hub was stopped (or was never started)."""
