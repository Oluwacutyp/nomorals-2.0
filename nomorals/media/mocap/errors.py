"""Errors for the cross-engine mocap/state organ.

Every failure names what was asked for and why it was refused — an
unconfigured engine, a channel owned by someone else, or data too
stale to trust are all honest errors, never silent fallbacks.
"""

from __future__ import annotations

__all__ = [
    "MocapError",
    "EngineNotConfigured",
    "ChannelOwnershipError",
    "StaleChannelError",
    "UnknownChannelError",
    "UnknownEngineError",
    "UnknownSubscriberError",
]


class MocapError(Exception):
    """Base error for the mocap organ."""

    def __init__(self, message: str, *, kind: str = "mocap_error") -> None:
        super().__init__(message)
        self.kind = kind


class EngineNotConfigured(MocapError):
    """An engine was asked for that is not configured or not present."""

    def __init__(self, engine: str, *, reason: str = "") -> None:
        detail = f"engine {engine!r} is not configured"
        if reason:
            detail += f": {reason}"
        super().__init__(detail, kind="engine_not_configured")
        self.engine = engine


class ChannelOwnershipError(MocapError):
    """A write/claim/release was refused because another engine owns
    the channel."""

    def __init__(self, channel: str, *, owner: str | None,
                 action: str = "write") -> None:
        if owner:
            detail = (f"cannot {action} channel {channel!r}: owned by "
                      f"engine {owner!r}")
        else:
            detail = (f"cannot {action} channel {channel!r}: channel "
                      "has no owner")
        super().__init__(detail, kind="channel_ownership")
        self.channel = channel
        self.owner = owner


class StaleChannelError(MocapError):
    """A read with ``require_fresh`` found data older than its limit."""

    def __init__(self, channel: str, *, age_s: float,
                 max_age_s: float) -> None:
        super().__init__(
            f"channel {channel!r} is stale: {age_s:.3f}s old, limit "
            f"{max_age_s:.3f}s — re-sync before trusting it",
            kind="stale_channel")
        self.channel = channel
        self.age_s = age_s
        self.max_age_s = max_age_s


class UnknownChannelError(MocapError):
    """A channel was named that the store has never seen."""

    def __init__(self, channel: str) -> None:
        super().__init__(f"unknown channel {channel!r}",
                         kind="unknown_channel")
        self.channel = channel


class UnknownEngineError(MocapError):
    """An engine was named that never registered."""

    def __init__(self, engine: str) -> None:
        super().__init__(f"unknown engine {engine!r}",
                         kind="unknown_engine")
        self.engine = engine


class UnknownSubscriberError(MocapError):
    """A subscriber id was named that never subscribed."""

    def __init__(self, subscriber_id: str) -> None:
        super().__init__(f"unknown subscriber {subscriber_id!r}",
                         kind="unknown_subscriber")
        self.subscriber_id = subscriber_id
