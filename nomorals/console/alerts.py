"""One alert policy, many channels.

A single :class:`AlertPolicy` decides, per severity, which channels an
alert goes to and how often it may repeat. The actual delivery is the
caller's job: ``send_alert`` takes a ``channels_sender`` callable
``fn(channel, severity, message) -> bool | None`` so this module never
imports Telegram / console / email internals — one policy, any set of
channels.

Throttling is per (severity, message): an identical alert inside the
throttle window is deduplicated (counted as suppressed, not sent), so
a flapping subsystem pages once per window instead of once per tick.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

Sender = Callable[[str, str, str], Any]

#: Default policy: severity -> (channels, throttle seconds).
DEFAULT_POLICY: dict[str, tuple[list[str], float]] = {
    "critical": (["telegram", "console"], 300.0),
    "error": (["telegram", "console"], 300.0),
    "warning": (["console"], 900.0),
    "info": (["console"], 3600.0),
}
_FALLBACK_CHANNELS = ["console"]
_FALLBACK_THROTTLE_S = 600.0


@dataclass(frozen=True)
class SeverityRule:
    """One severity's routing: which channels, how often at most."""
    channels: tuple[str, ...]
    throttle_s: float


@dataclass
class AlertRecord:
    severity: str
    message: str
    channels: tuple[str, ...]
    ts: float
    delivered: dict[str, bool] = field(default_factory=dict)


class AlertPolicy:
    """Severity -> channels + throttle, with dedup inside the window."""

    def __init__(self, rules: dict[str, SeverityRule | tuple[list[str], float] | dict[str, Any]] | None = None,
                 *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._rules: dict[str, SeverityRule] = {}
        source = rules if rules is not None else DEFAULT_POLICY
        for severity, rule in source.items():
            self._rules[str(severity).lower()] = self._coerce_rule(rule)
        self._last_sent: dict[tuple[str, str], float] = {}
        self._history: list[AlertRecord] = []
        self.sent_count = 0
        self.suppressed_count = 0
        self._lock = threading.RLock()

    @staticmethod
    def _coerce_rule(rule: SeverityRule | tuple[list[str], float] | dict[str, Any]) -> SeverityRule:
        if isinstance(rule, SeverityRule):
            return rule
        if isinstance(rule, dict):
            return SeverityRule(
                channels=tuple(rule.get("channels") or _FALLBACK_CHANNELS),
                throttle_s=float(rule.get("throttle_s", _FALLBACK_THROTTLE_S)))
        channels, throttle = rule
        return SeverityRule(channels=tuple(channels), throttle_s=float(throttle))

    def rule_for(self, severity: str) -> SeverityRule:
        return self._rules.get(str(severity).lower(), SeverityRule(
            channels=tuple(_FALLBACK_CHANNELS), throttle_s=_FALLBACK_THROTTLE_S))

    def send_alert(self, severity: str, msg: str,
                   channels_sender: Sender | None = None) -> bool:
        """Send an alert through its severity's channels.

        Returns True if the alert was sent (at least one channel
        attempted), False if it was deduplicated inside the throttle
        window. ``channels_sender(channel, severity, msg)`` is called
        once per channel; a sender returning False marks that channel
        undelivered but does not retry — delivery retries belong to the
        channel layer, not the policy.
        """
        sev = str(severity).lower()
        msg = str(msg)
        rule = self.rule_for(sev)
        now = self._clock()
        key = (sev, msg)
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and (now - last) < rule.throttle_s:
                self.suppressed_count += 1
                return False
            self._last_sent[key] = now
        delivered: dict[str, bool] = {}
        if channels_sender is not None:
            for channel in rule.channels:
                try:
                    delivered[channel] = bool(channels_sender(channel, sev, msg))
                except Exception:
                    delivered[channel] = False
        with self._lock:
            self._history.append(AlertRecord(
                severity=sev, message=msg, channels=rule.channels,
                ts=now, delivered=delivered))
            self.sent_count += 1
        return True

    def history(self) -> list[AlertRecord]:
        with self._lock:
            return list(self._history)


__all__ = ["AlertPolicy", "AlertRecord", "DEFAULT_POLICY", "SeverityRule"]
