"""Fail-open telemetry for the power module.

One tiny helper so every power module emits the same way: best-effort,
never raises, never breaks scheduling when the bus or a subscriber is
broken (fail-open telemetry, fail-closed function).
"""

from __future__ import annotations

from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger

_log = get_logger(__name__)


def emit_event(topic: str, data: dict[str, Any], source: str = __name__) -> None:
    """Publish a telemetry event. Never raises."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=source))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("power event %s failed", topic, exc_info=True)
