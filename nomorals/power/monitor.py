"""Power monitor: battery/thermal state for scheduling decisions.

Layer L5. This module does NOT import ``nomorals.os`` (L6) — the sampler
is injected by the caller (L7 entry points wire in
``nomorals.os.resources.ResourceManager``). Integration via dependency
injection, never upward imports.

The sampler is a zero-arg callable returning an object with:
  ``battery_percent`` (float|None), ``thermal_state`` (str|None),
  and a ``consult()`` method returning ``{throttled, ok, reasons}`` —
  i.e., the :class:`nomorals.os.resources.ResourceManager` interface.
  When no sampler is given, the monitor reports "unknown" and fails open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..core.logging_setup import get_logger

__all__ = ["PowerStatus", "PowerMonitor", "ResourceSampler"]

_log = get_logger(__name__)


class ResourceSampler(Protocol):
    """The ResourceManager interface the monitor needs."""

    def sample(self) -> Any: ...
    def consult(self, mission: Any = None) -> dict[str, Any]: ...


@dataclass
class PowerStatus:
    battery_pct: float | None      # None = no readable battery (e.g., desktop)
    thermal_state: str | None      # "nominal" | "warm" | "hot" | None
    throttled: bool                # advisory: ease off heavy work
    ok: bool                       # advisory: false under critical pressure
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_pct": self.battery_pct,
            "thermal_state": self.thermal_state,
            "throttled": self.throttled,
            "ok": self.ok,
            "reasons": list(self.reasons),
        }

    @property
    def should_defer_heavy(self) -> bool:
        """True when heavy work should wait for better conditions."""
        return self.throttled or not self.ok


class PowerMonitor:
    """Reads power state via an injected sampler. Never raises."""

    def __init__(self, sampler: Callable[[], ResourceSampler] | None = None) -> None:
        # sampler is a factory (fresh manager per read avoids shared state).
        self._sampler = sampler

    def status(self) -> PowerStatus:
        if self._sampler is None:
            return PowerStatus(
                battery_pct=None, thermal_state=None,
                throttled=False, ok=True,
                reasons=["no sampler configured; assuming ok"],
            )
        try:
            manager = self._sampler()
            sample = manager.sample()
            advisory = manager.consult()
        except Exception as exc:  # noqa: BLE001 - sampling never fails hard
            _log.warning("power sampling failed: %s", exc)
            return PowerStatus(
                battery_pct=None, thermal_state=None,
                throttled=False, ok=True,
                reasons=["sampling failed; assuming ok"],
            )
        return PowerStatus(
            battery_pct=getattr(sample, "battery_percent", None),
            thermal_state=getattr(sample, "thermal_state", None),
            throttled=bool(advisory.get("throttled")),
            ok=bool(advisory.get("ok", True)),
            reasons=list(advisory.get("reasons") or []),
        )

    def battery_pct(self) -> float | None:
        return self.status().battery_pct

    def thermal_state(self) -> str | None:
        return self.status().thermal_state
