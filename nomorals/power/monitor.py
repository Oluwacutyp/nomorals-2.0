"""Power monitor: battery/thermal/degradation state for scheduling decisions.

Layer L5. This module does NOT import ``nomorals.os`` (L6) — the sampler
is injected by the caller (L7 entry points wire in
``nomorals.os.resources.ResourceManager``). Integration via dependency
injection, never upward imports.

The sampler is a zero-arg callable returning an object with:
  ``battery_percent`` (float|None), ``thermal_state`` (str|None),
  and a ``consult()`` method returning ``{throttled, ok, reasons, ...}`` —
  i.e., the :class:`nomorals.os.resources.ResourceManager` interface.
  When the sampler is a full ResourceManager, the monitor also surfaces
  its degradation ladder (``degradation_level``, ``allowed_tiers``),
  battery drain forecast (``charging``, ``drain_rate_pct_per_h``,
  ``time_to_empty_min``), and environment info.  When no sampler is
  given, the monitor reports "unknown" and fails open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # -- full-signal additions ------------------------------------------------
    charging: bool | None = None          # True when the battery is charging
    drain_rate_pct_per_h: float | None = None
    time_to_empty_min: float | None = None
    degradation_level: int = 0            # 0=full .. 4=essential
    degradation_name: str = "full"
    allowed_tiers: list[str] = field(default_factory=lambda: [
        "critical", "important", "background", "bulk"])
    environment: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_pct": self.battery_pct,
            "thermal_state": self.thermal_state,
            "throttled": self.throttled,
            "ok": self.ok,
            "reasons": list(self.reasons),
            "charging": self.charging,
            "drain_rate_pct_per_h": self.drain_rate_pct_per_h,
            "time_to_empty_min": self.time_to_empty_min,
            "degradation_level": self.degradation_level,
            "degradation_name": self.degradation_name,
            "allowed_tiers": list(self.allowed_tiers),
            "environment": self.environment,
        }

    @property
    def should_defer_heavy(self) -> bool:
        """True when heavy work should wait for better conditions."""
        return self.throttled or not self.ok

    @property
    def should_defer_medium(self) -> bool:
        """True when medium work should also wait (degradation level >= 2)."""
        return self.degradation_level >= 2 or not self.ok

    def tier_allowed(self, tier: str) -> bool:
        """True when a task of ``tier`` may run at the current level."""
        return str(tier).lower() in self.allowed_tiers


class PowerMonitor:
    """Reads power state via an injected sampler. Never raises.

    Damping/hysteresis lives in the sampler's degradation ladder
    (slow-start recovery), so this monitor reports the ladder state
    verbatim — one damped machine, not two.
    """

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
        return self._from_advisory(sample, advisory)

    @staticmethod
    def _from_advisory(sample: Any, advisory: dict[str, Any]) -> PowerStatus:
        batt_fc = advisory.get("battery_forecast") or {}
        degr = advisory.get("degradation") or {}
        charging = batt_fc.get("charging")
        if charging is None:
            status = str(getattr(sample, "battery_status", "") or "")
            if status:
                low = status.lower()
                charging = "charg" in low and "dis" not in low
        allowed = degr.get("allowed_tiers") or [
            "critical", "important", "background", "bulk"]
        return PowerStatus(
            battery_pct=getattr(sample, "battery_percent", None),
            thermal_state=getattr(sample, "thermal_state", None),
            throttled=bool(advisory.get("throttled")),
            ok=bool(advisory.get("ok", True)),
            reasons=list(advisory.get("reasons") or []),
            charging=charging,
            drain_rate_pct_per_h=batt_fc.get("drain_pct_per_h"),
            time_to_empty_min=batt_fc.get("time_to_empty_min"),
            degradation_level=int(advisory.get("degradation_level") or 0),
            degradation_name=str(degr.get("name") or "full"),
            allowed_tiers=list(allowed),
            environment=getattr(sample, "environment", None),
        )

    def battery_pct(self) -> float | None:
        return self.status().battery_pct

    def thermal_state(self) -> str | None:
        return self.status().thermal_state

    def degradation_level(self) -> int:
        return self.status().degradation_level
