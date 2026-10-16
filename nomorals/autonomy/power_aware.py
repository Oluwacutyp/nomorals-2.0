"""Power-aware gate for organ ticks (Tier 0 wiring).

``nomorals.autonomy.coordinator`` and ``idle`` previously had zero
references to ``nomorals.power`` — background organs ticked regardless
of battery/thermal state. This module is the gate the coordinator calls
before every tick: :meth:`PowerAwareGate.should_tick` consults the
power monitor and :meth:`PowerAwareGate.gate_tick` enforces the
invariant — **no background organ ticks while power says heavy shed**.

Shed levels derive from the monitor's degradation ladder:
level 0 → ``none``, 1 → ``light``, 2 → ``medium``, ≥3 → ``heavy``.
Under heavy shed only ``critical``-tier organs may tick; everything
else (``important`` / ``background`` / ``bulk``) is blocked. Below
heavy shed, the monitor's own ``allowed_tiers`` decides per tier — the
gate never second-guesses the monitor, it only adds the heavy-shed
floor.

The monitor is injected (a :class:`nomorals.power.monitor.PowerMonitor`
or any duck-typed object with a ``status()`` method). With no monitor
the default is a sampler-less ``PowerMonitor`` which fails open
(``ok=True``, all tiers allowed) — exactly the old behaviour, so wiring
the gate in cannot newly block work on a machine with no telemetry.
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..power.monitor import PowerMonitor, PowerStatus
from ..tier0_rollback import record_promotion

__all__ = [
    "PowerAwareGate",
    "SHED_NONE",
    "SHED_LIGHT",
    "SHED_MEDIUM",
    "SHED_HEAVY",
    "TIERS",
    "check_power",
    "should_tick",
    "register_promotion",
    "promotion_active",
]

_log = get_logger(__name__)

PROMOTION_NAME = "power_aware_gate"

TIERS = ("critical", "important", "background", "bulk")

SHED_NONE = "none"
SHED_LIGHT = "light"
SHED_MEDIUM = "medium"
SHED_HEAVY = "heavy"

_promotion_active = False


def promotion_active() -> bool:
    """True while the power-aware gate promotion is active."""
    return _promotion_active


def register_promotion() -> None:
    """Record this wiring in the Tier 0 rollback registry."""
    global _promotion_active
    _promotion_active = True

    def _rollback() -> None:
        global _promotion_active
        _promotion_active = False

    record_promotion(
        PROMOTION_NAME, _rollback,
        description="autonomy organ ticks gated on power monitor (heavy shed)",
        metadata={"module": __name__},
    )


def shed_level_for(status: PowerStatus) -> str:
    """Map a PowerStatus degradation level to a shed level name."""
    level = int(getattr(status, "degradation_level", 0) or 0)
    if level >= 3:
        return SHED_HEAVY
    if level == 2:
        return SHED_MEDIUM
    if level == 1:
        return SHED_LIGHT
    return SHED_NONE


class PowerAwareGate:
    """Decide whether an organ may tick, from live power state."""

    def __init__(self, monitor: Any | None = None) -> None:
        # No sampler → PowerMonitor fails open (ok=True, all tiers).
        self.monitor = monitor if monitor is not None else PowerMonitor()

    # ── raw state ────────────────────────────────────────────────────────────
    def check_power(self) -> PowerStatus:
        """Live power status. Call this before a tick; never raises.

        A monitor failure degrades to a fail-open status (the monitor
        itself already fails open; this is the second net).
        """
        try:
            return self.monitor.status()
        except Exception as exc:  # noqa: BLE001 - gate never breaks ticks
            _log.warning("power gate: monitor failed (%s); failing open",
                         exc)
            return PowerStatus(
                battery_pct=None, thermal_state=None,
                throttled=False, ok=True,
                reasons=[f"monitor failed ({exc}); assuming ok"],
            )

    def shed_level(self, status: PowerStatus | None = None) -> str:
        """Current shed level: none | light | medium | heavy."""
        return shed_level_for(status or self.check_power())

    def heavy_shed(self, status: PowerStatus | None = None) -> bool:
        """True when power says heavy shed — background organs must stop.

        Heavy shed = degradation level ≥ 3, or the monitor reporting
        both heavy and medium work deferred (``not ok`` + throttled +
        medium deferral) even if the ladder number lags.
        """
        st = status or self.check_power()
        if int(getattr(st, "degradation_level", 0) or 0) >= 3:
            return True
        return (
            not bool(getattr(st, "ok", True))
            and bool(getattr(st, "throttled", False))
            and bool(getattr(st, "should_defer_medium", False))
        )

    # ── the gate ─────────────────────────────────────────────────────────────
    def should_tick(
        self,
        organ: str,
        tier: str = "background",
    ) -> tuple[bool, str, str]:
        """Return ``(allowed, reason, shed_level)`` for an organ tick.

        ``tier`` is the organ's power tier (``critical`` | ``important`` |
        ``background`` | ``bulk``). Unknown tiers raise — silently
        treating a typo'd tier as background would mis-gate work.
        """
        tier = str(tier).lower()
        if tier not in TIERS:
            raise ValueError(
                f"unknown tier {tier!r}; expected {TIERS}")
        status = self.check_power()
        shed = shed_level_for(status)
        if self.heavy_shed(status):
            if tier != "critical":
                return (
                    False,
                    f"heavy shed (degradation "
                    f"{status.degradation_level}, "
                    f"{'; '.join(status.reasons) or 'power constrained'}): "
                    f"{organ} ({tier}) blocked",
                    SHED_HEAVY,
                )
            return (
                True,
                f"heavy shed but {organ} is critical-tier: allowed",
                SHED_HEAVY,
            )
        if not status.tier_allowed(tier):
            return (
                False,
                f"tier {tier} shed at degradation "
                f"{status.degradation_level} "
                f"({status.degradation_name}): {organ} blocked",
                shed,
            )
        return (
            True,
            f"power flowing (shed={shed}, degradation "
            f"{status.degradation_level}): {organ} may tick",
            shed,
        )

    def gate_tick(
        self,
        organ: str,
        fn: Callable[..., Any],
        *args: Any,
        tier: str = "background",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run ``fn(*args, **kwargs)`` only if the gate allows the tick.

        Returns ``{"skipped": True, "organ", "reason", "shed_level"}`` when
        blocked (``fn`` is never called), or ``{"skipped": False,
        "result": ...}`` when it ran. ``fn``'s own exceptions propagate —
        the gate decides admission, it does not swallow organ errors.
        """
        allowed, reason, shed = self.should_tick(organ, tier=tier)
        if not allowed:
            _log.info("power gate: skipping %s tick: %s", organ, reason)
            return {"skipped": True, "organ": organ, "reason": reason,
                    "shed_level": shed}
        return {"skipped": False, "organ": organ, "reason": reason,
                "shed_level": shed, "result": fn(*args, **kwargs)}


def check_power(monitor: Any | None = None) -> PowerStatus:
    """Module-level helper: live power status for a pre-tick check."""
    return PowerAwareGate(monitor).check_power()


def should_tick(
    organ: str,
    tier: str = "background",
    monitor: Any | None = None,
) -> tuple[bool, str, str]:
    """Module-level helper: ``(allowed, reason, shed_level)`` for a tick."""
    return PowerAwareGate(monitor).should_tick(organ, tier=tier)


register_promotion()
