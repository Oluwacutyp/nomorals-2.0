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

:func:`local_sampler` builds a real on-machine sampler with zero new
mandatory dependencies: it prefers ``psutil`` (``sensors_battery`` /
``sensors_temperatures``), falls back to the Linux sysfs power-supply ABI
(``/sys/class/power_supply/*/capacity|status`` — type-file driven, not
hardcoded ``BAT0``), and reports "unknown" when neither exists.

Hysteresis: deferral flags are damped — a transition needs
``stability_reads`` consecutive agreeing readings before the monitor
flips, so one noisy sample can't flap the deferral plan.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..core.logging_setup import get_logger
from .telemetry import emit_event

__all__ = [
    "PowerStatus",
    "PowerMonitor",
    "ResourceSampler",
    "local_sampler",
    "SYSFS_POWER_SUPPLY",
    "THERMAL_BANDS",
    "DEGRADATION_TIERS",
]

_log = get_logger(__name__)


class ResourceSampler(Protocol):
    """The ResourceManager interface the monitor needs."""

    def sample(self) -> Any: ...
    def consult(self, mission: Any = None) -> dict[str, Any]: ...


SYSFS_POWER_SUPPLY = "/sys/class/power_supply"

# (name, enter_c, exit_c): hysteresis bands for the max core/package temp.
THERMAL_BANDS = (
    ("nominal", None, None),
    ("warm", 70.0, 65.0),
    ("hot", 85.0, 80.0),
)

# Degradation levels derived by the local sampler (mirror the ladder
# shape: 0=full .. 3=critical).  Level 4 (essential) is left to a full
# ResourceManager, which owns richer signals.
DEGRADATION_TIERS: dict[int, tuple[str, list[str]]] = {
    0: ("full", ["critical", "important", "background", "bulk"]),
    1: ("reduced", ["critical", "important", "background", "bulk"]),
    2: ("low", ["critical", "important"]),
    3: ("critical", ["critical"]),
}

LOW_BATTERY_PCT = 30.0
CRITICAL_BATTERY_PCT = 15.0


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

    def format(self) -> str:
        """Compact human-readable status card for chat/CLI surfaces."""
        bar = _battery_bar(self.battery_pct, self.charging)
        thermal = self.thermal_state or "unknown"
        flow = " ".join(
            f"{t}✓" if self.tier_allowed(t) else f"{t}✗"
            for t in ("critical", "important", "background", "bulk"))
        lines = [
            f"⚡ power  {bar}",
            f"   thermal: {thermal} · degradation: {self.degradation_name} "
            f"(level {self.degradation_level})",
            f"   tiers: {flow}",
        ]
        if self.drain_rate_pct_per_h is not None:
            lines.append(
                f"   drain: {self.drain_rate_pct_per_h:.1f}%/h" +
                (f" · ~{self.time_to_empty_min:.0f} min to empty"
                 if self.time_to_empty_min else ""))
        verdict = ("HEAVY+MEDIUM DEFERRED" if self.should_defer_heavy and
                   self.should_defer_medium
                   else "HEAVY DEFERRED" if self.should_defer_heavy
                   else "all classes flowing")
        lines.append(f"   → {verdict}")
        if self.reasons:
            lines.append("   reasons: " + "; ".join(self.reasons[:3]))
        return "\n".join(lines)


def _battery_bar(pct: float | None, charging: bool | None) -> str:
    if pct is None:
        return "n/a (no battery)"
    filled = int(round(max(0.0, min(100.0, pct)) / 10.0))
    bolt = "⚡" if charging else ""
    return f"{pct:.0f}% [{'▓' * filled}{'░' * (10 - filled)}]{bolt}"


# ---------------------------------------------------------------------------
# Local on-machine sampler (psutil → sysfs → unknown), zero new deps
# ---------------------------------------------------------------------------

@dataclass
class _LocalSample:
    battery_percent: float | None = None
    battery_status: str = ""          # Charging|Discharging|Full|...
    power_plugged: bool | None = None
    secs_left: int | None = None
    thermal_state: str | None = None
    thermal_max_c: float | None = None
    environment: dict[str, Any] | None = None


def _read_sysfs_battery(root: str) -> dict[str, Any]:
    """Read batteries + mains from the sysfs power-supply ABI.

    Enumerates ``root/*`` by the ``type`` file (``Battery``/``Mains``) —
    never hardcodes ``BAT0`` — so multi-battery laptops and oddly named
    AC adapters (``ACAD``, ...) work.
    """
    batteries: list[dict[str, Any]] = []
    mains_online: bool | None = None
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return {"batteries": [], "mains_online": None}
    for name in names:
        base = os.path.join(root, name)

        def _read(fname: str) -> str | None:
            try:
                with open(os.path.join(base, fname)) as fh:
                    return fh.read().strip()
            except OSError:
                return None

        kind = _read("type") or ""
        if kind == "Battery":
            cap = _read("capacity")
            status = _read("status") or ""
            try:
                pct = max(0.0, min(100.0, float(cap))) if cap else None
            except ValueError:
                pct = None
            batteries.append({"percent": pct, "status": status})
        elif kind == "Mains":
            online = _read("online")
            if online in ("0", "1"):
                mains_online = online == "1"
    return {"batteries": batteries, "mains_online": mains_online}


def _read_sysfs_thermal() -> float | None:
    """Max thermal-zone temperature in °C (millidegree sysfs values)."""
    root = "/sys/class/thermal"
    best: float | None = None
    try:
        names = os.listdir(root)
    except OSError:
        return None
    for name in names:
        if not name.startswith("thermal_zone"):
            continue
        try:
            with open(os.path.join(root, name, "temp")) as fh:
                milli = float(fh.read().strip())
            c = milli / 1000.0
            best = c if best is None else max(best, c)
        except (OSError, ValueError):
            continue
    return best


def _classify_thermal(max_c: float | None,
                      high: float | None,
                      critical: float | None) -> str | None:
    if max_c is None:
        return None
    hi = high if high else 70.0
    crit = critical if critical else 85.0
    if max_c >= crit:
        return "hot"
    if max_c >= hi:
        return "warm"
    return "nominal"


class _LocalSampler:
    """Real on-machine sampler. Never raises; degrades to unknown."""

    def __init__(self, psutil_mod: Any = None,
                 sysfs_root: str = SYSFS_POWER_SUPPLY) -> None:
        self._psutil = psutil_mod
        self._sysfs_root = sysfs_root
        self._last: _LocalSample = _LocalSample()

    # -- collection ------------------------------------------------------
    def sample(self) -> _LocalSample:
        s = _LocalSample()
        try:
            self._collect_battery(s)
        except Exception:  # noqa: BLE001 - sampling never fails hard
            _log.debug("local battery sampling failed", exc_info=True)
        try:
            self._collect_thermal(s)
        except Exception:  # noqa: BLE001 - sampling never fails hard
            _log.debug("local thermal sampling failed", exc_info=True)
        self._last = s
        return s

    def _collect_battery(self, s: _LocalSample) -> None:
        ps = self._psutil
        if ps is not None and hasattr(ps, "sensors_battery"):
            try:
                b = ps.sensors_battery()
            except Exception:  # noqa: BLE001
                b = None
            if b is not None:
                s.battery_percent = float(b.percent)
                s.power_plugged = bool(b.power_plugged)
                s.secs_left = (int(b.secsleft)
                               if b.secsleft not in (None, -1, -2) else None)
                s.battery_status = ("Charging" if b.power_plugged
                                    else "Discharging")
                return
        # sysfs fallback
        data = _read_sysfs_battery(self._sysfs_root)
        batts = data["batteries"]
        if batts:
            # Worst case wins: the emptiest battery gates the decision.
            worst = min((b for b in batts if b["percent"] is not None),
                        key=lambda b: b["percent"], default=None)
            if worst is not None:
                s.battery_percent = worst["percent"]
                s.battery_status = worst["status"]
            plugged = data["mains_online"]
            if plugged is None:
                plugged = any(b["status"] == "Charging" for b in batts)
            s.power_plugged = plugged

    def _collect_thermal(self, s: _LocalSample) -> None:
        ps = self._psutil
        temps: dict[str, list[Any]] = {}
        if ps is not None and hasattr(ps, "sensors_temperatures"):
            try:
                temps = ps.sensors_temperatures() or {}
            except Exception:  # noqa: BLE001
                temps = {}
        best: float | None = None
        hi: float | None = None
        crit: float | None = None
        for entries in temps.values():
            for e in entries:
                try:
                    c = float(e.current)
                except (TypeError, ValueError, AttributeError):
                    continue
                best = c if best is None else max(best, c)
                try:
                    if e.high:
                        hi = e.high if hi is None else min(hi, float(e.high))
                    if e.critical:
                        crit = (e.critical if crit is None
                                else min(crit, float(e.critical)))
                except (TypeError, ValueError, AttributeError):
                    pass
        if best is None:
            best = _read_sysfs_thermal()
        s.thermal_max_c = best
        s.thermal_state = _classify_thermal(best, hi, crit)

    # -- advisory ---------------------------------------------------------
    def consult(self, mission: Any = None) -> dict[str, Any]:
        s = self._last
        pct = s.battery_percent
        charging = (s.power_plugged or s.battery_status == "Charging"
                    or s.battery_status == "Full")
        reasons: list[str] = []
        throttled = False
        ok = True

        if pct is not None and not charging:
            if pct < CRITICAL_BATTERY_PCT:
                ok = False
                throttled = True
                reasons.append(f"battery critical ({pct:.0f}%)")
            elif pct < LOW_BATTERY_PCT:
                throttled = True
                reasons.append(f"battery low ({pct:.0f}%)")
        if s.thermal_state == "hot":
            throttled = True
            reasons.append("thermal hot")
            if pct is not None and pct < LOW_BATTERY_PCT and not charging:
                ok = False
                reasons.append("hot + low battery")
        elif s.thermal_state == "warm":
            reasons.append("thermal warm (watching)")

        level = 0
        if not ok:
            level = 3
        elif throttled:
            level = 2 if pct is not None and pct < LOW_BATTERY_PCT else 1
        if s.thermal_state == "hot" and level < 2:
            level = 2
        name, tiers = DEGRADATION_TIERS[level]

        forecast: dict[str, Any] = {"charging": bool(charging)}
        if pct is not None and not charging and s.secs_left:
            forecast["time_to_empty_min"] = s.secs_left / 60.0
        return {
            "throttled": throttled,
            "ok": ok,
            "reasons": reasons,
            "degradation_level": level,
            "degradation": {"level": level, "name": name,
                            "allowed_tiers": list(tiers)},
            "battery_forecast": forecast,
        }


def local_sampler(psutil_mod: Any = None,
                  sysfs_root: str = SYSFS_POWER_SUPPLY
                  ) -> Callable[[], _LocalSampler]:
    """Build a real on-machine sampler factory (dependency-optional).

    Pass the ``psutil`` module explicitly when available; otherwise the
    sampler tries a lazy import itself and falls back to sysfs.
    """
    def _factory() -> _LocalSampler:
        mod = psutil_mod
        if mod is None:
            try:
                import psutil as _ps  # type: ignore[import]
                mod = _ps
            except ImportError:
                mod = None
        return _LocalSampler(psutil_mod=mod, sysfs_root=sysfs_root)
    return _factory


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class PowerMonitor:
    """Reads power state via an injected sampler. Never raises.

    Damping/hysteresis for the *degradation ladder* lives in the sampler
    (slow-start recovery) — this monitor reports the ladder state verbatim.
    The *advisory flags* (throttled/ok → should_defer_*) get an extra
    debounce here: ``stability_reads`` consecutive agreeing readings are
    required before a flag flips, so a single noisy sample can't flap the
    deferral plan.  ``stability_reads=1`` preserves the old behavior.
    """

    def __init__(
        self,
        sampler: Callable[[], ResourceSampler] | None = None,
        *,
        stability_reads: int = 1,
        history_size: int = 120,
    ) -> None:
        # sampler is a factory (fresh manager per read avoids shared state).
        self._sampler = sampler
        self.stability_reads = max(1, int(stability_reads))
        self._history: deque[tuple[float, float, bool]] = deque(
            maxlen=max(1, int(history_size)))
        # Hysteresis state: damped flags + consecutive-disagreement counters.
        self._damped_throttled: bool | None = None
        self._damped_ok: bool | None = None
        self._pending_throttled = 0
        self._pending_ok = 0
        self._last_level: int | None = None
        self._last_charging: bool | None = None

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
        st = self._from_advisory(sample, advisory)
        st = self._apply_hysteresis(st)
        self._record_sample(st)
        self._maybe_emit_transitions(st)
        return st

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

    # -- hysteresis ------------------------------------------------------
    def _damp(self, raw: bool, damped: bool | None,
              pending: int) -> tuple[bool, bool | None, int]:
        """Debounce one flag. Returns (effective, new_damped, new_pending)."""
        if damped is None:
            return raw, raw, 0
        if raw == damped:
            return damped, damped, 0
        pending += 1
        if pending >= self.stability_reads:
            return raw, raw, 0
        return damped, damped, pending

    def _apply_hysteresis(self, st: PowerStatus) -> PowerStatus:
        eff_t, self._damped_throttled, self._pending_throttled = self._damp(
            st.throttled, self._damped_throttled, self._pending_throttled)
        eff_o, self._damped_ok, self._pending_ok = self._damp(
            st.ok, self._damped_ok, self._pending_ok)
        if eff_t != st.throttled or eff_o != st.ok:
            st.reasons.append(
                f"awaiting {self.stability_reads} stable reads before "
                "flipping power advisory")
        st.throttled = eff_t
        st.ok = eff_o
        return st

    # -- history / local drain forecast -----------------------------------
    def _record_sample(self, st: PowerStatus) -> None:
        if st.battery_pct is None:
            return
        try:
            self._history.append(
                (time.time(), float(st.battery_pct), bool(st.charging)))
        except (TypeError, ValueError):
            pass
        # Local forecast when the sampler didn't provide one.
        if st.drain_rate_pct_per_h is None:
            rate = self._history_drain_rate()
            if rate is not None:
                st.drain_rate_pct_per_h = rate
                if not st.charging and rate > 0 and st.battery_pct:
                    st.time_to_empty_min = (
                        st.battery_pct / rate * 60.0)

    def _history_drain_rate(self) -> float | None:
        """Linear drain %/h from history (discharging samples only).

        Needs ≥2 samples spanning ≥120s; ignores charging stretches.
        Positive = draining.
        """
        pts = [(t, p) for t, p, ch in self._history if not ch]
        if len(pts) < 2:
            return None
        (t0, p0), (t1, p1) = pts[0], pts[-1]
        span = t1 - t0
        if span < 120:
            return None
        drop = p0 - p1
        if drop <= 0:
            return None  # not actually draining
        return drop / (span / 3600.0)

    def drain_rate_pct_per_h(self) -> float | None:
        return self.status().drain_rate_pct_per_h

    def projected_drain_pct(self, minutes: float) -> float | None:
        """Projected battery % spent over ``minutes`` at current drain."""
        rate = self.drain_rate_pct_per_h()
        if rate is None or rate <= 0:
            return None
        return rate * (max(0.0, minutes) / 60.0)

    def can_sustain(self, minutes: float, pct_budget: float) -> bool:
        """True when a ``minutes``-long run fits inside ``pct_budget`` %.

        Unknown drain is optimistic (fail-open): without a rate we can't
        prove the run *won't* fit, and blocking work on missing telemetry
        would be the worse failure.
        """
        projected = self.projected_drain_pct(minutes)
        if projected is None:
            return True
        return projected <= max(0.0, pct_budget)

    def _maybe_emit_transitions(self, st: PowerStatus) -> None:
        if self._last_level is None:
            self._last_level = st.degradation_level
            self._last_charging = st.charging
            return
        if st.degradation_level != self._last_level:
            emit_event("power.degradation.changed", {
                "from_level": self._last_level,
                "to_level": st.degradation_level,
                "name": st.degradation_name,
                "battery_pct": st.battery_pct,
                "thermal_state": st.thermal_state,
            }, source=__name__)
            self._last_level = st.degradation_level
        if st.charging != self._last_charging and st.charging is not None:
            emit_event("power.charging.changed", {
                "charging": st.charging,
                "battery_pct": st.battery_pct,
            }, source=__name__)
            self._last_charging = st.charging

    def battery_pct(self) -> float | None:
        return self.status().battery_pct

    def thermal_state(self) -> str | None:
        return self.status().thermal_state

    def degradation_level(self) -> int:
        return self.status().degradation_level
