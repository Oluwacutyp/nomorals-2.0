"""Resource manager: advisory, stdlib-only sampling of machine resources.

Layer L6 of the ``nomorals.os`` control plane.  This module answers one
question: *is this machine under enough pressure that a mission should
throttle itself?*  It never enforces anything — :meth:`ResourceManager.consult`
returns an advisory dict, and it **never raises and never hard-blocks**.

Sampling is pure stdlib (no psutil dependency):

* CPU — :func:`os.getloadavg` (1-minute load) normalized by CPU count.
* Memory — ``/proc/meminfo`` parse (``MemTotal`` / ``MemAvailable``).
* Disk — :func:`shutil.disk_usage` on the user's home directory.
* Battery — ``/sys/class/power_supply`` capacity reads (all ``Battery``-type
  supplies are tried; includes the Termux/Android paths when the kernel
  exposes them).  ``None`` when no readable battery exists.
* Thermal — ``/sys/class/thermal/thermal_zone*/temp`` (millidegree Celsius)
  plus the ``/sys/devices/virtual/thermal`` aliases used on some Android
  kernels.  ``None`` when nothing readable exists.
* Network — ``/sys/class/net/*/operstate`` for a non-loopback interface in
  state ``up``; falls back to a short-timeout socket probe on platforms
  without sysfs.

Every individual sampler is defensive: a failed sampler yields ``None``
(or ``False``/``unknown``), never an exception.

Profile-aware thresholds: on battery-powered platforms (``termux``,
``android``, ``mobile`` per :func:`nomorals.core.platform.detect_platform`)
the advisory is more conservative — battery and thermal margins are
widened and the throttled thresholds are lowered.

Thresholds are configurable through constructor kwargs and through
environment overrides.  Environment variables, highest precedence first
for each value (env wins over constructor kwarg which wins over the
default):

* ``NM_RESOURCE_CPU_THROTTLE``    — cpu pressure (0..1) that throttles
* ``NM_RESOURCE_CPU_CRITICAL``    — cpu pressure (0..1) that fails advisory
* ``NM_RESOURCE_MEM_THROTTLE``    — memory pressure (0..1) throttling
* ``NM_RESOURCE_MEM_CRITICAL``    — memory pressure (0..1) critical
* ``NM_RESOURCE_DISK_THROTTLE``   — disk pressure (0..1) throttling
* ``NM_RESOURCE_DISK_CRITICAL``   — disk pressure (0..1) critical
* ``NM_RESOURCE_BATTERY_MIN``     — battery percent; below → not ok
* ``NM_RESOURCE_BATTERY_THROTTLE``— battery percent; below → throttled
* ``NM_RESOURCE_THERMAL_WARM_C``  — Celsius at/above which thermal is "warm"
* ``NM_RESOURCE_THERMAL_HOT_C``   — Celsius at/above which thermal is "hot"
* ``NM_RESOURCE_MAX_PRESSURE``    — overall pressure (0..1) cap for ok=True
* ``NM_RESOURCE_METERED``         — "1"/"true" → metered, "0"/"false" → not

The ``MissionRunner._resource_advisor`` hook (owned by another wave — this
module only needs a callable ``fn(mission) -> dict``) is fed by
:func:`advisor_callable`.
"""

from __future__ import annotations

import os
import shutil
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = [
    "ResourceSample",
    "ResourceManager",
    "default_manager",
    "reset_default_manager",
    "advisor_callable",
]

_THERMAL_NOMINAL = "nominal"
_THERMAL_WARM = "warm"
_THERMAL_HOT = "hot"


# ---------------------------------------------------------------------------
# defensive helpers
# ---------------------------------------------------------------------------

def _read_float(path: str) -> Optional[float]:
    """Read a number from a sysfs-ish file.  None on any failure."""
    try:
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            return float(fh.read().strip())
    except (OSError, ValueError, TypeError):
        return None


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            return fh.read().strip()
    except (OSError, TypeError):
        return None


def _battery_capacity_paths() -> list[str]:
    """Candidate sysfs battery capacity paths, Termux/Android included."""
    root = "/sys/class/power_supply"
    paths: list[str] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        entries = []
    for entry in entries:
        base = os.path.join(root, entry)
        typ = _read_text(os.path.join(base, "type")) or ""
        if "battery" in typ.lower():
            paths.append(os.path.join(base, "capacity"))
    # common Android/Termux kernel paths, even when `type` is unreadable
    paths.extend([
        "/sys/class/power_supply/battery/capacity",
        "/sys/class/power_supply/Battery/capacity",
        "/sys/class/power_supply/bms/capacity",
    ])
    seen: set[str] = set()
    ordered: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _thermal_zone_temps() -> list[float]:
    """All readable thermal-zone temps in Celsius."""
    roots = ["/sys/class/thermal", "/sys/devices/virtual/thermal"]
    temps: list[float] = []
    for root in roots:
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith("thermal_zone"):
                continue
            raw = _read_float(os.path.join(root, entry, "temp"))
            if raw is None:
                continue
            # sysfs reports millidegree Celsius on virtually all kernels
            temps.append(raw / 1000.0 if raw > 1000.0 else raw)
    return temps


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceSample:
    """One snapshot of machine resources.  ``None`` = unreadable, never fatal."""

    cpu_percent: Optional[float]        # 1-min loadavg / ncpu * 100
    mem_percent: Optional[float]        # used / total * 100
    disk_percent: Optional[float]       # used / total * 100 (home volume)
    battery_percent: Optional[float]    # remaining charge %, None if no battery
    thermal_state: Optional[str]        # "nominal" | "warm" | "hot" | None
    network_online: bool
    metered: Optional[bool]             # True/False/None(unknown)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "mem_percent": self.mem_percent,
            "disk_percent": self.disk_percent,
            "battery_percent": self.battery_percent,
            "thermal_state": self.thermal_state,
            "network_online": self.network_online,
            "metered": self.metered,
            "ts": self.ts,
        }


# ---------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------

_METERED_TRUE = {"1", "true", "yes", "on"}
_METERED_FALSE = {"0", "false", "no", "off"}


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _env_metered(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    low = str(raw).strip().lower()
    if low in _METERED_TRUE:
        return True
    if low in _METERED_FALSE:
        return False
    return None


class ResourceManager:
    """Sample resources and give advisory consults.  Advisory only.

    Every public method is exception-safe: samplers are defensive and
    :meth:`consult` / :meth:`pressure` never raise.
    """

    def __init__(
        self,
        *,
        cpu_throttle: float = 0.70,
        cpu_critical: float = 0.92,
        mem_throttle: float = 0.80,
        mem_critical: float = 0.93,
        disk_throttle: float = 0.85,
        disk_critical: float = 0.95,
        battery_min: float = 15.0,
        battery_throttle: float = 30.0,
        thermal_warm_c: float = 55.0,
        thermal_hot_c: float = 70.0,
        max_pressure: float = 0.95,
        metered: Optional[bool] = None,
        disk_path: Optional[str] = None,
        platform: Any = None,
    ) -> None:
        # env overrides win over kwargs (see module docstring)
        self.cpu_throttle = _env_float("NM_RESOURCE_CPU_THROTTLE") or cpu_throttle
        self.cpu_critical = _env_float("NM_RESOURCE_CPU_CRITICAL") or cpu_critical
        self.mem_throttle = _env_float("NM_RESOURCE_MEM_THROTTLE") or mem_throttle
        self.mem_critical = _env_float("NM_RESOURCE_MEM_CRITICAL") or mem_critical
        self.disk_throttle = _env_float("NM_RESOURCE_DISK_THROTTLE") or disk_throttle
        self.disk_critical = _env_float("NM_RESOURCE_DISK_CRITICAL") or disk_critical
        self.battery_min = _env_float("NM_RESOURCE_BATTERY_MIN") or battery_min
        self.battery_throttle = (
            _env_float("NM_RESOURCE_BATTERY_THROTTLE") or battery_throttle
        )
        self.thermal_warm_c = _env_float("NM_RESOURCE_THERMAL_WARM_C") or thermal_warm_c
        self.thermal_hot_c = _env_float("NM_RESOURCE_THERMAL_HOT_C") or thermal_hot_c
        self.max_pressure = _env_float("NM_RESOURCE_MAX_PRESSURE") or max_pressure
        self.metered = _env_metered("NM_RESOURCE_METERED")
        if self.metered is None:
            self.metered = metered
        self.disk_path = disk_path or os.path.expanduser("~")
        self._platform = platform  # lazy detect_platform() when None

        # battery-powered platforms get a more conservative advisory
        if self._is_battery_powered():
            self.battery_throttle = max(self.battery_throttle, 40.0)
            self.cpu_throttle = min(self.cpu_throttle, 0.60)
            self.mem_throttle = min(self.mem_throttle, 0.70)

    # -- platform ----------------------------------------------------------
    def _platform_obj(self) -> Any:
        if self._platform is None:
            from ..core.platform import detect_platform

            self._platform = detect_platform()
        return self._platform

    def _is_battery_powered(self) -> bool:
        try:
            plat = self._platform_obj()
            name = str(getattr(plat, "name", "") or "").lower()
            caps = getattr(plat, "capabilities", None)
            wake = bool(getattr(caps, "wake_lock", False)) if caps else False
            return name in {"termux", "android", "mobile"} or wake
        except Exception:  # noqa: BLE001 - platform detection must not break us
            return False

    # -- samplers (each defensive; never raises) ---------------------------
    def _sample_cpu(self) -> Optional[float]:
        try:
            load1 = os.getloadavg()[0]
            ncpu = os.cpu_count() or 1
            pct = load1 / max(1, ncpu) * 100.0
            return max(0.0, min(100.0, pct))
        except (OSError, AttributeError, ValueError):
            return None

    def _sample_mem(self) -> Optional[float]:
        total: Optional[int] = None
        avail: Optional[int] = None
        try:
            with open("/proc/meminfo", "r", encoding="ascii",
                      errors="replace") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1])
                    if total and avail:
                        break
        except (OSError, ValueError, IndexError):
            return None
        if not total or avail is None:
            return None
        used = max(0, total - avail)
        return max(0.0, min(100.0, used / total * 100.0))

    def _sample_disk(self) -> Optional[float]:
        try:
            usage = shutil.disk_usage(self.disk_path)
            if not usage.total:
                return None
            used = usage.total - usage.free
            return max(0.0, min(100.0, used / usage.total * 100.0))
        except (OSError, ValueError):
            return None

    def _sample_battery(self) -> Optional[float]:
        try:
            for path in _battery_capacity_paths():
                pct = _read_float(path)
                if pct is not None:
                    return max(0.0, min(100.0, pct))
        except Exception:  # noqa: BLE001 - battery probe is best-effort
            pass
        return None

    def _sample_thermal(self) -> Optional[str]:
        try:
            temps = _thermal_zone_temps()
        except Exception:  # noqa: BLE001 - thermal probe is best-effort
            return None
        if not temps:
            return None
        hottest = max(temps)
        if hottest >= self.thermal_hot_c:
            return _THERMAL_HOT
        if hottest >= self.thermal_warm_c:
            return _THERMAL_WARM
        return _THERMAL_NOMINAL

    def _sample_network(self) -> bool:
        try:
            net_root = "/sys/class/net"
            ifaces = os.listdir(net_root)
            for iface in ifaces:
                if iface == "lo":
                    continue
                state = _read_text(os.path.join(net_root, iface, "operstate"))
                if state and state.lower() == "up":
                    return True
            # sysfs readable but nothing up — only count `up` as online
            return False
        except OSError:  # noqa: E103 - no sysfs; falls through to socket probe
            pass
        # no sysfs (macOS/Windows/odd kernels): short socket probe
        try:
            with socket.create_connection(("8.8.8.8", 53), timeout=1.5):
                return True
        except (OSError, socket.timeout):
            return False

    # -- public API --------------------------------------------------------
    def sample(self) -> ResourceSample:
        """Take a resource snapshot.  Never raises on Exception failures."""
        def safe(fn, fallback):
            try:
                return fn()
            except Exception:  # noqa: BLE001 - sampler failed -> fallback
                return fallback

        return ResourceSample(
            cpu_percent=safe(self._sample_cpu, None),
            mem_percent=safe(self._sample_mem, None),
            disk_percent=safe(self._sample_disk, None),
            battery_percent=safe(self._sample_battery, None),
            thermal_state=safe(self._sample_thermal, None),
            network_online=safe(self._sample_network, False),
            metered=self.metered,
        )

    @staticmethod
    def _pct_to_pressure(pct: Optional[float]) -> float:
        if pct is None:
            return 0.0  # unknown => no pressure; consult() notes the gap
        return max(0.0, min(1.0, pct / 100.0))

    @staticmethod
    def _thermal_to_pressure(state: Optional[str]) -> float:
        return {
            None: 0.0,
            _THERMAL_NOMINAL: 0.0,
            _THERMAL_WARM: 0.5,
            _THERMAL_HOT: 1.0,
        }.get(state, 0.0)

    def pressure(self, sample: Optional[ResourceSample] = None) -> dict[str, float]:
        """0..1 pressure per resource plus ``overall`` = max.  Never raises."""
        try:
            s = sample or self.sample()
            battery_pct = s.battery_percent
            battery_p = 0.0 if battery_pct is None else 1.0 - battery_pct / 100.0
            parts = {
                "cpu": self._pct_to_pressure(s.cpu_percent),
                "mem": self._pct_to_pressure(s.mem_percent),
                "disk": self._pct_to_pressure(s.disk_percent),
                "battery": max(0.0, min(1.0, battery_p)),
                "thermal": self._thermal_to_pressure(s.thermal_state),
            }
            parts["overall"] = max(parts.values()) if parts else 0.0
            return parts
        except Exception:  # noqa: BLE001 - pressure must never raise
            return {"cpu": 0.0, "mem": 0.0, "disk": 0.0, "battery": 0.0,
                    "thermal": 0.0, "overall": 0.0}

    def consult(self, mission: Optional[Any] = None) -> dict[str, Any]:
        """Advisory consult: ``{ok, throttled, reasons, pressure}``.

        Advisory only — the caller decides what to do.  Never raises,
        never hard-blocks.
        """
        try:
            s = self.sample()
            p = self.pressure(s)
            reasons: list[str] = []
            throttled = False
            ok = True

            def note(text: str) -> None:
                reasons.append(text)

            if s.cpu_percent is None:
                note("cpu unreadable")
            elif p["cpu"] >= self.cpu_critical:
                ok = False
                note(f"cpu pressure {p['cpu']:.2f} >= critical {self.cpu_critical:.2f}")
            elif p["cpu"] >= self.cpu_throttle:
                throttled = True
                note(f"cpu pressure {p['cpu']:.2f} >= throttle {self.cpu_throttle:.2f}")

            if s.mem_percent is None:
                note("memory unreadable")
            elif p["mem"] >= self.mem_critical:
                ok = False
                note(f"memory pressure {p['mem']:.2f} >= critical {self.mem_critical:.2f}")
            elif p["mem"] >= self.mem_throttle:
                throttled = True
                note(f"memory pressure {p['mem']:.2f} >= throttle {self.mem_throttle:.2f}")

            if s.disk_percent is None:
                note("disk unreadable")
            elif p["disk"] >= self.disk_critical:
                ok = False
                note(f"disk pressure {p['disk']:.2f} >= critical {self.disk_critical:.2f}")
            elif p["disk"] >= self.disk_throttle:
                throttled = True
                note(f"disk pressure {p['disk']:.2f} >= throttle {self.disk_throttle:.2f}")

            if s.battery_percent is None:
                note("battery unknown (no readable battery)")
            elif s.battery_percent <= self.battery_min:
                ok = False
                note(f"battery {s.battery_percent:.0f}% <= minimum {self.battery_min:.0f}%")
            elif s.battery_percent <= self.battery_throttle:
                throttled = True
                note(f"battery {s.battery_percent:.0f}% <= throttle level {self.battery_throttle:.0f}%")

            if s.thermal_state is None:
                note("thermal unknown")
            elif s.thermal_state == _THERMAL_HOT:
                ok = False
                note("thermal state hot")
            elif s.thermal_state == _THERMAL_WARM:
                throttled = True
                note("thermal state warm")

            if p["overall"] > self.max_pressure:
                ok = False
                note(f"overall pressure {p['overall']:.2f} > cap {self.max_pressure:.2f}")

            # mission may carry its own pressure cap (duck-typed)
            cap = self._mission_cap(mission)
            if cap is not None and p["overall"] > cap:
                throttled = True
                note(f"mission pressure cap {cap:.2f} exceeded ({p['overall']:.2f})")

            if not s.network_online:
                note("network offline (local-only work)")

            if s.metered if s.metered is not None else self.metered:
                throttled = True
                note("metered network: avoid heavy transfers")

            return {
                "ok": ok,
                "throttled": throttled,
                "reasons": reasons,
                "pressure": p,
                "sample": s.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001 - consult never raises
            return {
                "ok": False,
                "throttled": True,
                "reasons": [f"consult failed: {exc}"],
                "pressure": {},
                "sample": {},
            }

    @staticmethod
    def _mission_cap(mission: Optional[Any]) -> Optional[float]:
        try:
            if mission is None:
                return None
            if isinstance(mission, dict):
                cap = mission.get("max_pressure")
            else:
                cap = getattr(mission, "max_pressure", None)
            if cap is None:
                return None
            cap = float(cap)
            return max(0.0, min(1.0, cap))
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# module-level convenience
# ---------------------------------------------------------------------------

_cached_manager: Optional[ResourceManager] = None


def default_manager() -> ResourceManager:
    """Process-cached singleton (reads env thresholds at first creation)."""
    global _cached_manager
    if _cached_manager is None:
        _cached_manager = ResourceManager()
    return _cached_manager


def reset_default_manager() -> None:
    """Forget the cached singleton (tests)."""
    global _cached_manager
    _cached_manager = None


def advisor_callable(
    manager: Optional[ResourceManager] = None,
) -> Callable[[Optional[Any]], dict[str, Any]]:
    """Return ``fn(mission) -> dict`` suitable for ``MissionRunner._resource_advisor``.

    Duck-typed: we do not import or depend on ``nomorals.missions`` here.
    The returned callable itself never raises — an internal failure
    degrades to a throttled advisory.
    """
    mgr = manager if manager is not None else default_manager()

    def advise(mission: Optional[Any] = None) -> dict[str, Any]:
        try:
            return mgr.consult(mission)
        except Exception as exc:  # noqa: BLE001 - advisor never raises
            return {
                "ok": False,
                "throttled": True,
                "reasons": [f"resource advisor failed: {exc}"],
                "pressure": {},
                "sample": {},
            }

    return advise
