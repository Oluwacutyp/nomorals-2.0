"""Resource manager: full-signal advisory sampling of machine resources.

Layer L6 of the ``nomorals.os`` control plane.  This module answers one
question: *is this machine under enough pressure that a mission should
throttle itself?*  It never enforces anything — :meth:`ResourceManager.consult`
returns an advisory dict, and it **never raises and never hard-blocks**.

Full signal surface (every platform's signals; the profile system decides
what to use where — this module never designs down to a deployment target):

* CPU — ``os.getloadavg`` (1-min load) normalized by CPU count **plus**
  delta-sampled true utilization and **steal time** from ``/proc/stat``
  (steal = the vCPU was ready but the hypervisor scheduled elsewhere; on
  burstable cloud types sustained steal ≈ credit depletion).
* Memory — ``/proc/meminfo`` (``MemTotal`` / ``MemAvailable``), swap
  presence/saturation, **cgroup-aware effective memory**
  (``/proc/self/cgroup`` walk over v2 ``memory.max`` / v1
  ``memory.limit_in_bytes`` — ``/proc/meminfo`` is not namespaced, so raw
  meminfo is a lie inside containers), and our own RSS from
  ``/proc/self/status``.
* Pressure Stall Information — ``/proc/pressure/{cpu,memory,io}``
  (kernel 4.20+): ``some``/``full`` stall percentages over avg10/60/300.
  Utilization lies; PSI measures *experienced* stalls.
* Disk — :func:`shutil.disk_usage` on the configured path.
* Battery — ``/sys/class/power_supply`` capacity **plus** ``status``
  (Charging/Discharging/Full — a plugged-in phone at 25% is fine, a
  discharging one is not), ``power_now``/``energy_now`` for drain-rate and
  time-to-empty forecasting.
* Thermal — ``/sys/class/thermal/thermal_zone*/temp`` plus Android aliases.
* Network — ``/sys/class/net/*/operstate`` with socket-probe fallback.
* Environment — cloud/container/phone detection from DMI, cgroup, and
  userspace markers (``detect_environment``).

Beyond snapshots, the manager keeps a rolling sample history and adds:

* **Trend + forecast** — least-squares slope per resource, predicted
  pressure at a horizon, minutes-to-critical.  Predictive, not just
  reactive.
* **Composite memory pressure** — 55% PSI-memory stall + 20% swap
  saturation + 15% memory utilization + 10% swap I/O rate (adapted from
  the lmkd-linux composite-score design).
* **Degradation ladder** — five damped levels (FULL → LIGHT_SHED →
  DEGRADED → SURVIVAL → ESSENTIAL) with task-tier shedding and
  slow-start recovery: step up after sustained pressure (instant on
  critical), step down only after sustained calm.
* **Subsystem budgets** — named admission-control budgets (``llm``,
  ``media``, ``scheduler``, ``missions``, ``bulk``): "does this load fit,
  including the reserve?" sized from *effective* (cgroup-aware) resources
  with an oma-style reserve margin (6% of effective RAM, configurable
  floor).

Profile-aware thresholds: on battery-powered platforms (``termux``,
``android``, ``mobile`` per :func:`nomorals.core.platform.detect_platform`)
the advisory is more conservative — battery and thermal margins are
widened and the throttled thresholds are lowered.  Charging state relaxes
battery gating (plugged in = power available).

Every individual sampler is defensive: a failed sampler yields ``None``
(or ``False``/``unknown``), never an exception.

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
* ``NM_RESOURCE_HISTORY``         — rolling sample history size (default 120)
* ``NM_RESOURCE_RESERVE_PCT``     — budget reserve fraction of effective RAM
* ``NM_RESOURCE_RESERVE_MIN_MB``  — budget reserve floor in MB
* ``NM_RESOURCE_CALM_SECONDS``    — sustained calm before ladder steps down
* ``NM_RESOURCE_UP_TICKS``        — sustained ticks before ladder steps up
* ``NM_RESOURCE_STEAL_WARN``      — steal fraction (0..1) that warns
* ``NM_RESOURCE_STEAL_CRITICAL``  — steal fraction (0..1) that is critical
* ``NM_RESOURCE_PSI_MEM_WARN``    — memory PSI some-avg10 % that warns
* ``NM_RESOURCE_PSI_MEM_FULL_CRIT``— memory PSI full-avg10 % that is critical

The ``MissionRunner._resource_advisor`` hook (owned by another wave — this
module only needs a callable ``fn(mission) -> dict``) is fed by
:func:`advisor_callable`.
"""

from __future__ import annotations

import math
import os
import shutil
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = [
    "ResourceSample",
    "ResourceManager",
    "ResourceBudgets",
    "SubsystemBudget",
    "DegradationLadder",
    "DegradationStep",
    "DEGRADATION_LEVELS",
    "EnvironmentInfo",
    "PressureStall",
    "default_manager",
    "reset_default_manager",
    "advisor_callable",
    "detect_environment",
    "reset_environment_cache",
    "sample_psi",
    "cgroup_memory_limit_bytes",
    "cgroup_cpu_quota_cores",
]

_THERMAL_NOMINAL = "nominal"
_THERMAL_WARM = "warm"
_THERMAL_HOT = "hot"

# task priority tiers, highest first.  The degradation ladder sheds from
# the bottom: level 1 sheds "bulk", level 2 sheds "background", ...
TASK_TIERS = ("critical", "important", "background", "bulk")

# sentinel: cgroup v1 reports LONG_MAX when no limit is set
_CGROUP_V1_UNLIMITED = (1 << 62)


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


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None


_METERED_TRUE = {"1", "true", "yes", "on"}
_METERED_FALSE = {"0", "false", "no", "off"}


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


# ---------------------------------------------------------------------------
# environment detection (stdlib only; best-effort; never raises)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvironmentInfo:
    """Where this process runs, as best as stdlib probing can tell.

    ``kind`` is one of: ``aws-ec2``, ``gcp``, ``azure``, ``other-cloud``,
    ``container``, ``termux``, ``android``, ``desktop``, ``unknown``.
    ``signals`` records which probes fired, so callers can judge
    confidence.
    """

    kind: str
    virtualized: bool
    containerized: bool
    signals: dict[str, str]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "virtualized": self.virtualized,
            "containerized": self.containerized,
            "signals": dict(self.signals),
            "detail": self.detail,
        }


_cached_environment: Optional[EnvironmentInfo] = None


def reset_environment_cache() -> None:
    """Forget the cached environment detection (tests)."""
    global _cached_environment
    _cached_environment = None


def _self_cgroup_text() -> str:
    return _read_text("/proc/self/cgroup") or ""


def _detect_environment() -> EnvironmentInfo:
    signals: dict[str, str] = {}
    containerized = False
    virtualized = False
    kind = "unknown"

    # -- container markers -------------------------------------------------
    cgroup = _self_cgroup_text().lower()
    if os.path.exists("/.dockerenv"):
        containerized = True
        signals["dockerenv"] = "/.dockerenv present"
    if os.path.exists("/run/.containerenv"):
        containerized = True
        signals["containerenv"] = "/run/.containerenv present"
    for marker in ("docker", "kubepods", "containerd", "lxc", "podman"):
        if marker in cgroup:
            containerized = True
            signals["cgroup"] = f"/proc/self/cgroup mentions {marker}"
            break
    # systemd-nspawn / cgroup v2 scope names
    if ".scope" in cgroup and ("docker" in cgroup or "podman" in cgroup):
        containerized = True

    # -- virtualization markers --------------------------------------------
    cpuinfo = _read_text("/proc/cpuinfo") or ""
    if "hypervisor" in cpuinfo.lower():
        virtualized = True
        signals["cpuinfo"] = "hypervisor flag present"
    if os.path.exists("/sys/hypervisor/type"):
        virtualized = True
        signals["hypervisor"] = _read_text("/sys/hypervisor/type") or "present"

    # -- cloud DMI markers --------------------------------------------------
    sys_vendor = (_read_text("/sys/devices/virtual/dmi/id/sys_vendor") or "")
    product = (_read_text("/sys/devices/virtual/dmi/id/product_name") or "")
    board = (_read_text("/sys/devices/virtual/dmi/id/board_vendor") or "")
    dmi = f"{sys_vendor} {product} {board}".lower()
    if "amazon ec2" in dmi or "amazon" in sys_vendor.lower() and "ec2" in dmi:
        kind = "aws-ec2"
        virtualized = True
        signals["dmi"] = f"sys_vendor={sys_vendor!r} product={product!r}"
        # Xen-based EC2 also exposes an ec2-prefixed hypervisor uuid
        uuid = (_read_text("/sys/hypervisor/uuid") or "").lower()
        if uuid.startswith("ec2"):
            signals["hypervisor_uuid"] = "ec2-prefixed"
    elif "google compute engine" in dmi or "google" in sys_vendor.lower():
        kind = "gcp"
        virtualized = True
        signals["dmi"] = f"sys_vendor={sys_vendor!r} product={product!r}"
    elif ("microsoft corporation" in dmi and "virtual machine" in dmi) or \
            "microsoft-hyperv" in cgroup:
        kind = "azure"
        virtualized = True
        signals["dmi"] = f"sys_vendor={sys_vendor!r} product={product!r}"
    elif virtualized and kind == "unknown":
        kind = "other-cloud"

    # -- phone markers -------------------------------------------------------
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix:
        kind = "termux"
        signals["prefix"] = "PREFIX mentions com.termux"
    else:
        try:
            import platform as _plat

            if _plat.system() == "Android":
                kind = "android"
                signals["platform"] = "platform.system() == Android"
        except Exception:  # noqa: BLE001 - platform probe is best-effort
            pass

    if kind == "unknown":
        kind = "container" if containerized else "desktop"

    detail_bits = [f"{k}={v}" for k, v in signals.items()]
    detail = "; ".join(detail_bits) if detail_bits else "no markers found"
    return EnvironmentInfo(
        kind=kind,
        virtualized=virtualized,
        containerized=containerized,
        signals=signals,
        detail=detail,
    )


def detect_environment(*, refresh: bool = False) -> EnvironmentInfo:
    """Detect the runtime environment (cached per process).  Never raises."""
    global _cached_environment
    if _cached_environment is None or refresh:
        try:
            _cached_environment = _detect_environment()
        except Exception:  # noqa: BLE001 - detection must not break callers
            _cached_environment = EnvironmentInfo(
                kind="unknown", virtualized=False, containerized=False,
                signals={}, detail="detection failed",
            )
    return _cached_environment


# ---------------------------------------------------------------------------
# PSI — pressure stall information
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PressureStall:
    """Stall percentages for one resource (cpu/memory/io).

    ``some`` = share of time at least one task stalled; ``full`` = share of
    time ALL runnable tasks stalled (zero productive cycles).  ``None`` =
    unreadable.  CPU has no ``full`` line by definition.
    """

    some_avg10: Optional[float] = None
    some_avg60: Optional[float] = None
    some_avg300: Optional[float] = None
    full_avg10: Optional[float] = None
    full_avg60: Optional[float] = None
    full_avg300: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "some_avg10": self.some_avg10,
            "some_avg60": self.some_avg60,
            "some_avg300": self.some_avg300,
            "full_avg10": self.full_avg10,
            "full_avg60": self.full_avg60,
            "full_avg300": self.full_avg300,
        }


def _parse_psi_line(line: str) -> dict[str, float]:
    vals: dict[str, float] = {}
    for token in line.split()[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            try:
                vals[k] = float(v)
            except (ValueError, TypeError):
                pass
    return vals


def sample_psi() -> Optional[dict[str, PressureStall]]:
    """Read ``/proc/pressure/{cpu,memory,io}``.  None when PSI unavailable."""
    try:
        out: dict[str, PressureStall] = {}
        for name in ("cpu", "memory", "io"):
            text = _read_text(f"/proc/pressure/{name}")
            if not text:
                continue
            some: dict[str, float] = {}
            full: dict[str, float] = {}
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("some"):
                    some = _parse_psi_line(line)
                elif line.startswith("full"):
                    full = _parse_psi_line(line)
            if not some:
                continue
            out[name] = PressureStall(
                some_avg10=some.get("avg10"),
                some_avg60=some.get("avg60"),
                some_avg300=some.get("avg300"),
                full_avg10=full.get("avg10"),
                full_avg60=full.get("avg60"),
                full_avg300=full.get("avg300"),
            )
        return out or None
    except Exception:  # noqa: BLE001 - PSI probe is best-effort
        return None


# ---------------------------------------------------------------------------
# cgroup-aware effective resources
# ---------------------------------------------------------------------------

def _cgroup_mount() -> Optional[str]:
    """Cgroup filesystem mount point (v2 unified or v1)."""
    try:
        with open("/proc/self/mountinfo", "r", encoding="ascii",
                  errors="replace") as fh:
            for line in fh:
                parts = line.split()
                # format: ... - fstype source target options
                if "- cgroup2 " in line:
                    for i, p in enumerate(parts):
                        if p == "-" and i + 2 < len(parts):
                            return parts[i + 2]
                if "- cgroup " in line and "memory" in line:
                    for i, p in enumerate(parts):
                        if p == "-" and i + 2 < len(parts):
                            return parts[i + 2]
    except OSError:
        pass
    for cand in ("/sys/fs/cgroup",):
        if os.path.isdir(cand):
            return cand
    return None


def _self_cgroup_relpath() -> Optional[str]:
    """Our cgroup path relative to the mount (v2 ``0::<path>`` or v1)."""
    text = _read_text("/proc/self/cgroup")
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("0:"):
            # v2 unified: 0::<path>
            path = line[2:]
            return path if path.startswith("/") else "/" + path
        # v1: N:controllers:<path> — prefer the memory controller entry
        parts = line.split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            path = parts[2]
            return path if path.startswith("/") else "/" + path
    return None


def _walk_cgroup_chain(mount: str, rel: str) -> list[str]:
    """Leaf-to-root cgroup dirs; an ancestor limit binds the leaf too."""
    chain: list[str] = []
    cur = rel.rstrip("/") or "/"
    while True:
        chain.append(os.path.join(mount, cur.lstrip("/")) if cur != "/" else mount)
        if cur == "/":
            break
        cur = os.path.dirname(cur) or "/"
    return chain


def cgroup_memory_limit_bytes() -> Optional[int]:
    """Effective cgroup memory limit in bytes.

    Walks our cgroup chain and takes the minimum over v2 ``memory.max``
    / v1 ``memory.limit_in_bytes``.  ``max`` / LONG_MAX sentinels mean
    "unlimited at this level" and are skipped.  None = no limit found.
    Never raises.
    """
    try:
        mount = _cgroup_mount()
        rel = _self_cgroup_relpath()
        if not mount or not rel:
            return None
        best: Optional[int] = None
        for d in _walk_cgroup_chain(mount, rel):
            for fname in ("memory.max", "memory/memory.limit_in_bytes",
                          "memory.limit_in_bytes"):
                raw = _read_text(os.path.join(d, fname))
                if raw is None:
                    continue
                raw = raw.strip().lower()
                if raw in ("max", ""):
                    break  # unlimited at this level; try next dir up
                try:
                    val = int(raw)
                except (ValueError, TypeError):
                    break
                if val >= _CGROUP_V1_UNLIMITED or val <= 0:
                    break  # sentinel / nonsense = unlimited here
                best = val if best is None else min(best, val)
                break
        return best
    except Exception:  # noqa: BLE001 - best-effort probe
        return None


def cgroup_cpu_quota_cores() -> Optional[float]:
    """Effective cgroup CPU quota in cores (v2 ``cpu.max`` / v1
    ``cpu.cfs_quota_us`` ÷ ``cpu.cfs_period_us``).  None = unlimited."""
    try:
        mount = _cgroup_mount()
        rel = _self_cgroup_relpath()
        if not mount or not rel:
            return None
        for d in _walk_cgroup_chain(mount, rel):
            raw = _read_text(os.path.join(d, "cpu.max"))
            if raw:
                parts = raw.split()
                if parts and parts[0].lower() != "max":
                    try:
                        quota = float(parts[0])
                        period = float(parts[1]) if len(parts) > 1 else 100000.0
                        if period > 0 and quota > 0:
                            return max(0.1, quota / period)
                    except (ValueError, TypeError, IndexError):
                        pass
                break
            quota = _read_float(os.path.join(d, "cpu/cpu.cfs_quota_us"))
            if quota is None:
                quota = _read_float(os.path.join(d, "cpu.cfs_quota_us"))
            if quota is not None and quota > 0:
                period = _read_float(os.path.join(d, "cpu/cpu.cfs_period_us"))
                if period is None:
                    period = _read_float(os.path.join(d, "cpu.cfs_period_us"))
                period = period or 100000.0
                if period > 0:
                    return max(0.1, quota / period)
            if quota is not None:
                break
        return None
    except Exception:  # noqa: BLE001 - best-effort probe
        return None


# ---------------------------------------------------------------------------
# /proc/stat cpu + steal, swap, self RSS
# ---------------------------------------------------------------------------

def _read_cpu_stat() -> Optional[tuple[int, int, int]]:
    """(total_jiffies, idle_jiffies, steal_jiffies) for aggregate cpu."""
    try:
        with open("/proc/stat", "r", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("cpu "):
                    f = line.split()
                    nums = [int(x) for x in f[1:9]]
                    while len(nums) < 8:
                        nums.append(0)
                    user, nice, system, idle, iowait, irq, softirq, steal = nums
                    total = sum(nums)
                    return total, idle + iowait, steal
                if not line.startswith("cpu"):
                    break
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_swap() -> tuple[Optional[int], Optional[int]]:
    """(SwapTotal_kb, SwapFree_kb) from /proc/meminfo."""
    total = free = None
    try:
        with open("/proc/meminfo", "r", encoding="ascii",
                  errors="replace") as fh:
            for line in fh:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1])
                if total is not None and free is not None:
                    break
    except (OSError, ValueError, IndexError):
        pass
    return total, free


def _read_swap_io() -> Optional[tuple[int, int]]:
    """(pswpin, pswpout) cumulative page swap counters from /proc/vmstat."""
    pin = pout = None
    try:
        with open("/proc/vmstat", "r", encoding="ascii",
                  errors="replace") as fh:
            for line in fh:
                if line.startswith("pswpin"):
                    pin = int(line.split()[1])
                elif line.startswith("pswpout"):
                    pout = int(line.split()[1])
                if pin is not None and pout is not None:
                    break
    except (OSError, ValueError, IndexError):
        pass
    if pin is None or pout is None:
        return None
    return pin, pout


def _self_rss_mb() -> Optional[float]:
    """Our own resident set size in MB (for budget accounting)."""
    try:
        with open("/proc/self/status", "r", encoding="ascii",
                  errors="replace") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# battery detail
# ---------------------------------------------------------------------------

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


def _battery_detail() -> dict[str, Any]:
    """Capacity, status, power and energy for the first readable battery.

    Keys: ``capacity`` (float|None), ``status`` (str|None — Charging /
    Discharging / Full / Not charging / Unknown), ``power_w``
    (float|None, signed: +charging/-discharging on some drivers),
    ``energy_now_wh``, ``energy_full_wh`` (float|None).
    """
    detail: dict[str, Any] = {
        "capacity": None, "status": None, "power_w": None,
        "energy_now_wh": None, "energy_full_wh": None,
    }
    try:
        for cap_path in _battery_capacity_paths():
            base = os.path.dirname(cap_path)
            cap = _read_float(cap_path)
            if cap is None:
                continue
            detail["capacity"] = max(0.0, min(100.0, cap))
            status = _read_text(os.path.join(base, "status"))
            if status:
                detail["status"] = status.strip()
            power_uw = _read_float(os.path.join(base, "power_now"))
            if power_uw is not None:
                detail["power_w"] = power_uw / 1e6
            else:
                # current_now (µA) * voltage_now (µV) = pW -> W
                cur = _read_float(os.path.join(base, "current_now"))
                volt = _read_float(os.path.join(base, "voltage_now"))
                if cur is not None and volt is not None:
                    detail["power_w"] = abs(cur * volt) / 1e12
            e_now = _read_float(os.path.join(base, "energy_now"))
            e_full = _read_float(os.path.join(base, "energy_full"))
            if e_now is not None:
                detail["energy_now_wh"] = e_now / 1e6
            if e_full is not None:
                detail["energy_full_wh"] = e_full / 1e6
            # charge_now (µAh) fallback when energy files are absent
            if detail["energy_now_wh"] is None:
                c_now = _read_float(os.path.join(base, "charge_now"))
                c_full = _read_float(os.path.join(base, "charge_full"))
                v = _read_float(os.path.join(base, "voltage_now"))
                if c_now is not None and v:
                    detail["energy_now_wh"] = c_now / 1e6 * (v / 1e6)
                if c_full is not None and v:
                    detail["energy_full_wh"] = c_full / 1e6 * (v / 1e6)
            return detail
    except Exception:  # noqa: BLE001 - battery probe is best-effort
        pass
    return detail


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
    mem_percent: Optional[float]        # used / total * 100 (host meminfo)
    disk_percent: Optional[float]       # used / total * 100 (disk_path volume)
    battery_percent: Optional[float]    # remaining charge %, None if no battery
    thermal_state: Optional[str]        # "nominal" | "warm" | "hot" | None
    network_online: bool
    metered: Optional[bool]             # True/False/None(unknown)
    ts: float = field(default_factory=time.time)
    # -- full-signal additions (all optional; None = unreadable) ----------
    cpu_util_percent: Optional[float] = None   # delta-sampled true CPU util %
    steal_percent: Optional[float] = None      # delta-sampled steal % (vCPU ready, hypervisor busy)
    mem_available_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None
    swap_used_percent: Optional[float] = None  # None = no swap configured
    effective_mem_mb: Optional[float] = None   # min(cgroup limit, host RAM)
    effective_cpu_count: Optional[float] = None  # min(cgroup quota, cpu_count)
    self_rss_mb: Optional[float] = None        # our own RSS
    battery_status: Optional[str] = None       # Charging|Discharging|Full|...
    battery_power_w: Optional[float] = None    # signed instantaneous W
    energy_now_wh: Optional[float] = None
    energy_full_wh: Optional[float] = None
    thermal_c: Optional[float] = None          # hottest zone, Celsius
    disk_free_mb: Optional[float] = None
    psi: Optional[dict[str, dict[str, Any]]] = None  # cpu/memory/io -> PressureStall dict
    environment: Optional[dict[str, Any]] = None    # EnvironmentInfo dict

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
            "cpu_util_percent": self.cpu_util_percent,
            "steal_percent": self.steal_percent,
            "mem_available_mb": self.mem_available_mb,
            "mem_total_mb": self.mem_total_mb,
            "swap_used_percent": self.swap_used_percent,
            "effective_mem_mb": self.effective_mem_mb,
            "effective_cpu_count": self.effective_cpu_count,
            "self_rss_mb": self.self_rss_mb,
            "battery_status": self.battery_status,
            "battery_power_w": self.battery_power_w,
            "energy_now_wh": self.energy_now_wh,
            "energy_full_wh": self.energy_full_wh,
            "thermal_c": self.thermal_c,
            "disk_free_mb": self.disk_free_mb,
            "psi": self.psi,
            "environment": self.environment,
        }


# ---------------------------------------------------------------------------
# degradation ladder — damped, five levels, slow-start recovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DegradationStep:
    level: int
    name: str
    description: str
    shed_tiers: tuple[str, ...] = ()  # task tiers shed at this level (bottom-up)
    actions: tuple[str, ...] = ()     # recommended operator actions


DEGRADATION_LEVELS: tuple[DegradationStep, ...] = (
    DegradationStep(0, "full",
                    "all work flows",
                    (),
                    ("no action",)),
    DegradationStep(1, "light-shed",
                    "bulk background tasks deferred; heavy work throttled",
                    ("bulk",),
                    ("defer bulk tasks", "throttle heavy work", "reduce media quality")),
    DegradationStep(2, "degraded",
                    "background tasks deferred; heavy LLM/media gated",
                    ("bulk", "background"),
                    ("defer background tasks", "gate heavy LLM/media on budget admission",
                     "pause model downloads")),
    DegradationStep(3, "survival",
                    "only critical and important work; LLM must be light",
                    ("bulk", "background", "degraded"),
                    ("suspend media pipeline", "LLM: small/quantized only",
                     "stop prefetching and indexing")),
    DegradationStep(4, "essential",
                    "critical only: interactive chat, alarms, safety",
                    ("bulk", "background", "degraded", "important"),
                    ("pause everything non-critical", "alert the operator")),
)

_MAX_LEVEL = len(DEGRADATION_LEVELS) - 1


class DegradationLadder:
    """Damped degradation state machine.

    ``update(target, *, critical)`` moves toward ``target``: non-critical
    pressure steps up one level after ``up_ticks`` sustained ticks;
    critical pressure jumps immediately.  Recovery is slow-start: the
    ladder steps down one level only after ``calm_seconds`` of sustained
    calm, so a recovering system is never stampeded.
    Never raises.
    """

    def __init__(self, *, up_ticks: int = 2, calm_seconds: float = 120.0) -> None:
        self.up_ticks = max(1, int(up_ticks))
        self.calm_seconds = max(0.0, float(calm_seconds))
        self.level = 0
        self._up_streak = 0
        self._calm_since: Optional[float] = None
        self._last_update = time.time()

    def step(self) -> DegradationStep:
        return DEGRADATION_LEVELS[max(0, min(_MAX_LEVEL, self.level))]

    def allowed_tiers(self) -> tuple[str, ...]:
        """Task tiers that may run at the current level."""
        shed = set(self.step().shed_tiers)
        return tuple(t for t in TASK_TIERS if t not in shed)

    def tier_allowed(self, tier: str) -> bool:
        return str(tier).lower() in self.allowed_tiers()

    def update(self, target: int, *, critical: bool = False,
               now: Optional[float] = None) -> int:
        """Move toward ``target`` with damping.  Returns the new level."""
        try:
            now = time.time() if now is None else float(now)
            target = max(0, min(_MAX_LEVEL, int(target)))
            self._last_update = now
            if target > self.level:
                self._calm_since = None
                if critical:
                    self.level = target
                    self._up_streak = 0
                else:
                    self._up_streak += 1
                    if self._up_streak >= self.up_ticks:
                        self.level = min(self.level + 1, target)
                        self._up_streak = 0
            elif target < self.level:
                self._up_streak = 0
                if self._calm_since is None:
                    self._calm_since = now
                if now - self._calm_since >= self.calm_seconds:
                    self.level = max(self.level - 1, target)
                    self._calm_since = now  # each step down needs its own calm window
            else:
                # target == level: hold steady, reset the up-streak.
                self._up_streak = 0
            return self.level
        except Exception:  # noqa: BLE001 - ladder never raises
            return self.level

    def to_dict(self) -> dict[str, Any]:
        s = self.step()
        return {
            "level": self.level,
            "name": s.name,
            "description": s.description,
            "shed_tiers": list(s.shed_tiers),
            "allowed_tiers": list(self.allowed_tiers()),
            "actions": list(s.actions),
        }


# ---------------------------------------------------------------------------
# subsystem budgets — admission control ("does this load fit?")
# ---------------------------------------------------------------------------

@dataclass
class SubsystemBudget:
    """Tracked usage against a named subsystem's caps."""

    name: str
    tier: str = "background"
    max_mem_mb: Optional[float] = None
    max_cpu_pct: Optional[float] = None
    max_concurrent: Optional[int] = None
    used_mem_mb: float = 0.0
    used_cpu_pct: float = 0.0
    running: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "max_mem_mb": self.max_mem_mb,
            "max_cpu_pct": self.max_cpu_pct,
            "max_concurrent": self.max_concurrent,
            "used_mem_mb": self.used_mem_mb,
            "used_cpu_pct": self.used_cpu_pct,
            "running": self.running,
        }


class ResourceBudgets:
    """Per-subsystem admission control.

    ``acquire(name, mem_mb, cpu_pct)`` answers "does this load fit?" —
    within the subsystem's own caps AND within the machine's effective
    resources minus the reserve margin.  It mutates tracked usage on
    success; :meth:`admission` is the read-only variant.  Never raises.
    """

    #: default subsystems (caps are fractions of effective resources unless
    #: overridden via define())
    _DEFAULTS: tuple[tuple[str, str, float, float, Optional[int]], ...] = (
        # name, tier, mem_frac, cpu_frac, max_concurrent
        ("system", "critical", 0.10, 0.15, None),
        ("llm", "important", 0.70, 0.80, 2),
        ("media", "background", 0.40, 0.60, 3),
        ("scheduler", "background", 0.10, 0.20, 8),
        ("missions", "important", 0.50, 0.60, 4),
        ("bulk", "bulk", 0.15, 0.25, 4),
    )

    def __init__(
        self,
        manager: "ResourceManager",
        *,
        reserve_pct: float = 0.06,
        reserve_min_mb: float = 256.0,
    ) -> None:
        self._manager = manager
        self.reserve_pct = max(0.0, min(0.5, float(reserve_pct)))
        self.reserve_min_mb = max(0.0, float(reserve_min_mb))
        self._budgets: dict[str, SubsystemBudget] = {}
        self._sized = False

    # -- definition ------------------------------------------------------
    def define(
        self,
        name: str,
        *,
        tier: str = "background",
        max_mem_mb: Optional[float] = None,
        max_cpu_pct: Optional[float] = None,
        max_concurrent: Optional[int] = None,
    ) -> SubsystemBudget:
        try:
            b = SubsystemBudget(
                name=name,
                tier=str(tier).lower(),
                max_mem_mb=max_mem_mb,
                max_cpu_pct=max_cpu_pct,
                max_concurrent=max_concurrent,
            )
            self._budgets[name] = b
            return b
        except Exception:  # noqa: BLE001 - never raises
            return SubsystemBudget(name=name)

    def ensure_defaults(self) -> None:
        """Size the default subsystems from effective resources (once)."""
        if self._sized:
            return
        self._sized = True
        try:
            eff_mem = self._manager.effective_memory_mb() or 1024.0
            eff_cpu = self._manager.effective_cpu_count() or 1.0
            for name, tier, mem_frac, cpu_frac, conc in self._DEFAULTS:
                if name in self._budgets:
                    continue
                self._budgets[name] = SubsystemBudget(
                    name=name,
                    tier=tier,
                    max_mem_mb=eff_mem * mem_frac,
                    max_cpu_pct=min(100.0, eff_cpu * 100.0 * cpu_frac),
                    max_concurrent=conc,
                )
        except Exception:  # noqa: BLE001 - never raises
            pass

    def get(self, name: str) -> Optional[SubsystemBudget]:
        self.ensure_defaults()
        return self._budgets.get(name)

    # -- reserve ----------------------------------------------------------
    def reserve_mb(self) -> float:
        """MB that must stay free machine-wide (oma-style reserve)."""
        try:
            self.ensure_defaults()
            eff = self._manager.effective_memory_mb() or 0.0
            return max(eff * self.reserve_pct, self.reserve_min_mb)
        except Exception:  # noqa: BLE001 - never raises
            return self.reserve_min_mb

    def machine_headroom_mb(self) -> Optional[float]:
        """Free MB beyond the reserve.  None when unreadable."""
        try:
            s = self._manager.sample()
            if s.effective_mem_mb is None:
                return None
            avail = s.mem_available_mb
            if avail is None:
                return None
            return avail - self.reserve_mb()
        except Exception:  # noqa: BLE001 - never raises
            return None

    # -- admission ---------------------------------------------------------
    def admission(
        self,
        name: str,
        *,
        mem_mb: float = 0.0,
        cpu_pct: float = 0.0,
    ) -> tuple[bool, list[str]]:
        """Read-only "would this fit?" check.  Never raises."""
        reasons: list[str] = []
        try:
            self.ensure_defaults()
            b = self._budgets.get(name)
            if b is None:
                return True, [f"no budget defined for {name!r}; allowed"]
            if b.max_concurrent is not None and b.running >= b.max_concurrent:
                reasons.append(
                    f"budget {name}: {b.running}/{b.max_concurrent} slots in use")
            if b.max_mem_mb is not None and b.used_mem_mb + mem_mb > b.max_mem_mb:
                reasons.append(
                    f"budget {name}: mem {b.used_mem_mb + mem_mb:.0f}MB > "
                    f"cap {b.max_mem_mb:.0f}MB")
            if b.max_cpu_pct is not None and b.used_cpu_pct + cpu_pct > b.max_cpu_pct:
                reasons.append(
                    f"budget {name}: cpu {b.used_cpu_pct + cpu_pct:.0f}% > "
                    f"cap {b.max_cpu_pct:.0f}%")
            headroom = self.machine_headroom_mb()
            if mem_mb > 0 and headroom is not None and mem_mb > headroom:
                reasons.append(
                    f"machine headroom {headroom:.0f}MB < requested {mem_mb:.0f}MB "
                    f"(reserve {self.reserve_mb():.0f}MB)")
            return (not reasons), reasons
        except Exception as exc:  # noqa: BLE001 - never raises
            return False, [f"budget check failed: {exc}"]

    def acquire(
        self,
        name: str,
        *,
        mem_mb: float = 0.0,
        cpu_pct: float = 0.0,
    ) -> bool:
        """Admit a load: checks fit, then tracks it.  Returns True if admitted."""
        try:
            ok, _ = self.admission(name, mem_mb=mem_mb, cpu_pct=cpu_pct)
            if not ok:
                return False
            b = self._budgets.get(name)
            if b is None:
                b = self.define(name)
            b.used_mem_mb += max(0.0, mem_mb)
            b.used_cpu_pct += max(0.0, cpu_pct)
            b.running += 1
            return True
        except Exception:  # noqa: BLE001 - never raises
            return False

    def release(
        self,
        name: str,
        *,
        mem_mb: float = 0.0,
        cpu_pct: float = 0.0,
    ) -> None:
        """Release tracked usage.  Never raises, never goes negative."""
        try:
            b = self._budgets.get(name)
            if b is None:
                return
            b.used_mem_mb = max(0.0, b.used_mem_mb - max(0.0, mem_mb))
            b.used_cpu_pct = max(0.0, b.used_cpu_pct - max(0.0, cpu_pct))
            b.running = max(0, b.running - 1)
        except Exception:  # noqa: BLE001 - never raises
            pass

    def snapshot(self) -> dict[str, Any]:
        try:
            self.ensure_defaults()
            return {
                name: b.to_dict() for name, b in sorted(self._budgets.items())
            }
        except Exception:  # noqa: BLE001 - never raises
            return {}


# ---------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------

class ResourceManager:
    """Sample resources and give advisory consults.  Advisory only.

    Every public method is exception-safe: samplers are defensive and
    :meth:`consult` / :meth:`pressure` / :meth:`forecast` never raise.
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
        # -- full-signal additions --------------------------------------
        history_size: int = 120,
        reserve_pct: float = 0.06,
        reserve_min_mb: float = 256.0,
        calm_seconds: float = 120.0,
        up_ticks: int = 2,
        steal_warn: float = 0.10,
        steal_critical: float = 0.25,
        psi_mem_warn: float = 25.0,
        psi_mem_full_critical: float = 10.0,
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

        v = _env_int("NM_RESOURCE_HISTORY")
        self.history_size = v if v is not None else int(history_size)
        v = _env_float("NM_RESOURCE_RESERVE_PCT")
        self.reserve_pct = v if v is not None else float(reserve_pct)
        v = _env_float("NM_RESOURCE_RESERVE_MIN_MB")
        self.reserve_min_mb = v if v is not None else float(reserve_min_mb)
        v = _env_float("NM_RESOURCE_CALM_SECONDS")
        self.calm_seconds = v if v is not None else float(calm_seconds)
        v = _env_int("NM_RESOURCE_UP_TICKS")
        self.up_ticks = v if v is not None else int(up_ticks)
        v = _env_float("NM_RESOURCE_STEAL_WARN")
        self.steal_warn = v if v is not None else float(steal_warn)
        v = _env_float("NM_RESOURCE_STEAL_CRITICAL")
        self.steal_critical = v if v is not None else float(steal_critical)
        v = _env_float("NM_RESOURCE_PSI_MEM_WARN")
        self.psi_mem_warn = v if v is not None else float(psi_mem_warn)
        v = _env_float("NM_RESOURCE_PSI_MEM_FULL_CRIT")
        self.psi_mem_full_critical = (
            v if v is not None else float(psi_mem_full_critical)
        )

        # battery-powered platforms get a more conservative advisory
        if self._is_battery_powered():
            self.battery_throttle = max(self.battery_throttle, 40.0)
            self.cpu_throttle = min(self.cpu_throttle, 0.60)
            self.mem_throttle = min(self.mem_throttle, 0.70)

        # rolling history for trend/forecast (ts -> key -> value)
        self._history: deque[dict[str, Any]] = deque(
            maxlen=max(2, self.history_size))
        # delta samplers
        self._last_cpu_stat: Optional[tuple[float, tuple[int, int, int]]] = None
        self._last_swap_io: Optional[tuple[float, tuple[int, int]]] = None
        # effective-resource cache (cgroup walks are cheap but not free)
        self._eff_cache: dict[str, tuple[float, Any]] = {}
        # damped degradation ladder + subsystem budgets
        self.ladder = DegradationLadder(up_ticks=self.up_ticks,
                                       calm_seconds=self.calm_seconds)
        self.budgets = ResourceBudgets(
            self, reserve_pct=self.reserve_pct,
            reserve_min_mb=self.reserve_min_mb)
        # alert subscriptions: list of (key, threshold, comparator, fn)
        self._alerts: list[dict[str, Any]] = []
        self._alert_state: dict[str, bool] = {}  # alert-id -> currently firing

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

    # -- effective resources (cgroup-aware) --------------------------------
    def effective_memory_mb(self) -> Optional[float]:
        """min(cgroup memory limit, host RAM) in MB.  Cached ~60s."""
        try:
            now = time.time()
            hit = self._eff_cache.get("mem")
            if hit and now - hit[0] < 60.0:
                return hit[1]
            host_mb: Optional[float] = None
            try:
                with open("/proc/meminfo", "r", encoding="ascii",
                          errors="replace") as fh:
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            host_mb = int(line.split()[1]) / 1024.0
                            break
            except (OSError, ValueError, IndexError):
                pass
            limit_b = cgroup_memory_limit_bytes()
            if host_mb is None:
                val = (limit_b / (1024.0 * 1024.0)) if limit_b else None
            elif limit_b:
                val = min(limit_b / (1024.0 * 1024.0), host_mb)
            else:
                val = host_mb
            self._eff_cache["mem"] = (now, val)
            return val
        except Exception:  # noqa: BLE001 - never raises
            return None

    def effective_cpu_count(self) -> Optional[float]:
        """min(cgroup CPU quota, cpu_count()).  Cached ~60s."""
        try:
            now = time.time()
            hit = self._eff_cache.get("cpu")
            if hit and now - hit[0] < 60.0:
                return hit[1]
            host = float(os.cpu_count() or 1)
            quota = cgroup_cpu_quota_cores()
            val = min(quota, host) if quota else host
            self._eff_cache["cpu"] = (now, val)
            return val
        except Exception:  # noqa: BLE001 - never raises
            return None

    # -- samplers (each defensive; never raises) ---------------------------
    def _sample_cpu(self) -> Optional[float]:
        try:
            load1 = os.getloadavg()[0]
            ncpu = self.effective_cpu_count() or os.cpu_count() or 1
            pct = load1 / max(1.0, ncpu) * 100.0
            return max(0.0, min(100.0, pct))
        except (OSError, AttributeError, ValueError):
            return None

    def _sample_cpu_deltas(self) -> tuple[Optional[float], Optional[float]]:
        """(true utilization %, steal %) from /proc/stat deltas.

        First call primes the pump and returns (None, None) — deltas need
        two readings.  Never raises.
        """
        try:
            now = time.time()
            cur = _read_cpu_stat()
            prev = self._last_cpu_stat
            self._last_cpu_stat = (now, cur) if cur else prev
            if not cur or not prev or prev[1] is None:
                return None, None
            _pts, (ptotal, pidle, psteal) = prev
            total, idle, steal = cur
            dt = total - ptotal
            if dt <= 0:
                return None, None
            util = max(0.0, min(100.0, (dt - (idle - pidle)) / dt * 100.0))
            st = max(0.0, min(100.0, (steal - psteal) / dt * 100.0))
            return util, st
        except Exception:  # noqa: BLE001 - never raises
            return None, None

    def _sample_mem(self) -> tuple[Optional[float], Optional[float],
                                   Optional[float]]:
        """(used %, available MB, total MB) from /proc/meminfo."""
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
            return None, None, None
        if not total or avail is None:
            return None, None, None
        used = max(0, total - avail)
        pct = max(0.0, min(100.0, used / total * 100.0))
        return pct, avail / 1024.0, total / 1024.0

    def _sample_swap(self) -> Optional[float]:
        """Swap used %.  None when no swap is configured."""
        try:
            total, free = _read_swap()
            if not total:
                return None
            used = max(0, total - (free or 0))
            return max(0.0, min(100.0, used / total * 100.0))
        except Exception:  # noqa: BLE001 - never raises
            return None

    def _sample_swap_io_rate(self) -> Optional[float]:
        """Swap pages/s (in+out) from /proc/vmstat deltas.  None on 1st call."""
        try:
            now = time.time()
            cur = _read_swap_io()
            prev = self._last_swap_io
            self._last_swap_io = (now, cur) if cur else prev
            if not cur or not prev or prev[1] is None:
                return None
            pts, (ppin, ppout) = prev
            dt = now - pts
            if dt <= 0:
                return None
            return max(0.0, ((cur[0] - ppin) + (cur[1] - ppout)) / dt)
        except Exception:  # noqa: BLE001 - never raises
            return None

    def _sample_disk(self) -> tuple[Optional[float], Optional[float]]:
        try:
            usage = shutil.disk_usage(self.disk_path)
            if not usage.total:
                return None, None
            used = usage.total - usage.free
            pct = max(0.0, min(100.0, used / usage.total * 100.0))
            return pct, usage.free / (1024.0 * 1024.0)
        except (OSError, ValueError):
            return None, None

    def _sample_battery(self) -> dict[str, Any]:
        return _battery_detail()

    def _sample_thermal(self) -> tuple[Optional[str], Optional[float]]:
        try:
            temps = _thermal_zone_temps()
        except Exception:  # noqa: BLE001 - thermal probe is best-effort
            return None, None
        if not temps:
            return None, None
        hottest = max(temps)
        if hottest >= self.thermal_hot_c:
            return _THERMAL_HOT, hottest
        if hottest >= self.thermal_warm_c:
            return _THERMAL_WARM, hottest
        return _THERMAL_NOMINAL, hottest

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

        mem_pct, mem_avail_mb, mem_total_mb = safe(self._sample_mem,
                                                  (None, None, None))
        disk_pct, disk_free_mb = safe(self._sample_disk, (None, None))
        batt = safe(self._sample_battery, {})
        thermal_state, thermal_c = safe(self._sample_thermal, (None, None))
        cpu_util, steal = safe(self._sample_cpu_deltas, (None, None))
        psi_raw = safe(sample_psi, None)
        psi = ({k: v.to_dict() for k, v in psi_raw.items()}
               if psi_raw else None)
        try:
            env = detect_environment().to_dict()
        except Exception:  # noqa: BLE001 - never raises
            env = None
        s = ResourceSample(
            cpu_percent=safe(self._sample_cpu, None),
            mem_percent=mem_pct,
            disk_percent=disk_pct,
            battery_percent=batt.get("capacity"),
            thermal_state=thermal_state,
            network_online=safe(self._sample_network, False),
            metered=self.metered,
            cpu_util_percent=cpu_util,
            steal_percent=steal,
            mem_available_mb=mem_avail_mb,
            mem_total_mb=mem_total_mb,
            swap_used_percent=safe(self._sample_swap, None),
            effective_mem_mb=safe(self.effective_memory_mb, None),
            effective_cpu_count=safe(self.effective_cpu_count, None),
            self_rss_mb=safe(_self_rss_mb, None),
            battery_status=batt.get("status"),
            battery_power_w=batt.get("power_w"),
            energy_now_wh=batt.get("energy_now_wh"),
            energy_full_wh=batt.get("energy_full_wh"),
            thermal_c=thermal_c,
            disk_free_mb=disk_free_mb,
            psi=psi,
            environment=env,
        )
        self._record(s)
        if self._alerts:
            try:
                self._eval_alerts(s, self.pressure(s))
            except Exception:  # noqa: BLE001 — alerts never break sampling
                pass
        return s

    def _record(self, s: ResourceSample) -> None:
        """Append a compact row to the rolling history.  Never raises."""
        try:
            psi_mem = None
            if s.psi and "memory" in s.psi:
                psi_mem = s.psi["memory"].get("some_avg10")
            self._history.append({
                "ts": s.ts,
                "cpu": s.cpu_percent,
                "mem": s.mem_percent,
                "disk": s.disk_percent,
                "steal": s.steal_percent,
                "psi_mem": psi_mem,
                "composite": self.composite_pressure(s),
                "battery": s.battery_percent,
                "battery_status": s.battery_status,
            })
        except Exception:  # noqa: BLE001 - history must not break sampling
            pass

    def history(self) -> list[dict[str, Any]]:
        """Copy of the rolling sample history (oldest first)."""
        try:
            return [dict(row) for row in self._history]
        except Exception:  # noqa: BLE001 - never raises
            return []

    # -- trend + forecast (predictive, not just reactive) ------------------
    def trend(self, key: str, window_s: float = 300.0) -> Optional[float]:
        """Least-squares slope per *second* for ``key`` over the window.

        Positive = rising.  None when fewer than 2 usable points.
        Never raises.
        """
        try:
            now = time.time()
            pts = [(r["ts"], r[key]) for r in self._history
                   if r.get("ts", 0) >= now - window_s
                   and r.get(key) is not None]
            if len(pts) < 2:
                return None
            n = len(pts)
            # centered form: raw epoch timestamps (~1.7e9) would lose all
            # precision to catastrophic cancellation in float64
            mx = sum(p[0] for p in pts) / n
            my = sum(p[1] for p in pts) / n
            num = sum((p[0] - mx) * (p[1] - my) for p in pts)
            den = sum((p[0] - mx) ** 2 for p in pts)
            if not den:
                return 0.0
            return num / den
        except Exception:  # noqa: BLE001 - never raises
            return None

    def forecast(self, horizon_s: float = 300.0,
                 sample: Optional[ResourceSample] = None) -> dict[str, Any]:
        """Predict pressures at ``horizon_s`` and minutes-to-critical.

        ``predicted`` clamps the linear extrapolation to [0, 100];
        ``minutes_to_critical`` is None when the trend is flat/falling or
        the critical threshold is unreachable at the current slope.
        Never raises.
        """
        out: dict[str, Any] = {"horizon_s": horizon_s, "predicted": {},
                               "minutes_to_critical": {}}
        try:
            specs = (
                ("cpu", self.cpu_critical * 100.0),
                ("mem", self.mem_critical * 100.0),
                ("disk", self.disk_critical * 100.0),
            )
            # current values come from the freshest history row, or the
            # caller-supplied sample when history is empty
            cur: dict[str, Optional[float]] = {}
            if self._history:
                last = self._history[-1]
                cur = {"cpu": last.get("cpu"), "mem": last.get("mem"),
                       "disk": last.get("disk")}
            elif sample is not None:
                cur = {"cpu": sample.cpu_percent, "mem": sample.mem_percent,
                       "disk": sample.disk_percent}
            for key, crit in specs:
                slope = self.trend(key)  # per second
                c = cur.get(key)
                if slope is None or c is None:
                    continue
                pred = max(0.0, min(100.0, c + slope * horizon_s))
                out["predicted"][key] = round(pred, 1)
                if slope > 1e-9 and c < crit:
                    mins = (crit - c) / slope / 60.0
                    out["minutes_to_critical"][key] = round(mins, 1)
                else:
                    out["minutes_to_critical"][key] = None
            return out
        except Exception:  # noqa: BLE001 - never raises
            return out

    def battery_forecast(self) -> dict[str, Any]:
        """Drain rate (%/h) and time-to-empty from sample history.

        Uses only discharging segments; charging periods are excluded from
        the drain average.  Never raises.
        """
        out: dict[str, Any] = {"drain_pct_per_h": None,
                               "time_to_empty_min": None,
                               "charging": None}
        try:
            rows = [r for r in self._history if r.get("battery") is not None]
            if len(rows) < 2:
                return out
            first, last = rows[0], rows[-1]
            dt_h = (last["ts"] - first["ts"]) / 3600.0
            if dt_h <= 0:
                return out
            statuses = {r.get("battery_status") for r in rows}
            charging = any(s and "charg" in s.lower() for s in statuses
                           if s)
            out["charging"] = charging or None
            drop = first["battery"] - last["battery"]
            if charging and drop <= 0:
                return out  # net charging: no drain to report
            rate = drop / dt_h
            if rate <= 0:
                return out
            out["drain_pct_per_h"] = round(rate, 2)
            if last["battery"] and last["battery"] > 0:
                out["time_to_empty_min"] = round(
                    last["battery"] / rate * 60.0, 1)
            return out
        except Exception:  # noqa: BLE001 - never raises
            return out

    # -- composite pressure -------------------------------------------------
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

    def composite_pressure(
            self, sample: Optional[ResourceSample] = None) -> float:
        """Memory-domain composite score 0..1.

        55% PSI-memory stall (some avg10) + 20% swap saturation +
        15% memory utilization + 10% swap I/O rate (normalized).  Catches
        fast-onset thrashing before raw utilization reacts.  Never raises.
        """
        try:
            s = sample or self.sample()
            psi_mem = 0.0
            if s.psi and "memory" in s.psi:
                v = s.psi["memory"].get("some_avg10")
                if v is not None:
                    psi_mem = max(0.0, min(1.0, v / 100.0))
            swap_sat = self._pct_to_pressure(s.swap_used_percent)
            mem_u = self._pct_to_pressure(s.mem_percent)
            swap_io = self._sample_swap_io_rate()
            # ~50 pages/s sustained swapping is already painful; normalize
            swap_io_p = 0.0 if swap_io is None else max(
                0.0, min(1.0, swap_io / 50.0))
            score = (0.55 * psi_mem + 0.20 * swap_sat
                     + 0.15 * mem_u + 0.10 * swap_io_p)
            return max(0.0, min(1.0, score))
        except Exception:  # noqa: BLE001 - never raises
            return 0.0

    def pressure(self, sample: Optional[ResourceSample] = None) -> dict[str, float]:
        """0..1 pressure per resource plus ``overall`` = max.  Never raises."""
        try:
            s = sample or self.sample()
            battery_pct = s.battery_percent
            battery_p = 0.0 if battery_pct is None else 1.0 - battery_pct / 100.0
            steal_p = 0.0 if s.steal_percent is None else max(
                0.0, min(1.0, s.steal_percent / 100.0))
            psi_mem_p = 0.0
            if s.psi and "memory" in s.psi:
                v = s.psi["memory"].get("some_avg10")
                if v is not None:
                    psi_mem_p = max(0.0, min(1.0, v / 100.0))
            parts = {
                "cpu": self._pct_to_pressure(s.cpu_percent),
                "cpu_util": self._pct_to_pressure(s.cpu_util_percent),
                "mem": self._pct_to_pressure(s.mem_percent),
                "disk": self._pct_to_pressure(s.disk_percent),
                "battery": max(0.0, min(1.0, battery_p)),
                "thermal": self._thermal_to_pressure(s.thermal_state),
                "steal": steal_p,
                "psi_mem": psi_mem_p,
                "composite_mem": self.composite_pressure(s),
            }
            parts["overall"] = max(parts.values()) if parts else 0.0
            return parts
        except Exception:  # noqa: BLE001 - pressure must never raise
            return {"cpu": 0.0, "cpu_util": 0.0, "mem": 0.0, "disk": 0.0,
                    "battery": 0.0, "thermal": 0.0, "steal": 0.0,
                    "psi_mem": 0.0, "composite_mem": 0.0, "overall": 0.0}

    # -- degradation target -------------------------------------------------
    def _degradation_target(self, s: ResourceSample,
                            p: dict[str, float]) -> tuple[int, bool]:
        """Map pressures to a ladder target level + critical flag."""
        try:
            target = 0
            critical = False

            # level 1: any throttle condition
            if (p["cpu"] >= self.cpu_throttle or p["mem"] >= self.mem_throttle
                    or p["disk"] >= self.disk_throttle
                    or s.thermal_state == _THERMAL_WARM
                    or (s.battery_percent is not None
                        and s.battery_percent <= self.battery_throttle)
                    or (s.steal_percent is not None
                        and s.steal_percent / 100.0 >= self.steal_warn)
                    or (s.metered if s.metered is not None else self.metered)):
                target = max(target, 1)

            # level 2: composite memory pressure or strong PSI / drain
            psi_some = None
            psi_full = None
            if s.psi and "memory" in s.psi:
                psi_some = s.psi["memory"].get("some_avg10")
                psi_full = s.psi["memory"].get("full_avg10")
            if (p["composite_mem"] >= 0.50
                    or (psi_some is not None and psi_some >= self.psi_mem_warn)
                    or (s.steal_percent is not None
                        and s.steal_percent / 100.0 >= self.steal_critical / 2)):
                target = max(target, 2)

            # level 3: critical conditions (non-instant unless combined)
            if (p["cpu"] >= self.cpu_critical
                    or p["mem"] >= self.mem_critical
                    or p["disk"] >= self.disk_critical
                    or s.thermal_state == _THERMAL_HOT
                    or (psi_full is not None
                        and psi_full >= self.psi_mem_full_critical)
                    or (s.battery_percent is not None
                        and s.battery_percent <= self.battery_min)):
                target = max(target, 3)

            # level 4: hard caps / combined crisis -> instant
            if (p["overall"] > self.max_pressure
                    or (p["mem"] >= self.mem_critical
                        and (self.trend("mem", 120.0) or 0.0) > 0)
                    or (s.thermal_state == _THERMAL_HOT
                        and p["cpu"] >= self.cpu_critical)):
                target = max(target, 4)
                critical = True

            return target, critical
        except Exception:  # noqa: BLE001 - never raises
            return 0, False

    def consult(
        self,
        mission: Optional[Any] = None,
        subsystem: Optional[str] = None,
    ) -> dict[str, Any]:
        """Advisory consult.

        Returns ``{ok, throttled, reasons, pressure, sample}`` plus
        ``degradation_level``, ``degradation`` (ladder state),
        ``forecast``, ``psi``, ``environment``, ``budgets``,
        ``budget_ok``/``budget_reasons`` (when ``subsystem`` is given),
        ``composite`` and ``battery`` forecast.  Advisory only — the
        caller decides what to do.  Never raises, never hard-blocks.
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

            # steal: the hypervisor is taking our cycles (burstable credit
            # depletion shows up here long before loadavg looks alarming)
            if s.steal_percent is not None:
                steal_f = s.steal_percent / 100.0
                if steal_f >= self.steal_critical:
                    ok = False
                    note(f"cpu steal {s.steal_percent:.1f}% >= critical "
                         f"{self.steal_critical*100:.0f}% — hypervisor contention "
                         f"(likely credit-depleted on burstable types)")
                elif steal_f >= self.steal_warn:
                    throttled = True
                    note(f"cpu steal {s.steal_percent:.1f}% >= warn "
                         f"{self.steal_warn*100:.0f}% — host contention")

            if s.mem_percent is None:
                note("memory unreadable")
            elif p["mem"] >= self.mem_critical:
                ok = False
                note(f"memory pressure {p['mem']:.2f} >= critical {self.mem_critical:.2f}")
            elif p["mem"] >= self.mem_throttle:
                throttled = True
                note(f"memory pressure {p['mem']:.2f} >= throttle {self.mem_throttle:.2f}")

            # composite memory pressure warns below the oomd band
            if p["composite_mem"] >= 0.50:
                throttled = True
                note(f"composite memory pressure {p['composite_mem']:.2f} "
                     f">= 0.50 — thrashing likely before raw mem% reacts")
            # no swap: memory pressure has no cushion
            if s.swap_used_percent is None and s.mem_percent is not None \
                    and p["mem"] >= self.mem_throttle:
                note("no swap configured — memory pressure has no cushion")

            if s.disk_percent is None:
                note("disk unreadable")
            elif p["disk"] >= self.disk_critical:
                ok = False
                note(f"disk pressure {p['disk']:.2f} >= critical {self.disk_critical:.2f}")
            elif p["disk"] >= self.disk_throttle:
                throttled = True
                note(f"disk pressure {p['disk']:.2f} >= throttle {self.disk_throttle:.2f}")

            # battery: charging relaxes the failure floor (plugged in = power)
            charging = bool(s.battery_status and
                            "charg" in s.battery_status.lower()
                            and "dis" not in s.battery_status.lower())
            batt_min = self.battery_min / 2.0 if charging else self.battery_min
            if s.battery_percent is None:
                note("battery unknown (no readable battery)")
            elif s.battery_percent <= batt_min:
                ok = False
                note(f"battery {s.battery_percent:.0f}% <= minimum {batt_min:.0f}%"
                     + (" (charging)" if charging else ""))
            elif s.battery_percent <= self.battery_throttle:
                throttled = True
                note(f"battery {s.battery_percent:.0f}% <= throttle level "
                     f"{self.battery_throttle:.0f}%"
                     + (" (charging)" if charging else ""))

            if s.thermal_state is None:
                note("thermal unknown")
            elif s.thermal_state == _THERMAL_HOT:
                ok = False
                note(f"thermal state hot ({s.thermal_c:.0f}C)"
                     if s.thermal_c else "thermal state hot")
            elif s.thermal_state == _THERMAL_WARM:
                throttled = True
                note(f"thermal state warm ({s.thermal_c:.0f}C)"
                     if s.thermal_c else "thermal state warm")

            # PSI memory: reactive stall signal
            if s.psi and "memory" in s.psi:
                some10 = s.psi["memory"].get("some_avg10")
                full10 = s.psi["memory"].get("full_avg10")
                if full10 is not None and full10 >= self.psi_mem_full_critical:
                    ok = False
                    note(f"memory PSI full avg10 {full10:.1f}% >= "
                         f"{self.psi_mem_full_critical:.0f}% — total stall")
                elif some10 is not None and some10 >= self.psi_mem_warn:
                    throttled = True
                    note(f"memory PSI some avg10 {some10:.1f}% >= "
                         f"{self.psi_mem_warn:.0f}% — tasks stalling on memory")

            if p["overall"] > self.max_pressure:
                ok = False
                note(f"overall pressure {p['overall']:.2f} > cap {self.max_pressure:.2f}")

            # degradation ladder (damped)
            target, critical = self._degradation_target(s, p)
            level = self.ladder.update(target, critical=critical)
            degradation = self.ladder.to_dict()
            if level > 0:
                throttled = True
                note(f"degradation level {level} ({degradation['name']}): "
                     f"shed tiers {degradation['shed_tiers']}")

            # mission may carry its own pressure cap (duck-typed)
            cap = self._mission_cap(mission)
            if cap is not None and p["overall"] > cap:
                throttled = True
                note(f"mission pressure cap {cap:.2f} exceeded ({p['overall']:.2f})")

            # subsystem budget admission (duck-typed needs on the mission)
            budget_ok = True
            budget_reasons: list[str] = []
            if subsystem:
                need_mem, need_cpu = self._mission_needs(mission)
                budget_ok, budget_reasons = self.budgets.admission(
                    subsystem, mem_mb=need_mem, cpu_pct=need_cpu)
                if not budget_ok:
                    throttled = True
                    for br in budget_reasons:
                        note(br)

            if not s.network_online:
                note("network offline (local-only work)")

            if s.metered if s.metered is not None else self.metered:
                throttled = True
                note("metered network: avoid heavy transfers")

            # predictive layer: forecast + battery drain
            fc = self.forecast(300.0)
            for key, mins in fc.get("minutes_to_critical", {}).items():
                if mins is not None and mins <= 30.0:
                    throttled = True
                    note(f"{key} trending to critical in ~{mins:.0f} min")
            batt_fc = self.battery_forecast()

            env_kind = (s.environment or {}).get("kind", "unknown")
            if env_kind in ("aws-ec2", "gcp", "azure", "other-cloud") \
                    and s.steal_percent is not None \
                    and s.steal_percent / 100.0 >= self.steal_warn:
                note(f"cloud VM ({env_kind}) under host contention — "
                     f"consider lighter scheduling")

            return {
                "ok": ok,
                "throttled": throttled,
                "reasons": reasons,
                "pressure": p,
                "sample": s.to_dict(),
                "degradation_level": level,
                "degradation": degradation,
                "forecast": fc,
                "battery_forecast": batt_fc,
                "psi": s.psi,
                "environment": s.environment,
                "budgets": self.budgets.snapshot(),
                "budget_ok": budget_ok,
                "budget_reasons": budget_reasons,
                "composite": p["composite_mem"],
            }
        except Exception as exc:  # noqa: BLE001 - consult never raises
            return {
                "ok": False,
                "throttled": True,
                "reasons": [f"consult failed: {exc}"],
                "pressure": {},
                "sample": {},
                "degradation_level": 0,
                "degradation": {},
                "forecast": {},
                "battery_forecast": {},
                "psi": None,
                "environment": None,
                "budgets": {},
                "budget_ok": False,
                "budget_reasons": [f"consult failed: {exc}"],
                "composite": 0.0,
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

    @staticmethod
    def _mission_needs(mission: Optional[Any]) -> tuple[float, float]:
        """(mem_mb, cpu_pct) a mission says it needs.  Duck-typed."""
        try:
            if mission is None:
                return 0.0, 0.0
            if isinstance(mission, dict):
                mem = mission.get("mem_mb", mission.get("memory_mb", 0.0))
                cpu = mission.get("cpu_pct", 0.0)
            else:
                mem = getattr(mission, "mem_mb",
                              getattr(mission, "memory_mb", 0.0))
                cpu = getattr(mission, "cpu_pct", 0.0)
            return max(0.0, float(mem or 0.0)), max(0.0, float(cpu or 0.0))
        except (TypeError, ValueError):
            return 0.0, 0.0

    # -- alerts ------------------------------------------------------------
    def on_alert(self, key: str, threshold: float,
                 fn: Callable[[dict[str, Any]], Any], *,
                 above: bool = True,
                 alert_id: str = "") -> str:
        """Subscribe to a resource alert.

        ``key`` is a pressure key (``"cpu"``, ``"mem"``, ``"disk"``,
        ``"overall"``) or a sample key (``"battery_percent"``,
        ``"thermal_c"``...).  ``fn`` receives
        ``{"key", "value", "threshold", "firing"}`` whenever the value
        crosses the threshold (edge-triggered: it fires once per crossing,
        and again when it re-arms).  Evaluated on every :meth:`sample`.
        Returns the alert id (for :meth:`clear_alert`).
        """
        aid = alert_id or f"{key}:{threshold:g}:{'above' if above else 'below'}"
        self._alerts.append({"id": aid, "key": key, "threshold": threshold,
                             "above": above, "fn": fn})
        self._alert_state.setdefault(aid, False)
        return aid

    def clear_alert(self, alert_id: str) -> bool:
        """Remove an alert subscription.  Returns True when one existed."""
        before = len(self._alerts)
        self._alerts = [a for a in self._alerts if a["id"] != alert_id]
        self._alert_state.pop(alert_id, None)
        return len(self._alerts) < before

    def _eval_alerts(self, sample: "ResourceSample",
                     pressures: dict[str, float]) -> None:
        sample_d = sample.to_dict()
        for alert in self._alerts:
            key = alert["key"]
            value = pressures.get(key, sample_d.get(key))
            if value is None:
                continue
            try:
                firing = (float(value) >= alert["threshold"]
                          if alert["above"]
                          else float(value) <= alert["threshold"])
            except (TypeError, ValueError):
                continue
            was = self._alert_state.get(alert["id"], False)
            self._alert_state[alert["id"]] = firing
            if firing != was:  # edge-triggered
                try:
                    alert["fn"]({"key": key, "value": float(value),
                                 "threshold": alert["threshold"],
                                 "firing": firing,
                                 "above": alert["above"]})
                except Exception:  # noqa: BLE001 — alerts never break sampling
                    pass

    # -- presentation ------------------------------------------------------
    @staticmethod
    def _gauge(frac: Optional[float], width: int = 16) -> str:
        """Unicode pressure bar.  ``None`` renders as unknown."""
        if frac is None:
            return "░" * width + "  n/a"
        frac = max(0.0, min(1.0, float(frac)))
        filled = int(round(frac * width))
        bar = "█" * filled + "░" * (width - filled)
        if frac >= 0.92:
            mark = "!!"
        elif frac >= 0.70:
            mark = " !"
        else:
            mark = "  "
        return f"{bar}{mark} {frac * 100:5.1f}%"

    def render(self, sample: Optional["ResourceSample"] = None,
               pressures: Optional[dict[str, float]] = None) -> str:
        """God-tier plain-text resource dashboard with pressure gauges."""
        try:
            sample = sample or self.sample()
            pressures = (pressures if pressures is not None
                         else self.pressure(sample))
        except Exception:  # noqa: BLE001 — render degrades, never raises
            return "resources — unavailable"
        g = self._gauge
        lines = ["resources"]
        cpu = sample.cpu_util_percent if sample.cpu_util_percent is not None \
            else sample.cpu_percent
        lines.append(f"  cpu     {g((cpu or 0) / 100)}"
                     + (f"  steal {sample.steal_percent:.1f}%"
                        if sample.steal_percent else ""))
        mem_line = f"  mem     {g(pressures.get('mem'))}"
        if sample.mem_available_mb is not None:
            mem_line += f"  {sample.mem_available_mb:.0f}MB avail"
        if sample.effective_mem_mb is not None:
            mem_line += f" / {sample.effective_mem_mb:.0f}MB effective"
        lines.append(mem_line)
        lines.append(f"  disk    {g(pressures.get('disk'))}"
                     + (f"  {sample.disk_free_mb:.0f}MB free"
                        if sample.disk_free_mb else ""))
        if sample.battery_percent is not None:
            status = f" ({sample.battery_status})" if sample.battery_status \
                else ""
            lines.append(f"  battery {g(sample.battery_percent / 100)}"
                         f"{status}")
        if sample.thermal_c is not None or sample.thermal_state:
            lines.append(f"  thermal {sample.thermal_c if sample.thermal_c is not None else '?'}°C"
                         f" [{sample.thermal_state or 'unknown'}]")
        psi = sample.psi or {}
        mem_psi = (psi.get("memory") or {})
        if mem_psi:
            some = mem_psi.get("some", {}).get("avg10")
            full = mem_psi.get("full", {}).get("avg10")
            lines.append(f"  psi     mem some {some if some is not None else '?'}%"
                         f" / full {full if full is not None else '?'}% (avg10)")
        overall = pressures.get("overall")
        ladder = self.ladder.step()
        lines.append(f"  overall {g(overall)}  ladder: {ladder.name}")
        lines.append(f"  net     {'online' if sample.network_online else 'offline'}"
                     + (" · metered" if sample.metered else ""))
        return "\n".join(lines)

    def summary_line(self) -> str:
        """One-line status-bar summary: ``cpu 34% · mem 61% · ok``."""
        try:
            sample = self.sample()
            pressures = self.pressure(sample)
            overall = pressures.get("overall")
            if overall is None:
                return "resources n/a"
            state = "ok"
            if overall >= 0.92:
                state = "critical"
            elif overall >= 0.70:
                state = "pressured"
            cpu = sample.cpu_util_percent if sample.cpu_util_percent is not None \
                else sample.cpu_percent
            return (f"cpu {cpu:.0f}% · mem {(pressures.get('mem') or 0) * 100:.0f}%"
                    f" · disk {(pressures.get('disk') or 0) * 100:.0f}%"
                    f" · {state}")
        except Exception:  # noqa: BLE001
            return "resources n/a"


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
    degrades to a throttled advisory.  Pass ``subsystem=`` through a
    mission dict key ``subsystem`` for budget admission.
    """
    mgr = manager if manager is not None else default_manager()

    def advise(mission: Optional[Any] = None) -> dict[str, Any]:
        try:
            subsystem = None
            if isinstance(mission, dict):
                subsystem = mission.get("subsystem")
            else:
                subsystem = getattr(mission, "subsystem", None)
            return mgr.consult(mission, subsystem=subsystem)
        except Exception as exc:  # noqa: BLE001 - advisor never raises
            return {
                "ok": False,
                "throttled": True,
                "reasons": [f"resource advisor failed: {exc}"],
                "pressure": {},
                "sample": {},
                "degradation_level": 0,
                "degradation": {},
                "forecast": {},
                "battery_forecast": {},
                "psi": None,
                "environment": None,
                "budgets": {},
                "budget_ok": False,
                "budget_reasons": [f"resource advisor failed: {exc}"],
                "composite": 0.0,
            }

    return advise
