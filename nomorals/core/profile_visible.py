"""Profile-visible one line (GROUPS Tier 0).

Console and dashboard show *one line* for the machine, not a dump::

    termux 2 vcpu 1.4GB free heavy shed

That line is :func:`profile_visible_line` / ``profile_summary()["line"]``:
profile kind, live worker count from the envelope, free RAM, and the
power shed level. :func:`profile_summary` returns the same line plus
the full dict (vcpu envelope, cpu, RAM total/free, power shed, cloud)
for dashboards that want the breakdown.

NOTE on layering: this module lives in ``nomorals.core`` (L1) but is
deliberately *not* imported by ``nomorals/core/__init__.py`` — L1 may
not import other subpackages, and the shed level consults
``nomorals.autonomy.power_aware``. The power lookup is a lazy,
best-effort import inside :func:`_power_shed` (fails open to ``"none"``),
so importing this module directly never pulls in autonomy at import
time. Import it as ``from nomorals.core.profile_visible import ...``.
"""

from __future__ import annotations

from typing import Any

from .profile import EnvironmentProfile, detect_profile, resolve_profile

__all__ = [
    "profile_summary",
    "profile_visible_line",
    "free_memory_mb",
    "power_shed",
]


def free_memory_mb() -> int:
    """Free (available) RAM in MB, or 0 when unreadable."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            avail = free = total = 0
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                elif line.startswith("MemFree:"):
                    free = int(line.split()[1]) // 1024
                elif line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
            if avail:
                return avail
            if free:
                return free
            return total
    except (OSError, ValueError, IndexError):
        pass
    try:
        from .profile import memory_mb

        return int(memory_mb())
    except Exception:  # noqa: BLE001
        return 0


def power_shed(monitor: Any = None, status: Any = None) -> str:
    """Current power shed level: none | light | medium | heavy.

    Lazy import of the autonomy power gate; any failure fails open to
    ``"none"``. An explicit ``status`` (string or object with a
    ``shed``/``degradation``/``level`` attribute) is honoured directly.
    """
    if isinstance(status, str) and status.strip().lower() in (
            "none", "light", "medium", "heavy"):
        return status.strip().lower()
    if status is not None:
        for attr in ("shed", "shed_level", "degradation", "level"):
            val = getattr(status, attr, None)
            if isinstance(val, str) and val.lower() in (
                    "none", "light", "medium", "heavy"):
                return val.lower()
            if isinstance(val, (int, float)):
                lvl = int(val)
                return ("none" if lvl <= 0 else "light" if lvl == 1
                        else "medium" if lvl == 2 else "heavy")
    try:
        # No import of autonomy/power modules: core must not import outside
        # core (layering). If the caller supplies a monitor object with a
        # shed_level() method, use it; otherwise fail open to "none".
        get = getattr(monitor, "shed_level", None)
        if callable(get):
            lvl = get(status)
            if isinstance(lvl, str) and lvl.lower() in (
                    "none", "light", "medium", "heavy"):
                return lvl.lower()
    except Exception:  # noqa: BLE001 - power telemetry is best-effort
        pass
    return "none"


def _workers_for(profile: EnvironmentProfile, shed: str) -> int:
    """Envelope workers under ``shed`` (mirrors runtime_profile)."""
    if shed == "heavy":
        return profile.min_vcpus
    if shed == "medium":
        return max(profile.min_vcpus, profile.target_vcpus - 1)
    if shed == "light":
        return profile.target_vcpus
    return profile.target_vcpus


def profile_summary(profile: EnvironmentProfile | str | None = None,
                    shed: str | None = None,
                    monitor: Any = None) -> dict[str, Any]:
    """Full profile dict + one-line string (key ``"line"``).

    ``profile``: a detected/pinned :class:`EnvironmentProfile`, a kind
    name (``"termux"``), or None (detect). ``shed``: explicit shed
    level, else read from the power gate.
    """
    if isinstance(profile, str):
        prof = resolve_profile(profile)
    else:
        prof = profile or detect_profile()
    shed_lvl = (shed.strip().lower() if isinstance(shed, str) and shed
                else power_shed(monitor))
    if shed_lvl not in ("none", "light", "medium", "heavy"):
        shed_lvl = "none"
    free_mb = free_memory_mb()
    total_mb = int(prof.memory_mb or 0)
    workers = _workers_for(prof, shed_lvl)
    line = (f"{prof.kind} {workers} vcpu {free_mb / 1024:.1f}GB free "
            f"{shed_lvl} shed")
    return {
        "profile": prof.kind,
        "kind": prof.kind,
        "vcpu": {
            "workers": workers,
            "min": prof.min_vcpus,
            "target": prof.target_vcpus,
            "max": prof.max_vcpus,
        },
        "workers": workers,
        "cpu": prof.cpu,
        "memory_mb": total_mb,
        "free_mb": free_mb,
        "free_gb": round(free_mb / 1024, 2),
        "power_shed": shed_lvl,
        "shed": shed_lvl,
        "cloud": prof.cloud,
        "detail": prof.detail,
        "line": line,
    }


def profile_visible_line(profile: EnvironmentProfile | str | None = None,
                         shed: str | None = None,
                         monitor: Any = None) -> str:
    """The one line: ``termux 2 vcpu 1.4GB free heavy shed``."""
    return str(profile_summary(profile, shed=shed,
                               monitor=monitor)["line"])
