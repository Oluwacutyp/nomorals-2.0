"""Runtime profile envelope + autoscale (GROUPS Tier 0).

The old worker math was ``cpu // 2`` / ``max(2, cpu - 1)`` — a formula
that knows nothing about the machine it runs on. On a Termux phone
that spins up a pool the phone cannot afford; on a workstation it
leaves cores idle.

This module sizes the runtime from the *profile envelope*
(:mod:`nomorals.core.profile` — termux ``(1, 2, 3)``, pc ``(2, 4, 8)``,
workstation ``(4, 8, 16)``), not from a raw CPU count:

* :func:`workers_for` — process/thread workers inside the envelope,
  shed-aware (heavy shed -> envelope minimum, never zero).
* :func:`process_workers` — 0 on Termux/mobile/embedded (fork is
  unreliable there; the platform backend already declares
  ``process_pool=False``), envelope-sized elsewhere.
* :class:`RuntimeProfileManager` — owns the live profile, re-detects
  on change (:meth:`re_detect`, logged), and exposes the one-line
  profile summary (:meth:`profile_visible_line`).
* :func:`hybrid_kwargs` — the ``(threads, processes, use_processes)``
  triple to hand to :class:`HybridExecutor`, so the executor stops
  guessing from ``cpu_count``.

Beyond the spec: load-based autoscale inside the envelope
(:func:`autoscale`), an explicit ``NM_WORKERS``/``NM_PROCESS_WORKERS``
override that still clamps to the envelope maximum (an env var must
never let Termux allocate 16 workers), and a change callback so the
dashboard can repaint when the profile moves.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.profile import (
    ENVELOPES,
    EnvironmentProfile,
    detect_profile,
    resolve_profile,
)

__all__ = [
    "SHED_LEVELS",
    "workers_for",
    "process_workers",
    "thread_workers",
    "autoscale",
    "shed_level",
    "free_memory_mb",
    "hybrid_kwargs",
    "RuntimeProfileManager",
]

_log = get_logger(__name__)

#: Shed levels, weakest -> strongest. Unknown values behave as "none".
SHED_LEVELS = ("none", "light", "medium", "heavy")


def _process_pool_ok(kind: str) -> bool:
    """Mirror of :mod:`nomorals.core.platform`'s rule, without touching
    its process-global cache: fork is unreliable on termux / mobile /
    embedded, fine everywhere else."""
    return (kind or "").strip().lower() not in {
        "termux", "mobile", "embedded"}


def _max_thread_workers(kind: str, cpu: int) -> int:
    """Mirror of the platform backend's ``max_workers`` cap."""
    if (kind or "").strip().lower() in {"termux", "mobile", "embedded"}:
        return 4
    return max(4, int(cpu or 4))


def shed_level(monitor: Any = None, status: Any = None) -> str:
    """Current power shed level: none | light | medium | heavy.

    Consults :class:`PowerAwareGate` when importable; any failure
    (no power module, no monitor, no telemetry) fails open to
    ``"none"`` — a missing battery sensor must never zero the pool.
    An explicit ``status`` object with a ``degradation``/``level``
    attribute, or a plain string, is honoured directly (tests and
    embedders).
    """
    if isinstance(status, str) and status.strip().lower() in SHED_LEVELS:
        return status.strip().lower()
    if status is not None:
        for attr in ("shed", "shed_level", "degradation", "level"):
            val = getattr(status, attr, None)
            if isinstance(val, str) and val.lower() in SHED_LEVELS:
                return val.lower()
            if isinstance(val, (int, float)):
                lvl = int(val)
                return ("none" if lvl <= 0 else "light" if lvl == 1
                        else "medium" if lvl == 2 else "heavy")
    try:
        from ..autonomy.power_aware import PowerAwareGate

        gate = PowerAwareGate(monitor)
        lvl = gate.shed_level(status)
        if isinstance(lvl, str) and lvl.lower() in SHED_LEVELS:
            return lvl.lower()
    except Exception:  # noqa: BLE001 - power telemetry is best-effort
        pass
    return "none"


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
        from ..core.profile import memory_mb

        return int(memory_mb())
    except Exception:  # noqa: BLE001
        return 0


def _shed_cap(profile: EnvironmentProfile, shed: str) -> int:
    """Shed-aware worker ceiling inside the envelope."""
    shed = (shed or "none").strip().lower()
    if shed == "heavy":
        return profile.min_vcpus
    if shed == "medium":
        return max(profile.min_vcpus, profile.target_vcpus - 1)
    if shed == "light":
        return profile.target_vcpus
    return profile.max_vcpus  # none: full envelope available


def _env_override() -> int:
    """NM_WORKERS / NM_PROCESS_WORKERS pin, or 0 when unset/invalid."""
    for key in ("NM_WORKERS", "NM_PROCESS_WORKERS"):
        try:
            val = int(os.environ.get(key, "") or "")
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return 0


def workers_for(profile: EnvironmentProfile | None = None,
                shed: str = "none",
                load: float | None = None) -> int:
    """Process-pool worker count for ``profile`` under ``shed``.

    Base = envelope target; ``load`` in [0, 1] autoscales between
    envelope min and the shed ceiling (low load -> min, high load ->
    ceiling). The result is always inside ``[min_vcpus, max_vcpus]``
    — never ``cpu // 2``, never zero, never above the envelope even
    when ``NM_WORKERS`` pins higher (the pin clamps to the max).
    """
    prof = profile or detect_profile()
    ceiling = _shed_cap(prof, shed)
    if load is None:
        workers = prof.target_vcpus
    else:
        try:
            frac = max(0.0, min(1.0, float(load)))
        except (TypeError, ValueError):
            frac = 0.5
        workers = prof.min_vcpus + round(
            (ceiling - prof.min_vcpus) * frac)
    workers = max(prof.min_vcpus, min(ceiling, int(workers)))
    override = _env_override()
    if override:
        # An explicit pin wins, but never above the envelope maximum —
        # an env var must not let Termux allocate workstation workers.
        workers = max(prof.min_vcpus,
                      min(prof.max_vcpus, override))
        workers = min(workers, ceiling) if (shed or "none") in (
            "medium", "heavy") else workers
    return workers


def process_workers(profile: EnvironmentProfile | None = None,
                    shed: str = "none",
                    load: float | None = None) -> int:
    """Process-pool workers: 0 where fork is unreliable (Termux etc.)."""
    prof = profile or detect_profile()
    if not _process_pool_ok(prof.kind):
        return 0
    return workers_for(prof, shed=shed, load=load)


def thread_workers(profile: EnvironmentProfile | None = None,
                   shed: str = "none") -> int:
    """I/O thread workers: 2x the process workers, capped per platform."""
    prof = profile or detect_profile()
    cap = _max_thread_workers(prof.kind, prof.cpu)
    workers = workers_for(prof, shed=shed) * 2
    return max(1, min(workers, max(cap, 1)))


def autoscale(profile: EnvironmentProfile | None = None,
              shed: str = "none",
              load: float | None = None) -> dict[str, int]:
    """Full autoscale decision: threads + processes inside the envelope."""
    prof = profile or detect_profile()
    return {
        "threads": thread_workers(prof, shed=shed),
        "processes": process_workers(prof, shed=shed, load=load),
        "workers": workers_for(prof, shed=shed, load=load),
        "min_vcpus": prof.min_vcpus,
        "target_vcpus": prof.target_vcpus,
        "max_vcpus": prof.max_vcpus,
    }


def hybrid_kwargs(profile: EnvironmentProfile | None = None,
                  shed: str = "none",
                  load: float | None = None) -> dict[str, Any]:
    """Kwargs for :class:`HybridExecutor` from the profile envelope."""
    prof = profile or detect_profile()
    use_processes = _process_pool_ok(prof.kind)
    return {
        "threads": thread_workers(prof, shed=shed),
        "processes": process_workers(prof, shed=shed, load=load),
        "use_processes": use_processes,
    }


class RuntimeProfileManager:
    """Owns the live runtime profile: detect, re-detect, report.

    ``re_detect()`` re-runs detection and logs + reports any change
    (kind, envelope, or CPU/RAM movement) so a phone that docks, a VPS
    that gets resized, or a profile pin change is visible instead of
    silently stale. ``on_change`` callbacks receive the
    ``(old, new, changes)`` triple.
    """

    def __init__(self,
                 profile: EnvironmentProfile | None = None,
                 on_change: Callable[[Any, Any, dict[str, Any]], None]
                 | None = None) -> None:
        self._profile = profile or detect_profile()
        self._on_change = on_change
        self._shed = "none"
        self.history: list[dict[str, Any]] = []

    @property
    def profile(self) -> EnvironmentProfile:
        return self._profile

    @property
    def shed(self) -> str:
        return self._shed

    def refresh_shed(self, monitor: Any = None,
                     status: Any = None) -> str:
        """Re-read the power shed level (best-effort, fails open)."""
        self._shed = shed_level(monitor, status)
        return self._shed

    def re_detect(self,
                  profile: EnvironmentProfile | None = None,
                  kind: str = "") -> dict[str, Any]:
        """Re-detect the profile; log + report when anything moved.

        ``profile`` pins an explicit profile (tests); ``kind`` pins a
        named profile via :func:`resolve_profile`; otherwise full
        re-detection runs. Returns ``{"changed": bool, ...}``.
        """
        old = self._profile
        if profile is not None:
            new = profile
        elif kind:
            new = resolve_profile(kind)
        else:
            new = detect_profile()
        changes: dict[str, Any] = {}
        for attr in ("kind", "min_vcpus", "target_vcpus", "max_vcpus",
                     "cpu", "memory_mb", "cloud"):
            o, n = getattr(old, attr, None), getattr(new, attr, None)
            if o != n:
                changes[attr] = {"old": o, "new": n}
        changed = bool(changes)
        if changed:
            _log.info("runtime profile changed: %s -> %s (%s)",
                      old.kind, new.kind, changes)
            self.history.append({"old": old.to_dict(),
                                 "new": new.to_dict(),
                                 "changes": changes})
            self._profile = new
            if self._on_change is not None:
                try:
                    self._on_change(old, new, changes)
                except Exception:  # noqa: BLE001 - callback must not break
                    _log.debug("runtime profile on_change failed",
                               exc_info=True)
        return {"changed": changed, "changes": changes,
                "profile": new.to_dict()}

    def workers(self, load: float | None = None) -> dict[str, int]:
        """Current autoscale decision under the live shed level."""
        return autoscale(self._profile, shed=self._shed, load=load)

    def profile_visible_line(self) -> str:
        """One line: ``termux 2 vcpu 1.4GB free heavy shed``."""
        from ..core.profile_visible import profile_visible_line

        return profile_visible_line(profile=self._profile,
                                    shed=self._shed)

    def summary(self) -> dict[str, Any]:
        """Full live summary: profile + workers + shed + free RAM."""
        from ..core.profile_visible import profile_summary

        out = profile_summary(profile=self._profile, shed=self._shed)
        out["workers"] = self.workers()
        return out
