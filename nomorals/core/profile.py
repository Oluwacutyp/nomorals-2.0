"""Environment profiles (wave 84): what kind of machine are we on, and
how many Virtual CPUs does it earn.

Detection is stdlib-only and conservative: it reads the platform,
architecture, CPU count, and (where readable) total memory, and maps
them onto one of five profiles — termux/mobile, pc, vps, workstation,
embedded — each carrying a ``min/target/max`` VCPU envelope.  The
workspace starts at ``target`` and may autoscale inside the envelope;
the envelope is what "profile-aware scaling" means: nothing is
hardcoded per deployment, and a 2-core Android phone never spins up
the 16-core pool a bare-metal workstation gets.

Every field can be overridden from config (``workspace.*``), and
:meth:`~nomorals.workspace.profile.detect_profile` takes explicit
``cpu``/``memory_mb``/``system`` arguments so tests and embedders can
pin the environment.
"""
from __future__ import annotations

import os
import platform as _platform
import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = ["EnvironmentProfile", "detect_profile", "memory_mb", "cpu_count",
           "is_termux"]

#: profile name → (min, target, max) VCPU envelope
ENVELOPES: dict[str, tuple[int, int, int]] = {
    # Android/Termux: fork() is unreliable, RAM is precious — small pool
    "termux": (1, 2, 3),
    "mobile": (1, 2, 3),
    # ordinary laptop/desktop: solid but not endless
    "pc": (2, 4, 8),
    # small cloud box: 2-16 GB RAM
    "vps": (2, 4, 8),
    # bare-metal / big cloud: 32 GB+
    "workstation": (4, 8, 16),
    # Raspberry Pi class / very constrained
    "embedded": (1, 1, 2),
}

_KNOWN = set(ENVELOPES)


def cpu_count() -> int:
    try:
        return int(os.cpu_count() or 2)
    except (TypeError, ValueError):
        return 2


def memory_mb() -> int:
    """Total physical memory in MB, or 0 when unreadable."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(int(line.split()[1]) / 1024.0)
    except (OSError, ValueError, IndexError):  # noqa: E103 - memory probe falls through to next method
        pass
    try:  # macOS / anything else
        import subprocess
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True,
            text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            return int(int(out.stdout.strip()) / (1024.0 * 1024.0))
    except Exception:  # noqa: BLE001
        pass
    return 0


@dataclass(frozen=True)
class EnvironmentProfile:
    """A detected (or configured) environment and its VCPU envelope."""

    kind: str                      # termux|mobile|pc|vps|workstation|embedded
    min_vcpus: int
    target_vcpus: int
    max_vcpus: int
    cpu: int = 0
    memory_mb: int = 0
    detail: str = ""
    detected: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "min_vcpus": self.min_vcpus,
            "target_vcpus": self.target_vcpus,
            "max_vcpus": self.max_vcpus,
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "detail": self.detail,
            "detected": self.detected,
        }

    @classmethod
    def from_config(cls, kind: str) -> "EnvironmentProfile":
        """A profile pinned by configuration, not by detection."""
        kind = (kind or "").strip().lower()
        if kind not in _KNOWN:
            kind = "pc"
        lo, mid, hi = ENVELOPES[kind]
        return cls(kind=kind, min_vcpus=lo, target_vcpus=mid, max_vcpus=hi,
                   cpu=cpu_count(), memory_mb=memory_mb(),
                   detail=f"configured as {kind}", detected=False)


def _looks_like_termux(system: str, machine: str,
                       environ: dict[str, str]) -> bool:
    if system == "Android":
        return True
    if str(environ.get("TERMUX_VERSION") or ""):
        return True
    if "termux" in (sys.platform or "").lower():
        return True
    # a Termux build reports Linux/aarch64 with a termux path in sys.prefix
    if "termux" in (getattr(sys, "prefix", "") or "").lower():
        return True
    return False


def is_termux() -> bool:
    """True when this process is running inside Termux on Android.

    Public wrapper over :func:`_looks_like_termux` so other modules
    (e.g. the notifier's Termux-notification fallback) don't duplicate
    the detection heuristics.  Stdlib-only, never raises.
    """
    try:
        return bool(_looks_like_termux(_platform.system(), _platform.machine(),
                                       dict(os.environ)))
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def detect_profile(*, cpu: int | None = None, mem_mb: int | None = None,
                   system: str | None = None,
                   machine: str | None = None) -> EnvironmentProfile:
    """Inspect the environment and return the matching profile.

    Order of precedence: Android/Termux → embedded (very small) →
    workstation (≥32 GB) → vps (few cores, ≤16 GB) → pc (default).
    """
    ncpu = cpu if cpu is not None else cpu_count()
    mem = mem_mb if mem_mb is not None else memory_mb()
    system = system or _platform.system()
    machine = machine or _platform.machine() or ""
    env = dict(os.environ)

    if _looks_like_termux(system, machine, env):
        return EnvironmentProfile(
            kind="termux", **dict(zip(("min_vcpus", "target_vcpus", "max_vcpus"),
                                      ENVELOPES["termux"])),
            cpu=ncpu, memory_mb=mem,
            detail=f"Android/Termux ({machine or 'arm'})",
        )

    # very small boxes: ≤1 core or ≤1 GB of RAM
    if ncpu <= 1 or (0 < mem <= 1024):
        return EnvironmentProfile(
            kind="embedded", **dict(zip(("min_vcpus", "target_vcpus", "max_vcpus"),
                                        ENVELOPES["embedded"])),
            cpu=ncpu, memory_mb=mem,
            detail=f"constrained ({ncpu} cpu, {mem} MB)",
        )

    if mem >= 32 * 1024:
        return EnvironmentProfile(
            kind="workstation",
            **dict(zip(("min_vcpus", "target_vcpus", "max_vcpus"),
                       ENVELOPES["workstation"])),
            cpu=ncpu, memory_mb=mem,
            detail=f"{ncpu} cpu, {mem // 1024} GB RAM",
        )

    # small cloud boxes are usually 1-4 cores with ≤16 GB
    if ncpu <= 4 and mem <= 16 * 1024:
        return EnvironmentProfile(
            kind="vps", **dict(zip(("min_vcpus", "target_vcpus", "max_vcpus"),
                                   ENVELOPES["vps"])),
            cpu=ncpu, memory_mb=mem,
            detail=f"{ncpu} cpu, {mem // 1024} GB RAM (small box)",
        )

    return EnvironmentProfile(
        kind="pc", **dict(zip(("min_vcpus", "target_vcpus", "max_vcpus"),
                              ENVELOPES["pc"])),
        cpu=ncpu, memory_mb=mem,
        detail=f"{ncpu} cpu, {mem // 1024 if mem else '?'} GB RAM",
    )


def resolve_profile(kind: str = "", *, cpu: int | None = None,
                    mem_mb: int | None = None,
                    system: str | None = None,
                    machine: str | None = None) -> EnvironmentProfile:
    """Config first (``workspace.profile``), detection second.

    ``kind`` = "auto"/"" → detect; a known name → pin that envelope
    (still carrying the detected cpu/memory for reporting).
    """
    kind = (kind or "auto").strip().lower()
    if not kind or kind == "auto":
        return detect_profile(cpu=cpu, mem_mb=mem_mb,
                              system=system, machine=machine)
    return EnvironmentProfile.from_config(kind)


def known_profiles() -> list[str]:
    return sorted(_KNOWN)
