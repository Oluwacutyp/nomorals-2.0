"""Wave 84: the Workspace — the machine's virtual CPU farm.

* :mod:`~nomorals.core.profile` (moved from here; this module re-exports it) — environment detection
  (termux/mobile/pc/vps/workstation) with per-profile VCPU envelopes;
* :mod:`~nomorals.workspace.vcpu` — VirtualCPU: one independent
  execution engine (own queue, own context, status, pause/resume);
* :mod:`~nomorals.workspace.workspace` — Workspace: registry,
  affinity-aware dispatch, autoscaling, observability.
"""
from ..core.profile import (EnvironmentProfile, cpu_count, detect_profile,
                      known_profiles, memory_mb, resolve_profile)
from .vcpu import KINDS, VcpuStatus, VirtualCPU
from .workspace import Workspace, build_workspace

__all__ = [
    "EnvironmentProfile",
    "Workspace",
    "VirtualCPU",
    "VcpuStatus",
    "KINDS",
    "build_workspace",
    "cpu_count",
    "detect_profile",
    "known_profiles",
    "memory_mb",
    "resolve_profile",
]
