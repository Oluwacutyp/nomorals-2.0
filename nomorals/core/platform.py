"""Platform backend: what this machine can do, as data.

Instead of scattering ``platform.system() == "Android"`` checks through the
codebase, every capability question goes through :func:`detect_platform`::

    from nomorals.core.platform import detect_platform

    plat = detect_platform()
    if plat.supports("process_pool"):
        ...
    workers = plat.capabilities.max_workers

Termux is a first-class platform here, not an afterthought: the phone is a
deployment target, so its limits (no process pool, no GPU, no background
service without a wake lock) are declared once and consulted everywhere.
"""

from __future__ import annotations

import os
import platform as _stdlib_platform
from dataclasses import dataclass, field
from typing import Any

from .profile import detect_profile


@dataclass(frozen=True)
class PlatformCapabilities:
    """Concrete limits of a platform.  Anything not listed is assumed fine."""

    process_pool: bool = True      # multiprocessing pool usable
    gpu: bool = False              # GPU compute available
    max_workers: int = 8           # sane default thread/process count
    storage_backend: str = "sqlite"
    background_service: bool = True   # can run unattended in background
    wake_lock: bool = False           # needs an explicit wake lock to stay alive
    low_memory: bool = False          # < 2 GB class device; keep buffers small


@dataclass(frozen=True)
class Platform:
    """A named platform with its capabilities."""

    name: str                       # linux | termux | android | windows | macos
    capabilities: PlatformCapabilities = field(default_factory=PlatformCapabilities)
    detail: str = ""

    def supports(self, capability: str) -> bool:
        """Answer ``platform.supports("process_pool")`` style questions.

        Unknown capability names return True (assume fine) — the backend
        declares *limits*, not permissions.
        """
        return bool(getattr(self.capabilities, capability, True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detail": self.detail,
            "capabilities": {
                "process_pool": self.capabilities.process_pool,
                "gpu": self.capabilities.gpu,
                "max_workers": self.capabilities.max_workers,
                "storage_backend": self.capabilities.storage_backend,
                "background_service": self.capabilities.background_service,
                "wake_lock": self.capabilities.wake_lock,
                "low_memory": self.capabilities.low_memory,
            },
        }


def _from_profile_kind(kind: str, detail: str) -> Platform:
    kind = (kind or "").lower()
    if kind == "termux":
        return Platform(
            name="termux",
            capabilities=PlatformCapabilities(
                process_pool=False,   # fork() unreliable under Termux/Android
                gpu=False,
                max_workers=4,
                storage_backend="sqlite",
                background_service=False,  # needs wake lock + foreground
                wake_lock=True,
                low_memory=True,
            ),
            detail=detail,
        )
    if kind in {"mobile", "embedded"}:
        return Platform(
            name="android" if _stdlib_platform.system() == "Android" else kind,
            capabilities=PlatformCapabilities(
                process_pool=False,
                gpu=False,
                max_workers=4,
                storage_backend="sqlite",
                background_service=False,
                wake_lock=True,
                low_memory=True,
            ),
            detail=detail,
        )
    system = _stdlib_platform.system()
    if system == "Windows":
        return Platform(
            name="windows",
            capabilities=PlatformCapabilities(
                process_pool=True,    # spawn only; slower but works
                max_workers=max(4, (os.cpu_count() or 4)),
                detail=detail,
            ),
        )
    if system == "Darwin":
        return Platform(
            name="macos",
            capabilities=PlatformCapabilities(
                process_pool=True,    # spawn default since 3.8
                max_workers=max(4, (os.cpu_count() or 4)),
            ),
            detail=detail,
        )
    return Platform(
        name="linux",
        capabilities=PlatformCapabilities(
            process_pool=True,
            max_workers=max(4, (os.cpu_count() or 4)),
        ),
        detail=detail,
    )


_cached: Platform | None = None


def detect_platform(*, profile: Any = None, refresh: bool = False) -> Platform:
    """Detect the current platform backend (cached per process)."""
    global _cached
    if _cached is not None and not refresh:
        return _cached
    prof = profile or detect_profile()
    kind = getattr(prof, "kind", "pc") or "pc"
    detail = getattr(prof, "detail", "") or ""
    # A Termux userspace on top of Android reports Linux — the profile
    # detector already knows the difference; trust it over uname.
    if _stdlib_platform.system() == "Android" and kind not in {"termux", "mobile", "embedded"}:
        kind = "android"
    plat = _from_profile_kind(kind, detail)
    _cached = plat
    return plat


def reset_platform_cache() -> None:
    """Forget the cached detection (tests)."""
    global _cached
    _cached = None
