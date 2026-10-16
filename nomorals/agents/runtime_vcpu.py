"""Agents runtime pointed at the VCPU farm (Tier 0 wiring).

:mod:`nomorals.agents.runtime.HybridExecutor` never imported the VCPU
farm — it sized its own pools from ``os.cpu_count()`` (currently
``max(2, cpu - 1)`` for processes; the old ``max(1, cpu // 2)`` formula
the task brief cites was already replaced). This adapter is the bridge:
agent work submitted here runs on :class:`nomorals.workspace.Workspace`
(the VirtualCPU farm) with affinity-aware dispatch, pause/resume, and
worker counts taken from the detected environment profile's envelope —
never from a raw ``cpu // 2`` guess.

Fallback is honest, never fake: when the farm cannot be built (import
failure, construction error), the adapter runs on a local
:class:`~concurrent.futures.ThreadPoolExecutor`, reports
``using_farm == False`` with the real reason, and never claims farm
semantics (pause/resume report unsupported on the fallback path).
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.profile import resolve_profile
from ..tier0_rollback import record_promotion
from ..workspace.vcpu import KINDS

__all__ = ["RuntimeVCPUAdapter", "register_promotion", "promotion_active"]

_log = get_logger(__name__)

PROMOTION_NAME = "runtime_vcpu"

_promotion_active = False


def promotion_active() -> bool:
    """True while the runtime→VCPU promotion is active (not rolled back)."""
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
        description="agents runtime submits to the VCPU farm (Workspace)",
        metadata={"module": __name__},
    )


class RuntimeVCPUAdapter:
    """Submit agent work to the VCPU farm, honouring the profile envelope."""

    def __init__(
        self,
        *,
        profile: str = "",
        max_workers: int | None = None,
        farm: Any | None = None,
        farm_factory: Callable[..., Any] | None = None,
        autoscale: bool = False,
    ) -> None:
        """Build the adapter.

        ``profile`` pins an environment profile (``"termux"``, ``"pc"``,
        ...; ``""`` auto-detects). ``max_workers`` caps the farm size and
        is clamped into the profile envelope — a Termux envelope caps at
        3 VCPUs no matter what is requested. ``farm`` injects an
        already-built Workspace (the adapter does not shut it down);
        ``farm_factory`` overrides how a farm is built (test seam — a
        factory that raises exercises the honest fallback path).
        """
        self.profile = resolve_profile(profile)
        envelope_max = self.profile.max_vcpus
        envelope_min = self.profile.min_vcpus
        if max_workers is None:
            effective = self.profile.target_vcpus
        else:
            effective = max(envelope_min, min(int(max_workers), envelope_max))
        self.max_workers = effective
        self.envelope = (
            envelope_min, self.profile.target_vcpus, envelope_max)
        self._owns_farm = farm is None
        self._closed = False
        self._fallback_paused = False
        self._lock = threading.Lock()
        self.stats = {"submitted": 0, "farm_tasks": 0, "fallback_tasks": 0}

        self._farm: Any = None
        self._fallback: cf.ThreadPoolExecutor | None = None
        self.fallback_reason = ""
        if farm is not None:
            self._farm = farm
        else:
            self._farm = self._build_farm(farm_factory, effective)
        if self._farm is None:
            self._fallback = cf.ThreadPoolExecutor(
                max_workers=max(1, effective),
                thread_name_prefix="nm-runtime-vcpu-fallback",
            )
            _log.warning(
                "RuntimeVCPUAdapter: VCPU farm unavailable (%s); "
                "using local thread-pool fallback — NOT the farm",
                self.fallback_reason or "unknown")

    def _build_farm(
        self,
        farm_factory: Callable[..., Any] | None,
        effective: int,
    ) -> Any | None:
        try:
            if farm_factory is not None:
                return farm_factory(
                    profile=self.profile.kind,
                    vcpus=effective,
                    min_vcpus=effective,
                    max_vcpus=effective,
                    autoscale=False,
                )
            from ..workspace.workspace import Workspace
            return Workspace(
                profile=self.profile.kind,
                vcpus=effective,
                min_vcpus=effective,
                max_vcpus=effective,
                autoscale=False,
            )
        except Exception as exc:  # noqa: BLE001 - fallback, never die
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            return None

    # ── identity ─────────────────────────────────────────────────────────────
    @property
    def using_farm(self) -> bool:
        """True only when submissions really run on the VCPU farm."""
        return self._farm is not None

    @property
    def backend(self) -> str:
        return "vcpu-farm" if self.using_farm else "thread-pool-fallback"

    @property
    def worker_count(self) -> int:
        """Live workers, always within the profile envelope."""
        if self._farm is not None:
            try:
                return len(self._farm.vcpus())
            except Exception:  # noqa: BLE001 - report, never raise
                return 0
        return max(1, self.max_workers)

    # ── submission ───────────────────────────────────────────────────────────
    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        affinity: str = "balanced",
        priority: int = 0,
        name: str = "",
        timeout: float = 0.0,
        vcpu: str = "",
        **kwargs: Any,
    ) -> cf.Future:
        """Submit ``fn(*args, **kwargs)``. Returns a standard Future.

        ``affinity`` (``"io"`` | ``"cpu"`` | ``"balanced"``) steers the
        farm's dispatch; anything else raises — silently ignoring a bad
        affinity would misroute work. On the farm path the return is the
        VCPU's Future; on the fallback path a thread-pool Future.
        """
        if self._closed:
            raise RuntimeError("RuntimeVCPUAdapter is shut down")
        if affinity not in KINDS:
            raise ValueError(
                f"unknown affinity {affinity!r}; expected {KINDS}")
        with self._lock:
            self.stats["submitted"] += 1
        if self._farm is not None:
            _, future = self._farm.submit(
                fn, *args, affinity=affinity, priority=priority,
                vcpu=vcpu, name=name or getattr(fn, "__name__", ""),
                timeout=timeout, **kwargs,
            )
            with self._lock:
                self.stats["farm_tasks"] += 1
            return future
        assert self._fallback is not None  # noqa: S101 - set in __init__
        future = self._fallback.submit(fn, *args, **kwargs)
        with self._lock:
            self.stats["fallback_tasks"] += 1
        return future

    def submit_named(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, cf.Future]:
        """Like :meth:`submit` but also returns the VCPU that took the work.

        On the fallback path the name is ``"fallback"`` — never a fake
        VCPU name.
        """
        if self._farm is not None:
            # Reuse submit() for validation/stats, then find the VCPU by
            # re-dispatching is wasteful — so dispatch directly here.
            affinity = kwargs.pop("affinity", "balanced")
            priority = kwargs.pop("priority", 0)
            name = kwargs.pop("name", "")
            timeout = kwargs.pop("timeout", 0.0)
            vcpu = kwargs.pop("vcpu", "")
            if affinity not in KINDS:
                raise ValueError(
                    f"unknown affinity {affinity!r}; expected {KINDS}")
            with self._lock:
                self.stats["submitted"] += 1
                self.stats["farm_tasks"] += 1
            vcpu_name, future = self._farm.submit(
                fn, *args, affinity=affinity, priority=priority,
                vcpu=vcpu, name=name or getattr(fn, "__name__", ""),
                timeout=timeout, **kwargs,
            )
            return vcpu_name, future
        return "fallback", self.submit(fn, *args, **kwargs)

    def map(
        self,
        fn: Callable[[Any], Any],
        items: list[Any],
        *,
        affinity: str = "balanced",
        priority: int = 0,
    ) -> list[cf.Future]:
        """Submit ``fn(x)`` for every item. Returns Futures in order."""
        return [self.submit(fn, x, affinity=affinity, priority=priority)
                for x in items]

    # ── pause / resume ───────────────────────────────────────────────────────
    def pause(self) -> int:
        """Pause every farm VCPU (queues keep accepting work).

        Returns how many VCPUs paused. On the fallback path pausing is
        unsupported — returns 0 and records the flag; submissions keep
        running rather than silently queueing forever.
        """
        if self._farm is not None:
            return int(self._farm.pause_all())
        self._fallback_paused = True
        return 0

    def resume(self) -> int:
        """Resume every farm VCPU. Returns how many resumed."""
        if self._farm is not None:
            return int(self._farm.resume_all())
        self._fallback_paused = False
        return 0

    def pause_vcpu(self, name: str) -> bool:
        """Pause one VCPU by name. False when unknown or on fallback."""
        if self._farm is None:
            return False
        vcpu = self._farm.get(name)
        if vcpu is None:
            return False
        vcpu.pause()
        return True

    def resume_vcpu(self, name: str) -> bool:
        """Resume one VCPU by name. False when unknown or on fallback."""
        if self._farm is None:
            return False
        vcpu = self._farm.get(name)
        if vcpu is None:
            return False
        vcpu.resume()
        return True

    # ── lifecycle / reporting ────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        farm_status: dict[str, Any] = {}
        if self._farm is not None:
            try:
                farm_status = self._farm.status()
            except Exception:  # noqa: BLE001 - report, never raise
                farm_status = {"error": "farm status unavailable"}
        return {
            "backend": self.backend,
            "using_farm": self.using_farm,
            "profile": self.profile.kind,
            "envelope": {
                "min": self.envelope[0],
                "target": self.envelope[1],
                "max": self.envelope[2],
            },
            "max_workers": self.max_workers,
            "worker_count": self.worker_count,
            "fallback_reason": self.fallback_reason or None,
            "fallback_paused": self._fallback_paused,
            "stats": dict(self.stats),
            "farm": farm_status,
        }

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut down. Only stops a farm this adapter owns (built itself);
        an injected farm is left running — its owner manages it."""
        if self._closed:
            return
        self._closed = True
        if self._farm is not None and self._owns_farm:
            try:
                for vcpu in self._farm.vcpus():
                    vcpu.stop(drain=True, timeout=5.0)
            except Exception:  # noqa: BLE001 - shutdown never raises
                _log.debug("farm shutdown error", exc_info=True)
        if self._fallback is not None:
            self._fallback.shutdown(wait=wait)

    def __enter__(self) -> "RuntimeVCPUAdapter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


register_promotion()
