"""OS kernel: process lifecycle for the Devon control plane.

:class:`OSKernel` owns the three things every Devon process needs:

* the global :mod:`~nomorals.core.events` bus dispatcher thread,
* the :class:`~nomorals.os.services.ServiceRegistry`,
* the :class:`~nomorals.os.health.HealthMonitor`.

``start()`` is idempotent: it starts the bus dispatcher, wires each
registered service's health check into the monitor, and runs the checks
once so a broken service is visible immediately.  ``stop()`` drains the bus
and stops background threads; it is idempotent too.
"""

from __future__ import annotations

import threading
from typing import Any

from ..core.events import EventBus, global_bus
from .health import HealthMonitor
from .services import ServiceRegistry

__all__ = ["OSKernel"]


class OSKernel:
    """Lifecycle owner for one Devon process."""

    def __init__(self, *, db_path: str | None = None,
                 bus: EventBus | None = None) -> None:
        self.bus: EventBus = bus if bus is not None else global_bus
        self.services = ServiceRegistry(db_path=db_path)
        self.health = HealthMonitor()
        self._started = False
        self._lock = threading.RLock()
        self._wired_checks: list[str] = []

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> "OSKernel":
        """Start the kernel.  Safe to call more than once."""
        with self._lock:
            if self._started:
                return self
            # 1. Event bus dispatcher first: everything else publishes here.
            self.bus.start()
            # 2. Wire each service's health check into the monitor.
            for name in self.services.list_services():
                check = self.services.health_check_for(name)
                if check is not None:
                    check_name = self.health.register_check(check)
                    self._wired_checks.append(check_name)
            # 3. Run every check once; recovery hooks fire for failures.
            self.health.check_all()
            self._started = True
            return self

    def stop(self) -> None:
        """Drain the bus and stop background threads.  Idempotent."""
        with self._lock:
            if not self._started:
                return
            # EventBus.stop() drains queued events before joining, so
            # async subscribers see everything published before stop().
            self.bus.stop()
            self._started = False

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    # ── convenience ──────────────────────────────────────────────────────
    def service(self, name: str) -> Any:
        """Shortcut for ``self.services.lookup(name)``."""
        return self.services.lookup(name)

    def status(self) -> dict[str, Any]:
        """Snapshot: started flag, services, last health results."""
        last = {name: (s.to_dict() if s else None)
                for name, s in ((n, self.health.last_status(n))
                                for n in self.health.check_names())}
        return {
            "started": self.started,
            "services": self.services.list_services(),
            "health": last,
        }

    def __enter__(self) -> "OSKernel":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
