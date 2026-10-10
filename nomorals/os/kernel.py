"""OS kernel: process lifecycle for the Devon control plane.

:class:`OSKernel` owns the four things every Devon process needs:

* the global :mod:`~nomorals.core.events` bus dispatcher thread,
* the :class:`~nomorals.os.services.ServiceRegistry`,
* the :class:`~nomorals.os.health.HealthMonitor`,
* the shutdown path: ordered teardown, shutdown hooks, signal wiring.

``start()`` is idempotent: it starts the bus dispatcher, starts services
in dependency order, wires each registered service's health check into
the monitor, and runs the checks once so a broken service is visible
immediately.  ``stop()`` runs shutdown hooks (LIFO), stops services in
reverse dependency order, drains the bus, and stops background threads;
it is idempotent too.

Call :meth:`arm_signals` (or pass ``handle_signals=True``) so SIGTERM /
SIGINT trigger the same graceful shutdown — a Ctrl-C stops the world
cleanly instead of abandoning it mid-write.
"""

from __future__ import annotations

import atexit
import signal
import threading
import time
from typing import Any, Callable

from ..core.events import EventBus, global_bus
from .health import HealthMonitor
from .services import ServiceRegistry

__all__ = ["OSKernel"]


class OSKernel:
    """Lifecycle owner for one Devon process."""

    def __init__(self, *, db_path: str | None = None,
                 bus: EventBus | None = None,
                 handle_signals: bool = False) -> None:
        self.bus: EventBus = bus if bus is not None else global_bus
        self.services = ServiceRegistry(db_path=db_path)
        self.health = HealthMonitor()
        self._started = False
        self._ready = False
        self._lock = threading.RLock()
        self._wired_checks: list[str] = []
        self._shutdown_hooks: list[tuple[str, Callable[[], Any]]] = []
        self._signals_armed = False
        self._atexit_registered = False
        self._start_ts: float | None = None
        if handle_signals:
            self.arm_signals()

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> "OSKernel":
        """Start the kernel.  Safe to call more than once."""
        with self._lock:
            if self._started:
                return self
            self._start_ts = time.time()
            # 1. Event bus dispatcher first: everything else publishes here.
            self.bus.start()
            # 2. Start services in dependency order.
            self.services.start_all()
            # 3. Wire each service's health check into the monitor.
            for name in self.services.list_services():
                check = self.services.health_check_for(name)
                if check is not None:
                    check_name = self.health.register_check(
                        check, name=f"service:{name}", role="readiness")
                    self._wired_checks.append(check_name)
            # 4. Run every check once; recovery hooks fire for failures.
            self.health.check_all()
            # 5. Background health loop so staleness is caught at runtime.
            self.health.start_background()
            self._started = True
            self._ready = True
            return self

    def stop(self, *, timeout_s: float = 10.0) -> None:
        """Graceful shutdown.  Idempotent.

        Order: shutdown hooks (LIFO) → services (reverse dependency order)
        → health background loop → bus drain.  Hooks get ``timeout_s``
        total; a hanging hook is abandoned, not waited on forever.
        """
        with self._lock:
            if not self._started:
                return
            self._ready = False
            hooks = list(reversed(self._shutdown_hooks))
            self._started = False
        deadline = time.time() + max(0.5, timeout_s)
        for name, hook in hooks:
            if time.time() >= deadline:
                break
            try:
                hook()
            except Exception:  # noqa: BLE001 — shutdown keeps moving
                pass
        try:
            self.services.stop_all()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.health.stop_background()
        except Exception:  # noqa: BLE001
            pass
        # EventBus.stop() drains queued events before joining, so
        # async subscribers see everything published before stop().
        try:
            self.bus.stop()
        except Exception:  # noqa: BLE001
            pass

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def ready(self) -> bool:
        """True between a successful start and the beginning of stop."""
        with self._lock:
            return self._ready

    @property
    def uptime_s(self) -> float | None:
        with self._lock:
            if self._start_ts is None:
                return None
            return time.time() - self._start_ts

    # ── shutdown plumbing ────────────────────────────────────────────────
    def on_shutdown(self, hook: Callable[[], Any], *,
                    name: str = "") -> Callable[[], Any]:
        """Register a shutdown hook (runs LIFO at :meth:`stop`)."""
        with self._lock:
            self._shutdown_hooks.append(
                (name or getattr(hook, "__name__", "hook"), hook))
        return hook

    def arm_signals(self) -> "OSKernel":
        """SIGTERM/SIGINT → graceful :meth:`stop`.  Also registers atexit.

        Safe to call twice; only works on the main thread (signal
        restriction) — elsewhere it just arms atexit.
        """
        with self._lock:
            if self._signals_armed:
                return self
            self._signals_armed = True
            if not self._atexit_registered:
                atexit.register(self._atexit_stop)
                self._atexit_registered = True

        def _handler(signum: int, _frame: Any) -> None:
            self.stop()

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError, RuntimeError):
            # Not the main thread (or no signal support): atexit still covers
            # normal interpreter shutdown.
            pass
        return self

    def _atexit_stop(self) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001 — never fail interpreter shutdown
            pass

    # ── convenience ──────────────────────────────────────────────────────
    def service(self, name: str) -> Any:
        """Shortcut for ``self.services.lookup(name)``."""
        return self.services.lookup(name)

    def restart_service(self, name: str) -> Any:
        """Recycle one service without touching the rest of the kernel."""
        return self.services.restart(name)

    def wait_ready(self, timeout_s: float = 30.0) -> bool:
        """Block until the kernel is ready or the timeout expires."""
        deadline = time.time() + max(0.0, timeout_s)
        while time.time() < deadline:
            if self.ready and self.health.summary()["ready"]:
                return True
            time.sleep(0.05)
        return self.ready

    def status(self) -> dict[str, Any]:
        """Snapshot: started flag, services, last health results."""
        last = {name: (s.to_dict() if s else None)
                for name, s in ((n, self.health.last_status(n))
                                for n in self.health.check_names())}
        return {
            "started": self.started,
            "ready": self.ready,
            "uptime_s": self.uptime_s,
            "services": self.services.list_services(),
            "service_details": self.services.describe_all(),
            "health": last,
            "health_summary": self.health.summary(),
        }

    def render(self) -> str:
        """Plain-text kernel dashboard."""
        st = self.status()
        uptime = st["uptime_s"]
        uptime_s = (f"{uptime:.0f}s" if uptime is not None else "?")
        lines = [
            f"devon kernel — {'READY' if st['ready'] else 'STOPPED'}"
            f" (uptime {uptime_s})",
            self.services.render(),
            self.health.render(),
        ]
        return "\n".join(lines)

    def __enter__(self) -> "OSKernel":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
