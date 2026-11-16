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
from ..core.logging_setup import get_logger
from .health import HealthMonitor
from .services import ServiceRegistry

_log = get_logger(__name__)

__all__ = ["OSKernel"]


class OSKernel:
    """Lifecycle owner for one Devon process."""

    def __init__(self, *, db_path: str | None = None,
                 bus: EventBus | None = None,
                 handle_signals: bool = False,
                 start_bus: bool = False) -> None:
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
        # The bus dispatcher is OFF by default (gap-14 point d): inline
        # delivery is the proven behavior, and starting the dispatcher
        # would change global event-delivery semantics. A caller that
        # wants the async dispatcher opts in explicitly.
        self._start_bus = bool(start_bus)
        # Supervision state (see supervise()): one episode per service
        # currently down, plus the capped restart ledger.
        self._supervise_thread: threading.Thread | None = None
        self._supervise_stop = threading.Event()
        self._heartbeat: Callable[[], Any] | None = None
        self._stop_requested = False
        self._restart_delays: tuple[float, ...] = self.DEFAULT_RESTART_DELAYS
        self._max_restarts = self.DEFAULT_MAX_RESTARTS
        self._episodes: dict[str, dict[str, Any]] = {}
        self._restarts: list[dict[str, Any]] = []
        self._restarts_total = 0
        if handle_signals:
            self.arm_signals()

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> "OSKernel":
        """Start the kernel.  Safe to call more than once."""
        with self._lock:
            if self._started:
                return self
            self._start_ts = time.time()
            # 1. Event bus dispatcher first — ONLY when opted in (the
            # wired default leaves inline delivery alone).
            if self._start_bus:
                self.bus.start()
            # 2. Start services in dependency order.
            self.services.start_all()
            # 3. Wire each service's health check into the monitor, with
            # the descriptor's role (liveness is restart-worthy,
            # readiness never restarts). Liveness checks need two
            # consecutive failures to declare down, so one slow tick
            # never triggers a restart.
            for name in self.services.list_services():
                check = self.services.health_check_for(name)
                if check is None:
                    continue
                try:
                    role = self.services.describe(name).health_role
                except Exception:  # noqa: BLE001 — describe never fails here
                    role = "readiness"
                check_name = self.health.register_check(
                    check, name=f"service:{name}", role=role,
                    interval_s=15.0 if role == "liveness" else 30.0,
                    failure_threshold=2 if role == "liveness" else 1)
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

        Order: supervisor thread → shutdown hooks (LIFO) → services
        (reverse dependency order) → health background loop → bus drain.
        Hooks get ``timeout_s`` total; a hanging hook is abandoned, not
        waited on forever.
        """
        with self._lock:
            if not self._started:
                return
            self._ready = False
            hooks = list(reversed(self._shutdown_hooks))
            self._started = False
        self._stop_supervision()
        deadline = time.time() + max(0.5, timeout_s)
        for name, hook in hooks:
            if time.time() >= deadline:
                break
            try:
                hook()
            except Exception:  # noqa: BLE001 — shutdown keeps moving
                _log.debug("kernel: shutdown hook %s failed", name,
                           exc_info=True)
        try:
            self.services.stop_all()
        except Exception:  # noqa: BLE001
            _log.debug("kernel: services stop_all failed", exc_info=True)
        try:
            self.health.stop_background()
        except Exception:  # noqa: BLE001
            _log.debug("kernel: health stop_background failed",
                       exc_info=True)
        # EventBus.stop() drains queued events before joining, so
        # async subscribers see everything published before stop().
        try:
            self.bus.stop()
        except Exception:  # noqa: BLE001
            _log.debug("kernel: bus stop failed", exc_info=True)

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

        CHAINED, never overwriting (gap-14 point c): the previous
        handler is captured and still invoked — the entry points'
        SIGTERM-as-interrupt handlers (final beacon,
        ``httpd.shutdown()``, the exit-130 contract) keep working
        because the previous handler still runs. The signal handler
        itself does no I/O: it requests a kernel stop through a
        thread-safe flag, and the supervisor thread (or the next
        :meth:`stop` call) performs the teardown on a normal thread.

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

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous = signal.getsignal(sig)
                if getattr(previous, "_devon_kernel_handler", False):
                    continue
                # Bind this signal's previous handler into the closure
                # without sharing one dict across signals.
                _prev: dict[int, Any] = {sig: previous}

                def _bound(signum: int, _frame: Any,
                           _prev: dict[int, Any] = _prev) -> None:
                    self._stop_requested = True
                    prev = _prev.get(signum)
                    if callable(prev):
                        prev(signum, _frame)

                _bound._devon_kernel_handler = True  # type: ignore[attr-defined]
                signal.signal(sig, _bound)
        except (ValueError, OSError, RuntimeError) as exc:
            # Not the main thread (or no signal support): atexit still covers
            # normal interpreter shutdown.
            _log.debug("kernel.arm_signals failed", exc_info=True)
        return self

    # ── supervision ──────────────────────────────────────────────────────
    #: Default restart budget: exponential backoff, 5 attempts per
    #: episode, then the service stays down until a manual start.
    DEFAULT_RESTART_DELAYS: tuple[float, ...] = (5.0, 15.0, 45.0, 120.0, 300.0)
    DEFAULT_MAX_RESTARTS = 5

    def supervise(self, *, interval_s: float = 10.0,
                  restart_delays: tuple[float, ...] | None = None,
                  max_restarts: int = DEFAULT_MAX_RESTARTS,
                  heartbeat: Callable[[], Any] | None = None) -> OSKernel:
        """Start the supervisor thread: watch liveness, restart the dead.

        Each cycle runs every health check once, calls ``heartbeat``
        (the boot site passes the runtime's beacon tick, so the status
        beacon goes live on the supervisor's cadence), and restarts any
        liveness-role service whose check newly declares down — through
        the registry's ``restart()``, i.e. the service's own stop/start
        path, never a re-implementation. Restarts use exponential
        backoff and stop after ``max_restarts`` attempts per episode;
        the service then stays down, loudly journaled, until a manual
        start. Services marked ``restartable=False`` are watched and
        journaled but never restarted. Idempotent.
        """
        with self._lock:
            if (self._supervise_thread is not None
                    and self._supervise_thread.is_alive()):
                return self
            self._restart_delays = tuple(
                restart_delays if restart_delays is not None
                else self.DEFAULT_RESTART_DELAYS)
            self._max_restarts = max(0, int(max_restarts))
            if heartbeat is not None:
                self._heartbeat = heartbeat
            self._supervise_stop.clear()
            self._supervise_thread = threading.Thread(
                target=self._supervise_loop, args=(max(0.5, interval_s),),
                daemon=True, name="kernel-supervisor")
            self._supervise_thread.start()
        return self

    @property
    def supervising(self) -> bool:
        with self._lock:
            thread = self._supervise_thread
        return thread is not None and thread.is_alive()

    def _stop_supervision(self) -> None:
        # Defensive getattr: stop() is pinned to survive a
        # partially-constructed kernel (see S12: built via __new__
        # without __init__), so missing supervisor state is a no-op,
        # never an exception escaping stop().
        thread = getattr(self, "_supervise_thread", None)
        stop_event = getattr(self, "_supervise_stop", None)
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=10.0)

    def _supervise_loop(self, interval_s: float) -> None:
        while not self._supervise_stop.is_set():
            if self._heartbeat is not None:
                try:
                    self._heartbeat()
                except Exception:  # noqa: BLE001 — heartbeat never kills supervision
                    _log.debug("kernel heartbeat failed", exc_info=True)
            try:
                self.health.check_all()
            except Exception:  # noqa: BLE001 — a failed sweep never kills supervision
                _log.debug("kernel health sweep failed", exc_info=True)
            try:
                self._supervise_once()
            except Exception:  # noqa: BLE001 — supervision keeps watching
                _log.debug("kernel supervise cycle failed", exc_info=True)
            self._supervise_stop.wait(interval_s)

    def _supervise_once(self) -> None:
        """One supervision pass: restart newly-down liveness services."""
        now = time.time()
        for name in self.services.list_services():
            try:
                descriptor = self.services.describe(name)
            except Exception:  # noqa: BLE001 — a missing descriptor is skipped
                continue
            if descriptor.health_role != "liveness":
                continue
            check_name = f"service:{name}"
            if self.health.is_down(check_name):
                self._handle_down(name, now)
            else:
                episode = self._episodes.get(name)
                if episode is not None:
                    last = self.health.last_status(check_name)
                    if last is not None and last.ok:
                        _log.info("kernel: service %r recovered", name)
                        del self._episodes[name]

    def _handle_down(self, name: str, now: float) -> None:
        """Restart a down service within its episode budget."""
        with self._lock:
            episode = self._episodes.get(name)
            if episode is None:
                episode = {"attempts": 0, "next_attempt_at": now,
                           "gave_up": False}
                self._episodes[name] = episode
                self._journal_death(name)
            if episode["gave_up"]:
                return
            if now < episode["next_attempt_at"]:
                return
            if episode["attempts"] >= self._max_restarts:
                episode["gave_up"] = True
                _log.error(
                    "kernel: service %r gave up after %d restart attempts; "
                    "manual start required", name, episode["attempts"])
                self._journal_gave_up(name, episode["attempts"])
                return
            attempt = episode["attempts"] + 1
        try:
            self.services.restart(name)
            ok, error = True, ""
        except Exception as exc:  # noqa: BLE001 — a failed restart is an attempt, not a crash
            ok, error = False, f"{type(exc).__name__}: {exc}"
        with self._lock:
            episode = self._episodes.get(name)
            if episode is None:
                return
            episode["attempts"] = attempt
            delay = self._restart_delays[
                min(attempt - 1, len(self._restart_delays) - 1)]
            episode["next_attempt_at"] = time.time() + delay
            self._record_restart(name, attempt, ok, error)
        if ok:
            _log.warning("kernel: restarted service %r (attempt %d)",
                         name, attempt)
        else:
            _log.warning("kernel: restart of service %r failed "
                         "(attempt %d): %s", name, attempt, error)

    def _journal_death(self, name: str) -> None:
        """Record a service death in the durable failure ledger."""
        try:
            from ..console.failures import record_failure
        except Exception:  # noqa: BLE001 — journaling never breaks supervision
            _log.warning("kernel: service %r is down (ledger unavailable)",
                         name)
            return
        try:
            detail = ""
            status = self.health.last_status(f"service:{name}")
            if status is not None:
                detail = status.detail
            record_failure("kernel.supervise." + name,
                           RuntimeError(f"service {name!r} down: {detail}"))
        except Exception:  # noqa: BLE001 — journaling never breaks supervision
            _log.debug("kernel: failure ledger write failed", exc_info=True)

    def _journal_gave_up(self, name: str, attempts: int) -> None:
        try:
            from ..console.failures import record_failure
        except Exception:  # noqa: BLE001
            return
        try:
            record_failure(
                "kernel.supervise." + name,
                RuntimeError(f"service {name!r} gave up after "
                             f"{attempts} restart attempts"))
        except Exception:  # noqa: BLE001
            _log.debug("kernel: failure ledger write failed", exc_info=True)

    def _record_restart(self, name: str, attempt: int, ok: bool,
                        error: str) -> None:
        entry = {"service": name, "attempt": attempt, "ok": ok,
                 "error": error, "ts": time.time()}
        self._restarts.append(entry)
        self._restarts_total += 1
        del self._restarts[:-50]

    def supervision_summary(self) -> dict[str, Any]:
        """Supervision state for the beacon / status surfaces."""
        with self._lock:
            restarts = list(self._restarts)
            total = self._restarts_total
            gave_up = {n: {"attempts": e["attempts"]}
                       for n, e in self._episodes.items() if e["gave_up"]}
            down = sorted(
                n for n in self.services.list_services()
                if self.health.is_down(f"service:{n}"))
        return {
            "started": self.started,
            "ready": self.ready,
            "supervising": self.supervising,
            "services": self.services.list_services(),
            "down": down,
            "restarts": restarts[-10:],
            "restarts_total": total,
            "gave_up": gave_up,
        }

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
        """Block until the kernel is ready or the timeout expires.

        Returns True only when the kernel is actually ready (its own
        flag AND the health summary's readiness over the real checks).
        A timeout with an unready service returns False — returning
        ``self.ready`` here would claim readiness that the health
        checks contradict.
        """
        deadline = time.time() + max(0.0, timeout_s)
        while time.time() < deadline:
            if self.ready and self.health.summary()["ready"]:
                return True
            time.sleep(0.05)
        return self.ready and self.health.summary()["ready"]

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
            "supervising": self.supervising,
            "restarts": list(self._restarts[-10:]),
            "restarts_total": self._restarts_total,
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
