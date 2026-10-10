"""Graceful shutdown: ordered lifecycle hooks, signal handling, readiness.

Two layers:

1. :func:`install_sigterm_as_interrupt` — the original one-trick mapping of
   SIGTERM onto KeyboardInterrupt. Kept byte-for-byte for the entry points
   that already shut down cleanly on Ctrl-C (the CLI's top-level
   ``except KeyboardInterrupt``, ``APIServer.serve()``'s finally block,
   ``PartnerRuntime.run()``'s finally block).

2. :class:`ShutdownCoordinator` — the full lifecycle. Hooks register with a
   priority and a per-hook timeout; startup runs them in *ascending* priority
   order and shutdown in *descending* order (Spring SmartLifecycle
   semantics: the thing that started first stops last). The shutdown
   sequence is "no new work, then finish work, then clean up": requesting
   shutdown first flips :attr:`ready` to False (readiness probes fail from
   that instant so load balancers stop sending traffic), then hooks run in
   order, each with its own timeout — a straggler is recorded as a timeout
   and shutdown moves on instead of blocking a deploy forever. The run is
   idempotent: a second SIGTERM returns the first report instead of
   re-running hooks.

Signal handlers do exactly one thing — record the request. All real work
happens in :meth:`ShutdownCoordinator.run`, on the calling thread.

Stdlib only, no package imports at module level beyond logging — this is
L1 core and must stay importable from anywhere.
"""

from __future__ import annotations

import asyncio
import inspect
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .logging_setup import get_logger

__all__ = [
    "install_sigterm_as_interrupt",
    "ShutdownCoordinator",
    "ShutdownHook",
    "get_coordinator",
    "reset_coordinator",
]

_log = get_logger(__name__)


def install_sigterm_as_interrupt() -> bool:
    """Map SIGTERM to KeyboardInterrupt in the main thread.

    Returns True when the handler was installed. Safe to call from any
    thread — signal handlers can only be installed on the main thread, so
    elsewhere this is a no-op returning False — and on platforms without
    SIGTERM. Idempotent: calling twice keeps a single handler.

    Every long-running entry point already shuts down cleanly on
    KeyboardInterrupt (the CLI's top-level ``except KeyboardInterrupt``,
    ``APIServer.serve()``'s finally block, ``PartnerRuntime.run()``'s
    finally block), so this one mapping gives SIGTERM the same graceful
    treatment: context teardown runs, the event bus drains, the gateway
    stops, and the final status beacon is written with ``stopped: true``.
    """
    try:
        sigterm = signal.SIGTERM
    except AttributeError:  # pragma: no cover - Windows has no SIGTERM
        return False
    previous = signal.getsignal(sigterm)

    def _sigterm_handler(signum: int, frame: object) -> None:  # noqa: ARG001
        raise KeyboardInterrupt()

    # Mark our own closure so a repeat install stays silent.
    _sigterm_handler._devon_sigterm_handler = True  # type: ignore[attr-defined]
    if getattr(previous, "_devon_sigterm_handler", False):
        return True
    try:
        signal.signal(sigterm, _sigterm_handler)
    except (OSError, ValueError, RuntimeError) as exc:
        # ValueError: not the main thread. OSError/RuntimeError: the
        # interpreter refuses (e.g. embedded). Either way, degrade to the
        # default SIGTERM behavior rather than breaking the boot.
        _log.debug("SIGTERM handler not installed: %s", exc)
        return False
    if previous not in (signal.SIG_DFL, None):
        _log.debug("replaced existing SIGTERM handler %r", previous)
    return True


@dataclass
class ShutdownHook:
    """One lifecycle hook: ``fn(reason)`` or ``fn()``, sync or async."""

    name: str
    fn: Callable[..., Any]
    priority: int = 0
    timeout_s: float = 10.0
    phase: str = "shutdown"  # "startup" | "shutdown"
    takes_reason: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.phase not in ("startup", "shutdown"):
            raise ValueError(f"phase must be 'startup'|'shutdown', got {self.phase!r}")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        try:
            params = inspect.signature(self.fn).parameters
            kinds = [p.kind for p in params.values()]
            positional = [k for k in kinds if k in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            self.takes_reason = bool(positional) or any(
                k == inspect.Parameter.VAR_POSITIONAL for k in kinds)
        except (TypeError, ValueError):
            self.takes_reason = False


class ShutdownCoordinator:
    """Ordered, timeout-bounded application lifecycle.

    Suggested priority bands (higher runs first on shutdown):

    * 100 — stop accepting new work (close listeners, pause schedulers)
    *  50 — drain in-flight work (finish requests, flush queues)
    *   0 — close resources (DB pools, HTTP sessions, file handles)
    * -100 — final beacon / last log line
    """

    def __init__(self, *, name: str = "nomorals",
                 default_hook_timeout_s: float = 10.0) -> None:
        self.name = name
        self.default_hook_timeout_s = default_hook_timeout_s
        self._hooks: list[ShutdownHook] = []
        self._lock = threading.RLock()
        self._shutdown_requested = False
        self._shutdown_reason = ""
        self._shutdown_started_at = 0.0
        self._shutdown_report: dict[str, Any] | None = None
        self._startup_report: dict[str, Any] | None = None
        self._booted_at = time.time()
        self._installed: tuple[int, ...] = ()

    # -- registration ----------------------------------------------------------
    def register(self, name: str, fn: Callable[..., Any] | None = None, *,
                 priority: int = 0, timeout_s: float | None = None,
                 phase: str = "shutdown") -> Callable[..., Any]:
        """Register a hook; usable as a decorator.

        ``@coord.on_shutdown("db", priority=50)`` — wait, that's
        :meth:`on_shutdown`. ``register`` itself also works as a decorator
        factory: ``@coord.register("db", priority=50)``.
        """
        def do_register(func: Callable[..., Any]) -> Callable[..., Any]:
            hook = ShutdownHook(
                name=name, fn=func, priority=priority,
                timeout_s=timeout_s if timeout_s is not None
                else self.default_hook_timeout_s,
                phase=phase)
            with self._lock:
                self._hooks.append(hook)
            return func

        if fn is not None:
            return do_register(fn)
        return do_register

    def on_shutdown(self, name: str, *, priority: int = 0,
                    timeout_s: float | None = None) -> Callable[..., Any]:
        """Decorator: ``@coord.on_shutdown("db", priority=50)``."""
        return self.register(name, priority=priority, timeout_s=timeout_s,
                             phase="shutdown")

    def on_startup(self, name: str, *, priority: int = 0,
                   timeout_s: float | None = None) -> Callable[..., Any]:
        """Decorator: ``@coord.on_startup("db", priority=50)``."""
        return self.register(name, priority=priority, timeout_s=timeout_s,
                             phase="startup")

    def unregister(self, name: str, phase: str = "shutdown") -> bool:
        """Remove a hook by name. Returns True when one was removed."""
        with self._lock:
            before = len(self._hooks)
            self._hooks = [h for h in self._hooks
                           if not (h.name == name and h.phase == phase)]
            return len(self._hooks) < before

    def hooks(self, phase: str = "shutdown") -> list[ShutdownHook]:
        with self._lock:
            return [h for h in self._hooks if h.phase == phase]

    # -- state -------------------------------------------------------------------
    @property
    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown_requested

    @property
    def reason(self) -> str:
        with self._lock:
            return self._shutdown_reason

    @property
    def ready(self) -> bool:
        """Readiness: False from the instant shutdown is requested.

        Liveness (the process is running) is separate — readiness failing
        during shutdown is what tells load balancers to stop sending
        traffic while in-flight work drains.
        """
        return not self.shutdown_requested

    def uptime_s(self) -> float:
        return time.time() - self._booted_at

    def request_shutdown(self, reason: str = "") -> bool:
        """Request shutdown. Thread-safe and signal-handler-safe.

        Returns True when this call was the first request; later calls are
        no-ops returning False (the original reason is kept).
        """
        with self._lock:
            if self._shutdown_requested:
                return False
            self._shutdown_requested = True
            self._shutdown_reason = reason or "requested"
            self._shutdown_started_at = time.time()
            return True

    # -- execution ---------------------------------------------------------------
    def run_startup(self) -> dict[str, Any]:
        """Run startup hooks in ASCENDING priority order. Idempotent."""
        with self._lock:
            if self._startup_report is not None:
                return self._startup_report
        report = self._run_phase("startup", reverse=False)
        with self._lock:
            self._startup_report = report
        return report

    def run(self, reason: str = "") -> dict[str, Any]:
        """Run shutdown hooks in DESCENDING priority order. Idempotent.

        A second call (a second SIGTERM, a finally block after the handler
        already ran) returns the first report instead of re-running hooks.
        """
        with self._lock:
            if self._shutdown_report is not None:
                return self._shutdown_report
        if reason:
            self.request_shutdown(reason)
        else:
            self.request_shutdown("run() invoked")
        report = self._run_phase("shutdown", reverse=True)
        with self._lock:
            self._shutdown_report = report
        return report

    def _run_phase(self, phase: str, *, reverse: bool) -> dict[str, Any]:
        with self._lock:
            ordered = sorted(
                (h for h in self._hooks if h.phase == phase),
                key=lambda h: (-h.priority if reverse else h.priority, h.name))
            reason = self._shutdown_reason
        started = time.time()
        results = [self._invoke(hook, reason) for hook in ordered]
        duration = time.time() - started
        failed = [r for r in results if r["status"] != "ok"]
        return {
            "coordinator": self.name,
            "phase": phase,
            "reason": reason,
            "hooks": results,
            "hook_count": len(results),
            "failed": len(failed),
            "duration_s": round(duration, 3),
            "stopped": phase == "shutdown",
            "uptime_s": round(self.uptime_s(), 3),
        }

    def _invoke(self, hook: ShutdownHook, reason: str) -> dict[str, Any]:
        """Run one hook with its timeout. A straggler is recorded as a
        timeout and shutdown moves on — never block a deploy forever."""
        outcome: dict[str, Any] = {
            "name": hook.name, "priority": hook.priority,
            "timeout_s": hook.timeout_s,
            "status": "ok", "error": "", "duration_s": 0.0,
        }
        box: dict[str, Any] = {}
        started = time.perf_counter()

        def target() -> None:
            try:
                res = hook.fn(reason) if hook.takes_reason else hook.fn()
                if asyncio.iscoroutine(res):
                    res = asyncio.run(res)
                box["result"] = res
            except BaseException as exc:  # noqa: BLE001 - hooks report, never raise
                box["error"] = exc

        thread = threading.Thread(target=target, daemon=True,
                                  name=f"shutdown-{hook.name}")
        thread.start()
        thread.join(hook.timeout_s)
        outcome["duration_s"] = round(time.perf_counter() - started, 3)
        if thread.is_alive():
            outcome["status"] = "timeout"
            outcome["error"] = (f"hook exceeded {hook.timeout_s:g}s; "
                                f"continuing shutdown without it")
            _log.warning("shutdown hook %r timed out after %gs",
                         hook.name, hook.timeout_s)
        elif "error" in box:
            outcome["status"] = "error"
            exc = box["error"]
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            _log.warning("shutdown hook %r failed: %s", hook.name,
                         outcome["error"])
        return outcome

    # -- signals -------------------------------------------------------------------
    def install(self, signals: tuple[int, ...] | None = None, *,
                raise_interrupt: bool = True) -> bool:
        """Install signal handlers that request shutdown.

        The handler records the request and — when ``raise_interrupt`` —
        raises KeyboardInterrupt so existing ``except KeyboardInterrupt`` /
        finally teardown paths keep working; entry points then call
        :meth:`run` in their finally block to execute the ordered hooks.
        Idempotent per signal. Returns False off the main thread or when a
        signal is unavailable.
        """
        if signals is None:
            candidates = []
            for name in ("SIGTERM", "SIGINT"):
                sig = getattr(signal, name, None)
                if isinstance(sig, int):
                    candidates.append(sig)
            signals = tuple(candidates)
        try:
            for sig in signals:
                previous = signal.getsignal(sig)
                if getattr(previous, "_devon_coord_handler", False):
                    continue

                def _handler(signum: int, frame: object,
                             _sig: int = sig) -> None:  # noqa: ARG001
                    self.request_shutdown(
                        f"signal {signal.Signals(_sig).name}")
                    if raise_interrupt:
                        raise KeyboardInterrupt()

                _handler._devon_coord_handler = True  # type: ignore[attr-defined]
                signal.signal(sig, _handler)
                with self._lock:
                    self._installed = tuple(sorted(set(self._installed + (sig,))))
        except (OSError, ValueError, RuntimeError) as exc:
            _log.debug("coordinator signal install failed: %s", exc)
            return False
        if not signals:
            return False
        return True

    # -- reporting -------------------------------------------------------------------
    @property
    def last_report(self) -> dict[str, Any] | None:
        with self._lock:
            return self._shutdown_report

    def beacon(self) -> dict[str, Any]:
        """Final status beacon payload: ``stopped: true`` once shutdown ran."""
        report = self.last_report or {}
        return {
            "service": self.name,
            "stopped": bool(report),
            "reason": self.reason,
            "ready": self.ready,
            "uptime_s": round(self.uptime_s(), 3),
            "report": report,
        }


_coordinator: ShutdownCoordinator | None = None
_coordinator_lock = threading.Lock()


def get_coordinator() -> ShutdownCoordinator:
    """Process-wide coordinator, created on first use."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = ShutdownCoordinator()
        return _coordinator


def reset_coordinator() -> None:
    """Forget the process-wide coordinator (used by tests)."""
    global _coordinator
    with _coordinator_lock:
        _coordinator = None
