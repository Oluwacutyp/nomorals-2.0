"""Health monitoring for the OS control plane.

A :class:`HealthCheck` is anything with a ``name`` and a zero-argument
``check()`` that returns a :class:`HealthStatus`.  :class:`HealthMonitor`
runs a set of checks on demand via :meth:`check_all` and fires recovery
hooks registered with :meth:`on_failure` whenever a check reports not-ok.

Checks never take the monitor down: a raising check becomes a failed
:class:`HealthStatus`, and a raising recovery hook is logged, not
propagated.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

_log = logging.getLogger(__name__)

__all__ = [
    "HealthStatus",
    "HealthCheck",
    "HealthMonitor",
    "as_health_check",
]


@dataclass
class HealthStatus:
    """The outcome of one health check."""

    name: str
    ok: bool
    detail: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "ts": self.ts}

    @property
    def failed(self) -> bool:
        return not self.ok


@runtime_checkable
class HealthCheck(Protocol):
    """Structural protocol: ``name`` + ``check() -> HealthStatus``."""

    @property
    def name(self) -> str: ...

    def check(self) -> HealthStatus: ...


class _CallableHealthCheck:
    """Adapt a zero-arg callable into the :class:`HealthCheck` protocol.

    The callable may return:

    * ``bool`` — True is healthy;
    * ``str`` — returned as the detail, healthy unless it starts with
      ``"FAIL"`` / ``"ERROR"`` (case-insensitive);
    * ``(bool, str)`` tuple — ``(ok, detail)``;
    * a :class:`HealthStatus` — passed through.
    """

    def __init__(self, name: str, fn: Callable[[], Any]) -> None:
        self._name = name
        self._fn = fn

    @property
    def name(self) -> str:
        return self._name

    def check(self) -> HealthStatus:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 — a raising check is a failed check
            return HealthStatus(name=self._name, ok=False,
                                detail=f"check raised: {exc!r}")
        if isinstance(result, HealthStatus):
            if not result.name:
                result.name = self._name
            return result
        if isinstance(result, bool):
            return HealthStatus(name=self._name, ok=result,
                                detail="ok" if result else "check returned False")
        if isinstance(result, tuple) and len(result) == 2:
            ok, detail = result
            return HealthStatus(name=self._name, ok=bool(ok), detail=str(detail))
        if isinstance(result, str):
            lowered = result.strip().lower()
            ok = not (lowered.startswith("fail") or lowered.startswith("error"))
            return HealthStatus(name=self._name, ok=ok, detail=result)
        return HealthStatus(name=self._name, ok=bool(result),
                            detail=f"returned {result!r}")


def as_health_check(name: str, check: HealthCheck | Callable[[], Any]) -> HealthCheck:
    """Normalize a HealthCheck instance or plain callable into a HealthCheck."""
    if isinstance(check, HealthCheck):
        return check
    return _CallableHealthCheck(name, check)


class HealthMonitor:
    """Owns a set of named health checks plus failure recovery hooks."""

    def __init__(self) -> None:
        self._checks: dict[str, HealthCheck] = {}
        self._failure_hooks: dict[str, list[Callable[[HealthStatus], Any]]] = {}
        self._lock = threading.RLock()
        self._last: dict[str, HealthStatus] = {}

    # ── checks ───────────────────────────────────────────────────────────
    def register_check(self, check: HealthCheck | Callable[[], Any], *,
                       name: str = "") -> str:
        """Register a check; returns its name. Re-registering replaces."""
        hc = as_health_check(name or getattr(check, "name", "") or "check", check)
        if not hc.name:
            raise ValueError("health check needs a name")
        with self._lock:
            self._checks[hc.name] = hc
        return hc.name

    def unregister_check(self, name: str) -> bool:
        with self._lock:
            return self._checks.pop(name, None) is not None

    def check_names(self) -> list[str]:
        with self._lock:
            return sorted(self._checks)

    def check_all(self) -> dict[str, HealthStatus]:
        """Run every registered check once; fire failure hooks for not-ok."""
        with self._lock:
            checks = list(self._checks.values())
        results: dict[str, HealthStatus] = {}
        for hc in checks:
            try:
                status = hc.check()
            except Exception as exc:  # noqa: BLE001 — defensive; adapter already guards
                status = HealthStatus(name=hc.name, ok=False,
                                      detail=f"check raised: {exc!r}")
            if not status.name:
                status.name = hc.name
            results[status.name] = status
            with self._lock:
                self._last[status.name] = status
            if status.failed:
                self._fire_failure_hooks(status)
        return results

    def last_status(self, name: str) -> HealthStatus | None:
        with self._lock:
            return self._last.get(name)

    # ── recovery hooks ───────────────────────────────────────────────────
    def on_failure(self, check_name: str,
                   handler: Callable[[HealthStatus], Any]) -> None:
        """Run ``handler(status)`` whenever ``check_name`` reports not-ok.

        The special name ``"*"`` matches every failing check.
        """
        with self._lock:
            self._failure_hooks.setdefault(check_name, []).append(handler)

    def _fire_failure_hooks(self, status: HealthStatus) -> None:
        with self._lock:
            handlers = list(self._failure_hooks.get(status.name, ()))
            handlers += list(self._failure_hooks.get("*", ()))
        for handler in handlers:
            try:
                handler(status)
            except Exception:  # noqa: BLE001 — a broken hook must not break monitoring
                _log.exception("health failure hook for %r raised", status.name)
