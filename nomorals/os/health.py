"""Health monitoring for the OS control plane.

A :class:`HealthCheck` is anything with a ``name`` and a zero-argument
``check()`` that returns a :class:`HealthStatus`.  :class:`HealthMonitor`
runs a set of checks on demand via :meth:`check_all` (or continuously via
:meth:`start_background`) and fires recovery hooks registered with
:meth:`on_failure` whenever a check reports not-ok.

Checks carry a *role* borrowed from the Kubernetes probe triad:

* ``liveness`` — is the thing irrecoverably broken? A failing liveness
  check is restart-worthy.
* ``readiness`` — is the thing ready to take traffic/work right now? A
  failing readiness check sheds load but does not restart.
* ``startup`` — has slow initialization finished? Gates the other roles.

Checks never take the monitor down: a raising check becomes a failed
:class:`HealthStatus`, and a raising recovery hook is logged, not
propagated.  Consecutive-failure thresholds keep transient blips from
flapping the aggregate: a check is only *down* after ``failure_threshold``
failures in a row, and only *up* again after ``success_threshold``
successes in a row.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

_log = logging.getLogger(__name__)

__all__ = [
    "HealthStatus",
    "HealthCheck",
    "HealthMonitor",
    "as_health_check",
    "LIVENESS",
    "READINESS",
    "STARTUP",
]

#: Check roles (Kubernetes probe triad).
LIVENESS = "liveness"
READINESS = "readiness"
STARTUP = "startup"


@dataclass
class HealthStatus:
    """The outcome of one health check."""

    name: str
    ok: bool
    detail: str = ""
    ts: float = field(default_factory=time.time)
    #: Probe role: "liveness" | "readiness" | "startup".
    role: str = READINESS
    #: True when this status came from a stale background sample.
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "ok": self.ok, "detail": self.detail,
            "ts": self.ts, "role": self.role, "stale": self.stale,
        }

    @property
    def failed(self) -> bool:
        return not self.ok

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.ts)


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

    def __init__(self, name: str, fn: Callable[[], Any],
                 *, role: str = READINESS) -> None:
        self._name = name
        self._fn = fn
        self._role = role

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> str:
        return self._role

    def check(self) -> HealthStatus:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 — a raising check is a failed check
            return HealthStatus(name=self._name, ok=False, role=self._role,
                                detail=f"check raised: {exc!r}")
        if isinstance(result, HealthStatus):
            if not result.name:
                result.name = self._name
            return result
        if isinstance(result, bool):
            ok, detail = result, "ok" if result else "check returned False"
        elif isinstance(result, tuple) and len(result) == 2:
            ok, detail = bool(result[0]), str(result[1])
        elif isinstance(result, str):
            lowered = result.strip().lower()
            ok = not (lowered.startswith("fail") or lowered.startswith("error"))
            detail = result
        else:
            ok, detail = bool(result), f"returned {result!r}"
        return HealthStatus(name=self._name, ok=ok, role=self._role,
                            detail=detail)


def as_health_check(name: str, check: HealthCheck | Callable[[], Any], *,
                    role: str = READINESS) -> HealthCheck:
    """Normalize a HealthCheck instance or plain callable into a HealthCheck.

    ``role`` is one of ``liveness`` / ``readiness`` / ``startup``.
    """
    if isinstance(check, HealthCheck):
        return check
    return _CallableHealthCheck(name, check, role=role)


class _ManualCheck:
    """Application-controlled check: the app sets ok/fail explicitly.

    For initialization states ("cache warming", "model loaded") that only
    the application itself can judge.
    """

    def __init__(self, name: str, *, role: str = STARTUP,
                 detail: str = "not yet set") -> None:
        self._name = name
        self._role = role
        self._ok: bool | None = None
        self._detail = detail
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> str:
        return self._role

    def set(self, ok: bool, detail: str = "") -> None:
        with self._lock:
            self._ok = bool(ok)
            self._detail = detail or ("ok" if ok else "manually marked failed")

    def check(self) -> HealthStatus:
        with self._lock:
            ok, detail = self._ok, self._detail
        if ok is None:
            return HealthStatus(name=self._name, ok=False, role=self._role,
                                detail="manual check not set: " + detail)
        return HealthStatus(name=self._name, ok=ok, role=self._role,
                            detail=detail)


@dataclass
class _CheckMeta:
    """Per-check scheduling/threshold metadata."""

    check: HealthCheck
    role: str = READINESS
    interval_s: float = 30.0
    timeout_s: float = 5.0
    failure_threshold: int = 3
    success_threshold: int = 2
    manual: bool = False
    consec_fail: int = 0
    consec_ok: int = 0
    declared_down: bool = False  # flapping-suppressed aggregate state
    history: deque = field(default_factory=lambda: deque(maxlen=50))


class HealthMonitor:
    """Owns a set of named health checks plus failure recovery hooks.

    Checks can run on demand (:meth:`check_all`) or continuously in a
    background loop (:meth:`start_background`) with per-check intervals,
    timeouts, and flapping suppression via consecutive-failure /
    consecutive-success thresholds.
    """

    def __init__(self) -> None:
        self._checks: dict[str, _CheckMeta] = {}
        self._failure_hooks: dict[str, list[Callable[[HealthStatus], Any]]] = {}
        self._lock = threading.RLock()
        self._last: dict[str, HealthStatus] = {}
        self._bg_thread: threading.Thread | None = None
        self._bg_stop = threading.Event()
        self._bg_last_run: dict[str, float] = {}

    # ── checks ───────────────────────────────────────────────────────────
    def register_check(self, check: HealthCheck | Callable[[], Any], *,
                       name: str = "", role: str = READINESS,
                       interval_s: float = 30.0, timeout_s: float = 5.0,
                       failure_threshold: int = 1,
                       success_threshold: int = 2) -> str:
        """Register a check; returns its name. Re-registering replaces.

        ``role`` is ``liveness`` / ``readiness`` / ``startup``.
        ``failure_threshold`` consecutive failures mark the check down;
        ``success_threshold`` consecutive successes clear it (flap guard).
        With the default ``failure_threshold=1``, failure hooks fire on
        every failure (legacy semantics); set it higher to suppress flaps.
        """
        hc = as_health_check(name or getattr(check, "name", "") or "check",
                             check, role=role)
        if not hc.name:
            raise ValueError("health check needs a name")
        with self._lock:
            self._checks[hc.name] = _CheckMeta(
                check=hc, role=role, interval_s=max(0.5, float(interval_s)),
                timeout_s=max(0.1, float(timeout_s)),
                failure_threshold=max(1, int(failure_threshold)),
                success_threshold=max(1, int(success_threshold)),
            )
        return hc.name

    def register_manual(self, name: str, *, role: str = STARTUP,
                        detail: str = "not yet set") -> _ManualCheck:
        """Register an application-controlled check; returns it so the app
        can call ``.set(ok, detail)`` as initialization progresses."""
        manual = _ManualCheck(name, role=role, detail=detail)
        with self._lock:
            self._checks[name] = _CheckMeta(check=manual, role=role,
                                            manual=True)
        return manual

    def unregister_check(self, name: str) -> bool:
        with self._lock:
            return self._checks.pop(name, None) is not None

    def check_names(self, *, role: str | None = None) -> list[str]:
        with self._lock:
            names = [n for n, m in self._checks.items()
                     if role is None or m.role == role]
        return sorted(names)

    def check_all(self, *, role: str | None = None) -> dict[str, HealthStatus]:
        """Run every registered check once; fire failure hooks for not-ok.

        Applies flapping suppression: failure hooks fire only when a check
        newly *declares* down (after ``failure_threshold`` consecutive
        failures), not on every blip.
        """
        with self._lock:
            metas = [m for m in self._checks.values()
                     if role is None or m.role == role]
        results: dict[str, HealthStatus] = {}
        for meta in metas:
            status = self._run_one(meta, timeout_s=meta.timeout_s)
            self._record(meta, status)
            results[status.name] = status
        return results

    def check_one(self, name: str) -> HealthStatus | None:
        """Run a single check now."""
        with self._lock:
            meta = self._checks.get(name)
        if meta is None:
            return None
        status = self._run_one(meta, timeout_s=meta.timeout_s)
        self._record(meta, status)
        return status

    def last_status(self, name: str) -> HealthStatus | None:
        with self._lock:
            return self._last.get(name)

    def history(self, name: str, limit: int = 20) -> list[HealthStatus]:
        """Recent raw outcomes for ``name`` (newest first)."""
        with self._lock:
            meta = self._checks.get(name)
            if meta is None:
                return []
            items = list(meta.history)
        return list(reversed(items))[:max(0, int(limit))]

    def is_down(self, name: str) -> bool:
        """Flap-suppressed aggregate state for ``name``."""
        with self._lock:
            meta = self._checks.get(name)
            return bool(meta and meta.declared_down)

    # ── execution ────────────────────────────────────────────────────────
    @staticmethod
    def _run_one(meta: _CheckMeta, *, timeout_s: float) -> HealthStatus:
        """Run one check with a timeout; a timeout is a failed status."""
        if meta.manual:
            status = meta.check.check()
            status.role = meta.role
            return status
        box: list[HealthStatus] = []
        err: list[BaseException] = []

        def _target() -> None:
            try:
                box.append(meta.check.check())
            except BaseException as exc:  # noqa: BLE001 — adapter guards anyway
                err.append(exc)

        worker = threading.Thread(target=_target, daemon=True,
                                  name=f"health-{meta.check.name}")
        worker.start()
        worker.join(timeout=timeout_s)
        if worker.is_alive():
            return HealthStatus(name=meta.check.name, ok=False,
                                role=meta.role,
                                detail=f"check timed out after {timeout_s:g}s")
        if err:
            return HealthStatus(name=meta.check.name, ok=False, role=meta.role,
                                detail=f"check raised: {err[0]!r}")
        status = box[0] if box else HealthStatus(
            name=meta.check.name, ok=False, role=meta.role,
            detail="check returned nothing")
        if not status.name:
            status.name = meta.check.name
        status.role = meta.role
        return status

    def _record(self, meta: _CheckMeta, status: HealthStatus) -> None:
        with self._lock:
            meta.history.append(status)
            if status.ok:
                meta.consec_ok += 1
                meta.consec_fail = 0
                if meta.declared_down and meta.consec_ok >= meta.success_threshold:
                    meta.declared_down = False
                    _log.info("health check %r recovered", status.name)
            else:
                meta.consec_fail += 1
                meta.consec_ok = 0
                if (not meta.declared_down
                        and meta.consec_fail >= meta.failure_threshold):
                    meta.declared_down = True
            self._last[status.name] = status
            newly_down = status.failed and meta.declared_down
            # Default threshold (1) keeps the legacy semantics: hooks fire
            # on every failure.  With a higher threshold, hooks fire only
            # when the check newly declares down (flap suppression).
            if meta.failure_threshold <= 1:
                should_fire = status.failed
            else:
                should_fire = newly_down and meta.consec_fail == meta.failure_threshold
        if should_fire:
            self._fire_failure_hooks(status)

    # ── background loop ──────────────────────────────────────────────────
    def start_background(self) -> None:
        """Start the background loop: each check runs every ``interval_s``.

        Idempotent. Results older than 2× the check's interval are reported
        ``stale`` in :meth:`summary`.
        """
        with self._lock:
            if self._bg_thread is not None and self._bg_thread.is_alive():
                return
            self._bg_stop.clear()
            self._bg_thread = threading.Thread(
                target=self._background_loop, daemon=True,
                name="health-monitor")
            self._bg_thread.start()

    def stop_background(self) -> None:
        with self._lock:
            thread = self._bg_thread
        self._bg_stop.set()
        if thread is not None:
            thread.join(timeout=5.0)

    def _background_loop(self) -> None:
        while not self._bg_stop.is_set():
            now = time.time()
            with self._lock:
                due = [m for m in self._checks.values()
                       if now - self._bg_last_run.get(m.check.name, 0.0)
                       >= m.interval_s]
            for meta in due:
                if self._bg_stop.is_set():
                    break
                status = self._run_one(meta, timeout_s=meta.timeout_s)
                self._record(meta, status)
                with self._lock:
                    self._bg_last_run[meta.check.name] = time.time()
            self._bg_stop.wait(0.5)

    # ── aggregate ────────────────────────────────────────────────────────
    def summary(self, *, stale_after: float | None = None) -> dict[str, Any]:
        """Aggregate view: alive (liveness), ready (readiness), started.

        Checks whose last result is older than 2× their interval (or
        ``stale_after`` seconds when given) are reported stale and do not
        count toward readiness.
        """
        now = time.time()
        per_role: dict[str, dict[str, Any]] = {}
        overall_down: list[str] = []
        with self._lock:
            metas = list(self._checks.values())
            last = dict(self._last)
        for role in (LIVENESS, READINESS, STARTUP):
            names = [m.check.name for m in metas if m.role == role]
            downs: list[str] = []
            stales: list[str] = []
            for n in names:
                status = last.get(n)
                meta = next((m for m in metas if m.check.name == n), None)
                limit = (stale_after if stale_after is not None
                         else (meta.interval_s * 2 if meta else 60.0))
                if status is None or (now - status.ts) > limit:
                    stales.append(n)
                    continue
                if meta is not None and meta.declared_down:
                    downs.append(n)
            per_role[role] = {
                "total": len(names), "down": sorted(downs),
                "stale": sorted(stales),
                "ok": not downs and not stales and bool(names),
            }
            overall_down.extend(f"{role}:{n}" for n in downs)
        return {
            "alive": per_role[LIVENESS]["ok"],
            "ready": per_role[READINESS]["ok"],
            "startup_done": per_role[STARTUP]["ok"],
            "roles": per_role,
            "down": sorted(overall_down),
            "background_running": self._bg_thread is not None
            and self._bg_thread.is_alive(),
        }

    def render(self) -> str:
        """Plain-text health dashboard."""
        lines = ["health"]
        with self._lock:
            metas = sorted(self._checks.values(),
                           key=lambda m: (m.role, m.check.name))
            last = dict(self._last)
        for meta in metas:
            status = last.get(meta.check.name)
            if status is None:
                mark, detail = "?", "never ran"
            elif meta.declared_down:
                mark, detail = "✗", status.detail
            elif status.ok:
                mark, detail = "✓", status.detail
            else:
                mark, detail = f"! ({meta.consec_fail}/{meta.failure_threshold})", status.detail
            stale = " [stale]" if (status is not None and status.stale) else ""
            lines.append(f"  {mark} [{meta.role:8s}] {meta.check.name}{stale}")
            if detail:
                lines.append(f"      {detail}")
        summ = self.summary()
        lines.append(f"  alive={'yes' if summ['alive'] else 'no'}"
                     f" ready={'yes' if summ['ready'] else 'no'}"
                     f" startup_done={'yes' if summ['startup_done'] else 'no'}")
        return "\n".join(lines)

    # ── recovery hooks ───────────────────────────────────────────────────
    def on_failure(self, check_name: str,
                   handler: Callable[[HealthStatus], Any]) -> None:
        """Run ``handler(status)`` whenever ``check_name`` newly declares
        down (after its failure threshold).

        The special name ``"*"`` matches every newly-down check.
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
