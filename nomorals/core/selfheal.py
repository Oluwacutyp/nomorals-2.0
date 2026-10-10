"""Self-healing executor: the error catcher that fights back.

The old ``error_intelligence`` classified and explained. This module
*attempts recovery*:

    analyze → consult memory → recover → verify → record

Recovery ladder (tried in order):
1. Known verified fix from the incident journal (mined rule: memory first).
2. Retry with exponential backoff + jitter (uses ``core.retry``).
3. Fallback chain — alternative implementations, best quality first.
4. Degraded mode — partial result, honestly labeled.
5. Escalation — structured failure with everything the human needs.

Every recovery ends with a verification probe. A recovery that doesn't
verify is recorded as *unverified*, never promoted to the fix journal.

Design target: t3.small CPU. No new dependencies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .error_doctor import diagnose
from .error_intelligence import (
    ErrorAnalysis,
    ErrorCategory,
    ErrorIntelligence,
    ErrorSeverity,
)
from .incidents import Incident, IncidentJournal, RecoveryRecord
from .logging_setup import get_logger
from .retry import BackoffPolicy, CircuitBreaker, default_retryable

__all__ = [
    "RecoveryOutcome",
    "RecoveryStatus",
    "SelfHealingExecutor",
    "SubsystemSupervisor",
    "RestartStrategy",
]

_log = get_logger(__name__)


class RecoveryStatus:
    OK = "ok"                    # primary succeeded, no recovery needed
    RECOVERED = "recovered"      # a strategy fixed it, verifier confirmed
    DEGRADED = "degraded"        # partial result, honestly labeled
    FAILED = "failed"            # nothing worked; structured escalation


@dataclass
class RecoveryOutcome:
    """The result of a healed (or unhealable) operation."""

    status: str
    result: Any = None
    analysis: ErrorAnalysis | None = None
    incident: Incident | None = None
    strategy_used: str = ""
    attempts: int = 0
    verified: bool = False
    verify_note: str = ""
    degraded_reason: str = ""
    similar_past: int = 0          # how many times we've seen this before
    chronic: bool = False          # recurring pattern flag

    @property
    def ok(self) -> bool:
        return self.status in (RecoveryStatus.OK, RecoveryStatus.RECOVERED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "strategy_used": self.strategy_used,
            "attempts": self.attempts,
            "verified": self.verified,
            "verify_note": self.verify_note,
            "degraded_reason": self.degraded_reason,
            "similar_past": self.similar_past,
            "chronic": self.chronic,
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }

    def format_outcome(self, *, color: bool | None = None,
                       theme: Any = None) -> str:
        """A styled one-screen recovery report."""
        from .style import paint, supports_color, status_dot, header

        if color is None:
            color = supports_color()
        dot = status_dot("ok" if self.ok else "error", theme, color=color)
        head = {RecoveryStatus.OK: "primary succeeded",
                RecoveryStatus.RECOVERED: "recovered",
                RecoveryStatus.DEGRADED: "degraded",
                RecoveryStatus.FAILED: "FAILED"}.get(self.status, self.status)
        lines = [header(f"{dot} recovery: {head}", theme, color=color), ""]
        if self.strategy_used:
            lines.append(f"  strategy: {paint(self.strategy_used, 'info', theme, color=color)}"
                         f" ({self.attempts} attempt(s))")
        if self.verify_note:
            mark = paint("verified", "ok", theme, color=color) if self.verified \
                else paint("unverified", "warn", theme, color=color)
            lines.append(f"  {mark}: {self.verify_note}")
        if self.degraded_reason:
            lines.append(f"  degraded: {self.degraded_reason}")
        if self.chronic:
            lines.append(f"  {paint('chronic pattern', 'warn', theme, color=color)}"
                         f" — seen {self.similar_past}x before")
        elif self.similar_past:
            lines.append(f"  seen {self.similar_past}x before")
        if self.analysis is not None:
            lines.append(f"  error: {type(self.analysis).__name__}")
        return "\n".join(lines).rstrip()


# ── recovery strategies ──────────────────────────────────────────────────

class RecoveryStrategy:
    """One way to recover. Subclass and override ``attempt``."""

    name = "base"

    def can_handle(self, analysis: ErrorAnalysis) -> bool:
        return True

    def attempt(self, fn: Callable[[], Any], analysis: ErrorAnalysis,
                ctx: dict[str, Any]) -> tuple[bool, Any, str]:
        """Try to recover. Returns (recovered, result_or_none, note)."""
        raise NotImplementedError


class RetryStrategy(RecoveryStrategy):
    """Retry with exponential backoff + jitter. For transient failures."""

    name = "retry"

    def __init__(self, policy: BackoffPolicy | None = None) -> None:
        self.policy = policy or BackoffPolicy()

    def can_handle(self, analysis: ErrorAnalysis) -> bool:
        return analysis.retryable

    def attempt(self, fn, analysis, ctx):
        last: BaseException | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                return True, fn(), f"succeeded on attempt {attempt}"
            except Exception as exc:  # noqa: BLE001 - classified per attempt
                last = exc
                if attempt >= self.policy.max_attempts:
                    break
                retry_after = getattr(exc, "retry_after", None)
                delay = self.policy.delay(attempt, retry_after)
                if delay > 0:
                    time.sleep(delay)
        return False, None, f"exhausted {self.policy.max_attempts} attempts ({last})"


class FallbackChainStrategy(RecoveryStrategy):
    """Try alternative implementations, best quality first.

    ``fallbacks``: ordered list of (name, callable). The first that
    produces an acceptable result wins. Mirrors the mined degradation
    pattern: each rung sacrifices some quality for reliability.
    """

    name = "fallback_chain"

    def __init__(self, fallbacks: list[tuple[str, Callable[[], Any]]]) -> None:
        self.fallbacks = fallbacks

    def can_handle(self, analysis: ErrorAnalysis) -> bool:
        return bool(self.fallbacks)

    def attempt(self, fn, analysis, ctx):
        errors: list[str] = []
        for fb_name, fb_fn in self.fallbacks:
            try:
                result = fb_fn()
                return True, result, f"fallback {fb_name!r} succeeded"
            except Exception as exc:  # noqa: BLE001 - next rung
                errors.append(f"{fb_name}: {type(exc).__name__}: {exc}")
        return False, None, "all fallbacks failed: " + " | ".join(errors)


class CredentialRefreshStrategy(RecoveryStrategy):
    """For auth failures: run the refresh hook, then retry once."""

    name = "credential_refresh"

    def __init__(self, refresh: Callable[[dict[str, Any]], bool]) -> None:
        self._refresh = refresh

    def can_handle(self, analysis: ErrorAnalysis) -> bool:
        return analysis.category == ErrorCategory.AUTH

    def attempt(self, fn, analysis, ctx):
        try:
            refreshed = self._refresh(ctx)
        except Exception as exc:  # noqa: BLE001 - refresh itself failed
            return False, None, f"credential refresh raised {exc}"
        if not refreshed:
            return False, None, "credential refresh declined"
        try:
            return True, fn(), "retry after credential refresh succeeded"
        except Exception as exc:  # noqa: BLE001
            return False, None, f"retry after refresh failed: {exc}"


class DegradedResultStrategy(RecoveryStrategy):
    """Last resort before escalation: produce a partial, honest result."""

    name = "degraded"

    def __init__(self, degrade: Callable[[ErrorAnalysis, dict], Any]) -> None:
        self._degrade = degrade

    def attempt(self, fn, analysis, ctx):
        try:
            result = self._degrade(analysis, ctx)
            return True, result, "degraded result produced"
        except Exception as exc:  # noqa: BLE001
            return False, None, f"degraded path failed: {exc}"


# ── the executor ─────────────────────────────────────────────────────────

class SelfHealingExecutor:
    """Runs an operation with full analyze → recover → verify → record.

    Usage:
        ex = SelfHealingExecutor(subsystem="telegram", journal=journal)
        out = ex.execute(
            lambda: send_message(...),
            context={"operation": "send_message", "chat_id": 123},
            fallbacks=[("queue_for_later", lambda: enqueue(...))],
            verify=lambda r: r is not None,
        )
        if out.status == RecoveryStatus.DEGRADED:
            ...  # result is partial; out.degraded_reason says why
    """

    def __init__(
        self,
        subsystem: str,
        *,
        journal: IncidentJournal | None = None,
        intelligence: ErrorIntelligence | None = None,
        breaker: CircuitBreaker | None = None,
        strategies: list[RecoveryStrategy] | None = None,
    ) -> None:
        self.subsystem = subsystem
        self.journal = journal or IncidentJournal()
        self.intel = intelligence or ErrorIntelligence()
        self.breaker = breaker
        self.strategies = strategies or [RetryStrategy()]
        self._lock = threading.RLock()
        #: per-strategy track record: name → {attempts, recovered}
        self.strategy_stats: dict[str, dict[str, int]] = {}

    def execute(
        self,
        fn: Callable[[], Any],
        *,
        context: dict[str, Any] | None = None,
        fallbacks: list[tuple[str, Callable[[], Any]]] | None = None,
        verify: Callable[[Any], bool] | None = None,
        on_credential_refresh: Callable[[dict], bool] | None = None,
        on_degrade: Callable[[ErrorAnalysis, dict], Any] | None = None,
    ) -> RecoveryOutcome:
        """Execute ``fn`` with self-healing. Never raises for *handled*
        failures — returns a FAILED outcome instead. Programming errors
        (bugs in ``fn`` itself, e.g. TypeError from our own code) are still
        recorded but also returned, not raised: the caller decides.
        """
        ctx = {"subsystem": self.subsystem, **(context or {})}

        # Fast path: breaker already open.
        if self.breaker is not None:
            try:
                self.breaker.before_call()
            except Exception as exc:  # noqa: BLE001 - CircuitOpen
                return self._escalate(exc, ctx, strategy_used="circuit_open")

        # Primary attempt.
        error: BaseException | None = None
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - the whole point
            error = exc
        else:
            if self.breaker is not None:
                self.breaker.record_success()
            self.journal.record_heartbeat(self.subsystem, ok=True)
            return RecoveryOutcome(status=RecoveryStatus.OK, result=result,
                                   attempts=1, verified=True,
                                   verify_note="primary succeeded")

        # Failure path: analyze deeply.
        assert error is not None  # noqa: S101 - except always sets it
        exc = error
        analysis = self.intel.analyze(exc, context=ctx)
        try:
            doctor = diagnose(exc)  # deep AST-level diagnosis, never raises
        except Exception:  # noqa: BLE001 - diagnose is bulletproofed anyway
            doctor = {}
        incident = self.journal.record_incident(
            exc, subsystem=self.subsystem,
            category=analysis.category.value,
            severity=analysis.severity.value, context=ctx)
        if self.breaker is not None:
            self.breaker.record_failure(exc)
        # Note: no heartbeat(ok=False) here — the incident row above is the
        # failure signal. Heartbeats are for operations that don't raise.

        similar = self.journal.find_similar(exc, self.subsystem)
        chronic = self.journal.is_chronic(exc, self.subsystem)

        # Strategy 0: memory — a verified fix for this exact shape goes first.
        known = self.journal.known_fix(exc, self.subsystem)
        strategies: list[RecoveryStrategy] = list(self.strategies)
        if known:
            # Move the strategy that produced the verified fix to the front.
            match = [s for s in strategies if s.name == known["strategy"]]
            rest = [s for s in strategies if s.name != known["strategy"]]
            if match:
                _log.info("selfheal: known verified fix %r for %s — trying first",
                          known["strategy"], incident.signature)
                strategies = match + rest
        if fallbacks:
            strategies.append(FallbackChainStrategy(fallbacks))
        if on_credential_refresh:
            strategies.append(CredentialRefreshStrategy(on_credential_refresh))
        degraded_strategy: DegradedResultStrategy | None = None
        if on_degrade:
            degraded_strategy = DegradedResultStrategy(on_degrade)
            strategies.append(degraded_strategy)

        attempts = 1  # the primary attempt
        for strategy in strategies:
            if not strategy.can_handle(analysis):
                continue
            # Memory-first: if we know a verified fix and this strategy
            # matches its name, it goes first (already ordered by caller,
            # but the known fix is authoritative).
            ok, result, note = strategy.attempt(fn, analysis, ctx)
            attempts += 1
            with self._lock:
                stat = self.strategy_stats.setdefault(
                    strategy.name, {"attempts": 0, "recovered": 0})
                stat["attempts"] += 1
                if ok:
                    stat["recovered"] += 1
            verified, vnote = self._verify(result, ok, verify)
            rec = RecoveryRecord(
                incident_id=incident.id,
                signature=incident.signature,
                strategy=strategy.name, detail=note,
                verified=verified, verify_note=vnote)
            self.journal.record_recovery(rec)
            if ok and verified:
                status = (RecoveryStatus.DEGRADED
                          if strategy is degraded_strategy
                          else RecoveryStatus.RECOVERED)
                return RecoveryOutcome(
                    status=status, result=result, analysis=analysis,
                    incident=incident, strategy_used=strategy.name,
                    attempts=attempts, verified=True, verify_note=vnote,
                    degraded_reason=(note if status == RecoveryStatus.DEGRADED
                                     else ""),
                    similar_past=len(similar), chronic=chronic)
            _log.warning("selfheal: strategy %s failed for %s: %s",
                         strategy.name, incident.signature, note)

        # Nothing worked: structured escalation.
        if known:
            analysis.suggested_fix = (
                f"Known fix ({known['strategy']}, worked"
                f" {known['successes']}x): {known['detail']}"
                f" — {analysis.suggested_fix}")
        return self._escalate(exc, ctx, analysis=analysis,
                              incident=incident, attempts=attempts,
                              similar_past=len(similar), chronic=chronic,
                              doctor=doctor)

    def _verify(self, result: Any, ok: bool,
                verify: Callable[[Any], bool] | None) -> tuple[bool, str]:
        """Independent verification gate. Unverified recoveries are never
        promoted to the fix journal (mined rule 3)."""
        if not ok:
            return False, "strategy reported failure"
        if verify is None:
            return True, "no verifier supplied; strategy success accepted"
        try:
            if verify(result):
                return True, "verifier confirmed"
            return False, "verifier rejected the result"
        except Exception as exc:  # noqa: BLE001 - verifier itself broke
            return False, f"verifier raised {type(exc).__name__}: {exc}"

    def _escalate(self, exc: BaseException, ctx: dict[str, Any], *,
                  analysis: ErrorAnalysis | None = None,
                  incident: Incident | None = None,
                  strategy_used: str = "",
                  attempts: int = 0,
                  similar_past: int = 0,
                  chronic: bool = False,
                  doctor: dict[str, Any] | None = None) -> RecoveryOutcome:
        analysis = analysis or self.intel.analyze(exc, context=ctx)
        if incident is None:
            incident = self.journal.record_incident(
                exc, subsystem=self.subsystem,
                category=analysis.category.value,
                severity=analysis.severity.value, context=ctx)
        level = {
            ErrorSeverity.CRITICAL: "critical",
            ErrorSeverity.HIGH: "error",
            ErrorSeverity.MEDIUM: "warning",
            ErrorSeverity.LOW: "info",
        }[analysis.severity]
        getattr(_log, level)(
            "selfheal ESCALATE [%s] %s: %s (chronic=%s, seen=%dx)",
            self.subsystem, analysis.exception_type,
            analysis.explanation, chronic, similar_past + 1)
        return RecoveryOutcome(
            status=RecoveryStatus.FAILED, analysis=analysis,
            incident=incident, strategy_used=strategy_used,
            attempts=attempts, verified=False,
            similar_past=similar_past, chronic=chronic)


# ── OTP-inspired subsystem supervisor ────────────────────────────────────

class RestartStrategy:
    """How a supervisor restarts a crashed worker (Erlang/OTP, adapted)."""
    ONE_FOR_ONE = "one_for_one"    # restart just the crashed worker
    ONE_FOR_ALL = "one_for_all"    # restart all workers (shared state)
    REST_FOR_ONE = "rest_for_one"  # restart crashed + workers started after it


class SubsystemSupervisor:
    """Watches worker callables; restarts them on crash per strategy.

    This is the "let it crash" half of the design: workers focus on the
    happy path, the supervisor owns recovery. Thread-safe, stdlib only.

    Usage:
        sup = SubsystemSupervisor("scheduler")
        sup.add_worker("ticker", tick_loop, strategy=RestartStrategy.ONE_FOR_ONE)
        sup.start_all()   # each worker runs on a daemon thread, supervised
    """

    def __init__(self, subsystem: str, *,
                 journal: IncidentJournal | None = None,
                 max_restarts: int = 5,
                 restart_window_s: float = 300.0) -> None:
        self.subsystem = subsystem
        self.journal = journal or IncidentJournal()
        self.max_restarts = max_restarts
        self.restart_window_s = restart_window_s
        self._workers: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()

    def add_worker(self, name: str, fn: Callable[[], None], *,
                   strategy: str = RestartStrategy.ONE_FOR_ONE,
                   restart_delay_s: float = 2.0) -> None:
        with self._lock:
            self._workers[name] = {
                "fn": fn, "strategy": strategy,
                "restart_delay_s": restart_delay_s,
                "restarts": [],  # timestamps of restarts in window
                "thread": None, "running": False,
                "stop": threading.Event(),
                "accepts_stop": self._fn_accepts_stop(fn),
            }

    @staticmethod
    def _fn_accepts_stop(fn: Callable) -> bool:
        """True if the worker declares a ``stop_event`` parameter.

        Cooperative workers get the event and should exit their loop when
        it is set — this is what makes ONE_FOR_ALL / REST_FOR_ONE real
        restarts instead of hopeful ones.
        """
        try:
            import inspect as _inspect
            return "stop_event" in _inspect.signature(fn).parameters
        except Exception:  # noqa: BLE001 - uninspectable callable
            return False

    def start_all(self) -> None:
        with self._lock:
            names = list(self._workers.keys())
        for name in names:
            self._start_worker(name)

    def stop_all(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            names = list(self._workers.keys())
        for name in names:
            self._stop_worker(name, timeout_s=timeout_s)

    def _start_worker(self, name: str) -> None:
        with self._lock:
            w = self._workers[name]
            if w["running"]:
                return
            w["stop"].clear()
            w["running"] = True
            t = threading.Thread(target=self._supervise, args=(name,),
                                 name=f"sup-{self.subsystem}-{name}",
                                 daemon=True)
            w["thread"] = t
        t.start()

    def _stop_worker(self, name: str, timeout_s: float = 5.0) -> bool:
        """Signal a worker to stop and wait for its thread. Returns True
        if the thread exited cleanly."""
        with self._lock:
            w = self._workers[name]
            t = w["thread"]
            w["stop"].set()
        if t is not None and t is not threading.current_thread():
            t.join(timeout=timeout_s)
            alive = t.is_alive()
        else:
            alive = False
        with self._lock:
            if not alive:
                w["running"] = False
                w["thread"] = None
        if alive:
            _log.error("supervisor: worker %s/%s ignored stop signal",
                       self.subsystem, name)
        return not alive

    def _supervise(self, name: str) -> None:
        order = self._worker_order()
        w = self._workers[name]
        stop_ev: threading.Event = w["stop"]
        call_kwargs: dict[str, Any] = (
            {"stop_event": stop_ev} if w["accepts_stop"] else {})
        while not stop_ev.is_set() and not self._stop.is_set():
            try:
                w["fn"](**call_kwargs)
                # Worker returned cleanly: not a crash, don't restart.
                with self._lock:
                    w["running"] = False
                return
            except Exception as exc:  # noqa: BLE001 - supervise, don't die
                self.journal.record_incident(
                    exc, subsystem=f"{self.subsystem}/{name}",
                    category="supervisor", severity="high",
                    context={"worker": name})
                _log.error("supervisor: worker %s/%s crashed: %s",
                           self.subsystem, name, exc)
                if not self._note_restart(name):
                    _log.critical(
                        "supervisor: worker %s/%s exceeded %d restarts in %.0fs"
                        " — giving up (escalate to human)",
                        self.subsystem, name, self.max_restarts,
                        self.restart_window_s)
                    with self._lock:
                        w["running"] = False
                    return
                targets = self._restart_targets(name, order,
                                                w["strategy"])
                delay = w["restart_delay_s"]
                _log.warning("supervisor: restarting %s (strategy %s) in %.1fs",
                             targets, w["strategy"], delay)
                # Stop sibling targets first so ONE_FOR_ALL / REST_FOR_ONE
                # are real restarts, not hopeful ones.
                for t in targets:
                    if t != name:
                        self._stop_worker(t)
                time.sleep(delay)
                for t in targets:
                    if t != name and not self._stop.is_set():
                        self._start_worker(t)
                # loop continues: this worker's fn runs again

    def _worker_order(self) -> list[str]:
        with self._lock:
            return list(self._workers.keys())

    def _restart_targets(self, crashed: str, order: list[str],
                         strategy: str) -> list[str]:
        if strategy == RestartStrategy.ONE_FOR_ALL:
            return list(order)
        if strategy == RestartStrategy.REST_FOR_ONE:
            idx = order.index(crashed)
            return order[idx:]
        return [crashed]

    def _note_restart(self, name: str) -> bool:
        """True if another restart is allowed (within budget)."""
        now = time.time()
        with self._lock:
            w = self._workers[name]
            w["restarts"] = [t for t in w["restarts"]
                             if now - t < self.restart_window_s]
            if len(w["restarts"]) >= self.max_restarts:
                return False
            w["restarts"].append(now)
            return True

    def watch(self, name: str,
              is_alive: Callable[[], bool],
              restart: Callable[[], None], *,
              check_interval_s: float = 30.0,
              strategy: str = RestartStrategy.ONE_FOR_ONE) -> None:
        """Watchdog for an externally-managed thread.

        Some components (scheduler, autonomy) own their threads already —
        rewriting their lifecycle would be invasive. Instead the supervisor
        watches: ``is_alive()`` reports thread health, ``restart()`` brings
        it back. A dead thread records an incident, counts against the
        restart budget, and triggers ``restart()``. Exceeding the budget
        escalates (logs critical, stops watching).

        The watcher itself runs on a daemon thread owned by the supervisor.
        """
        def _watch_loop(stop_event: threading.Event) -> None:
            while not stop_event.is_set():
                try:
                    alive = is_alive()
                except Exception:  # noqa: BLE001 - a broken probe is a failure
                    alive = False
                if not alive and not stop_event.is_set():
                    try:
                        raise RuntimeError(
                            f"watched thread {self.subsystem}/{name} died")
                    except RuntimeError as exc:
                        self.journal.record_incident(
                            exc, subsystem=f"{self.subsystem}/{name}",
                            category="supervisor", severity="high",
                            context={"worker": name, "mode": "watchdog"})
                    _log.error("supervisor: watched %s/%s died", self.subsystem, name)
                    if not self._note_restart(name):
                        _log.critical(
                            "supervisor: watched %s/%s exceeded %d restarts "
                            "in %.0fs — giving up (escalate to human)",
                            self.subsystem, name, self.max_restarts,
                            self.restart_window_s)
                        with self._lock:
                            if name in self._workers:
                                self._workers[name]["running"] = False
                        return
                    _log.warning("supervisor: restarting watched %s/%s",
                                 self.subsystem, name)
                    try:
                        restart()
                    except Exception:  # noqa: BLE001
                        _log.exception("supervisor: restart of %s/%s failed",
                                       self.subsystem, name)
                stop_event.wait(check_interval_s)

        with self._lock:
            self._workers[name] = {
                "fn": _watch_loop, "strategy": strategy,
                "restart_delay_s": 0.0,
                "restarts": [],
                "thread": None, "running": False,
                "stop": threading.Event(),
                "accepts_stop": True,
                "watchdog": True,
            }
        self._start_worker(name)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "running": w["running"],
                    "strategy": w["strategy"],
                    "restarts_in_window": len(w["restarts"]),
                    "watchdog": w.get("watchdog", False),
                }
                for name, w in self._workers.items()
            }
