"""Degradation ladders: planned, honest graceful degradation.

Each subsystem declares its ladder *before* the outage: an ordered list
of rungs from full capability down to honest failure. When the primary
fails, the system steps down the ladder instead of crashing; when the
primary heals, it climbs back up automatically.

Every rung carries:
- trigger: what moves the system to this rung (error signal)
- capability trade: what the user loses at this rung
- honesty clause: the user-visible message (users forgive slower, not silent)
- probe: how to detect the primary has healed (auto-recovery)

Mined from agent-reliability ADRs (see ERROR_MINING.md).
Pure stdlib.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import NoMoralsError
from .logging_setup import get_logger

__all__ = [
    "DegradationLadder",
    "LadderExhausted",
    "Rung",
    "LadderManager",
]

_log = get_logger(__name__)


@dataclass
class Rung:
    """One step on the ladder. Lower index = better experience."""

    name: str
    # What to call at this rung. Receives the same args the primary got.
    handler: Callable[..., Any]
    # Human description of what's lost at this rung.
    capability_trade: str = ""
    # User-visible honesty clause. REQUIRED for every rung below 0.
    honesty: str = ""
    # Probe: returns True when the primary is healthy again. Only needed
    # on rung 0 (the primary) — used for auto-recovery climbs.
    probe: Callable[[], bool] | None = None
    # Seconds between auto-recovery probes while degraded.
    probe_interval_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.handler:
            raise ValueError(f"rung {self.name!r} needs a handler")


class DegradationLadder:
    """One subsystem's ladder.

    Usage:
        ladder = DegradationLadder("llm", rungs=[
            Rung("groq", call_groq, probe=groq_healthy),
            Rung("openrouter", call_openrouter,
                 capability_trade="slower, smaller model",
                 honesty="Primary model unavailable — answering with a fallback model."),
            Rung("cached", answer_from_cache,
                 capability_trade="answers may be stale",
                 honesty="Live models unavailable — answering from cached knowledge."),
        ])
        answer, meta = ladder.call(prompt)
        # meta: {"rung": 1, "rung_name": "openrouter", "degraded": True, ...}
    """

    def __init__(self, subsystem: str, rungs: list[Rung],
                 journal: Any | None = None, *,
                 exhausted_hint: Callable[[], float | None] | None = None,
                 breaker_floor_s: float = 1.0,
                 breaker_cap_s: float = 600.0,
                 breaker_default_s: float = 20.0,
                 warn_dedupe_s: float = 120.0) -> None:
        if not rungs:
            raise ValueError("ladder needs at least one rung")
        for i, rung in enumerate(rungs):
            if i > 0 and not rung.honesty:
                raise ValueError(
                    f"rung {i} ({rung.name!r}) needs an honesty clause —"
                    " degraded results must say so")
        self.subsystem = subsystem
        self.rungs = rungs
        self.journal = journal
        # All-down circuit breaker: after a call exhausts every rung,
        # further calls fast-fail with the same LadderExhausted instead
        # of paying a full rung walk per caller (the boot-storm failure:
        # dozens of callers × full attempts × per-rung warnings inside
        # seconds). ``exhausted_hint`` reports how long a retry is worth
        # waiting for (the llm ladder wires it to the router's earliest
        # provider-cooldown expiry); None → ``breaker_default_s``. The
        # window is clamped to [floor, cap]. A healthy primary probe
        # closes the breaker early (see _breaker_probe).
        self._exhausted_hint = exhausted_hint
        self.breaker_floor_s = max(0.0, float(breaker_floor_s))
        self.breaker_cap_s = max(self.breaker_floor_s, float(breaker_cap_s))
        self.breaker_default_s = max(0.0, float(breaker_default_s))
        self.warn_dedupe_s = max(0.0, float(warn_dedupe_s))
        self._lock = threading.RLock()
        self._rung_idx = 0
        self._last_probe_ts = 0.0
        self._consecutive_failures = 0
        self._transitions: list[dict[str, Any]] = []  # rung-change history
        self._on_transition: list[Callable[[str, int, int], None]] = []
        # Breaker state (monotonic-free: wall clock, like the probe ts).
        self._breaker_until = 0.0
        self._breaker_opened_at = 0.0
        self._breaker_errors: list[str] = []
        self._fast_fails = 0
        # Rung-failure log dedupe: (rung idx, error text) → last logged
        # ts / suppressed count. Identical failures log once per window;
        # the suppressed count is summarised when the text changes, the
        # window lapses, or a call succeeds.
        self._warn_last: dict[tuple[int, str], float] = {}
        self._warn_suppressed: dict[tuple[int, str], int] = {}

    def on_transition(self, fn: Callable[[str, int, int], None]) -> Callable[[str, int, int], None]:
        """Observe rung changes: ``fn(subsystem, old_idx, new_idx)``."""
        with self._lock:
            self._on_transition.append(fn)
        return fn

    def _set_rung(self, idx: int, why: str) -> None:
        with self._lock:
            old = self._rung_idx
            if old == idx:
                return
            self._rung_idx = idx
            self._transitions.append({
                "ts": time.time(), "from": old, "to": idx,
                "from_name": self.rungs[old].name, "to_name": self.rungs[idx].name,
                "why": why,
            })
            del self._transitions[:-50]  # bounded history
            hooks = list(self._on_transition)
        _log.info("ladder %s: rung %d (%s) → %d (%s) [%s]",
                  self.subsystem, old, self.rungs[old].name,
                  idx, self.rungs[idx].name, why)
        for hook in hooks:
            try:
                hook(self.subsystem, old, idx)
            except Exception:  # noqa: BLE001 - observability must not break calls
                _log.exception("ladder on_transition hook failed")

    def transitions(self) -> list[dict[str, Any]]:
        """Rung-change history, oldest first (bounded to the last 50)."""
        with self._lock:
            return list(self._transitions)

    @property
    def rung_idx(self) -> int:
        with self._lock:
            return self._rung_idx

    @property
    def degraded(self) -> bool:
        return self.rung_idx > 0

    def _record(self, ok: bool, rung_idx: int) -> None:
        if self.journal is None:
            return
        try:
            self.journal.record_heartbeat(
                f"{self.subsystem}/ladder", ok=ok)
        except Exception:  # noqa: BLE001 - journal must not break calls
            pass

    def _primary_healthy(self) -> bool | None:
        """Probe rung 0, honouring its probe interval.

        Returns True/False when a probe actually ran, None when no
        probe exists or the interval has not elapsed (no information).
        """
        primary = self.rungs[0]
        if primary.probe is None:
            return None
        with self._lock:
            now = time.time()
            if now - self._last_probe_ts < primary.probe_interval_s:
                return None
            self._last_probe_ts = now
        try:
            return bool(primary.probe())
        except Exception:  # noqa: BLE001 - a broken probe means "not healthy"
            return False

    def _try_recover(self) -> None:
        """Probe the primary; climb back to rung 0 when it's healthy."""
        with self._lock:
            if self._rung_idx == 0:
                return
        if self._primary_healthy():
            self._set_rung(0, "primary healed")
            with self._lock:
                self._consecutive_failures = 0
            _log.info("ladder %s: primary healed, climbed back to rung 0",
                      self.subsystem)

    # ── all-down circuit breaker ─────────────────────────────────────
    def _open_breaker(self, errors: list[str]) -> None:
        """Open the breaker after an exhausting call. Logs ONCE."""
        delay: float | None = None
        if self._exhausted_hint is not None:
            try:
                delay = self._exhausted_hint()
            except Exception:  # noqa: BLE001 - a broken hint must not break calls
                delay = None
        if delay is None or delay <= 0:
            delay = self.breaker_default_s
        delay = min(self.breaker_cap_s,
                    max(self.breaker_floor_s, float(delay)))
        now = time.time()
        with self._lock:
            self._breaker_until = now + delay
            self._breaker_opened_at = now
            self._breaker_errors = list(errors)
            self._fast_fails = 0
        window_s = f"{delay:.0f}s" if delay >= 10 else f"{delay:.1f}s"
        _log.warning(
            "ladder %s: all rungs failed — circuit open for %s; "
            "calls fast-fail until it closes", self.subsystem, window_s)

    def _breaker_gate(self) -> None:
        """Raise LadderExhausted while the breaker is open.

        A healthy primary probe closes the breaker early — the breaker
        floors the retry rate, it does not replace probe-driven
        recovery. When the window has simply expired, the episode is
        summarised once and the call proceeds to the rungs.
        """
        now = time.time()
        with self._lock:
            until = self._breaker_until
        if until <= 0.0:
            return
        if until > now:
            if self._primary_healthy():
                self._close_breaker_episode("primary healed")
                self._set_rung(0, "primary healed")
                with self._lock:
                    self._consecutive_failures = 0
                return
            with self._lock:
                self._fast_fails += 1
                errors = list(self._breaker_errors)
                retry_in = until - now
            raise LadderExhausted(
                f"{self.subsystem}: all {len(self.rungs)} ladder rungs "
                f"failed (circuit open — fast-failing, retry in "
                f"{retry_in:.0f}s)",
                details={"subsystem": self.subsystem, "errors": errors,
                         "circuit_open": True,
                         "retry_in_s": round(retry_in, 3)})
        self._close_breaker_episode("window expired")
        with self._lock:
            # The post-breaker attempt is a FULL round from the primary:
            # exhaustion parked the ladder on its last rung, and without
            # a healthy probe nothing else would climb back to retry
            # the providers this breaker exists to protect.
            self._rung_idx = 0

    def _close_breaker_episode(self, why: str) -> None:
        with self._lock:
            fast = self._fast_fails
            opened = self._breaker_opened_at
            self._breaker_until = 0.0
            self._breaker_opened_at = 0.0
            self._breaker_errors = []
            self._fast_fails = 0
        if fast:
            _log.info(
                "ladder %s: circuit episode over (%s) — %d call(s) "
                "fast-failed over %.0fs; retrying rungs",
                self.subsystem, why, fast, max(0.0, time.time() - opened))

    # ── rung-failure log dedupe ──────────────────────────────────────
    def _log_rung_failure(self, idx: int, rung: Rung, exc: Exception) -> None:
        """Log a rung failure, deduped per (rung, error) per window."""
        key = (idx, f"{type(exc).__name__}: {exc}")
        now = time.time()
        with self._lock:
            last = self._warn_last.get(key, 0.0)
            if now - last < self.warn_dedupe_s:
                self._warn_suppressed[key] = \
                    self._warn_suppressed.get(key, 0) + 1
                return
            suppressed = self._warn_suppressed.pop(key, 0)
            self._warn_last[key] = now
        suffix = (f" (+{suppressed} identical suppressed)"
                  if suppressed else "")
        _log.warning("ladder %s: rung %d (%s) failed: %s%s",
                     self.subsystem, idx, rung.name, exc, suffix)

    def _flush_warn_summaries(self) -> None:
        """On success, report how many identical warnings were held back."""
        with self._lock:
            items = [(k, v) for k, v in self._warn_suppressed.items() if v]
            for k, _ in items:
                self._warn_suppressed.pop(k, None)
        for (idx, err), count in items:
            _log.info("ladder %s: suppressed %d identical rung-%d "
                      "failure warning(s): %s",
                      self.subsystem, count, idx, err)

    def call(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Call the subsystem, stepping down the ladder on failure.

        Returns (result, meta). Meta always includes rung info and, when
        degraded, the honesty clause for user-visible messages.
        """
        self._breaker_gate()
        self._try_recover()
        with self._lock:
            start_idx = self._rung_idx
        errors: list[str] = []
        for idx in range(start_idx, len(self.rungs)):
            rung = self.rungs[idx]
            try:
                result = rung.handler(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - step down the ladder
                errors.append(f"{rung.name}: {type(exc).__name__}: {exc}")
                self._log_rung_failure(idx, rung, exc)
                continue
            # Success: settle on this rung.
            self._set_rung(idx, "settled" if idx != start_idx else "primary ok")
            with self._lock:
                self._consecutive_failures = 0
            self._flush_warn_summaries()
            self._record(True, idx)
            return result, self._meta(idx, errors)
        # Every rung failed.
        self._set_rung(len(self.rungs) - 1, "all rungs failed")
        with self._lock:
            self._consecutive_failures += 1
        self._record(False, len(self.rungs) - 1)
        self._open_breaker(errors)
        raise LadderExhausted(
            f"{self.subsystem}: all {len(self.rungs)} ladder rungs failed",
            details={"subsystem": self.subsystem, "errors": errors})

    def _meta(self, idx: int, errors: list[str]) -> dict[str, Any]:
        rung = self.rungs[idx]
        return {
            "subsystem": self.subsystem,
            "rung": idx,
            "rung_name": rung.name,
            "degraded": idx > 0,
            "honesty": rung.honesty if idx > 0 else "",
            "capability_trade": rung.capability_trade,
            "rungs_failed": errors,
        }

    def force_rung(self, idx: int, reason: str = "") -> None:
        """Manually pin the ladder (testing, maintenance)."""
        if not 0 <= idx < len(self.rungs):
            raise ValueError(f"rung {idx} out of range")
        self._set_rung(idx, f"manual pin: {reason}" if reason else "manual pin")

    def status(self) -> dict[str, Any]:
        with self._lock:
            rung = self.rungs[self._rung_idx]
            circuit_open = self._breaker_until > time.time()
            return {
                "subsystem": self.subsystem,
                "rung": self._rung_idx,
                "rung_name": rung.name,
                "degraded": self._rung_idx > 0,
                "honesty": rung.honesty,
                "rungs": [r.name for r in self.rungs],
                "consecutive_failures": self._consecutive_failures,
                # All-down circuit breaker ("" / 0 when closed).
                "circuit_open": circuit_open,
                "retry_in_s": (round(self._breaker_until - time.time(), 1)
                               if circuit_open else 0.0),
                "fast_fails": self._fast_fails if circuit_open else 0,
            }


class LadderExhausted(NoMoralsError):
    """All rungs failed. Carries structured details for escalation.

    Now part of the one error dialect (:class:`NoMoralsError`, code
    ``"ladder.exhausted"``) instead of a bare ``Exception`` — still
    catchable as ``Exception``, and ``retryable`` because the primary may
    heal while probes keep running.
    """

    code = "ladder.exhausted"
    retryable = True

    def __init__(self, message: str,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details or {})


class LadderManager:
    """Owns every subsystem's ladder."""

    def __init__(self) -> None:
        self._ladders: dict[str, DegradationLadder] = {}
        self._lock = threading.RLock()

    def register(self, ladder: DegradationLadder) -> DegradationLadder:
        with self._lock:
            self._ladders[ladder.subsystem] = ladder
        return ladder

    def get(self, subsystem: str) -> DegradationLadder | None:
        with self._lock:
            return self._ladders.get(subsystem)

    def status_all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {n: l.status() for n, l in self._ladders.items()}

    def degraded_subsystems(self) -> list[str]:
        with self._lock:
            return [n for n, l in self._ladders.items() if l.degraded]

    def format_report(self, theme: Any = None) -> str:
        """Every ladder as a styled status card."""
        from .style import header, status_dot, styled_table

        with self._lock:
            ladders = list(self._ladders.values())
        if not ladders:
            return "no degradation ladders registered"
        rows = []
        for lad in ladders:
            st = lad.status()
            dot = status_dot("warn" if st["degraded"] else "ok", theme)
            rows.append([
                f"{dot} {lad.subsystem}",
                f"{st['rung']} ({st['rung_name']})",
                str(st["consecutive_failures"]),
                " → ".join(st["rungs"]),
            ])
        return (header("degradation ladders", theme) + "\n"
                + styled_table(["subsystem", "rung", "failures", "ladder"], rows, theme))
