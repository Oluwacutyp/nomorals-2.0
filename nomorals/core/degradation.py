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

from .logging_setup import get_logger

__all__ = [
    "DegradationLadder",
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
                 journal: Any | None = None) -> None:
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
        self._lock = threading.RLock()
        self._rung_idx = 0
        self._last_probe_ts = 0.0
        self._consecutive_failures = 0

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

    def _try_recover(self) -> None:
        """Probe the primary; climb back to rung 0 when it's healthy."""
        with self._lock:
            if self._rung_idx == 0:
                return
            now = time.time()
            primary = self.rungs[0]
            if (primary.probe is None
                    or now - self._last_probe_ts < primary.probe_interval_s):
                return
            self._last_probe_ts = now
        try:
            healthy = primary.probe()
        except Exception:  # noqa: BLE001 - a broken probe means "not healthy"
            healthy = False
        if healthy:
            with self._lock:
                self._rung_idx = 0
                self._consecutive_failures = 0
            _log.info("ladder %s: primary healed, climbed back to rung 0",
                      self.subsystem)

    def call(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Call the subsystem, stepping down the ladder on failure.

        Returns (result, meta). Meta always includes rung info and, when
        degraded, the honesty clause for user-visible messages.
        """
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
                _log.warning("ladder %s: rung %d (%s) failed: %s",
                             self.subsystem, idx, rung.name, exc)
                continue
            # Success: settle on this rung.
            with self._lock:
                if idx != self._rung_idx:
                    _log.info("ladder %s: settled on rung %d (%s)",
                              self.subsystem, idx, rung.name)
                self._rung_idx = idx
                self._consecutive_failures = 0
            self._record(True, idx)
            return result, self._meta(idx, errors)
        # Every rung failed.
        with self._lock:
            self._rung_idx = len(self.rungs) - 1
            self._consecutive_failures += 1
        self._record(False, len(self.rungs) - 1)
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
        with self._lock:
            if not 0 <= idx < len(self.rungs):
                raise ValueError(f"rung {idx} out of range")
            self._rung_idx = idx
        _log.warning("ladder %s: manually pinned to rung %d (%s)",
                     self.subsystem, idx, reason)

    def status(self) -> dict[str, Any]:
        with self._lock:
            rung = self.rungs[self._rung_idx]
            return {
                "subsystem": self.subsystem,
                "rung": self._rung_idx,
                "rung_name": rung.name,
                "degraded": self._rung_idx > 0,
                "honesty": rung.honesty,
                "rungs": [r.name for r in self.rungs],
                "consecutive_failures": self._consecutive_failures,
            }


class LadderExhausted(Exception):
    """All rungs failed. Carries structured details for escalation."""

    def __init__(self, message: str,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}
        self.retryable = True  # the primary may heal; probes keep running


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
