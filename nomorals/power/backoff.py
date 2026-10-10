"""Retry/deferral backoff with jitter.

Implements the strategies from Marc Brooker's AWS Architecture Blog post
"Exponential Backoff and Jitter" (the same family the storage queue's
``fail()`` path already uses):

- ``full`` (recommended default): ``uniform(0, min(cap, base * 2**attempt))``
- ``equal``: ``cap_delay/2 + uniform(0, cap_delay/2)``
- ``decorrelated``: ``min(cap, uniform(base, prev * 3))`` — stateful per key
- ``none``: ``min(cap, base * 2**attempt)`` — deterministic baseline; not
  recommended for production (worst case under contention)

Fixed delays keep deferrals marching in lockstep (the thundering herd);
exponential spreads them; jitter dissolves the schedule.  Every strategy
accepts an injected RNG (any object with ``uniform(a, b)``) so tests stay
deterministic.
"""

from __future__ import annotations

import random
from typing import Any, Protocol

__all__ = ["BackoffPolicy", "STRATEGIES"]


class _Rng(Protocol):
    def uniform(self, a: float, b: float) -> float: ...


STRATEGIES = ("full", "equal", "decorrelated", "none")


class BackoffPolicy:
    """Jittered exponential backoff for deferral requeues."""

    def __init__(
        self,
        base: float = 30.0,
        cap: float = 1800.0,
        strategy: str = "full",
        rng: _Rng | None = None,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(
                f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        self.base = max(0.0, float(base))
        self.cap = max(0.0, float(cap))
        self.strategy = strategy
        self.rng: _Rng = rng if rng is not None else random
        # Decorrelated jitter is stateful; track the last delay per key so
        # interleaved tasks don't corrupt each other's random walk.
        self._prev: dict[Any, float] = {}

    def _expo(self, attempt: int) -> float:
        return min(self.cap, self.base * (2.0 ** max(0, int(attempt))))

    def next_delay(self, attempt: int = 0, *, key: Any = None) -> float:
        """Delay in seconds after ``attempt`` deferrals (0-based count of
        deferrals so far; the first deferral uses ``attempt=1``)."""
        attempt = max(0, int(attempt))
        if self.strategy == "none":
            return self._expo(attempt)
        if self.strategy == "decorrelated":
            prev = self._prev.get(key, self.base)
            delay = min(self.cap, self.rng.uniform(self.base, prev * 3.0))
            self._prev[key] = delay
            return max(0.0, delay)
        cap_delay = self._expo(attempt)
        if self.strategy == "equal":
            return cap_delay / 2.0 + self.rng.uniform(0.0, cap_delay / 2.0)
        # full jitter
        return self.rng.uniform(0.0, cap_delay)

    def reset(self, key: Any = None) -> None:
        """Forget decorrelated state (all keys when ``key`` is None)."""
        if key is None:
            self._prev.clear()
        else:
            self._prev.pop(key, None)

    def to_dict(self) -> dict[str, Any]:
        return {"base": self.base, "cap": self.cap,
                "strategy": self.strategy}
