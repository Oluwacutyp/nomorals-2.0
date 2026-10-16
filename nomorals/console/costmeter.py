"""Live cost meter: tokens + NGN per organ, and NGN/hour burn rate.

Every LLM / tool call that costs something is recorded with
:meth:`CostMeter.record_cost`. The meter attributes each entry to one
organ so the per-organ breakdown always sums to the total — that
invariant is what the tests check, and what makes the meter trustworthy
in the cockpit.

Rates are computed over a sliding window (default: last 60 minutes),
so ``total_per_hour()`` is the *current* burn rate, not an average
since process start.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _Entry:
    organ: str
    tokens: int
    ngn: float
    ts: float


@dataclass
class OrganCost:
    organ: str
    tokens: int = 0
    ngn: float = 0.0
    calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"organ": self.organ, "tokens": self.tokens,
                "ngn": round(self.ngn, 4), "calls": self.calls}


class CostMeter:
    """Thread-safe token + NGN cost ledger, attributed per organ."""

    def __init__(self, *, clock: Callable[[], float] | None = None,
                 window_s: float = 3600.0) -> None:
        self._clock = clock or time.time
        self.window_s = float(window_s)
        self._entries: list[_Entry] = []
        self._lock = threading.RLock()

    def record_cost(self, organ: str, tokens: int, ngn: float,
                    ts: float | None = None) -> None:
        """Record one charged call. ``tokens``/``ngn`` must be >= 0."""
        if tokens < 0 or ngn < 0:
            raise ValueError("tokens and ngn must be >= 0")
        with self._lock:
            self._entries.append(_Entry(
                organ=str(organ), tokens=int(tokens), ngn=float(ngn),
                ts=self._clock() if ts is None else float(ts)))

    # ── aggregation ──

    def _window_entries(self, now: float, window_s: float | None) -> list[_Entry]:
        if window_s is None:
            return list(self._entries)
        cutoff = now - window_s
        return [e for e in self._entries if e.ts >= cutoff]

    def per_organ(self, *, window_s: float | None = None) -> dict[str, OrganCost]:
        """Per-organ totals. Sums always equal the meter totals."""
        now = self._clock()
        with self._lock:
            entries = self._window_entries(now, window_s)
        out: dict[str, OrganCost] = {}
        for e in entries:
            row = out.setdefault(e.organ, OrganCost(organ=e.organ))
            row.tokens += e.tokens
            row.ngn += e.ngn
            row.calls += 1
        return out

    def total_tokens(self, *, window_s: float | None = None) -> int:
        return sum(r.tokens for r in self.per_organ(window_s=window_s).values())

    def total_ngn(self, *, window_s: float | None = None) -> float:
        return sum(r.ngn for r in self.per_organ(window_s=window_s).values())

    def total_calls(self, *, window_s: float | None = None) -> int:
        return sum(r.calls for r in self.per_organ(window_s=window_s).values())

    def per_hour(self, organ: str | None = None, *,
                 window_s: float | None = None) -> float:
        """NGN/hour over the sliding window (default: last 60 minutes).

        With ``organ`` set, the rate for that organ only; otherwise the
        whole-meter rate.
        """
        window = self.window_s if window_s is None else float(window_s)
        if window <= 0:
            return 0.0
        now = self._clock()
        with self._lock:
            entries = self._window_entries(now, window)
        ngn = sum(e.ngn for e in entries
                  if organ is None or e.organ == organ)
        return ngn / (window / 3600.0)

    def total_per_hour(self, *, window_s: float | None = None) -> float:
        """Whole-meter NGN/hour over the sliding window."""
        return self.per_hour(None, window_s=window_s)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()

    # ── render ──

    def render(self, *, window_s: float | None = None) -> str:
        rows = self.per_organ(window_s=window_s)
        lines = ["== cost meter =="]
        if not rows:
            lines.append("  (no costs recorded)")
            return "\n".join(lines)
        for name in sorted(rows):
            r = rows[name]
            lines.append(
                f"  {name:<18} {r.tokens:>7} tok  "
                f"₦{r.ngn:>9.2f}  {r.calls:>3} calls  "
                f"₦{self.per_hour(name, window_s=window_s):.2f}/h")
        lines.append(
            f"  {'TOTAL':<18} {self.total_tokens(window_s=window_s):>7} tok  "
            f"₦{self.total_ngn(window_s=window_s):>9.2f}  "
            f"{self.total_calls(window_s=window_s):>3} calls  "
            f"₦{self.total_per_hour(window_s=window_s):.2f}/h")
        return "\n".join(lines)


__all__ = ["CostMeter", "OrganCost"]
