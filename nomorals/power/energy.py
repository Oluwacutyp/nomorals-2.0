"""Per-task-type energy ledger (userspace analog of the kernel Energy Model).

The Linux Energy Aware Scheduler keeps a per-performance-domain table of
operating points (capacity vs power cost) and simulates placements against
it before migrating work.  We can't read RAPL counters portably, so this
module keeps the portable stand-in: a learned ledger of wall-clock
duration per task type, converted to a watt-hour estimate with a clearly
documented heuristic power draw.

The estimate is an *advisory* — good for ordering cheap work first and
for answering "will this batch fit in the remaining battery", not for
billing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["EnergyLedger", "BASE_WATTS", "WATTS_PER_MEM_MB"]

# Heuristic draw for a busy worker slot on a typical laptop-class machine.
# Portable, documented, and deliberately conservative — not a measurement.
BASE_WATTS = 8.0
# Extra draw per MB of declared task memory (larger working set ≈ more
# cache/memory-subsystem activity).
WATTS_PER_MEM_MB = 0.001


class EnergyLedger:
    """Learned duration-per-task-type ledger → Wh estimates."""

    def __init__(self) -> None:
        # task_type -> {"count": int, "total_s": float, "total_mem_mb": float}
        self._stats: dict[str, dict[str, float]] = {}

    def record(self, task_type: str, duration_s: float,
               mem_mb: float = 0.0) -> None:
        """Record one finished run. Never raises; ignores bad input."""
        try:
            if not task_type or duration_s is None:
                return
            d = max(0.0, float(duration_s))
            m = max(0.0, float(mem_mb))
            st = self._stats.setdefault(
                str(task_type), {"count": 0.0, "total_s": 0.0,
                                 "total_mem_mb": 0.0})
            st["count"] += 1.0
            st["total_s"] += d
            st["total_mem_mb"] += m
        except (TypeError, ValueError):
            pass

    def avg_duration_s(self, task_type: str) -> float | None:
        st = self._stats.get(str(task_type))
        if not st or st["count"] <= 0:
            return None
        return st["total_s"] / st["count"]

    def estimated_wh(self, task_type: str, mem_mb: float = 0.0) -> float | None:
        """Estimated energy for one run, or None when unseen (unknown ≠ 0)."""
        avg = self.avg_duration_s(task_type)
        if avg is None:
            return None
        try:
            m = max(0.0, float(mem_mb))
        except (TypeError, ValueError):
            m = 0.0
        watts = BASE_WATTS + WATTS_PER_MEM_MB * m
        return watts * (avg / 3600.0)

    def batch_wh(self, tasks: list[tuple[str, float]]) -> float | None:
        """Sum of estimates for ``[(task_type, mem_mb)]``; None if any
        task type is unseen (refuses to invent numbers)."""
        total = 0.0
        for task_type, mem_mb in tasks:
            wh = self.estimated_wh(task_type, mem_mb)
            if wh is None:
                return None
            total += wh
        return total

    def summary(self) -> dict[str, Any]:
        return {
            t: {
                "runs": int(st["count"]),
                "avg_duration_s": round(st["total_s"] / st["count"], 2),
                "avg_mem_mb": round(st["total_mem_mb"] / st["count"], 1),
                "estimated_wh_per_run": round(
                    self.estimated_wh(t, st["total_mem_mb"] / st["count"])
                    or 0.0, 4),
            }
            for t, st in sorted(self._stats.items())
            if st["count"] > 0
        }

    def clear(self) -> None:
        self._stats.clear()
