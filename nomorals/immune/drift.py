"""Behaviour drift — 3-day watch, suggestions not actions.

Sibling in shape to ``health/drift.py``: needs 3 full days of data
before any verdict (``MIN_DAYS = 3``), and it NEVER acts — it returns
a :class:`DriftReport` and the caller decides what, if anything, to
do with it.

Signals (latest day vs the first recorded day as baseline, plus a
monotonic check for grant-hoarding):

* ``token-burn`` — tokens > 3x baseline.
* ``process-growth`` — processes spawned > 2x baseline.
* ``grant-hoarding`` — grants held growing across all recorded days.
* ``error-spike`` — errors > 3x baseline (or >= 5 from a zero baseline).

Verdict: ``insufficient`` (< 3 days) | ``ok`` (no signals) |
``watch`` (one signal) | ``act`` (two or more signals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = ["MIN_DAYS", "DriftReport", "BehaviourDriftMonitor"]

MIN_DAYS = 3


@dataclass
class DriftReport:
    verdict: str = "insufficient"  # insufficient | ok | watch | act
    signals: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    days_considered: int = 0


class BehaviourDriftMonitor:
    """Accumulate one metrics dict per day; report drift, never act."""

    def __init__(self) -> None:
        self._days: List[Tuple[Any, Dict[str, float]]] = []

    def add_day(self, date: Any, metrics: Dict[str, Any]) -> DriftReport:
        clean = {
            "tool_calls": float(metrics.get("tool_calls", 0) or 0),
            "tokens": float(metrics.get("tokens", 0) or 0),
            "processes_spawned": float(metrics.get("processes_spawned", 0) or 0),
            "grants_held": float(metrics.get("grants_held", 0) or 0),
            "errors": float(metrics.get("errors", 0) or 0),
        }
        self._days.append((date, clean))
        return self.current_report()

    def current_report(self) -> DriftReport:
        days = len(self._days)
        if days < MIN_DAYS:
            return DriftReport(
                verdict="insufficient",
                signals=[],
                reasons=[f"need {MIN_DAYS} full days before a verdict; have {days}"],
                days_considered=days,
            )
        baseline = self._days[0][1]
        latest = self._days[-1][1]
        signals: List[str] = []
        reasons: List[str] = []

        # token-burn: tokens > 3x baseline
        base_tokens = baseline["tokens"]
        if base_tokens > 0 and latest["tokens"] > 3 * base_tokens:
            signals.append("token-burn")
            reasons.append(f"tokens {latest['tokens']:g} > 3x baseline {base_tokens:g}")
        elif base_tokens == 0 and latest["tokens"] > 0:
            signals.append("token-burn")
            reasons.append(f"tokens {latest['tokens']:g} from a zero baseline")

        # process-growth: spawned > 2x baseline
        base_spawned = baseline["processes_spawned"]
        if base_spawned > 0 and latest["processes_spawned"] > 2 * base_spawned:
            signals.append("process-growth")
            reasons.append(
                f"processes_spawned {latest['processes_spawned']:g} > 2x baseline {base_spawned:g}"
            )
        elif base_spawned == 0 and latest["processes_spawned"] >= 20:
            signals.append("process-growth")
            reasons.append(
                f"processes_spawned {latest['processes_spawned']:g} from a zero baseline"
            )

        # grant-hoarding: grants_held strictly growing across all days
        grants = [m["grants_held"] for _, m in self._days]
        if all(later > earlier for earlier, later in zip(grants, grants[1:])):
            signals.append("grant-hoarding")
            reasons.append(f"grants_held growing every day: {grants}")

        # error-spike: errors > 3x baseline (or >= 5 from zero)
        base_errors = baseline["errors"]
        if base_errors > 0 and latest["errors"] > 3 * base_errors:
            signals.append("error-spike")
            reasons.append(f"errors {latest['errors']:g} > 3x baseline {base_errors:g}")
        elif base_errors == 0 and latest["errors"] >= 5:
            signals.append("error-spike")
            reasons.append(f"errors {latest['errors']:g} from a zero baseline")

        if not signals:
            verdict = "ok"
        elif len(signals) == 1:
            verdict = "watch"
        else:
            verdict = "act"
        return DriftReport(
            verdict=verdict, signals=signals, reasons=reasons, days_considered=days
        )
