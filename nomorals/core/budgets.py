"""Error budgets: SRE-style burn-rate alerting per subsystem.

Each subsystem gets an SLO (e.g. 99.9% success over 30 days). The budget
tracks *how fast the error budget is being consumed* — the burn rate —
not absolute error counts.

Canonical Google SRE multi-window thresholds (30-day window):

| Severity | Burn rate | Short window | Long window | Budget consumed |
|----------|-----------|--------------|-------------|-----------------|
| page     | 14.4x     | 5m           | 1h          | 2% in 1h        |
| page     | 6x        | 30m          | 6h          | 5% in 6h        |
| ticket   | 3x        | 2h           | 24h         | 10% in 1d       |
| ticket   | 1x        | 6h           | 72h         | 10% in 3d       |

The two-window rule: an alert fires only when BOTH the short and long
window exceed the threshold. The short window reacts fast; the long
window prevents flapping on brief spikes.

Budget policies (what changes behavior, not just what pages):
- budget < 20% remaining  -> warn: reliability work takes priority
- budget exhausted        -> freeze: no risky changes until recovery

Backed by the IncidentJournal (heartbeats give true success ratios).
Pure stdlib.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .incidents import IncidentJournal
from .logging_setup import get_logger

__all__ = [
    "ErrorBudget",
    "BudgetManager",
    "BudgetAlert",
    "BurnRate",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class BurnRate:
    """One alert rule: burn-rate threshold over two windows."""
    name: str
    burn_rate: float       # e.g. 14.4 means budget exhausts 14.4x too fast
    short_window_s: float  # fast-reacting window
    long_window_s: float   # flap-suppressing window
    severity: str          # "page" or "ticket"


# Canonical Google SRE thresholds.
CANONICAL_BURN_RATES = (
    BurnRate("fast_burn", 14.4, 300, 3600, "page"),
    BurnRate("medium_burn", 6.0, 1800, 21600, "page"),
    BurnRate("slow_burn", 3.0, 7200, 86400, "ticket"),
    BurnRate("chronic_burn", 1.0, 21600, 259200, "ticket"),
)


@dataclass
class BudgetAlert:
    subsystem: str
    rule: str
    severity: str          # page | ticket | warning
    burn_rate: float
    observed_ratio: float
    budget_remaining: float  # 1.0 = full, 0.0 = exhausted
    message: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subsystem": self.subsystem, "rule": self.rule,
            "severity": self.severity, "burn_rate": round(self.burn_rate, 2),
            "observed_ratio": round(self.observed_ratio, 5),
            "budget_remaining": round(self.budget_remaining, 4),
            "message": self.message, "ts": self.ts,
        }


class ErrorBudget:
    """One subsystem's error budget.

    Args:
        subsystem: name, e.g. "telegram", "llm", "voice"
        slo: success ratio target, e.g. 0.999 for 99.9%
        window_s: SLO window in seconds (default 30 days)
        burn_rates: alert rules (default: canonical SRE set)
    """

    def __init__(self, subsystem: str, slo: float = 0.999,
                 window_s: float = 30 * 86400,
                 journal: IncidentJournal | None = None,
                 burn_rates: tuple[BurnRate, ...] = CANONICAL_BURN_RATES,
                 on_alert: Callable[[BudgetAlert], None] | None = None) -> None:
        if not 0.0 < slo < 1.0:
            raise ValueError("slo must be in (0, 1)")
        self.subsystem = subsystem
        self.slo = slo
        self.window_s = window_s
        self.journal = journal or IncidentJournal()
        self.burn_rates = burn_rates
        self.on_alert = on_alert
        self._lock = threading.RLock()
        self._fired: dict[str, float] = {}  # rule -> last fired ts
        self._cooldown_s = 900.0            # don't re-fire within 15m

    @property
    def error_ratio_budget(self) -> float:
        """Allowed failure ratio, e.g. 0.001 for 99.9%."""
        return 1.0 - self.slo

    def budget_remaining(self, now: float | None = None) -> float:
        """Fraction of the window's error budget still unspent (1.0 = full)."""
        now = now if now is not None else time.time()
        fails, total, _ = self.journal.subsystem_ratio(
            self.subsystem, self.window_s, now)
        if total == 0:
            return 1.0
        allowed = total * self.error_ratio_budget
        if allowed == 0:
            return 0.0 if fails > 0 else 1.0
        return max(0.0, 1.0 - fails / allowed)

    def time_to_exhaustion(self, burn_rate: float) -> float | None:
        """Seconds until the budget is gone at the given burn rate."""
        if burn_rate <= 0:
            return None
        return self.window_s / burn_rate

    def check(self, now: float | None = None) -> list[BudgetAlert]:
        """Evaluate all burn-rate rules. Returns fired alerts.

        Applies the two-window rule: both short and long windows must
        exceed the burn-rate threshold before firing.
        """
        now = now if now is not None else time.time()
        alerts: list[BudgetAlert] = []
        budget = self.error_ratio_budget
        remaining = self.budget_remaining(now)

        for rule in self.burn_rates:
            _, _, short_ratio = self.journal.subsystem_ratio(
                self.subsystem, rule.short_window_s, now)
            _, _, long_ratio = self.journal.subsystem_ratio(
                self.subsystem, rule.long_window_s, now)
            # Not enough traffic to judge: skip (no data != healthy, but
            # paging on zero samples is worse).
            if short_ratio == 0 and long_ratio == 0:
                continue
            short_burn = short_ratio / budget if budget else 0.0
            long_burn = long_ratio / budget if budget else 0.0
            if short_burn >= rule.burn_rate and long_burn >= rule.burn_rate:
                last = self._fired.get(rule.name, 0.0)
                if now - last < self._cooldown_s:
                    continue
                burn = min(short_burn, long_burn)
                tte = self.time_to_exhaustion(burn)
                tte_str = (f"{tte / 3600:.1f}h" if tte and tte < 86400 * 2
                           else f"{tte / 86400:.1f}d" if tte else "n/a")
                alert = BudgetAlert(
                    subsystem=self.subsystem, rule=rule.name,
                    severity=rule.severity, burn_rate=burn,
                    observed_ratio=max(short_ratio, long_ratio),
                    budget_remaining=remaining,
                    message=(f"{self.subsystem}: {rule.name} — burning"
                             f" {burn:.1f}x budget (SLO {self.slo:.3%}),"
                             f" exhaustion in ~{tte_str} at this rate."),
                    ts=now)
                with self._lock:
                    self._fired[rule.name] = now
                alerts.append(alert)
                _log.warning("budget alert: %s", alert.message)
                if self.on_alert:
                    try:
                        self.on_alert(alert)
                    except Exception:  # noqa: BLE001 - alert hook must not break checks
                        _log.exception("budget on_alert hook failed")

        # Budget policy levels (independent of burn rate).
        if remaining <= 0.0:
            alerts.append(BudgetAlert(
                subsystem=self.subsystem, rule="budget_exhausted",
                severity="page", burn_rate=float("inf"),
                observed_ratio=0.0, budget_remaining=0.0,
                message=(f"{self.subsystem}: error budget EXHAUSTED —"
                         " freeze risky changes, reliability work only."),
                ts=now))
        elif remaining < 0.2:
            alerts.append(BudgetAlert(
                subsystem=self.subsystem, rule="budget_low",
                severity="warning", burn_rate=0.0,
                observed_ratio=0.0, budget_remaining=remaining,
                message=(f"{self.subsystem}: error budget at"
                         f" {remaining:.0%} — reliability work takes priority."),
                ts=now))
        return alerts

    def status(self, now: float | None = None) -> dict[str, Any]:
        """Full budget snapshot for dashboards."""
        now = now if now is not None else time.time()
        fails, total, ratio = self.journal.subsystem_ratio(
            self.subsystem, self.window_s, now)
        return {
            "subsystem": self.subsystem,
            "slo": self.slo,
            "window_s": self.window_s,
            "total_events": total,
            "failures": fails,
            "failure_ratio": round(ratio, 6),
            "budget_remaining": round(self.budget_remaining(now), 4),
            "rules": [r.name for r in self.burn_rates],
        }


class BudgetManager:
    """Owns budgets for every subsystem. One place to check them all."""

    def __init__(self, journal: IncidentJournal | None = None,
                 on_alert: Callable[[BudgetAlert], None] | None = None) -> None:
        self.journal = journal or IncidentJournal()
        self.on_alert = on_alert
        self._budgets: dict[str, ErrorBudget] = {}
        self._lock = threading.RLock()

    def budget_for(self, subsystem: str,
                   slo: float = 0.999) -> ErrorBudget:
        with self._lock:
            if subsystem not in self._budgets:
                self._budgets[subsystem] = ErrorBudget(
                    subsystem, slo=slo, journal=self.journal,
                    on_alert=self.on_alert)
            return self._budgets[subsystem]

    def check_all(self) -> list[BudgetAlert]:
        with self._lock:
            budgets = list(self._budgets.values())
        alerts: list[BudgetAlert] = []
        for b in budgets:
            alerts.extend(b.check())
        return alerts

    def status_all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: b.status() for name, b in self._budgets.items()}
