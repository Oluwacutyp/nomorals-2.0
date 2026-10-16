"""Budget governor — per-organ budgets + a daily ceiling + preemption.

The existing ``nomorals.core.budgets`` tracks *error* budgets (SLO burn
rates). This module tracks *spend* budgets: tokens, money cost, and
wall-seconds, per organ, with a daily ceiling across all organs. When
an organ is over its budget — or the day is over its ceiling — work is
**deferred**, not executed, and the deferral is reported honestly
instead of silently dropped.

* :meth:`BudgetGovernor.track_spend` — record spend after work runs.
  ``time`` is accepted as an alias for ``seconds`` so callers can pass
  ``track_spend(organ, tokens=..., cost=..., time=...)``.
* :meth:`BudgetGovernor.check` — ask before work runs. Returns a dict
  with ``allowed`` (bool), ``status`` (``"allowed"`` / ``"deferred"``),
  ``reason``, and the measured spend vs. ceilings. Never raises on a
  missing organ: an unregistered organ gets the daily ceiling only.
* :meth:`BudgetGovernor.preempt` — log a preemption: low-priority work
  displaced so higher-priority work can run under the ceiling. The log
  is in-memory (``.preemptions``); pass the records on to the
  :class:`nomorals.autonomy.preemption.PreemptionLog` for a durable
  append-only copy when one is wired in.

Day boundaries use UTC dates so a restart mid-day does not reset the
spend (the governor is in-memory; a restart resets it — for durable
spend accounting, wire ``track_spend`` calls through a persisted
ledger as well. The in-memory state is the gate, not the audit trail).
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["BudgetGovernor"]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


class BudgetGovernor:
    """Per-organ spend budgets + a daily ceiling + preemption log."""

    def __init__(self, *,
                 daily_token_ceiling: float | None = None,
                 daily_cost_ceiling: float | None = None,
                 daily_seconds_ceiling: float | None = None,
                 per_organ_budgets: dict[str, dict[str, Any]] | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.daily_token_ceiling = daily_token_ceiling
        self.daily_cost_ceiling = daily_cost_ceiling
        self.daily_seconds_ceiling = daily_seconds_ceiling
        self.per_organ_budgets: dict[str, dict[str, float]] = {
            str(organ): {
                "tokens": _as_float(spec.get("tokens")),
                "cost": _as_float(spec.get("cost")),
                "seconds": _as_float(spec.get("seconds")),
            }
            for organ, spec in (per_organ_budgets or {}).items()
            if isinstance(spec, dict)
        }
        self._clock = clock or time.time
        self._day = self._today()
        #: organ -> {"tokens", "cost", "seconds"} spent today.
        self._spent: dict[str, dict[str, float]] = {}
        #: Preemption records, newest last. See :meth:`preempt`.
        self.preemptions: list[dict[str, Any]] = []

    # ── day rollover ─────────────────────────────────────────────

    def _today(self) -> str:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            self._clock(), tz=_dt.timezone.utc).strftime("%Y-%m-%d")

    def _rollover(self) -> None:
        today = self._today()
        if today != self._day:
            _log.info("budget governor: day rollover %s -> %s (spend reset)",
                      self._day, today)
            self._day = today
            self._spent = {}

    # ── spend ────────────────────────────────────────────────────

    def track_spend(self, organ: str, *,
                    tokens: float = 0.0, cost: float = 0.0,
                    seconds: float = 0.0, time: float | None = None,
                    **_ignored: Any) -> dict[str, Any]:
        """Record spend for ``organ``. ``time`` is an alias for
        ``seconds``. Returns the organ's spend snapshot after recording."""
        self._rollover()
        organ = str(organ or "unknown")
        entry = self._spent.setdefault(
            organ, {"tokens": 0.0, "cost": 0.0, "seconds": 0.0})
        entry["tokens"] += _as_float(tokens)
        entry["cost"] += _as_float(cost)
        entry["seconds"] += _as_float(seconds) + _as_float(time)
        return {"organ": organ, "day": self._day, **dict(entry)}

    def spent(self, organ: str = "") -> dict[str, Any]:
        """Spend so far today: per-organ when ``organ`` is given, else
        the daily totals across all organs."""
        self._rollover()
        if organ:
            entry = self._spent.get(str(organ), {})
            return {"organ": str(organ), "day": self._day,
                    "tokens": entry.get("tokens", 0.0),
                    "cost": entry.get("cost", 0.0),
                    "seconds": entry.get("seconds", 0.0)}
        totals = {"tokens": 0.0, "cost": 0.0, "seconds": 0.0}
        for entry in self._spent.values():
            for key in totals:
                totals[key] += entry.get(key, 0.0)
        return {"organ": "*", "day": self._day, **totals}

    def remaining(self, organ: str = "") -> dict[str, Any]:
        """Ceiling minus spend. ``None`` where no ceiling is set."""
        spent = self.spent(organ)
        ceilings: dict[str, float | None] = {"tokens": None, "cost": None,
                                             "seconds": None}
        if organ:
            caps = self.per_organ_budgets.get(str(organ), {})
            ceilings = {"tokens": caps.get("tokens") or None,
                        "cost": caps.get("cost") or None,
                        "seconds": caps.get("seconds") or None}
        else:
            ceilings = {"tokens": self.daily_token_ceiling,
                        "cost": self.daily_cost_ceiling,
                        "seconds": self.daily_seconds_ceiling}
        return {
            "organ": organ or "*",
            "tokens": (None if ceilings["tokens"] is None
                       else ceilings["tokens"] - spent["tokens"]),
            "cost": (None if ceilings["cost"] is None
                     else ceilings["cost"] - spent["cost"]),
            "seconds": (None if ceilings["seconds"] is None
                        else ceilings["seconds"] - spent["seconds"]),
        }

    # ── gate ─────────────────────────────────────────────────────

    def check(self, organ: str) -> dict[str, Any]:
        """Allowed / deferred verdict for ``organ`` before work runs."""
        self._rollover()
        organ = str(organ or "unknown")
        organ_spent = self.spent(organ)
        day_spent = self.spent()
        reasons: list[str] = []

        daily = (("tokens", self.daily_token_ceiling),
                 ("cost", self.daily_cost_ceiling),
                 ("seconds", self.daily_seconds_ceiling))
        for key, ceiling in daily:
            if ceiling is not None and day_spent[key] >= ceiling:
                reasons.append(
                    f"daily {key} ceiling reached "
                    f"({day_spent[key]:.1f}/{ceiling:.1f})")
        caps = self.per_organ_budgets.get(organ, {})
        for key in ("tokens", "cost", "seconds"):
            cap = caps.get(key) or 0.0
            if cap and organ_spent[key] >= cap:
                reasons.append(
                    f"organ {organ!r} {key} budget reached "
                    f"({organ_spent[key]:.1f}/{cap:.1f})")
        allowed = not reasons
        return {
            "allowed": allowed,
            "status": "allowed" if allowed else "deferred",
            "reason": "; ".join(reasons) if reasons else "within budget",
            "organ": organ,
            "day": self._day,
            "spent": organ_spent,
            "day_spent": day_spent,
            "remaining": self.remaining(organ),
        }

    # ── preemption ───────────────────────────────────────────────

    def preempt(self, low_priority: str, *,
                preempted_by: str = "budget-ceiling",
                reason: str = "") -> dict[str, Any]:
        """Log that ``low_priority`` work was preempted (displaced) so
        higher-priority work could run under the ceiling. Returns the
        log record. Logging is the governor's job; the caller owns the
        actual displacement (defer / cancel / pause)."""
        record = {
            "id": f"preempt-{len(self.preemptions) + 1}-{int(self._clock())}",
            "ts": self._clock(),
            "day": self._day,
            "victim": str(low_priority),
            "preempted_by": str(preempted_by),
            "reason": reason or "over budget ceiling — deferred for higher-priority work",
        }
        self.preemptions.append(record)
        _log.info("budget preemption: %s displaced by %s (%s)",
                  record["victim"], record["preempted_by"], record["reason"])
        return dict(record)

    def preemptions_for(self, organ: str = "") -> list[dict[str, Any]]:
        if not organ:
            return [dict(r) for r in self.preemptions]
        return [dict(r) for r in self.preemptions
                if r["victim"] == str(organ)
                or r["preempted_by"] == str(organ)]

    def reset(self) -> None:
        """Clear today's spend + preemption log (tests / operator reset)."""
        self._spent = {}
        self.preemptions = []
        self._day = self._today()
