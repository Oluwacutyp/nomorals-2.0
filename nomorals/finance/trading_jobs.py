"""Durable scheduler registration for Devon's trading system
(Pass 6).

Sibling of ``finance/jobs.py`` / ``finance/price_jobs.py``, same
``ensure_*`` pattern: two jobs whose payloads call the ``trading``
tool — the pipeline tick and the audit cycle — so the Pass 1–5
organs run on cadence without a parallel scheduler. The jobs live
in the Scheduler's DB store: they survive restarts, and
re-registration is idempotent by name.

Jobs:

* ``trading pipeline`` — the shadow-pipeline tick (market read →
  setup lifecycle → forecast → judgment shadow → management
  poll). Default ``every 15m``; ``settings.trading
  .pipeline_tick_spec`` overrides. Registered with
  ``overlap_policy="skip"`` and the tick itself is slot-keyed
  idempotent, so a doubled fire can never double-act.
* ``trading audit`` — the validation cycle (retrain the trained
  forecast estimators, Pass 5 audit, rung comparison). Default
  ``every 6h``; ``settings.trading.pipeline_audit_spec``
  overrides.

Both marked ``routine``: their output lands in the trading
journal and the pipeline status surface, not in owner alerts —
but a FAILED run still records as a failed run (pass #71
semantics). Start/stop is NOT job deletion: the pipeline's
control state lives in the trading journal (agents/trade_pipeline)
and the tick honors it, so the cadence registration stays stable
across control changes and restarts.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_trading_jobs", "TRADING_PIPELINE_JOB",
           "TRADING_AUDIT_JOB"]

TRADING_PIPELINE_JOB = "trading pipeline"
TRADING_AUDIT_JOB = "trading audit"

#: Mirror agents.trade_pipeline's defaults (kept literal here so
#: this finance module needs no agents import; the pipeline reads
#: the same settings fields when it runs).
_DEFAULT_TICK_SPEC = "every 15m"
_DEFAULT_AUDIT_SPEC = "every 6h"


def _spec(settings: Any, field: str, default: str) -> str:
    trading = getattr(settings, "trading", None)
    value = getattr(trading, field, None)
    return str(value) if value else default


def ensure_trading_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the trading pipeline + audit jobs.

    Idempotent by name. Returns one status dict per job.
    """
    settings = getattr(getattr(scheduler, "context", None),
                       "settings", None)
    jobs = (
        (TRADING_PIPELINE_JOB,
         _spec(settings, "pipeline_tick_spec", _DEFAULT_TICK_SPEC),
         {"tool": "trading", "args": {"action": "pipeline_tick"}}),
        (TRADING_AUDIT_JOB,
         _spec(settings, "pipeline_audit_spec", _DEFAULT_AUDIT_SPEC),
         {"tool": "trading", "args": {"action": "pipeline_audit"}}),
    )
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for name, spec, payload in jobs:
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(name, spec, "tool", payload, routine=True,
                            overlap_policy="skip")
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out
