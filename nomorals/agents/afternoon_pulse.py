"""Afternoon pulse — SUPERSEDED by the multi-slot briefing engine.

AA-44 consolidation (2026-11-05): this module's standalone pipeline
(news → market snapshot → LLM compose → deliver) and its registrar
were a parallel system for the same job.  The live engine is
:mod:`nomorals.agents.morning_pulse`: ``BRIEFING_SLOTS`` carries the
afternoon slot (job ``afternoon-pulse``, default 13:00, anchored like
every slot), ``ensure_pulse_jobs`` registers it at partner boot, and
the ``pulse`` tool dispatches ``run_slot(context, "afternoon")``.
Evidence at consolidation time: the partner runtime imported only
``morning_pulse`` (runtime.py ``ensure_pulse_jobs``), and this
module's job payload (``{"tool": "pulse", "args": {"action":
"run_afternoon"}}``) named an action the pulse tool cannot dispatch
(its vocabulary is run|status|config|catchup with ``slot=``), so the
job this registrar wrote could never have run.

Unique-remainder check: none.  The live afternoon composer covers
the same ground (fresh headlines, the AFTERNOON_SYMBOLS market
snapshot — the same symbol set this module used — and realtalk),
from the shared delivery ledger, so slots never repeat each other.

What remains here is a documented delegation shim: the historical
``ensure_afternoon_pulse_job`` / ``run_afternoon_pulse`` names keep
working by delegating to the live engine (the registrar of record),
and the flat-pref helpers stay for import compatibility.  This
module registers NOTHING on its own.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

PULSE_JOB_NAME = "afternoon-pulse"
DEFAULT_PULSE_TIME = "13:00"
PULSE_TIMEZONE = "America/Denver"

PULSE_POLICIES = {
    "missed_fire_policy": "fire_now",
    "overlap_policy": "skip",
    "run_timeout_s": 1800,
}


def pulse_time(context: Any) -> str:
    prefs = _prefs(context)
    return str(prefs.get("afternoon_pulse_time") or DEFAULT_PULSE_TIME)


def pulse_timezone(context: Any) -> str:
    prefs = _prefs(context)
    return str(prefs.get("pulse_timezone") or PULSE_TIMEZONE)


def pulse_enabled(context: Any) -> bool:
    prefs = _prefs(context)
    return bool(prefs.get("afternoon_pulse_enabled", True))


def _prefs(context: Any) -> dict[str, Any]:
    try:
        from .morning_pulse import _prefs as _morning_prefs
        return _morning_prefs(context)
    except Exception:
        return {}


def ensure_afternoon_pulse_job(context: Any) -> dict[str, Any]:
    """Register the afternoon briefing job — via the live engine.

    Delegates to ``morning_pulse.ensure_slot_job(context,
    "afternoon")``: the one registrar of record for every briefing
    slot.  Idempotent; the payload it writes is the pulse tool's real
    vocabulary (``action=run, slot=afternoon``).
    """
    from .morning_pulse import ensure_slot_job

    return ensure_slot_job(context, "afternoon")


def run_afternoon_pulse(context: Any) -> dict[str, Any]:
    """Run the afternoon briefing — via the live engine's slot runner."""
    from .morning_pulse import run_slot

    return run_slot(context, "afternoon")
