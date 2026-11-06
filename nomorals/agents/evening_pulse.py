"""Evening pulse — SUPERSEDED by the multi-slot briefing engine.

AA-44 consolidation (2026-11-05): this module's standalone pipeline
(day's news → market close → LLM compose → deliver) and its
registrar were a parallel system for the same job.  The live engine
is :mod:`nomorals.agents.morning_pulse`: ``BRIEFING_SLOTS`` carries
the evening slot (job ``evening-pulse``, default 19:00, anchored
like every slot), ``ensure_pulse_jobs`` registers it at partner
boot, and the ``pulse`` tool dispatches ``run_slot(context,
"evening")``.  Evidence at consolidation time: the partner runtime
imported only ``morning_pulse`` (runtime.py ``ensure_pulse_jobs``),
and this module's job payload (``{"tool": "pulse", "args":
{"action": "run_evening"}}``) named an action the pulse tool cannot
dispatch (its vocabulary is run|status|config|catchup with
``slot=``), so the job this registrar wrote could never have run.

Unique remainders — both carried into the live engine before this
module retired, so nothing the evening wrap did is lost:

* the F7 alert-digest fold (this module's formatting note): the
  evening wrap was the second daily fold of the DIGEST tier.  It
  now lives in ``morning_pulse.run_slot`` via
  ``_fold_evening_digest`` / ``_mark_evening_digest_folded`` —
  rendered through the shared ``alert_digest_lines``, marked folded
  only on confirmed delivery.
* tomorrow's Lagos weather in the Tomorrow outlook: now appended by
  ``morning_pulse._tomorrow_preview``.

What remains here is a documented delegation shim: the historical
``ensure_evening_pulse_job`` / ``run_evening_pulse`` names keep
working by delegating to the live engine (the registrar of record),
and the flat-pref helpers stay for import compatibility.  This
module registers NOTHING on its own.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

PULSE_JOB_NAME = "evening-pulse"
DEFAULT_PULSE_TIME = "21:00"
PULSE_TIMEZONE = "America/Denver"

PULSE_POLICIES = {
    "missed_fire_policy": "fire_now",
    "overlap_policy": "skip",
    "run_timeout_s": 1800,
}


def pulse_time(context: Any) -> str:
    prefs = _prefs(context)
    return str(prefs.get("evening_pulse_time") or DEFAULT_PULSE_TIME)


def pulse_timezone(context: Any) -> str:
    prefs = _prefs(context)
    return str(prefs.get("pulse_timezone") or PULSE_TIMEZONE)


def pulse_enabled(context: Any) -> bool:
    prefs = _prefs(context)
    return bool(prefs.get("evening_pulse_enabled", True))


def _prefs(context: Any) -> dict[str, Any]:
    try:
        from .morning_pulse import _prefs as _morning_prefs
        return _morning_prefs(context)
    except Exception:
        return {}


def ensure_evening_pulse_job(context: Any) -> dict[str, Any]:
    """Register the evening briefing job — via the live engine.

    Delegates to ``morning_pulse.ensure_slot_job(context,
    "evening")``: the one registrar of record for every briefing
    slot.  Idempotent; the payload it writes is the pulse tool's real
    vocabulary (``action=run, slot=evening``).
    """
    from .morning_pulse import ensure_slot_job

    return ensure_slot_job(context, "evening")


def run_evening_pulse(context: Any) -> dict[str, Any]:
    """Run the evening briefing — via the live engine's slot runner."""
    from .morning_pulse import run_slot

    return run_slot(context, "evening")
