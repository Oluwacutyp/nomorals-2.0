"""Commerce dunning driver: the scheduled tick that retries failed charges.

Register §8 L969: "subscriptions are records, not live billing, dunning
has no scheduled driver yet." ``PaymentGateway.dunning_due()`` lists the
subscriptions whose retry time has arrived, ``record_failed_charge()``
advances the 1/3/7-day schedule, ``record_successful_charge()`` clears
it — but nothing ever called any of them. This module is the driver:

* :func:`run_dunning_tick_once` — one tick: load the durable commerce
  store, charge every due subscription through the real gateway,
  journal every attempt (``charged`` / ``failed_retry`` / ``past_due`` /
  ``skipped_no_key`` / ``skipped_bad_record``) into the store's dunning
  journal, and explain the firing into the why store when the
  boot-built autonomy stack is on the context (duck-typed — commerce
  is L4, autonomy is L5, so no import crosses the layer).
* :func:`ensure_dunning_job` — one durable "dunning tick" job (daily
  09:00), idempotent by name, replacing a stale spec.

Honesty rules (the payments module's own):
* a missing provider key is a WALL, not a failed charge — the attempt
  is skipped, the wall is named (the env var), and the retry is NOT
  burned;
* a bad record (amount <= 0, unknown provider) is likewise skipped,
  never charged;
* a live charge needs the provider's real key AND a real HTTP
  round-trip — without either the gateway raises and the tick records
  the failure, never a fabricated success.

Stdlib only, besides the commerce organs themselves.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .payments import PaymentError
from .store import CommerceStore, default_commerce_path

_log = get_logger(__name__)

__all__ = [
    "DUNNING_JOB",
    "DUNNING_SPEC",
    "DUNNING_TOOL",
    "ensure_dunning_job",
    "run_dunning_tick_once",
]

DUNNING_JOB = "dunning tick"
DUNNING_SPEC = "daily 09:00"
DUNNING_TOOL = "__dunning_tick__"

#: Outcomes the tick counts (and journals) per subscription.
#: Journal outcome → result count key (a failed retry counts as "failed").
_OUTCOME_COUNTS: dict[str, str] = {
    "charged": "charged",
    "failed_retry": "failed",
    "past_due": "past_due",
    "skipped_no_key": "skipped_no_key",
    "skipped_bad_record": "skipped_bad_record",
}


def _home(context: Any) -> Path:
    """The nomorals home for this context (settings, else the default)."""
    settings = getattr(context, "settings", None) if context is not None else None
    home = getattr(settings, "home", None) or getattr(
        settings, "home_path", None) or "~/.nomorals"
    return Path(str(home)).expanduser()


def _explain(context: Any, result: dict[str, Any]) -> None:
    """Explain this firing into the why store, when one is on the context.

    The boot-built autonomy stack rides on ``context.extras`` (the same
    seam the unified-scheduler tick tool reads). Duck-typed on purpose:
    commerce must not import autonomy. When boot never ran there is no
    stack — the tick still runs and the durable scheduler still journals
    the run itself; the why record is best-effort, never fatal.
    """
    try:
        stack = (getattr(context, "extras", None) or {}).get("autonomy_stack")
        why = getattr(stack, "why_store", None) if stack is not None else None
        if why is None:
            return
        evidence = {key: result.get(key, 0)
                    for key in ("charged", "failed", "past_due",
                                "skipped_no_key", "skipped_bad_record")}
        evidence["attempted"] = result.get("attempted", 0)
        did = (f"ran dunning over {result.get('attempted', 0)} due "
               f"subscription(s)")
        happened = (f"charged {result.get('charged', 0)}, "
                    f"failed {result.get('failed', 0)}, "
                    f"past_due {result.get('past_due', 0)}, "
                    f"skipped_no_key {result.get('skipped_no_key', 0)}, "
                    f"skipped_bad_record "
                    f"{result.get('skipped_bad_record', 0)}")
        why.record("dunning tick", evidence=evidence, did=did,
                   happened=happened)
    except Exception:  # noqa: BLE001 - explanation never breaks execution
        _log.debug("dunning why record failed", exc_info=True)


def run_dunning_tick_once(context: Any, *, http: Any = None,
                          now: Any = None) -> dict[str, Any]:
    """Run one dunning tick. Never raises.

    ``http`` is the gateway's HTTP client (``http(method, url, headers,
    json_body)``). None builds the live HttpClient adapter — production.
    Tests inject a fake. ``now`` is an injectable ``() -> float`` clock.
    """
    clock = now or time.time
    result: dict[str, Any] = {
        "ok": True, "attempted": 0, "charged": 0, "failed": 0,
        "past_due": 0, "skipped_no_key": 0, "skipped_bad_record": 0,
        "outcomes": [],
    }
    try:
        store = CommerceStore(default_commerce_path(_home(context)))
        if http is None:
            from .commands import _gateway_http

            http = _gateway_http()
        gw = store.gateway(http=http, now=clock)
        due = gw.dunning_due()
        for sub in due:
            result["attempted"] += 1
            attempt_no = sub.failed_attempts
            reference = f"{sub.id}-retry-{attempt_no}"
            outcome = ""
            reason = ""
            try:
                gw.require_configured(sub.provider)
            except PaymentError as exc:
                reason = str(exc)
                outcome = ("skipped_no_key" if "secret key missing" in reason
                           else "skipped_bad_record")
            else:
                if sub.amount_minor <= 0:
                    outcome = "skipped_bad_record"
                    reason = (f"amount_minor {sub.amount_minor} is not "
                              "chargeable — refusing to charge a bad record")
                else:
                    try:
                        payment = gw.charge(
                            sub.provider, sub.amount_minor,
                            currency=sub.currency, customer=sub.customer,
                            reference=reference, channel="online")
                    except PaymentError as exc:
                        reason = str(exc)
                        gw.record_failed_charge(sub)
                        outcome = ("past_due" if sub.status == "past_due"
                                   else "failed_retry")
                    else:
                        gw.record_successful_charge(sub)
                        outcome = "charged"
                        reference = payment.reference
                        reason = f"payment {payment.id}"
            result[_OUTCOME_COUNTS[outcome]] += 1
            entry = {
                "id": "dun_" + uuid.uuid4().hex[:12],
                "subscription_id": sub.id,
                "customer": sub.customer,
                "provider": sub.provider,
                "amount_minor": sub.amount_minor,
                "currency": sub.currency,
                "outcome": outcome,
                "attempt": attempt_no,
                "reference": reference,
                "reason": reason,
                "ts": clock(),
            }
            result["outcomes"].append({
                "subscription_id": sub.id, "outcome": outcome,
                "attempt": attempt_no, "reason": reason,
            })
            store.save_gateway(gw)
            store.record_dunning_attempt(entry)
            _log.info("dunning %s %s (%s): %s",
                      sub.id, sub.customer, outcome, reason)
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        return {"ok": False, "error": f"dunning tick failed: {exc}"}
    _explain(context, result)
    return result


def ensure_dunning_job(scheduler: Any, *,
                       spec: str = DUNNING_SPEC) -> Any:
    """Register the durable dunning tick job.

    Idempotent — mirrors ``ensure_delegation_manager_job``: an existing
    job at the requested cadence is returned as-is, a stale one is
    replaced. Best-effort: never raises.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == DUNNING_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        return scheduler.add(DUNNING_JOB, spec, "tool",
                             {"tool": DUNNING_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
