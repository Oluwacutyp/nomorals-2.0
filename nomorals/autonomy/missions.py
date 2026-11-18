"""Live work on the unified scheduler (audit gap R1-11 remainder).

The R1-11 ship built the unified scheduler's 5-minute driver, but
nothing ever moved onto it: ``register_organ_tick`` had zero
production callers, nothing ever called ``schedule_mission``, and no
handler was ever registered at boot. This module is the first live
load on that scheduler — one migrated organ tick, one mission
producer, one boot-registered handler:

* **Migrated tick — WAL checkpoint.** The hourly ``wal checkpoint
  tick`` durable job on the live scheduler is retired (see
  ``remove_wal_checkpoint_job`` in ``storage.db``) and the same work
  runs as the ``wal-checkpoint`` organ tick on the boot-built stack.
  The tick carries a durable last-run marker under ``<home>/autonomy``
  so two runtimes sharing one home still checkpoint at most once an
  hour: the marker, not in-memory ``next_due``, is the claim.
* **Producer — due commitments.** The commitment ledger already knew
  how to nudge (``CommitmentLedger.nudges``) and nothing in production
  ever called it. The ``commitment-producer`` organ tick scans the
  ledger hourly and schedules one ``notify_owner`` mission per due
  commitment. Dedupe is by commitment id against the DURABLE queue
  (the stack refreshes from disk before producing, so a second
  runtime sees the first one's pending mission): a pending or running
  mission blocks a second, a mission completed within the last day
  blocks a repeat nudge, anything else is a fresh nudge.
* **Handler — notify the owner.** ``notify_owner`` delivers a mission
  step's ``(kind, title, body)`` through the notifier the runtime
  passes in at boot. Handlers are callables, not imports: this module
  never imports the notifier (or the ledger's storage) — the runtime
  injects its collaborators, so the module stays testable with fakes
  and the layering stays downward (``commitments`` is layer 0).

Exactly-once, stated plainly: the old durable job is removed, so the
checkpoint fires through the unified path only; the marker makes the
tick once-an-hour even across two runtimes; the producer dedupes
against the durable queue, so a commitment gets one pending nudge
mission at a time; a done mission is never re-fired (the scheduler
itself guarantees that). What stays at-least-once, by the scheduler's
own design, is a mission killed mid-run: it stays ``running`` and a
later tick retries it — a nudge delivered twice after a crash is
better than a nudge never delivered.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..commitments.ledger import nudge_text
from ..storage.db import remove_wal_checkpoint_job
from .unified import atomic_write_json

__all__ = [
    "WAL_TICK_NAME",
    "WAL_TICK_INTERVAL_S",
    "PRODUCER_TICK_NAME",
    "PRODUCER_TICK_INTERVAL_S",
    "NOTIFY_HANDLER_NAME",
    "NUDGE_REPEAT_AFTER_S",
    "commitment_mission",
    "produce_commitment_missions",
    "notify_owner_handler",
    "make_wal_checkpoint_tick",
    "register_boot_autonomy",
]

#: Organ-tick name for the migrated WAL checkpoint.
WAL_TICK_NAME = "wal-checkpoint"
#: The old durable job ran hourly; the tick keeps the same cadence.
WAL_TICK_INTERVAL_S = 3600.0
#: Organ-tick name for the commitment producer.
PRODUCER_TICK_NAME = "commitment-producer"
#: The producer scans the ledger hourly.
PRODUCER_TICK_INTERVAL_S = 3600.0
#: Mission-plan handler name for owner delivery.
NOTIFY_HANDLER_NAME = "notify_owner"
#: A completed nudge is not repeated inside this window.
NUDGE_REPEAT_AFTER_S = 24 * 3600.0
#: Default scan window for due commitments (matches the ledger).
DUE_WINDOW_S = 86400.0


# ── missions ─────────────────────────────────────────────────────────


def commitment_mission(commitment: Any, text: str, *,
                       now: float | None = None) -> dict[str, Any]:
    """Build the nudge mission for one due commitment.

    Pure dict-building: the plan is a single ``notify_owner`` step
    carrying the nudge text, the commitment id rides in the metadata
    so the producer can dedupe on it.
    """
    ts = float(now if now is not None else time.time())
    what = str(getattr(commitment, "what", "") or "")
    promisee = str(getattr(commitment, "promisee", "") or "")
    who = f" for {promisee}" if promisee else ""
    return {
        "goal": f"Remind the owner: {what}{who}",
        "name": f"commitment-nudge:{getattr(commitment, 'id', '')}",
        "plan": [{"name": NOTIFY_HANDLER_NAME,
                  "handler": NOTIFY_HANDLER_NAME,
                  "kind": "commitment",
                  "title": f"Commitment due{who}: {what}",
                  "body": text}],
        "priority": "scheduled",
        "organ": "commitments",
        "due_at": ts,
        "metadata": {"commitment_id": str(getattr(commitment, "id", "")),
                     "kind": "commitment-nudge"},
    }


def _nudge_blocked(existing: list[dict[str, Any]], commitment_id: str,
                   now: float) -> bool:
    """True when a mission for this commitment already covers it: a
    pending or running one blocks forever (until it resolves); a done
    one blocks repeats inside the daily window."""
    for mission in existing:
        metadata = mission.get("metadata") or {}
        if metadata.get("commitment_id") != commitment_id:
            continue
        status = str(mission.get("status") or "")
        if status in ("pending", "running"):
            return True
        if status == "done":
            created = float(mission.get("created_at") or 0.0)
            if now - created < NUDGE_REPEAT_AFTER_S:
                return True
    return False


def produce_commitment_missions(stack: Any, ledger: Any, *,
                                within_seconds: float = DUE_WINDOW_S,
                                now: float | None = None) -> list[str]:
    """Schedule one nudge mission per due commitment, deduplicated.

    The stack refreshes the durable queue from disk FIRST, so a
    second runtime's producer sees the first runtime's pending
    mission instead of scheduling a duplicate. Returns the ids of
    the missions this call scheduled (empty when nothing was due or
    everything was already covered). Never raises for a bad ledger
    row — one unbuildable commitment is skipped, the rest still
    produce.
    """
    ts = float(now if now is not None else time.time())
    scheduler = stack.scheduler
    scheduler.refresh()
    existing = scheduler.all_missions()
    try:
        due = ledger.due_soon(within_seconds, ts)
    except Exception:
        return []
    produced: list[str] = []
    for commitment in due or []:
        cid = str(getattr(commitment, "id", "") or "")
        if not cid or _nudge_blocked(existing, cid, ts):
            continue
        try:
            text = nudge_text(commitment, ts)
        except Exception:
            continue
        mission = commitment_mission(commitment, text, now=ts)
        produced.append(scheduler.schedule_mission(mission))
        existing.append(scheduler.get_mission(produced[-1]) or mission)
    return produced


# ── handler ──────────────────────────────────────────────────────────


def notify_owner_handler(
        notify_fn: Callable[[str, str, str], Any]) -> Callable[..., Any]:
    """Build the ``notify_owner`` mission handler around a notify
    callable ``(kind, title, body) -> result`` (the runtime passes a
    closure over its Notifier). The step's payload carries the text;
    the mission's goal is the fallback. A raising notify fails the
    step — recorded on the mission, never silent."""

    def _handle(mission: dict[str, Any],
                step: dict[str, Any]) -> Any:
        kind = str(step.get("kind") or "commitment")
        title = str(step.get("title")
                    or mission.get("name") or "mission")
        body = str(step.get("body") or mission.get("goal") or "")
        return notify_fn(kind, title, body)

    return _handle


# ── the migrated WAL tick ─────────────────────────────────────────────


def _read_marker(path: Path) -> float | None:
    """Last-run timestamp from the marker file, or None when there is
    none / it is unreadable — fail-open: a missing marker means run."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return float(raw.get("last_run_ts"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _write_marker(path: Path, now: float) -> None:
    """Record this run's timestamp, atomically. A failure here must
    never fail the checkpoint that already ran — swallow it."""
    with contextlib.suppress(Exception):
        atomic_write_json(path, {"last_run_ts": float(now)})


def make_wal_checkpoint_tick(db: Any, *, marker_path: str | Path,
                             clock: Callable[[], float] | None = None,
                             ) -> Callable[[], Any]:
    """Build the WAL-checkpoint organ tick around a workspace DB.

    The tick runs ``checkpoint_maintenance`` (PASSIVE first, TRUNCATE
    only when no reader pins the log) unless the durable marker says
    another runtime already ran it inside the interval — then it
    reports a skip and touches nothing. The marker is the cross-stack
    claim; ``next_due`` alone is per-stack memory and cannot be."""

    tick_clock = clock or time.time
    marker = Path(marker_path)

    def _tick() -> dict[str, Any]:
        now = float(tick_clock())
        last = _read_marker(marker)
        if last is not None and now - last < WAL_TICK_INTERVAL_S:
            return {"ok": True, "skipped": True,
                    "reason": ("checkpoint ran "
                               f"{now - last:.0f}s ago on this home; "
                               "next in "
                               f"{WAL_TICK_INTERVAL_S - (now - last):.0f}s")}
        report = db.checkpoint_maintenance()
        _write_marker(marker, now)
        return {"ok": True, **(report if isinstance(report, dict)
                               else {"result": report})}

    return _tick


# ── boot wiring ──────────────────────────────────────────────────────


def register_boot_autonomy(stack: Any, *, db: Any, ledger: Any,
                           notify_fn: Callable[[str, str, str], Any],
                           scheduler: Any = None,
                           clock: Callable[[], float] | None = None,
                           ) -> dict[str, Any]:
    """Register the unified scheduler's first live load on a
    boot-built stack. Idempotent: re-registering replaces the ticks
    and the handler instead of doubling them.

    * Retires the old durable ``wal checkpoint tick`` job from the
      live scheduler (when one is passed) — the unified tick is the
      only firing path from here on.
    * Registers the ``wal-checkpoint`` organ tick (hourly, storage).
    * Registers the ``notify_owner`` mission handler.
    * Registers the ``commitment-producer`` organ tick (hourly,
      commitments) — the producer scan itself.

    Both ticks fire on the next driver pass after boot (``start_now``
    one interval in the past), then hourly. ``clock`` defaults to the
    stack's own clock, so a test stack and the production stack agree
    on "now".
    """
    tick_clock = clock or getattr(stack.scheduler, "_clock", None) or time.time
    now = float(tick_clock())
    removed_old_job = None
    if scheduler is not None:
        removed_old_job = remove_wal_checkpoint_job(scheduler)
    marker_path = Path(stack.base_dir) / "autonomy" / "wal_checkpoint_last_run.json"
    stack.scheduler.register_organ_tick(
        WAL_TICK_NAME,
        make_wal_checkpoint_tick(db, marker_path=marker_path,
                                 clock=tick_clock),
        WAL_TICK_INTERVAL_S,
        priority="routine", organ="storage",
        start_now=now - WAL_TICK_INTERVAL_S)
    stack.scheduler.register_handler(
        NOTIFY_HANDLER_NAME, notify_owner_handler(notify_fn))

    def _produce() -> list[str]:
        return produce_commitment_missions(stack, ledger, now=tick_clock())

    stack.scheduler.register_organ_tick(
        PRODUCER_TICK_NAME, _produce, PRODUCER_TICK_INTERVAL_S,
        priority="routine", organ="commitments",
        start_now=now - PRODUCER_TICK_INTERVAL_S)
    return {"ok": True, "wal_tick": WAL_TICK_NAME,
            "producer_tick": PRODUCER_TICK_NAME,
            "handler": NOTIFY_HANDLER_NAME,
            "old_job_removed": removed_old_job}
