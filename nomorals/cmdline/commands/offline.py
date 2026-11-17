"""``nm offline`` — the offline-fallback state from the console.

Read-only. Surfaces the real durable state through
``agents.offline_host.summarize_offline_state``: the persisted offline
queue (rows exist only after a failed send), the durable 5-minute
sync job, the BrainGate held/recovered episodes from the failure
ledger, and the degraded-capability matrix. There is deliberately no
drain verb — the 5-minute tick already drains the queue; status is
the whole scope.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_offline"]


def _age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown age"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"
    return f"{int(seconds // 86400)}d"


def _render_status(summary: dict[str, Any]) -> str:
    q = summary["queue"]
    job = summary["sync_job"]
    bg = summary["braingate"]
    lines = ["offline:"]
    kinds = ", ".join(f"{k}: {v}" for k, v in sorted(q["by_kind"].items()))
    lines.append(
        f"  queue: {q['pending']} pending, {q['failed']} failed, "
        f"{q['synced']} synced, {q['total']} total"
        + (f" ({kinds})" if kinds else ""))
    if q["oldest_pending_id"]:
        lines.append(
            f"  oldest pending: {q['oldest_pending_id']} "
            f"({_age(q['oldest_pending_age_s'])} old)")
    if q["error"]:
        lines.append(f"  queue read error: {q['error']}")
    if job["scheduled"]:
        lines.append(
            f"  sync job: scheduled ({job['spec']})"
            + ("" if job["enabled"] else " — disabled")
            + (f", next run {job['next_run_iso']}"
               if job["next_run_iso"] else ", next run not recorded"))
    else:
        lines.append("  sync job: not scheduled")
    if job["error"]:
        lines.append(f"  sync job read error: {job['error']}")
    lines.append(
        f"  brain gate: {bg['episodes']} episode(s), {bg['open']} open, "
        f"{bg['held_closed']} held in closed episodes"
        + (" — the open episode's held count is unknown"
           if bg["held_open_unknown"] else ""))
    if bg["error"]:
        lines.append(f"  brain gate read error: {bg['error']}")
    lines.append("  when offline:")
    for row in summary["matrix"]:
        lines.append(f"    {row['feature']}: {row['level']} — {row['notes']}")
    for note in summary.get("notes") or []:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def _render_queue(summary: dict[str, Any], queue: Any) -> str:
    try:
        actions = list(queue.all())
    except Exception as exc:  # noqa: BLE001 - a broken queue lists as empty
        return f"queue read error: {type(exc).__name__}: {exc}"
    if not actions:
        return "queue empty — no actions waiting"
    lines = ["offline queue:"]
    for action in actions:
        entity = getattr(action, "entity", "") or ""
        error = getattr(action, "error", "") or ""
        lines.append(
            f"  {getattr(action, 'id', '?')} · {getattr(action, 'kind', '?')} "
            f"· {getattr(action, 'status', '?')}"
            + (f" · {entity}" if entity else "")
            + (f" · {error}" if error else ""))
    return "\n".join(lines)


def _cmd_offline(args: argparse.Namespace, context: Any) -> int:
    """``nm offline [status|queue]``."""
    from ...agents.offline_host import summarize_offline_state
    from ...offline import OfflineQueue

    action = (getattr(args, "offline_action", None) or "status").lower()
    if action not in ("status", "queue"):
        print("usage: nm offline [status|queue]", file=sys.stderr)
        return 2
    queue = getattr(context, "offline_queue", None)
    if queue is None:
        queue = OfflineQueue()
    scheduler = getattr(context, "offline_scheduler", None)
    if scheduler is None:
        try:
            from ...agents.scheduler import Scheduler

            scheduler = Scheduler(context)
        except Exception:  # noqa: BLE001 - scheduler read is best-effort
            scheduler = None
    summary = summarize_offline_state(queue=queue, scheduler=scheduler)
    if bool(getattr(args, "json", False)):
        print(json.dumps(summary, indent=2))
        return 0
    if action == "queue":
        print(_render_queue(summary, queue))
    else:
        print(_render_status(summary))
    return 0
