"""Watch loop (wave 83): the resident operator.

``nm watch`` keeps the system on its feet between human check-ins:

* every tick re-scans the ops surfaces (stuck missions with the
  self-heal escalation, chronic roles, unrecovered pivots) and pages
  the operator through the notifier when something new is wrong;
* every tick redelivers the durable pending-alert queue the moment a
  platform is reachable again (the "2 undelivered in the queue" gap
  from one-shot scans closes itself);
* every tick re-mines the reasoning journal, so new repeated aborts
  become prevention skills without a human running --learn.

One line per tick on the console (or --json for machine feeds).  Run it
with systemd/Termux:ac/supervisor like any daemon; ``--once`` is a
single tick for cron.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any

#: Durable scheduler job + tool for the resident loop (audit gap R2-22).
WATCH_LOOP_JOB = "watch loop tick"
WATCH_LOOP_TOOL = "__watch_loop_tick__"
#: Live cadence — the dominant resident-tick cadence in this runtime
#: (group rhythm, delegation manager, unified scheduler all ~5m). The
#: module's 60s default governs the standalone daemon's run() loop
#: only; the scheduler tool runs one tick per job fire.
WATCH_LOOP_SPEC = "every 5m"

__all__ = ["WatchLoop", "WATCH_LOOP_JOB", "WATCH_LOOP_TOOL",
           "WATCH_LOOP_SPEC", "ensure_watch_loop_job"]


class WatchLoop:
    """One resident loop over the ops + alert + learning surfaces."""

    def __init__(self, context: Any, *, interval: float = 60.0,
                 heal_background: bool = True) -> None:
        self.context = context
        self.interval = max(1.0, float(interval))
        self.heal_background = heal_background
        self.ticks = 0

    def tick(self) -> dict[str, Any]:
        """One sweep of everything the operator would check by hand."""
        from .notifier import Notifier
        from .ops_alerts import ops_alerts
        from .reasoning import ReasoningAgent

        report = ops_alerts(self.context, heal_background=self.heal_background)
        redelivered = 0
        try:
            redelivered = Notifier(self.context).redeliver()
        except Exception:  # noqa: BLE001 - delivery is best-effort
            pass
        mined = 0
        try:
            mined = len(ReasoningAgent(self.context).mine_preventions())
        except Exception:  # noqa: BLE001 - learning is best-effort
            pass
        workspace = ""
        try:
            ws = getattr(self.context, "workspace", None)
            if ws is not None:
                workspace = ws.summary_line()
                ws.autoscale_tick()
        except Exception:  # noqa: BLE001 - the farm report is best-effort
            pass
        self.ticks += 1
        return {
            "tick": self.ticks,
            "ts": round(time.time(), 1),
            "paged": [f["title"] for f in report.get("sent", [])],
            "self_healed": report.get("self_healed", []),
            "cooldown": len(report.get("cooldown", [])),
            "findings": len(report.get("findings", [])),
            "redelivered": redelivered,
            "mined": mined,
            "pending": report.get("pending", 0),
            "workspace": workspace,
        }

    @staticmethod
    def format(result: dict[str, Any]) -> str:
        parts = [
            f"tick {result['tick']} — {len(result['paged'])} paged, "
            f"{len(result['self_healed'])} self-heal(s), "
            f"{result['cooldown']} in cooldown",
        ]
        if result["redelivered"]:
            parts.append(f"{result['redelivered']} alert(s) redelivered")
        if result["mined"]:
            parts.append(f"{result['mined']} prevention skill(s) mined")
        if result["pending"]:
            parts.append(f"{result['pending']} still in the delivery queue")
        if result.get("workspace"):
            parts.append(result["workspace"])
        if not (result["paged"] or result["self_healed"]
                or result["redelivered"] or result["mined"]
                or result["pending"]):
            parts[0] += " — all quiet"
        return "; ".join(parts)

    def run(self, *, max_ticks: int | None = None, once: bool = False,
            emit=None) -> int:
        """Run until interrupted (or max_ticks/once).  Returns 0.

        ``emit(result)`` receives each tick report (defaults to the
        formatted line on stdout).
        """
        import sys

        out = emit or (lambda r: print(WatchLoop.format(r), flush=True))
        try:
            while True:
                started = time.time()
                out(self.tick())
                if once or (max_ticks is not None
                            and self.ticks >= max_ticks):
                    break
                time.sleep(max(1.0, self.interval - (time.time() - started)))
        except KeyboardInterrupt:  # pragma: no cover  # noqa: E106 - deliberate top-level shutdown
            print(f"\nstopped after {self.ticks} tick(s)", file=sys.stderr)
        return 0


def ensure_watch_loop_job(scheduler: Any, *,
                          spec: str = WATCH_LOOP_SPEC) -> Any:
    """Register the durable watch-loop tick job. Idempotent — mirrors
    ``ensure_group_digest_job``: an existing job at the requested
    cadence is returned as-is, a stale one is replaced.
    Best-effort: never raises."""
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == WATCH_LOOP_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        return scheduler.add(WATCH_LOOP_JOB, spec, "tool",
                             {"tool": WATCH_LOOP_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
