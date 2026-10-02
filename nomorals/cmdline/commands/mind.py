"""``nm mind`` — cognition surfaces."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any



def _cmd_mind(args: argparse.Namespace, context: Any) -> int:
    """`nm mind [status]` — inspect CoreMind's persisted state.

    Shows pending clarification questions, recent routed jobs with
    outcomes, the last objective, and the persisted router telemetry
    (per-route decision counts, model-check counts, last plan_error —
    written by the running bot via ``nomorals.storage.router_telemetry``,
    migration 65).
    """
    from ...agents.coremind import CoreMind
    from ...storage.router_telemetry import snapshot as telemetry_snapshot

    mind = CoreMind(context)
    state = mind._state  # persisted state.json: pending / jobs / last_objective
    now = time.time()

    def _age(ts: float) -> str:
        secs = max(0.0, now - float(ts or 0))
        if secs < 90:
            return f"{secs:.0f}s ago"
        if secs < 5400:
            return f"{secs / 60:.0f}m ago"
        if secs < 172800:
            return f"{secs / 3600:.1f}h ago"
        return f"{secs / 86400:.1f}d ago"

    pending_raw: dict[str, Any] = state.get("pending") or {}
    pending: list[dict[str, Any]] = []
    for chat_key, p in pending_raw.items():
        created = float(p.get("created", 0) or 0)
        stale = (now - created) > CoreMind.PENDING_TTL
        pending.append({"chat": chat_key, "kind": p.get("kind"),
                        "question": p.get("question"), "age": _age(created),
                        "stale": stale})

    jobs_raw: list[dict[str, Any]] = list(mind._jobs)[-10:]
    jobs = [{"id": j.get("id"), "kind": j.get("kind"),
             "target": (j.get("target") or "")[:70],
             "route": j.get("route"), "status": j.get("status"),
             "note": (j.get("note") or "")[:90],
             "age": _age(j.get("created", 0))}
            for j in jobs_raw]
    last_objective = state.get("last_objective")
    telemetry = telemetry_snapshot(getattr(context, "db", None))
    route_counts = telemetry.get("routes") or {}
    last_plan_error = telemetry.get("last_plan_error") or {}
    last_reeval = telemetry.get("last_reevaluation") or {}

    payload = {
        "command": "mind",
        "pending_clarifications": pending,
        "recent_jobs": jobs,
        "last_objective": last_objective,
        # Persisted router telemetry (migration 65) — written by the
        # running bot on every route decision, readable from any process.
        "router_calls": {
            "per_route": route_counts,
            "total": sum(route_counts.values()),
            "model_consults": telemetry.get("model_consults", 0),
            "model_timeouts": telemetry.get("model_timeouts", 0),
        },
        "last_plan_error": last_plan_error or None,
        "last_reevaluation": last_reeval or None,
    }

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
        return 0

    print(f"coremind state: {mind.state_file or 'no state file'}")
    print(f"pending clarifications: {len(pending)}")
    for p in pending:
        stale_mark = " (stale)" if p["stale"] else ""
        print(f"  [{p['kind']}] {p['chat']}: {p['question']}{stale_mark} "
              f"— {p['age']}")
    if not pending:
        print("  none")
    print(f"recent jobs: {len(jobs_raw)} kept (showing {len(jobs)})")
    for j in jobs:
        print(f"  {j['id']} [{j['status']}] {j['kind']} → {j['route']}: "
              f"{j['target']} — {j['age']}"
              + (f" — {j['note']}" if j["note"] else ""))
    if not jobs:
        print("  none")
    if last_objective:
        print(f"last objective: {last_objective.get('text')} "
              f"(route={last_objective.get('route')}, "
              f"{_age(last_objective.get('created', 0))})")
    else:
        print("last objective: none")
    if route_counts:
        top = sorted(route_counts.items(), key=lambda kv: -kv[1])[:8]
        print(f"router calls: {sum(route_counts.values())} total "
              f"(model consulted {telemetry.get('model_consults', 0)}×, "
              f"timed out {telemetry.get('model_timeouts', 0)}×)")
        for route, n in top:
            print(f"  {route}: {n}×")
    else:
        print("router calls: none recorded yet")
    if last_plan_error:
        print(f"last plan_error ({_age(last_plan_error.get('at', 0))}, "
              f"route={last_plan_error.get('route') or 'n/a'}): "
              f"{last_plan_error.get('error')}")
    else:
        print("last plan_error: none recorded")
    if last_reeval:
        print(f"last re-evaluation ({_age(last_reeval.get('at', 0))}): "
              f"{last_reeval.get('action')} — {last_reeval.get('reason')}")
    else:
        print("last re-evaluation: none recorded")
    return 0
