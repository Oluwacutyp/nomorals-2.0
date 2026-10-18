"""Owner views + reconciled agenda (audit §8).

Three views, separate rhythms (execution / deliberation /
notification / external-communication cadences are NOT the same):
quiet mode sends fewer messages — it does NOT suspend initiative.

* Now: what is running, why it matters, what is waiting.
* Since last check: outcomes, changed decisions, blockers, costs.
* Why: evidence → objective → plan → actions → verification → receipt.

Every day ends with a reconciled agenda, not a fresh forgotten list.
Count closed loops and valuable outcomes, not autonomous messages.
"""
from __future__ import annotations

import time
from typing import Any

from .initiatives import InitiativeStore

__all__ = ["now_view", "since_view", "why_view", "reconciled_agenda"]


def now_view(store: InitiativeStore, *, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    active = store.list_active()
    running = [i for i in active if i.lease_owner]
    waiting = [i for i in active if i.blocker]
    return {
        "running": [{"outcome": i.outcome, "why": i.objective,
                     "owner": i.lease_owner, "next": i.next_action}
                    for i in running],
        "waiting": [{"outcome": i.outcome, "blocker": i.blocker,
                     "next_review_at": i.next_review_at} for i in waiting],
        "queued": len(active) - len(running) - len(waiting),
        "total_active": len(active),
    }


def since_view(store: InitiativeStore, *, since_ts: float) -> dict[str, Any]:
    events = [e for e in store.events() if e["ts"] >= since_ts]
    by_kind: dict[str, int] = {}
    for e in events:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    completed = [e for e in events if e["kind"] == "completed"]
    return {"events": len(events), "by_kind": by_kind,
            "outcomes": [e["detail"] for e in completed],
            "blockers": [e["detail"] for e in events if e["kind"] == "deferred"]}


def why_view(store: InitiativeStore, initiative_id: str) -> dict[str, Any]:
    it = store.get(initiative_id)
    if it is None:
        raise KeyError(initiative_id)
    lineage = store.lineage(initiative_id)
    events = store.events(initiative_id)
    return {
        "evidence": it.evidence,
        "objective": it.objective,
        "plan": it.next_action,
        "actions": [e for e in events if e["kind"] in ("advanced", "claimed")],
        "verification": it.verification,
        "receipt": it.verification,
        "lineage": [p.outcome for p in lineage],
        "stop_reason": it.stop_reason,
        "status": it.status,
    }


def reconciled_agenda(store: InitiativeStore, *,
                      now: float | None = None) -> dict[str, Any]:
    """End-of-day reconciliation: open, closed, carried-over — nothing forgotten."""
    now = time.time() if now is None else now
    all_items = store.list_all()
    open_items = [i for i in all_items if i.status not in ("completed", "cancelled")]
    closed = [i for i in all_items if i.status == "completed"]
    return {
        "open": len(open_items),
        "closed": len(closed),
        "closed_loops": [{"outcome": i.outcome, "stop": i.stop_reason,
                          "verification": i.verification} for i in closed],
        "carried_over": [{"outcome": i.outcome, "next": i.next_action,
                          "blocker": i.blocker, "review": i.next_review_at}
                         for i in open_items],
        "reconciled_at": now,
    }
