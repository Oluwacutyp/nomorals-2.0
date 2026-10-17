"""Always-on digest — the idle-driven tick, budgeted, question-led.

Pattern mirrors autonomy/coordinator.py (idle-driven ticks, budget plan,
backoff) without duplicating it: the digest runs ONE tick per call, reads
open problems from the queue, ingests prior art for the most-ready problem,
runs a transfer search, and records progress. No motor, no hoarding:
ingest to answer, not to hoard. A problem worked N cycles with no delta
goes cold (see attempt.AttemptQueue.mark_cold).

Budget: max_seconds + max_problems per tick. Profile-gating happens at the
caller (power/ budgets) — this module never deletes capability for a small
device, it just does less per tick.
"""

from __future__ import annotations

import time

from .attempt import AttemptQueue
from .ledger import HonestyLedger


def digest_tick(queue: AttemptQueue | None = None, ledger: HonestyLedger | None = None,
                max_seconds: float = 60.0, max_problems: int = 2,
                prior_art_fn=None) -> dict:
    """One always-on tick. Returns what moved (problems moved, not docs ingested)."""
    started = time.time()
    queue = queue or AttemptQueue()
    ledger = ledger or HonestyLedger()
    moved: list[dict] = []
    open_items = [i for i in queue.queue("open")]
    for item in open_items[:max_problems]:
        if time.time() - started > max_seconds:
            break
        entry = ledger.get(item["claim_id"])
        before = len((entry or {}).get("attempts", []))
        # Digest: fresh prior-art pass for this open problem.
        if prior_art_fn is not None:
            try:
                art = prior_art_fn(item["problem"])
                new_hits = sum(1 for v in (art.get("results") or {}).values()
                               for r in v if r.get("status") == "found")
            except Exception as exc:  # noqa: BLE001 — digest must never crash the idle loop
                new_hits = 0
                art = {"summary": f"prior-art pass failed: {exc}"}
        else:
            art = {"summary": "no prior-art function wired — tick skipped ingestion"}
            new_hits = 0
        after = len((ledger.get(item["claim_id"]) or {}).get("attempts", []))
        delta = (new_hits > 0) or (after != before)
        if delta:
            moved.append({"claim_id": item["claim_id"], "problem": item["problem"],
                          "new_prior_art_hits": new_hits,
                          "prior_art_summary": art.get("summary", "")})
        else:
            # No delta this cycle — count toward dormancy.
            item["cycles_no_delta"] = item.get("cycles_no_delta", 0) + 1
            if item["cycles_no_delta"] >= 3:
                queue.mark_cold(item["claim_id"], cycles=item["cycles_no_delta"])
    return {"tick_seconds": round(time.time() - started, 2),
            "problems_checked": min(len(open_items), max_problems),
            "problems_moved": moved,
            "metric": "problems moved (not documents ingested)"}


def register_with_coordinator(coordinator) -> dict:
    """Wire the digest tick into an IdleCoordinator-like object, if it
    supports tick registration. Returns what was wired (honest if none)."""
    for attr in ("register_tick", "add_tick", "subscribe_tick"):
        fn = getattr(coordinator, attr, None)
        if callable(fn):
            fn("impossible_digest", digest_tick)
            return {"wired": True, "via": attr}
    return {"wired": False,
            "reason": "Coordinator exposes no tick-registration hook — call digest_tick() from the idle loop directly."}
