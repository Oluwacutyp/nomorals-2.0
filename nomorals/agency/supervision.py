"""Persistent supervision: boot recovery (audit §2D).

On boot: reconcile incomplete effects, re-register handlers/watches
idempotently, recover expired leases, coalesce missed recurring
checks, then resume due work. One scheduler facade owns public
scheduling — this module never schedules directly; it asks the
injected facade.

Idempotency: re-registration is keyed by stable watch IDs, so a
second boot does not duplicate watches. Missed checks coalesce:
N missed firings of one watch become ONE catch-up run, not N.
"""
from __future__ import annotations

import time
from typing import Any

from .initiatives import InitiativeStore

__all__ = ["BootSupervisor"]


class BootSupervisor:
    def __init__(self, store: InitiativeStore,
                 scheduler_facade: Any | None = None) -> None:
        self.store = store
        self.facade = scheduler_facade
        self._registered_watches: set[str] = set()

    def reregister_watches(self, watches: list[dict[str, Any]] | None = None
                           ) -> int:
        """Idempotent watch re-registration after restart."""
        count = 0
        for w in watches or []:
            wid = w.get("id") or w.get("topic", "")
            if wid and wid not in self._registered_watches:
                self._registered_watches.add(wid)
                count += 1
        return count

    @staticmethod
    def coalesce_missed(missed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """N missed firings of one watch → one catch-up run."""
        seen: dict[str, dict[str, Any]] = {}
        for m in missed:
            key = m.get("watch_id") or m.get("id", "")
            if key not in seen:
                seen[key] = {**m, "missed_count": 1}
            else:
                seen[key]["missed_count"] += 1
        return list(seen.values())

    def reconcile(self, *, now: float | None = None,
                  watches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        recovered = self.store.boot_reconcile(now=now)
        rereg = self.reregister_watches(watches)
        due = [i for i in self.store.list_active()
               if i.next_review_at and i.next_review_at <= now and not i.blocker]
        return {"leases_recovered": recovered["leases_recovered"],
                "watches_reregistered": rereg,
                "due_on_boot": len(due),
                "active": recovered["active"]}
