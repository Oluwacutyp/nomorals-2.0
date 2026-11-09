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

from .failures import record_failure
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
        """Idempotent watch re-registration after restart.

        Registration is REAL, not bookkeeping (F12): when a scheduler
        facade is injected, each new watch is handed to it —
        ``facade.register_watch(watch)`` when the facade offers that
        method, else the agent-scheduler shape the agency wiring uses
        (``facade.add(name, spec, payload_kind, payload)``, which
        needs the watch to carry a ``spec``/``schedule``/``cron``).
        Only a facade-ACCEPTED watch counts in the return value and
        is marked registered; a rejected/errored watch stays
        unregistered so a later boot retries it. Before this, the
        count was incremented from the supervisor's own set while the
        scheduler received zero calls. Without a facade there is no
        scheduler to call: the supervisor's own registry is the
        record, and the count is tracked-only.
        """
        count = 0
        for w in watches or []:
            wid = w.get("id") or w.get("topic", "")
            if not wid or wid in self._registered_watches:
                continue
            if self.facade is not None \
                    and not self._register_with_facade(w):
                continue
            self._registered_watches.add(wid)
            count += 1
        return count

    def _register_with_facade(self, watch: dict[str, Any]) -> bool:
        """One real registration call into the injected facade."""
        facade = self.facade
        register = getattr(facade, "register_watch", None)
        if callable(register):
            try:
                return bool(register(watch))
            except Exception as exc:  # noqa: BLE001 - refused = retry later
                record_failure("BootSupervisor.register_watch", exc)
                return False
        add = getattr(facade, "add", None)
        if callable(add):
            spec = watch.get("spec") or watch.get("schedule") \
                or watch.get("cron")
            if not spec:
                return False  # this facade cannot schedule a spec-less watch
            name = watch.get("name") or watch.get("id") \
                or watch.get("topic", "")
            try:
                add(name, spec, watch.get("payload_kind", "tool"),
                    watch.get("payload") or {"watch": watch})
                return True
            except Exception as exc:  # noqa: BLE001 - refused = retry later
                record_failure("BootSupervisor.register_watch.add", exc)
                return False
        return False

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
