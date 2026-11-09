"""Deliberation loop, SEPARATE from execution (audit §2C).

Cheap deterministic checks for ordinary scheduling; the planner is
invoked (by the caller) only when evidence changes the decision, a
plan fails, a deadline approaches, or capacity becomes available.

Ranking (conceptual, audit §2C):
  goal contribution + urgency + expected outcome + information value
  − effort − cash cost − disruption − duplication.
Scores are estimates, calibrated from outcomes — never presented as
objective expected value.

Guards:
* Exploration gets a bounded share of capacity (default 20%).
* Waiting items age: overdue critical work is never starved by
  research/exploration.
* Never start more commitments than capacity allows.
"""
from __future__ import annotations

import time
from typing import Any

from .initiatives import Initiative, InitiativeStore

__all__ = ["DeliberationLoop", "score"]

#: Max share of a deliberation batch that may be exploration.
EXPLORATION_SHARE = 0.20
#: Overdue past this is "critical" — it outranks everything else.
OVERDUE_CRITICAL_S = 3600.0


def score(it: Initiative, *, now: float) -> float:
    overdue = max(0.0, now - it.next_review_at) if it.next_review_at else 0.0
    urgency = 0.0
    if it.deadline:
        remaining = max(1.0, it.deadline - now)
        urgency = min(2.0, 86400.0 / remaining)
    if overdue:
        urgency += min(2.0, overdue / 3600.0)
    expected = it.benefit * (1.0 - it.uncertainty) * max(0.1, it.priority)
    info_value = 0.3 if it.kind == "exploration" else 0.0
    cost = it.effort * 0.2 + it.cost * 0.1
    # Ageing: waiting items gain score so nothing starves silently.
    age_bonus = min(1.0, overdue / 86400.0)
    return expected + urgency + info_value + age_bonus - cost


class DeliberationLoop:
    """Chooses what to start / continue / defer / stop this cycle."""

    def __init__(self, store: InitiativeStore,
                 exploration_share: float = EXPLORATION_SHARE) -> None:
        self.store = store
        self.exploration_share = exploration_share

    def choose(self, *, now: float | None = None,
               capacity: int = 3) -> list[Initiative]:
        now = time.time() if now is None else now
        if capacity <= 0:
            return []
        active = self.store.list_active()
        # Blocked initiatives are not executable — but they stay visible
        # with their wake-up; deliberation never drops them. Neither
        # are initiatives under a live lease: another entrypoint owns
        # them this cycle (F03), so choosing them would double-work.
        # Nor is a FUTURE WAIT executable (F12): an initiative with no
        # next action whose only plan is a wake-up in the future has
        # nothing to start yet — selecting it early starts nothing and
        # crowds out work that is actually due. (An initiative WITH a
        # next action is workable now even if its review is future.)
        executable = [it for it in active if not it.blocker
                      and not (it.lease_owner and it.lease_until > now)
                      and not (not (it.next_action or "").strip()
                               and it.next_review_at
                               and it.next_review_at > now)]
        blocked = [it for it in active if it.blocker]
        # Starvation guard: overdue-critical first — but capacity is a
        # CONTRACT ("never start more commitments than capacity
        # allows", this module's own header), not a suggestion the
        # critical lane may override (F12: five critical initiatives
        # were returned for capacity=1). Critical work wins the slots
        # by score; the overflow is deferred with its event, like
        # everything else, and its overdue score keeps it first next
        # cycle.
        critical = [it for it in executable
                    if it.next_review_at and now - it.next_review_at > OVERDUE_CRITICAL_S
                    and it.priority >= 0.8]
        critical.sort(key=lambda it: score(it, now=now), reverse=True)
        rest = [it for it in executable if it not in critical]
        rest.sort(key=lambda it: score(it, now=now), reverse=True)
        explore = [it for it in rest if it.kind == "exploration"]
        exploit = [it for it in rest if it.kind != "exploration"]
        chosen: list[Initiative] = critical[:capacity]
        room = max(0, capacity - len(chosen))
        # Exploration's bounded share (F12): the share is a RESERVATION,
        # not a remainder — exploit must not consume the whole batch
        # first and leave exploration with the int() truncation of a
        # small capacity (capacity=3 × 20% truncated to 0, so a lone
        # exploration candidate was selected zero times). When
        # exploration candidates exist and the share is positive, at
        # least one slot is reserved for them; the share still bounds
        # exploration when both lanes compete for a large batch.
        max_explore = int(capacity * self.exploration_share)
        if explore and self.exploration_share > 0:
            max_explore = max(1, max_explore)
        max_explore = min(max_explore, len(explore), room)
        explore_pick = explore[:max_explore]
        room -= len(explore_pick)
        chosen += exploit[:room]
        chosen += explore_pick
        # Any remaining room stays unfilled: the share is a bound, and
        # an under-filled batch is honest — capacity is a ceiling on
        # commitments started, not a quota to fill.
        # Deferred set is implicit: everything active not chosen. Record
        # the decision cheaply so "Why?" can explain it — and record the
        # SELECTIONS too. Before this, only deferrals were logged: the
        # choice itself evaporated, so a deliberation that changed nothing
        # else was indistinguishable from one that never ran.
        for it in chosen:
            self.store._event(it.id, "selected",
                              f"deliberation chose for this cycle "
                              f"(capacity={capacity})")
        for it in active:
            if it not in chosen:
                self.store._event(it.id, "deferred",
                                  f"not chosen this cycle; blocked={bool(it.blocker)}")
        return chosen

    def stop_stale(self, *, now: float | None = None,
                   max_age_s: float = 30 * 86400) -> list[Initiative]:
        """Cancel plans whose deadline passed or whose value expired."""
        now = time.time() if now is None else now
        stopped = []
        for it in self.store.list_active():
            if it.deadline and it.deadline < now:
                stopped.append(self.store.cancel(
                    it.id, stop_reason="expired: deadline passed"))
            elif it.created_at and now - it.created_at > max_age_s:
                stopped.append(self.store.cancel(
                    it.id, stop_reason="expired: stale, no progress"))
        return stopped
