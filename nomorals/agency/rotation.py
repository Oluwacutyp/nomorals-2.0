"""Rotation: HOW all AI systems take turns (owner refinement).

This is the centerpiece. Every AI system — digestion, wisdom,
research, Impossible Made Possible, planner, and the domain loops
(deals/weather/news/social/leads/gigs) — works on its own, and they
share one capacity pool through this scheduler:

* **Weighted fair queuing** (virtual time): each system accrues
  virtual time proportional to ``cost / weight`` per turn. The
  scheduler always picks the runnable system with the LOWEST virtual
  time. Shares are respected proportionally; nobody starves, because
  a system that hasn't run has the lowest virtual time.
* **Capacity sharing**: ``allocate_cycle`` splits one cycle budget
  by weight, clamped to per-system ``[min_share, max_share]`` — one
  system can never hog the whole cycle.
* **Yielding**: a system that blocks (external wait, no useful work
  mid-turn) calls ``yield_slot`` and hands its remaining slice to the
  next system. Yielding advances its virtual time so it doesn't get
  picked again immediately.
* **Preemption**: an urgent set (overdue-critical initiatives,
  deadline pressure) jumps the queue — checked before rotation order.
  Preemption is bounded: urgency must be real (caller-supplied set),
  not a priority label a system grants itself.
* **Standing down**: a system with no useful work is NOT picked and
  consumes NO budget. Standing down is recorded, not punished — and
  it must not manufacture busywork to look active. Empty runs back
  off: ``cooldown_after_empty`` consecutive useless turns put the
  system on cooldown; real useful work resets the streak.
* **Duplicate-work prevention**: ``claim_work`` / ``complete_work``
  guard work keys (e.g. ``gig:<id>``). A key runs in one system at a
  time; after completion it stays reserved for ``dedupe_ttl_s`` so a
  later rotation doesn't blindly redo it.

The scheduler is deterministic and dependency-free: it decides WHO
runs, never WHAT the work means. Tick callables live in
:mod:`nomorals.agency.systems`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SystemSlot", "WorkResult", "RotationScheduler"]


@dataclass
class SystemSlot:
    name: str
    weight: float = 1.0
    min_share: float = 0.05  # guaranteed floor of a cycle budget
    max_share: float = 0.5   # hogging cap: never more than half a cycle
    cooldown_after_empty: int = 3
    cooldown_s: float = 1800.0
    virtual_time: float = 0.0
    empty_streak: int = 0
    cooldown_until: float = 0.0
    last_run: float = 0.0


@dataclass
class WorkResult:
    useful: bool
    work_key: str = ""
    cost: float = 1.0  # abstract cost units consumed by this turn
    knowledge_items: int = 0


class RotationScheduler:
    def __init__(self, *, dedupe_ttl_s: float = 86400.0,
                 claim_ttl_s: float = 3600.0) -> None:
        self._slots: dict[str, SystemSlot] = {}
        self._yields: dict[str, int] = {}
        self._stand_downs: dict[str, int] = {}
        self._preemptions: int = 0
        self.cycles_run: int = 0
        self._active_work: dict[str, tuple[str, float]] = {}
        self._done_work: dict[str, float] = {}
        self.dedupe_ttl_s = dedupe_ttl_s
        self.claim_ttl_s = claim_ttl_s

    # ── registration ───────────────────────────────────────────────
    def register(self, slot: SystemSlot) -> None:
        self._slots[slot.name] = slot

    def slot(self, name: str) -> SystemSlot:
        return self._slots[name]

    def names(self) -> list[str]:
        return list(self._slots)

    # ── picking: preemption → fair rotation → stand-down ───────────
    def next_system(self, *, runnable: set[str] | frozenset[str],
                    urgent: set[str] | frozenset[str] | None = None,
                    now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        for name in self._slots:
            if name not in runnable:
                self._stand_downs[name] = self._stand_downs.get(name, 0) + 1
        candidates = [self._slots[n] for n in runnable
                      if n in self._slots and self._slots[n].cooldown_until <= now]
        if not candidates:
            return None
        urgent_set = set(urgent or set()) & {c.name for c in candidates}
        if urgent_set:
            self._preemptions += 1
            return min((c for c in candidates if c.name in urgent_set),
                       key=lambda c: c.virtual_time).name
        return min(candidates, key=lambda c: c.virtual_time).name

    def yield_slot(self, name: str, *, reason: str = "") -> None:
        """Voluntary yield: hand the rest of this slice to the next system."""
        slot = self._slots.get(name)
        if slot is None:
            return
        self._yields[name] = self._yields.get(name, 0) + 1
        # Advance virtual time so the yielder is not re-picked at once.
        slot.virtual_time += 2.0 / max(0.1, slot.weight)

    def yields(self, name: str) -> int:
        return self._yields.get(name, 0)

    def stand_downs(self, name: str) -> int:
        return self._stand_downs.get(name, 0)

    @property
    def preemptions(self) -> int:
        return self._preemptions

    # ── result accounting: fairness + empty-run backoff ─────────────
    def record_result(self, name: str, result: WorkResult, *,
                      now: float | None = None) -> None:
        now = time.time() if now is None else now
        slot = self._slots.get(name)
        if slot is None:
            return
        slot.last_run = now
        slot.virtual_time += max(0.0, result.cost) / max(0.1, slot.weight)
        if result.useful:
            slot.empty_streak = 0
            slot.cooldown_until = 0.0
        else:
            slot.empty_streak += 1
            if slot.empty_streak >= slot.cooldown_after_empty:
                slot.cooldown_until = now + slot.cooldown_s

    # ── capacity sharing ───────────────────────────────────────────
    def allocate_cycle(self, *, total_budget: float) -> dict[str, float]:
        total_w = sum(s.weight for s in self._slots.values()) or 1.0
        alloc: dict[str, float] = {}
        for name, s in self._slots.items():
            raw = total_budget * s.weight / total_w
            alloc[name] = min(max(raw, total_budget * s.min_share),
                              total_budget * s.max_share)
        over = sum(alloc.values()) - total_budget
        if over > 0:
            # Scale down only the above-minimum portion, preserving floors.
            floors = {n: total_budget * self._slots[n].min_share for n in alloc}
            headroom = sum(alloc[n] - floors[n] for n in alloc) or 1.0
            for n in alloc:
                alloc[n] = floors[n] + (alloc[n] - floors[n]) * max(
                    0.0, (headroom - over) / headroom)
        return alloc

    # ── duplicate-work prevention ──────────────────────────────────
    def claim_work(self, key: str, *, owner: str, now: float) -> bool:
        active = self._active_work.get(key)
        if active is not None:
            _owner, ts = active
            if now - ts < self.claim_ttl_s:
                return False  # already running elsewhere
            del self._active_work[key]  # stale claim: owner died
        done_at = self._done_work.get(key)
        if done_at is not None and now - done_at < self.dedupe_ttl_s:
            return False  # completed recently: do not blindly redo
        self._active_work[key] = (owner, now)
        return True

    def complete_work(self, key: str, *, owner: str, now: float) -> bool:
        active = self._active_work.get(key)
        if active is None or active[0] != owner:
            return False
        del self._active_work[key]
        self._done_work[key] = now
        return True
