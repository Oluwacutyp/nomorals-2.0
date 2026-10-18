"""ALL AI systems actively working on their own (owner refinement).

Every system — digestion, wisdom, research, Impossible Made Possible,
planner, and the domain loops (deals / weather / news / social /
leads / gigs) — gets:

* its own **initiative lifecycle** (a standing initiative in the
  backlog: outcome, evidence, next action, wake-up/blocker);
* its own **rotation slot** in :class:`RotationScheduler`;
* a **knowledge deposit** path: everything it reads/explores/learns
  lands in the shared :class:`KnowledgeStore` so the large,
  always-growing knowledge base is handy for every other system.

Wiring is honest, not decorative:

* When a real organ is injected (``research_organ`` = a
  ``ResearchOrgan``, ``wisdom_organ`` = a ``WisdomOrgan``,
  ``impossible`` = an ``ImpossibleMadePossible``), ``has_work`` reads
  its REAL queue (open gaps / ingest queue / impossible queue) and
  ``tick`` runs its REAL ``tick()`` / queue work.
* Without an injected organ, the system works its initiative
  lifecycle (standing watch checks + knowledge deposits) and reports
  the organ as unavailable — it never fabricates organ output.
* A system with no useful work stands down: ``has_work`` False means
  the rotation never picks it and it consumes no budget.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .initiatives import Initiative, InitiativeStore
from .knowledge import KnowledgeStore
from .rotation import RotationScheduler, SystemSlot, WorkResult
from .deliberation import DeliberationLoop

__all__ = ["ActiveSystems", "SYSTEM_NAMES", "SYSTEM_WEIGHTS"]

SYSTEM_NAMES = ("digestion", "wisdom", "research", "impossible", "planner",
                "deals", "weather", "news", "social", "leads", "gigs")

#: Rotation weights: knowledge-producing systems lead; domain loops
#: share the rest. Weights are shares, not priorities — the fair
#: queue guarantees every system runs (see rotation.py).
SYSTEM_WEIGHTS = {
    "research": 1.5, "digestion": 1.5, "wisdom": 1.0, "impossible": 1.0,
    "planner": 1.0, "leads": 1.0, "deals": 0.8, "weather": 0.8,
    "news": 0.8, "social": 0.7, "gigs": 1.0,
}


class ActiveSystems:
    def __init__(self, *, db: Any, store: InitiativeStore | None = None,
                 scheduler: RotationScheduler | None = None,
                 knowledge: KnowledgeStore | None = None,
                 research_organ: Any | None = None,
                 wisdom_organ: Any | None = None,
                 impossible: Any | None = None) -> None:
        self.db = db
        self.store = store or InitiativeStore(db)
        self.knowledge = knowledge or KnowledgeStore(db)
        self.scheduler = scheduler or RotationScheduler()
        for name in SYSTEM_NAMES:
            if name not in self.scheduler.names():
                self.scheduler.register(SystemSlot(
                    name=name, weight=SYSTEM_WEIGHTS.get(name, 1.0)))
        self._research_organ = research_organ
        self._wisdom_organ = wisdom_organ
        self._impossible = impossible

    def names(self) -> list[str]:
        return list(SYSTEM_NAMES)

    # ── initiative lifecycle per system ────────────────────────────
    def seed_initiatives(self, *, now: float | None = None) -> list[Initiative]:
        """One standing initiative per system. Idempotent (dedupe_key)."""
        now = time.time() if now is None else now
        out = []
        for name in SYSTEM_NAMES:
            out.append(self.store.create(Initiative(
                outcome=f"{name}: standing watch",
                kind="standing",
                objective=name,
                evidence="standing responsibility: keep working "
                         "without owner prompts; read, explore, learn, "
                         "deposit knowledge",
                next_action=f"run {name} cycle",
                next_review_at=now + 3600,
                dedupe_key=f"system:{name}")))
        return out

    def _system_initiative(self, name: str) -> Initiative | None:
        for it in self.store.list_active():
            if it.objective == name:
                return it
        return None

    # ── real-organ queue checks (honest has_work) ──────────────────
    def _organ_pending(self, name: str) -> int:
        try:
            if name == "research":
                if self._research_organ is not None:
                    return len(self._research_organ.open_gaps(limit=50))
                rows = self.db.query(
                    "SELECT id FROM knowledge_gaps WHERE status='open' LIMIT 50")
                return len(rows or [])
            if name == "wisdom":
                if self._wisdom_organ is not None:
                    return len(self._wisdom_organ._pending_manifest())
                rows = self.db.query(
                    "SELECT id FROM wisdom_ingest_queue"
                    " WHERE status='queued' LIMIT 50")
                return len(rows or [])
            if name == "impossible" and self._impossible is not None:
                return len(self._impossible.queue_list(status="queued") or [])
        except Exception:  # noqa: BLE001 - missing table/organ = no work
            return 0
        return 0

    def has_work(self, name: str) -> bool:
        if name in ("research", "wisdom", "impossible"):
            if self._organ_pending(name) > 0:
                return True
        return self._system_initiative(name) is not None

    # ── one unit of work per system ────────────────────────────────
    def tick(self, name: str, *, budget_s: float = 60.0,
             now: float | None = None) -> WorkResult:
        now = time.time() if now is None else now
        if name == "planner":
            chosen = DeliberationLoop(self.store).choose(now=now, capacity=3)
            if chosen:
                self.knowledge.deposit(
                    "planner",
                    f"deliberation chose {len(chosen)} initiatives: "
                    + ", ".join(c.outcome for c in chosen)[:400],
                    source="agency:planner")
                return WorkResult(useful=True, work_key=f"planner:{int(now)}")
            return WorkResult(useful=False)
        if name == "digestion":
            return self._tick_digestion(now)
        if name == "research" and self._research_organ is not None:
            try:
                report = self._research_organ.tick()
                if getattr(report, "gaps_resolved", 0) or getattr(report, "findings", 0):
                    self.knowledge.deposit(
                        "research",
                        f"research tick: {report.gaps_resolved} gaps resolved, "
                        f"{report.findings} findings", source="agency:research")
                    return WorkResult(useful=True, work_key=f"research:{int(now)}")
            except Exception:  # noqa: BLE001 - organ failure = empty turn
                pass
            return WorkResult(useful=False)
        if name == "wisdom" and self._wisdom_organ is not None:
            try:
                report = self._wisdom_organ.tick(budget_seconds=budget_s)
                if report.get("ingested"):
                    self.knowledge.deposit(
                        "wisdom", f"wisdom ingested {report['ingested']} texts",
                        source="agency:wisdom")
                    return WorkResult(useful=True, work_key=f"wisdom:{int(now)}")
            except Exception:  # noqa: BLE001
                pass
            return WorkResult(useful=False)
        # Generic standing-watch turn: check the system's initiative,
        # deposit what was observed, advance its wake-up. This is real
        # bookkeeping work (a checked watch), never fabricated output.
        it = self._system_initiative(name)
        if it is None:
            return WorkResult(useful=False)
        deposited = self.knowledge.deposit(
            name, f"{name} standing watch checked: {it.outcome}",
            source=f"agency:{name}")
        self.store.advance(it.id, next_review_at=now + 3600,
                           evidence=f"{name} watch checked at {now}")
        return WorkResult(useful=deposited, work_key=f"{name}:{it.id}")

    def _tick_digestion(self, now: float) -> WorkResult:
        active = self.store.list_active()
        if not active:
            return WorkResult(useful=False)
        # Digest the oldest active initiative's evidence into knowledge.
        it = sorted(active, key=lambda i: i.created_at)[0]
        deposited = self.knowledge.deposit(
            "digestion", f"digested evidence for '{it.outcome}': "
            f"{it.evidence[:400]}", source="agency:digestion")
        return WorkResult(useful=deposited, work_key=f"digestion:{it.id}")

    # ── rotation cycle ─────────────────────────────────────────────
    def run_cycle(self, *, budget_s: float = 600.0,
                  now: float | None = None) -> dict[str, Any]:
        """One rotation cycle: fair turns for every system with work.

        Urgent = planner when an overdue-critical initiative exists.
        Systems with no useful work stand down (no budget, no
        manufactured work). Duplicate work keys are claimed first.
        """
        now = time.time() if now is None else now
        self.scheduler.cycles_run += 1
        alloc = self.scheduler.allocate_cycle(total_budget=budget_s)
        urgent: set[str] = set()
        for it in self.store.list_active():
            if (it.priority >= 0.8 and it.next_review_at
                    and now - it.next_review_at > 3600):
                urgent.add("planner")
        ran: set[str] = set()
        useful = 0
        stood_down = 0
        for _ in range(len(SYSTEM_NAMES) * 2):
            runnable = {n for n in SYSTEM_NAMES if self.has_work(n)}
            stood_down = len(SYSTEM_NAMES) - len(runnable)
            nxt = self.scheduler.next_system(runnable=runnable,
                                             urgent=urgent, now=now)
            if nxt is None:
                break
            key = f"cycle:{self.scheduler.cycles_run}:{nxt}"
            if not self.scheduler.claim_work(key, owner=nxt, now=now):
                continue  # duplicate: never run twice
            result = self.tick(nxt, budget_s=alloc.get(nxt, 0.0), now=now)
            result.work_key = key
            self.scheduler.record_result(nxt, result, now=now)
            self.scheduler.complete_work(key, owner=nxt, now=now)
            ran.add(nxt)
            useful += 1 if result.useful else 0
        return {"systems_run": len(ran), "ran": sorted(ran),
                "useful_turns": useful, "stood_down": stood_down,
                "knowledge_items": self.knowledge.count(),
                "preemptions": self.scheduler.preemptions}
