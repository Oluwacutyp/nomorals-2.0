"""Second-wave autonomy: the REST of the repo works on its own.

The sibling agency build wired 11 systems (digestion, wisdom,
research, impossible, planner, deals, weather, news, social, leads,
gigs) into the initiative lifecycle + rotation. This module wires the
remaining systems the same way — media, code, books, games, commerce,
health, travel, property, legal, finance, learn, marketing, community,
characters, documents, datasci, plus ``sourcehealth`` itself — WITHOUT
duplicating the core: it reuses :class:`InitiativeStore`,
:class:`KnowledgeStore`, and :class:`RotationScheduler` from the core
modules, and registers its slots on the SAME scheduler instance the
core systems use, so all systems share one capacity pool.

Wiring is honest, not decorative:

* When a real organ is injected (``organs={"games": <organ>}``), the
  system reads the organ's REAL queue and runs its REAL ``tick()``.
  Queue discovery tries the organ's own API in order —
  ``pending_count()``, ``open_items()``, ``queue_list()``, a
  ``pending`` attribute — and falls back to standing-watch
  bookkeeping when the organ exposes no queue. It never fabricates
  organ output.
* A system with no useful work stands down: ``has_work`` False means
  the rotation never picks it and it consumes no budget.
* ``sourcehealth`` is a first-class system: its tick re-verifies every
  registered source (see :mod:`nomorals.agency.sourcehealth` and
  :mod:`nomorals.agency.sources_registry`), deposits the summary into
  shared knowledge, and stands down dead sources via the circuit
  breaker instead of deleting evidence.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .initiatives import Initiative, InitiativeStore
from .knowledge import KnowledgeStore
from .rotation import RotationScheduler, SystemSlot, WorkResult
from .sourcehealth import SourceHealth

__all__ = ["ExpansionSystems", "EXPANSION_SYSTEMS", "EXPANSION_WEIGHTS"]

EXPANSION_SYSTEMS = (
    "media", "code", "books", "games", "commerce", "health", "travel",
    "property", "legal", "finance", "learn", "marketing", "community",
    "characters", "documents", "datasci", "sourcehealth",
)

#: Rotation weights: media + source-health lead (they feed every other
#: system); the rest share evenly. Weights are shares, not priorities —
#: the fair queue guarantees every system runs (see rotation.py).
EXPANSION_WEIGHTS = {
    "media": 1.2, "sourcehealth": 1.2, "code": 1.0, "books": 1.0,
    "games": 1.0, "commerce": 0.8, "health": 0.8, "travel": 0.8,
    "property": 0.7, "legal": 0.7, "finance": 0.8, "learn": 0.8,
    "marketing": 0.7, "community": 0.7, "characters": 0.7,
    "documents": 0.7, "datasci": 0.7,
}


class ExpansionSystems:
    def __init__(self, *, db: Any,
                 store: InitiativeStore | None = None,
                 scheduler: RotationScheduler | None = None,
                 knowledge: KnowledgeStore | None = None,
                 organs: dict[str, Any] | None = None,
                 source_health: SourceHealth | None = None,
                 registry_collect: Callable[[], list] | None = None,
                 registry_fetcher: Any = None,
                 authority: Any | None = None,
                 explicit_go_aheads: Any | None = None) -> None:
        self.db = db
        self.store = store or InitiativeStore(db)
        self.knowledge = knowledge or KnowledgeStore(db)
        self.scheduler = scheduler or RotationScheduler()
        for name in EXPANSION_SYSTEMS:
            if name not in self.scheduler.names():
                self.scheduler.register(SystemSlot(
                    name=name, weight=EXPANSION_WEIGHTS.get(name, 1.0)))
        self._organs = dict(organs or {})
        self._source_health = source_health
        self._registry_collect = registry_collect
        self._registry_fetcher = registry_fetcher
        #: Standing authority (trial pattern): an expansion organ that
        #: performs a GATED action must go through this authority.
        #: None = fail closed — gated organs never run without one.
        self.authority = authority
        #: Explicit go-aheads for THIS run (trial ``--yes`` pattern):
        #: action ids (``f"{system}:{capability}"``) the owner has
        #: explicitly confirmed, so the tick may proceed. A go-ahead
        #: covers that action only — never a blanket permission.
        self.explicit_go_aheads = frozenset(explicit_go_aheads or ())

    def names(self) -> list[str]:
        return list(EXPANSION_SYSTEMS)

    # ── initiative lifecycle per system ────────────────────────────
    def seed_initiatives(self, *, now: float | None = None) -> list[Initiative]:
        """One standing initiative per system. Idempotent (dedupe_key)."""
        now = time.time() if now is None else now
        out = []
        for name in EXPANSION_SYSTEMS:
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
    def _organ_pending(self, name: str) -> int | None:
        """Pending work on the injected organ, or None if unknowable."""
        organ = self._organs.get(name)
        if organ is None:
            return None
        try:
            if callable(getattr(organ, "pending_count", None)):
                return int(organ.pending_count() or 0)
            if callable(getattr(organ, "open_items", None)):
                return len(organ.open_items() or [])
            if callable(getattr(organ, "queue_list", None)):
                try:
                    return len(organ.queue_list(status="queued") or [])
                except TypeError:
                    return len(organ.queue_list() or [])
            pending = getattr(organ, "pending", None)
            if isinstance(pending, (list, tuple, set, dict)):
                return len(pending)
            if isinstance(pending, int):
                return pending
        except Exception:  # noqa: BLE001 — organ probe failure = no work
            return 0
        return None

    def has_work(self, name: str) -> bool:
        if name == "sourcehealth":
            # Periodic re-verification is itself the work.
            return self._registry_collect is not None or \
                self._system_initiative(name) is not None
        pending = self._organ_pending(name)
        if pending is not None:
            if pending > 0:
                return True
            # Empty organ queue: work only if a standing initiative exists.
            return self._system_initiative(name) is not None
        return self._system_initiative(name) is not None

    # ── gated actions go through standing authority ─────────────
    @staticmethod
    def _gated_capabilities(organ: Any) -> list[str]:
        """Capabilities this organ's tick performs (duck-typed).

        An organ declares them via ``gated_capability`` (one string)
        or ``gated_capabilities()`` / ``gated_capabilities`` (many).
        No declaration = no gated actions = no gate.
        """
        caps: list[str] = []
        single = getattr(organ, "gated_capability", None)
        if isinstance(single, str) and single:
            caps.append(single)
        many = getattr(organ, "gated_capabilities", None)
        if callable(many):
            try:
                many = many()
            except Exception:  # noqa: BLE001 - declaration failure = none
                many = None
        if isinstance(many, (list, tuple, set, frozenset)):
            caps.extend(str(c) for c in many if c)
        return caps

    def _authority_allows(self, name: str, capability: str, *,
                          now: float) -> tuple[bool, str, str]:
        """Consult standing authority for one gated action.

        Returns ``(allowed, reason, proposal_id)``. The trial pattern:
        the action id is ``f"{system}:{capability}"``; an explicit
        go-ahead for THAT action id (owner confirmed this exact
        action) is passed as ``explicit_override``. No authority
        configured = fail closed (denied).
        """
        action_id = f"{name}:{capability}"
        if self.authority is None:
            return False, "gated action requires standing authority", ""
        check = getattr(self.authority, "check", None)
        if not callable(check):
            return False, "authority has no check() method", ""
        context = {"action_id": action_id, "system": name,
                   "capability": capability}
        try:
            decision = check(
                capability, explicit_override=(
                    action_id in self.explicit_go_aheads),
                draft=f"expansion {name}: {capability}", context=context)
        except TypeError:
            # Authority without the trial-pattern kwarg (e.g. a plain
            # AgencyAuthority): go-ahead maps to a direct check.
            decision = check(capability, draft=f"expansion {name}: "
                             f"{capability}", context=context)
        return (bool(getattr(decision, "allowed", False)),
                str(getattr(decision, "reason", "") or ""),
                str(getattr(decision, "proposal_id", "") or ""))

    # ── one unit of work per system ────────────────────────────────
    def tick(self, name: str, *, budget_s: float = 60.0,
             now: float | None = None) -> WorkResult:
        now = time.time() if now is None else now
        if name == "sourcehealth":
            return self._tick_sourcehealth(now)
        organ = self._organs.get(name)
        if organ is not None and callable(getattr(organ, "tick", None)):
            # Gated organs: the tick must pass standing authority
            # BEFORE it runs. Denied = no organ work, blocker +
            # proposal recorded, watch advanced. The organ never runs
            # on a denied gate.
            for capability in self._gated_capabilities(organ):
                allowed, reason, proposal_id = self._authority_allows(
                    name, capability, now=now)
                if not allowed:
                    text = (f"gated action {capability} blocked: {reason}"
                            + (f" (proposal {proposal_id})"
                               if proposal_id else ""))
                    self.knowledge.deposit(
                        name, text, source=f"agency:{name}")
                    it = self._system_initiative(name)
                    if it is not None:
                        self.store.advance(
                            it.id, next_review_at=now + 3600, blocker=text,
                            evidence=text)
                    return WorkResult(useful=False)
            try:
                report = organ.tick()
                done = _report_done(report)
            except Exception:  # noqa: BLE001 — organ failure = empty turn
                return WorkResult(useful=False)
            if done:
                self.knowledge.deposit(
                    name, f"{name} organ tick: {done}",
                    source=f"agency:{name}")
                self._advance_watch(name, now, f"{name} organ tick: {done}")
                return WorkResult(useful=True, work_key=f"{name}:{int(now)}")
            # Organ ran but did nothing useful: still a checked watch.
            self._advance_watch(name, now, f"{name} organ tick: no work done")
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

    def _advance_watch(self, name: str, now: float, evidence: str) -> None:
        it = self._system_initiative(name)
        if it is not None:
            self.store.advance(it.id, next_review_at=now + 3600,
                               evidence=evidence)

    def _tick_sourcehealth(self, now: float) -> WorkResult:
        health = self._source_health
        if health is None:
            health = SourceHealth(self.db)
            self._source_health = health
        if self._registry_collect is None:
            return WorkResult(useful=False)
        try:
            specs = self._registry_collect()
        except Exception:  # noqa: BLE001 — registry failure = empty turn
            return WorkResult(useful=False)
        summary = health.verify_all(specs, fetcher=self._registry_fetcher,
                                    now=now)
        text = ("sourcehealth verified "
                f"{summary['checked']} sources: "
                f"{summary['verified_live']} live, "
                f"{summary['dead']} dead, "
                f"{summary['bot_blocked']} bot-blocked, "
                f"{summary['stood_down']} stood down, "
                f"{summary['disabled']} disabled")
        deposited = self.knowledge.deposit(
            "sourcehealth", text, source="agency:sourcehealth")
        self._advance_watch("sourcehealth", now, text)
        return WorkResult(useful=deposited or bool(summary["changed"]),
                          work_key=f"sourcehealth:{int(now)}")

    # ── rotation cycle ─────────────────────────────────────────────
    def run_cycle(self, *, budget_s: float = 600.0,
                  now: float | None = None) -> dict[str, Any]:
        """One rotation cycle over the expansion systems.

        Same contract as the core ``ActiveSystems.run_cycle``: fair
        turns for every system with work, duplicate-work claims,
        stand-down for systems without work.
        """
        now = time.time() if now is None else now
        self.scheduler.cycles_run += 1
        alloc = self.scheduler.allocate_cycle(total_budget=budget_s)
        ran: set[str] = set()
        useful = 0
        stood_down = 0
        for _ in range(len(EXPANSION_SYSTEMS) * 2):
            runnable = {n for n in EXPANSION_SYSTEMS if self.has_work(n)}
            stood_down = len(EXPANSION_SYSTEMS) - len(runnable)
            nxt = self.scheduler.next_system(runnable=runnable, now=now)
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


def _report_done(report: Any) -> str:
    """Human-readable 'what got done' from an organ tick report."""
    if report is None:
        return ""
    if isinstance(report, dict):
        parts = []
        for key in ("done", "completed", "ingested", "resolved",
                    "processed", "findings", "items"):
            val = report.get(key)
            if val:
                parts.append(f"{key}={val}")
        if parts:
            return ", ".join(parts)
        if report.get("ok") or report.get("useful"):
            return "organ tick ok"
        return ""
    for attr in ("done", "completed", "gaps_resolved", "findings"):
        val = getattr(report, attr, None)
        if val:
            return f"{attr}={val}"
    return ""
