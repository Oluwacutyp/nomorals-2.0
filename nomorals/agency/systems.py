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
  ``impossible`` = an ``ImpossibleMadePossible``, or an ``adapters``
  dict of real organ adapters), ``has_work`` reads REAL queues and
  ``tick`` runs REAL organ work.
* Without an injected organ, the system works its initiative
  lifecycle (standing watch checks + knowledge deposits) and reports
  the organ as unavailable — it never fabricates organ output.
* A system with no useful work stands down: ``has_work`` False means
  the rotation never picks it and it consumes no budget.

Proactive (human-worker) behavior: systems don't wait to be poked.
``scan_proactive`` observes every system through its adapter, turns
triggers into initiatives, and the rotation then WORKS them —
``work_shift`` is observe → see what needs doing → do it → verify →
follow through → look for the next thing, with no prompt.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .initiatives import Initiative, InitiativeStore
from .knowledge import KnowledgeStore
from .checker import Checker
from .rotation import RotationScheduler, SystemSlot, WorkResult
from .deliberation import DeliberationLoop
from .domains import LEAD_TRANSITIONS, social_next_action
from .proactive import (
    detect_triggers, fire_triggers, load_proactive_state,
    save_proactive_state,
)

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

#: Blocker markers the leads handler sets ONLY after a successful
#: stage advance (see ``_work_leads_trigger``). When one is present,
#: the lead was already advanced on a previous turn and the trigger
#: is waiting on the DM step — a retry must resume there, never
#: advance the lead a second time.
_LEADS_DM_WAIT_MARKERS = (
    "DM send needs owner-approved draft",
    "DM send blocked:",
)


class ActiveSystems:
    def __init__(self, *, db: Any, store: InitiativeStore | None = None,
                 scheduler: RotationScheduler | None = None,
                 knowledge: KnowledgeStore | None = None,
                 research_organ: Any | None = None,
                 wisdom_organ: Any | None = None,
                 impossible: Any | None = None,
                 adapters: dict[str, Any] | None = None,
                 authority: Any | None = None,
                 checker: Any | None = None) -> None:
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
        #: Real organ adapters, keyed by system name (+ "planner").
        #: Organ-less adapters degrade to empty observations.
        self.adapters = dict(adapters or {})
        #: Standing authority (AgencyAuthority) — consulted at
        #: execution time, never widened by this layer.
        self.authority = authority
        #: Maker-checker (a SEPARATE verifier, never the worker):
        #: every proposed completion is independently re-read here
        #: before it is allowed to complete. Defaults to a real
        #: :class:`Checker`; inject a fake only in tests.
        self.checker = checker if checker is not None else Checker()
        #: Durable proactive detector state (survives restarts).
        self._proactive_state = load_proactive_state(db)
        #: Execution hints keyed by initiative dedupe_key. Hints only:
        #: handlers re-derive from live adapter state when missing
        #: (payloads don't survive a restart).
        self._trigger_payloads: dict[str, dict[str, Any]] = {}

    def _persist_state(self) -> None:
        save_proactive_state(self.db, self._proactive_state)

    def _complete_verified(self, it: Initiative, system: str, *,
                           stop_reason: str, verification: dict[str, Any],
                           now: float) -> WorkResult:
        """Maker-checker gate for a proposed completion.

        The handler proposes (``stop_reason`` + claimed
        ``verification``); the CHECKER — a separate code path that
        re-reads the world — decides. Pass → the initiative completes
        with the checker's checks recorded. Fail → the initiative
        ADVANCES with a ``maker-checker rejected`` blocker: the work
        is NOT completed, the discrepancy is visible, and the next
        turn can retry. The worker never grades its own work.
        """
        # The checker re-reads through adapters AND the injected organs:
        # research / wisdom / impossible organs are not in the adapters
        # dict (they are separate constructor args), but their re-reads
        # (open_gaps, queue_list) are the independent observations the
        # checker needs. Same objects the worker used — independence comes
        # from re-observing state, not from a different handle.
        check_adapters = dict(self.adapters)
        if self._research_organ is not None:
            check_adapters.setdefault("research", self._research_organ)
        if self._wisdom_organ is not None:
            check_adapters.setdefault("wisdom", self._wisdom_organ)
        if self._impossible is not None:
            check_adapters.setdefault("impossible", self._impossible)
        check = self.checker.verify(
            system, verification, adapters=check_adapters, db=self.db)
        if check.passed:
            merged = dict(verification)
            merged["maker_checker"] = {"passed": True,
                                       "checks": list(check.checks)}
            self.store.complete(it.id, stop_reason=stop_reason,
                                verification=merged)
            return WorkResult(useful=True, work_key=f"{system}:{it.id}")
        self.store.advance(
            it.id, next_review_at=now + 3600,
            blocker=f"maker-checker rejected: {check.reason}",
            evidence=f"completion claim rejected: "
                     f"{str(verification)[:400]}")
        return WorkResult(useful=False, work_key=f"{system}:{it.id}")

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

    def _trigger_initiatives(self, name: str,
                             now: float | None = None) -> list[Initiative]:
        # Only actionable triggers: fresh ones (never worked) plus
        # backed-off ones whose wake-up has arrived. A trigger the code
        # itself advanced with a future next_review_at is NOT re-worked
        # early — otherwise every blocked/failed trigger is retried on
        # every pick and the backoff the handlers set is meaningless.
        # ``now`` defaults to wall-clock; tick paths pass their own.
        now = time.time() if now is None else now
        out = []
        for it in self.store.list_active():
            if it.objective != name or it.kind != "evidence_opportunity":
                continue
            if (it.last_action_at > 0 and it.next_review_at
                    and it.next_review_at > now):
                continue
            out.append(it)
        return sorted(out, key=lambda i: i.created_at)

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

    def _adapter_pending(self, name: str) -> int:
        """Cheap pending-work read from an injected adapter (guarded)."""
        adapter = self.adapters.get(name)
        if adapter is None:
            return 0
        try:
            if name == "deals":
                return len(adapter.check_alerts() or [])
            if name == "weather":
                return len(adapter.places() or [])
            if name == "leads":
                return len(adapter.live_leads() or [])
            if name == "gigs":
                return len(adapter.follow_ups_due() or [])
            if name == "planner":
                return len(adapter.pending_graphs() or [])
        except Exception:  # noqa: BLE001 - adapter failure = no work
            return 0
        return 0

    def has_work(self, name: str) -> bool:
        if name in ("research", "wisdom", "impossible"):
            if self._organ_pending(name) > 0:
                return True
        if self._trigger_initiatives(name):
            return True
        if self._adapter_pending(name) > 0:
            return True
        return self._system_initiative(name) is not None

    # ── proactive: observe → initiate ──────────────────────────────
    def _detector_source(self, name: str) -> Any:
        if name in ("deals", "weather", "news", "social", "leads", "gigs",
                    "planner"):
            return self.adapters.get(name)
        if name == "research":
            if self._research_organ is not None:
                return self._research_organ
            try:
                rows = self.db.query(
                    "SELECT id FROM knowledge_gaps WHERE status='open' "
                    "LIMIT 50") or []
                return {"gaps": [{"id": r.get("id", "")} for r in rows],
                        "watches": []}
            except Exception:  # noqa: BLE001
                return {"gaps": [], "watches": []}
        if name == "wisdom":
            if self._wisdom_organ is not None:
                return self._wisdom_organ
            try:
                rows = self.db.query(
                    "SELECT id FROM wisdom_ingest_queue "
                    "WHERE status='queued' LIMIT 50") or []
                return {"queued": len(rows)}
            except Exception:  # noqa: BLE001
                return {"queued": 0}
        if name == "impossible":
            return self._impossible
        if name == "digestion":
            return {"knowledge_count": self.knowledge.count()}
        if name == "planner":
            return {"initiatives": self.store.list_active()}
        return None

    def _detector_places(self, name: str) -> list[str]:
        if name != "weather":
            return []
        adapter = self.adapters.get("weather")
        if adapter is None:
            return []
        try:
            return list(adapter.places() or [])
        except Exception:  # noqa: BLE001
            return []

    def scan_proactive(self, *, now: float | None = None) -> list[Any]:
        """Observe every system, turn triggers into initiatives."""
        now = time.time() if now is None else now
        fired_all = []
        for name in SYSTEM_NAMES:
            triggers = detect_triggers(
                name, self._detector_source(name), self._proactive_state,
                now=now, places=self._detector_places(name))
            if not triggers:
                continue
            fired = fire_triggers(self.store, triggers, now)
            for t in triggers:
                self._trigger_payloads[t.dedupe_key] = t.payload
            fired_all.extend(triggers)
        self._persist_state()
        return fired_all

    # ── one unit of work per system ────────────────────────────────
    def tick(self, name: str, *, budget_s: float = 60.0,
             now: float | None = None) -> WorkResult:
        now = time.time() if now is None else now
        try:
            return self._tick_inner(name, budget_s=budget_s, now=now)
        finally:
            self._persist_state()

    def _tick_inner(self, name: str, *, budget_s: float,
                    now: float) -> WorkResult:
        if name == "planner":
            return self._tick_planner(now, budget_s)
        if name == "digestion":
            return self._tick_digestion(now)
        if name == "research" and self._research_organ is not None:
            # Per-trigger handlers FIRST: a pending research trigger
            # (a specific gap / a specific due watch) gets its own
            # handler, not a generic organ tick. The generic tick is
            # only the fallback when no trigger is pending.
            triggers = self._trigger_initiatives("research", now)
            if triggers:
                return self._work_trigger(triggers[0], now)
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
            # Per-trigger handler first (a specific ingest-queue
            # trigger), generic tick only as fallback.
            triggers = self._trigger_initiatives("wisdom", now)
            if triggers:
                return self._work_trigger(triggers[0], now)
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
        if name == "impossible" and self._impossible is not None:
            return self._tick_impossible(now)
        if name in SYSTEM_NAMES:
            return self._tick_domain(name, now)
        # Unknown system: standing-watch bookkeeping only.
        return self._tick_standing_watch(name, now)

    def _tick_planner(self, now: float, budget_s: float) -> WorkResult:
        # A pending plan graph is real work: execute ONE first.
        adapter = self.adapters.get("planner")
        if adapter is not None:
            try:
                graphs = adapter.pending_graphs() or []
            except Exception:  # noqa: BLE001
                graphs = []
            if graphs:
                try:
                    result = adapter.execute_graph(graphs[0])
                except Exception as exc:  # noqa: BLE001
                    result = {"completed": [], "failed": [str(exc)],
                              "attempts": 0}
                self.knowledge.deposit(
                    "planner",
                    f"plan graph executed: {len(result.get('completed', []))} "
                    f"completed, {len(result.get('failed', []))} failed",
                    source="agency:planner")
                return WorkResult(useful=True, work_key=f"planner:{int(now)}")
        # Else deliberation: choose the next initiatives to work.
        chosen = DeliberationLoop(self.store).choose(now=now, capacity=3)
        if chosen:
            self.knowledge.deposit(
                "planner",
                f"deliberation chose {len(chosen)} initiatives: "
                + ", ".join(c.outcome for c in chosen)[:400],
                source="agency:planner")
            return WorkResult(useful=True, work_key=f"planner:{int(now)}")
        return WorkResult(useful=False)

    def _tick_impossible(self, now: float) -> WorkResult:
        try:
            queued = self._impossible.queue_list(status="queued") or []
        except Exception:  # noqa: BLE001
            queued = []
        if not queued:
            return self._tick_standing_watch("impossible", now)
        claim = queued[0]
        claim_id = str(claim.get("id", "") if isinstance(claim, dict)
                       else getattr(claim, "id", ""))
        try:
            result = self._impossible.attempt(claim_id)
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc)}
        self.knowledge.deposit(
            "impossible",
            f"attempted impossible claim {claim_id}: "
            f"{str(result)[:400]}",
            source="agency:impossible")
        # Complete the matching trigger initiative — through the
        # maker-checker gate, not the worker's own say-so.
        for it in self._trigger_initiatives("impossible", now):
            payload = self._trigger_payloads.get(it.dedupe_key, {})
            if payload.get("claim_id", claim_id) == claim_id:
                return self._complete_verified(
                    it, "impossible",
                    stop_reason="impossible claim attempted",
                    verification={"claim_id": claim_id,
                                  "attempt": str(result)[:1000]},
                    now=now)
        return WorkResult(useful=True, work_key=f"impossible:{int(now)}")

    def _tick_weather_baseline(self, now: float) -> list[Any]:
        """Weather first-sight: deposit baselines (no trigger yet)."""
        triggers = detect_triggers(
            "weather", self._detector_source("weather"),
            self._proactive_state, now=now,
            places=self._detector_places("weather"))
        for t in triggers:
            self._trigger_payloads[t.dedupe_key] = t.payload
        return triggers

    def _tick_domain(self, name: str, now: float) -> WorkResult:
        # Work the oldest trigger initiative first (follow-through).
        triggers = self._trigger_initiatives(name, now)
        if triggers:
            return self._work_trigger(triggers[0], now)
        # None pending: observe now — a new trigger may initiate work.
        if name == "weather":
            fresh = self._tick_weather_baseline(now)
        else:
            fresh = detect_triggers(
                name, self._detector_source(name), self._proactive_state,
                now=now, places=self._detector_places(name))
            for t in fresh:
                self._trigger_payloads[t.dedupe_key] = t.payload
        if fresh:
            fired = fire_triggers(self.store, fresh, now)
            return self._work_trigger(fired[0], now)
        return self._tick_standing_watch(name, now)

    def _tick_standing_watch(self, name: str, now: float) -> WorkResult:
        # Generic standing-watch turn: check the system's standing
        # initiative, deposit what was observed, advance its wake-up.
        # Real bookkeeping (a checked watch), never fabricated output.
        it = None
        for cand in self.store.list_active():
            if cand.objective == name and cand.kind == "standing":
                it = cand
                break
        if it is None:
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

    # ── working one trigger initiative ─────────────────────────────
    def _work_trigger(self, it: Initiative, now: float) -> WorkResult:
        try:
            return self._work_trigger_inner(it, now)
        except Exception as exc:  # noqa: BLE001 - trigger work never crashes
            self.store.advance(
                it.id, next_review_at=now + 3600,
                blocker=f"trigger work failed: {exc}")
            return WorkResult(useful=False, work_key=f"{it.objective}:{it.id}")

    def _work_trigger_inner(self, it: Initiative, now: float) -> WorkResult:
        name = it.objective
        payload = self._trigger_payloads.get(it.dedupe_key) or {}
        if name == "deals":
            return self._work_deals_trigger(it, payload, now)
        if name == "weather":
            return self._work_weather_trigger(it, payload, now)
        if name == "news":
            return self._work_news_trigger(it, payload, now)
        if name == "social":
            return self._work_social_trigger(it, payload, now)
        if name == "leads":
            return self._work_leads_trigger(it, payload, now)
        if name == "gigs":
            return self._work_gigs_trigger(it, payload, now)
        if name == "research":
            return self._work_research_trigger(it, payload, now)
        if name == "wisdom":
            return self._work_wisdom_trigger(it, payload, now)
        # Impossible / digestion / planner triggers are worked by their
        # organ ticks, not per-trigger handlers: note the trigger, keep
        # the lifecycle moving.
        self.store.advance(it.id, next_review_at=now + 3600,
                            evidence=f"{name} trigger noted; organ tick "
                                     "handles execution")
        return WorkResult(useful=True, work_key=f"{name}:{it.id}")

    def _work_research_trigger(self, it: Initiative, payload: dict,
                               now: float) -> WorkResult:
        """Per-trigger research handlers: one handler per trigger kind.

        * ``new_gap`` → ``organ.investigate_gap(gap_id)`` (that gap
          only, not a generic tick over the open list).
        * ``watch_due`` → ``organ.run_watch(watch_id)`` (that watch
          only, cadence bypassed because the trigger IS the due-ness).
        Organs without the per-trigger method fall back to a generic
        tick. The investigated result is deposited as knowledge, and
        completion goes through the maker-checker gate.
        """
        organ = self._research_organ
        if organ is None:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="no research organ injected")
            return WorkResult(useful=False, work_key=f"research:{it.id}")
        gap = payload.get("gap") or {}
        watch = payload.get("watch") or {}
        try:
            if gap and callable(getattr(organ, "investigate_gap", None)):
                result = organ.investigate_gap(
                    gap.get("id", gap.get("gap_id", "")))
                gap_id = str(result.get("gap_id", gap.get("id", "")))
                self.knowledge.deposit(
                    "research",
                    f"gap {gap_id} investigated: "
                    f"status={result.get('status')}; "
                    f"{str(result.get('synthesis', ''))[:400]}",
                    source="agency:research")
                return self._complete_verified(
                    it, "research",
                    stop_reason=f"research gap {gap_id} investigated: "
                                f"{result.get('status')}",
                    verification={"gap_id": gap_id,
                                  "status": result.get("status"),
                                  "synthesis": str(result.get(
                                      "synthesis", ""))[:500]},
                    now=now)
            if watch and callable(getattr(organ, "run_watch", None)):
                result = organ.run_watch(
                    watch.get("id", watch.get("watch_id", "")))
                watch_id = str(result.get("watch_id", watch.get("id", "")))
                self.knowledge.deposit(
                    "research",
                    f"watch {watch_id} run: {result.get('findings', 0)} "
                    f"findings, {result.get('delivered', 0)} delivered",
                    source="agency:research")
                return self._complete_verified(
                    it, "research",
                    stop_reason=f"research watch {watch_id} run: "
                                f"{result.get('status')}",
                    verification={"gap_id": watch_id,
                                  "watch_id": watch_id,
                                  "findings": result.get("findings", 0),
                                  "status": result.get("status")},
                    now=now)
        except Exception as exc:  # noqa: BLE001 - per-trigger failure
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker=f"research trigger work failed: {exc}")
            return WorkResult(useful=False, work_key=f"research:{it.id}")
        # No per-trigger method on this organ (or no gap/watch in the
        # payload): fall back to a generic tick and note the trigger.
        try:
            report = organ.tick()
            self.knowledge.deposit(
                "research",
                f"research tick (fallback for trigger {it.dedupe_key}): "
                f"{getattr(report, 'gaps_resolved', 0)} gaps resolved, "
                f"{getattr(report, 'findings', 0)} findings",
                source="agency:research")
        except Exception as exc:  # noqa: BLE001 - per-trigger failure
            # A failed fallback tick must NOT be recorded as a noted
            # trigger + useful turn (tail-1 convert): the organ did
            # not handle anything, so the trigger blocks like every
            # other per-trigger failure in this module.
            self.store.advance(it.id, next_review_at=now + 3600,
                               blocker=f"research trigger fallback tick "
                                       f"failed: {exc}")
            return WorkResult(useful=False, work_key=f"research:{it.id}")
        self.store.advance(it.id, next_review_at=now + 3600,
                            evidence="research trigger noted; organ tick "
                                     "handles execution (no per-trigger "
                                     "method)")
        return WorkResult(useful=True, work_key=f"research:{it.id}")

    def _work_wisdom_trigger(self, it: Initiative, payload: dict,
                             now: float) -> WorkResult:
        """Per-trigger wisdom handler: ``ingest_queue`` →
        ``organ.ingest_next()`` (exactly one pending item), not a
        generic tick. Falls back to a generic tick when the organ has
        no per-item method."""
        organ = self._wisdom_organ
        if organ is None:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="no wisdom organ injected")
            return WorkResult(useful=False, work_key=f"wisdom:{it.id}")
        try:
            if callable(getattr(organ, "ingest_next", None)):
                result = organ.ingest_next() or {}
                ingested = list(result.get("ingested") or [])
                self.knowledge.deposit(
                    "wisdom", f"wisdom ingested {ingested} (trigger)",
                    source="agency:wisdom")
                if not ingested:
                    self.store.advance(it.id, next_review_at=now + 3600,
                                       evidence="wisdom ingest trigger: "
                                               "nothing pending")
                    return WorkResult(useful=True,
                                      work_key=f"wisdom:{it.id}")
                return self._complete_verified(
                    it, "wisdom",
                    stop_reason=f"wisdom ingested {ingested}",
                    verification={"ingested": ingested},
                    now=now)
        except Exception as exc:  # noqa: BLE001 - per-trigger failure
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker=f"wisdom trigger work failed: {exc}")
            return WorkResult(useful=False, work_key=f"wisdom:{it.id}")
        try:
            report = organ.tick(budget_seconds=60.0)
            self.knowledge.deposit(
                "wisdom",
                f"wisdom tick (fallback for trigger {it.dedupe_key}): "
                f"ingested {report.get('ingested')}",
                source="agency:wisdom")
        except Exception as exc:  # noqa: BLE001 - per-trigger failure
            # Same tail-1 convert as the research fallback: a failed
            # fallback tick blocks, it is never recorded as noted +
            # useful.
            self.store.advance(it.id, next_review_at=now + 3600,
                               blocker=f"wisdom trigger fallback tick "
                                       f"failed: {exc}")
            return WorkResult(useful=False, work_key=f"wisdom:{it.id}")
        self.store.advance(it.id, next_review_at=now + 3600,
                            evidence="wisdom trigger noted; organ tick "
                                     "handles execution (no per-item "
                                     "method)")
        return WorkResult(useful=True, work_key=f"wisdom:{it.id}")

    def _work_deals_trigger(self, it: Initiative, payload: dict,
                            now: float) -> WorkResult:
        adapter = self.adapters.get("deals")
        if adapter is None:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="no deals adapter injected")
            return WorkResult(useful=False, work_key=f"deals:{it.id}")
        alert = payload.get("alert") or {}
        if not alert:
            alerts = adapter.check_alerts() or []
            alert = alerts[0] if alerts else {}
        product = str(alert.get("product", alert.get("query", "")))
        if not product:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="deal trigger has no product")
            return WorkResult(useful=False, work_key=f"deals:{it.id}")
        need: dict[str, Any] = dict(alert.get("need") or {})
        need.setdefault("product", product)
        need.setdefault("needed", True)
        for key in ("shipping", "fees", "historical_low"):
            if key in alert:
                need.setdefault(key, alert[key])
        result = adapter.check_need(need)
        evaluation = result.get("evaluation") or {}
        observed = result.get("observed") or {}
        self.knowledge.deposit(
            "deals",
            f"deal check {product}: {evaluation.get('reason', '')} "
            f"(landed {observed.get('landed_price')}, "
            f"historical low {observed.get('historical_low')})",
            source="agency:deals")
        if evaluation.get("is_saving"):
            amount = float(observed.get("landed_price", 0) or 0)
            # Purchase path (adapter with an executor): COMPLETE the
            # purchase when authority allows — don't just report
            # readiness. The executor itself re-gates policy + mandate;
            # a denial comes back as a blocker, never a silent buy.
            if callable(getattr(adapter, "execute_purchase", None)):
                res = adapter.execute_purchase(need, observed,
                                               authority=self.authority)
                if res.get("purchased"):
                    receipt = res.get("receipt") or {}
                    order = receipt.get("order_id", "") or ""
                    return self._complete_verified(
                        it, "deals",
                        stop_reason=f"deal purchased: order {order}",
                        verification={"product": product, "need": need,
                                      "observed": observed,
                                      "evaluation": evaluation,
                                      "purchased": True, "receipt": receipt},
                        now=now)
                blocker = res.get("blocker") or res.get("reason") \
                    or "purchase not completed"
                self.store.advance(
                    it.id, next_review_at=now + 3600, blocker=blocker,
                    evidence=f"saving verified; purchase blocked: "
                             f"{res.get('reason', blocker)}")
                return WorkResult(useful=True, work_key=f"deals:{it.id}")
            # Legacy path (adapter without an executor, e.g. a test
            # double): verify + report readiness, never buy.
            if self.authority is not None \
                    and self.authority.can_spend(amount):
                return self._complete_verified(
                    it, "deals",
                    stop_reason="deal verified: saving within mandate",
                    verification={"product": product, "need": need,
                                  "observed": observed,
                                  "evaluation": evaluation},
                    now=now)
            reason = self.authority.spend_reason(amount) \
                if self.authority is not None else \
                "no authority configured — cannot verify spending"
            self.store.advance(
                it.id, next_review_at=now + 3600,
                blocker=f"saving verified but {reason} (no purchase made)")
            return WorkResult(useful=True, work_key=f"deals:{it.id}")
        return self._complete_verified(
            it, "deals", stop_reason="deal checked: not a saving",
            verification={"product": product, "need": need,
                          "observed": observed, "evaluation": evaluation},
            now=now)

    def _work_weather_trigger(self, it: Initiative, payload: dict,
                              now: float) -> WorkResult:
        adapter = self.adapters.get("weather")
        place = str(payload.get("place", ""))
        if not place:
            places = self._detector_places("weather")
            place = places[0] if places else ""
        if adapter is None or not place:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="no weather adapter / place configured")
            return WorkResult(useful=False, work_key=f"weather:{it.id}")
        snap = adapter.snapshot(place) or {}
        alerts = snap.get("alerts") or []
        self.knowledge.deposit(
            "weather",
            f"weather briefing for {place}: "
            f"{len(snap.get('days') or [])} forecast days, "
            f"{len(alerts)} alerts — "
            + "; ".join(a.get("text", "") for a in alerts)[:300],
            source="agency:weather")
        return self._complete_verified(
            it, "weather", stop_reason="weather change briefed",
            verification={"place": place,
                          "days": len(snap.get("days") or []),
                          "alerts": len(alerts)},
            now=now)

    def _work_news_trigger(self, it: Initiative, payload: dict,
                           now: float) -> WorkResult:
        adapter = self.adapters.get("news")
        if adapter is None:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="no news adapter injected")
            return WorkResult(useful=False, work_key=f"news:{it.id}")
        digest = adapter.fetch_digest() or {}
        clusters = adapter.clusters() or []
        self.knowledge.deposit(
            "news",
            f"news digest: {len(clusters)} story clusters; "
            f"digest keys: {', '.join(sorted(digest.keys())[:8])}",
            source="agency:news")
        return self._complete_verified(
            it, "news", stop_reason="news digest deposited",
            verification={"clusters": len(clusters),
                          "digest_keys": sorted(digest.keys())[:8]},
            now=now)

    def _work_social_trigger(self, it: Initiative, payload: dict,
                             now: float) -> WorkResult:
        adapter = self.adapters.get("social")
        if adapter is None:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="no social adapter injected")
            return WorkResult(useful=False, work_key=f"social:{it.id}")
        backlog = adapter.backlog() or {}
        if backlog.get("stop_signal"):
            return self._complete_verified(
                it, "social",
                stop_reason="social silence honoured (stop signal)",
                verification={"backlog": backlog, "silence": "stop_signal"},
                now=now)
        if not backlog.get("cadence_ok", True):
            return self._complete_verified(
                it, "social",
                stop_reason="social silence honoured (cadence closed)",
                verification={"backlog": backlog, "silence": "cadence"},
                now=now)
        action = social_next_action(
            unanswered=int(backlog.get("unanswered", 0) or 0),
            promised_replies=int(backlog.get("promised_replies", 0) or 0),
            milestones=int(backlog.get("milestones", 0) or 0),
            stop_signal=False, cadence_ok=True)
        draft = adapter.draft_for(action)
        decision = self.authority.may_post(draft=str(draft)) \
            if self.authority is not None else None
        if decision is not None and decision.allowed:
            published = adapter.publish_due() or []
            self.knowledge.deposit(
                "social", f"social published {len(published)} due posts",
                source="agency:social")
            return self._complete_verified(
                it, "social", stop_reason="social action published",
                verification={"draft": draft, "published": len(published)},
                now=now)
        if decision is not None:
            blocker = ("post needs approval: "
                       f"{decision.reason} "
                       f"(proposal {getattr(decision, 'proposal_id', '') or '—'})")
        else:
            blocker = "no authority configured — draft held, nothing sent"
        self.store.advance(it.id, next_review_at=now + 3600,
                            blocker=blocker,
                            evidence=f"draft ready: {draft}")
        self.knowledge.deposit(
            "social", f"social draft held for approval: {action}",
            source="agency:social")
        return WorkResult(useful=True, work_key=f"social:{it.id}")

    def _work_leads_trigger(self, it: Initiative, payload: dict,
                            now: float) -> WorkResult:
        adapter = self.adapters.get("leads")
        if adapter is None:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="no leads adapter injected")
            return WorkResult(useful=False, work_key=f"leads:{it.id}")
        lead = dict(payload.get("lead") or {})
        if not lead:
            live = adapter.live_leads() or []
            lead = live[0] if live else {}
        if not lead:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="lead trigger has no lead")
            return WorkResult(useful=False, work_key=f"leads:{it.id}")
        kind = it.dedupe_key.split(":")[1] if ":" in it.dedupe_key else ""
        if "quiet" in it.dedupe_key or kind == "quiet":
            res = adapter.follow_up(lead)
            self.knowledge.deposit(
                "leads", f"quiet lead {lead.get('lead_id', '')} followed up",
                source="agency:leads")
            return self._complete_verified(
                it, "leads",
                stop_reason="quiet lead followed up (draft held)",
                verification={"lead": lead, "follow_up": res},
                now=now)
        current = str(lead.get("stage", lead.get("status", "discovered")))
        nxt_options = [t for t in LEAD_TRANSITIONS.get(current, frozenset())
                       if t != "lost"]
        if not nxt_options:
            self.store.complete(
                it.id, stop_reason=f"lead {current}: no forward transition",
                verification={"lead": lead})
            return WorkResult(useful=True, work_key=f"leads:{it.id}")
        res = None
        # Idempotency: when the blocker is one of the DM-wait markers,
        # this handler already advanced the lead on a previous turn.
        # Resume at the contact step below — never advance twice for
        # one trigger (the retry re-derives from the stale trigger
        # payload, so a blind re-advance would push the lead another
        # stage with no DM ever sent).
        already_advanced = bool(it.blocker) and it.blocker.startswith(
            _LEADS_DM_WAIT_MARKERS)
        if not already_advanced:
            res = adapter.advance(lead, nxt_options[0])
            if res.get("status") != "advanced":
                self.store.advance(it.id, next_review_at=now + 3600,
                                    blocker=f"lead advance failed: "
                                            f"{res.get('error', res)}")
                return WorkResult(useful=False, work_key=f"leads:{it.id}")
            self.knowledge.deposit(
                "leads",
                f"lead {lead.get('lead_id', '')} advanced "
                f"{current} → {nxt_options[0]}",
                source="agency:leads")
        # Contact step: the DM-send executor sends the OWNER-APPROVED
        # draft when authority allows (``payload["approved_draft"]``
        # or a default lead draft), and the completion is
        # maker-checker verified against the store. Without an
        # executor on the adapter, the legacy blocker path stands.
        if callable(getattr(adapter, "send_dm", None)):
            approved = payload.get("approved_draft")
            draft = str(approved or
                        f"Hi {lead.get('name', '') or lead.get('lead_id', '')}"
                        " — following up on your enquiry.")
            # An owner-approved draft is the trial pattern: the owner
            # explicitly approved THIS draft, so the send goes through
            # the explicit-override path (deny/grant/biometric still
            # apply). A default draft is autonomous → confirmable gate.
            sent = adapter.send_dm(lead, draft=draft,
                                   authority=self.authority,
                                   explicit=bool(approved))
            if sent.get("sent"):
                return self._complete_verified(
                    it, "leads", stop_reason="lead contacted: DM sent",
                    verification={"lead": lead, "sent": True,
                                  "store_status": "dmed",
                                  "evidence": sent.get("evidence", {})},
                    now=now)
            reason = str(sent.get("reason", "DM not sent"))
            proposal = sent.get("proposal_id") or ""
            self.store.advance(
                it.id, next_review_at=now + 3600,
                next_action=f"send approved DM draft for lead "
                            f"{lead.get('lead_id', '')}",
                blocker=f"DM send blocked: {reason}"
                        + (f" (proposal {proposal})" if proposal else ""),
                evidence=f"lead advanced {current} → {nxt_options[0]}; "
                         f"draft held: {draft}")
            return WorkResult(useful=True, work_key=f"leads:{it.id}")
        # No DM-send executor on this adapter: the next action is a
        # DM draft held for owner approval, recorded as a blocker.
        self.store.advance(
            it.id, next_review_at=now + 3600,
            next_action=f"send approved DM draft for lead {lead.get('lead_id', '')}",
            blocker="DM send needs owner-approved draft (no DM executor)",
            evidence=f"lead advanced {current} → {nxt_options[0]}")
        return WorkResult(useful=True, work_key=f"leads:{it.id}")

    def _work_gigs_trigger(self, it: Initiative, payload: dict,
                           now: float) -> WorkResult:
        adapter = self.adapters.get("gigs")
        if adapter is None:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="no gigs adapter injected")
            return WorkResult(useful=False, work_key=f"gigs:{it.id}")
        if "followup" in it.dedupe_key:
            gig_id = str(payload.get("gig_id", ""))
            if not gig_id:
                due = adapter.follow_ups_due() or []
                gig_id = str(due[0].get("gig_id", "")) if due else ""
            draft = adapter.follow_up_draft(gig_id) if gig_id else {}
            self.knowledge.deposit(
                "gigs", f"gig follow-up drafted for {gig_id}",
                source="agency:gigs")
            return self._complete_verified(
                it, "gigs", stop_reason="gig follow-up drafted",
                verification={"gig_id": gig_id, "draft": draft},
                now=now)
        opp = dict(payload.get("opp") or {})
        if not opp:
            opps = adapter.scan() or []
            opp = opps[0] if opps else {}
        if not opp:
            self.store.advance(it.id, next_review_at=now + 3600,
                                blocker="gig trigger has no opportunity")
            return WorkResult(useful=False, work_key=f"gigs:{it.id}")
        score = adapter.score(opp)
        draft = adapter.draft(opp)
        self.knowledge.deposit(
            "gigs",
            f"gig scored {opp.get('title', opp.get('id', ''))}: "
            f"expected_net {score.get('expected_net')} "
            f"({score.get('recommendation', '')}), draft "
            f"{draft.get('status', '')}",
            source="agency:gigs")
        return self._complete_verified(
            it, "gigs", stop_reason="gig scored + drafted",
            verification={"opp": opp, "score": score, "draft": draft},
            now=now)

    # ── rotation cycle ─────────────────────────────────────────────
    def run_cycle(self, *, budget_s: float = 600.0,
                  now: float | None = None,
                  scan_first: bool = True) -> dict[str, Any]:
        """One rotation cycle: proactive scan first, then fair turns.

        Urgent = planner when an overdue-critical initiative exists.
        Systems with no useful work stand down (no budget, no
        manufactured work). Duplicate work keys are claimed first.
        """
        now = time.time() if now is None else now
        if scan_first:
            self.scan_proactive(now=now)
        self.scheduler.cycles_run += 1
        alloc = self.scheduler.allocate_cycle(total_budget=budget_s)
        urgent: set[str] = set()
        for it in self.store.list_active():
            if (it.priority >= 0.8 and it.next_review_at
                    and now - it.next_review_at > 3600
                    and not it.blocker):
                # Blocked initiatives are excluded: a blocker names work
                # that cannot proceed (owner approval, failed check), so
                # preempting the planner for it starves the whole cycle
                # for work that will never happen — the wake-up
                # (next_review_at) re-surfaces it when the blocker clears.
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

    def work_shift(self) -> dict[str, Any]:
        """One human-worker shift: observe → do → verify → next thing.

        No prompt, no owner poke: scan for triggers, run a rotation
        cycle that works them, report what happened.
        """
        now = time.time()
        triggers = self.scan_proactive(now=now)
        cycle = self.run_cycle(now=now, scan_first=False)
        self._persist_state()
        return {"triggers_fired": len(triggers),
                "triggers": [f"{t.system}:{t.kind}" for t in triggers],
                "cycle": cycle,
                "knowledge_items": self.knowledge.count()}
