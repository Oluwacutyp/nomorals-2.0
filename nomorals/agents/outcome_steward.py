"""Outcome Steward (L5) — doc 15 P4: cases, discovery, verdicts.

The steward is the SINGLE CASE OWNER above Devon's existing
reconcile/self-heal mechanisms — it is not a fifth reconciler. The
mechanisms keep their authority over the operations themselves:

* ``MissionStore.reconcile()`` (missions/mission.py) owns runner
  liveness: it flips a dead runner's mission to FAILED, on read. The
  steward never calls it and never edits mission state; it OBSERVES
  the mission record afterwards, because a liveness failure is not an
  effect outcome (doc 15 §8 — the evaluation's G6 catch).
* ``core/jobs.py``'s JobRegistry owns thread-job liveness: its
  dead-man's switch marks silent jobs presumed-dead /
  ``outcome_unknown`` and collects late results honestly. The steward
  reads snapshots only; it never cancels, reaps or re-issues a job.
* ``core/selfheal.py`` owns error recovery for tool/step failures
  (analyze → recover → verify → record) and ``core/self_heal.py``
  owns conservative repair of broken chat commands. Neither accounts
  for the outcome of forgotten work; the steward does not repair
  anything.
* ``agency/supervision.py``'s Supervisor owns boot-time lease/watch
  recovery for initiative work. The steward opens no leases and
  re-registers no watches.
* The delegation substrate (agents/delegation.py) owns accountable
  work contracts. The steward RUNS ON it: every investigation is a
  contract with an independent verifier, a skeptic pass and
  generation fencing — it forks none of that machinery.

What the steward adds (and only this): durable outcome CASES
(:mod:`outcome_cases`) for operations whose outcome is in question,
discovery sweeps that open those cases from domain records, and
evidence-gathering runs whose verdicts are either evidence-cited
resolutions or an honest ``closed-unknown``. Domain adapters are
READ-ONLY in P4 — they gather evidence from the domain's own records
and never call a mutating domain path. P5 adds the remaining domains
and the mission effect-outcome projection.

Runs are deterministic by design: deterministic read-back is
sufficient evidence for the domains covered here, and no model call
is spent where records already answer the question (doc 15 §9). The
contract's verifier judges the EVIDENCE TRAIL — a proposed verdict
stands only when it cites an evidence receipt gathered in the same
run — never the worker's say-so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.jobs import JobRegistry, shared_registry
from ..missions.mission import MissionStatus, MissionStore, mission_liveness
from .delegation import DelegationStore
from .outcome_cases import (
    RESOLVABLE_VERDICTS,
    OutcomeCase,
    OutcomeCaseStore,
)

__all__ = [
    "STEWARD_JOB",
    "Candidate",
    "EvidenceItem",
    "MissionsAdapter",
    "JobsAdapter",
    "OutcomeSteward",
    "run_stewardship_cycle",
    "ensure_outcome_steward_job",
    "inspect_case",
    "render_case",
    "register",
]

#: Durable sweep job name (idempotent registration, the ensure_*
#: pattern). Every 6h — the metrics/engagement rhythm: stewardship
#: questions (did forgotten work actually land?) ripen over hours,
#: and a tighter loop would spend delegation contracts to learn
#: nothing sooner.
STEWARD_JOB = "outcome steward sweep"
STEWARD_SCHEDULE = "every 6h"

#: Contract role holders for steward runs. Worker and verifier are
#: distinct by construction (the delegation store raises otherwise).
WORKER = "outcome-steward:worker"
VERIFIER = "outcome-steward:verifier"
SKEPTIC = "outcome-steward:skeptic"
SUPERVISOR = "outcome-steward:supervisor"

#: A mission still PENDING this long with no movement is stalled.
PENDING_STALE_S = 24 * 3600.0
#: A terminal core job whose result nobody collected for this long is
#: orphaned — well inside the registry's own terminal TTL, so the
#: record is still there to read when the case investigates.
UNSEEN_RESULT_S = 600.0
#: Hard cap on evidence items per run (the contract's step budget is
#: the second bound; both are stated, neither is silent).
MAX_EVIDENCE_PER_RUN = 6


@dataclass
class Candidate:
    """One operation a domain adapter believes needs a case."""

    domain: str
    kind: str
    subject: str
    subject_key: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    """One read-only observation from a domain's own records.

    ``establishes`` names the outcome verdict this observation, on its
    own, is sufficient evidence for — empty when the observation is
    context (liveness, checkpoints) that establishes nothing by
    itself. Liveness and timing NEVER establish an outcome.
    """

    source: str
    summary: str
    establishes: str = ""
    ref: str = ""


def _first_establishing(items: list[EvidenceItem]) -> EvidenceItem | None:
    for item in items:
        if item.establishes in RESOLVABLE_VERDICTS:
            return item
    return None


def _is_liveness_failure(mission: Any) -> bool:
    """True when a FAILED mission's record rests on liveness alone —
    the reconcile marker (missions/mission.py writes status_note
    'reconciled: …' + a 'runner died' last_error), with no step or
    effect failure recorded beside it."""
    state = mission.state or {}
    note = str(state.get("status_note") or "")
    error = str(state.get("last_error") or "")
    return note.startswith("reconciled:") or "runner died" in error


class MissionsAdapter:
    """Read-only observation of MissionStore records.

    Discovery: a RUNNING mission whose runner is gone (the shared
    ``mission_liveness`` probe says not alive) is ORPHANED — its
    effect outcome is exactly what reconcile() cannot say. A FAILED
    mission carrying only the liveness marker is UNKNOWN-OUTCOME. A
    PENDING mission unmoved for a day is STALLED. Nothing here calls
    ``reconcile()`` or ``save()`` — observation only.
    """

    domain = "missions"

    def __init__(self, db: Any) -> None:
        self.store = MissionStore(db)

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        for mission in self.store.list(status=MissionStatus.RUNNING,
                                       limit=200):
            live = mission_liveness(mission)
            if not live["alive"]:
                out.append(Candidate(
                    domain=self.domain, kind="mission",
                    subject=mission.name or mission.goal[:60],
                    subject_key=f"mission:{mission.id}",
                    reason="orphaned",
                    detail={"mission_id": mission.id,
                            "status": mission.status,
                            "heartbeat_age_s": live["heartbeat_age_s"],
                            "process_alive": live["process_alive"]}))
        for mission in self.store.list(status=MissionStatus.FAILED,
                                       limit=200):
            if _is_liveness_failure(mission):
                out.append(Candidate(
                    domain=self.domain, kind="mission",
                    subject=mission.name or mission.goal[:60],
                    subject_key=f"mission:{mission.id}",
                    reason="unknown-outcome",
                    detail={"mission_id": mission.id,
                            "status": mission.status,
                            "failure_basis": "liveness only"}))
        for mission in self.store.list(status=MissionStatus.PENDING,
                                       limit=200):
            age = now - float(mission.updated_at
                              or mission.created_at or now)
            if age > PENDING_STALE_S:
                out.append(Candidate(
                    domain=self.domain, kind="mission",
                    subject=mission.name or mission.goal[:60],
                    subject_key=f"mission:{mission.id}",
                    reason="stalled",
                    detail={"mission_id": mission.id,
                            "status": mission.status,
                            "age_s": round(age, 1)}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        mission_id = str(case.detail.get("mission_id") or "")
        try:
            mission = self.store.get(mission_id)
        except Exception as exc:  # noqa: BLE001 - a vanished record is data
            return [EvidenceItem(
                source="missions.store",
                summary=f"mission record unreadable: {exc}")]
        items: list[EvidenceItem] = []
        status = mission.status
        establishes = ""
        summary = (f"mission status is {status}; "
                   f"iterations {mission.iterations}; "
                   f"last updated {max(0.0, now - float(mission.updated_at or now)):.0f}s ago")
        if status == MissionStatus.DONE:
            establishes = "succeeded"
            summary += "; the mission record itself shows completion"
        elif status == MissionStatus.CANCELLED:
            establishes = "stopped"
            summary += "; the mission record shows cancellation"
        elif status == MissionStatus.FAILED \
                and not _is_liveness_failure(mission):
            establishes = "failed"
            error = str((mission.state or {}).get("last_error") or "")
            summary += f"; recorded failure: {error[:200]}"
        items.append(EvidenceItem(source="missions.store",
                                  summary=summary,
                                  establishes=establishes))
        live = mission_liveness(mission)
        items.append(EvidenceItem(
            source="missions.liveness",
            summary=(f"runner alive={live['alive']}; "
                     f"heartbeat age {live['heartbeat_age_s']}s; "
                     f"process alive={live['process_alive']} "
                     "(liveness is context, never an outcome)")))
        if status == MissionStatus.FAILED \
                and _is_liveness_failure(mission):
            items.append(EvidenceItem(
                source="missions.store",
                summary=("the recorded failure rests on liveness "
                         "evidence alone (reconcile marker); no step "
                         "failure or effect record exists beside it, "
                         "so the effect outcome is not established")))
        try:
            history = self.store.checkpoint_history(mission_id, limit=5)
        except Exception:  # noqa: BLE001 - checkpoints are best-effort
            history = []
        if history:
            latest = history[0]
            items.append(EvidenceItem(
                source="missions.checkpoints",
                summary=(f"{len(history)} recent checkpoint(s); "
                         f"latest: {latest.label or 'checkpoint'}")))
        return items


class JobsAdapter:
    """Read-only observation of the core JobRegistry's snapshots.

    Discovery: a presumed-dead job is UNKNOWN-OUTCOME by the
    registry's own honest verdict; a terminal job whose result nobody
    collected is ORPHANED (the result exists, uncollected — doc 15
    §7's 'artifact/result exists but no collection receipt'). The
    adapter never cancels, collects or re-issues a job; ``poll`` is
    the registry's read path (its dead-man evaluation is the
    registry's own business, unchanged).
    """

    domain = "jobs"

    def __init__(self, registry: JobRegistry | None = None) -> None:
        self.registry = registry if registry is not None \
            else shared_registry()

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        for snap in self.registry.list():
            status = snap["status"]
            if status == "presumed-dead":
                out.append(Candidate(
                    domain=self.domain, kind="job",
                    subject=f"{snap['tool'] or 'job'} job",
                    subject_key=f"job:{snap['id']}",
                    reason="unknown-outcome",
                    detail={"job_id": snap["id"], "tool": snap["tool"],
                            "outcome_unknown": True}))
            elif status in ("done", "failed") and not snap["collected"] \
                    and snap.get("ended_at") \
                    and now - float(snap["ended_at"]) > UNSEEN_RESULT_S:
                out.append(Candidate(
                    domain=self.domain, kind="job",
                    subject=f"{snap['tool'] or 'job'} job",
                    subject_key=f"job:{snap['id']}",
                    reason="orphaned",
                    detail={"job_id": snap["id"], "tool": snap["tool"],
                            "status": status,
                            "late_result": snap["late_result"]}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        job_id = str(case.detail.get("job_id") or "")
        job = self.registry.poll(job_id)
        if job is None:
            return [EvidenceItem(
                source="core.jobs",
                summary=("job is no longer in the registry — its "
                         "terminal record was swept; nothing can be "
                         "read back"))]
        snap = job.snapshot()
        status = snap["status"]
        establishes = ""
        if status == "done":
            establishes = "succeeded"
        elif status == "failed" and not snap["outcome_unknown"]:
            establishes = "failed"
        elif snap["late_result"] and job.result is not None:
            # The registry's own read-back: the worker DID finish and
            # its result is recorded — that record, not the
            # presumed-dead status, is the evidence (doc 15 §8: a
            # timeout never meant no effect; safe read-back resolves).
            establishes = "succeeded"
        summary = (f"job status is {status}; "
                   f"outcome_unknown={snap['outcome_unknown']}; "
                   f"late_result={snap['late_result']}; "
                   f"collected={snap['collected']}")
        if snap["error"]:
            summary += f"; registry error note: {snap['error'][:200]}"
        if status == "done" and snap["late_result"]:
            summary += ("; the result arrived AFTER the job was "
                        "presumed dead and is recorded by the registry "
                        "as a late result — the effect did complete")
        return [EvidenceItem(source="core.jobs", summary=summary,
                             establishes=establishes)]


class OutcomeSteward:
    """Opens cases from domain records; investigates them under
    delegation contracts; resolves only on cited evidence."""

    def __init__(self, db: Any, *, cases: OutcomeCaseStore | None = None,
                 delegation: DelegationStore | None = None,
                 adapters: list[Any] | None = None,
                 registry: JobRegistry | None = None,
                 idempotency: Any = None) -> None:
        self.db = db
        self.cases = cases if cases is not None else OutcomeCaseStore(db)
        self.delegation = delegation if delegation is not None \
            else DelegationStore(db)
        if adapters is None:
            adapters = [MissionsAdapter(db), JobsAdapter(registry)]
        self.adapters = {a.domain: a for a in adapters}
        self._idempotency = idempotency

    @property
    def idempotency(self) -> Any:
        if self._idempotency is None:
            from ..missions.idempotency import IdempotencyStore

            self._idempotency = IdempotencyStore(self.db)
        return self._idempotency

    # -- discovery ---------------------------------------------------------

    def sweep(self, now: float | None = None) -> dict[str, Any]:
        """One discovery pass: open a case per candidate operation.

        Re-sweeps never duplicate a case (the store dedupes by
        operation identity + reason). A broken adapter is recorded
        and never blocks the others.
        """
        now = time.time() if now is None else now
        report: dict[str, Any] = {"opened": 0, "existing": 0,
                                  "reopened": 0, "candidates": 0,
                                  "errors": []}
        for domain, adapter in self.adapters.items():
            try:
                candidates = adapter.discover(now)
            except Exception as exc:  # noqa: BLE001 - contained per domain
                report["errors"].append(
                    {"domain": domain, "error": str(exc)[:200]})
                continue
            for candidate in candidates:
                report["candidates"] += 1
                before = self.cases.case_for(
                    domain=candidate.domain,
                    subject_key=candidate.subject_key,
                    reason=candidate.reason)
                case, created = self.cases.open_case(
                    domain=candidate.domain, kind=candidate.kind,
                    subject=candidate.subject,
                    subject_key=candidate.subject_key,
                    reason=candidate.reason, detail=candidate.detail,
                    actor="outcome-steward")
                if created:
                    report["opened"] += 1
                elif before is not None and before.closed_at \
                        and case.state == "open":
                    report["reopened"] += 1
                else:
                    report["existing"] += 1
        return report

    # -- investigation -------------------------------------------------------

    def run_case(self, case_id: str, now: float | None = None
                 ) -> dict[str, Any]:
        """Investigate one case under a fresh delegation contract.

        Flow: claim the case (generation bumps — older runs are
        fenced) → open a contract (worker ≠ verifier, skeptic named,
        consequential) → the worker gathers read-only domain evidence
        (each item also lands as a case receipt; the receipt id is the
        claim's evidence ref) → the worker proposes the verdict the
        evidence establishes, or states plainly that it establishes
        none → the skeptic challenges the verdict claim → the verifier
        judges the TRAIL: a proposal stands only when its establishing
        evidence receipt exists on the case; an "unestablished" claim
        stands only when checks actually ran and none establishes an
        outcome → sign-off → completion gate → ONLY THEN the case
        transitions (resolve with the cited receipt, or close-unknown
        with the checks recorded).
        """
        now = time.time() if now is None else now
        case = self.cases.get(case_id)
        if case is None:
            return {"case_id": case_id, "outcome": "missing"}
        if case.terminal:
            return {"case_id": case_id, "outcome": "already-terminal",
                    "state": case.state, "verdict": case.verdict}
        adapter = self.adapters.get(case.domain)
        if adapter is None:
            return {"case_id": case_id, "outcome": "no-adapter",
                    "domain": case.domain}
        case, generation = self.cases.claim(case_id, actor=WORKER)
        contract = self.delegation.create_contract(
            goal=(f"establish the outcome of {case.domain} operation "
                  f"{case.subject} (case opened: {case.reason})"),
            title=f"outcome case: {case.subject}"[:60],
            deliverable=("an evidence-cited verdict, or a recorded "
                         "unknown with the checks named"),
            worker=WORKER, verifier=VERIFIER, skeptic=SKEPTIC,
            integration_owner="outcome-steward",
            recovery_supervisor=SUPERVISOR, arbiter="owner",
            specialist_class="outcome-steward",
            consequential=True,
            required_evidence=("domain-evidence", "verdict"),
            scope=(case.domain, case.kind), step_budget=8,
            deadline=now + 3600.0)
        self.cases.note_run(case_id, contract_id=contract.id,
                            generation=generation)
        try:
            items = adapter.collect(case, now)[:MAX_EVIDENCE_PER_RUN]
        except Exception as exc:  # noqa: BLE001 - a failed probe is data
            self.cases.add_receipt(
                case_id, kind="check", source="outcome-steward",
                actor=WORKER, generation=generation,
                note=f"evidence collection failed: {exc}"[:300])
            self.delegation.report_failure(
                contract.id, by=WORKER,
                detail=f"evidence collection failed: {exc}"[:300])
            return {"case_id": case_id, "outcome": "incomplete",
                    "state": "investigating",
                    "reason": "evidence collection failed"}
        if not items:
            items = [EvidenceItem(
                source=case.domain,
                summary="the domain adapter gathered no observations")]

        receipt_by_item: dict[int, str] = {}
        claims = []
        for index, item in enumerate(items):
            receipt = self.cases.add_receipt(
                case_id, kind="evidence", source=item.source,
                actor=WORKER, generation=generation, note=item.summary,
                evidence={"establishes": item.establishes,
                          "ref": item.ref})
            receipt_by_item[index] = receipt.id
            claims.append(self.delegation.submit_evidence(
                contract.id, kind="domain-evidence", claim=item.summary,
                producer=WORKER, evidence_ref=receipt.id,
                generation=contract.generation))

        establishing = _first_establishing(items)
        if establishing is not None:
            est_index = items.index(establishing)
            est_receipt = receipt_by_item[est_index]
            verdict_claim_text = (
                f"outcome is {establishing.establishes}: "
                f"{establishing.summary}")
        else:
            est_receipt = ""
            verdict_claim_text = (
                "outcome could not be established from the domain's "
                "records; the checks are on the case and none of them "
                "establishes a result")
        verdict_claim = self.delegation.submit_evidence(
            contract.id, kind="verdict", claim=verdict_claim_text,
            producer=WORKER, evidence_ref=est_receipt,
            generation=contract.generation)

        challenge = self.delegation.challenge(
            contract.id, challenger=SKEPTIC,
            claim=("the verdict must rest on cited domain evidence — "
                   "liveness, timing or absence alone establishes no "
                   "outcome"),
            target_claim_id=verdict_claim.id)
        # The verifier judges the trail, deterministically: a proposed
        # verdict stands only when its cited receipt is a real
        # evidence receipt on this case from THIS run; the unknown
        # claim stands only when evidence was actually gathered and
        # no item establishes an outcome.
        trail_ok = False
        if establishing is not None:
            cited = self.cases.get_receipt(est_receipt)
            trail_ok = cited is not None and cited.case_id == case_id \
                and cited.kind == "evidence" \
                and cited.generation == generation
        else:
            trail_ok = bool(claims)
        if trail_ok:
            for claim in claims:
                self.delegation.verify_claim(
                    claim.id, validator=VERIFIER, verdict="supported")
            self.delegation.verify_claim(
                verdict_claim.id, validator=VERIFIER,
                verdict="supported")
            self.delegation.resolve_challenge(
                challenge.id, resolver=VERIFIER, verdict="supported",
                note="verdict rests on the cited evidence trail")
        else:
            self.delegation.resolve_challenge(
                challenge.id, resolver=VERIFIER, verdict="contradicted",
                note="the evidence trail does not support the verdict "
                     "claim — no transition is taken")
            self.cases.add_receipt(
                case_id, kind="check", source="outcome-steward",
                actor=VERIFIER, generation=generation,
                note="verifier contradicted the verdict claim; the "
                     "case stays open and untransitioned")
            return {"case_id": case_id, "outcome": "incomplete",
                    "state": "investigating",
                    "reason": "evidence trail contradicted the verdict"}
        self.delegation.sign_off(contract.id, verifier=VERIFIER,
                                 note="evidence trail reviewed")
        decision = self.delegation.complete(
            contract.id, by=VERIFIER, idempotency=self.idempotency)
        if not decision.ok:
            self.cases.add_receipt(
                case_id, kind="check", source="outcome-steward",
                actor=VERIFIER, generation=generation,
                note=("completion gate refused: "
                      + "; ".join(decision.reasons))[:300])
            return {"case_id": case_id, "outcome": "incomplete",
                    "state": "investigating",
                    "reason": "completion gate refused",
                    "blockers": decision.reasons}
        if establishing is not None:
            closed = self.cases.resolve(
                case_id, verdict=establishing.establishes,
                evidence_receipt_id=est_receipt, generation=generation,
                actor=VERIFIER,
                note=f"resolved: {establishing.establishes} "
                     f"(contract completed; evidence cited)")
            return {"case_id": case_id, "outcome": "resolved",
                    "state": closed.state, "verdict": closed.verdict}
        checked = "; ".join(
            f"{item.source}: {item.summary}" for item in items)
        closed = self.cases.close_unknown(
            case_id, generation=generation, actor=VERIFIER,
            note=("outcome unknown after the steward's checks — "
                  f"checked: {checked}")[:900])
        return {"case_id": case_id, "outcome": "closed-unknown",
                "state": closed.state, "verdict": closed.verdict}

    def run_pending(self, *, limit: int = 25, now: float | None = None
                    ) -> dict[str, Any]:
        """Run every claimable case once, oldest first."""
        now = time.time() if now is None else now
        pool = self.cases.list_cases(state="open", limit=200) \
            + self.cases.list_cases(state="investigating", limit=200)
        pool.sort(key=lambda c: c.opened_at)
        results = []
        for case in pool[: int(limit)]:
            results.append(self.run_case(case.id, now=now))
        report: dict[str, Any] = {"ran": len(results), "results": results}
        for key in ("resolved", "closed-unknown", "incomplete"):
            report[key] = sum(1 for r in results
                              if r.get("outcome") == key)
        return report


def run_stewardship_cycle(db: Any, *, now: float | None = None,
                          registry: JobRegistry | None = None
                          ) -> dict[str, Any]:
    """The durable job's entry: sweep for candidates, then run every
    claimable case once. Returns the combined report."""
    steward = OutcomeSteward(db, registry=registry)
    sweep = steward.sweep(now=now)
    runs = steward.run_pending(now=now)
    return {"sweep": sweep, "runs": runs,
            "counts": steward.cases.counts()}


# ── inspection read API (P6 puts chat/CLI clothes on this) ──────────────


def inspect_case(cases: OutcomeCaseStore,
                 delegation: DelegationStore, case_id: str
                 ) -> dict[str, Any] | None:
    """One case with its full receipt trail and, when a steward run
    investigated it, the delegation contract's summary. None when the
    case does not exist."""
    case = cases.get(case_id)
    if case is None:
        return None
    out: dict[str, Any] = {
        "id": case.id, "domain": case.domain, "kind": case.kind,
        "subject": case.subject, "reason": case.reason,
        "state": case.state, "verdict": case.verdict,
        "generation": case.generation, "opened_at": case.opened_at,
        "closed_at": case.closed_at, "detail": dict(case.detail),
        "evidence_receipt_id": case.evidence_receipt_id,
        "receipts": [
            {"kind": r.kind, "source": r.source, "actor": r.actor,
             "note": r.note, "at": r.created_at,
             "generation": r.generation}
            for r in cases.receipts(case_id)],
        "contract": None,
    }
    if case.contract_id:
        contract = delegation.get_contract(case.contract_id)
        if contract is not None:
            out["contract"] = delegation.summary(contract)
    return out


def render_case(case: OutcomeCase,
                receipts: list[Any]) -> str:
    """Owner-readable case text. Identity handles (case/contract ids,
    subject keys) never print — subject, state, verdict and the
    receipt trail are the surface."""
    verdict_line = case.verdict if case.state == "resolved" \
        else "outcome unknown"
    lines = [
        f"Outcome case — {case.subject} ({case.domain})",
        f"opened for: {case.reason} · state: {case.state} · "
        f"verdict: {verdict_line}",
    ]
    for receipt in receipts:
        note = receipt.note
        if len(note) > 140:
            note = note[:137] + "..."
        lines.append(f"• {receipt.kind} — {receipt.source}: {note}")
    return "\n".join(lines)


# ── durable job + tool wiring ────────────────────────────────────────────


def ensure_outcome_steward_job(scheduler: Any) -> list[dict[str, Any]]:
    """Register the durable steward sweep (idempotent by name, the
    social/jobs.py pattern): every 6h, fire the ``outcome_steward``
    tool's ``sweep`` action. Routine: findings surface through the
    case records and the P6 inspection surface, not job alerts — a
    failed sweep still records as a failed run."""
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    if STEWARD_JOB in have:
        return [{"name": STEWARD_JOB, "already_scheduled": True}]
    job = scheduler.add(
        STEWARD_JOB, STEWARD_SCHEDULE, "tool",
        {"tool": "outcome_steward", "args": {"action": "sweep"}},
        missed_fire_policy="skip", overlap_policy="skip",
        run_timeout_s=1800, routine=True)
    out.append({"name": STEWARD_JOB, "scheduled": True,
                "job_id": job.get("id")})
    return out


def register(registry: Any) -> None:
    """The ``outcome_steward`` tool — the durable job's payload and
    the brain's route to the steward. action=sweep runs one full
    stewardship cycle (discovery + investigations); action=status
    reports case counts without running anything."""
    context = registry.context

    @registry.register(
        "outcome_steward",
        description=(
            "Outcome stewardship for long-running or forgotten work: "
            "sweeps missions and background jobs for operations whose "
            "outcome is overdue, stalled, orphaned or unknown, opens "
            "a reconciliation case for each, and investigates under "
            "an accountable contract — a case resolves only on cited "
            "evidence from the domain's own records; anything else "
            "closes as honestly unknown with the checks recorded. "
            "action=sweep runs one full cycle (this is what the "
            "durable 'outcome steward sweep' job fires every 6h); "
            "action=status reports case counts only. Call directly "
            "when the owner asks what became of work that was left "
            "running."
        ),
        capability="memory.write",
        parameters={
            "action": "str — sweep | status",
        },
    )
    def _outcome_steward(action: str = "sweep") -> dict[str, Any]:
        action = (action or "sweep").strip().lower()
        db = getattr(context, "db", None)
        if db is None:
            return {"ok": False,
                    "error": "no database on the context — the "
                             "steward has no records to keep"}
        if action == "status":
            return {"ok": True,
                    "counts": OutcomeCaseStore(db).counts()}
        if action == "sweep":
            report = run_stewardship_cycle(db)
            return {"ok": True, **report}
        return {"ok": False, "error": f"unknown action {action!r}"}
