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
READ-ONLY — they gather evidence from the domain's own records and
never call a mutating domain path.

P5 completes the domain set the records can honestly support —
delegation contracts, the durable scheduler's persisted jobs and the
chat outbox — and wires the mission effect-outcome projection
(:meth:`MissionStore.effect_outcome`) into the missions adapter: a
liveness failure whose effect the mission's own records establish no
longer opens an unknown-outcome case, and a case that does exist
cites the projection as its evidence. Four briefed domains were
deliberately NOT adapted at P5 — browser runs, downloads/uploads,
media render jobs, artifacts — because none of them kept a durable
record an honest discovery could read; :func:`adapter_health` named
each absence and the missing record, on the record, instead of
papering over it with a fabricated adapter. Doc 18 P1 built the two
missing records (``browser_run_records``, ``media_job_records``), so
the browser and render adapters now work against them. The last two
absences — transfers and artifacts — were closed by the delivery
record registry (:mod:`nomorals.storage.delivery_records`): every
artifact hand-off journals a transfer entry at the seam where it
happens and a delivery receipt when it lands, and every
artifact-carrying publication journals a publication receipt tied
to its post — so the transfers and artifacts adapters now work
against those records, and :func:`adapter_health` names no
unavailable domain.

Runs are deterministic by design: deterministic read-back is
sufficient evidence for the domains covered here, and no model call
is spent where records already answer the question (doc 15 §9). The
contract's verifier judges the EVIDENCE TRAIL — a proposed verdict
stands only when it cites an evidence receipt gathered in the same
run — never the worker's say-so.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.jobs import JobRegistry, shared_registry
from ..missions.mission import MissionStatus, MissionStore, mission_liveness
from ..storage.delivery_records import DeliveryRecordStore
from .delegation import DelegationStore
from .outcome_cases import (
    RESOLVABLE_VERDICTS,
    OutcomeCase,
    OutcomeCaseStore,
    ack_changed,
    ack_snapshot,
)

__all__ = [
    "STEWARD_JOB",
    "Candidate",
    "EvidenceItem",
    "MissionsAdapter",
    "JobsAdapter",
    "ContractsAdapter",
    "SchedulerAdapter",
    "OutboxAdapter",
    "BrowserAdapter",
    "RenderAdapter",
    "TransfersAdapter",
    "ArtifactsAdapter",
    "OutcomeSteward",
    "adapter_health",
    "run_stewardship_cycle",
    "ensure_outcome_steward_job",
    "inspect_case",
    "render_case",
    "case_board",
    "render_case_board",
    "render_case_detail",
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

#: A delegation contract still open this long past creation, with no
#: deadline to judge it by, is stalled (P5).
CONTRACT_STALE_S = 7 * 24 * 3600.0
#: An enabled scheduler job whose next firing is this far in the past
#: was missed — the tick records every firing that happens, so a
#: next_run this stale has no run record because it never ran (P5).
SCHED_OVERDUE_S = 3600.0
#: A one-shot scheduler job whose only firing failed this long ago,
#: with nothing scheduled after it, is abandoned (P5).
SCHED_ABANDONED_S = 3600.0
#: An outbox send still queued — or leased with an expired lease —
#: this long after submission was never confirmed delivered (P5).
OUTBOX_STUCK_S = 3600.0
#: A browser run whose durable record has not transitioned for this
#: long is presumed abandoned by its process (doc 18 P1) — the live
#: manager is in-memory, so a surviving run lands its terminal
#: record; silence this long means nobody is left to land it.
BROWSER_STUCK_S = 3600.0
#: A media job whose durable record has not transitioned for this
#: long is presumed abandoned by its process (doc 18 P1).
RENDER_STUCK_S = 3600.0
#: A transfer still recorded as begun this long after the send was
#: attempted never landed its outcome record — the process died
#: mid-send or the completion write was lost, so whether the
#: artifact arrived is genuinely unknown (delivery records pass).
TRANSFER_STUCK_S = 3600.0
#: An artifact whose latest hand-off failed this long ago, with no
#: delivery since, is abandoned — the failure was surfaced at send
#: time; a day later it is forgotten work (delivery records pass).
ARTIFACT_ABANDONED_S = 24 * 3600.0


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
    mission carrying only the liveness marker is UNKNOWN-OUTCOME —
    unless the mission effect-outcome projection
    (:meth:`MissionStore.effect_outcome`, P5) establishes the effect
    from the mission's own records, in which case there is no unknown
    to open a case about. A PENDING mission unmoved for a day is
    STALLED. Nothing here calls ``reconcile()`` or ``save()`` —
    observation only.
    """

    domain = "missions"

    def __init__(self, db: Any) -> None:
        self.store = MissionStore(db)

    def _effect_projection(self, mission: Any) -> Any | None:
        """The mission's effect-outcome projection, or None when it
        cannot be computed — a projection that fails to read falls
        back to P4 behaviour (discovery proceeds, collection gathers
        the raw record), never to a guessed effect."""
        try:
            return self.store.effect_outcome(mission.id)
        except Exception:  # noqa: BLE001 - unreadable = no projection
            return None

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
                projection = self._effect_projection(mission)
                if projection is not None \
                        and projection.effect != "unknown":
                    # The mission's own records establish the effect
                    # (P5 projection) — the outcome is not unknown,
                    # so no unknown-outcome case opens. The mission
                    # status stays FAILED; both facts coexist.
                    continue
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
            projection = self._effect_projection(mission)
            evidenced = projection is not None \
                and projection.effect in RESOLVABLE_VERDICTS
            if not evidenced:
                items.append(EvidenceItem(
                    source="missions.store",
                    summary=("the recorded failure rests on liveness "
                             "evidence alone (reconcile marker); no step "
                             "failure or effect record exists beside it, "
                             "so the effect outcome is not established")))
            if projection is not None:
                for entry in projection.evidence:
                    items.append(EvidenceItem(
                        source=entry["source"],
                        summary=entry["summary"]))
                if evidenced:
                    items.append(EvidenceItem(
                        source="missions.effect_outcome",
                        summary=(
                            "effect-outcome projection: the mission's "
                            f"own records establish its effect as "
                            f"{projection.effect} — the runner status "
                            "stays FAILED (a liveness verdict about "
                            "the runner); this is a verdict about the "
                            "effect, resting on the records cited "
                            "above"),
                        establishes=projection.effect))
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


class ContractsAdapter:
    """Read-only observation of delegation contract records (P5).

    Discovery: a contract still open past its deadline is OVERDUE; a
    contract that has spent its whole step budget without completing
    is OVERDUE (the budget was its bound); a contract with no
    deadline still open a week after creation is STALLED. The
    adapter reads through the DelegationStore's own read paths and
    never creates, completes, waives or refuses a contract — the
    contracts it observes include the steward's own investigation
    contracts, which complete inside their runs and so never
    discover; one left open by a crashed run honestly would.
    """

    domain = "contracts"

    def __init__(self, db: Any) -> None:
        self.store = DelegationStore(db)

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        for contract in self.store.open_contracts(""):
            deadline = float(contract.deadline or 0.0)
            budget_spent = bool(contract.step_budget) \
                and contract.steps_used >= contract.step_budget
            if deadline and now > deadline:
                reason = "overdue"
                basis = f"deadline passed {now - deadline:.0f}s ago"
            elif budget_spent:
                reason = "overdue"
                basis = (f"step budget spent ({contract.steps_used}/"
                         f"{contract.step_budget}) without completion")
            elif now - float(contract.created_at or now) \
                    > CONTRACT_STALE_S:
                reason = "stalled"
                basis = "open a week with no deadline to judge it by"
            else:
                continue
            out.append(Candidate(
                domain=self.domain, kind="contract",
                subject=contract.title or contract.goal[:60],
                subject_key=f"contract:{contract.id}",
                reason=reason,
                detail={"contract_id": contract.id,
                        "status": contract.status,
                        "worker": contract.worker,
                        "basis": basis}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        contract_id = str(case.detail.get("contract_id") or "")
        contract = self.store.get_contract(contract_id)
        if contract is None:
            return [EvidenceItem(
                source="delegation.contracts",
                summary="the contract record no longer exists")]
        establishes = ""
        if contract.status == "completed":
            establishes = "succeeded"
        elif contract.status == "refused":
            establishes = "failed"
        elif contract.status == "waived":
            establishes = "stopped"
        summary = (f"contract status is {contract.status}; "
                   f"steps {contract.steps_used}/"
                   f"{contract.step_budget or 'unbounded'}; "
                   f"worker {contract.worker}, verifier "
                   f"{contract.verifier}")
        if contract.refusal:
            summary += f"; refusal recorded: {contract.refusal[:200]}"
        if contract.completed_at:
            summary += ("; completed "
                        f"{max(0.0, now - float(contract.completed_at)):.0f}s ago")
        items = [EvidenceItem(source="delegation.contracts",
                              summary=summary,
                              establishes=establishes)]
        claims = self.store.claims(contract_id)
        if claims:
            kinds: dict[str, int] = {}
            for claim in claims:
                kinds[claim.kind] = kinds.get(claim.kind, 0) + 1
            parts = ", ".join(f"{k} ×{n}"
                              for k, n in sorted(kinds.items()))
            items.append(EvidenceItem(
                source="delegation.claims",
                summary=(f"claim ledger holds {len(claims)} entr(ies) "
                         f"for this contract ({parts}) — context for "
                         "the contract record, never a verdict by "
                         "itself")))
        return items


class SchedulerAdapter:
    """Read-only observation of the durable scheduler's persisted
    records — the ``schedule_jobs`` / ``schedule_runs`` tables the
    agents Scheduler owns (P5).

    Discovery: an enabled job whose next firing is far in the past is
    OVERDUE — the scheduler journals every firing that happens, so a
    stale ``next_run`` with no run row means the firing never ran. A
    one-shot job whose only firing failed, with nothing scheduled
    after it, is ABANDONED. The adapter reads the tables directly
    (the scheduler's own receipt paths read the same rows) and never
    constructs the Scheduler — building one runs its lazy schema
    guards, and observation must write nothing.
    """

    domain = "scheduler"

    def __init__(self, db: Any) -> None:
        self.db = db

    def _jobs(self) -> list[dict[str, Any]]:
        try:
            return [dict(r) for r in self.db.query(
                "SELECT * FROM schedule_jobs")]
        except Exception:  # noqa: BLE001 - no table = no records
            return []

    def _runs(self, job_id: str, *, limit: int = 3
              ) -> list[dict[str, Any]]:
        try:
            return [dict(r) for r in self.db.query(
                "SELECT * FROM schedule_runs WHERE job_id = ?"
                " ORDER BY started_at DESC LIMIT ?",
                (job_id, int(limit)))]
        except Exception:  # noqa: BLE001 - no table = no records
            return []

    @staticmethod
    def _run_state(row: dict[str, Any]) -> str:
        """A run row's true state — mirrors the scheduler's AA-67
        derivation: the stored status, or ran/failed from the ok bit
        for legacy rows."""
        status = str(row.get("status") or "").strip().lower()
        if status:
            return status
        return "ran" if row.get("ok") else "failed"

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        for job in self._jobs():
            job_id = str(job.get("id") or "")
            name = str(job.get("name") or "scheduled job")
            next_run = job.get("next_run")
            if job.get("enabled") and next_run \
                    and now - float(next_run) > SCHED_OVERDUE_S:
                out.append(Candidate(
                    domain=self.domain, kind="scheduled job",
                    subject=name,
                    subject_key=f"schedjob:{job_id}",
                    reason="overdue",
                    detail={"job_id": job_id,
                            "next_run": float(next_run),
                            "last_run": job.get("last_run")}))
                continue
            if str(job.get("kind") or "") == "at" and not next_run:
                runs = self._runs(job_id, limit=1)
                if runs and self._run_state(runs[0]) == "failed" \
                        and now - float(runs[0].get("finished_at")
                                        or now) > SCHED_ABANDONED_S:
                    out.append(Candidate(
                        domain=self.domain, kind="scheduled job",
                        subject=name,
                        subject_key=f"schedjob:{job_id}",
                        reason="abandoned",
                        detail={"job_id": job_id,
                                "last_run_state": "failed"}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        job_id = str(case.detail.get("job_id") or "")
        job = None
        for row in self._jobs():
            if str(row.get("id") or "") == job_id:
                job = row
                break
        if job is None:
            return [EvidenceItem(
                source="scheduler.jobs",
                summary="the job record no longer exists")]
        items = [EvidenceItem(
            source="scheduler.jobs",
            summary=(f"job '{job.get('name')}' kind "
                     f"{job.get('kind')}; enabled="
                     f"{bool(job.get('enabled'))}; next_run="
                     f"{job.get('next_run')}; last_run="
                     f"{job.get('last_run')}"))]
        runs = self._runs(job_id)
        if runs:
            latest = runs[0]
            state = self._run_state(latest)
            establishes = ""
            if state == "ran" and latest.get("ok"):
                establishes = "succeeded"
            elif state == "failed":
                establishes = "failed"
            summary = (f"latest recorded run: {state} "
                       f"(ok={bool(latest.get('ok'))}), "
                       f"{max(0.0, now - float(latest.get('started_at') or now)):.0f}s ago")
            result = str(latest.get("result") or "").strip()
            if result:
                summary += f"; result: {result[:200]}"
            if state in ("missed", "skipped", "suppressed", "throttled"):
                summary += ("; the firing never executed — a non-run "
                            "establishes no outcome either way")
            items.append(EvidenceItem(source="scheduler.runs",
                                      summary=summary,
                                      establishes=establishes))
        else:
            items.append(EvidenceItem(
                source="scheduler.runs",
                summary=("no run was ever recorded for this job — "
                         "the scheduler journals every firing that "
                         "happens, so nothing has run")))
        return items


class OutboxAdapter:
    """Read-only observation of chat outbox sends (P5).

    The outbox's rows live in the shared ``work_queue`` table under
    per-chat topics (``chat_outbox:<platform>:<chat>``) — every
    durable send is a row before it is a network call, and the row is
    its receipt. Discovery: a send still ``ready`` long after
    submission is OVERDUE (queued, never sent); a send still
    ``leased`` with an expired lease is UNKNOWN-OUTCOME — the send
    may or may not have happened, which is exactly the steward's
    question. Collection re-reads the row: ``done`` establishes
    ``delivered``, ``dead`` establishes ``failed``, ``cancelled``
    establishes ``stopped``. The subject names the platform and the
    message's own opening words — never the chat key, which can be a
    phone number or JID.
    """

    domain = "outbox"
    _TOPIC_PREFIX = "chat_outbox:"

    def __init__(self, db: Any) -> None:
        self.db = db

    def _rows(self, where: str, params: tuple = (),
              limit: int = 200) -> list[dict[str, Any]]:
        try:
            return [dict(r) for r in self.db.query(
                "SELECT * FROM work_queue WHERE topic LIKE ? "
                f"AND {where} ORDER BY created_at LIMIT ?",
                (self._TOPIC_PREFIX + "%", *params, int(limit)))]
        except Exception:  # noqa: BLE001 - no table = no records
            return []

    @staticmethod
    def _subject(row: dict[str, Any]) -> str:
        try:
            payload = json.loads(row.get("payload") or "{}")
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        platform = str(payload.get("platform") or "chat")
        text = str(payload.get("text") or "").strip()
        if text:
            return f"{platform} message: {text[:40]}"
        if payload.get("media"):
            return f"{platform} media send"
        return f"{platform} queued send"

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        for row in self._rows("status IN ('ready', 'leased')"):
            created = float(row.get("created_at") or now)
            age = now - created
            if age <= OUTBOX_STUCK_S:
                continue
            status = str(row.get("status") or "")
            lease_until = float(row.get("lease_until") or 0.0)
            if status == "leased" and lease_until \
                    and now <= lease_until:
                continue  # a live lease — a send in flight, not stuck
            reason = "unknown-outcome" if status == "leased" \
                else "overdue"
            out.append(Candidate(
                domain=self.domain, kind="send",
                subject=self._subject(row),
                subject_key=f"outbox:{row.get('id')}",
                reason=reason,
                detail={"job_id": str(row.get("id") or ""),
                        "queue_status": status,
                        "attempts": int(row.get("attempts") or 0),
                        "age_s": round(age, 1)}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        job_id = str(case.detail.get("job_id") or "")
        rows = self._rows("id = ?", (job_id,), limit=1)
        if not rows:
            return [EvidenceItem(
                source="outbox.queue",
                summary=("the send's queue row no longer exists "
                         "(purged after completion, or never "
                         "persisted) — delivery cannot be read back"))]
        row = rows[0]
        status = str(row.get("status") or "")
        establishes = {"done": "delivered", "dead": "failed",
                       "cancelled": "stopped"}.get(status, "")
        summary = (f"send row status is {status}; attempts "
                   f"{int(row.get('attempts') or 0)}/"
                   f"{int(row.get('max_attempts') or 0)}; queued "
                   f"{max(0.0, now - float(row.get('created_at') or now)):.0f}s ago")
        error = str(row.get("last_error") or "").strip()
        if error:
            summary += f"; last error: {error[:200]}"
        return [EvidenceItem(source="outbox.queue", summary=summary,
                              establishes=establishes)]


class BrowserAdapter:
    """Read-only observation of durable browser-run records (doc 18
    P1 — the ``browser_run_records`` table the BrowserJobManager now
    writes at run start and on every transition).

    Discovery: a record still in a non-terminal state whose last
    transition is far in the past belongs to a process that died
    mid-run (the live manager is in-memory; a surviving process would
    have landed a terminal record). A run that never left ``queued``
    is OVERDUE — nothing has happened yet. A run stuck at ``running``
    / ``waiting_owner`` is UNKNOWN-OUTCOME — it may already have
    acted on the site, which is exactly the steward's question.
    Collection re-reads the record: ``done`` establishes
    ``succeeded``, ``blocked`` / ``failed`` establish ``failed`` (the
    summary names the wall outcome / stop reason), ``cancelled``
    establishes ``stopped``. The adapter reads the table directly
    (the scheduler precedent) and never constructs the manager —
    building one starts from in-memory state and observation must
    write nothing. Subjects use the record's redacted URL — never a
    raw URL, which can carry credentials in its query.
    """

    domain = "browser"

    def __init__(self, db: Any) -> None:
        self.db = db

    def _rows(self, where: str, params: tuple = (),
              limit: int = 200) -> list[dict[str, Any]]:
        try:
            return [dict(r) for r in self.db.query(
                "SELECT * FROM browser_run_records "
                f"WHERE {where} ORDER BY created_at LIMIT ?",
                (*params, int(limit)))]
        except Exception:  # noqa: BLE001 - no table = no records
            return []

    @staticmethod
    def _subject(row: dict[str, Any]) -> str:
        host = ""
        try:
            from urllib.parse import urlsplit
            host = urlsplit(str(row.get("url") or "")).hostname or ""
        except ValueError:
            host = ""
        goal = str(row.get("goal") or "").strip()
        where = host or "browser run"
        return f"browser run: {where} — {goal[:40]}" if goal \
            else f"browser run: {where}"

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        rows = self._rows(
            "status IN ('queued', 'running', 'waiting_owner')")
        for row in rows:
            updated = float(row.get("updated_at")
                            or row.get("created_at") or now)
            age = now - updated
            if age <= BROWSER_STUCK_S:
                continue
            status = str(row.get("status") or "")
            reason = "overdue" if status == "queued" \
                else "unknown-outcome"
            out.append(Candidate(
                domain=self.domain, kind="browser run",
                subject=self._subject(row),
                subject_key=f"browser:{row.get('run_id')}",
                reason=reason,
                detail={"run_id": str(row.get("run_id") or ""),
                        "record_status": status,
                        "cancel_state": str(row.get("cancel_state")
                                            or ""),
                        "age_s": round(age, 1)}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        run_id = str(case.detail.get("run_id") or "")
        rows = self._rows("run_id = ?", (run_id,), limit=1)
        if not rows:
            return [EvidenceItem(
                source="browser.run_records",
                summary=("the run's durable record no longer exists "
                         "— what the run did cannot be read back"))]
        row = rows[0]
        status = str(row.get("status") or "")
        establishes = {"done": "succeeded", "blocked": "failed",
                       "failed": "failed",
                       "cancelled": "stopped"}.get(status, "")
        summary = (f"run record status is {status}; "
                   f"{int(row.get('steps') or 0)} steps; wall: "
                   f"{str(row.get('wall_outcome') or 'no wall')}")
        stopped = str(row.get("stopped_reason") or "").strip()
        if stopped:
            summary += f"; stopped: {stopped[:200]}"
        error = str(row.get("error") or "").strip()
        if error:
            summary += f"; error: {error[:200]}"
        if str(row.get("cancel_state") or "") == "unhonored":
            summary += ("; a cancel was requested but the run ended "
                        "without honoring it")
        return [EvidenceItem(source="browser.run_records",
                             summary=summary,
                             establishes=establishes)]


class RenderAdapter:
    """Read-only observation of durable media job records (doc 18
    P1 — the ``media_job_records`` table the media JobManager now
    writes at submit and on every transition).

    Discovery mirrors the browser adapter: a record stuck at
    ``queued`` past the threshold is OVERDUE (never picked up); a
    record stuck at ``running`` is UNKNOWN-OUTCOME — the render may
    have partially written its output, and only the record's terminal
    transition (which never landed) could say otherwise. Collection:
    ``done`` establishes ``succeeded`` (the summary names the output
    the job produced), ``failed`` establishes ``failed``,
    ``cancelled`` establishes ``stopped``. Reads the table directly;
    never constructs the JobManager (construction starts a worker
    thread — observation must write nothing and start nothing).
    """

    domain = "render"

    def __init__(self, db: Any) -> None:
        self.db = db

    def _rows(self, where: str, params: tuple = (),
              limit: int = 200) -> list[dict[str, Any]]:
        try:
            return [dict(r) for r in self.db.query(
                "SELECT * FROM media_job_records "
                f"WHERE {where} ORDER BY created_at LIMIT ?",
                (*params, int(limit)))]
        except Exception:  # noqa: BLE001 - no table = no records
            return []

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        for row in self._rows("status IN ('queued', 'running')"):
            updated = float(row.get("updated_at")
                            or row.get("created_at") or now)
            age = now - updated
            if age <= RENDER_STUCK_S:
                continue
            status = str(row.get("status") or "")
            reason = "overdue" if status == "queued" \
                else "unknown-outcome"
            label = str(row.get("label") or "media job").strip()
            out.append(Candidate(
                domain=self.domain, kind="media job",
                subject=f"media job: {label[:40]}",
                subject_key=f"render:{row.get('job_id')}",
                reason=reason,
                detail={"job_id": str(row.get("job_id") or ""),
                        "record_status": status,
                        "job_kind": str(row.get("kind") or ""),
                        "age_s": round(age, 1)}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        job_id = str(case.detail.get("job_id") or "")
        rows = self._rows("job_id = ?", (job_id,), limit=1)
        if not rows:
            return [EvidenceItem(
                source="render.job_records",
                summary=("the job's durable record no longer exists "
                         "— what the job produced cannot be read "
                         "back"))]
        row = rows[0]
        status = str(row.get("status") or "")
        establishes = {"done": "succeeded", "failed": "failed",
                       "cancelled": "stopped"}.get(status, "")
        summary = f"job record status is {status}"
        output = str(row.get("output_ref") or "").strip()
        if output:
            summary += f"; output: {output[:200]}"
        error = str(row.get("error") or "").strip()
        if error:
            summary += f"; error: {error[:200]}"
        return [EvidenceItem(source="render.job_records",
                             summary=summary,
                             establishes=establishes)]


class TransfersAdapter:
    """Read-only observation of the delivery record registry's
    transfer journal (:mod:`nomorals.storage.delivery_records`).

    Every artifact hand-off (``/artifact send`` / ``export``) writes
    a transfer entry BEFORE the send is attempted and lands its
    outcome when the send returns. Discovery: an entry still
    ``begun`` long past its attempt is UNKNOWN-OUTCOME — the process
    died mid-send or the completion write was lost, so whether the
    artifact arrived is exactly the steward's question. Collection
    re-reads the entry: ``delivered`` establishes ``delivered``
    (citing the delivery receipt the completion minted), ``failed``
    establishes ``failed`` with the recorded error, still ``begun``
    establishes nothing. The subject names the platform and the
    artifact's label — never the chat key, which can be a phone
    number or JID (the outbox adapter's discipline).
    """

    domain = "transfers"

    def __init__(self, db: Any) -> None:
        self.records = DeliveryRecordStore(db)

    @staticmethod
    def _subject(entry: Any) -> str:
        label = str(entry.artifact_label or "artifact").strip()
        return f"{entry.platform or 'chat'} delivery: {label[:48]}"

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        for entry in self.records.stuck_transfers(
                now, older_than_s=TRANSFER_STUCK_S):
            age = now - float(entry.begun_at or now)
            out.append(Candidate(
                domain=self.domain, kind="transfer",
                subject=self._subject(entry),
                subject_key=f"transfer:{entry.id}",
                reason="unknown-outcome",
                detail={"transfer_id": entry.id,
                        "artifact_id": entry.artifact_id,
                        "platform": entry.platform,
                        "age_s": round(age, 1)}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        transfer_id = str(case.detail.get("transfer_id") or "")
        if not transfer_id and case.subject_key.startswith("transfer:"):
            transfer_id = case.subject_key.split(":", 1)[1]
        entry = self.records.get_transfer(transfer_id) if transfer_id \
            else None
        if entry is None:
            return [EvidenceItem(
                source="transfers.registry",
                summary=("the transfer record no longer exists — "
                         "whether the hand-off landed cannot be "
                         "read back"))]
        age = max(0.0, now - float(entry.begun_at or now))
        if entry.state == "delivered":
            receipt = self.records.delivery_receipt_for_transfer(
                entry.id)
            summary = (f"the registry records this {entry.kind} "
                       f"delivered to {entry.platform or 'chat'} "
                       f"{age:.0f}s after it began; the delivery "
                       "receipt minted at completion is the citation")
            return [EvidenceItem(
                source="transfers.registry", summary=summary,
                establishes="delivered",
                ref=receipt.id if receipt is not None else entry.id)]
        if entry.state == "failed":
            summary = (f"the registry records this {entry.kind} to "
                       f"{entry.platform or 'chat'} as failed")
            if entry.error:
                summary += f"; recorded error: {entry.error[:200]}"
            return [EvidenceItem(
                source="transfers.registry", summary=summary,
                establishes="failed", ref=entry.id)]
        return [EvidenceItem(
            source="transfers.registry",
            summary=(f"the transfer is still recorded as begun "
                     f"{age:.0f}s after the send was attempted and "
                     "no outcome was ever recorded — the send may "
                     "or may not have happened; the record cannot "
                     "say"))]


class ArtifactsAdapter:
    """Read-only observation of the delivery record registry from
    the artifact's side: did this artifact ever reach anyone?

    Discovery: an artifact whose latest hand-off failed a day ago
    with no delivery since is ABANDONED — the failure was surfaced
    at send time; a day later it is forgotten work, and the case
    asks what became of the delivery. Collection reads the
    artifact's receipts: a delivery receipt establishes
    ``delivered`` (citing the receipt), a posted publication
    receipt establishes ``delivered`` (citing the receipt — the
    artifact reached its platform audience), failure records alone
    establish ``failed``, and no records at all establishes
    nothing — the honest answer that nothing in the registry
    records this artifact ever being delivered.
    """

    domain = "artifacts"

    def __init__(self, db: Any) -> None:
        self.records = DeliveryRecordStore(db)

    def discover(self, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        for entry in self.records.abandoned_deliveries(
                now, older_than_s=ARTIFACT_ABANDONED_S):
            label = str(entry.artifact_label or "artifact").strip()
            out.append(Candidate(
                domain=self.domain, kind="artifact delivery",
                subject=f"artifact delivery: {label[:48]}",
                subject_key=f"artifact:{entry.artifact_id}",
                reason="abandoned",
                detail={"artifact_id": entry.artifact_id,
                        "platform": entry.platform,
                        "failure_age_s": round(
                            max(0.0, now - float(entry.finished_at
                                                 or now)), 1)}))
        return out

    def collect(self, case: OutcomeCase, now: float) -> list[EvidenceItem]:
        artifact_id = str(case.detail.get("artifact_id") or "")
        if not artifact_id and case.subject_key.startswith("artifact:"):
            artifact_id = case.subject_key.split(":", 1)[1]
        if not artifact_id:
            return [EvidenceItem(
                source="artifacts.delivery_records",
                summary="the case names no artifact — nothing can "
                        "be read back")]
        deliveries = self.records.delivery_receipts_for_artifact(
            artifact_id)
        publications = self.records.publication_receipts_for_artifact(
            artifact_id)
        transfers = self.records.transfers_for_artifact(artifact_id)
        items: list[EvidenceItem] = []
        if deliveries:
            latest = deliveries[-1]
            age = max(0.0, now - float(latest.created_at or now))
            items.append(EvidenceItem(
                source="artifacts.delivery_receipts",
                summary=(f"a delivery receipt records this artifact "
                         f"delivered to {latest.platform or 'chat'} "
                         f"{age:.0f}s ago — the send's own reference "
                         "is on the receipt"),
                establishes="delivered", ref=latest.id))
        posted = [p for p in publications if p.outcome == "posted"]
        if posted:
            latest = posted[-1]
            age = max(0.0, now - float(latest.created_at or now))
            items.append(EvidenceItem(
                source="artifacts.publication_receipts",
                summary=(f"a publication receipt records this "
                         f"artifact posted to "
                         f"{latest.platform or 'a platform'} "
                         f"{age:.0f}s ago as part of a published "
                         f"draft (media carried "
                         f"{latest.media_carried} of "
                         f"{latest.media_offered} on that platform; "
                         "the post's own reference is on the "
                         "receipt)"),
                establishes="delivered", ref=latest.id))
        if items:
            failures = [t for t in transfers if t.state == "failed"]
            failures += [p for p in publications
                         if p.outcome != "posted"]
            if failures:
                items.append(EvidenceItem(
                    source="artifacts.delivery_records",
                    summary=(f"the registry also holds "
                             f"{len(failures)} failed attempt(s) for "
                             "this artifact — recorded, and superseded "
                             "by the delivery above where one landed")))
            return items
        failed_transfers = [t for t in transfers if t.state == "failed"]
        failed_pubs = [p for p in publications if p.outcome != "posted"]
        if failed_transfers or failed_pubs:
            latest_error = ""
            ref = ""
            if failed_transfers:
                ref = failed_transfers[-1].id
                latest_error = failed_transfers[-1].error
            if failed_pubs:
                ref = failed_pubs[-1].id
                latest_error = latest_error or failed_pubs[-1].error
            summary = ("the registry holds only failure(s) for this "
                       "artifact — "
                       f"{len(failed_transfers)} failed hand-off(s), "
                       f"{len(failed_pubs)} failed publication(s); "
                       "no delivery or publication ever landed")
            if latest_error:
                summary += f"; latest recorded error: " \
                           f"{latest_error[:200]}"
            return [EvidenceItem(
                source="artifacts.delivery_records", summary=summary,
                establishes="failed", ref=ref)]
        if transfers:
            states = ", ".join(sorted({t.state for t in transfers}))
            return [EvidenceItem(
                source="artifacts.delivery_records",
                summary=(f"the registry holds {len(transfers)} "
                         f"transfer entr(ies) for this artifact "
                         f"(states: {states}) but no delivery or "
                         "publication receipt — no completed delivery "
                         "is recorded"))]
        return [EvidenceItem(
            source="artifacts.delivery_records",
            summary=("no delivery or publication record exists for "
                     "this artifact — nothing in the registry "
                     "records it ever being delivered or published"))]


#: Adapter coverage, stated on the record: the domains the steward
#: observes, and the briefed domains it deliberately does NOT —
#: each with the durable record whose absence makes honest discovery
#: impossible today. Surfaced by the tool's status action; no domain
#: is ever silently missing.
_WORKING_ADAPTERS: tuple[tuple[str, str], ...] = (
    ("missions", "MissionStore records, incl. the effect-outcome "
                 "projection"),
    ("jobs", "core JobRegistry snapshots"),
    ("contracts", "delegation contract records + claim ledger"),
    ("scheduler", "the durable scheduler's schedule_jobs / "
                  "schedule_runs records"),
    ("outbox", "chat outbox rows in the shared work_queue"),
    ("browser", "browser_run_records — the browser job manager's "
                "durable run records (doc 18 P1)"),
    ("render", "media_job_records — the media job manager's durable "
               "job records (doc 18 P1); renders run as missions stay "
               "covered by the missions adapter"),
    ("transfers", "transfer_entries — the delivery record registry's "
                  "transfer journal: one entry per artifact hand-off, "
                  "begun at the send seam and landed delivered/failed "
                  "with its outcome"),
    ("artifacts", "delivery_receipts + publication_receipts — the "
                  "delivery record registry's receipts: whether an "
                  "artifact was delivered to a chat or published to a "
                  "platform, under the send/post reference it landed "
                  "with"),
)

#: Briefed domains the steward deliberately does NOT observe, each
#: with the durable record whose absence makes honest discovery
#: impossible. Both P5 absences (transfers, artifacts) were closed
#: by the delivery record registry and moved to the working set
#: above; the mechanism stays — a future gap is named here, never
#: silently missing.
_UNAVAILABLE_ADAPTERS: tuple[tuple[str, str], ...] = ()


def adapter_health() -> list[dict[str, str]]:
    """Per-domain adapter health for the status surface: every
    domain the steward was briefed on, named — working with its
    record source, or unavailable with the missing record."""
    rows = [{"domain": domain, "status": "working", "note": note}
            for domain, note in _WORKING_ADAPTERS]
    rows += [{"domain": domain, "status": "unavailable", "note": note}
             for domain, note in _UNAVAILABLE_ADAPTERS]
    return rows


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
            adapters = [MissionsAdapter(db), JobsAdapter(registry),
                        ContractsAdapter(db), SchedulerAdapter(db),
                        OutboxAdapter(db), BrowserAdapter(db),
                        RenderAdapter(db), TransfersAdapter(db),
                        ArtifactsAdapter(db)]
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
        held = 0
        for case in pool[: int(limit)]:
            # P6: a case the owner acknowledged, unchanged since, is
            # not re-run by the sweep — re-investigating it every
            # cycle would re-surface it in its own record.  An
            # explicit run_case (the owner's verify request) always
            # runs; a case whose state/verdict moved since the ack
            # (ack_changed) runs and resurfaces.
            if ack_snapshot(case) is not None and not ack_changed(case):
                held += 1
                results.append({"case_id": case.id,
                                "outcome": "acknowledged-held",
                                "subject": case.subject})
                continue
            results.append(self.run_case(case.id, now=now))
        report: dict[str, Any] = {"ran": len(results) - held,
                                  "results": results,
                                  "acknowledged_held": held}
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
            "counts": steward.cases.counts(),
            "adapters": adapter_health()}


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


def case_board(cases: OutcomeCaseStore, *,
               limit: int = 50) -> list[OutcomeCase]:
    """The ONE ordering every owner surface numbers cases by (P6).

    Needs-attention first: open/investigating cases that are not
    acknowledged-quiet, oldest first (the order the sweep works
    them).  Then closed cases the owner has not acknowledged,
    newest first.  Acknowledged-and-unchanged cases ride at the
    tail — they keep their board numbers so show/verify/ack resolve
    against exactly the board the owner saw, but they sit quiet
    until :func:`ack_changed` says they moved.
    """
    pool: list[OutcomeCase] = []
    for state in ("open", "investigating", "resolved", "closed-unknown"):
        pool.extend(cases.list_cases(state=state, limit=200))

    def quiet(case: OutcomeCase) -> bool:
        return ack_snapshot(case) is not None and not ack_changed(case)

    attention = sorted(
        (c for c in pool if c.state in ("open", "investigating")
         and not quiet(c)),
        key=lambda c: c.opened_at)
    closed = sorted(
        (c for c in pool if c.state in ("resolved", "closed-unknown")
         and not quiet(c)),
        key=lambda c: -(c.closed_at or c.opened_at))
    tail = sorted((c for c in pool if quiet(c)),
                  key=lambda c: c.opened_at)
    return (attention + closed + tail)[: max(1, int(limit))]


def _case_line(case: OutcomeCase) -> str:
    verdict = case.verdict if case.state == "resolved" \
        else "outcome unknown"
    line = (f"{case.subject} ({case.domain}) — opened for "
            f"{case.reason} · {case.state} · {verdict}")
    ack = ack_snapshot(case)
    if ack is not None:
        if ack_changed(case):
            line += " · ⚠ changed since you acknowledged it"
        else:
            line += " · acknowledged"
    return line


def render_case_board(cases: OutcomeCaseStore, *,
                      limit: int = 50) -> str:
    """The /cases board: needs-attention first, acknowledged cases
    quiet at the tail. Ordinals are :func:`case_board`'s — the same
    numbers show/verify/ack take."""
    board = case_board(cases, limit=limit)
    counts = cases.counts()
    header = (f"outcome cases — {counts.get('open', 0)} open · "
              f"{counts.get('investigating', 0)} investigating · "
              f"{counts.get('resolved', 0)} resolved · "
              f"{counts.get('closed-unknown', 0)} closed unknown")
    if not board:
        return (header + "\nno cases — nothing forgotten is waiting "
                "on an answer.")
    lines = [header]
    quiet_from = None
    for idx, case in enumerate(board, 1):
        if (ack_snapshot(case) is not None and not ack_changed(case)
                and quiet_from is None):
            quiet_from = idx
            lines.append("— acknowledged (quiet unless they change) —")
        lines.append(f"{idx}. {_case_line(case)}")
    lines.append("/cases <n> shows the evidence · /cases verify <n> "
                 "re-checks now · /cases ack <n> marks seen")
    return "\n".join(lines)


#: Detail keys that are identity handles, never owner text.
_DETAIL_HIDDEN_KEYS = frozenset({
    "mission_id", "job_id", "contract_id", "subject_key",
    "acknowledged", "ack_count",
})


def _detail_facts(case: OutcomeCase) -> list[str]:
    facts = []
    for key in sorted(case.detail or {}):
        if key in _DETAIL_HIDDEN_KEYS or key.endswith("_id"):
            continue
        value = case.detail[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        text = str(value)
        if len(text) > 120:
            text = text[:117] + "..."
        facts.append(f"{key.replace('_', ' ')}: {text}")
    return facts


def render_case_detail(case: OutcomeCase,
                       receipts: list[Any],
                       contract: dict[str, Any] | None = None) -> str:
    """One case in full: what it is, what the records say, and the
    evidence trail — who checked, from which source, and what each
    piece of evidence established. No internal ids anywhere."""
    lines = [f"Outcome case — {case.subject} ({case.domain})"]
    opened = time.strftime("%H:%M %d %b",
                           time.localtime(case.opened_at))
    verdict_line = case.verdict if case.state == "resolved" \
        else "outcome unknown"
    lines.append(f"opened {opened} for: {case.reason} · "
                 f"state: {case.state} · verdict: {verdict_line}")
    for fact in _detail_facts(case):
        lines.append(f"· {fact}")
    ack = ack_snapshot(case)
    if ack is not None:
        when = time.strftime("%H:%M %d %b",
                             time.localtime(float(ack.get("at") or 0)))
        state_note = "changed since" if ack_changed(case) else "as seen"
        lines.append(f"· acknowledged {when} ({state_note}) — "
                     "acknowledgement records that you saw it; it "
                     "never resolves the case")
    if contract:
        lines.append(f"· investigation: {contract.get('status', '')} "
                     f"— worker {contract.get('worker', '')}, "
                     f"verifier {contract.get('verifier', '')}")
    if receipts:
        lines.append("evidence trail:")
        for receipt in receipts:
            when = time.strftime(
                "%H:%M", time.localtime(receipt.created_at))
            note = receipt.note
            if len(note) > 300:
                note = note[:297] + "..."
            establishes = (receipt.evidence or {}).get("establishes")
            suffix = f" [establishes: {establishes}]" if establishes \
                else ""
            lines.append(f"• {when} {receipt.kind} — "
                         f"{receipt.source} ({receipt.actor}): "
                         f"{note}{suffix}")
    else:
        lines.append("evidence trail: nothing gathered yet")
    if case.state in ("open", "investigating"):
        lines.append("ask for a fresh check any time: "
                     "/cases verify re-runs the investigation")
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
    reports case counts without running anything; action=inspect /
    show / verify / ack are the P6 inspection surface at parity with
    the chat /cases verbs and the nm outcomes CLI (internal ids are
    fine here — this is the machine consumer)."""
    context = registry.context

    @registry.register(
        "outcome_steward",
        description=(
            "Outcome stewardship for long-running or forgotten work: "
            "sweeps missions, background jobs, delegation contracts, "
            "scheduled jobs, queued chat sends, browser runs, media "
            "jobs, artifact transfers and artifact deliveries for "
            "operations "
            "whose outcome is overdue, stalled, orphaned or unknown, "
            "opens a reconciliation case for each, and investigates "
            "under an accountable contract — a case resolves only on "
            "cited evidence from the domain's own records; anything "
            "else closes as honestly unknown with the checks "
            "recorded. action=sweep runs one full cycle (this is "
            "what the durable 'outcome steward sweep' job fires "
            "every 6h); action=status reports case counts and "
            "per-domain adapter health (domains without a durable "
            "record are named unavailable, never silently absent); "
            "action=inspect lists cases on the shared board order; "
            "action=show returns one case with its evidence trail; "
            "action=verify runs that case's investigation now, "
            "through the same delegation substrate; action=ack "
            "records the owner's acknowledgement of a case — it "
            "quiets the sweep for that case until it changes, and it "
            "never resolves anything. Call directly when the owner "
            "asks what became of work that was left running."
        ),
        capability="memory.write",
        parameters={
            "action": "str — sweep | status | inspect | show | "
                      "verify | ack",
            "case_id": "str — the case id for show / verify / ack",
            "note": "str — optional note recorded with an ack",
        },
    )
    def _outcome_steward(action: str = "sweep", case_id: str = "",
                         note: str = "") -> dict[str, Any]:
        action = (action or "sweep").strip().lower()
        db = getattr(context, "db", None)
        if db is None:
            return {"ok": False,
                    "error": "no database on the context — the "
                             "steward has no records to keep"}
        if action == "status":
            return {"ok": True,
                    "counts": OutcomeCaseStore(db).counts(),
                    "adapters": adapter_health()}
        if action == "sweep":
            report = run_stewardship_cycle(db)
            return {"ok": True, **report}
        cases = OutcomeCaseStore(db)
        if action == "inspect":
            board = case_board(cases)
            return {"ok": True,
                    "counts": cases.counts(),
                    "adapters": adapter_health(),
                    "cases": [
                        inspect_case(cases, DelegationStore(db), c.id)
                        for c in board]}
        if action in ("show", "verify", "ack"):
            if not case_id:
                return {"ok": False,
                        "error": f"action {action} needs a case_id"}
            if cases.get(case_id) is None:
                return {"ok": False,
                        "error": f"no case {case_id!r}"}
            if action == "show":
                return {"ok": True,
                        **inspect_case(cases, DelegationStore(db),
                                       case_id)}
            if action == "verify":
                result = OutcomeSteward(db).run_case(case_id)
                return {"ok": True, **result,
                        **inspect_case(cases, DelegationStore(db),
                                       case_id)}
            case = cases.acknowledge(case_id, actor="owner",
                                     note=note)
            return {"ok": True, "acknowledged": case.id,
                    "state": case.state, "verdict": case.verdict}
        return {"ok": False, "error": f"unknown action {action!r}"}
