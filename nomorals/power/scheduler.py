"""Power-aware scheduler: heavy work waits for good power.

Wraps :class:`nomorals.storage.queue.WorkQueue` with power-class topics:
``power:light`` always flows; ``power:medium`` flows while degradation is
mild (level <= 1); ``power:heavy`` only flows at degradation level 0 with
a clean advisory.  Deferred tasks wait durably — nothing is dropped.

On top of power classes, every task carries a *tier*
(``critical`` | ``important`` | ``background`` | ``bulk``) and an optional
*subsystem* (``llm``, ``media``, ``scheduler``, ``missions``, ``bulk``).
The degradation ladder sheds tiers bottom-up, and — when a
resource-manager-like ``resources`` object is injected — subsystem
budgets gate admission: a task whose subsystem budget is exhausted is
requeued with backoff instead of being started.

New in this sweep:

- **Jittered deferral backoff** (:class:`BackoffPolicy`, full jitter by
  default — the AWS "Exponential Backoff and Jitter" family): deferral
  delays grow exponentially with each deferral, capped, instead of the
  old fixed 30s/60s.
- **WorkManager-style constraints** per task: ``requires_charging``,
  ``requires_battery_not_low``, ``requires_idle`` — declared needs, the
  system picks the time.
- **Aging / anti-starvation**: every deferral is counted; past
  ``max_deferrals`` the task *escalates* (event + admission), so low
  tiers can't wait forever under sustained pressure.
- **Non-preemptible tasks** (``preemptible=False``): refuse tier-shed /
  budget deferral and run when their class flows — the
  ``preemptionPolicy: Never`` analog.
- **Energy ledger** (:class:`EnergyLedger`, the userspace Energy Model):
  learned duration per task type → Wh estimates; batches are ordered
  cheapest-first within a tier.

- **Generation-fenced claims** (doc 18 P3): every task :meth:`poll`
  hands out carries a lease from :class:`nomorals.core.leases.LeaseStore`
  on the resource ``power-task:<job_id>``, held by this instance
  (``power-scheduler:<worker>:<instance>``) at a monotonic generation.
  The terminal acts (:meth:`complete` / :meth:`fail` / :meth:`cancel`)
  revalidate the generation at the act seam — a holder whose lease
  lapsed or was superseded is refused there: the act lands in the
  record as ``power.task.fenced``, never as the work's own outcome,
  and the queue row is left for its rightful holder.
  :meth:`renew_claim` is the heartbeat that keeps a live claim's
  lease (and its queue lease) from lapsing mid-execution.

This module never imports ``nomorals.os`` (layer rule): the optional
``resources`` object is duck-typed (needs ``.budgets`` with
``acquire``/``release``/``admission``/``snapshot``), and L7 entry points
wire in the real ``ResourceManager``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.leases import LeaseStore
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.queue import Job, WorkQueue
from .backoff import BackoffPolicy
from .energy import EnergyLedger
from .errors import InvalidPowerSpecError, PowerError
from .monitor import PowerMonitor
from .telemetry import emit_event

__all__ = [
    "PowerTask",
    "PowerAwareScheduler",
    "LIGHT_TOPIC",
    "MEDIUM_TOPIC",
    "HEAVY_TOPIC",
    "POWER_CLASSES",
    "TASK_TIERS",
    "CONSTRAINT_KEYS",
    "LEASE_RESOURCE_PREFIX",
]

_log = get_logger(__name__)

LIGHT_TOPIC = "power:light"
MEDIUM_TOPIC = "power:medium"
HEAVY_TOPIC = "power:heavy"
POWER_CLASSES = ("light", "medium", "heavy")
TASK_TIERS = ("critical", "important", "background", "bulk")
TIER_RANK = {"critical": 0, "important": 1, "background": 2, "bulk": 3}
# WorkManager-style declared constraints a task may require.
CONSTRAINT_KEYS = ("requires_charging", "requires_battery_not_low",
                   "requires_idle")
# Doc 18 lease resource for one claimed task: the open-vocabulary
# resource name is ``power-task:<job_id>`` — one claim per job, so a
# job's execution right is a named, generation-fenced resource.
LEASE_RESOURCE_PREFIX = "power-task"


@dataclass
class PowerTask:
    job_id: str
    task_type: str
    payload: dict[str, Any]
    power_class: str  # "light" | "medium" | "heavy"
    priority: int = 0
    attempts: int = 0
    subsystem: str | None = None   # budget owner: llm|media|scheduler|...
    tier: str = "background"       # critical|important|background|bulk
    mem_mb: float = 0.0            # expected memory need (budget admission)
    cpu_pct: float = 0.0           # expected cpu need (budget admission)
    # -- sweep additions -------------------------------------------------
    constraints: dict[str, bool] = field(default_factory=dict)
    preemptible: bool = True       # False: tier/budget deferral can't shed it
    deferred: int = 0              # deferral count so far (anti-starvation)
    max_deferrals: int = 10        # past this the task escalates, not waits
    deadline: float | None = None  # epoch seconds; overdue tasks escalate
    enqueued_at: float = 0.0
    # -- claim fencing (doc 18 P3), set by poll() on admitted tasks ---
    lease_id: str = ""             # LeaseStore lease id for this claim
    lease_generation: int = 0      # fencing generation of the claim
    lease_resource: str = ""       # "power-task:<job_id>"
    lease_token: str = ""          # the WorkQueue F02 token of the claim

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_type": self.task_type,
            "payload": dict(self.payload),
            "power_class": self.power_class,
            "priority": self.priority,
            "attempts": self.attempts,
            "subsystem": self.subsystem,
            "tier": self.tier,
            "mem_mb": self.mem_mb,
            "cpu_pct": self.cpu_pct,
            "constraints": dict(self.constraints),
            "preemptible": self.preemptible,
            "deferred": self.deferred,
            "max_deferrals": self.max_deferrals,
            "deadline": self.deadline,
            "enqueued_at": self.enqueued_at,
        }

    @property
    def overdue(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline

    @property
    def escalated(self) -> bool:
        """True when the task has waited past its deferral budget."""
        return (self.max_deferrals > 0 and self.deferred >= self.max_deferrals) \
            or self.overdue


@dataclass
class _Claim:
    """One live claim this instance holds: the doc 18 lease fencing a
    polled task plus the WorkQueue F02 token of the same claim.  Kept
    in memory only — a dead instance's claims die with it, and the
    durable truth (lease rows + queue rows) is re-read on every act."""

    job_id: str
    resource: str
    lease_id: str
    generation: int
    holder_ref: str
    worker: str
    queue_token: str = ""
    ttl_s: float = 0.0
    fenced: bool = False           # True once an act was refused here


class PowerAwareScheduler:
    """Defer heavy work when power, degradation, or budgets say so."""

    def __init__(
        self,
        db: Database,
        monitor: PowerMonitor | None = None,
        queue: WorkQueue | None = None,
        resources: Any | None = None,
        *,
        deferral_backoff_s: float = 30.0,
        budget_backoff_s: float = 60.0,
        deferral_backoff_cap: float = 1800.0,
        budget_backoff_cap: float = 3600.0,
        backoff_strategy: str = "full",
        backoff_rng: Any | None = None,
    ) -> None:
        self.db = db
        # Default monitor has no sampler (fails open). L7 callers inject
        # a ResourceManager-backed sampler for real readings.
        if monitor is None and resources is not None:
            # Build a real monitor from the injected resource manager when
            # it quacks like one (duck-typed: .sample() + .consult()).
            if callable(getattr(resources, "sample", None)) and callable(
                    getattr(resources, "consult", None)):
                monitor = PowerMonitor(sampler=lambda: resources)
        self.monitor = monitor or PowerMonitor()
        self.queue = queue or WorkQueue(db)
        # Duck-typed resource manager for subsystem budget admission.
        # None = budgets unavailable -> fail-open (tasks flow).
        self.resources = resources
        # Jittered deferral backoff (replaces the old fixed delays).
        # ``deferral_backoff_s``/``budget_backoff_s`` are the base delays.
        self.deferral_backoff = BackoffPolicy(
            base=deferral_backoff_s, cap=deferral_backoff_cap,
            strategy=backoff_strategy, rng=backoff_rng)
        self.budget_backoff = BackoffPolicy(
            base=budget_backoff_s, cap=budget_backoff_cap,
            strategy=backoff_strategy, rng=backoff_rng)
        self.energy = EnergyLedger()
        # Claim fencing (doc 18 P3): this instance's identity is part
        # of every holder_ref — a worker NAME is not an identity (the
        # same name on a later process must not inherit the claim).
        # The lease store is built lazily on first claim so scheduler
        # construction stays side-effect-free.
        self._instance_id = uuid.uuid4().hex[:12]
        self._claims: dict[str, _Claim] = {}
        self._lease_store: LeaseStore | None = None

    # -- budget helpers --------------------------------------------------
    def _budgets(self) -> Any | None:
        try:
            b = getattr(self.resources, "budgets", None)
            return b
        except Exception:  # noqa: BLE001 - never raises
            return None

    def try_acquire(self, subsystem: str, *, mem_mb: float = 0.0,
                    cpu_pct: float = 0.0) -> bool:
        """Admit a subsystem load against its budget.  Fail-open when no
        resource manager is wired (returns True).  Never raises."""
        try:
            b = self._budgets()
            if b is None:
                return True
            return bool(b.acquire(subsystem, mem_mb=mem_mb, cpu_pct=cpu_pct))
        except Exception:  # noqa: BLE001 - never raises
            return True

    def release_budget(self, subsystem: str, *, mem_mb: float = 0.0,
                       cpu_pct: float = 0.0) -> None:
        """Release tracked budget usage.  Never raises."""
        try:
            b = self._budgets()
            if b is not None:
                b.release(subsystem, mem_mb=mem_mb, cpu_pct=cpu_pct)
        except Exception:  # noqa: BLE001 - never raises
            pass

    def budget_snapshot(self) -> dict[str, Any]:
        try:
            b = self._budgets()
            return b.snapshot() if b is not None else {}
        except Exception:  # noqa: BLE001 - never raises
            return {}

    # -- dispatch ---------------------------------------------------------
    def dispatch(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        *,
        power_class: str = "light",
        priority: int = 0,
        delay: float = 0.0,
        max_attempts: int = 5,
        subsystem: str | None = None,
        tier: str = "background",
        mem_mb: float = 0.0,
        cpu_pct: float = 0.0,
        constraints: dict[str, bool] | None = None,
        preemptible: bool = True,
        max_deferrals: int = 10,
        deadline: float | None = None,
    ) -> str:
        """Queue a task.

        ``power_class`` is ``"light"`` | ``"medium"`` | ``"heavy"``;
        ``tier`` is ``"critical"`` | ``"important"`` | ``"background"`` |
        ``"bulk"``.  ``subsystem``/``mem_mb``/``cpu_pct`` feed budget
        admission at poll time (dispatch itself never blocks).

        ``constraints`` declares WorkManager-style needs, e.g.
        ``{"requires_charging": True}`` — the task only runs when the
        live power status satisfies them.  ``preemptible=False`` makes the
        task refuse tier-shed/budget deferral (it still respects the
        power-class plan).  ``max_deferrals`` caps how long a task can be
        deferred before it escalates instead of waiting.
        """
        if power_class not in POWER_CLASSES:
            raise InvalidPowerSpecError(
                f"power_class must be one of {POWER_CLASSES}, "
                f"got {power_class!r}",
                details={"power_class": power_class})
        if not task_type:
            raise InvalidPowerSpecError("task_type is required")
        tier = str(tier).lower()
        if tier not in TASK_TIERS:
            raise InvalidPowerSpecError(
                f"tier must be one of {TASK_TIERS}, got {tier!r}",
                details={"tier": tier})
        clean_constraints = self._clean_constraints(constraints)
        topic = {"light": LIGHT_TOPIC, "medium": MEDIUM_TOPIC,
                 "heavy": HEAVY_TOPIC}[power_class]
        job_id = self.queue.enqueue(
            topic,
            {
                "task_type": task_type,
                "payload": payload or {},
                "power_class": power_class,
                "subsystem": subsystem,
                "tier": tier,
                "mem_mb": max(0.0, float(mem_mb)),
                "cpu_pct": max(0.0, float(cpu_pct)),
                "constraints": clean_constraints,
                "preemptible": bool(preemptible),
                "deferred": 0,
                "max_deferrals": max(0, int(max_deferrals)),
                "deadline": deadline,
                "enqueued_at": time.time(),
            },
            priority=priority,
            delay=delay,
            max_attempts=max_attempts,
        )
        _log.info("power task %s queued: %s (%s/%s)", job_id, task_type,
                  power_class, tier)
        emit_event("power.task.dispatched", {
            "job_id": job_id,
            "task_type": task_type,
            "power_class": power_class,
            "subsystem": subsystem,
            "tier": tier,
            "constraints": clean_constraints,
        }, source=__name__)
        return job_id

    @staticmethod
    def _clean_constraints(
            constraints: dict[str, bool] | None) -> dict[str, bool]:
        if not constraints:
            return {}
        clean: dict[str, bool] = {}
        for key, want in constraints.items():
            if key not in CONSTRAINT_KEYS:
                raise InvalidPowerSpecError(
                    f"unknown constraint {key!r}; expected one of "
                    f"{CONSTRAINT_KEYS}", details={"constraint": key})
            clean[key] = bool(want)
        return clean

    # -- claim fencing (doc 18 P3) --------------------------------------
    def _leases(self) -> LeaseStore:
        """The lease authority over this scheduler's database, built
        lazily (its DDL is runtime-created, migrations untouched)."""
        if self._lease_store is None:
            self._lease_store = LeaseStore(self.db)
        return self._lease_store

    @staticmethod
    def _claim_resource(job_id: str) -> str:
        return f"{LEASE_RESOURCE_PREFIX}:{job_id}"

    def _holder_ref(self, worker: str) -> str:
        return f"power-scheduler:{worker}:{self._instance_id}"

    def _acquire_claim(self, job: Job, worker: str
                       ) -> tuple[_Claim | None, dict[str, Any]]:
        """Take the doc 18 lease fencing one queue-claimed job.

        Returns ``(claim, {})`` on a grant, ``(None, refusal)`` when
        another holder's live lease fences this job out (or the lease
        store itself failed — fencing fails closed: a task is never
        handed out unfenced).  The lease TTL tracks the queue claim's
        own remaining lease, so the two lapse together; renewal is
        :meth:`renew_claim`'s job."""
        resource = self._claim_resource(job.id)
        holder = self._holder_ref(worker)
        now = time.time()
        ttl = float(job.lease_until or 0.0) - now
        if ttl <= 0:
            ttl = float(self.queue.default_lease_seconds)
        try:
            result = self._leases().acquire(resource, holder, ttl, now=now)
        except Exception:  # noqa: BLE001 - fail closed, never unfenced
            _log.warning("lease acquire failed for %s; task not handed "
                         "out unfenced", job.id, exc_info=True)
            return None, {"reason": "lease_unavailable",
                          "resource": resource}
        if not result.get("granted"):
            return None, result
        lease = result["lease"]
        return _Claim(
            job_id=job.id,
            resource=resource,
            lease_id=str(lease["lease_id"]),
            generation=int(lease["generation"]),
            holder_ref=holder,
            worker=worker,
            queue_token=str(job.lease_token or ""),
            ttl_s=ttl,
        ), {}

    def _contend(self, job: Job, refusal: dict[str, Any]) -> None:
        """Claim-time fencing: the queue handed this instance a job
        whose execution right another holder still leases.  The task
        is NOT handed out; it returns to the pool (fenced by this
        claim's own queue token, timed to the holder's lease expiry)
        so the rightful holder — or a later claim once the fence
        frees — can take it.  Recorded as fencing, never as the
        work's failure, and never counted as a power deferral."""
        delay = 5.0
        expires_at = refusal.get("expires_at")
        if expires_at:
            delay = min(300.0, max(0.05, float(expires_at) - time.time()))
        try:
            self.queue.requeue(job.id, delay=delay,
                               lease_token=job.lease_token or None)
        except Exception:  # noqa: BLE001 - the queue lease lapses on its own
            _log.warning("contention requeue failed for %s", job.id,
                         exc_info=True)
        _log.warning(
            "power task %s fenced at claim: %s held by %s (generation "
            "%s); returned to the pool for its rightful holder",
            job.id, refusal.get("resource", self._claim_resource(job.id)),
            refusal.get("holder_ref", "?"), refusal.get("generation", "?"))
        emit_event("power.task.fenced", {
            "job_id": job.id,
            "phase": "claim",
            "reason": str(refusal.get("reason") or "held"),
            "holder_ref": refusal.get("holder_ref", ""),
            "generation": refusal.get("generation", 0),
        }, source=__name__)

    def _release_claim(self, claim: _Claim) -> None:
        """Release a claim's lease (terminal act landed, job deferred
        back to the pool, ...).  Best-effort: a refusal here means the
        lease already lapsed or moved on — nothing to free."""
        try:
            self._leases().release(claim.lease_id, claim.generation)
        except Exception:  # noqa: BLE001 - release is bookkeeping
            _log.debug("lease release failed for %s", claim.job_id,
                       exc_info=True)

    def _claim_still_held(self, claim: _Claim) -> bool:
        """The fencing question at the act seam: does this instance
        STILL hold the claim's generation, live and unexpired?  A
        lease-store failure answers False — an act that cannot prove
        its right does not land."""
        try:
            return self._leases().check(
                claim.resource, claim.holder_ref, claim.generation)
        except Exception:  # noqa: BLE001 - fail closed at the act seam
            _log.warning("lease check failed for %s; treating the "
                         "claim as fenced", claim.job_id, exc_info=True)
            return False

    def _fence_reason(self, claim: _Claim) -> str:
        """Why a claim stopped being held, in the lease organ's
        vocabulary: superseded (``stale_generation``), ``expired``,
        ``released`` or ``not_found``."""
        try:
            store = self._leases()
            current = store.current(claim.resource)
            if current is not None and (
                    current["holder_ref"] != claim.holder_ref
                    or current["generation"] != claim.generation):
                return "stale_generation"
            mine = store.get(claim.lease_id)
            if mine is not None:
                return str(mine["state"])
        except Exception:  # noqa: BLE001 - classification is best-effort
            pass
        return "stale_generation"

    def _fenced_act(self, job_id: str, claim: _Claim, phase: str) -> None:
        """Record a refused terminal act.  The queue row is NOT
        touched (it belongs to its rightful holder), the act is NOT
        recorded as the work's completion/failure, and this instance's
        own budget acquisition is freed exactly once.  The claim
        record stays as a tombstone: later acts on it fence again
        rather than falling through to the unfenced path."""
        first = not claim.fenced
        claim.fenced = True
        reason = self._fence_reason(claim)
        if first:
            self._release_for_job(job_id)
        _log.warning(
            "power task %s %s fenced (%s): claim generation %d is no "
            "longer held by this instance; the act did not land",
            job_id, phase, reason, claim.generation)
        emit_event("power.task.fenced", {
            "job_id": job_id,
            "phase": phase,
            "reason": reason,
            "generation": claim.generation,
            "resource": claim.resource,
        }, source=__name__)

    def _act_fenced(self, job_id: str, phase: str,
                    lease_token: str | None) -> bool:
        """The act-seam gate for complete/fail/cancel.  True = the act
        is fenced and must not land.  No claim record (a job this
        instance never polled) = the deliberate administrative path,
        unfenced exactly as before.  A presented ``lease_token`` that
        is not the live claim's token belongs to a superseded claim
        of this very instance and fences the act on its own."""
        claim = self._claims.get(job_id)
        if claim is None:
            return False
        if lease_token is not None and lease_token != claim.queue_token:
            self._fenced_act(job_id, claim, phase)
            return True
        if not self._claim_still_held(claim):
            self._fenced_act(job_id, claim, phase)
            return True
        return False

    def renew_claim(self, job_id: str, extra_seconds: float | None = None
                    ) -> bool:
        """Heartbeat a live claim: extend its doc 18 lease (and its
        queue lease, fenced by the claim's token) so a long execution
        is not fenced mid-flight.  Returns False — and records the
        claim fenced — when the lease organ refuses (the generation
        moved on, or the lease already lapsed: an expired lease is
        NOT revived, the organ's semantics) or the job is not claimed
        by this instance.  A refused renewal means stop: later acts
        on the claim fence at the act seam."""
        claim = self._claims.get(job_id)
        if claim is None or claim.fenced:
            return False
        ttl = float(extra_seconds) if extra_seconds is not None \
            else claim.ttl_s
        if ttl <= 0:
            return False
        try:
            result = self._leases().renew(
                claim.lease_id, claim.generation, ttl_s=ttl)
        except Exception:  # noqa: BLE001 - a failed heartbeat proves nothing
            _log.warning("lease renew failed for %s", job_id,
                         exc_info=True)
            return False
        if not result.get("renewed"):
            self._fenced_act(job_id, claim, "renew")
            return False
        try:
            self.queue.extend_lease(
                job_id, claim.worker, ttl,
                lease_token=claim.queue_token or None)
        except Exception:  # noqa: BLE001 - queue heartbeat is best-effort
            _log.debug("queue lease extension failed for %s", job_id,
                       exc_info=True)
        return True

    # -- poll --------------------------------------------------------------
    def poll(
        self,
        worker: str = "worker",
        *,
        batch: int = 5,
        lease_seconds: float | None = None,
        force_heavy: bool = False,
        force: bool = False,
    ) -> list[PowerTask]:
        """Claim tasks.  Heavy tasks flow only when power allows; medium
        tasks flow while degradation is mild; tasks whose tier is shed or
        whose subsystem budget is exhausted are requeued with jittered
        backoff.

        Set ``force_heavy=True`` to override heavy gating (e.g., plugged in
        and the operator explicitly wants the work now); ``force=True``
        overrides all power-class gating.  Escalated tasks (past
        ``max_deferrals`` or overdue) are admitted regardless of tier shed
        or budget pressure.

        Every admitted task carries a generation-fenced claim (doc 18
        P3): the lease on ``power-task:<job_id>`` held by this instance.
        A job whose claim another live holder fences is returned to
        the pool for its rightful holder instead of being handed out.
        """
        if not worker:
            raise ValueError("worker is required")
        status = self.monitor.status()
        plan = [LIGHT_TOPIC]
        if force or status.ok and status.degradation_level <= 1:
            plan.append(MEDIUM_TOPIC)
        elif status.should_defer_medium:
            _log.info("degradation level %d: deferring medium tasks",
                      status.degradation_level)
        if force or force_heavy or (status.degradation_level == 0
                                    and not status.should_defer_heavy):
            plan.append(HEAVY_TOPIC)
        else:
            _log.info("power constrained: deferring heavy tasks")

        tasks: list[PowerTask] = []
        for topic in plan:
            if len(tasks) >= batch:
                break
            jobs = self.queue.lease(
                topic, worker=worker, lease_seconds=lease_seconds,
                batch=batch - len(tasks),
            )
            for job in jobs:
                task = self._to_task(job)
                claim, refusal = self._acquire_claim(job, worker)
                if claim is None:
                    self._contend(job, refusal)
                    continue
                if self._admit(task, job, status):
                    self._claims[task.job_id] = claim
                    task.lease_id = claim.lease_id
                    task.lease_generation = claim.generation
                    task.lease_resource = claim.resource
                    task.lease_token = claim.queue_token
                    tasks.append(task)
                else:
                    # Deferred back to the pool: the claim ends with
                    # the hand-out that didn't happen, so the next
                    # claim of this job starts unfenced by this one.
                    self._release_claim(claim)
                if len(tasks) >= batch:
                    break
        # Cheapest work first within a tier (energy-model ordering), then
        # tier rank, then priority — keeps the batch power-efficient.
        tasks.sort(key=lambda t: (
            TIER_RANK.get(t.tier, 2),
            -t.priority,
            self.energy.estimated_wh(t.task_type, t.mem_mb) or 0.0,
        ))
        return tasks

    def _admit(self, task: PowerTask, job: Job, status: Any) -> bool:
        """Decide whether a leased task runs now.  Deferred tasks are
        requeued with jittered backoff and counted (anti-starvation)."""
        escalated = task.escalated
        if escalated:
            emit_event("power.task.escalated", {
                "job_id": task.job_id,
                "task_type": task.task_type,
                "deferred": task.deferred,
                "overdue": task.overdue,
            }, source=__name__)
            _log.warning("task %s escalated after %d deferrals",
                         task.job_id, task.deferred)
            return True
        if not status.tier_allowed(task.tier):
            if task.preemptible:
                self._defer(job, task, "tier_shed", self.deferral_backoff,
                            tier=task.tier,
                            degradation_level=status.degradation_level)
                return False
            emit_event("power.task.admitted_unpreemptible", {
                "job_id": task.job_id, "reason": "tier_shed",
                "tier": task.tier}, source=__name__)
            return True
        if task.subsystem and not self.try_acquire(
                task.subsystem, mem_mb=task.mem_mb, cpu_pct=task.cpu_pct):
            if task.preemptible:
                self._defer(job, task, "budget_exhausted", self.budget_backoff,
                            subsystem=task.subsystem)
                return False
            emit_event("power.task.admitted_unpreemptible", {
                "job_id": task.job_id, "reason": "budget_exhausted",
                "subsystem": task.subsystem}, source=__name__)
            return True
        unmet = self._unmet_constraints(task, status)
        if unmet:
            self._defer(job, task, "constraint_unmet", self.deferral_backoff,
                        unmet=unmet)
            return False
        return True

    @staticmethod
    def _unmet_constraints(task: PowerTask, status: Any) -> list[str]:
        """WorkManager-style gating against the live power status."""
        unmet: list[str] = []
        c = task.constraints or {}
        if c.get("requires_charging") and not status.charging:
            unmet.append("requires_charging")
        if c.get("requires_battery_not_low"):
            pct = getattr(status, "battery_pct", None)
            # Unknown battery = fail-open (desktop): can't prove it's low.
            if pct is not None and pct < 20.0:
                unmet.append("requires_battery_not_low")
        if c.get("requires_idle"):
            idle = getattr(status, "idle", None)
            if idle is False:
                unmet.append("requires_idle")
        return unmet

    def _defer(self, job: Job, task: PowerTask, reason: str,
               policy: BackoffPolicy, **detail: Any) -> None:
        """Requeue with jittered backoff; count the deferral durably."""
        deferred = task.deferred + 1
        delay = policy.next_delay(deferred, key=job.id)
        try:
            row = self.db.query_one(
                f"SELECT payload FROM {self.queue.TABLE} WHERE id = ?",
                (job.id,))
            payload: dict[str, Any] = {}
            if row and row.get("payload"):
                try:
                    payload = json.loads(row["payload"])
                except (TypeError, ValueError):
                    payload = {}
            payload["deferred"] = deferred
            payload["defer_reason"] = reason
            payload["next_retry_in_s"] = round(delay, 2)
            now = time.time()
            self.db.execute(
                f"UPDATE {self.queue.TABLE} SET status='ready', "
                f"payload=?, available_at=?, lease_owner='', "
                f"lease_until=0, updated_at=? WHERE id=?",
                (json.dumps(payload, default=str), now + delay, now, job.id),
            )
        except Exception:  # noqa: BLE001 - fall back to plain requeue
            _log.warning("deferral bookkeeping failed for %s", job.id,
                         exc_info=True)
            self.queue.requeue(job.id, delay=delay)
        _log.info("deferring task %s (%s; attempt %d, retry in %.1fs)",
                  job.id, reason, deferred, delay)
        emit_event("power.task.deferred", {
            "job_id": job.id,
            "reason": reason,
            "deferred": deferred,
            "retry_in_s": round(delay, 2),
            **{k: v for k, v in detail.items()},
        }, source=__name__)

    # -- completion ---------------------------------------------------------
    def _release_for_job(self, job_id: str) -> None:
        """Release budget usage tracked for a finished job, if any."""
        try:
            job = self.queue.get(job_id)
            if job is None:
                return
            p = job.payload or {}
            sub = p.get("subsystem")
            if sub:
                self.release_budget(
                    str(sub),
                    mem_mb=float(p.get("mem_mb") or 0.0),
                    cpu_pct=float(p.get("cpu_pct") or 0.0))
        except Exception:  # noqa: BLE001 - never raises
            pass

    def complete(self, job_id: str, result: Any = None, *,
                 lease_token: str | None = None) -> None:
        """Record a claimed task's successful execution.

        For a task this instance claimed, the act is fenced (doc 18
        P3): the claim's generation is revalidated here, at the act
        seam, and the queue write carries the claim's F02 token.  A
        fenced act lands in the record as ``power.task.fenced`` — the
        queue row keeps its rightful holder's state and this is never
        recorded as the work's completion.  ``lease_token`` pins the
        act to the exact claim the executor was handed (from
        :attr:`PowerTask.lease_token`); a token from a superseded
        claim fences the act.  Jobs this instance never claimed take
        the administrative path, exactly as before."""
        if self._act_fenced(job_id, "complete", lease_token):
            return
        claim = self._claims.get(job_id)
        if claim is not None:
            landed = self.queue.complete(
                job_id, result=result,
                lease_token=claim.queue_token or None)
            if not landed:
                if self._claim_still_held(claim):
                    # The queue generation moved under a still-held
                    # lease generation (a successor touched the row
                    # without taking the fence): the rightful holder's
                    # write is the administrative one.
                    self.queue.complete(job_id, result=result)
                else:
                    self._fenced_act(job_id, claim, "complete")
                    return
            self._record_energy(job_id)
            self._release_for_job(job_id)
            self._claims.pop(job_id, None)
            self._release_claim(claim)
        else:
            self._record_energy(job_id)
            self._release_for_job(job_id)
            self.queue.complete(job_id, result=result)
        emit_event("power.task.completed", {"job_id": job_id},
                   source=__name__)

    def fail(self, job_id: str, error: str = "", *, retry: bool = True,
             lease_token: str | None = None) -> None:
        """Record a claimed task's failed execution — fenced at the
        act seam exactly like :meth:`complete`: a stale holder's late
        failure must not requeue or dead-letter work its rightful
        holder is executing."""
        if self._act_fenced(job_id, "fail", lease_token):
            return
        claim = self._claims.get(job_id)
        if claim is not None:
            outcome = self.queue.fail(
                job_id, error=error, retry=retry,
                lease_token=claim.queue_token or None)
            if outcome == "stale":
                if self._claim_still_held(claim):
                    self.queue.fail(job_id, error=error, retry=retry)
                else:
                    self._fenced_act(job_id, claim, "fail")
                    return
            self._release_for_job(job_id)
            self._claims.pop(job_id, None)
            self._release_claim(claim)
        else:
            self._release_for_job(job_id)
            self.queue.fail(job_id, error=error, retry=retry)
        emit_event("power.task.failed", {"job_id": job_id, "error": error,
                                         "retry": retry}, source=__name__)

    def cancel(self, job_id: str, reason: str = "cancelled", *,
               lease_token: str | None = None) -> None:
        """Drop a deferred/waiting task (moves it to dead).

        Fenced like the other terminal acts: cancelling a claim this
        instance no longer holds would kill its rightful holder's
        work, so a fenced cancel is recorded and the row is left."""
        if self._act_fenced(job_id, "cancel", lease_token):
            return
        claim = self._claims.get(job_id)
        try:
            if claim is not None:
                outcome = self.queue.fail(
                    job_id, error=reason, retry=False,
                    lease_token=claim.queue_token or None)
                if outcome == "stale":
                    if self._claim_still_held(claim):
                        self.queue.fail(job_id, error=reason, retry=False)
                    else:
                        self._fenced_act(job_id, claim, "cancel")
                        return
                self._release_for_job(job_id)
                self._claims.pop(job_id, None)
                self._release_claim(claim)
            else:
                self._release_for_job(job_id)
                self.queue.fail(job_id, error=reason, retry=False)
        except Exception as exc:
            raise PowerError(f"cannot cancel {job_id}: {exc}",
                             details={"job_id": job_id}) from exc
        emit_event("power.task.cancelled",
                   {"job_id": job_id, "reason": reason}, source=__name__)

    def _record_energy(self, job_id: str) -> None:
        """Feed the energy ledger from a finished job's wall time."""
        try:
            job = self.queue.get(job_id)
            if job is None:
                return
            p = job.payload or {}
            enqueued = p.get("enqueued_at")
            if not enqueued:
                return
            duration = max(0.0, time.time() - float(enqueued))
            self.energy.record(
                str(p.get("task_type") or "unknown"), duration,
                mem_mb=float(p.get("mem_mb") or 0.0))
        except Exception:  # noqa: BLE001 - ledger is advisory only
            pass

    # -- inspection ----------------------------------------------------------
    def deferred_heavy_count(self) -> int:
        return self._deferred_count(HEAVY_TOPIC)

    def deferred_medium_count(self) -> int:
        return self._deferred_count(MEDIUM_TOPIC)

    def _deferred_count(self, topic: str) -> int:
        try:
            rows = self.db.query(
                f"SELECT COUNT(*) AS n FROM {self.queue.TABLE} "
                "WHERE topic=? AND status='ready'",
                (topic,),
            )
            return int(rows[0]["n"]) if rows else 0
        except Exception:  # noqa: BLE001 - never raises
            return 0

    def deferred_tasks(self, power_class: str | None = None) -> list[dict[str, Any]]:
        """Inspect waiting work: what is deferred, why, and for how long."""
        topics = {
            "light": LIGHT_TOPIC, "medium": MEDIUM_TOPIC, "heavy": HEAVY_TOPIC,
        }
        if power_class is not None:
            if power_class not in topics:
                raise InvalidPowerSpecError(
                    f"power_class must be one of {POWER_CLASSES}",
                    details={"power_class": power_class})
            wanted = [topics[power_class]]
        else:
            wanted = list(topics.values())
        out: list[dict[str, Any]] = []
        try:
            for topic in wanted:
                rows = self.db.query(
                    f"SELECT id, payload, priority, available_at "
                    f"FROM {self.queue.TABLE} "
                    f"WHERE topic=? AND status='ready'",
                    (topic,))
                for row in rows:
                    try:
                        p = json.loads(row["payload"]) if row["payload"] else {}
                    except (TypeError, ValueError):
                        p = {}
                    wait_s = max(0.0, float(row["available_at"] or 0)
                                 - time.time())
                    out.append({
                        "job_id": row["id"],
                        "task_type": p.get("task_type"),
                        "power_class": p.get("power_class"),
                        "tier": p.get("tier"),
                        "deferred": int(p.get("deferred") or 0),
                        "defer_reason": p.get("defer_reason"),
                        "retry_in_s": round(wait_s, 1),
                        "priority": row["priority"],
                    })
        except Exception:  # noqa: BLE001 - never raises
            return []
        return sorted(out, key=lambda d: (d["retry_in_s"], -d["priority"]))

    def next_window_hint(self) -> str:
        """Human hint: when will deferred classes flow again?"""
        status = self.monitor.status()
        if not status.should_defer_heavy and not status.should_defer_medium:
            return "all power classes flowing now"
        bits = []
        if status.charging:
            bits.append("charging — classes should reopen as the battery "
                        "recovers")
        elif status.battery_pct is not None and status.battery_pct < 20:
            bits.append(f"battery at {status.battery_pct:.0f}% — plug in or "
                        "wait for degradation to clear")
        if status.thermal_state == "hot":
            bits.append("thermal hot — heavy work waits for cooldown")
        bits.append(f"degradation level {status.degradation_level} "
                    f"({status.degradation_name})")
        if status.time_to_empty_min:
            bits.append(f"~{status.time_to_empty_min:.0f} min to empty at "
                        "current drain")
        return "heavy/medium deferred: " + "; ".join(bits)

    def stats(self) -> dict[str, Any]:
        """Pending/deferred counts per class plus budget snapshot."""
        try:
            pending: dict[str, int] = {}
            for cls, topic in (("light", LIGHT_TOPIC),
                               ("medium", MEDIUM_TOPIC),
                               ("heavy", HEAVY_TOPIC)):
                try:
                    pending[cls] = int(self.queue.pending(topic))
                except Exception:  # noqa: BLE001 - per-topic best-effort
                    pending[cls] = 0
            return {
                "pending": pending,
                "deferred_heavy": self.deferred_heavy_count(),
                "deferred_medium": self.deferred_medium_count(),
                "budgets": self.budget_snapshot(),
                "energy": self.energy.summary(),
                "backoff": {
                    "deferral": self.deferral_backoff.to_dict(),
                    "budget": self.budget_backoff.to_dict(),
                },
            }
        except Exception:  # noqa: BLE001 - never raises
            return {"pending": {}, "deferred_heavy": 0,
                    "deferred_medium": 0, "budgets": {}}

    def summary(self) -> str:
        """Rendered queue-state card for chat/CLI surfaces."""
        st = self.stats()
        pend = st.get("pending", {})
        lines = ["🔋 power scheduler"]
        lines.append(
            "   pending  " + " · ".join(
                f"{c}: {pend.get(c, 0)}" for c in ("light", "medium", "heavy")))
        lines.append(
            f"   deferred medium: {st.get('deferred_medium', 0)} · "
            f"heavy: {st.get('deferred_heavy', 0)}")
        waiting = self.deferred_tasks()
        if waiting:
            lines.append("   waiting:")
            for w in waiting[:8]:
                why = w["defer_reason"] or "power plan"
                lines.append(
                    f"     • {w['task_type']} ({w['power_class']}/"
                    f"{w['tier']}) — {why}, retry in {w['retry_in_s']:.0f}s "
                    f"[deferred ×{w['deferred']}]")
            if len(waiting) > 8:
                lines.append(f"     … +{len(waiting) - 8} more")
        energy = st.get("energy") or {}
        if energy:
            top = sorted(energy.items(),
                         key=lambda kv: kv[1]["estimated_wh_per_run"],
                         reverse=True)[:3]
            lines.append("   priciest tasks: " + ", ".join(
                f"{t} (~{v['estimated_wh_per_run']:.3f} Wh/run)"
                for t, v in top))
        lines.append("   " + self.next_window_hint())
        return "\n".join(lines)

    @staticmethod
    def _to_task(job: Job) -> PowerTask:
        p = job.payload or {}
        tier = str(p.get("tier") or "background").lower()
        if tier not in TASK_TIERS:
            tier = "background"
        pc = str(p.get("power_class") or "light")
        if pc not in POWER_CLASSES:
            pc = "light"
        sub = p.get("subsystem")
        try:
            mem_mb = max(0.0, float(p.get("mem_mb") or 0.0))
        except (TypeError, ValueError):
            mem_mb = 0.0
        try:
            cpu_pct = max(0.0, float(p.get("cpu_pct") or 0.0))
        except (TypeError, ValueError):
            cpu_pct = 0.0
        constraints = p.get("constraints") or {}
        if not isinstance(constraints, dict):
            constraints = {}
        constraints = {k: bool(v) for k, v in constraints.items()
                       if k in CONSTRAINT_KEYS}
        try:
            deferred = max(0, int(p.get("deferred") or 0))
        except (TypeError, ValueError):
            deferred = 0
        try:
            max_def = max(0, int(p.get("max_deferrals", 10)))
        except (TypeError, ValueError):
            max_def = 10
        deadline = p.get("deadline")
        try:
            deadline = float(deadline) if deadline is not None else None
        except (TypeError, ValueError):
            deadline = None
        try:
            enqueued_at = float(p.get("enqueued_at") or 0.0)
        except (TypeError, ValueError):
            enqueued_at = 0.0
        return PowerTask(
            job_id=job.id,
            task_type=str(p.get("task_type") or ""),
            payload=dict(p.get("payload") or {}),
            power_class=pc,
            priority=job.priority,
            attempts=job.attempts,
            subsystem=str(sub) if sub else None,
            tier=tier,
            mem_mb=mem_mb,
            cpu_pct=cpu_pct,
            constraints=constraints,
            preemptible=bool(p.get("preemptible", True)),
            deferred=deferred,
            max_deferrals=max_def,
            deadline=deadline,
            enqueued_at=enqueued_at,
        )
