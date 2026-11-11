"""The serial integration queue (Lane 2, dossier §2.6).

The monitor's integration hand: it takes ``submitted``
candidates in submit order and promotes them, **one at a
time**, onto the promoted tree. Serial is the whole point —
two promotions never race, so the promoted tree's history is a
single line of journaled landings, and every later staging
bases on it.

A promotion:

1. **Reconciles tree and journal first.** The store's
   ``promoted_revision`` is the truth; the repo's HEAD must
   equal it. A HEAD ahead of the journal can only be this
   task's own earlier promotion commit whose journaling was
   interrupted (crash between the git commit and the store
   transaction) — recognised by its parent and its bench
   trailers, it is *adopted*: journaled, not re-applied. Any
   other divergence is discarded back to the journaled
   revision. The tree and the journal agree before anything
   new is attempted.
2. **Applies** the candidate onto the current promoted HEAD
   (cherry-pick semantics — a candidate captured against an
   older base is rebased onto whatever HEAD has become), runs
   the optional acceptance smoke on the staged result, and
   commits as ONE commit carrying the candidate's bench
   trailers.
3. **Journals atomically**: task ``verifying -> promoted``,
   ``note_promoted_revision``, lease release, and the
   ``task_promoted`` event land in one store transaction.

Exactly-once rides ``missions/idempotency`` (the dossier's
assignment): the promotion runs under the key
``bench-promotion(run, task, candidate_ref)``. A completed key
never re-executes; a crashed attempt leaves the key failed or
stale-running, the retry steals it, and step 1's reconciliation
makes the re-execution land the promotion exactly once.

**Conflict is a named event, never a merge strategy.** A
candidate that does not apply cleanly (or whose smoke fails)
is restored out of the tree untouched; the task goes
``conflicted`` with the conflicting files and the apply output
in the journal, a ``monitor_defect`` event records that the
write-set admission missed an overlap, the task's leases are
released, and the queue CONTINUES with the next candidate.
:meth:`BenchIntegrator.return_for_rebase` hands a conflicted
task back (``requeued``) so its cursor can re-claim it at the
new promoted base and redo the change against current code.
There is no --ours/--theirs and no model merge anywhere here.

Verification gating is Lane 3's, and it sits at exactly one
point: :meth:`BenchIntegrator.pending` returns only tasks in
``verifying`` whose LATEST verdict for the CURRENT candidate
ref is ``supported`` — the verifier's word, written to
``bench_verdicts`` by the verifier organ (verifier.py), is the
gate's only input. A merely submitted candidate, a returned
one, a resubmission whose new ref carries no verdict yet: all
absent by construction, and :meth:`BenchIntegrator._promote_once`
re-checks the same gate at the moment of promotion, so a
verdict written between listing and promoting still rules.
The queue mechanics (serial order, exactly-once, conflict
honesty) are unchanged by the gate; promotion and conflict
journaling now transition from ``verifying``, the state the
verifier leaves a supported task in.

Layer: agents (L5) — composes the bench model/monitor/staging
and the missions peer's idempotency store.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ...missions.idempotency import IdempotencyStore, dedupe, idempotency_key
from .model import (
    MONITOR_REF,
    TASK_CONFLICTED,
    TASK_PROMOTED,
    TASK_REQUEUED,
    TASK_SUBMITTED,
    TASK_VERIFYING,
    BenchError,
)

__all__ = ["BenchIntegrator", "promotion_key"]

#: A promotion attempt whose idempotency claim goes stale after
#: this long may be stolen by a retry (the crashed owner's
#: window — the command-claims logic at promotion grain).
DEFAULT_PROMOTION_STALE_AFTER_S = 300.0


def promotion_key(run_id: str, task: dict[str, Any]) -> str:
    """The idempotency key one promotion runs under: one key
    per (run, task, candidate) — a re-submit is a NEW candidate
    and earns a new key; a retry of the same promotion is the
    same key and collapses to exactly once."""
    return idempotency_key(
        "bench-promotion",
        {"run": str(run_id), "task": str(task["id"]),
         "candidate": str(task.get("candidate_ref") or "")})


class BenchIntegrator:
    """Promotes submitted candidates, serially, for one
    monitor. Requires the monitor's staging manager (the
    integration repo lives there)."""

    def __init__(self, monitor: Any, *,
                 smoke: Callable[..., tuple[bool, str]] | None = None,
                 stale_after_s: float
                 = DEFAULT_PROMOTION_STALE_AFTER_S) -> None:
        if getattr(monitor, "staging", None) is None:
            raise BenchError(
                "the integration queue needs the monitor's "
                "staging manager (it owns the integration repo)")
        self.monitor = monitor
        self.store = monitor.store
        self.staging = monitor.staging
        self.idempotency = IdempotencyStore(monitor.db)
        #: Optional promotion smoke: ``smoke(repo_path, task) ->
        #: (ok, output)`` run on the staged candidate before it
        #: is committed (dossier §2.6 step 2 — the fast subset;
        #: full verification is Lane 3). None = no smoke.
        self.smoke = smoke
        self.stale_after_s = float(stale_after_s)

    # -- queue shape ---------------------------------------------------

    def pending(self, run_id: str) -> list[dict[str, Any]]:
        """The queue's whole input (Lane 3's verdict gate):
        tasks in ``verifying`` whose latest verdict for the
        CURRENT candidate ref is ``supported``, in verdict
        order. Nothing else promotes — not a submitted-but-
        unverified candidate, not a returned one, not a
        resubmission whose new ref carries no verdict yet."""
        out: list[tuple[float, dict[str, Any]]] = []
        for task in self.store.list_tasks(run_id):
            if task["state"] != TASK_VERIFYING:
                continue
            verdict = self.store.latest_verdict(
                task["id"], str(task.get("candidate_ref") or ""))
            if verdict is None or verdict["verdict"] != "supported":
                continue
            out.append((float(verdict["created_at"]), task))
        out.sort(key=lambda pair: pair[0])
        return [task for _, task in out]

    def blocked(self, run_id: str) -> list[dict[str, Any]]:
        """Submitted/verifying tasks the gate is holding, each
        with its reason — observability for the run surface
        (Lane 4 renders it), never a second promotion path."""
        out: list[dict[str, Any]] = []
        for task in self.store.list_tasks(run_id):
            if task["state"] not in (TASK_SUBMITTED, TASK_VERIFYING):
                continue
            verdict = self.store.latest_verdict(
                task["id"], str(task.get("candidate_ref") or ""))
            if task["state"] == TASK_VERIFYING and verdict is not None \
                    and verdict["verdict"] == "supported":
                continue
            if verdict is None:
                reason = ("awaiting_verdict"
                          if task["state"] == TASK_SUBMITTED
                          else "verdict_missing")
            else:
                reason = "verdict_" + str(verdict["verdict"])
            out.append({
                "task_id": task["id"], "key": task["key"],
                "state": task["state"],
                "candidate_ref": str(task.get("candidate_ref") or ""),
                "reason": reason,
                "verdict": (verdict or {}).get("verdict", "")})
        return out

    def promote_all(self, run_id: str, *,
                    now: float | None = None) -> list[dict[str, Any]]:
        """Drain the queue in submit order. A conflict is one
        candidate's outcome, never the queue's: draining
        continues past it."""
        outcomes: list[dict[str, Any]] = []
        while True:
            outcome = self.promote_next(run_id, now=now)
            if outcome is None:
                return outcomes
            outcomes.append(outcome)

    def promote_next(self, run_id: str, *,
                     now: float | None = None) -> dict[str, Any] | None:
        """Promote (or conflict) the oldest gate-cleared
        candidate. Returns its outcome record, or None when no
        task passes the verdict gate. The promotion itself runs
        under its idempotency
        key — a retry after a crash returns the recorded
        outcome or re-executes into reconciliation, landing
        exactly once either way."""
        now = time.time() if now is None else float(now)
        queue = self.pending(run_id)
        if not queue:
            return None
        task = queue[0]
        key = promotion_key(run_id, task)
        result = dedupe(
            self.idempotency, key,
            lambda: self._promote_once(task["id"], now),
            owner=MONITOR_REF, stale_after=self.stale_after_s,
            wait_timeout=max(30.0, 4.0 * self.stale_after_s),
            fingerprint=str(task.get("candidate_ref") or ""),
            label="bench-promotion")
        outcome = dict(result.value or {})
        outcome["executed"] = bool(result.executed)
        return outcome

    # -- the promotion (runs under the idempotency key) -----------------

    def _promote_once(self, task_id: str,
                      now: float) -> dict[str, Any]:
        task = self.store.get_task_by_id(task_id)
        if task is None:
            raise BenchError("promotion of unknown task: "
                             + str(task_id))
        run = self.store.get_run(task["run_id"])
        if run is None:
            raise BenchError("promotion on unknown run: "
                             + str(task["run_id"]))
        if task["state"] == TASK_PROMOTED:
            # The journal already landed this promotion (a crash
            # between journaling and the idempotency record):
            # report the recorded outcome, apply nothing.
            return {"promoted": True, "task": task["key"],
                    "task_id": task_id,
                    "revision": str(run["promoted_revision"]),
                    "already_journaled": True}
        if task["state"] != TASK_VERIFYING:
            return {"promoted": False, "task": task["key"],
                    "task_id": task_id,
                    "reason": "not_verifying",
                    "state": task["state"]}
        # The gate, re-checked at the moment of promotion: the
        # latest verdict for the CURRENT candidate ref must be
        # supported — a listing from a moment ago does not rule.
        verdict = self.store.latest_verdict(
            task_id, str(task.get("candidate_ref") or ""))
        if verdict is None or verdict["verdict"] != "supported":
            return {"promoted": False, "task": task["key"],
                    "task_id": task_id,
                    "reason": "verdict_gate",
                    "verdict": (verdict or {}).get("verdict", ""),
                    "state": task["state"]}
        # Step 1 — reconcile tree and journal.
        head = self.staging.head()
        if head != str(run["promoted_revision"]):
            if self._adoptable_head(task, run, head):
                return self._journal_promotion(task, run, head,
                                               now, adopted=True)
            self.staging.reset_tree(str(run["promoted_revision"]))
            head = str(run["promoted_revision"])
        # Step 2 — apply (+ smoke), then commit.
        candidate_ref = str(task.get("candidate_ref") or "")
        if not candidate_ref:
            # An empty candidate lands as itself: nothing to
            # apply, the promoted revision does not move.
            return self._journal_promotion(task, run, head, now,
                                           adopted=False)
        applied = self.staging.apply_candidate(candidate_ref)
        if applied.get("conflicted"):
            return self._journal_conflict(task, run, applied, now)
        if applied.get("already_present"):
            return self._journal_promotion(task, run, head, now,
                                           adopted=False)
        if self.smoke is not None:
            ok, output = self.smoke(self.staging.repo, dict(task))
            if not ok:
                self.staging.reset_tree(head)
                return self._journal_conflict(
                    task, run,
                    {"files": list(applied.get("files") or []),
                     "output": str(output)[-2000:],
                     "reason": "smoke_failed"},
                    now)
        message = self.staging.commit_message(candidate_ref)
        revision = self.staging.commit_staged(message)
        self._promotion_checkpoint("after_git_commit", task,
                                   revision)
        # Step 3 — journal atomically.
        return self._journal_promotion(task, run, revision, now,
                                       adopted=False)

    def _promotion_checkpoint(self, phase: str,
                              task: dict[str, Any],
                              revision: str = "") -> None:
        """Observability seam inside a promotion (phases:
        ``after_git_commit``). The base implementation does
        nothing; embedders and tests use it to watch — or, in
        tests, to interrupt — a promotion at its exact seams."""

    def _adoptable_head(self, task: dict[str, Any],
                        run: dict[str, Any], head: str) -> bool:
        """True when ``head`` is THIS task's own interrupted
        promotion commit: its parent is the journaled promoted
        revision and its bench trailers name this run + task.
        Nothing else is ever adopted."""
        if self.staging.commit_parent(head) \
                != str(run["promoted_revision"]):
            return False
        message = self.staging.commit_message(head)
        return ("Bench-Run: " + str(run["id"]) in message
                and "Bench-Task-Id: " + str(task["id"]) in message)

    def _journal_promotion(self, task: dict[str, Any],
                           run: dict[str, Any], revision: str,
                           now: float, *,
                           adopted: bool) -> dict[str, Any]:
        with self.monitor.db.transaction():
            moved = self.store.transition(
                task["id"], [TASK_VERIFYING], TASK_PROMOTED,
                now=now, holder="", pid=0, claim_generation=0,
                leases=[])
            if not moved:
                raise BenchError(
                    "promotion lost the state race for task "
                    + str(task["key"]))
            self.store.note_promoted_revision(run["id"], revision)
            for record in task["leases"]:
                self.monitor.leases.release(
                    record["lease_id"], int(record["generation"]),
                    now=now)
            self.store.append_event(
                run["id"], task["id"], "task_promoted", MONITOR_REF,
                generation=int(task["claim_generation"]),
                detail={"revision": revision,
                        "candidate_ref":
                            str(task.get("candidate_ref") or ""),
                        "adopted": bool(adopted)},
                now=now)
        self.staging.discard(run["id"], task["id"])
        return {"promoted": True, "task": task["key"],
                "task_id": task["id"], "revision": revision,
                "adopted": bool(adopted)}

    def _journal_conflict(self, task: dict[str, Any],
                          run: dict[str, Any],
                          applied: dict[str, Any],
                          now: float) -> dict[str, Any]:
        files = [str(f) for f in applied.get("files") or []]
        write_set = {str(e) for e in task["write_set"]}
        evidence = {
            "files": files,
            "output": str(applied.get("output") or ""),
            "reason": str(applied.get("reason") or "apply_conflict"),
            "candidate_ref": str(task.get("candidate_ref") or ""),
            "base_revision": str(task.get("base_revision") or ""),
            "promoted_revision": str(run["promoted_revision"]),
            "in_write_set": {f: f in write_set for f in files},
        }
        with self.monitor.db.transaction():
            moved = self.store.transition(
                task["id"], [TASK_VERIFYING], TASK_CONFLICTED,
                now=now, holder="", pid=0, claim_generation=0,
                leases=[])
            if not moved:
                raise BenchError(
                    "conflict lost the state race for task "
                    + str(task["key"]))
            for record in task["leases"]:
                self.monitor.leases.release(
                    record["lease_id"], int(record["generation"]),
                    now=now)
            self.store.append_event(
                run["id"], task["id"], "task_conflicted",
                MONITOR_REF,
                generation=int(task["claim_generation"]),
                reason=evidence["reason"], detail=evidence,
                now=now)
            # The admission check was supposed to make this
            # impossible for declared write-sets; a conflict is
            # therefore also a defect in the monitor's own
            # partitioning, recorded as data (dossier §2.6).
            self.store.append_event(
                run["id"], task["id"], "monitor_defect",
                MONITOR_REF,
                generation=int(task["claim_generation"]),
                reason="promotion_conflict_after_admission",
                detail={"files": files,
                        "in_write_set": evidence["in_write_set"]},
                now=now)
        self.staging.quarantine(run["id"], task["id"])
        return {"promoted": False, "conflicted": True,
                "task": task["key"], "task_id": task["id"],
                "files": files, "reason": evidence["reason"]}

    # -- the return loop --------------------------------------------------

    def return_for_rebase(self, run_id: str, key: str, *,
                          now: float | None = None) -> dict[str, Any]:
        """Hand a conflicted task back for rebase: it becomes
        claimable again (``requeued``), its stale candidate is
        dropped, and a fresh claim pins the CURRENT promoted
        revision as its base — the cursor redoes its change
        against current promoted code, never against the base
        that conflicted."""
        now = time.time() if now is None else float(now)
        task = self.store.get_task(run_id, key)
        if task is None:
            return {"returned": False, "reason": "unknown_task"}
        moved = self.store.transition(
            task["id"], [TASK_CONFLICTED], TASK_REQUEUED,
            now=now, candidate_ref="")
        if not moved:
            return {"returned": False, "reason": "not_conflicted",
                    "state": task["state"], "task_id": task["id"]}
        self.store.append_event(
            run_id, task["id"], "task_requeued", MONITOR_REF,
            generation=int(task["claim_generation"]),
            reason="conflict_rebase",
            detail={"promoted_revision":
                    self.store.get_run(run_id)["promoted_revision"]
                    if self.store.get_run(run_id) else ""},
            now=now)
        return {"returned": True, "task_id": task["id"],
                "state": TASK_REQUEUED}
