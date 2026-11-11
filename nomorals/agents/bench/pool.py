"""The verifier pool (Lane 4 wiring for Lane 3's organ).

Promotion is serial, but verification need not be — and on a
live run someone has to actually hand submitted candidates to
verifier cursors. The pool is that hand: a set of
:class:`BenchVerifier` instances with distinct ids, an
assignment pass that gives the oldest submitted task to a
verifier that is **not its worker**, and a ``core/jobs.py``
JobRegistry loop that runs assignment + the submitted-state
timeout over every open run on a cadence (the watchdog's
idiom, one organ over).

Independence is structural, as everywhere on this bench: the
assignment skips the task's holder by construction, and if the
pool's only verifier IS the worker, the task is reported
``no_independent_verifier`` and left submitted — it is never
verified by its producer, and the delegation store would
refuse the verdict even if a caller tried (Lane 3). Each
verifier instance judges from its own scratch under the pool
root, so pool members never share a checkout.

The pool verifies; it does not promote. Draining the
integration queue behind the verdicts is the integrator's job
— :mod:`.wiring`'s bench cycle composes the two.

Layer: agents (L5) — composes the bench peers and core/jobs.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ...core.jobs import JobRegistry, shared_registry
from .model import TASK_SUBMITTED, BenchError
from .verifier import (
    DEFAULT_ATTEMPT_BUDGET,
    DEFAULT_VERIFY_TIMEOUT_S,
    BenchVerifier,
)

__all__ = ["VerifierPool", "DEFAULT_VERIFIER_IDS"]

#: The pool's standing verifier identities. Instances register
#: verdicts under their own ids; the delegation contract names
#: the POOL (``bench-verifier``) as verifier, and the store's
#: independence law is what binds each instance: never the
#: producer.
DEFAULT_VERIFIER_IDS = ("bench-verifier-1", "bench-verifier-2")


class VerifierPool:
    """Assigns submitted bench tasks to independent verifier
    cursors, for one monitor."""

    def __init__(self, monitor: Any, scratch_root: str | Path, *,
                 verifier_ids: tuple[str, ...] = DEFAULT_VERIFIER_IDS,
                 attempt_budget: int = DEFAULT_ATTEMPT_BUDGET,
                 verify_timeout_s: float = DEFAULT_VERIFY_TIMEOUT_S,
                 test_timeout_s: float = 300.0) -> None:
        if getattr(monitor, "staging", None) is None:
            raise BenchError(
                "the verifier pool needs the monitor's staging "
                "manager (verifiers materialise candidates from "
                "the integration repo)")
        if not verifier_ids:
            raise BenchError("a verifier pool needs at least "
                             "one verifier id")
        self.monitor = monitor
        self.store = monitor.store
        self.scratch_root = Path(scratch_root)
        self.verifier_ids = tuple(str(v) for v in verifier_ids)
        self.attempt_budget = int(attempt_budget)
        self.verify_timeout_s = float(verify_timeout_s)
        self.test_timeout_s = float(test_timeout_s)
        self._verifiers: dict[str, BenchVerifier] = {}
        self._job_stop = threading.Event()
        self._job: Any = None

    # -- instances -----------------------------------------------------

    def verifier_for(self, verifier_id: str) -> BenchVerifier:
        """The pool member with this id (constructed lazily,
        with its own scratch directory under the pool root)."""
        verifier = self._verifiers.get(verifier_id)
        if verifier is None:
            verifier = BenchVerifier(
                self.monitor,
                self.scratch_root / str(verifier_id),
                verifier_id=str(verifier_id),
                attempt_budget=self.attempt_budget,
                verify_timeout_s=self.verify_timeout_s,
                test_timeout_s=self.test_timeout_s)
            self._verifiers[verifier_id] = verifier
        return verifier

    # -- assignment ------------------------------------------------------

    def assign_next(self, run_id: str, *,
                    now: float | None = None) -> dict[str, Any] | None:
        """Verify the oldest submitted task on a run with the
        first pool verifier that is not the task's worker.
        Returns the verifier's outcome plus ``verifier`` (the
        assignee), a ``no_independent_verifier`` record when
        every pool member is the worker, or None when nothing
        waits."""
        waiting = [t for t in self.store.list_tasks(run_id)
                   if t["state"] == TASK_SUBMITTED]
        if not waiting:
            return None
        waiting.sort(key=lambda t: (t["submitted_at"], t["key"]))
        task = waiting[0]
        holder = str(task.get("holder") or "")
        assignee = next(
            (v for v in self.verifier_ids if v != holder), None)
        if assignee is None:
            return {"assigned": False,
                    "reason": "no_independent_verifier",
                    "task_id": task["id"], "key": task["key"],
                    "holder": holder}
        outcome = self.verifier_for(assignee).verify_task(
            run_id, task["key"], now=now)
        return {**outcome, "assigned": True, "verifier": assignee}

    def assign_all(self, run_id: str, *,
                   now: float | None = None) -> list[dict[str, Any]]:
        """Drain a run's submitted queue. Stops at the first
        task no independent verifier can take — it would be
        picked again forever, and its record names why it
        waits."""
        outcomes: list[dict[str, Any]] = []
        while True:
            outcome = self.assign_next(run_id, now=now)
            if outcome is None:
                return outcomes
            outcomes.append(outcome)
            if not outcome.get("assigned") \
                    or not outcome.get("verified"):
                return outcomes

    # -- the live-run cycle ------------------------------------------------

    def cycle(self, *, now: float | None = None) -> dict[str, Any]:
        """One pool pass over every open run on this store:
        assign what waits, then expire what sat submitted past
        its budget (the verifier phase's timeout, Lane 3).
        Safe to run any time; the registry loop runs it on a
        cadence."""
        verified: list[dict[str, Any]] = []
        for run in self.store.list_runs(state="open"):
            verified.extend(self.assign_all(run["id"], now=now))
        expired = self.verifier_for(
            self.verifier_ids[0]).expire_stale(now=now)
        return {"verified": verified, "expired": expired}

    def start(self, registry: JobRegistry | None = None, *,
              interval_s: float = 30.0) -> Any:
        """Run :meth:`cycle` on a cadence inside a JobRegistry
        job (its heartbeat/reap plumbing included). Idempotent:
        a second call returns the already-running job."""
        if self._job is not None and not self._job.done():
            return self._job
        self._job_stop.clear()
        stop = self._job_stop

        def _loop() -> str:
            while not stop.wait(max(1.0, float(interval_s))):
                self.cycle()
            return "stopped"

        registry = registry if registry is not None \
            else shared_registry()
        self._job = registry.submit(
            _loop, tool="bench-verifier-pool",
            labels={"bench": "verifier-pool"})
        return self._job

    def stop(self) -> None:
        self._job_stop.set()
