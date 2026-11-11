"""The verifier cursor (Lane 3, dossier §3).

A standing role, not a step in a builder's pipeline. The
verifier's only job is checking finished candidates, because
the failure it exists to eliminate is specific: defects a
builder's own tests bless — up to and including the
fixture/decoder pair that share one misreading and both report
green — landing as "the result" because nothing independent
ever re-ran the evidence.

Structural independence, enforced rather than promised:

- **The input is a type.** :class:`VerifyRequest` carries
  exactly the task spec (instructions + acceptance), the base
  revision, and the immutable candidate ref. There is no field
  for the builder's self-report, summary, or claimed test
  counts — a verify request cannot smuggle them in, and the
  verdict is earned from the verifier's own runs, at proof
  grade G5 (built artifact passes), never from a report.
- **Read-only on production code, by capability.** The
  ``verifier`` RoleSpec (role_specs.py) is held by
  ``RoleEnforcingRegistry`` like every role: its write tools
  are path-guarded to test paths, and the runner here executes
  tests only inside its own verification scratch. It cannot
  fix what it finds. It can only judge.
- **Self-verification is unwritable.** The verdict is written
  twice — a ``bench_verdicts`` row (the gate's input) and a
  delegation-ledger claim validated through
  ``DelegationStore.verify_claim``, whose independence law
  raises when validator == producer. A verifier constructed
  with the builder's own id cannot write a verdict over that
  builder's candidate; the store refuses, not the runner's
  politeness.
- **Verdict before integration.** The integration queue
  (``integration.py``) promotes only ``verifying`` tasks whose
  latest verdict for the CURRENT candidate ref is
  ``supported``. A resubmission is a new candidate ref, and an
  old verdict — even a supported one — has no power over it.

The procedure (dossier §3.2): materialise the candidate alone
on a clean checkout; read the full diff (the roster Reviewer
rule pass runs over it); re-run the acceptance itself; run the
adjoining suites for the touched files itself; apply the
per-class check table (§3.3). Counts in the evidence bundle
are the verifier's own counts from its own run logs.

The return loop (dossier §3.4): a non-``supported`` verdict
returns the task to its cursor (state ``returned``, leases
retained — the claimant reworks its own staging against the
evidence bundle and resubmits as a new candidate).
``contradicted`` verdicts burn an attempt budget (default 3);
on exhaustion the task escalates — a skeptic challenge on the
delegation contract and a journal event naming the arbiter
(the owner; the surface that presents it is Lane 4's).
``unavailable`` (verification itself could not run) escalates
immediately: it blocks promotion and never passes silently.
A task left ``submitted``/``verifying`` past its budget is
returned to the pool by :meth:`BenchVerifier.expire_stale` —
the submitted-state timeout Lane 1 deferred to this lane.

Layer: agents (L5) — composes the bench model/monitor, the
delegation peer, the roster Reviewer, role_specs, and the
tools layer's pytest runner.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...tools.pytest_runner import run_tests
from ..delegation import DelegationError
from ..subagents import Reviewer, ReviewInput
from .model import (
    ARBITER_REF,
    TASK_REQUEUED,
    TASK_RETURNED,
    TASK_SUBMITTED,
    TASK_VERIFYING,
    VERIFIER_REF,
    BenchError,
    overlapping_entries,
)

__all__ = [
    "BenchVerifier",
    "VerifyRequest",
    "MIGRATIONS_PATH",
    "task_class",
    "parse_acceptance",
]

#: The frozen file (AGENTS.md): a candidate that moves it is
#: contradicted by hash, never reviewed into acceptance.
MIGRATIONS_PATH = "nomorals/storage/migrations.py"

#: Default attempt budget per task (dossier §3.4): this many
#: contradicted verdicts exhausts the task and escalates it.
DEFAULT_ATTEMPT_BUDGET = 3

#: Default budget for a task sitting submitted/verifying
#: before the verifier phase returns it to the pool.
DEFAULT_VERIFY_TIMEOUT_S = 1800.0

#: Task classes the check table knows (dossier §3.3).
_CLASS_WORDS = (
    ("bugfix", ("bugfix", "bug", "fix")),
    ("format", ("format", "parser", "decoder")),
    ("schema", ("schema", "persistence", "migration")),
    ("refactor", ("refactor",)),
    ("tests", ("test",)),
    ("docs", ("doc",)),
)


def task_class(role: str) -> str:
    """The check-table class for a task's role string. Unknown
    roles land on ``feature`` — the general row, never a free
    pass."""
    text = str(role or "").lower()
    for name, words in _CLASS_WORDS:
        if any(word in text for word in words):
            return name
    return "feature"


def parse_acceptance(
        acceptance: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    """Split acceptance criteria into the executable shapes the
    verifier can run itself and the prose it can only note.

    - ``test:<path>`` (or a bare ``*.py`` path whose name says
      test) — a test file to run;
    - ``cmd:<command>`` — a shell command to run in the
      candidate tree, judged by exit code;
    - ``artifact:<path>`` — a stored real artifact the
      format-class rule judges evidence against;
    - ``repin:<test id>`` — a test whose pass/fail change a
      refactor deliberately makes (dossier §3.3);
    - anything else — prose: recorded in the evidence as noted,
      never silently treated as checked.
    """
    out: dict[str, list[str]] = {
        "tests": [], "cmds": [], "artifacts": [], "repins": [],
        "prose": []}
    for entry in acceptance:
        text = str(entry or "").strip()
        low = text.lower()
        if low.startswith("cmd:"):
            out["cmds"].append(text[4:].strip())
        elif low.startswith("test:"):
            out["tests"].append(text[5:].strip())
        elif low.startswith("artifact:"):
            out["artifacts"].append(text[9:].strip())
        elif low.startswith("repin:"):
            out["repins"].append(text[6:].strip())
        elif text.endswith(".py") and "test" in Path(text).name:
            out["tests"].append(text)
        else:
            out["prose"].append(text)
    return out


@dataclass(frozen=True)
class VerifyRequest:
    """Everything a verification may look at — and, by
    construction, nothing else. Frozen, with exactly these
    fields: the builder's self-report, summary, and claimed
    counts have no field to ride in on (dossier §3.1)."""

    run_id: str
    task_id: str
    task_key: str
    instructions: str
    acceptance: tuple[str, ...]
    role: str
    write_set: tuple[str, ...]
    base_revision: str
    candidate_ref: str

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> VerifyRequest:
        return cls(
            run_id=str(task["run_id"]), task_id=str(task["id"]),
            task_key=str(task["key"]),
            instructions=str(task["instructions"]),
            acceptance=tuple(task["acceptance"]),
            role=str(task["role"]),
            write_set=tuple(task["write_set"]),
            base_revision=str(task.get("base_revision") or ""),
            candidate_ref=str(task.get("candidate_ref") or ""))


def _git(cwd: Path, args: list[str], *,
         timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=timeout)


def _purge_pycache(root: Path) -> None:
    """Drop every ``__pycache__`` under ``root`` (and any stray
    ``.pyc`` beside sources). A revision switch inside one
    scratch must never let bytecode compiled from the other
    revision survive: same-size files and second-granularity
    ``.pyc`` validation make that a silent wrong-code run
    (see :meth:`BenchVerifier._run_on_base`)."""
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    for stale in root.rglob("*.pyc"):
        with contextlib.suppress(OSError):
            stale.unlink()


def _failed_ids(run: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for entry in run.get("failed") or []:
        if isinstance(entry, dict):
            out.append(str(entry.get("test_id") or entry.get("id")
                           or "?"))
        else:
            out.append(str(entry))
    return out


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {"ok": bool(run.get("ok")),
            "ran": int(run.get("ran") or 0),
            "passed": int(run.get("passed") or 0),
            "failed": _failed_ids(run),
            "errors": int(run.get("errors") or 0),
            "nothing_ran": bool(run.get("nothing_ran")),
            "note": str(run.get("note") or "")}


class BenchVerifier:
    """Runs verification jobs for one monitor. Constructed per
    bench (a pool is several instances with distinct ids, all
    registering verdicts under the contract's verifier pool)."""

    def __init__(self, monitor: Any, scratch_root: str | Path, *,
                 verifier_id: str = VERIFIER_REF,
                 attempt_budget: int = DEFAULT_ATTEMPT_BUDGET,
                 verify_timeout_s: float = DEFAULT_VERIFY_TIMEOUT_S,
                 test_timeout_s: float = 300.0) -> None:
        if getattr(monitor, "staging", None) is None:
            raise BenchError(
                "the verifier needs the monitor's staging manager "
                "(it owns the integration repo candidates live in)")
        self.monitor = monitor
        self.store = monitor.store
        self.staging = monitor.staging
        self.repo = Path(monitor.staging.repo)
        self.scratch_root = Path(scratch_root)
        self.verifier_id = str(verifier_id)
        self.attempt_budget = int(attempt_budget)
        self.verify_timeout_s = float(verify_timeout_s)
        self.test_timeout_s = float(test_timeout_s)

    # -- job intake -----------------------------------------------------

    def verify_next(self, run_id: str, *,
                    now: float | None = None) -> dict[str, Any] | None:
        """Verify the oldest submitted task on a run, or None
        when none waits."""
        waiting = [t for t in self.store.list_tasks(run_id)
                   if t["state"] == TASK_SUBMITTED]
        if not waiting:
            return None
        waiting.sort(key=lambda t: (t["submitted_at"], t["key"]))
        return self.verify_task(run_id, waiting[0]["key"], now=now)

    def verify_all(self, run_id: str, *,
                   now: float | None = None) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        while True:
            outcome = self.verify_next(run_id, now=now)
            if outcome is None:
                return outcomes
            outcomes.append(outcome)

    def verify_task(self, run_id: str, key: str, *,
                    now: float | None = None) -> dict[str, Any]:
        """One verification job: submitted -> verifying, the
        checks run, the verdict is written (bench journal +
        delegation ledger), and a non-supported verdict returns
        the task to its cursor with the evidence bundle."""
        now = time.time() if now is None else float(now)
        task = self.store.get_task(run_id, key)
        if task is None:
            return {"verified": False, "reason": "unknown_task",
                    "key": str(key)}
        if task["state"] != TASK_SUBMITTED:
            return {"verified": False, "reason": "not_submitted",
                    "state": task["state"], "task_id": task["id"]}
        request = VerifyRequest.from_task(task)
        moved = self.store.transition(
            task["id"], [TASK_SUBMITTED], TASK_VERIFYING, now=now)
        if not moved:
            return {"verified": False, "reason": "state_race",
                    "task_id": task["id"]}
        self.store.append_event(
            run_id, task["id"], "verification_started",
            self.verifier_id,
            generation=int(task["claim_generation"]),
            detail={"candidate_ref": request.candidate_ref,
                    "verifier": self.verifier_id},
            now=now)
        started = time.perf_counter()
        verdict, evidence = self._verify(request)
        evidence["verdict"] = verdict
        evidence["duration_s"] = round(time.perf_counter() - started, 3)
        evidence_ref = self._write_evidence(request, evidence, now)
        fresh = self.store.get_task_by_id(task["id"]) or task
        try:
            record = self._write_verdict(fresh, request, verdict,
                                         evidence_ref, evidence, now)
        except DelegationError:
            # The store refused the verdict itself (the
            # independence law: this verifier IS the producer).
            # Leave no half-state: the task goes back to
            # submitted, no bench verdict row exists, and the
            # refusal propagates as the typed error it is.
            self.store.transition(
                task["id"], [TASK_VERIFYING], TASK_SUBMITTED, now=now)
            raise
        outcome = {"verified": True, "verdict": verdict,
                   "task_id": task["id"], "key": str(key),
                   "candidate_ref": request.candidate_ref,
                   "evidence_ref": evidence_ref,
                   "verdict_id": record["id"],
                   "claim_id": record.get("claim_id", "")}
        self.store.append_event(
            run_id, task["id"], "verdict_recorded",
            self.verifier_id,
            generation=int(task["claim_generation"]),
            reason=verdict,
            detail={"candidate_ref": request.candidate_ref,
                    "evidence_ref": evidence_ref,
                    "failed_checks": evidence.get("failed_checks", []),
                    "task_class": evidence.get("task_class", "")},
            now=now)
        if verdict == "supported":
            return outcome
        # The return loop: the task goes back to its cursor with
        # the evidence; its leases stay (it remains the live
        # claimant — returned is a held state).
        self.store.transition(
            task["id"], [TASK_VERIFYING], TASK_RETURNED, now=now)
        self.store.append_event(
            run_id, task["id"], "task_returned", self.verifier_id,
            generation=int(task["claim_generation"]),
            reason=verdict,
            detail={"candidate_ref": request.candidate_ref,
                    "evidence_ref": evidence_ref,
                    "failed_checks": evidence.get("failed_checks", []),
                    "verdict_id": record["id"]},
            now=now)
        outcome["returned"] = True
        if verdict == "contradicted":
            burned = [row for row in self.store.verdicts(task["id"])
                      if row["verdict"] == "contradicted"]
            if len(burned) >= self.attempt_budget:
                self._escalate(fresh, request, record,
                               "attempt_budget_exhausted", now)
                outcome["escalated"] = True
        elif verdict == "unavailable":
            self._escalate(fresh, request, record,
                           "verification_unavailable", now)
            outcome["escalated"] = True
        return outcome

    def _escalate(self, task: dict[str, Any], request: VerifyRequest,
                  record: dict[str, Any], reason: str,
                  now: float) -> None:
        """Open the skeptic challenge on the task's contract and
        journal the escalation naming the arbiter (the owner —
        dossier §3.4; the presenting surface is Lane 4's)."""
        contract_id = str(task.get("contract_id") or "")
        challenge_id = ""
        if contract_id:
            entry = self.monitor.delegation.challenge(
                contract_id, challenger=self.verifier_id,
                claim=("bench verification escalated: " + reason
                       + " after verdict "
                       + str(record.get("verdict") or "")),
                target_claim_id=str(record.get("claim_id") or ""))
            challenge_id = entry.id
        self.store.append_event(
            request.run_id, request.task_id,
            "verification_escalated", self.verifier_id,
            reason=reason,
            detail={"arbiter": ARBITER_REF,
                    "challenge_id": challenge_id,
                    "verdict_id": str(record.get("id") or "")},
            now=now)

    # -- the submitted-state timeout (Lane 1's deferral) -----------------

    def expire_stale(self, *,
                     now: float | None = None) -> list[dict[str, Any]]:
        """Return tasks that sat submitted/verifying past the
        budget to the pool: leases released, candidate dropped,
        state ``requeued`` — the next cursor starts from the
        promoted base, and the timeout is journaled with the
        age it fired at. Idempotent and safe to run any time."""
        now = time.time() if now is None else float(now)
        out: list[dict[str, Any]] = []
        for task in self.store.list_tasks():
            if task["state"] not in (TASK_SUBMITTED, TASK_VERIFYING):
                continue
            submitted_at = float(task.get("submitted_at") or 0)
            if not submitted_at:
                continue
            age = now - submitted_at
            if age <= self.verify_timeout_s:
                continue
            for record in task["leases"]:
                self.monitor.leases.release(
                    record["lease_id"], int(record["generation"]),
                    now=now)
            moved = self.store.transition(
                task["id"], [TASK_SUBMITTED, TASK_VERIFYING],
                TASK_REQUEUED, now=now, holder="", pid=0,
                claim_generation=0, leases=[], candidate_ref="",
                last_seen=now)
            if not moved:
                continue
            self.store.append_event(
                task["run_id"], task["id"], "verification_timeout",
                self.verifier_id,
                generation=int(task["claim_generation"]),
                reason="submitted_budget_exceeded",
                detail={"age_s": round(age, 3),
                        "budget_s": self.verify_timeout_s,
                        "candidate_ref":
                            str(task.get("candidate_ref") or "")},
                now=now)
            out.append({"task_id": task["id"], "key": task["key"],
                        "run_id": task["run_id"], "age_s": age})
        return out

    # -- verdict writing ---------------------------------------------------

    def _write_verdict(self, task: dict[str, Any],
                       request: VerifyRequest, verdict: str,
                       evidence_ref: str, evidence: dict[str, Any],
                       now: float) -> dict[str, Any]:
        """Write the verdict to BOTH ledgers, before the gate
        may act on it: the delegation claim (producer = the
        candidate's worker; validator = this verifier — the
        store raises IndependenceError when they are the same
        party) and the bench_verdicts row the queue consumes."""
        claim_id = ""
        contract_id = str(task.get("contract_id") or "")
        if contract_id:
            entry = self.monitor.delegation.submit_evidence(
                contract_id, kind="verdict",
                claim=("bench candidate "
                       + (request.candidate_ref[:12] or "empty")
                       + " for task " + request.task_key),
                producer=str(task.get("holder") or ""),
                evidence_ref=evidence_ref,
                revision=request.candidate_ref)
            moved = self.monitor.delegation.verify_claim(
                entry.id, validator=self.verifier_id, verdict=verdict)
            claim_id = moved.id
        detail = {
            "task_class": evidence.get("task_class", ""),
            "checks": evidence.get("checks", []),
            "failed_checks": evidence.get("failed_checks", []),
            "counts": evidence.get("counts", {}),
            "claim_id": claim_id,
            "duration_s": evidence.get("duration_s", 0.0),
        }
        record = self.store.record_verdict(
            request.run_id, request.task_id,
            candidate_ref=request.candidate_ref,
            verifier=self.verifier_id, verdict=verdict,
            evidence_ref=evidence_ref, detail=detail, now=now)
        record["claim_id"] = claim_id
        record["verdict"] = verdict
        return record

    def _write_evidence(self, request: VerifyRequest,
                        evidence: dict[str, Any], now: float) -> str:
        """The evidence bundle as a file in the verifier's own
        evidence dir — verdict rows and delegation claims cite
        it by path (evidence_ref), so the record survives as an
        artifact, not as an agent's memory of having checked."""
        dest_dir = self.scratch_root / "evidence"
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = (request.task_id + "-"
                + (request.candidate_ref[:8] or "empty")
                + "-" + str(int(now)) + ".json")
        path = dest_dir / name
        bundle = {"request": {
            "run_id": request.run_id, "task_id": request.task_id,
            "task_key": request.task_key,
            "instructions": request.instructions,
            "acceptance": list(request.acceptance),
            "role": request.role, "write_set": list(request.write_set),
            "base_revision": request.base_revision,
            "candidate_ref": request.candidate_ref},
            "verifier": self.verifier_id, **evidence}
        path.write_text(json.dumps(bundle, indent=1, sort_keys=True),
                        encoding="utf-8")
        return str(path)

    # -- the procedure -----------------------------------------------------

    def _verify(self, request: VerifyRequest
                ) -> tuple[str, dict[str, Any]]:
        cls = task_class(request.role)
        plan = parse_acceptance(request.acceptance)
        checks: list[dict[str, Any]] = []
        contradictions: list[str] = []
        reservations: list[str] = []
        counts: dict[str, Any] = {}

        def check(name: str, status: str, detail: Any = "") -> None:
            checks.append({"name": name, "status": status,
                           "detail": detail})
            if status == "fail":
                contradictions.append(name)

        evidence: dict[str, Any] = {
            "task_class": cls, "checks": checks, "counts": counts,
            "acceptance_plan": plan}
        # Materialise the candidate alone.
        try:
            scratch = self._materialise(request)
        except BenchError as exc:
            check("materialise", "fail", str(exc))
            evidence["failed_checks"] = contradictions
            return "unavailable", evidence
        try:
            verdict = self._verify_in_scratch(
                request, cls, plan, scratch, check, contradictions,
                reservations, counts, evidence)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        evidence["failed_checks"] = contradictions
        evidence["reservations"] = reservations
        return verdict, evidence

    def _verify_in_scratch(self, request: VerifyRequest, cls: str,
                           plan: dict[str, list[str]], scratch: Path,
                           check: Any, contradictions: list[str],
                           reservations: list[str],
                           counts: dict[str, Any],
                           evidence: dict[str, Any]) -> str:
        base = request.base_revision
        cand = request.candidate_ref
        # The diff, read by the verifier itself (full, in the
        # integration repo — a read-only resolution of two
        # immutable revisions).
        files: list[str] = []
        diff_text = ""
        if cand:
            names = _git(self.repo, ["diff", "--name-only", base, cand])
            if names.returncode != 0:
                check("diff", "fail",
                      "candidate does not resolve against its base: "
                      + names.stderr.strip()[-300:])
                return "contradicted"
            files = [line.strip() for line in
                     names.stdout.splitlines() if line.strip()]
            diff = _git(self.repo, ["diff", base, cand])
            diff_text = diff.stdout if diff.returncode == 0 else ""
            parent = _git(self.repo, ["rev-list", "--parents", "-n",
                                       "1", cand])
            parts = parent.stdout.strip().split()
            if len(parts) < 2 or parts[1] != base:
                check("candidate_base", "fail",
                      "the candidate's parent is not the task's "
                      "pinned base revision — it is not the artifact "
                      "that was submitted against this spec")
                return "contradicted"
        evidence["files"] = files
        evidence["diff_stat"] = (
            _git(self.repo, ["diff", "--stat", base, cand]).stdout
            if cand else "")

        # Hard check: the diff stays inside the declared
        # write-set. Any escape is an automatic contradiction
        # naming the file.
        escapes = [f for f in files
                   if not overlapping_entries((f,),
                                              request.write_set)]
        if escapes:
            check("write_set", "fail",
                  "candidate touches files outside its declared "
                  "write-set: " + ", ".join(sorted(escapes)))
        elif files:
            check("write_set", "pass",
                  str(len(files)) + " file(s), all inside the "
                  "declared write-set")

        # Hard check: the frozen migrations file, by hash.
        mig = self._migrations_check(base, cand)
        if mig is not None:
            check("migrations_frozen", mig[0], mig[1])

        # The roster Reviewer's rule pass over the full diff.
        if diff_text:
            reviewer = Reviewer(router=None, project_root=str(scratch))
            result = reviewer.run(ReviewInput(diff=diff_text))
            flaws = [f.to_dict() for f in result.flaws]
            evidence["reviewer_flaws"] = flaws
            evidence["reviewer_model"] = result.model_review
            hard = [f for f in flaws
                    if f["severity"] in ("critical", "high")]
            if hard:
                check("reviewer", "fail",
                      "; ".join(
                          f["file"] + ":" + str(f["line"]) + " "
                          + f["message"] for f in hard[:5]))
            else:
                check("reviewer", "pass",
                      str(len(flaws)) + " flaw(s), none critical/high")
            soft = [f for f in flaws
                    if f["severity"] not in ("critical", "high")]
            if soft and not hard:
                reservations.append(
                    "reviewer noted " + str(len(soft))
                    + " low/medium flaw(s)")

        # Stub markers in added code lines (feature row of the
        # check table; applied to every code-bearing class).
        if cls not in ("tests", "docs"):
            stubs = [line[1:].strip()[:80] for line in
                     diff_text.splitlines()
                     if line.startswith("+")
                     and not line.startswith("+++")
                     and "NotImplementedError" in line]
            if stubs:
                check("no_stubs", "fail",
                      "stub marker in the diff: " + stubs[0])

        executed = False

        # The acceptance, re-run by the verifier itself.
        cand_tests = self._run_acceptance_tests(plan, scratch)
        if cand_tests is not None:
            counts["acceptance_tests"] = _run_summary(cand_tests)
            executed = executed or not cand_tests.get("nothing_ran")
            if cand_tests.get("failed") or \
                    int(cand_tests.get("errors") or 0) > 0 \
                    or not cand_tests.get("ok"):
                check("acceptance_tests", "fail",
                      "failing: " + ", ".join(
                          _failed_ids(cand_tests)[:5]))
            elif cand_tests.get("nothing_ran"):
                check("acceptance_tests", "fail",
                      "the named acceptance tests ran nothing — "
                      "zero executed tests is not a pass")
            else:
                check("acceptance_tests", "pass",
                      str(cand_tests.get("passed")) + " passed")
        for command in plan["cmds"]:
            proc = subprocess.run(
                command, shell=True, cwd=str(scratch),
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=int(self.test_timeout_s))
            executed = True
            tail = (proc.stdout + proc.stderr).strip()[-300:]
            if proc.returncode != 0:
                check("acceptance_cmd", "fail",
                      "cmd " + repr(command) + " exited "
                      + str(proc.returncode) + ": " + tail)
            else:
                check("acceptance_cmd", "pass",
                      "cmd " + repr(command) + " exited 0")
        if plan["prose"]:
            evidence["prose_acceptance"] = plan["prose"]

        # Adjoining suites for the touched files, run by the
        # verifier itself (selection: stem + import graph, the
        # house runner's changed-file selection).
        adj = None
        if files:
            adj = run_tests(changed_only=True, changed_files=files,
                            repo=str(scratch),
                            timeout=self.test_timeout_s)
            counts["adjoining"] = _run_summary(adj)
            executed = executed or not adj.get("nothing_ran")
            adj_failed = _failed_ids(adj)
            if adj_failed or int(adj.get("errors") or 0) > 0:
                base_adj = self._run_on_base(
                    request, scratch,
                    lambda: run_tests(
                        changed_only=True, changed_files=files,
                        repo=str(scratch),
                        timeout=self.test_timeout_s))
                base_failed = set(_failed_ids(base_adj)) \
                    if base_adj is not None else set()
                new_failures = [f for f in adj_failed
                                if f not in base_failed]
                if cls == "refactor":
                    vanished = [f for f in base_failed
                                if f not in set(adj_failed)
                                and not any(f.startswith(r)
                                            for r in plan["repins"])]
                    if vanished:
                        check("adjoining_refactor", "fail",
                              "tests that failed on base pass on "
                              "the candidate without a repin: "
                              + ", ".join(sorted(vanished)[:5]))
                if new_failures:
                    check("adjoining", "fail",
                          "new failures vs base: "
                          + ", ".join(new_failures[:5]))
                elif adj_failed:
                    reservations.append(
                        "adjoining failures are pre-existing on "
                        "base: " + ", ".join(adj_failed[:5]))
                    check("adjoining", "pass",
                          "failures identical on base (pre-existing)")
                else:
                    check("adjoining", "fail",
                          "adjoining run errored on the candidate")
            elif adj.get("nothing_ran"):
                check("adjoining", "na",
                      "no adjoining tests matched the touched files")
            else:
                check("adjoining", "pass",
                      str(adj.get("passed")) + " passed")

        # Layering, where the candidate tree carries the law.
        # (A structural check that runs on every candidate — it
        # does not count as executed acceptance evidence for the
        # uncertainty rule below.)
        if (scratch / "tests" / "test_layering.py").is_file():
            layering = run_tests(paths=["tests/test_layering.py"],
                                 repo=str(scratch),
                                 timeout=self.test_timeout_s)
            counts["layering"] = _run_summary(layering)
            if _failed_ids(layering) \
                    or int(layering.get("errors") or 0) > 0 \
                    or not layering.get("ok"):
                check("layering", "fail",
                      "test_layering is not green on the candidate")
            else:
                check("layering", "pass", "")

        # Per-class rows of the check table.
        if cls in ("bugfix", "tests"):
            self._check_proving_tests(request, plan, scratch, check,
                                      counts)
        if cls == "tests":
            self._check_test_hygiene(diff_text, check)
        if cls == "format":
            self._check_format_evidence(request, plan, files, check)
        if cls == "docs":
            self._check_docs(scratch, files, check)
        if cls == "feature" and files and not plan["tests"] \
                and not plan["cmds"] \
                and not any("test" in Path(f).name for f in files):
            reservations.append(
                "no test in the candidate exercises the new "
                "behaviour and the acceptance declares no "
                "executable check")

        if contradictions:
            return "contradicted"
        if files and not executed:
            return "uncertain"
        if reservations:
            return "partial"
        return "supported"

    # -- class checks ------------------------------------------------------

    def _check_proving_tests(self, request: VerifyRequest,
                             plan: dict[str, list[str]], scratch: Path,
                             check: Any,
                             counts: dict[str, Any]) -> None:
        """Bugfix/tests row: the proving test must FAIL on the
        base revision and PASS on the candidate — both runs by
        the verifier. A fix whose test passes on the base proves
        nothing and is contradicted by rule."""
        if not plan["tests"]:
            check("proving_test", "fail",
                  "a " + task_class(request.role) + " task names "
                  "no proving test in its acceptance — there is "
                  "nothing whose base-failure could prove the fix")
            return
        # The proving tests are the CANDIDATE's test files, run
        # against the BASE's production code: hold their bytes
        # (a base checkout would otherwise delete a test file
        # the candidate itself added) and lay them back over
        # the base tree for the run.
        held: dict[str, bytes] = {}
        for rel in plan["tests"]:
            source = scratch / rel
            if source.is_file():
                held[rel] = source.read_bytes()

        def _with_candidate_tests() -> dict[str, Any] | None:
            for rel, data in held.items():
                target = scratch / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return self._run_acceptance_tests(plan, scratch)

        base_run = self._run_on_base(request, scratch,
                                      _with_candidate_tests)
        if base_run is None:
            check("proving_test", "fail",
                  "the proving tests could not be run on base")
            return
        counts["proving_tests_on_base"] = _run_summary(base_run)
        base_failed = _failed_ids(base_run)
        if base_run.get("nothing_ran"):
            check("proving_test", "fail",
                  "the proving tests ran NOTHING against the base "
                  "revision — no base-side evidence exists")
        elif not base_failed and not int(base_run.get("errors") or 0):
            check("proving_test", "fail",
                  "the proving tests PASS on the base revision — "
                  "they pin nothing the candidate changes")

    def _check_test_hygiene(self, diff_text: str, check: Any) -> None:
        """Tests-only row: no bare ``except: pass`` in pins (the
        E103 census rule) and no absolute tree paths smuggled
        into sys.path pins (the tail-83 binding hazard)."""
        added = [line[1:] for line in diff_text.splitlines()
                 if line.startswith("+") and not line.startswith("+++")]
        bare = [line.strip()[:80] for line in added
                if re.search(r"except\s*(\w+\s*)?:\s*pass\b", line)
                or re.search(r"except\s*:\s*pass\b", line)]
        if bare:
            check("pin_hygiene", "fail",
                  "bare except: pass in a pin: " + bare[0])
        absolute = [line.strip()[:80] for line in added
                    if "sys.path" in line
                    and re.search(r"[\"']/(home|Users|tmp|var)/", line)]
        if absolute:
            check("pin_hygiene", "fail",
                  "absolute tree path in a sys.path pin: "
                  + absolute[0])

    def _check_format_evidence(self, request: VerifyRequest,
                               plan: dict[str, list[str]],
                               files: list[str], check: Any) -> None:
        """Format/parser row — the format-code rule (AGENTS.md):
        acceptance must run against a REAL artifact of the
        format, stored in the repo. Evidence that rests only on
        fixtures the candidate itself carries is contradicted
        by rule, however green."""
        if not plan["artifacts"]:
            check("real_artifact", "fail",
                  "format-class evidence names no real artifact "
                  "(acceptance needs an artifact:<path> entry) — "
                  "fixture-only evidence is contradicted by rule")
            return
        for relpath in plan["artifacts"]:
            if relpath in files:
                check("real_artifact", "fail",
                      "the named artifact " + relpath + " is "
                      "authored by the candidate itself — a "
                      "fixture sharing the decoder's reading, "
                      "contradicted by rule")
                return
            try:
                self.staging.read_at_revision(request.base_revision,
                                              relpath)
            except BenchError:
                check("real_artifact", "fail",
                      "the named artifact " + relpath + " does "
                      "not exist at the base revision")
                return
        check("real_artifact", "pass",
              "evidence runs against stored real artifact(s): "
              + ", ".join(plan["artifacts"]))

    def _check_docs(self, scratch: Path, files: list[str],
                    check: Any) -> None:
        """Docs row: changed docs exist and are non-empty, and
        repo paths they cite exist in the candidate tree."""
        docs = [f for f in files if f.endswith(".md")]
        missing: list[str] = []
        cited_missing: list[str] = []
        for rel in docs:
            path = scratch / rel
            if not path.is_file() or not path.read_text(
                    encoding="utf-8", errors="replace").strip():
                missing.append(rel)
                continue
            text = path.read_text(encoding="utf-8",
                                  errors="replace")
            for cited in re.findall(
                    r"`([A-Za-z0-9_/.-]+\.(?:py|md))`", text):
                if cited.startswith("http"):
                    continue
                if not (scratch / cited).exists():
                    cited_missing.append(cited)
        if missing:
            check("docs", "fail",
                  "changed docs missing or empty: "
                  + ", ".join(missing))
        elif cited_missing:
            check("docs", "fail",
                  "docs cite paths that do not exist: "
                  + ", ".join(sorted(set(cited_missing))[:5]))
        elif docs:
            check("docs", "pass", str(len(docs)) + " doc(s) sound")

    # -- runs on the other revision -----------------------------------------

    def _run_acceptance_tests(self, plan: dict[str, list[str]],
                              scratch: Path) -> dict[str, Any] | None:
        if not plan["tests"]:
            return None
        return run_tests(paths=list(plan["tests"]),
                         repo=str(scratch),
                         timeout=self.test_timeout_s)

    def _run_on_base(self, request: VerifyRequest, scratch: Path,
                     fn: Any) -> dict[str, Any] | None:
        """Run ``fn`` with the scratch checked out at the BASE
        revision, then restore the candidate checkout. The
        base-side half of every two-revision check — the
        verifier runs both sides itself.

        Every revision switch purges the scratch's bytecode
        caches first. A candidate-side run compiles the
        candidate's modules into ``__pycache__``; CPython
        validates a ``.pyc`` by source mtime in whole seconds
        plus size, so a same-size one-character fix whose base
        checkout lands in the same second as the compile would
        otherwise execute the CANDIDATE's bytecode on the base
        side — the proving test then passes on "base" and a
        sound fix is contradicted (Lane 5, Proof 8). The purge
        makes each side run the source actually checked out."""
        try:
            proc = _git(scratch, ["checkout", "--quiet",
                                  request.base_revision])
            if proc.returncode != 0:
                return None
            _purge_pycache(scratch)
            return fn()
        finally:
            target = request.candidate_ref or request.base_revision
            _git(scratch, ["checkout", "--quiet", target])
            _purge_pycache(scratch)

    # -- frozen file ---------------------------------------------------------

    def _migrations_check(self, base: str,
                          cand: str) -> tuple[str, str] | None:
        """(status, detail) for the frozen-file check, or None
        when the repo carries no such file at base (nothing to
        freeze). Hash equality — never a review judgement."""
        try:
            base_bytes = self.staging.read_at_revision(
                base, MIGRATIONS_PATH)
        except BenchError:
            return None
        base_hash = hashlib.sha256(base_bytes).hexdigest()
        if not cand:
            return ("pass", "migrations.py untouched (empty "
                    "candidate)")
        try:
            cand_bytes = self.staging.read_at_revision(
                cand, MIGRATIONS_PATH)
        except BenchError:
            return ("fail", "migrations.py is MISSING from the "
                    "candidate — the frozen file was removed")
        cand_hash = hashlib.sha256(cand_bytes).hexdigest()
        if cand_hash != base_hash:
            return ("fail", "migrations.py differs from base "
                    "(sha256 " + cand_hash[:12] + " != "
                    + base_hash[:12] + ") — the file is frozen")
        return ("pass", "migrations.py hash-identical to base")

    # -- materialisation ------------------------------------------------------

    def _materialise(self, request: VerifyRequest) -> Path:
        """The candidate alone, on a clean checkout in the
        verifier's own scratch. Raises BenchError when the
        candidate cannot be materialised at all (an
        environment-grade failure — the verdict is
        ``unavailable``, never a silent pass)."""
        dest = self.scratch_root / "work" / (
            request.task_id + "-"
            + (request.candidate_ref[:8] or "empty"))
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = _git(self.repo, ["clone", "--quiet", str(self.repo),
                                str(dest)])
        if proc.returncode != 0 or not dest.is_dir():
            raise BenchError(
                "verifier clone failed: "
                + proc.stderr.strip()[-300:])
        target = request.candidate_ref or request.base_revision
        if target:
            have = _git(dest, ["cat-file", "-t", target])
            if have.returncode != 0 and request.candidate_ref:
                # A plain clone may not carry the candidate ref's
                # objects; fetch the pinned ref explicitly (the
                # staging copy-fallback's own transfer idiom).
                _git(dest, ["fetch", "--quiet", str(self.repo),
                            "+refs/bench/candidates/"
                            + request.task_id
                            + ":refs/bench/candidates/"
                            + request.task_id])
            proc = _git(dest, ["checkout", "--quiet", target])
            if proc.returncode != 0:
                raise BenchError(
                    "verifier checkout of "
                    + target[:12] + " failed: "
                    + proc.stderr.strip()[-300:])
        return dest
