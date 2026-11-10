"""Repair pipeline — doc 20 rungs 7–10 (candidate patch → promotion).

The shipped healing ladder (rungs 1–6) lives in
:mod:`nomorals.core.selfheal` (the executor: classify, bounded retry,
restart-class recovery, verified-fix promotion through the
:class:`~nomorals.core.incidents.IncidentJournal`) and is wired into
the one tool-call funnel by doc 20 P2.  Those rungs recover a call;
they never change code.  This module is the ladder's continuation
for the one repair class they cannot perform — a *code defect* —
exactly as the doc-20 ladder table specifies:

* rung 7 — generate a candidate patch in a sandbox (a detached git
  worktree at the production HEAD); no production write yet;
* rung 8 — run the mandatory regression/property/security suite in
  the sandbox; a failing candidate never leaves it;
* rung 9 — canary on a single instance/profile: a second worktree
  exercised with the update health checks plus the incident's
  reproduction probes.  A canary failure is rolled back
  AUTOMATICALLY (the canary tree is reset to the pre-patch commit
  and the reset is verified) before anything is recorded as done;
* rung 10 — promote, monitor, retain rollback: policy-gated (the
  shipped :class:`PolicyGate`; consequential changes park for the
  owner), the production write happens ONLY here, a pre-repair
  snapshot is retained through :class:`~nomorals.os.snapshots.SnapshotManager`
  (the UpdateManager machinery — no second snapshot system), the
  post-promotion health monitor runs the same
  :func:`~nomorals.os.update.default_health_checks` the self-update
  gate uses, and any monitor failure restores the tree with the
  UpdateManager ``_restore_tree`` shape (only the repair's own files
  are touched; HEAD/index move with ``reset --mixed``).

Records: every rung transition is a
:class:`~nomorals.core.incidents.RecoveryRecord` on the incident in
the shipped journal — the same records doc 20 P1's healing-ticket
projection reads, so the trail (patch proposed → suite verdicts →
canary → gate → promoted / rolled back) is complete for the ticket
surface with no new store.  ``verified=True`` is set ONLY on the
``repair_promote`` record: the journal promotes verified recoveries
into its verified-fix table, and only a completed promotion is a
fix.  Every other record states its outcome in its detail payload
and ``verify_note`` instead of claiming the incident recovered.

Write-path discipline (doc 20 §9: zero production writes without a
ticket receipt): the only functions in this module that write to
the production tree are the promotion path and the tree-restore
helper, and promotion re-derives the run's journal trail from the
journal itself and refuses — writing nothing — unless the trail
contains a clean proposal, a fully green suite and a green canary
for THIS candidate hash.  A repair without an incident is refused
by construction: there is nothing to journal a receipt against.

Scope binds, named: this pipeline repairs CODE (unified-diff
patches).  It does not widen mission auto-repair — it never calls
``MissionRunner.self_heal``, never writes mission state, and
imports nothing from ``nomorals.missions`` or
``nomorals.os.mission_state``.  Doc 9's two narrow mission fixes
therefore remain the stated prerequisite of any FUTURE pass that
raises mission-repair autonomy; they are not part of this one.
Rung 11 (regression-test + antibody/playbook update after a
verified outcome) is likewise not here — it is the doc's P8/M-G4
territory.

Sandbox confinement: suite steps and probes run in worktrees via an
injected command runner.  The default runner is a subprocess with a
process-group kill on timeout (the execbox fallback posture);
composition roots in higher layers may inject an execbox-backed
runner for bwrap confinement — the pipeline itself sits at layer 5
beside the healing ticket and may not import the layer-7 execbox.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..codews.patch import apply_patch, check_patch, review_patch
from ..codews.run import run_tests
from ..core.incidents import RecoveryRecord
from ..core.own_processes import mark_spawned_exited, register_spawned
from ..os.update import default_health_checks
from ..security.audit_patterns import entropy_hits, scan_text_for_secrets

__all__ = [
    "FROZEN_PATHS",
    "GateVerdict",
    "PatchCandidate",
    "PolicyGate",
    "RepairError",
    "RepairPipeline",
    "RepairRefused",
    "RunState",
    "assess_patch",
    "candidate_hash_of",
]

#: Files no candidate may ever touch: the frozen migration history
#: (its tamper guard refuses to boot on any edit) and the owner's
#: persona record.  House law, enforced as a pipeline property.
FROZEN_PATHS = frozenset({
    "nomorals/storage/migrations.py",
    "nomorals/memory/persona.py",
})

#: Recovery-record strategy names, one per rung transition.  The
#: healing-ticket projection renders these verbatim, so they are
#: part of the record surface — rename only with the projection.
STRATEGY_REFUSED = "repair_patch_refused"
STRATEGY_PROPOSED = "repair_patch_proposed"
STRATEGY_SUITE = "repair_suite"
STRATEGY_CANARY = "repair_canary"
STRATEGY_CANARY_ROLLBACK = "repair_canary_rollback"
STRATEGY_GATE = "repair_gate"
STRATEGY_PROMOTE = "repair_promote"
STRATEGY_ROLLBACK = "repair_rollback"

#: Run statuses (the fold of a run's records, latest wins).
STATUS_REFUSED = "PATCH_REFUSED"
STATUS_REJECTED_PATCH = "PATCH_REJECTED"
STATUS_PROPOSED = "PROPOSED"
STATUS_SUITE_FAILED = "SUITE_FAILED"
STATUS_SUITE_PASSED = "SUITE_PASSED"
STATUS_CANARY_ROLLED_BACK = "CANARY_ROLLED_BACK"
STATUS_CANARY_PASSED = "CANARY_PASSED"
STATUS_AWAITING = "AWAITING_APPROVAL"
STATUS_REJECTED = "REJECTED"
STATUS_PROMOTED = "PROMOTED"
STATUS_PROMOTION_ROLLED_BACK = "PROMOTION_ROLLED_BACK"
STATUS_ROLLED_BACK = "ROLLED_BACK"

_TERMINAL = frozenset({
    STATUS_REFUSED, STATUS_REJECTED_PATCH, STATUS_SUITE_FAILED,
    STATUS_CANARY_ROLLED_BACK, STATUS_REJECTED, STATUS_PROMOTED,
    STATUS_PROMOTION_ROLLED_BACK, STATUS_ROLLED_BACK,
})

# ── consequentiality assessment ────────────────────────────────────
# Mirrored from agents/coding.py's structural diff metrics (the
# house pattern for cross-module constants: mirror + cite, pinned
# equal by tests/test_doc20p6_repair_pipeline.py).  The assessment
# reads diff STRUCTURE only — never the producer's claims.
_SENSITIVE_SEGMENTS = frozenset({"auth", "credentials", "secrets", "vault", ".ssh"})
_SENSITIVE_STEMS = frozenset({
    "config", "settings", "secrets", "credentials", "auth", "vault",
    "password", "passwords", "token", "tokens", "api_key", "apikey",
})
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".env")
_WIDE_FILES = 5
_WIDE_LINES = 400
_HEAVY_DELETED = 50
_API_DEF_RE = re.compile(r"^(?:def|class)\s+[A-Za-z_]")
_HARD_RISKS = frozenset({
    "sensitive-path", "public-api-change", "file-deletion", "heavy-deletion",
})


class RepairError(RuntimeError):
    """The pipeline cannot run at all (environment, not outcome)."""


class RepairRefused(RepairError):
    """A gated write was refused; nothing was written."""


def _sensitive_path(path: str) -> bool:
    parts = [p for p in re.split(r"[/\\]", path) if p]
    if any(p in _SENSITIVE_SEGMENTS for p in parts):
        return True
    if not parts:
        return False
    name = parts[-1]
    if name == ".env" or name.startswith(".env."):
        return True
    if name.endswith(_SENSITIVE_SUFFIXES):
        return True
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return stem in _SENSITIVE_STEMS


def assess_patch(diff_text: str) -> dict[str, Any]:
    """Structural risk assessment of a candidate diff.

    Returns file/line counts, the risk signals present, and whether
    the change is *consequential* (an owner-gate trigger): any hard
    risk (sensitive path, public-API change, file deletion, heavy
    deletion) or a wide change (>= 5 files or >= 400 changed lines).
    """
    review = review_patch(diff_text)
    files = [str(f.get("path", "")) for f in review["files"]]
    added = int(review["totals"]["additions"])
    deleted = int(review["totals"]["deletions"])
    risks: list[str] = []
    if any(_sensitive_path(p) for p in files):
        risks.append("sensitive-path")
    if any(f.get("is_removed") for f in review["files"]):
        risks.append("file-deletion")
    if deleted >= _HEAVY_DELETED and deleted >= added:
        risks.append("heavy-deletion")
    api_change = False
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line[:1] in ("+", "-"):
            body = line[1:]
            if _API_DEF_RE.match(body) or "__all__" in body:
                api_change = True
                break
    if api_change:
        risks.append("public-api-change")
    wide = len(files) >= _WIDE_FILES or (added + deleted) >= _WIDE_LINES
    if wide:
        risks.append("wide-change")
    consequential = bool(_HARD_RISKS.intersection(risks)) or wide
    return {
        "files": files, "file_count": len(files),
        "added": added, "deleted": deleted,
        "risks": risks, "consequential": consequential,
    }


def candidate_hash_of(diff_text: str) -> str:
    return hashlib.sha256(diff_text.encode("utf-8")).hexdigest()[:16]


def _fit_payload(payload: dict, limit: int = 3600) -> dict:
    """Shrink a record payload until its JSON fits the journal.

    Record details must stay parseable — a truncated JSON string
    would make the whole run unreadable to the projection — so
    oversized string values are shortened structurally instead of
    slicing the serialized text.  The keys the pipeline's own
    guards re-derive state from (run, candidate, status, ok,
    snapshot, shas) are all short and survive untouched.
    """
    def shrink(value: Any, cap: int) -> Any:
        if isinstance(value, str) and len(value) > cap:
            return value[:cap] + "…"
        if isinstance(value, dict):
            return {k: shrink(v, cap) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [shrink(v, cap) for v in value]
        return value

    fitted = payload
    for cap in (800, 300, 120, 60):
        if len(json.dumps(fitted, default=str)) <= limit:
            return fitted
        fitted = shrink(payload, cap)
    return fitted


# ── candidates, producers, gate ────────────────────────────────────


@dataclass
class PatchCandidate:
    """One candidate repair: a unified diff plus its provenance.

    ``probes`` are reproduction commands (argv tuples) run in the
    canary worktree — for a code defect, the commands that exercised
    the failing behaviour.  ``suite_selectors`` explicitly narrows
    the rung-8 test steps; the narrowing is recorded on the ticket.
    """

    diff_text: str
    producer: str
    rationale: str = ""
    incident_id: int | None = None
    probes: tuple[tuple[str, ...], ...] = ()
    suite_selectors: tuple[str, ...] | None = None


class PatchProducer(Protocol):
    """Anything that can propose a candidate for an incident."""

    def produce(self, incident: Any) -> PatchCandidate | None:
        ...


@dataclass
class StaticPatchProducer:
    """Hands the pipeline one explicit candidate (a surface/owner)."""

    candidate: PatchCandidate
    name: str = "static"

    def produce(self, incident: Any) -> PatchCandidate | None:
        return self.candidate


@dataclass
class CallablePatchProducer:
    """Wraps any callable ``incident -> PatchCandidate | None``.

    The composition seam for brain/CodingAgent producers: whatever
    authors the patch, the pipeline's rungs treat it identically.
    """

    fn: Callable[[Any], PatchCandidate | None]
    name: str = "callable"

    def produce(self, incident: Any) -> PatchCandidate | None:
        return self.fn(incident)


@dataclass
class GateVerdict:
    approved: bool
    by: str
    reason: str


class PolicyGate:
    """The doc's rung-9/10 policy gate, as code.

    Non-consequential candidates that reached the gate (suite and
    canary already green) are approved by policy.  Consequential
    ones are NOT refused — they park for the owner, whose approval
    arrives through the shipped surfaces (the console operator is
    the owner; see cmdline/commands/repair.py).  ``owner_always``
    parks every candidate.
    """

    def __init__(self, *, owner_always: bool = False) -> None:
        self.owner_always = owner_always

    def decide(self, assessment: dict[str, Any]) -> GateVerdict:
        if self.owner_always:
            return GateVerdict(False, "policy:owner-required",
                               "policy parks every repair for the owner")
        if assessment.get("consequential"):
            risks = ", ".join(assessment.get("risks") or [])
            return GateVerdict(
                False, "policy:owner-required",
                "consequential change (" + risks + ") needs the owner")
        return GateVerdict(True, "policy:auto",
                           "non-consequential; suite and canary green")


# ── run state (read-computed from the journal, P1-style) ──────────


@dataclass
class RunState:
    run_id: str
    incident_id: int
    signature: str
    candidate_hash: str
    status: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)

    def detail_of(self, strategy: str) -> dict[str, Any]:
        """The detail payload of the LAST record with ``strategy``."""
        for rec in reversed(self.records):
            if rec.get("strategy") == strategy:
                return rec.get("detail") or {}
        return {}

    def has(self, strategy: str) -> bool:
        return any(r.get("strategy") == strategy for r in self.records)


# ── the pipeline ───────────────────────────────────────────────────

HealthFactory = Callable[[Path, Any], list]
RunnerFn = Callable[..., tuple]


def _subprocess_runner(argv: list, *, cwd: Path, timeout: float,
                       env_extra: dict | None = None) -> tuple:
    """Default command runner: subprocess, process-group kill.

    Mirrors the execbox fallback posture (no network control — the
    worktree is the confinement at this layer; a composition root
    may inject an execbox-backed runner for bwrap isolation).
    Returns (returncode, stdout_tail, stderr_tail).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(
        argv, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True)
    # Doc 20 P7: this child is one of Devon's own spawned
    # processes — registered (fail-open; the record never breaks
    # the run) so the immune police's fenced kill can name it if
    # it ever runs away, and marked exited when it ends.
    proc_row = register_spawned(
        proc.pid, label="repair pipeline command",
        spawned_by="repair_pipeline",
        argv0=str(argv[0]) if argv else "")
    try:
        out, err = proc.communicate(timeout=timeout)
        mark_spawned_exited(proc_row)
        return proc.returncode, out[-1500:], err[-1500:]
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
        mark_spawned_exited(proc_row, note="timeout kill by the runner")
        return 124, (out or "")[-1500:], "TIMEOUT after " + str(timeout) + "s"


class RepairPipeline:
    """Rungs 7–10 for one install: journal + repo + work root."""

    def __init__(self, *, journal: Any, repo_dir: Path, work_root: Path,
                 snapshots: Any = None, health_factory: HealthFactory = None,
                 gate: PolicyGate = None, runner: RunnerFn = None,
                 git_bin: str = "git", db_path: Any = None,
                 suite_timeout: float = 600.0,
                 probe_timeout: float = 180.0) -> None:
        self.journal = journal
        self.repo_dir = Path(repo_dir)
        self.work_root = Path(work_root)
        self.snapshots = snapshots
        self.health_factory = health_factory or (
            lambda root, db: default_health_checks(root, db))
        self.gate = gate or PolicyGate()
        self.runner = runner or _subprocess_runner
        self.git_bin = git_bin
        self.db_path = db_path
        self.suite_timeout = suite_timeout
        self.probe_timeout = probe_timeout
        self._cache: dict[int, list[dict[str, Any]]] = {}

    # ── journal plumbing ──────────────────────────────────────────

    def _record(self, state: RunState, strategy: str, status: str, *,
                verified: bool = False, verify_note: str = "",
                **detail: Any) -> None:
        payload = {"run": state.run_id, "candidate": state.candidate_hash,
                   "status": status}
        payload.update(detail)
        text = json.dumps(_fit_payload(payload), default=str)
        rec = RecoveryRecord(
            incident_id=state.incident_id, signature=state.signature,
            strategy=strategy, detail=text, verified=verified,
            verify_note=verify_note)
        self.journal.record_recovery(rec)
        self._cache.setdefault(state.incident_id, []).append({
            "strategy": strategy, "detail": payload,
            "verified": verified, "verify_note": verify_note,
            "ts": time.time()})
        state.records.append({
            "strategy": strategy, "detail": payload,
            "verified": verified, "verify_note": verify_note,
            "ts": time.time()})
        state.status = status

    def _recoveries_for(self, incident_id: int) -> list[dict[str, Any]]:
        """All recovery records for an incident, in journal order.

        File-backed journals are read read-only from the journal
        file (the healing-ticket projection's own precedent — runs
        span processes: propose on one invocation, approve on
        another).  In-memory journals fall back to this instance's
        write cache.
        """
        path = getattr(self.journal, "_path", "")
        if path and path != ":memory:" and Path(str(path)).exists():
            rows: list[dict[str, Any]] = []
            conn = sqlite3.connect(
                "file:" + str(path) + "?mode=ro", uri=True)
            try:
                cur = conn.execute(
                    "SELECT strategy, detail, verified, verify_note, ts"
                    " FROM recoveries WHERE incident_id = ? ORDER BY id",
                    (incident_id,))
                for strategy, detail, verified, note, ts in cur.fetchall():
                    try:
                        payload = json.loads(detail or "{}")
                    except (ValueError, TypeError):
                        payload = {}
                    rows.append({"strategy": strategy, "detail": payload,
                                 "verified": bool(verified),
                                 "verify_note": note or "", "ts": ts or 0.0})
            finally:
                conn.close()
            return rows
        return list(self._cache.get(incident_id, []))

    def _load_run(self, run_id: str) -> RunState | None:
        parts = run_id.split("-")
        if len(parts) != 3 or parts[0] != "rp":
            return None
        try:
            incident_id = int(parts[1])
        except ValueError:
            return None
        try:
            incident = self.journal.get_incident(incident_id)
            signature = incident.signature
        except Exception:  # noqa: BLE001 - unknown incident, no run
            return None
        records = [r for r in self._recoveries_for(incident_id)
                   if (r.get("detail") or {}).get("run") == run_id]
        if not records:
            return None
        state = RunState(
            run_id=run_id, incident_id=incident_id, signature=signature,
            candidate_hash=str((records[0].get("detail") or {}).get(
                "candidate", parts[2])),
            records=records)
        for rec in records:
            status = (rec.get("detail") or {}).get("status")
            if status:
                state.status = str(status)
        return state

    def list_runs(self, incident_id: int | None = None) -> list[RunState]:
        """Every repair run reconstructable from the journal."""
        run_ids: list[str] = []
        if incident_id is not None:
            candidates = [incident_id]
        else:
            candidates = []
            path = getattr(self.journal, "_path", "")
            if path and path != ":memory:" and Path(str(path)).exists():
                conn = sqlite3.connect(
                    "file:" + str(path) + "?mode=ro", uri=True)
                try:
                    cur = conn.execute(
                        "SELECT DISTINCT incident_id FROM recoveries"
                        " WHERE strategy LIKE 'repair_%'")
                    candidates = [int(r[0]) for r in cur.fetchall()]
                finally:
                    conn.close()
            else:
                candidates = list(self._cache.keys())
        for inc in candidates:
            seen: set[str] = set()
            for rec in self._recoveries_for(inc):
                run = (rec.get("detail") or {}).get("run")
                if run and run not in seen:
                    seen.add(run)
                    run_ids.append(str(run))
        states = []
        for run_id in run_ids:
            state = self._load_run(run_id)
            if state is not None:
                states.append(state)
        return states

    # ── git plumbing ──────────────────────────────────────────────

    def _git(self, args: list, *, cwd: Path) -> tuple:
        proc = subprocess.run(
            [self.git_bin] + args, cwd=str(cwd), capture_output=True,
            text=True, timeout=120)
        return proc.returncode, proc.stdout, proc.stderr

    def _head_sha(self, root: Path) -> str:
        rc, out, err = self._git(["rev-parse", "HEAD"], cwd=root)
        if rc != 0:
            raise RepairError("cannot read HEAD of " + str(root)
                              + ": " + err.strip()[:300])
        return out.strip()

    def _worktree_add(self, path: Path, sha: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rc, _out, err = self._git(
            ["worktree", "add", "--detach", str(path), sha],
            cwd=self.repo_dir)
        if rc != 0:
            raise RepairError("git worktree add failed: "
                              + err.strip()[:300])

    def _worktree_remove(self, path: Path) -> None:
        if not path.exists():
            return
        rc, _out, _err = self._git(
            ["worktree", "remove", "--force", str(path)], cwd=self.repo_dir)
        if rc != 0 or path.exists():
            shutil.rmtree(path, ignore_errors=True)
            self._git(["worktree", "prune"], cwd=self.repo_dir)

    def _restore_tree(self, root: Path, from_sha: str,
                      touched: list) -> tuple:
        """UpdateManager ``_restore_tree`` shape, scoped to a run.

        Only files the repair itself changed are touched (checked
        out from ``from_sha``, or removed when the repair added
        them); HEAD and the index move with ``reset --mixed``, which
        never touches other working-tree content.  The restore is
        VERIFIED: HEAD must be back at ``from_sha`` and every touched
        path must match its ``from_sha`` state (content or absence).
        Returns (ok, detail).
        """
        current = self._head_sha(root)
        if current != from_sha:
            # The guarded callers guarantee HEAD is the repair's own
            # promotion commit, so this delta is exactly the repair.
            rc, out, _err = self._git(
                ["diff", "--name-only", from_sha, current], cwd=root)
            changed = [n for n in out.splitlines() if n] if rc == 0 \
                else list(touched)
            for name in sorted(set(changed)):
                rc, _o, _e = self._git(
                    ["cat-file", "-e", from_sha + ":" + name], cwd=root)
                if rc == 0:
                    rc2, _o, err = self._git(
                        ["checkout", from_sha, "--", name], cwd=root)
                    if rc2 != 0:
                        return False, "checkout of " + name + " failed: " \
                            + err.strip()[:200]
                else:
                    try:
                        (root / name).unlink(missing_ok=True)
                    except OSError as exc:
                        return False, "could not remove added file " \
                            + name + ": " + str(exc)
            rc, _o, err = self._git(["reset", "--mixed", from_sha], cwd=root)
            if rc != 0:
                return False, "git reset --mixed failed: " \
                    + err.strip()[:200]
        # Uncommitted leftovers on touched paths (apply without a
        # commit, or a commit that failed midway).
        for name in touched:
            rc, out, _e = self._git(
                ["status", "--porcelain", "--", name], cwd=root)
            if rc == 0 and out.strip():
                rc2, _o, _e2 = self._git(
                    ["cat-file", "-e", from_sha + ":" + name], cwd=root)
                if rc2 == 0:
                    self._git(["checkout", from_sha, "--", name], cwd=root)
                else:
                    with contextlib.suppress(OSError):
                        (root / name).unlink(missing_ok=True)
        if self._head_sha(root) != from_sha:
            return False, "HEAD did not return to the pre-repair commit"
        for name in touched:
            rc, out, _e = self._git(
                ["status", "--porcelain", "--", name], cwd=root)
            if rc == 0 and out.strip():
                return False, "touched path still dirty after restore: " \
                    + name
        return True, "tree restored to " + from_sha[:12] \
            + " (" + str(len(touched)) + " repair file(s) put back)"

    # ── rung 8 helpers ────────────────────────────────────────────

    def _suite_selectors(self, candidate: PatchCandidate,
                         root: Path, files: list) -> tuple:
        """Test selectors for the mandatory suite.

        An explicit candidate narrowing wins (and is recorded);
        otherwise selectors map from the touched files — the tests
        the patch itself touches, plus ``tests/test_<stem>*.py`` for
        each touched module.  No mapping at all means the WHOLE
        suite of the target root: mandatory means mandatory.
        """
        if candidate.suite_selectors is not None:
            return tuple(candidate.suite_selectors), "candidate"
        selectors: list[str] = []
        for name in files:
            if name.startswith("tests/") and name.endswith(".py") \
                    and name not in selectors:
                selectors.append(name)
        tests_dir = root / "tests"
        for name in files:
            if name.startswith("tests/"):
                continue
            stem = Path(name).stem
            if not stem:
                continue
            for cand in sorted(tests_dir.glob("test_" + stem + "*.py")):
                rel = "tests/" + cand.name
                if rel not in selectors:
                    selectors.append(rel)
        selectors = selectors[:8]
        if not selectors:
            return ("",), "whole-suite-fallback"
        return tuple(selectors), "mapped"

    def _run_suite(self, state: RunState, candidate: PatchCandidate,
                   sandbox: Path, files: list) -> bool:
        """Rung 8: regression + security + property steps, in order.

        Every step journals its own ``repair_suite`` record; the
        first failure ends the run (the candidate never canaries).
        """
        steps: list[tuple] = []
        selectors, mapping = self._suite_selectors(candidate, sandbox, files)
        for selector in selectors:
            steps.append(("tests:" + (selector or "<whole-suite>"),
                          "tests", selector))
        steps.append(("security:added-lines", "security", ""))
        steps.append(("property:touched-code", "property", ""))
        for idx, (name, kind, arg) in enumerate(steps):
            last = idx == len(steps) - 1
            if kind == "tests":
                result = run_tests(sandbox, arg, timeout=self.suite_timeout)
                passed = int(result.get("passed", 0) or 0)
                failed = int(result.get("failed", 0) or 0)
                skipped = int(result.get("skipped", 0) or 0)
                total = passed + failed + skipped
                ok = bool(result.get("ok")) and total > 0 and failed == 0
                evidence = {
                    "selector": arg, "mapping": mapping,
                    "runner": result.get("runner", ""),
                    "tests": total, "passed": passed,
                    "failed": failed, "skipped": skipped,
                    "failed_cases": [
                        str(c)[:120] for c in
                        (result.get("failed_tests") or [])][:5],
                }
            elif kind == "security":
                added_lines = "\n".join(
                    line[1:] for line in candidate.diff_text.splitlines()
                    if line.startswith("+") and not line.startswith("+++"))
                hits = scan_text_for_secrets(added_lines) \
                    + entropy_hits(added_lines)
                ok = not hits
                evidence = {"hits": len(hits),
                            "samples": [h.redacted for h in hits[:3]]}
            else:  # property: touched code still compiles and imports
                ok, evidence = self._property_step(sandbox, files)
            status = STATUS_PROPOSED
            if not ok:
                status = STATUS_SUITE_FAILED
            elif last:
                status = STATUS_SUITE_PASSED
            self._record(
                state, STRATEGY_SUITE, status,
                verify_note=("suite step " + name + " passed") if ok
                else ("suite step " + name + " FAILED — candidate "
                      "stops here and never reaches canary"),
                step=name, kind=kind, ok=ok, evidence=evidence)
            if not ok:
                return False
        return True

    def _property_step(self, root: Path, files: list) -> tuple:
        """Touched .py files compile; touched packages still import
        in a fresh interpreter (the update gate's own import shape).
        """
        compiled = 0
        for name in files:
            if not name.endswith(".py"):
                continue
            target = root / name
            if not target.exists():
                continue  # a deletion: nothing to compile
            rc, _out, err = self.runner(
                [sys.executable, "-m", "py_compile", name],
                cwd=root, timeout=60)
            if rc != 0:
                return False, {"file": name, "stage": "compile",
                               "error": err[-400:]}
            compiled += 1
        packages = sorted({Path(n).parts[0] for n in files
                           if len(Path(n).parts) > 1
                           and (root / Path(n).parts[0]
                                / "__init__.py").exists()})
        imported = []
        for pkg in packages:
            rc, _out, err = self.runner(
                [sys.executable, "-c", "import " + pkg],
                cwd=root, timeout=120)
            if rc != 0:
                return False, {"package": pkg, "stage": "import",
                               "error": err[-400:]}
            imported.append(pkg)
        return True, {"compiled": compiled, "imported": imported}

    def _health_results(self, root: Path, db_path: Any) -> list:
        results = []
        for check in self.health_factory(root, db_path):
            name = getattr(check, "__name__", "health_check")
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001 - a raising check fails
                ok, detail = False, "raised " + type(exc).__name__ \
                    + ": " + str(exc)
            results.append({"name": name, "ok": bool(ok),
                            "detail": str(detail)[:300]})
        return results

    # ── the pipeline ──────────────────────────────────────────────

    def run(self, candidate: PatchCandidate) -> RunState:
        """Rungs 7–9 plus the gate; promotion when policy approves.

        Idempotent per (incident, candidate): an existing run for
        the same pair is returned as recorded, never re-executed.
        """
        if candidate.incident_id is None:
            raise RepairRefused(
                "a repair candidate without an incident has nothing "
                "to journal its receipt against — refused")
        try:
            incident = self.journal.get_incident(candidate.incident_id)
        except Exception as exc:  # noqa: BLE001 - unknown incident
            raise RepairRefused(
                "unknown incident " + str(candidate.incident_id)
                + ": " + str(exc)) from exc
        chash = candidate_hash_of(candidate.diff_text)
        run_id = "rp-" + str(candidate.incident_id) + "-" + chash[:10]
        existing = self._load_run(run_id)
        if existing is not None:
            return existing
        state = RunState(run_id=run_id, incident_id=candidate.incident_id,
                         signature=incident.signature, candidate_hash=chash)

        # Rung 7, review half: parse + frozen-file guard.
        try:
            review = review_patch(candidate.diff_text)
        except Exception as exc:  # noqa: BLE001 - unparseable candidate
            self._record(
                state, STRATEGY_REFUSED, STATUS_REFUSED,
                verify_note="candidate diff could not be parsed; "
                            "nothing was written anywhere",
                producer=candidate.producer,
                reason=type(exc).__name__ + ": " + str(exc)[:300])
            return state
        files = [str(f.get("path", "")) for f in review["files"]]
        frozen = [f for f in files if f in FROZEN_PATHS]
        if frozen:
            self._record(
                state, STRATEGY_REFUSED, STATUS_REFUSED,
                verify_note="candidate touches frozen file(s); "
                            "refused before any worktree was created",
                producer=candidate.producer, files=files,
                frozen=frozen)
            return state
        if not files:
            self._record(
                state, STRATEGY_REFUSED, STATUS_REFUSED,
                verify_note="candidate diff has no file sections",
                producer=candidate.producer)
            return state

        run_dir = self.work_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "candidate.diff").write_text(
            candidate.diff_text, encoding="utf-8")
        base_sha = self._head_sha(self.repo_dir)

        # Rung 7: materialize in the sandbox worktree.
        sandbox = run_dir / "sandbox"
        self._worktree_add(sandbox, base_sha)
        applied, apply_note = self._apply_to_worktree(
            candidate.diff_text, sandbox)
        if not applied:
            self._record(
                state, STRATEGY_PROPOSED, STATUS_REJECTED_PATCH,
                verify_note="candidate does not apply cleanly to the "
                            "current tree; rejected before the suite",
                producer=candidate.producer,
                rationale=candidate.rationale[:400], files=files,
                added=int(review["totals"]["additions"]),
                deleted=int(review["totals"]["deletions"]),
                base_sha=base_sha, ok=False, note=apply_note[:400])
            self._worktree_remove(sandbox)
            return state
        self._record(
            state, STRATEGY_PROPOSED, STATUS_PROPOSED,
            verify_note="candidate materialized in the sandbox "
                        "worktree only; no production write yet",
            producer=candidate.producer,
            rationale=candidate.rationale[:400], files=files,
            added=int(review["totals"]["additions"]),
            deleted=int(review["totals"]["deletions"]),
            base_sha=base_sha, ok=True, sandbox=str(sandbox))

        # Rung 8: the mandatory suite, in the sandbox.
        suite_ok = self._run_suite(state, candidate, sandbox, files)
        self._worktree_remove(sandbox)
        if not suite_ok:
            return state

        # Rung 9: canary on a fresh worktree (single profile).
        canary = run_dir / "canary"
        self._worktree_add(canary, base_sha)
        applied, apply_note = self._apply_to_worktree(
            candidate.diff_text, canary)
        if not applied:
            probes = [{"name": "apply", "ok": False,
                       "detail": apply_note[:300]}]
            self._record(
                state, STRATEGY_CANARY, STATUS_PROPOSED,
                verify_note="candidate would not apply in the canary "
                            "worktree — automatic rollback follows",
                ok=False, canary=str(canary), probes=probes)
            return self._fail_canary(state, canary, base_sha, files,
                                     probes)
        probes = self._health_results(canary, None)
        for argv in candidate.probes:
            rc, out, err = self.runner(
                list(argv), cwd=canary, timeout=self.probe_timeout)
            probes.append({
                "name": "probe:" + " ".join(str(a) for a in argv)[:80],
                "ok": rc == 0,
                "detail": ("rc=" + str(rc) + " " + (err or out))[-300:]})
        canary_ok = all(p["ok"] for p in probes)
        self._record(
            state, STRATEGY_CANARY,
            STATUS_CANARY_PASSED if canary_ok else STATUS_PROPOSED,
            verify_note="canary probes green on the single-profile "
                        "worktree" if canary_ok else
                        "canary probe failure — automatic rollback "
                        "follows in the next record",
            ok=canary_ok, canary=str(canary), probes=probes)
        if not canary_ok:
            return self._fail_canary(state, canary, base_sha, files,
                                     probes)

        # Rung 10, gate half.
        assessment = assess_patch(candidate.diff_text)
        verdict = self.gate.decide(assessment)
        self._record(
            state, STRATEGY_GATE,
            STATUS_CANARY_PASSED if verdict.approved else STATUS_AWAITING,
            verify_note="policy gate: " + verdict.reason,
            decision="approved" if verdict.approved else "owner-required",
            by=verdict.by, reason=verdict.reason,
            assessment={k: assessment[k] for k in
                        ("file_count", "added", "deleted", "risks",
                         "consequential")})
        if not verdict.approved:
            return state
        return self._promote(state, by=verdict.by)

    def _apply_to_worktree(self, diff_text: str, root: Path) -> tuple:
        """check_patch + apply_patch inside a scratch worktree."""
        try:
            check = check_patch(diff_text, root=root)
            if not check.get("ok"):
                return False, "check_patch failed: " + json.dumps(
                    check.get("totals", {}), default=str)
            results = apply_patch(diff_text, dry_run=False, root=root)
        except Exception as exc:  # noqa: BLE001 - refused escapes etc.
            return False, type(exc).__name__ + ": " + str(exc)[:300]
        bad = [r for r in results if not r.get("ok")]
        if bad:
            return False, "apply failed for " + ", ".join(
                str(r.get("path", "?")) for r in bad)
        return True, "applied " + str(len(results)) + " file(s)"

    def _fail_canary(self, state: RunState, canary: Path, base_sha: str,
                     files: list, probes: list) -> RunState:
        """Automatic canary rollback: reset, verify, record, remove.

        The production tree was never written; the rollback restores
        the CANARY tree to the pre-patch commit and proves it.
        """
        ok, detail = self._restore_tree(canary, base_sha, files)
        self._record(
            state, STRATEGY_CANARY_ROLLBACK, STATUS_CANARY_ROLLED_BACK,
            verify_note="canary automatically rolled back after a "
                        "probe failure; production was never written. "
                        "Restore: " + detail,
            ok=ok, restore=detail,
            failed=[p.get("name", "") for p in probes if not p.get("ok")])
        self._worktree_remove(canary)
        return state

    # ── rung 10: promotion, approval, rejection, rollback ─────────

    def _require_trail(self, state: RunState) -> None:
        """The §9 audit guard, in code: promotion demands the run's
        OWN journal trail — clean proposal, fully green suite, green
        canary, all for this candidate hash — re-derived from the
        journal, never from caller-held state."""
        fresh = self._load_run(state.run_id)
        if fresh is None:
            raise RepairRefused("no journal trail for run " + state.run_id)
        proposed = fresh.detail_of(STRATEGY_PROPOSED)
        if not proposed.get("ok"):
            raise RepairRefused(
                "run " + state.run_id + " has no clean patch proposal "
                "on its ticket — refusing the production write")
        suite = [r for r in fresh.records
                 if r.get("strategy") == STRATEGY_SUITE]
        if not suite or not all(
                (r.get("detail") or {}).get("ok") for r in suite):
            raise RepairRefused(
                "run " + state.run_id + " has no fully green sandbox "
                "suite on its ticket — refusing the production write")
        canary = fresh.detail_of(STRATEGY_CANARY)
        if not canary.get("ok"):
            raise RepairRefused(
                "run " + state.run_id + " has no green canary on its "
                "ticket — refusing the production write")

    def _promote(self, state: RunState, *, by: str) -> RunState:
        self._require_trail(state)
        if self.snapshots is None:
            raise RepairRefused(
                "no snapshot manager wired: promotion without the "
                "retained snapshot rollback is refused")
        run_dir = self.work_root / state.run_id
        diff_path = run_dir / "candidate.diff"
        if not diff_path.exists():
            raise RepairRefused(
                "candidate artifact missing for run " + state.run_id)
        diff_text = diff_path.read_text(encoding="utf-8")
        if candidate_hash_of(diff_text) != state.candidate_hash:
            raise RepairRefused(
                "candidate artifact no longer matches the ticketed "
                "hash for run " + state.run_id)
        proposed = state.detail_of(STRATEGY_PROPOSED)
        files = list(proposed.get("files") or [])
        check = check_patch(diff_text, root=self.repo_dir)
        if not check.get("ok"):
            self._record(
                state, STRATEGY_GATE, STATUS_AWAITING,
                verify_note="promotion refused: the candidate no "
                            "longer applies cleanly to the current "
                            "tree (the tree moved under the parked "
                            "run); nothing was written",
                decision="refused-stale", by="pipeline")
            return state
        rc, out, _err = self._git(
            ["status", "--porcelain", "--"] + files, cwd=self.repo_dir)
        if rc == 0 and out.strip():
            self._record(
                state, STRATEGY_GATE, STATUS_AWAITING,
                verify_note="promotion refused: a file the candidate "
                            "touches is dirty in the production tree; "
                            "nothing was written",
                decision="refused-dirty", by="pipeline")
            return state

        snap = self.snapshots.create(label="repair " + state.run_id)
        snap_id = str(getattr(snap, "id", snap))
        from_sha = self._head_sha(self.repo_dir)
        # THE production write of the whole pipeline: reachable only
        # through _require_trail above, on a candidate the gate
        # approved, with the rollback snapshot already retained.
        results = apply_patch(diff_text, dry_run=False, root=self.repo_dir)
        bad = [r for r in results if not r.get("ok")]
        applied = not bad
        apply_note = "apply failed for " + ", ".join(
            str(r.get("path", "?")) for r in bad)
        if not applied:
            ok, detail = self._restore_tree(self.repo_dir, from_sha, files)
            self._record(
                state, STRATEGY_ROLLBACK, STATUS_PROMOTION_ROLLED_BACK,
                verify_note="promotion apply failed (" + apply_note[:200]
                            + "); tree restored: " + detail,
                phase="promotion", stage="apply", snapshot=snap_id,
                from_sha=from_sha, ok=ok, restore=detail, by=by)
            return state
        rc, _out, err = self._git(["add", "--"] + files, cwd=self.repo_dir)
        if rc == 0:
            msg = "repair " + state.run_id + ": " \
                + (state.detail_of(STRATEGY_PROPOSED).get("rationale")
                   or "pipeline promotion")[:120] \
                + "\n\nIncident: " + str(state.incident_id) \
                + "\nRun: " + state.run_id \
                + "\nCandidate: " + state.candidate_hash
            rc, _out, err = self._git(["commit", "-m", msg],
                                      cwd=self.repo_dir)
        if rc != 0:
            ok, detail = self._restore_tree(self.repo_dir, from_sha, files)
            self._record(
                state, STRATEGY_ROLLBACK, STATUS_PROMOTION_ROLLED_BACK,
                verify_note="promotion commit failed (" + err.strip()[:200]
                            + "); tree restored: " + detail,
                phase="promotion", stage="commit", snapshot=snap_id,
                from_sha=from_sha, ok=ok, restore=detail, by=by)
            return state
        to_sha = self._head_sha(self.repo_dir)

        monitor = self._health_results(self.repo_dir, self.db_path)
        if not all(m["ok"] for m in monitor):
            ok, detail = self._restore_tree(self.repo_dir, from_sha, files)
            restored, restore_note = self._restore_snapshot(snap_id)
            self._record(
                state, STRATEGY_ROLLBACK, STATUS_PROMOTION_ROLLED_BACK,
                verify_note="post-promotion monitor failed; promotion "
                            "automatically rolled back. Tree: " + detail
                            + " Snapshot: " + restore_note,
                phase="promotion", stage="monitor", snapshot=snap_id,
                snapshot_restored=restored, from_sha=from_sha,
                to_sha=to_sha, ok=ok, restore=detail,
                monitor=monitor, by=by)
            self._worktree_remove(run_dir / "canary")
            return state
        self._record(
            state, STRATEGY_PROMOTE, STATUS_PROMOTED, verified=True,
            verify_note="promoted through the gate; post-promotion "
                        "monitor green; snapshot " + snap_id
                        + " retained for rollback",
            snapshot=snap_id, from_sha=from_sha, to_sha=to_sha,
            files=files, by=by, monitor=monitor)
        self._worktree_remove(run_dir / "canary")
        return state

    def _restore_snapshot(self, snap_id: str) -> tuple:
        if self.snapshots is None:
            return False, "no snapshot manager wired"
        try:
            self.snapshots.restore(snap_id)
            return True, "snapshot " + snap_id + " restored"
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            return False, "snapshot restore raised " \
                + type(exc).__name__ + ": " + str(exc)[:200]

    def approve(self, run_id: str, *, by: str) -> RunState:
        """The owner's gate decision, arriving via a shipped surface."""
        state = self._load_run(run_id)
        if state is None:
            raise RepairRefused("unknown repair run " + run_id)
        if state.status != STATUS_AWAITING:
            raise RepairRefused(
                "run " + run_id + " is " + (state.status or "?")
                + ", not awaiting approval")
        if not by:
            raise RepairRefused("an approval must name who approved")
        self._record(
            state, STRATEGY_GATE, STATUS_AWAITING,
            verify_note="owner approved promotion of run " + run_id,
            decision="approved", by=by)
        return self._promote(state, by=by)

    def reject(self, run_id: str, *, by: str, reason: str = "") -> RunState:
        state = self._load_run(run_id)
        if state is None:
            raise RepairRefused("unknown repair run " + run_id)
        if state.status in _TERMINAL:
            raise RepairRefused(
                "run " + run_id + " is already " + state.status)
        self._record(
            state, STRATEGY_GATE, STATUS_REJECTED,
            verify_note="gate declined promotion; candidate discarded, "
                        "production never written",
            decision="declined", by=by, reason=reason[:400])
        self._worktree_remove(self.work_root / run_id / "canary")
        return state

    def rollback(self, run_id: str, *, by: str,
                 reason: str = "") -> RunState:
        """Exercise the retained rollback of a PROMOTED run."""
        state = self._load_run(run_id)
        if state is None:
            raise RepairRefused("unknown repair run " + run_id)
        if state.status != STATUS_PROMOTED:
            raise RepairRefused(
                "run " + run_id + " is " + (state.status or "?")
                + "; only a promoted run can be rolled back")
        promoted = state.detail_of(STRATEGY_PROMOTE)
        from_sha = str(promoted.get("from_sha", ""))
        to_sha = str(promoted.get("to_sha", ""))
        files = list(promoted.get("files") or [])
        if not from_sha or self._head_sha(self.repo_dir) != to_sha:
            raise RepairRefused(
                "HEAD has moved past the promotion commit of run "
                + run_id + " — a scoped rollback is no longer safe; "
                "roll back manually with the retained snapshot "
                + str(promoted.get("snapshot", "?")))
        ok, detail = self._restore_tree(self.repo_dir, from_sha, files)
        restored, restore_note = self._restore_snapshot(
            str(promoted.get("snapshot", "")))
        self._record(
            state, STRATEGY_ROLLBACK, STATUS_ROLLED_BACK,
            verify_note="promoted repair rolled back by " + by
                        + ". Tree: " + detail + " Snapshot: "
                        + restore_note,
            phase="post-promotion", snapshot=promoted.get("snapshot", ""),
            snapshot_restored=restored, from_sha=from_sha,
            to_sha=to_sha, ok=ok, restore=detail, by=by,
            reason=reason[:400])
        return state
