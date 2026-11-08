"""Transactional self-update: pull, migrate, health-check, or roll back.

:class:`UpdateManager` owns the whole update as one transaction:

1. take a pre-update snapshot (via :class:`SnapshotManager`),
2. ``git pull --ff-only`` in the repo (skippable),
3. run database migrations,
4. run post-update health checks (importable package, layering vs the
   grandfathered baseline, DB integrity),

then advisory reporting (the error scan — information only, never a
gate).  A pull that lands nothing (HEAD unchanged) short-circuits
right after the pull: no migrate, no health gate, no rollback, no
quarantine, no restart — an up-to-date tree has nothing to prove.

If *any* step fails, the pre-update snapshot is restored automatically and
the report says so.  The system is never left half-updated.  Before the
pull, the update QUIESCES the children that execute files from the tree
(the WhatsApp bridge processes — a running bridge locks its own program
file on Windows, and git cannot replace a locked file); a quiesce that
cannot stop a child aborts the update before anything is pulled.  The
pull itself is NOT all-or-nothing at the filesystem level: git writes
the new tree file by file, so a pull that dies mid-checkout leaves a
half-written tree behind even though HEAD never moved (the 2026-11-07
EC2 incident: the pull died on the locked bridge binary after writing
most of the tree, and the follow-up "nothing was applied" report was
false).  A failed pull therefore restores the working tree to the exact
status captured before it (the pull's own writes undone, the owner's
disjoint local edits never touched) and verifies the status matches
before reporting; a successful pull is verified against the landed
commit before anything runs on the new code.  A failed migration or
health check restores BOTH the state snapshot and the code tree (the
files the pull changed are put back and HEAD returns to the pre-update
commit).  One exception, by design: when the state restore
cannot swap the live database because the running process holds it open
(Windows WinError 32), the database half is deferred — a pending-restore
marker is written and applied by the next boot (see
:meth:`UpdateManager.recover_interrupted`) before the runtime serves on
the database — and the report carries a deferral note, never a rollback
failure.

A deferred restore is not immortal.  The marker records its provenance
(the run that wrote it and the version that run targeted), and a later
SUCCESSFUL update supersedes it: that update migrated and
health-checked the live database, adopting it as the base, so applying
the old snapshot afterwards would regress live data to serve a
rollback nobody is running anymore.  The successful run clears a
marker it did not write and says so in its report; recovery
independently discards a marker the journal shows superseded —
silently, because booting on the live database is the correct state,
not an incident.

Recovery also discards a marker whose premise is gone when the
journal cannot prove it — the case of an install updated by hand
(``git pull``), where no update ever completes in the journal.  The
test is the pairing the marker itself records: the restore exists
to pair the pre-update snapshot with the pre-update CODE, the
version the failed run started from.  When the code actually
running is provably not that version, the pairing the restore
would serve no longer exists, and applying the snapshot could only
regress the live database under code it does not belong to — so
the marker is discarded before any restore is attempted and the
owner is told with an informational notice (not a failure page:
the live database is the correct state, kept untouched).  When the
recorded facts cannot decide — the marker names no starting
version, or the running version cannot be read — recovery keeps
the old behaviour exactly: attempt the restore, and report the
outcome honestly.

The owner's own uncommitted changes are never silently destroyed: before
pulling, the update fetches and refuses up front when a locally modified
file collides with the incoming change set; disjoint local edits are left
untouched by the apply, by a rollback, and by crash recovery (the tree
restore only ever touches files the pull itself changed).

Crash recovery: every run appends to an update journal
(``<home>/update-journal.jsonl``).  :meth:`recover_interrupted` replays the
journal at boot — a run with no terminal event whose last step reached a
dangerous point (migrations begun/applied, health checks running) is
rolled back automatically; a run that died before that is marked aborted.
Failed versions are quarantined so a bad update is never retried blindly,
and a pull that lands on a quarantined version is undone again.
Quarantine only ever names a version an update itself pulled: an entry
naming the version already checked out when a run starts is stale (no
update installed it) and is voided at run start, with the report saying
so.

Health checks are plain callables ``() -> (ok: bool, detail: str)`` so tests
can inject a broken one to prove the rollback path.  Advisory checks use
the same shape but are contractually non-blocking: they are recorded in
the report as information and can never fail an update, trigger a
rollback, or quarantine a version.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import diskspace
from ..core.filelock import lock_exclusive_nb, unlock as _unlock_file
from ..core.logging_setup import get_logger
from .snapshots import SnapshotError, SnapshotManager, SnapshotRefused
from .verify_install import verify_install

__all__ = [
    "HealthCheckFn",
    "QuiesceFn",
    "UpdateReport",
    "UpdateManager",
    "default_health_checks",
    "default_advisory_checks",
]

_log = get_logger(__name__)

#: A health check returns (ok, detail).
HealthCheckFn = Callable[[], "tuple[bool, str]"]

#: A quiesce callable stops the children that execute files from the
#: repo tree (the WhatsApp bridge processes) so a pull can replace
#: those files; it returns (ok, detail) in the health-check shape.
#: ok=False ABORTS the update before the pull — the update never
#: pulls into a tree a running child still locks.
QuiesceFn = Callable[[], "tuple[bool, str]"]

JOURNAL_FILENAME = "update-journal.jsonl"
HISTORY_FILENAME = "update-history.jsonl"
QUARANTINE_FILENAME = "update-quarantine.json"
UPDATE_LOCK_FILENAME = "update.lock"
#: Written when a rollback's database restore is deferred because the
#: running process holds the live database open; consumed at boot by
#: :meth:`UpdateManager.apply_pending_restore` (via recover_interrupted).
PENDING_RESTORE_FILENAME = "pending-db-restore.json"
#: Written by the runtime just before it restarts into a successful
#: update; consumed at boot by
#: :meth:`UpdateManager.consume_restart_intent`, which is how the NEW
#: process — not the dying one — confirms the restart landed.
RESTART_INTENT_FILENAME = "update-restart-intent.json"

#: Steps at/after which a crash leaves the system in a state that must be
#: rolled back (migrations may have partially applied).
_DANGEROUS_STEPS = frozenset({"migrate", "health"})


@dataclass
class UpdateReport:
    """What the update did, step by step."""

    ok: bool = False
    rolled_back: bool = False
    pre_update_snapshot: str = ""
    #: Revisions the tree moved between this run ("" when unknown —
    #: no pull ran, or git could not say). Equal values mean the pull
    #: landed nothing: the tree was already up to date.
    from_sha: str = ""
    to_sha: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0
    #: Version/commit quarantined by this run ("" when none).
    quarantined: str = ""
    #: Why the rollback's database restore was deferred to the next
    #: boot ("" when it was not): the live database was locked by the
    #: running process, so the restore is pending, not failed.
    db_restore_deferred: str = ""

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append({"step": name, "ok": ok, "detail": detail})

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rolled_back": self.rolled_back,
            "pre_update_snapshot": self.pre_update_snapshot,
            "from_sha": self.from_sha,
            "to_sha": self.to_sha,
            "steps": self.steps,
            "error": self.error,
            "seconds": round(self.seconds, 3),
            "quarantined": self.quarantined,
            "db_restore_deferred": self.db_restore_deferred,
        }

    def render(self) -> str:
        """Plain-text update report."""
        lines = [f"update — {'✓ OK' if self.ok else '✗ FAILED'}"
                 f" ({self.seconds:.1f}s)"]
        if self.from_sha and self.to_sha:
            if self.from_sha == self.to_sha:
                lines.append(f"  at {self.to_sha[:12]} — already up to date")
            else:
                lines.append(f"  {self.from_sha[:12]} → {self.to_sha[:12]}")
        elif self.to_sha:
            lines.append(f"  at {self.to_sha[:12]}")
        for step in self.steps:
            mark = "✓" if step.get("ok") else "✗"
            lines.append(f"  {mark} {step.get('step')}")
            if step.get("detail"):
                lines.append(f"      {step['detail']}")
        if self.rolled_back:
            lines.append("  ↩ rolled back to pre-update snapshot"
                         f" {self.pre_update_snapshot}")
        if self.db_restore_deferred:
            lines.append("  ⏳ database restore deferred — "
                         f"{self.db_restore_deferred}; a pending-restore "
                         "marker was written and the restore applies at "
                         "the next boot, before the database opens")
        if self.quarantined:
            lines.append(f"  ⚠ version quarantined: {self.quarantined}")
        if self.error:
            lines.append(f"  error: {self.error}")
        return "\n".join(lines)


def _interp_tag() -> str:
    """Interpreter identity for health failure lines.

    A health failure on the owner's host must be diagnosable from the
    report alone — which Python, at which path, produced it.
    """
    return f"python {platform.python_version()} ({sys.executable})"


#: Errnos that mean "the file is held by another process", not "broken".
_LOCK_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EBUSY})


def _is_lock_error(exc: BaseException) -> bool:
    """True when a restore failed because the live database is held open.

    On Windows, replacing a file another process has open raises
    ``PermissionError`` (WinError 32); SQLite/locking layers surface
    EACCES/EBUSY, and the snapshot layer refuses with a lock-worded
    :class:`SnapshotRefused`.  All of those are the same fact — the
    running process owns the database right now — and none of them is
    a restore *failure*: the restore is deferred instead.
    """
    if isinstance(exc, PermissionError):
        return True
    if getattr(exc, "winerror", None) == 32:  # WinError 32: file in use
        return True
    if isinstance(exc, OSError) and exc.errno in _LOCK_ERRNOS:
        return True
    return isinstance(exc, SnapshotRefused) and "lock" in str(exc).lower()


def _load_tree_layering_module(tests_dir: Path):
    """Load the TARGET tree's tests/test_layering.py — never a stale
    process cache.

    The health gate judges the tree the update just pulled, but a
    plain ``import test_layering`` answers from ``sys.modules``:
    after a FAILED update the process keeps running (no restart on
    failure) with the failed tree's module cached, so a retry would
    judge the new tree by the OLD tree's KNOWN_VIOLATIONS baseline
    and scanner — the gate rolling back the very fix for the failure
    it just reported. So the tree's file is always loaded fresh and
    its baseline compared with the cached module's: when they agree,
    the cached module is returned untouched, which keeps the seam
    the layering pins inject through (a test double whose baseline
    matches the tree's is honored, scanner and all); when they
    disagree — or the cache holds another tree's module, or
    nothing — the fresh module takes the cache's place and is
    returned. A SyntaxError in the tree's file propagates with its
    filename, as the gate's report requires.
    """
    import importlib.util

    path = tests_dir / "test_layering.py"
    spec = importlib.util.spec_from_file_location("test_layering", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the layering baseline at {path}")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    cached = sys.modules.get("test_layering")
    if cached is not None and cached is not fresh:
        try:
            same_file = Path(
                str(getattr(cached, "__file__", ""))).resolve() \
                == path.resolve()
        except OSError:
            same_file = False
        if same_file and getattr(cached, "KNOWN_VIOLATIONS", None) \
                == getattr(fresh, "KNOWN_VIOLATIONS", None):
            return cached
    sys.modules["test_layering"] = fresh
    return fresh


def default_health_checks(
    repo_dir: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> list[HealthCheckFn]:
    """The standard post-update health gate.

    * the ``nomorals`` package imports and reports a version,
    * the layering check finds no upward imports in the tracked tree
      beyond the grandfathered baseline the test suite carries
      (``tests/test_layering.py``'s ``KNOWN_VIOLATIONS`` — the one
      shared set; subset semantics, so a repaired grandfathered
      violation never fails an update and a new one always does),
    * the live database passes ``PRAGMA integrity_check``.

    The error scan is NOT part of the gate: it is an advisory smell
    report (see :func:`default_advisory_checks`).  The healthy tree
    carries hundreds of standing findings, so gating on it made every
    update unpassable by construction — the 2026-10-23 EC2 incident,
    where a no-op update "failed" health, failed its rollback, and
    quarantined the healthy running version.

    Every failure detail names the full offending file path (when a
    file is implicated) and the interpreter that produced the verdict.
    """
    checks: list[HealthCheckFn] = []

    if repo_dir is not None:

        def _import() -> tuple[bool, str]:
            # Exercise the code IN THE TREE, in a fresh interpreter. This
            # process already imported the OLD nomorals before the pull,
            # so an in-process `import nomorals` would return the cached
            # (pre-update) package and the check would pass vacuously
            # even when the pulled code cannot import at all.
            env = dict(os.environ)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(repo_dir) + (
                os.pathsep + existing if existing else "")
            try:
                proc = subprocess.run(
                    [sys.executable, "-c",
                     "import nomorals;"
                     " from nomorals.version import __version__;"
                     " print(__version__)"],
                    cwd=str(repo_dir), env=env,
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=180,
                )
            except subprocess.TimeoutExpired:
                return False, ("import check timed out after 180s "
                               f"[{_interp_tag()}]")
            except Exception as exc:  # noqa: BLE001 - the check IS the report
                return False, (f"import check could not run: {exc} "
                               f"[{_interp_tag()}]")
            if proc.returncode != 0:
                tail = "\n".join(
                    (proc.stderr or "").strip().splitlines()[-3:])
                return False, (f"import failed in tree "
                               f"{Path(repo_dir) / 'nomorals'}: {tail} "
                               f"[{_interp_tag()}]")
            version = (proc.stdout or "").strip().splitlines()
            return True, (f"nomorals {version[-1] if version else '?'}"
                          " imports (fresh interpreter, tree code)")

    else:

        def _import() -> tuple[bool, str]:
            try:
                import nomorals  # noqa: F401

                from ..version import __version__

                return True, f"nomorals {__version__} imports"
            except Exception as exc:  # noqa: BLE001 - the check IS the report
                return False, f"import failed: {exc} [{_interp_tag()}]"

    checks.append(_import)

    if repo_dir is not None:

        def _layering() -> tuple[bool, str]:
            # Subset semantics over the ONE shared baseline:
            # tests/test_layering.py's KNOWN_VIOLATIONS is the
            # accepted grandfathered set, read by the test suite and
            # by this gate alike.  The update is healthy when every
            # violation found is in that set — a grandfathered pair
            # that a repair made disappear must NOT fail the update
            # (the test's stale-ratchet is a suite concern, not a
            # health verdict) — and unhealthy, naming them, on any
            # violation beyond the set.  Running the raw test method
            # here instead compared against the emptied set and
            # rolled back the healthy 2026-11-08 EC2 update.
            try:
                test_layering = _load_tree_layering_module(
                    Path(repo_dir) / "tests")

                new, known, _seen, _scanned = \
                    test_layering.scan_layering()
            except SyntaxError as exc:  # noqa: BLE001
                # test_layering parses with filename=<full path>, so
                # the report names the exact offending file — the EC2
                # incident report showed a bare "build.py", which is
                # undiagnosable from the report alone.
                return False, (f"layering failed: {exc.msg} in "
                               f"{exc.filename} (line {exc.lineno}) "
                               f"[{_interp_tag()}]")
            except Exception as exc:  # noqa: BLE001
                return False, (f"layering failed: {exc} "
                               f"[package root: {Path(repo_dir) / 'nomorals'}] "
                               f"[{_interp_tag()}]")
            if new:
                return False, (
                    f"layering failed: {len(new)} new violation(s) "
                    f"beyond the {len(test_layering.KNOWN_VIOLATIONS)} "
                    "grandfathered pairs:\n  " + "\n  ".join(new) +
                    f" [{_interp_tag()}]")
            return True, (f"layering ok: {len(known)} known "
                          "grandfathered, 0 new")

        checks.append(_layering)

    if db_path is not None:

        def _db() -> tuple[bool, str]:
            try:
                from ..storage.db import Database

                db = Database(str(db_path))
                try:
                    result = db.integrity_check()
                finally:
                    db.close()
                if str(result).lower() == "ok":
                    return True, "db integrity ok"
                return False, (f"db integrity_check: {result} "
                               f"[db: {db_path}] [{_interp_tag()}]")
            except Exception as exc:  # noqa: BLE001
                return False, (f"db check failed: {exc} "
                               f"[db: {db_path}] [{_interp_tag()}]")

        checks.append(_db)

    return checks


def default_advisory_checks(
    repo_dir: str | os.PathLike[str] | None = None,
) -> list[HealthCheckFn]:
    """Advisory post-update reporting — NEVER part of the health gate.

    The error scan runs over the tracked tree and its counts land in
    the update report as information.  These callables always report
    ok: an advisory finding can never fail an update, trigger a
    rollback, or quarantine a version.
    """
    checks: list[HealthCheckFn] = []

    if repo_dir is not None:

        def _error_scan() -> tuple[bool, str]:
            try:
                from ..tools.error_scan import scan

                report = scan([str(Path(repo_dir) / "nomorals")],
                              tracked_only=True)
                errors = [f for f in report.findings
                          if f.severity == "error"]
                warnings = len(report.findings) - len(errors)
                return True, (
                    f"error_scan (advisory, non-blocking): {len(errors)} "
                    f"error(s), {warnings} warning(s) across "
                    f"{report.files_scanned} tracked file(s) — findings "
                    "never gate an update")
            except Exception as exc:  # noqa: BLE001 - advisory is a report
                return True, (f"error_scan (advisory, non-blocking): "
                              f"could not run: {exc}")

        checks.append(_error_scan)

    return checks


class UpdateManager:
    """Run a transactional update of the Devon installation."""

    def __init__(
        self,
        repo_dir: str | os.PathLike[str],
        snapshots: SnapshotManager,
        *,
        health_checks: list[HealthCheckFn] | None = None,
        advisory_checks: list[HealthCheckFn] | None = None,
        git_bin: str = "git",
        verifier: Any | None = None,
        quiesce: QuiesceFn | None = None,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.snapshots = snapshots
        self.git_bin = git_bin
        # Quiesce (update hardening): the callable that stops the
        # repo-file children (the WhatsApp bridge processes) before
        # the pull replaces the files they execute. The chat runtime
        # wires its bridge supervisor in; a construction without one
        # (the console update, tests) runs no such children of its
        # own and the quiesce step records exactly that — it never
        # pretends a stop happened.
        self._quiesce = quiesce
        # Verify-then-install (Wave 8): after a pull moves HEAD, the
        # landed state is verified before the update proceeds. The
        # default follows the health-gate idiom above: the production
        # construction (no injected health checks) gets the real
        # verifier; an injected construction's gates are scripted
        # wholes (their git seams are stubbed, so a real-repo verifier
        # could only lie about them) and gets one only when it passes
        # a verifier explicitly.
        if verifier is not None:
            self._verifier = verifier
        elif health_checks is None:
            self._verifier = verify_install
        else:
            self._verifier = None
        if health_checks is not None:
            self.health_checks = list(health_checks)
            # Injected gates (tests, embedders) get no advisory checks
            # unless they ask: advisory defaults belong to the
            # production construction only.
            self.advisory_checks = (
                list(advisory_checks) if advisory_checks is not None else [])
        else:
            self.health_checks = default_health_checks(
                repo_dir, snapshots.db_path)
            self.advisory_checks = (
                list(advisory_checks) if advisory_checks is not None
                else default_advisory_checks(repo_dir))
        home = Path(snapshots.home)
        self.journal_path = home / JOURNAL_FILENAME
        self.history_path = home / HISTORY_FILENAME
        self.quarantine_path = home / QUARANTINE_FILENAME
        self.lock_path = home / UPDATE_LOCK_FILENAME
        self.pending_restore_path = home / PENDING_RESTORE_FILENAME
        self.restart_intent_path = home / RESTART_INTENT_FILENAME
        #: Code-move bookkeeping for the transaction in flight ("" when
        #: this run did not move the tree): the commit to restore on
        #: rollback and the commit the pull landed on.
        self._tx_from_sha = ""
        self._tx_to_sha = ""
        #: True when THIS run's rollback wrote the pending-restore
        #: marker. A successful run clears a stale marker it did not
        #: write; its own marker is never cleared (a run that rolled
        #: back never reaches the success path, and this flag keeps
        #: that true by construction).
        self._tx_wrote_pending_restore = False

    # ── public ─────────────────────────────────────────────────────────────

    def run(self, *, pull: bool = True) -> UpdateReport:
        """Execute the update transaction. Never raises on update failure.

        One update at a time per install home: a second concurrent run
        is refused (two transactions interleaving pulls, migrations and
        rollbacks over one tree/journal cannot both be honest).
        """
        lock_fd = self._acquire_update_lock()
        if lock_fd is None:
            report = UpdateReport()
            report.error = ("another update is already running over this "
                            "install — refusing to run two updates at once")
            report.record("lock", False, report.error)
            self._append_history(report)
            return report
        try:
            return self._run(pull=pull)
        finally:
            self._release_update_lock(lock_fd)

    def _acquire_update_lock(self) -> int | None:
        """Take the per-home update lock (non-blocking). None = held."""
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            _log.debug("could not open update lock", exc_info=True)
            return None
        if lock_exclusive_nb(fd):
            return fd
        os.close(fd)
        return None

    @staticmethod
    def _release_update_lock(fd: int) -> None:
        try:
            _unlock_file(fd)
        finally:
            os.close(fd)

    def _run(self, *, pull: bool = True) -> UpdateReport:
        started = time.time()
        report = UpdateReport()
        run_id = uuid.uuid4().hex[:12]
        self._journal({"run_id": run_id, "event": "start", "ts": time.time()})

        # Step 0: snapshot of the current tree sha (for post-pull checks).
        previous_sha = self._git_sha() if pull else ""
        self._tx_from_sha = previous_sha
        self._tx_to_sha = ""
        self._tx_wrote_pending_restore = False
        report.from_sha = previous_sha

        # Step 0.25: quarantine truth. An update may only quarantine a
        # version it pulled itself, so an entry naming the version
        # ALREADY checked out predates this run and is void when no
        # journal record shows an update ever pulled that version
        # (the EC2 incident: a no-op update's failing health gate
        # quarantined the healthy running version). Void it here —
        # before the pull, so the quarantined-pull check below never
        # "undoes" a pull that landed on the running version — and say
        # so in the report.
        head_now = previous_sha or self._git_sha()
        if head_now and self._is_quarantined(head_now) \
                and self._quarantine_is_stale(head_now):
            self._quarantine_void(head_now)
            report.record(
                "quarantine", True,
                f"voided a stale quarantine entry for the running "
                f"version {head_now[:12]} — no update ever pulled it, "
                "so no update may quarantine it")
            self._journal({"run_id": run_id, "event": "quarantine_voided",
                           "ts": time.time(), "sha": head_now})

        # Step 0.5: preflight — never destroy the owner's uncommitted
        # work. A locally modified file that the incoming update also
        # changes refuses the update up front (nothing is pulled);
        # disjoint local edits are fine: neither the pull nor a later
        # tree restore touches them.
        if pull:
            proceed, detail = self._preflight_pull()
            if not proceed:
                report.record("working_tree", False, detail)
                report.error = detail
                report.seconds = time.time() - started
                self._journal({"run_id": run_id, "event": "aborted",
                               "ts": time.time(), "error": report.error})
                self._append_history(report)
                return report
            if detail:
                report.record("working_tree", True, detail)

        # Step 0.6: disk preflight — an update first takes a full
        # pre-update snapshot (database + WAL + the blob store,
        # copied wholesale) and then rewrites the tree. On a disk
        # without room for that, step 1 is where the update used
        # to discover the problem by failing (2026-11-08 EC2:
        # "database or disk is full" at the snapshot step, after
        # snapshots had piled up unbounded for weeks). Measure
        # FIRST and refuse here — before the snapshot, before the
        # pull, and before the quiesce: a refused update changes
        # nothing on disk and stops nothing that is running.
        d_ok, d_detail = self._preflight_disk()
        report.record("disk_space", d_ok, d_detail)
        self._journal({"run_id": run_id, "event": "step",
                       "step": "disk_space", "ts": time.time(),
                       "ok": d_ok, "detail": d_detail})
        if not d_ok:
            report.error = d_detail
            report.seconds = time.time() - started
            self._journal({"run_id": run_id, "event": "aborted",
                           "ts": time.time(), "error": report.error})
            self._append_history(report)
            return report

        # Step 0.75: quiesce — stop the children that execute files
        # from the tree BEFORE anything tries to replace those files.
        # The WhatsApp bridge is the proven locker (2026-11-07 EC2:
        # git could not unlink the running bridge binary, died
        # mid-checkout, and left the tree half-written). A quiesce
        # that cannot stop a child aborts the update HERE, before
        # the snapshot and the pull: pulling into a locked tree is
        # how a tree gets half-applied. (After preflight on purpose:
        # an update preflight refuses must not stop the owner's
        # bridges on its way out.)
        if pull:
            q_ok, q_detail = self._run_quiesce()
            report.record("quiesce", q_ok, q_detail)
            self._journal({"run_id": run_id, "event": "step",
                           "step": "quiesce", "ts": time.time(),
                           "ok": q_ok, "detail": q_detail})
            if not q_ok:
                report.error = ("update stopped before pulling — "
                                + q_detail)
                report.seconds = time.time() - started
                self._journal({"run_id": run_id, "event": "aborted",
                               "ts": time.time(), "error": report.error})
                self._append_history(report)
                return report

        # Step 1: pre-update snapshot — the rollback anchor.
        try:
            snap = self.snapshots.create(label="pre-update")
            report.pre_update_snapshot = snap.id
            report.record("snapshot", True, f"pre-update snapshot {snap.id}")
            self._journal({"run_id": run_id, "event": "step", "step": "snapshot",
                           "ts": time.time(), "snapshot": snap.id})
            # Retention (disk discipline, 2026-11-08): the pile of
            # pre-update snapshots is bounded the moment a new one
            # lands — newest 3, plus the snapshot a pending
            # deferred restore names (a rollback anchor, not
            # clutter). Pruning failures journal here and never
            # fail the update.
            try:
                protect: list[str] = []
                pending = self._read_pending_restore()
                if pending and pending.get("snapshot"):
                    protect.append(str(pending["snapshot"]))
                pruned = self.snapshots.prune_pre_update(
                    protect=protect)
                if pruned:
                    self._journal({"run_id": run_id, "event": "retention",
                                   "ts": time.time(), "pruned": pruned})
            except Exception as exc:  # noqa: BLE001 - never fails the update
                self._journal({"run_id": run_id,
                               "event": "retention_failed",
                               "ts": time.time(),
                               "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # noqa: BLE001 - snapshot failure aborts
            report.record("snapshot", False, f"{type(exc).__name__}: {exc}")
            report.error = f"could not take pre-update snapshot: {exc}"
            report.seconds = time.time() - started
            self._journal({"run_id": run_id, "event": "aborted",
                           "ts": time.time(), "error": report.error})
            self._append_history(report)
            return report

        # Step 2: pull. --ff-only decides the MERGE atomically, but
        # its checkout is file-by-file: a pull that dies mid-checkout
        # (the locked bridge binary, 2026-11-07) leaves files written,
        # deleted and created behind even though HEAD never moved.
        # So: capture the working-tree status first, journal the
        # pull's BEGIN entry before it runs (the migrate-begin
        # reasoning — a crash inside the pull must be recoverable as
        # "died in the pull", not misread as "died before anything
        # changed"), and on failure restore the tree to exactly the
        # captured status before reporting.
        new_sha = ""
        if pull:
            pre_tracked, pre_untracked = self._status_paths()
            self._journal({"run_id": run_id, "event": "step",
                           "step": "git_pull", "phase": "begin",
                           "ts": time.time(), "from_sha": previous_sha,
                           "pre_dirty": sorted(pre_tracked),
                           "pre_untracked": sorted(pre_untracked)})
            ok, detail = self._git_pull()
            report.record("git_pull", ok, detail)
            if not ok:
                self._journal({"run_id": run_id, "event": "step",
                               "step": "git_pull", "ts": time.time(),
                               "ok": ok, "detail": detail})
                restored, restore_detail = (
                    self._restore_tree_after_failed_pull(
                        previous_sha, pre_tracked, pre_untracked))
                report.record("git_restore", restored, restore_detail)
                self._journal({"run_id": run_id, "event": "step",
                               "step": "git_restore", "ts": time.time(),
                               "ok": restored, "detail": restore_detail})
                if restored:
                    report.error = (
                        f"git pull failed: {detail} Nothing was "
                        f"applied: {restore_detail} — the tree is "
                        "exactly as it was before the update and "
                        "your local changes are untouched.")
                else:
                    report.error = (
                        f"git pull failed: {detail} The working tree "
                        f"could NOT be put back ({restore_detail}) — "
                        "do not restart until it is fixed; "
                        f"`git -C {self.repo_dir} status` shows what "
                        "still differs from the pre-update state.")
                report.seconds = time.time() - started
                self._journal({"run_id": run_id, "event": "aborted",
                               "ts": time.time(), "error": report.error})
                self._append_history(report)
                return report
            new_sha = self._git_sha()
            self._tx_to_sha = new_sha
            report.to_sha = new_sha
            self._journal({"run_id": run_id, "event": "step", "step": "git_pull",
                           "ts": time.time(), "ok": ok, "detail": detail,
                           "from_sha": previous_sha, "to_sha": new_sha})
            if new_sha and self._is_quarantined(new_sha):
                # Undo the pull: the tree must not be left sitting on a
                # version already known to fail — any restart would run
                # it. The restore only touches files the pull changed.
                undone, undo_detail = (
                    False, "tree was already at the quarantined commit "
                    "before this run; nothing to undo")
                if previous_sha and new_sha != previous_sha:
                    undone, undo_detail = self._restore_tree(previous_sha)
                report.record("git_restore", undone, undo_detail)
                report.error = (
                    f"pulled version {new_sha[:12]} is quarantined"
                    " (it failed a previous update) — refusing to apply it."
                    + (f" The pull was undone; the tree is back at"
                       f" {previous_sha[:12]}." if undone else
                       f" Undo the pull with: git -C {self.repo_dir} reset"
                       f" --hard {previous_sha[:12]}" if previous_sha else "")
                    + " Clear the quarantine explicitly to retry.")
                report.seconds = time.time() - started
                self._journal({"run_id": run_id, "event": "aborted",
                               "ts": time.time(), "error": report.error})
                self._append_history(report)
                return report
        else:
            report.record("git_pull", True, "skipped (--no-pull)")
            self._journal({"run_id": run_id, "event": "step", "step": "git_pull",
                           "ts": time.time(), "ok": True,
                           "detail": "skipped (--no-pull)"})

        # Step 2.25: verify-then-install. The pull claims it landed
        # new_sha; a transport's success line is not proof (the
        # push_nm20 lesson), so grep the resulting state — HEAD,
        # ancestry, object presence, worktree content, new-blob
        # sizes — before anything else runs on the new code. A failed
        # verification undoes the pull exactly like the quarantined
        # pull above: the tree must not sit on unverified code.
        if pull and new_sha and previous_sha and new_sha != previous_sha \
                and self._verifier is not None:
            verdict = self._verifier(self.repo_dir, new_sha, previous_sha,
                                     git_bin=self.git_bin)
            report.record("verify_install", bool(verdict.get("ok")),
                          str(verdict.get("detail", "")))
            self._journal({"run_id": run_id, "event": "step",
                           "step": "verify_install", "ts": time.time(),
                           "ok": bool(verdict.get("ok")),
                           "detail": str(verdict.get("detail", ""))})
            if not verdict.get("ok"):
                undone, undo_detail = self._restore_tree(previous_sha)
                report.record("git_restore", undone, undo_detail)
                report.error = (
                    f"pulled version {new_sha[:12]} failed install "
                    f"verification ({verdict.get('detail', '')}) — "
                    "refusing to run unverified code."
                    + (f" The pull was undone; the tree is back at"
                       f" {previous_sha[:12]}." if undone else
                       f" Undo the pull with: git -C {self.repo_dir} "
                       f"reset --hard {previous_sha[:12]}"))
                report.seconds = time.time() - started
                self._journal({"run_id": run_id, "event": "aborted",
                               "ts": time.time(), "error": report.error})
                self._append_history(report)
                return report

        # Step 2.3: the working tree must BE the pulled commit for
        # everything the pull changed. A transport's exit code is
        # not proof of the landed state (the push_nm20 lesson), and
        # this check runs for EVERY construction — scripted-gate
        # constructions skip the deep verifier above, but no update
        # proceeds on a tree that merely CLAIMS to be the new commit.
        # Scoped to the incoming change set: the owner's disjoint
        # local edits are supposed to be dirty and never fail this.
        # A mismatch undoes the pull through the same tree restore.
        if pull and new_sha and previous_sha and new_sha != previous_sha:
            tree_ok, tree_detail = self._verify_tree_matches(
                previous_sha, new_sha)
            report.record("verify_tree", tree_ok, tree_detail)
            self._journal({"run_id": run_id, "event": "step",
                           "step": "verify_tree", "ts": time.time(),
                           "ok": tree_ok, "detail": tree_detail})
            if not tree_ok:
                undone, undo_detail = self._restore_tree(previous_sha)
                report.record("git_restore", undone, undo_detail)
                report.error = (
                    f"pulled version {new_sha[:12]} failed the "
                    f"working-tree check ({tree_detail}) — refusing "
                    "to run a tree that is not the pulled commit."
                    + (f" The pull was undone; the tree is back at"
                       f" {previous_sha[:12]}." if undone else
                       f" Undo the pull with: git -C {self.repo_dir} "
                       f"reset --hard {previous_sha[:12]}"))
                report.seconds = time.time() - started
                self._journal({"run_id": run_id, "event": "aborted",
                               "ts": time.time(), "error": report.error})
                self._append_history(report)
                return report

        # Step 2.5: no-op short-circuit. The pull landed nothing —
        # HEAD is exactly where it started. Migrating, health-gating,
        # rolling back or quarantining here is precisely how a healthy
        # running version got quarantined in the EC2 incident: there is
        # no new code to prove, so the update reports "already up to
        # date" and STOPS (the chat layer sees from_sha == to_sha and
        # does not restart either).
        if pull and new_sha and previous_sha and new_sha == previous_sha:
            report.record(
                "up_to_date", True,
                f"already up to date at {new_sha[:12]} — no migration, "
                "health gate, rollback or restart needed")
            report.ok = True
            report.seconds = time.time() - started
            self._journal({"run_id": run_id, "event": "complete",
                           "ts": time.time(), "sha": new_sha, "noop": True})
            self._append_history(report)
            _log.info("update: already up to date at %s", new_sha[:12])
            return report

        # Step 3: migrations. The begin entry is journaled BEFORE the
        # step runs: a crash inside the migration itself is the most
        # dangerous interruption there is (schema half-applied), and
        # crash recovery can only know that if the journal says the
        # step was entered — journaling only the outcome would mark a
        # mid-migration death as "died before dangerous steps".
        self._journal({"run_id": run_id, "event": "step", "step": "migrate",
                       "ts": time.time(), "phase": "begin"})
        try:
            applied, version = self._migrate()
            report.record(
                "migrate", True,
                f"schema v{version}" + (f" (+{len(applied)} applied)" if applied else ""),
            )
            self._journal({"run_id": run_id, "event": "step", "step": "migrate",
                           "ts": time.time(), "ok": True,
                           "schema_version": version,
                           "applied": list(applied)})
        except Exception as exc:  # noqa: BLE001 - migrate failure rolls back
            report.record("migrate", False, f"{type(exc).__name__}: {exc}")
            self._journal({"run_id": run_id, "event": "step", "step": "migrate",
                           "ts": time.time(), "ok": False,
                           "error": f"{type(exc).__name__}: {exc}"})
            report = self._rollback(report, started, f"migration failed: {exc}",
                                    run_id=run_id, quarantine_sha=new_sha)
            self._append_history(report)
            return report

        # Step 4: health checks. ANY failure rolls back.
        failed: list[str] = []
        for check in self.health_checks:
            name = getattr(check, "__name__", "health_check")
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001 - a raising check is a failure
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            report.record(f"health:{name}", ok, detail)
            self._journal({"run_id": run_id, "event": "step",
                           "step": f"health:{name}", "ts": time.time(),
                           "ok": ok, "detail": detail})
            if not ok:
                failed.append(f"{name}: {detail}")
        if failed:
            report = self._rollback(
                report, started, "health check failed: " + "; ".join(failed),
                run_id=run_id, quarantine_sha=new_sha)
            self._append_history(report)
            return report

        # Advisory reporting runs only once the update itself is
        # proven: it informs the report, it can never change the
        # outcome (see _run_advisory).
        self._run_advisory(report, run_id=run_id)

        # This run just migrated and health-checked the LIVE database
        # and adopted it as the base — so a pending-restore marker an
        # EARLIER failed update left behind is now stale: applying it
        # at a later boot would regress the live database to that
        # update's pre-update snapshot to serve a rollback this update
        # superseded (the 2026-10-23 EC2 sequence: update A deferred
        # its restore, Devon ran for hours, update B succeeded, and
        # B's boot tried to apply A's restore and paged a failure
        # whose premise — rolled-back code — no longer existed).
        # Clear the stale marker and say so. A marker THIS run wrote
        # is untouched (see _clear_stale_pending_restore).
        cleared = self._clear_stale_pending_restore()
        if cleared is not None:
            who = (cleared.get("target_sha") or cleared.get("run_id")
                   or cleared.get("snapshot") or "an earlier update")
            report.record(
                "pending_restore", True,
                f"cleared stale deferred database restore from "
                f"superseded update {str(who)[:12]} — this update "
                "migrated and health-checked the live database, so "
                "the deferred snapshot restore no longer applies")
            self._journal({"run_id": run_id,
                           "event": "pending_restore_cleared",
                           "ts": time.time(),
                           "cleared_snapshot": cleared.get("snapshot", ""),
                           "cleared_run_id": cleared.get("run_id", ""),
                           "cleared_target_sha":
                               cleared.get("target_sha", ""),
                           "cleared_from_sha": cleared.get("from_sha", "")})

        report.ok = True
        report.seconds = time.time() - started
        self._journal({"run_id": run_id, "event": "complete",
                       "ts": time.time(), "sha": new_sha})
        self._append_history(report)
        _log.info("update completed cleanly")
        return report

    def _run_advisory(self, report: UpdateReport, *,
                      run_id: str = "") -> None:
        """Run the advisory checks into the report. NEVER gates.

        Every advisory step is recorded ok=True whatever the check
        returns or raises: advisory output is information for the
        owner (e.g. the error scan's standing finding counts), and the
        moment it could fail an update we are back to the EC2 incident
        — an unpassable gate wired to a rollback and a quarantine.
        """
        for check in self.advisory_checks:
            name = getattr(check, "__name__", "advisory_check")
            try:
                _ok, detail = check()
            except Exception as exc:  # noqa: BLE001 - advisory is a report
                detail = (f"advisory check raised {type(exc).__name__}: "
                          f"{exc} (non-blocking)")
            report.record(f"advisory:{name}", True, detail)
            if run_id:
                self._journal({"run_id": run_id, "event": "step",
                               "step": f"advisory:{name}",
                               "ts": time.time(), "ok": True,
                               "detail": detail})

    def dry_run(self) -> UpdateReport:
        """Run the health checks without pulling, migrating, or snapshotting.

        Answers "would this update survive the health gate?" safely.
        """
        started = time.time()
        report = UpdateReport()
        report.record("snapshot", True, "skipped (dry run)")
        report.record("git_pull", True, "skipped (dry run)")
        report.record("migrate", True, "skipped (dry run)")
        failed: list[str] = []
        for check in self.health_checks:
            name = getattr(check, "__name__", "health_check")
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            report.record(f"health:{name}", ok, detail)
            if not ok:
                failed.append(f"{name}: {detail}")
        self._run_advisory(report)
        report.ok = not failed
        if failed:
            report.error = "dry run health gate failed: " + "; ".join(failed)
        report.seconds = time.time() - started
        return report

    # ── crash recovery ─────────────────────────────────────────────────────
    def recover_interrupted(self) -> dict[str, Any]:
        """Replay the journal at boot and recover interrupted updates.

        A run with no terminal event (complete / rolled_back / aborted):
        * last step at/after a dangerous point (migrate, health) → roll
          back to its pre-update snapshot automatically;
        * died earlier → mark aborted (nothing user-visible changed).

        Returns a receipt dict describing what was done.
        """
        # A deferred database restore is applied FIRST: boot is the
        # one moment the database is not yet being served on, which is
        # exactly what the deferral was waiting for. Its receipt stands
        # on its own when the journal itself needs no recovery.
        pending = self.apply_pending_restore()
        pending_receipt: dict[str, Any] | None = None
        if pending.get("applied"):
            pending_receipt = {
                "recovered": True, "action": "pending_restore_applied",
                "restored_snapshot": pending["snapshot"],
                "snapshot": pending["snapshot"], "run_id": ""}
        elif pending.get("error"):
            pending_receipt = {
                "recovered": True, "action": "pending_restore_failed",
                "error": pending["error"],
                "snapshot": pending.get("snapshot", ""), "run_id": "",
                # The facts the failure notice must state honestly:
                # which update the marker belongs to, which version is
                # actually running, and how the two pair up.
                "update_sha": pending.get("update_sha", ""),
                "running_sha": pending.get("running_sha", ""),
                "state_note": pending.get("state_note", "")}
        elif pending.get("discarded_premise"):
            # A premise-gone marker was discarded before any restore
            # was attempted. The receipt stands on its own so the
            # boot notice fires: information, not a failure — the
            # live database was kept, exactly as it is.
            pending_receipt = {
                "recovered": True,
                "action": "pending_restore_discarded",
                "snapshot": pending.get("snapshot", ""), "run_id": "",
                "update_sha": pending.get("update_sha", ""),
                "from_sha": pending.get("from_sha", ""),
                "running_sha": pending.get("running_sha", ""),
                "state_note": pending.get("state_note", "")}
        elif pending.get("discarded_snapshot_missing"):
            # The marker's snapshot was deleted after the restore
            # was deferred: the restore can never be applied, so the
            # marker was discarded. The receipt stands on its own so
            # the boot notice fires — the live database was kept,
            # but the rollback's database half is gone for good,
            # which the owner should hear once, plainly.
            pending_receipt = {
                "recovered": True,
                "action": "pending_restore_unfulfillable",
                "snapshot": pending.get("snapshot", ""), "run_id": "",
                "update_sha": pending.get("update_sha", ""),
                "from_sha": pending.get("from_sha", ""),
                "running_sha": pending.get("running_sha", ""),
                "state_note": pending.get("state_note", "")}
        runs = self._journal_runs()
        if not runs:
            return pending_receipt or {"recovered": False,
                                       "reason": "no journal entries"}
        last = runs[-1]
        events = last["events"]
        # "pending_restore_applied" is journaled under a synthetic
        # "boot" run by apply_pending_restore — a complete record in
        # itself, never an interrupted update to recover. The stale-
        # discard entry ("pending_restore_discarded_stale") is the
        # same shape: the marker is gone, the boot is clean, and the
        # discard must not surface as an "aborted" recovery.
        terminal = next((e for e in events
                         if e.get("event") in ("complete", "rolled_back",
                                               "aborted", "crash_recovery",
                                               "pending_restore_applied",
                                               "pending_restore_discarded_stale")),
                        None)
        if terminal is not None:
            return pending_receipt or {
                "recovered": False, "reason": "last run terminated cleanly",
                "run_id": last["run_id"]}
        steps = [str(e.get("step", "")) for e in events
                 if e.get("event") == "step"]
        # "health:<name>" steps belong to the health phase; a bare
        # prefix test keeps them dangerous without mangling the names.
        dangerous = any(
            s in _DANGEROUS_STEPS or s.split(":", 1)[0] in _DANGEROUS_STEPS
            for s in steps)
        snap_id = next((e.get("snapshot") for e in events
                        if e.get("event") == "step"
                        and e.get("step") == "snapshot"), "")
        from_sha = next((str(e.get("from_sha", "")) for e in events
                         if e.get("event") == "step"
                         and e.get("step") == "git_pull"), "")
        # A run that journaled the pull's BEGIN entry but never its
        # completion died INSIDE the pull: migrations never began
        # (no state to restore), but the checkout may have
        # half-written the tree, so the "died before anything
        # changed" verdict would be false for it. The begin entry
        # carries the pre-pull status the tree restore needs.
        pull_begin = next((e for e in events
                           if e.get("event") == "step"
                           and e.get("step") == "git_pull"
                           and e.get("phase") == "begin"), None)
        pull_done = any(e.get("event") == "step"
                        and e.get("step") == "git_pull"
                        and "phase" not in e for e in events)
        receipt: dict[str, Any] = {
            "recovered": True, "run_id": last["run_id"],
            "last_steps": steps, "snapshot": snap_id,
        }
        if pending.get("applied") or pending.get("error") \
                or pending.get("discarded_stale"):
            receipt["pending_restore"] = pending
        if snap_id and dangerous:
            try:
                self.snapshots.restore(snap_id, force=True)
                receipt["action"] = "rolled_back"
                receipt["restored_snapshot"] = snap_id
                _log.warning("crash recovery: rolled back interrupted update"
                             " %s to %s", last["run_id"], snap_id)
                if from_sha:
                    tree_ok, tree_detail = self._restore_tree(from_sha)
                    receipt["code_restored"] = tree_ok
                    if not tree_ok:
                        receipt["action"] = "rolled_back_state_only"
                        receipt["code_restore_error"] = tree_detail
                        _log.error("crash recovery tree restore failed: %s",
                                   tree_detail)
            except Exception as exc:  # noqa: BLE001 — recovery must report
                receipt["action"] = "rollback_failed"
                receipt["error"] = f"{type(exc).__name__}: {exc}"
                _log.error("crash recovery rollback failed: %s", exc,
                           exc_info=True)
        elif snap_id and from_sha and pull_begin is not None \
                and not pull_done:
            # Died inside the pull — tree only, from the begin
            # entry's journaled pre-pull status (see above).
            pre_dirty = {str(p) for p in pull_begin.get("pre_dirty") or []}
            pre_untracked = {str(p)
                             for p in pull_begin.get("pre_untracked") or []}
            tree_ok, tree_detail = self._restore_tree_after_failed_pull(
                from_sha, pre_dirty, pre_untracked)
            receipt["action"] = ("tree_restored" if tree_ok
                                  else "tree_restore_failed")
            receipt["tree_restore"] = tree_detail
            if tree_ok:
                _log.warning("crash recovery: interrupted update %s "
                             "died inside git pull; working tree "
                             "restored to its pre-update state",
                             last["run_id"])
            else:
                receipt["error"] = tree_detail
                _log.error("crash recovery: interrupted update %s "
                           "died inside git pull and the tree "
                           "restore failed: %s", last["run_id"],
                           tree_detail)
        else:
            receipt["action"] = "aborted"
            _log.warning("crash recovery: marked interrupted update %s"
                         " aborted (died before dangerous steps)",
                         last["run_id"])
        self._journal({"run_id": last["run_id"], "event": "crash_recovery",
                       "ts": time.time(), **{k: v for k, v in receipt.items()
                                             if k != "recovered"}})
        return receipt

    # ── deferred database restore ──────────────────────────────────────
    def _write_pending_restore(self, snapshot_id: str, reason: str, *,
                               run_id: str = "", target_sha: str = "",
                               from_sha: str = "") -> None:
        """Record a database restore deferred to the next boot.

        Written by :meth:`_rollback` when the live database cannot be
        swapped from inside the process holding it open.  The marker
        is the whole contract: the snapshot to restore, why it was
        deferred, and its PROVENANCE — the run that wrote it, the
        version that run targeted, and the version it started from.
        Provenance is what lets a later successful update (and boot
        recovery) recognise this marker as superseded instead of
        applying a restore that no longer belongs to the install's
        history.
        """
        self._tx_wrote_pending_restore = True
        payload = {"snapshot": snapshot_id, "reason": reason,
                   "ts": time.time(), "run_id": run_id,
                   "target_sha": target_sha, "from_sha": from_sha}
        try:
            self.pending_restore_path.parent.mkdir(parents=True,
                                                   exist_ok=True)
            self.pending_restore_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:  # noqa: BLE001 — the deferral note still reports
            _log.error("could not write pending-restore marker at %s",
                       self.pending_restore_path, exc_info=True)

    def _read_pending_restore(self) -> dict[str, Any] | None:
        """The pending-restore marker payload — None when no marker
        file exists, an empty dict when the file exists but is
        malformed (such a marker can never be applied, but its
        presence still matters to the successful-update cleanup)."""
        try:
            raw = self.pending_restore_path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            payload = json.loads(raw)
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _clear_stale_pending_restore(self) -> dict[str, Any] | None:
        """Remove a pending-restore marker this run superseded.

        Called only from the success path of :meth:`_run`: the run
        migrated and health-checked the live database, so a deferred
        restore left by an earlier failed update no longer applies.
        Returns the cleared marker's provenance for the report and
        the journal, or None when there was nothing to clear — no
        marker, or the marker is THIS run's own (written by this
        run's rollback; such a run never reaches the success path,
        and the flag keeps that true by construction even if a check
        wrote one as a side effect).
        """
        if self._tx_wrote_pending_restore:
            return None
        payload = self._read_pending_restore()
        if payload is None:
            return None
        try:
            self.pending_restore_path.unlink()
        except OSError:  # noqa: BLE001 — a marker we cannot remove
            # stays for recovery's provenance check to discard; the
            # update itself has already succeeded.
            _log.warning("could not remove stale pending-restore "
                         "marker at %s", self.pending_restore_path,
                         exc_info=True)
            return None
        return {"snapshot": str(payload.get("snapshot", "")),
                "run_id": str(payload.get("run_id", "")),
                "target_sha": str(payload.get("target_sha", "")),
                "from_sha": str(payload.get("from_sha", ""))}

    def _pending_restore_superseded_by(
            self, payload: dict[str, Any]) -> str:
        """SHA (or run id) of a later successful update that makes
        this pending-restore marker stale, or "" when it is current.

        The journal is the evidence: a run that COMPLETED the full
        transaction after the marker was written migrated and
        health-checked the live database, adopting it as the base —
        the deferred restore's whole premise (revert the database to
        pair with the rolled-back code) is gone.  A no-op completion
        (``noop``) proves nothing about the database and never
        supersedes.  The marker's own run never completes (it rolled
        back), so its run id is excluded by construction as well as
        by the explicit check.
        """
        try:
            marker_ts = float(payload.get("ts") or 0)
        except (TypeError, ValueError):
            marker_ts = 0.0
        marker_run = str(payload.get("run_id") or "")
        for run in self._journal_runs():
            if marker_run and run["run_id"] == marker_run:
                continue
            for event in run["events"]:
                if event.get("event") != "complete" \
                        or event.get("noop"):
                    continue
                try:
                    event_ts = float(event.get("ts") or 0)
                except (TypeError, ValueError):
                    event_ts = 0.0
                if event_ts >= marker_ts:
                    return str(event.get("sha") or "") or run["run_id"]
        return ""

    def _pending_restore_premise_gone(
            self, payload: dict[str, Any]) -> dict[str, str] | None:
        """The marker's pairing facts when its premise is provably
        gone, or None when the premise holds or cannot be judged.

        The deferred restore exists to pair the failed update's
        pre-update snapshot with its pre-update CODE — the rollback
        restored the tree to ``from_sha`` and deferred the database
        half to match.  That premise is gone when the code actually
        running is not ``from_sha``: the install demonstrably moved
        on (a hand pull, a checkout — however the code arrived), the
        live database belongs to the running code's own history, and
        applying the snapshot would regress it to serve a rollback
        nobody is running.  Every input is a recorded fact — the
        marker's ``from_sha`` and the running HEAD this boot resolves
        (the same resolution the failure notice quotes).  A marker
        that records no ``from_sha`` (written by an older build) or
        a running version that cannot be read leaves the premise
        UNJUDGED — None — and the caller keeps today's behaviour
        exactly: a discard on a guess is never acceptable.
        """
        from_sha = str(payload.get("from_sha") or "")
        if not from_sha:
            return None
        running = self._git_sha()
        if not running or running == from_sha:
            return None
        return {"from_sha": from_sha, "running_sha": running}

    def _pending_restore_state_note(self, payload: dict[str, Any]) -> str:
        """Describe the ACTUAL code/database state for a marker.

        The boot failure notice quotes this verbatim, so it is built
        only from facts: which update the deferred restore belongs
        to, which version is running now, and whether that running
        code is the rolled-back version the restore was meant to
        pair with.  "Rolled-back code" is asserted only when the
        running HEAD is the marker update's own pre-update version.
        """
        target = str(payload.get("target_sha") or "")
        from_sha = str(payload.get("from_sha") or "")
        running = self._git_sha()
        parts: list[str] = []
        if target:
            parts.append("the deferred restore belongs to the failed "
                         f"update that targeted {target[:12]}")
        elif payload.get("run_id"):
            parts.append("the deferred restore belongs to failed "
                         f"update run {str(payload['run_id'])[:12]}")
        if running:
            if from_sha and running == from_sha:
                parts.append(
                    f"the running code is {running[:12]} — the "
                    "rolled-back pre-update version — while the "
                    "database is still the post-update one")
            else:
                parts.append(
                    f"the running code is {running[:12]}, which is "
                    "NOT the rolled-back version, and the database "
                    "in use is the unrestored live database")
        return "; ".join(parts)

    def apply_pending_restore(self) -> dict[str, Any]:
        """Apply a deferred database restore, if one is pending.

        Called at boot via :meth:`recover_interrupted` — before the
        runtime serves on the database, which is the only moment the
        swap is guaranteed not to fight the running process for the
        file.  No marker → nothing to do.  A STALE marker — one a
        later successful update superseded — is discarded, never
        applied: the successful run is expected to clear it itself,
        but a marker that survived (written by an older build, or its
        clearing failed) must not regress the live database at boot.
        A marker whose PREMISE is gone — the running code is provably
        not the rolled-back version the marker pairs with, the case
        of an install updated by hand where no journal completion
        ever supersedes it — is likewise discarded before any
        restore is attempted, but WITH an operator notice: neither
        the wrongful apply nor the every-boot failure page a locked
        database would otherwise produce can occur.  When the
        recorded provenance cannot judge the premise, nothing is
        discarded.  A marker naming a snapshot that no longer EXISTS
        is discarded the same way: a deleted snapshot can never be
        restored, so treating it as a transient failure would page
        the operator at every boot forever.  A current marker whose restore still cannot run
        keeps the marker (the next boot retries) and reports why,
        with the actual code/database state attached.
        """
        try:
            payload = json.loads(
                self.pending_restore_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"applied": False, "reason": "no pending restore"}
        if not isinstance(payload, dict):
            payload = {}
        snap_id = str(payload.get("snapshot", ""))
        if not snap_id:
            return {"applied": False,
                    "reason": "pending-restore marker names no snapshot"}
        superseded_by = self._pending_restore_superseded_by(payload)
        if superseded_by:
            try:
                self.pending_restore_path.unlink()
            except OSError:  # noqa: BLE001 — the discard stands
                _log.debug("could not remove stale pending-restore "
                           "marker", exc_info=True)
            self._journal({"run_id": "boot",
                           "event": "pending_restore_discarded_stale",
                           "ts": time.time(), "snapshot": snap_id,
                           "marker_run_id":
                               str(payload.get("run_id", "")),
                           "marker_target_sha":
                               str(payload.get("target_sha", "")),
                           "superseded_by": superseded_by})
            _log.warning(
                "discarded stale deferred database restore (snapshot "
                "%s, from the update that targeted %s): superseded "
                "by successful update %s — booting on the live "
                "database", snap_id,
                str(payload.get("target_sha", ""))[:12] or "?",
                superseded_by[:12])
            return {"applied": False, "discarded_stale": True,
                    "snapshot": snap_id, "superseded_by": superseded_by}
        premise = self._pending_restore_premise_gone(payload)
        if premise is not None:
            try:
                self.pending_restore_path.unlink()
            except OSError:  # noqa: BLE001 — the discard stands
                _log.debug("could not remove premise-gone "
                           "pending-restore marker", exc_info=True)
            self._journal({"run_id": "boot",
                           "event": "pending_restore_discarded_stale",
                           "ts": time.time(), "snapshot": snap_id,
                           "marker_run_id":
                               str(payload.get("run_id", "")),
                           "marker_target_sha":
                               str(payload.get("target_sha", "")),
                           "marker_from_sha": premise["from_sha"],
                           "running_sha": premise["running_sha"],
                           "discarded_reason": "premise_gone"})
            _log.warning(
                "discarded deferred database restore (snapshot %s, "
                "from the update that targeted %s): its premise is "
                "gone — the running code is %s, not the rolled-back "
                "version %s it pairs with — booting on the live "
                "database", snap_id,
                str(payload.get("target_sha", ""))[:12] or "?",
                premise["running_sha"][:12], premise["from_sha"][:12])
            return {"applied": False, "discarded_stale": True,
                    "discarded_premise": True, "snapshot": snap_id,
                    "update_sha": str(payload.get("target_sha", "")),
                    "from_sha": premise["from_sha"],
                    "running_sha": premise["running_sha"],
                    "state_note":
                        self._pending_restore_state_note(payload)}
        try:
            self.snapshots.restore(snap_id, force=True)
        except Exception as exc:  # noqa: BLE001 — reported below
            if isinstance(exc, SnapshotError) \
                    and str(exc).startswith("no snapshot"):
                # The restore target is GONE — the snapshot was
                # deleted after the restore was deferred (the owner
                # hand-deleted snapshots in the 2026-11-08 disk
                # clear-out). That is the store's own missing-id
                # wording (snapshots.get, the first thing restore()
                # calls), and it is permanent, not transient:
                # keeping the marker would re-report a restore
                # failure at every boot forever, with no path to
                # clear it short of a successful update. Discard it
                # like the premise-gone case — marker out, journal
                # entry, operator notice — and boot on the live
                # database. A snapshot that EXISTS but fails
                # verification raises a different SnapshotError and
                # stays on the error path below, marker kept.
                try:
                    self.pending_restore_path.unlink()
                except OSError:  # noqa: BLE001 — the discard stands
                    _log.debug("could not remove pending-restore "
                               "marker whose snapshot is gone",
                               exc_info=True)
                self._journal({"run_id": "boot",
                               "event": "pending_restore_discarded_stale",
                               "ts": time.time(), "snapshot": snap_id,
                               "marker_run_id":
                                   str(payload.get("run_id", "")),
                               "marker_target_sha":
                                   str(payload.get("target_sha", "")),
                               "discarded_reason": "snapshot_missing"})
                _log.warning(
                    "discarded deferred database restore (snapshot "
                    "%s, from the update that targeted %s): the "
                    "snapshot no longer exists — it was deleted "
                    "after the restore was deferred, so the restore "
                    "can never be applied — booting on the live "
                    "database", snap_id,
                    str(payload.get("target_sha", ""))[:12] or "?")
                return {"applied": False, "discarded_stale": True,
                        "discarded_snapshot_missing": True,
                        "snapshot": snap_id,
                        "update_sha": str(payload.get("target_sha", "")),
                        "from_sha": str(payload.get("from_sha", "")),
                        "running_sha": self._git_sha(),
                        "state_note":
                            self._pending_restore_state_note(payload)}
            detail = f"{type(exc).__name__}: {exc}"
            _log.error("pending database restore of %s failed: %s",
                       snap_id, detail)
            return {"applied": False, "snapshot": snap_id,
                    "error": detail,
                    "update_sha": str(payload.get("target_sha", "")),
                    "from_sha": str(payload.get("from_sha", "")),
                    "running_sha": self._git_sha(),
                    "state_note":
                        self._pending_restore_state_note(payload)}
        try:
            self.pending_restore_path.unlink()
        except OSError:  # noqa: BLE001 — applied is what matters
            _log.debug("could not remove pending-restore marker",
                       exc_info=True)
        self._journal({"run_id": "boot", "event": "pending_restore_applied",
                       "ts": time.time(), "snapshot": snap_id})
        _log.warning("applied deferred database restore: snapshot %s",
                     snap_id)
        return {"applied": True, "snapshot": snap_id}

    # ── restart confirmation ───────────────────────────────────────────────
    def note_restart_intent(self, report: UpdateReport) -> bool:
        """Record that the runtime is about to restart into ``report``.

        The first half of the restart journey. The chat layer
        restarts the process after a successful update, and the
        process making that promise is the one about to die — its
        "restarting now" cannot confirm anything. This marker (the
        version the update landed, written before the restart) is
        what the NEXT process checks at boot
        (:meth:`consume_restart_intent`): the restart counts as
        complete only when a process actually comes up running the
        landed version. Returns True when a marker was written —
        only for a real code move; a no-op update restarts into the
        same code and needs no confirmation.
        """
        to_sha = str(getattr(report, "to_sha", "") or "")
        from_sha = str(getattr(report, "from_sha", "") or "")
        if not to_sha or (from_sha and from_sha == to_sha):
            return False
        try:
            self.restart_intent_path.parent.mkdir(parents=True,
                                                  exist_ok=True)
            self.restart_intent_path.write_text(
                json.dumps({"to_sha": to_sha, "from_sha": from_sha,
                            "ts": time.time()}, indent=2),
                encoding="utf-8")
        except OSError:
            _log.error("could not write the restart-intent marker at "
                       "%s", self.restart_intent_path, exc_info=True)
            return False
        return True

    def consume_restart_intent(self) -> dict[str, Any] | None:
        """Settle a restart-intent marker at boot (the second half).

        Returns None when no update restart was in flight. Otherwise
        the marker is ALWAYS consumed — one-shot, so a later boot
        never re-reports an old restart — and the receipt states the
        verdict from facts: CONFIRMED when the running HEAD is the
        version the update landed, UNCONFIRMED when it is anything
        else (a hand checkout, an unreadable HEAD, a boot that never
        took the new code). Silence after "restarting now" is
        exactly how a dead boot used to hide.
        """
        try:
            raw = self.restart_intent_path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                payload = {}
        except ValueError:
            payload = {}
        try:
            self.restart_intent_path.unlink()
        except OSError:  # noqa: BLE001 — the verdict still reports
            _log.debug("could not remove the restart-intent marker",
                       exc_info=True)
        to_sha = str(payload.get("to_sha", "") or "")
        running = self._git_sha()
        confirmed = bool(to_sha) and running == to_sha
        receipt = {"recovered": True,
                   "action": ("restart_confirmed" if confirmed
                              else "restart_unconfirmed"),
                   "to_sha": to_sha,
                   "from_sha": str(payload.get("from_sha", "") or ""),
                   "running_sha": running}
        self._journal({"run_id": "boot", "event": receipt["action"],
                       "ts": time.time(), "to_sha": to_sha,
                       "running_sha": running})
        return receipt

    # ── quarantine ─────────────────────────────────────────────────────────
    def _quarantine(self) -> dict[str, Any]:
        try:
            return json.loads(self.quarantine_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"quarantined": []}

    def _is_quarantined(self, sha: str) -> bool:
        return any(q.get("sha") == sha
                   for q in self._quarantine().get("quarantined", []))

    def quarantine_list(self) -> list[dict[str, Any]]:
        """Versions that failed an update and will not be retried."""
        return list(self._quarantine().get("quarantined", []))

    def quarantine_clear(self, sha: str = "") -> int:
        """Clear the quarantine (one sha, or all when empty). Returns count
        removed."""
        data = self._quarantine()
        entries = data.get("quarantined", [])
        if sha:
            kept = [q for q in entries if q.get("sha") != sha]
        else:
            kept = []
        removed = len(entries) - len(kept)
        data["quarantined"] = kept
        self._write_quarantine(data)
        return removed

    def _write_quarantine(self, data: dict[str, Any]) -> None:
        self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        self.quarantine_path.write_text(json.dumps(data, indent=2),
                                        encoding="utf-8")

    def _quarantine_add(self, sha: str, reason: str) -> None:
        if not sha or self._is_quarantined(sha):
            return
        data = self._quarantine()
        data.setdefault("quarantined", []).append(
            {"sha": sha, "reason": reason, "ts": time.time()})
        self._write_quarantine(data)

    def _quarantine_is_stale(self, sha: str) -> bool:
        """True when no update ever PULLED ``sha``.

        The journal is the evidence: a version an update genuinely
        pulled and rolled back appears as a ``git_pull`` step whose
        ``to_sha`` is the version and whose ``from_sha`` is something
        else.  A version every run *started* at (``from_sha == sha``,
        the EC2 no-op incident) was never installed by an update, so
        quarantining it was never legitimate.
        """
        for run in self._journal_runs():
            for event in run["events"]:
                if event.get("event") == "step" \
                        and event.get("step") == "git_pull" \
                        and event.get("to_sha") == sha \
                        and event.get("from_sha") not in ("", sha):
                    return False
        return True

    def _quarantine_void(self, sha: str) -> int:
        """Remove quarantine entries for ``sha`` (stale-entry cleanup).

        Returns the number removed.  Used at run start for an entry
        naming the version already checked out: an update may only
        quarantine a version it pulled itself.
        """
        data = self._quarantine()
        entries = data.get("quarantined", [])
        kept = [q for q in entries if q.get("sha") != sha]
        removed = len(entries) - len(kept)
        if removed:
            data["quarantined"] = kept
            self._write_quarantine(data)
        return removed

    # ── history ────────────────────────────────────────────────────────────
    def _append_history(self, report: UpdateReport) -> None:
        entry = {"ts": time.time(), **report.to_dict()}
        try:
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:  # noqa: BLE001 — history is best effort
            _log.debug("could not append update history", exc_info=True)

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Past update reports, newest first."""
        entries: list[dict[str, Any]] = []
        try:
            with open(self.history_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            return []
        return list(reversed(entries))[:max(0, int(limit))]

    # ── journal internals ──────────────────────────────────────────────────
    def _journal(self, entry: dict[str, Any]) -> None:
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:  # noqa: BLE001 — the journal must never break updates
            _log.debug("could not append update journal", exc_info=True)

    def _journal_runs(self) -> list[dict[str, Any]]:
        runs: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        try:
            with open(self.journal_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    run_id = str(entry.get("run_id", ""))
                    if run_id not in runs:
                        runs[run_id] = []
                        order.append(run_id)
                    runs[run_id].append(entry)
        except OSError:
            return []
        return [{"run_id": rid, "events": runs[rid]} for rid in order]

    # ── git helpers ──────────────────────────────────────────────────────

    def _git_out(self, args: list[str], timeout: int = 60) -> tuple[int, str, str]:
        """Run git in the repo; (returncode, stdout, stderr). Never raises."""
        try:
            proc = subprocess.run(
                [self.git_bin, "-C", str(self.repo_dir), *args],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return 1, "", str(exc)
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    def _status_paths(self) -> tuple[set[str], set[str]]:
        """(dirty tracked paths, untracked paths) from one status pass.

        Renames contribute BOTH paths: the source is a deletion and
        the destination a creation, and a tree restore must know
        both names. Untracked files are listed individually
        (``--untracked-files=all``) — a collapsed ``newdir/`` entry
        could not be restored file-by-file. (empty, empty) when git
        cannot say (not a repo, git missing): callers treat that as
        "nothing known dirty", which is exactly right for
        scripted-seam constructions whose repo dir is not a repo.
        """
        rc, out, _ = self._git_out(
            ["status", "--porcelain", "--untracked-files=all"])
        tracked: set[str] = set()
        untracked: set[str] = set()
        if rc != 0:
            return tracked, untracked
        for line in out.splitlines():
            if len(line) < 4:
                continue
            code, path = line[:2], line[3:].strip()
            paths = [path]
            if " -> " in path:  # rename: both names are touched
                paths = [p.strip() for p in path.split(" -> ", 1)]
            for name in paths:
                name = name.strip('"')
                if not name:
                    continue
                if code == "??":
                    untracked.add(name)
                else:
                    tracked.add(name)
        return tracked, untracked

    def _git_changed_files(self, from_ref: str, to_ref: str) -> list[str]:
        rc, out, _ = self._git_out(["diff", "--name-only", from_ref, to_ref])
        if rc != 0:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]

    def _preflight_disk(self) -> tuple[bool, str]:
        """Decide whether the disk has room for this update.

        Returns (proceed, detail). The need is the shared measure
        (core.diskspace): the pre-update snapshot step 1 is about to
        take — database + WAL + the blob store it copies — plus an
        equal headroom for the pull/checkout churn. A MEASURED
        shortage refuses here, with both numbers, before anything
        on disk changes. A measurement FAILURE never bricks an
        update: it is named in the step detail and the update
        proceeds — the snapshot step still fails honestly if the
        disk truly is full."""
        try:
            rep = diskspace.disk_report(
                self.snapshots.home, self.snapshots.db_path,
                getattr(self.snapshots, "blob_dir", None))
        except Exception as exc:  # noqa: BLE001 - measure, don't brick
            return True, (f"could not measure free disk space "
                          f"({type(exc).__name__}: {exc}) — proceeding; "
                          "the snapshot step remains the guard")
        free = diskspace.human_bytes(rep["free_bytes"])
        need = diskspace.human_bytes(rep["need_bytes"])
        snap = diskspace.human_bytes(rep["snapshot_bytes"])
        # The need keeps its meaning (the snapshot this update is
        # about to take, plus headroom) — but when the home
        # tree's biggest area outweighs that snapshot, the
        # refusal names it: the pile that filled the owner's
        # disk was a WhatsApp media cache, and a preflight that
        # only ever discussed snapshots would have left the next
        # pile just as invisible (media-cache discipline,
        # 2026-11-08).
        area = diskspace.largest_area_note(rep)
        if rep["low"]:
            return False, (
                f"not enough free disk to update safely — free {free}, "
                f"this update needs ~{need} (pre-update snapshot ~{snap} "
                "plus equal headroom for the pull). Nothing was "
                "changed. Free space on the install volume and retry — "
                "pre-update snapshots are pruned automatically now, "
                "but a pile already on disk may need one manual clear."
                + (f" Biggest consumer on disk: {area}." if area
                   else ""))
        return True, (f"free {free} · needs ~{need}"
                      + (f" · {area}" if area else ""))

    def _preflight_pull(self) -> tuple[bool, str]:
        """Decide whether pulling is safe for the owner's local changes.

        Returns (proceed, detail). No local changes → proceed
        silently. Changes that collide with the incoming change set →
        refuse (detail explains; nothing has been pulled or restored).
        Disjoint changes → proceed, with the detail naming what is
        being preserved.

        Untracked files count as local changes for the collision
        test: an untracked file sitting at a path the incoming
        change set CREATES would stop the pull mid-checkout ("the
        following untracked working tree file would be
        overwritten"), leaving the cleanup machinery to undo a
        half-applied pull — the refusal belongs here, up front,
        exactly as for a modified tracked file. A disjoint untracked
        file can neither be overwritten nor destroyed by the pull,
        so it never blocks and is not named.
        """
        tracked, untracked = self._status_paths()
        if not tracked and not untracked:
            return True, ""
        rc, _out, err = self._git_out(["fetch", "--quiet"], timeout=300)
        if rc != 0:
            return False, (
                "uncommitted local changes present and the incoming "
                "update could not be fetched to check them against it "
                f"(git fetch failed: {err.strip().splitlines()[-1] if err.strip() else 'unknown error'})"
                " — refusing to pull blind. Commit or stash the local "
                "changes and retry.")
        rc, upstream, _ = self._git_out(["rev-parse", "@{upstream}"])
        upstream = upstream.strip()
        if rc != 0 or not upstream:
            return False, (
                "uncommitted local changes present but the upstream "
                "branch cannot be determined to check them against the "
                "incoming update — refusing to pull blind. Commit or "
                "stash the local changes and retry.")
        incoming = set(self._git_changed_files("HEAD", upstream))
        tracked_hits = sorted(tracked & incoming)
        untracked_hits = sorted(untracked & incoming)
        if tracked_hits or untracked_hits:
            parts = []
            if tracked_hits:
                parts.append("modified: " + ", ".join(tracked_hits))
            if untracked_hits:
                parts.append("untracked (the update adds a file at "
                             "this path): " + ", ".join(untracked_hits))
            return False, (
                "uncommitted local changes collide with the incoming "
                "update: " + "; ".join(parts)
                + " — commit or stash them first; nothing was pulled "
                "and nothing was overwritten.")
        if tracked:
            return True, ("local changes preserved (untouched by this "
                          "update): " + ", ".join(sorted(tracked)))
        return True, ""

    def _restore_tree(self, from_sha: str) -> tuple[bool, str]:
        """Put the code tree back to ``from_sha``.

        Only files that differ between the current HEAD and
        ``from_sha`` are touched (checked out from ``from_sha``, or
        removed when the update added them), so uncommitted local
        edits to any other file survive. HEAD and the index are then
        moved with ``reset --mixed``, which never touches the working
        tree. Returns (ok, detail).
        """
        if not from_sha:
            return False, "no pre-update commit recorded"
        current = self._git_sha()
        if not current:
            return False, "cannot read current HEAD"
        if current == from_sha:
            return True, f"tree already at {from_sha[:12]}"
        changed = self._git_changed_files(from_sha, current)
        for name in changed:
            rc, _, _ = self._git_out(["cat-file", "-e", f"{from_sha}:{name}"])
            if rc == 0:
                rc, _, err = self._git_out(["checkout", from_sha, "--", name])
                if rc != 0:
                    return False, f"checkout of {name} failed: {err.strip()}"
            else:
                # Added by the update: remove it from the working tree.
                try:
                    (self.repo_dir / name).unlink(missing_ok=True)
                except OSError as exc:
                    return False, f"could not remove added file {name}: {exc}"
        rc, _, err = self._git_out(["reset", "--mixed", from_sha])
        if rc != 0:
            return False, f"git reset --mixed {from_sha[:12]} failed: {err.strip()}"
        if self._git_sha() != from_sha:
            return False, "HEAD did not return to the pre-update commit"
        return True, (f"tree restored to {from_sha[:12]} "
                      f"({len(changed)} file(s) put back)")

    def _restore_tree_after_failed_pull(
        self, from_sha: str, pre_tracked: set[str],
        pre_untracked: set[str],
    ) -> tuple[bool, str]:
        """Put the working tree back to its pre-pull state.

        A failed pull can leave files written, deleted or created
        even though HEAD never moved (git's checkout is file-by-file;
        the 2026-11-07 EC2 pull died on the locked bridge binary
        after writing most of the tree). The HEAD-based restore
        (:meth:`_restore_tree`) is blind to that state — unmoved
        HEAD, empty diff, "nothing to restore" — so this works off
        the STATUS DELTA against the status captured before the
        pull: anything dirty-or-untracked now that was
        clean-or-absent before is the pull's own work and is undone
        (checked out from the pre-update commit, or removed when
        the pull created it); anything already dirty or untracked
        before the pull is the owner's and is never touched
        (preflight proved those files disjoint from the incoming
        change set, so the pull cannot have written them). Success
        is VERIFIED, not assumed: the post-restore status must
        equal the captured pre-pull status exactly.
        """
        ref = from_sha or "HEAD"
        head = self._git_sha()
        if head and from_sha and head != from_sha:
            # The pull moved HEAD before failing (not the --ff-only
            # shape, but never assume): the HEAD-based restore owns
            # the tracked content; the delta sweep below still runs
            # for the untracked leftovers it cannot see.
            moved_ok, moved_detail = self._restore_tree(from_sha)
            if not moved_ok:
                return False, moved_detail
        now_tracked, now_untracked = self._status_paths()
        put_back = 0
        removed = 0
        problems: list[str] = []
        for name in sorted(now_tracked - pre_tracked):
            rc, _, _ = self._git_out(["cat-file", "-e", f"{ref}:{name}"])
            if rc == 0:
                rc, _, err = self._git_out(["checkout", ref, "--", name])
                if rc != 0:
                    problems.append(
                        f"{name} (checkout failed: {err.strip()})")
                else:
                    put_back += 1
            else:
                # Not in the pre-update tree: the pull created it
                # (staged-new or an index ghost) — remove it, the
                # same treatment _restore_tree gives added files.
                try:
                    (self.repo_dir / name).unlink(missing_ok=True)
                    removed += 1
                except OSError as exc:
                    problems.append(f"{name} (remove failed: {exc})")
        for name in sorted(now_untracked - pre_untracked):
            try:
                (self.repo_dir / name).unlink(missing_ok=True)
                removed += 1
            except OSError as exc:
                problems.append(f"{name} (remove failed: {exc})")
        if problems:
            return False, "could not restore: " + "; ".join(problems)
        after_tracked, after_untracked = self._status_paths()
        if after_tracked != pre_tracked or after_untracked != pre_untracked:
            delta = sorted((after_tracked ^ pre_tracked)
                           | (after_untracked ^ pre_untracked))
            return False, ("working tree still differs from its "
                           "pre-update state at: "
                           + ", ".join(delta[:8])
                           + ("…" if len(delta) > 8 else ""))
        return True, (f"working tree restored to its pre-update "
                      f"state ({put_back} file(s) put back, "
                      f"{removed} added file(s) removed; the status "
                      "matches the pre-update status exactly)")

    def _verify_tree_matches(self, from_sha: str,
                             to_sha: str) -> tuple[bool, str]:
        """The working tree IS ``to_sha`` for everything the pull
        changed (the post-pull verification).

        Scoped to the incoming change set on purpose: the owner's
        disjoint local edits (preflight-verified) are supposed to
        be dirty and must not fail this check — but a changed file
        that is modified, missing, or shadowed by an untracked
        leftover means what would run is not what was pulled.
        """
        head = self._git_sha()
        if head != to_sha:
            return False, (f"HEAD is "
                           f"{head[:12] if head else 'unreadable'}, "
                           f"not the pulled commit {to_sha[:12]}")
        changed = set(self._git_changed_files(from_sha, to_sha))
        tracked, untracked = self._status_paths()
        offenders = sorted((tracked | untracked) & changed)
        if offenders:
            return False, ("working tree differs from the pulled "
                           "commit at: " + ", ".join(offenders[:8])
                           + ("…" if len(offenders) > 8 else ""))
        return True, (f"working tree matches {to_sha[:12]} "
                      f"({len(changed)} changed file(s) clean)")

    # ── steps ──────────────────────────────────────────────────────────────

    def _run_quiesce(self) -> tuple[bool, str]:
        """Run the wired quiesce (see the constructor seam).

        No hook attached is a recorded fact, not a silent pass: this
        process runs no repo-file children of its own (the console
        update assumes Devon is stopped — the bridges belong to the
        running Devon process, whose chat runtime wires the real
        quiesce). A hook that RAISES cannot prove the tree is
        unlocked, so it refuses the pull like a hook that reports a
        child still running.
        """
        if self._quiesce is None:
            return True, ("no quiesce hook attached — this process "
                          "runs no repo-file children to stop (the "
                          "chat runtime wires the WhatsApp bridge "
                          "quiesce; a console update assumes Devon "
                          "is stopped)")
        try:
            ok, detail = self._quiesce()
        except Exception as exc:  # noqa: BLE001 - an unprovable stop
            return False, (f"the quiesce step itself failed: "
                           f"{type(exc).__name__}: {exc} — refusing "
                           "to pull while a repo-file child may "
                           "still be running")
        return bool(ok), str(detail)

    def _git_pull(self) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                [self.git_bin, "-C", str(self.repo_dir), "pull", "--ff-only"],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=300,
            )
        except FileNotFoundError:
            return False, f"{self.git_bin} not found"
        except subprocess.TimeoutExpired:
            return False, "git pull timed out after 300s"
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join(out.strip().splitlines()[-3:])
        if proc.returncode != 0:
            return False, tail or f"exit {proc.returncode}"
        return True, tail or "already up to date"

    def _migrate(self) -> tuple[list[str], int]:
        from ..storage.db import Database

        db = Database(str(self.snapshots.db_path))
        try:
            summary = db.migrate()
            return list(summary.applied), int(summary.version)
        finally:
            db.close()

    def _rollback(
        self, report: UpdateReport, started: float, error: str, *,
        run_id: str = "", quarantine_sha: str = "",
    ) -> UpdateReport:
        """Restore the pre-update snapshot AND the pre-update code tree.

        The update is NOT ok, and it only counts as rolled back when
        both halves are back: state (snapshot) and code (the files the
        pull changed, with HEAD returned to the pre-pull commit). A
        state-only restore would leave the new — failed — code in
        place for the next boot to run.
        """
        report.error = error
        state_ok = False
        try:
            self.snapshots.restore(report.pre_update_snapshot, force=True)
            state_ok = True
            report.record("rollback", True,
                          f"restored {report.pre_update_snapshot}")
            _log.warning("update rolled back: %s", error)
        except Exception as exc:  # noqa: BLE001 - rollback failure is critical
            if _is_lock_error(exc):
                # The running process holds the live database open, so
                # the swap cannot happen from inside it (WinError 32
                # on the EC2 host). That is not a failed rollback: the
                # database half is DEFERRED via a pending-restore
                # marker the next boot applies before the database
                # opens, and the code-tree restore below — the half
                # that must not wait — still runs now.
                reason = f"{type(exc).__name__}: {exc}"
                self._write_pending_restore(
                    report.pre_update_snapshot, reason, run_id=run_id,
                    target_sha=quarantine_sha or self._tx_to_sha,
                    from_sha=self._tx_from_sha)
                report.db_restore_deferred = reason
                report.record(
                    "rollback_db", True,
                    "database restore deferred — the live database is "
                    f"locked by the running process ({reason}); a "
                    "pending-restore marker was written and the restore "
                    "applies at the next boot, before the database opens")
                state_ok = True
                _log.warning("update rollback: database restore deferred "
                             "(%s)", reason)
            else:
                report.record("rollback", False,
                              f"{type(exc).__name__}: {exc}")
                _log.error("ROLLBACK FAILED: %s", exc, exc_info=True)
        tree_ok = True
        if self._tx_from_sha:
            tree_ok, detail = self._restore_tree(self._tx_from_sha)
            report.record("git_restore", tree_ok, detail)
            if not tree_ok:
                _log.error("TREE ROLLBACK FAILED: %s", detail)
        report.rolled_back = state_ok and tree_ok
        # Quarantine names ONLY a version this update itself pulled:
        # never the version the tree was already at (that is the
        # running version — quarantining it was the EC2 incident).
        if quarantine_sha and quarantine_sha != self._tx_from_sha:
            self._quarantine_add(quarantine_sha, error)
            report.quarantined = quarantine_sha
        report.seconds = time.time() - started
        if run_id:
            self._journal({"run_id": run_id, "event": "rolled_back",
                           "ts": time.time(), "error": error,
                           "restored": report.pre_update_snapshot,
                           "quarantined": quarantine_sha,
                           "db_restore_deferred":
                               report.db_restore_deferred})
        return report

    def _git_sha(self) -> str:
        """Current HEAD sha of the repo ("" when unknowable)."""
        try:
            proc = subprocess.run(
                [self.git_bin, "-C", str(self.repo_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""
