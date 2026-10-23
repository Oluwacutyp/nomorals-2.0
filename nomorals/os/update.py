"""Transactional self-update: pull, migrate, health-check, or roll back.

:class:`UpdateManager` owns the whole update as one transaction:

1. take a pre-update snapshot (via :class:`SnapshotManager`),
2. ``git pull --ff-only`` in the repo (skippable),
3. run database migrations,
4. run post-update health checks (importable package, layering subset,
   DB integrity),

then advisory reporting (the error scan — information only, never a
gate).  A pull that lands nothing (HEAD unchanged) short-circuits
right after the pull: no migrate, no health gate, no rollback, no
quarantine, no restart — an up-to-date tree has nothing to prove.

If *any* step fails, the pre-update snapshot is restored automatically and
the report says so.  The system is never left half-updated: a failed pull
leaves the tree untouched (``--ff-only`` is all-or-nothing), and a failed
migration or health check restores BOTH the state snapshot and the code
tree (the files the pull changed are put back and HEAD returns to the
pre-update commit).  One exception, by design: when the state restore
cannot swap the live database because the running process holds it open
(Windows WinError 32), the database half is deferred — a pending-restore
marker is written and applied by the next boot (see
:meth:`UpdateManager.recover_interrupted`) before the runtime serves on
the database — and the report carries a deferral note, never a rollback
failure.

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

from ..core.filelock import lock_exclusive_nb, unlock as _unlock_file
from ..core.logging_setup import get_logger
from .snapshots import SnapshotManager, SnapshotRefused

__all__ = [
    "HealthCheckFn",
    "UpdateReport",
    "UpdateManager",
    "default_health_checks",
    "default_advisory_checks",
]

_log = get_logger(__name__)

#: A health check returns (ok, detail).
HealthCheckFn = Callable[[], "tuple[bool, str]"]

JOURNAL_FILENAME = "update-journal.jsonl"
HISTORY_FILENAME = "update-history.jsonl"
QUARANTINE_FILENAME = "update-quarantine.json"
UPDATE_LOCK_FILENAME = "update.lock"
#: Written when a rollback's database restore is deferred because the
#: running process holds the live database open; consumed at boot by
#: :meth:`UpdateManager.apply_pending_restore` (via recover_interrupted).
PENDING_RESTORE_FILENAME = "pending-db-restore.json"

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


def default_health_checks(
    repo_dir: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> list[HealthCheckFn]:
    """The standard post-update health gate.

    * the ``nomorals`` package imports and reports a version,
    * the layering subset passes (no upward imports in the tracked tree),
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
            try:
                tests_dir = str(Path(repo_dir) / "tests")
                if tests_dir not in sys.path:
                    sys.path.insert(0, tests_dir)
                import test_layering

                case = test_layering.TestLayering("test_no_upward_imports")
                case.test_no_upward_imports()
                return True, "layering subset green (tracked tree)"
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
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.snapshots = snapshots
        self.git_bin = git_bin
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
        #: Code-move bookkeeping for the transaction in flight ("" when
        #: this run did not move the tree): the commit to restore on
        #: rollback and the commit the pull landed on.
        self._tx_from_sha = ""
        self._tx_to_sha = ""

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

        # Step 1: pre-update snapshot — the rollback anchor.
        try:
            snap = self.snapshots.create(label="pre-update")
            report.pre_update_snapshot = snap.id
            report.record("snapshot", True, f"pre-update snapshot {snap.id}")
            self._journal({"run_id": run_id, "event": "step", "step": "snapshot",
                           "ts": time.time(), "snapshot": snap.id})
        except Exception as exc:  # noqa: BLE001 - snapshot failure aborts
            report.record("snapshot", False, f"{type(exc).__name__}: {exc}")
            report.error = f"could not take pre-update snapshot: {exc}"
            report.seconds = time.time() - started
            self._journal({"run_id": run_id, "event": "aborted",
                           "ts": time.time(), "error": report.error})
            self._append_history(report)
            return report

        # Step 2: pull. --ff-only is all-or-nothing: a failed pull changes
        # nothing, so no rollback is needed for it.
        new_sha = ""
        if pull:
            ok, detail = self._git_pull()
            report.record("git_pull", ok, detail)
            if not ok:
                self._journal({"run_id": run_id, "event": "step",
                               "step": "git_pull", "ts": time.time(),
                               "ok": ok, "detail": detail})
                report.error = f"git pull failed: {detail}"
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
                "snapshot": pending.get("snapshot", ""), "run_id": ""}
        runs = self._journal_runs()
        if not runs:
            return pending_receipt or {"recovered": False,
                                       "reason": "no journal entries"}
        last = runs[-1]
        events = last["events"]
        # "pending_restore_applied" is journaled under a synthetic
        # "boot" run by apply_pending_restore — a complete record in
        # itself, never an interrupted update to recover.
        terminal = next((e for e in events
                         if e.get("event") in ("complete", "rolled_back",
                                               "aborted", "crash_recovery",
                                               "pending_restore_applied")),
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
        receipt: dict[str, Any] = {
            "recovered": True, "run_id": last["run_id"],
            "last_steps": steps, "snapshot": snap_id,
        }
        if pending.get("applied") or pending.get("error"):
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
    def _write_pending_restore(self, snapshot_id: str, reason: str) -> None:
        """Record a database restore deferred to the next boot.

        Written by :meth:`_rollback` when the live database cannot be
        swapped from inside the process holding it open.  The marker
        is the whole contract: snapshot id + why; applying it is
        :meth:`apply_pending_restore`'s job.
        """
        payload = {"snapshot": snapshot_id, "reason": reason,
                   "ts": time.time()}
        try:
            self.pending_restore_path.parent.mkdir(parents=True,
                                                   exist_ok=True)
            self.pending_restore_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:  # noqa: BLE001 — the deferral note still reports
            _log.error("could not write pending-restore marker at %s",
                       self.pending_restore_path, exc_info=True)

    def apply_pending_restore(self) -> dict[str, Any]:
        """Apply a deferred database restore, if one is pending.

        Called at boot via :meth:`recover_interrupted` — before the
        runtime serves on the database, which is the only moment the
        swap is guaranteed not to fight the running process for the
        file.  No marker → nothing to do.  A restore that still cannot
        run keeps its marker (the next boot retries) and reports why.
        """
        try:
            payload = json.loads(
                self.pending_restore_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"applied": False, "reason": "no pending restore"}
        snap_id = str(payload.get("snapshot", ""))
        if not snap_id:
            return {"applied": False,
                    "reason": "pending-restore marker names no snapshot"}
        try:
            self.snapshots.restore(snap_id, force=True)
        except Exception as exc:  # noqa: BLE001 — reported, marker kept
            detail = f"{type(exc).__name__}: {exc}"
            _log.error("pending database restore of %s failed: %s",
                       snap_id, detail)
            return {"applied": False, "snapshot": snap_id, "error": detail}
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

    def _dirty_tracked_files(self) -> list[str]:
        """Locally modified tracked files (untracked files excluded:
        neither a pull nor a tree restore can destroy those)."""
        rc, out, _ = self._git_out(
            ["status", "--porcelain", "--untracked-files=no"])
        if rc != 0:
            return []  # not a repo / git missing — the pull reports it
        files: list[str] = []
        for line in out.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if " -> " in path:  # rename: the destination is what's dirty
                path = path.split(" -> ", 1)[1]
            files.append(path.strip('"'))
        return files

    def _git_changed_files(self, from_ref: str, to_ref: str) -> list[str]:
        rc, out, _ = self._git_out(["diff", "--name-only", from_ref, to_ref])
        if rc != 0:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]

    def _preflight_pull(self) -> tuple[bool, str]:
        """Decide whether pulling is safe for the owner's local changes.

        Returns (proceed, detail). No local tracked changes → proceed
        silently. Changes that collide with the incoming change set →
        refuse (detail explains; nothing has been pulled or restored).
        Disjoint changes → proceed, with the detail naming what is
        being preserved.
        """
        dirty = self._dirty_tracked_files()
        if not dirty:
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
        incoming = self._git_changed_files("HEAD", upstream)
        collisions = sorted(set(dirty) & set(incoming))
        if collisions:
            return False, (
                "uncommitted local changes collide with the incoming "
                "update: " + ", ".join(collisions)
                + " — commit or stash them first; nothing was pulled "
                "and nothing was overwritten.")
        return True, ("local changes preserved (untouched by this "
                      "update): " + ", ".join(sorted(dirty)))

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

    # ── steps ──────────────────────────────────────────────────────────────

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
                self._write_pending_restore(report.pre_update_snapshot,
                                            reason)
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
