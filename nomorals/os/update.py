"""Transactional self-update: pull, migrate, health-check, or roll back.

:class:`UpdateManager` owns the whole update as one transaction:

1. take a pre-update snapshot (via :class:`SnapshotManager`),
2. ``git pull --ff-only`` in the repo (skippable),
3. run database migrations,
4. run post-update health checks (importable package, layering subset,
   error scan, DB integrity).

If *any* step fails, the pre-update snapshot is restored automatically and
the report says so.  The system is never left half-updated: a failed pull
leaves the tree untouched (``--ff-only`` is all-or-nothing), and a failed
health check triggers a full state restore.

Health checks are plain callables ``() -> (ok: bool, detail: str)`` so tests
can inject a broken one to prove the rollback path.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .snapshots import SnapshotManager

__all__ = [
    "HealthCheckFn",
    "UpdateReport",
    "UpdateManager",
    "default_health_checks",
]

_log = get_logger(__name__)

#: A health check returns (ok, detail).
HealthCheckFn = Callable[[], "tuple[bool, str]"]


@dataclass
class UpdateReport:
    """What the update did, step by step."""

    ok: bool = False
    rolled_back: bool = False
    pre_update_snapshot: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append({"step": name, "ok": ok, "detail": detail})

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rolled_back": self.rolled_back,
            "pre_update_snapshot": self.pre_update_snapshot,
            "steps": self.steps,
            "error": self.error,
            "seconds": round(self.seconds, 3),
        }


def default_health_checks(
    repo_dir: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> list[HealthCheckFn]:
    """The standard post-update health checks.

    * the ``nomorals`` package imports and reports a version,
    * the layering subset passes (no upward imports in the changed tree),
    * the error scanner reports no findings on first-party code,
    * the live database passes ``PRAGMA integrity_check``.
    """
    checks: list[HealthCheckFn] = []

    def _import() -> tuple[bool, str]:
        try:
            import nomorals  # noqa: F401
            from ..version import __version__

            return True, f"nomorals {__version__} imports"
        except Exception as exc:  # noqa: BLE001 - the check IS the report
            return False, f"import failed: {exc}"

    checks.append(_import)

    if repo_dir is not None:

        def _layering() -> tuple[bool, str]:
            try:
                import sys

                tests_dir = str(Path(repo_dir) / "tests")
                if tests_dir not in sys.path:
                    sys.path.insert(0, tests_dir)
                import test_layering

                case = test_layering.TestLayering("test_no_upward_imports")
                case.test_no_upward_imports()
                return True, "layering subset green"
            except Exception as exc:  # noqa: BLE001
                return False, f"layering failed: {exc}"

        def _error_scan() -> tuple[bool, str]:
            try:
                from ..tools.error_scan import scan

                report = scan([str(Path(repo_dir) / "nomorals")])
                errors = [f for f in report.findings if f.severity == "error"]
                if errors:
                    return False, f"{len(errors)} error_scan errors"
                return True, "error_scan clean"
            except Exception as exc:  # noqa: BLE001
                return False, f"error_scan failed: {exc}"

        checks.extend([_layering, _error_scan])

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
                return False, f"db integrity_check: {result}"
            except Exception as exc:  # noqa: BLE001
                return False, f"db check failed: {exc}"

        checks.append(_db)

    return checks


class UpdateManager:
    """Run a transactional update of the Devon installation."""

    def __init__(
        self,
        repo_dir: str | os.PathLike[str],
        snapshots: SnapshotManager,
        *,
        health_checks: list[HealthCheckFn] | None = None,
        git_bin: str = "git",
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.snapshots = snapshots
        self.git_bin = git_bin
        self.health_checks = (
            list(health_checks)
            if health_checks is not None
            else default_health_checks(repo_dir, snapshots.db_path)
        )

    # ── public ─────────────────────────────────────────────────────────────

    def run(self, *, pull: bool = True) -> UpdateReport:
        """Execute the update transaction. Never raises on update failure."""
        started = time.time()
        report = UpdateReport()

        # Step 1: pre-update snapshot — the rollback anchor.
        try:
            snap = self.snapshots.create(label="pre-update")
            report.pre_update_snapshot = snap.id
            report.record("snapshot", True, f"pre-update snapshot {snap.id}")
        except Exception as exc:  # noqa: BLE001 - snapshot failure aborts
            report.record("snapshot", False, f"{type(exc).__name__}: {exc}")
            report.error = f"could not take pre-update snapshot: {exc}"
            report.seconds = time.time() - started
            return report

        # Step 2: pull. --ff-only is all-or-nothing: a failed pull changes
        # nothing, so no rollback is needed for it.
        if pull:
            ok, detail = self._git_pull()
            report.record("git_pull", ok, detail)
            if not ok:
                report.error = f"git pull failed: {detail}"
                report.seconds = time.time() - started
                return report
        else:
            report.record("git_pull", True, "skipped (--no-pull)")

        # Step 3: migrations.
        try:
            applied, version = self._migrate()
            report.record(
                "migrate", True,
                f"schema v{version}" + (f" (+{len(applied)} applied)" if applied else ""),
            )
        except Exception as exc:  # noqa: BLE001 - migrate failure rolls back
            report.record("migrate", False, f"{type(exc).__name__}: {exc}")
            return self._rollback(report, started, f"migration failed: {exc}")

        # Step 4: health checks. ANY failure rolls back.
        failed: list[str] = []
        for check in self.health_checks:
            name = getattr(check, "__name__", "health_check")
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001 - a raising check is a failure
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            report.record(f"health:{name}", ok, detail)
            if not ok:
                failed.append(f"{name}: {detail}")
        if failed:
            return self._rollback(
                report, started, "health check failed: " + "; ".join(failed)
            )

        report.ok = True
        report.seconds = time.time() - started
        _log.info("update completed cleanly")
        return report

    # ── steps ──────────────────────────────────────────────────────────────

    def _git_pull(self) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                [self.git_bin, "-C", str(self.repo_dir), "pull", "--ff-only"],
                capture_output=True,
                text=True,
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
        self, report: UpdateReport, started: float, error: str
    ) -> UpdateReport:
        """Restore the pre-update snapshot. The update is NOT ok."""
        report.error = error
        try:
            self.snapshots.restore(report.pre_update_snapshot, force=True)
            report.rolled_back = True
            report.record("rollback", True,
                          f"restored {report.pre_update_snapshot}")
            _log.warning("update rolled back: %s", error)
        except Exception as exc:  # noqa: BLE001 - rollback failure is critical
            report.rolled_back = False
            report.record("rollback", False, f"{type(exc).__name__}: {exc}")
            _log.error("ROLLBACK FAILED: %s", exc, exc_info=True)
        report.seconds = time.time() - started
        return report
