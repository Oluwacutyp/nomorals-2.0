"""Scheduled-backup tools — the daily backup + monthly restore drill.

Scheduler-internal tools (``__backup_tick__`` / ``__backup_drill__``):
they run the encrypted offsite 3-2-1 pipeline
(:mod:`nomorals.backup.coordinator`) on the live scheduler's tick.
Registered via closure in :func:`register`, NOT via the ``TOOLS`` dict:
the registry calls fns with no ``context`` arg, and a
``context``-taking fn would fail with TypeError (the pre-existing
``presence_sense`` failure mode). Same closure pattern as
``pulse``/``briefing`` and the other ``__*_tick__`` tools in
:mod:`nomorals.tools.autonomy`.

Fail-closed by construction: a tick that cannot even build its
coordinator (vault locked, no backup key, no settings) records the
failure in the scheduler failure ledger, attempts a notifier report so
the owner hears about it, and returns ``{"ok": False}`` — the
scheduler records that as a failed run. It never raises, and never
silently skips. A run that gets far enough to have a backup id is
already reported by the coordinator itself
(:func:`nomorals.backup.report.report_backup_failure`), so the tool
does not report twice — it only records the ledger entry.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..scheduler.failures import record_failure

__all__ = ["register"]

_log = get_logger(__name__)


def _build_coordinator(context: Any) -> Any:
    """Build the backup coordinator for this home.

    Root is the Devon home (the plan snapshots ``data/nomorals.db`` and
    the user-data dirs under it); the work dir is the configured backup
    dir, so generations land where ``/backup status`` looks. The vault
    holds the encryption key; the GitHub connector is attached only
    when a GitHub credential is actually stored — without one the
    coordinator queues the generation in the outbox (the designed
    offline path) instead of failing every day.
    """
    from ..accounts.vault import open_vault
    from ..agents.notifier import Notifier
    from ..backup.coordinator import BackupCoordinator, open_s3_store
    from ..backup.errors import BackupError
    from ..connectors.github import GitHubConnector

    settings = getattr(context, "settings", None)
    if settings is None:
        raise BackupError(
            "backup tick needs settings (home + backup dir) — no "
            "settings on this context")
    vault = open_vault(getattr(context, "db", None), context=context,
                       actor="backup")
    connector = None
    github = GitHubConnector(vault)
    if github._load_credential() is not None:  # noqa: SLF001 - token via vault
        connector = github
    return BackupCoordinator(
        root=settings.home_path, work_dir=settings.backup_dir,
        vault=vault, connector=connector, notifier=Notifier(context),
        s3_store=open_s3_store(),
        # getattr, not a hard read: a settings object that cannot
        # name a workspace skips the session seal (the coordinator
        # treats None as "no session step") instead of failing a
        # tick that was building fine before the wiring.
        session_workspace_dir=getattr(settings, "workspace_dir", None),
    )


def _report_failure_best_effort(context: Any, backup_id: str,
                                error: BaseException) -> None:
    """Attempt a notifier report for a tick that never got a coordinator.

    Best effort only — reporting never raises, and a failed report must
    not mask the failure it reports.
    """
    try:
        from ..agents.notifier import Notifier
        from ..backup.report import report_backup_failure

        report_backup_failure(Notifier(context), backup_id, error)
    except Exception:  # noqa: BLE001 - reporting never raises
        _log.warning("backup failure report failed", exc_info=True)


def register(registry: Any) -> None:
    _context = registry.context

    @registry.register(
        "__backup_tick__",
        description="Run one scheduled backup generation (daily): "
        "snapshot -> encrypt -> ship to GitHub Releases or queue in "
        "the outbox when offline. Scheduler-internal; runs daily at "
        "02:00. Routine and silent on success — the run reports "
        "itself through the notifier.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _backup_tick_tool() -> dict[str, Any]:
        from ..backup.coordinator import run_scheduled_backup

        try:
            coordinator = _build_coordinator(_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            record_failure("backup.tick", exc)
            _report_failure_best_effort(_context, "scheduled", exc)
            return {"ok": False,
                    "error": f"backup tick could not start: {exc}"}
        try:
            result = run_scheduled_backup(coordinator)
        except Exception as exc:  # noqa: BLE001 - coordinator already reported via notifier
            record_failure("backup.tick", exc)
            return {"ok": False, "error": f"backup tick failed: {exc}"}
        return {"ok": True,
                "backup_id": result.get("backup_id"),
                "shipped": bool(result.get("shipped")),
                "queued": bool(result.get("queued")),
                "manifest_pushed": bool(result.get("manifest_pushed")),
                "s3": bool(result.get("s3")),
                "s3_error": str(result.get("s3_error", "")),
                "total_encrypted_bytes": result.get(
                    "total_encrypted_bytes", 0)}

    @registry.register(
        "__backup_drill__",
        description="Run the monthly restore drill over the latest "
        "backup generation: decrypt + verify + restore-check. "
        "Scheduler-internal; runs monthly on the 1st at 03:00. "
        "Reports itself through the notifier; a failed drill is a "
        "failed run.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _backup_drill_tool(backup_id: str = "") -> dict[str, Any]:
        try:
            coordinator = _build_coordinator(_context)
        except Exception as exc:  # noqa: BLE001 - one bad tick never kills the job
            record_failure("backup.drill", exc)
            _report_failure_best_effort(_context, "scheduled", exc)
            return {"ok": False,
                    "error": f"backup drill could not start: {exc}"}
        try:
            # run_drill directly (not run_scheduled_drill) so an explicit
            # backup_id reaches the drill; with no id it drills the latest
            # generation, exactly as the scheduled run does.
            report = coordinator.run_drill(backup_id=backup_id)
        except Exception as exc:  # noqa: BLE001 - coordinator already reported via notifier
            record_failure("backup.drill", exc)
            return {"ok": False, "error": f"backup drill failed: {exc}"}
        return {"ok": bool(report.ok),
                "backup_id": report.backup_id,
                "summary": report.summary()}
