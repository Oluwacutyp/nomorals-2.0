"""Coordinator: snapshot -> protect -> manifest -> ship -> report -> prune.

The one entry point for a backup generation. Pipeline:

1. :mod:`nomorals.backup.snapshot` — consistent local snapshot, verified
   (corrupt snapshots abort here, never shipped).
2. :mod:`nomorals.backup.protect` — gzip -> AES-256 encrypt -> SHA-256
   (ciphertext) -> split into 100MB parts.
3. :mod:`nomorals.backup.manifest` — git-native record (checksums only).
4. :mod:`nomorals.backup.ship` — ensure_repo -> create_release -> upload
   assets; manifest-only commit into the repo. Network down -> outbox
   queue + retry, never a silent skip.
5. :mod:`nomorals.backup.report` — outcome via the notifier organ.
6. :mod:`nomorals.backup.policy` — retention, local + GitHub.

3-2-1: copy 1 = live DB + continuous replication
(:func:`snapshot_via_replicator`); copy 2 = local/external archive (the
staging dir + outbox under ``work_dir``); copy 3 = GitHub encrypted.
Optional copy 4 = S3-compatible store (B2/R2/Wasabi) via
``ship_to_blob_store`` when ``s3_store`` is given.

Scheduling: :func:`schedule_specs` returns the two jobs (daily backup,
monthly drill); :func:`register_jobs` installs them on a
:class:`SchedulerFacade` / :class:`UnifiedScheduler` via real
``UnifiedJob`` rows. Termux-killed runs are safe: the local snapshot is
already taken before any network work, and the outbox drains on the
next run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .errors import BackupError, DiskFullError
from .manifest import build_manifest, write_manifest
from .plan import REPO_NAME
from .policy import prune_github_releases, prune_local, select_keepers
from .protect import load_backup_key, protect_snapshot
from .report import (
    report_backup_failure,
    report_backup_result,
)
from .ship import (
    Outbox,
    ensure_repo,
    ship_manifest_to_repo,
    ship_snapshot,
    ship_to_blob_store,
)
from .snapshot import take_snapshot

__all__ = [
    "BackupCoordinator",
    "schedule_specs",
    "register_jobs",
    "run_scheduled_backup",
    "run_scheduled_drill",
]

_log = get_logger(__name__)

DAILY_CRON = "0 2 * * *"
MONTHLY_DRILL_CRON = "0 3 1 * *"


def schedule_specs() -> list[dict[str, Any]]:
    """The two scheduled jobs for this coordinator."""
    return [
        {"job_id": "github-backup-daily", "kind": "cron",
         "recurrence": {"type": "cron", "spec": DAILY_CRON},
         "payload": {"handler": "nomorals.backup.coordinator:run_scheduled_backup"},
         "name": "GitHub backup (daily)"},
        {"job_id": "github-backup-drill-monthly", "kind": "cron",
         "recurrence": {"type": "cron", "spec": MONTHLY_DRILL_CRON},
         "payload": {"handler": "nomorals.backup.coordinator:run_scheduled_drill"},
         "name": "Restore drill (monthly)"},
    ]


def register_jobs(facade: Any) -> list[dict[str, Any]]:
    """Install the two jobs on a SchedulerFacade / UnifiedScheduler."""
    from ..scheduler.unified_scheduler import UnifiedJob

    installed: list[dict[str, Any]] = []
    for spec in schedule_specs():
        try:
            job = UnifiedJob(
                job_id=spec["job_id"], kind="cron",
                recurrence=spec["recurrence"], payload=spec["payload"],
                name=spec["name"],
            )
            facade.add_job(job)
            installed.append({"job_id": spec["job_id"], "installed": True})
        except ValueError as exc:
            # Already installed — idempotent re-register is not an error.
            installed.append({"job_id": spec["job_id"], "installed": False,
                              "reason": str(exc)[:200]})
    return installed


@dataclass
class BackupCoordinator:
    """Owns one backup configuration: root + work dir + vault + shipper."""

    root: str | Path
    work_dir: str | Path
    vault: Any = None
    connector: Any = None  # GitHubConnector (vault-backed credential)
    notifier: Any = None  # Notifier or context (report target)
    s3_store: Any = None  # optional 4th copy (S3-compatible blob store)
    repo_name: str = REPO_NAME

    def _key(self) -> bytes:
        if self.vault is None:
            raise BackupError(
                "no vault configured — the backup key lives in the vault, "
                "never in .env, never in the repo"
            )
        return load_backup_key(self.vault)

    def run_backup(self, *, label: str = "", prune: bool = True) -> dict[str, Any]:
        """Run one full generation. Returns the result dict; never silent."""
        snapshot = take_snapshot(self.root, self.work_dir, label=label)
        backup_id = snapshot.backup_id
        try:
            key = self._key()
            out_dir = Path(self.work_dir) / backup_id / "protected"
            protected = protect_snapshot(snapshot, out_dir, key)
            manifest = build_manifest(
                backup_id=backup_id, created_at=snapshot.created_at,
                protected=protected, plan=snapshot.plan,
                release_tag=backup_id,
            )
            manifest_path = write_manifest(
                Path(self.work_dir) / backup_id, manifest
            )
            blob_files = [p for item in protected for p in item.parts]
            result: dict[str, Any] = {
                "backup_id": backup_id,
                "manifest": manifest,
                "manifest_path": str(manifest_path),
                "protected": [p.to_dict() for p in protected],
                "total_encrypted_bytes": manifest["total_encrypted_bytes"],
                "shipped": False, "queued": False,
                "manifest_pushed": False,
            }
            # Drain the outbox first: older generations ship before newer.
            if self.connector is not None:
                outbox = Outbox(Path(self.work_dir) / "outbox")
                for queued in outbox.pending():
                    try:
                        self._ship_queued(queued, outbox)
                    except Exception as exc:  # noqa: BLE001 - one bad
                        # queued item must not kill the current generation
                        _log.warning("outbox drain failed for %s: %s",
                                     queued.get("backup_id"), exc)
                        outbox.mark_failed(str(queued.get("backup_id")))
                shipped = self._ship_current(manifest, blob_files)
                result.update(shipped)
            else:
                outbox = Outbox(Path(self.work_dir) / "outbox")
                entry = outbox.enqueue({
                    "backup_id": backup_id, "manifest": manifest,
                    "blob_files": [str(p) for p in blob_files],
                    "error": "no GitHub connector configured",
                })
                result.update({"queued": True, "outbox_entry": entry,
                               "reason": "no GitHub connector — queued"})
            if self.s3_store is not None and not result.get("queued"):
                try:
                    result["s3"] = ship_to_blob_store(
                        self.s3_store, blob_files
                    )
                except Exception as exc:  # noqa: BLE001 - 4th copy is bonus
                    _log.warning("S3 4th copy failed for %s: %s", backup_id, exc)
                    result["s3_error"] = str(exc)[:300]
            if prune:
                self._prune_local()
            report_backup_result(self.notifier, result)
            return result
        except DiskFullError:
            raise
        except Exception as exc:  # noqa: BLE001 - report, then re-raise typed
            report_backup_failure(self.notifier, backup_id, exc)
            raise

    def _ship_current(
        self, manifest: dict[str, Any], blob_files: list[str]
    ) -> dict[str, Any]:
        ensured = ensure_repo(self.connector, self.repo_name)
        full_name = str(ensured["full_name"])
        shipped = ship_snapshot(
            self.connector, full_name, manifest, blob_files,
            outbox_dir=Path(self.work_dir) / "outbox",
        )
        if shipped.get("shipped"):
            pushed = ship_manifest_to_repo(
                self.connector, full_name, manifest,
                work_dir=Path(self.work_dir),
            )
            shipped["manifest_pushed"] = bool(pushed.get("pushed"))
            shipped["tag"] = shipped.get("tag") or manifest.get("backup_id")
        return shipped

    def _ship_queued(self, queued: dict[str, Any], outbox: Outbox) -> None:
        ensured = ensure_repo(self.connector, self.repo_name)
        shipped = ship_snapshot(
            self.connector, str(ensured["full_name"]),
            queued["manifest"], queued["blob_files"],
            outbox_dir=Path(self.work_dir) / "outbox",
        )
        if shipped.get("shipped"):
            outbox.mark_sent(str(queued.get("backup_id")))
            ship_manifest_to_repo(
                self.connector, str(ensured["full_name"]),
                queued["manifest"], work_dir=Path(self.work_dir),
            )

    def _prune_local(self) -> dict[str, Any]:
        work = Path(self.work_dir)
        entries: list[dict[str, Any]] = []
        for child in sorted(work.iterdir()):
            if not child.is_dir() or not child.name.startswith("backup-"):
                continue
            manifest_file = child / "manifest.json"
            created = child.stat().st_mtime
            if manifest_file.is_file():
                try:
                    import json

                    created = float(json.loads(
                        manifest_file.read_text(encoding="utf-8")
                    ).get("created_at", created))
                except (ValueError, OSError):
                    pass
            entries.append({"backup_id": child.name, "created_at": created,
                            "staging_dir": str(child)})
        pruned = prune_local(entries)
        # Prune GitHub releases under the same policy (best effort: a
        # retention failure must not fail the backup that just shipped).
        if self.connector is not None:
            try:
                ensured = ensure_repo(self.connector, self.repo_name)
                pruned["github"] = prune_github_releases(
                    self.connector, str(ensured["full_name"])
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("GitHub retention prune failed: %s", exc)
                pruned["github_error"] = str(exc)[:300]
        _ = select_keepers(entries)  # policy single source of truth
        return pruned

    def run_drill(self, *, backup_id: str = "") -> Any:
        """Monthly restore drill over the latest (or named) generation."""
        from .protect import load_backup_key as _load_key
        from .verify import run_monthly_drill

        work = Path(self.work_dir)
        candidates = sorted(
            (d for d in work.iterdir()
             if d.is_dir() and d.name.startswith("backup-")),
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        if backup_id:
            candidates = [d for d in candidates if d.name == backup_id]
        if not candidates:
            raise BackupError("no local generations to drill — nothing to verify")
        target = candidates[0]
        key = _load_key(self.vault) if self.vault is not None else self._key()
        blobs = target / "protected"
        return run_monthly_drill(
            target / "manifest.json", blobs, key, report=self.notifier
        )


def run_scheduled_backup(coordinator: BackupCoordinator) -> dict[str, Any]:
    """Scheduler payload handler: daily backup."""
    return coordinator.run_backup(label="scheduled")


def run_scheduled_drill(coordinator: BackupCoordinator) -> Any:
    """Scheduler payload handler: monthly restore drill."""
    return coordinator.run_drill()
