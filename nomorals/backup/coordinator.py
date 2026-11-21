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
    clear_backup_failure_streak,
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
    "open_s3_store",
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


def open_s3_store() -> Any | None:
    """Open the optional S3 4th-copy store from the environment, or None.

    S3 is configured purely by environment (``S3_ENDPOINT`` /
    ``S3_BUCKET`` / ``S3_ACCESS_KEY`` / ``S3_SECRET_KEY`` — see
    :class:`S3Config.from_env`): fully set returns a real
    :class:`S3BlobStore` (construction touches no network); anything
    missing or malformed returns None and the backup runs exactly as
    before, with no 4th copy. Never raises for a missing or malformed
    S3 env — an optional copy must never kill the daily backup.
    """
    try:
        from ..core.errors import ConfigError
        from ..storage.s3blob import S3Config, open_blob_store

        return open_blob_store("s3", s3_config=S3Config.from_env())
    except ConfigError:
        _log.info("S3 4th copy not configured — skipping offsite copy")
        return None
    except ValueError as exc:
        _log.warning("S3 4th copy misconfigured (%s) — skipping "
                     "offsite copy", exc)
        return None


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
    #: Workspace dir for the WhatsApp session seal (settings
    #: ``workspace_dir`` — the pairing registry + session dirs live
    #: under it). None skips the session step: the coordinator
    #: predates the wiring, and a caller that cannot name a
    #: workspace must not gain a guessed one.
    session_workspace_dir: str | Path | None = None

    def _backup_whatsapp_session(self) -> dict[str, Any]:
        """Seal the linked WhatsApp session(s) for this generation.

        Fail-open by design: the session copy is additive — a run
        with no WhatsApp at all, no passphrase in the vault, or an
        unreadable registry still ships its snapshot. The outcome
        (written rows, or the skip reason) rides the result dict so
        the report line appears only when a copy was actually
        written, and the skip reason is in the log, never silent.
        """
        from ..social.chat.baileys_guard import seal_whatsapp_sessions
        from ..social.pairing import read_registry_snapshot, registry_path_for
        from .protect import load_session_passphrase

        workspace = self.session_workspace_dir
        if workspace is None:
            return {"skipped": [{"account": "",
                                 "reason": "no session workspace "
                                           "configured"}]}
        if self.vault is None:
            return {"skipped": [{"account": "",
                                 "reason": "no vault configured — the "
                                           "session passphrase lives "
                                           "in the vault"}]}
        try:
            passphrase = load_session_passphrase(self.vault)
        except Exception as exc:  # noqa: BLE001 - skip, with the reason
            _log.warning("WhatsApp session seal skipped: %s", exc)
            return {"skipped": [{"account": "", "reason": str(exc)[:300]}]}
        try:
            snapshot = read_registry_snapshot(
                registry_path_for(workspace))
        except Exception as exc:  # noqa: BLE001 - skip, with the reason
            _log.warning("WhatsApp session seal skipped: %s", exc)
            return {"skipped": [{"account": "", "reason": str(exc)[:300]}]}
        if snapshot.error:
            _log.warning("WhatsApp session seal skipped: %s",
                         snapshot.error)
            return {"skipped": [{"account": "",
                                 "reason": snapshot.error[:300]}]}
        try:
            sealed = seal_whatsapp_sessions(
                snapshot.records("whatsapp"), passphrase)
        except Exception as exc:  # noqa: BLE001 - skip, with the reason
            _log.warning("WhatsApp session seal failed: %s", exc)
            return {"skipped": [{"account": "", "reason": str(exc)[:300]}]}
        if not sealed["written"] and not sealed["skipped"]:
            sealed["skipped"] = [{
                "account": "",
                "reason": "no WhatsApp accounts are paired"}]
        return sealed

    def _key(self) -> bytes:
        if self.vault is None:
            raise BackupError(
                "no vault configured — the backup key lives in the vault, "
                "never in .env, never in the repo"
            )
        return load_backup_key(self.vault)

    def run_backup(self, *, label: str = "", prune: bool = True) -> dict[str, Any]:
        """Run one full generation. Returns the result dict; never silent.

        The encryption key is checked FIRST, before the snapshot: a
        run with no key used to take the whole snapshot and only then
        die on ``BackupKeyError`` (2026-11-21: the daily job did the
        expensive half of the work every morning and failed anyway).
        A missing key is a setup state, not a snapshot failure — fail
        fast, before any work, with the key error naming the fix.
        """
        # Fail fast on the precondition: no key, no snapshot.
        key = self._key()
        snapshot = take_snapshot(self.root, self.work_dir, label=label)
        backup_id = snapshot.backup_id
        try:
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
            # The sealed WhatsApp session copy rides this generation:
            # refreshed in place every run, reported only when one
            # was actually written. A skip never fails the run.
            result["whatsapp_session"] = self._backup_whatsapp_session()
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
            clear_backup_failure_streak(self.work_dir)
            return result
        except DiskFullError:
            raise
        except Exception as exc:  # noqa: BLE001 - report, then re-raise typed
            report_backup_failure(self.notifier, backup_id, exc,
                                  state_dir=self.work_dir)
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
                except (ValueError, OSError) as _exc:
                    _log.debug("coordinator._prune_local failed", exc_info=True)
            entries.append({"backup_id": child.name, "created_at": created,
                            "staging_dir": str(child)})
        # A generation still queued in the outbox has NOT shipped —
        # its local blobs are the only copy. Retention must never
        # delete it, however old it is; unqueued generations prune
        # under the normal policy.
        queued_ids = Outbox(work / "outbox").queued_ids()
        if queued_ids:
            spared = [e["backup_id"] for e in entries
                      if e["backup_id"] in queued_ids]
            if spared:
                _log.info("retention: sparing outbox-queued "
                          "generation(s) from prune: %s", spared)
            entries = [e for e in entries
                       if e["backup_id"] not in queued_ids]
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
