"""GitHub backup coordinator: encrypted offsite backups, 3-2-1.

Repo holds manifests/checksums/schema/restore recipe ONLY (text, KB,
diffs). Blobs — encrypted gzipped snapshots — go to GitHub Releases
(2GB per asset, split into 100MB parts). Binary snapshots NEVER in
commit history.

Encryption is MANDATORY: the backup contains live session cookies
(= credentials). Key lives in the vault, never in the repo, never in
.env. Lose the key, lose the backup — keep an offline copy
(paper / password manager).
"""

from __future__ import annotations

from .coordinator import (
    BackupCoordinator,
    register_jobs,
    run_scheduled_backup,
    run_scheduled_drill,
    schedule_specs,
)
from .errors import (
    KEY_LOSS_WARNING,
    BackupError,
    BackupKeyError,
    ChecksumMismatchError,
    CorruptBackupError,
    DiskFullError,
    RateLimitedBackupError,
    TokenExpiredError,
    TokenScopeError,
)
from .manifest import (
    BACKUP_REPO_GITIGNORE,
    BACKUP_REPO_README,
    build_manifest,
    load_manifest,
    render_restore_recipe,
    write_manifest,
)
from .plan import REPO_NAME, SCHEMA_VERSION, build_plan
from .policy import KEEP_DAILY, KEEP_LAST, KEEP_MONTHLY, select_keepers
from .protect import (
    generate_backup_key,
    load_backup_key,
    store_backup_key,
)
from .ship import Outbox, check_token_scopes, ensure_repo
from .snapshot import take_snapshot
from .verify import DrillReport, drill, run_monthly_drill

__all__ = [
    "BackupCoordinator",
    "BackupError",
    "BackupKeyError",
    "ChecksumMismatchError",
    "CorruptBackupError",
    "DiskFullError",
    "DrillReport",
    "KEY_LOSS_WARNING",
    "KEEP_DAILY",
    "KEEP_LAST",
    "KEEP_MONTHLY",
    "Outbox",
    "REPO_NAME",
    "SCHEMA_VERSION",
    "BACKUP_REPO_GITIGNORE",
    "BACKUP_REPO_README",
    "RateLimitedBackupError",
    "TokenExpiredError",
    "TokenScopeError",
    "build_manifest",
    "build_plan",
    "check_token_scopes",
    "drill",
    "ensure_repo",
    "generate_backup_key",
    "load_backup_key",
    "load_manifest",
    "register_jobs",
    "render_restore_recipe",
    "run_monthly_drill",
    "run_scheduled_backup",
    "run_scheduled_drill",
    "schedule_specs",
    "select_keepers",
    "store_backup_key",
    "take_snapshot",
    "write_manifest",
]
