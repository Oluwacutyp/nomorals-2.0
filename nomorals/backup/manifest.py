"""Manifest: the git-native record of a backup generation.

The backup repo holds manifests/checksums/schema/restore recipe/small
config/docs ONLY (text, KB, diffs). Blobs live in Releases. The manifest
is plaintext JSON — it holds checksums of ciphertext, sizes, schema
versions, and table counts, never secrets, never plaintext data.

A manifest is the unit the restore drill verifies against, and the unit
retention prunes by.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .errors import BackupError
from .plan import SCHEMA_VERSION

__all__ = [
    "MANIFEST_NAME",
    "BACKUP_REPO_GITIGNORE",
    "BACKUP_REPO_README",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "verify_manifest_files",
    "render_restore_recipe",
]

MANIFEST_NAME = "manifest.json"

#: .gitignore for the backup repo — binary snapshots NEVER in commit
#: history on schedule. This is written into the backup repo before the
#: first commit so no accident can land a blob in history.
BACKUP_REPO_GITIGNORE = """# nomorals-backup: manifests + restore recipe ONLY.
# Blobs (encrypted gzipped snapshots) live in GitHub Releases, never here.
*.db
*.db-wal
*.db-shm
*.sqlite
*.sqlite3
*.gz
*.enc
*.part*
*.zip
*.tar
cache/
logs/
llm/
vault*
.vault*
.env
.env.*
backups/
__pycache__/
"""

BACKUP_REPO_README = """# nomorals-backup

Encrypted offsite backups of Devon. This repo holds **manifests,
checksums, schema versions, and the restore recipe ONLY** — text, KB,
diffs. The encrypted snapshot blobs live in **GitHub Releases** (2GB per
asset cap; blobs are split into 100MB parts), so the repo never bloats
and a binary snapshot never lands in commit history.

## Restore

1. Pick a release (one per backup generation) and download its
   `*.gz.enc` parts.
2. Reassemble parts in name order, then decrypt with the backup key:
   the key is in the vault (`github-backup` / `backup-key`) AND in the
   owner's offline copy (paper / password manager). **Lose the key,
   lose the backup — there is no reset.**
3. Verify the SHA-256 of the ciphertext against `manifest.json`
   BEFORE decrypting.
4. Decrypt + gunzip, then `PRAGMA integrity_check` the SQLite files.
5. See `RESTORE.md` in the release for the full from-zero recipe.

## Layout

- `manifests/<backup-id>/manifest.json` — what, when, size, checksum,
  schema version per file.
- `RESTORE.md` — the from-zero restore recipe.
- Releases — encrypted blobs, one release per generation.
"""


def build_manifest(
    *,
    backup_id: str,
    created_at: float | None = None,
    protected: list[Any],
    plan: Any | None = None,
    release_tag: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build the git-native record for one generation."""
    files = []
    for item in protected:
        entry = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        files.append(entry)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backup_id": backup_id,
        "created_at": created_at or time.time(),
        "created_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at or time.time())
        ),
        "release_tag": release_tag,
        "files": files,
        "total_encrypted_bytes": sum(f["size"] for f in files),
        "file_count": len(files),
        "notes": notes,
    }
    if plan is not None:
        manifest["plan"] = (
            plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
        )
    return manifest


def write_manifest(directory: str | Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest.json`` into ``directory`` (plaintext, checksums only)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load + schema-check a manifest. Refuses unknown schemas outright."""
    path = Path(path)
    if not path.is_file():
        raise BackupError(f"manifest missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BackupError(f"manifest {path} is not valid JSON: {exc}") from exc
    if data.get("schema_version") != SCHEMA_VERSION:
        raise BackupError(
            f"manifest {path} schema_version "
            f"{data.get('schema_version')!r} != supported {SCHEMA_VERSION} "
            "— refusing to guess"
        )
    return data


def verify_manifest_files(
    base_dir: str | Path, manifest: dict[str, Any]
) -> list[str]:
    """Check every manifest file exists under ``base_dir`` with matching hash.

    Mirrors :func:`nomorals.memory.backup.verify_backup`: returns a list
    of problems; empty means clean. Hash mismatches stop at the report —
    a mismatch is FAILED, not partial.
    """
    import hashlib

    base = Path(base_dir)
    problems: list[str] = []
    for entry in manifest.get("files", []):
        for candidate in entry.get("parts", []) or [
            entry.get("encrypted_path", "")
        ]:
            path = Path(candidate)
            if not path.is_absolute():
                path = base / path.name
            if not path.is_file():
                problems.append(f"missing: {path.name}")
                continue
            # Only the single-file case carries a per-file hash; split
            # parts hash to the whole-file ciphertext hash after joining,
            # which the drill does explicitly.
            if len(entry.get("parts", []) or []) <= 1:
                h = hashlib.sha256()
                with path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                if h.hexdigest() != entry.get("sha256"):
                    problems.append(
                        f"checksum mismatch: {path.name} "
                        "(FAILED, not partial — do not trust these bytes)"
                    )
    return problems


def render_restore_recipe(manifest: dict[str, Any]) -> str:
    """From-zero restore recipe for one generation (goes in the repo)."""
    lines = [
        "# Restore — from zero",
        "",
        f"Backup: `{manifest.get('backup_id')}` "
        f"created {manifest.get('created_utc', '')}",
        f"Release tag: `{manifest.get('release_tag', '')}`",
        f"Files: {manifest.get('file_count')} — "
        f"{manifest.get('total_encrypted_bytes', 0)} encrypted bytes",
        "",
        "## You need",
        "",
        "1. The backup key: vault `github-backup` / `backup-key`, or the "
        "offline copy (paper / password manager).",
        "   **Lose the key, lose the backup. There is no reset.**",
        "2. This repo (manifests) + the Release assets for this backup.",
        "3. A clean machine with Python + this codebase.",
        "",
        "## Steps",
        "",
        "1. Download every `*.gz.enc` part from the Release.",
        "2. Reassemble split parts in filename order "
        "(`.part00of03`, `.part01of03`, …).",
        "3. SHA-256 the reassembled ciphertext; it MUST equal the "
        "`sha256` in `manifest.json`. Mismatch = FAILED, do not decrypt.",
        "4. Decrypt with the backup key (AES-256, `nmc1` blob), then gunzip.",
        "5. For each `.db` file: open with sqlite3, run "
        "`PRAGMA integrity_check` (must be `ok`), confirm the table count "
        "matches the manifest.",
        "6. Put files back at their `rel` paths relative to the project "
        "root. Vault files go back encrypted-only — never plaintext.",
        "7. Start Devon; run `nm backup verify` as a second opinion.",
        "",
        "## Files in this generation",
        "",
    ]
    for entry in manifest.get("files", []):
        lines.append(
            f"- `{entry.get('rel')}` — {entry.get('kind')}, "
            f"{entry.get('size')} bytes encrypted, "
            f"sha256 `{str(entry.get('sha256'))[:16]}…`, "
            f"schema v{entry.get('schema_version')}, "
            f"{entry.get('tables')} tables"
        )
    lines.append("")
    return "\n".join(lines)
