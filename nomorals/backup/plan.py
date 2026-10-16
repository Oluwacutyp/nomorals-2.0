"""Snapshot plan: what a backup contains, what it excludes, and why.

The plan is data, not code paths: :mod:`nomorals.backup.snapshot` walks
exactly what :func:`build_plan` returns, so the manifest, the drill, and
the restore recipe all agree on what "a backup" means.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "REPO_NAME",
    "SCHEMA_VERSION",
    "INCLUDED_DATABASES",
    "INCLUDED_DIRS",
    "EXCLUDE_DIRS",
    "EXCLUDE_SUFFIXES",
    "EXCLUDE_NAMES",
    "VAULT_PATTERNS",
    "SNAPSHOT_CONTENTS",
    "BackupPlan",
    "build_plan",
]

#: The backup repo holds manifests/checksums/schema/restore recipe/small
#: config/docs ONLY. Blobs (encrypted gzipped snapshots) go to Releases.
#: Never push code into it — it is not a code repo.
REPO_NAME = "nomorals-backup"

#: Manifest schema version. Bump when the manifest format changes; the
#: restore drill refuses manifests it cannot parse instead of guessing.
SCHEMA_VERSION = 1

#: Canonical database locations, relative to the project root. Only
#: ``data/nomorals.db`` is required — the rest are recorded as
#: ``missing`` (not fatal) when a deployment never created them, so a
#: phone install without e.g. ``social_drafts.db`` still backs up.
INCLUDED_DATABASES: tuple[str, ...] = (
    "data/nomorals.db",  # required: the live brain (see LIVE_SCHEMA below)
    "data/representation.db",
    "data/social_drafts.db",
    "data/health/timeline.db",
    "data/planning/world_graph.db",
)

REQUIRED_DATABASE = "data/nomorals.db"

#: User-data directories, relative to the project root. Absent dirs are
#: recorded as absent, not errors.
INCLUDED_DIRS: tuple[str, ...] = (
    "memory",
    "characters",
    "games",
    "finance",
    "legal",
    "matching",
    "social",
    "marketing",
    "audio",
    "workspace/books",
    "dj_mixes",
    "config",
)

#: Never walked, never snapshotted.
EXCLUDE_DIRS: frozenset[str] = frozenset({
    "cache", "logs", "llm", "__pycache__", ".git", "node_modules",
    ".venv", "venv", "temp", "tmp", ".pytest_cache", ".ruff_cache",
})

#: File suffixes that are runtime sidecars / locks / sockets, never data.
EXCLUDE_SUFFIXES: tuple[str, ...] = (
    ".db-wal", ".db-shm", ".lock", ".sock", ".pyc", ".pyo",
)

#: Basenames never snapshotted in any form.
EXCLUDE_NAMES: frozenset[str] = frozenset({
    ".env", ".env.local", ".DS_Store", "Thumbs.db",
})

#: Vault files match these patterns. They are included ONLY through the
#: encrypted path (:mod:`nomorals.backup.protect`) — never plaintext,
#: never in git. A vault file that would land in a snapshot unencrypted
#: is a plan error, not a warning.
VAULT_PATTERNS: tuple[str, ...] = ("vault", "vaults", ".vault")

#: What a snapshot contains and why. Kept as data so the manifest and
#: the restore recipe can render it without drifting from the code.
SNAPSHOT_CONTENTS: tuple[dict[str, str], ...] = (
    {"what": "data/nomorals.db",
     "why": "Live brain: conversations, memory, goals, jobs, trading journal."},
    {"what": "data/representation.db",
     "why": "Representation / persona state, if the deployment created it."},
    {"what": "data/social_drafts.db",
     "why": "Drafted-but-unsent social posts; losing them loses owner intent."},
    {"what": "data/health/timeline.db",
     "why": "Health timeline — irreplaceable personal history."},
    {"what": "data/planning/world_graph.db",
     "why": "WorldGraph: people, commitments, extracted facts."},
    {"what": "memory/ characters/ games/ finance/ legal/ matching/ social/ "
             "marketing/ audio/ workspace/books/ dj_mixes/ config/",
     "why": "User data + configuration that the databases reference. "
            "Without config, a restored DB points at settings that no "
            "longer exist."},
    {"what": "vault (encrypted always)",
     "why": "Credentials. Encrypted with the backup key, never plaintext, "
            "never in git — the backup contains live session cookies "
            "(browser_login persists into SessionManager/DB), which ARE "
            "credentials."},
    {"what": "llm/ manifest only",
     "why": "Model WEIGHTS are not backed up (gigabytes, re-downloadable). "
            "The manifest records what was downloaded (name/size/source) "
            "so a from-zero restore re-fetches exactly the same files."},
)


def _is_vault_path(rel: str) -> bool:
    lower = rel.lower()
    name = Path(lower).name
    return any(
        name == pat or name.startswith(pat + ".") or name.startswith(pat + "_")
        for pat in VAULT_PATTERNS
    ) or "vault" in Path(lower).parts


def _excluded(rel_path: Path) -> str | None:
    """Reason the path is excluded, or None when it is includable."""
    parts = {p.lower() for p in rel_path.parts}
    if parts & {d.lower() for d in EXCLUDE_DIRS}:
        return "excluded dir"
    if rel_path.name in EXCLUDE_NAMES:
        return "excluded name (.env / OS junk — never backed up)"
    if rel_path.suffix.lower() in (".db-wal", ".db-shm"):
        return "sqlite sidecar — snapshot is checkpointed, self-contained"
    if rel_path.name.endswith(EXCLUDE_SUFFIXES):
        return "lock/socket/bytecode sidecar"
    if rel_path.suffix.lower() in (".gguf", ".safetensors", ".bin", ".pt",
                                   ".ckpt", ".onnx"):
        return "model weights — manifest only, never the bytes"
    return None


@dataclass
class BackupPlan:
    """A resolved plan for one snapshot run."""

    root: Path
    databases: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    vault_files: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    llm_manifest: dict[str, Any] = field(default_factory=dict)
    contents: list[dict[str, str]] = field(
        default_factory=lambda: [dict(c) for c in SNAPSHOT_CONTENTS]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "databases": self.databases,
            "files": self.files,
            "vault_files": self.vault_files,
            "excluded": self.excluded,
            "llm_manifest": self.llm_manifest,
            "contents": self.contents,
        }


def _llm_manifest(root: Path) -> dict[str, Any]:
    """What models were downloaded — names/sizes/sources, never weights."""
    llm_dir = root / "llm"
    entries: list[dict[str, Any]] = []
    if llm_dir.is_dir():
        for path in sorted(llm_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in (".gguf", ".safetensors", ".bin",
                                       ".pt", ".ckpt", ".onnx"):
                entries.append({
                    "name": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    # No hash: hashing multi-GB weights on every snapshot
                    # costs more than the record is worth; size + name
                    # identifies the re-download.
                    "note": "weights not backed up — re-download on restore",
                })
            elif path.name.endswith((".json", ".txt", ".md")):
                entries.append({
                    "name": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                })
    return {"dir": "llm", "models": entries,
            "note": "Manifest of what was downloaded, not the weights."}


def _sha256_small(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_plan(root: str | os.PathLike[str]) -> BackupPlan:
    """Resolve include/exclude for ``root`` into a concrete plan.

    Never raises on a missing optional source: missing entries are
    recorded with ``status: "missing"`` so the manifest is honest about
    what the snapshot did NOT contain.
    """
    root = Path(root).expanduser()
    plan = BackupPlan(root=root)

    for rel in INCLUDED_DATABASES:
        path = root / rel
        entry: dict[str, Any] = {"rel": rel, "kind": "database"}
        if path.is_file():
            entry.update({"status": "present", "size": path.stat().st_size,
                          "required": rel == REQUIRED_DATABASE})
        else:
            entry.update({"status": "missing", "size": 0,
                          "required": rel == REQUIRED_DATABASE})
        plan.databases.append(entry)

    for rel_dir in INCLUDED_DIRS:
        base = root / rel_dir
        if not base.is_dir():
            plan.excluded.append({"rel": rel_dir, "reason": "absent"})
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            reason = _excluded(path.relative_to(root))
            if reason:
                plan.excluded.append({"rel": rel, "reason": reason})
                continue
            if path.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
                # A stray DB inside a data dir still goes through the
                # SQLite online backup API, never a raw copy.
                plan.databases.append({"rel": rel, "kind": "database",
                                       "status": "present",
                                       "size": path.stat().st_size,
                                       "required": False})
                continue
            entry = {"rel": rel, "kind": "file",
                     "size": path.stat().st_size}
            if _is_vault_path(rel):
                plan.vault_files.append(entry)
            else:
                plan.files.append(entry)

    # Vault material that lives outside the included dirs (e.g. a
    # top-level vault file) is still encrypt-only, never plaintext.
    for path in sorted(root.glob("vault*")):
        if path.is_file() and not any(
            v["rel"] == path.name for v in plan.vault_files
        ):
            plan.vault_files.append({"rel": path.name, "kind": "file",
                                     "size": path.stat().st_size})
    plan.llm_manifest = _llm_manifest(root)
    return plan
